"""Historical frozen encoders and evidence analysis.

Vocabulary selection/fitting now lives in TVA. Compatibility wrappers below
resolve those trainers lazily; DTLR evidence generation remains independent.
"""
from collections import Counter, defaultdict
import math
import unicodedata
from .selection import text_sha256
from ._tva_training import load_training

SCHEMA = "dtlr.handwriting-subwords.v1"
SPAN_METHOD = "pairwise-consistent-dominant-component-v1"
TEXT_SCHEMA = "dtlr.text-subwords.v1"


def prepare_corpus(records, rows, selection, manifest):
    if (selection.get("schema_version") != "dtlr.iam-selection.v1"
            or selection.get("dataset") != "IAM" or selection.get("split") != "train"):
        raise ValueError("selection must be IAM train")
    if (manifest.get("schema_version") != "dtlr.export-manifest.v5"
            or manifest.get("datasets") != ["IAM"] or manifest.get("splits") != ["train"]
            or manifest.get("primary_connectivity_method") != "dominant-core-v3"):
        raise ValueError("manifest must declare sole IAM train dominant-core-v3 evidence")
    selected = selection["lines"]
    if (len(selected) != selection["requested_count"] or len(records) != len(selected)
            or len(records) != manifest["line_count"] or len(rows) != manifest["evidence_count"]):
        raise ValueError("selection, line, or evidence counts disagree")
    corpus, by_id = [], {}
    for record, item in zip(records, selected):
        if (record.get("dataset") != "IAM" or record.get("split") != "train"
                or record["line_id"] != item["id"]
                or text_sha256(record["transcription"]) != item["transcription_sha256"]):
            raise ValueError("detection text/order differs from frozen IAM train selection")
        text = record["transcription"]
        if unicodedata.normalize("NFC", text) != text:
            raise ValueError("non-NFC evidence text needs an explicit offset remapping")
        if record["line_id"] in by_id:
            raise ValueError("duplicate line ID")
        line = {"line_id": record["line_id"], "text": text, "edges": {}}
        corpus.append(line)
        by_id[line["line_id"]] = line
    for row in rows:
        if (row.get("schema_version") != "dtlr.bigram-evidence.v5"
                or row.get("dataset") != "IAM" or row.get("split") != "train"
                or row.get("connectivity_method") != "dominant-core-v3"):
            raise ValueError("row is not IAM train dominant-core-v3 evidence")
        if row["line_id"] not in by_id:
            raise ValueError("evidence line is outside frozen selection")
        line = by_id[row["line_id"]]
        start = row["left_gt_index"]
        if (not isinstance(start, int) or start < 0 or start + 1 >= len(line["text"])
                or row["right_gt_index"] != start + 1
                or line["text"][start:start + 2] != row["pair"]
                or any(c.isspace() for c in row["pair"])):
            raise ValueError("evidence offsets/text do not match transcription")
        if start in line["edges"]:
            raise ValueError("duplicate edge evidence")
        usable = (row["usable"] is True
                  and row["left_alignment"] == row["right_alignment"] == "match")
        left, right = row.get("left_dominant_component_id"), row.get("right_dominant_component_id")
        if usable and (type(left) is not int or left <= 0 or type(right) is not int or right <= 0
                       or row["connected"] is not (left == right)):
            raise ValueError("usable edge has inconsistent dominant component evidence")
        line["edges"][start] = (left, right) if usable else None
    for line in corpus:
        expected = {i for i in range(len(line["text"]) - 1)
                    if not any(c.isspace() for c in line["text"][i:i + 2])}
        if set(line["edges"]) != expected:
            raise ValueError("incomplete per-line non-whitespace edge evidence")
    return corpus


def span_connected(line, start, end):
    """Return None for unusable/non-exact evidence, rather than False."""
    edges = [line["edges"].get(i) for i in range(start, end - 1)]
    if not edges or any(edge is None for edge in edges):
        return None
    return len({component for edge in edges for component in edge}) == 1


def _apply_merge(sequence, pair):
    result, i = [], 0
    while i < len(sequence):
        if i + 1 < len(sequence) and (sequence[i][0], sequence[i + 1][0]) == pair:
            result.append((pair[0] + pair[1], sequence[i][1], sequence[i + 1][2]))
            i += 2
        else:
            result.append(sequence[i])
            i += 1
    return result


def _lattice(text, log_probs):
    max_length = max(map(len, log_probs))
    return [[(end, text[start:end], log_probs[text[start:end]])
             for end in range(start + 1, min(len(text), start + max_length) + 1)
             if text[start:end] in log_probs] for start in range(len(text))]


class HandwritingSubwordTokenizer:
    def __init__(self, model):
        if (model.get("status") == "frozen-experiment-policy" or "freeze" in model
                or str(model.get("model_version", "")).startswith("iam-english-handwriting-")):
            from .subword_freeze import validate_frozen_subword_artifact
            validate_frozen_subword_artifact(model)
        algorithms = {SCHEMA: ("handwriting-aware-bpe", "handwriting-aware-unigram"),
                      TEXT_SCHEMA: ("text-only-bpe", "text-only-unigram")}
        if model.get("algorithm") not in algorithms.get(model.get("schema_version"), ()):
            raise ValueError("unsupported handwriting subword model")
        vocab = model["vocab"]
        inverse = {int(i): t for i, t in model["idx_token"].items()}
        if (vocab.get("") != 0 or model.get("blank_id") != 0 or model.get("blank_token") != ""
                or model.get("size") != len(vocab) or set(vocab.values()) != set(range(len(vocab)))
                or inverse != {i: t for t, i in vocab.items()}
                or model.get("unicode_normalization") != "NFC"):
            raise ValueError("invalid CTC vocabulary or normalization")
        subwords = {t for t in vocab if len(t) > 1}
        if any(not all(c.isalpha() and c in vocab for c in t) for t in subwords):
            raise ValueError("subwords must be letters with singleton fallback")
        if model["algorithm"].endswith("-bpe"):
            available = {t for t in vocab if len(t) == 1}
            for merge in model["merges"]:
                if (merge["left"] not in available or merge["right"] not in available
                        or merge["token"] != merge["left"] + merge["right"]
                        or merge["token"] not in subwords or merge["token"] in available):
                    raise ValueError("invalid BPE merge order")
                available.add(merge["token"])
            if available != set(vocab) - {""}:
                raise ValueError("BPE vocabulary differs from learned merges")
        else:
            scores = model["log_probs"]
            if (set(scores) != set(vocab) - {""}
                    or any(not math.isfinite(v) or v > 0 for v in scores.values())
                    or not math.isclose(sum(math.exp(v) for v in scores.values()), 1, abs_tol=1e-8)):
                raise ValueError("invalid unigram probabilities")
        self.model, self.vocab, self.idx_token = model, vocab, inverse
        self.size = len(vocab)

    def segment(self, text):
        text = unicodedata.normalize("NFC", text)
        missing = sorted(set(text) - set(self.vocab))
        if missing:
            raise ValueError(f"characters are absent from tokenizer vocabulary: {missing!r}")
        if self.model["algorithm"].endswith("-bpe"):
            sequence = [(c, i, i + 1) for i, c in enumerate(text)]
            for merge in self.model["merges"]:
                sequence = _apply_merge(sequence, (merge["left"], merge["right"]))
        else:
            arcs = _lattice(text, self.model["log_probs"])
            best, back = [-math.inf] * (len(text) + 1), [None] * (len(text) + 1)
            best[0] = 0.0
            for start, edges in enumerate(arcs):
                for end, token, score in edges:
                    candidate = best[start] + score
                    if candidate > best[end]:
                        best[end], back[end] = candidate, (token, start, end)
            sequence, end = [], len(text)
            while end:
                step = back[end]
                if step is None:
                    raise ValueError("text cannot be segmented")
                sequence.append(step)
                end = step[1]
            sequence.reverse()
        return {"text": text, "tokens": [t for t, _, _ in sequence],
                "segments": [{"token": t, "start": s, "end": e} for t, s, e in sequence]}

    def encode(self, text):
        return [self.vocab[t] for t in self.segment(text)["tokens"]]

    def decode(self, ids):
        try:
            return "".join(self.idx_token[i] for i in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID: {error.args[0]}") from error


def compare_tokenizers(corpus, tokenizers):
    reports = {}
    for name, tokenizer in tokenizers.items():
        chars = token_count = failures = 0
        lengths = Counter()
        internal_known = internal_connected = boundary_known = boundary_connected = 0
        span_known = span_supported = 0
        for line in corpus:
            result = tokenizer.segment(line["text"])
            segments, tokens = result["segments"], result["tokens"]
            ids = tokenizer.encode(line["text"])
            failures += int(tokenizer.decode(ids) != line["text"] or 0 in ids)
            chars += len(line["text"])
            token_count += len(tokens)
            lengths.update(map(len, tokens))
            boundaries = {s["end"] - 1 for s in segments[:-1]}
            for index, edge in line["edges"].items():
                if edge is None:
                    continue
                connected = edge[0] == edge[1]
                if index in boundaries:
                    boundary_known += 1
                    boundary_connected += int(connected)
                else:
                    internal_known += 1
                    internal_connected += int(connected)
            for segment in segments:
                if len(segment["token"]) > 1:
                    connected = span_connected(line, segment["start"], segment["end"])
                    if connected is not None:
                        span_known += 1
                        span_supported += int(connected)
        known_connected = internal_connected + boundary_connected
        reports[name] = {"ctc_class_count": tokenizer.size, "line_count": len(corpus),
                         "character_count": chars, "token_count": token_count,
                         "token_reduction_fraction": 1 - token_count / chars if chars else 0,
                         "token_length_histogram": dict(sorted(lengths.items())),
                         "round_trip_or_blank_failures": failures,
                         "known_internal_edges": internal_known,
                         "connected_internal_edges": internal_connected,
                         "known_boundary_edges": boundary_known,
                         "connected_edges_cut": boundary_connected,
                         "connected_edge_cut_fraction": boundary_connected / known_connected
                         if known_connected else None,
                         "known_multi_token_spans": span_known,
                         "same_component_multi_token_spans": span_supported,
                         "multi_token_same_component_fraction": span_supported / span_known
                         if span_known else None}
    return {"evaluation_scope": "IAM train descriptive; not held-out recognition performance",
            "connectivity_scope": "exact-alignment usable edges only; abstentions excluded",
            "tokenizers": reports}


def extend_bigram_training_alphabet(model, characters):
    """Derived experiment copy: append singleton IDs, preserving all frozen IDs."""
    import copy
    result = copy.deepcopy(model)
    additions = sorted(set(characters) - set(result["vocab"]))
    if any(len(c) != 1 for c in additions):
        raise ValueError("alphabet additions must be single characters")
    original_size = result["size"]
    for char in additions:
        index = len(result["vocab"])
        result["vocab"][char] = index
        result["idx_token"][str(index)] = char
    result["size"] = len(result["vocab"])
    result["status"] = "experimental-derived-training-alphabet"
    result["model_version"] = model.get("model_version", "bigram") + "-iam-train-alphabet-poc"
    result["alphabet_adaptation"] = {"sole_dataset": "IAM", "sole_split": "train",
                                    "source_size": original_size, "added_singletons": additions,
                                    "all_original_ids_preserved": True,
                                    "bigram_inventory_and_utilities_unchanged": True}
    result["reconstruction"] = {"source_model_exact_match": False,
                                "reason": "experimental IAM train singleton expansion only"}
    result["downstream_contract"] = {"vocabulary_selection": "frozen IAM bigrams plus IAM train singletons",
                                     "reference_segmentation": "maximum-total-utility-non-overlapping-v1"}
    result["note"] = "Experimental derived alphabet copy; this is not the unchanged frozen artifact."
    return result


def collect_candidates(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().collect_candidates(*args, **kwargs)


def collect_text_candidates(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().collect_text_candidates(*args, **kwargs)


def train_bpe(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().train_bpe(*args, **kwargs)


def forward_backward(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().forward_backward(*args, **kwargs)


def train_unigram(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().train_unigram(*args, **kwargs)


def removal_losses(*args, **kwargs):
    """Historical API; vocabulary fitting is implemented in TVA."""
    return load_training().removal_losses(*args, **kwargs)
