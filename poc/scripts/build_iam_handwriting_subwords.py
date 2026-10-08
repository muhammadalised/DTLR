#!/usr/bin/env python3
"""Build experimental handwriting-aware BPE/Unigram from pinned IAM CCL exports."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import subprocess

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))
from dtlr_poc.handwriting_subwords import (
    HandwritingSubwordTokenizer, collect_candidates, compare_tokenizers,
    prepare_corpus, train_bpe, train_unigram, extend_bigram_training_alphabet,
)
from dtlr_poc.selection import sha256_file
from dtlr_poc.tokenizer import (
    FROZEN_IAM_MANIFEST_SHA256, FROZEN_IAM_SCORES_SHA256, HandwritingBigramTokenizer,
)

FROZEN_ARTIFACT_SHA256 = "2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iam-train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-token-length", type=int, default=6)
    parser.add_argument("--minimum-count", type=int, default=20)
    parser.add_argument("--rate-threshold", type=float, default=0.5)
    parser.add_argument("--subword-budget", type=int, default=145)
    parser.add_argument("--em-iterations", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if REPO not in output.parents:
        raise ValueError("outputs must stay within DTLR")
    if not output.is_relative_to(REPO / "outputs"):
        raise ValueError("write generated models/reports only under ignored DTLR/outputs")
    source = args.iam_train_dir
    paths = {"manifest": source / "bigrams-dominant-core-v3/manifest.json",
             "scores": source / "bigrams-dominant-core-v3/bigram_scores.json",
             "evidence": source / "bigrams-dominant-core-v3/bigram_evidence.json",
             "detections": source / "detections.jsonl", "selection": source / "selection.json",
             "frozen_bigram": REPO / "poc/frozen/iam-english-handwriting-bigram-v1.json"}
    hashes = {}
    for key, expected in (("manifest", FROZEN_IAM_MANIFEST_SHA256),
                          ("scores", FROZEN_IAM_SCORES_SHA256),
                          ("frozen_bigram", FROZEN_ARTIFACT_SHA256)):
        hashes[key] = sha256_file(paths[key])
        if hashes[key] != expected:
            raise ValueError(f"pinned {key} SHA-256 differs")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    if Path(manifest["source"]).resolve() != paths["detections"].resolve():
        raise ValueError("detection input differs from pinned manifest source")
    print("Reading and validating frozen IAM train occurrence evidence...", flush=True)
    records = [json.loads(line) for line in paths["detections"].read_text(encoding="utf-8").splitlines() if line]
    rows = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    corpus = prepare_corpus(records, rows, selection, manifest)
    # Reconstruct each primary aggregate to bind the unpinned row export to pinned scores.
    counts = {}
    for row in rows:
        counts.setdefault(row["pair"], [0, 0, 0])
        item = counts[row["pair"]]
        item[0] += 1
        if row["usable"] and row["left_alignment"] == row["right_alignment"] == "match":
            item[1] += 1
            item[2] += int(row["connected"])
    scores = json.loads(paths["scores"].read_text(encoding="utf-8"))
    if len(scores) != manifest["score_count"] or set(counts) != {r["pair"] for r in scores}:
        raise ValueError("score inventory differs from occurrence evidence")
    for score in scores:
        total, exact, connected = counts[score["pair"]]
        rate = connected / exact if exact else None
        if (score["n_total"] != total or score["n_exact_alignment"] != exact
                or score["exact_alignment_connected_rate"] != rate):
            raise ValueError("primary aggregate does not reconstruct pinned score")
    for key in ("evidence", "detections", "selection"):
        hashes[key] = sha256_file(paths[key])
    del records, rows
    gc.collect()
    bigram_model = json.loads(paths["frozen_bigram"].read_text(encoding="utf-8"))
    observed_characters = {c for line in corpus for c in line["text"]}
    additions = sorted(observed_characters - set(bigram_model["vocab"]))
    alphabet_audit = {"frozen_ctc_class_count": bigram_model["size"],
                      "missing_characters": {c: sum(line["text"].count(c) for line in corpus) for c in additions},
                      "affected_lines": sum(any(c in line["text"] for c in additions) for line in corpus)}
    bigram_model = extend_bigram_training_alphabet(bigram_model, observed_characters)
    singletons = {t for t in bigram_model["vocab"] if len(t) == 1}
    candidates = collect_candidates(corpus, args.max_token_length, args.minimum_count, args.rate_threshold)
    print(f"Validated {len(corpus)} lines; {len(candidates)} eligible multi-character pieces.", flush=True)
    policy = {"max_token_length": args.max_token_length, "minimum_count": args.minimum_count,
              "rate_threshold": args.rate_threshold, "pair_policy": "letters-only",
              "span_method": "pairwise-consistent-dominant-component-v1",
              "threshold_status": "provisional inherited count/rate; new length cap is exploratory"}
    provenance = {"source_sha256": hashes, "source_paths": {k: str(p) for k, p in paths.items()},
                  "line_count": len(corpus), "sole_dataset": "IAM", "sole_split": "train",
                  "reference_singleton_count": len(singletons),
                  "source_repository_commit": subprocess.check_output(
                      ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                  "python_version": sys.version,
                  "implementation_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in (
                      REPO / "poc/dtlr_poc/handwriting_subwords.py", Path(__file__).resolve(),
                      REPO / "poc/dtlr_poc/tokenizer.py", REPO / "poc/dtlr_poc/selection.py")},
                  "implementation_status": "uncommitted prototype; source tree must accompany results"}
    print("Training handwriting-aware BPE...", flush=True)
    bpe = train_bpe(corpus, candidates, args.subword_budget, singletons)
    print("Training CCL-constrained Unigram EM...", flush=True)
    unigram = train_unigram(corpus, candidates, args.subword_budget, singletons, args.em_iterations)
    for model in (bpe, unigram):
        model["candidate_policy"] = policy
        model["evidence_provenance"] = provenance
    tokenizers = {"hw_bigram": HandwritingBigramTokenizer(bigram_model),
                  "hw_bpe": HandwritingSubwordTokenizer(bpe),
                  "hw_unigram": HandwritingSubwordTokenizer(unigram)}
    print("Comparing all tokenizers on IAM train (descriptive only)...", flush=True)
    comparison = compare_tokenizers(corpus, tokenizers)
    comparison["candidate_policy"], comparison["evidence_provenance"] = policy, provenance
    comparison["eligible_candidates_by_length"] = dict(sorted(Counter(map(len, candidates)).items()))
    comparison["requested_subword_budget"] = args.subword_budget
    comparison["frozen_alphabet_audit"] = alphabet_audit
    comparison["examples"] = []
    for text in ("the handwriting", "connected character components", "minimum evidence"):
        comparison["examples"].append({"text": text, "source": "illustrative non-evaluation text",
                                      "tokens": {name: tok.segment(text)["tokens"]
                                                 for name, tok in tokenizers.items()}})
    if any(r["round_trip_or_blank_failures"] for r in comparison["tokenizers"].values()):
        raise ValueError("round-trip or blank emission failures")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "hw-bigram-iam-train-alphabet.json", bigram_model)
    write_json(output / "hw-bpe.json", bpe)
    write_json(output / "hw-unigram.json", unigram)
    write_json(output / "candidates.json", candidates)
    write_json(output / "comparison.json", comparison)
    lines = ["# Experimental IAM handwriting-aware subwords", "",
             "Training-set descriptive results; no held-out evaluation or recognition model training.", "",
             "| Tokenizer | CTC classes | Tokens | Reduction vs characters | Connected edges cut | Multi-token spans sharing one component |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in comparison["tokenizers"].items():
        lines.append(f"| {name} | {metrics['ctc_class_count']} | {metrics['token_count']} | "
                     f"{metrics['token_reduction_fraction']:.2%} | "
                     f"{percent(metrics['connected_edge_cut_fraction'])} | "
                     f"{percent(metrics['multi_token_same_component_fraction'])} |")
    lines += ["", f"IAM train lines: {len(corpus)}. Characters: {sum(len(l['text']) for l in corpus)}.",
              f"Eligible candidates: {len(candidates)}. Requested subword budget: {args.subword_budget}.",
              "Actual class counts can differ when eligible BPE merge paths run out.", "",
              "The same IAM-train singleton alphabet and blank ID 0 are shared across models. The bigram model is a derived IAM-train alphabet copy; the frozen artifact is unchanged.",
              "Missing/non-exact evidence is excluded from connectivity denominators, never labelled disconnected.",
              "Long spans use a new conservative extension of pair-facing v3 cores, not a validated span detector.",
              "Count 20, rate 0.5, length cap and training settings are exploratory, not optimality claims.",
              "Unigram uses exact EM and individual fixed-weight removal-loss pruning; batch interactions are omitted. This custom implementation is not SentencePiece.",
              "Inference uses learned vocabularies/merge rules and text, without requiring images.",
              "No ranking of recognition accuracy can be inferred from these measurements.", "",
              "## Illustrative segmentations", ""]
    for example in comparison["examples"]:
        lines.append(f"Text: `{example['text']}`")
        lines.extend(f"- {name}: `{tokens!r}`" for name, tokens in example["tokens"].items())
        lines.append("")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison["tokenizers"], indent=2))
    print(f"Outputs: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
