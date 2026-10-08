"""Deterministic, train-only diagnostic sampling for longer CCL spans."""
from collections import defaultdict
import hashlib

from .handwriting_subwords import span_connected


def span_group(line, start, end):
    value = span_connected(line, start, end)
    if value is None:
        return "abstained"
    if value:
        return "supported"
    edges = [line["edges"][i] for i in range(start, end - 1)]
    if all(left == right for left, right in edges):
        return "interior-component-disagreement"
    return "disconnected"


def select_span_review(corpus, tokens, seed, per_stratum=2, stress_count=4):
    """Hash sample by length/status, preferring different tokens and lines.

    The special interior-core disagreement group is sampled separately. This
    deliberately stratified pilot is not a population accuracy sample.
    """
    if not seed or type(per_stratum) is not int or per_stratum < 1:
        raise ValueError("a seed and positive per-stratum count are required")
    if type(stress_count) is not int or stress_count < 0:
        raise ValueError("stress_count must be nonnegative")
    if any(not 3 <= len(t) <= 6 or not t.isalpha() for t in tokens):
        raise ValueError("review tokens must be letter spans of length 3 through 6")
    pools = defaultdict(list)
    for line in corpus:
        for start in range(len(line["text"])):
            for length in range(3, 7):
                end = start + length
                token = line["text"][start:end]
                if len(token) != length or token not in tokens:
                    continue
                group = span_group(line, start, end)
                identifier = f"{line['line_id']}:{start}:{end}"
                rank = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
                pools[(length, group)].append({"span_id": identifier,
                    "line_id": line["line_id"], "start": start, "end": end,
                    "token": token, "group": group, "selection_rank_sha256": rank})
    selected = []
    for key, pool in sorted(pools.items()):
        if key[1] != "interior-component-disagreement":
            selected.extend(_diverse_sample(pool, per_stratum))
    stress = [row for (length, group), pool in pools.items()
              if group == "interior-component-disagreement" for row in pool]
    selected.extend(_diverse_sample(stress, stress_count))
    counts = {f"length-{length}/{group}": len(pool)
              for (length, group), pool in sorted(pools.items())}
    return selected, counts


def _diverse_sample(pool, count):
    ranked = sorted(pool, key=lambda row: (row["selection_rank_sha256"], row["span_id"]))
    chosen, used_tokens, used_lines = [], set(), set()
    for row in ranked:
        if len(chosen) >= count:
            break
        if row["token"] in used_tokens or row["line_id"] in used_lines:
            continue
        chosen.append(row)
        used_tokens.add(row["token"])
        used_lines.add(row["line_id"])
    selected_ids = {row["span_id"] for row in chosen}
    chosen.extend(row for row in ranked if row["span_id"] not in selected_ids)
    return chosen[:count]
