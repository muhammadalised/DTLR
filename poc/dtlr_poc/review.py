"""Deterministic construction of a manual QA review queue."""

import hashlib


def pair_id(line_id: str, pair: dict) -> str:
    return f"{line_id}:{pair['left_gt_index']}:{pair['right_gt_index']}"


def pair_rank(seed: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def build_review_rows(lines: list[dict], agreement_audit_count: int, seed: str) -> list[dict]:
    if agreement_audit_count < 0:
        raise ValueError("agreement_audit_count cannot be negative")
    mandatory = []
    agreements = []
    for line in lines:
        for pair in line["pairs"]:
            identifier = pair_id(line["line_id"], pair)
            if not pair["usable"]:
                group = "unusable"
            elif pair["connected_exclusive_core_v2_1"] != pair["connected_dominant_core_v3"]:
                group = "v2.1-v3-disagreement"
            else:
                group = "agreement-audit"
            row = {
                "pair_id": identifier,
                "queue_group": group,
                "selection_rank_sha256": pair_rank(seed, identifier),
                "line_id": line["line_id"],
                **pair,
            }
            (agreements if group == "agreement-audit" else mandatory).append(row)

    if agreement_audit_count > len(agreements):
        raise ValueError(
            f"agreement audit count {agreement_audit_count} exceeds pool size {len(agreements)}"
        )
    group_order = {"v2.1-v3-disagreement": 0, "unusable": 1}
    mandatory.sort(key=lambda row: (group_order[row["queue_group"]], row["pair_id"]))
    audit = sorted(agreements, key=lambda row: (row["selection_rank_sha256"], row["pair_id"]))[
        :agreement_audit_count
    ]
    return mandatory + audit
