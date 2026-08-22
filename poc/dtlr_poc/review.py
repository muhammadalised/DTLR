"""Deterministic construction of a manual QA review queue."""

import hashlib
from collections import Counter


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


def summarize_review(queue: dict, review: dict) -> dict:
    if review.get("schema_version") != "dtlr.qa-review.v1":
        raise ValueError("unsupported manual review schema")
    if review.get("queue_schema_version") != queue.get("schema_version"):
        raise ValueError("manual review queue schema does not match")
    for key in ("qa_manifest_sha256", "seed"):
        if review.get(key) != queue.get(key):
            raise ValueError(f"manual review {key} does not match the frozen queue")

    annotations = review.get("annotations", {})
    known_ids = {row["pair_id"] for row in queue["rows"]}
    unknown_ids = sorted(set(annotations) - known_ids)
    if unknown_ids:
        raise ValueError(f"manual review contains unknown pair IDs: {unknown_ids}")

    required_fields = ("alignment", "visual_connectivity", "v3_assessment")
    incomplete = []
    consistency_issues = []
    reviewed = []
    for row in queue["rows"]:
        annotation = annotations.get(row["pair_id"], {})
        missing = [field for field in required_fields if annotation.get(field, "pending") == "pending"]
        if missing:
            incomplete.append({"pair_id": row["pair_id"], "pending_fields": missing})
            continue
        combined = {**row, "manual": annotation}
        reviewed.append(combined)
        visual = annotation["visual_connectivity"]
        assessment = annotation["v3_assessment"]
        if row["usable"] and assessment in ("appropriate-abstention", "unnecessary-abstention"):
            consistency_issues.append({"pair_id": row["pair_id"], "issue": "usable pair marked as abstention"})
        if not row["usable"] and assessment in ("correct", "incorrect"):
            consistency_issues.append({"pair_id": row["pair_id"], "issue": "unusable pair given binary v3 assessment"})
        if row["usable"] and visual in ("connected", "disconnected") and assessment in ("correct", "incorrect"):
            expected_correct = row["connected_dominant_core_v3"] == (visual == "connected")
            if (assessment == "correct") != expected_correct:
                consistency_issues.append({
                    "pair_id": row["pair_id"],
                    "issue": "v3 assessment conflicts with visual connectivity and v3 output",
                })

    groups = {}
    for group in sorted({row["queue_group"] for row in queue["rows"]}):
        group_rows = [row for row in reviewed if row["queue_group"] == group]
        groups[group] = {
            "queue_count": sum(row["queue_group"] == group for row in queue["rows"]),
            "completed_count": len(group_rows),
            "alignment_counts": dict(sorted(Counter(row["manual"]["alignment"] for row in group_rows).items())),
            "visual_connectivity_counts": dict(sorted(Counter(
                row["manual"]["visual_connectivity"] for row in group_rows
            ).items())),
            "v3_assessment_counts": dict(sorted(Counter(
                row["manual"]["v3_assessment"] for row in group_rows
            ).items())),
        }

    evaluable = [
        row for row in reviewed
        if row["usable"]
        and row["manual"]["alignment"] == "correct"
        and row["manual"]["visual_connectivity"] in ("connected", "disconnected")
        and row["manual"]["v3_assessment"] in ("correct", "incorrect")
    ]
    errors = [row for row in evaluable if row["manual"]["v3_assessment"] == "incorrect"]
    false_disconnected = [
        row for row in errors
        if not row["connected_dominant_core_v3"] and row["manual"]["visual_connectivity"] == "connected"
    ]
    false_connected = [
        row for row in errors
        if row["connected_dominant_core_v3"] and row["manual"]["visual_connectivity"] == "disconnected"
    ]
    return {
        "schema_version": "dtlr.qa-review-summary.v1",
        "queue_count": len(queue["rows"]),
        "completed_count": len(reviewed),
        "status": "complete" if not incomplete and not consistency_issues else "needs-attention",
        "incomplete": incomplete,
        "consistency_issues": consistency_issues,
        "group_summaries": groups,
        "alignment_failure_count": sum(row["manual"]["alignment"] == "incorrect" for row in reviewed),
        "evaluable_v3_count": len(evaluable),
        "observed_v3_correct_count": len(evaluable) - len(errors),
        "observed_v3_error_count": len(errors),
        "observed_false_disconnected_count": len(false_disconnected),
        "observed_false_connected_count": len(false_connected),
        "observed_error_pair_ids": [row["pair_id"] for row in errors],
        "note": (
            "The queue is stratified: all v2.1/v3 disagreements and unusable pairs, "
            "plus a hash-selected agreement audit. Raw reviewed accuracy is not a "
            "population accuracy estimate. Findings outside the frozen queue must be "
            "reported separately."
        ),
    }
