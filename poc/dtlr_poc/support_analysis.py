"""Exploratory scoring helpers for shared-component support analysis."""

from collections import Counter

import numpy as np


def shared_component_support(evidence: dict) -> dict:
    """Return the shared component with strongest bidirectional core support.

    The score is the smaller of the component's shares in the two exclusive
    cores. It is high only when one physical component has substantial support
    on both sides. This reports a feature; it does not modify v3.
    """
    left_counts = evidence["left_core_component_pixel_counts"]
    right_counts = evidence["right_core_component_pixel_counts"]
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    candidates = []
    for component in evidence["shared_core_components"]:
        left_share = left_counts[component] / left_total if left_total else 0.0
        right_share = right_counts[component] / right_total if right_total else 0.0
        candidates.append({
            "component_id": int(component),
            "left_share": left_share,
            "right_share": right_share,
            "bidirectional_support": min(left_share, right_share),
        })
    if not candidates:
        return {
            "component_id": None,
            "left_share": 0.0,
            "right_share": 0.0,
            "bidirectional_support": 0.0,
        }
    return max(
        candidates,
        key=lambda item: (
            item["bidirectional_support"],
            item["left_share"] + item["right_share"],
            -item["component_id"],
        ),
    )


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minimum": None, "q1": None, "median": None, "q3": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q3": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def threshold_candidates(rows: list[dict]) -> list[dict]:
    """Evaluate score cutoffs on manually evaluable development rows."""
    if not rows:
        return []
    thresholds = sorted({0.0, *(float(row["bidirectional_support"]) for row in rows)})
    maximum = max(thresholds)
    thresholds.append(float(np.nextafter(maximum, np.inf)))
    output = []
    for threshold in thresholds:
        counts = Counter()
        for row in rows:
            predicted = row["bidirectional_support"] >= threshold
            actual = row["manual_visual_connectivity"] == "connected"
            counts["tp" if predicted and actual else "fp" if predicted else "fn" if actual else "tn"] += 1
        tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        output.append({
            "threshold": threshold,
            "true_connected": tp,
            "false_connected": fp,
            "true_disconnected": tn,
            "false_disconnected": fn,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        })
    return output


def best_exploratory_candidate(candidates: list[dict]) -> dict | None:
    """Select one diagnostic cutoff deterministically; this does not freeze it."""
    if not candidates:
        return None
    return max(candidates, key=lambda row: (
        row["balanced_accuracy"] if row["balanced_accuracy"] is not None else -1.0,
        row["f1"],
        row["threshold"],
    ))
