from collections import deque

import numpy as np


def otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    total = values.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg = weight_bg = 0.0
    best_variance, best = -1.0, 127
    for threshold in range(256):
        weight_bg += hist[threshold]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += threshold * hist[threshold]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance, best = variance, threshold
    return best


def label_ink(gray: np.ndarray, threshold: int | None = None) -> tuple[np.ndarray, int]:
    """Return 8-connected labels for dark ink on a light background."""
    gray = np.asarray(gray)
    if gray.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    threshold = otsu_threshold(gray) if threshold is None else threshold
    ink = gray <= threshold
    labels = np.zeros(ink.shape, dtype=np.int32)
    next_label = 0
    height, width = ink.shape
    for y, x in zip(*np.nonzero(ink & (labels == 0))):
        if labels[y, x]:
            continue
        next_label += 1
        labels[y, x] = next_label
        queue = deque([(y, x)])
        while queue:
            cy, cx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if (dy or dx) and 0 <= ny < height and 0 <= nx < width:
                        if ink[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = next_label
                            queue.append((ny, nx))
    return labels, next_label


def component_ids_in_box(labels: np.ndarray, box_xyxy: list[float]) -> set[int]:
    height, width = labels.shape
    x0, y0, x1, y1 = box_xyxy
    left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    if right <= left or bottom <= top:
        return set()
    ids = set(np.unique(labels[top:bottom, left:right]).tolist())
    ids.discard(0)
    return ids


def component_pixel_counts_in_box(
    labels: np.ndarray, box_xyxy: list[float]
) -> dict[int, int]:
    """Count pixels from each non-background component inside a box."""
    height, width = labels.shape
    x0, y0, x1, y1 = box_xyxy
    left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
    if right <= left or bottom <= top:
        return {}
    ids, counts = np.unique(labels[top:bottom, left:right], return_counts=True)
    return {int(component): int(count) for component, count in zip(ids, counts) if component != 0}


def unique_dominant_component(counts: dict[int, int]) -> int | None:
    """Return the unique largest component, or None for empty/tied support."""
    if not counts:
        return None
    largest = max(counts.values())
    winners = [component for component, count in counts.items() if count == largest]
    return winners[0] if len(winners) == 1 else None


def core_unusable_reasons(
    side: str,
    box: list[float],
    counts: dict[int, int],
    dominant: int | None,
) -> list[str]:
    """Describe an unusable core mechanically, without guessing visual cause."""
    x0, y0, x1, y1 = box
    if x1 < x0 or y1 < y0:
        return [f"{side}-core-inverted"]
    if x1 == x0 or y1 == y0:
        return [f"{side}-core-collapsed"]
    if not counts:
        return [f"{side}-core-no-ink"]
    if dominant is None:
        return [f"{side}-core-dominant-tie"]
    return []


def exclusive_core_boxes_v2(
    left_box: list[float], right_box: list[float]
) -> tuple[list[float], list[float]]:
    """Original float-core definition retained to reproduce v2 outputs."""
    lx0, ly0, lx1, ly1 = left_box
    rx0, ry0, rx1, ry1 = right_box
    return [lx0, ly0, min(lx1, rx0), ly1], [max(rx0, lx1), ry0, rx1, ry1]


def raster_safe_exclusive_core_boxes(
    left_box: list[float], right_box: list[float]
) -> tuple[list[float], list[float]]:
    """Return cores whose integer pixel slices cannot overlap.

    Full boxes are rasterized outward elsewhere. Core-facing boundaries must be
    rasterized inward: floor the left core's right edge and ceil the right
    core's left edge. Otherwise distinct fractional boundaries can both include
    the same pixel column.
    """
    lx0, ly0, lx1, ly1 = left_box
    rx0, ry0, rx1, ry1 = right_box
    left_inner_edge = int(np.floor(min(lx1, rx0)))
    right_inner_edge = int(np.ceil(max(rx0, lx1)))
    return [lx0, ly0, left_inner_edge, ly1], [right_inner_edge, ry0, rx1, ry1]


def pair_component_evidence(
    labels: np.ndarray, left_box: list[float], right_box: list[float]
) -> dict:
    """Compare rejected full-box connectivity with exclusive-core connectivity.

    Earlier region-intersection results are retained as rejected provenance. The
    v3 candidate requires the same component to have unique largest pixel
    support in both raster-safe character cores.
    """
    left_full = component_ids_in_box(labels, left_box)
    right_full = component_ids_in_box(labels, right_box)
    shared_full = left_full & right_full
    left_core_box_v2, right_core_box_v2 = exclusive_core_boxes_v2(left_box, right_box)
    left_core_v2 = component_ids_in_box(labels, left_core_box_v2)
    right_core_v2 = component_ids_in_box(labels, right_core_box_v2)
    shared_core_v2 = left_core_v2 & right_core_v2
    core_usable_v2 = bool(left_core_v2) and bool(right_core_v2)
    left_core_box, right_core_box = raster_safe_exclusive_core_boxes(left_box, right_box)
    left_core = component_ids_in_box(labels, left_core_box)
    right_core = component_ids_in_box(labels, right_core_box)
    shared_core = left_core & right_core
    core_usable = bool(left_core) and bool(right_core)
    left_core_counts = component_pixel_counts_in_box(labels, left_core_box)
    right_core_counts = component_pixel_counts_in_box(labels, right_core_box)
    left_dominant = unique_dominant_component(left_core_counts)
    right_dominant = unique_dominant_component(right_core_counts)
    unusable_reason_codes = (
        core_unusable_reasons("left", left_core_box, left_core_counts, left_dominant)
        + core_unusable_reasons("right", right_core_box, right_core_counts, right_dominant)
    )
    dominant_usable = left_dominant is not None and right_dominant is not None
    return {
        "left_full_components": left_full,
        "right_full_components": right_full,
        "shared_full_components": shared_full,
        "left_core_box_v2": left_core_box_v2,
        "right_core_box_v2": right_core_box_v2,
        "left_core_components_v2": left_core_v2,
        "right_core_components_v2": right_core_v2,
        "shared_core_components_v2": shared_core_v2,
        "exclusive_core_usable_v2": core_usable_v2,
        "left_core_box": left_core_box,
        "right_core_box": right_core_box,
        "left_core_components": left_core,
        "right_core_components": right_core,
        "shared_core_components": shared_core,
        "exclusive_core_usable": core_usable,
        "left_core_component_pixel_counts": left_core_counts,
        "right_core_component_pixel_counts": right_core_counts,
        "left_dominant_component": left_dominant,
        "right_dominant_component": right_dominant,
        "left_dominant_share": (
            left_core_counts[left_dominant] / sum(left_core_counts.values())
            if left_dominant is not None else None
        ),
        "right_dominant_share": (
            right_core_counts[right_dominant] / sum(right_core_counts.values())
            if right_dominant is not None else None
        ),
        "dominant_core_usable": dominant_usable,
        "unusable_reason_codes": unusable_reason_codes,
        "connected_box_intersection_v1": bool(shared_full),
        "connected_exclusive_core_v2": bool(shared_core_v2) if core_usable_v2 else None,
        "connected_exclusive_core_v2_1": bool(shared_core) if core_usable else None,
        "connected_dominant_core_v3": (
            left_dominant == right_dominant if dominant_usable else None
        ),
    }
