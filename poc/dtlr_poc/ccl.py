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
