from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Alignment:
    gt_index: Optional[int]
    detection_index: Optional[int]
    operation: str


def align_monotonic(ground_truth: str, detected: Sequence[str]) -> list[Alignment]:
    """Levenshtein-align ordered DTLR labels to transcription characters.

    Ties prefer a diagonal match/substitution, then a missing detection, then an
    extra detection. This is a localization bridge, not a correction to DTLR.
    """
    m, n = len(ground_truth), len(detected)
    costs = [[0] * (n + 1) for _ in range(m + 1)]
    back: list[list[Optional[str]]] = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        costs[i][0], back[i][0] = i, "delete"
    for j in range(1, n + 1):
        costs[0][j], back[0][j] = j, "insert"
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            op = "match" if ground_truth[i - 1] == detected[j - 1] else "substitute"
            options = [
                (costs[i - 1][j - 1] + (op == "substitute"), 0, op),
                (costs[i - 1][j] + 1, 1, "delete"),
                (costs[i][j - 1] + 1, 2, "insert"),
            ]
            costs[i][j], _, back[i][j] = min(options)

    result: list[Alignment] = []
    i, j = m, n
    while i or j:
        op = back[i][j]
        if op in ("match", "substitute"):
            result.append(Alignment(i - 1, j - 1, op))
            i, j = i - 1, j - 1
        elif op == "delete":
            result.append(Alignment(i - 1, None, "missing_detection"))
            i -= 1
        elif op == "insert":
            result.append(Alignment(None, j - 1, "extra_detection"))
            j -= 1
        else:
            raise RuntimeError("alignment backtrace is incomplete")
    return list(reversed(result))


def gt_detection_map(ground_truth: str, detected: Sequence[str]) -> dict[int, Alignment]:
    return {
        item.gt_index: item
        for item in align_monotonic(ground_truth, detected)
        if item.gt_index is not None
    }
