"""Shared relevance math for client and document search.

Each search maps its signals to calibrated confidences in [0, 1] and combines them with a
noisy-OR, `1 - prod(1 - c_i)`: one strong signal is enough, and agreeing signals reinforce each
other while the score stays in [0, 1], so clients and documents can share one ranked list.
"""


def clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def noisy_or(confidences: list[float]) -> float:
    remaining = 1.0
    for c in confidences:
        remaining *= 1.0 - clip01(c)
    return 1.0 - remaining
