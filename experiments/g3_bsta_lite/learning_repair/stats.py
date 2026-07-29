"""R2/R3 statistics — paired LCB95 and Spearman correlation.

PREREGISTRATION §9 requires for every claimed metric:
  - within-scenario aggregation across action reps first;
  - paired delta across policies on the same scenario;
  - mean, SE, LCB95 reported;
  - raw per-replicate rows saved;
  - no test-set checkpoint selection;
  - no pseudo-replication.

The functions here are pure and have unit-style assertions in
test_learning_repair_r2.py.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def _t_critical_95_one_sided(df: int) -> float:
    """One-sided 95% Student-t critical value via approximation.

    For df >= 30 we use the normal approximation (1.645). For smaller df
    we use a small lookup table; this is a self-contained impl with no
    scipy dependency. Values are conservative (rounded up).
    """
    if df <= 0:
        return float("inf")
    if df >= 30:
        return 1.645
    table = {
        1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476,
        6: 1.440, 7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372,
        15: 1.341, 20: 1.325, 25: 1.316, 29: 1.311,
    }
    return table.get(df, 1.645)


def paired_delta_stats(
    per_seed_a: Sequence[float],
    per_seed_b: Sequence[float],
) -> dict:
    """Paired delta a − b per seed. Returns mean, SE, LCB95 (one-sided),
    point improvement, and the raw deltas.

    per_seed_a / per_seed_b must be aligned: same seed order, each value
    already aggregated across action reps.
    """
    n = len(per_seed_a)
    if n != len(per_seed_b):
        raise ValueError(
            f"paired mismatch: len(a)={len(per_seed_a)}, len(b)={len(per_seed_b)}"
        )
    deltas = [float(a) - float(b) for a, b in zip(per_seed_a, per_seed_b)]
    if n == 0:
        return {"mean": float("nan"), "se": float("nan"), "lcb95": float("nan"),
                "n": 0, "point_pp": float("nan"), "deltas": []}
    mean = sum(deltas) / float(n)
    if n >= 2:
        var = sum((d - mean) ** 2 for d in deltas) / float(n - 1)
        se = math.sqrt(var / float(n))
    else:
        se = float("inf")
    tcrit = _t_critical_95_one_sided(n - 1)
    lcb95 = mean - tcrit * se
    return {
        "mean": mean, "se": se, "lcb95": lcb95, "n": n,
        "point_pp": mean * 100.0, "deltas": deltas,
    }


def spearman_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation. Returns NaN if either sequence is constant.

    Pure-python implementation: rank, then Pearson on ranks.
    """
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")

    def rank(seq):
        order = sorted(range(n), key=lambda i: seq[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and seq[order[j + 1]] == seq[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0   # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = rank(list(x))
    ry = rank(list(y))
    mx = sum(rx) / float(n)
    my = sum(ry) / float(n)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)
