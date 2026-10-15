"""Paired bootstrap over test images.

Each replicate resamples test images with replacement once and applies the same resample to
every model and every seed, so differences between models are paired. The reported statistic for
a model is the mean over its seeds, so the interval reflects test-image sampling noise of the
seed-averaged metric (it does not include training-seed variance; that is reported separately).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def paired_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    differences: list[tuple[str, str]] | None = None,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return {model: {metric: (point, lo, hi)}} plus "a - b" entries for each difference."""
    y = np.asarray(y)
    n = len(y)
    scores = {k: np.atleast_2d(np.asarray(v)) for k, v in scores.items()}
    rng = np.random.default_rng(seed)

    def stat(idx: np.ndarray, model: str, name: str) -> float:
        fn = metrics[name]
        return float(np.mean([fn(y[idx], s[idx]) for s in scores[model]]))

    full = np.arange(n)
    point = {m: {k: stat(full, m, k) for k in metrics} for m in scores}
    reps: dict[str, dict[str, list[float]]] = {m: {k: [] for k in metrics} for m in scores}
    for _ in range(n_boot):
        while True:
            idx = rng.integers(0, n, n)
            if 0 < y[idx].sum() < n:
                break
        for m in scores:
            for k in metrics:
                reps[m][k].append(stat(idx, m, k))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for m in scores:
        out[m] = {
            k: (point[m][k], float(np.percentile(reps[m][k], lo_q)), float(np.percentile(reps[m][k], hi_q)))
            for k in metrics
        }
    for a, b in differences or []:
        d = {}
        for k in metrics:
            diff = np.array(reps[a][k]) - np.array(reps[b][k])
            d[k] = (
                point[a][k] - point[b][k],
                float(np.percentile(diff, lo_q)),
                float(np.percentile(diff, hi_q)),
            )
        out[f"{a} - {b}"] = d
    return out
