"""Calibration: reliability curve, ECE, temperature scaling (fit on validation only)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: bins by confidence of the predicted class, equal-width bins."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= 0.5).astype(int)
    conf = np.where(pred == 1, prob, 1.0 - prob)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence of the predicted class is >= 0.5
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(
    y: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive-class reliability: (mean predicted prob, observed positive rate, count) per bin."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if m.any():
            conf[b] = prob[m].mean()
            acc[b] = y[m].mean()
    return conf, acc, cnt


def _nll(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    z = logits / t
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> float:
    """Single scalar T > 0 minimising validation NLL (never touches test data)."""
    logits = np.asarray(val_logits, dtype=np.float64)
    y = np.asarray(val_y, dtype=np.float64)
    res = minimize_scalar(
        lambda lt: _nll(logits, y, float(np.exp(lt))),
        bounds=(np.log(0.05), np.log(50.0)),
        method="bounded",
    )
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / t


def nll(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2))
