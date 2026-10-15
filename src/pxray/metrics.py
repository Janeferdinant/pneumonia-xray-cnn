"""Classification metrics beyond accuracy."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes present")
    ranks = rankdata(score)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_matrix(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp)."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return tn, fp, fn, tp


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y, (np.asarray(prob) >= threshold).astype(int))
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * prec * sens, prec + sens)
    return {
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_for_specificity(y: np.ndarray, score: np.ndarray, specificity: float) -> float:
    """Smallest threshold (predict positive if score >= thr) with specificity >= target."""
    y = np.asarray(y)
    neg = np.sort(np.asarray(score)[y == 0])
    n = len(neg)
    allowed_fp = int(np.floor((1.0 - specificity) * n + 1e-9))
    if allowed_fp >= n:
        return float(neg[0])
    # threshold just above the (allowed_fp+1)-th highest negative score
    cutoff = neg[n - allowed_fp - 1]
    above = np.asarray(score)[np.asarray(score) > cutoff]
    return float(above.min()) if above.size else float(cutoff) + 1e-12


def sensitivity_at_specificity(y: np.ndarray, score: np.ndarray, specificity: float) -> float:
    """Highest ROC-curve sensitivity with specificity >= target (no interpolation)."""
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    ok = fpr <= (1.0 - specificity) + 1e-12
    return float(tpr[ok].max())


def sens_spec_at_threshold(y: np.ndarray, score: np.ndarray, thr: float) -> tuple[float, float]:
    pred = np.asarray(score) >= thr
    y = np.asarray(y)
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())
