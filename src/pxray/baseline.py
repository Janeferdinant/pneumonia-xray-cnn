"""Classical baseline: logistic regression on block-averaged pixels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import Split
from .metrics import roc_auc

CANDIDATE_C = (0.0005, 0.005, 0.05, 0.5)


def _pool(images: np.ndarray, k: int) -> np.ndarray:
    n, h, w = images.shape
    h2, w2 = h // k, w // k
    x = images[:, : h2 * k, : w2 * k].astype(np.float32) / 255.0
    return x.reshape(n, h2, k, w2, k).mean(axis=(2, 4)).reshape(n, -1)


def fit_logreg(
    splits: dict[str, Split],
    pool: int = 4,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Fit on train, choose C by validation AUC, return decision values for val and test.

    `transform` (for example a border mask) is applied to the images of every split first.
    """
    feats = {}
    for name in ("train", "val", "test"):
        imgs = splits[name].images
        feats[name] = _pool(transform(imgs) if transform else imgs, pool)
    mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-6
    feats = {k: (v - mu) / sd for k, v in feats.items()}
    best = None
    for c in CANDIDATE_C:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
        clf.fit(feats["train"], splits["train"].labels)
        val = clf.decision_function(feats["val"])
        auc = roc_auc(splits["val"].labels, val)
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    assert best is not None
    _, c, clf = best
    return {
        "C": c,
        "val_logits": clf.decision_function(feats["val"]).astype(np.float64),
        "test_logits": clf.decision_function(feats["test"]).astype(np.float64),
    }
