import numpy as np
import pytest

from pxray.bootstrap import paired_bootstrap
from pxray.metrics import roc_auc


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = rng.normal(size=(3, n)) + 1.5 * y
    bad = rng.normal(size=(3, n)) + 0.2 * y
    return y, good, bad


def test_ci_contains_point_estimate_and_is_reproducible():
    y, good, bad = _data()
    kw = {"y": y, "scores": {"good": good, "bad": bad}, "metrics": {"auc": roc_auc},
          "n_boot": 200, "seed": 5}
    a = paired_bootstrap(**kw)
    b = paired_bootstrap(**kw)
    assert a == b
    pt, lo, hi = a["good"]["auc"]
    assert lo <= pt <= hi
    assert pt == pytest.approx(np.mean([roc_auc(y, s) for s in good]))
    assert a["good"]["auc"][0] > a["bad"]["auc"][0]


def test_paired_difference_ci_excludes_zero_for_clear_gap():
    y, good, bad = _data()
    res = paired_bootstrap(
        y, {"good": good, "bad": bad}, {"auc": roc_auc}, n_boot=300, seed=1,
        differences=[("good", "bad")],
    )
    pt, lo, hi = res["good - bad"]["auc"]
    assert lo > 0 and pt > 0
