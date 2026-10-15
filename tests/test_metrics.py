import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pxray.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)
from pxray.metrics import (
    binary_metrics,
    confusion_matrix,
    roc_auc,
    sensitivity_at_specificity,
    threshold_for_specificity,
)


def test_auc_perfect_reversed_and_ties():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + 0.8 * y
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        roc_auc(np.ones(5, int), np.arange(5.0))


def test_confusion_and_binary_metrics_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.1, 0.2, 0.3, 0.4])
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int))
    assert (tn, fp, fn, tp) == (5, 1, 1, 3)
    m = binary_metrics(y, p, threshold=0.5)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(5 / 6)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)


def test_binary_metrics_no_positive_predictions_is_zero_not_nan():
    m = binary_metrics(np.array([1, 0]), np.array([0.1, 0.1]), 0.5)
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_sensitivity_at_fixed_specificity():
    y = np.array([0] * 10 + [1] * 10)
    s = np.concatenate([np.arange(10) / 10, np.arange(5, 15) / 10])
    sens = sensitivity_at_specificity(y, s, 0.9)
    thr = threshold_for_specificity(y, s, 0.9)
    pred = s >= thr
    assert pred[y == 0].mean() <= 0.1 + 1e-9
    assert sens == pytest.approx(pred[y == 1].mean())
    assert 0.0 <= sens <= 1.0
    assert sensitivity_at_specificity(y, s, 1.0) <= sens


def test_ece_zero_for_perfectly_calibrated_and_known_value_for_overconfident():
    y = np.array([1, 0, 1, 0])
    assert expected_calibration_error(y, np.array([1.0, 0.0, 1.0, 0.0])) == pytest.approx(0.0)
    # always predict class 1 with p=0.9, but only 50% are positive -> |0.9 - 0.5| = 0.4
    y = np.array([1, 0] * 50)
    assert expected_calibration_error(y, np.full(100, 0.9), n_bins=10) == pytest.approx(0.4)
    # calibrated by construction: p=0.8 with 80% positives
    y = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(y, np.full(10, 0.8), n_bins=10) == pytest.approx(0.0)


def test_reliability_curve_shapes_and_counts():
    rng = np.random.default_rng(1)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(int)
    conf, acc, cnt = reliability_curve(y, p, n_bins=10)
    assert conf.shape == acc.shape == cnt.shape == (10,)
    assert cnt.sum() == 200


def test_temperature_scaling_softens_overconfident_logits_and_keeps_ranking():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    logits = (2 * y - 1) * 1.0 + rng.normal(0, 1.5, 2000)
    over = logits * 4.0
    t = fit_temperature(over, y)
    assert t > 2.0
    cal = apply_temperature(over, t)
    p_raw = 1 / (1 + np.exp(-over))
    p_cal = 1 / (1 + np.exp(-cal))
    assert expected_calibration_error(y, p_cal) < expected_calibration_error(y, p_raw)
    assert roc_auc(y, cal) == pytest.approx(roc_auc(y, over))


def test_temperature_close_to_one_when_already_calibrated():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-logits))).astype(int)
    assert fit_temperature(logits, y) == pytest.approx(1.0, abs=0.1)
