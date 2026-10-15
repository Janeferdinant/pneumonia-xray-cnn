import numpy as np

from pxray.explain import compute_cams, outcome_of, select_cases
from pxray.models import build_model
from pxray.report import seed_metrics


def test_outcomes_and_case_selection_rules():
    y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    logit = np.array([3.0, -2.0, -0.5, -1.0, 4.0, 1.5, 0.2, -3.0])
    assert list(outcome_of(y, logit)) == ["TP", "FN", "FN", "TN", "FP", "FP", "TP", "TN"]
    sel = select_cases(y, logit, per_row=2)
    # errors are the most confident ones
    assert sel["FN"] == [1, 2]
    assert sel["FP"] == [4, 5]
    assert set(sel["TP"]) <= {0, 6} and set(sel["TN"]) <= {3, 7}
    assert select_cases(y, logit, per_row=2) == sel  # deterministic


def test_compute_cams_batches_match_single_pass():
    import torch

    torch.manual_seed(0)
    model = build_model("small_cnn", pretrained=False).eval()
    x = torch.randn(5, 1, 32, 32)
    cams, logits = compute_cams(model, x, batch=2)
    one, _ = compute_cams(model, x, batch=5)
    assert cams.shape == (5, 32, 32) and logits.shape == (5,)
    assert np.allclose(cams, one, atol=1e-5)


def test_seed_metrics_has_expected_keys_and_ranges():
    rng = np.random.default_rng(0)
    yv, yt = rng.integers(0, 2, 200), rng.integers(0, 2, 300)
    lv, lt = rng.normal(size=200) + yv, rng.normal(size=300) + yt
    m = seed_metrics(yv, lv, yt, lt)
    for k in ("auc", "sens_at_spec90", "ece_raw", "ece_cal", "temperature", "valthr90_test_sensitivity"):
        assert k in m
    assert 0.5 < m["auc"] < 1.0 and m["temperature"] > 0
