import numpy as np

from pxray.baseline import fit_logreg
from pxray.cli import build_parser
from pxray.metrics import roc_auc

from .conftest import make_toy_splits


def test_logreg_baseline_learns_toy_task_and_is_deterministic():
    s = make_toy_splits(sizes=(200, 60, 80))
    a = fit_logreg(s, pool=4)
    b = fit_logreg(s, pool=4)
    assert np.array_equal(a["test_logits"], b["test_logits"])
    assert roc_auc(s["test"].labels, a["test_logits"]) > 0.8


def test_cli_parser_has_documented_commands():
    p = build_parser()
    for cmd in ("download", "check-data", "train", "evaluate", "explain", "export-demo"):
        assert p.parse_args([cmd]).command == cmd
