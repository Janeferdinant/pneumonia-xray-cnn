from pathlib import Path

import numpy as np
import pytest

from pxray.data import (
    DATASETS,
    DataIntegrityError,
    Split,
    Standardizer,
    check_splits,
    class_counts,
    load_splits,
    pos_weight,
)

from .conftest import make_toy_splits


def test_clean_splits_pass_and_report_counts(toy_splits):
    rep = check_splits(toy_splits)
    assert rep["sizes"] == {"train": 60, "val": 20, "test": 30}
    for name, s in toy_splits.items():
        assert rep["label_counts"][name] == class_counts(s.labels)
        assert sum(rep["label_counts"][name].values()) == len(s.labels)
    assert rep["cross_split_duplicates"] == 0


def test_leakage_train_into_test_is_detected(toy_splits):
    toy_splits["test"].images[3] = toy_splits["train"].images[5].copy()
    with pytest.raises(DataIntegrityError, match="leak"):
        check_splits(toy_splits)


def test_leakage_train_into_val_is_detected(toy_splits):
    toy_splits["val"].images[0] = toy_splits["train"].images[0].copy()
    with pytest.raises(DataIntegrityError, match="leak"):
        check_splits(toy_splits)


def test_non_binary_labels_rejected(toy_splits):
    toy_splits["train"].labels[0] = 2
    with pytest.raises(DataIntegrityError, match="binary"):
        check_splits(toy_splits)


def test_missing_split_and_shape_mismatch_rejected(toy_splits):
    with pytest.raises(DataIntegrityError, match="splits"):
        check_splits({"train": toy_splits["train"]})
    bad = dict(toy_splits)
    bad["val"] = Split(images=toy_splits["val"].images[:, :16, :16], labels=toy_splits["val"].labels)
    with pytest.raises(DataIntegrityError, match="shape"):
        check_splits(bad)


def test_images_labels_length_mismatch_rejected():
    with pytest.raises(DataIntegrityError, match="length"):
        Split(images=np.zeros((3, 4, 4), np.uint8), labels=np.zeros(2, np.int64))


def test_within_split_duplicates_are_reported_not_fatal(toy_splits):
    toy_splits["train"].images[1] = toy_splits["train"].images[0].copy()
    rep = check_splits(toy_splits)
    assert rep["within_split_duplicates"]["train"] >= 1


def test_pos_weight_is_negatives_over_positives():
    y = np.array([0, 0, 0, 1])
    assert class_counts(y) == {0: 3, 1: 1}
    assert pos_weight(y) == pytest.approx(3.0)


def test_standardizer_uses_train_statistics_only(toy_splits):
    st = Standardizer.fit(toy_splits["train"].images)
    z = st.transform(toy_splits["train"].images)
    assert z.shape == (60, 1, 32, 32)
    assert abs(float(z.mean())) < 1e-4
    assert float(z.std()) == pytest.approx(1.0, abs=1e-3)
    other = make_toy_splits(seed=9)
    # fitting on another split gives different numbers, i.e. stats are data dependent
    assert Standardizer.fit(other["test"].images).mean != st.mean


@pytest.mark.data
def test_real_pneumoniamnist_official_split_integrity():
    root = Path("data")
    if not (root / "pneumoniamnist_128.npz").exists():
        pytest.skip("dataset not downloaded; run `pxray download`")
    splits = load_splits(root, 128)
    rep = check_splits(splits)
    assert rep["sizes"] == {"train": 4708, "val": 524, "test": 624}
    assert rep["label_counts"]["test"] == {0: 234, 1: 390}
    assert rep["label_counts"]["train"] == {0: 1214, 1: 3494}
    assert rep["cross_split_duplicates"] == 0
    assert DATASETS[128]["md5"] == "05b46931834c231683c68f40c47b2971"
