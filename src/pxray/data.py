"""PneumoniaMNIST (MedMNIST v2) download, loading and integrity checks.

Source: https://zenodo.org/records/10519652 (CC BY 4.0). The official train/val/test split
shipped in the file is used as is; nothing is re-split here.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ZENODO = "https://zenodo.org/records/10519652/files"
DATASETS: dict[int, dict[str, str]] = {
    64: {"url": f"{ZENODO}/pneumoniamnist_64.npz?download=1",
         "md5": "8f4eceb4ccffa70c672198ea285246c6"},
    128: {"url": f"{ZENODO}/pneumoniamnist_128.npz?download=1",
          "md5": "05b46931834c231683c68f40c47b2971"},
}
SPLITS = ("train", "val", "test")
LABEL_NAMES = {0: "normal", 1: "pneumonia"}


class DataIntegrityError(ValueError):
    """Raised when the splits violate the evaluation protocol (leakage, bad labels, shapes)."""


@dataclass
class Split:
    images: np.ndarray  # (N, H, W) uint8
    labels: np.ndarray  # (N,) int64

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise DataIntegrityError(
                f"length mismatch: {len(self.images)} images vs {len(self.labels)} labels"
            )


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity checksum published by the dataset host
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(root: str | Path, size: int) -> Path:
    return Path(root) / f"pneumoniamnist_{size}.npz"


def ensure_dataset(root: str | Path, size: int = 128) -> Path:
    """Download the npz when missing and verify its MD5 checksum."""
    if size not in DATASETS:
        raise ValueError(f"size must be one of {sorted(DATASETS)}")
    path = dataset_path(root, size)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.tmp")
        urllib.request.urlretrieve(DATASETS[size]["url"], tmp)  # noqa: S310 - fixed https URL
        tmp.replace(path)
    got = md5_file(path)
    if got != DATASETS[size]["md5"]:
        raise DataIntegrityError(f"checksum mismatch for {path.name}: {got}")
    return path


def load_splits(root: str | Path, size: int = 128) -> dict[str, Split]:
    path = ensure_dataset(root, size)
    z = np.load(path)
    return {
        s: Split(images=z[f"{s}_images"], labels=z[f"{s}_labels"].reshape(-1).astype(np.int64))
        for s in SPLITS
    }


def class_counts(labels: np.ndarray) -> dict[int, int]:
    return {c: int((labels == c).sum()) for c in (0, 1)}


def pos_weight(labels: np.ndarray) -> float:
    c = class_counts(labels)
    return c[0] / c[1]


def _hashes(images: np.ndarray) -> list[str]:
    return [hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images]  # noqa: S324


def check_splits(splits: dict[str, Split]) -> dict:
    """Validate split integrity; raise DataIntegrityError on any protocol violation."""
    if set(splits) != set(SPLITS):
        raise DataIntegrityError(f"expected splits {SPLITS}, got {sorted(splits)}")
    shapes = {s: splits[s].images.shape[1:] for s in SPLITS}
    if len(set(shapes.values())) != 1:
        raise DataIntegrityError(f"image shape differs between splits: {shapes}")
    for s in SPLITS:
        if not set(np.unique(splits[s].labels)) <= {0, 1}:
            raise DataIntegrityError(f"labels in {s} are not binary")
    hashes = {s: _hashes(splits[s].images) for s in SPLITS}
    cross = 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            cross += len(set(hashes[a]) & set(hashes[b]))
    if cross:
        raise DataIntegrityError(f"leakage: {cross} identical images appear in more than one split")
    return {
        "sizes": {s: len(splits[s].labels) for s in SPLITS},
        "image_shape": list(shapes["train"]),
        "label_counts": {s: class_counts(splits[s].labels) for s in SPLITS},
        "positive_rate": {s: float(splits[s].labels.mean()) for s in SPLITS},
        "cross_split_duplicates": cross,
        "within_split_duplicates": {s: len(hashes[s]) - len(set(hashes[s])) for s in SPLITS},
    }


def deduplicate_splits(splits: dict[str, Split]) -> tuple[dict[str, Split], dict[str, int]]:
    """Drop validation/test images that are byte-identical to an image in an earlier split.

    Order of precedence is train > val > test, so training data is never touched and the
    held-out test set only shrinks if it duplicates train or val. Returns the cleaned splits and
    the number of images removed from each split.
    """
    seen: set[str] = set()
    out: dict[str, Split] = {}
    removed: dict[str, int] = {}
    for name in SPLITS:
        hs = _hashes(splits[name].images)
        keep = np.array([h not in seen for h in hs]) if name != "train" else np.ones(len(hs), bool)
        removed[name] = int((~keep).sum())
        out[name] = Split(images=splits[name].images[keep], labels=splits[name].labels[keep])
        seen.update(hs)
    return out, removed


def prepare_splits(root: str | Path, size: int = 128) -> tuple[dict[str, Split], dict]:
    """Load the official split, remove cross-split duplicates, verify, and return a report."""
    raw = load_splits(root, size)
    clean, removed = deduplicate_splits(raw)
    report = check_splits(clean)
    report["removed_cross_split_duplicates"] = removed
    report["official_sizes"] = {k: len(v.labels) for k, v in raw.items()}
    return clean, report


@dataclass
class Standardizer:
    """Scalar mean/std of pixel/255, fitted on the training images only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, train_images: np.ndarray) -> Standardizer:
        x = train_images.astype(np.float64) / 255.0
        return cls(mean=float(x.mean()), std=float(x.std()))

    def transform(self, images: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(images.astype(np.float32) / 255.0)
        return ((x - self.mean) / self.std).unsqueeze(1)
