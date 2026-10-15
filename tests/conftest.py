import numpy as np
import pytest

from pxray.data import Split


def make_toy_splits(seed=0, sizes=(60, 20, 30), hw=32):
    """Synthetic 'X-rays': positives carry a bright blob, so the task is learnable."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, n in zip(("train", "val", "test"), sizes, strict=True):
        y = (rng.random(n) < 0.7).astype(np.int64)
        y[:2] = (0, 1)
        x = rng.integers(40, 90, size=(n, hw, hw)).astype(np.uint8)
        for i in np.flatnonzero(y == 1):
            r, c = rng.integers(8, hw - 8, size=2)
            x[i, r - 4 : r + 4, c - 4 : c + 4] = 220
        out[name] = Split(images=x, labels=y)
    return out


@pytest.fixture()
def toy_splits():
    return make_toy_splits()
