"""Border-mask test: does the model (or the data) rely on the image border?"""

from __future__ import annotations

import numpy as np


def ring_mask(h: int, w: int, frac: float) -> np.ndarray:
    """Boolean (h, w) mask, True on a border ring `frac` of the short side thick."""
    b = int(round(frac * min(h, w)))
    m = np.ones((h, w), dtype=bool)
    m[b : h - b, b : w - b] = False
    return m


def mask_border(images: np.ndarray, frac: float, fill: int) -> np.ndarray:
    """Return a copy with the border ring replaced by a constant (centre kept)."""
    out = images.copy()
    out[:, ring_mask(*images.shape[1:], frac)] = fill
    return out


def keep_border_only(images: np.ndarray, frac: float, fill: int) -> np.ndarray:
    """Return a copy with everything except the border ring replaced by a constant."""
    out = images.copy()
    out[:, ~ring_mask(*images.shape[1:], frac)] = fill
    return out


def border_mass_fraction(cams: np.ndarray, frac: float) -> np.ndarray:
    """Share of each Grad-CAM map's total mass that lies in the border ring (nan if all zero)."""
    cams = np.asarray(cams, dtype=np.float64)
    ring = ring_mask(*cams.shape[1:], frac)
    total = cams.sum(axis=(1, 2))
    inside = cams[:, ring].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, inside / total, np.nan)
