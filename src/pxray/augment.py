"""Radiograph-appropriate augmentation on standardised (B, 1, H, W) tensors.

Small rotations, scale and shifts mimic patient positioning; brightness/contrast jitter mimics
exposure differences. There are deliberately no flips: a vertical flip produces anatomically
impossible images, and a horizontal flip moves the heart to the right side (rare situs variants).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AugmentConfig:
    max_rotation_deg: float = 10.0
    scale: tuple[float, float] = (0.9, 1.1)
    max_translate: float = 0.05  # fraction of half image size in normalised coords
    brightness: float = 0.1  # additive, in standardised units
    contrast: float = 0.15  # multiplicative around 1


def _uniform(n: int, lo: float, hi: float, g: torch.Generator) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=g)


def augment_batch(x: torch.Tensor, g: torch.Generator, cfg: AugmentConfig) -> torch.Tensor:
    n = x.shape[0]
    ang = _uniform(n, -cfg.max_rotation_deg, cfg.max_rotation_deg, g) * math.pi / 180.0
    s = _uniform(n, cfg.scale[0], cfg.scale[1], g)
    tx = _uniform(n, -cfg.max_translate, cfg.max_translate, g)
    ty = _uniform(n, -cfg.max_translate, cfg.max_translate, g)
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack(
        [torch.stack([cos, -sin, tx], dim=1), torch.stack([sin, cos, ty], dim=1)], dim=1
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    c = _uniform(n, 1 - cfg.contrast, 1 + cfg.contrast, g).view(n, 1, 1, 1)
    b = _uniform(n, -cfg.brightness, cfg.brightness, g).view(n, 1, 1, 1)
    return out * c + b
