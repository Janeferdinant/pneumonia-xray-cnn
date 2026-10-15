"""Class-imbalance-aware losses on logits."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    pw = torch.tensor(float(pos_weight), dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def focal_loss_with_logits(
    logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """Binary focal loss (Lin et al., 2017); alpha weights the positive class."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * y + (1 - alpha) * (1 - y)) * loss
    return loss.mean()


def make_loss(kind: str, pos_weight: float, gamma: float = 2.0) -> LossFn:
    """kind: 'bce' (class-weighted), 'bce_unweighted', or 'focal' (class-balanced alpha)."""
    if kind == "bce":
        return lambda lg, y: weighted_bce(lg, y, pos_weight)
    if kind == "bce_unweighted":
        return lambda lg, y: weighted_bce(lg, y, 1.0)
    if kind == "focal":
        alpha = pos_weight / (1.0 + pos_weight)
        return lambda lg, y: focal_loss_with_logits(lg, y, gamma=gamma, alpha=alpha)
    raise ValueError(f"unknown loss {kind!r}")
