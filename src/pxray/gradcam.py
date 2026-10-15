"""Grad-CAM (Selvaraju et al., 2017) for a single-logit binary classifier."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps


class GradCAM:
    """Context manager: registers a forward hook on `layer`, removes it on exit.

    The CAM explains the logit of the chosen class: `pneumonia` uses +logit, `normal` uses
    -logit, `auto` uses whichever class the model predicts for each image.
    """

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model, self.layer = model, layer
        self._act: torch.Tensor | None = None
        self._handle = None

    def __enter__(self) -> GradCAM:
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _m, _inp, out: torch.Tensor) -> None:
        self._act = out

    def __call__(self, x: torch.Tensor, target: str = "auto") -> tuple[torch.Tensor, torch.Tensor]:
        if target not in {"auto", "pneumonia", "normal"}:
            raise ValueError("target must be auto, pneumonia or normal")
        with torch.enable_grad():
            logits = self.model(x)
            act = self._act
            assert act is not None
            if target == "auto":
                sign = torch.where(logits.detach() >= 0, 1.0, -1.0)
            else:
                sign = torch.full_like(logits.detach(), 1.0 if target == "pneumonia" else -1.0)
            (grad,) = torch.autograd.grad((logits * sign).sum(), act)
        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act.detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        peak = cam.amax(dim=(1, 2), keepdim=True)
        cam = torch.where(peak > 0, cam / peak.clamp_min(1e-12), cam)
        return cam.detach(), logits.detach()


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] heatmap over a grayscale uint8 image; low-attention pixels stay gray."""
    gray = np.repeat(img[..., None].astype(np.float32), 3, axis=2)
    heat = colormaps["jet"](np.clip(cam, 0, 1))[..., :3].astype(np.float32) * 255.0
    a = (alpha * np.clip(cam, 0, 1))[..., None]
    return np.clip((1 - a) * gray + a * heat, 0, 255).astype(np.uint8)
