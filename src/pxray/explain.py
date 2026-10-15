"""Grad-CAM figures for correct and incorrect cases, plus border-attention statistics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import prepare_splits  # noqa: E402
from .experiment import IMAGE_SIZE, RUNS, load_trained, weights_path  # noqa: E402
from .gradcam import GradCAM, overlay_heatmap  # noqa: E402
from .models import gradcam_layer  # noqa: E402
from .report import NAMES  # noqa: E402
from .shortcut import border_mass_fraction, ring_mask  # noqa: E402

RING = 0.10
EXPLAIN_RUNS = ("small_cnn", "mobilenet_v3_small")
OUTCOMES = ("TP", "TN", "FP", "FN")
OUTCOME_TITLES = {
    "TP": "True positive (pneumonia found)",
    "TN": "True negative (normal)",
    "FP": "False positive (normal called pneumonia)",
    "FN": "False negative (pneumonia missed)",
}


def compute_cams(model: torch.nn.Module, x: torch.Tensor, batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for the pneumonia logit on every image; returns (cams, logits)."""
    cams, logits = [], []
    with GradCAM(model, gradcam_layer(model)) as cam:
        for i in range(0, len(x), batch):
            c, lg = cam(x[i : i + batch], target="pneumonia")
            cams.append(c.numpy())
            logits.append(lg.numpy())
    return np.concatenate(cams), np.concatenate(logits).astype(np.float64)


def outcome_of(y: np.ndarray, logit: np.ndarray) -> np.ndarray:
    pred = (logit >= 0).astype(int)
    out = np.empty(len(y), dtype="<U2")
    out[(y == 1) & (pred == 1)] = "TP"
    out[(y == 0) & (pred == 0)] = "TN"
    out[(y == 0) & (pred == 1)] = "FP"
    out[(y == 1) & (pred == 0)] = "FN"
    return out


def select_cases(y: np.ndarray, logit: np.ndarray, per_row: int = 4, seed: int = 0) -> dict[str, list[int]]:
    """Correct cases: seeded random draw. Errors: the most confident ones (largest |logit|)."""
    rng = np.random.default_rng(seed)
    oc = outcome_of(y, logit)
    sel = {}
    for k in OUTCOMES:
        idx = np.flatnonzero(oc == k)
        if k in ("TP", "TN"):
            sel[k] = sorted(rng.choice(idx, size=min(per_row, len(idx)), replace=False).tolist())
        else:
            sel[k] = idx[np.argsort(-np.abs(logit[idx]))][:per_row].tolist()
    return sel


def explain_all(root: Path, results: Path, art_dir: Path, docs: Path) -> dict:
    splits, _ = prepare_splits(root, IMAGE_SIZE)
    yt, imgs = splits["test"].labels, splits["test"].images
    fig_dir = docs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ring = ring_mask(IMAGE_SIZE, IMAGE_SIZE, RING)
    stats: dict = {"ring_fraction": RING, "ring_area_share": float(ring.mean()), "models": {}}
    for name in EXPLAIN_RUNS:
        seeds = [s for s in range(10) if weights_path(art_dir, name, s).exists()]
        if not seeds:
            continue
        per_seed = []
        cam0 = lg0 = None
        for s in seeds:
            model, st = load_trained(name, s, art_dir)
            cams, logits = compute_cams(model, st.transform(imgs))
            frac = border_mass_fraction(cams, RING)
            oc = outcome_of(yt, logits)
            per_seed.append(
                {
                    "seed": s,
                    "mean_border_share": float(np.nanmean(frac)),
                    "by_outcome": {
                        k: float(np.nanmean(frac[oc == k])) if (oc == k).any() else None for k in OUTCOMES
                    },
                    "outcome_counts": {k: int((oc == k).sum()) for k in OUTCOMES},
                }
            )
            if s == 0:
                cam0, lg0 = cams, logits
        stats["models"][name] = {
            "seeds": seeds,
            "mean_border_share": float(np.mean([p["mean_border_share"] for p in per_seed])),
            "per_seed": per_seed,
        }
        if cam0 is None:
            continue
        sel = select_cases(yt, lg0)
        stats["models"][name]["figure_cases_seed0"] = sel
        fig, axes = plt.subplots(4, 4, figsize=(8.2, 9.2))
        for r, k in enumerate(OUTCOMES):
            for c in range(4):
                ax = axes[r, c]
                ax.axis("off")
                if c < len(sel[k]):
                    i = sel[k][c]
                    p = 1 / (1 + np.exp(-lg0[i]))
                    ax.imshow(overlay_heatmap(imgs[i], cam0[i], 0.55))
                    ax.set_title(f"#{i}  P(pneumonia)={p:.2f}", fontsize=8)
            axes[r, 0].text(
                -0.08,
                0.5,
                OUTCOME_TITLES[k].replace(" (", "\n("),
                transform=axes[r, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
            )
        fig.suptitle(
            f"Grad-CAM for the pneumonia logit, {NAMES[name]}, seed 0, test images\n"
            "(correct rows: seeded random draw; error rows: most confident errors)",
            fontsize=9,
        )
        fig.tight_layout(rect=(0.08, 0, 1, 0.96))
        fig.savefig(fig_dir / f"gradcam_{name}.png", dpi=110)
        plt.close(fig)
    (results / "explain.json").write_text(json.dumps(stats, indent=1))
    print(
        json.dumps(
            {k: {"mean_border_share": v["mean_border_share"]} for k, v in stats["models"].items()}, indent=1
        ),
        "ring area share:",
        round(stats["ring_area_share"], 3),
    )
    return stats


__all__ = ["RUNS", "explain_all", "select_cases", "compute_cams", "outcome_of"]
