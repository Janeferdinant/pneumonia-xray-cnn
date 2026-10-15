"""Write the static explorer data: predictions JSON plus small PNGs of PneumoniaMNIST test images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .calibration import fit_temperature
from .data import prepare_splits
from .experiment import IMAGE_SIZE, load_trained
from .explain import EXPLAIN_RUNS, compute_cams
from .gradcam import overlay_heatmap
from .report import NAMES

THUMB = 96
GALLERY_PER_CLASS = 60
SEED = 0

DISCLAIMER = (
    "Research exercise on a public paediatric dataset. Not a medical device. "
    "Not for diagnosis, screening or any clinical decision."
)


def export_demo(root: Path, results: Path, art_dir: Path, docs: Path) -> Path:
    splits, _ = prepare_splits(root, IMAGE_SIZE)
    yv, yt, imgs = splits["val"].labels, splits["test"].labels, splits["test"].images
    data_dir = docs / "data"
    (data_dir / "img").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    gallery = sorted(
        np.concatenate(
            [rng.choice(np.flatnonzero(yt == c), GALLERY_PER_CLASS, replace=False) for c in (0, 1)]
        ).tolist()
    )
    for i in gallery:
        Image.fromarray(imgs[i]).resize((THUMB, THUMB), Image.LANCZOS).save(
            data_dir / "img" / f"{i}.png", optimize=True
        )

    models = {}
    for name in EXPLAIN_RUNS:
        model, st = load_trained(name, SEED, art_dir)
        x_te, x_va = st.transform(imgs), st.transform(splits["val"].images)
        cams, logits = compute_cams(model, x_te)
        from .train import predict_logits

        t = fit_temperature(predict_logits(model, x_va), yv)
        (data_dir / "cam" / name).mkdir(parents=True, exist_ok=True)
        for i in gallery:
            rgb = Image.fromarray(overlay_heatmap(imgs[i], cams[i], 0.55)).resize(
                (THUMB, THUMB), Image.LANCZOS
            )
            rgb.quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(
                data_dir / "cam" / name / f"{i}.png", optimize=True
            )
        models[name] = {
            "label": NAMES[name],
            "logits": [round(float(v), 4) for v in logits],
            "temperature": round(float(t), 4),
        }
    payload = {
        "disclaimer": DISCLAIMER,
        "dataset": "PneumoniaMNIST (MedMNIST v2), 128 px, official test split, CC BY 4.0",
        "seed": SEED,
        "n_test": int(len(yt)),
        "labels": [int(v) for v in yt],
        "gallery": gallery,
        "gallery_note": (
            f"{GALLERY_PER_CLASS} normal and {GALLERY_PER_CLASS} pneumonia test images drawn at "
            f"random (seed {SEED}); not selected by correctness. Normal is over-represented here "
            "relative to the full test set (37.5% normal)."
        ),
        "cam_note": "Grad-CAM for the pneumonia logit, from the final convolutional block.",
        "models": models,
    }
    out = data_dir / "explorer.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    size = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file())
    print(f"wrote {out} ; data dir {size / 1e6:.2f} MB")
    return out
