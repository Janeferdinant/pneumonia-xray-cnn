"""Experiment registry and runner: trains every configuration over several seeds."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from .baseline import fit_logreg
from .data import Split, Standardizer, prepare_splits
from .metrics import roc_auc
from .models import build_model
from .shortcut import keep_border_only, mask_border
from .train import RunResult, TrainConfig, predict_logits

SEEDS = (0, 1, 2, 3, 4)
IMAGE_SIZE = 128
MASK_FRACS = (0.05, 0.10, 0.20)

RUNS: dict[str, TrainConfig] = {
    "small_cnn": TrainConfig(model="small_cnn", epochs=30, lr=2e-3, loss="bce"),
    "mobilenet_v3_small": TrainConfig(
        model="mobilenet_v3_small", pretrained=True, epochs=10, lr=5e-4, loss="bce"
    ),
    # ablations
    "small_cnn_unweighted": TrainConfig(model="small_cnn", epochs=30, lr=2e-3, loss="bce_unweighted"),
    "small_cnn_focal": TrainConfig(model="small_cnn", epochs=30, lr=2e-3, loss="focal"),
    "small_cnn_noaug": TrainConfig(model="small_cnn", epochs=30, lr=2e-3, loss="bce", augment=False),
    "mobilenet_v3_small_scratch": TrainConfig(
        model="mobilenet_v3_small", pretrained=False, epochs=10, lr=5e-4, loss="bce"
    ),
}
MAIN_RUNS = ("logreg", "small_cnn", "mobilenet_v3_small")
ABLATION_RUNS = tuple(k for k in RUNS if k not in MAIN_RUNS)


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def fill_value(st: Standardizer) -> int:
    return int(round(st.mean * 255))


def masked_test_logits(model: torch.nn.Module, st: Standardizer, test: Split) -> dict[str, np.ndarray]:
    fill = fill_value(st)
    out = {}
    for f in MASK_FRACS:
        out[f"centre_{f:.2f}"] = predict_logits(
            model, st.transform(mask_border(test.images, f, fill))
        )
        out[f"border_{f:.2f}"] = predict_logits(
            model, st.transform(keep_border_only(test.images, f, fill))
        )
    return out


def pred_path(out_dir: Path, name: str, seed: int) -> Path:
    return out_dir / "predictions" / f"{name}_seed{seed}.npz"


def weights_path(art_dir: Path, name: str, seed: int) -> Path:
    return art_dir / f"{name}_seed{seed}.pt"


def load_trained(name: str, seed: int, art_dir: Path) -> tuple[torch.nn.Module, Standardizer]:
    blob = torch.load(weights_path(art_dir, name, seed), map_location="cpu", weights_only=True)
    cfg = RUNS[name]
    model = build_model(cfg.model, pretrained=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, Standardizer(mean=float(blob["mean"]), std=float(blob["std"]))


def run_baseline(splits: dict[str, Split], out_dir: Path) -> dict:
    """Logistic regression (deterministic, one run) plus border/centre-only data probes."""
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    res = fit_logreg(splits, pool=4)
    fill = fill_value(Standardizer.fit(splits["train"].images))
    probes = {}
    for f in MASK_FRACS:
        for kind, fn in (("centre", mask_border), ("border", keep_border_only)):
            r = fit_logreg(splits, pool=4, transform=lambda im, fn=fn, f=f: fn(im, f, fill))
            probes[f"{kind}_{f:.2f}"] = r["test_logits"]
    np.savez_compressed(
        pred_path(out_dir, "logreg", 0), val_logits=res["val_logits"], test_logits=res["test_logits"],
        **{f"probe_{k}": v for k, v in probes.items()},
    )
    info = {"C": res["C"]}
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs" / "logreg_seed0.json").write_text(json.dumps(info, indent=1))
    return info


def run_training(
    name: str, seeds: tuple[int, ...], root: Path, out_dir: Path, art_dir: Path, force: bool = False
) -> list[dict]:
    from .train import train_one

    splits, _ = prepare_splits(root, IMAGE_SIZE)
    if name == "logreg":
        return [{"seed": 0, **run_baseline(splits, out_dir)}]
    cfg = RUNS[name]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for seed in seeds:
        pp = pred_path(out_dir, name, seed)
        if pp.exists() and weights_path(art_dir, name, seed).exists() and not force:
            infos.append(json.loads((out_dir / "runs" / f"{name}_seed{seed}.json").read_text()))
            continue
        res: RunResult = train_one(cfg, splits, seed)
        torch.save(
            {"state_dict": res.state_dict, "mean": res.standardizer.mean, "std": res.standardizer.std},
            weights_path(art_dir, name, seed),
        )
        model, st = load_trained(name, seed, art_dir)
        masked = masked_test_logits(model, st, splits["test"])
        np.savez_compressed(
            pp, val_logits=res.val_logits, test_logits=res.test_logits,
            **{f"mask_{k}": v for k, v in masked.items()},
        )
        info = {
            "run": name, "seed": seed, "config": cfg.to_dict(), "best_epoch": res.best_epoch,
            "best_val_auc": max(h["val_auc"] for h in res.history),
            "test_auc": roc_auc(splits["test"].labels, res.test_logits),
            "seconds": round(res.seconds, 1), "history": res.history,
        }
        (out_dir / "runs" / f"{name}_seed{seed}.json").write_text(json.dumps(info, indent=1))
        infos.append(info)
        print(f"{name} seed {seed}: best epoch {res.best_epoch}, val AUC "
              f"{info['best_val_auc']:.4f}, {res.seconds:.0f}s", flush=True)
    return infos


def write_run_log(name: str, infos: list[dict], docs_dir: Path) -> Path:
    today = dt.date.today().isoformat()
    path = docs_dir / "runs" / f"{today}-train-{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    hw = hardware_info()
    lines = [f"# Run log: {name} ({today})", "", f"- hardware: `{json.dumps(hw)}`"]
    if name != "logreg":
        lines += [f"- config: `{json.dumps(RUNS[name].to_dict())}`",
                  f"- data: PneumoniaMNIST {IMAGE_SIZE}x{IMAGE_SIZE}, official train/val/test split",
                  "", "| seed | best epoch | best val AUC | test AUC | seconds |",
                  "|---|---|---|---|---|"]
        for i in infos:
            lines.append(f"| {i['seed']} | {i['best_epoch']} | {i['best_val_auc']:.4f} | "
                         f"{i['test_auc']:.4f} | {i['seconds']} |")
        lines += ["", "Test AUC is listed for the record only; the checkpoint was chosen on "
                      "validation AUC."]
    else:
        lines += ["- config: logistic regression, 4x4 block-averaged pixels, balanced class weights, "
                  "C chosen on validation AUC", f"- chosen C: {infos[0].get('C', 'see results/runs')}"]
    path.write_text("\n".join(lines) + "\n")
    return path
