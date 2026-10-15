"""Evaluation beyond accuracy: metrics, calibration, bootstrap CIs, mask test, figures, tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .bootstrap import paired_bootstrap  # noqa: E402
from .calibration import (  # noqa: E402
    apply_temperature,
    brier,
    expected_calibration_error,
    fit_temperature,
    nll,
    reliability_curve,
    sigmoid,
)
from .data import prepare_splits  # noqa: E402
from .experiment import ABLATION_RUNS, IMAGE_SIZE, MAIN_RUNS, MASK_FRACS, hardware_info  # noqa: E402
from .metrics import (  # noqa: E402
    binary_metrics,
    roc_auc,
    sens_spec_at_threshold,
    sensitivity_at_specificity,
    threshold_for_specificity,
)

COLORS = {"logreg": "#666666", "small_cnn": "#0072B2", "mobilenet_v3_small": "#D55E00"}
NAMES = {
    "logreg": "Logistic regression (pixels)",
    "small_cnn": "Small CNN (from scratch)",
    "mobilenet_v3_small": "MobileNetV3-Small (ImageNet, fine-tuned)",
    "small_cnn_unweighted": "Small CNN, no class weights",
    "small_cnn_focal": "Small CNN, focal loss",
    "small_cnn_noaug": "Small CNN, no augmentation",
    "mobilenet_v3_small_scratch": "MobileNetV3-Small, random init",
}


def load_run(results: Path, name: str) -> dict | None:
    files = sorted(
        (results / "predictions").glob(f"{name}_seed*.npz"), key=lambda p: int(p.stem.split("seed")[1])
    )
    if not files:
        return None
    zs = [np.load(f) for f in files]
    out = {
        "seeds": [int(f.stem.split("seed")[1]) for f in files],
        "val": np.stack([z["val_logits"] for z in zs]),
        "test": np.stack([z["test_logits"] for z in zs]),
        "mask": {},
    }
    for k in zs[0].files:
        if k.startswith("mask_"):
            out["mask"][k[5:]] = np.stack([z[k] for z in zs])
        if k.startswith("probe_"):
            out.setdefault("probe", {})[k[6:]] = np.stack([z[k] for z in zs])
    return out


def seed_metrics(yv: np.ndarray, lv: np.ndarray, yt: np.ndarray, lt: np.ndarray) -> dict:
    t = fit_temperature(lv, yv)
    p_raw, p_cal = sigmoid(lt), sigmoid(apply_temperature(lt, t))
    m = binary_metrics(yt, p_raw, 0.5)
    thr = threshold_for_specificity(yv, lv, 0.90)
    s_v, sp_v = sens_spec_at_threshold(yt, lt, thr)
    m.update(
        auc=roc_auc(yt, lt),
        val_auc=roc_auc(yv, lv),
        sens_at_spec90=sensitivity_at_specificity(yt, lt, 0.90),
        sens_at_spec95=sensitivity_at_specificity(yt, lt, 0.95),
        ece_raw=expected_calibration_error(yt, p_raw),
        ece_cal=expected_calibration_error(yt, p_cal),
        nll_raw=nll(yt, p_raw),
        nll_cal=nll(yt, p_cal),
        brier_raw=brier(yt, p_raw),
        brier_cal=brier(yt, p_cal),
        temperature=t,
        valthr90_test_sensitivity=s_v,
        valthr90_test_specificity=sp_v,
        valthr90_val_threshold_logit=thr,
        val_ece_raw=expected_calibration_error(yv, sigmoid(lv)),
    )
    return {k: float(v) for k, v in m.items()}


def _agg(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    ddof = 1 if len(rows) > 1 else 0
    return {
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "sd": {k: float(np.std([r[k] for r in rows], ddof=ddof)) for k in keys},
    }


def _f1(y, p):
    return binary_metrics(y, p, 0.5)["f1"]


def _sens(y, p):
    return binary_metrics(y, p, 0.5)["sensitivity"]


def _spec(y, p):
    return binary_metrics(y, p, 0.5)["specificity"]


def _s90(y, p):
    return sensitivity_at_specificity(y, p, 0.90)


def _s95(y, p):
    return sensitivity_at_specificity(y, p, 0.95)


BOOT_METRICS = {
    "auc": roc_auc,
    "f1": _f1,
    "sensitivity": _sens,
    "specificity": _spec,
    "sens_at_spec90": _s90,
    "sens_at_spec95": _s95,
    "ece_raw": expected_calibration_error,
}


def _ci(t: tuple[float, float, float], d: int = 3) -> str:
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def evaluate_all(root: Path, results: Path, docs: Path, n_boot: int = 2000) -> dict:
    splits, data_report = prepare_splits(root, IMAGE_SIZE)
    yv, yt = splits["val"].labels, splits["test"].labels
    runs = {n: load_run(results, n) for n in (*MAIN_RUNS, *ABLATION_RUNS)}
    runs = {k: v for k, v in runs.items() if v is not None}

    summary: dict = {
        "hardware": hardware_info(),
        "n_test": int(len(yt)),
        "n_val": int(len(yv)),
        "n_boot": n_boot,
        "data": data_report,
        "runs": {},
    }
    for name, r in runs.items():
        rows = [seed_metrics(yv, r["val"][i], yt, r["test"][i]) for i in range(len(r["seeds"]))]
        summary["runs"][name] = {"seeds": r["seeds"], "per_seed": rows, **_agg(rows)}
        mask = {}
        for k, arr in r["mask"].items():
            aucs = [roc_auc(yt, a) for a in arr]
            mask[k] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs, ddof=1 if len(aucs) > 1 else 0)),
                "auc_per_seed": [float(a) for a in aucs],
            }
        for k, arr in r.get("probe", {}).items():
            mask[k] = {
                "auc_mean": float(roc_auc(yt, arr[0])),
                "auc_sd": 0.0,
                "auc_per_seed": [float(roc_auc(yt, arr[0]))],
            }
        summary["runs"][name]["mask"] = mask
        sc = summary["runs"][name]
        sc["mask_original_auc"] = sc["mean"]["auc"]

    main = [n for n in MAIN_RUNS if n in runs]
    probs = {f"{n}": sigmoid(runs[n]["test"]) for n in main}
    diffs = [
        (a, b)
        for a, b in (
            ("mobilenet_v3_small", "small_cnn"),
            ("small_cnn", "logreg"),
            ("mobilenet_v3_small", "logreg"),
        )
        if a in probs and b in probs
    ]
    boot = paired_bootstrap(yt, probs, BOOT_METRICS, n_boot=n_boot, seed=0, differences=diffs)
    cal_probs = {
        n: sigmoid(
            np.stack(
                [
                    apply_temperature(runs[n]["test"][i], summary["runs"][n]["per_seed"][i]["temperature"])
                    for i in range(len(runs[n]["seeds"]))
                ]
            )
        )
        for n in main
    }
    boot_cal = paired_bootstrap(yt, cal_probs, {"ece": expected_calibration_error}, n_boot=n_boot, seed=0)
    for n in main:
        summary["runs"][n]["bootstrap_ci"] = boot[n]
        summary["runs"][n]["bootstrap_ci"]["ece_cal"] = boot_cal[n]["ece"]
    summary["differences"] = {k: v for k, v in boot.items() if " - " in k}

    # ablations: same seeds 0-2 only, paired bootstrap against the matching reference
    abl_pairs = [
        ("small_cnn", ["small_cnn_unweighted", "small_cnn_focal", "small_cnn_noaug"]),
        ("mobilenet_v3_small", ["mobilenet_v3_small_scratch"]),
    ]
    abl = {}
    for ref, others in abl_pairs:
        others = [o for o in others if o in runs]
        if ref not in runs or not others:
            continue
        seeds = [s for s in runs[ref]["seeds"] if all(s in runs[o]["seeds"] for o in others)]
        sel = {ref: sigmoid(runs[ref]["test"][[runs[ref]["seeds"].index(s) for s in seeds]])}
        for o in others:
            sel[o] = sigmoid(runs[o]["test"][[runs[o]["seeds"].index(s) for s in seeds]])
        m = {
            "auc": roc_auc,
            "sensitivity": _sens,
            "specificity": _spec,
            "f1": _f1,
            "ece_raw": expected_calibration_error,
        }
        res = paired_bootstrap(
            yt, sel, m, n_boot=max(500, n_boot // 2), seed=1, differences=[(o, ref) for o in others]
        )
        abl[ref] = {"seeds": seeds, "results": res}
    summary["ablations"] = abl

    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=1))
    (results / "summary.md").write_text(render_markdown(summary))
    figures(runs, summary, yv, yt, docs / "figures")
    print((results / "summary.md").read_text())
    return summary


def render_markdown(s: dict) -> str:
    L: list[str] = []
    runs = s["runs"]
    main = [n for n in MAIN_RUNS if n in runs]
    L += [
        "## Test-set performance (mean over seeds, bootstrap 95% CI over test images; seed SD in parentheses)",
        "",
        "| Model | Seeds | ROC AUC | Sens. @ 90% spec | Sens. @ 95% spec | F1 @ 0.5 | Sensitivity @ 0.5 | Specificity @ 0.5 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in main:
        r = runs[n]
        b = r["bootstrap_ci"]
        sd = r["sd"]

        def cell(k: str, b=b, sd=sd) -> str:
            return f"{_ci(b[k])} ({sd[k]:.3f})"

        L.append(
            f"| {NAMES[n]} | {len(r['seeds'])} | {cell('auc')} | {cell('sens_at_spec90')} | "
            f"{cell('sens_at_spec95')} | {cell('f1')} | {cell('sensitivity')} | {cell('specificity')} |"
        )
    L += [
        "",
        "## Other metrics at threshold 0.5 (mean over seeds)",
        "",
        "| Model | Accuracy | Balanced acc. | Precision | Mean TN / FP / FN / TP |",
        "|---|---|---|---|---|",
    ]
    for n in main:
        m = runs[n]["mean"]
        L.append(
            f"| {NAMES[n]} | {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['precision']:.3f} | "
            f"{m['tn']:.1f} / {m['fp']:.1f} / {m['fn']:.1f} / {m['tp']:.1f} |"
        )
    L += [
        "",
        "## Threshold chosen on validation (specificity >= 0.90 on validation), applied to test (mean over seeds)",
        "",
        "| Model | Val-chosen threshold: test sensitivity | Test specificity | Validation AUC |",
        "|---|---|---|---|",
    ]
    for n in main:
        m, sd = runs[n]["mean"], runs[n]["sd"]
        L.append(
            f"| {NAMES[n]} | {m['valthr90_test_sensitivity']:.3f} ({sd['valthr90_test_sensitivity']:.3f}) | "
            f"{m['valthr90_test_specificity']:.3f} ({sd['valthr90_test_specificity']:.3f}) | {m['val_auc']:.3f} |"
        )
    L += [
        "",
        "## Calibration (test; temperature fitted on validation only)",
        "",
        "| Model | T (mean, SD) | ECE raw [95% CI] | ECE after scaling [95% CI] | NLL raw -> scaled | Brier raw -> scaled |",
        "|---|---|---|---|---|---|",
    ]
    for n in main:
        r = runs[n]
        m, b = r["mean"], r["bootstrap_ci"]
        L.append(
            f"| {NAMES[n]} | {m['temperature']:.2f}, {r['sd']['temperature']:.2f} | {_ci(b['ece_raw'])} | "
            f"{_ci(b['ece_cal'])} | {m['nll_raw']:.3f} -> {m['nll_cal']:.3f} | {m['brier_raw']:.3f} -> {m['brier_cal']:.3f} |"
        )
    if s.get("differences"):
        L += [
            "",
            "## Paired differences (bootstrap 95% CI over test images)",
            "",
            "| Comparison | ROC AUC | Sens. @ 90% spec | F1 @ 0.5 |",
            "|---|---|---|---|",
        ]
        for k, v in s["differences"].items():
            L.append(
                f"| {NAMES.get(k.split(' - ')[0], k)} minus {NAMES.get(k.split(' - ')[1], k)} | "
                f"{_ci(v['auc'])} | {_ci(v['sens_at_spec90'])} | {_ci(v['f1'])} |"
            )
    L += [
        "",
        "## Border-mask test (test AUC, mean over seeds, SD in parentheses)",
        "",
        "Border ring width as a fraction of the image side. `centre` = ring masked (model sees the centre only); "
        "`border` = only the ring visible. Models were not retrained on masked inputs; logistic regression rows are "
        "separate classifiers trained on the masked pixels (data-level probe).",
        "",
        "| Model | Original | "
        + " | ".join(f"centre {f:.2f}" for f in MASK_FRACS)
        + " | "
        + " | ".join(f"border {f:.2f}" for f in MASK_FRACS)
        + " |",
        "|---|---|" + "---|" * (2 * len(MASK_FRACS)),
    ]
    for n in main:
        r = runs[n]
        mk = r["mask"]
        cells = []
        for kind in ("centre", "border"):
            for f in MASK_FRACS:
                key = f"{kind}_{f:.2f}"
                cells.append(f"{mk[key]['auc_mean']:.3f} ({mk[key]['auc_sd']:.3f})" if key in mk else "n/a")
        orig = f"{r['mean']['auc']:.3f} ({r['sd']['auc']:.3f})"
        L.append(f"| {NAMES[n]} | {orig} | " + " | ".join(cells) + " |")
    if s.get("ablations"):
        L += ["", "## Ablations (seeds listed per block; paired bootstrap over test images)", ""]
        for ref, blk in s["ablations"].items():
            L += [
                f"Reference: {NAMES[ref]}; seeds {blk['seeds']}",
                "",
                "| Variant | ROC AUC | Sensitivity @ 0.5 | Specificity @ 0.5 | F1 @ 0.5 | ECE raw |",
                "|---|---|---|---|---|---|",
            ]
            for k, v in blk["results"].items():
                if " - " in k:
                    nm = f"{NAMES.get(k.split(' - ')[0])} minus reference (difference)"
                else:
                    nm = NAMES.get(k, k) + (" (reference)" if k == ref else "")
                L.append(
                    f"| {nm} | {_ci(v['auc'])} | {_ci(v['sensitivity'])} | {_ci(v['specificity'])} | "
                    f"{_ci(v['f1'])} | {_ci(v['ece_raw'])} |"
                )
            L.append("")
    return "\n".join(L) + "\n"


def figures(runs: dict, s: dict, yv: np.ndarray, yt: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    main = [n for n in MAIN_RUNS if n in runs]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for n in main:
        for i in range(len(runs[n]["seeds"])):
            fpr, tpr, _ = roc_curve(yt, runs[n]["test"][i])
            ax.plot(fpr, tpr, color=COLORS[n], alpha=0.35, lw=1, label=NAMES[n] if i == 0 else None)
    ax.plot([0, 1], [0, 1], ls=":", color="#999999")
    ax.axvline(0.10, ls="--", color="#999999", lw=0.8)
    ax.text(0.11, 0.05, "specificity 90%", fontsize=8, color="#555555")
    ax.set(
        xlabel="False positive rate (1 - specificity)",
        ylabel="True positive rate (sensitivity)",
        title="ROC on the official test set (one line per seed)",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=130)
    plt.close(fig)

    # reliability (pooled over seeds)
    cnn = [n for n in main if n != "logreg"]
    fig, axes = plt.subplots(1, len(cnn), figsize=(4.8 * len(cnn), 4.4), squeeze=False)
    for ax, n in zip(axes[0], cnn, strict=True):
        r = runs[n]
        raw = sigmoid(r["test"]).ravel()
        cal = np.concatenate(
            [
                sigmoid(apply_temperature(r["test"][i], s["runs"][n]["per_seed"][i]["temperature"]))
                for i in range(len(r["seeds"]))
            ]
        )
        yy = np.tile(yt, len(r["seeds"]))
        ax.plot([0, 1], [0, 1], ls=":", color="#999999", label="perfect")
        for p, lab, col in ((raw, "raw", COLORS[n]), (cal, "temperature scaled (val)", "#009E73")):
            c, a, k = reliability_curve(yy, p, 10)
            ok = k > 0
            ax.plot(c[ok], a[ok], marker="o", ms=4, color=col, label=lab)
        ece = s["runs"][n]["mean"]
        ax.set(
            xlabel="Mean predicted P(pneumonia)",
            ylabel="Observed fraction pneumonia",
            title=f"{NAMES[n]}\nECE {ece['ece_raw']:.3f} -> {ece['ece_cal']:.3f}",
        )
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=130)
    plt.close(fig)

    # confusion matrices (mean over seeds at threshold 0.5)
    fig, axes = plt.subplots(1, len(main), figsize=(3.6 * len(main), 3.5), squeeze=False)
    for ax, n in zip(axes[0], main, strict=True):
        m = s["runs"][n]["mean"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["normal", "pneumonia"],
            yticklabels=["normal", "pneumonia"],
            xlabel="Predicted",
            ylabel="True",
            title=NAMES[n].split(" (")[0],
        )
        ax.title.set_fontsize(9)
    fig.suptitle("Mean test confusion matrix at threshold 0.5 (counts, colour = row share)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)

    # mask test
    keys = ["original"] + [f"centre_{f:.2f}" for f in MASK_FRACS] + [f"border_{f:.2f}" for f in MASK_FRACS]
    labels = (
        ["original"]
        + [f"centre only\n(ring {int(f * 100)}% masked)" for f in MASK_FRACS]
        + [f"ring {int(f * 100)}% only" for f in MASK_FRACS]
    )
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.8 / len(main)
    for j, n in enumerate(main):
        r = s["runs"][n]
        vals, errs = [], []
        for k in keys:
            if k == "original":
                vals.append(r["mean"]["auc"])
                errs.append(r["sd"]["auc"])
            else:
                vals.append(r["mask"][k]["auc_mean"])
                errs.append(r["mask"][k]["auc_sd"])
        ax.bar(
            np.arange(len(keys)) + (j - (len(main) - 1) / 2) * w,
            vals,
            w,
            yerr=errs,
            color=COLORS[n],
            label=NAMES[n],
            capsize=2,
        )
    ax.axhline(0.5, ls=":", color="#999999")
    ax.set_xticks(range(len(keys)), labels, fontsize=8)
    ax.set(ylabel="Test ROC AUC", ylim=(0.3, 1.0), title="Border-mask test (error bars: SD over seeds)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "mask_test.png", dpi=130)
    plt.close(fig)
