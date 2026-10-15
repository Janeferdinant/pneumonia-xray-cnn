"""Command line: download, check-data, train, evaluate, explain, export-demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pxray",
        description="PneumoniaMNIST pneumonia CNN study. Research exercise, not a diagnostic tool.",
    )
    p.add_argument("--root", default="data", help="dataset directory (never committed)")
    p.add_argument("--results", default="results")
    p.add_argument("--artifacts", default="artifacts", help="local model weights (never committed)")
    p.add_argument("--docs", default="docs")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download PneumoniaMNIST 128 px and verify its MD5")
    sub.add_parser("check-data", help="verify split integrity, print and save a data report")
    t = sub.add_parser("train", help="train configurations over several seeds")
    t.add_argument("--run", nargs="+", default=None,
                   help="run names (default: logreg small_cnn mobilenet_v3_small)")
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    t.add_argument("--force", action="store_true")
    e = sub.add_parser("evaluate", help="metrics, calibration, bootstrap CIs, figures")
    e.add_argument("--n-boot", type=int, default=2000)
    sub.add_parser("explain", help="Grad-CAM figures and border-attention statistics")
    sub.add_parser("export-demo", help="write the static explorer data to docs/")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root, results, art, docs = (Path(args.root), Path(args.results), Path(args.artifacts),
                                Path(args.docs))
    if args.command == "download":
        from .data import ensure_dataset

        print(ensure_dataset(root, 128))
    elif args.command == "check-data":
        from .data import prepare_splits

        _, rep = prepare_splits(root, 128)
        results.mkdir(parents=True, exist_ok=True)
        (results / "data_report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    elif args.command == "train":
        from .experiment import MAIN_RUNS, run_training, write_run_log

        for name in args.run or MAIN_RUNS:
            infos = run_training(name, tuple(args.seeds), root, results, art, args.force)
            print("log:", write_run_log(name, infos, docs))
    elif args.command == "evaluate":
        from .report import evaluate_all

        evaluate_all(root, results, docs, n_boot=args.n_boot)
    elif args.command == "explain":
        from .explain import explain_all

        explain_all(root, results, art, docs)
    elif args.command == "export-demo":
        from .export_demo import export_demo

        export_demo(root, results, art, docs)


if __name__ == "__main__":
    main()
