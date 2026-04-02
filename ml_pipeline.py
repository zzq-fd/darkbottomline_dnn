#!/usr/bin/env python3
"""End-to-end pipeline: analyzer -> DNN training -> (optional) apply score.

This script is a lightweight convenience wrapper that stitches together the
existing entrypoints in this repository:
- ppbbchichi_analyzer_par.py
- dnn/train_classifier.py
- dnn/apply_classifier.py

It is designed to support MG5/LHEF inputs as well as parquet/ntuples, as long as
ppbbchichi_analyzer_par.py can read them.

Typical usage (MG5/LHEF signal+background directories):
  python ml_pipeline.py \
    --year 2023 --era preBPix --tree LHEF --pid-chi 52 \
    -i /path/to/signal/results -i /path/to/background/results \
    --tag train_mix \
    --region preselection \
    --label-csv labels.csv \
    --drop-constant-features \
    --outdir outputs_dnn_train_mix

The `labels.csv` should contain columns: sample,label
where sample is the top-level directory name inside ppbbchichi-trees.root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run analyzer + train classifier + apply scores.")

    # Analyzer
    ap.add_argument("--year", required=True)
    ap.add_argument("--era", required=True)
    ap.add_argument("-i", "--inFile", action="append", required=True, help="Input file or directory (repeatable)")
    ap.add_argument("--tree", default=None, help="ROOT TTree name (e.g. LHEF). If omitted, analyzer picks the first TTree.")
    ap.add_argument("--pid-chi", type=int, default=52, help="LHEF mode only: abs(PID) used as chi (default: 52)")
    ap.add_argument("--tag", required=True, help="Output tag under outputfiles/merged/<tag>")

    # Training
    ap.add_argument("--region", default="preselection")
    ap.add_argument("--features", default=None, help="Comma-separated feature list (defaults to dnn/train_classifier.py defaults)")
    ap.add_argument("--signal-pattern", action="append", default=None, help="Repeatable regex; used if --label-csv not provided")
    ap.add_argument("--label-csv", default=None, help="CSV mapping sample->label (columns: sample,label)")
    ap.add_argument("--drop-constant-features", action="store_true", help="Forwarded to training: drop near-constant features")
    ap.add_argument("--outdir", default="outputs_dnn", help="Output dir for training artifacts")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Apply
    ap.add_argument("--skip-apply", action="store_true", help="Only train; do not write scored ROOT")
    ap.add_argument("--score-branch", default="ml_score")
    ap.add_argument("--apply-step", type=int, default=200000)

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    analyzer = repo_root / "ppbbchichi_analyzer_par.py"
    trainer = repo_root / "dnn" / "train_classifier.py"
    applier = repo_root / "dnn" / "apply_classifier.py"

    out_dir = repo_root / "outputfiles" / "merged" / str(args.tag)
    trees_root = out_dir / "ppbbchichi-trees.root"

    # 1) Run analyzer
    cmd_an = [sys.executable, str(analyzer), "--year", str(args.year), "--era", str(args.era), "--tag", str(args.tag)]
    for p in args.inFile:
        cmd_an += ["-i", str(p)]
    if args.tree:
        cmd_an += ["--tree", str(args.tree)]
    cmd_an += ["--pid-chi", str(int(args.pid_chi))]
    _run(cmd_an)

    # 2) Train
    cmd_tr = [
        sys.executable,
        str(trainer),
        "--root",
        str(trees_root),
        "--region",
        str(args.region),
        "--outdir",
        str(args.outdir),
        "--device",
        str(args.device),
    ]
    if args.features:
        cmd_tr += ["--features", str(args.features)]
    if args.label_csv:
        cmd_tr += ["--label-csv", str(args.label_csv)]
    else:
        if args.signal_pattern:
            for pat in args.signal_pattern:
                cmd_tr += ["--signal-pattern", str(pat)]
    if args.drop_constant_features:
        cmd_tr += ["--drop-constant-features"]
    _run(cmd_tr)

    # 3) Apply
    if not args.skip_apply:
        outdir = Path(args.outdir)
        cmd_ap = [
            sys.executable,
            str(applier),
            "--root",
            str(trees_root),
            "--region",
            str(args.region),
            "--model",
            str(outdir / "dnn_model.pt"),
            "--scaler",
            str(outdir / "scaler.json"),
            "--features",
            str(outdir / "features.json"),
            "--device",
            str(args.device),
            "--score-branch",
            str(args.score_branch),
            "--step",
            str(int(args.apply_step)),
        ]
        _run(cmd_ap)

    print("[OK] Pipeline completed")
    print(f"[OK] Trees: {trees_root}")
    if not args.skip_apply:
        print("[OK] Scored ROOT is '<input>-scored.root' next to the trees ROOT")


if __name__ == "__main__":
    main()
