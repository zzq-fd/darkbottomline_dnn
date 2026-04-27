#!/usr/bin/env python3
"""One-click DNN workflow launcher.

Inputs:
- year
- signal (category or full prefix)
- save-name (used as tag for output naming)

This script forwards to dnn/train_from_eos_outputs.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dnn.feature_engineering import get_default_feature_csv


def _is_signal_prefix(sig: str) -> bool:
    s = sig.strip().lower()
    return s.startswith("new")


def _torch_cuda_available(python_exe: str) -> bool:
    cmd = [
        str(python_exe),
        "-c",
        "import torch; print('1' if torch.cuda.is_available() and torch.cuda.device_count()>0 else '0')",
    ]
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("1")


def main() -> None:
    ap = argparse.ArgumentParser(description="One-click EOS->DNN workflow")
    ap.add_argument("--year", required=True, help="Target year, e.g. 2022")
    ap.add_argument(
        "--signal",
        required=True,
        help="Signal category (e.g. diboson) or full prefix (e.g. newdiboson)",
    )
    ap.add_argument(
        "--save-name",
        required=True,
        help="Output naming tag used for merged trees, plots, and model directory",
    )
    ap.add_argument(
        "--base-dir",
        default="/eos/user/x/xdu/dbl_praveen/DarkBottomLine/outputs",
        help="EOS base directory",
    )
    ap.add_argument("--region", default="preselection")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-events-per-sample", type=int, default=200000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--auto-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --device auto, probe torch CUDA and upgrade to cuda when available.",
    )
    ap.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated features. Default uses the canonical 25-feature set.",
    )
    ap.add_argument("--top-k-significance", type=int, default=5)
    ap.add_argument("--single-feature-epochs", type=int, default=20)
    ap.add_argument("--topology-decorrelation-weight", type=float, default=0.0)
    ap.add_argument("--topology-decorrelation-features", default="M_Jet1Jet2,dRJet12")
    ap.add_argument("--topology-decorrelation-min-signal", type=int, default=16)
    ap.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reweight signal/background to contribute equally to the training loss.",
    )
    ap.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help="Class-balance strength in [0,1], passed through to the trainer.",
    )
    ap.add_argument("--allow-missing-run", action="store_true")
    ap.add_argument("--date-dir", default=None, help="Optional fixed date dir like 4-2")
    ap.add_argument("--clean-output", action="store_true", help="Clean existing outputs for this save-name")
    args = ap.parse_args()

    effective_device = str(args.device)
    if str(args.device) == "auto" and bool(args.auto_gpu):
        if _torch_cuda_available(sys.executable):
            effective_device = "cuda"
            print("[INFO] CUDA detected in current python environment, using --device cuda")
        else:
            print("[INFO] CUDA not detected in current python environment, keeping --device auto")

    feature_csv = str(args.features) if args.features else get_default_feature_csv()

    repo_root = Path(__file__).resolve().parent
    target = repo_root / "dnn" / "train_from_eos_outputs.py"

    cmd = [
        sys.executable,
        str(target),
        "--base-dir",
        str(args.base_dir),
        "--year",
        str(args.year),
        "--region",
        str(args.region),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-events-per-sample",
        str(args.max_events_per_sample),
        "--device",
        str(effective_device),
        "--features",
        str(feature_csv),
        "--top-k-significance",
        str(args.top_k_significance),
        "--single-feature-epochs",
        str(args.single_feature_epochs),
        "--topology-decorrelation-weight",
        str(args.topology_decorrelation_weight),
        "--topology-decorrelation-features",
        str(args.topology_decorrelation_features),
        "--topology-decorrelation-min-signal",
        str(args.topology_decorrelation_min_signal),
        "--tag",
        str(args.save_name),
    ]
    if args.balance_classes:
        train_cmd.append("--balance-classes")
    else:
        train_cmd.append("--no-balance-classes")
    train_cmd.extend(["--balance-strength", str(args.balance_strength)])

    if _is_signal_prefix(args.signal):
        cmd.extend(["--signal-prefix", str(args.signal)])
    else:
        cmd.extend(["--signal-category", str(args.signal)])

    if args.allow_missing_run:
        cmd.append("--allow-missing-run")
    if args.date_dir:
        cmd.extend(["--date-dir", str(args.date_dir)])
    if args.clean_output:
        cmd.append("--clean-output")

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
