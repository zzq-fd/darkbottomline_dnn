#!/usr/bin/env python3
"""Train a classifier using ppbbchichi output trees.

Purpose:
- Read trees from `ppbbchichi-trees.root` (per sample / region).
- Build a feature matrix from selected branches and train a binary classifier.
- Write the trained model + training metadata (AUC, features) to an output directory.

Notes:
- This script trains a small tabular DNN (MLP) using PyTorch.
- Output artifacts are written to an output directory (checkpoint, scaler, metrics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import csv

import numpy as np
import uproot

try:
    # When executed as a module: `python -m dnn.train_classifier ...`
    from dnn.common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from dnn.data import list_sample_region_trees, read_tree_as_arrays
    from dnn.model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint
    from dnn.scaler import StandardScaler
except ModuleNotFoundError:
    # When executed as a script: `python dnn/train_classifier.py ...`
    from common import DEFAULT_SIGNAL_PATTERNS, is_signal, sanitize_feature_frame
    from data import list_sample_region_trees, read_tree_as_arrays
    from model import ModelSpec, build_mlp, parse_hidden_layers, save_checkpoint
    from scaler import StandardScaler


DEFAULT_FEATURES = [
    "puppiMET_pt",
    "DeltaPhi_j1MET",
    "DeltaPhi_j2MET",
    "lead_bjet_PNetB",
    "sublead_bjet_PNetB",
    "lead_bjet_pt",
    "sublead_bjet_pt",
    "dijet_mass",
    "dijet_pt",
    "n_jets",
    "lepton1_pt",
]


def _load_label_csv(path: str) -> dict[str, int]:
    """Load sample->label mapping from a CSV.

    Expected columns: sample,label
    - sample: top-level directory name inside ppbbchichi-trees.root
    - label: 0 (background) or 1 (signal)
    """
    mapping: dict[str, int] = {}
    with open(path, "r", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "sample" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("label CSV must have columns: sample,label")
        for row in reader:
            s = str(row["sample"]).strip()
            if not s:
                continue
            v = int(row["label"])
            if v not in (0, 1):
                raise ValueError(f"Invalid label for sample '{s}': {v} (expected 0 or 1)")
            mapping[s] = v
    if not mapping:
        raise ValueError(f"No labels loaded from: {path}")
    return mapping


def _drop_constant_features(df, *, missing_sentinel: float = -9999.0, eps: float = 1e-12):
    """Drop near-constant features (common for LHEF-derived btag placeholders).

    Uses std computed on finite, non-sentinel values.
    Returns: (df_kept, kept_features, dropped_features)
    """
    import numpy as _np

    kept: list[str] = []
    dropped: list[str] = []
    for c in list(df.columns):
        a = _np.asarray(df[c].to_numpy(), dtype="f8")
        m = _np.isfinite(a) & (a != float(missing_sentinel))
        s = float(_np.std(a[m])) if _np.any(m) else 0.0
        if s <= float(eps):
            dropped.append(str(c))
        else:
            kept.append(str(c))

    if not kept:
        raise ValueError("All features were dropped as (near-)constant; cannot train.")

    return df[kept], kept, dropped


def _resolve_device(arg: str):
    import torch

    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {arg}")


def main():
    ap = argparse.ArgumentParser(description="Train a tabular DNN classifier for pp→bbχχ using ppbbchichi output trees.")
    ap.add_argument("--root", required=True, help="Path to ppbbchichi-trees.root")
    ap.add_argument("--region", default="preselection", help="Region tree to use for training")
    ap.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature branch names",
    )
    ap.add_argument(
        "--signal-pattern",
        action="append",
        default=None,
        help="Regex to identify signal sample names (repeatable). Default matches: " + ", ".join(DEFAULT_SIGNAL_PATTERNS),
    )
    ap.add_argument(
        "--label-csv",
        default=None,
        help="Optional CSV mapping sample->label with columns 'sample,label'. If provided, overrides --signal-pattern.",
    )
    ap.add_argument("--max-events-per-sample", type=int, default=200000, help="Cap events per sample for training")
    ap.add_argument("--test-size", type=float, default=0.3, help="Fraction for test split")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="outputs_dnn", help="Output directory")
    ap.add_argument("--hidden", default="128,128", help="Hidden layer sizes, e.g. '128,128'")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-clip", type=float, default=100.0, help="Clip event weights to this maximum")
    ap.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--drop-constant-features",
        action="store_true",
        help="Drop near-constant features automatically (useful for LHEF inputs with placeholder btag).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise ValueError("No features specified")

    region = args.region
    weight_branch = f"weight_{region}"
    patterns = tuple(args.signal_pattern) if args.signal_pattern else DEFAULT_SIGNAL_PATTERNS
    label_map = _load_label_csv(args.label_csv) if args.label_csv else None

    # Read dataset
    X_parts = []
    y_parts = []
    w_parts = []
    sample_parts = []

    with uproot.open(args.root) as f:
        trees = list_sample_region_trees(f, region)
        if not trees:
            raise FileNotFoundError(f"No trees found for region '{region}' in {args.root}")

        for sample, tpath in trees:
            branches = features + [weight_branch]
            arrs = read_tree_as_arrays(f, tpath, branches=branches, max_events=args.max_events_per_sample)

            # Build X
            import pandas as pd

            df = pd.DataFrame({k: np.asarray(arrs[k], dtype="f8") for k in features})
            df = sanitize_feature_frame(df)

            w = np.asarray(arrs[weight_branch], dtype="f8")
            w = np.where(np.isfinite(w), w, 0.0)
            # Training with negative weights is ill-defined for BCE; clamp to 0.
            w = np.maximum(w, 0.0)
            if args.weight_clip is not None:
                w = np.minimum(w, float(args.weight_clip))

            if label_map is not None:
                if sample not in label_map:
                    raise KeyError(
                        f"Sample '{sample}' not found in --label-csv mapping. "
                        "Add it to the CSV or remove --label-csv to use --signal-pattern."
                    )
                label = int(label_map[sample])
            else:
                label = 1 if is_signal(sample, patterns) else 0
            y = np.full(df.shape[0], label, dtype=np.int8)

            X_parts.append(df)
            y_parts.append(y)
            w_parts.append(w)
            sample_parts.append(np.array([sample] * df.shape[0], dtype=object))

    import pandas as pd

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)
    _sample_names = np.concatenate(sample_parts)

    # Optionally drop constant features (common for LHEF placeholder branches)
    if args.drop_constant_features:
        X, kept, dropped = _drop_constant_features(X, missing_sentinel=-9999.0)
        if dropped:
            print(f"[INFO] Dropped near-constant features: {dropped}")
        features = kept

    if np.unique(y).size < 2:
        raise ValueError(
            "Training labels contain only one class. "
            "Check --signal-pattern and the sample names inside the ROOT file."
        )
    if not np.isfinite(w).all():
        w = np.where(np.isfinite(w), w, 0.0)
    if float(np.sum(w)) <= 0.0:
        raise ValueError("All event weights are zero after sanitation/clipping; cannot train.")

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y,
    )

    # Convert to numpy arrays and fit scaler on TRAIN only
    X_train_np = X_train.to_numpy(dtype="f8")
    X_test_np = X_test.to_numpy(dtype="f8")
    scaler = StandardScaler.fit(X_train_np, missing_sentinel=-9999.0)
    X_train_np = scaler.transform(X_train_np).astype("float32")
    X_test_np = scaler.transform(X_test_np).astype("float32")

    # Torch setup
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(args.seed))
    device = _resolve_device(args.device)

    spec = ModelSpec(
        n_inputs=int(X_train_np.shape[1]),
        hidden_layers=parse_hidden_layers(args.hidden),
        dropout=float(args.dropout),
    )
    model = build_mlp(spec).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    Xtr = torch.from_numpy(X_train_np)
    ytr = torch.from_numpy(np.asarray(y_train, dtype="float32"))
    wtr = torch.from_numpy(np.asarray(w_train, dtype="float32"))

    Xte = torch.from_numpy(X_test_np)
    yte = torch.from_numpy(np.asarray(y_test, dtype="float32"))
    wte = torch.from_numpy(np.asarray(w_test, dtype="float32"))

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr, wtr),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )

    # Train with early stopping on weighted AUC
    from sklearn.metrics import roc_auc_score, roc_curve

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss_vec = bce(logits, yb)
            wnorm = wb / (wb.mean() + 1e-12)
            loss = (loss_vec * wnorm).mean()
            loss.backward()
            optim.step()

            running += float(loss.detach().cpu().item())
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(Xte.to(device)).squeeze(1)
            y_score = torch.sigmoid(logits).cpu().numpy()

        auc = float(roc_auc_score(np.asarray(y_test, dtype="int8"), y_score, sample_weight=np.asarray(w_test, dtype="f8")))
        avg_loss = running / max(1, n_batches)

        print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f}  auc={auc:.6f}  device={device}")

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if int(args.patience) > 0 and bad_epochs >= int(args.patience):
                print(f"[EarlyStop] No AUC improvement for {bad_epochs} epochs")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation + ROC curve
    model.eval()
    with torch.no_grad():
        logits = model(Xte.to(device)).squeeze(1)
        y_score = torch.sigmoid(logits).cpu().numpy()

    auc = float(roc_auc_score(np.asarray(y_test, dtype="int8"), y_score, sample_weight=np.asarray(w_test, dtype="f8")))
    fpr, tpr, thr = roc_curve(np.asarray(y_test, dtype="int8"), y_score, sample_weight=np.asarray(w_test, dtype="f8"))

    # Save artifacts
    model_path = outdir / "dnn_model.pt"
    save_checkpoint(str(model_path), model=model, spec=spec)
    (outdir / "scaler.json").write_text(json.dumps(scaler.to_jsonable(), indent=2) + "\n")

    # Evaluate
    from sklearn.metrics import roc_auc_score, roc_curve

    metrics = {
        "root": str(args.root),
        "region": region,
        "features": features,
        "weight_branch": weight_branch,
        "signal_patterns": list(patterns),
        "model_type": "pytorch_mlp",
        "model_path": str(model_path),
        "scaler_path": str(outdir / "scaler.json"),
        "model_spec": {
            "n_inputs": int(spec.n_inputs),
            "hidden_layers": list(spec.hidden_layers),
            "dropout": float(spec.dropout),
        },
        "auc": auc,
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
    }

    (outdir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (outdir / "features.json").write_text(json.dumps(features, indent=2) + "\n")

    # ROC plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC ({region})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc.png", dpi=150)

    print(f"[OK] Trained DNN model. AUC={auc:.4f}")
    print(f"[OK] Wrote: {model_path}")
    print(f"[OK] Wrote: {outdir / 'train_metrics.json'}")


if __name__ == "__main__":
    main()
