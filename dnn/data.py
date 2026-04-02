"""Dataset reading helpers for DNN training.

Purpose:
- Enumerate per-sample trees inside the ppbbchichi output ROOT file.
- Read requested branches into numpy arrays with basic validation.

This module intentionally stays small so both training and inference scripts can
reuse the same ROOT IO logic.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import uproot


def list_sample_region_trees(root_file: uproot.ReadOnlyFile, region: str) -> list[tuple[str, str]]:
    """Return list of (sample, tree_path) for trees ending with '/<region>'."""

    out: list[tuple[str, str]] = []
    classmap = root_file.classnames()
    suffix = f"/{region}"
    for k, cls in classmap.items():
        if cls != "TTree":
            continue
        path = k.split(";")[0]
        if not path.endswith(suffix):
            continue
        sample = path.split("/")[0]
        out.append((sample, path))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def read_tree_as_arrays(
    root_file: uproot.ReadOnlyFile,
    tree_path: str,
    branches: list[str],
    max_events: int | None,
) -> dict:
    tree = root_file[tree_path]
    available = set(map(str, tree.keys()))

    present = [b for b in branches if b in available]
    missing = [b for b in branches if b not in available]
    if missing:
        raise KeyError(f"Tree '{tree_path}' missing branches: {missing}")

    arrays = tree.arrays(present, library="np")
    n = len(next(iter(arrays.values()))) if arrays else 0
    if max_events is not None:
        n = min(n, int(max_events))
        arrays = {k: np.asarray(v)[:n] for k, v in arrays.items()}
    return arrays
