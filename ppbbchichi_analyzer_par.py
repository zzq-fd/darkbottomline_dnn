"""Main analyzer to build bbχχ trees and histograms from input files.

Purpose:
- Read input ROOT/Parquet (MC, data, and data-driven templates).
- Build event-level variables, apply region selections, and compute weights.
- Write histogram ROOT output and per-sample per-region trees (`ppbbchichi-trees.root`).
"""

import os
import re
import time
import argparse
from pathlib import Path

import numpy as np
import awkward as ak
import pyarrow.parquet as pq

import uproot

try:
    import ROOT  # type: ignore

    ROOT.gROOT.SetBatch(True)
    ROOT.TH1.AddDirectory(False)
except ModuleNotFoundError:
    ROOT = None

from config.config import RunConfig
from config.utils import lVector
from normalisation import getXsec, getLumi

from bbchichi_regions import (
    get_mask_preselection,
    get_mask_srbbchichi,
    get_mask_cr_antibtag,
    get_mask_cr_lowmet,
    get_mask_cr_lepton,
)
from bbchichi_variables import vardict, regions, variables_common
from bbchichi_binning import binning


def _ensure_1d(a):
    a = np.asarray(a)
    return a.ravel()


def make_th1_pyroot(values, weights, name, title, binning_):
    if ROOT is None:
        raise RuntimeError(
            "PyROOT (module 'ROOT') is not available. "
            "Install ROOT or run ppbbchichi_analyzer_par.py with --skip-hists."
        )
    v = _ensure_1d(values)
    w = None if weights is None else _ensure_1d(weights)

    if w is not None:
        m = np.isfinite(v) & np.isfinite(w)
        v = v[m]
        w = w[m]
    else:
        v = v[np.isfinite(v)]

    # build edges
    if isinstance(binning_, (list, tuple)) and len(binning_) == 3 and all(np.isscalar(x) for x in binning_):
        nb, lo, hi = int(binning_[0]), float(binning_[1]), float(binning_[2])
        edges = np.linspace(lo, hi, nb + 1, dtype="f8")
        h = ROOT.TH1D(name, title, nb, lo, hi)
    else:
        edges = np.asarray(binning_, dtype="f8").ravel()
        if edges.size < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"[{name}] invalid variable bin edges")
        h = ROOT.TH1D(name, title, int(edges.size - 1), edges)

    if v.dtype == np.bool_:
        v = v.astype("f8")
    if w is not None and w.dtype == np.bool_:
        w = w.astype("f8")

    counts, _ = np.histogram(v, bins=edges, weights=w)
    if w is None:
        sumw2 = counts.astype("f8")
    else:
        sumw2, _ = np.histogram(v, bins=edges, weights=w * w)

    h.Sumw2()
    for i in range(1, h.GetNbinsX() + 1):
        c = float(counts[i - 1])
        e2 = float(sumw2[i - 1])
        h.SetBinContent(i, c)
        h.SetBinError(i, float(np.sqrt(e2) if e2 >= 0 else 0.0))

    h.SetDirectory(0)
    return h


def ensure_dir_in_tfile(tfile, path):
    curr = tfile
    if not path:
        return curr
    for part in path.split("/"):
        d = curr.GetDirectory(part)
        curr = d if d else curr.mkdir(part)
    return curr


def normalize_sample_name(name: str) -> str:
    base = os.path.basename(name)
    base = re.sub(r"\.(parquet|root)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"(_part\d+|_chunk\d+|_\d+of\d+)$", "", base, flags=re.IGNORECASE)
    return base


def sample_name_from_path(path: str) -> str:
    """Derive a stable sample name.

    For MadGraph scans, many files are named identically (e.g. unweighted_events.root).
    In that case we take the parameter-point directory name (e.g. Events_*).
    """
    p = Path(path)

    # EOS EVENTSELECTION production convention:
    #   .../<SampleName>_EVENTSELECTION/<uuid>.root
    # where the ROOT file name is not meaningful. Use the dataset directory.
    if "EVENTSELECTION" in p.parent.name.upper():
        return p.parent.name

    base = normalize_sample_name(p.name)
    if base.lower() == "unweighted_events":
        # Typical structure: .../results/Events_<params>/run_01/unweighted_events.root
        if len(p.parents) >= 2:
            cand = p.parents[1].name
            if cand.startswith("Events_"):
                return cand
        # fallback: nearest parent that looks like a point name
        for parent in p.parents:
            if parent.name.startswith("Events_"):
                return parent.name
    return base


def is_signal_from_name(name: str) -> bool:
    """Heuristic signal/background flag from a sample or file name.

    Rules (in order):
    - Data-like names (Run20XX* / data*) are never treated as signal.
    - For EVENTSELECTION-style samples: signal-like physics keywords are treated
      as signal (e.g. diboson, bbchichi, dark, invisible, dm).
    - Otherwise: keyword-based matching (legacy).
    """
    s = os.path.basename(str(name))
    s_low = s.lower()
    if is_data_from_name(s_low):
        return False
    if "eventselection" in s_low:
        signal_keys = ("diboson", "chichi", "bbchichi", "dm", "dark", "invisible")
        return any(k in s_low for k in signal_keys)
    return any(k in s_low for k in ("chichi", "bbchichi", "dm", "dark", "invisible"))


def is_data_from_name(name: str) -> bool:
    s = os.path.basename(str(name)).lower()
    # Common CMS data naming: Run20XX*
    if s.startswith("run"):
        return True
    # Fallback: explicit 'data' tokens
    return ("data" in s) or ("_data" in s) or ("data_" in s)


def _build_bbchichi_from_eventselection(tree_: ak.Array, *, file_event_offset: int = 0) -> ak.Array:
    """Map EVENTSELECTION ROOT schema (Events tree) to internal bbχχ event record.

    Expected branches:
      - PFMET_pt, PFMET_phi
      - jet_pt, jet_eta, jet_phi, jet_btag

    Optional branches:
      - run, luminosityBlock, event
      - n_jets
      - muon_pt, electron_pt
      - weight
    """
    fields = set(tree_.fields)
    required = {"PFMET_pt", "PFMET_phi", "jet_pt", "jet_eta", "jet_phi", "jet_btag"}
    missing = sorted(required - fields)
    if missing:
        raise KeyError(
            "EVENTSELECTION ROOT input is missing required branches:\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    n_entries = len(tree_)

    # run/lumi/event
    run = tree_["run"] if "run" in fields else ak.Array(np.full(n_entries, 1, dtype=np.int64))
    lumi = tree_["luminosityBlock"] if "luminosityBlock" in fields else ak.Array(np.full(n_entries, 1, dtype=np.int64))
    if "event" in fields:
        event = tree_["event"]
    else:
        event = ak.Array(np.arange(int(file_event_offset), int(file_event_offset) + n_entries, dtype=np.int64))

    # MET
    met_pt = tree_["PFMET_pt"]
    met_phi = tree_["PFMET_phi"]

    # Jets -> select 2 "b jets" as the two highest btag jets
    jet_pt = tree_["jet_pt"]
    jet_eta = tree_["jet_eta"]
    jet_phi = tree_["jet_phi"]
    jet_btag = tree_["jet_btag"]

    order = ak.argsort(jet_btag, axis=1, ascending=False)
    jet_pt = jet_pt[order]
    jet_eta = jet_eta[order]
    jet_phi = jet_phi[order]
    jet_btag = jet_btag[order]

    def _get2(arr, idx, fill):
        return ak.fill_none(ak.pad_none(arr, 2, axis=1, clip=True)[:, idx], fill)

    lead_bjet_pt = _get2(jet_pt, 0, -9999.0)
    lead_bjet_eta = _get2(jet_eta, 0, -9999.0)
    lead_bjet_phi = _get2(jet_phi, 0, -9999.0)
    sublead_bjet_pt = _get2(jet_pt, 1, -9999.0)
    sublead_bjet_eta = _get2(jet_eta, 1, -9999.0)
    sublead_bjet_phi = _get2(jet_phi, 1, -9999.0)

    lead_bjet_btag = _get2(jet_btag, 0, -9999.0)
    sublead_bjet_btag = _get2(jet_btag, 1, -9999.0)

    # n_jets (prefer provided scalar, else compute)
    if "n_jets" in fields:
        n_jets = tree_["n_jets"]
    else:
        n_jets = ak.values_astype(ak.num(tree_["jet_pt"], axis=1), np.int64)

    # lepton1_pt = max(leading muon, leading electron)
    if "muon_pt" in fields:
        mu_max = ak.fill_none(ak.max(tree_["muon_pt"], axis=1), -9999.0)
    else:
        mu_max = ak.Array(np.full(n_entries, -9999.0, dtype=np.float64))
    if "electron_pt" in fields:
        el_max = ak.fill_none(ak.max(tree_["electron_pt"], axis=1), -9999.0)
    else:
        el_max = ak.Array(np.full(n_entries, -9999.0, dtype=np.float64))
    lepton1_pt = ak.where(mu_max > el_max, mu_max, el_max)

    # weight (optional)
    weight = tree_["weight"] if "weight" in fields else ak.Array(np.ones(n_entries, dtype=np.float64))

    ev = ak.zip(
        {
            "run": run,
            "lumi": lumi,
            "event": event,
            "puppiMET_pt": met_pt,
            "puppiMET_phi": met_phi,
            "lead_bjet_pt": lead_bjet_pt,
            "lead_bjet_eta": lead_bjet_eta,
            "lead_bjet_phi": lead_bjet_phi,
            "lead_bjet_mass": ak.Array(np.full(n_entries, 0.0, dtype=np.float64)),
            "sublead_bjet_pt": sublead_bjet_pt,
            "sublead_bjet_eta": sublead_bjet_eta,
            "sublead_bjet_phi": sublead_bjet_phi,
            "sublead_bjet_mass": ak.Array(np.full(n_entries, 0.0, dtype=np.float64)),
            # treat jet_btag as PNetB-like score
            "lead_bjet_PNetB": lead_bjet_btag,
            "sublead_bjet_PNetB": sublead_bjet_btag,
            "weight": weight,
            "n_jets": n_jets,
            "lepton1_pt": lepton1_pt,
        },
        depth_limit=1,
    )
    return ev


def is_dd_template(path: str) -> bool:
    b = normalize_sample_name(os.path.basename(path))
    if "rescaled" in b:
        return True
    # Avoid matching random "dd" substrings in UUID/hash-like fragments.
    return bool(re.search(r"(^|[_-])(dd|data_driven|datadriven|fake|fakes)([_-]|$)", b))


DD_WEIGHT_COLUMNS = ("weight", "evt_weight", "w", "fake_weight")


def _get_dd_weight_col(all_columns) -> str:
    cols = set(map(str, all_columns))
    for c in DD_WEIGHT_COLUMNS:
        if c in cols:
            return c
    raise KeyError(
        "No DD weight column found in DD template. "
        f"Tried: {', '.join(DD_WEIGHT_COLUMNS)}; available: {sorted(cols)}"
    )


def ak_to_numpy_dict(arr: ak.Array) -> dict:
    out = {}
    for key in arr.fields:
        filled = ak.fill_none(arr[key], -9999)
        np_arr = ak.to_numpy(filled)
        if np.issubdtype(np_arr.dtype, np.integer):
            np_arr = np.nan_to_num(np_arr.astype("int64"), nan=-9999, posinf=999999999, neginf=-999999999)
        else:
            np_arr = np.nan_to_num(np_arr.astype("float64"), nan=-9999.0, posinf=9.999e306, neginf=-9.999e306)
        out[key] = np_arr
    return out


def sanitize_for_uproot(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        a = np.asarray(v)
        if a.dtype == np.bool_:
            a = a.astype("int8")
        elif a.dtype.kind in ("i", "u"):
            a = a.astype("int64")
        elif a.dtype.kind == "f":
            a = a.astype("float64")
        else:
            a = a.astype("float64")
        if a.dtype.kind in ("i", "u"):
            a = np.nan_to_num(a, nan=-9999, posinf=999999999, neginf=-999999999)
        else:
            a = np.nan_to_num(a, nan=-9999.0, posinf=9.999e306, neginf=-9.999e306)
        out[k] = a
    return out


def _ensure_tree(upfile, treedir, treename, first_piece):
    curr = upfile
    if treedir:
        for part in [p for p in treedir.split("/") if p]:
            curr = curr.mkdir(part) if part not in curr.keys() else curr[part]

    def _btype(a):
        a = np.asarray(a)
        if a.dtype == np.bool_:
            return "int8"
        if a.dtype.kind in ("i", "u"):
            return "int64"
        if a.dtype.kind == "f":
            return "float64"
        raise TypeError(f"Unsupported dtype for branch: {a.dtype}")

    types = {k: _btype(v) for k, v in first_piece.items()}
    return curr.mktree(treename, types)


HIST_CACHE = {}
TREE_HANDLES = {}
PROC_TREE = {}


def _delta_phi(phi1, phi2):
    d = np.abs(phi1 - phi2)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def _missing_columns(parquet_file: pq.ParquetFile, required: list[str]) -> list[str]:
    cols = set(parquet_file.schema.names)
    return [c for c in required if c not in cols]


def _find_first_ttree_name(upfile: uproot.ReadOnlyFile) -> str:
    """Return the first TTree key name (without ;cycle)."""
    classmap = upfile.classnames()
    for k, cls in classmap.items():
        if cls == "TTree":
            return k.split(";")[0]
    raise KeyError("No TTree found in ROOT file")


def _iterate_root_batches(root_path: str, expressions: list[str], tree_name: str | None, step_size: int = 10000):
    with uproot.open(root_path) as fin:
        tname = tree_name or _find_first_ttree_name(fin)
        tree = fin[tname]
        available = set(map(str, tree.keys()))

        present = [e for e in expressions if e in available]
        missing = [e for e in expressions if e not in available]
        if missing:
            # Do not hard-fail here; the caller may fill some optional columns.
            pass

        for arrays in tree.iterate(present, step_size=step_size, library="ak"):
            # uproot may return either dict[str, ak.Array] or an ak.Array record
            if isinstance(arrays, dict):
                yield ak.zip(arrays, depth_limit=1)
            else:
                yield arrays


def _iterate_parquet_batches(parquet_path: str, columns: list[str], batch_size: int = 10000):
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield ak.from_arrow(batch)


def _safe_eta(px, py, pz):
    pt = np.sqrt(px * px + py * py)
    # eta = asinh(pz/pt) is stable; guard pt=0
    return ak.where(pt != 0, np.arcsinh(pz / pt), 0.0)


def _build_bbchichi_from_lhef(tree_: ak.Array, pid_chi: int = 52) -> ak.Array:
    """Map an LHEF tree batch to the internal bbχχ event record.

    Expects branches like:
      - Particle/Particle.PID, Particle/Particle.Status
      - Particle/Particle.Px/Py/Pz/E
      - Event/Event.Weight, Event/Event.Number (optional)
    """
    fields = set(tree_.fields)
    required = {
        "Particle/Particle.PID",
        "Particle/Particle.Status",
        "Particle/Particle.Px",
        "Particle/Particle.Py",
        "Particle/Particle.Pz",
        "Particle/Particle.E",
    }
    missing = sorted(required - fields)
    if missing:
        raise KeyError(
            "LHEF ROOT input is missing required branches:\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    pid = tree_["Particle/Particle.PID"]
    status = tree_["Particle/Particle.Status"]
    px = tree_["Particle/Particle.Px"]
    py = tree_["Particle/Particle.Py"]
    pz = tree_["Particle/Particle.Pz"]
    e = tree_["Particle/Particle.E"]

    is_final = status == 1
    is_b = is_final & (np.abs(pid) == 5)
    is_chi = is_final & (np.abs(pid) == int(pid_chi))

    # b-quarks (expect 2)
    b_px = px[is_b]
    b_py = py[is_b]
    b_pz = pz[is_b]
    b_e = e[is_b]
    b_pt = np.sqrt(b_px * b_px + b_py * b_py)
    b_phi = np.arctan2(b_py, b_px)
    b_eta = _safe_eta(b_px, b_py, b_pz)

    # order by pt
    order = ak.argsort(b_pt, axis=1, ascending=False)
    b_px = b_px[order]
    b_py = b_py[order]
    b_pz = b_pz[order]
    b_e = b_e[order]
    b_pt = b_pt[order]
    b_phi = b_phi[order]
    b_eta = b_eta[order]

    # if something odd happens, fill with -9999
    def _get2(arr, idx, fill):
        return ak.fill_none(ak.pad_none(arr, 2, axis=1, clip=True)[:, idx], fill)

    lead_bjet_px = _get2(b_px, 0, 0.0)
    lead_bjet_py = _get2(b_py, 0, 0.0)
    lead_bjet_pz = _get2(b_pz, 0, 0.0)
    lead_bjet_e = _get2(b_e, 0, 0.0)
    sublead_bjet_px = _get2(b_px, 1, 0.0)
    sublead_bjet_py = _get2(b_py, 1, 0.0)
    sublead_bjet_pz = _get2(b_pz, 1, 0.0)
    sublead_bjet_e = _get2(b_e, 1, 0.0)

    lead_bjet_pt = _get2(b_pt, 0, -9999.0)
    lead_bjet_eta = _get2(b_eta, 0, -9999.0)
    lead_bjet_phi = _get2(b_phi, 0, -9999.0)
    sublead_bjet_pt = _get2(b_pt, 1, -9999.0)
    sublead_bjet_eta = _get2(b_eta, 1, -9999.0)
    sublead_bjet_phi = _get2(b_phi, 1, -9999.0)

    # dijet 4-vector
    dijet_px = lead_bjet_px + sublead_bjet_px
    dijet_py = lead_bjet_py + sublead_bjet_py
    dijet_pz = lead_bjet_pz + sublead_bjet_pz
    dijet_e = lead_bjet_e + sublead_bjet_e
    dijet_pt = np.sqrt(dijet_px * dijet_px + dijet_py * dijet_py)
    dijet_phi = np.arctan2(dijet_py, dijet_px)
    dijet_eta = _safe_eta(dijet_px, dijet_py, dijet_pz)
    m2 = dijet_e * dijet_e - (dijet_px * dijet_px + dijet_py * dijet_py + dijet_pz * dijet_pz)
    dijet_mass = np.sqrt(np.maximum(m2, 0.0))

    # chi chi -> MET proxy (vector sum)
    chi_px = px[is_chi]
    chi_py = py[is_chi]
    met_px = ak.sum(chi_px, axis=1)
    met_py = ak.sum(chi_py, axis=1)
    met_pt = np.sqrt(met_px * met_px + met_py * met_py)
    met_phi = np.arctan2(met_py, met_px)

    # event id & weight (optional)
    n_entries = len(tree_)
    if "Event/Event.Number" in fields:
        # These branches come as var-len (usually length=1). Convert to scalar per event.
        event_number = ak.fill_none(ak.firsts(tree_["Event/Event.Number"]), -1)
    else:
        event_number = ak.Array(np.arange(n_entries, dtype=np.int64))

    if "Event/Event.Weight" in fields:
        weight = ak.fill_none(ak.firsts(tree_["Event/Event.Weight"]), 1.0)
    else:
        weight = ak.Array(np.ones(n_entries, dtype=np.float64))

    # build internal record
    ev = ak.zip(
        {
            "run": ak.Array(np.full(n_entries, 1, dtype=np.int64)),
            "lumi": ak.Array(np.full(n_entries, 1, dtype=np.int64)),
            "event": event_number,
            "puppiMET_pt": met_pt,
            "puppiMET_phi": met_phi,
            "lead_bjet_pt": lead_bjet_pt,
            "lead_bjet_eta": lead_bjet_eta,
            "lead_bjet_phi": lead_bjet_phi,
            "lead_bjet_mass": ak.Array(np.full(n_entries, 0.0, dtype=np.float64)),
            "sublead_bjet_pt": sublead_bjet_pt,
            "sublead_bjet_eta": sublead_bjet_eta,
            "sublead_bjet_phi": sublead_bjet_phi,
            "sublead_bjet_mass": ak.Array(np.full(n_entries, 0.0, dtype=np.float64)),
            # LHEF doesn't have b-tag; set to 1 so selection works by default
            "lead_bjet_PNetB": ak.Array(np.full(n_entries, 1.0, dtype=np.float64)),
            "sublead_bjet_PNetB": ak.Array(np.full(n_entries, 1.0, dtype=np.float64)),
            "weight": weight,
            "n_jets": ak.Array(np.full(n_entries, 2, dtype=np.int64)),
            "lepton1_pt": ak.Array(np.full(n_entries, -9999.0, dtype=np.float64)),
            "dijet_mass": dijet_mass,
            "dijet_pt": dijet_pt,
            "dijet_eta": dijet_eta,
            "dijet_phi": dijet_phi,
        },
        depth_limit=1,
    )
    return ev


def process_input_file(
    inputfile,
    cli_year,
    cli_era,
    xsec_lumi_cache,
    out_files,
    tree_name: str | None = None,
    pid_chi: int = 52,
    make_hists: bool = True,
):
    ext = str(inputfile).lower()
    is_root = ext.endswith(".root")
    is_parquet = ext.endswith(".parquet")
    if not (is_root or is_parquet):
        print(f"[WARN] Skipping unsupported input: {inputfile}")
        return

    print(f"[INFO] Processing {'ROOT' if is_root else 'parquet'}: {inputfile}")

    lhef_mode = False
    lhef_tree_name = None
    schema_kind = "parquet" if is_parquet else "unknown"
    if is_root:
        with uproot.open(inputfile) as fin:
            lhef_tree_name = tree_name or _find_first_ttree_name(fin)
            tree = fin[lhef_tree_name]
            available = set(map(str, tree.keys()))
            lhef_mode = "Particle/Particle.PID" in available and "Particle/Particle.Px" in available

            if lhef_mode:
                schema_kind = "lhef"
            else:
                hh_required = {
                    "puppiMET_pt",
                    "puppiMET_phi",
                    "Res_lead_bjet_pt",
                    "Res_lead_bjet_eta",
                    "Res_lead_bjet_phi",
                    "Res_lead_bjet_mass",
                    "Res_sublead_bjet_pt",
                    "Res_sublead_bjet_eta",
                    "Res_sublead_bjet_phi",
                    "Res_sublead_bjet_mass",
                    "Res_lead_bjet_btagPNetB",
                    "Res_sublead_bjet_btagPNetB",
                }
                eventsel_required = {"PFMET_pt", "PFMET_phi", "jet_pt", "jet_eta", "jet_phi", "jet_btag"}
                if hh_required.issubset(available):
                    schema_kind = "hh"
                elif eventsel_required.issubset(available):
                    schema_kind = "eventselection"
                else:
                    schema_kind = "unknown"

    # Minimal required schema
    if is_parquet:
        required_columns = [
            "puppiMET_pt",
            "puppiMET_phi",
            "Res_lead_bjet_pt",
            "Res_lead_bjet_eta",
            "Res_lead_bjet_phi",
            "Res_lead_bjet_mass",
            "Res_sublead_bjet_pt",
            "Res_sublead_bjet_eta",
            "Res_sublead_bjet_phi",
            "Res_sublead_bjet_mass",
            "Res_lead_bjet_btagPNetB",
            "Res_sublead_bjet_btagPNetB",
        ]
        optional_columns = [
            "run",
            "lumi",
            "event",
            "weight",
            "n_jets",
            "lepton1_pt",
        ]
        want_columns = required_columns + optional_columns
    elif is_root and not lhef_mode and schema_kind == "eventselection":
        # EVENTSELECTION schema (Run3_bbMET_2HDM_a storage output)
        required_columns = [
            "PFMET_pt",
            "PFMET_phi",
            "jet_pt",
            "jet_eta",
            "jet_phi",
            "jet_btag",
        ]
        optional_columns = [
            "run",
            "luminosityBlock",
            "event",
            "n_jets",
            "muon_pt",
            "electron_pt",
            "weight",
        ]
        want_columns = required_columns + optional_columns
    elif is_root and not lhef_mode:
        # ROOT ntuple in an HH-like nano/parquet-derived schema
        required_columns = [
            "puppiMET_pt",
            "puppiMET_phi",
            "Res_lead_bjet_pt",
            "Res_lead_bjet_eta",
            "Res_lead_bjet_phi",
            "Res_lead_bjet_mass",
            "Res_sublead_bjet_pt",
            "Res_sublead_bjet_eta",
            "Res_sublead_bjet_phi",
            "Res_sublead_bjet_mass",
            "Res_lead_bjet_btagPNetB",
            "Res_sublead_bjet_btagPNetB",
        ]
        optional_columns = [
            "run",
            "lumi",
            "event",
            "weight",
            "n_jets",
            "lepton1_pt",
        ]
        want_columns = required_columns + optional_columns
    else:
        # LHEF mode uses Particle/* branches
        want_columns = [
            "Particle/Particle.PID",
            "Particle/Particle.Status",
            "Particle/Particle.Px",
            "Particle/Particle.Py",
            "Particle/Particle.Pz",
            "Particle/Particle.E",
            "Event/Event.Weight",
            "Event/Event.Number",
        ]

    if is_parquet:
        parquet_file = pq.ParquetFile(inputfile)
        missing = _missing_columns(parquet_file, required_columns)
        if missing:
            raise KeyError(
                "ppbbchichi_analyzer_par.py: missing required parquet columns:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\n\nExpected a schema with resolved b-jet branches (e.g. 'Res_*') and MET branches (e.g. 'puppiMET_*')."
            )

    base = os.path.basename(inputfile)
    sample_name_norm = sample_name_from_path(str(inputfile))

    # EVENTSELECTION samples use UUID-like file names; do not use them for
    # template/data/signal heuristics (they can randomly contain substrings
    # like 'dd' and trigger misclassification).
    base_norm = normalize_sample_name(base)
    base_is_uuid = bool(
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(base_norm),
            flags=re.IGNORECASE,
        )
    )

    isdata = is_data_from_name(sample_name_norm) or (is_data_from_name(base) if not base_is_uuid else False)
    isdd = is_dd_template(sample_name_norm) or (is_dd_template(base) if not base_is_uuid else False)
    sigflag = is_signal_from_name(sample_name_norm) or (is_signal_from_name(base) if not base_is_uuid else False)
    if isdata:
        sigflag = False

    # normalization cache
    if inputfile not in xsec_lumi_cache:
        if isdata or isdd:
            xsec_lumi_cache[inputfile] = (1.0, 1.0)
        else:
            xsec_lumi_cache[inputfile] = (
                float(getXsec(sample_name_norm)),
                float(getLumi(str(cli_year), str(cli_era))) * 1000.0,  # pb^-1
            )

    xsec_, lumi_ = xsec_lumi_cache[inputfile]
    print(
        f"[NORM] sample={sample_name_norm} file={base} xsec={xsec_} pb, lumi={lumi_/1000.0:.3f} fb^-1 "
        f"({cli_year} {cli_era}) [flags: data={isdata} dd={isdd} sig={sigflag}]"
    )

    # iterate batches
    if is_parquet:
        batches = _iterate_parquet_batches(inputfile, want_columns, batch_size=10000)
    elif not lhef_mode:
        batches = _iterate_root_batches(inputfile, want_columns, tree_name=lhef_tree_name, step_size=10000)
    else:
        # LHEF mode: read Particle/* from the selected tree
        def _lhef_batches():
            with uproot.open(inputfile) as fin:
                tree = fin[lhef_tree_name]
                available = set(map(str, tree.keys()))
                present = [c for c in want_columns if c in available]
                for arrays in tree.iterate(present, step_size=10000, library="ak"):
                    if isinstance(arrays, dict):
                        yield ak.zip(arrays, depth_limit=1)
                    else:
                        yield arrays

        batches = _lhef_batches()

    file_event_offset = 0
    for tree_ in batches:
        n_entries = len(tree_)

        if lhef_mode:
            cms_events = _build_bbchichi_from_lhef(tree_, pid_chi=int(pid_chi))
        elif schema_kind == "eventselection":
            cms_events = _build_bbchichi_from_eventselection(tree_, file_event_offset=int(file_event_offset))
        else:
            # optional columns filled if absent
            fields = set(tree_.fields)
            n_jets = tree_["n_jets"] if "n_jets" in fields else ak.Array(np.full(n_entries, -9999, dtype=np.int64))
            lepton1_pt = tree_["lepton1_pt"] if "lepton1_pt" in fields else ak.Array(np.full(n_entries, -9999.0, dtype=np.float64))

            if "run" in fields:
                run = tree_["run"]
            else:
                run = ak.Array(np.full(n_entries, 1, dtype=np.int64))
            if "lumi" in fields:
                lumi = tree_["lumi"]
            else:
                lumi = ak.Array(np.full(n_entries, 1, dtype=np.int64))
            if "event" in fields:
                event = tree_["event"]
            else:
                # Make a deterministic per-file event index
                event = ak.Array(np.arange(file_event_offset, file_event_offset + n_entries, dtype=np.int64))

            cms_events = ak.zip(
                {
                    "run": run,
                    "lumi": lumi,
                    "event": event,
                    "puppiMET_pt": tree_["puppiMET_pt"],
                    "puppiMET_phi": tree_["puppiMET_phi"],

                    "lead_bjet_pt": tree_["Res_lead_bjet_pt"],
                    "lead_bjet_eta": tree_["Res_lead_bjet_eta"],
                    "lead_bjet_phi": tree_["Res_lead_bjet_phi"],
                    "lead_bjet_mass": tree_["Res_lead_bjet_mass"],

                    "sublead_bjet_pt": tree_["Res_sublead_bjet_pt"],
                    "sublead_bjet_eta": tree_["Res_sublead_bjet_eta"],
                    "sublead_bjet_phi": tree_["Res_sublead_bjet_phi"],
                    "sublead_bjet_mass": tree_["Res_sublead_bjet_mass"],

                    "lead_bjet_PNetB": tree_["Res_lead_bjet_btagPNetB"],
                    "sublead_bjet_PNetB": tree_["Res_sublead_bjet_btagPNetB"],

                    # weight may be missing for data; handled below
                    "weight": tree_["weight"] if "weight" in fields else ak.Array(np.ones(n_entries, dtype=np.float64)),

                    "n_jets": n_jets,
                    "lepton1_pt": lepton1_pt,
                },
                depth_limit=1,
            )

        cms_fields = set(cms_events.fields)

        cms_events["signal"] = ak.Array(np.full(n_entries, 1 if sigflag else 0, dtype=np.int8))
        cms_events["isdata"] = ak.Array(np.full(n_entries, 1 if isdata else 0, dtype=np.int8))
        cms_events["isdd"] = ak.Array(np.full(n_entries, 1 if isdd else 0, dtype=np.int8))

        dijet = lVector(
            cms_events["lead_bjet_pt"],
            cms_events["lead_bjet_eta"],
            cms_events["lead_bjet_phi"],
            cms_events["sublead_bjet_pt"],
            cms_events["sublead_bjet_eta"],
            cms_events["sublead_bjet_phi"],
            cms_events["lead_bjet_mass"],
            cms_events["sublead_bjet_mass"],
        )
        cms_events["dijet_mass"] = dijet.mass
        cms_events["dijet_pt"] = dijet.pt
        cms_events["dijet_eta"] = dijet.eta
        cms_events["dijet_phi"] = dijet.phi

        cms_events["DeltaPhi_j1MET"] = _delta_phi(ak.to_numpy(cms_events["lead_bjet_phi"]), ak.to_numpy(cms_events["puppiMET_phi"]))
        cms_events["DeltaPhi_j2MET"] = _delta_phi(ak.to_numpy(cms_events["sublead_bjet_phi"]), ak.to_numpy(cms_events["puppiMET_phi"]))

        # region masks
        cms_events["preselection"] = get_mask_preselection(cms_events)
        cms_events["srbbchichi"] = get_mask_srbbchichi(cms_events)
        cms_events["cr_antibtag"] = get_mask_cr_antibtag(cms_events)
        cms_events["cr_lowmet"] = get_mask_cr_lowmet(cms_events)
        cms_events["cr_lepton"] = get_mask_cr_lepton(cms_events)

        # weights
        if isdata:
            base_w = np.ones(n_entries, dtype="f8")
        elif isdd:
            dd_wname = _get_dd_weight_col(tree_.fields)
            dd_w = ak.to_numpy(tree_[dd_wname])
            base_w = np.where(np.isfinite(dd_w), dd_w, 0.0)
        else:
            if "weight" not in cms_fields:
                raise KeyError(
                    f"MC input '{base}' is missing 'weight' branch/column. "
                    "Either add a weight branch or name the file as Data so it is treated as unweighted."
                )
            w = ak.to_numpy(cms_events["weight"])
            w = np.where(np.isfinite(w), w, 0.0)
            base_w = w * float(xsec_) * float(lumi_)

        # output record: keep it small but complete for bbchichi
        keys_to_copy = [
            "puppiMET_pt",
            "puppiMET_phi",
            "lead_bjet_pt",
            "lead_bjet_eta",
            "lead_bjet_phi",
            "sublead_bjet_pt",
            "sublead_bjet_eta",
            "sublead_bjet_phi",
            "lead_bjet_PNetB",
            "sublead_bjet_PNetB",
            "dijet_mass",
            "dijet_pt",
            "dijet_eta",
            "dijet_phi",
            "DeltaPhi_j1MET",
            "DeltaPhi_j2MET",
            "n_jets",
            "lepton1_pt",
            "preselection",
            "srbbchichi",
            "cr_antibtag",
            "cr_lowmet",
            "cr_lepton",
            "signal",
            "isdata",
            "isdd",
        ]

        out_events = ak.zip(
            {k: cms_events[k] for k in keys_to_copy}
            | {"run": cms_events["run"], "lumi": cms_events["lumi"], "event": cms_events["event"]},
            depth_limit=1,
        )

        # attach per-region weights
        for r in regions:
            out_events = ak.with_field(out_events, base_w, "weight_" + r)

        # processed_events tree
        proc_piece = sanitize_for_uproot(ak_to_numpy_dict(out_events))
        if sample_name_norm not in PROC_TREE:
            PROC_TREE[sample_name_norm] = _ensure_tree(
                out_files["tree"],
                treedir=f"{sample_name_norm}",
                treename="processed_events",
                first_piece=proc_piece,
            )
        PROC_TREE[sample_name_norm].extend(proc_piece)

        # per-region trees + histograms
        for reg in regions:
            thisreg = out_events[out_events[reg]]
            if len(thisreg) == 0:
                continue

            weight_name = "weight_" + reg

            # hists (optional)
            if make_hists:
                for var in variables_common[reg]:
                    hname = vardict[var]
                    vals = ak.to_numpy(thisreg[var])
                    wts = ak.to_numpy(thisreg[weight_name])
                    wts = np.where(np.isfinite(wts), wts, 0.0)
                    h = make_th1_pyroot(vals, wts, hname, hname, binning[reg][var])

                    key = (sample_name_norm, reg, hname)
                    if key not in HIST_CACHE:
                        acc = h.Clone(f"{hname}__acc")
                        acc.Reset()
                        acc.SetDirectory(0)
                        HIST_CACHE[key] = acc
                    HIST_CACHE[key].Add(h)
                    del h

            # trees
            tree_piece = sanitize_for_uproot(ak_to_numpy_dict(thisreg))
            key = (sample_name_norm, reg)
            if key not in TREE_HANDLES:
                TREE_HANDLES[key] = _ensure_tree(
                    out_files["tree"],
                    treedir=f"{sample_name_norm}",
                    treename=reg,
                    first_piece=tree_piece,
                )
            TREE_HANDLES[key].extend(tree_piece)

        file_event_offset += n_entries


def _infer_output_prefix(in_paths: list[Path]) -> str:
    """Infer output file prefix from a single input path.

    Examples:
    - input dir .../3-16/2022   -> prefix "2022_"
    - input dir .../3-16/2022EE -> prefix "2022EE_"
    """
    if len(in_paths) != 1:
        return ""

    p = in_paths[0]
    base = (p.name if p.is_dir() else p.parent.name).strip()
    if re.fullmatch(r"20\d{2}(EE)?", base, flags=re.IGNORECASE):
        return f"{base}_"
    return ""


def main():
    ap = argparse.ArgumentParser(description="pp -> bb chi chi analyzer (parquet), bb+MET")
    ap.add_argument("-i", "--inFile", action="append", help="Parquet/ROOT file or directory. Can be repeated.")
    ap.add_argument("--tree", default=None, help="ROOT TTree name (if input is .root). If omitted, uses the first TTree found.")
    ap.add_argument("--year", required=True, help="e.g. 2022 or 2023")
    ap.add_argument("--era", required=True, help="e.g. PreEE, PostEE, preBPix, postBPix, All")
    ap.add_argument("--tag", default=None, help="If multiple -i are given, output goes to outputfiles/merged/<tag>")
    ap.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional output filename prefix. "
            "Example: --output-prefix 2022 creates 2022_ppbbchichi-trees.root. "
            "If omitted and a single input path is a year-like directory (e.g. 2022, 2022EE), the prefix is auto-inferred."
        ),
    )
    ap.add_argument("--pid-chi", type=int, default=52, help="In LHEF mode: abs(PID) used as chi for MET proxy (default: 52)")
    ap.add_argument(
        "--skip-hists",
        action="store_true",
        help="Skip histogram output (trees only). Useful when PyROOT is unavailable.",
    )
    ap.add_argument(
        "--skip-bad-files",
        action="store_true",
        help="Skip unreadable/corrupted input files instead of aborting the full run.",
    )
    args = ap.parse_args()

    if ROOT is None and not args.skip_hists:
        raise RuntimeError(
            "PyROOT (module 'ROOT') is not available in this Python environment. "
            "Either install ROOT/PyROOT or rerun with --skip-hists."
        )

    cfg = RunConfig(args.year, args.era)

    # discover inputs
    in_paths = []
    if args.inFile:
        in_paths = [Path(p).resolve() for p in args.inFile]
    else:
        in_paths = cfg.raw_paths if hasattr(cfg, "raw_paths") else [Path(cfg.raw_path)]

    if args.tag:
        out_dir = cfg.outputs_root / "merged" / args.tag
    elif len(in_paths) > 1:
        out_dir = cfg.outputs_root / "merged" / f"bbchichi_{cfg.year}_{cfg.era}"
    else:
        out_dir = cfg.outputs_root / f"bbchichi_{cfg.year}_{cfg.era}"
    os.makedirs(out_dir, exist_ok=True)

    # collect parquet/root files
    inputfiles = []
    for p in in_paths:
        if p.is_file():
            if str(p).lower().endswith((".parquet", ".root")):
                inputfiles.append(str(p))
            else:
                print(f"[WARN] Skipping non-parquet: {p}")
        else:
            # Recursive discovery is important for MG5-style outputs (results/**/unweighted_events.root)
            inputfiles.extend([str(x) for x in sorted(p.rglob("*.parquet"))])
            inputfiles.extend([str(x) for x in sorted(p.rglob("*.root"))])

    if not inputfiles:
        raise FileNotFoundError("No .parquet/.root files found in provided inputs")

    n_parquet = sum(1 for x in inputfiles if x.lower().endswith(".parquet"))
    n_root = len(inputfiles) - n_parquet
    print(f"[INFO] Will process {len(inputfiles)} file(s): {n_parquet} parquet, {n_root} root")

    if args.output_prefix is not None:
        output_prefix = str(args.output_prefix).strip()
        if output_prefix and not output_prefix.endswith("_"):
            output_prefix = output_prefix + "_"
    else:
        output_prefix = _infer_output_prefix(in_paths)

    hist_path = os.path.join(out_dir, f"{output_prefix}ppbbchichi-histograms.root")
    tree_path = os.path.join(out_dir, f"{output_prefix}ppbbchichi-trees.root")

    hist_tfile = None if args.skip_hists else ROOT.TFile(hist_path, "RECREATE")
    tree_upfile = uproot.recreate(tree_path)
    out_files = {"hist": hist_tfile, "tree": tree_upfile}

    xsec_lumi_cache = {}
    n_ok = 0
    failed_files = []
    for fn in inputfiles:
        try:
            process_input_file(
                fn,
                args.year,
                args.era,
                xsec_lumi_cache,
                out_files,
                tree_name=args.tree,
                pid_chi=int(args.pid_chi),
                make_hists=not args.skip_hists,
            )
            n_ok += 1
        except Exception as exc:
            if args.skip_bad_files:
                print(f"[WARN] Skipping failed file: {fn}")
                print(f"[WARN] Reason: {exc}")
                failed_files.append((fn, str(exc)))
                continue
            raise

    if n_ok == 0:
        raise RuntimeError("No input files were processed successfully.")

    if not args.skip_hists:
        print("[INFO] Writing accumulated histograms...")
        for (sample, reg, hname), h in HIST_CACHE.items():
            ensure_dir_in_tfile(out_files["hist"], f"{sample}/{reg}").cd()
            h_clone = h.Clone(hname)
            h_clone.SetDirectory(ROOT.gDirectory)
            h_clone.Write()
            del h_clone

    out_files["tree"].close()
    if out_files["hist"] is not None:
        out_files["hist"].Write()
        out_files["hist"].Close()

    print(f"[OK] Wrote trees to:      {tree_path}")
    if not args.skip_hists:
        print(f"[OK] Wrote histograms to: {hist_path}")
    else:
        print("[OK] Histograms skipped (--skip-hists)")
    if failed_files:
        print(f"[WARN] Failed files skipped: {len(failed_files)}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Execution time: {end - start:.2f} s")
