# pp → bbχχ (bb + MET) analysis overview

This repository is cleaned to support only the **pp → bbχχ** analysis.

## Inputs
Supported inputs for `ppbbchichi_analyzer_par.py`:
- **Parquet** files (typical HiggsDNA-style tables)
- **ROOT** files containing a **TTree** (optionally pass `--tree` to pick the name; otherwise the first TTree is used)

You can pass either:
- a single file via `-i /path/to/file.parquet` or `-i /path/to/file.root`
- a directory via `-i /path/to/dir` (it will recursively pick up `*.parquet` and `*.root`)
- multiple inputs by repeating `-i`

## Core processing flow
1. Read input in batches (Parquet via `pyarrow`, ROOT via `uproot.iterate`).
2. Build / validate the event record needed for bbχχ (resolved b-jets + MET).
3. Apply region masks defined in `bbchichi_regions.py`:
   - `preselection`
   - `srbbchichi`
   - `cr_antibtag`
   - `cr_lowmet`
   - `cr_lepton`
4. Compute weights per region (uses `normalisation.getXsec()` and `normalisation.getLumi(year, era)` where applicable).
5. Fill per-sample, per-region histograms (definitions in `bbchichi_variables.py`, binnings in `bbchichi_binning.py`).
6. Write outputs:
   - ROOT histograms file (PyROOT)
   - ROOT trees file (uproot)

## Outputs
By default the analyzer writes two files:
- `ppbbchichi-histograms.root`
- `ppbbchichi-trees.root`

Output directory selection:
- Single input (or no `--tag`): `outputfiles/bbchichi_<year>_<era>/`
- With `--tag`: `outputfiles/merged/<tag>/`

## Typical commands
Run analyzer:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/inputs
```
Merge multiple inputs into one output folder:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i /path/to/preBPix \
  -i /path/to/postBPix \
  --tag bbchichi_Combined2023
```
Quick yield table from the output trees:
```bash
python yields_check.py -i outputfiles/merged/bbchichi_Combined2023/ppbbchichi-trees.root --region preselection --print-table
```
Make stack plots from histograms:
```bash
python bbchichi_Plotter.py \
  --root outputfiles/merged/bbchichi_Combined2023/ppbbchichi-histograms.root \
  --out stack_plots_bbchichi \
  --data Data \
  --mc TT --mc QCD
```
