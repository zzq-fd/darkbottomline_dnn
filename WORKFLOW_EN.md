# hhbbgg_AwkwardAnalyzer (pp → bbχχ / bb + MET) Workflow Notes (English)

> Goal: Provide a detailed, repo-wide workflow explanation: what each step does, what goes in/out, how outputs are structured on disk (including inside the ROOT files), and a per-file purpose table.

## 1. What this repo does (high level)
- The main entrypoint `ppbbchichi_analyzer_par.py` reads **Parquet/ROOT** inputs, builds bbχχ event variables, applies region selections, computes weights, and writes two ROOT outputs:
  - `ppbbchichi-trees.root` (TTrees: per sample and per region)
  - `ppbbchichi-histograms.root` (TH1: per sample, per region, per variable)
- Then you typically:
  - sanity-check yields with `yields_check.py` (reads trees)
  - make stack plots with `bbchichi_Plotter.py` (reads histograms)
  - train a tabular DNN with `dnn/train_classifier.py` (reads trees)
  - apply the model and write `ml_score` back with `dnn/apply_classifier.py` (writes a *new* scored ROOT)

## 2. Environments
Two conda/micromamba env specs are provided:
- `environment.yml` → `bbchichi-awk` (core analysis + plotting)
- `environment-ml.yml` → `bbchichi-awk-ml` (adds PyTorch / scikit-learn)

Typical usage:
```bash
micromamba create -f environment.yml
micromamba activate bbchichi-awk

micromamba create -f environment-ml.yml
micromamba activate bbchichi-awk-ml
```

## 3. Configuration and input discovery
### 3.1 Dataset mapping: `config/datasets.yaml`
- Maps `year/era` → default `raw_path` input folders.
- Defines default roots:
  - `outputs_root: outputfiles`
  - `plots_root: stack_plots`

### 3.2 Runtime config object: `config/config.py` (`RunConfig`)
- Loads `config/datasets.yaml`
- Expands `era=All` into a list of eras:
  - 2022 All → PreEE + PostEE
  - 2023 All → preBPix + postBPix
- Exposes:
  - `raw_paths`: default inputs when the analyzer is run without `-i`
  - `outputs_root`: base output directory

### 3.3 Analyzer CLI (core entrypoint)
`ppbbchichi_analyzer_par.py` key args:
- `--year`, `--era` (required)
- `-i/--inFile` repeatable; each can be a file or directory (directories are searched recursively for `*.parquet` and `*.root`)
- `--tree` optional; ROOT TTree name (defaults to the first TTree found)
- `--pid-chi` LHEF-only; which abs(PID) is treated as χ for the MET proxy (default: 52)
- `--tag` optional; forces merged output folder name

Input discovery:
1) if `-i` is provided: use those paths
2) else: use `RunConfig.raw_paths` derived from `config/datasets.yaml`

## 4. Supported input schemas
The analyzer supports three modes (auto-detected for ROOT):

### 4.1 Parquet inputs
Required columns:
- `puppiMET_pt`, `puppiMET_phi`
- `Res_lead_bjet_pt/eta/phi/mass`
- `Res_sublead_bjet_pt/eta/phi/mass`
- `Res_lead_bjet_btagPNetB`, `Res_sublead_bjet_btagPNetB`

Optional columns (filled with defaults if missing):
- `run`, `lumi`, `event`, `weight`, `n_jets`, `lepton1_pt`

### 4.2 ROOT ntuples in the same schema
If the ROOT TTree has the same branch names as above, it is treated like the Parquet schema.

### 4.3 LHEF ROOT mode (MadGraph-style)
If the ROOT tree contains `Particle/Particle.PID` and `Particle/Particle.Px` (etc.), the analyzer switches to LHEF mode:
- uses final-state b-quarks (PID=±5) as “b-jets”
- MET proxy is the vector sum of χ px/py (default χ PID=52; configurable via `--pid-chi`)
- b-tag is not available → PNetB is set to 1.0 (so regions won’t reject everything)

## 5. End-to-end workflow
### Step A: run the analyzer (produce trees + histograms)
Single input:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/inputs
```

For MG5 LHEF ROOT inputs (typical tree name `LHEF`), it is safer to be explicit:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix \
  -i /path/to/unweighted_events.root --tree LHEF --pid-chi 52
```

Merge multiple inputs with a tag:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i /path/to/preBPix \
  -i /path/to/postBPix \
  --tag bbchichi_Combined2023
```

Output directory rules:
- with `--tag`: `outputfiles/merged/<tag>/`
- else if multiple `-i`: `outputfiles/merged/bbchichi_<year>_<era>/`
- else: `outputfiles/bbchichi_<year>_<era>/`

Outputs:
- `ppbbchichi-trees.root`
- `ppbbchichi-histograms.root`

### Step B: what happens inside the analyzer (key logic)
1) **Stable sample naming**
   - `sample_name_from_path()` normalizes filenames.
   - For MG5 `unweighted_events.root`, it prefers the parent `Events_*` directory name.

2) **Heuristic flags**
   - `isdata`: inferred from filename containing `data` keywords
   - `isdd`: inferred from filename containing `dd` / `rescaled` keywords
   - `signal`: inferred from filename containing `chichi/bbchichi/dm/dark/invisible`

3) **Event variable construction**
   - reads MET and two resolved b-jets
   - computes dijet 4-vector with `config/utils.py:lVector()` → `dijet_mass/pt/eta/phi`
   - computes `DeltaPhi_j1MET`, `DeltaPhi_j2MET`

4) **Region masks** (`bbchichi_regions.py`)
   - `preselection`, `srbbchichi`, `cr_antibtag`, `cr_lowmet`, `cr_lepton`

5) **Weights** (important I/O convention)
   - Data: base weight = 1
   - DD templates: reads a weight column among `weight/evt_weight/w/fake_weight`
   - MC: `weight * xsec(pb) * lumi(pb^-1)`
     - `xsec` via `normalisation.getXsec(inputfile)`
     - `lumi` via `normalisation.getLumi(year, era) * 1000` (fb^-1 → pb^-1)
   - The output record always contains `weight_<region>` for every region.

6) **Histograms**
   - variable list: `bbchichi_variables.py:variables_common`
   - stable on-disk names: `bbchichi_variables.py:vardict`
   - binning: `bbchichi_binning.py:binning`

7) **Trees** (written with uproot)
   - per sample:
     - `<sample>/processed_events`
     - `<sample>/<region>` for each region that has events

### Step C: ROOT output structure
#### 5.1 `ppbbchichi-trees.root`
- top-level dirs: one per sample, e.g. `TT/`, `QCD/`, `Data/` …
- each sample contains:
  - `processed_events` (TTree)
  - region trees: `preselection`, `srbbchichi`, `cr_*` (TTrees, if non-empty)

Typical branches:
- kinematics: `puppiMET_pt`, `lead_bjet_pt`, `dijet_mass`, `DeltaPhi_j1MET`, …
- region masks: `preselection`, `srbbchichi`, `cr_antibtag`, `cr_lowmet`, `cr_lepton`
- weights: `weight_preselection`, `weight_srbbchichi`, …
- flags: `signal`, `isdata`, `isdd`

#### 5.2 `ppbbchichi-histograms.root`
- directory hierarchy: `<sample>/<region>/<variable>`

### Step D: yields check (no reprocessing)
```bash
python yields_check.py -i outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root \
  --region preselection --print-table
```
Outputs:
- `yields/yields_<region>.csv` (per sample)
- `yields/yields_categories_<region>.csv` (category aggregation)

### Step E: stacked plots from histograms
```bash
python bbchichi_Plotter.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-histograms.root \
  --out stack_plots_bbchichi \
  --data Data \
  --mc TT --mc QCD
```
Outputs:
- `stack_plots_bbchichi/<region>/<var>.png/.pdf`

### Step F: DNN training and score writing
#### F.1 Train (`dnn/train_classifier.py`)
Inputs:
- `--root <.../ppbbchichi-trees.root>`
- `--region` (default `preselection`)
- `--features` (comma-separated list; defaults exist)
- expects a weight branch named `weight_<region>`

Labeling (signal/background), choose one:
- Recommended: `--label-csv labels.csv` with columns `sample,label` (0/1), where `sample` is the top-level directory name in the ROOT file.
- Fallback: `--signal-pattern` (repeatable regex) for heuristic labeling based on the sample name.

Common for LHEF-derived training:
- `--drop-constant-features` to automatically remove near-constant features (e.g. placeholder btag=1.0).

Outputs (in `--outdir`):
- `dnn_model.pt`, `scaler.json`, `features.json`, `train_metrics.json`, `roc.png`

#### F.2 Apply (`dnn/apply_classifier.py`)
Inputs:
- original `ppbbchichi-trees.root`
- the training artifacts above

Outputs:
- `<input>-scored.root` by default
- adds a per-event branch `ml_score` (configurable name)

#### F.3 One-command end-to-end (Analyzer → Train → Apply)
Use `ml_pipeline.py` from the repo root to stitch the steps together:
```bash
python ml_pipeline.py \
  --year 2023 --era preBPix --tree LHEF --pid-chi 52 \
  -i /path/to/signal/results -i /path/to/background/results \
  --tag train_mix \
  --region preselection \
  --label-csv labels.csv \
  --drop-constant-features \
  --outdir outputs_dnn_train_mix
```

## 6. Per-file purpose table
> Note: `.git/`, `__pycache__/`, and similar directories are metadata/caches and not part of the physics workflow.

| Path | Type | Main inputs | Main outputs | Purpose |
|---|---|---|---|---|
| README.md | Doc | — | — | Quick-start, inputs/outputs, ML instructions |
| analysis_full_overview.md | Doc | — | — | High-level workflow overview |
| command.md | Doc | — | — | Command cheat sheet |
| analysis_script.sh | Shell | local input dirs | ROOT outputs | Example wrapper to run the analyzer |
| environment.yml | Env | — | conda env | Core analysis + plotting dependencies |
| environment-ml.yml | Env | — | conda env | Adds ML dependencies (PyTorch/sklearn) |
| ppbbchichi_analyzer_par.py | Entry script | Parquet/ROOT (+ datasets.yaml defaults) | trees.root + histograms.root | Main analyzer: variables, regions, weights, trees, histograms |
| bbchichi_regions.py | Module | event record | masks | Region selections |
| bbchichi_variables.py | Module | — | — | Region list + variable list + name mapping |
| bbchichi_binning.py | Module | — | — | Histogram binning definitions |
| normalisation.py | Module | sample name; year/era | xsec/lumi | `getXsec()` and `getLumi()` helpers |
| yields_check.py | Tool | trees.root (or dir/glob) | CSV + printed table | Yield aggregation per sample/category |
| bbchichi_Plotter.py | Tool | histograms.root | png/pdf | Stack plots from histogram ROOT |
| config/datasets.yaml | Config | — | — | year/era → raw_path + default output roots |
| config/config.py | Module | datasets.yaml | RunConfig | Loads config, resolves raw paths and output roots |
| config/utils.py | Module | — | — | Helper math/kinematics; provides `lVector()` |
| dnn/__init__.py | Package | — | — | DNN package description |
| dnn/common.py | Module | sample name | signal label | Signal naming rules + feature sanitation |
| dnn/data.py | Module | trees.root | numpy arrays | Enumerate/read `<sample>/<region>` trees |
| dnn/model.py | Module | — | checkpoint | MLP spec + save/load checkpoint |
| dnn/scaler.py | Module | numpy arrays | scaler JSON | Simple serializable standard scaler |
| dnn/train_classifier.py | Script | trees.root | outputs_dnn/* | Train PyTorch MLP classifier |
| dnn/apply_classifier.py | Script | trees.root + artifacts | *-scored.root | Add `ml_score` branch to new ROOT |
| ml_pipeline.py | Script | raw inputs + labels.csv | trees.root + outputs_dnn/* + scored.root | One-command analyzer→train→apply pipeline |
| outputfiles/ | Output dir | — | ROOT files | Default analyzer output root (includes example outputs) |
| outputfiles/merged/ | Output dir | — | ROOT files | Output location for merged runs (`--tag` / multi `-i`) |
| .gitignore | Meta | — | — | Git ignore rules |
| .vscode/ | Meta dir | — | — | VS Code settings (if present) |
| .venv/ | Meta dir | — | — | Local virtual env dir (if you created it) |

## 7. Quick I/O cheat sheet
- Raw inputs: `data/inputs/<year>/<era>/**/*.(parquet|root)` (provided by you)
- Core outputs:
  - `outputfiles/bbchichi_<year>_<era>/ppbbchichi-trees.root`
  - `outputfiles/bbchichi_<year>_<era>/ppbbchichi-histograms.root`
- Merged outputs: `outputfiles/merged/<tag>/ppbbchichi-*.root`
- Plots: `stack_plots_bbchichi/<region>/*.{png,pdf}`
- Yields: `yields/*.csv`
- DNN artifacts: `outputs_dnn/*` (or `--outdir`)

---
If you want, I can also add a Mermaid dependency diagram and/or summarize branch lists by opening one of the example ROOT files under `outputfiles/`.