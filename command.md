# pp → bbχχ command cheat sheet

## Environment
Create + activate:
```bash
micromamba create -f environment.yml
micromamba activate bbchichi-awk
```

## Run analyzer
Single input file/folder:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/inputs
```
Multiple inputs merged:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i /path/to/preBPix \
  -i /path/to/postBPix \
  --tag bbchichi_Combined2023
```
If your input is ROOT and you need a specific TTree:
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/file.root --tree Events
```

## Outputs
- `outputfiles/bbchichi_<year>_<era>/ppbbchichi-histograms.root`
- `outputfiles/bbchichi_<year>_<era>/ppbbchichi-trees.root`

## Quick yields from trees
```bash
python yields_check.py -i outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root --region preselection --print-table
```

## Plots
```bash
python bbchichi_Plotter.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-histograms.root \
  --out stack_plots_bbchichi \
  --data Data \
  --mc TT --mc QCD
```

## ML (train + apply score)
Create ML environment:
```bash
micromamba create -f environment-ml.yml
micromamba activate bbchichi-awk-ml
```
Train:
```bash
python dnn/train_classifier.py --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root --region preselection --outdir outputs_dnn
```
Apply (write `ml_score` into a new ROOT):
```bash
python dnn/apply_classifier.py --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root --region preselection --model outputs_dnn/dnn_model.pt --scaler outputs_dnn/scaler.json --features outputs_dnn/features.json
```
