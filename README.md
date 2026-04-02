# pp → bbχχ AwkwardAnalyzer

本仓库已清理为 **只支持 pp → bbχχ（bb + MET）** 的分析流程。

## 1) 环境
推荐使用 conda/micromamba：
```bash
micromamba create -f environment.yml
micromamba activate bbchichi-awk
```

## 2) 输入（Input）
`ppbbchichi_analyzer_par.py` 支持两类输入：
- `.parquet`（常见的列式表格）
- `.root`（包含 TTree；可用 `--tree` 指定树名，不指定则自动选第一个 TTree）

如果输入是 MadGraph/MG5 的 LHEF ROOT（常见树名 `LHEF`，包含 `Particle/Particle.PID` 等分支），分析器会自动进入 LHEF 模式。此时建议显式指定：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/unweighted_events.root --tree LHEF --pid-chi 52
```
其中 `--pid-chi` 用于指定 abs(PID)（默认 52）来构造 MET proxy。

`-i/--inFile` 的用法：
- 传单个文件：`-i /path/to/file.parquet` 或 `-i /path/to/file.root`
- 传目录：`-i /path/to/dir`（会递归搜集 `*.parquet` 和 `*.root`）
- 合并多个输入：重复给 `-i` 多次

如果 **不提供 `-i`**，会去读 [config/datasets.yaml](config/datasets.yaml) 里对应 `--year/--era` 的 `raw_path`。

## 3) 运行分析（Analyzer）
单一路径：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/inputs
```
合并多路径并固定输出目录名：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i /path/to/preBPix \
  -i /path/to/postBPix \
  --tag bbchichi_Combined2023
```

## 4) 输出（Output）
分析器会写出两份 ROOT 文件：
- `ppbbchichi-histograms.root`：按 sample/region/variable 存的 TH1
- `ppbbchichi-trees.root`：按 sample/region 存的 TTree（包含 `weight_<region>` 等分支）

输出目录规则：
- 默认：`outputfiles/bbchichi_<year>_<era>/`
- 如果传了 `--tag`：`outputfiles/merged/<tag>/`

## 5) 流程（Pipeline）
代码主链路：
1. 读取输入（Parquet 用 `pyarrow`，ROOT 用 `uproot`）
2. 构建 bbχχ 事件变量（bb jets + MET 等）
3. 按区域切分（见 [bbchichi_regions.py](bbchichi_regions.py)）：
   - `preselection`, `srbbchichi`, `cr_antibtag`, `cr_lowmet`, `cr_lepton`
4. 对每个 region 计算权重并填充直方图
   - 变量清单： [bbchichi_variables.py](bbchichi_variables.py)
   - 分箱： [bbchichi_binning.py](bbchichi_binning.py)
   - xsec/lumi： [normalisation.py](normalisation.py)
5. 写出 histogram ROOT + tree ROOT

## 6) 快速检查（Yields / Plots）
Yields（从输出树读取，不重新跑分析）：
```bash
python yields_check.py -i outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root --region preselection --print-table
```
Plot（堆叠图）：
```bash
python bbchichi_Plotter.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-histograms.root \
  --out stack_plots_bbchichi \
  --data Data \
  --mc TT --mc QCD
```

## 7) 机器学习（ML）分类（代码位于 dnn/）

建议的做法是：**先跑 analyzer 输出树** → **用树训练分类器得到 `ml_score`** → **把 `ml_score` 写回新的 ROOT**。

### 7.1 训练分类器
需要 `environment-ml.yml`（包含 `pytorch` 等 DNN 依赖）：
```bash
micromamba create -f environment-ml.yml
micromamba activate bbchichi-awk-ml
```

训练（默认用 `preselection` 区域）：
```bash
python dnn/train_classifier.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root \
  --region preselection \
  --outdir outputs_dnn
```

标签（signal/background）建议用显式 CSV，避免仅靠样本名启发式：
- `--label-csv labels.csv`：两列 `sample,label`（label 为 0/1；sample 是 `ppbbchichi-trees.root` 里顶层目录名）
- 不提供 `--label-csv` 时，仍支持 `--signal-pattern`（可重复正则）自动打标签

如果训练输入来自 LHEF（例如 btag 占位为常数），建议加：
- `--drop-constant-features`：自动丢弃近似常数特征
产物：
- `outputs_dnn/dnn_model.pt`
- `outputs_dnn/scaler.json`
- `outputs_dnn/features.json`
- `outputs_dnn/train_metrics.json` + `roc.png`

### 7.2 回写打分到 ROOT（生成 scored trees）
```bash
python dnn/apply_classifier.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root \
  --region preselection \
  --model outputs_dnn/dnn_model.pt \
  --scaler outputs_dnn/scaler.json \
  --features outputs_dnn/features.json
```
默认输出：`outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees-scored.root`
并在每个 `<sample>/preselection` 的树里新增分支：`ml_score`（范围 0~1）。

更完整的说明见：[analysis_full_overview.md](analysis_full_overview.md)

### 7.3 一键跑通（Analyzer → Train → Apply）
如果你想把“从原始输入 → 训练 → 回写分数”串成一个命令，可以用根目录的 `ml_pipeline.py`：
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
`labels.csv` 需要包含两列：`sample,label`（label 为 0/1），其中 sample 是 `ppbbchichi-trees.root` 里顶层目录名。
