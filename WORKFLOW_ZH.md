# hhbbgg_AwkwardAnalyzer（pp → bbχχ / bb + MET）工作流说明（中文）

> 目标：解释这个仓库从“输入数据（Parquet/ROOT）→ 事件变量构建 → 区域选择 → 产出 ROOT(树+直方图) → 产额检查/画图 → DNN 训练与回写打分”的完整逻辑，并尽量覆盖到每一个文件，标明输入与输出。

## 1. 这个仓库在做什么（一句话）
- 主程序 `ppbbchichi_analyzer_par.py` 读取 **Parquet/ROOT** 输入，构建 pp→bbχχ 分析需要的事件变量，按 region 切分，计算权重，最终写出：
  - `ppbbchichi-trees.root`（树：按 sample/region 存分支）
  - `ppbbchichi-histograms.root`（直方图：按 sample/region/variable 存 TH1）
- 随后可以：
  - 用 `yields_check.py` 从 trees 快速做产额/统计误差表
  - 用 `bbchichi_Plotter.py` 从 histograms 画 stack plot
  - 用 `dnn/train_classifier.py` 从 trees 训练一个表格型 DNN
  - 用 `dnn/apply_classifier.py` 把 DNN 分数 `ml_score` 写回到一个新的 scored ROOT

## 2. 环境与依赖（Conda/Micromamba）
仓库提供两套环境：
- 基础分析/画图环境：`environment.yml` → `bbchichi-awk`
- 机器学习环境：`environment-ml.yml` → `bbchichi-awk-ml`（包含 PyTorch / sklearn 等）

典型用法：
```bash
micromamba create -f environment.yml
micromamba activate bbchichi-awk

micromamba create -f environment-ml.yml
micromamba activate bbchichi-awk-ml
```

## 3. 配置与输入数据发现逻辑
### 3.1 数据集路径配置：`config/datasets.yaml`
- 通过 year/era 映射到原始输入目录：
  - 2022: `PreEE`, `PostEE`
  - 2023: `preBPix`, `postBPix`
- 默认输出根目录：
  - `outputs_root: outputfiles`
  - `plots_root: stack_plots`

### 3.2 运行时配置对象：`config/config.py` → `RunConfig`
`RunConfig(year, era)` 的职责：
- 读 `config/datasets.yaml`
- 解析 `--year/--era`，并把 `era=All` 展开为该年对应的多个 era：
  - 2022 All → PreEE + PostEE
  - 2023 All → preBPix + postBPix
- 给出：
  - `raw_paths`: 输入目录列表（当命令行不传 `-i` 时使用）
  - `outputs_root`: 输出根目录（默认 `outputfiles/`）

### 3.3 分析器输入参数（核心入口）
`ppbbchichi_analyzer_par.py` 主要参数：
- `--year`, `--era`：必填，用于 lumi 选择与默认输入路径选择
- `-i/--inFile`：可重复；可以是文件或目录（目录会递归找 `*.parquet` 与 `*.root`）
- `--tree`：当输入是 ROOT 时，指定 TTree 名称；不指定会自动取第一个 TTree
- `--pid-chi`：仅 LHEF 模式使用；指定用哪个 abs(PID) 当作 χ 来构造 MET proxy（默认 52）
- `--tag`：输出合并目录名（见下文输出规则）

输入发现规则（优先级）：
1) 若提供 `-i`：使用这些路径
2) 否则：使用 `RunConfig.raw_paths`（来自 `config/datasets.yaml` 的 `raw_path`）

## 4. 输入数据格式（Analyzer 支持的两类 schema）
分析器支持三种“输入模式”（自动判断）：

### 4.1 Parquet（推荐/常见）
要求至少包含这些列（**必须**）：
- `puppiMET_pt`, `puppiMET_phi`
- `Res_lead_bjet_pt/eta/phi/mass`
- `Res_sublead_bjet_pt/eta/phi/mass`
- `Res_lead_bjet_btagPNetB`, `Res_sublead_bjet_btagPNetB`

可选列（缺失会填默认值）：
- `run`, `lumi`, `event`, `weight`, `n_jets`, `lepton1_pt`

### 4.2 ROOT（nano/parquet-derive 类 schema）
若 ROOT 的 TTree 拥有与 4.1 同名分支，则按同样逻辑读取。

### 4.3 LHEF ROOT（MadGraph / LHE 事件树）
如果 ROOT 中存在 `Particle/Particle.PID`、`Particle/Particle.Px` 等分支，分析器会进入 LHEF 模式：
- 从 final-state 粒子中取 b-quark（PID=±5）构建两条“bjet”
- 从 χ（默认 PID=52，可用 `--pid-chi` 修改）的 px/py 矢量和构建 MET proxy
- b-tag（PNetB）在 LHEF 中不存在：代码里默认填成 1.0 以便 region 选择不会把事件都切掉

## 5. 端到端工作流（从输入到产物）
### Step A：运行 analyzer 产出 trees + histograms
示例（单一路径输入）：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix -i /path/to/inputs
```

如果输入是 MG5 的 LHEF ROOT（常见树名 `LHEF`），建议显式指定：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era preBPix \
  -i /path/to/unweighted_events.root --tree LHEF --pid-chi 52
```

示例（多路径合并并指定输出 tag）：
```bash
python ppbbchichi_analyzer_par.py --year 2023 --era All \
  -i /path/to/preBPix \
  -i /path/to/postBPix \
  --tag bbchichi_Combined2023
```

输出目录规则：
- 若传了 `--tag`：`outputfiles/merged/<tag>/`
- 否则若 `-i` 多个路径：`outputfiles/merged/bbchichi_<year>_<era>/`
- 否则：`outputfiles/bbchichi_<year>_<era>/`

输出文件：
- `ppbbchichi-trees.root`
- `ppbbchichi-histograms.root`

### Step B：分析器内部做了什么（关键逻辑）
1) **样本名(sample)归一化**
   - 用 `sample_name_from_path()` 从文件名/路径推导稳定 sample 名
   - 对 MG5 常见的 `unweighted_events.root`，会优先用上级 `Events_*` 目录名作为 sample

2) **信号/数据/模板(dd)标志**
   - `isdata`: 从文件名含 `data` 等关键词推断（注意：这是“文件名启发式”）
   - `isdd`: 文件名包含 `dd` / `rescaled` 等关键词视为 data-driven 模板
   - `signal`: 文件名含 `chichi/bbchichi/dm/dark/invisible` 视为 signal

3) **构建事件变量**
   - 直接从输入取 MET、两条 resolved bjet
   - 用 `config/utils.py:lVector()` 计算 dijet 四矢量，得到：
     - `dijet_mass`, `dijet_pt`, `dijet_eta`, `dijet_phi`
   - 计算：
     - `DeltaPhi_j1MET`, `DeltaPhi_j2MET`

4) **region 切分**（`bbchichi_regions.py`）
   - `preselection`: 基础 bb+MET
   - `srbbchichi`: SR（高 MET + 双 btag + Δφ）
   - `cr_antibtag`: 反 btag CR
   - `cr_lowmet`: 低 MET CR
   - `cr_lepton`: 有 lepton 的 CR

5) **权重计算（核心 I/O 约定）**
   - Data：`base_w = 1`
   - DD 模板：从输入里寻找 `weight/evt_weight/w/fake_weight` 之一作为权重
   - MC：
     - 取输入的 `weight` 分支/列
     - 乘以 `normalisation.getXsec(inputfile)`（pb）
     - 再乘以 `normalisation.getLumi(year, era) * 1000`（把 fb⁻¹ 转 pb⁻¹）
   - 最终写入：每个 region 都会带一个同名权重分支：`weight_<region>`

6) **填充直方图**
   - 变量列表：`bbchichi_variables.py:variables_common`
   - 变量键→磁盘名字：`bbchichi_variables.py:vardict`
   - 分箱：`bbchichi_binning.py:binning`
   - 直方图用 PyROOT `TH1D` 构建/累加，最后写入文件

7) **写树（uproot）**
   - 每个 sample 会写：
     - `<sample>/processed_events`（包含所有事件 + region mask + 各 region 的 `weight_<region>`）
     - `<sample>/<region>`（仅保存通过该 region 的事件子集）

### Step C：输出 ROOT 的内部结构（非常重要）
#### 5.1 `ppbbchichi-trees.root`
- 顶层目录：每个 sample 一个目录，例如：`TT/`, `QCD/`, `Data/`…
- 每个 sample 下至少：
  - `processed_events`（TTree）
  - `preselection`、`srbbchichi`、`cr_*`（TTree，若该 region 有事件）

树的典型分支包括：
- 运动学：`puppiMET_pt`, `lead_bjet_pt`, `dijet_mass`, `DeltaPhi_j1MET` 等
- region mask：`preselection`, `srbbchichi`, `cr_antibtag`, `cr_lowmet`, `cr_lepton`（布尔/0-1）
- 权重：`weight_preselection`, `weight_srbbchichi`, ...
- 标志：`signal`, `isdata`, `isdd`

#### 5.2 `ppbbchichi-histograms.root`
- 目录层级：`<sample>/<region>/<variable>`
- `<variable>` 是 `bbchichi_variables.py:vardict[var]` 定义的稳定名字

### Step D：快速产额检查（不重跑 analyzer）
用 `yields_check.py` 从 trees 汇总：
```bash
python yields_check.py -i outputfiles/bbchichi_2023_preBPix/ppbbchichi-trees.root \
  --region preselection --print-table
```
输出：
- `yields/yields_<region>.csv`（按 sample）
- `yields/yields_categories_<region>.csv`（按物理类别聚合）
- 可选打印 pretty table

### Step E：画图（stack plot）
用 `bbchichi_Plotter.py` 从 histograms 画堆叠图：
```bash
python bbchichi_Plotter.py \
  --root outputfiles/bbchichi_2023_preBPix/ppbbchichi-histograms.root \
  --out stack_plots_bbchichi \
  --data Data \
  --mc TT --mc QCD
```
输出：
- `stack_plots_bbchichi/<region>/<var>.png/.pdf`

### Step F：DNN 训练与打分回写
#### F.1 训练（`dnn/train_classifier.py`）
输入：
- `--root <.../ppbbchichi-trees.root>`
- `--region preselection`（默认）
- `--features`（默认给了一组特征）
- 权重分支固定约定为：`weight_<region>`

标签（signal/background）来源（二选一）：
- 推荐：`--label-csv labels.csv`，CSV 两列 `sample,label`（label 为 0/1；sample 是 ROOT 里顶层目录名）
- 兼容：不传 `--label-csv` 时，用 `--signal-pattern`（可重复的正则）对 sample 名启发式打标签

LHEF 训练的常见选项：
- `--drop-constant-features`：自动丢弃近似常数的特征（比如 LHEF 中占位的 btag=1.0），避免训练被无意义特征干扰

输出（写入 `--outdir`）：
- `dnn_model.pt`（PyTorch checkpoint，包含网络结构 spec）
- `scaler.json`（StandardScaler 的 mean/std）
- `features.json`（训练使用的特征列表）
- `train_metrics.json`（AUC 等元信息）
- `roc.png`

#### F.2 回写打分（`dnn/apply_classifier.py`）
输入：
- 原始 `ppbbchichi-trees.root`
- 训练产物：`features.json`, `scaler.json`, `dnn_model.pt`

#### F.3 一键闭环（Analyzer → Train → Apply）

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
输出：
- 默认写出 `<input>-scored.root`
- 在每个 `<sample>/<region>` 树里新增分支：`ml_score`（默认分支名）

## 6. 文件用途总表（尽量覆盖整个仓库）
> 说明：`.git/`、`__pycache__/`、`dnn/__pycache__/`、`config/__pycache__/` 等是元数据/缓存目录，通常不参与分析逻辑。

| 路径 | 类型 | 主要输入 | 主要输出 | 作用/说明 |
|---|---|---|---|---|
| README.md | 文档 | — | — | 仓库使用说明（环境、入口命令、输出物） |
| analysis_full_overview.md | 文档 | — | — | 工作流总览（英文） |
| command.md | 文档 | — | — | 常用命令速查表 |
| analysis_script.sh | 脚本 | 本地输入路径 | ROOT 输出目录 | 一键跑 analyzer 的示例 shell 脚本 |
| environment.yml | 环境 | — | conda env | 基础分析依赖（awkward/uproot/ROOT/plot） |
| environment-ml.yml | 环境 | — | conda env | ML 依赖（pytorch/sklearn 等） |
| ppbbchichi_analyzer_par.py | 主程序 | Parquet/ROOT（或 datasets.yaml 默认路径） | ppbbchichi-trees.root + ppbbchichi-histograms.root | 核心分析器：变量构建、region、权重、填图、写树 |
| bbchichi_regions.py | 模块 | 事件 record | mask(bool) | 定义 preselection/SR/CR 的选择 |
| bbchichi_variables.py | 模块 | — | — | 定义 regions、变量清单、变量名映射 |
| bbchichi_binning.py | 模块 | — | — | 定义每个变量的直方图分箱 |
| normalisation.py | 模块 | sample 名/路径；year/era | xsec、lumi | `getXsec()` 与 `getLumi()`，供 analyzer 计算 MC 权重 |
| yields_check.py | 工具脚本 | ppbbchichi-trees.root 或目录/glob | CSV + 终端表格 | 读取 `weight_<region>`，统计 per-sample / per-category 产额 |
| bbchichi_Plotter.py | 工具脚本 | ppbbchichi-histograms.root | PNG/PDF | 从 uproot 读 hist，画 Data/MC 堆叠图 |
| config/datasets.yaml | 配置 | — | — | year/era → raw_path；默认输出根目录 |
| config/config.py | 模块 | datasets.yaml | RunConfig 对象 | 解析 year/era/All，给出输入路径与输出根目录 |
| config/utils.py | 模块 | — | — | 一些通用函数；分析器用到 `lVector()` 做四矢量 |
| dnn/__init__.py | 包说明 | — | — | 说明 dnn 包用途 |
| dnn/common.py | 模块 | sample 名 | label(0/1) | `is_signal()` 与特征数据清洗 |
| dnn/data.py | 模块 | ppbbchichi-trees.root | numpy arrays | 列举 `<sample>/<region>` 树并读取指定分支 |
| dnn/model.py | 模块 | — | checkpoint | 定义 MLP 结构与 save/load checkpoint |
| dnn/scaler.py | 模块 | numpy features | 标准化矩阵 | 简易 StandardScaler（可序列化到 JSON） |
| dnn/train_classifier.py | 训练脚本 | ppbbchichi-trees.root + region + features | outputs_dnn/* | 训练 PyTorch MLP，输出模型/特征/scaler/指标/ROC |
| dnn/apply_classifier.py | 推理脚本 | trees.root + 模型产物 | *-scored.root | 计算 `ml_score` 并写入新 ROOT |
| ml_pipeline.py | 工具脚本 | 原始输入 + labels.csv | trees.root + outputs_dnn/* + scored.root | 一键串联 analyzer→train→apply 的端到端流程 |
| outputfiles/ | 目录（产物） | — | ROOT 文件 | 默认 analyzer 输出根目录；仓库里包含示例输出 |
| outputfiles/merged/ | 目录（产物） | — | ROOT 文件 | 多输入合并输出目录（`--tag` 或多 `-i`） |
| .gitignore | 元文件 | — | — | git 忽略规则 |
| .vscode/ | 目录 | — | — | VS Code 配置（如存在） |
| .venv/ | 目录 | — | — | 本地虚拟环境目录（如果你创建了） |

## 7. 常见“输入/输出”对照（快速查）
- 原始输入：`data/inputs/<year>/<era>/**/*.(parquet|root)`（由你本地提供/下载）
- 核心输出：
  - `outputfiles/bbchichi_<year>_<era>/ppbbchichi-trees.root`
  - `outputfiles/bbchichi_<year>_<era>/ppbbchichi-histograms.root`
- 合并输出（tag）：
  - `outputfiles/merged/<tag>/ppbbchichi-*.root`
- 画图输出：`stack_plots_bbchichi/<region>/*.png|pdf`
- 产额输出：`yields/*.csv`
- DNN 产物：`outputs_dnn/*`（或你指定的 `--outdir`）

---
如果你希望我再补一张“从脚本到脚本的依赖图（Mermaid）”或把 `ppbbchichi-trees.root` 的分支列表也自动总结出来（通过读取一个示例 ROOT 文件），我也可以继续做。