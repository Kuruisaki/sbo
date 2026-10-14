# SBO 代码目录与接口设计

版本：1.6（M5 实施记录，与流程指南同步） · 日期：2026-10-14  
项目目录：`D:\codex\sbo`  
配套文档：[SBO_流程与部署指南.md](SBO_流程与部署指南.md)

## 1. 实现范围

本文描述完整流水线的目录、接口和验收契约。截至本版，M1–M5 及入口 01–07 均已生成。主实验完成 8 个真实成功新增点、25 点训练/GP/LOO 更新、冻结最终模型后开放 6 点留出、12 次独立多面元/重复复核及收益判断，已导出最终报告和推荐几何。实际运行入口和当前状态见 [README.md](README.md)。

目标流水线为：配置→连续 NACA 几何→XFOIL 真实评估→样本库→GP→EI 提议→真实加点→独立复核→报告。使用单机串行进程、两个设计变量、一个目标，首版不做分布式调度、Web 界面或神经网络。

## 2. Conda 环境与外部依赖

### 2.1 建议的保守版本组合

采用独立环境 `sbo-xfoil`、Python 3.11，统一使用 conda-forge。下表 Python 库版本已在本机核对，并通过导入、合成数据 GP 拟合与真实 XFOIL 基准点预检；实际构建锁已导出为 `environment-win-64.lock.txt`，每个成功预检实验也保留独立副本。

| 依赖 | 拟定版本 | 用途 |
|---|---|---|
| Python | 3.11 | 主程序、标准库和进程管理 |
| NumPy | 1.26.4 | 几何与数值数组 |
| SciPy | 1.14.1 | LHS、Sobol、正态分布、EI 搜索 |
| scikit-learn | 1.5.2 | GaussianProcessRegressor 与核函数 |
| pandas | 2.2.3 | 样本导出、统计表 |
| matplotlib | 3.9.2 | 几何、误差和优化历程绘图，使用无界面后端 |
| PyYAML | 6.0.2 | 配置读取 |
| joblib | 1.4.2 | 保存已训练 GP 和相关元数据 |
| psutil | 6.0.0 | 超时后清理子进程，核对断点状态 |
| pytest | 8.3.3 | 关键数值和求解器接口验证 |
| setuptools | 75.1.0 | 本地包可编辑安装 |
| pip | 24.2 | 仅用于本地项目安装 |
| XFOIL | 官方 Windows 6.99 包 | 环境外的独立气动求解器 |

Windows 环境额外显式约束 `libblas=*=*openblas`、`libcblas=*=*openblas`、`liblapack=*=*openblas`。首次实测中 MKL 2026.1.0 构建未通过 Cholesky/GP 数值探针；按 [Conda-forge 官方 BLAS 切换说明](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation) 切换到 OpenBLAS 后通过。Python 库的上述精确版本没有改变；最终构建号以实际锁文件为准。数值探针在独立、有时限且可清理进程树的子进程中执行，底层崩溃不得被当作环境验收成功。

`sqlite3`、`subprocess`、`pathlib`、`hashlib`、`json`、`logging`、`argparse`、`dataclasses` 为 Python 标准库，不另行安装。不需要 PyTorch、TensorFlow、OpenMDAO、pymoo、Gmsh、MPI 或商业 CFD。直接封装 XFOIL 可执行程序，避免额外 Python XFOIL 绑定的构建依赖。

`environment.yml` 已写入环境名、conda-forge 渠道、上述精确版本和 OpenBLAS 构建选择。新机器仍需先做依赖求解与预检；本机通过不代表所有 Windows 平台都自动兼容。只导入成功仍不能代替 GP 数值探针及真实 XFOIL 单点测试。

以下是新机器的环境部署命令；本机环境已部署，无需重复创建。相关配置与包文件已实现，仍须在目标机器重新执行预检：

| 顺序 | 命令 | 前提 |
|---|---|---|
| 1 | `conda create --dry-run -n sbo-xfoil --override-channels -c conda-forge python=3.11 numpy=1.26.4 scipy=1.14.1 'libblas=*=*openblas' 'libcblas=*=*openblas' 'liblapack=*=*openblas' scikit-learn=1.5.2 pandas=2.2.3 matplotlib=3.9.2 pyyaml=6.0.2 joblib=1.4.2 psutil=6.0.0 pytest=8.3.3 setuptools=75.1.0 pip=24.2` | 网络可达，确认依赖可解 |
| 2 | `conda env create -f environment.yml` | environment.yml 已按通过的组合编写 |
| 3 | `conda activate sbo-xfoil` | 环境创建成功 |
| 4 | `python -m pip install --no-deps --no-build-isolation -e .` | pyproject.toml 与包目录已实现；避免 pip 重解析依赖 |
| 5 | `conda export --format=explicit --file=environment-win-64.lock.txt` | 通过预检后导出真实构建锁 |

Conda 锁是平台相关的；换到 Linux 要重新创建并导出，不能复用 Windows 二进制锁。依据见 [Conda 环境管理文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。若版本组合在目标渠道不可用，先调整最小必要依赖、重做预检并更新两份文档，不能把未解出的版本宣称为验证环境。

XFOIL 的路径配置为 `xfoil.executable`，不放在 Conda 包列表里。下载入口及 Windows 兼容性预检见流程指南；保留官方压缩包和所需运行库，禁止只拷一个 exe 后假定部署完整。

## 3. 规划的目录树

```text
D:\codex\sbo\
├─ SBO_流程与部署指南.md              已交付
├─ SBO_代码目录与接口设计.md          已交付
├─ README.md                         已实现：M1 入口和结果导航
├─ environment.yml                   已实现：环境声明及 OpenBLAS 选择
├─ environment-win-64.lock.txt        已导出：通过预检后的实际构建锁
├─ pyproject.toml                    已实现：可编辑安装及 pytest 配置
├─ .gitignore                        已实现：缓存、工具和大型运行输出
├─ configs/
│  └─ naca2412.yaml                   已实现：唯一默认实验配置
├─ src/sbo/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ geometry.py
│  ├─ xfoil_runner.py
│  ├─ polar.py
│  ├─ evaluator.py
│  ├─ records.py                     已实现：共享结果和预算异常
│  ├─ store.py
│  ├─ sampling.py
│  ├─ stages.py                      已实现：前置关卡、租约、产物凭据
│  ├─ baseline.py
│  ├─ doe.py
│  ├─ m2_cli.py
│  ├─ surrogate.py                   已实现：原尺度 GP、独立 LOO、模型校验
│  ├─ acquisition.py
│  ├─ fitting.py                     已实现：初始拟合/LOO/提议协调与恢复
│  ├─ optimizer.py
│  ├─ verification.py
│  └─ reporting.py
├─ scripts/
│  ├─ 01_preflight.py
│  ├─ 02_baseline.py
│  ├─ 03_doe.py
│  ├─ 04_fit.py
│  ├─ 05_optimize.py
│  ├─ 06_verify.py
│  └─ 07_report.py
├─ tests/
│  ├─ test_geometry.py
│  ├─ test_polar.py
│  ├─ test_evaluator.py
│  ├─ test_evaluator_integration.py
│  ├─ test_m2.py
│  ├─ test_m2_integration.py
│  ├─ test_surrogate_acquisition.py
│  ├─ test_m3_stage.py
│  ├─ test_m3_integration.py
│  ├─ test_optimizer.py
│  ├─ test_optimizer_integration.py
│  ├─ test_verification.py
│  ├─ test_verification_integration.py
│  ├─ test_xfoil_integration.py
│  └─ fixtures/                      已保存：真实正常、延续和未收敛样本
├─ tools/xfoil/                      已部署：官方 6.99 可执行包及源码
├─ data/reference/                   已保存：UIUC 参考几何及来源记录
└─ runs/<run_id>/                    已实现：M1–M5 独立目录与最终报告
   ├─ config.frozen.yaml
   ├─ manifest.json
   ├─ samples.sqlite
   ├─ samples.csv
   ├─ failures.csv
   ├─ holdout.csv
   ├─ checkpoint.json
   ├─ baseline.json / baseline.md / baseline.png
   ├─ doe.json / doe.md / doe.png / doe-plan.json
   ├─ datasets.json
   ├─ fit.json / fit.md
   ├─ optimization.json / optimization.md / history.csv / best-so-far.png
   ├─ verification.json / verification.csv / verification.png
   ├─ holdout-predictions.csv
   ├─ report.md / report.json / report.png
   ├─ best_airfoil.dat / best_airfoil.json
   ├─ proposals/iter_<k>.json
   ├─ evaluations/<eval_id>/attempt_<n>/
   │  ├─ airfoil.dat
   │  ├─ input.txt
   │  ├─ stdout.log
   │  ├─ stderr.log
   │  ├─ polar.txt
   │  ├─ cp.txt                       可选，复核阶段要求生成
   │  └─ result.json
   ├─ models/iter_<k>/
   │  ├─ gp.joblib
   │  ├─ metadata.json
   │  ├─ training.json / training.csv
   │  ├─ loo.json / loo.csv / loo.png
   │  └─ proposal.json
   └─ reports/
      ├─ summary.md
      ├─ metrics.json
      ├─ final_comparison.csv
      ├─ best_airfoil.dat
      ├─ best_airfoil.json
      └─ figures/
```

树中的数据文件是预期产物，不是让使用者手工填写的虚构结果。源码文件数量刻意保持有限，入口脚本只协调业务模块，不能各自复制求解、筛选和建模逻辑。

## 4. 配置与基础文件

| 文件 | 输入 | 输出/作用 | 验收要点 |
|---|---|---|---|
| `environment.yml` | 通过求解的依赖版本 | 可创建 Conda 环境的声明 | 与真实环境锁相符 |
| `pyproject.toml` | 包名、版本、src 布局、依赖和测试配置 | 本地包 `sbo` 可安装 | 从根目录运行入口时可导入包 |
| `configs/naca2412.yaml` | 下表中的所有物理与算法参数 | 默认实验配置 | 不在代码中重复硬编码 |
| `README.md` | 两份设计文档与最终 CLI | 最短运行路径、前提、产物位置 | 显式区分已实现与计划功能 |
| `.gitignore` | 运行目录、环境和工具路径 | 控制版本库体积 | 不能忽略必须版本化的配置和文档 |
| `src/sbo/__init__.py` | 包版本信息 | 统一版本标识 | 版本写入实验 manifest |

### 4.1 配置字段契约

| 字段组 | 必需内容 |
|---|---|
| `design` | 变量顺序固定为 [m,p]；边界 [[0.01,0.04],[0.25,0.60]]；基准 [0.02,0.40]；t=0.12 |
| `geometry` | 每侧161点；余弦分布；尾缘系数−0.1015；生成器版本；英文命名规则 |
| `flow` | Re=1e6；Mach=0.1；target_cl=0.5；TYPE=1；Ncrit=9；XTR=[1,1]；CL容差0.001 |
| `xfoil` | 可执行文件绝对路径；标准N=160；冻结的其余PPAR设置；200/400迭代；60/120秒超时；最多2次尝试 |
| `doe` | 16初始点+基准；6留出点；种子42与2026；距离阈值0.001 |
| `gp` | Matérn ν=2.5、ARD长度尺度、幅值边界、normalize_y、alpha及最大正则、5次重启、种子43 |
| `acquisition` | EI ξ=1e−5；Sobol池4096；补边界；差分进化种子1000+k；每4次成功加点探索一次 |
| `budget` | 成功序贯点最多12；不同标准设计最多45；总进程启动最多120；预留最终12次 |
| `stopping` | 至少8个成功新增点后检查；最近5点改善≤2e−5且最大EI<1e−5；连续3失败；最近10点失败率>20% |
| `quality` | 基准重复与面元误差门槛；LOO/留出误差门槛；需要人工复查的异常标记 |
| `verification` | N=[160,200,240]；重复200；候选数2；不确定度下限1e−5 |
| `runtime` | 串行；根目录；日志级别；预算计数策略；人工复查点的显式处置规则 |

所有“角度”字段名称加 `_deg` 并使用度。m、p、t、CL、CD、CM、Re、Mach 为无量纲量。GP 的正则 `alpha` 命名为 `gp.alpha_regularization`，避免与 `alpha_deg` 混淆。配置未知字段直接报错，缺少关键值不能猜默认。

## 5. 业务模块：输入、输出、职责

| 文件 | 输入 | 输出 | 功能与主要依赖 |
|---|---|---|---|
| `config.py` | 默认YAML、run-id、路径 | 校验后的不可变配置；冻结YAML；配置哈希 | 校验边界、预算、版本、路径和单位；PyYAML、dataclasses、hashlib |
| `geometry.py` | m、p、固定t、离散约定 | 坐标数组、airfoil.dat、几何检查结果与哈希 | 连续NACA公式、轮廓拼接、自交/重复检查；NumPy |
| `xfoil_runner.py` | exe、几何、评估配置、attempt目录 | 会话输入、日志、返回码、计时、输出文件路径 | 创建全新进程、菜单输入、超时、进程树清理；subprocess、psutil、pathlib |
| `polar.py` | polar文件、日志、目标CL、返回码 | 结构化气动记录或解析失败原因 | 按列名识别数据；支持科学记数法、Fortran D指数、空文件、多点和损坏行；标准库、NumPy |
| `evaluator.py` | 设计向量、标准或验证配置、预算与store | EvaluationResult及attempt记录 | 几何→缓存查询→运行→解析→接受检查→一次重试；集中实现所有有效性规则 |
| `store.py` | 配置、设计/attempt/模型/预算事件 | SQLite事务、CSV快照、断点检查点 | 唯一标识、缓存、去重、原子提交、恢复核对；sqlite3、json、hashlib、csv |
| `sampling.py` | 边界、数量、种子、已占用点 | DOE、留出、候选池、替补点 | LHS/Sobol、归一化、距离过滤、maximin；SciPy qmc、NumPy |
| `baseline.py` | 既有实验、预检基准、质量阈值 | 几何/Cp/转捩及离散/重复性报告与图像 | 独立 200/240/160-repeat 求解；UIUC 对照；NumPy、matplotlib |
| `doe.py` | 已通过基准关卡的实验、持久采样方案 | 17+6 有效数据、失败记录、恢复检查点 | 训练先于留出；提议先提交；预算保护、maximin 替补与封存标签 |
| `surrogate.py` | 同配置成功训练样本、GP配置 | GP模型、原尺度均值/标准差接口、LOO报告、模型元数据 | 核拟合、警告记录、交叉验证、正则回退；scikit-learn、joblib |
| `acquisition.py` | GP、f_best、边界、排除点、轮次 | 一个候选及EI/探索原因/距离诊断 | EI及σ≈0处理、候选池与DE比较、maximin后备；SciPy stats/optimize |
| `optimizer.py` | 冻结配置、store、checkpoint、M3 模型 | 每轮真实新增样本、GP/LOO、历史、检查点、停止原因 | 串行状态机、原始证据绑定、预算保护、设计级失败率、停止条件；调用 runner/polar 既有接口，不复制解析规则 |
| `verification.py` | 冻结模型、6点留出、前2候选、基准 | 泛化指标、独立复核结果、数值差异和排名 | 使用相同物理配置、多面元评估；阻止复核数据回流训练 |
| `reporting.py` | store、模型诊断、verification结果 | Markdown、CSV、JSON、PNG和推荐几何 | 原始证据复核、图表、来源与限制；csv、NumPy、matplotlib；不启动求解器或重训 |

依赖方向为：入口→流程协调→评估/模型/采集→底层几何/进程/解析/存储。`acquisition.py` 不依赖 XFOIL，`polar.py` 不修改数据集，`reporting.py` 不发起新气动计算。候选优化使用 [SciPy differential_evolution](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.differential_evolution.html)，其迭代次数不计为真实评估次数。

## 6. 入口脚本与阶段契约

所有入口返回明确退出状态：成功为0，阶段未完成/质量关卡失败为非0，并保留机器可读原因。不能将“完成了程序执行”与“达到质量门槛”共用一条无条件成功消息。

| 文件 | 输入/参数 | 输出 | 放行条件 |
|---|---|---|---|
| `01_preflight.py` | `--config`、新`--run-id` | 冻结配置、manifest、环境诊断、基准单点尝试 | exe存在且可启动、依赖可导入、气动目标点成功、无挂起；记录预算 |
| `02_baseline.py` | `--run-id`，可选`--resume` | 基准几何对比、160/200/240与160重复、质量报告 | 流程指南中的基准门槛通过；预检合格单点允许复用 |
| `03_doe.py` | `--run-id`，可选`--resume` | 16+1训练、6封存留出、失败记录 | 有效数量达标，否则标记不完整并停止 |
| `04_fit.py` | `--run-id`，可选`--resume` | iter_000模型、LOO、元数据、首个未评估提议 | 训练集合法、模型可预测、提议合法；LOO较差则记录探索标志 |
| `05_optimize.py` | `--run-id`，可选`--resume` | 序贯历史、模型、检查点、停止原因 | 至少一次成功代理驱动加点后才可称闭环有更新 |
| `06_verify.py` | `--run-id`、可选`--resume`和明确复查参数 | 6点最终诊断、基准和2候选的12次独立复核 | 数据冻结完成且有足够预算；缺点或质量失败返回非0 |
| `07_report.py` | `--run-id` | reports目录 | 缺失阶段也能生成诊断报告，但不得伪装成完整结果 |

`01_preflight` 负责创建新实验，已有run-id不能被它覆盖。入口02–06要求已有且前置关卡通过的run-id；第一次执行可以不带 `--resume`，完成后只核验/复用，不新增气动调用，中断时用 `--resume`。07为可重复的只读证据报告入口，不需要恢复参数，缺阶段也能生成诊断并返回非0。配套流程表中的 `05_optimize --resume` 表示允许基于该实验已有阶段/检查点继续，若无序贯检查点则从已验收的初始模型开始。后续脚本只读取 `config.frozen.yaml`，并验证其哈希。

## 7. 数据格式与接口契约

### 7.1 参数、结果与状态

| 对象 | 必需字段 | 约束 |
|---|---|---|
| Design | design_id、m、p、t、u_m、u_p、split | split为train或holdout；变量顺序不可随列序改变 |
| Evaluation | eval_id、design_id、fidelity、config_hash、status、selected_attempt | 标准160与验证200/240是不同evaluation；fidelity至少含N和求解器身份 |
| Attempt | attempt_id、eval_id、开始结束时间、耗时、iteration_limit、timeout、returncode、日志路径、失败原因 | 每次启动一个新目录；全局预算按启动事件计数 |
| 气动结果 | alpha_deg、cl、cd、cdp、cm、top_xtr、bot_xtr、cl_error | 不存在的可选列为null；cl/cd/cm/alpha必须有效 |
| 模型元数据 | iteration、训练ID列表和数据哈希、kernel、正则、版本、种子、LOO指标 | 单独保存，不只保存joblib二进制 |
| Proposal | iteration、m、p、mu_cd、sigma_cd、ei、选择原因、排重距离 | proposal不是成功样本；须经过真实评估 |

计算状态采用 planned→running→success/failed；被异常阈值拦截时为 needs_review，当前阶段保存现场并以非0状态退出，待显式复查结论写入事件表后续跑；预算不足为 skipped。split、数据用途（baseline/doe/sequential/verification）和状态分字段保存，不能用一个状态混装不同含义。

SQLite 以 designs、evaluations、attempts、models、events 为最小逻辑表。M2 已将 `SQLiteEvaluationStore` 移入 `store.py`，保留旧导入，并增加 stages 和进程身份租约；M3 已使用 models 表注册数据哈希、模型文件哈希与完整元数据。M1 账本自动迁移且不重置启动计数。失败 JSON 的气动对象为 null，禁止将 NaN 作为可交换格式。CSV 为 UTF-8，包含单位明确的列名，不用行号作为样本身份。

### 7.2 缓存和精度隔离

缓存键至少覆盖：几何文件哈希、m/p/t、生成器版本、Re/M/CL、转捩、所有面元设置、求解器文件哈希、评估策略版本。设计变量用稳定精度序列化，不能用只保留两位小数的字符串作缓存键。

同一设计在160与240、不同转捩或不同exe下不能命中同一缓存。只有通过质量检查的成功记录允许命中；失败记录保留用于排重和诊断。最终独立重复评估显式绕过缓存，但仍保存它与原evaluation的关联。

`samples.csv` 导出可训练标准结果和状态，不包含留出标签；`holdout.csv` 单独保存且仅供最终验证读取。训练查询必须显式要求 split=train、status=success、标准fidelity及匹配的配置哈希；不能只把全部cd非空行拿来拟合。

### 7.3 断点续跑

每次完成真实评估后先以数据库事务保存attempt与最终evaluation，再保存模型和checkpoint。checkpoint记录阶段、成功序贯数、不同设计计数、进程启动总数、随机数生成器状态、排除点、停止历史和当前模型数据哈希。

恢复时数据库为结果和预算事实源；校验文件哈希与模型训练ID，不一致则从已完成样本重训模型。发现running状态先核对进程身份与时间，不能误杀无关程序；已退出但输出完整的attempt先解析并入库，无有效输出则标为中断，按剩余重试和预算处理。不能重置预算继续无限重试。

检查点用临时文件写入后原子替换。随机状态和候选提议先落盘，再启动评估，避免崩溃后不断更换同一轮候选。标准数据更新前后均保留可追溯事件。

## 8. 关键实现细节

### 8.1 XFOIL 封装

采用 `subprocess` 启动绝对路径exe，参数列表与工作目录分开指定，不通过字符串拼接shell指令。单个attempt目录预先保证没有同名输出文件。英文ASCII几何名、会话文件名，日志可容错解码。

会话生成须覆盖关闭绘图、加载几何、统一布点、固定TYPE、黏性工况、转捩、PACC及CL等操作；准确的菜单层级和空行用所选二进制验证后保留一份成功会话fixture。输出解析按列标题定位，不能固定“跳过前12行”，也不能盲取最后一行。

首次目标点失败后允许一次全新进程延续重试（CL=0.2→0.3→0.4→0.5）。只接受0.5对应且满足容差的最终结果，保存全部延续点但不把它们当作额外设计训练数据。超过时限必须结束进程树并归档失败原因。

### 8.2 GP 与采集函数

在固定边界上归一化输入，不用每轮样本的最小最大值改变坐标系。GP可以内部标准化CD，预测输出必须回到原尺度；EI的f_best、ξ、均值和标准差必须同尺度。

LOO每次只拿当前训练样本，独立重新拟合，不把全数据拟合的均值/方差泄漏到被留出样本。最终6点验证集不参与训练、策略门槛或最优候选选择。

采集优化搜索的是便宜的GP。候选点必须过边界、几何和归一化距离检查；邻域排除包含已评估成功点、失败点、保留验证点、正在运行点。DE落在禁区时返回不可选值，再与有效候选池比较；不得“保留同一点再算一次”冒充新加点。可重复评估只在验证用途下显式允许。

### 8.3 预算与停止

分别维护成功设计数、不同标准设计尝试数、真实进程启动数，缓存命中不增加进程数。重试不增加不同设计数，但增加进程数。离散验证不增加标准设计数，增加进程数。预检中的基准标准点通过检查后复用，不能在统计中重复占初始样本名额。

最多35个计划有效标准设计、45个不同标准设计硬上限、120次总进程启动硬上限；最终12次独立复核必须预留。重试前再次检查剩余预算。失败率按一个设计在允许重试后的最终结果统计，不能把首次失败但重试成功算成失败设计。

## 9. 必须完成的验证

测试只围绕会使优化结论错误的风险；无须为简单转发脚本逐行写测试。

| 文件 | 要验证的风险 | 输入与验收 |
|---|---|---|
| `test_geometry.py` | 型号取整、尾缘误用、点序错误 | 基准和边界参数；321点、单前缘、有限尾缘、连续参数扰动确实改变几何；解析特征及参考轮廓一致 |
| `test_polar.py` | 把失败/旧数据当有效 | 正常、多CL、无目标、NaN、D指数、缺列、崩溃/超时日志；只提取合格目标行 |
| `test_surrogate_acquisition.py` | 归一化/符号错误、留出泄漏 | 人工已知函数只用于算法测试；检查最小化EI方向、σ≈0、预测单位、重复过滤、训练ID不含holdout |
| `test_store_resume.py` | 重复算、错缓存、预算重置 | 模拟中断；恢复结果与不中断一致；N/工况改变不命中；强制复核确实新启动 |
| `test_xfoil_integration.py` | 实机菜单不兼容、挂起、求解无效 | 标记为integration，真实NACA2412单点及连续调用；保存版本、日志、CL误差和耗时 |

推荐执行顺序为 `python -m pytest -m "not integration"`，再 `python -m pytest -m integration`。pyproject.toml必须注册integration标记。没有exe时单元验证仍可运行，集成验证明确skip而不是标记通过；预检总状态仍为未完成。

此外，基准160/200/240离散对比、重复性、6点独立验证和最终12次复核属于数值实验验收，不能用单元测试通过替代。

## 10. 建议开发顺序与完成标准

| 里程碑 | 编写范围 | 可审查产物 |
|---|---|---|
| M1 能算一个点 | 环境、配置、geometry、runner、polar、evaluator最小实现、预检 | 一份真实几何和完整XFOIL会话、有效目标CD |
| M2 能可靠存数据 | store、baseline、sampling、DOE、恢复验证 | 基准质量报告、17+6样本、失败状态、可复现续跑 |
| M3 能建模提议 | surrogate、acquisition、初始拟合 | LOO、核参数、一个未评估候选及选择理由 |
| M4 能闭环更新 | optimizer、序贯脚本、预算与停止 | 至少一次真实新点使训练集及模型更新，best-so-far可追溯 |
| M5 能判断收益 | verification、reporting | 独立验证误差、同精度气动对比、面元敏感性、最终结论 |

最终报告应逐项回答：基准是什么；在何种条件下优化；花了多少真实调用；哪些点失败；代理误差多大；最佳候选是否经过独立真实求解；改善是否超过数值差异。若任何关键关卡未通过，输出具体未完成状态及下一步，不将程序成功退出写成“优化成功”。

最终复核对每个翼型应用流程指南中的200/240差值及200重复性门槛，再比较改善与数值差异。`best_airfoil.dat` 表示最终推荐几何：只有候选通过复核且改善超过双方不确定度尺度之和时才导出该候选，否则导出基准，并在metadata/报告中说明推荐理由。若基准自身复核失败，则不生成“最终推荐”文件，只保留诊断和候选数据。

## 11. 两份文档的一致性审查

本版已按以下项目进行方案级交叉修订：2个连续变量、固定t=0.12、CL=0.5、Re/M/转捩一致；321几何点与160面元设置区分；17训练+6验证+最多12新增=最多35计划有效设计；45设计与120进程预算区分；所有失败只留档；6点留出不参与寻优；最后3翼型×4次=12次独立复核；所有入口读取冻结配置。

M1 已核实本机 Windows 6.99 二进制、Python 依赖和 OpenBLAS 数值运行库，保存了真实基准 CD、五次独立调用的计时及预算。主实验 `runs/m1_20261013_02/` 的基准 CD=0.00599、CL=0.5000；共 6 次进程启动、1 个不同标准设计，缓存复用没有增加调用。原 MKL 探针失败记录保留在独立的 `m1_20261013_01` 中，没有混入成功实验。

M2 已在同一主实验完成：基准 CD_160/CD_200/CD_240 为 0.00599/0.00599/0.00600，160 重复差为 0，200/240 差为 0.00001≤0.00006；几何参考、转捩和 Cp 筛查通过。DOE 获得 17 个训练点和 6 个封存留出点，无失败替补；累计 31 次进程启动、23 个不同标准设计，最小归一化距离约 0.0114124。4 点后跨进程恢复保持坐标、已完成评估 ID 和随机状态，仅新增剩余 18 次调用；已完成阶段重跑不增加调用。证据为 `baseline.md`、`doe.md`、`recovery-validation.json`。

M2 新模块为 `records.py`（共享结果类型）、`store.py`（事务/CSV/阶段状态）、`sampling.py`（LHS/Sobol/排重/maximin）、`stages.py`（关卡与租约）、`baseline.py`、`doe.py` 和 `m2_cli.py`。`--stop-after N` 仅用于显式恢复验证；正常 DOE 不设置它。异常点通过 `--resume --review-eval-id … --review-decision accept|reject --review-reason …` 显式处理。Cp 筛查工程阈值及独立 UIUC 几何对照阈值保存在基准报告，图像供进一步审查。

当前恢复范围为 M2–M5 和有持久请求的 evaluator 尝试；01只创建新实验，07从核验过的源证据重建报告。恢复核验 PID 与创建时间，存活原进程、身份缺失或已提交原始产物变更时停止；完整退出审计及有效输出存在时先入库，中断尝试只使用剩余重试和预算。真实硬退出测试覆盖进程完成但结果未入库、启动事件已入库但求解器尚未运行、evaluation已提交但优化历史尚未更新，以及M5 evaluation已提交但复核槽位尚未更新。M5最终模型冻结后不再自动重训该模型。

M3 已生成 `surrogate.py`、`acquisition.py`、`fitting.py`、`04_fit.py`。主实验使用 17 个真实成功标准训练点，核为 Constant×ARD Matérn(2.5)，长度尺度约 [0.386200,0.744768]，alpha=1e-6。17 折独立重新拟合的 LOO MAE=9.81823e-5、RMSE=1.41062e-4、最大绝对误差=3.75049e-4；MAE≤2.995e-4 门槛，未触发下一轮强制探索。

首个提议 m=0.0296878183732、p=0.467042903972，预测 CD≈0.00549696、σ≈8.61643e-5、EI≈9.07321e-5。差分进化的 EI 高于有效池最大 EI≈9.04931e-5，最小排除距离≈0.177177；提议仅持久保存，不写入真实样本库。M3 未新增 XFOIL 调用，主实验仍为 31 次/23 个不同标准设计。证据见 `fit.md`、`models/iter_000/`、`m3-recovery-validation.json`。

模型、训练快照、依赖版本与原始会话身份均校验后才复用；损坏派生产物需 `--resume` 从核实的样本重建，并保留旧文件。LOO 每折提交到 SQLite，恢复不重新拟合已完成折；提议提交后复用同一坐标/种子。`--stop-after-folds N` 仅为恢复诊断。真实数据集的独立恢复验证在复制账本/只读原始会话的新目录执行，不伪装成新采集的数据集。M3 正常及失败路径均不读取留出标签。

候选池与 DE 策略：默认 Sobol 4096+去重边界网格后 4352 点；DE 使用 maxiter=128、popsize=15、tol=1e-6、atol=1e-12、polish=True、workers=1，seed=1000+k。原配置未给出 DE 收敛选项，以上实现选择写入每条提议。目标求值仅使用 GP；选中坐标额外通过几何和距离检查。无效/退化 EI、LOO 触发或第四个成功新增点使用距训练点最远的有效池点。

M4 已生成 `optimizer.py` 与 `05_optimize.py`：持久提交候选/种子/评估 ID，再由 evaluator 执行标准真实求解，核验标签与原始会话，成功后更新训练集及 GP/LOO。失败样本不进入训练，不消耗成功名额；重试成功按一个成功设计计数。已完成折和 evaluation 可恢复，派生模型损坏时重建并保留旧文件；原始证据受损停止。M4 的训练 CSV 导出、复查和异常路径保持留出文件封存。

主实验新增 8 个真实成功点、无最终失败，可训练点从 17 增至 25，最终模型 `iter_008`。最近五次成功新增动作的 best-so-far 改善 B[n−5]−B[n]=0，当前合法池最大 EI=2.51108e-6<1e-5；已完成至少 8 次成功，故以 `converged` 正常停止。第 4/8 点采用周期 maximin 探索，未把暂时无改善视为失败。最终 LOO MAE≈1.43947e-4≤2.995e-4，最大绝对误差仍保存在逐折报告中。

累计 39 次进程启动、31 个不同标准设计，12 次最终复核预留未动用。标准 N=160 下最佳已接受 m=0.0244207128215、p=0.488133342722，CD=0.00544；仅为标准求解排名，未认证收益。首点/model `iter_001` 后跨进程恢复保留历史和模型，仅新增剩余七次调用。证据为 `optimization.md`、`history.csv`、`proposals/`、`models/iter_001`–`iter_008`、`m4-recovery-validation.json`。

M4 首次和恢复命令为 `python scripts/05_optimize.py --run-id …` / `--resume`；`--stop-after-evaluations N` 为显式恢复诊断。异常点用 `--resume --review-eval-id … --review-decision accept|reject --review-reason …` 提交结论。达到合法预算上限且已有真实更新时可正常停止；失败率、连续失败或环境错误返回非 0。已完成阶段核验并复用历史预算快照，后续验证调用不会被记成序贯花费。

M5 已实现 `run_verify(run_dir, resume=False, stop_after_evaluations=None)` 和 `run_report(run_dir)`。冻结对象包含最终模型注册记录/二进制哈希、训练 ID/数据哈希、按标准 CD 选出的两名非基准候选及留出坐标；12 个独立槽位先提交 SQLite，再记录解封事件、读取真实标签。最终阶段不重训、不换候选，不向训练表回填复核数据；M5 冻结后禁止新增训练/留出评估和改变最终 GP 注册记录。

`holdout_metrics`、`shape_quality`、`certify` 分别负责独立误差、全目标点/压力/离散/重复质量关卡及同精度收益判断。R² 在点数不足或目标方差≤默认 1e-10 时为 null；六点覆盖率只作诊断。默认12个新 eval_id 用预留预算、全新进程/目录和 Cp；恢复复用同一槽位已提交证据，不冒充新的独立重复。needs_review 停机，明确 accept/reject 事件后才继续，Cp 在等待复查时也保留原始哈希。

根目录阶段摘要和 `reports/` 约定目录同时生成 Markdown/JSON/CSV/PNG 及几何元数据。07 重新核验源证据但不启动 XFOIL 或拟合 GP；缺阶段/损坏证据只生成诊断并撤回活动推荐，基准复核失败不导出 best_airfoil。派生 JSON 使用稳定字段顺序，报告重建后更新其文件 receipt；替换损坏产物前留档。基准保留和候选推荐都带明确原因、冻结身份和独立 eval_id。

主实验 M5：留出 MAE=5.12602e-5≤2.995e-4，RMSE=8.17385e-5、最大误差=1.89826e-4、R²=.938411、2σ覆盖6/6。基准 CD240=.00600；候选1 CD240=.00545，候选2=.00546；三者质量均通过，重复200差均为0，ε各约1e-5。第一候选 m=.0244207128215,p=.488133342722,t=.12，同精度 ΔCD240=.00055>ε之和≈.00002，改善约9.17%。累计51次启动、31个不同标准设计；25个训练点和最终模型未改变。结论限于冻结的二维 XFOIL 工况，不证明实验/三维性能或全局最优。

证据见 `report.md`、`reports/summary.md`、`verification.csv`、`holdout-predictions.csv`、`best_airfoil.dat/json`、`m5-recovery-validation.json` 和测试摘要。主实验四次后跨进程续跑只新增八次；独立复制账本另以真实硬退出验证 evaluation 提交与槽位提交之间的恢复，不重复启动原求解。M1 的合成 GP 探针仍只检查数值依赖，没有写入真实样本库。
