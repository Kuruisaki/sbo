# SBO + XFOIL

本项目已实现 M1–M5：连续 NACA 几何、XFOIL/目标点验收、部署预检、基准质量关卡、持久样本库、DOE/留出采样、GP/LOO、真实序贯更新、独立留出与多面元复核、收益判断及报告导出。主实验完成 12 次独立最终复核，已生成推荐几何。

开发过程、软件架构、实际执行流程、SBO 数学流程和文件职责见[项目开发归档](docs/development_record/README.md)。

## 运行预检

在项目根目录执行：

```powershell
conda activate sbo-xfoil
python -m pip install --no-deps --no-build-isolation -e .
python scripts/01_preflight.py --config configs/naca2412.yaml --run-id demo_001
```

`run-id` 必须是新名称；已有实验不会被覆盖。先读取并严格校验 202 个配置字段，再冻结配置并保存哈希。预检验证声明中的依赖版本和一次合成数据 GP 拟合，然后计算真实基准点，执行五次独立串行调用，检查基准缓存复用及调用预算，最后导出实际 Conda 构建锁。数值探针在有时限的独立进程中执行，其底层崩溃不会被当作通过。

成功返回 0；异常点、环境问题、预算不足及求解失败返回非 0，并保留原因。预检只代表 M1，不代替 160/200/240 面元敏感性与收益复核。

Windows 的运行库显式选用 OpenBLAS；Python 库版本仍采用设计文档中的精确版本。`environment.yml` 记录运行库选择，实际构建号以成功实验导出的 `environment-win-64.lock.txt` 为准。已有 MKL 环境可按 Conda-forge 官方支持的方式切换：

```powershell
conda install -n sbo-xfoil --override-channels -c conda-forge "libblas=*=*openblas" "libcblas=*=*openblas" "liblapack=*=*openblas"
```

依据：[Conda-forge BLAS 切换说明](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation)。

## 评估器接口

```python
from sbo.config import load_config, create_run
from sbo.evaluator import Evaluator

config = load_config("configs/naca2412.yaml")
run_dir = create_run(config, "example_001")
with Evaluator(run_dir) as evaluator:
    baseline = evaluator.evaluate(save_cp=True)
    point = evaluator.evaluate([0.0237, 0.436], purpose="doe")
    independent = evaluator.evaluate(
        [0.0237, 0.436], purpose="verification", panel_count=240, save_cp=True
    )
```

评估器只读取该实验的冻结配置。标准点最多两次尝试，第二次用同样物理/几何/面元设置和 0.2→0.3→0.4→0.5 的 CL 延续路径。失败点没有阻力罚值；`needs_review` 保留测得气动值并立即停在复查状态，不自动重试或接受。

仅 `result.usable_for_training` 为真的结果才可进入训练。留出点和复核点均不可训练；最终复核必定绕过缓存。一般调用保护 12 次最终复核预算，只有显式 `purpose="verification", use_reserved_budget=True` 才可使用该预留。新增 Cp 输出必须通过独立复核调用获得，不能伪造旧缓存的压力数据。

`store.py` 中的 `SQLiteEvaluationStore` 保存 designs/evaluations/attempts/events/stages/metadata，并预留 models 表。打开原 M1 账本时迁移到 M2，不重置预算。旧导入 `from sbo.evaluator import SQLiteEvaluationStore` 仍可用。缓存命中重新校验原始文件、输入会话和求解器身份；`Evaluator.resume(eval_id)` 根据已落盘的请求恢复同一个设计及剩余尝试。

## 运行 M2

完成 M1 后，在同一个 run-id 上执行：

```powershell
python scripts/02_baseline.py --run-id demo_001
python scripts/03_doe.py --run-id demo_001
```

基准关卡复用预检的 160 面元点，独立计算 200、240 和一次 160 重复，检查 CD 差值、UIUC 几何对照、转捩位置及 Cp；生成 `baseline.json`、`baseline.md`、`baseline.png`。几何/Cp 使用明确记录阈值的自动异常筛查，并保存图像供审查。

DOE 使用训练种子 42 生成 16 个 LHS 点，加入已验收基准得到 17 点；留出种子 2026 独立生成 6 点，坐标在任何 DOE 求解前预留。失败点保留为空标签，按持久随机状态进行 maximin 替补，并记录替补后不再严格满足 LHS。所有已有成功、失败和预留坐标均参与排重。可训练查询 `store.training_data()` 只返回当前冻结配置下的成功标准训练点；`holdout.csv` 单独保存，仅供后续最终验证读取。

中断后恢复对应阶段：

```powershell
python scripts/02_baseline.py --run-id demo_001 --resume
python scripts/03_doe.py --run-id demo_001 --resume
```

SQLite 是结果、预算、提议及随机状态的事实源；`checkpoint.json` 和 CSV 是原子写出的可重建快照。完整进程退出审计和有效输出存在时直接重新解析入库；中断尝试保留启动计数，只使用剩余重试。仍存活的原 PID/创建时间身份会阻止恢复；身份缺失或已提交产物被修改时明确停止。旧 M1 的未完成请求缺少恢复元数据时也会停止。已完成阶段重跑只核验/复用，不增加调用。`--stop-after N` 是 DOE 的显式恢复验证选项，完成 N 个新评估结果后返回非 0 并保留现场。

遇到 `needs_review`，由用户检查原始结果后提交明确结论，例如：

```powershell
python scripts/03_doe.py --run-id demo_001 --resume --review-eval-id eval_ID --review-decision reject --review-reason "已检查原始输出并决定排除该异常点"
```

`accept`/`reject` 和非空理由写入事件表；不会自动接受异常点。启动、求解器身份或清理错误会立即停止，避免通过不停换设计掩盖环境问题。

## 已验证实验

`runs/m1_20261013_02/` 在原 M1 基础上完成 M2：基准 CD_160=0.00599、CD_200=0.00599、CD_240=0.00600，160 重复差=0，200/240 差=0.00001≤0.00006。获得 17 个训练点、6 个封存留出点，累计 31 次 XFOIL 启动、23 个不同标准设计，无失败替补或挂起占位。DOE 在 4 点处主动保存现场，再由新进程恢复；坐标、已有评估 ID、随机状态保持一致，只新增剩余 18 次调用。证据见 `baseline.md`、`doe.md`、`recovery-validation.json`。

采样 API 依据 [SciPy 1.14.1 LatinHypercube](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.qmc.LatinHypercube.html) 与 [Sobol](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.qmc.Sobol.html) 官方文档；独立几何数据来自已保存哈希及来源的 [UIUC NACA 2412](https://m-selig.ae.illinois.edu/ads/coord/naca2412.dat)。没有生成模拟气动标签填充正式样本。

## 运行 M3

在 M2 已通过的同一实验上执行：

```powershell
python scripts/04_fit.py --run-id demo_001
# 恢复中断，或从已核验的训练数据重建损坏的模型产物：
python scripts/04_fit.py --run-id demo_001 --resume
```

`surrogate.py` 使用固定设计边界归一化输入，采用 ConstantKernel×ARD Matérn(ν=2.5)，内部标准化 CD，五次超参数重启、seed=43。预测均值和标准差由 [scikit-learn 1.5.2 GPR](https://scikit-learn.org/1.5/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) 自动还原到 CD 原尺度；EI 使用同一尺度。Cholesky 失败才按声明的 alpha=1e-6→1e-5→1e-4 回退；重复/过近点、混合配置、非标准结果和未接受异常点会先被拒绝。所有拟合警告、回退尝试、核参数、库版本及训练 ID/数据哈希均保存。

LOO 每次仅使用其余训练点，独立重新拟合核和输出标准化，记录 MAE、RMSE、最大绝对误差和残差图。MAE 超过 max(5e-5,5%×基准 CD_160) 时保存下一轮探索标志；不把这一策略门槛称为独立泛化认证。初始模型始终绑定 M2 的基准+DOE 训练 ID，后续序贯样本由 M4 的新模型处理。

`acquisition.py` 生成 4096 个 Sobol 点，补充 65 点/边的边界网格和四角（去重后默认总计 4352 点）。按归一化距离排除成功、失败、运行及预留留出坐标，比较有效候选池与 [SciPy 1.14.1 differential_evolution](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.differential_evolution.html) 的 EI。差分进化在单位方形内以 seed=1000+k 搜索，显式 maxiter=128、popsize=15、tol=1e-6、atol=1e-12、polish=True、workers=1；这些搜索策略写入提议。最终候选必须通过几何及距离检查。LOO 触发、每第 4 个成功新增点、EI 无效/退化时使用合法池中距训练点最远的 maximin 后备。

该阶段输出 `fit.json`/`fit.md`，以及 `models/iter_000/` 中的 `gp.joblib`、`metadata.json`、`training.json`/`training.csv`、`loo.json`/`loo.csv`/`loo.png` 和 `proposal.json`。模型注册表、LOO 已完成折和候选保存在 SQLite；checkpoint 是可重建投影。候选状态为 `proposed`/`assessed=false`，不写入真实设计/训练表，也不启动 XFOIL。模型加载前核验数据库记录、文件哈希、训练快照和依赖版本；损坏派生产物需要 `--resume` 重建，原文件保留在 `models/iter_000/recovery/`。原始求解输出受损或标签与日志不符时停止。

`--stop-after-folds N` 是显式恢复验证选项，在 N 个新 LOO 折提交后返回非 0，再用 `--resume` 接续。M3 正常及失败路径均不打开封存留出标签。留出坐标用于排重；标签继续等待 M5。

主实验在 M3 阶段完成的历史记录为：17 折 LOO MAE=9.81823e-5、RMSE=1.41062e-4、最大绝对误差=3.75049e-4，MAE 门槛=2.995e-4，通过该门槛。拟合 ARD 长度尺度约 [0.386200,0.744768]，alpha=1e-6。首个提议 m=0.0296878183732、p=0.467042903972，预测 CD≈0.00549696、σ≈8.61643e-5、EI≈9.07321e-5，来自差分进化；最小排除距离约 0.177177。该候选在 M3 结束时尚未求解，后续真实结果见 M4 记录。M3 新增 XFOIL 调用为 0，主实验累计仍为 31 次、23 个不同标准设计。恢复记录见 `m3-recovery-validation.json`；独立真实数据恢复测试另生成 `m3-real-validation-*/m3-validation.json`，复制账本并只读原始会话，明确区别于独立采集的新数据集。

## 运行 M4

在 M3 已通过的同一实验上执行：

```powershell
python scripts/05_optimize.py --run-id demo_001
# 恢复中断或重建损坏的派生模型：
python scripts/05_optimize.py --run-id demo_001 --resume
```

`optimizer.py` 实现串行闭环：提交提议/种子/固定评估 ID→标准 XFOIL 求解→核验原始证据→更新训练集→重新拟合 GP 和逐折 LOO→更新下一点与停止判断。首次沿用仍合法的 M3 提议；失败设计坐标继续排除，训练样本不变时复用同一模型并用新的轮次种子生成替补。成功新增点依次生成 `models/iter_001/`、`iter_002/` 等，模型 ID 与成功点数对应，失败不占成功名额。

最多新增 12 个成功点；有效标准设计计划上限 35、不同标准设计上限 45、总启动上限 120，始终保护最终复核的 12 次预留。失败率按一个设计在重试后的最终结果计算，重试成功不会算失败设计。连续 3 个最终失败即停止；最近 10 个序贯设计窗口完整且失败率>20% 时停止。异常点暂停等待显式 `accept`/`reject`；启动、求解器身份或清理故障立即停止。

完成至少 8 个成功新增点后，最近 5 次成功新增动作的 best-so-far 总改善≤2e-5，并且当前合法池最大 EI<1e-5 时停止。实现的改善窗口明确为 B[n−5]−B[n]，包含五次更新；无效 EI 不会被当作 0 来提前收敛。第 4、8 等成功新增点及 LOO 触发时采用 maximin 探索。

`--stop-after-evaluations N` 是显式恢复诊断：N 个新设计结果及相应模型更新完成后保存现场并返回非 0。恢复使用 SQLite 的提议、评估 ID、历史和已完成 LOO 折；求解完成但历史尚未提交时先复用该 evaluation，不重复启动。模型文件损坏时从已核实的原始数据重建，保留旧文件；原始会话受损则停止。已完成阶段复用历史报告和停止时的预算快照，后续 M5 调用不会回写成 M4 花费。

如需处理异常点，先检查原始输出，再明确提交结论：

```powershell
python scripts/05_optimize.py --run-id demo_001 --resume --review-eval-id eval_ID --review-decision reject --review-reason "已检查输出，排除该异常点"
```

M4 只更新训练 CSV，保留 M2 的封存留出文件及其原摘要；正常、失败、恢复和复查路径均不打开留出标签。新增产物为 `optimization.json`/`optimization.md`、`history.csv`、`best-so-far.png`、`proposals/iter_<k>.json` 及各成功轮次的模型/训练快照/LOO 结果。终止报告区分正常停止、失败阈值及未发生真实更新；只有至少一次成功真实更新且正常停止才标记 M4 通过。

主实验已完成 M4：8 个真实成功新增点，无最终失败，训练集达到 25 点，封存留出仍为 6 点；最终模型 `iter_008`。最近五次成功更新改善为 0，合法池最大 EI=2.51108e-6，满足收敛条件。累计 39 次 XFOIL 启动、31 个不同标准设计，最终 12 次预留未动用。标准 160 面元下最佳已接受点为 m=0.0244207128215、p=0.488133342722，CD=0.00544；这是未经 M5 认证的标准结果。最终 LOO MAE≈1.43947e-4，门槛=2.995e-4。首点后跨进程恢复保留历史和 `iter_001` 模型，仅新增剩余七次调用；证据见 `m4-recovery-validation.json`。真实硬退出测试另保存在 `m4-real-validation-*/m4-validation.json`，明确记录继承源账本及一次独立新求解。

## 运行 M5

在 M4 正常结束的同一实验上执行：

```powershell
python scripts/06_verify.py --run-id demo_001
python scripts/07_report.py --run-id demo_001
# 中断后继续同一批独立复核任务：
python scripts/06_verify.py --run-id demo_001 --resume
```

`verification.py` 先核验全部训练原始证据和最终 GP，按标准 CD 选择两名不同非基准训练候选，把模型二进制/元数据、训练 ID/数据哈希、候选及 12 个新评估 ID 提交到 SQLite，再记录 `holdout_unsealed` 事件，开放封存标签。`final_holdout_data(freeze_hash)` 验证这一冻结记录；留出指标不进入训练、调参、停止或候选选择。最终模型注册表和新训练/留出求解从此锁定，若要改变实验须创建新 run-id。

留出报告包含 MAE、RMSE、最大绝对误差、R²、均值±2σ覆盖率及逐点残差。MAE 门槛为 max(5e-5,5%×基准 CD160)。目标方差≤数值差异下限的平方（默认 1e-10）或点数不足时，R² 为 null 并说明原因；覆盖率只作六点诊断，不宣称严格校准。误差超限如实标记代理质量失败，不重新拟合或更换候选。

基准及两名候选各自独立求解 N=160、200、240，另重复一次 N=200，共正常 12 次启动，使用预留预算并保存 Cp/转捩。每个槽位有固定的新 eval_id；正常复核绕过旧缓存，显式恢复才复用同一已提交结果。每次重试仍计入全局启动预算。失败点没有罚值；异常点立即暂停，只有明确检查后可提交 `--resume --review-eval-id … --review-decision accept|reject --review-reason …`。

所有目标点有效、Cp/转捩筛查、200/240 面元差及 200 重复差均为关卡。ε=max(|CD160−CD200|,|CD200−CD240|,|两次CD200之差|,1e-5)，是工程对比尺度。合格候选按 CD240 排名，只有同精度改善 ΔCD240>ε基准+ε候选才认证；同时保存独立 N=160 的比较。浮点边界使用 CD 幅值×16 倍机器精度的舍入容差，远小于 CD 输出分辨率。没有可认证改善则推荐基准；基准复核失败则不导出推荐几何。

`reporting.py`/07 不启动求解器或拟合模型，只重新核验证据并输出报告。产物包括根目录的 `verification.json/csv/png`、`holdout-predictions.csv`、`report.md/json/png`、`best_airfoil.dat/json`，以及设计文档约定的 `reports/summary.md`、`metrics.json`、`final_comparison.csv`、`best_airfoil.dat/json`、`figures/`。报告缺阶段或源证据受损时返回非 0，保存诊断并撤回活动推荐几何；派生报告可以重建并保留被替换文件。`--stop-after-evaluations N` 用于显式恢复诊断。

主实验 `m1_20261013_02` 的 M5 结果：留出 MAE=5.12602e-5、RMSE=8.17385e-5、最大绝对误差=1.89826e-4、R²=0.938411，六点均在均值±2σ内；MAE≤2.995e-4 门槛。25 个训练点和 `iter_008` 二进制保持不变，留出 CSV 未改写。

| 独立复核 | CD160 | CD200 | CD240 | CD200重复 | ε |
|---|---:|---:|---:|---:|---:|
| 基准 m=.02,p=.4 | .00599 | .00599 | .00600 | .00599 | 1e-5 |
| 候选1 m=.0244207128215,p=.488133342722 | .00544 | .00545 | .00545 | .00545 | 1e-5 |
| 候选2 m=.0233189234746,p=.497234726530 | .00546 | .00546 | .00546 | .00546 | 1e-5 |

三者全部通过，第一候选 ΔCD240=.00055，超过 ε 之和≈.00002；同精度相对改善约 9.17%，推荐第一候选，厚度仍为 .12。结论限于本次二维 XFOIL 工况和设计边界，不是实验/RANS/三维验证，也不证明全局最优。主实验累计 51 次启动、31 个不同标准设计，无失败评估或重试。

主实验在首四次复核后跨进程恢复，仅新增剩余八次，见 `m5-recovery-validation.json`。真实硬退出测试在独立复制账本中验证 evaluation 已提交、槽位尚未提交的窗口，结果在 `m5-real-validation-*/m5-validation.json`；继承的历史数据和新发起的 12 次求解分别说明。测试证据与源码哈希见 `m5-test-summary.json`。

## 实验产物

`runs/<run_id>/` 包含冻结配置、manifest、预检/环境诊断与构建锁、`samples.sqlite`，以及各 evaluation/attempt 的几何、输入、stdout/stderr、polar、Cp、`process.json` 和 `result.json`。M2 增加基准/DOE、checkpoint、datasets 元数据和分离 CSV；M3 增加初始模型、训练快照、逐折 LOO 和未评估提议；M4 增加真实序贯历史、各轮模型、停止诊断和恢复审计。失败气动值在 JSON 中为 null，在 CSV 中为空。各阶段的报告和模型保留当时的数据/预算快照；M3 的未评估提议是 M4 启动前的历史记录。

## 测试

```powershell
python -m pytest -m "not integration"
python -m pytest -m integration
```

集成测试使用配置中的真实 XFOIL，可执行文件缺失时明确跳过，不代表部署预检通过。测试产生的独立运行目录和已脱敏的真实会话 fixture 都与正式实验隔离。
