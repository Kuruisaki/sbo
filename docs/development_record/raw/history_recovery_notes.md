# 历史恢复说明：哪些事能查证，哪些事还不知道

采集范围：`D:\codex\sbo`，项目 `Kuruisaki/sbo`；归档日期2026-10-15（Asia/Shanghai）。本次仅只读分析与文档生成，没有启动XFOIL，没有新SBO实验，没有修改既有源码/配置/运行产物，没有运行pytest/项目入口，没有commit/push。

收录说明：下文描述首次归档的采集过程。`local_history_sources.json`、`powershell_project_history.json` 和 `terminal_recovery.txt` 为本机元数据，仅在本地保留，由 `.gitignore` 排除；其余项目证据随开发文档入库。文中的 Git 状态与检查结果都是当时的快照。

这份说明不是新的开发日志，而是解释归档材料从哪里来、能证明什么。简单说：保存的聊天和执行输出能证明“当时做过什么”；源码能说明“现在怎样工作”；两者都没有留下的信息，就只能写“无法确认”。

如果只是想了解开发过程，读上一级目录的第 01 章即可。需要核对某一句历史描述的依据时，再看这里和相应的原始记录。

## 1. 能恢复的历史（A）

### Codex应用会话

首次归档时，通过应用的会话列表和 `read_thread` 接口读取了三个由用户发起、工作目录属于本项目的会话：

| 线程标题（原样） | ID | 已完成轮次 | 用途 |
|---|---|---:|---|
| 编写SBO流程与代码目录指南 | 01a133bb-c9f7-7ba2-ba1d-5db0b83e1224 | 1 | 原始目标、选择XFOIL、两文档方案 |
| Agent1-basic | 01a13494-856c-7fd2-84bb-26ef79b25c2a | 9 | YAML→geometry→runner→polar→M1→M2→M3→M4→M5 |
| Agent2-use | 01a13ac2-29d7-7df0-a1d7-ceaa9b88c3a3 | 3 | Git初始化、作者/仓库建议、local配置并commit/push |

总共13个已完成轮次、14个用户消息条目（方案轮次含一次提问选择回复）、210个终端执行条目。`codex_conversation.json`保留公开用户文字、Codex commentary/final回复、fileChange、commandExecution、webSearch；不导出reasoning/analysis、系统/开发者指令、账号标识。没有另建Conversation Summary；公开回复与操作只作为01章开发事件的依据。

首次采集时，归档任务本身还在进行，因此导出只收录该任务开始以前的记录；截止时间是（UTC `2026-10-14T15:54:48+00:00`，本地23:54:48）；避免把归档自己读取/写文档的工具日志递归混入项目开发历史。原始当前用户归档需求在本会话可见，但未将整份长需求重复导出。若之后继续开发，当前归档不是实时同步器。

### 本地Codex日志与命令输出

可合法读取用户授权的项目会话JSONL。三个来源文件及采集时SHA见 `local_history_sources.json`；活跃原文件会继续追加，所以其hash只对应采集瞬间。来源角色先检查：只导出三个人类交互线程，不把其他非用户线程的内部记录当作用户Prompt，也不读取不相关项目内容。

`execution_log.jsonl`保留236个历史工具调用与220个对应输出；差额是外部网页正文输出未转存。导出只保留记录中的文字输出（input_text/text），不复制图片；原来已经截断的工具输出仍是截断版本。每条包含thread、原调用ID、UTC时间和源日志行号，可与命令/fileChange对应。原始函数执行代码可能同时含多个子调用，不能将条目数当作高保真启动次数。

`command_history.json`来自应用commandExecution记录：命令、cwd、状态、shell封装exitCode和duration。时间字段 `turn_started_at_utc` 表示那一轮对话开始的时间，并不是每条命令精确开始的时间。脚本通过Python subprocess间接启动时，原记录保留父Python命令，子CLI命令与returncode通过父代码及保存的stdout/stderr确认。这里有三层程序：终端外层命令、项目脚本和 XFOIL。它们各有自己的退出码；外层返回 1，并不能据此断言内部脚本或 XFOIL 也返回 1。

已脱敏Authorization头和已知API/GitHub token模式；排除凭据文件读取、无关会话、外部网页响应正文、图片和内部推理。保留用户自己提供的项目远程地址、Git作者信息以及项目路径；没有复制账号/认证文件。

### 终端历史补充

应用当前线程没有附加terminal session，实际读取返回“No app terminal session is attached to this thread yet.”，保存于 `terminal_recovery.txt`。因此无法恢复一份该面板的持续终端屏幕/完整历史。

另外读取现存ConsoleHost PSReadLine历史，只导出明确包含项目路径、`sbo-xfoil`、项目远端或run-id的3行至 `powershell_project_history.json`。它们是：Conda dry-run输入、激活sbo-xfoil、裸xfoil.exe路径。没有时间、输出或退出码：只能确认保存了这些输入，不能证明它们何时执行成功，不能把裸exe这一行加入主实验的51次账本，也没有在本次执行这些行。不导出该全局历史中的其他项目命令。

### Git

执行并保存了 `git status --short --branch`、`git log --oneline --decorate --graph --all`、`git diff`、`git diff --staged`、`git branch -vv`，另保存log --stat/name-status和remote。完整调用/退出码/分别的stdout在 `git_commands.json`。

Git 在采集时只有一次初始提交 `d670b51`（73 个文件、11995 行新增）；本地 main 分支与本地保存的 origin/main 指向同一提交。没有M1–M5各自commit/diff；工作区/暂存区diff为空，因此 `git_diff.patch`为空是实际结果，不是遗漏。未跟踪的docs不会出现在git diff中。归档前工作区干净；新增归档后为`?? docs/`，未暂存/提交。未联网fetch，不能把本地origin/main说成实时服务器验证。

### 运行与测试

只读URI `samples.sqlite?mode=ro`查询主实验，实际51次launched、0预留、25 train/6 holdout、9 models、6成功stages。所有原始run、receipt、CSV、hash、恢复验证和测试XML是已经存在的证据。本地共有36个run目录，包括失败/诊断和继承账本测试；不能把它们都叫独立成功SBO实验。

`run_evidence.json`保存主要JSON/hash和SQLite查询，`run_inventory.json`说明其他目录的来源/状态。主实验阶段预算是6→9→31→31→39→51；M4报告保持39，不把M5追加费用改写历史。真实恢复测试复制账本而只读源会话文件；继承启动数不代表该验证目录新求解了全部历史点。

测试统计的过程是：先有一次包含 299 项的测试，其中 7 项失败；修复后又分别复测 17、3、2 项。因为这些批次有重复测试，统计时按测试名称识别同一项，只保留其最后一次结果，最终得到 300 个不同测试通过。不能把批次数量直接相加，也不能说第一次就全部通过。归档只再次读取/计数这些XML，不重新测试，也不删除旧失败输出。

## 2. 当前代码重建内容（B）

02–05章的职责、调用关系、argparse参数、数学公式、数值门槛、模型/数据/状态流直接来自当前19个核心模块、7入口、测试、YAML和产物。四层是职责划分，不是所有imports都严格单向。架构/数学说明不是恢复到的用户原话，不能据此反推某个未记录的修改动机。

01章恢复27个主要事件：一个原始方案节点，M1–M5中的文件实现、实际问题/测试/恢复/验收节点，以及Git交付。一个用户Prompt对应多个事件时明确“沿用任务，没有单独Prompt”，没有拼造新的需求对话。

## 3. 无法确认的历史（C）

- 未保存在三个线程/日志中的早期草稿、完整连续键盘操作、所有人工检查结论；无法从当前保存记录确认。
- Miniforge最初安装、完整环境create成功命令、此前手动XFOIL交互的时刻/结果；PSReadLine三行不足以证明这些过程。
- BLAS故障的具体DLL/符号/ABI根因；可确认原生探针失败与换OpenBLAS后的通过，不能补造底层诊断。
- 逐阶段Git提交/分支演进：当前Git只保存最终初始快照，不存在可查询的M1–M5独立commits。
- 原工具已截断的stdout全文、未记录输出、未保存的外部手动命令；不能从最终代码补写成实际运行。
- GitHub在本地引用之后是否发生变化，其他未查询ChatGPT讨论或非用户内部记录是否还有背景；本归档没有声称全部账户历史都恢复。

## 4. 怎样确认归档没有改动原实验

SHA256 可以理解为文件内容的指纹：内容变化时，指纹通常也会改变。`baseline_integrity.json` 在首次文档生成早期记录核心源码、配置、原README/设计文档、环境/pyproject/.gitignore、整个主实验已有文件及SQLite SHA256；`consistency_check.json`末次重新比较，任何变化都单独报错，不依赖“没有运行solver”的口头承诺。

捕获与检查辅助脚本只写docs/development_record，使用 `python -B`避免生成项目pyc，AST/JSON/YAML/CSV/XML/哈希读取，SQLite mode=ro。它们不导入sbo、不执行01–07、pytest、Conda安装、GP fit或求解器。重新运行捕获会更新归档快照本身，绝不改变旧实验；不应将它当成真实运行入口。

## 5. 后续文字优化的范围

2026-10-15 对这些 Markdown 做了面向本科生的文字优化：增加术语解释，把密集段落拆开，并解释公式、阶段和文件的实际用途。原始用户引文、命令、历史数值及原始证据文件继续保留。文字易懂不等于降低证据要求：没有保存的历史仍然不补写，模型预测也仍然不能当作真实 XFOIL 结果。

这里涉及的 Git 状态、会话条目数和测试数量，都是首次采集时的快照；它们不是自动更新的实时状态。原始 `consistency_check.json` 同样记录首次归档检查，不应冒充后续每次文字修改都重新执行了全部检查。
