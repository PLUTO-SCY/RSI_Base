# Harness evolution：benchmark 与实验设置核查

核查日期：2026-11-16。读取原文方法、实验与相关附录，并交叉核对官方配置。以下是文献事实和本项目建议；未运行第三方 RSI 仓库，不将公开脚本视为已经复现成功。模型、任务、编辑空间、数据划分和预算需要一起对齐。固定文献版本与下载指纹见[source manifest](harness-benchmark-sources.json)；原文缓存位于本地忽略目录，不上传原文或第三方实现。

## 先给实施判断

当前原型是单次模型调用上的策略提示搜索。正式 harness evolution 应让任务真正经过可改进的记忆、工具、检索或执行流程。下一步优先参考 **Meta-Harness 的记忆分类**；再接 **HarnessForge 的 ToolHop**。数学保留为对照，若继续做，应参考 AFlow/WHALE 的代码执行环境。

这并不要求继承这些论文的 RSI 代码。我们独立实现小型运行器，复用公开 benchmark 数据、固定评分规则及可核查的实验协议。

## 快速对照

| 工作与核查版本 | benchmark | 实验设置中的关键点 | 对当前原型的启示 |
| --- | --- | --- | --- |
| [Meta-Harness，2603.28052v1](https://arxiv.org/html/2603.28052v1)，§4.1–4.3 | LawBench、Symptom2Disease、USPTO；数学检索；TerminalBench-2 | 分类20轮×2候选；改记忆/检索/上下文代码。数学用250题搜索、200道未见题测试，并有大型解题检索库 | 最适合从公开、短输出的记忆分类开始 |
| [AFlow，2410.10762v3](https://arxiv.org/html/2410.10762v3)，§4–5、A.4/A.6 | GSM8K、MATH、HumanEval、MBPP、HotpotQA、DROP | 20%搜索验证/80%测试，seed42；20轮；候选验证5次；主表测试3次。可改多节点工作流 | 数学实验也能改程序执行和集成，而非只写检查提示 |
| [GEPA，2507.19457v2](https://arxiv.org/html/2507.19457v2)，§3–4、E.1/E.4 | HotpotQA、HoVer、IFBench、PUPA、AIME等 | HotpotQA/HoVer：150 feedback-train /300 val /300 test；反思minibatch=3；验证前沿选择，按rollout预算比较 | 将反思反馈和候选选择分开；保留多个互补候选 |
| [HarnessForge，2606.01779v1](https://arxiv.org/html/2606.01779v1)，§4、B/C | ToolHop、RestBench-TMDB、API-Bank；HotpotQA/2Wiki | 3,800训练/演化题，其中ToolHop800；ToolHop测试195；三轮harness与LoRA对齐 | 更接近后续 model–harness 共演化目标 |
| [WHALE，2609.00196v1](https://arxiv.org/html/2609.00196v1)，§5–6、B | SearchQA、DAPO-Math/AIME、Lichess | 数学17,917权重训练/256 harness搜索/AIME60测试；Python环境，最多16 assistant turns；测试mean@8 | 纯harness和权重联合优化要分开；不能照抄联合优化曲线作为prompt-only预期 |
| [DGM，2505.22954v1](https://arxiv.org/html/2505.22954v1)，§4、B | SWE-bench Verified、Polyglot | 80轮，代码变异archive；分级10/60/200或10/50任务评估，候选可从历史分支继续 | 可借鉴测试驱动和分级评估，运行成本较高 |

上述轮数不是等价预算：每轮候选数、每题调用次数、验证重复次数均不同。需要同时画准确率—调用数/生成token/时间曲线。

## 逐篇阅读要点

### 第1篇｜Meta-Harness（Lee等，2026）

1. **问题/定位：** 自动优化固定模型外部的信息组织程序，属于已有任务上的harness搜索。
2. **方法：** Claude Code proposer读取历史源码、分数、完整执行记录，修改单文件Python harness；不是只有一个短prompt。分类部分20轮、每轮2候选。
3. **数据/验证：** 分类覆盖LawBench215类、S2D22类、USPTO180类；数学部分是检索增强推理，并非普通裸题；TerminalBench-2明确在同一89题上搜索和评估，属于discovery设置，不能解释为严格未见任务泛化。
4. **核心贡献：** 展示历史轨迹与可执行程序搜索的价值；分类和数学另有跨数据/模型验证。[原文§3–4](https://arxiv.org/html/2603.28052v1)

**当前官方参考代码**单独核查：S2D为200 train/50 val/212 test；LawBench 200/50/100；USPTO 50/30/100。这里的train用于记忆学习，不是模型权重训练。配置为offline、1 epoch、seed42；论文正文按在线分类任务介绍，不能把整理后的默认参数都当成原始实验参数。[config.yaml](https://github.com/stanford-iris-lab/meta-harness/blob/main/reference_examples/text_classification/config.yaml)

公开包直接包含此实验数据；显式test finalization后阻止同run继续演化。官方仓库当前提供分类、TerminalBench两个参考实验，并未把数学检索部分同等列为可运行reference。[分类README](https://github.com/stanford-iris-lab/meta-harness/blob/main/reference_examples/text_classification/README.md)、[根README](https://github.com/stanford-iris-lab/meta-harness/blob/main/README.md)

### 第2篇｜AFlow（Zhang等，2025）

1. **问题/定位：** 自动构造多节点agent workflow，任务本身来自已有benchmark。
2. **方法：** MCTS式搜索代码表示的流程；可使用Generate、Review/Revise、Ensemble、Test、Programmer等操作。
3. **数据/验证：** QA各抽1,000题；MATH取四种题型level5共617题；GSM8K/HumanEval/MBPP按论文描述用完整数据。20/80切分后，还用初始流程的5次运行选择有区分度的搜索题。候选验证5次、测试3次。
4. **核心贡献：** 优化执行结构与成本前沿，而不只改指令。[原文§4–5](https://arxiv.org/html/2410.10762v3)

实现差异：当前`run.py --validation_rounds`默认1，而`Optimizer`类默认5、论文也写5。对齐论文需要显式设置。数学启用Programmer；代码任务启用Test。MBPP取第一条测试作公开反馈，不能把最终隐藏测试交给Task。[run.py](https://github.com/FoundationAgents/AFlow/blob/main/run.py)、[optimizer.py](https://github.com/FoundationAgents/AFlow/blob/main/scripts/optimizer.py)

### 第3篇｜GEPA（Agrawal等，2025/2026 v2）

1. **问题/定位：** 在固定多模块语言程序上优化提示，与自由代码演化范围不同。
2. **方法：** 在feedback minibatch上反思，改进后才进入更大验证集；按逐题Pareto前沿采样父代。
3. **数据/验证：** HotpotQA/HoVer用150/300/300；IFBench用150/300/294，采用回答后重写的两阶段结构。公开模型Task为Qwen3-8B；反思batch=3，比较按rollout预算匹配。
4. **核心贡献：** 分离反思数据与选择数据，并展示模块级反馈的效率。[原文§3、附录E](https://arxiv.org/html/2507.19457v2)

HotpotQA是多跳检索程序；改为只提供候选段落的distractor任务会改变原设置，应另标协议，不能沿用“原样复现”。

### 第4篇｜HarnessForge（Chen等，2026）

1. **问题/定位：** harness变更后使小模型适应规划、动作和记忆接口。
2. **方法：** 失败归因、历史引导修改与筛选，再训练harness-specific LoRA，三轮成对演化。
3. **数据/验证：** 3,800训练/演化题由EnvScaler-RL 2,000、ToolHop800、离线QA1,000组成；ToolHop held-out195。训练/演化与测试按ID及归一化题目去重。
4. **核心贡献：** 将外部执行结构和内部策略适应连接；Task为Qwen3-4B/8B，适合作为后续小模型共演化机制参考。[原文§3–4、附录B/C](https://arxiv.org/html/2606.01779v1)

ToolHop原始benchmark共995题、3,912个本地可执行工具。800/195是HarnessForge所用分法，不能说成ToolHop原作者提供的官方train/test划分；工具族是否不重叠需另行检验。[ToolHop论文](https://arxiv.org/html/2501.02506v2)、[官方数据与评测入口](https://huggingface.co/datasets/bytedance-research/ToolHop)

### 第5篇｜WHALE（Kim等，2026）

1. **问题/定位：** 已有工具任务中的harness–weight交替优化。
2. **方法：** 成功轨迹RSFT与完整harness搜索交替；每次H搜索3候选，保存archive。
3. **数据/验证：** 数学DAPO17,917/搜索256/AIME60；可用Python和执行错误反馈。H-only数学60轮；测试每题8次取mean@8，非pass@8。正文指出数学H-only收益有限，小步权重训练后搜索才有效。
4. **核心贡献：** 研究冻结侧瓶颈与交替节奏，不能把其联合训练结果直接当作prompt-only的预期。[原文§5–6、附录B](https://arxiv.org/html/2609.00196v1)

官方代码说明SearchQA与Math的harness训练题从weight训练池采样，三个数据角色并非都互斥。算法返回按test表现最优组合，与本项目冻结后最终评估的承诺有差异，应单列协议。SearchQA部分训练中间资产未发布，当前不适合做首个可复现入口。[数据说明](https://github.com/krafton-ai/WHALE/blob/main/docs/data.md)、[复现说明](https://github.com/krafton-ai/WHALE/blob/main/docs/reproducing.md)

### 第6篇｜Darwin Gödel Machine（Zhang等，2025）

1. **问题/定位：** 固定模型的coding agent自动改自身代码。
2. **方法：** 代码变异、实测验证、历史archive分支搜索，运行80轮。
3. **数据/验证：** SWE先做10题功能检查，再到60题；超过阈值且位列前列才扩大到200题。Polyglot先10后50，最终还评全集。扩大的200题包含先前60题，不应把全部 headline 提升当成完全未见测试。
4. **核心贡献：** 将开放式探索与经验检验连接；工程成本和长时执行要求高。[原文§4、附录B](https://arxiv.org/html/2505.22954v1)

### 第7篇｜ADAS（Hu、Lu、Clune，ICLR 2025）

1. **问题/定位：** 用Meta Agent编写agent系统，而非提出新任务。
2. **方法：** 根据archive生成程序，经自反思、执行和评估继续搜索。
3. **数据/验证：** ARC网格题取20搜索/60测试，各评5次，25轮；DROP/MGSM/MMLU一般128搜索/800测试、30轮；GPQA为32/166、评5次。ARC指网格变换任务。
4. **核心贡献：** 说明可搜索程序组合，并研究跨模型/任务迁移。[原文§4、附录D/E](https://arxiv.org/html/2408.08435v2)、[官方仓库](https://github.com/ShengranHu/ADAS)

### 第8篇｜SICA（Robeyns等，2025）

1. **问题/定位：** 研究agent改自己的实现，执行与改进角色合一。
2. **方法：** 自修改代码、工具及子agent，依据性能、成本和速度反馈继续迭代。
3. **数据/验证：** 主实验含固定50题SWE-bench Verified、50题LiveCodeBench及合成编辑/导航任务；另测AIME2024、GPQA Diamond。§4.1明确展示推理任务上增益有限的饱和现象。
4. **核心贡献：** 长时agent任务上的工程改进证据，同时提示scaffolding增益依赖基础模型与任务，不保证数学必涨。[原文§4–5](https://arxiv.org/html/2504.15228v1)

### 第9篇｜RRSI（2026预印本）

1. **问题/定位：** 演化集涨分却不迁移的适应性过拟合问题，与当前诊断直接相关。
2. **方法：** 限制修改幅度、筛除benchmark特化、估计基线噪声带，并约束成本增长。
3. **数据/验证：** 编码、工作空间、工程设计八个benchmark；如Harvey LAB固定120 evolve/40 ID held-out，并另设OOD任务。各方法统一初始harness、冻结策略与候选预算。
4. **核心贡献：** 将稳定选择、编辑复杂度和OOD效果联系起来，适合作为后续evaluator/泛化课题的近邻和选择器参考。[原文§3–4、附录A](https://arxiv.org/html/2609.24972v1)

## 建议落地的首个正式任务

**主参考为Meta-Harness当前公开记忆分类实验；小模型和预算缩减是本项目改动，不宣称复现其主表。** 初始设计文件：[meta_harness_s2d_pilot.json](../../configs/design/meta_harness_s2d_pilot.json)。后续运行器已独立实现，实际配置和去重后的协议见[记忆演化实验](../memory-evolution.md)；下面保留当时的设计建议。旧design-only文件不作为可执行配置。

- 数据按官方参考配置使用S2D 200/50/212，固定版本、ID、标签集合及seed42。所有候选只从200条train构建记忆，test标签不进入记忆或proposer。
- proposer反馈预设为train内32题leave-one-out诊断，检索时去掉当前题自身；val用于候选选择并只回传汇总指标。这个更严格的观察边界是我们的新增设置，不能写成论文原参数。
- Meta沿用已验证的DeepSeek + Claude Code入口；Task先用现有Qwen3-4B-Instruct-2507，权重固定。calibration用训练侧样本，校验非零能力、格式和上下文预算。
- 初始对照包括no-memory、固定few-shot、BM25检索和固定两次调用的验证策略；从固定强基线出发，避免用人为削弱的父代制造增长。
- 搜索前只用训练/验证侧校准难度；若强基线已接近满分，或各基线均接近随机猜测且缺少可修复错误，则转向LawBench或ToolHop并重新锁定协议，不削弱父代，也不依据test分数换任务。
- 演化范围是记忆筛选/去重/组织、检索、上下文构造及是否验证草稿；固定标签输出合同、评分器、数据与每题调用/token上限。
- 首个pilot为5轮×2候选，候选评估重复3次；之后才对齐20轮×2候选。重复次数是借鉴AFlow的本项目设置，不能写成Meta-Harness原参数。
- 报告accuracy、输入/输出token、调用数、耗时；保留Static与同预算随机提案对照。搜索结束冻结选择，再补评历史选中版本的测试曲线；测试成绩不用于继续选策略。
- 先独立实现`memory/context builder → Task predict → evaluator`；下一小阶段才开放受限代码编辑。benchmark数据和评分可复用，RSI循环不继承论文实现。

**第二个任务接ToolHop。** 可参考HarnessForge的800/195，再在800内部预先划分反馈/选择；该内部划分是我们新增的协议。固定工具集合和隔离执行环境，允许改变调用编排、错误反馈和上下文策略，使用官方答案/路径指标。它更适合之后做harness–LoRA成对适应。

数学路线如保留，先实现Python工具与标准verifier，再参考AFlow/WHALE设置；代码路线可先做MBPP/HumanEval的测试反馈，SWE/TerminalBench放在基础runner稳定之后。

## 与当前增长试验的关系

当前八轮已有dev 35→37→40/48，但独立曲线25→19/32，final40→41/64且区间跨0；两次同题复测增益也较小。其结果与脚手架饱和、开发集适应及噪声风险相关，不能靠增加轮数保证解决。完整结果见[增长诊断](../growth-diagnosis.md)。

优先级调整为：先选有实际harness发挥空间的文献任务，完善数据角色和重复选择协议，再接权重训练。论文自己的discovery、held-out、OOD和best-test记录口径也要分开；不统一包装为独立泛化增长。
