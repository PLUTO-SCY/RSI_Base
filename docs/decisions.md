# 架构与技术选型决策

状态：2026-11-16 的实施建议，已按增量路线修订。未经集成验证的组合都属于待验证选型；依赖在引入对应功能的阶段锁定并记录。

## D01｜RSI 编排从零独立实现

依据：用户已尝试 WHALE，明确要求参考方案但不在论文代码上搭建。

我们实现领域对象、角色接口、演化状态机、算子组合、候选/谱系、预算和实验记录。论文中的交替搜索、历史继承、技能更新等分别写成独立 recipe。

不 fork、vendor 或以论文仓库作父工程。允许使用官方引擎包，但其内部类型不泄漏到 core。只有确有必要且可维护的小型补丁才进入独立 adapter patch，并记录上游问题和移除条件。

## D02｜先固定模型的改进循环，再 RSFT + LoRA SFT，后 GRPO

依据：本项目先验证通用闭环；用户已确认单机四卡，尚未确认可用显存和实际吞吐。

v0.0/v0.1 先固定 Task 模型，完成单任务执行、评分、Meta 配置提案和三轮串行选择。v0.2 加入一种 LoRA SFT 后端；不将训练引擎接入作为第一个可运行版本的前提。

RSFT 拆成四个通用步骤：collect → verify → curate → SFT。SFT learner 只接收训练作业，不承担整个演化循环。这样可把训练替换为 GRPO，而不修改 Meta 提案或 benchmark 接口。

第一候选后端是官方 [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) + PEFT，使用 assistant 目标 mask；LoRA 支持以底层模型与版本矩阵实测为准。需要人工构造一条多轮工具轨迹核对 loss token，不能仅依赖库默认值。

后续 GRPO 首选官方 [verl](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)。在单次 GPU spike 中证明 rollout prefix、old logprobs、mask 与行为版本一致，才声明适配成功。若无法通过已有 agent loop 扩展点接入，再比较 Agent Lightning 或 SkyRL；不同时维护三个 RL 后端。

**边界**：RSFT 与 GRPO 不等价；使用 RSFT 不代表复现了某论文的 GRPO 结果。成功轨迹覆盖不足时要记录失败，不能期待 SFT 自动产生新能力。

## D03｜本地推理先 vLLM，精确版本后锁定

采用官方 [vLLM](https://docs.vllm.ai/en/latest/features/lora/) 作为第一本地服务后端。model ID、base/adapter、tokenizer、chat template、工具解析器和采样设置都进入 ModelRef/RuntimeSpec。

v0.0/v0.1 可直接使用已配置的模型端点。v0.2 引入本地训练时，先用完整服务重载验证阶段交接，再优化热切 adapter。若特定模型无法在锁定版本中正确加载，则选较小/成熟支持的模型，或另接官方推理后端；不将特殊处理写进主循环。

标准 HTTP 调用足够做 SFT 数据采集，但不自动满足 RL 对 token IDs/logprobs 的要求。

## D04｜接入本地训练后，四卡分时使用

生成阶段可使用四个小模型副本，或在显存不足时用模型并行；训练阶段暂停生成并释放服务，然后由四卡训练作业独占。

是否使用 DDP、FSDP 或 ZeRO 由模型/显存实测决定。默认从 dense 小模型与 LoRA 开始。四卡数据并行不是把四张卡的显存拼成一张卡。

这牺牲部分流水线吞吐，换取首版更容易验证的 checkpoint 边界和资源占用；异步训练在基线稳定后再做。

## D05｜配置与环境分离

- 框架目标 Python 3.11；core 不依赖 CUDA。
- 配置采用 Pydantic schema + YAML/JSON 输入，拒绝未知字段并输出 resolved config。
- 编排/推理/SFT/RL/部分任务环境可以使用不同虚拟环境或镜像，以 manifest/JSONL 交换数据。
- 不预先承诺所有引擎能装进同一个环境；每个被验证的组合有自己的 lock 或镜像 digest。
- 现有 JSON 是后续设计示例。v0.0/v0.1 只校验实际需要的少量字段，后续逐步扩展 schema 与 CLI。

当前执行工具中的默认 Python 为 3.8，nvidia-smi 不能与驱动通信。这只说明本轮工具环境尚不能验证宿主 GPU 栈，不否认用户说明的四张实体 GPU。

## D06｜本地 artifact 与事件记录优先

v0.0/v0.1 用文件系统保存配置、候选、逐题结果与决策，用 JSONL 记录调用和用量。SQLite、通用 ArtifactStore 和外部 exporter 按实际规模与查询需求后置。

数据/权重/轨迹不进 Git；Git 保存代码、实验定义、数据与产物的 manifest、验证摘要。异步后台服务不是首版依赖。

## D07｜评分角色分离，允许 evaluator 研究

TrainingVerifier 决定训练样本/奖励；SearchEvaluator 决定候选排名；FinalEvaluator 固定用于最终测试。三者可以共享算法，但其数据权限、版本和调用记录分离。

evaluator 课题可替换前两者、训练 proxy judge，甚至演化评分策略；最终 gold/官方指标保持固定，从而量化偏差、误选和 reward hacking。若研究对象就是 gold verifier，则另外冻结更高层评价集，不能同时修改自己和自己的最终考题。

## D08｜先一个任务跑通，再用第二任务整理接口

v0.0/v0.1 从 GSM8K 的单次生成与评分起步，v0.2 在同一任务接训练。v0.3 再接 ALFWorld 和固定交互 loop，用真实差异检验接口；计算工具、ToolHop/BFCL 和外部 bridge 逐项加入。

增加一个 benchmark 必须明确其支持模式：最终评测、在线执行，还是训练闭环。不能把一个通用评测器的任务列表直接作为本项目的支持列表。

## D09｜通用扩展点通过替换实验验收

第一目标是少量清楚的协议，而不是任意动态插件系统。注册表先显式注册，后续确有外部包需要时再加入 entry point discovery。

验收包括：

1. 添加第二个任务，不修改核心演化状态机。
2. 将 SearchEvaluator 换为 proxy judge，任务 runner 不改。
3. 将 LoRA SFT learner 换为已验证的 GRPO learner，snapshot/trajectory 格式不分叉。
4. 从 external_meta 切到 self_meta，只改变角色解析与 recipe，原始证据格式不变。

## D10｜方法研究与基础设施分开发布结论

基线性能不提升时仍可证明基础设施交付完成；要主张某组件有效，则必须用固定预算对照、独立最终测试和统计误差给出证据。

低资源 recipe 命名为项目自己的实验定义，清楚列出与原论文的模型、数据、phase、奖励和预算差异。保留方法引用，不沿用“原样复现”的承诺。

## D11｜每阶段只增加一个主要变化来源

先固定权重验证 harness/config 改进，再加训练，再加第二任务，之后每次只改变一个研究组件。完整架构保持可演化方向，工程抽象由已出现的复用需求驱动。

第一阶段后置通用算子 DAG、插件注册、数据库和复杂调度；数据隔离、固定评分、候选版本与调用上限从一开始保留。版本范围以 roadmap 为准。

## D12｜Meta 首选 Claude Code，Task 使用自研薄 harness

状态：当前推荐的起步方案，尚未实现或执行真实模型调用。用户提出 Claude Code 作为 Meta、Task 使用简单 agent 架构，与增量路线一致。

### 模型与 harness 分开选择

| 层次 | Meta Agent | Task Agent |
| --- | --- | --- |
| 模型 | 现有可用、擅长代码/轨迹分析的强模型，固定具体 ID/版本 | 首个候选为本地可训练的 Qwen3-4B，按实测支持与任务表现确定 |
| 执行 harness | Claude Code CLI，封装为一个轻量 proposer 适配函数 | 自己实现少量上下文构造、模型调用和终止逻辑 |
| v0.1 是否训练 | 固定模型与配置 | 固定权重，只改受限 prompt/config |
| v0.2 是否训练 | 继续固定 | LoRA SFT，并绑定训练后的 harness 版本 |

Claude Code 的名字不代表底层实际模型 ID；实验记录同时保存 CLI 版本、实际模型/服务标识、提案指令、设置、可用工具和用量。具体强模型依据已有接入与预算决定，不把动态 model alias 当成固定实验条件。

### Meta 先做受约束提案

Python 外循环负责执行、评分、选择、训练和归档。每轮新开一个 Meta 会话，宿主显式提供当前配置、开发侧成功/失败样例、指标、成本和必要历史摘要；Meta 返回一个候选及修改理由。

v0.1 初期采用 proposal-only 方式：关闭内建工具与额外 MCP，宿主将允许的证据作为输入传入。输出只包含可编辑字段，宿主校验后物化为新候选。这里不需要 Meta 修改实验仓库或自行运行评分。

未来扩大到代码级 harness 编辑时，再开放独立候选工作区的文件能力，并实施执行隔离、协议 smoke 和修改范围校验。CLI 的当前目录与工具自动批准列表不能单独充当文件系统隔离。

在只改短 prompt 的阶段，直接模型 API 也是可替换实现；初期先保持一个 Claude Code 后端，不同时维护多个 Meta 框架。

### Task 先做最小执行，再逐步加入交互

v0.0/v0.1：task + system prompt → 一次模型调用 → 原始回答与调用记录。答案提取和正确性判断归固定 evaluator。

交互任务接入后：构造上下文 → 模型输出工具调用或最终答案 → 执行允许工具 → 回传观察 → 继续或结束。工具集合、最大步数、token/时间上限由宿主固定；prompt、观察组织和受限策略逐步开放为可编辑部分。

模型服务与 Task harness 分离：后续将基础权重替换为 LoRA checkpoint，不重写整个 Task Agent。先选择可遵守输出格式、可训练、又有任务改进空间的模型；先测合理 seed harness 的基线，再决定是否需要调整任务难度。

### 接入验证与来源

2026-11-16 本机只读检查发现 Claude Code 2.1.169 已安装；help 包含 print、JSON 输出、JSON schema、model、tools 和预算相关参数。未执行模型 prompt，尚未验证登录、真实调用或其成本记录。

官方文档说明 CLI 支持非交互运行和结构化输出：[程序化运行](https://code.claude.com/docs/en/headless)、[CLI 参数](https://code.claude.com/docs/en/cli-reference)。工具权限与执行隔离分别按[权限文档](https://code.claude.com/docs/en/permissions)和宿主能力实现。

接入时以锁定的本机版本为准，验证退出码、结果中的错误标记、结构化提案、超时与实际用量；CLI 正常退出不能单独证明提案有效。记录实际加载的项目/用户设置，不让会话历史、自动发现的扩展或版本升级成为未记录的变量。
