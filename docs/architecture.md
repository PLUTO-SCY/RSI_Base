# 架构方案

状态：拟实施的设计，尚无运行时实现。约束：单机四卡、独立实现、可替换实验组件。

## 1. 实验对象与角色

一次可评估的 agent 由模型、harness 和外部资源快照共同定义：

**AgentSnapshot = (ModelRef, HarnessRef, ResourceRefs, RuntimeSpec)**。

数据生成、数据配比、训练参数和改进策略属于产生新快照的过程；它们记录在谱系和 ImprovementPlan 中，不偷偷塞进模型名。一次实验另外固定 TaskManifest、EvalProtocol、Budget 和 Recipe。

- **Task Agent**：在指定快照下执行任务，产出原始调用记录、环境交互和最终输出。
- **Meta Agent**：根据允许访问的证据提出改进计划；计划可以包含代码变更、训练作业、数据生成和配比调整。
- **Host Orchestrator**：校验计划、执行算子、管理预算和恢复，实施评分与权限边界。它是固定实验基础设施。
- **Evaluator**：给轨迹或最终产物评分。训练奖励、搜索信号与最终指标通过各自接口接入。

“两个 agent”不等于两套大模型。首版可用独立 Meta 模型，Task 模型用本地小模型；也支持让同一个服务承担两个角色，使用独立上下文与工具集合。

### 递归范围必须写明

| 模式 | 谁生成下一轮改进？ | 结论边界 |
| --- | --- | --- |
| external_meta | 固定外部 Meta 模型 | 验证 task model 与 harness 的联合自适应 |
| self_meta | 更新后的 Task 模型成为下一轮 proposer | 可研究任务学习是否改善后续改进能力 |
| learned_meta（后续） | Meta 自身也通过独立训练作业更新 | 另需改进任务数据、独立验证与预算 |

首版 external_meta 足以做基础设施验证。self_meta 在同一协议上实现，不能因名称里有 RSI 就声称已经证明改进器能力递增。

## 2. 分层与依赖方向

~~~mermaid
flowchart TB
    CLI["CLI / 有类型配置"] --> Engine["核心：状态机、预算、版本谱系"]
    Recipe["Recipe：交替顺序、父代选择、停止规则"] --> Engine
    Engine --> Improve["改进算子：harness / skill / data / model"]
    Improve --> Ports["协议：Meta / TaskRunner / Learner / Benchmark / Evaluator"]
    Ports --> Adapters["适配器：本地推理、SFT、GRPO、外部评测"]
    Adapters --> Official["官方引擎与任务环境"]
    Engine --> Store["ArtifactStore + 事件日志 + 索引"]
    Adapters --> Store
~~~

依赖单向：基础领域对象和协议不认识外部引擎；适配器实现协议。引擎通过注册表获得实例，不按任务名写 if/else。

不把自研 RSI 核心建立在 WHALE、CoEvo-RSI 或其他论文仓库上。引用其方法时，落在独立 recipe 的文档与配置上。官方训练、推理库作为包或隔离作业使用。

## 3. 最小稳定对象

| 对象 | 必要内容 | 解决的问题 |
| --- | --- | --- |
| ModelRef | 基础权重 revision/digest、adapter digest、tokenizer、chat template、生成/解析配置 | 避免“同名模型”指向不同策略 |
| HarnessRef | bundle digest、父版本、接口版本、入口、资源依赖、变更集 | harness 可重放、可比较、可分支 |
| AgentSnapshot | model/harness/resource/runtime 的固定引用 | 选择和部署必须绑定完整组合 |
| TaskManifest | 来源/revision、样本 ID、group ID、划分、工具/语料/环境版本 | 数据去重、划分和环境重建 |
| Trajectory | 任务与快照、事件、实际请求/响应、终止状态、用量 | 不丢训练与诊断信号 |
| ImprovementPlan | 目标、父代、算子 DAG、产物契约、预算、理由和证据引用 | Meta 输出可检查、可执行 |
| TrainingJobSpec | 初始模型、数据 manifest、目标、mask 规则、优化参数、预算、引擎 | 训练与搜索流程解耦 |
| EvaluationRecord | 协议、数据、评分器、逐样本结果、缺失/错误、成本 | 比较只发生在兼容的评测条件下 |
| CandidateRecord | 计划、谱系、状态、快照、所有评估引用 | 失败候选也有完整记录 |
| BudgetLedger | 预留/结算、调用与作业 ID、资源维度、重试 | 限制实际总成本 |
| RunManifest | 源码 commit、配置、依赖、硬件、随机源、角色关系、锁定策略 | 重放实验设置 |
| ResourceRef | skill bank、检索语料/索引、工具 schema、环境镜像 digest | 外部资源变化不被遗漏 |

这些是数据契约，不要求一开始做十二个大型管理器。首版可用少量 dataclass/Pydantic schema、文件 artifact 和 SQLite 索引实现。

## 4. 关键协议

以下为签名示意；具体类型在 P0 用两个任务的实际需求定稿。

~~~python
class MetaAgent(Protocol):
    async def propose(self, evidence, editable_state, limits) -> ImprovementPlan: ...

class TaskRunner(Protocol):
    async def run(self, snapshot, task, context) -> TrajectoryRef: ...

class BenchmarkAdapter(Protocol):
    def capabilities(self) -> BenchmarkCapabilities: ...
    def load_manifest(self, split) -> TaskManifest: ...
    async def create_episode(self, task, limits) -> EpisodeHandle: ...

class Learner(Protocol):
    def capabilities(self) -> LearnerCapabilities: ...
    async def train(self, job, context) -> TrainingResult: ...

class Evaluator(Protocol):
    async def score(self, prediction_or_artifact, task_ref, protocol) -> EvaluationRecord: ...

class ImprovementOperator(Protocol):
    async def execute(self, operation, context) -> OperatorResult: ...

class Selector(Protocol):
    def choose(self, records, comparison_protocol) -> SelectionDecision: ...
~~~

TaskRunner 不拿 reference answer。Evaluator 从独立的任务答案存储读取 target。训练端和最终评分端可以采用相同的评分逻辑，但不能共享最终测试样本的可见性。

## 5. 一个清楚的默认循环

固定轮次只是默认 Recipe，不是写死在核心里的唯一方式。

~~~text
初始化父代快照、划分清单和总预算
记录基线；用训练侧轨迹形成诊断证据
对每轮：
    Meta 根据证据提出最多 K 个改进计划
    Host 校验改动范围、能力兼容与剩余预算
    独立生成 harness 候选；执行契约检查和小规模试跑
    在 D_search 上比较候选，保留少量 shortlist
    在选中的新 harness 下，使用 D_weight 重新收集轨迹
    通过 TrainingVerifier 的成功轨迹构建 SFT 数据
    数据不足：记录 NO_UPDATE；按配置结束或保留 harness 候选
    训练新的 adapter，验证导出物并重新加载推理服务
    在 D_select 上比较完整的新 model–harness 组合
    记录接受、拒绝、无结论或故障，保留全部候选与实际成本
最终：冻结选择，单独运行 D_test / D_ood
~~~

默认用 H→W，先让模型适应新接口；W→H、H-only、W-only、H→W→H 都用 Recipe 配置表达。预算包含全部诊断、筛选、生成、训练、重载验证和最终评测。

**MVP 的计算节约假设**：只训练 shortlist 中的一个候选。它可能错过“当前较弱但训练后更强”的 harness，因此后续以少量双候选训练检查这一偏差；不能把预筛排名当成最终潜力的金标准。

## 6. 可演化范围

首版实现以下三个必要操作，再扩展其他算子：

| 算子 | 输入 → 输出 | 首版范围 |
| --- | --- | --- |
| EditHarness | 父 bundle + patch → 新 bundle | prompt、上下文策略、重试、步骤分配、受约束控制流 |
| CurateData / SetDataMix | 原始轨迹 + 标签 + 配比 → 数据 manifest | 当前成功轨迹 + 兼容历史回放；按 token 计配比 |
| TrainModel | ModelRef + TrainingJobSpec → 新 ModelRef | LoRA SFT；RSFT 由收集、验证、筛选和 SFT 组合 |
| EvolveSkills | 轨迹 → skill bank 新版本 | 后续，作为独立 ResourceRef |
| GenerateTasks | 任务生成计划 → 带来源的数据集 | 后续，须单独验证生成任务/标签 |
| InheritModel | 历史 ModelRef → 初始化选择 | 后续；默认选 checkpoint，权重合并另行实现 |
| TuneTraining | 类型化训练参数变更 → 作业 | 后续，只允许已支持的范围 |
| TrainMeta | 改进任务数据 → Meta ModelRef | 研究扩展，不作为 v0.1 依赖 |

数据混合必须区分“训练哪些轨迹”和“用哪些任务收集轨迹”。同名配比在两个层面会产生不同干预，配置应分别命名。

## 7. Harness 的边界与隔离

Harness 是可执行 Python bundle，至少包含 manifest、入口、prompt/配置和 smoke 示例。先提供一个简单 ReAct 实现，再允许在明确入口下编辑整个控制流。

可改：模型可见的上下文、工具说明的表述、有效 schema 内的路由、动作解析、错误反馈、重试、记忆和终止建议。

宿主固定：真实工具权限与语义、环境状态转移、最大资源上限、任务划分、最终 verifier、评分记录和预算。Harness 的 finish 只是提交请求；是否成功由外部 evaluator 决定。

仅做路径白名单不足以隔离任意 Python。可执行候选使用独立进程/容器，给予候选目录写权限、只读任务输入、受限工具代理和必要的模型端点访问。最终答案、隐藏测试、宿主文件系统和容器控制接口不进入候选空间。CPU 调试模式只执行仓库中受信任的固定 harness。

候选通过静态契约检查、导入/协议检查、工具调用 smoke、资源限制试跑后进入评分。这些检查验证可执行性，不证明任务能力或代码安全。

## 8. 训练轨迹是最容易出问题的接口

每一步同时保留两种表示：

1. 原始消息、tool_call_id、工具 schema、实际渲染后 prompt，以及 tokenizer/template 版本；
2. 可用时的 token IDs、生成 token mask、模型调用 logprobs、截断与上下文压缩事件。

SFT 只监督真实 assistant 生成的目标片段；不把 system、user 和工具返回值误当成策略目标。多次模型调用可以构成多个 decision sample，避免把经过裁剪的上下文强行还原成一条未发生过的长对话。

GRPO 还需要行为策略版本、采样参数、每次调用的真实前缀和 old logprobs。不能仅凭文本轨迹事后重新 tokenization 就声称是原来的 on-policy batch。核心采集协议为这些字段留位，只有适配器通过一致性检查后才声明支持 RL。

默认 GRPO 分组键为 task ID + harness digest + behavior policy digest + rollout protocol digest。同一组不混用不同 harness 或中途切换的策略。新 harness 只在 batch 边界生效。工具观察不参与 policy loss。

## 9. 模型加载与回放规则

ModelRef 明确区分 base+adapter、merged checkpoint 和 full checkpoint。默认继续训练父 adapter 的复制品并保存完整 adapter 快照，不悄悄把多轮 adapter 堆叠。

导出后做可加载检查、tokenizer/template 一致性检查和真实小样本推理；服务完成重载并报告预期 digest 后才能处理下一阶段请求。先重启并重新加载，动态 adapter 热切换是后续性能优化。

历史轨迹只有在工具 schema、观测格式和训练目标兼容时才能回放。默认优先使用当前 harness 的成功轨迹。跨 harness 旧轨迹要么在新接口下重新生成，要么通过显式迁移和重验证；不做字符串拼接后直接送训。

## 10. 状态、恢复与预算

建议状态机：

~~~text
PROPOSED → MATERIALIZED → CONTRACT_PASSED → SEARCH_EVALUATED
         → COLLECTED → CURATED → TRAINED → RELOADED → CONFIRM_EVALUATED
         → ACCEPTED | REJECTED | INCONCLUSIVE
任一步可进入 FAILED / BUDGET_EXHAUSTED / NO_UPDATE
~~~

- 每个阶段的输入哈希构成幂等键；完成标记必须在产物校验后原子写入。
- 目录存在不代表阶段成功；恢复校验 manifest、产物哈希和完成事件。
- 所有外部 job/request 使用稳定 ID，重试不会覆盖旧结果或漏算成本。
- 已产生 checkpoint 与“完整优化器状态可续训”分别记录；跨阶段默认从模型快照开启新训练作业。
- JSONL 追加事件保存原始事实；SQLite 用于检索和状态索引，损坏后可据事件与 manifest 重建。
- 训练/推理分时持有 GPU lease；释放、进程退出和设备显存回收检查完成后进入下一阶段。
- 调用前预留最大生成量；结算真实用量，超时或不可确认的调用保留保守估计。恢复继续使用原账本。
- 开始优化前预留冻结最终评估所需预算；候选搜索只能消费剩余部分。并发作业的预留也算已占用预算。
- 缓存键包含完整 ModelRef、HarnessRef、数据/环境/评分协议和随机设置；在共享 policy 更新后重新评估活跃 harness，不能以旧策略分数替代新策略分数。
- 终止任务、取消外部作业和回收环境属于状态机职责，不依赖研究脚本最后一行。

## 11. 计划代码结构

~~~text
src/rsi_base/
  core/               # 对象、协议、状态机、预算；无 GPU 框架依赖
  engine/             # 作业执行、阶段交接、恢复、选择与归档
  agents/             # Meta 提案接口、Task 调用协调
  operators/          # edit_harness / curate / collect / train
  harnesses/          # 固定 seed harness、bundle 契约与加载
  data/               # 轨迹转换、mask、去重、数据混合与兼容性
  evaluation/         # reward / selection / final 角色与结果模型
  benchmarks/         # 独立任务插件与外部评测桥接
  adapters/
    inference/        # 官方推理引擎
    training/         # 官方 SFT / RL 引擎
    execution/        # 作业与容器执行
  recipes/            # 方法组合；不复制核心循环
  storage/            # artifact、事件日志与索引
  cli.py
configs/               # 有类型配置及少量可运行 recipe
experiments/           # 固定版本实验清单；不放大结果文件
tests/                 # 契约、故障恢复、CPU 集成
scripts/               # 薄启动与诊断入口
~~~

目录随真实功能出现，不预先堆空文件。可扩展性的验收是接入第二个不同类型的任务、更换一个 evaluator 和一个 learner 时无需改核心状态机。
