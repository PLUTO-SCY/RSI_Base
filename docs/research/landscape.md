# 文献与基础设施调研

检索日期：2026-11-16。目标是为 evaluator、效率和泛化课题提供共同的实验 substrate。用户明确要求从零独立实现，以下工作用于机制对照，不作为本仓库的代码基座。

## 1. 先回答核心判断

从复现论文建立研究起点是合理路径，但“复现其算法”“复现其公开分数”和“继承其工程实现”是三个不同的交付。本项目先独立实现可比较的算法机制，并为结果复现保存协议差异；软件结构不继承某一篇论文。

最接近的机制并不少：WHALE 的交替 RSFT、HarnessForge 的 harness-specific adapter、CoEvo-RSI 的历史继承、SkillRL 的技能与策略共演化，以及 CoHARL 的共享策略与多 harness 种群。因此，Meta/Task 两个角色或 model–harness 联合更新本身不适合作为“全新方法”的贡献声明。

**建议的基础设施价值**：让不同更新对象、调度和评估器在同一套版本、预算、数据边界与实验协议下组合和比较。这个价值仍需用第二个任务和组件替换实验来证明。

## 2. 覆盖范围与证据层级

本轮扫描 18 项原始文献，重点核对用户提供的 5 篇正文方法/实验与对应代码片段，并补查 CoHARL 的方法与评估约束、HELIX 的限制。未在本环境执行这些仓库的 GPU 复现。

- L3：已读方法、实验或限制的相关正文，必要时交叉核对代码；不表示逐行审计全仓库。
- L2：正文部分章节或官方实现/说明，尚不足以完整评价全部实验。
- L1：摘要/元数据级，用于定位路线，不据此判断实现完整性。

| 工作 | 更新对象与与本项目的关系 | 本轮层级 |
| --- | --- | --- |
| [Population Based Training，2017](https://arxiv.org/abs/1711.09846) | 同时优化模型与超参数调度；历史继承与预算调度的邻近机制 | L1 |
| [STaR，2022](https://arxiv.org/abs/2203.14465) | 生成、筛选、训练的自举循环；成功样本再训练的先例 | L1 |
| [ADAS，2024](https://arxiv.org/abs/2408.08435) | Meta Agent 写代码定义的新 agent，结合历史 archive 搜索 | L1 |
| [SICA，2025](https://arxiv.org/abs/2504.15228) | coding agent 编辑自身代码；固定权重的程序改进路线 | L1 |
| [Darwin Gödel Machine，2025](https://arxiv.org/abs/2505.22954) | agent 代码变异、经验验证、历史 archive | L1 |
| [SEAL，2025](https://arxiv.org/abs/2506.10943) | 自生成训练数据与更新指令，并学习适应策略 | L1 |
| [GEPA，2025](https://arxiv.org/abs/2507.19457) | 轨迹反思与 Pareto prompt 演化；低成本 harness/prompt 基线 | L1 |
| [SkillRL，2026](https://arxiv.org/html/2602.08234v2) | 技能库与策略 RL 联合演化 | L3 + 代码片段 |
| [Meta-Harness，2026](https://arxiv.org/abs/2603.28052) | 利用源码、历史分数和执行轨迹进行 harness code 搜索 | L1 |
| [HarnessForge，2026](https://arxiv.org/html/2606.01779v1) | harness 种群 + 针对 harness 的 LoRA 对齐 | L3 + 代码片段 |
| [EvoTrainer，2026](https://arxiv.org/abs/2606.03108) | 训练侧诊断/干预 harness 与 policy 的共同改进；需区分训练侧和任务侧 harness | L1 + 官方说明 |
| [HASE，2026](https://arxiv.org/abs/2607.03935) | 一个模型在多轮动作空间里求解任务或编辑 harness | L1 |
| [Co-Harness，2026](https://arxiv.org/abs/2607.22688) | harness 和权重联合优化的直接先例 | L1 |
| [HELIX，2026](https://arxiv.org/html/2608.13951v1) | source-traceable harness 组合和验证轨迹导出 | L2 + 限制核查 |
| [WHALE，2026](https://arxiv.org/html/2609.00196v1) | 小步交替 RSFT / harness 搜索，固定或自适应 phase 调度 | L3 + 代码片段 |
| [Self-HPO，2026](https://arxiv.org/html/2610.22791v1) | 更新后的 policy 充当下一轮 proposer，困难任务驱动搜索 | L3 + 代码片段 |
| [CoEvo-RSI，2026](https://arxiv.org/html/2610.25078v1) | harness 历史继承 + checkpoint 选择/合并 + SFT 适应 | L3 + 代码片段 |
| [CoHARL，2026](https://arxiv.org/html/2610.36469v1) | 多个可执行 harness 的 rollout 共同训练一个共享 policy | L3，未审计代码 |

表中按本轮实际读取的 arXiv/官方版本引用；不把未核实的会议口头报告、引用次数或 GitHub star 作为选型依据。用户五篇工作详见[逐篇核查](reproduction-audit.md)。

## 3. 对架构影响最大的补充证据

### CoHARL：核心不能假设“一模型只配一个活跃 harness”

其 §3 把共享 policy、训练 harness pool 和历史 Pareto archive 分开。相同 task–harness 的 rollout 在组内计算相对奖励，新 harness 在训练 step 边界加入；评估覆盖 SWE-rebench、SWE-bench Verified 和 Terminal-Bench 2.0。贡献是持续的共享策略种群训练，不能只用单活跃 harness 的循环表达。原文的隐藏测试访问检查也说明可编辑代码需要明确的 evaluator 边界。[方法与附录 B](https://arxiv.org/html/2610.36469v1)

**设计推论**：v0.1 可以只有一个活跃快照，但对象协议要允许多个 harness 引用同一个 ModelRef；pool 调度和不可变历史 archive 分开建模。不能复用跨策略时期的旧分数构造一个看似可比较的 Pareto 前沿。

### HELIX：轨迹导出不等于完成了权重共演化

HELIX 将 harness 拆为可追溯的 ports/atoms/recipes，保留 sibling successes、failures 和验证信息。原文 §9.5 明确表示该研究生成了模型更新数据，但没有训练更新后的模型；§9.1–9.3 也限制了测试范围和 portfolio coverage 的解释。[原文限制](https://arxiv.org/html/2608.13951v1#S9)

**设计推论**：需要分别验证“harness 能运行”“轨迹可训练”“模型已训练”“新快照回到下一轮”。Portfolio 的事后 any-success 不应报为单个部署 agent 的 pass@1。

### EvoTrainer：训练侧 harness 与 Task harness 应是两类干预

其官方说明把诊断、干预回测和经验积累纳入训练流程，这与任务执行时的 prompt/tool/控制流不同。[官方项目说明](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/main/EvoTrainer/README.md)

**设计推论**：计划使用 SetDataMix、TuneTraining 等类型化算子，而不是让所有修改都伪装成 EditHarness。首版不允许任意重写训练引擎。

## 4. 官方工具的复用边界

下列是引擎/工具能力证据，不代表已在本项目集成成功。

| 工具 | 可复用能力与主来源 | 本项目定位 |
| --- | --- | --- |
| [TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) + PEFT | SFT、对话数据与 adapter 训练 | 首选 LoRA SFT 后端；验证 assistant/tool masks |
| [vLLM](https://docs.vllm.ai/en/latest/features/lora/) | 本地推理服务与受支持模型的 LoRA serving | 第一推理适配器；先采用进程重载 |
| [verl 多轮 rollout](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html) | 工具交互、多轮训练流程 | 后续 GRPO 后端，锁定官方版本 |
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | API Gateway、真实 agent 轨迹采集、训练解耦 | 后续黑盒 harness/RL 桥接备选，不作为 RSI 核心 |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | 自定义 Generator、环境与训练接口 | 后续替换 learner 的备选 |
| [Inspect](https://inspect.aisi.org.uk/tasks.html) | 可替换 solver、scorer 和工具的任务组织 | 多类外部评测桥接 |
| [Harbor](https://docs.harborframework.com/) | agent、任务与沙箱的执行体系 | 后续终端/软件工程任务的执行桥接 |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 语言模型的生成/似然评估 | 通用能力与遗忘测试；默认不当作任意 agent runtime |

具体选型与替代触发条件见[架构决策](../decisions.md)。不把支持某个模型端点误写成支持任意 harness 的训练和评估。

还查看了 [GaryTang32/RSI](https://github.com/GaryTang32/RSI) 与 [HELIX 官方仓库](https://github.com/HKUDS/HELIX) 的 README，确认已有统一改进循环/可组合 harness 的相关工程探索。本轮未验证其全部实现与对照结果；因此不作“市面没有通用框架”的结论。

## 5. 检索轨迹与停止边界

实际使用的检索路径包括：

- 问题：model harness co evolution recursive self improvement agent。
- 用户指定最近邻：五个 arXiv 标识与官方 GitHub 仓库，逐个读取。
- 同构机制：self-improving coding agent、ADAS、SEAL、GEPA、STaR、PBT。
- 逆向核查：model harness co-evolution reproducibility、harness reward hacking、WHALE bugs；低相关检索结果不作为事实来源。
- 引用扩展：从 CoHARL/CoEvo-RSI 的 related work 发现并定位 EvoTrainer、Co-Harness、HELIX。
- 工程：官方 SFT、LoRA serving、multi-turn rollout、benchmark runner 与环境文档。

本轮足以支持“独立的小核心 + 适配器 + 机制 recipe”的工程选型；不构成全领域系统综述，也没有证明某种方法在本地四卡机器上最优。HASE、EvoTrainer、Co-Harness 等仅作定位，实施相应 recipe 前再读其方法、实验和官方配置。即使后续出现更近工作，独立实现与可替换协议的选择仍可保留。
