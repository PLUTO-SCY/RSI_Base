# RSI Base

面向 evaluator、效率和泛化研究的递归自我改进（RSI）实验基础框架。

**当前状态：调研与设计阶段。尚未实现演化运行时、真实模型训练或 benchmark 适配器。** 本仓库中的配置是接口设计示例，不能用于启动训练。完成情况见[进度记录](docs/progress.md)。

## 已确认的边界

- 从零独立实现 RSI 编排、Meta/Task Agent 接口、候选管理和实验记录；论文实现只用于理解与核对机制。
- 单机四卡优先，首个模型以 Qwen3-4B 或更小的同系列模型为候选；显存、驱动和实际吞吐经实测后确定。
- 官方维护的训练、推理、评测引擎通过适配层使用；不复制论文仓库中的训练框架或任务专用控制逻辑。
- 首版实现成功轨迹筛选 + LoRA SFT + harness 搜索的交替闭环，随后扩展 GRPO、技能演化、数据配比和共享策略种群。
- 不把分数提升设为软件交付的必要条件：零增益、退化和失败也是需要完整保留的实验结果。

## 建议先读

| 文档 | 回答的问题 |
| --- | --- |
| [调研地图](docs/research/landscape.md) | 相关工作已经做了什么，哪些部分值得参考？ |
| [重点论文与代码核查](docs/research/reproduction-audit.md) | 用户列出的五篇工作提供了哪些真实资产，复现边界是什么？ |
| [架构方案](docs/architecture.md) | Meta Agent、Task Agent、训练、harness、数据如何解耦？ |
| [技术选型](docs/decisions.md) | 哪些自己写，哪些通过官方引擎复用？ |
| [实验协议与四卡预算](docs/experiment-protocol.md) | 如何防止评估泄漏、做归因、控制成本和验证泛化？ |
| [Benchmark 接入计划](docs/benchmarks.md) | 各任务支持到什么程度，先做哪几个？ |
| [开发计划与验收](docs/roadmap.md) | 每阶段交付什么，什么条件算完成？ |
| [进度记录](docs/progress.md) | 本轮完成了什么，下一步是什么，GitHub 同步状态如何？ |

## 目标闭环

~~~mermaid
flowchart LR
    A["版本化 AgentSnapshot"] --> B["Task Agent 执行训练侧任务"]
    B --> C["轨迹、反馈和成本"]
    C --> D["Meta Agent 提出 ImprovementPlan"]
    D --> E["执行 harness / 数据 / 训练算子"]
    E --> F["固定协议下验证候选"]
    F --> G["选择新快照或保留父代"]
    G --> A
~~~

Meta Agent 是提出改进的角色，Task Agent 是执行任务的角色；两者可以使用不同模型，也可以在后续实验中共享同一份更新后的权重。角色关系必须记录在实验清单中。

## 仓库组织

当前只提交设计文档、设计示例和协作约定。实现按里程碑逐步新增，避免先建大量空模块。

~~~text
README.md
CONTRIBUTING.md
configs/design/                 # 明确标为 design-only 的示例
docs/
  research/                     # 文献证据、代码核查与检索边界
  architecture.md
  decisions.md
  experiment-protocol.md
  benchmarks.md
  roadmap.md
  progress.md
~~~

计划采用 Python 的 src 布局。未来的 src/rsi_base/core 只包含领域对象、协议和状态机；第三方引擎、benchmark 规则和具体研究方法分别进入 adapters、benchmarks 和 recipes，详见[架构](docs/architecture.md)。

## 开发与进度同步

每个里程碑提交可审阅的代码、配置、验证结果，并更新 docs/progress.md。仅在远端实际接受提交并核对提交 SHA 后，标记为“已同步 GitHub”。数据、权重和完整轨迹存于外部 artifact 目录，Git 中保存来源、哈希、配置和摘要。

远端仓库：[PLUTO-SCY/RSI_Base](https://github.com/PLUTO-SCY/RSI_Base)。

调研日期：2026-11-16。文献结论、代码检查和待验证设计分别标识；没有执行过的实验不写成复现结果。
