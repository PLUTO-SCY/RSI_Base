# 开发计划与验收

2026-11-16 初版。人力估算按一名熟悉 Python/LLM 训练的主要开发者，加一名研究同学参与任务/结果核对。四卡为共享实验资源；以下是开发工作日估算，不是模型运行时长保证。

## 1. 版本目标

| 阶段 | 工作量估算 | 交付 | 完成判据 |
| --- | --- | --- | --- |
| M0：调研与设计 | 本轮 | 文献证据、独立架构、四卡实验协议、任务与计划 | 文档/设计配置一致，可据此分派开发 |
| M1：P0 核心契约 | 3–5 天 | 类型对象、状态机、账本、artifact、CPU runner、故障恢复 | 无模型调用的完整模拟循环；恢复不重复结算 |
| M2：第一个真实闭环 | 5–7 天 | 数学工具任务、Task/Meta 接口、LoRA SFT 与推理适配 | 一个真实小模型完成 H→collect→verify→SFT→reload→eval |
| M3：v0.1 四卡 backbone | 5–8 天 | 第二个任务 ALFWorld、3 轮循环、基础对照、恢复与报告 | 两种任务无需改核心；真实四卡运行、固定协议、完整谱系 |
| M4：v0.2 研究扩展 | 5–8 天 | evaluator 替换、数据混合/回放、历史父代、ToolHop/BFCL 或一个 eval bridge | 完成可替换组件的对照与兼容性检查 |
| M5：v0.2 GRPO | 5–10 天 | 官方 RL 后端、多轮分组、token/logprob 一致性、资源交接 | 在相同 snapshot/trajectory 协议上完成真实小模型 RL 更新 |
| M6：v0.3 可选 | 另行估算 | self_meta、共享 policy 多 harness pool、adaptive 调度 | 分别经过对照与预算验证，再合入稳定 recipe |

预期 v0.1 约 3–4 周；v0.2 再约 2–4 周。引擎/驱动适配或资源排队可能延长，按 gate 通过情况更新。M6 是研究迭代，不纳入首版必须交付。

## 2. M1：核心先于框架堆叠

实现少量稳定领域对象：ModelRef、HarnessRef、AgentSnapshot、TaskManifest、Trajectory、ImprovementPlan、TrainingJobSpec、EvaluationRecord、RunManifest。

实现显式注册、配置解析、原子 artifact 写入、JSONL 事件、SQLite 索引、稳定 job ID 和预算预留/结算。模拟的 Meta/Task/Learner 用于验证状态路径，所有输出带 simulated 标记。

必要检查：

- 同配置/同输入的重复执行能复用已校验 artifact。
- checkpoint 目录存在但缺 manifest 时不能跳过训练阶段。
- 超时重试、进程中止与恢复不漏算或重复消费预算。
- 子候选失败不覆盖父代；空成功集得到 NO_UPDATE。
- 更换 SearchEvaluator 不改变任务执行代码。
- 从零输入到候选决策能在 CPU CI 中完成，不下载模型和任务数据。

## 3. M2：尽早暴露真实训练问题

先做运行环境诊断与依赖组合锁定：Python、PyTorch/CUDA、TRL/PEFT、vLLM、模型支持、GPU 可见性、CPU 工具容器。

Task Agent 先用一个小而清楚的 ReAct harness；Meta 通过 provider-neutral 接口产出受约束 patch 与改进理由。模型端点和预算由运行配置提供，不绑定某个付费服务或 coding CLI。

第一个任务使用 GSM8K 的训练侧小集合和受控计算工具。完成数据转换后，人工逐 token 核对一条多轮轨迹的训练 mask，再跑真实的小 SFT 作业。

必须取得：

1. 原始轨迹、rendered prompt、工具输出和训练样本的一致性证据。
2. 训练前后可识别的不同 ModelRef 与实际 adapter 权重更新。
3. 导出物可以重新加载，推理请求确实命中新版本。
4. 新 model–harness 组合完成独立确认集评估。
5. 运行有准确预算、时间分解和错误记录。

不要求该微型作业产生显著能力提升；要求数据和版本链正确。只看到 loss 下降或成功启动进程不算闭环完成。

## 4. M3：v0.1 的验收重点

### 四卡和多任务

- 分时 GPU lease：生成停止、训练启动、训练退出、推理重载。
- 用 32–64 个真实 episodes 测 throughput、显存、工具延迟、样本通过率。
- 接 ALFWorld 原生 episode，复用相同核心；如果需要修改 engine 中的任务分支，回到协议设计处理。
- 至少 Static、H-only、W-only、Alternating 四个配置；比较 phase 调度时补 Sequential。
- 完成一个小规模 3 轮运行，并对论文机制与项目 recipe 的差异留说明。

### 故障注入

在采集、训练、导出和确认评估后分别中断/恢复；验证候选状态、产物哈希、累计预算和父代引用一致。

### 报告

输出完整快照谱系、逐题指标、coverage、错误与成本、选择理由、2×2 配对和固定最终评估。主研究结论需要多 seed；单 seed 运行清楚标为工程 pilot。

**v0.1 的定义**：两个任务能跑同一闭环，组件可替换，训练产物真正回到下一轮，失败可恢复。不是“所有主流 benchmark 已支持”。

## 5. M4/M5：扩展次序

M4 先做成本低且能直接服务研究的组件：

1. 独立 Gold/Proxy evaluator 接口与离线重评分，不必重跑全部轨迹。
2. 数据配比按训练 token 计量，记录当前轨迹、兼容历史回放和来源；先做固定配比对照。
3. 从历史选择 parent checkpoint/harness，先不做权重合并。
4. 一个轻量工具任务或外部评测桥，验证插件扩展成本。
5. 在冻结快照上运行 transfer/forgetting 集合。

M5 通过官方引擎扩展点接入 GRPO。先验证 task–harness 分组和行为版本，再增加长上下文/并发。保留 RSFT 作为轻量基线，不因 RL 接入而分叉成第二套框架。

若 RL 接入需要对官方引擎做大范围 fork，应暂停该技术路径，记录约束并比较现有桥接方案，其他独立开发继续进行。

## 6. 可直接分派的 backlog

状态均为 planned，M0 文档项除外。负责人在实际分工后填写，不虚设已完成任务。

| ID | 任务 | 依赖 | 可验收产物 |
| --- | --- | --- | --- |
| P0-01 | 类型协议、配置 schema 与 capabilities | M0 | 非法配置/不兼容组合提前失败 |
| P0-02 | ArtifactStore、内容 digest、事件与索引 | P0-01 | 可重建索引与不可变原始记录 |
| P0-03 | 候选状态机与谱系 | P0-01/02 | 接受/拒绝/无更新/失败全部可表示 |
| P0-04 | 多维 BudgetLedger 与幂等 job ID | P0-02 | 中断恢复与重试正确记账 |
| P0-05 | CPU 模拟闭环、取消与恢复 | P0-03/04 | 无网络的故障路径验证 |
| P1-01 | 环境诊断与官方依赖锁 | 无；可提前 | GPU/驱动/模型支持证据 |
| P1-02 | Task runner、seed harness、隔离工具 | P0-01 | 真实多轮轨迹与 episode 上限 |
| P1-03 | Meta 提案协议、patch/bundle 物化 | P0-01/03 | 可重放的有效/无效候选 |
| P1-04 | GSM8K manifest、split、verifier | P0-01 | 去重/划分/正确错误样例一致性 |
| P1-05 | 轨迹转换、assistant/tool loss mask | P1-02/04 | 逐 token 金样例与过长样本记录 |
| P1-06 | 官方 LoRA SFT learner 与导出 | P1-01/05 | 真正更新的可加载 ModelRef |
| P1-07 | 推理服务生命周期与 digest 握手 | P1-01/06 | 新请求命中新 checkpoint |
| P1-08 | 固定交替 recipe 与确认选择 | P0、P1-03/07 | 一个真实循环与可审阅报告 |
| P2-01 | 四卡资源 lease 与故障恢复 | P1-08 | 3 轮 pilot，无显存残留冲突 |
| P2-02 | ALFWorld adapter | P1-02 | 第二任务不修改核心 |
| P2-03 | 对照、成本与 2×2 报告 | P2-01/02 | 同协议比较及明确统计限制 |
| P3-01 | evaluator 替换/离线重评分 | P2-03 | 固定候选池的选择质量对照 |
| P3-02 | 数据混合、兼容回放与历史父代 | P2-03 | token 配比/来源/兼容性报告 |
| P3-03 | ToolHop/BFCL 或外部评测 bridge | P2-02 | capabilities 与官方评分一致 |
| P3-04 | 官方 GRPO 适配 | P1-05、P2-01 | prefix/logprob/mask 与真实更新 |
| P4-01 | self_meta 与多 harness pool | P3 | 独立消融、冻结 batch 边界 |

## 7. 第一周的具体工作

- 第 1 天：在本机真实运行上下文诊断 GPU；冻结核心协议、两个任务用例和 artifact 格式。
- 第 2 天：事件、状态与预算；做 CPU 模拟循环。
- 第 3 天：中断/恢复；GSM8K 数据划分与固定 seed harness。
- 第 4 天：模型服务与真实轨迹，核对训练样本和 token masks。
- 第 5 天：争取跑小 SFT 和重载；若依赖栈尚未就绪，给出明确诊断结果，不把模拟路径算作真实闭环。

第一周的主要风险是环境与数据接口，而不是搜索算法复杂度。M2 的完整验收允许跨到第二周。

## 8. 每个里程碑的 GitHub 交付

一个清楚的提交或功能分支包括：功能代码、最小配置、实际验证摘要和更新后的 progress。研究结果附 manifest、指标摘要和 artifact 位置；大文件留在外部。

源码提交成功后核对远端 SHA。无写权限或推送失败时记录本地文件状态、具体障碍和可恢复的提交包，不能标为已同步。

本轮不创建一批空 GitHub issue，也不提前运行昂贵训练。backlog 已足够分派；实施时可按需要拆成 issue。
