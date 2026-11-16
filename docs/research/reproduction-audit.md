# 重点论文与公开代码核查

核查日期：2026-11-16。用户此前试跑 WHALE 遇到多处问题，要求从零搭建。本轮做文献与源码静态核查，未重跑这些项目；用户既有运行观察与本轮可直接验证的事实分别记录。

下面四个问题用于说明每篇工作的科研定位；代码部分用于确定能参考什么。**无一篇作为本项目代码基座。**

## 第 1 篇｜HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems

来源：Chen 等，arXiv:2606.01779v1，2026。重点读取 §3–4、附录 C/D/F。[论文](https://arxiv.org/html/2606.01779v1)

### 1. 问题与任务定位

研究已有 agent 系统的联合适应：执行结构变更后，如何使 policy 适应该 harness。不是全新 benchmark 任务。

### 2. 方法与核心思想

对 Planning、Action、Memory 做失败定位和历史引导编辑；预算内筛选候选，再以成功轨迹训练 harness-specific LoRA。重点是保存成对系统和 adapter 谱系。

### 3. 验证数据与任务

ToolHop、HotpotQA/2Wiki、RestBench-TMDB、API-Bank 等，覆盖多跳工具和检索/API 任务。主实验使用 Qwen3-4B/8B；附录给出 LoRA 配置。不能从这些设置推出四卡实际耗时。

### 4. Core contributions

把模块故障诊断、harness 种群筛选和成对 policy 对齐连成方法；对本项目的参考点是 bundle 边界与 harness-specific adapter，效果需在独立实现上重新验证。

### 公开代码核查

固定快照：[8bef9488c35a21ef510a6f069476305025ba5519](https://github.com/mingju-c/HarnessForge/tree/8bef9488c35a21ef510a6f069476305025ba5519)。

- [根 README](https://github.com/mingju-c/HarnessForge/blob/8bef9488c35a21ef510a6f069476305025ba5519/README.md)将 4B/8B 工作区分开，包含生成 harness、pair manifests、benchmark runner 和 SFT 数据工具；LlamaFactory 作为共享目录存在。
- [harness production 入口](https://github.com/mingju-c/HarnessForge/blob/8bef9488c35a21ef510a6f069476305025ba5519/HarnessForge_4B/harness_production/run_harness_production.py)规定分阶段输入和生成文件契约。它是机制参考，不证明所有阶段可一键闭环。
- 根 README 说明大 checkpoint 不随仓库提交、需自行服务对应 policy；还需核对实际 adapter 来源、数据来源和 split。
- **独立实现要求**：模型大小进入配置；任务插件不复制 4B/8B 工作区。一个 runner 能处理不同 ModelRef。

## 第 2 篇｜SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

来源：Xia 等，arXiv:2602.08234v2，2026。重点读取 §3–4、附录 D。[论文](https://arxiv.org/html/2602.08234v2)

### 1. 问题与任务定位

已有交互任务中，如何把历史经验压缩为可用技能，并随 policy 学习更新。

### 2. 方法与核心思想

从成功/失败轨迹提炼分层 SkillBank；冷启动 SFT 教模型使用技能，再用 GRPO 与失败驱动技能更新共同训练。技能演化比任意可执行 harness 编辑的范围窄。

### 3. 验证数据与任务

ALFWorld、WebShop 及搜索任务。附录 D.2 报告 8×H100 80GB、约 30 小时/实验，其中 RL 24 小时、SFT 2 小时；不能直接换算为任意四卡配置。

### 4. Core contributions

把经验提炼、层次化检索和技能/策略递归更新连接；适合作为 skill-memory 研究 recipe，不应要求所有任务都具有技能类别模板。

### 公开代码核查

固定快照：[4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d](https://github.com/aiming-lab/SkillRL/tree/4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d)。

- [README](https://github.com/aiming-lab/SkillRL/blob/4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d/README.md)给出 SFT 数据、任务 checkpoint、GRPO 启动和 template/embedding 检索模式。
- [skill_updater.py](https://github.com/aiming-lab/SkillRL/blob/4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d/agent_system/memory/skill_updater.py)构造 AzureOpenAI client，内部指定 o3。这一片段说明改换 Meta provider 需要处理实现耦合；并非在本轮运行中发现故障。
- **独立实现要求**：SkillBank 是版本化资源；Meta provider、检索方式、类别映射和更新策略分别配置，GRPO 只是一个 learner。

## 第 3 篇｜Self-Evolving Coding Agents via Harness-Policy Optimization（Self-HPO）

来源：Wei 等，arXiv:2610.22791v1，2026。重点读取 §2–3、附录 A。[论文](https://arxiv.org/html/2610.22791v1)

### 1. 问题与任务定位

已有软件工程任务上，让 agent 自己针对不断变化的失败模式改进 harness 与 policy。

### 2. 方法与核心思想

H-step 由当前 policy 分析失败并改 harness；P-step 做 GRPO；新 policy 成为下一轮 proposer，未解决训练题更新搜索集。原文还使用 BAR 改善稀疏奖励分组。

### 3. 验证数据与任务

SWE 系列训练任务；评估 SWE-bench Verified、Multilingual、Pro。Qwen3.5-35B-A3B，附录 A 使用 8×B200。四卡小模型方案属于缩小规模的独立实验。

### 4. Core contributions

递归改变 proposer 的 policy 与搜索任务分布，区别于固定外部 Meta 模型；self_meta 是本项目值得保留的研究模式。

### 公开代码核查

固定快照：[3c3d0bd93a8a5a72399c855783460b141ea27d49](https://github.com/MSR-Orchard/Self-HPO/tree/3c3d0bd93a8a5a72399c855783460b141ea27d49)。

- [README](https://github.com/MSR-Orchard/Self-HPO/blob/3c3d0bd93a8a5a72399c855783460b141ea27d49/README.md)区分 self-proposer 与外部 HPO，并依赖 Orchard 远程任务沙箱、SGLang 服务及 slime 训练。
- [P-step 脚本](https://github.com/MSR-Orchard/Self-HPO/blob/3c3d0bd93a8a5a72399c855783460b141ea27d49/scripts/p_step/run_p_step.sh)包含具体模型并行、SWE 和八卡配置；不能仅把 GPU 数字改成 4 就视为通用迁移。
- 用户笔记中的外部 proposer 型号需按版本校准：本轮读取的原文与 README 写 **Opus 4.8**。
- **独立实现要求**：角色关系显式化；困难任务采样器为可替换组件；软件工程沙箱放在 benchmark/execution adapter。

## 第 4 篇｜When Harnesses Evolve, Which Models Should Follow?（CoEvo-RSI / CoEvoX）

来源：Su 等，arXiv:2610.25078v1，2026。重点读取 §3–4、附录 A。[论文](https://arxiv.org/html/2610.25078v1)

### 1. 问题与任务定位

已有垂直任务中，变更 harness 后如何处理模型与接口失配，而非机械继承最新 checkpoint。

### 2. 方法与核心思想

从历史 archive 选 harness 起点，再选择或合并历史模型，收集验证通过轨迹做 SFT 适应，以验证结果决定晋升。

### 3. 验证数据与任务

失败归因、带角色权限的 Text-to-SQL、智能家居指令；包含域外测试。主设置为 Qwen3.5-9B、10 轮、每轮 8 候选、4×H100 全参 SFT。17.48–20.71 小时对应三轮成本分析，不是十轮总耗时。

### 4. Core contributions

把模型历史继承与 harness 继承共同考虑；TIMR 是接口失配诊断，不等于任务质量。参考点是可查询的多父代谱系和成对验证。

### 公开代码核查

固定快照：[b9928a973af68781f0ea7fccfbb8a6ee7af0f3f1](https://github.com/SuJQ0722/CoEvo-RSI/tree/b9928a973af68781f0ea7fccfbb8a6ee7af0f3f1)。

- [README](https://github.com/SuJQ0722/CoEvo-RSI/blob/b9928a973af68781f0ea7fccfbb8a6ee7af0f3f1/README.md)列明三个 domain、模型与配套 harness。
- [领域协议](https://github.com/SuJQ0722/CoEvo-RSI/blob/b9928a973af68781f0ea7fccfbb8a6ee7af0f3f1/src/domains/base.py)已分离部分评估、采样和选择接口；可参考职责划分。
- [base.yaml](https://github.com/SuJQ0722/CoEvo-RSI/blob/b9928a973af68781f0ea7fccfbb8a6ee7af0f3f1/configs/base.yaml)可直接查到四卡、长上下文、ZeRO-3、Meta CLI 和继承设置。数据准备和外部引擎仍是独立依赖。
- **独立实现要求**：不继承其代码与默认任务假设。接入训练时先做简单父代继续训练，checkpoint merging 后置，避免一开始引入合并兼容性和额外显存成本。

## 第 5 篇｜WHALE: A Simple Recipe for Joint Harness–Weight Optimization

来源：Kim 等，arXiv:2609.00196v1，2026。重点读取 §3–6、附录 B。[论文](https://arxiv.org/html/2609.00196v1)

### 1. 问题与任务定位

已有工具任务中，只优化权重或只优化 harness 会受到冻结侧限制，研究交替优化及切换时机。

### 2. 方法与核心思想

W-step 在当前 harness 下做在线成功轨迹筛选与 SFT；H-step 固定新模型做 Meta-Harness 搜索。包含固定小步与基于训练信号 patience 的调度。

### 3. 验证数据与任务

SearchQA、DAPO-Math/AIME、Lichess puzzles，使用 Qwen3.5-2B/4B。主比较用 (E,I)=(0.6,6)，并报告其他 schedule；测试 mean@8 不等同于 pass@8。

### 4. Core contributions

可解释的交替基线与 phase 时长分析；适合参考调度思想，其算法价值与公开工程可运行程度应分别判断。

### 公开代码核查

固定快照：[63faf4deb2dd2938775af6e2b020a9030f3aa697](https://github.com/krafton-ai/WHALE/tree/63faf4deb2dd2938775af6e2b020a9030f3aa697)。

- [README](https://github.com/krafton-ai/WHALE/blob/63faf4deb2dd2938775af6e2b020a9030f3aa697/README.md)说明三个 domain 各有一份修改后的训练框架，属于实验代码布局。
- [复现说明](https://github.com/krafton-ai/WHALE/blob/63faf4deb2dd2938775af6e2b020a9030f3aa697/docs/reproducing.md)写明原实验每节点 8×H200；整理后的 launcher 只做了 stub 验证，尚未执行端到端 GPU 验证。
- 同一说明给出的当前默认是 (0.2,6)；代码的 5 次搜索加上初始 harness 评估才是论文的 I=6。启动脚本默认不直接等于主表设置。
- [数据说明](https://github.com/krafton-ai/WHALE/blob/63faf4deb2dd2938775af6e2b020a9030f3aa697/docs/data.md)明确 SearchQA 部分中间训练资产未发布，无法仅靠公开仓库完整重建；数据源也需固定 snapshot。
- [交替 driver](https://github.com/krafton-ai/WHALE/blob/63faf4deb2dd2938775af6e2b020a9030f3aa697/run/lib/alternate.sh)通过目录和 accepted_harness 文件交接状态。这里只记录读到的实现，不把潜在脆弱点写成已复现 bug。
- **独立实现要求**：我们自己写循环、数据转换、训练适配和恢复；真实小模型训练、阶段产物校验与中断恢复均列为交付项。

## 综合比较与实施选择

| 参考机制 | 我们如何表达 | 实施顺序 |
| --- | --- | --- |
| WHALE 式交替 | phase 顺序和切换规则 | v0.2 接入训练后固定交替；adaptive 后置 |
| HarnessForge 式成对适应 | model–harness 组合与 adapter 谱系 | v0.1 记配置组合，v0.2 加权重/adapter |
| SkillRL 式技能演化 | 版本化 SkillBank + EvolveSkills 算子 | 后续按课题单独增加 |
| CoEvo-RSI 式历史继承 | archive + checkpoint parent selector | 后续单独增加；合并另做 |
| Self-HPO 式 self-proposal | Meta role 解析到更新后的 Task ModelRef | 基础共演化稳定后单独检验 |
| CoHARL 式共享模型种群 | 多 harness 指向共享模型，单独的 pool scheduler | 研究需要时单独引入 |

这是机制参考顺序，不是对原文效果的横向排名。不同工作数据、模型、预算和指标差异较大，不比较其表面分数。

## 复现承诺分级

1. **软件闭环可重放**：阶段交接、记录与恢复正确。
2. **机制复现**：在明确缩小的模型/数据设定下，执行与原算法相同的关键步骤。
3. **结果复现**：匹配原文协议、资产和预算后核验分数与统计误差。

首版交付 1，并为部分机制提供 2 的实验定义；不承诺仅凭四卡直接复现所有论文主表。没有公开资产的部分记录为协议差异，不能静默用替代数据后沿用“原样复现”的名称。
