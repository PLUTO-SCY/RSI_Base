# Benchmark 接入计划

**以下全部是计划接入项，目前没有已实现的 benchmark adapter。** 本文区分支持模式和完成标准，避免“能调用模型”被算成“支持整个实验”。

## 1. 三种能力分别声明

- **eval**：能把冻结快照的预测/产物交给官方或经校验的 scorer；未必能产生可训练轨迹。
- **episode**：能 reset/step/终止真实环境，通过 TaskRunner 记录完整交互。
- **learn**：轨迹、动作格式、mask 与 learner 对齐，并通过真实训练和模型重载验证。

一个 adapter 可以只有 eval，也可以支持三者。是否支持可编辑 harness、token logprobs、工具调用、分组 reset、沙箱等，通过 capabilities 显式声明；不支持的组合在运行前报错。

## 2. 按任务形态实施

| 优先级 | 任务族与候选 benchmark | 计划模式 | 评分与关键工程成本 |
| --- | --- | --- | --- |
| P1 | GSM8K + 受控计算工具 | 原生 episode/learn/eval | 数值答案；工具执行由宿主隔离，先做第一个真实闭环 |
| P2 | ALFWorld 文本环境 | 原生 episode/learn/eval | 环境目标完成；reset、步数、状态和 action grammar |
| P3 | ToolHop；BFCL 指定子集 | ToolHop 原生；BFCL 先 eval bridge | 结构化工具调用、依赖链、官方任务判分；逐子集确认在线环境能力 |
| P3 | MATH-500、AIME | 先 eval；数学训练另用训练来源 | 数学等价/官方规则；小测试集不能承担反复候选搜索 |
| P4 | HotpotQA、Natural Questions、2WikiMultiHopQA | 原生检索环境，或 Inspect solver bridge | 答案指标；固定 corpus/index，必要时另记 judge 成本 |
| P4 | HumanEval+/MBPP+、LiveCodeBench | 官方 evaluator bridge，后续 repair episode | 隔离执行测试、时间窗口、release 和生成次数 |
| P5 | SWE-bench Verified、Multilingual/Pro；Terminal-Bench | 官方 runner / Harbor bridge | 仓库/终端环境、镜像、CPU/磁盘、隐藏测试和长轨迹成本 |
| P5 | WebShop；后续真实网页环境 | WebShop 原生，浏览器环境独立 adapter | 状态 reset、环境依赖、用户/会话隔离；需专门资源验证 |
| 辅助 | lm-eval 中选定的语言/推理任务 | model-only eval bridge | 通用能力与遗忘；不代表任意工具 agent 的结果 |

首版 v0.1 的承诺范围是 **GSM8K 工具推理 + ALFWorld**，不是上表全部完成。第二个任务是检验架构边界的必要条件，而非为了凑 benchmark 数量。

## 3. 原生插件契约

每个 benchmark 插件给出：

1. 数据获取与版本 manifest；无网络 CPU CI 另用自己创建的微型 fixture。
2. 训练、搜索、确认、最终测试的合法来源和 group split 规则。
3. EpisodeHandle 的输入/输出、工具 schema、reset/seed、状态终止与回收。
4. evaluator 输入契约、reference 的独立存储、官方指标与错误政策。
5. 任务能力描述、最小可执行样例和适配器一致性证据。
6. 数据和执行资产所需资源、许可证/访问条件、下载校验。

算法不能从插件名推断其能力，插件也不能在内部决定用哪个 Meta 模型、多少代或如何做 SFT。

## 4. 外部桥接的实际边界

### Inspect

可以把 Task Agent 包装为自定义 solver，复用任务的数据、scorer 和 sandbox；逐个任务核对 solver 是否可替换、能否取得原始调用轨迹。[官方 Task 文档](https://inspect.aisi.org.uk/tasks.html)

评测任务可执行，不代表 training split 存在。最终测试内容不因 bridge 方便就导回 Meta evidence。

### lm-evaluation-harness

用于指定模型在固定语言任务上的生成/似然评估。工具 harness 的控制流不会自然进入标准 model API，因此第一版只声明 model-only eval。[官方仓库](https://github.com/EleutherAI/lm-evaluation-harness)

### Harbor / 软件工程官方 runner

将完整 Task Agent 作为 agent 接口接入，使用对应任务环境与 verifier，导出原始过程记录。调用代理仅开放模型访问，隐藏测试不对候选 harness 可见。[Harbor 文档](https://docs.harborframework.com/)

先验证单任务、单实例执行；再扩大并发。四卡 GPU 不决定容器 CPU、内存、镜像缓存和磁盘容量是否足够。

## 5. 评测来源与解释边界

- [GSM8K 官方数据仓库](https://github.com/openai/grade-school-math)：从训练部分分出开发集合，测试部分冻结。
- [ALFWorld 官方仓库](https://github.com/alfworld/alfworld)：先用文本环境，保留环境原有的训练/评估边界，记录任务类型和 reset。
- [ToolHop 原始论文](https://arxiv.org/abs/2501.02506)：实施前固定官方数据/工具资产来源并核对 scorer；本轮未审计它的独立适配代码。
- [BFCL 官方说明](https://gorilla.cs.berkeley.edu/leaderboard.html)：包含不同版本/类别；固定版本与子集，不将所有结果统称为一个无版本的 BFCL accuracy。
- [MATH-500 数据卡](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)：作为 held-out math 测试候选；若训练源含 MATH，必须排除相同题及已知变体。
- [EvalPlus](https://evalplus.github.io/)：使用增强测试与明确的执行限制，不能仅运行模型样例。
- [LiveCodeBench 官方仓库](https://github.com/LiveCodeBench/LiveCodeBench)：固定 release/date window；使用持续更新的 latest 无法保证相同任务集合。
- 检索与 SWE 等较重任务，在对应里程碑补充官方版本、样本划分和环境证据，不预先把论文中的复现资产当作本项目资产。

数学评估需区分可稳定做数值匹配的 GSM8K 与需要表达式等价的其他任务；不能只比较最后一个字符串。AIME 题量较小，单次变化噪声大，报告重复尝试和区间；不据几道题的波动决定全部开发方向。

## 6. “已接入”的验收

一个任务只有以下证据齐全，才能从计划表转到支持列表：

- 与固定官方 scorer 在人工构造的正确/错误/格式错/超时样本上一致；
- 在真实小模型上完成非空任务集，记录完成覆盖和错误；
- 可换两个不同的 harness；得到的 prompt、工具观察和产物与预期一致；
- 若声明 learn，成功样本通过转换、mask 核对、真实训练、导出和重载；
- reset 后无上一任务状态或未授权的参考内容残留；
- 数据/协议/环境版本可追溯，相关回归检查能在对应环境复跑。

每次新增版本或更换 scorer 后重新记录兼容性，不沿用旧的支持声明。
