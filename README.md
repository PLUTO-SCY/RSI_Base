# RSI Base

独立实现的最小 RSI 实验框架，按阶段扩展，面向 evaluator、效率和泛化研究。

**当前优先做固定权重的 harness evolution。** 已实现真实 Meta/Task 调用、记忆检索与上下文策略演化、重复选择、随机搜索对照和冻结后的独立测试；LoRA 后置。

首个 backbone 使用 Qwen3-4B-Instruct-2507 + S2D 记忆分类：200 条训练记忆、300 道选择题、291 道测试题；三次独立搜索，每次 5 轮 × 2 候选，每候选重复验证 3 次。三张 A100 分别运行同权重的推理服务，第四张卡不占用。

**当前结论：基础框架已可运行和复用；自动演化方法的研究效果还不够好，尚未达到稳定优于强对照的目标。**

下面是在同一291题测试集上的三次搜索平均结果：

| 方案 | 测试准确率 | 含义 |
| --- | ---: | --- |
| 固定few-shot | 52.04% | 简单起点，始终使用固定示例 |
| 强固定harness | 78.81% | 预设七个固定策略中，在选择集上最好的对照 |
| 随机搜索 | 80.72% | 同候选数和单题上限的随机配置搜索 |
| 当前演化（Meta＋局部结构搜索） | 79.34% | 本框架当前搜索策略的结果 |

79.34%相对简单起点提高了27.30个百分点，说明自动调整检索和上下文结构确实有用。但它相对强固定harness只高0.53个百分点，相对随机搜索低1.37个百分点，这两项差异的95%区间都跨0，**没有证据证明当前演化策略优于这两个强对照**。主要增长发生在第一轮，后面大多进入平台；目前不把它称为持续自我改进成功。各方案实际token/调用成本不同，详见[完整实验报告](docs/harness-backbone.md)。

回查已确认搜索端的源码/执行反馈不足、提案探索过窄，以及部分局部候选不改变实际输入，详见[失败诊断与证据](docs/harness-search-diagnosis.md)。这些问题尚未通过修复后的对照实验证明各自的性能影响。

此前还有一组**历史失败实验**：用50题选择候选时，选择集分数41→43/50，但在211题独立测试上平均退步3.63个百分点。它与上表是两组不同实验，不是同一个测试集上的矛盾结论。当前划分为200条训练记忆、300题候选选择、291题最终测试；旧负结果保留在[历史记录](docs/memory-evolution-expanded.md)。训练记忆用于检索，没有训练模型权重。

## 当前闭环

~~~text
Task 执行训练诊断与候选选择样本 → 固定规则评分
→ Meta 提案 + 局部结构候选
→ 重复评估；每次不退步且平均净增至少一题才接受
→ 下一轮；全部搜索冻结后统一评估独立测试集
~~~

- Meta：Claude Code CLI + 可配置底层模型；本次接通 deepseek-v4.1-flash。
- Task：自研薄 harness，通过 OpenAI 兼容接口调用本地模型；支持直接回答或最多两次调用的复核/改写。
- 可演化字段：示例检索/数量/类别分配、上下文顺序、覆盖与对比布局、复核流程和 system prompt。当前是配置演化，代码编辑后续单独接入。
- 评分、模型、生成预算和数据划分由宿主固定；Meta 只接收训练诊断和选择集汇总。
- 保存配置、源代码/依赖锁哈希、逐题记录、提案、决策、调用账本和校验后的轮次 checkpoint。
- 支持完成轮次与部分逐题结果恢复；拒绝输入变更和已封存产物损坏。

## 运行记忆 harness backbone

完整入口见[从全新clone开始的上手指南](docs/getting-started.md)：环境版本、隐藏凭据、数据重建、模型下载、单卡服务、两题smoke、三卡实验与结果导出。统一使用`sh scripts/harness.sh`，无需activate或修改全局Claude/shell配置。

~~~sh
sh scripts/harness.sh setup --gpu --claude
sh scripts/harness.sh data
sh scripts/harness.sh model
~~~

接着按指南配置自己的凭据，启动单卡服务并运行`doctor`和`smoke`。上述准备命令不会自动启动演化或调用Meta API。只做无GPU/无API检查：`setup`后运行`check`和`doctor`。研究结果与验收边界另见[实验报告](docs/harness-backbone.md)。

## 最小 API 连通性检查

需要 uv；Meta 还需已有的 Claude Code CLI。本机验证版本为 Python 3.11.15、uv 0.11.17、Claude Code 2.1.169。所有 Python 依赖安装到项目 .venv。

~~~sh
sh scripts/setup.sh
.venv/bin/python scripts/configure_credentials.py
.venv/bin/python scripts/fetch_gsm8k.py \
  --revision 3101c7d5072418e28b9008a6636bde82a006892c \
  --dev-size 6 --test-size 6
sh scripts/run_smoke.sh
~~~

凭据脚本隐藏输入，将密钥写入被 Git 忽略的 .env.local，权限 0600。已有文件不会被覆盖。无需修改用户的 shell 或全局 Claude 配置。

早期数学pilot的baseline和恢复命令见[历史指南](docs/legacy-math-quickstart.md)。运行真实命令会消费所配置服务的API用量。

## 配置

| 配置 | 状态与用途 |
| --- | --- |
| configs/deepseek_smoke.json | 实测：Meta 和 Task 分别通过 Claude Code / OpenAI SDK 调用同一 DeepSeek 服务，验证软件闭环 |
| configs/local_qwen.json | 本地 Task endpoint 模板，尚未在 Qwen 服务上实测；Meta 仍使用 DeepSeek |
| configs/qwen_math_growth.json | 4B 六轮无增长对照；历史评分协议见诊断记录 |
| configs/qwen_1_7b_math_growth.json | 1.7B 六轮历史对照，策略与格式耦合 |
| configs/qwen_1_7b_boxed_growth.json | 历史数学诊断：固定格式后缀、600字符策略、八轮搜索，未确认稳定泛化 |
| configs/memory_s2d.json | 记忆分类harness：固定4B模型、5轮×2候选、重复验证、独立搜索与随机对照 |
| configs/memory_s2d_expanded.json | 扩大选择集协议；三张卡分别运行独立搜索，训练记忆和模型权重固定 |
| configs/memory_s2d_starter.json | 当前默认：从固定 few-shot 演化，与强固定基线、随机搜索分别比较 |
| configs/calibration/ | 失败和中间校准配置，供核查，不作为增长成绩 |
| configs/design/ | 后续 benchmark / 训练方案设计，不能执行当前 CLI |

Meta 的凭据、状态和调用参数仅设置在子进程中。使用项目 .local/claude、bare 模式、关闭模型工具/MCP，并监测全局 Claude/shell 配置哈希。Meta 不直接操作实验仓库或评分器。

## 验证

~~~sh
sh scripts/check.sh
~~~

离线测试覆盖评分、数据隔离、选择/回退、非法提案、恢复、产物校验、预算保留和 Claude 配置隔离，不使用真实凭据或模型服务。

真实运行结果与复现范围见[验证记录](docs/validation-v0.1.md)。小集合上跑通循环不代表获得了能力提升，当前版本也没有更新模型权重。

八轮实验出现开发集 35→37→40/48；独立观察集 25→19/32，最终验收 40→41/64，尚未确认稳定泛化收益。难度校准、失败对照、复测、完整曲线和重跑命令见[增长诊断](docs/growth-diagnosis.md)。

后续任务选型参考[论文 benchmark 与协议核查](docs/research/harness-benchmark-setups.md)。

## 代码与文档

~~~text
src/rsi_base/        # Meta/Task、评分、记忆执行、选择/搜索、冻结审计与 checkpoint
scripts/            # 环境创建、凭据输入、数据准备、接入探测、检查与运行
tests/              # 无网络的契约和故障路径测试
configs/            # 当前可运行配置与独立的 design-only 目录
uv.lock             # Python 依赖锁
docs/               # 上手、验证、调研、架构和开发计划
~~~

优先阅读[上手指南](docs/getting-started.md)、[增量路线](docs/roadmap.md)、[进度](docs/progress.md)与[选型决策](docs/decisions.md)。

历史提交的作者与提交者已按仓库所有者要求更正，文件内容与实验记录保留；旧报告中的提交ID可通过[新旧映射](docs/history/README.md)追溯。

调研参考：[文献地图](docs/research/landscape.md)、[五篇论文与代码核查](docs/research/reproduction-audit.md)。长期方案：[架构](docs/architecture.md)、[实验协议](docs/experiment-protocol.md)、[benchmark 计划](docs/benchmarks.md)。

代码从零实现，论文只作机制参考；官方 SDK/CLI 作为依赖。数据、权重、密钥和完整运行记录不进 Git，验证摘要与复现脚本随代码提交。
