# RSI Base

独立实现的最小 RSI 实验框架，按阶段扩展，面向 evaluator、效率和泛化研究。

**v0.0/v0.1 已实现，完成真实 API 与本地单卡 GPU 推理验证。** 当前支持固定权重的 prompt/harness 配置改进；LoRA SFT、代码级 harness 编辑和多任务训练尚未实现。

当前集中验证[记忆harness演化](docs/memory-evolution.md)：已实现检索/上下文/两次调用流程搜索、重复选择、随机对照及测试封存；正在运行三次独立搜索。稳定性结果完成后据实更新，LoRA后置。

50题选择集上的三次开发增长未通过211题独立测试；完整保留负结果。当前按[扩大选择集协议](docs/memory-evolution-expanded.md)使用200记忆/300选择/291全新测试，继续只做harness。

验收区分[标准起点的基础backbone与强固定基线挑战](docs/harness-backbone.md)，两类对照与结果分别保留。

## 当前闭环

~~~text
Task 执行开发样本 → 固定规则评分
→ Meta 提出一个新 system prompt
→ 子代在相同样本与协议下执行
→ 严格改善则接受，否则保留父代
→ 下一轮；最后冻结选择并独立评估
~~~

- Meta：Claude Code CLI + 可配置底层模型；本次接通 deepseek-v4.1-flash。
- Task：自研单次模型调用 harness，通过 OpenAI 兼容接口执行。
- 默认每轮一个候选，三轮串行运行；评分、模型、生成预算和数据划分固定。
- 保存配置、源代码/依赖锁哈希、逐题记录、提案、决策、调用账本和校验后的轮次 checkpoint。
- 支持完成轮次与部分逐题结果恢复；拒绝输入变更和已封存产物损坏。

## 快速开始

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

运行说明、独立 baseline 和恢复命令见[上手指南](docs/getting-started.md)。运行真实命令会消费所配置服务的 API 用量。

## 配置

| 配置 | 状态与用途 |
| --- | --- |
| configs/deepseek_smoke.json | 实测：Meta 和 Task 分别通过 Claude Code / OpenAI SDK 调用同一 DeepSeek 服务，验证软件闭环 |
| configs/local_qwen.json | 本地 Task endpoint 模板，尚未在 Qwen 服务上实测；Meta 仍使用 DeepSeek |
| configs/qwen_math_growth.json | 4B 六轮无增长对照；历史评分协议见诊断记录 |
| configs/qwen_1_7b_math_growth.json | 1.7B 六轮历史对照，策略与格式耦合 |
| configs/qwen_1_7b_boxed_growth.json | 当前增长试验：固定格式后缀、600字符策略、八轮搜索 |
| configs/memory_s2d.json | 记忆分类harness：固定4B模型、5轮×2候选、重复验证、独立搜索与随机对照 |
| configs/memory_s2d_expanded.json | 扩大选择集协议；三张卡分别运行独立搜索，训练记忆和模型权重固定 |
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
src/rsi_base/        # 类型、Task、Meta、评分、串行循环、文件记录与 CLI
scripts/            # 环境创建、凭据输入、数据准备、接入探测、检查与运行
tests/              # 无网络的契约和故障路径测试
configs/            # 当前可运行配置与独立的 design-only 目录
uv.lock             # Python 依赖锁
docs/               # 上手、验证、调研、架构和开发计划
~~~

优先阅读[上手指南](docs/getting-started.md)、[增量路线](docs/roadmap.md)、[进度](docs/progress.md)与[选型决策](docs/decisions.md)。

调研参考：[文献地图](docs/research/landscape.md)、[五篇论文与代码核查](docs/research/reproduction-audit.md)。长期方案：[架构](docs/architecture.md)、[实验协议](docs/experiment-protocol.md)、[benchmark 计划](docs/benchmarks.md)。

代码从零实现，论文只作机制参考；官方 SDK/CLI 作为依赖。数据、权重、密钥和完整运行记录不进 Git，验证摘要与复现脚本随代码提交。
