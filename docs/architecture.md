# 实现结构

数学联合流程按轮次调度 harness、离线 SFT、在线 SFT 和 verl GRPO，并把收录的 harness 与模型权重一同传给下一轮。分类 harness 演化由 Meta 提出改动，宿主执行任务、评分并选择候选。SFT 和 GRPO 也有独立入口，均支持 LoRA adapter 和全参 Hugging Face checkpoint。 [联合演化](pipeline.md)

## 分类 harness 运行流程

```text
配置 + 数据锁 + 固定 Task 模型
            ↓
评测初始 harness 与固定对照
            ↓
Meta 读取源码 / 开发数据 / 完整轨迹 / 历史结果
            ↓
编辑 harness → 小样本探测 → 提交候选
            ↓
完整开发集重复评测 → 接受改进或保留父代
            ↓
继续下一轮 → 冻结搜索 → 最终评测与本地报告
```

初始 harness 使用任务描述与标签，单次生成分类结果。`MemoryPolicy` 显式表示检索、示例选择与排序、标签覆盖、prompt、direct/verify/rewrite 流程等设置。一次任务最多两次模型调用；标签协议、数据划分、评分器与预算由宿主固定。

## 代码位置

| 模块 | 职责 |
| --- | --- |
| `classification.py` | 分类 harness、记忆检索、policy、输出解析和评分 |
| `agent.py` | OpenAI 兼容 Task 调用、凭据读取 |
| `improve.py` | 启动隔离配置的 Claude Code，收集 Meta 结果和失败 |
| `memory_agent.py` | MCP 演化流程、候选验证与选择、匹配预算的随机对照 |
| `memory_agent_tools.py`、`meta_tools.py` | 宿主开发工具和 stdio MCP 桥接 |
| `memory_code.py`、`sandbox_worker.py` | Python 检索函数的隔离执行和资源限制 |
| `memory_loop.py` | 配置校验、固定对照、评测、初始策略选择与恢复约束 |
| `memory_search.py` | 搜索入口、结构邻域与其他提案策略 |
| `memory_audit.py` | 冻结后评测、按题配对统计与汇总 |
| `sft.py` | 训练集轨迹筛选、模型/harness 交叉评测、模型选择和下一轮配置 |
| `lora_train.py` | 独立环境中的 Transformers/PEFT 训练、checkpoint 恢复与 adapter 产物 |
| `rl.py`、`scripts/train_grpo.py` | 固定数学 harness、verl GRPO、模型导出和父代配对评测 |
| `pipeline.py` | 按策略接口编排轮次、提交父版本、运行配对评测与冻结测试 |
| `pipeline_setup.py`、`pipeline_types.py`、`components.py` | 配置装配、可加载组件接口、Python 组件入口与指纹 |
| `pipeline_strategies.py`、`pipeline_data.py` | 内置策略、划分校验、离线与在线 SFT 数据准备 |
| `pipeline_runtime.py`、`pipeline_weights.py` | 本地或外接 Task 服务、训练进程与模型权重身份 |
| `weights.py` | adapter 与完整 checkpoint 的文件校验和父模型解析 |
| `storage.py`、`types.py` | 原子写入、调用账本、预算和类型 |
| `loop.py`、`benchmark.py`、`math_scoring.py`、`cli.py` | 通用记录辅助函数，以及独立的数学任务运行与评分接口 |

`scripts/harness.sh` 统一对外入口。脚本层负责环境安装、参数解析、服务启动、运行调度和产物导出。独立 GPU 环境分别服务于 Task 推理、SFT 训练和 verl GRPO。 [环境与运行](getting-started.md#1-安装)

## Meta 的信息和权限

MCP 工具包括 `list_artifacts`、`read_artifact`、`get_case`、`evaluate_candidate`、`submit_candidates`。Meta 可查看训练记忆、开发集及其标签、当前与历史轨迹、对照分数和相关源码；大文件分页读取，单个完整案例可按 ID 查询。最终测试不在工具可访问范围内。

Meta 可提交配置，也可定义 `rank(query, memory)` 检索函数。函数只处理宿主传入的训练记忆，返回排序后的训练 ID 或带分数的结果。执行环境无网络、宿主目录、凭据和 GPU，并限制 CPU、内存和执行时间。宿主检查返回值、去除题目自身和重复项，再完成示例组织与 Task 调用。

Claude Code 的配置目录、工作目录、模型和认证环境变量在子进程内设置；全局用户配置不修改。Claude 内置文件与 shell 工具关闭，只提供宿主 MCP 工具。

## 记录和恢复

每次运行固定核心源码、依赖锁、配置、数据和标签的内容哈希。恢复必须匹配；已有评测复用，未完成请求保留账本。选择只依据完整开发集重复评测，探测分数只用于调试。失败提案保留错误与成本，下一轮可以继续使用当前父代。

搜索结束写入 `frozen.json`。三次搜索全部冻结后才允许汇总测试；测试回放不改变已选策略。报告区分初始 harness 增益、固定对照和随机搜索比较，数据用途由 `test_status` 标识。

## 扩展方式

- 更换模型/API：修改 `task`、`meta` 配置和运行环境版本锁。
- 修改 harness 搜索空间：修改 `MemoryPolicy` 与 `MemoryHarness`，同步宿主工具 schema、约束和相关契约测试。
- 更换分类数据：复用当前文件布局、标签协议和评测器，通过 `--data` 指定目录。
- 增加任务类型：新增任务执行和评分模块，再接入独立运行入口；从已有明确用例出发调整共用逻辑。

SFT 用留一法收集训练集轨迹，支持通过验证的最终回答或训练集标准标签作为监督目标。它交叉评测模型和 harness，按开发集选择模型，再把训练配置和开发集反馈交给下一轮 Meta。LoRA 模式可在一个推理服务中比较父代和候选，全参模式分别加载其 checkpoint。 [LoRA SFT](lora-sft.md) · [全参微调](full-tuning.md)

GRPO 把固定数学提示和训练题交给 verl 在线采样。`grpo_reward.py` 调用项目数值评分器，`train_grpo.py` 从 FSDP2 checkpoint 导出模型，`rl.py` 通过 Task 服务做父代配对评测与模型选择。 [LoRA GRPO](lora-grpo.md) · [全参微调](full-tuning.md)
