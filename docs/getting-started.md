# 环境与运行

所有命令均在仓库根目录执行。环境、依赖缓存、模型、Claude Code 状态和凭据均保存在项目目录；脚本不修改 shell profile、系统 Python 或用户的全局 Claude 配置。

## 1. 安装

宿主机需要 Linux x86_64、Git、curl、Node.js ≥18/npm。本地推理和训练需要 NVIDIA 驱动和支持 BF16 的 GPU。verl GRPO 的依赖需要支持 CUDA 13.0 的 NVIDIA 驱动。

只运行本地 harness 演化时，安装 Task、Meta 和隔离执行环境。

```sh
sh scripts/harness.sh setup --gpu --claude --sandbox
```

同时运行 harness 演化、SFT 与 GRPO 时，安装全套环境。

```sh
sh scripts/harness.sh setup --gpu --claude --sandbox --train --rl
```

已有 harness 环境时，用 `sh scripts/harness.sh setup --train --rl` 补装训练环境。LoRA 和全参微调使用相同的训练依赖。

默认 4B 模型以单卡服务，显存需求取决于上下文和并发，可以通过 `serve` 参数调整。独立分类流程可使用远程 Task 服务，省略 `--gpu` 并修改配置中的 `task`。数学联合流程在本机管理 Task 服务。 [联合演化](pipeline.md)

| 内容 | 脚本 / 版本文件 | 本地位置 |
| --- | --- | --- |
| uv 0.11.17、Python 3.11.15、核心与开发依赖 | `bootstrap.sh`、`setup.sh`、`uv.lock` | `.local/`、`.venv/` |
| vLLM 0.11.0 与 GPU 依赖 | `setup_task.sh`、`requirements/task.lock` | `.venv-task/` |
| SFT（`--train`） | `setup_train.sh`、`requirements/train.lock` | `.venv-train/` |
| verl GRPO（`--rl`） | `setup_rl.py`、`requirements/verl.json`、`requirements/rl-numpy.lock` | `.venvs/rl/` |
| Claude Code 2.1.169 | `requirements/claude-code/package-lock.json` | `.local/claude-cli/` |
| Python 检索执行隔离 | `setup_sandbox.py`、`requirements/sandbox.json` | `.local/bin/bwrap` |

隔离执行依赖内核允许用户命名空间。提供的 bubblewrap 包针对 Linux x86_64/Ubuntu 20.04；其他系统可提供兼容的 `bwrap`。`doctor --ready` 会实际运行隔离探针，失败时先解决宿主兼容性。

安装后可按照 [verl 环境排查](verl-troubleshooting.md)运行 CUDA 算子检查。训练流程见 [LoRA SFT](lora-sft.md)、[LoRA GRPO](lora-grpo.md)和 [全参微调](full-tuning.md)。

仅做开发和离线检查：

```sh
sh scripts/harness.sh setup
sh scripts/harness.sh check
```

## 2. 配置 API 与数据

```sh
sh scripts/harness.sh credentials
sh scripts/harness.sh data
```

凭据通过隐藏输入保存在 `.env.local`，权限为 600，不进入 Git。Meta 使用 `RSI_API_KEY`；默认网关、模型和 token/时间预算见 `configs/default.json`。Claude Code 需要 **Anthropic 兼容入口**，因此 Meta 的 `base_url` 是网关根地址；Task 使用 OpenAI 兼容 `/v1` 入口。替换服务时分别核对协议。

数据下载只读取公开数据，不调用模型。`scripts/prepare_data.py` 根据 `requirements/data/s2d.json` 中固定的 Git/Hugging Face 版本、去重规则和随机种子，生成：

```text
data/s2d/
  train.jsonl       200 条训练记忆
  val.jsonl         300 条开发样本
  test.jsonl        291 条冻结后评测样本
  labels.json      22 个标签
  manifest.json    数据锁与校验值
```

重复执行会校验现有文件，文件发生变化或不完整时直接报错，不覆盖。数据处理先移除归一化重复及字符三元组 Jaccard ≥0.9 的近重复，再按标签分层划分开发和测试样本。示例采用公开合成数据，划分曾用于开发，因此默认 `test_status=historical_replay`；需独立泛化验证时换用未参与开发的数据并设置 `held_out`。

运行 GRPO 时，再准备默认数学数据。脚本按数据锁下载并校验 GSM8K，生成 `data/gsm8k-grpo/`。

```sh
.venv/bin/python scripts/prepare_grpo_data.py
```

联合演化使用同一数据目录。加 `--with-references` 可同时生成离线 SFT 所需的训练题参考解答。

## 3. 启动 Task 服务

```sh
sh scripts/harness.sh model
RSI_GPU=0 RSI_TASK_PORT=18000 sh scripts/harness.sh serve
```

模型下载支持缓存续传，固定 revision 为 `cdbee75f17c01a7cc42f958dc650907174af0554`。服务默认只监听本机。需要更换模型目录、服务名称、端口或 GPU 时，可设置 `RSI_TASK_MODEL_DIR`、`RSI_TASK_MODEL_NAME`、`RSI_TASK_PORT`、`RSI_GPU`，并同步修改 JSON 中的 Task 地址与模型名称。额外 vLLM 参数可附加在 `serve` 后面。

在另一终端检查：

```sh
sh scripts/harness.sh doctor --ready --local-gpu
sh scripts/harness.sh smoke --output runs/smoke
```

`doctor` 不生成内容；`smoke` 请求两次 Task，检查格式，不读取最终测试。加 `--meta` 会额外调用一次 Meta 检查连接；完整 MCP 演化流程由下面的 `run search` 验证。smoke 输出目录须不存在，重跑时换一个路径。

## 4. 执行与恢复

单次演化：

```sh
sh scripts/harness.sh run search --output runs/demo
sh scripts/harness.sh inspect runs/demo
```

默认从初始 harness 开始，执行 5 轮；每个候选用完整开发集重复评测 3 次。Meta 可读取源码、训练记忆、开发题目与标签、完整轨迹、历史候选和对照分数，通过 MCP 提交有限次数的小样本探测和候选。宿主评分与选择下一轮版本，测试内容不向 Meta 提供。

中断后执行相同命令恢复。配置、数据、核心源码或依赖锁变化时会拒绝混用已有记录；新配置应使用新的输出目录。已冻结的搜索不再调用模型。模型/API 请求失败和已发生的调用保留在本地账本中。

三次独立搜索与冻结后测试：

```sh
sh scripts/harness.sh run suite --output runs/demo-suite
sh scripts/harness.sh run finalize --output runs/demo-suite
sh scripts/harness.sh runtime
sh scripts/harness.sh report --suite runs/demo-suite
```

三次种子固定为 42、123、2026。`suite` 只搜索；`finalize` 要求三次均冻结后才执行测试，生成按题配对的增益、区间与对照。导出的 JSON/PNG/SVG 默认为 `runs/demo-suite/reports/`，不进入 Git。`runtime` 用本地 Task 环境记录权重与依赖指纹；远程服务需要提供对应指纹文件并通过 `report --runtime PATH` 指定。导出还会校验实验核心源码和当前提交的一致性，源码变化后的旧实验应在对应提交导出。

4 卡机器可在三个终端分别启动 GPU 0/1/2、端口 18000/18001/18002 的相同 Task 服务，第四卡留给其他实验；然后运行：

```sh
sh scripts/harness.sh run suite --output runs/demo-suite --parallel \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18001/v1 http://127.0.0.1:18002/v1
sh scripts/harness.sh run finalize --output runs/demo-suite --parallel
```

并行服务是独立推理副本，不涉及模型训练。

## 5. 修改实验

复制 `configs/default.json`，通过 `--config PATH` 传入。常用字段：

| 字段 | 用途 |
| --- | --- |
| `task` / `meta` | 模型、地址、凭据环境变量名、token 与超时限制 |
| `initial_baseline` | 初始 harness，默认 `no_memory` |
| `rounds` / `validation_repeats` | 演化轮数 / 完整开发集重复次数 |
| `max_probes` / `probe_queries` | 每轮小样本探测数量 / 样本数 |
| `max_tool_calls` | 每轮 Meta 工具调用上限 |
| `max_task_calls` / `wall_seconds` | 整次运行的调用 / 时间预算 |
| `proposal_strategy` | 默认 `agent`，使用源码与轨迹可见的 MCP 搜索 |

替换分类数据时，通过 `--data DIRECTORY` 指定同样布局的文件。每条 JSONL 包含唯一 `id`、`question`、`answer`，答案须属于 `labels.json`；不同划分禁止重复题目，`manifest.json` 记录来源和处理规则。默认 `data`/`doctor` 命令校验内置数据锁，其他数据需自行校验。其他任务类型需要增加自己的 harness、数据加载和评分实现，见 [实现结构](architecture.md)。

## 本地输出

`calls.jsonl` 是调用账本，`baselines/` 是固定对照，`evolution/` 保留每轮提案、工具调用、轨迹和选择，`random/` 是随机搜索对照，`frozen.json` 保存最终搜索状态。`finalize` 生成评测与汇总报告。原始记录保持不变；实验资产均保存在 `runs/` 等被忽略的目录。
