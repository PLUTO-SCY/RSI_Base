# LoRA GRPO

verl 在一个节点内交替用 vLLM 采样、用 FSDP2 更新 LoRA。RSI Base 提供固定的单次 Task 提示和数值评分器，负责数据划分、PEFT 产物和模型选择。训练、部署与评测使用同一份 `configs/lora_grpo.json`。

## 安装与数据

```sh
sh scripts/harness.sh setup --gpu --rl
sh scripts/harness.sh model
.venv/bin/python scripts/prepare_grpo_data.py
```

`requirements/verl.json` 固定 verl 版本、源码和上游依赖锁。RL 环境位于 `.venvs/rl/`，需要能运行 CUDA 13.0 的 NVIDIA 驱动。训练使用 PyTorch SDPA，安装跳过上游依赖组里的独立 FlashAttention wheel 和 TRL。Task 服务从 `.venv-task/` 启动。

默认数据以固定种子从 GSM8K 官方训练池划出 400 题作开发集，剩余 7,073 题用于训练。官方 1,319 题用作最终测试。也可用 `--data DIR` 接入相同格式的数值任务，目录提供 `train.jsonl`、`val.jsonl`、`test.jsonl`。每行包含字符串 `id`、`question`、`answer`。训练只读取训练划分，答案由验证器使用。训练阶段的 rollout 和 checkpoint 保存在指定的运行目录。

## 训练与回载

```sh
RSI_RL_GPUS=0,1 sh scripts/harness.sh rl train --output runs/grpo-demo
RSI_GPU=2 sh scripts/harness.sh serve-adapter --run runs/grpo-demo
```

训练可以改为 `RSI_RL_GPUS=0,1,2,3`。每批题数 `prompts_per_batch` 要能整除所选 GPU 数。`samples_per_prompt` 控制组内采样数。训练超参数、GPU、代码、数据和基座哈希在运行开始时固定。同一目录重试会用最近的 verl checkpoint 恢复，已导出的 adapter 直接校验后返回。不同设置使用新的目录。

从已有 adapter 开始时，把配置的 `parent_adapter` 设为其 `adapter/` 目录，并使用 `--config PATH`。父 adapter 和本轮的基座、rank、alpha 及目标模块必须匹配。GRPO 使用组内标准化优势，当前配方关闭 KL。日志保留每步采样与分数，组内全对或全错时 GRPO 没有该组的奖励学习信号。

## 评测与版本选择

```sh
sh scripts/harness.sh rl evaluate --output runs/grpo-demo
sh scripts/harness.sh rl test --output runs/grpo-demo
```

模型服务同时加载基础模型、候选 adapter 和可选的父 adapter。开发集对比固定任务提示与解码参数，逐题配对报告增益和区间。候选正确数更高才写为选中模型。测试要求开发集决策已经冻结，并复用相同模型与协议。`decision.json` 中的 `selected_model` 可用于下一轮相同数学任务的 Task 配置。训练奖励与独立测试精度应分别解读。
