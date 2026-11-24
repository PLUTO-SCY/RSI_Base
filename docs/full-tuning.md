# 全参 SFT 与全参 GRPO

用 `training_mode` 选择训练方式。配置省略该字段时使用 LoRA。全参模式从已有 Hugging Face 模型初始化，训练时更新全部可训练权重，并把完整 checkpoint 写到输出目录的 `model/` 下。

| 模式 | 配置值 | 前一轮产物 | 本轮产物 |
| --- | --- | --- | --- |
| LoRA | `"lora"` | `parent_adapter` | `adapter/` |
| 全参 | `"full"` | `parent_checkpoint` | `model/` |

`base_directory` 选择初始基座。全参续训保留基座配置，把上一轮 SFT 或 GRPO 的 `model/` 路径填入 `parent_checkpoint`。训练器校验父模型文件、基座身份和训练数据。每一轮使用独立的输出目录。

## 全参 SFT

安装项目训练环境，并完成一次 harness 搜索以生成 `runs/demo/frozen.json`。采集阶段使用搜索配置对应的父模型服务。采集与验证规则见 [SFT 工作流](lora-sft.md)。

```sh
sh scripts/harness.sh setup --gpu --train --rl
sh scripts/harness.sh sft collect --config configs/full_sft.json \
  --source-run runs/demo --output runs/full-sft
RSI_TRAIN_GPUS=2,3 RSI_TRAIN_PROCESSES=2 sh scripts/harness.sh sft train \
  --config configs/full_sft.json --output runs/full-sft
```

## 全参 GRPO

使用已经准备好的 GSM8K 数据，或通过 `--data` 指定数值任务。依赖和数据格式见 [GRPO 工作流](lora-grpo.md)。

```sh
RSI_RL_GPUS=2,3 sh scripts/harness.sh rl train \
  --config configs/full_grpo.json --output runs/full-grpo
```

从全参 SFT 接力时，把以下字段加入一份 GRPO 配置并通过 `--config` 传入。相同字段也用于从 GRPO checkpoint 继续全参 SFT。

```json
{
  "training_mode": "full",
  "parent_checkpoint": "runs/full-sft/model",
  "learning_rate": 0.000003
}
```

GRPO 通过 verl FSDP2 保存可恢复的分片 checkpoint。完成后使用 verl 的模型合并器导出 BF16 Hugging Face 权重，并校验相对训练父代的有效张量变化。 [verl checkpoint 文档](https://verl.readthedocs.io/en/latest/advance/checkpoint.html)

## 启动父模型与候选并配对评测

在两个终端分别加载父模型和候选。把下方的 `runs/full-sft` 换成 `runs/full-grpo` 即可加载对应轮次的模型。

```sh
RSI_GPU=0 sh scripts/harness.sh serve-model --run runs/full-sft --parent
RSI_GPU=1 sh scripts/harness.sh serve-model --run runs/full-sft
```

全参模式的父模型默认使用 `18000`，候选默认使用 `18001`。使用自定义端口时，通过 `--parent-base-url` 和 `--base-url` 传给评测命令。两个模型在相同的题目、harness 和解码设置下评测。

```sh
sh scripts/harness.sh sft evaluate --config configs/full_sft.json --output runs/full-sft
sh scripts/harness.sh sft test --config configs/full_sft.json --output runs/full-sft
sh scripts/harness.sh rl evaluate --config configs/full_grpo.json --output runs/full-grpo
sh scripts/harness.sh rl test --config configs/full_grpo.json --output runs/full-grpo
```

评测和测试分开执行。测试要求先冻结开发集选择，继续执行 harness 搜索可使用 SFT 评测生成的 `next-search.json`，其中包含被选中模型的名称和服务地址。

全参 SFT 使用 FP32 参数与 BF16 混合精度，DDP 的每张训练卡均保存一份参数、梯度和优化器状态。配置显存预算时计入序列长度和 batch 的激活开销。GRPO 使用 FSDP2 分片。完整模型和训练 checkpoint 一并写入本地运行目录。 [Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
