# LoRA SFT：从 harness 演化到模型更新

相同的 SFT 入口也支持 [全参微调](full-tuning.md)。

这一阶段复用已冻结的 harness，完成 **训练集采样 → 正确轨迹筛选 → LoRA SFT → 开发集模型选择 → 下一轮 harness 演化**。训练策略由配置固定，Meta 继续负责 harness；当前不搜索训练超参数。

## 1. 准备独立训练环境

```sh
sh scripts/harness.sh setup --train
```

训练环境放在 `.venv-train/`，与核心 `.venv/`、推理 `.venv-task/` 分开。版本固定为 PyTorch 2.8.0、Transformers 4.57.6、PEFT 0.18.0、Accelerate 1.12.0，完整依赖见 `requirements/train.lock`；安装入口为 `scripts/setup_train.sh`。

默认配置在 [`configs/lora_sft.json`](../configs/lora_sft.json)：BF16 LoRA、rank 16、alpha 32、attention 的 q/k/v/o 投影、1 epoch、学习率 2e-4、每卡 batch 1、梯度累积 8。默认 4B 模型可先用单张 A100 80GB；其他硬件需要根据序列长度和 batch 调整。

## 2. 从训练集采样

先完成一次 harness 搜索，保持对应 Task 服务可用：

```sh
sh scripts/harness.sh run search --output runs/demo
sh scripts/harness.sh sft collect --source-run runs/demo --output runs/lora-demo
```

`collect` 要求源运行已有 `frozen.json`，验证训练记忆和标签的校验值，再用冻结的 harness 重新执行 **训练集**。它不读取开发题目或测试内容，不将开发／测试轨迹用于 SFT。

- 检索采用留一法，排除题目本身与相同文本的训练记忆。
- 默认 `training_target="verified"`，保留模型实际生成且通过固定评分器的回答。
- 有标注的数据可用 `training_target="reference"`。使用同一个最终调用的输入，以训练集标准标签做监督。采样统计会区分这些样本在训练前是正确、错误还是格式不符。
- direct 流程监督唯一一次回答；verify/rewrite 流程只监督最终通过验证的分类调用。错误草稿可以保留在最终调用的输入中，不作为训练目标。
- 所有调用正常结束才可入选。重复输入和目标只保留一条；`verified` 还会过滤错误和格式不符的回答。样本不足时停止并保留已有记录。

`collection_repeats` 控制每题采样次数。用 `collection_temperature` 可单独调节采样温度，例如设置为 `0.7` 并采样 3 次，评测仍使用源运行的模型参数。`dataset.json` 记录样本数、覆盖的不同题目数和筛选原因，便于发现重复采样或困难题覆盖不足。

产物为 `train.jsonl`、`dataset.json`、`collection_identity.json` 和 `collection/` 中的原始轨迹。重复相同命令会复用已完成采样。若源服务端口变化，可指定 `collect --base-url URL`，模型名称必须仍匹配实际基础模型或父 adapter。

## 3. 训练并保存 adapter

```sh
RSI_TRAIN_GPUS=1 sh scripts/harness.sh sft train --output runs/lora-demo
```

训练器使用模型原生 chat template，**只对 assistant 输出及其结束标记计算 loss**；system/user 内容和 padding 的 labels 为 `-100`。超过 `max_length` 的整条样本丢弃，不截断回答。仅 LoRA 参数可训练，基础权重冻结。

所有产物在同一个运行目录：

```text
runs/lora-demo/
├── train.jsonl                 经验证的训练消息
├── dataset.json               采样统计与校验值
├── collection_identity.json   父模型、harness、数据和源码来源
├── collection/                采样轨迹、调用账本与缓存
├── training/                  优化器 checkpoint、token 统计、训练与失败记录
└── adapter/                   不可变 adapter、文件哈希和模型名称
```

训练身份固定配置、数据、源码、依赖锁、基础 checkpoint 和父 adapter 的文件哈希。中断后重跑同一命令，从最近保存的 Trainer checkpoint 恢复；首个 checkpoint 前中断则重新开始训练，已发生的尝试保留记录。完成后重跑只校验并返回 adapter。更改配置时使用新的输出目录。

多卡数据并行使用显式 GPU 列表和进程数，例如：

```sh
RSI_TRAIN_GPUS=1,3 RSI_TRAIN_PROCESSES=2 \
  sh scripts/harness.sh sft train --output runs/lora-demo
```

有效 batch 为 `每卡 batch × 梯度累积 × 进程数`。恢复需保持进程数一致；新实验应在采样前确定配置。推理与训练分别占用指定的 GPU，脚本不接管其他服务。

## 4. 加载与开发集选择

```sh
RSI_GPU=2 RSI_TASK_PORT=18001 \
  sh scripts/harness.sh serve-adapter --run runs/lora-demo
```

服务启动前校验基础 checkpoint 和 adapter 文件；vLLM 同时提供基础模型和以内容哈希命名的 LoRA。继续训练已有 adapter 时也会加载父 adapter，比较保持在同一推理服务内。

在另一终端运行：

```sh
sh scripts/harness.sh sft evaluate --output runs/lora-demo
```

评测固定开发集、重复次数和调用协议，运行四组对照：

| 模型 | 初始 harness | 演化后 harness |
| --- | --- | --- |
| 父模型 | 初始配置对照 | 直接衡量 harness 效果 |
| LoRA 候选 | 检查模型更新的迁移 | 与“父模型 + 演化后 harness”配对比较 |

接受规则为：在演化后 harness 固定时，候选每次重复的正确数均不下降，平均至少多答对一题。否则保留父模型。此规则用于基础设施的版本选择，不等价于统计显著性或泛化结论。

选择结果写入 `decision.json`，四组开发集报告在 `evaluation/val/report.json`。报告含固定 harness 时的按题配对差值和 95% bootstrap 区间，重复评测先按题平均。`next-search.json` 引用被选中的模型和当前 harness。

## 5. 继续演化或结束测试

继续下一轮 harness 搜索：

```sh
sh scripts/harness.sh run search \
  --config runs/lora-demo/next-search.json \
  --output runs/next-search
```

Meta 可以通过 `model/context.json` 查看上一次训练的配置、样本数量、训练指标、开发集对照和模型选择结果；测试结果不进入这些信息。

若继续下一次 SFT，可复制训练配置，将 `parent_adapter` 指向被选中的 adapter 路径，并保持 LoRA 结构一致，再以新的冻结搜索和新的输出目录运行。若更新被拒绝，继续使用原父模型／父 adapter。

本轮完整迭代结束、开发集选择冻结后，才执行对应的最终评测：

```sh
sh scripts/harness.sh sft test --output runs/lora-demo
```

`test` 先校验已冻结的开发集决策，再读取测试集，测试不改变模型选择。若还要继续基于反馈开发，不应把同一测试集再当作独立测试。内置 S2D 划分默认标记为历史回放。

## 配置与使用边界

所有阶段使用同一 `--config`、`--data` 和 `--output`。默认数据为 `data/s2d`，对照服务为 `http://127.0.0.1:18001/v1`；地址可用 `--base-url` 修改。训练采样和评测均有独立调用账本和预算。当前采样／评分适配分类任务，其他 benchmark 需要接入相应轨迹转换和验证器。

评估训练收益时，在同一个 harness 和相同解码参数下比较父模型与 adapter。可用 `reference` 做已标注任务的监督基线，再比较 `verified` 的采样覆盖率和独立测试增益。多随机种子的比较应使用同一份测试集，并按题汇总配对差值。训练消息、权重、日志与报告保存在被 Git 忽略的运行目录。

实现依据：[Transformers Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)、[Chat templates](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)、[PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)、[vLLM 0.11 LoRA serving](https://docs.vllm.ai/en/v0.11.0/features/lora.html)。
