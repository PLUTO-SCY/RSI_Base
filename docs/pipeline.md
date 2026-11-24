# 联合演化

`pipeline` 把数学 Task Agent 的 harness 和模型权重作为同一个父版本。每轮 Meta 读取当前版本的开发集分数、典型成功与失败轨迹，以及最近候选的收录结果。Meta 选择一个动作并给出理由，宿主负责执行、评测和提交版本。

| 策略 | 本轮更新 | 数据来源 |
| --- | --- | --- |
| `harness` | 单轮数学 harness 的系统提示 | 用开发集轨迹诊断问题 |
| `offline_sft` | SFT 权重 | 训练集题目和人工参考解答，按当前 harness 组装对话 |
| `online_sft` | SFT 权重 | 当前模型在当前 harness 下对训练题采样，保留答对且正常结束的轨迹并去重 |
| `grpo` | verl GRPO 权重 | 用当前模型和 harness 对训练题进行新的组内采样，以数值正确性为奖励 |

训练轮固定 harness，改 harness 的轮次固定权重。宿主把每次通过评测的候选设为下一轮的父版本，支持连续选择同一策略，也支持在 SFT、GRPO 与 harness 之间切换。默认要求候选在同一开发集上净多做对至少一题。训练样本不足、GRPO 无组内奖励差异或开发分数未达到门槛时，保留父版本并记录原因。独立的分类 harness 流程、SFT 和 GRPO 入口可用于单组件研究。

## 运行

在仓库根目录执行。环境文件和凭据保存在项目目录，联合入口负责启动和关闭本地 Task 服务。

```sh
sh scripts/harness.sh setup --gpu --claude --train --rl
sh scripts/harness.sh credentials
sh scripts/harness.sh model
.venv/bin/python scripts/prepare_grpo_data.py --with-references

sh scripts/harness.sh pipeline run --output runs/joint-adaptive
sh scripts/harness.sh pipeline test --output runs/joint-adaptive
```

`pipeline run` 默认使用 [`configs/pipeline.json`](../configs/pipeline.json) 和准备好的 GSM8K 数据。`pipeline test` 从运行目录读取冻结状态，在同一批测试题上对比初始和入选版本。宿主在运行前检查样本划分是否重叠，测试题和答案始终留在宿主侧，测试分数只在冻结之后产生。安装和依赖排查见 [环境与运行](getting-started.md)与 [verl 环境排查](verl-troubleshooting.md)。

## 配置和对照

- `schedule: null` 让 Meta 每轮自主选择。复制配置并设为 `"schedule": ["harness", "offline_sft", "online_sft", "grpo"]`、`"rounds": 4` 可跑固定顺序对照。对照的 harness 仍由 Meta 提案，其余策略与数据源由顺序指定。
- `sft` 和 `grpo` 中的基座、训练模式和初始父权重设置一致。两处均可设置 `"training_mode": "full"`，联合流程会把入选的完整 checkpoint 传给下一轮训练。LoRA 模式要求相同的 rank、alpha、目标模块和零 dropout。已有训练产物可通过两处相同的 `parent_adapter` 或 `parent_checkpoint` 作为初始权重。
- 默认用 GPU 0 推理、GPU 1 做 SFT、GPU 2 和 3 做 GRPO。可修改 `runtime` 中的卡号和端口。GRPO 的 `prompts_per_batch` 须能被卡数整除。训练和推理可复用同一张卡，宿主在训练前关闭占用该卡的 Task 服务。
- `limits` 约束 Meta 调用、Task 请求和运行时间。在线 SFT 的 Task 请求计入开发预算，verl 内部 rollout 的规模由 `grpo.steps`、`prompts_per_batch`、`samples_per_prompt` 和长度上限控制。冻结测试有独立账本，沿用同一配置的 Task 和时间上限。
- `min_gain_questions` 是开发集净增正确题数门槛。小样本验证可设为 `0` 以验证平分时的权重接续。报告应说明本次门槛、用时和采样规模。

自定义数据通过 `--data DIR` 指定。`train.jsonl`、`val.jsonl` 和 `test.jsonl` 每行包含字符串 `id`、`question`、`answer`。可选 `references.jsonl` 每行包含一个训练题 `id` 和一个带最终数字的 `response`。宿主验证参考答案的数值，覆盖样本达到 SFT 下限时才向 Meta 开放 `offline_sft`。纯数字和 `#### 42` 等数值输出均由同一评分器处理。

## 记录和恢复

| 位置 | 内容 |
| --- | --- |
| `run.json` | 配置、基座、数据、源码和依赖哈希 |
| `rounds/NNN/plan.json` | Meta 或固定顺序选择的动作、来源和理由 |
| `rounds/NNN/training/` | SFT 数据及来源、在线轨迹或 GRPO 分组奖励、训练日志和权重 |
| `rounds/NNN/decision.json` | 父代、候选、开发集配对分数和收录结果 |
| `frozen.json`、`test/report.json` | 冻结的版本谱系和配对测试结果 |

用原命令加 `--resume` 从同一路径继续。配置、源码和数据需与 `run.json` 一致。宿主复用已经落盘的提案、训练产物和有效 Task 响应。训练进程故障时，查看 `training/pipeline-attempts/` 后可恢复该轮，已接受的父版本保留。完整实验产物保存在本地忽略目录。
