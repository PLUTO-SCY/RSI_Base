# 扩大选择集的 harness-only 实验

目标仍是固定模型的harness evolution；LoRA、RL和新任务训练不启动。

## 为什么重做数据角色

上一组的三次搜索均在50题验证集上从41升至43，但211题独立测试退步：平均差−3.63个百分点，按题配对95%区间为[−6.95,−0.53]。重复相同50题没有解决候选对开发集的适应。负结果与曲线完整保留在[memory-evolution-v4.json](validation/memory-evolution-v4.json)。它不满足稳定性门槛。

原先另封存的591题尚未执行任何模型调用。由于原测试已经退步，要求两个集合都提升的旧门槛不可能通过，取消旧的额外确认。按新协议将这591题一次性分成 **300道选择题、291道全新最终测试题**；训练记忆仍为原200条，完全不变。按标签采用最大余数分配和固定种子20261116分层打乱，不使用模型结果切分。新划分不等同于论文官方分数。

旧50题和211题均不进入新搜索或新验收门槛。此次调整依据旧实验失败诊断；新的291题在选定方案并冻结三次搜索前，不进入模型执行或Meta反馈。固定manifest见[memory-expanded-manifest.json](validation/memory-expanded-manifest.json)。

## 固定设置

- Task仍为同一Qwen3-4B-Instruct-2507快照；128输出token、temperature0、12000字符记忆上下文，每题最多两次调用。
- 三次独立搜索分别使用GPU0/1/2上的同版本服务；权重、dtype、服务参数相同。Meta为项目隔离Claude Code + DeepSeek，设`--effort low`和180秒单次超时；参数仅影响子进程。
- 固定七个基线，按300道选择题的三次重复选择最强父代。继续包含上一阶段的32示例参考策略，避免削弱父代。
- 每次5轮，每轮局部结构候选加一个Meta候选；另有同候选数随机结构搜索对照。局部邻域优先检查调用流程，再检查示例数量与类别上限。
- 训练LOO诊断仍为64题，逐步结果与有序记忆清单传给Meta；Meta只获得选择集汇总。
- 接受仍要求每次重复不退步、平均至少多答对一道选择题。最终门槛保持：三次搜索在291道新测试题上分别有正增益，且按题聚类的配对95%区间下界大于0。
- 数据扩大、提案顺序、Meta努力级别及失败处理均是本轮工程调整；不把结果单独归因于其中一项。

## 运行

此强起点实验使用源码`de1adf0`。恢复其本地运行需检出该版本；新clone的数据准备见[完整环境与划分重建](harness-backbone.md)，不依赖历史日志。

```sh
.venv/bin/python scripts/prepare_memory_expanded.py
# 三个终端，各运行一个相同权重的服务；没有权重训练
RSI_GPU=0 RSI_TASK_PORT=18000 sh scripts/serve_task.sh
RSI_GPU=1 RSI_TASK_PORT=18001 sh scripts/serve_task.sh
RSI_GPU=2 RSI_TASK_PORT=18002 sh scripts/serve_task.sh
```

```sh
.venv/bin/python scripts/run_memory.py suite --parallel \
  --config configs/memory_s2d_expanded.json --data data/memory-s2d-expanded \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18001/v1 http://127.0.0.1:18002/v1 \
  --output runs/memory-s2d-expanded-v2
.venv/bin/python scripts/run_memory.py finalize \
  --config configs/memory_s2d_expanded.json --data data/memory-s2d-expanded \
  --output runs/memory-s2d-expanded-v2
.venv/bin/python scripts/audit_memory_ablations.py --suite runs/memory-s2d-expanded-v2 --data data/memory-s2d-expanded
.venv/bin/python scripts/verify_memory_resume.py \
  --suite runs/memory-s2d-expanded-v2 --data data/memory-s2d-expanded
.venv/bin/python scripts/export_memory.py --suite runs/memory-s2d-expanded-v2 --data data/memory-s2d-expanded --name memory-evolution-expanded
```

原始日志、数据和模型继续位于Git忽略目录；脚本、配置、指纹、曲线及正负结果进入Git。历史实验需检出对应源码提交恢复。

结果等待实际执行，不预填增长或通过状态。

实现检查发现：当总示例数为8时，每类上限32→31不会改变执行。v1在测试解封前停止，保留调用与部分结果；对这类等价策略做规范化后另开v2。该修正防止相同执行的推理波动被当成新策略收益，数据划分不变。
