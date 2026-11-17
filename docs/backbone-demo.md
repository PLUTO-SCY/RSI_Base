# 当前目标：从简单harness搭建RSI基础设施

本阶段提供可复用的Meta/Task闭环。默认使用自然、未经优化的起点；强固定和随机搜索属于研究参考，超过它们不作为基础设施验收条件。

默认`configs/memory_s2d_agent.json`指定`initial_baseline="no_memory"`：普通分类提示、合法标签与输出合同、无检索示例、单次调用。评分、模型和数据角色保持固定，Meta可逐步添加检索、上下文组织与执行流程。参考策略即使分数更高，也不会替换明确指定的初始父代。

先按[环境指南](getting-started.md)安装并启动Task服务，然后运行一个seed：

```sh
sh scripts/harness.sh run search --seed 42 --output runs/my-agent-demo
sh scripts/harness.sh inspect runs/my-agent-demo
```

`inspect`支持单run和suite；完整曲线与策略保存在`runs/my-agent-demo/frozen.json`，逐轮记录在`evolution/`。本命令不打开测试集。默认仍会计算六个固定参考和随机对照，账本记录其成本；这些参考不决定初始父代，不用于判断基础设施是否可用。

验收关注：

1. 真实Meta/Task能够执行，候选通过校验后被评估。
2. 评分、接受/拒绝和下一轮父代更新可追溯。
3. 配置、代码、实际输入/输出、成本和失败被保留。
4. 恢复不重复已完成调用，输入变化会被拒绝。
5. 用简单起点演示相对自身的增长，完整保留持平与退化结果。

候选仍按现有重复评估规则接受，不为制造曲线更改评分。三seed与置信区间用于后续研究分析；报告中的`experiment_completed`与`research_diagnostics`分别表达执行完成和统计比较，统计比较未通过不代表基础设施执行失败。

如需重复研究比较，运行`suite`再`finalize`；同一run必须保持配置/源码一致。旧291题继续标记历史回放。

```sh
sh scripts/harness.sh run suite --output runs/my-agent-suite
sh scripts/harness.sh run finalize --output runs/my-agent-suite
```

此前强起点配置保留为`configs/memory_s2d_agent_strong.json`，使用新输出目录显式运行即可。旧实验的精确复核需检出匹配的源码提交，详见[历史强起点报告](harness-agent-v1.md)。本次默认调整尚未重跑模型实验；已有曲线保持原样。
