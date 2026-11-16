# 基础 backbone 与强基线挑战分开验收

本轮仍只做固定权重的harness evolution，LoRA后置。两种问题分别报告，不能互相替代：

1. **基础backbone：** 从标准简单harness出发，Meta能否自动选择/改进记忆检索，获得可重复的独立测试收益。
2. **强基线挑战：** 在最强固定harness基础上，能否继续取得独立测试增益。

此前强起点实验完整保留：50题选择集上的提升没有通过211题测试；扩大到300选择/291新测试后的强起点实验也继续执行。结果不会被简单起点的收益覆盖。

## 标准起点实验

固定few-shot 8作为初始Task harness，训练记忆200条。few-shot 8是[Meta-Harness分类实验](https://arxiv.org/html/2603.28052v1)列出的标准固定基线；不是修改模型能力或评分器制造低分。所有候选仍可使用同一200条训练记忆，比较query-aware检索与固定示例选择。

配置 `configs/memory_s2d_starter.json` 显式设置`initial_baseline=fixed_fewshot`。框架默认仍从最强固定基线出发；选择标准起点需要显式配置。Meta只看到该起点及自身候选的验证汇总，不获得其他固定控制策略的成绩。

Task仍是同一4B模型；单题最多两次调用、每次128输出token、上下文12000字符。3次独立搜索，5轮×2候选，每候选重复验证3次，固定300题选择/291题新测试。所有实验先冻结，随后才打开新测试；对两个优化设置同时保持这一边界。

基础门槛：每次搜索均超过各自固定few-shot起点，且按题聚类的配对95%区间下界大于0。同时报告最强固定harness与选中harness的差距；基础门槛通过不表示优于强固定策略，也不表示发明了BM25或自由代码演化。

## 共享控制测量

为避免重复执行已完成的固定对照，标准起点实验复用强起点实验的固定控制测量。导入脚本检查模型/生成设置、数据、依赖锁、策略/评分实现和评估函数AST一致，并校验原始checkpoint哈希。只复用控制测量，不导入候选或测试结果。复用的观测不是新的独立样本；物理调用成本在来源账本中保留。

```sh
.venv/bin/python scripts/import_memory_baselines.py \
  --source runs/memory-s2d-expanded-v2 --source-commit de1adf0 \
  --output runs/memory-s2d-starter-v1 --config configs/memory_s2d_starter.json \
  --data data/memory-s2d-expanded
.venv/bin/python scripts/run_memory.py suite --parallel \
  --config configs/memory_s2d_starter.json --data data/memory-s2d-expanded \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18001/v1 http://127.0.0.1:18002/v1 \
  --output runs/memory-s2d-starter-v1
```

独立测试、消融和恢复检查只在两组搜索都冻结后执行；所有结果按真实证据填写。
