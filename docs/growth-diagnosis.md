# 无增长诊断与第二次实验

## 原实验为什么无法增长

`runs/deepseek-v0.1` 的父代和三个候选在同一开发集均为 6/6。严格大于父代才接受，因此没有任何候选能在该协议下被接受。加轮数不能突破满分上限。

逐题检查显示：三个候选 prompt 确实改变；所有 Task 回答正常结束，没有空回答或 token 截断；解析答案与父代相同。执行闭环和保留父代的行为符合实现。原来的 6/6 留出结果只能验证接入，不能证明改进有效。

框架也存在反馈不足：原先只给 Meta 前八条父代记录，没有优先挑错题，没有开发题标准答案，也没有被拒绝候选的具体回答。这使后续提案容易重复泛泛的“再检查一次”。已补齐这些反馈，但评分、模型和生成预算仍固定，最终题目不进入 Meta。

## 固定协议

- Meta：Claude Code 2.1.169，底层 DeepSeek API；无模型工具，项目隔离状态。
- Task：[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`；BF16 单卡推理，权重不训练。
- 本机为 4 × A100-SXM4-80GB；本次只使用 GPU 0，其他卡不需要占用。
- [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 固定 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。固定规则筛选 level ≥ 3、无 Asymptote 图、整数/小数/整数分数答案，共 255 题。随机种子 42 后固定划分 calibration 16、dev 48、monitor 32、final 64；余下 95 题不用。
- 这是 MATH-500 数值子集的重新划分实验，不能当作标准 MATH-500 成绩，也不能视为模型预训练未见过这些题。
- calibration 仅检查难度及截断；Meta 只看 dev。六轮、每轮单候选，严格提高 dev accuracy 才接受。
- 所有候选搜索结束后才补评 monitor 曲线，不用 monitor 选择候选。最终选中的 prompt 由 dev 决定；随后 baseline 和选中版本各运行一次 final。
- Task temperature=0、最多 8192 输出 token，所有轮次相同；并发 16。任务仍是一次生成，不增加工具或隐式重试。
- dev 上选中版本的单调上升是筛选规则的直接结果；是否泛化要看 monitor 和 final。报告原始候选曲线、被拒绝候选、逐题得失与 final 配对 bootstrap 区间。

校准先测试了 4096 输出 token：11/16，其中 4 题截断。随后在搜索前统一提升至 8192；4096 校准原始记录保存在 `runs/qwen-math-growth/calibration/`，配置见 `configs/qwen_math_calibration_4k.json`。这次预算调整不计作 RSI 增益。

## 复现

```sh
sh scripts/setup.sh
sh scripts/setup_task.sh
.venv-task/bin/python scripts/download_task.py
.venv/bin/python scripts/prepare_growth_data.py
sh scripts/serve_task.sh
```

保持服务运行，在另一个终端执行：

```sh
RSI_LOCAL_API_KEY=EMPTY .venv/bin/python scripts/run_growth.py calibrate
RSI_LOCAL_API_KEY=EMPTY .venv/bin/python scripts/run_growth.py search
RSI_LOCAL_API_KEY=EMPTY .venv/bin/python scripts/run_growth.py audit
```

`EMPTY` 仅用于本机无鉴权 loopback endpoint。DeepSeek 使用现有 `.env.local`，不写入命令或 Git。第二个 Python 环境 `.venv-task` 与依赖锁 `requirements/task.lock` 专用于 vLLM，不改变系统环境。

产物位于 `runs/qwen-math-growth-8k/`：逐题原始结果、候选 prompt、反馈、调用记录、校验和及最终图。数据、模型和完整产物继续留在忽略目录。准备数据脚本拒绝覆盖已有划分；已完成执行可恢复，输入变化会被拒绝。

## 结果

实验结果完成后由真实产物导出到 `docs/validation/growth.json` 和本目录增长图；未运行的数值不预填。
