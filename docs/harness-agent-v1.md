# 可读源码、可调试候选的 harness 演化

本版针对[旧搜索诊断](harness-search-diagnosis.md)修复优化端。Task仍为固定权重Qwen3-4B-Instruct-2507，不训练LoRA。接上工具不等于证明收益，实验结论单独报告。

## Meta的能力与边界

Claude Code由项目局部DeepSeek API配置驱动，新增五个宿主MCP工具：

| 工具 | 能力 |
| --- | --- |
| `list_artifacts` | 列出完整源码、训练记忆、开发题、失败编号、历史候选和固定对照 |
| `read_artifact` | 分页读取已登记的完整文件 |
| `get_case` | 指定开发题的gold、实际模型输入、检索ID及各步输出，支持父代和历史候选 |
| `evaluate_candidate` | 试跑配置或Python检索代码，返回逐题得失与完整轨迹 |
| `submit_candidates` | 提交最多两个本轮已试跑的候选，由宿主完整验证 |

每轮四次有效试跑，错误修正/重复候选最多八次尝试；40次工具调用、CLI内部最多40轮、600秒超时。固定24题分层试跑批次由seed和轮次确定，Meta与随机搜索使用相同批次。候选接受仍需300题×3次验证，各次均不退步且平均净增至少一题。

**协议变化：Meta现在能够看到全部300题开发集的题目、gold与轨迹。** 这300题同时用于优化和选择，不能再当作优化过程未见的验证集。最终291题、答案、密钥和宿主个人目录不在工具可读范围内。

除已有配置外，候选可设置`retrieval="python"`与`retrieval_code`：

```python
def rank(query, memory):
    # memory仅包含训练行的id/question/answer
    return [row["id"] for row in memory]
```

也可返回带有限`score`的ID字典列表。宿主核验ID只能来自允许的训练记忆且不得重复，再统一应用示例数量、布局、输出合同和评分。禁止嵌入开发题、ID或逐题答案表。代码编辑范围目前是检索/重排函数，尚未开放任意修改执行器、评分器或工作流源码。

代码在无网络、无宿主home、无密钥和GPU的独立Linux namespace内运行，仅提供Python标准库；单批512MiB内存、30秒CPU、45秒宿主超时。完成批次记录CPU/墙钟与内存峰值；冻结摘要明确该统计只覆盖当前进程，不含失败/中断批次，不能当作完整机器级成本。

## 去重、对照与证据

- 规范化对比布局7/8例等无效字段组合；单步候选比较全部开发请求，输入完全相同就跳过Task调用。
- 跨候选按实际模型请求复用结果；不同repeat分别保存。两步流程按实际逐步请求缓存，不把首步相同误判成整条工作流相同。
- 完整源码、实际输入、父子得失和失败子代轨迹均可回查。
- 从十个固定对照中开发分数最高者起跑，包含旧随机搜索得到的三个强配置，避免从弱起点制造增长。
- 随机搜索有相同轮数、试跑数和完整验证数上限，并包含通用Python检索模板。但随机模板空间小于Meta任意编写检索代码的空间，实际调用/token成本也不同，不能称严格等算力或全空间均匀随机。
- 保存工具调用、候选代码、逐题轨迹、决策、物理API调用账本与源码/数据/依赖哈希。评估summary中的`model_calls`为逻辑工作流调用数；实际新增请求以`calls.jsonl`为准。

## 安装与复现

完整环境说明见[上手指南](getting-started.md)。本模式另需支持非特权user namespace的Linux。x86_64脚本通过固定URL和SHA256提取bubblewrap到项目`.local/`，需要系统已有`dpkg-deb`；不安装系统包、不启动Docker daemon。

```sh
sh scripts/harness.sh setup --gpu --claude --sandbox
sh scripts/harness.sh credentials
sh scripts/harness.sh data
sh scripts/harness.sh model
# 另开终端启动服务
RSI_GPU=0 RSI_TASK_PORT=18000 sh scripts/harness.sh serve
# 服务就绪后
sh scripts/harness.sh doctor --ready --config configs/memory_s2d_agent.json
sh scripts/harness.sh run suite --config configs/memory_s2d_agent.json \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18000/v1 http://127.0.0.1:18000/v1 \
  --output runs/my-agent-suite
sh scripts/harness.sh run finalize --config configs/memory_s2d_agent.json \
  --output runs/my-agent-suite
```

三卡并行时另启动GPU1/18001与GPU2/18002，suite增加`--parallel`并替换对应端口，finalize也增加`--parallel`。全部搜索冻结后才回放测试。改代码/配置要使用新run；恢复需要完全相同的源文件和协议。旧协议仍可复现，省略`--config`时入口仍运行旧starter协议；本模式须显式指定新配置。

## 本次实验状态

72项离线测试通过；本机namespace隔离、候选返回值、跨候选缓存、repeat隔离和冻结恢复均有覆盖。真实DeepSeek/Claude Code预检完成18个内部轮次、17次宿主工具调用、两次Python检索试跑及提交；宿主232次Task请求、1次Meta CLI调用，未打开测试。8题的小预检未产生增长，仅用于验证闭环，不作为性能证据。首次预检因CLI轮次限制未完成提交，记录保留，随后增加限制并修正阅读/试跑顺序。

正式三次搜索尚未填入结果。291题在旧实验中已经使用，后续只能称**历史回放**，不能重新宣称全新独立验收。演化方法须与强固定/随机对照比较，再另行封存新任务或数据确认泛化。
