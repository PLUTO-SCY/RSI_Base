# RSI Base

用于研究和复用的最小 RSI 实验框架。当前版本实现固定模型权重下的 harness 演化：Meta Agent 查看运行证据、修改 harness、调用评测工具，宿主执行候选、评分并选择下一轮版本。

- **Meta Agent**：Claude Code 作为工具调用运行时，通过兼容 Anthropic 的网关接入 DeepSeek API；状态和配置限于项目目录。
- **Task Agent**：Qwen3-4B-Instruct-2507，通过 vLLM 的 OpenAI 兼容接口推理；也可替换为已有服务。
- **Harness**：支持 prompt、检索算法、示例组织和单步/复核流程的演化，也允许 Meta 编写受隔离执行的 Python 检索函数。
- **实验机制**：配置与数据校验、调用预算、候选记录、开发集选择、冻结后测试、断点恢复和曲线导出。

默认初始 harness 使用一次模型调用，根据任务描述和标签输出分类结果。演化过程中可以引入训练记忆检索和复核。当前只做 harness 演化，模型权重保持固定。

## 开始使用

在 Linux 仓库根目录运行。需要 `curl`、Git、Node.js ≥18/npm；本地部署 Task 模型还需要 NVIDIA 驱动和支持 BF16 的 GPU。

```sh
sh scripts/harness.sh setup --gpu --claude --sandbox
sh scripts/harness.sh credentials
sh scripts/harness.sh data
sh scripts/harness.sh model
sh scripts/harness.sh serve
```

保持服务运行，在另一个终端执行：

```sh
sh scripts/harness.sh doctor --ready --local-gpu
sh scripts/harness.sh smoke
sh scripts/harness.sh run search
sh scripts/harness.sh inspect
```

`smoke` 只请求两次 Task，用于检查连接和输出格式。`run search` 才启动真实 Meta/Task 演化，使用 [configs/default.json](configs/default.json)；默认 5 轮，输出到 `runs/demo/`。相同命令可以恢复运行，已完成的请求会被复用。三次独立搜索、冻结后测试和曲线导出见[使用文档](docs/getting-started.md)。

## 仓库结构

```text
configs/default.json      当前默认配置
src/rsi_base/             Agent、harness、评测、搜索、隔离执行与记录
scripts/                  安装、数据准备、启动、运行、检查与导出
requirements/             GPU/CLI/隔离工具及数据来源的版本锁
tests/                   离线契约测试
docs/                    使用说明与实现结构
```

运行日志、候选、图表、数据集、模型和本地环境都写入被 Git 忽略的目录。仓库保留当前代码、脚本、配置、测试和使用文档，不收录实验迭代档案。

默认示例是 Symptom2Disease 的 22 类文本分类，固定划分为 200 条训练记忆、300 条开发样本和 291 条测试样本。数据由脚本下载并校验，来源及处理规则见 [数据锁文件](requirements/data/s2d.json)。此划分曾用于开发，默认标为历史回放；新研究需要自行准备未参与开发的测试集。

更多说明：[环境与运行](docs/getting-started.md) · [实现结构](docs/architecture.md) · [开发约定](CONTRIBUTING.md)
