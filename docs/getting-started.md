# 从全新 clone 到首次 harness 实验

当前主入口是`sh scripts/harness.sh`。本指南不需要作者的环境、历史run、模型缓存或API key。先完成两题连接检查，再显式启动完整实验；默认不训练权重。实验结论与对照见[harness报告](harness-backbone.md)，旧数学pilot见[历史指南](legacy-math-quickstart.md)。

## 1. 机器与环境

已验证平台：Linux x86_64、A100-SXM4-80GB、NVIDIA驱动580.65.06。单卡即可按本指南串行运行；三卡用于并行三次搜索，四卡不是启动条件。其他显卡的显存与兼容性尚需实测，80GB是已验证配置，不是声称的最低显存。

| 部分 | 固定版本/位置 | 创建方式 |
| --- | --- | --- |
| 宿主工具 | Git、curl；本地npm安装CLI时还需Node.js≥18和npm | 机器预先提供；CLI安装已实测Node20.20.2 |
| Python管理 | uv0.11.17、Python3.11.15 | 缺失的uv与Python下载到项目`.local/` |
| 核心环境 | `.venv/`，精确依赖见`uv.lock` | `setup`，CPU检查也使用此环境 |
| 本地推理环境 | `.venv-task/`，vLLM0.11.0、PyTorch2.8.0/CUDA12.8 | `setup --gpu`，精确依赖见`requirements/task.lock` |
| Meta CLI | Claude Code2.1.169 | `setup --claude`安装到`.local/claude-cli/`，含npm依赖锁 |
| Task权重 | Qwen3-4B-Instruct-2507固定revision | `model`，位置`.local/models/qwen3-4b-instruct-2507/` |

NVIDIA驱动由宿主提供，脚本不安装系统驱动或系统Python包。核心环境与GPU环境分开，避免vLLM/PyTorch约束影响实验编排。GPU平台要求参考[vLLM0.11安装文档](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html)。

uv使用官方固定版本安装器和项目安装目录，不修改shell profile，见[uv安装选项](https://docs.astral.sh/uv/reference/installer/)。CLI的npm包与平台依赖固定为2.1.169，Node要求来自[该版本包元数据](https://registry.npmjs.org/@anthropic-ai/claude-code/2.1.169)；新版本安装方式可能不同，参考[Claude Code官方说明](https://code.claude.com/docs/en/setup)。

首次安装和下载需要访问Python包源、GitHub、Hugging Face，以及可选的npm源。GPU环境和权重较大，请预留磁盘空间。若已持有兼容的Claude CLI，可省略`--claude`；只做CPU检查或使用已有Task服务时，可省略`--gpu`。

## 2. Clone与安装

```sh
git clone https://github.com/PLUTO-SCY/RSI_Base.git
cd RSI_Base
sh scripts/harness.sh setup --gpu --claude
sh scripts/harness.sh check
```

`setup`本身只安装核心环境；两个可选参数分别添加GPU环境和项目内CLI。入口脚本也可使用绝对路径从其他工作目录调用，会自行定位仓库根目录。不需要activate，不需要sudo，不修改全局Claude配置或`.bashrc`。Python、uv、npm缓存也放在`.local/`；脚本保留在`scripts/`，依赖锁随Git提交。

## 3. 凭据与数据

```sh
sh scripts/harness.sh credentials
sh scripts/harness.sh data
sh scripts/harness.sh model
```

凭据命令隐藏输入，生成Git忽略的`.env.local`，权限0600；已有文件不会被覆盖。示例中不包含真实密钥。默认Meta为DeepSeek经已配置网关接入Claude Code，需使用自己的网关凭据。

`data`依次重建原始200/50/211数据、591条去重上游池、发布的200/300/291划分，并校验manifest与各数据文件哈希。已有的正确数据直接复用；不完整或修改过的目录会报错，不自动删除或覆盖。GitHub公共API限流时使用固定commit的raw文件，并继续校验相同Git blob哈希。整个准备过程不调用模型、不读取历史run。

`model`按以下固定快照下载，Hugging Face可复用已完成的下载文件：

```text
Qwen/Qwen3-4B-Instruct-2507
revision: cdbee75f17c01a7cc42f958dc650907174af0554
```

重建发布划分只能复现既有实验协议，不能把已公布的291题重新宣称为新发现的未见测试。

## 4. 单卡启动与两题检查

终端一，前台启动服务：

```sh
RSI_GPU=0 RSI_TASK_PORT=18000 sh scripts/harness.sh serve
```

终端二，等服务完成加载后执行：

```sh
sh scripts/harness.sh doctor --ready --local-gpu
sh scripts/harness.sh smoke
```

`doctor`检查Python/包版本；`--ready`增加CLI参数兼容性、凭据是否存在、数据完整性与Task的`GET /models`，`--local-gpu`增加CUDA/BF16及权重分片检查。它不生成回答、不调用Meta API，也不验证凭据的远端有效性；不会打印密钥。

`smoke`仅从选择集取两题，用生产harness检索训练记忆并调用Task两次，检查标签输出合同。它不演化、不打开最终测试、不应解释为准确率结果。若要验证Meta的真实接入，显式增加一次CLI调用：

```sh
sh scripts/harness.sh smoke --meta --output runs/harness-smoke-meta
```

这个可选命令会消费自己的Meta API用量。CLI内部可能重试，不能把一次CLI调用当成上游HTTP请求的绝对上限。smoke记录在`runs/harness-smoke*/`；重复检查请用新的`--output`，不会覆盖已有记录。

## 5. 完整实验：单卡串行或三卡并行

这是显式的正式运行：3次搜索，每次5轮×2候选，每个候选在300题上重复3次；另有七个固定控制与同候选数随机搜索。Meta最多15次CLI提案，Task评估远多于smoke，请依据`calls.jsonl`核对实际成本。

单卡服务继续运行，三个seed顺序执行：

```sh
sh scripts/harness.sh run suite \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18000/v1 http://127.0.0.1:18000/v1 \
  --output runs/my-memory-suite
sh scripts/harness.sh run finalize --output runs/my-memory-suite
```

若有三张可用卡，另外两个终端分别启动：

```sh
RSI_GPU=1 RSI_TASK_PORT=18001 sh scripts/harness.sh serve
RSI_GPU=2 RSI_TASK_PORT=18002 sh scripts/harness.sh serve
```

随后将完整实验换为以下命令；选择与数据协议相同，改变的是服务映射和调度：

```sh
sh scripts/harness.sh run suite --parallel \
  --task-base-urls http://127.0.0.1:18000/v1 http://127.0.0.1:18001/v1 http://127.0.0.1:18002/v1 \
  --output runs/my-memory-suite
sh scripts/harness.sh run finalize --parallel --output runs/my-memory-suite
```

两种模式选择其一，不能对同一run修改端口映射。`finalize`必须等全部搜索冻结；只复用框架不要求先运行作者的强起点对照，也不要求导入历史固定控制。没有缓存时会自动评估固定策略。

## 6. 看进度、保存指纹与导出

```sh
sh scripts/harness.sh inspect runs/my-memory-suite
sh scripts/harness.sh runtime --gpu 0 --port 18000 --output runs/my-memory-suite/task-runtime.json
sh scripts/harness.sh report --suite runs/my-memory-suite \
  --runtime runs/my-memory-suite/task-runtime.json \
  --output-dir runs/my-memory-suite/public
```

导出JSON/PNG/SVG到自己的run，避免覆盖仓库中已公开的实验报告。三卡运行时可额外分别记录`--gpu 1 --port 18001 --output runs/my-memory-suite/task-runtime-expanded-123.json`、`--gpu 2 --port 18002 --output runs/my-memory-suite/task-runtime-expanded-2026.json`，GPU0同理记录seed42；若三份齐全，导出器会核对各卡权重指纹一致。

记录包括模型快照、依赖锁、源码哈希、冻结策略、曲线、逐题与调用checkpoint。原始数据、权重、凭据和完整轨迹均在Git忽略目录；公开聚合结果可另行选择提交。不要提交`.env.local`。

实验前先提交代码；运行后变更源码、依赖、数据或配置会拒绝恢复。中断后重用完全相同的命令和输出目录继续；恢复历史run应检出其对应源码，或使用匹配版本的独立worktree。预算从最初启动计算，恢复不清零。详见[实验协议](experiment-protocol.md)。

完成后在各服务终端按Ctrl-C释放GPU，保留环境和权重缓存供下次使用。

## 7. 替换自己的模型或服务

复制`configs/memory_s2d_starter.json`到自己的配置路径，修改`task.model/base_url/api_key_env`、`meta.model/base_url/api_key_env/command`。运行、doctor和smoke都支持`--config`。若使用自带Task服务，可跳过GPU环境、模型下载和本地serve；导出可核验报告时需服务端提供相应runtime/权重指纹JSON。

Task端点需兼容OpenAI Chat Completions。当前Claude Code Meta端点需兼容Anthropic Messages；只有OpenAI协议的服务不能直接填入Meta的`base_url`。默认网关的Task路由带`/v1`，Meta配置使用网关根路径；不要修改全局`ANTHROPIC_*`变量。

默认starter协议的编辑空间是结构化记忆/上下文/流程配置。新增可读源码、完整开发反馈和沙箱Python检索代码的模式，见[工具化演化指南](harness-agent-v1.md)，需额外`setup --sandbox`并显式指定`configs/memory_s2d_agent.json`。任意工作流源码编辑、新benchmark adapter与LoRA训练仍后置。

## 8. 常见问题

| 现象 | 处理 |
| --- | --- |
| 没有uv | `setup`自动安装固定版本到项目；需curl和可用网络 |
| 没有Claude CLI或参数不匹配 | 安装Node/npm后运行`setup --claude`，或在配置中指定兼容的`meta.command` |
| 没有CUDA/BF16或驱动不可见 | 检查宿主驱动与GPU；脚本不会修改系统驱动 |
| 本地服务尚未启动、模型名不匹配 | 在终端一启动serve，检查端口及`task.model`，再运行doctor |
| 数据校验失败 | 保留现场，检查被指出的目录；不要修改发布manifest来绕过校验 |
| Meta 401/403或协议报错 | 检查自己的凭据和Anthropic协议路由；用显式`smoke --meta`验证 |
| 不能恢复已有run | 核对源码/依赖/配置/数据哈希；新协议使用新输出目录 |
| 想先做无GPU、无API检查 | `sh scripts/harness.sh setup`后运行`check`和`doctor`，不用安装GPU或CLI |

## 9. 本次实际验证范围

2026-11-17，在无凭据、无项目环境/缓存/数据/权重的隔离checkout中，使用精简PATH验证了uv/Python自动安装、核心依赖安装、公开数据重建及第二次幂等校验；主仓库与隔离checkout各通过66项离线测试。

本机另外验证了项目内CLI安装、既有GPU环境的锁定依赖同步、CUDA/BF16、权重分片、真实Qwen的两题Task smoke，以及显式runtime/自定义目录的报告导出。该smoke为2次Task、0次Meta调用，不打开最终测试。临时GPU服务已停止。没有在隔离checkout重复下载完整GPU依赖和权重，也没有重复完整搜索或训练。

机器可读的版本、文件哈希与检查范围见[reuse-setup.json](validation/reuse-setup.json)。封存实验的核心源码/依赖锁保持原字节，历史实验报告未被本次安装检查覆盖。
