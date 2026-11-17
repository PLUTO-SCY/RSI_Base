# 历史 v0.0 / v0.1 数学 pilot 指南

本页保留早期单次prompt数学实验；当前记忆harness的环境与复用入口见[新上手指南](getting-started.md)。

当前实现的范围是：固定权重、单任务、单候选、串行 prompt 改进。v0.2 的训练与更多任务按 roadmap 后续实现。

## 1. 项目环境

从仓库根目录运行：

~~~sh
sh scripts/setup.sh
~~~

脚本通过 uv 和 uv.lock 创建/同步 .venv，不使用系统 pip 安装依赖，不修改 shell 启动文件。需要已有 uv；项目 .python-version 固定 Python 版本，uv 可在缺失时下载所需解释器。

Meta 需要已有 Claude Code CLI，本机验证为 2.1.169，且已核对 bare、print、model、tools、JSON 输出等参数。当前脚本不安装或升级全局 Claude Code。其他版本先检查兼容性。

所有后续 Python 命令均使用 .venv/bin/python，不要求 activate。

## 2. 仅配置本项目凭据

~~~sh
.venv/bin/python scripts/configure_credentials.py
~~~

按提示输入 API key，不会回显。生成 .env.local，并设为 0600；文件已被 .gitignore 排除。脚本不覆盖已有文件。模板见 .env.example，勿在模板中写真实密钥。

RSI_API_KEY 用于给定的 DeepSeek 服务。也可通过同名环境变量提供，环境变量优先。无需修改 ANTHROPIC_BASE_URL、全局 Claude 设置或 shell profile。

若使用本地无鉴权 Task 服务，可以在本地 .env.local 中另设 RSI_LOCAL_API_KEY=EMPTY；configs/local_qwen.json 仅是接入模板，尚未在本机 Qwen 服务上验证。

## 3. 验证网关和 Claude Code

以下会产生少量真实调用：

~~~sh
.venv/bin/python scripts/probe_api.py
.venv/bin/python scripts/probe_claude.py
~~~

前者分别检查 OpenAI Chat Completions 与 Anthropic Messages；后者实际运行一条 Claude Code 提案。

本次网关同时支持两种协议：

- Task 的 OpenAI base_url 为 https://zgc.apihy.com/v1。
- Claude Code 的 ANTHROPIC_BASE_URL 为 https://zgc.apihy.com，客户端请求 /v1/messages。
- 两者的路由差异保留在配置中，无需协议代理或机器级配置。

Claude 子进程使用项目内独立 CLAUDE_CONFIG_DIR、显式 provider/model 环境、bare 模式、空工具集与空 MCP 配置。输入来自宿主提供的开发侧记录，输出须通过 Proposal 字段校验。

监测 .claude.json、.claude/settings.json、.bashrc、.profile 的内容哈希；检测到变化则停止。该检查是指定文件的变化检查，不是对整台机器文件系统的审计。

## 4. 准备小规模 GSM8K pilot

~~~sh
.venv/bin/python scripts/fetch_gsm8k.py \
  --revision 3101c7d5072418e28b9008a6636bde82a006892c \
  --dev-size 6 --test-size 6
~~~

脚本固定官方源 commit，按 seed 42 对训练池题目去重/打乱，分出互不重叠的开发与留出样本。保存来源、原始数据哈希、split 哈希和样本 ID。

这里的 test 是训练池中的独立留出 pilot，**不是官方完整测试集**。默认数据下载到忽略的 data/gsm8k。输出已存在时拒绝覆盖，重新准备需选新的 --output 目录。

评分协议：解析最后一个 #### 后的数值，或纯数值/单个 boxed 数值，按有理数比较，支持负数、小数、千分位和分数；其他格式计 format_error。它是当前固定数值评分规则，不宣称等同所有官方数学 benchmark 评分。

## 5. 跑基线与三轮循环

只跑固定 Task 基线：

~~~sh
.venv/bin/rsi baseline --config configs/deepseek_smoke.json \
  --dev data/gsm8k/dev.jsonl --test data/gsm8k/test.jsonl \
  --output runs/baseline
~~~

跑三轮：

~~~sh
sh scripts/run_smoke.sh
~~~

默认输出 runs/deepseek-v0.1。为保留历史，同名目录拒绝直接覆盖。可以指定新目录，或显式恢复：

~~~sh
sh scripts/run_smoke.sh runs/my-run
RSI_RESUME=1 sh scripts/run_smoke.sh
~~~

恢复要求配置、数据、实现与依赖锁哈希一致，并校验完成轮次的文件哈希。逐题结果可继续追加；不重做已记录的题目和提案。恢复不是逐 token 确定性复现，也不承诺网络请求 exactly-once。

## 6. 读结果

- run.json：配置、数据/实现/依赖锁哈希与固定选择协议。
- calls.jsonl：task API 调用与 Claude 调用的开始/结束、用量、错误、未确认状态。
- rounds/000：父代基线。
- rounds/001 等：提案、Meta 输入与运行参数、逐题执行、选择理由、产物校验。
- final：冻结选择之后的独立评估，不进入 Meta 输入。
- summary.json：轮数、选择版本、分数、实际应用层调用次数。

v0.1 只开放 system_prompt 字段，reason 用于解释；其他字段不能通过提案修改。Task 模型、temperature、输出上限和 scorer 保持固定。

## 7. 预算与已知边界

调用开始前记账，失败调用也消耗额度。开发评估不能消费预留给最终评估的 Task 调用数；恢复不清零调用次数。

wall_seconds 从首次运行开始计算，包含暂停时间；到期后不启动新调用，已经启动的调用由各自 timeout 控制。未确认的中断请求保留已消费调用记录，之后的新尝试另计。

Meta 的上限是 Claude CLI 调用次数；CLI 内部可能有服务重试，不能将它理解为上游 HTTP 请求数或真实美元硬上限。保存的 CLI cost 是 CLI 估计，对第三方模型不作为网关账单；主要依据 provider token usage 和外部账单核对成本。

第一版尚无：权重训练、工具 agent、任意代码修改、任务种群、完整原子网络事务、自动依赖升级或四卡调度。

## 8. 离线检查

~~~sh
sh scripts/check.sh
~~~

此命令运行格式/静态检查和 pytest，不下载数据、不访问 API，也不需要 Claude 登录。测试中的模型与任务均为显式 fake fixture；真实验证单独记录。
