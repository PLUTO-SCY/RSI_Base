# v0.1 验证记录

日期：2026-11-16。版本范围：v0.0 任务执行/评分 + v0.1 固定权重的受限 prompt 改进。没有训练权重、启动本地 Qwen 或使用四卡训练。

## 环境与接入

- 项目 .venv：Python 3.11.15、OpenAI SDK 2.54.0；完整 Python 依赖由 uv.lock 固定。
- Claude Code：已有 CLI 2.1.169；没有进行全局安装、升级或配置修改。
- 给定网关的 /v1/chat/completions 与 /v1/messages 均实测成功。
- Claude Code 实际使用 deepseek-v4.1-flash，Task 通过 OpenAI SDK 使用同一模型服务。
- Task 响应返回的实际模型标识为 deepseek-v4.1-flash-1017。
- 监测的四个全局 Claude/shell 配置文件在 Meta 调用前后哈希一致。

密钥保存在忽略的 .env.local，权限 0600；Claude 配置状态和探测产物在忽略的 .local 中。程序不修改用户的 shell profile 或全局 Claude 设置。

## 离线检查

~~~sh
sh scripts/check.sh
~~~

17 项 pytest 检查通过，ruff 静态/格式检查通过；源码和暂存区另有不回显密钥的扫描。

覆盖：数值评分、禁止修改模型字段、不支持参数拒绝、开发/测试交叉去重、候选接受/退化回退/非法提案拒绝、完成运行恢复不新增调用、部分失败恢复、产物校验、调用上限、跨恢复的 wall deadline、最终评估额度保留、Claude 子进程与全局环境隔离。

测试使用 fake fixture，既不访问 API，也不依赖机器是否安装 Claude。

## 最终真实 pilot

输入：固定官方 GSM8K 源 commit 的训练池，seed 42，开发与留出各 6 题。数据来源和哈希见[机器可读摘要](validation/v0.1.json)。这不是官方完整测试集。

命令：

~~~sh
sh scripts/run_smoke.sh
RSI_RESUME=1 sh scripts/run_smoke.sh
.venv/bin/python scripts/export_validation.py
~~~

| 项目 | 实测 |
| --- | --- |
| 演化轮数 | 3 |
| 每轮子候选 | 1 |
| 基线开发准确率 | 6/6 |
| 三轮候选开发准确率 | 每轮 6/6 |
| 选择结果 | 全部平局，保留父代（round 0） |
| 冻结后的独立留出 | 6/6，完成覆盖 6/6 |
| Task API 应用层调用 | 30 |
| Meta CLI 调用 | 3 |
| Task input/output tokens | 5,531 / 5,056 |
| Meta input/output tokens | 4,083 / 1,964 |
| 已完成运行恢复 | 汇总完全一致，新增模型调用 0 |
| 权重更新 | 无 |

调用数包括一次开发基线、三个开发候选和一次最终留出评估；独立连通性探测另记，不含在上表循环账本中。CLI 估计美元金额不是第三方网关账单，本记录不声明已核对真实美元成本。

## 保留的初测结果与结论边界

早期联调运行 runs/deepseek-smoke 同样完成三轮并保留父代，其留出结果为 5/6；整理后的最终验证 runs/deepseek-v0.1 为 6/6。两次均保留原始配置、轨迹和记录，没有挑选题目重划分，也没有借助最终题目修改 Meta 提案。

两次结果不同不代表 RSI 获得收益：未选入任何新 harness，且外部推理不保证逐 token 确定性。该小 pilot 仅验证软件路径、选择策略与记录恢复，不能证明方法有效或达到某 benchmark 分数。

原始题目、回答、完整日志和凭据不提交 Git。公开摘要不包含这些内容；由 scripts/export_validation.py 从当前实现哈希匹配的真实运行导出，并再次核对恢复无新增调用。

## 后续尚未验证

本地 Qwen endpoint、GPU serving、LoRA SFT、交互工具 agent、第二任务、代码级 harness 修改和正式多 seed 的能力/效率实验均属于后续阶段。
