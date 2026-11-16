# 开发进度

## 2026-11-16｜M0 调研与设计

### 已确认的需求

- 为 evaluator、RSI 效率和泛化研究提供可复用 backbone。
- Meta/Task Agent 架构，可改模型训练、harness、技能和数据配比。
- 单机四卡起步，不依赖大模型规模。
- 用户曾试跑 WHALE 并遇到问题，明确要求从零独立实现；论文仓库只作机制参考。
- 开发进度同步到 PLUTO-SCY/RSI_Base，保持代码结构清楚。

### 本轮已完成

- 扫描 18 项原始文献；重点核对五篇指定论文的方法/实验与对应代码片段。
- 记录相关代码快照、实际复现边界与本轮读取层级。
- 形成独立架构、接口、可演化对象、训练数据契约、状态恢复和预算设计。
- 制定四卡实验协议、对照/归因/泛化方案和三类研究入口。
- 制定 benchmark 优先级、分阶段开发计划及可验收 backlog。
- 提供明确标为 design-only 的配置示例和仓库协作规范。

### 尚未完成

- Python package、配置 schema、CLI、演化核心与适配器。
- 任何真实模型调用、SFT/GRPO 训练、GPU 闭环或 benchmark 运行。
- 论文结果复现或框架有效性的实验结论。
- 依赖版本锁定与本机显存/吞吐测量。

### 环境与同步记录（首次交付时）

- 远端检查：仓库为空，默认分支为 main；当前 GitHub 连接报告 pull=true、push=false。
- 本机 Git HTTPS 读取失败：Couldn't connect to server。
- 工作区 .git 元数据写入受限：git init 报 Read-only file system。
- 当前执行工具中 Python 为 3.8；nvidia-smi 不能与驱动通信，尚未验证宿主的四卡运行环境。
- 本地方案文件已落盘；远端写入结果如下。
- 对已准备好的 README 执行 GitHub Contents 写入，返回 403：Resource not accessible by integration。远端未创建本轮提交，不能标记为已同步。
- 已通过设计文档检查：13 个文件、15 个本地链接、Markdown 围栏配对、JSON 与关键配置不变量、18 条文献表记录；未发现凭据模式。这是文档验证，不是框架 CPU/GPU 测试。
- 交付采用本地文件与另附的 Git bundle；提交元数据保存在可写临时目录，避免修改受限的工作区 .git。

### 下一步

按 roadmap 的 M1/P0 开始核心协议、artifact、事件与预算状态机；并先在具备 GPU 可见性的真实运行上下文完成环境诊断。M0 文档完成不表示 M1 软件能力完成。

## 2026-11-16｜SSH 通道恢复与首次同步

- 用户将新建的专用公钥加入 GitHub 后，SSH 成功认证为 PLUTO-SCY。
- 工作区 Git 元数据已可写；从已验证的 Git bundle 恢复原始设计提交，保留其历史。
- 初始设计提交 [bd0c1a2](https://github.com/PLUTO-SCY/RSI_Base/commit/bd0c1a2fcc762ef61a8faa2a0e9685a3b2d53ecc) 已推送到 main；git ls-remote 核对远端提交 SHA 一致。
- 远端共有 13 个设计与协作文件，包括调研、架构、实验协议、开发计划和 design-only 配置。
- 仓库的 origin 使用 SSH，并在仓库局部配置中指定专用密钥，供后续进度同步使用。
- 项目仍处于 M0 设计交付阶段；本次同步未新增训练运行时，也未运行模型训练。

## 状态用语

- 设计完成：已形成可审阅的协议/方案。
- 已实现：存在对应可执行代码。
- CPU 验证：在无 GPU 环境中通过明确检查。
- GPU 验证：真实权重更新、导出、重载与执行通过。
- Benchmark 验证：固定协议下与任务评分器一致。
- 已同步：远端接受提交并核对 SHA。

只有达到相应证据层级才更新状态。
