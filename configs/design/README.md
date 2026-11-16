# 设计配置

此目录的文件用于审阅字段与实验边界，**不能执行训练**。尚无配置 schema、CLI、adapter 或环境锁；字段在 P0 实施时可能调整。

four_gpu_rsft.json 表示一个 GSM8K 工具推理的四卡 RSFT/LoRA pilot。预算数字是设计上限，不是实测消耗或运行时间估计。model revision、Meta provider、依赖组合和数据 manifest 在 bring-up 时固定。

设计重点：

- 明确区分 model、harness、角色关系、训练任务采样和训练轨迹混合。
- 最终测试与搜索/训练集合独立。
- 生成、训练、评估和重试共同消费总预算。
- 服务身份和 checkpoint 校验先于下一阶段。
- 超出能力的配置应拒绝，不能静默降级成不同实验。

完整规则见[架构](../../docs/architecture.md)、[实验协议](../../docs/experiment-protocol.md)。
