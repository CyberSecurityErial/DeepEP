# DeepEP-LB 思想迁移取舍

本轮不复制 `/home/chen/workspace/infra/code/DeepEP-LB` 的 V1 kernel。我们只比较它的核心思想：用紧凑 quota 描述 Rail 目标，并在数据扫描时解析去向，避免生成完整的逐 copy 路由表。

两个实验实现共用当前 V2 Hybrid 的 proxy、ticket、combine 和 Gin 数据面：

- **LB-hard**：现有 `rail_balance="legacy_exact"`。它只看 `[owner, destination]`，允许任意 Rail，代表直接采用 DeepEP-LB 调度语义的性能上限对照。
- **LB-minimal**：`rail_balance="one_hop"` 或 `"adaptive"` 加正数 `rail_balance_threshold_percent`。先用紧凑 source-load 判断是否值得规划；未触发时保留原 Rail，触发后仍运行现有 endpoint-aware planner。

## 功能取舍表

| 能力 | 实际价值 | LB-hard | LB-minimal | 取舍结论 |
|---|---|---:|---:|---|
| 紧凑 count/quota | 高；避免逐 copy 保存完整决策 | 保留 | 只用于激活判断 | 两版都复用 |
| 目标端 local rank / target mask | 高；判断一跳还是两跳的唯一依据 | 丢失 | 保留 | hard 为速度对照而牺牲 |
| 精确 endpoint one-hop | 高；保证 `e` 属于源/目标端点 | 丢失 | 保留 | 生产候选不能轻易删除 |
| selective 2-hop | 条件性；只对一跳解不开的热点有用 | 丢失 | `adaptive` 保留 | 可选，不应进入普通流量热路径 |
| direct path bypass | 高；避免无意义节点内搬运 | 仅保留 legacy 数据面可识别部分 | 保留完整 hop 分类 | 保留 |
| 静态 proxy slot | 必需；避免热点 atomic 和 slot 冲突 | 保留 | 保留 | 属于 V2 正确性/性能底座，不能照搬 LB 时删除 |
| combine ticket 逆路由 | 必需；结果必须回到原 owner/token | 保留 | 保留 | 删除会破坏 round trip，不是可选算法功能 |
| multi-target payload 复用 | 高；避免同一 token hidden 被重复搬运 | legacy 粒度，不能表达精确 endpoint 约束 | 保留 bounded split | hard 接受语义损失作为实验变量 |
| two-hop threshold/cap/penalty | 条件性；控制第三 Rail 成本 | 丢失 | 保留 | 只在 `adaptive` 使用 |
| path 分布与 autotune 统计 | 中；不直接执行通信，但决定多机调优能否审计 | 降级为 legacy moves/rail load | 保留 | 不放入逐 token 热路径即可 |
| V2 Hybrid/Gin 与 fail-closed gate | 必需 | 保留 | 保留 | 两版都不改 |

结论：表中没有可以无条件认定为“没用”的功能。六项属于正确性或主要性能语义，三项只在 adaptive/调优时有条件价值；LB-hard 有意舍弃 endpoint-aware、selective 2-hop 和精细路径统计，目的是测出紧凑 quota 的上限，不是替代最终实现。

## 单变量比较

必须使用同一 commit、workload、拓扑、SM/QP、warmup 和 steady 次数，只改变以下一项：

1. `off`：原始 DeepEP V2。
2. `legacy_exact`：LB-hard。
3. `one_hop, threshold=0`：当前黄金语义。
4. `one_hop, threshold>0`：LB-minimal。
5. `adaptive, threshold>0`：LB-minimal 加 selective 2-hop。

性能真值是无 profiler 的 rank-max steady-state 延迟。Nsys 只用于区分 planner、source shuffle、scale-out 和 destination forward；只有 Nsys 证明某个 kernel 有暴露关键路径时间后才使用 NCU。
