# RailBalance 论文实验对比矩阵（2026-08-01）

> 状态：预注册式实验设计，所有可运行实验的结果均为 `NOT_RUN`。
> `NO_PUBLIC_ARTIFACT`/`API_GAP` 只是截止日源码证据状态，不是运行结果。本文不包含
> 本机性能结果，也不把论文、README 或作者图中的数字改写成我们的结果。
>
> 证据基础：[`COMPETITOR_EVIDENCE_2026-08-01.md`](./COMPETITOR_EVIDENCE_2026-08-01.md)。本文只使用该账本中已经固定的作者论文、官方仓库、commit、公开入口和已确认缺口；缺失的命令、硬件、raw data 和计时 lineage 保持缺失，不补猜。
>
> 范围：设计 transport-only / placement-enabled、single-node / multi-node 四条赛道；不运行 GPU，不宣称已经打平或超过任何方案。

## 1. 实验要回答的问题

RailBalance 的核心假设是：在不改变 token 的最终专家语义、payload 和目标端点集合的前提下，允许发送端选择较空闲的 NIC/Rail，并在必要时增加一次受限的节点内转发，可以降低最拥塞 Rail 的负载和通信尾延迟。

论文实验必须分别回答四个问题，不能用一个端到端数字替代：

1. **传输收益**：固定物理目标端点后，RailBalance 是否比同树 DeepEP v2 `off` 和独立上游 DeepEP v2 更快？
2. **直接竞品**：在相同 GPU、NIC、trace、dtype、API 边界和资源预算下，RailBalance
   与 NCCL EP、SABRE、SwiftEP、UCCL-EP 和 fabric-lib 的差距来自哪里？
3. **适用边界**：收益是否只出现在人为 Rail 热点，还是在均衡、随机和真实 trace 上仍有可解释的收益/低开销？
4. **与整层/负载机制的关系**：UniEP、FEPLB、UltraEP、ECHO、EPLB、LPLB 等融合
   通信计算或改变专家副本/token位置后，RailBalance 的主效应和交互是什么？

截至证据账本截止日，已审计的 MoonEP公开入口只有单节点 8×H20 EP8，因此只回答单机
布局/通信开销问题，不进入多节点 Rail主结论。

## 2. 三类结果必须分开发布

| 轨道 | 目的 | 允许改变的内容 | 可以下的结论 |
|---|---|---|---|
| `AUTHOR_REPRO` | 复现作者公开 artifact | 严格使用作者代码、公开 shape、API 边界、计时方式和可获得硬件；只补记录，不替换定义 | “在我们的某硬件上复现了作者公开入口”；硬件不一致时只能称 port/re-run |
| `COMMON_FAIR` | 同机同 trace 横向比较 | 统一 correctness、输入、计时、rank 聚合、重复和资源预算；允许写薄 adapter，但不得改变算法语义 | “在冻结的共同口径下 A 比 B 快/慢多少” |
| `COMPOSED` | 评估 placement × Rail 的组合关系 | 固定逻辑 router trace 和副本预算，分别开关 placement 与 Rail | 两层的主效应、交互和额外成本 |

作者论文/README 数字只放在“作者报告结果”背景表中，并标明硬件和口径；它们不得与本地柱状图相连、不得参与 speedup、回归或显著性检验。`AUTHOR_REPRO` 和 `COMMON_FAIR` 产生独立 artifact 目录与图表。

## 3. 对手角色和优先级

| 优先级 | 系统（冻结版本） | 系统角色 | 主赛道 | 进入主性能图的前提 | 当前状态 |
|---:|---|---|---|---|---|
| P0-R | 同一 RailBalance 工作树的 DeepEP v2 `off` | 控制变量最少的首要因果基线；除 Rail 策略外二进制/JIT环境相同 | transport-only，单机/多节点 | 同 build、同 trace、成对 ABBA/BAAB | `NOT_RUN` |
| P0-R | DeepEP v2 `dd758caf...` | 干净上游可复现锚点 | transport-only，单机/多节点 | 独立 clean worktree、共同 adapter、共同 API | `NOT_RUN` |
| P0-R | NCCL EP `63cf786...` | NCCL Device API 原生 LL/HT | transport-only，多节点 | 论文LL与later v0.1分开；共同 public API rank-max，不混作者 host/Kineto 边界 | `NOT_RUN` |
| P0-R | SABRE `b3f2c5a...` | 完整通信矩阵驱动的 proxy/NIC 分流；与节点内转发有明确算法重叠 | AllToAllv，多节点 | 独立严格reference；能建立相同packing/API才进共同图，否则`NO_COMMON_API` | `NOT_RUN` |
| P0-R | SwiftEP branch `4bd6a8b...` / PR `6db603e...` | DeepEP buffer fusion、SGL、多QP、TMA | transport-only/组合，多节点 | 冻结真实代码lineage、fused buffer、SM/QP/NIC及API边界 | `NOT_RUN` |
| P0-R | UCCL-EP `61ee4240...` | 直接传输竞品，含 CPU proxy 与 QP/NIC 分配 | transport-only，多节点为主 | 同 GPU/NIC、同物理目标 trace、相同 SM/QP 预算，并披露 proxy CPU 资源 | `NOT_RUN` |
| P0-R | fabric-lib `2446003...` | CPU proxy、多NIC sharding/rotation | transport-only，多节点 | decode/prefill分开；固定proxy cores/NUMA/NIC并改用rank-max | `NOT_RUN` |
| P0-E | UBEP（无代码） | CM384一/两跳 AIV token 调度；与 hop-aware scheduling有明确 novelty重叠 | evidence/related work | 只做设计差异；不可把Ascend作者数放入NVIDIA性能图 | `NO_PUBLIC_ARTIFACT` |
| P1 | UniEP `1512a81...` | 冻结`ep_overlap`的单机前向通信+GEMM融合段 | integrated，single-node | `EP=2/4/8`公开前向测试已有Torch/CUDA reference；完整训练FWD/BWD单独记`API_GAP` | `NOT_RUN` |
| P1 | FEPLB（无代码） | Copy Engine动态专家/计算负载平衡 | placement-enabled，端到端 | 代码发布前只保留证据；不得与dispatch-only相除 | `NO_PUBLIC_ARTIFACT` |
| P1 | UltraEP `94cab099...` | 精确负载驱动副本与重路由；作用层不同，组合/交互待验证 | placement-enabled，多节点为主；8 Hopper 可先做功能/脚手架 | 形成 placement-only / Rail-only / both 的 2×2 因子实验 | `NOT_RUN` |
| P1 | MoonEP `0f385f03...` | 单机动态冗余专家、VMM/zero-copy | placement-enabled，single-node only | 同一 8×H20 节点；作者未冻结精确互联拓扑，同时报告作者 rank-mean 与共同 rank-max | `NOT_RUN` |
| P1 | EPLB `d52c72d5...` | 慢时标、历史负载驱动专家放置 | placement-enabled，单机 planner / 多节点 full-step | 先补公开代码 adapter、迁移成本和端到端边界 | `NOT_RUN` |
| P1 | LPLB `0490f794...` | 批次级 LP token 重定向 | placement-enabled，planner 与 full-step | 同 replica 预算，不能只比较 solver 的 max/mean | `NOT_RUN` |
| P1 | ECHO draft `a2b16b87...` | Megatron-Core 热专家弹性克隆 | placement-enabled，端到端 | 冻结 draft checkout，建立独立 on/off benchmark；不能借用完整优化栈表 | `NOT_RUN` |

`P0-R` 是论文通信主结论的直接候选集合。论文冻结时，每个 P0-R 都必须
有最终状态：要么通过共同正确性/性能门禁，要么保留明确的 build、API、硬件或
环境失败；不得因第三方 build 失败删行。最低可运行对比覆盖另见第16节。
`P0-E` 用于 novelty 边界，不强行制造跨硬件数字。P1 用于适用性与交互/组合假设；缺少代码
或共同 API 时保留 `NO_PUBLIC_ARTIFACT`/`NO_COMMON_API`，不能为了填表制造数值。

本文中“RailBalance best”特指只用预先分离的 tuning 数据选定、然后在
confirmatory 数据解封前冻结的单一版本；不得用 confirmatory 结果反向选“best”。

本机 UCCL/NCCL 的 check-only 基础设施快照固定为
`.cache/rail_balance/competitors/preflight-20260801-08/result.json`（SHA256
`8e46f41c081e7838aecfaf7a50c6a64ef4d3b73e23f9d872afb454f2d3e907d5`）。它只证明冻结
source/tool/dependency/GPU/NIC/topology/disk合同的静态检查可复现；当前状态是
`BLOCKED_DEPENDENCY_NANOBIND`并观察到`WAITING_GPU`，future build executor仍为
`NOT_IMPLEMENTED`。build、correctness、benchmark、profile和GPU kernel均为`NOT_RUN`，
所以该快照不改变上表任何实验状态，也不能作为竞品结果或RDMA runtime证据。

## 4. 四条赛道的主矩阵

### 4.1 transport-only × single-node

单机没有真实跨节点 Rail，作用是测额外规划、metadata、分支、节点内 forwarding 和兼容性开销，不能声称网络加速。

| ID | 系统/模式 | 冻结输入 | 主指标 | 论文位置 | 状态 |
|---|---|---|---|---|---|
| `TS-01` | same-tree DeepEP `off` / `legacy_exact` / `one_hop` / `adaptive` | 同一物理目标 trace，EP8；覆盖均衡、热点、多 target mask | rank-max dispatch/combine/roundtrip latency；额外本地 bytes；正确性 | 开销/边界消融 | `NOT_RUN` |
| `TS-02` | clean DeepEP v2 vs same-tree `off` | 同 shape/dtype/layout | rank-max latency与逻辑带宽；代码身份 | 可复现性附表 | `NOT_RUN` |
| `TS-03` | UCCL-EP intranode vs DeepEP v2 vs RailBalance | 官方 anchor 4096 token/rank、H7168、K8、E256；另做共同 trace | rank-max public-API latency；SM/CPU资源 | 补充材料 | `NOT_RUN` |

若本地使用 vnode、reference 或 C100 adapter，只能标成 `SCAFFOLD_DIAGNOSTIC`；它们不进入真实 Rail 图，也不与作者 GPU 数字相除。

### 4.2 transport-only × multi-node

这是 RailBalance 的通信主赛道，只有真实 `D>1` Gin/RDMA 路径且能取得同一计时窗口的 runtime path counters 与 per-NIC/QP counters 后才能进入论文主图。

| ID | 系统/模式 | 工作负载 | 公平条件 | 主结论位置 | 状态 |
|---|---|---|---|---|---|
| `TM-01` | same-tree `off` vs `one_hop` vs `adaptive` | 合成 2×2 trace：专家均衡/偏斜 × Rail均衡/热点 | 完全相同物理目标、payload、二进制/JIT、timed window；ABBA/BAAB | 主图：核心因果结果 | `NOT_RUN` |
| `TM-02` | same-tree `off` vs `legacy_exact` | 同 `TM-01` | 冻结 legacy 语义和新模式语义，不把兼容路径当优化 Leader | 补充：回归与兼容性 | `NOT_RUN` |
| `TM-03` | same-tree `off` / clean DeepEP v2 / tuning-frozen RailBalance best | UCCL官方HT anchor：4096 token/rank、H7168、K8、E288，FP8 dispatch/BF16 combine | 相同 GPU/NIC、route、SM/QP、API 与 rank-max | `AUTHOR_ARTIFACT_COMPAT`背景panel | `NOT_RUN` |
| `TM-04` | UCCL-EP vs `TM-03`三系统 | 与 `TM-03`相同；若支持再加 128-token LL anchor | 同硬件/NIC/trace；UCCL 4 proxy threads/GPU或共同调参预算并单独披露 | UCCL官方shape兼容图，不与NCCL横算 | `NOT_RUN` |
| `TM-05` | UCCL-EP / clean DeepEP v2 / tuning-frozen RailBalance best 三系统 scaling | nodes/EP、tokens、H、K、E、dtype 分轴变化 | 每次只改变一个轴；route seed/hash固定 | scaling 图 | `NOT_RUN` |
| `TM-06` | UCCL-EP / clean DeepEP v2 / tuning-frozen RailBalance best 真实 trace replay | 脱敏且版本化的 logical/physical trace | 同时间片、相同 endpoint，报告 trace覆盖度 | 主图或外部有效性图 | `NOT_RUN` |
| `TM-07` | same-tree `off` / clean DeepEP v2 / RailBalance best / UCCL-EP / NCCL EP LL | E256、H7168、K8、128 token/rank、全BF16；至少2/4 nodes | 统一逐iteration rank-max；route/layout/handle update、Dispatch(+Complete)、Combine(+Complete)分层；作者 mixed timing只进背景 | LL五系统主图 | `NOT_RUN` |
| `TM-08` | 与`TM-07`相同五系统的HT | E256、H7168、K8、4096 token/rank、全BF16；至少2/4 nodes | 与`TM-07`同一分层边界；论文无HT性能，later v0.1的HT优化/`>8` nodes修复不等于作者复现 | HT五系统主图 | `NOT_RUN` |
| `TM-09` | SABRE vs NCCL/PyTorch AllToAllv AUTHOR_REPRO；Rail adapter COMMON_FAIR | 完整 skew matrix 与相同最终 endpoint | packing/reorder/API 无法统一时记 `NO_COMMON_API` | 算法边界/补充图 | `NOT_RUN` |
| `TM-10` | fabric-lib vs DeepEP v2 vs RailBalance | decode 128 与 prefill 4096 分开 | 固定 proxy CPU/NUMA、NIC集合；逐 iteration rank-max | 多NIC直接竞品图 | `NOT_RUN` |
| `TM-11` | SwiftEP off/on × Rail off/on | 2K/4K/8K prefill与同一token trace | 必须合并到同一冻结源码基座、共享`off/off`并可独立开关；固定fused buffer/SM/QP/NIC | 共同基座成立才做2×2；否则只做两个独立对比 | `NOT_RUN` |
| `EV-01` | NCCL论文vLLM panel兼容复现；Rail接入仅在同一serving基座成立后加入 | Qwen3-30B-A3B、1/2/4 nodes、1000 requests、max concurrency 32 | 每backend 4 runs与IQR字段兼容；另保存逐请求raw/CI；不得代替transport主矩阵 | 外部有效性补充 | `NOT_RUN` |

UCCL 官方六类 testbed 的作者数据只做背景。只有本集群与其中某一硬件完全匹配时才称“官方硬件复现”；否则 `TM-04` 是共同口径的本地对比，不是 UCCL 论文曲线复现。论文 Fig. 8 每个点取 HT/LL 的最小值；这个 oracle 只允许在 `AUTHOR_REPRO` 兼容图中出现。`COMMON_FAIR` 必须把两种 mode 分开，或在 tuning 分区一次选定后冻结到 confirmatory。

### 4.3 placement-enabled × single-node

| ID | 系统/模式 | 冻结条件 | 主指标 | 限制 | 状态 |
|---|---|---|---|---|---|
| `PS-01` | MoonEP vs DeepEP v2（作者通信口径） | 仅精确 8×H20；S8192/rank、E384、H7168、K8、H'2048、B48、padding128、BF16、MaxVio .2/1/10/20 | 作者定义的跨-rank mean，20/50 eager event；另保存 raw | `AUTHOR_REPRO`，排除 grad_reduce | `NOT_RUN` |
| `PS-02` | MoonEP vs DeepEP v2（共同口径） | 与 `PS-01`共享输入，但增加统一 correctness 和 rank-max | dispatch/combine、planner、permute/prefetch 分项；rank-max p50/p95/p99 | API包含项必须逐项列明，不能拿 rank-mean 图直接横比 | `NOT_RUN` |
| `PS-02b` | MoonEP full-layer/FWD+BWD | 与`PS-01`相同H20 EP8合同 | grouped GEMM、prefetch、backward、grad_reduce、full-step、allocated/reserved峰值 | 通信图不含这些阶段；OOM/碎片是结果，不能静默改shape | `NOT_RUN` |
| `PS-03a` | UltraEP 8-Hopper demo功能/开销 | 官方 demo 的E32、K4、20 layers，精确冻结完整argv与router trace | placement/reroute/weight sync/grad reduce/token A2A 分项 | `AUTHOR_DEMO`；不冒充EP64表格或论文RSN | `NOT_RUN` |
| `PS-03b` | UltraEP EP64公开microbench anchor | E256、K8、8192 token/rank、2 redundant/rank | 同官方计时兼容字段，并另做共同rank-max | 需要EP64；不能缩成8卡后沿用官方ID | `NOT_RUN` |
| `PS-04` | EPLB vs LPLB planner | 同逻辑 load tensor、replica数、拓扑、seed；含 E256 EP16/32/64 | planner latency、最终 max/mean、迁移/复制量、解合法性 | planner质量不能代替 step time | `NOT_RUN` |
| `PS-05` | RailBalance 单机组合 sanity | placement off/on × Rail off/on | 语义一致性、额外 local forwarding、控制面开销 | 不做网络性能结论 | `NOT_RUN` |
| `IS-01a` | UniEP 公开前向融合段 | 单节点`EP=2/4/8`，dispatch+GEMM与GEMM+combine | 分EP的build/API/correctness；保留公开Torch/CUDA reference的严格/bitwise assert | 只是前向融合段，无多节点Rail | `NOT_RUN` |
| `IS-01b` | UniEP 完整训练FWD/BWD | 公开冻结提交 | 需要可运行backward API/harness和完整reference | 截止日未找到该公开入口 | `API_GAP` |

如果没有 H20，`PS-01/02` 必须记为 `UNSUPPORTED_HARDWARE`。在 H100/H200 上移植 MoonEP 可以作为 exploratory port，但必须另起 ID，不能冒充官方口径。

### 4.4 placement-enabled × multi-node

| ID | placement层 | transport层 | 设计 | 主指标 | 状态 |
|---|---|---|---|---|---|
| `PM-01` | UltraEP off/on | Rail `off`/best | 先把Ultra公开HybridEP/DeepEP-v1集成与RailBalance-v2移到同一冻结transport基座并提供两个独立开关，再做2×2；共享logical router trace、权重、replica预算 | full-step与通信rank-max、expert/Rail负载、权重同步和梯度归并成本；共同基座失败则只报独立结果 | `NOT_RUN` |
| `PM-02` | EPLB off/on | Rail `off`/best | 固定历史窗口、更新周期、replica预算；摊销与触发时延都报告 | steady step、placement迁移成本、Rail tail | `NOT_RUN` |
| `PM-03` | LPLB off/on | Rail `off`/best | 固定 batch trace 与副本拓扑；solver时间不从step中隐藏 | full-step、solver、dispatch/combine、grouped GEMM、Rail tail | `NOT_RUN` |
| `PM-04` | ECHO off/on | Rail `off`/best | 仅在冻结 draft PR 且有独立开关/正确性后运行 | clone、reroute、token A2A、grad reduce、full-step | `NOT_RUN` |
| `PM-05` | placement best | DeepEP / UCCL / RailBalance | 先固定 placement 输出，再 replay 同一物理 trace | 纯 transport rank-max | `NOT_RUN` |
| `PM-06` | FEPLB off/on | Rail `off`/best | 代码发布后做2×2；冻结CE复制、显存、CPU planner和backward | full-layer与transport分项 | `NO_PUBLIC_ARTIFACT` |

`PM-01..04`用于回答组合后的最终效果，`PM-05`用于在 placement 后重新隔离 transport 因果。两类结果必须同时存在，否则无法判断收益来自专家重映射还是 Rail 选择。

## 5. 两种 trace 合同

### 5.1 transport-only：冻结物理目标合同

每个样本冻结并 hash：

- hidden/input tensor、top-k indices、top-k weights、dtype、shape、layout、stride、alignment；
- 每个 token 的最终 owner/目的节点/目标 rank mask；
- dispatch payload、combine payload和 metadata；
- seed、生成器版本、trace文件 SHA256；
- 理论 logical bytes、去重后的跨节点 payload bytes。

所有 transport 必须交付相同的最终专家输入/输出与目标集合。系统可以改变 NIC/QP、源节点内 egress、目标节点内 forwarding 和分块方式，不得改变专家 placement、drop/capacity、token副本数或 payload 数值。若某系统的 API 无法表达同一合同，该点记 `NO_COMMON_API`，不降级成相似 shape 继续横比。

### 5.2 placement-enabled：冻结逻辑专家合同

每个样本冻结逻辑 router trace、logical expert id、模型权重、capacity/drop规则和正确输出。placement系统可以改变 logical→physical expert 映射及副本，但必须额外保存：

- 派生后的物理目标 trace及 SHA256；
- 每个 expert副本位置、replica预算和显存占用；
- reroute/clone/weight sync/grad reduce字节数；
- 迁移发生时刻、历史负载窗口及是否使用未来信息；
- 最终 logical expert输出的还原映射。

主公平轨采用 **matched-budget**：相同额外专家slot、显存和更新频率。各系统的 `native-best` 可作为第二轨，但必须单独画图并报告它多用了哪些资源。

## 6. Workload 矩阵

### 6.1 必做合成 trace

| 维度 | 水平 | 目的 |
|---|---|---|
| 专家负载 | balanced；moderate skew；severe skew；single-hot | 区分 expert imbalance 与 Rail imbalance |
| Rail负载 | balanced；single-Rail hot；alternating hot；random | 验证策略是否真正缓解物理出口拥塞 |
| 目标位置 | local；same-node remote；cross-node diagonal；cross-node off-diagonal | 覆盖端点/转发分支 |
| target mask | 单target；多target低fanout；高fanout | 验证同一 token/目的节点 payload共享语义 |
| shape | 对齐；非对齐边界；最小；最大可承载 | 正确性和尾部效率 |
| dtype | 共同支持的 BF16；共同支持时加 FP8 dispatch/BF16 combine | 不跨dtype算speedup |

论文主图至少包含 expert balanced/imbalanced × Rail balanced/hot 的 2×2。横轴标签使用**运行时实际测得**的 expert max/mean、per-Rail max/mean和path分布，不能只写生成器请求的名义 skew。

### 6.2 官方 shape anchors

| 系统 | 只在对应轨道使用的公开 anchor | 使用规则 |
|---|---|---|
| DeepEP v2 | 8K token、H7168、K8、FP8 dispatch/BF16 combine | 显式设置 K8；不能误用测试默认 K6 |
| UCCL-EP HT | EP32、4096 token/rank、H7168、K8、E288、FP8/BF16 | 作为 `TM-03/04`共同 anchor；固定HT，不使用Fig.8逐点mode oracle |
| UCCL-EP LL | EP32、128 token/rank、H7168、K8、E288、FP8/BF16 | 仅UCCL-EP、clean DeepEP v2、RailBalance三系统都支持相同 LL API 时进入横比 |
| NCCL EP LL | E256、H7168、K8、128 token/rank、BF16、8–64 GPU | `TM-07`；共同 timing，不复用作者 mixed boundary或tag `ep_bench`的rank-mean |
| NCCL EP HT | >=4096 token/rank；三方共同非量化dtype | `TM-08`；论文无HT性能；v0.1已有HT优化/`>8` nodes修复但公开表≤8 nodes，不能称`AUTHOR_REPRO` |
| SABRE | BF16、64–512 MB/rank、高/低skew、16–256 GPU | `TM-09`；先复现AllToAllv，再决定是否存在共同EP adapter |
| fabric-lib | H7168、K8、decode<=128/prefill4096、FP8/BF16 | `TM-10`；两种token域分开，rank pooling改为rank-max |
| SwiftEP | H7168、K8、2K/4K/8K、FP8/BF16 dispatch、BF16 combine | `TM-11`；H20作者轨与本机port分开 |
| MoonEP | EP8、8192 token/rank、E384、H7168、K8、H'2048、BF16 | 单节点 H20 独立赛道 |
| UltraEP | EP64 microbench：E256、K8、8192 token/rank、2 redundant/rank；另有8卡demo E32/K4/20 layers | 两个artifact分ID；保存实际trace/ratio，缩卡不冒充EP64 |
| LPLB | E256、EP16、4 redundant/rank；cube/hypercube；另有EP16/32/64算法case | 先复现planner，再接入full-step |
| EPLB | 2 layers×12 experts、16 replicas、2 nodes×4 GPU示例 | 只做功能anchor，不当性能workload |
| UniEP | 冻结提交公开的单节点`EP=2/4/8`前向融合段 | `IS-01a`；保留Torch/CUDA reference与严格/bitwise assert；完整FWD/BWD为`IS-01b`/`API_GAP` |
| UBEP | 256 Ascend dies、CM384一/两跳调度 | evidence-only，不生成NVIDIA COMMON_FAIR点 |

把官方 shape 缩小到更少 GPU 可以用于脚手架和正确性，但 ID 必须带
`DERIVED_SCALE_DOWN`，并在表中列出改变的 EP、node、topology 和 payload；它不属于
作者结果复现。卡数不足时优先为 NCCL EP、UCCL-EP、SABRE、SwiftEP、fabric-lib 建
2/4-GPU compile/API/correctness harness；UniEP 只做公开`EP=2/4/8`前向融合段，完整
FWD/BWD保留`API_GAP`。NCCL tag 的 stock `ep_test` 在2/4 ranks把
`top_k=min(8,nRanks)`降成K2/K4，因此不能验证K8合同；少卡K8正确性应使用
`ep_bench --top-k 8 --validate`或独立共同adapter，并明确其原生输出统计仍不是共同
rank-max。UCCL的2卡intranode入口会运行自带调优，只能标功能smoke，不能把其输出当
稳定benchmark。少卡时延只证明脚手架与趋势，不进入主性能结论。

### 6.3 真实 trace

若能取得训练/推理 trace，必须脱敏后冻结：采集模型/阶段、连续 microbatch窗口、采样率、是否包含 expert placement变化，并发布 trace hash和统计摘要。不能只挑最有利的热点片段；主结果使用预先固定的连续窗口，最坏case另作 stress test。

## 7. 公平环境合同

每个 `COMMON_FAIR` block 在运行前冻结以下 manifest；任一关键字段变化就分成新 block：

- 完整源码 SHA、dirty patch hash、submodule、编译参数、PTX/cubin/JIT/SASS hash；
- GPU型号/UUID、driver、CUDA、PyTorch、NCCL、NVSHMEM、Gin和固件版本；
- GPU时钟、power limit、MIG状态、ECC、温度范围和持久化模式；
- PCIe/NVLink/NVSwitch/NIC/QP/NUMA拓扑及 GPU↔NIC affinity；
- 节点/rank映射、CPU affinity、proxy线程、后台进程和独占状态；
- 输入、trace、reference、API起止点、stream和同步方式；
- SM数、QP数、channel/chunk/buffer配置和额外显存；
- cold-start/build/JIT状态。首次编译/JIT时间与 steady latency分开报告。

资源公平采用两条明确轨道：

1. `MATCHED_RESOURCE`：固定每 GPU可用 SM/QP/NIC、额外显存与 CPU cores；这是主结论。
2. `EQUAL_TUNE_BUDGET_NATIVE_BEST`：给每个系统相同 wall-clock调参预算、shape集合和试验次数，允许各自最佳配置；这是补充结论。

UCCL 的 CPU proxy时间、core数和NUMA绑定属于资源，不得隐藏。RailBalance增加的本地NVLink转发和buffer显存同样必须披露。

## 8. 正确性门禁

每个版本必须先通过共同 reference，再测速。至少覆盖常用、边界、非对齐、最小/最大 shape，多seed，零、极值、重复expert、空expert、多target mask，以及所有声明支持的dtype/layout。

记录：max absolute error、max relative error、首个错误位置、输入/参考/实测值、shape/dtype/seed、重复运行是否确定。共同浮点门禁默认 `relative error < 1e-3`；若竞品官方标准更严格（例如 UCCL LL 的 BF16/FP8门禁），同时报告并优先满足更严格标准。integer metadata、counts、indices、owner/target masks 应 exact。

placement实验还必须验证：最终 logical expert输出、token不丢不重、replica权重一致、梯度归并、capacity/drop语义和无未来trace泄漏。任何 correctness失败的候选不得测速；失败行仍保留。

## 9. 统一计时边界和指标

### 9.1 主计时定义

- 主单位是 profiler-free CUDA-event包围的 public API调用；事件、barrier和stream语义对所有系统相同。
- 每个 iteration先取得各rank时长，再取 `rank-max` 作为集群完成时间。rank mean只作为诊断或 MoonEP `AUTHOR_REPRO`兼容字段。
- dispatch、combine、dispatch+combine roundtrip分别报告；full-step不得替代transport分项。
- planner、reroute、permute/prefetch、weight sync、grad reduce、grouped GEMM独立计时，并说明是否包含在 full-step。
- compile/JIT/cold start独立报告，不混入steady样本。

主延迟统计为 rank-max样本的 p50/p95/p99、median absolute deviation和CV。吞吐从同一计时窗口和同一分子计算；不同系统自己的“logical bandwidth”定义只作兼容字段，不进入横向主图。

### 9.2 三套字节口径

1. **logical bytes**：用户可见tensor按统一公式计算；包含哪些local traffic必须写清。
2. **unique network payload bytes**：按冻结目标集合去重后的理论跨节点payload。
3. **physical NIC bytes**：同一timed window从每NIC/QP硬件/驱动counter读取，扣除可验证的基线流量，并保存raw counter。

任何 bandwidth必须在列名中写出分子。禁止把 logical GB/s 当作 NIC line rate，或用预测 path distribution 替代物理NIC bytes。

### 9.3 RailBalance 专项指标

- 每 NIC/Rail bytes、throughput、utilization、`max/mean`、CV；
- Jain fairness：`(sum(x_i))^2 / (N * sum(x_i^2))`；
- direct、source-forward、destination-forward、two-hop 的payload和token比例；
- 额外节点内NVLink bytes、额外hop数、two-hop cap触发次数；
- runtime选择分布与物理NIC counter是否一致；
- peak Rail load下降量与额外local forwarding的Pareto关系。

若没有同窗口物理counter，Rail负载结论标 `NO_PHYSICAL_COUNTERS`；可以发布软件路径正确性，但不能写“缓解了物理NIC拥塞”。

### 9.4 placement/full-step 指标

- expert/rank token load的max、mean、p95、max/mean；
- planner/solver/clone/reroute/weight sync/grad reduce latency和bytes；
- 每step吞吐与rank-max step latency；
- grouped GEMM时长和实际计算不平衡；
- 额外expert slots、显存、迁移频率及摊销期；
- placement后派生的physical trace和Rail指标。

LPLB只优化token count，EPLB依赖历史估计；因此`max/mean`改善只能解释planner输出，不能代替最终step收益。

## 10. 重复、配对和显著性

统一 protocol 在首轮GPU运行前写入manifest，之后不因结果好坏改变：

- 每个新进程固定 warmup和steady次数；初始建议使用现有共同脚手架的 10 warmup + 100 steady，但正式冻结后所有共同版本一致。
- 至少5次独立进程启动、至少10个paired blocks；系统顺序交替使用 ABBA 与 BAAB。
- 每个 block共享输入trace、环境采样和紧邻时间窗口；发生热降频、外部作业、链路错误或counter不闭合时整block标记失败，不单删慢样本。
- 对paired latency ratio取log，报告median speedup及block bootstrap 95% CI；原始逐iteration和逐rank样本全部保存。
- 先跑 A/A伪配对估计噪声。只有收益CI排除1.0，且绝对log-ratio超过A/A噪声95百分位，才称“真实性能提升”；否则写“与噪声不可区分”。
- 在生成候选前按 trace/seed/block 冻结互斥的 `tuning` 与 `confirmatory`
  分区并分别hash；候选生成、超参选择和“RailBalance best”排名只能读 tuning
  数据。冻结代码、参数和唯一候选ID后才解封 confirmatory；confirmatory 失败
  不能反向调参，如需继续优化必须新建实验ID和新的确证集。
- 预注册主对比为 `off vs one_hop`、`off vs adaptive`、`tuning-frozen Rail best vs UCCL`；同一figure内多重主检验用Holm校正。其余标exploratory。
- p95/p99使用block bootstrap或分层bootstrap，不能把同一进程中的iteration错误当成完全独立样本。

作者默认值（UltraEP 10/30、MoonEP 20/50、UCCL代码50/50或Kineto 30）仅用于各自 `AUTHOR_REPRO`，不混进 `COMMON_FAIR`统计。

## 11. Profile 和性能归因

Profile只对已正确、且由稳定计时选出的代表版本执行；profiled latency不替代profiler-free主结果。每个代表点保存工具版本、命令、trace和report hash。

| 层级 | 必采指标 | 需要回答的归因 |
|---|---|---|
| GPU kernel | duration、occupancy、register/thread、shared memory、DRAM/L1/L2、issue slots、warp stalls、指令数 | 是算力、访存、依赖、occupancy还是launch/调度瓶颈 |
| 节点内 | NVLink/NVSwitch bytes与utilization、forwarding kernel | 降低Rail热点用了多少本地搬运，是否挤占计算/通信 |
| 网络 | per-NIC/QP bytes、带宽、重传/错误、CPU proxy利用率 | 软件路径选择是否转化为物理链路均衡，UCCL资源成本是什么 |
| 时间线 | dispatch/combine/compute/forward overlap，bubble与rank tail | latency变化来自真正重叠还是减少某个阶段 |

归因必须形成可证链：例如“runtime路径计数转移到Rail 1 → NIC counter同窗口增加且Rail 0峰值下降 → rank-max p95下降”。只有软件oracle而没有NIC counter时，链条不完整，结论降级。

## 12. 预注册消融

### 12.1 RailBalance

| 消融 | 固定项 | 目的 |
|---|---|---|
| `off / legacy_exact / one_hop / adaptive` | trace、binary/JIT、resource、timing | 区分兼容路径、单跳和自适应策略 |
| two-hop开/关及受支持的cap | 其余policy参数 | 衡量尾部改善与额外NVLink成本 |
| 单target / 多target fanout | logical payload不变 | 验证per-token/per-destination共享 |
| H、K、token、alignment逐轴变化 | 其余shape固定 | 找到控制开销与payload主导区间 |
| balanced / hotspot Rail | expert分布固定 | 隔离物理Rail不均衡 |
| balanced / hotspot expert | Rail目标固定 | 隔离expert skew |

chunk、hop penalty、pipeline stage等参数只有在代码确实存在、接口冻结后才能加入；本文不预造flag。若能实现plan freeze/replay，则用它拆开policy选择成本与数据面收益；未实现时明确记缺失。

### 12.2 竞品/组合

- **UltraEP**：placement off/on × Rail off/best；在共同源码基座上消融redundant-slot预算、direct/adaptive relay，并分别报告placement、reroute、weight sync、grad reduce的字节/显存和token A2A。公开约1.5/2/3负载档以实际trace统计为准。
- **MoonEP**：在官方允许的真实选项内测zero-copy、B预算、planning/permute/prefetch与MaxVio；`grad_reduce`、full backward、静态内存/碎片收益另表，不能暗中加入作者通信图边界。
- **UCCL-EP**：native 4 proxy threads/GPU；1/2/4 threads需要独立源码构建，因为冻结代码把线程数编译为常量。另冻结CPU affinity/NUMA、SM、QP/NIC mapping、inflight bytes/count；主表matched-resource，native-best使用equal tune budget。
- **NCCL EP**：LL/HT严格分开；route/layout/handle create或update、`Dispatch(+Complete)`、`Combine(+Complete)`和one-time group/JIT分层。不能沿用作者NCCL-host/DeepEP-Kineto混合边界，也不能把tag `ep_bench`的跨rank mean或错误方向的kernel throughput min/max列当共同统计。论文LL与later v0.1 release的HT优化/`>8` nodes修复分开归档，release当前不支持quantization。
- **SABRE**：高/低skew、块拆分、overlap开关；补严格reference与rank-max，保留作者回退点。
- **fabric-lib**：NIC数、sharding/rotation、proxy core/NUMA；decode/prefill分panel。
- **SwiftEP**：buffer fusion、SGL、QP并行、TMA offload；只有合并到同一冻结源码基座、共享`off/off`并有两个独立开关时才做SwiftEP×Rail 2×2，否则只报两个独立对比。
- **EPLB**：hierarchical/global、历史窗口、更新周期和replica预算；报告迁移与摊销。
- **LPLB**：官方cube/hypercube及代码实际支持的其他topology；solver与完整step同时报告。
- **ECHO**：只有公开draft代码能独立开关时才做clone/reroute/gradient组件消融；完整Megatron优化栈不用于单项归因。

## 13. 论文图表清单

| 编号 | 图/表 | 数据来源 | 关键编码 | 进入条件 |
|---|---|---|---|---|
| Fig. 1 | transport rank-max latency与speedup | `TM-01/07/08` | 五系统 × trace class，p50/p95/p99 + 95% CI | T3真实网络、correctness全过；失败保留状态 |
| Fig. 2 | latency vs 实际Rail imbalance | `TM-01/05/06` | x=per-Rail max/mean，y=rank-max latency | 有物理counter |
| Fig. 3 | per-node/per-NIC heatmap | `TM-01/04` | before/after bytes与utilization | counter窗口闭合 |
| Fig. 4 | peak Rail load vs extra local forwarding Pareto | Rail消融 | 点=mode/cap/shape | 有NIC+NVLink counters |
| Fig. 5 | tail distribution | paired raw samples | CDF/violin，标block而非伪独立误差棒 | 至少10 paired blocks |
| Fig. 6 | phase/time-line breakdown | representative profiles | planner/dispatch/forward/combine/compute | profile lineage完整 |
| Fig. 7 | expert skew × Rail skew 2×2 | `PM-01` | placement主效应、Rail主效应、interaction | UltraEP组合跑通 |
| Fig. 8 | scaling | `TM-05` | nodes/tokens/H/K/E/dtype逐轴 | 每点同trace合同 |
| Fig. 9 | 扩展直接系统同机比较 | `TM-09..11` 中实际可比子集 | API-rank-max；失败/不支持保留空位和状态 | 主五系统之外的算法族至少一个同机通过或留下最终失败状态 |
| Fig. S1 | MoonEP单机组件/整层 | `PS-01/02/02b` | 作者rank-mean与共同rank-max分panel；另列FWD/BWD/GEMM/memory | 精确H20或明确port |
| Fig. S2 | SABRE/UBEP novelty边界 | 证据账本与`TM-09` | endpoint、调度粒度、hop、forwarding、API对照 | 不混跨硬件作者speedup |
| Table 1 | 系统范围与第一手证据 | 证据账本 | transport/placement、single/multi、SHA | 无GPU要求 |
| Table 2 | 环境与资源 | manifests | GPU/NIC/software/SM/QP/CPU/memory | 每个性能block必填 |
| Table 3 | 正确性与失败矩阵 | 全部候选 | pass/failure code、误差、shape/seed | 不删失败行 |
| Table 4 | 主性能摘要 | paired统计 | median speedup、CI、A/A噪声阈值 | 仅共同口径 |
| Table S1 | 作者报告结果 | 论文/README | 原硬件、原指标、证据链接 | 与本地结果视觉隔离 |
| Table S2 | 无效消融与资源代价 | leaderboard/profile | 结论、失败原因、extra bytes/memory/CPU | 所有候选保留 |

## 14. 失败、缺失和不可比披露

每个计划行只能处于以下显式状态之一：

| 状态 | 含义 | 发布方式 |
|---|---|---|
| `NOT_RUN` | 已预注册，尚未尝试 | 保留冻结输入和前置条件，不产生结果性措辞 |
| `NOT_RUN_ENV` | 尚未取得所需机器/软件 | 保留行并列缺失条件 |
| `UNSUPPORTED_HARDWARE` | 官方实现不支持当前GPU/NIC/topology | 不移植冒充复现；可另列port |
| `BUILD_FAIL` | 固定版本无法构建 | 保存完整命令、stdout/stderr、环境和返回值 |
| `CORRECTNESS_FAIL` | 未过共同reference | 不测速；保存首错与seed/shape |
| `OOM` | 固定资源下显存不足 | 保存分配、峰值显存与shape |
| `TIMEOUT_OR_HANG` | 达到预注册timeout | 保存rank日志和最后进度，不猜性能 |
| `INTERFERENCE` | 外部作业、降频、链路异常等污染block | 整个paired block排除，原因预先定义 |
| `NO_COMMON_API` | 无法表达相同语义/API边界 | 不做speedup，只做定性比较 |
| `API_GAP` | 冻结公开代码缺少计划所需入口 | 列明已搜索范围和缺口，不自造替代实现冒充作者artifact |
| `NO_PUBLIC_ARTIFACT` | 截止日没有可冻结代码/命令 | 保留证据和设计差异，不伪造实现 |
| `NO_PHYSICAL_COUNTERS` | 无法验证物理Rail流量 | 不做Rail均衡/线速结论 |
| `DIRTY_TREE` | 源码身份不满足manifest | 不进入正式结果 |
| `NOT_COMPARABLE` | 硬件、dtype、计时或系统边界不同 | 只放背景/相关工作表 |
| `CORRECTNESS_PASS` | build/API/全部共同reference门禁通过 | 只表示可以测速，本身不是性能结论 |
| `BENCHMARK_PASS` | correctness先通过，且稳定计时、干扰和噪声门禁全过 | 才能进入性能统计与共同图；仍需报告CI |

禁止只报最快成功配置、静默删除慢点、把OOM改小shape后沿用原ID，或把失败竞品写成零。修改shape、资源或代码后必须生成新manifest和新ID。

## 15. 证据成熟度与执行顺序

| 等级 | 最低条件 | 允许结论 |
|---:|---|---|
| T0 | CPU/reference、manifest/schema、trace hash、正确性脚手架 | 只能说脚手架可执行 |
| T1 | 单GPU planner或算法测试 | planner correctness/latency，不是通信结论 |
| T2 | 单节点真实8GPU | 节点内开销与功能；不做Rail网络结论 |
| T3 | 多节点真实Gin/RDMA + 同窗口NIC/QP counters | transport论文主结论 |
| T4 | placement/full-model组合与完整step | 端到端、主效应/交互和训练影响 |

执行顺序冻结为：

1. 完成T0：统一manifest、raw schema、trace生成/replay、reference和失败状态。
2. 卡数不足时完成T1/T2：为 NCCL EP、UCCL-EP、SABRE、SwiftEP、fabric-lib 建2/4-GPU
   `DERIVED_SCALE_DOWN` build/API/correctness harness；NCCL少卡K8走带`--validate`的
   `ep_bench`或共同adapter而不是会降K的stock `ep_test`；UniEP做公开`EP=2/4/8`
   前向融合段，不预注册不存在的完整backward入口；所有缩小结果不称作者复现。
3. 有8卡但只有单节点时跑T2，优先same-tree开销、DeepEP clean anchor，以及硬件支持的
   P0-R编译/正确性；Moon仅H20，Ultra只做功能/整层，UniEP只做公开前向融合段；
   不写多节点Rail收益。
4. 有多节点8卡/节点时先跑`TM-01/07/08`五系统共同主矩阵，再跑UCCL官方shape兼容
   `TM-03/04`与`TM-09..11`扩展对手；每个系统均走
   correctness→stable timing→profile闭环，失败行不删除；再跑Ultra/SwiftEP组合。
5. P0-E/P1的无代码系统不阻塞P0-R主线，但必须保留novelty/组合边界。

## 16. 最低论文交付与停止条件

通信主张至少需要：

- same-tree `off`、clean DeepEP v2 和 tuning-frozen RailBalance best 在同机共同口径下
  的完整正确性与 paired raw samples；
- UCCL-EP与NCCL EP都必须进入主表，或留下可审计的build/API/hardware最终失败状态；
  UCCL本身已有QP round-robin和多NIC聚合，属于原生EP与显式NIC-balancing的交集，但
  仍应尽量加入一个算法边界不同的NIC-balancing系统（SABRE或fabric-lib），不能用一项
  模糊的通用P0-R结果代表全部类别；
- 所有 P0-R 在论文冻结时都必须从通用`NOT_RUN`转为最终状态：
  `BENCHMARK_PASS`，或明确的build/correctness/API/hardware/environment失败状态；
- balanced、Rail-hot、expert-skew × Rail-skew，以及至少一个真实或连续采样trace；
- per-NIC/QP物理counter、runtime path counter和节点内forwarding bytes能相互闭合；
- A/A噪声、bootstrap 95% CI、环境manifest和失败矩阵；
- 一个placement系统（优先UltraEP）的2×2组合，或明确披露因环境无法完成。

满足以下任一条件可停止某一优化方向：达到预注册目标；与可估计硬件上限差距足够小；连续多轮提升不超过A/A噪声；主要假设均被验证或否定；实验预算耗尽。停止不等于成功，最终必须同时发布最佳版本、无效方向和限制。

在T3前，只能写“完成脚手架/正确性/单机诊断”。在全部共同口径结果和统计门禁完成前，
禁止写“优于 NCCL EP/SABRE/SwiftEP/UCCL-EP/fabric-lib/DeepEP v2”，也不得拿
UniEP/FEPLB/UltraEP/MoonEP/ECHO/EPLB/LPLB 的整层或 placement 数字冒充 transport
胜负。novelty 只能限定为 DeepEP/NVIDIA 多NIC RDMA数据面内、保持最终endpoint不变的
hop/cost约束 rail/vnode选择与有限节点内forwarding；不得写泛化的“首次”。

## 17. 每个结果点的最小artifact

每个结果目录至少包含：

- immutable manifest与环境快照；
- source/patch/build/binary/JIT/SASS identity；
- input、logical trace、physical trace及hash；
- reference和correctness逐case日志；
- 每iteration、每rank、每phase原始时间；
- GPU clocks/temperature/power和干扰检测；
- runtime path、NVLink/NIC/QP raw counters与窗口时间戳；
- profile report及其对应的unprofiled sample ID；
- 聚合脚本版本、统计输出和图表数据；
- `PASS`或失败状态及原因。

所有图表必须能从这些raw artifacts一键重算。没有raw lineage的作者PNG、README约数或单次终端输出只能作为外部证据，不能进入我们的统计结果。
