# RailBalance 竞品与相关系统证据账本（2026-08-01）

> 状态：论文实验的第一手证据冻结稿，不是最终实验方案，也不是性能结论。
>
> 截止时间：2026-08-01 UTC。本文只接受作者论文、作者仓库、作者提交的代码/命令和作者发布的图表；不引用媒体、聚合站或第三方复述。
>
> 当前没有在本机完成任何竞品 GPU 性能复现。论文图、README 表格和作者声称的加速比均不得写成“本机结果”，也不得据此声称 RailBalance 已经胜出。

## 1. 证据标签与判定规则

本文使用以下互不替代的标签：

- **P（Paper）**：作者论文或技术报告中的文字、表格和实验口径。
- **C（Code）**：固定到完整 commit SHA 的作者代码。默认分支名不能替代 SHA。
- **R（Run command）**：作者仓库中公开、可定位到具体文件的运行命令或入口。
- **A（Artifact）**：作者发布的图、表或报告。它只证明“作者发布过该结果”。
- **A-data（Raw artifact data）**：能够重新计算图表的逐样本 CSV/JSON/日志。PNG、论文曲线和 README 表格不属于原始数据。
- **Local**：本机已有 checkout 的路径、SHA 和脏状态。它只证明代码存在，不证明可构建或可运行。
- **Reproduced**：在本机/本集群按冻结口径实际运行，并保存命令、环境、正确性和原始计时。未达到这个标准统一写“未复现”。

“代码中有默认 warmup/iters”不等于“论文图使用了这些默认值”；只有论文、图表生成脚本或其原始 manifest 明确建立 lineage 时才能合并两者。缺失项保持缺失，禁止补猜。

## 2. 总览：系统角色与当前证据等级

| 系统 | 主要改变对象 | 与 RailBalance 的关系 | P | C | R | A | A-data | Local | Reproduced |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| UltraEP | 精确负载驱动的专家复制、token reroute、权重/梯度搬运 | 作用层不同；是否可组合且收益正交必须由2×2实验验证 | 有 | 有 | 有 | 有 | 未公开 | 有 | 否 |
| MoonEP | 单机动态冗余专家、静态 shape、零拷贝通信 | 截止日已审计入口仅单机；不是当前多节点 Rail 对手 | 无独立论文 | 有 | 有 | 有 | 未公开 | 有但落后一提交 | 否 |
| NCCL EP | NCCL Device API 原生 LL/HT dispatch/combine | 直接通信对手；HT同样含节点内聚合和跨节点阶段 | 有 | 有 | 有 | 有 | 未公开 | 有，冻结tag | 否 |
| SABRE | skew-aware AllToAllv proxy 分流、块拆分和节点内重排 | 与 proxy/NIC 分流及节点内转发有明确算法重叠 | 有 | 有 | 有 | 有 | 未公开 | 无 | 否 |
| fabric-lib | CPU proxy、多 NIC sharding/rotation | 直接多 NIC/Rail 对手 | 有 | 有 | 有 | 有 | 未公开 | 无 | 否 |
| SwiftEP | DeepEP buffer fusion、SGL、多 QP、TMA 节点内搬运 | 直接通信对手；与 Rail 的组合空间待共享源码基座验证 | 有 | 分支/开放 PR | 有 | 有 | 未公开 | 无 | 否 |
| UCCL-EP | 可移植的 EP 通信传输栈与 CPU proxy | 直接通信对手 | 有 | 有 | 有 | 有 | 未公开 | 有，冻结SHA | 否 |
| DeepEP v2 | EP dispatch/combine 基线传输栈 | 上游基线；RailBalance 的直接母体 | 无独立 v2 论文 | 有 | 有 | 有 | 未公开 | 有开发分支但不是冻结 SHA | 否 |
| UBEP | CM384 上按一/两跳距离调度 token 发送任务 | 核心思想直接，但硬件不同且无代码 | 有 | 无 | 无 | 有 | 未公开 | 无 | 否 |
| UniEP | 论文的dispatch/GEMM/combine mega-kernel；冻结代码公开单机前向融合段 | integrated；非纯 transport，公开完整训练backward入口未找到 | 有 | 有 | 有 | 有 | 未公开 | 无 | 否 |
| FEPLB | Copy Engine 动态专家/Token 节点内迁移 | 计算负载层；组合可行性与交互仍待共同实现验证 | 有 | 无 | 无 | 有 | 未公开 | 无 | 否 |
| ECHO | 热专家弹性克隆、sync-free dropless MoE | 专家负载/整栈机制；非直接 Rail 对手 | 技术报告章节 | 草案 PR | 单测入口 | 仅整栈表/示意图 | 未公开 | 无冻结 PR checkout | 否 |
| EPLB | 基于历史/估计负载的专家复制与静态放置 | 慢时标放置器；非直接通信对手 | DeepSeek-V3 部署章节 | 有 | 示例 | 示例图 | 无 | 无 | 否 |
| LPLB | LP 求解的批次级 token 重定向 | 动态负载规划器；非直接通信对手 | 无独立论文 | 有 | 有 | README 仅给约数 | 无 | 无 | 否 |

为避免把“相关性”和“当前能否运行”混成一个优先级，后续实验使用三类标签：

- `P0-R`：直接数据面竞品且有可冻结代码，进入共同硬件主实验；包括 DeepEP v2
  `off`、NCCL EP、SABRE、SwiftEP、UCCL-EP 和 fabric-lib；
- `P0-E`：与核心思想存在明确 novelty重叠但当前不可复现，进入 related-work 与设计差异
  审计；当前是 UBEP；
- `P1`：整层融合、专家放置或计算负载机制，进入交互/组合假设实验，不进入纯 transport
  主图；包括 UniEP、FEPLB、UltraEP、MoonEP、ECHO、EPLB 和 LPLB。

## 3. UltraEP

### 3.1 系统边界

UltraEP根据当前 microbatch 的 post-gating 精确负载复制热点专家，并把 token 重定向到物理副本；同时负责副本权重同步和训练反向的副本梯度归并。它不替代 token dispatch/combine 后端：论文实现明确把 DeepEP `hybrid-ep` 用作 token all-to-all。

RailBalance保持一个已经确定的物理目标端点不变，只在可用 NIC/Rail 间选择出口，必要时付出一次本地 NVLink 转发。因此两者作用于不同层：UltraEP先决定“这个 token 去哪个专家实例”，RailBalance再决定“去该实例的网络 payload 从哪条 Rail 出去”。这给出了可组合的实现假设，但是否存在资源竞争、主效应是否独立以及组合收益是否保留，都必须由共享基座的2×2实验验证；不能把 UltraEP 当成 RailBalance 的替代基线，也不能用二选一比较推导谁全面优于谁。

[UltraEP §7](https://arxiv.org/html/2606.04101v3#S7)的公开集成基于其 HybridEP/DeepEP-v1
分支，而 RailBalance位于本地 DeepEP-v2树。只有先把两个独立开关放到同一个冻结
transport基座，2×2交互才可归因；否则只能发布两个独立结果，不能解释组合效应。

### 3.2 第一手证据

- **P**：[UltraEP arXiv:2606.04101v3](https://arxiv.org/abs/2606.04101v3)，[第 8 节评测](https://arxiv.org/html/2606.04101v3#S8)。论文报告公共云 RSN：每 rack 64 GPU/16 server；研究原型的训练使用 2 或 4 racks、serving prefill 使用 1 rack，所以公开研究结果覆盖到 256 GPU。论文没有披露生产部署规模，也未披露具体 GPU 型号。论文评测统一写明 BF16。
- **C**：冻结 [Dots-Infra/UltraEP@94cab099b44fffa99a82fea99e7c12d89cf65e4f](https://github.com/Dots-Infra/UltraEP/tree/94cab099b44fffa99a82fea99e7c12d89cf65e4f)，tag `v1.0.0`。它能冻结公开 README/demo/test artifact，但论文没有证明所有 Fig. 11–17 都由该 tag 生成。
- **R**：[分布式 `tests/test_e2e.py`](https://github.com/Dots-Infra/UltraEP/blob/94cab099b44fffa99a82fea99e7c12d89cf65e4f/tests/test_e2e.py#L680-L729)、[8×Hopper demo 与 Qwen3-235B recipe](https://github.com/Dots-Infra/UltraEP/blob/94cab099b44fffa99a82fea99e7c12d89cf65e4f/examples/README.md)。README 的公共入口是 `torchrun ... tests/test_e2e.py --num-experts 256`。
- **A**：论文 Fig. 11–17、README 的 EP64 `test_e2e.py` 表格和端到端汇总图。它们是作者 artifact，不是本机测量。
- **A-data**：仓库未发布可重算论文图或 README 表格的逐样本日志/CSV/JSON。
- **Local**：`/home/chen/workspace/infra/UltraEP`，`94cab099b44fffa99a82fea99e7c12d89cf65e4f`，审计时 clean，与冻结代码一致。
- **Reproduced**：未复现；本轮未构建、未运行 GPU、未产生本机 UltraEP 性能数字。

### 3.3 可冻结的官方口径与缺口

| 项目 | 官方可证内容 | 仍缺失/不能推断 |
|---|---|---|
| 硬件 | 论文：64 GPU/16 server 的公共云 RSN，scale-up 带宽为 scale-out RDMA 的 8–10 倍；代码要求 SM90/SM100、NVLink、NVSHMEM | 论文 GPU 型号、NIC 型号、每 GPU/NIC 映射、时钟、CUDA/NVSHMEM 精确版本未完整披露 |
| shape | 论文：GLM4.5 E128/K8 EP64、Qwen3 E128/K8 EP64、GLM4.7 E160/K8 EP40、DeepSeek-V3 E256/K8 EP64；代码表格：E256、K8、8192 token/rank、4 master+2 redundant/rank | 论文端到端每个点的完整 tensor layout、alignment 和实际 router trace未发布 |
| dtype | 论文端到端写明 BF16；`test_e2e.py` 默认 weight 2 B（BF16）、grad FP32 | 论文图与公开 `test_e2e.py` 表格的完整低精度/scale layout lineage未发布 |
| timing | `test_e2e.py` 默认 warmup 10、bench 30；[测试计时工具](https://github.com/Dots-Infra/UltraEP/blob/94cab099b44fffa99a82fea99e7c12d89cf65e4f/tests/utils.py#L116-L184) 使用 CUDA events、barrier，去掉首样本和两端极值；Kineto 通信 kernel duration 做 rank-max reduce | 论文端到端图的 warmup、iterations、独立重复、置信区间及其是否复用测试默认值未披露；CUDA-event统计与Kineto rank-max不能混用 |
| API 边界 | 测试分别测 `update_placement`、`reroute`、`weight_sync`、`grad_reduce`；token A2A 仅在 `--include-token-a2a` 时加入 | 这些分项不是 DeepEP dispatch/combine 的同一 API 边界；论文端到端包含模型计算，不可直接和 RailBalance kernel latency 相除 |
| correctness | `test_e2e.py`检查 placement/reroute 合法性、weight sync bitwise、deterministic grad bitwise、非 deterministic grad `allclose` | 论文图没有逐点 correctness artifact hash |

## 4. MoonEP

### 4.1 系统边界

MoonEP通过动态冗余专家把每个 rank 的接收量固定为 `S×K`，使用 CUDA VMM、对称映射和
NVLink remote reads 提供单机静态 shape 与 zero-copy buffer view。公开 README/源码没有
给出 NVSwitch multicast 的实现证据，不能把它补写成系统机制。公开实现和 README 截止
本账本日期只验证单节点 8×H20 EP8，没有公开多节点 RDMA 后端。因此 MoonEP可以作为
“单机专家负载与通信布局”相关工作，但当前不能作为 RailBalance 多节点 Rail 加速的
直接横向对手。

### 4.2 第一手证据

- **P**：没有独立论文；官方 citation 将仓库本身作为 2026 年作品。不得把 README 当作经过同行评审的论文。
- **C**：冻结 [MoonshotAI/MoonEP@0f385f038fc33bec22e3bcf5a07a8a22693e754c](https://github.com/MoonshotAI/MoonEP/tree/0f385f038fc33bec22e3bcf5a07a8a22693e754c)。
- **R**：[MoonEP 对 DeepEP v2 的 benchmark](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/benchmarks/bench_vs_deepep.py)，公开命令为 `torchrun --nproc_per_node=8 benchmarks/bench_vs_deepep.py [--out r.csv] [--plot r.png]`；[分布式正确性测试](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/tests/test_e2e.py)。
- **A**：[通信对比图](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/figure/comm_vs_deepep.png)和[端到端训练图](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/figure/e2e_vs_deepep.png)。两张 PNG 都只是作者 artifact。
- **A-data**：仓库未提交生成上述 PNG 的 CSV/逐样本计时。benchmark支持新运行时写 CSV，不等于官方图的原始 CSV 已公开。
- **Local**：`/home/chen/workspace/infra/MoonEP`，`51e64aa55310f6c6b464deabd80de2e8b5426d3f`，审计时 clean；它落后冻结上游 `0f385f03...`，不得在 manifest 中冒充冻结 SHA。上游差异审计显示 benchmark/core 未变化，但正式运行前仍须 checkout 并记录冻结 SHA。
- **Reproduced**：未复现；图中数值不是本机结果。

### 4.3 公开通信图的精确口径

| 项目 | `bench_vs_deepep.py` 可证内容 | 缺口/限制 |
|---|---|---|
| 硬件 | README声明 8×H20，单节点 EP8 | 节点内互联的精确拓扑/固件、GPU clocks、CUDA/PyTorch版本未随图冻结；README未明确列出NVSwitch |
| shape | `S=8192` token/rank、`E=384`、`H=7168`、`K=8`、MoonEP expert inner `H'=2048`、32 SM；MaxVio target `.2,1,10,20` | 只覆盖 EP8 单机；不是 2×8 RDMA workload |
| dtype | hidden BF16、route weights FP32；DeepEP v2 走 expanded dispatch/reduced combine | 不是 DeepEP README 的 FP8 dispatch/BF16 combine口径 |
| 输入 | 固定 routing seed 1234；每 rank data seed `7777+rank`；两库共享 routing/hidden/weights | 官方图对应的最终 CSV 和环境 hash未公开 |
| warmup/iters | 默认 20/50 | 只跑一次 sweep，无独立 run、CI 或噪声判定 |
| 计时 | eager CUDA-event；每个 rank 对 back-to-back iterations 求均值，再对 8 ranks 求**算术均值** | RailBalance正式口径计划使用 rank-max；跨-rank mean 与 rank-max不可直接比较 |
| API 边界 | MoonEP forward dispatch包含 inter-rank sync、planning、dispatch、fused permute、`prefetch_weight`；combine用 zero-copy view。DeepEP v2用 expanded dispatch和reduced combine | `grad_reduce`明确排除；DeepEP v2 layout以 `d_f-d_b`估算；端到端训练图没有公开运行脚本、warmup/iters或raw data |
| 正确性 | 独立 `tests/test_e2e.py`覆盖通信路径 | `bench_vs_deepep.py`本身没有在每个计时点对两库输出做 reference gate；正确性与图表结果没有 artifact lineage |

通信图不能代替 full-layer/FWD+BWD 证据。正式单机补充实验还要按
[官方 buffer合同](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/README.md#L49-L70)
和[公开 benchmark 参数](https://github.com/MoonshotAI/MoonEP/blob/0f385f038fc33bec22e3bcf5a07a8a22693e754c/benchmarks/bench_vs_deepep.py#L342-L438)
冻结训练所需的 `B=E/R=48` 和 padding 128，分别保存 grouped GEMM、prefetch、完整 backward、
`grad_reduce`、峰值 allocated/reserved memory、OOM与碎片状态；这些分项不得偷偷并回
作者的 communication-only 柱状图。

## 5. UCCL-EP

### 5.1 系统边界

UCCL-EP保持 DeepEP 风格 dispatch/combine API，以 GPU→CPU lock-free FIFO 和多线程 CPU proxy 代替 GPU 直接控制 NIC，并在 EFA、CX7、Broadcom 及 NVIDIA/AMD GPU 上实现 EP 通信。它和 RailBalance都作用于通信数据面，且 UCCL-EP自身包含 QP/NIC 负载分配，因此列为直接通信系统对手。

公平比较仍要求同一硬件、同一物理 NIC 集合、同一 route trace、同一 dispatch/combine边界和相同 SM/QP/CPU资源预算；不能拿 UCCL-EP 的 EFA/AMD论文图与本机 CX7 结果直接相除。

### 5.2 第一手证据

- **P**：[UCCL-EP arXiv:2512.19849v2](https://arxiv.org/abs/2512.19849v2)，[第 5 节评测](https://arxiv.org/html/2512.19849v2#S5)。
- **C**：冻结 [uccl-project/uccl@61ee42402819cabba3ac2a56dd4addec3363976c/ep](https://github.com/uccl-project/uccl/tree/61ee42402819cabba3ac2a56dd4addec3363976c/ep)。该提交日期为 2026-07-31，晚于论文 v2 约六个月；它只是截止日官方代码，不是论文 Fig. 8–17 的已证 source lineage。
- **R**：[官方 EP README 命令](https://github.com/uccl-project/uccl/blob/61ee42402819cabba3ac2a56dd4addec3363976c/ep/README.md)、[HT internode benchmark](https://github.com/uccl-project/uccl/blob/61ee42402819cabba3ac2a56dd4addec3363976c/ep/bench/test_internode.py)、[LL internode benchmark](https://github.com/uccl-project/uccl/blob/61ee42402819cabba3ac2a56dd4addec3363976c/ep/bench/test_low_latency.py)。
- **A**：论文 Fig. 8–17 和官方 README 的硬件结果表。
- **A-data**：未发现论文图/README表格的逐样本原始数据、CI或可校验 manifest。
- **Local**：开发 checkout仍是 `/home/chen/workspace/source_code/uccl` / `f071f2e...`；
  另已从同一 object store建立 detached clean frozen worktree
  `/home/chen/workspace/infra/uccl-ep-61ee4240`，精确 HEAD `61ee4240...`，按实验约定不
  修改（文件系统并未强制只读），大小 15,930,322 bytes。`ep/` 子树内25个 tracked
  Python文件通过 AST parse、34个 tracked shell文件通过 `bash -n`；这只是语法预检，
  不是 import、依赖检查、build、correctness或 benchmark。
- **Reproduced**：未复现；本机未构建或运行 UCCL-EP。

### 5.3 官方硬件、shape、计时与正确性

论文 Table 2 公布六类 testbed：

| 名称 | 服务器/GPU | 网络/NIC |
|---|---|---|
| NV_EFA3 | 4×(8×H200) | EFAv3 200G×16/机 |
| NV_EFA4 | 4×(8×B200) | EFAv4 400G×8/机 |
| NV_IB | 4×(8×H100) | CX7 400G×8/机 InfiniBand |
| NV_C2C_IB | 2×(1×GH200) | CX7 200G×1/机 InfiniBand |
| AMD_CX7 | 4–16×(8×MI300X) | CX7 400G×8/机 |
| AMD_BRC | 4×(8×MI300X) | Broadcom Thor-2 400G×8/机 |

`AMD_BRC` 的节点数存在论文内部矛盾：[Table 2](https://arxiv.org/html/2512.19849v2#S4)
写 4 servers，[§5.3.2/Fig. 14](https://arxiv.org/html/2512.19849v2#S5.SS3.SSS2)
写 16 servers。该项标记为 `PAPER_INTERNAL_INCONSISTENCY`，不能自行挑选其一当作
冻结事实。

官方 README 可提取的预注册形状是：

- HT：4 nodes×8 ranks，4096 tokens/rank、H7168、K8、E288；DeepSeek-V3口径为 FP8 dispatch/BF16 combine。
- LL：4 nodes×8 ranks，128 tokens/rank、H7168、K8、E288；同样是 FP8 dispatch/BF16 combine。
- intranode：8 ranks，4096 tokens/rank、H7168、K8、E256。

计时与正确性必须区分“代码能力”和“论文图口径”：

- [代码计时工具](https://github.com/uccl-project/uccl/blob/61ee42402819cabba3ac2a56dd4addec3363976c/ep/bench/utils.py#L415-L544)提供 CUDA-event `warmup=50/test=50`（丢弃首个样本，返回 average/min/max）以及默认 30 次的 Kineto named-kernel计时。
- `test_internode.py`的调优主路径调用 Kineto并由 node-local rank 0打印本地 best；它不是统一的全局 rank-max聚合。论文没有披露每张图最终采用的 warmup/iters、rank aggregation、独立重复或 CI，因此不能把代码默认值无条件归给论文曲线。
- [论文 Fig. 8](https://arxiv.org/html/2512.19849v2#S5.SS2.SSS1)的每个点取 HT 与 LL 中较小的时延。这种逐点 mode-selection oracle 只能在 `AUTHOR_REPRO` 兼容图中复现；`COMMON_FAIR` 必须把 LL/HT 分开，或只用 tuning 集选择一次模式后冻结，不能在 confirmatory 数据上逐点挑最优。
- HT测试在 benchmark 前验证 layout、recv count、token/weight内容和 combine reference；LL测试对 FP8 combine使用 `<9e-4`、BF16使用 `<1e-5` 的差异门禁，并有多项 exact/structural assert。正式横比仍需使用一套共同 reference harness，不能只相信各自自测。
- 论文说明基线使用相同 GPU资源（与 DeepEP相同 SM数），UCCL-EP每 GPU使用4个 CPU proxy threads；后续 manifest必须同时冻结 CPU affinity/NUMA和实际线程数。
- [论文 §4](https://arxiv.org/html/2512.19849v2#S4)明确描述 CPU proxy 在 QP 间 round-robin，并跨多个 NIC 聚合带宽；实现细节被论文省略。因此 UCCL-EP本身也属于显式 QP/NIC-balancing 对手，不能只把 SABRE/fabric-lib放入该类别。

## 6. DeepEP v2

### 6.1 系统边界

DeepEP v2是 RailBalance 所基于的上游 EP transport；同树 `off` 是必须首先完成的因果基线。上游 README 宣布 v2切换到 NCCL Gin、以 `ElasticBuffer`统一 HT/LL API，并分析计算 SM/QP数。RailBalance结果必须首先对同一 checkout、同一 binary/JIT identity 下的 DeepEP v2 `off` 做成对比较。

### 6.2 第一手证据

- **P**：未发现 DeepEP v2 独立论文。DeepSeek-V3技术报告描述了前代跨节点 EP 通信，但不能替代 v2代码口径。
- **C**：冻结上游 [deepseek-ai/DeepEP@dd758caf451848bd150e1046af3d0a73e5fff38d](https://github.com/deepseek-ai/DeepEP/tree/dd758caf451848bd150e1046af3d0a73e5fff38d)。
- **R**：[官方 README](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/README.md)、[`tests/elastic/test_ep.py`](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/tests/elastic/test_ep.py)、[计时/reference工具](https://github.com/deepseek-ai/DeepEP/blob/dd758caf451848bd150e1046af3d0a73e5fff38d/deep_ep/utils/testing.py)。
- **A**：README性能表：V3风格 8K tokens、H7168、K8、FP8 dispatch/BF16 combine，列出 SM90/SM100、EP8/16/32 的 logical bottleneck bandwidth。
- **A-data**：README表格没有逐样本计时、运行 manifest、CI或 profiler原始文件。
- **Local**：本证据审计时本地 RailBalance实现/控制面检查点为 `448a727`，工作树另含
  待提交文档；正式实验必须由 manifest现场记录 HEAD、tree和patch hash。该功能分支
  不是干净的冻结上游 `dd758caf...`，不得把本地 RailBalance验证写成上游 v2复现。
- **Reproduced**：上游冻结版本的官方公开 workload未复现。当前项目的 planner/vnode/C100验证属于 RailBalance开发证据，不等同于该项。

### 6.3 口径与缺口

- README把带本地 rank traffic 的吞吐写成 **logical bandwidth**；它不是物理 NIC bytes/s。任何横比必须同时保存 user-visible logical bytes、去重后的网络 payload和 NIC counters。
- `test_ep.py`默认 CLI 是 4096 tokens、H7168、K6、E256；README性能表是 K8。复现 README形状时必须显式传参，不能依赖测试默认值。
- 测试对 reference dispatch/combine、cached/expanded路径、padding、metadata和 deterministic路径有细粒度正确性检查。
- 测试性能主要使用 Kineto named-kernel duration；README未给出表格与某条精确 argv、warmup/iters、rank聚合及 raw trace的 lineage。因此 README绝对数字只能作为作者 artifact，不能作为本机 baseline。

## 7. ECHO

### 7.1 系统边界

ECHO（Elastic Cloning for Hot Experts）在 Megatron-Core dropless MoE中识别热点专家，把权重克隆到低负载 rank 的空闲 slot，并将 overflow tokens重定向到克隆；反向再把克隆梯度归并回 home expert。其目标包括降低 rank计算尾部和静态 worst-case buffer浪费，以支持 sync-free/full CUDA Graph。

这与 RailBalance的物理 Rail选择不是同一问题。ECHO会改变专家实例和 token计算位置；RailBalance不改变专家语义或目标端点。ECHO还依赖 HybridEP做专家权重与 token通信，所以只能做分层归因或组合评估，不能用 ECHO整栈吞吐替代纯通信基线。

### 7.2 第一手证据

- **P**：[Megatron-Core MoE 技术报告 arXiv:2603.07685v2](https://arxiv.org/abs/2603.07685v2)，[ECHO章节](https://arxiv.org/html/2603.07685v2#S4.SS3.SSS7)。这是整套 Megatron-Core MoE技术报告，不是独立 ECHO论文。
- **C**：[NVIDIA/Megatron-LM draft PR #2368](https://github.com/NVIDIA/Megatron-LM/pull/2368)，冻结 PR head [`a2b16b8733bb4ced46880d6adee6f19124732991`](https://github.com/NVIDIA/Megatron-LM/tree/a2b16b8733bb4ced46880d6adee6f19124732991)。截至截止日期该 PR 标为 Draft。
- **R**：[ECHO unit test](https://github.com/NVIDIA/Megatron-LM/blob/a2b16b8733bb4ced46880d6adee6f19124732991/tests/unit_tests/transformer/moe/test_echo.py)可作为代码正确性入口；没有公开独立 ECHO性能 benchmark命令。
- **A**：论文 Fig. 27 是工作流示意；论文 Table 11 是开启完整优化栈后的 GB300/GB200/H100整栈吞吐，不是 ECHO独立消融。
- **A-data**：没有 ECHO独立 latency/throughput的 raw data。
- **Local**：本机两个 Megatron-LM checkout均未包含冻结 PR head；没有可声明的 ECHO冻结本地版本。
- **Reproduced**：未复现。

### 7.3 不可归因边界

技术报告性能章节启用了 FP8、memory优化、HybridEP/DeepEP、通信 overlap、Grouped GEMM、fusions和 CUDA Graph等完整优化栈，并对结果使用 force-balanced routing。报告没有 `ECHO on/off` 隔离表。因此：

- 不能把 Table 11 的 TFLOPS/token/s归因给 ECHO；
- 不能从整栈表反推 ECHO planner、clone transfer或gradient reduce latency；
- 不能把 force-balanced workload当作自然不均衡 workload；
- 后续若把 ECHO作为 anchor，必须先补独立 API边界、正确性、warmup/iters、rank-max和原始路由 trace。

## 8. EPLB

### 8.1 系统边界

EPLB根据估计的专家负载复制重载专家并做启发式 placement。hierarchical策略先把专家组放到 node，再在 node内复制/装箱；global策略忽略 expert group全局放置。上游明确说负载预测不在仓库范围内，常见来源是历史统计移动平均。因此 EPLB是较慢时标的专家布局器，不是实时 Rail出口调度器。

### 8.2 第一手证据

- **P**：[DeepSeek-V3 技术报告 arXiv:2412.19437v2](https://arxiv.org/abs/2412.19437v2)的部署章节描述周期性（例如每10分钟）依据在线统计调整冗余专家；它不是 EPLB代码的独立性能论文。
- **C**：冻结 [deepseek-ai/EPLB@d52c72d5b2f2fb4c41afbf8eb21366820239913d](https://github.com/deepseek-ai/EPLB/tree/d52c72d5b2f2fb4c41afbf8eb21366820239913d)，核心为 [`eplb.py`](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913d/eplb.py)。
- **R**：[README示例](https://github.com/deepseek-ai/EPLB/blob/d52c72d5b2f2fb4c41afbf8eb21366820239913d/README.md)给出2 layers×12 experts、16 replicas、2 nodes×4 GPUs的 `rebalance_experts`调用；没有性能 runner。
- **A**：仓库 `example.png`仅解释 placement输出，不是性能图。
- **A-data**：无。
- **Local**：未发现冻结 EPLB checkout。
- **Reproduced**：未复现。

### 8.3 缺口

官方仓库没有硬件、CUDA版本、dtype、warmup/iters、计时方法、rank聚合、端到端吞吐或 raw artifact。输入是估计负载而不是固定 router trace；比较时必须同时冻结负载统计窗口、更新周期、replica预算和placement迁移成本。没有这些字段时，EPLB只能作为算法相关工作，不能进入通信 latency主表。

## 9. LPLB

### 9.1 系统边界

LPLB扩展 EPLB：先按历史负载选择/重排冗余专家，再针对当前 batch用 GPU单-SM interior-point LP solver求 token重定向。它通过 cube/hypercube/torus等静态副本拓扑做动态负载平衡，关注的是 expert/rank token load，不是 NIC/Rail byte load。

### 9.2 第一手证据

- **P**：未发现独立论文；官方 README明确称项目仍处于 early research stage。
- **C**：冻结 [deepseek-ai/LPLB@0490f79452f7ef277e814449600b1b1dd4c663b3](https://github.com/deepseek-ai/LPLB/tree/0490f79452f7ef277e814449600b1b1dd4c663b3)。
- **R**：[README安装/pytest入口](https://github.com/deepseek-ai/LPLB/blob/0490f79452f7ef277e814449600b1b1dd4c663b3/README.md)、[2 nodes×8 GPUs EP16脚本](https://github.com/deepseek-ai/LPLB/blob/0490f79452f7ef277e814449600b1b1dd4c663b3/scripts/run_ep16_cube8p2e.py)、[`test_solve.py`](https://github.com/deepseek-ai/LPLB/blob/0490f79452f7ef277e814449600b1b1dd4c663b3/tests/test_solve.py)。
- **A**：README只给“intra-node solver约100 µs，inter-node更长”的文字约数，不是带环境和样本的 benchmark artifact。
- **A-data**：无。
- **Local**：未发现冻结 LPLB checkout。
- **Reproduced**：未复现。

### 9.3 可冻结口径与缺口

- 分布式脚本明确要求 2×8 GPUs，E256、EP16、4 redundant/rank，seed `42+rank`；对 cube与hypercube拓扑各跑一次，并用 1000 次 Kineto调用打印 `kernel_solve`平均时间。
- 单卡 `test_solve.py`覆盖 E256、EP16/32/64、多种 topology和不同 replica预算，以最终 rank load `max/mean`阈值做算法门禁。
- `test_idx_processing.py`用 `(4096,8)` indices验证 workload count，并检查映射合法性和误差。
- 官方未冻结 GPU型号、CUDA/cuSolverDx/cuBLASDx版本、时钟、warmup、跨rank聚合、逐样本分布或完整 dispatch/combine/GEMM端到端边界。
- README承认 solver只平衡 token总数，未建模 grouped GEMM非线性；极端全局不均衡时可能差于 EPLB。这一限制必须保留，不能用 planner的 `max/mean`代替最终 step time。

## 10. NCCL EP

NCCL EP 是 NCCL Device API 上的原生 `ncclEpDispatch`/`ncclEpCombine`，同时提供
decode 用 LL 和 prefill/training 用 HT。LL 使用直接 RDMA+NVLink mesh；HT 先在
NVLink domain 聚合，再跨节点传输；这种“节点内聚合后跨节点”的阶段划分与
RailBalance存在可比较的数据路径成分，但实现和API边界并不相同。

- **P**：[NCCL EP arXiv:2603.13606v3](https://arxiv.org/abs/2603.13606v3)。论文只公布
  LL 性能，并明确把 HT 优化和 Megatron 训练结果留作后续工作。这是
  **论文当时**的边界，不能转述成后续 release 仍未优化 HT。
- **C/R**：截止日官方 artifact 为
  [NVIDIA/nccl `nccl-ep-v0.1.0` / `63cf786...`](https://github.com/NVIDIA/nccl/tree/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep)。tag 晚于论文，论文没有冻结生成图表的精确 commit，二者不能冒充同一 lineage。
  [v0.1官方发布页](https://github.com/NVIDIA/nccl/releases/tag/nccl-ep-v0.1.0)已列出
  HT优化及 `>8` nodes HT修复，所以这两项必须与论文中的 future-work描述分开记录。
- **口径**：论文 LL 为 E256、H7168、K8、128 token/rank、BF16、8–64 GPU。硬件是
  EOS H100 80 GB、8 GPU/node、第四代 NVLink 900 GB/s、Xeon 8480CL 和每节点
  8×400 Gb/s InfiniBand，实验覆盖 1–8 nodes。论文用包含 launch 的 NCCL EP host C API
  时间对比 DeepEP Kineto GPU kernel 时间；这不是共同计时。正式实验必须把 LL/HT
  分开，改用同一 public API 边界、逐 iteration rank-max，并保存 raw samples。
- **release 与公开覆盖边界**：[v0.1官方发布页](https://github.com/NVIDIA/nccl/releases/tag/nccl-ep-v0.1.0)
  指出 CUDA 12用户需要从源码构建；[同 tag EP README prerequisites](https://github.com/NVIDIA/nccl/blob/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep/README.md)
  则把 prerequisites 和当前 Python `nccl4py` 支持列为 CUDA 13+。官方文本本身存在
  不一致；CUDA 12.8本机只能标为“source build待验证”：既不能把 README/wheel约束
  推广为源码绝对不支持，也不能在真正编译前声称可用。
  release 当前不支持 quantization，但这不等于可以笼统声称
  “BF16-only”。公开 reference/performance 表的实测覆盖至多到 8 nodes；即使 release
  修复了 `>8` nodes HT，也不能把该修复当成 `>8` nodes 公开性能证据。论文
  vLLM 结果同样不能改写成“NCCL EP 已胜出”。
- **tag 表格 lineage**：[v0.1 README 的性能表](https://github.com/NVIDIA/nccl/blob/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep/README.md#L307-L328)
  明确注明来自 NCCL v2.29u1，只覆盖 LL BF16、最多 8 nodes；release 的 HT优化又依赖
  后续 Device API。该表不是 `63cf786...` 当前 tag 二进制的完整性能 lineage。
- **本地 benchmark边界**：tag 的
  [`ep_bench`](https://github.com/NVIDIA/nccl/blob/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep/ep_bench.cu#L1797-L1906)
  默认 10 warmup/50 iterations，但先求每个 rank的均值，再做跨 rank算术平均，并不
  产生逐 iteration rank-max分布。其
  [kernel-only throughput](https://github.com/NVIDIA/nccl/blob/63cf786b015b2b6bff6cf263461621acf584bd18/contrib/nccl_ep/ep_bench.cu#L2185-L2202)
  还把 minimum latency对应列标成 minimum throughput、maximum latency对应列标成
  maximum throughput，数学方向相反；这些字段不能进入共同图。
  `COMMON_FAIR` 必须保存每个 rank每次 iteration的原始时延并自行计算 rank-max。
- **API边界**：[论文的 handle/layout 生命周期](https://arxiv.org/html/2603.13606v3#S3.SS2)
  要求把 route/layout/handle create或update、`Dispatch(+Complete)`、
  `Combine(+Complete)`、group creation/JIT分层记录。只计 Dispatch/Combine会在
  HT或staged模式隐藏 metadata exchange、handle update和 completion成本；正式表同时
  给出 steady data-plane 与包含每层控制开销的边界。
- **serving外部有效性**：[论文 §VII-B/Table VII](https://arxiv.org/html/2603.13606v3#S7.SS2)
  使用 Qwen3-30B-A3B、1/2/4 nodes、1000 requests、maximum concurrency 32，每个
  backend 4 runs并做 IQR filtering。它没有公开逐请求 raw data/CI，只能作为独立
  `AUTHOR_REPRO`/外部有效性轨，不能替代 transport共同主矩阵。
- **Local**：`/home/chen/workspace/infra/nccl-ep-v0.1.0` 是精确 tag
  `nccl-ep-v0.1.0` / `63cf786...` 的 detached clean checkout，大小18,028,459 bytes。
  本机 `mpicc`/`mpirun`可见，nvcc为 CUDA 12.8 / V12.8.61 / build 35404655；这只证明
  前置工具可见，尚未证明 NCCL EP可构建，也未运行 correctness或启动任何 GPU。

### 10.1 UCCL/NCCL共享本机静态预检状态

check-only入口与完整启动合同见
[`HOP_AWARE_EXPERIMENT_PROTOCOL.md` §8.1](HOP_AWARE_EXPERIMENT_PROTOCOL.md)。它冻结
UCCL `61ee4240...` 的 `ep+include` build-input closure和NCCL `63cf786...` 的完整
tracked tree，future build只允许从 pinned Git objects物化到 fresh stage；冻结
checkout、ignored/untracked working-tree文件、Git conversion filter与lazy fetch均不
进入构建输入。工具、future argv/env/output、依赖路径、GPU/NIC/topology和磁盘预算也
逐项 fail-close。

当前权威记录为
`.cache/rail_balance/competitors/preflight-20260801-08/result.json`，mode `0444`，
SHA256 `8e46f41c081e7838aecfaf7a50c6a64ef4d3b73e23f9d872afb454f2d3e907d5`。
它使用 root-owned系统 Python、exact empty-env argv和113文件 runtime closure，结果为
`BLOCKED_DEPENDENCY_NANOBIND`，并同时记录 Qwen PID `3047501/3047502`造成的
`WAITING_GPU`。source closure、工具、磁盘、GPU mapping、NV18/PIX、8个 NIC sysfs和
`ibv_devices`静态枚举通过；`ibv_devinfo`、build、correctness、benchmark和GPU kernel
全部 `NOT_RUN`。future build executor明确为`NOT_IMPLEMENTED`，因此这不是
`Reproduced`，也不证明 verbs/RDMA runtime ready。`preflight-20260801-01`至`-07`均为
`SUPERSEDED_NON_AUTHORITATIVE`，不得用 glob消费或引用其状态。

## 11. SABRE

SABRE 与 RailBalance在 proxy/NIC分流、节点内 gather/distribute和保持逻辑 endpoint
方面有明确算法重叠。它读取完整 rank-to-rank byte/count 矩阵；偏斜
较大时，把源节点到目标节点的数据块贪心分给本地 proxy GPU/NIC，必要时拆块，经源端
NVLink gather、直接跨节点、目标端 NVLink distribute/reassemble。逻辑 rank/expert
endpoint 不变。

- **P**：[ICS'26 DOI](https://doi.org/10.1145/3797905.3800541)与
  [作者论文/项目页](https://cunyangwei.github.io/assets/paper_web/alltoallv/)。
- **C/R**：冻结
  [hpcgroup/sabre@b3f2c5a3...](https://github.com/hpcgroup/sabre/tree/b3f2c5a3be05410145285b3feebb6d7cde46a0b9)，无 tag。
- **差异边界**：SABRE 是通用 `all_to_all_single`/AllToAllv，依赖完整通信矩阵和多阶段
  packing/reorder；RailBalance 位于 DeepEP fused dispatch/combine 内，在 route 已定后
  选择 rail/vnode 并限制本地 forwarding。SABRE 不读取物理 fabric path telemetry；其额外
  gather/distribute 发生在源、目标两端节点内，proxy GPU 之间的跨节点传输仍直达
  目标节点，不再经第三个网络节点/rank 中继。这个差异不能支持“首次用
  NVLink/local forwarding 做 NIC
  负载均衡”的表述。
- **作者口径**：Perlmutter，4×A100/node、4×25 GB/s Cassini NIC/node、16–256 GPU；
  论文报告 10 个独立 trial 的均值。作者结果不是全域收益：轻偏斜、小消息和大规模存在
  明显回退点。
- **公共 harness 缺口**：代码 benchmark 是 BF16、5 warmup/20 timed，但只记录 rank 0
  本地时延；`--test` 默认关闭，helper 还容许最多 2% 元素不 close。共同实验必须换成
  独立 exact/reference gate 和 rank-max，不能直接沿用这套正确性与统计。

## 12. fabric-lib / pplx-garden

fabric-lib 通过 CPU proxy 管理 RDMA，并把每 GPU 的 1–4 个 NIC 做 sharding/rotation，
与 RailBalance 的多 NIC/rail 分配直接重叠。

- **P**：[arXiv:2510.27656v2](https://arxiv.org/abs/2510.27656v2)。
- **C/R**：冻结
  [perplexityai/pplx-garden@2446003...](https://github.com/perplexityai/pplx-garden/tree/244600375fe93f136103e0e44a1618cf332a03dc)；旧 `pplx-kernels` 已废弃，不再作为当前 artifact。
- **anchor**：decode `<=128` token/rank 与 prefill 4096 token/rank 分开，H7168、K8、
  dispatch FP8、combine BF16。作者代码使用 10k warmup+10k measured 后池化各 rank event
  样本，不是逐 iteration rank-max；正式共同口径必须冻结 CPU cores/NUMA/proxy/NIC
  集合并重做 rank-max。作者结果中也存在 DeepEP 更快的 prefill/小规模点，禁止写成
  全面胜出。

## 13. SwiftEP

SwiftEP 直接扩展 DeepEP internode prefill dispatch/combine，使用 buffer fusion、RDMA
SGL、多 QP、CUDA IPC 和 TMA multicast/reduce 减少 staging copy 与节点内 SM 开销。
它不改变 expert endpoint，和 RailBalance 既是直接传输对手，也有组合空间。

- **P**：[NSDI'26 页面](https://www.usenix.org/conference/nsdi26/presentation/li-xingyi)与
  [官方 PDF](https://www.usenix.org/system/files/nsdi26-li-xingyi.pdf)。
- **C/R**：DeepEP `tencent-zcopy` 分支截止日
  [4bd6a8b...](https://github.com/deepseek-ai/DeepEP/tree/4bd6a8b1d6ecf9daa76b98804e31c2c6712b577d)，以及仍开放的
  [PR #453](https://github.com/deepseek-ai/DeepEP/pull/453)（PR head `6db603e...`）。论文
  没有声明图表精确 SHA，分支与 PR head 只能作为截止日代码，不能冒充论文 artifact。
- **anchor**：2/4 nodes、16/32×H20，每 GPU 2×200 Gb/s CX-7；H7168、K8、
  2K/4K/8K token，FP8/BF16 dispatch、BF16 combine。论文 algorithm bandwidth 使用
  “通信字节/对应 kernel 时间”，没有冻结 warmup、iters、rank aggregation、时钟或 raw
  samples；它不等于 public API latency。
- **共同实验**：只有先把 SwiftEP 和 RailBalance 合并/移植到同一个冻结
  源码基座，共享唯一 `off/off` 对照且两个 feature flag 能独立开关后，才能做
  `SwiftEP off/on × Rail off/on` 2×2；同时冻结 fused-buffer 合同、SM、QP、NIC、最终
  token trace、正确性和计时起止点。若无法建立这一共同源码基座，只能分别报告
  SwiftEP-vs-own-off 与 Rail-vs-own-off，不得估计因子交互。

## 14. UBEP

UBEP 面向 Huawei CM384/Ascend C，以 Data-as-Flag 替代全局 barrier，并按一跳/两跳
距离把细粒度 token 发送任务分给 AIV communication cores。它不改变 expert endpoint，
但改变发送任务承担者，是“hop-aware token scheduling”非常直接的先行工作。

- **P/A**：[arXiv:2607.06202v2](https://arxiv.org/abs/2607.06202v2)与
  [SIGCOMM'26 DOI](https://doi.org/10.1145/3789240.3829183)。
- **C/R/A-data**：截止日无公开代码、命令或 raw samples，不能构造 NVIDIA/H100 上的
  `COMMON_FAIR` 数值行。
- **边界**：UBEP 调度的是 CM384 unified-bus 上的 AIV core 与 UB hop，不是 NVIDIA
  RDMA 多 NIC rail/vnode；但硬件边界不支持“首次 hop-aware token scheduling”。其 256
  Ascend NPU dies 结果不能和本项目 H100/H800 数字直接相除。

## 15. UniEP

UniEP 论文描述融合 dispatch+GroupGEMM 与 GroupGEMM+combine，并用 relay
worker 去重同一目标 rank 上的多 expert selection。本账本冻结的 `1512a81...`
`ep_overlap` 公开入口只能支撑单节点前向融合段，不是纯 transport 基线。

- **P**：[arXiv:2604.19241v1](https://arxiv.org/abs/2604.19241v1)。
- **C/R**：截止日可冻结
  [ByteDance-Seed/Triton-distributed@1512a81...](https://github.com/ByteDance-Seed/Triton-distributed/tree/1512a81189315eca7ba38f884aaf239469a88ba3)，但无 UniEP tag 或论文 commit。
- **可验证的公开范围**：冻结提交的 `ep_overlap` 是 intra-node；公开
  dispatch+GEMM 和 GEMM+combine 前向测试包含真实 Torch/CUDA reference，并对
  支持输出做严格近似或 bitwise assert。因此可先对 `EP=2/4/8` 做前向
  build/API/correctness，不得再写成“PyTorch comparison TODO/精度恒为 true”。
- **未公开入口**：截止日未找到可组成完整训练 FWD/BWD 的公开 backward
  harness，也未找到 inter-node rail/RDMA 入口。因此公开前向段只进入单节点
  integrated 赛道；完整 FWD/BWD 记 `API_GAP`，transport-only Rail 指标另表。

## 16. FEPLB

FEPLB 的第一阶段继续使用 DeepEP 等现有跨节点 dispatch，第二阶段用 Hopper Copy
Engine 在节点内复制动态 expert 权重和 token，在欠载 GPU 执行。它改变动态 expert 的
物理执行 GPU，但保持跨节点 EP path，优化的是 GEMM/计算尾部而非 NIC/Rail 热点。

- **P/A**：[arXiv:2604.19654v1](https://arxiv.org/abs/2604.19654v1)。截止日无公开代码、
  命令或 raw samples。
- **边界**：只能比较完整 layer/step，并显式计入权重复制、显存、CPU scheduler 和
  backward；不能把作者 layer ms 与 RailBalance kernel us 相除。代码发布后可做
  `RailBalance phase-1 transport + FEPLB phase-2 compute rebalance` 组合实验。

## 17. 可比与不可比边界

### 17.1 在补齐统一 harness 后可以比较的对象

1. **DeepEP v2 `off` 与 RailBalance模式**：同一源码/JIT、同一硬件、同一 route tensor、同一 API边界下的直接 paired A/B。这是内部主基线。
2. **P0-R transport**：NCCL EP、SABRE、SwiftEP、UCCL-EP 和 fabric-lib 都必须统一
   GPU/NIC、CPU proxy资源、SM/QP预算、dtype、layout、route、warmup/iters、rank-max
   和物理 bytes；若 SABRE 的 packing/AllToAllv API 无法建立同边界 adapter，记
   `NO_COMMON_API`，不能硬造一列。
3. **UltraEP 与 RailBalance的组合关系**：只能分层记录 expert-placement/reroute收益和 Rail路径收益。架构分层只说明组合值得验证，不能预先证明收益正交；必须用共享基座2×2实验检验主效应与交互。任何结果都要标明两层是否开启，不能把其中一层的收益归给另一层。
4. **MoonEP单机通信**：只在同一 8×H20/NVLink机器、相同路由与相同 API边界下形成独立单机赛道；它不能证明多节点 Rail收益。
5. **UniEP/FEPLB/UltraEP/ECHO/EPLB/LPLB**：可比较 planner质量、最终 expert/rank
   load、复制/迁移成本和整栈 step time，但不能直接作为 NIC/Rail transport latency
   的同类替代品。

以上只是“若进入后续 manifest，什么条件下语义可比”的边界，不代表已经选定最终论文实验。

### 17.2 明确不可比较或不可直接下结论的情况

- 不同机器、不同 GPU/NIC、不同 NVLink/RSN/EFA/IB拓扑上的论文绝对数字；
- MoonEP的跨-rank mean与 RailBalance计划的 rank-max；
- PNG/论文表格与本机 raw samples；
- FP8 dispatch/BF16 combine与全 BF16；
- 单卡 planner、8卡 vnode或 private checked adapter与公开端到端 dispatch/combine；
- profiler内 kernel duration与 profiler-free public API latency；
- logical bandwidth分子不同、或把包含local traffic的 logical GB/s当作 NIC line rate；
- MoonEP排除 `grad_reduce`的通信图与包含完整训练反向的step time；
- UltraEP/ECHO/EPLB/LPLB改变专家实例或route后的整栈结果，与冻结目标端点的transport-only结果混在一列；
- ECHO完整优化栈表格与 ECHO单项收益；
- 未固定commit、环境、命令、正确性与原始样本的“复现”。
- SABRE 的 rank0、本地容差 helper，fabric-lib 的跨 rank 样本池化，NCCL EP 的
  host-C-API/Kineto 混合边界，或 SwiftEP kernel bandwidth，直接和共同 public-API
  rank-max 相除；
- UBEP 的 Ascend/CM384 结果与 NVIDIA/H100 结果横比，或把 UniEP/FEPLB full-layer
  延迟当作 dispatch-only latency。

## 18. 后续 manifest 的 competitor anchors（官方口径预注册输入）

下表只把官方公开值转成待冻结字段，便于脚手架生成 manifest。它们是**预注册输入候选**，不是已经决定的论文实验，也不是本机已验证配置。

| ID | 固定代码 | 官方 shape/负载 anchor | 官方硬件约束 | 官方 timing anchor | manifest 必须补齐 |
|---|---|---|---|---|---|
| `ultraep-v1.0.0-demo-ep8` | `94cab099...` | E32、K4、20 layers；官方8×Hopper demo | 8×Hopper + NVLink + NVSHMEM；非论文RSN | demo入口，只作功能/整层轨 | exact dataset/argv、CUDA/NVSHMEM、route hash、分项/raw samples |
| `ultraep-v1.0.0-ep64-microbench` | `94cab099...` | EP64、E256、K8、8192 token/rank、2 redundant/rank；rank max/mean 约1.5/2/3；BF16 weight、FP32 grad | SM90/100 + NVLink + NVSHMEM；论文 RSN硬件型号未知 | test默认10/30；CUDA-event带barrier，Kineto comm取rank-max | GPU/NIC型号、CUDA/NVSHMEM、exact argv、route hash、API边界、raw samples |
| `moonep-h20-ep8-vs-v2` | `0f385f03...` | EP8、S8192/rank、E384、H7168、K8、H'2048、MaxVio .2/1/10/20、BF16 | 单机8×H20；互联精确拓扑未公开 | 20/50 eager CUDA-event，跨rank mean；排除grad_reduce | CUDA/PyTorch、clocks、rank-max补测、correctness lineage、CSV/raw samples |
| `uccl-ep-ht-ep32` | `61ee4240...` | 4096 token/rank、H7168、K8、E288、FP8 dispatch/BF16 combine | 论文六类testbed且AMD_BRC节点数自相矛盾；本项目最近的是 NVIDIA+CX7 IB而非EFA/AMD | 代码有50/50 event和30-test Kineto；论文Fig.8逐点取HT/LL最小值 | exact GPU/NIC、proxy affinity、SM/QP、固定mode、rank-max、API边界、route/raw hash |
| `uccl-ep-ll-ep32` | `61ee4240...` | 128 token/rank、H7168、K8、E288、FP8 dispatch/BF16 combine | 同上 | 同上；COMMON_FAIR不得逐点挑HT/LL | 同上，另冻结LL buffer清理与zero-copy/logfmt选项 |
| `nccl-ep-ll-ep8-64` | `63cf786...` / tag `nccl-ep-v0.1.0` | 论文：E256、H7168、K8、128 token/rank、BF16 | 论文：EOS H100、8×400G IB/node、8–64 GPU；官方CUDA12/13文本冲突 | 论文 NCCL host API vs DeepEP Kineto；tag `ep_bench`跨rank mean，不可直接用 | 分层handle/update/Complete与data-plane、逐iteration rank-max、exact tag/build、correctness/raw samples |
| `nccl-ep-ht-prefill` | `63cf786...` / tag `nccl-ep-v0.1.0` | `>=4096` token/rank；使用三方共同非量化dtype | release已含HT优化与`>8` nodes修复；公开表仍只覆盖≤8 nodes | 论文无 HT 性能结果；release修复不是性能数据 | 本地共同口径首次实验，不能称作者复现 |
| `sabre-skew-alltoallv` | `b3f2c5a...` | BF16；高/低 skew；64–512 MB/rank | Perlmutter，4×A100+4×Cassini NIC/node，16–256 GPU | 论文10 trials mean；代码5/20且rank0 local | exact matrix/packing、rank-max、严格reference、API adapter或`NO_COMMON_API` |
| `fabric-lib-decode-prefill` | `2446003...` | H7168、K8；decode<=128与prefill4096；FP8/BF16 | 每GPU 1–4 NIC；CPU proxy | 10k/10k后pool ranks，不是rank-max | CPU affinity/NUMA、NIC集合、逐iteration rank-max、raw samples |
| `swiftep-prefill` | branch `4bd6a8b...`; PR head `6db603e...` | H7168、K8、2K/4K/8K；FP8/BF16 dispatch、BF16 combine | 2/4 nodes，16/32×H20，2×200G CX-7/GPU | algorithm bytes/kernel time；其余未冻结 | 精确代码lineage、fused buffer、SM/QP/NIC、public API rank-max |
| `deepep-v2-v3like` | `dd758caf...` | README：8K token、H7168、K8、FP8 dispatch/BF16 combine | SM90/SM100；NVLink + RDMA；NCCL Gin | README未给表格的完整计时lineage | exact CLI、EP拓扑、SM/QP生效值、JIT/SASS hash、rank-max/raw samples |
| `echo-draft-pr2368` | `a2b16b87...` | 无独立性能shape；单测只作为正确性入口 | 技术报告整栈GB200/H100，不能归因给ECHO | 无独立benchmark | 独立API、自然不均衡route、clone bytes、planner/transfer/grad边界、on/off raw |
| `eplb-static-placement` | `d52c72d5...` | README示例：2 layers×12 experts、16 replicas、2 nodes×4 GPUs | 未给 | 无 | load-history窗口、更新周期、replica预算、placement迁移成本、端到端边界 |
| `lplb-ep16-cube-hypercube` | `0490f794...` | E256、EP16、4 redundant/rank；2×8 GPU脚本；另有EP16/32/64单卡算法case | GPU型号未知；CUDA>=12.6.3 + MathDx | 分布式脚本1000次Kineto；README约100 µs | warmup、rank聚合、CUDA/MathDx、raw samples、dispatch/GEMM/step time |
| `uniep-intranode-forward` | `1512a81...` | `EP=2/4/8`公开dispatch+GEMM、GEMM+combine前向段 | 单节点；未找到internode或完整训练backward入口 | 公开测试有Torch/CUDA reference与严格/bitwise assert | build/API/correctness分EP记录；完整FWD/BWD单独记`API_GAP` |
| `ubep-cm384-evidence-only` | 无代码 | CM384一/两跳AIV token scheduling | 256 Ascend NPU dies | 论文结果 only | `NO_PUBLIC_ARTIFACT`；只做novelty/设计差异，禁止跨硬件speedup |
| `feplb-composed` | 无代码 | DeepEP phase-1 + CE动态expert phase-2 | H100，最多16 GPU | full layer/step，细节缺失 | `NO_PUBLIC_ARTIFACT`；发布后冻结复制/显存/planner/backward成本 |

每个 anchor 落到正式 manifest 时，还必须统一增加：

- `source_url`、完整 SHA、dirty patch hash、build flags、binary/JIT/SASS hash；
- GPU UUID、PCIe/NVLink/NIC/NUMA拓扑、driver/CUDA/PyTorch/NCCL/NVSHMEM版本；
- exact route tensor及 SHA256、tokens/experts/top-k/hidden/dtype/layout/stride/alignment；
- public API起止点、stream、同步、warmup、steady iterations、独立runs和rank聚合；
- reference实现、误差阈值、失败shape/seed、非确定性检查；
- logical payload、deduplicated network payload和per-NIC physical bytes三套分子；
- 全部raw samples、环境采样、profile lineage和artifact hashes。

在这些字段补齐并完成 `Reproduced` gate 前，本账本只支持“选择和冻结对手”，不支持“我们已经打平或超过”的论文陈述。

论文 novelty 也必须使用窄边界：**在 DeepEP/NVIDIA 多 NIC RDMA 数据面内，保持最终
expert endpoint 不变，进行带 hop/cost 约束的 rail/vnode 选择与有限节点内
forwarding。** 禁止写“首次用本地 NVLink forwarding 做网络负载均衡”（SABRE已覆盖）、
“首次 hop-aware token scheduling”（UBEP已覆盖）或“首次多 NIC/Rail 分配”
（fabric-lib、UCCL-EP等已有直接机制）。
