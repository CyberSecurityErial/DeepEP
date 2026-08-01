# RailBalance 论文性能实验清单（2026-08-01）

> 原则：本机只优化 plan CUDA 算子时间；真实多节点性能与真实任务在外部集群执行。
> 当前工作只记录外部实验的关键环境依赖，不安装或搭建这些环境。

详细版本与作者入口证据见
[`COMPETITOR_EVIDENCE_2026-08-01.md`](./COMPETITOR_EVIDENCE_2026-08-01.md)，
长版候选池见
[`PAPER_EXPERIMENT_MATRIX_2026-08-01.md`](./PAPER_EXPERIMENT_MATRIX_2026-08-01.md)。

## 实验 A：Plan 算子性能优化

**验证什么**

验证 hop-aware RailBalance 的 plan CUDA 算子在保持路径选择结果完全正确的前提下，
能否显著降低规划时间。这是当前单机的主要优化任务。

**和什么对比、比较什么指标**

- 对比当前 clean Leader `8c56939`、当前功能版本 `286da0a` 和后续每轮 2--4 个
  单假设候选，不与通信竞品比较。
- 主指标：相同输入下 plan transaction 的 rank-max latency。
- 辅助指标：p50/p95/p99、抖动、kernel duration、寄存器、occupancy、指令数和主要
  warp stall；同时确认 plan 输出与 CPU reference exact 一致。
- workload 保留三个代表：production-like volume、singleton/rotating 边界和 balanced
  control。

**关键环境依赖**

- 本机独占 8 GPU（SM90/H200 目标）；任一 GPU 有其他 compute 任务时不运行。
- CUDA 12.8、PyTorch 2.11.0+cu128、Nsys 2024.6.2、NCU 2025.1.1。
- 当前 DeepEP 分支、固定 benchmark/reference、10 warmup + 100 steady 和相同 GPU 时钟。
- 已有入口：`tests/elastic/bench_rail_balance_hop_plan.py`；正式候选仍走现有
  source-round 控制面与 Leaderboard。

## 实验 B：多节点 EP 通信微基准

**验证什么**

验证 plan 开销降低后，RailBalance 在真实多 Rail 网络上能否降低热点条件下的
dispatch/combine 尾延迟，同时在均衡流量下不产生明显回退。

**和什么库对比、比较什么指标**

- 对比 same-tree DeepEP `off`、clean DeepEP v2、NCCL EP、UCCL-EP 和 RailBalance best。
- 使用 LL 128 token/rank 与 HT 4096 token/rank，E256/H7168/K8/BF16；至少覆盖
  `balanced` 和 `rail_hot` 两种 route。
- 主指标：dispatch、combine、roundtrip 的逐 iteration rank-max p50/p95/p99，以及
  有效带宽。
- 归因指标：plan time、每 Rail bytes、NIC/QP counter、本地 forwarding bytes，以及
  GPU/CPU/显存资源开销。

**关键环境依赖**

- 至少 2 nodes x 8 GPUs/node；资源允许时增加 4 nodes x 8 GPUs/node 扩展点。
- 每 GPU 对应多 Rail IB/RoCE NIC，GPU Direct RDMA 可用，GPU/NIC/NUMA 拓扑可查询。
- 冻结的 DeepEP、NCCL EP `v0.1.0`、UCCL-EP `61ee4240...` 代码版本，以及一致的
  驱动和硬件；各库单独保存其 build/runtime closure。
- 所有系统使用相同输入、物理目标、资源预算、timed window 和原始逐 rank 计时格式。
- 已核对的最短入口分别是 DeepEP `tests/elastic/test_ep.py`、NCCL EP
  `contrib/nccl_ep/ep_bench`、UCCL-EP `ep/bench/test_internode.py`。关键版本要求是
  DeepEP 的 SM90/CUDA>=12.3/PyTorch>=2.10/NCCL>=2.30.4，NCCL EP 的
  Hopper/CUDA 13/NCCL>=2.29/MPI，以及 UCCL 对应 CUDA wheel、Python binding 和 RDMA
  transport；正式外部运行前再冻结兼容组合。

## 实验 C：真实 MoE 工作负载

**验证什么**

验证 plan 算子和通信微基准的收益能否转化为真实任务吞吐或延迟收益，而不是只在合成
route 上成立。首版只选一个最容易共同接入的真实框架，不同时维护多套环境。

**和什么库对比、比较什么指标**

- 首选 SGLang MoE serving：对比原生 DeepEP、RailBalance，以及能在同一框架接入的
  UCCL-EP；NCCL EP 若没有同框架 adapter，不强行放入这张图。
- 主指标：请求吞吐、TTFT、TPOT、端到端 p50/p95/p99。
- 分项指标：同一次运行中的 plan、dispatch、combine 时间、通信占比和显存峰值。
- 固定一个公开 MoE checkpoint 和一份请求 trace，固定并发、输入/输出长度、sampling
  参数和 seed。

**关键环境依赖**

- 与实验 B 相同的外部多节点 GPU/RDMA 集群。
- 冻结版本的 SGLang、模型 checkpoint、tokenizer、DeepEP/RailBalance/UCCL adapter。
- 足够的共享或本地模型存储，以及版本化请求 trace；各 backend 使用相同模型与请求。
- UCCL 冻结树已有 `ep/bench/sglang/` 启动入口；首选基于该入口冻结一套 SGLang 和
  Qwen3/DeepSeek MoE 配置，尚未由源码固定的模型、数据和 SGLang commit 标为 `PENDING`。

如果论文最终更偏训练，可把本实验等价替换为一个 Megatron MoE 训练任务：对比
DeepEP off 与 RailBalance，报告 step time、tokens/s、通信占比和显存；仍只保留一个
真实任务主环境。

## 实验 D：UltraEP / MoonEP 补充对比

**验证什么**

验证 RailBalance 与专家 placement/replication 方案是竞争还是互补，并界定论文的适用
范围。该实验是补充项，不阻塞实验 A--C。

**和什么库对比、比较什么指标**

- UltraEP：比较 placement off/on 与 RailBalance off/on；指标为 MoE layer latency、
  token throughput、load balance、迁移/复制开销和额外显存。
- MoonEP：在其支持的单机环境比较通信/placement latency、rank-max tail 和 full-layer
  time；不把单机数字与多节点 Rail 微基准直接相除。

**关键环境依赖**

- UltraEP 需要与其公开规模和 transport 匹配的 Hopper/RSN 或等价环境；8 GPU demo
  只能做功能与组合开销，不能代替 EP64 性能；代码冻结为 `94cab099...`。
- MoonEP 需要匹配的单机 8xH20、冻结 MoonEP `0f385f03...`/DeepEP 版本和相同模型
  shape。
- 若拿不到匹配硬件，该实验保留为 `NOT_RUN`，不影响主要论文结论。

## 当前状态

- 实验 A：等待本机独占 8 GPU 后重建 clean 基线并继续 CUDA 候选闭环。
- 实验 B--D：只编写环境依赖和运行口径文档；本机不安装、不构建、不执行。
- 当前没有新的 plan 性能 Leader，也没有本机竞品性能结果。
