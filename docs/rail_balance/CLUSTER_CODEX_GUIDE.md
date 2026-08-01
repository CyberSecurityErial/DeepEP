# RailBalance 多节点调通、实验与策略调优指南

> 交给集群 Codex 的执行入口。本文只负责“先做什么、运行什么、何时停止”；完整证据合同见
> [DISTRIBUTED_EXPERIMENT_RUNBOOK.md](./DISTRIBUTED_EXPERIMENT_RUNBOOK.md)，当前自动调优工具合同见
> [AUTOTUNE_GUIDE.md](./AUTOTUNE_GUIDE.md)。
>
> 冻结时间：2026-08-01 UTC。开始工作时必须用现场 Git 和 `--help` 重新核对，不得盲信本文日期。

## 1. 任务与完成定义

你的任务分三步：

1. 在真实多节点 Rail/Gin 环境把同树 `off`、`one_hop`、`adaptive` 的 dispatch/combine 调通；
2. 用合成压力用例验证机制，用连续真实路由验证性能，并完成 DeepEP/NCCL EP/UCCL-EP 与
   UltraEP 组合实验；
3. 只在 tuning split 调策略，冻结一个候选后在 held-out split 复测，不在确认集追参数。

完成不是“某个 hot case 赢了”，而是下面条件同时成立：

- 端点、payload、weight、mask、ticket、combine 均正确，无丢失、重复或静默 fallback；
- `one_hop -> adaptive` 在同进程、同二进制、同连续真实窗口中直接配对；
- profiler 关闭的每-rank原始样本、rank-max 和尾延迟显示稳定净收益；
- runtime path、NIC Rail bytes 和本地 forwarding bytes 能闭环；
- balanced/low-gap 路径没有超过预注册预算的回退；
- 至少在两个真实 workload/model family 和 2x8、4x8 上复现，才能谈泛化；
- 当前 public capability 仍保持关闭；解锁必须是多机验收后的独立提交。

如果缺真实 trace、直接配对或物理 counter，分别写 `TRACE_GAP`、`API_GAP`、
`COUNTER_GAP`。这些是有效结果，不允许用 synthetic 或 Python oracle 补成“通过”。

## 2. 先认清三个提交

| 身份 | 提交 | 用途 |
|---|---|---|
| 历史两机黄金 | `e5d49d328c85aa210e170a0587a7cb318df0dfb4` | 用户已观察到性能收益的跨提交回归锚点 |
| 当前算子冻结点 | `4e002766d4be7c319687e127a69c1811efe7fe49` | hop-aware/第三 Rail 与本机 planner 优化收口点 |
| 集群执行起点 | `0d87d9510a0dab7617fbbd3b56a9a536a6dc07b6` | 在 `4e00276` 上增加离线 autotune 和文档；生产 `csrc/deep_ep` 无变化 |

远端分支：
[feat/rail-balance-autotune-scaffold](https://github.com/CyberSecurityErial/DeepEP/tree/feat/rail-balance-autotune-scaffold)。

第三 Rail 的因果对照必须使用当前同一 checkout 内的 `one_hop` 与 `adaptive`。
`e5d49d` 只做历史回归，跨提交差异不能算成第三 Rail 贡献，也不能交给当前 `freeze`。

### 2.1 只用一个主分支

当前祖先链已经现场验证：

```text
e5d49d  历史两机黄金
   -> 5199a04  feat/rail-balance-prototype
   -> 4e00276  feat/rail-balance-hop-aware
   -> 0d87d95  feat/rail-balance-autotune-scaffold
```

因此集群开发和主测试只使用最末端的
`feat/rail-balance-autotune-scaffold`。它已经同时覆盖：

1. prototype 的原有实现与 `legacy_exact` 回归路径；
2. 新的 `one_hop` 和可选 `adaptive` 第三-Rail/2-hop 路径；
3. autotune plan/freeze 工具；
4. 单机、多机、基线、竞品与端到端实验手册。

prototype 分支不删除，但不再继续开发，也不需要复制一套测试。只有做历史二分、复现旧结果
或审计旧提交时，才从对应 SHA 建独立 clean worktree。正常实验在当前分支用
`off/legacy_exact/one_hop/adaptive` 模式切换即可；历史 `e5d49d` 性能锚点另开 worktree，
结果单独成表。

第一条命令必须是：

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git merge-base --is-ancestor 4e002766d4be7c319687e127a69c1811efe7fe49 HEAD
git diff 4e002766d4be7c319687e127a69c1811efe7fe49 HEAD -- csrc deep_ep
```

预期：clean、祖先检查返回 0，最后一条为空。若现场分支已推进，记录新的 SHA，并重新证明
生产 diff；不要把旧 handoff 标头当现场事实。

## 3. 最少阅读顺序

不要从头阅读几千行日志。按下面顺序：

1. 本文；
2. [HANDOFF_HOP_AWARE_2026-08-01.md](./HANDOFF_HOP_AWARE_2026-08-01.md) 的 §1–5、§10、§16，
   只理解问题、`o/d/T/e`、四种模式、ticket 和踩坑；其分支/HEAD 状态已过时；
3. [AUTOTUNE_GUIDE.md](./AUTOTUNE_GUIDE.md) 全文；
4. [DISTRIBUTED_EXPERIMENT_RUNBOOK.md](./DISTRIBUTED_EXPERIMENT_RUNBOOK.md) 的 §0–5、§6.2–10、
   §15–19；
5. 准备竞品时再读
   [COMPETITOR_EVIDENCE_2026-08-01.md](./COMPETITOR_EVIDENCE_2026-08-01.md) 和
   [PAPER_EXPERIMENT_MATRIX_2026-08-01.md](./PAPER_EXPERIMENT_MATRIX_2026-08-01.md)。

旧 `HANDOFF.md`、`DEVELOPMENT_LOG.md`、`OPTIMIZATION_LOG.md` 只用 `rg` 查关键词。

## 4. 当前已经存在与尚不存在的能力

### 4.1 现成可跑

- `tests/elastic/run_rail_balance_hybrid_multinode.py`：`off/force` correctness、capacity、
  watchdog；不是 hop-aware 性能 runner。
- `tests/elastic/bench_rail_balance_hybrid_multinode.py`：同树 `off/candidate`，支持
  `legacy_exact/one_hop/adaptive`，计时外 correctness probe、ABBA/BAAB、每-rank原始值、
  rank-max、source/binary identity 和 V2 resolved SM/QP。
- `tests/elastic/bench_rail_balance_hop.py`：reference/vnode/multinode 便捷 smoke；正式多机
  精确实验直接使用上一条 lower-level runner。
- `tools/rail_balance_autotune.py`：离线筛选、预注册至少 5 次 attempt、诊断封存；默认复用
  DeepEP V2 auto。

### 4.2 主实验前必须补齐

以下能力当前不存在，命令里不得假装已经有：

1. 连续 token-level trace v2 capture/reader/replay；当前 trace v1 只是 endpoint count，
   会重新合成 token 并反复重放同一路由；
2. 同一进程直接 `--baseline-mode one_hop --candidate-mode adaptive` 的配对 runner；
3. runtime direct/source-forward/destination-forward/third-Rail payload、proxy/local-copy、
   fallback counter，以及 NIC/QP/NVLink 物理闭环；
4. `P1*`/`Pall*` exact max-flow/min-cut 证书工具；
5. DeepEP/NCCL EP/UCCL-EP common-fair 薄适配器与执行器。

先调通可以乐观推进；但缺哪一项就给对应实验标 `*_GAP`，不能越过证据边界。

## 5. 集群启动语义：最容易犯错的地方

每个节点只启动一次 runner。runner 自己为本机每张 GPU `spawn` worker：

```text
WORLD_SIZE = 节点数
RANK       = 节点序号
```

绝对不要再套 `torchrun --nproc-per-node=8`。所有节点必须使用相同的绝对 repo、Python、
trace 和 output 路径字符串；autotune plan 会冻结这些绝对路径。

`MASTER_ADDR` 必须是 node 0 可达地址，不能是 loopback/`0.0.0.0`。`EP_DISABLE_GIN=0`。
每轮使用新的 `MASTER_PORT`、JIT 目录、run ID 和结果路径，原始结果不得覆盖。

## 6. S0：冻结环境与产物目录

每节点只改 `NODE_RANK`：

```bash
export REPO=<SAME_ABSOLUTE_REPO>
export PY=<SAME_ABSOLUTE_PINNED_PYTHON>
export RUN_ROOT=<SAME_ABSOLUTE_SHARED_ARTIFACT_ROOT>
export NODE_RANK=<0_OR_1>
export MASTER_ADDR=<NODE0_REACHABLE_IP>
export MASTER_PORT=31101

cd "$REPO"
export PYTHONPATH="$PWD/tests/elastic:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2
export RANK="$NODE_RANK"
export OMP_NUM_THREADS=1
export EP_DISABLE_GIN=0
```

在看性能结果之前建立 run root，并写 `preregistration.md`：主指标、最小实用收益、回退预算、
tuning/confirmatory split、搜索预算、排除条件和随机种子。目录合同直接复用 runbook §4。

每节点保存：

```bash
date -u --iso-8601=seconds
hostname -f
uname -a
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
ibdev2netdev
ibv_devinfo -v
rdma link show
"$PY" -VV
"$PY" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
nsys --version
ncu --version
```

还要显式记录 GPU -> NIC/Rail -> PCIe switch -> NUMA 的映射、driver/firmware、时钟、温度、
功耗限制、NCCL/NVSHMEM/Gin 环境。GPU 被其他作业占用时只跑功能 smoke 并标 diagnostic；
性能 run 必须等独占、P0、无 throttle、无 MPS 和空闲 fabric 窗口。

## 7. S1：构建与原生路径调通

每节点在 clean checkout 强制构建，并保存日志和二进制 hash：

```bash
cd "$REPO"
TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH="$PWD" \
  "$PY" setup.py build_ext --inplace --force
git status --short
sha256sum deep_ep/_C*.so
```

先保存本机实际 CLI：

```bash
"$PY" -B tests/elastic/run_rail_balance_hybrid_multinode.py --help
"$PY" -B tests/elastic/bench_rail_balance_hybrid_multinode.py --help
"$PY" -B tools/rail_balance_autotune.py plan --help
"$PY" -B tools/rail_balance_autotune.py freeze --help
```

先用 correctness runner 验证 Gin/RDMA 和最小 round trip。每节点同时执行同一条：

```bash
"$PY" -B tests/elastic/run_rail_balance_hybrid_multinode.py \
  --run-id rb-correct-balanced \
  --case balanced \
  --output "$RUN_ROOT/04_correctness/rb-correct-balanced.json" \
  --num-processes 8 --num-tokens 128 --hidden 7168 \
  --num-topk 8 --num-experts 256 \
  --num-sms 2 --num-allocated-qps 1 \
  --rail-policy all --rail-threshold-percent 0 \
  --sl-idx 3 --seed 106 --timeout 300 --watchdog-seconds 1800
```

该 runner 的 SM 必须 `>=2`、allocated QP 必须 `>0`。先跑 `balanced`，再跑
`two_hot/one_hot/offdiag_hot/diag_hot/closed_block/capacity`；capacity 要保持
`rail-policy=all`、threshold=0。合成 case 只证明机制。

若失败，保留每节点 stdout/stderr、退出码和无 JSON 的事实；按初始化、全局 rank、Gin、
QP、buffer、dispatch、combine 的第一处失败定位，不要同时改多个变量。

## 8. S2：hop-aware correctness 与首轮性能 smoke

下面是当前真实存在的 runner。它固定以同树 `off` 为 baseline：

```bash
"$PY" -B tests/elastic/bench_rail_balance_hybrid_multinode.py \
  --run-id <UNIQUE_RUN_ID> \
  --case <balanced_OR_offdiag_hot_OR_diag_hot_OR_closed_block> \
  --output <UNIQUE_ABSOLUTE_RESULT_JSON> \
  --candidate-mode <legacy_exact_OR_one_hop_OR_adaptive> \
  --num-processes 8 --num-tokens 4096 --hidden 7168 \
  --num-topk 8 --num-experts 256 \
  --num-sms 0 --num-allocated-qps 0 \
  --rail-policy all --rail-threshold-percent 0 \
  --two-hop-threshold-percent <T> \
  --max-two-hop-percent <CAP> \
  --hop-penalty-percent <P> \
  --warmup-iters 10 --steady-iters 100 --order-repeats 1 \
  --sl-idx 3 --seed 106 --timeout 300 --watchdog-seconds 7200
```

规则：

- `one_hop` 用 `T=0 CAP=0 P=0`；
- selective third-Rail 才使用 `candidate-mode=adaptive`；
- 先小 shape 做功能 smoke，再用 4096/7168/8 做无 profiler 稳态；
- 主结果先检查 `eligibility`，再看每-rank raw 和
  `comparison.roundtrip_cuda_ms.speedup_baseline_over_candidate`；
- `path_distribution`/`rail_load` 当前来自 Python expected oracle，不是物理 counter；
- `--allow-contended-smoke` 和 `--diagnostic-profiler` 的结果永远不能进入性能主表。

最小模式矩阵：

| Case | 模式 | 人话目的 |
|---|---|---|
| balanced | one_hop、adaptive | 正常均衡时不能白白变慢 |
| offdiag_hot | one_hop、adaptive | 端点 Rail 之间的一跳调度本应能解决 |
| diag_hot | one_hop、adaptive | 对角热点是一跳解不开、第三 Rail 才可能有用的机制例 |
| closed_block | one_hop、adaptive | 端点集合封闭时验证受限第三 Rail 逃逸和 cap |
| one_hot/two_hot | legacy、one_hop、adaptive | 保留与历史 hotspot 结果的机制回归 |

再在独立 clean worktree 构建 `e5d49d`，使用该提交自己的
`bench_rail_balance_hybrid_multinode.py` 重跑用户历史 shape。它是跨提交回归表，不能与
同树 `one_hop/adaptive` 因果表混在一起。

## 9. S3：正式主实验前的最小 harness 改动

只在 benchmark/test 层做四项，并各自提交；不要先改黄金 CUDA：

1. 把 hard-coded `off` baseline 泛化为 `--baseline-mode`，让同一次进程按
   `one_hop/adaptive` 做 A/A、ABBA、BAAB；
2. 增加 trace v2 reader，逐 iteration 消费连续 window，保留完整 top-k tuple、weight、
   mask/drop、placement 和顺序；
3. 增加 planner submit/ready、dispatch/combine/consumer-ready 的 CUDA event 与 NVTX；
4. 增加 opt-in diagnostic specialization/counter，统计四类路径 payload、local-copy、proxy、
   fallback；默认性能路径不得承担诊断原子或日志开销。

必须先有测试：v2 round-trip hash、multi-target、placement change、drop mask、窗口顺序、
相邻配对、payload 守恒。若真实 trace 尚未到位，可以先完成接口和 synthetic smoke，但正式
第三 Rail 结论保持 `TRACE_GAP`。

## 10. S4：策略调优，按层做而不是大网格

### 10.1 第一层：第三 Rail 算法

固定 DeepEP V2 auto：

```text
num_sms=0
num_allocated_qps=0
dispatch/combine num_qps=0
```

记录实际 resolved SM/QP。只在 tuning split 搜：

```text
one_hop control
adaptive.two_hop_threshold_percent
adaptive.max_two_hop_percent
adaptive.hop_penalty_percent
proxy capacity feasibility
```

先用 CPU oracle/Pareto 淘汰明显无效或容量超限点；真机只跑 `one_hop`、Pareto knee 和一个
aggressive 点。总有效 GPU 配置最多 30 个，非法/OOM 也计预算。不要逐 workload 各挑一个
冠军；最终冻结一个全局配置，或一个只依赖运行时可观测特征的简单 gate。

### 10.2 第二层：参与均衡的 Rail 集合

这层回答“只有 3/5/7 Rail 有真实连接时，是只在活跃 Rail 间均衡，还是达到阈值再招募新
Rail”。它与 selective two-hop 不是同一参数：

```text
rail-policy=all,      rail-threshold=0
rail-policy=active,   rail-threshold=0
rail-policy=adaptive, rail-threshold=<pre-registered value>
```

先冻结第一层 hop 参数，再做这层单变量消融。不要把两层直接相乘成大笛卡尔积。主证据必须
来自真实自然连接/路由；“4 hot + 4 idle”只作压力测试，不能证明泛化。

### 10.3 第三层：执行资源

默认复用 V2 auto。只有 Nsys 证明 source-forward、RDMA、destination-forward 或 planner 的
资源分配暴露在关键路径时，才以 resolved 值为中心一次只改一个 SM/QP 变量，并始终保留
V2 auto 对照。capacity 是正确性/显存门禁，不是性能故事；chunk/seed 当前固定为 8/0，
不进入 v1 搜索。

### 10.4 当前 autotune 的精确用法

当前工具只支持 synthetic/endpoint-count v1 诊断。plan 与 results 必须在仓库外：

```bash
"$PY" -B tools/rail_balance_autotune.py plan \
  --space tests/elastic/rail_balance_autotune_space_v1.json \
  --case trace --workload-json <TUNING_ENDPOINT_COUNT_V1_JSON> \
  --num-nodes 2 --gpus-per-node 8 \
  --tokens-per-rank 4096 --topk 8 --num-experts 256 --hidden 7168 \
  --warmup-iters 10 --steady-iters 100 \
  --minimum-speedup <PREREGISTERED_VALUE_GREATER_THAN_1> \
  --minimum-runs 5 \
  --results-dir "$RUN_ROOT/05_tuning/autotune-results" \
  --output "$RUN_ROOT/05_tuning/autotune-plan.json"
```

所有节点逐条执行 `candidate.attempts[].argv`，只替换该 candidate 的 environment 占位符。
不要手工重拼 run ID 或 result path；至少五个 attempt 必须完全匹配 plan。生成 plan 的机器与
执行节点必须有相同绝对 Python/repo/output 路径。

诊断封存：

```bash
"$PY" -B tools/rail_balance_autotune.py freeze \
  --plan "$RUN_ROOT/05_tuning/autotune-plan.json" \
  --candidate-id <CANDIDATE_ID> \
  --result <EXACT_ATTEMPT_01_PATH_FROM_PLAN> \
  --result <EXACT_ATTEMPT_02_PATH_FROM_PLAN> \
  --result <EXACT_ATTEMPT_03_PATH_FROM_PLAN> \
  --result <EXACT_ATTEMPT_04_PATH_FROM_PLAN> \
  --result <EXACT_ATTEMPT_05_PATH_FROM_PLAN> \
  --output "$RUN_ROOT/05_tuning/diagnostic-profile.json"
```

该 profile 固定是 `production_consumed=false`、
`production_promotion_eligible=false`。它不会自动改变生产路径。补齐 trace v2、直接配对、
runtime counter 和 held-out 后，应扩展工具合同，而不是手改这两个字段。

## 11. S5：held-out 确认实验

正式顺序：

1. `one_hop/one_hop` A/A，冻结噪声界；
2. 同进程 direct `one_hop/adaptive`，ABBA + BAAB；
3. 每 block 至少 10 warmup + 100 steady；
4. 至少 5 次独立进程重启；
5. 先 2x8，通过后再 4x8；
6. `off/adaptive` 只作为 RailBalance 总收益辅助表；
7. `adaptive(max_two_hop=0)` 作为实现一致性负控。

每 iteration 保存每-rank consumer-ready 时间，再计算 rank-max。至少报告 p50/p95/p99、
mean/std/MAD/CV、process/block 层级 bootstrap CI、planner total/hidden/exposed、四类路径 bytes、
extra local-copy、proxy high-water、fallback 和 NIC/Rail counter。

真实 workload 至少覆盖：

- 低 token/rank 的 decode/低延迟窗口；
- 高 token/rank 的 prefill 或训练窗口；
- 两个模型或数据域；
- 自然 low/mid/high skew，而不是人为只构造 4/4 Rail；
- tuning 前 40%、held-out 后 60%，再做 leave-one-workload-out 检查。

若真实 trace 的 exact one-hop optimum 与 unrestricted optimum 没有 gap，不再花昂贵资源证明
第三 Rail；优先优化 one-hop。若结构有 gap 但完整 E2E 不赢，结论是拒绝或 guarded，不能
只拿 Rail 柱状图宣布成功。

## 12. S6：竞品、端到端与组合实验

### 12.1 Transport common-fair

同一输入、consumer-ready 边界和资源账本比较：

```text
same-tree off / one_hop / frozen adaptive
clean upstream DeepEP V2
NCCL EP
UCCL-EP
```

先跑 BF16 共同交集：LL 128 token/rank、HT 4096 token/rank，256 experts、hidden 7168、
top-k 8，2x8/4x8。common-fair 与作者原生命令分表；UCCL CPU proxy 的 core/NUMA/功耗必须
计资源。当前 adapter 未实现，先按 runbook §11 做薄适配器；改变库传输语义就降级为 port。

### 12.2 End-to-end

只有接入同一个冻结模型、checkpoint、请求/训练 batch 后才报告 TTFT/TPOT/goodput 或训练
step/tokens/s。通信 microbenchmark 不能推算模型 E2E。

### 12.3 UltraEP 与 MoonEP

UltraEP 改 placement/replication，RailBalance 改运输路径，做 2x2：placement off/on ×
one_hop/adaptive，并报告 interaction。不要把 UltraEP 当纯 transport 排名对象。

MoonEP 只做单节点 EP8 planning/layout 附录，不外推多机 RDMA。

## 13. S7：性能归因与算子调优

严格遵循：

```text
无 profiler 端到端真值
  -> Nsys 定位暴露关键路径
  -> NCU 只看被证明暴露的 1–3 个 kernel
  -> 单变量修改
  -> 正确性
  -> 同窗口无 profiler 复测
```

Nsys 先选一个 low-gap 和一个 high-gap 真实窗口，A/B 各跑一次。NCU 不对整轮 `--set full`，
也不把 replay 时间放入主表。每个 kernel dossier 必须包含 Nsys exposed time、launch shape、
NCU primary/secondary limiter、源码/SASS、反证、预期端到端上限和下一项证伪实验。

## 14. 每轮 checkpoint 与汇报格式

每完成一个阶段：

1. 原始 stdout/stderr、命令、退出码、JSON、环境和 hash 进入新的只追加目录；
2. 更新 `STATUS.json`，失败、OOM、超时和负结果不得删除；
3. 测量脚手架、生产候选和文档分别 commit；
4. 不修改已封存 raw artifact；派生表必须能一条命令重建；
5. 先发简短进度，再继续下一不依赖阶段。

进度模板：

```text
当前 SHA / build SHA:
拓扑与环境门禁:
本阶段 PASS/FAIL/*_GAP:
已跑模式、workload、样本数:
无 profiler rank-max p50/p95/p99:
正确性与物理 counter:
失败与保留日志:
下一步和停止条件:
RUN_ROOT:
```

## 15. 最终停止门禁

- 2x8 correctness 不稳定：不扩 4x8；
- trace v2 未通过：停止第三 Rail 业务/泛化结论；
- direct `one_hop/adaptive` 未实现：两个独立 job 只能是 diagnostic；
- runtime/physical counter 不可用：可报 E2E，但不得声称物理 Rail 已均衡；
- balanced/low-gap 回退超过预算：候选不能默认启用；
- Nsys 未证明执行资源是关键路径：不覆盖 V2 auto；
- confirmatory 结果不赢：保留 one-hop 或历史黄金，不为“做出正结果”改 trace/阈值；
- 所有多机门禁通过后，才用独立 commit 讨论 capability enable。

最重要的原则只有一句：先让真实端到端和物理路径证据决定策略，再调参数；不能让参数搜索
替代真实 workload，也不能让 profiler 或 synthetic hotspot 替代性能真值。
