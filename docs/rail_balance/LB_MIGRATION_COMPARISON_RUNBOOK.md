# LB-hard 与 LB-minimal 对比手册

## 1. 比较对象

本分支用一份 V2 Hybrid 数据面比较三种 planner，不复制 DeepEP-LB 的 V1 kernel：

| 名称 | 运行参数 | 说明 |
|---|---|---|
| DeepEP V2 baseline | benchmark 内置 `off` | 每次候选测试的配对基线 |
| 当前黄金版 | `one_hop`, threshold `0` | endpoint-aware，一跳约束，不做激活 gate |
| LB-hard | `legacy_exact`, threshold `0` | 紧凑 `[owner,destination]` quota，可选任意 Rail；有意舍弃 endpoint hop 语义 |
| LB-minimal | `one_hop`, threshold `>0` | source load 未超过阈值时零搬运；触发后运行当前黄金 planner |
| LB-minimal + 2-hop | `adaptive`, threshold `>0` | 同一激活 gate，触发后保留 selective 2-hop |

完整功能取舍见 [LB_MIGRATION_TRADEOFFS.md](LB_MIGRATION_TRADEOFFS.md)。静态 proxy slot、combine ticket、V2 Gin 和 fail-closed gate 是共同正确性底座，不作为实验变量。

## 2. 本机已有证据

以下只是 2026-08-02 单机 H200 diagnostic，不是多机性能结论：

| 检查 | 结果 |
|---|---|
| `setup.py build_ext --inplace --force` | PASS |
| hop-aware CUDA 回归 | 23/23 PASS，含 singleton/multi-target gate |
| public lifecycle CPU fake runtime | PASS |
| unified reference contracts | 9/9 PASS |
| 4+4 vnode balanced，threshold 20 | dispatch/combine round trip PASS |
| 4+4 vnode offdiag_hot，threshold 20 | dispatch/combine round trip PASS |

Planner 私有 API，`N=512,K=8,C=8,G=8`，10 warmup + 100 steady：

| workload | threshold | median | p95 | 解释 |
|---|---:|---:|---:|---|
| balanced | 0 | 525.332 us | 530.144 us | 当前黄金 planner |
| balanced | 20 | 313.121 us | 318.715 us | gate 关闭，median 减少 40.4% |
| rotating hotspot | 0 | 385.481 us | 392.196 us | 当前黄金 planner |
| rotating hotspot | 20 | 380.911 us | 389.262 us | gate 打开，差异约 -1.2%，视为噪声内 |

该 microbenchmark 包含 Python 私有 API、tensor allocation、planner kernel 和 status sync；它不能替代 profiler-free 多机 rank-max dispatch/combine。
原始 100 个样本保存在 `docs/rail_balance/artifacts/lb_comparison_20260802/`；
它们与上表一起绑定到实现提交 `d85d957`，最终文档提交不改动被测代码。

保留的失败证据：本机 venv 没有 `pytest`，因此使用各测试文件自带 runner；完整 Hybrid API CPU 测试中的旧 force constructor case 在当前 Torch 环境报 `Expected a cuda device, but got: cpu`，与新增 CUDA gate 无关，未伪造 PASS。

## 3. 本机复现

```bash
cd /home/chen/workspace/source_code/DeepEP
PY=/home/chen/.cache/deepep-sjlgpt/bin/python
export PYTHONPATH="$PWD/tests/elastic:$PWD"

TORCH_CUDA_ARCH_LIST=9.0 "$PY" setup.py build_ext --inplace --force

"$PY" tests/elastic/test_rail_balance_hop_one_hop_cuda.py --device 0
"$PY" tests/elastic/test_rail_balance_hybrid_public_lifecycle.py
"$PY" tests/elastic/test_rail_balance_hop_bench.py
```

这组命令分别证明：GPU planner 全部不变量、公开 dispatch/ticket 参数贯通、统一 reference schema 和激活语义。

比较当前黄金与 LB-minimal 的 planner：

```bash
"$PY" tests/elastic/bench_rail_balance_hop_plan.py \
  --mode one_hop --pattern balanced --tokens 512 --channels 8 \
  --planner-chunk-size 8 --activation-threshold-percent 0 \
  --warmup 10 --steady 100 --device 0 \
  --output-json /tmp/rb-current-balanced.json

"$PY" tests/elastic/bench_rail_balance_hop_plan.py \
  --mode one_hop --pattern balanced --tokens 512 --channels 8 \
  --planner-chunk-size 8 --activation-threshold-percent 20 \
  --warmup 10 --steady 100 --device 0 \
  --output-json /tmp/rb-minimal-balanced.json
```

完整单机虚拟双节点回环：

```bash
"$PY" tests/elastic/bench_rail_balance_hop.py \
  --backend vnode --mode one_hop --case balanced \
  --num-nodes 2 --gpus-per-node 4 --tokens-per-rank 16 \
  --topk 1 --num-experts 16 --hidden 256 --chunk-size 8 \
  --rail-threshold-percent 20 --timeout 300 --watchdog-seconds 900 \
  --output-json /tmp/rb-minimal-vnode-balanced.json
```

它证明 LSA 模拟路径的 metadata、proxy 和 combine 逆路由，不证明 NIC/RDMA 加速。

## 4. 两节点统一环境

每个节点只启动一次 runner；runner 自己 spawn 本机 8 个 GPU worker，禁止再套一层 `torchrun`。两个节点只改变 `NODE_RANK`：

```bash
export REPO=<两节点相同绝对路径>
export PY=<两节点相同固定Python>
export RUN_ROOT=<两节点相同共享结果目录>
export NODE_RANK=<0或1>
export MASTER_ADDR=<node0可达IP>

cd "$REPO"
export PYTHONPATH="$PWD/tests/elastic:$PWD"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2
export RANK="$NODE_RANK"
export OMP_NUM_THREADS=1
export EP_DISABLE_GIN=0

TORCH_CUDA_ARCH_LIST=9.0 "$PY" setup.py build_ext --inplace --force
git status --short
git rev-parse HEAD
sha256sum deep_ep/_C*.so
```

两节点必须 clean、HEAD 相同、`.so` hash 相同。性能运行前还要保存 GPU/NIC 拓扑、driver/CUDA/NCCL、功耗/时钟、温度和其他 GPU 进程；GPU 或 fabric 被占用时只跑 correctness，并标记 diagnostic。

## 5. 同树 baseline 对比命令

下面函数每次都把候选与原始 DeepEP V2 `off` 做 ABBA/BAAB 配对；两节点用完全相同参数同时执行：

```bash
run_rb() {
  label=$1
  mode=$2
  rail_threshold=$3
  two_hop_threshold=$4
  two_hop_cap=$5
  hop_penalty=$6
  case_name=$7
  port=$8

  run_id="${label}-${case_name}-${port}"
  jit_dir="/tmp/deepep-jit-${run_id}"
  test ! -e "$jit_dir" || { echo "JIT目录已存在: $jit_dir" >&2; return 1; }
  mkdir -m 700 "$jit_dir"

  MASTER_PORT="$port" EP_JIT_CACHE_DIR="$jit_dir" \
  "$PY" -B tests/elastic/bench_rail_balance_hybrid_multinode.py \
    --run-id "$run_id" --case "$case_name" \
    --output "$RUN_ROOT/${run_id}.json" \
    --candidate-mode "$mode" \
    --num-processes 8 --num-tokens 4096 --hidden 7168 \
    --num-topk 8 --num-experts 256 \
    --num-sms 0 --num-allocated-qps 0 \
    --rail-policy all --rail-threshold-percent "$rail_threshold" \
    --two-hop-threshold-percent "$two_hop_threshold" \
    --max-two-hop-percent "$two_hop_cap" \
    --hop-penalty-percent "$hop_penalty" \
    --warmup-iters 10 --steady-iters 100 --order-repeats 1 \
    --sl-idx 3 --seed 106 --timeout 300 --watchdog-seconds 7200
}
```

先跑 balanced，回答“正常流量是否白白变慢”：

```bash
run_rb hard legacy_exact 0 0 0 0 balanced 32101
run_rb gold one_hop    0 0 0 0 balanced 32102
run_rb min  one_hop   20 0 0 0 balanced 32103
```

再跑 endpoint one-hop 可解决的机制 case：

```bash
run_rb hard legacy_exact 0 0 0 0 offdiag_hot 32201
run_rb gold one_hop    0 0 0 0 offdiag_hot 32202
run_rb min  one_hop   20 0 0 0 offdiag_hot 32203
```

只有 one-hop 结果仍有结构性残余热点时才测第三 Rail：

```bash
run_rb min2 adaptive 20 5 10 50 diag_hot     32301
run_rb min2 adaptive 20 5 10 50 closed_block 32302
```

`diag_hot`/`closed_block` 只证明机制，不得作为业务收益主结论。正式结论必须 replay 真实 router trace 和专家均衡后的 trace；训练/调参 trace 与最终报告 trace 必须分开。

## 6. 严肃比较方法

1. 每个候选都以同次运行中的 `off` 为 baseline，主指标是 profiler-free `rank-max roundtrip_cuda_ms`；同时报告 dispatch、combine、median、p95、p99 和全部 raw samples。
2. 运行顺序做正反两轮：第一轮 hard→gold→minimal，第二轮 minimal→gold→hard；每轮使用新 port、run ID 和 JIT 目录。
3. correctness probe、JIT 和 warmup 必须在 steady timing 外；任何 fallback、timeout、NaN/Inf、drop/duplicate 或 rank 不一致都使该 run 无效。
4. hard 与 minimal 的算法差异同时看：Rail peak、source/destination local-forward bytes、moved copies、two-hop ratio、proxy high-water。不能只看柱状图是否更平。
5. synthetic 只定位机制。主实验至少包含多个 layer/window 的真实 route，以及 expert balance 后的 holdout route，避免阈值只拟合某一种 4/4 或 one-hot 分布。
6. threshold 先用小集合 `{5,10,20,30}` 在 tuning traces 搜索，再在未见过的 holdout traces 固定复测；不得逐 case 报各自最优值。

建议验收门槛：

- balanced：LB-minimal 相对 `off` 的 rank-max median/p95 回退不超过 2%；
- offdiag/真实热点：LB-minimal 至少保留当前黄金 one-hop 收益的 95%，且 correctness 完全一致；
- hard：只有在真实 holdout 上的额外收益稳定覆盖新增 local-hop 成本时，才进入候选；否则只保留为上限对照；
- adaptive：只有真实 trace 的 one-hop residual 明显、two-hop cap 未打满且 roundtrip 有稳定收益时启用。

## 7. Profiler 使用边界

先完成无 profiler 结果，再对同一冻结 case 单独采 Nsys，量化 planner、source shuffle、Gin/RDMA、destination forward、combine 和 rank skew。只有 Nsys 证明某个具体 kernel 有暴露关键路径时间，才对一个稳定 invocation 使用 NCU；NCU replay 时间不能当端到端性能。

如果 balanced 仍回退，优先检查：首次 recheck 的 record gather/precount 是否暴露、零 move cache 是否命中、是否真的退回 native Hybrid。若热点退化，检查 gate 是否误关；不要先调 SM/TMA。若 hard 更快但 roundtrip 更慢，检查任意 third-Rail 带来的两侧 local forwarding，而不是继续压 planner 微秒数。
