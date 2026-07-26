# DeepEP V2 baseline comparison

## Question answered

The production comparison has one baseline:

```text
DeepEP V2 Hybrid: rail_balance="off"
```

The candidate uses the same checkout, inputs, topology, SM/QP geometry and
precision, changing only:

```text
rail_balance="force"
rail_balance_policy=<all|active|adaptive>
rail_balance_threshold_percent=<integer>
```

Local checked-adapter or vnode measurements are component diagnostics.  They
must not be reported as a speedup over DeepEP V2 because they do not execute
the real Gin/RDMA path.

## Measurement boundary

Each measured iteration is the public Hybrid round trip:

```text
dispatch -> combine with that dispatch's handle
```

The force handle is consumed exactly once in the same iteration.  Buffer
construction, JIT warmup, reference generation, correctness checks, report
writing and process-group barriers are outside the steady measurement block.
CUDA events are recorded for dispatch, combine and the full round trip, then
synchronized once at the end of the block so the benchmark does not turn the
pipeline into one synchronized launch at a time.

Host and GPU clocks are not comparable across nodes.  Each rank therefore
computes its own elapsed durations; the distributed iteration result is the
maximum rank-local elapsed duration for that ordinal.  Every rank-local raw
sample is retained.

## Run order and statistics

Fresh buffers are measured in symmetric blocks so the V2 baseline is not
always colder or hotter than the candidate:

```text
off, force, force, off, force, off, off, force
```

Formal collection uses at least 10 warmup and 100 steady iterations per block.
Reports retain raw samples plus median, p95, p99, mean, population standard
deviation, CV, minimum and maximum.  The primary result is full round-trip
latency; dispatch and combine are supporting breakdowns.

The comparison reports candidate speedup as:

```text
DeepEP V2 off median / rail-balance force median
```

Balanced traffic is required as a negative control.  A useful optimization
must not buy skewed-case speedup by imposing material overhead when no move is
needed.

## Acceptance gate

A report is not performance evidence when any of these hold:

- fewer than 10 warmup or 100 steady iterations per block;
- Git is dirty or ranks do not share the same commit and extension binary;
- a profiler or diagnostic NVTX mode is enabled;
- any node has an unexpected GPU compute process, MPS/MIG mismatch, non-benign
  throttle state, or materially changing clocks/power state;
- off and force do not pass the same public dispatch/combine correctness
  reference and exact output comparison;
- topology, input, precision, SM/QP configuration, or run order differs;
- the route is the capacity-failure fixture rather than a valid workload.

Failure of the gate preserves the raw report for debugging but forbids a
speedup claim.  Nsys and NCU runs are separate diagnostic experiments; the
final truth is always the profiler-free off/force rerun.

## Commands

### 1. CPU contract tests

This test says, in plain language: the benchmark still calls DeepEP V2
`off` the baseline, computes speedup in the right direction, alternates the
run order, rejects contaminated GPUs, and does not synchronize every steady
iteration.

```bash
cd /home/chen/workspace/source_code/DeepEP
PY=/home/chen/.cache/deepep-sjlgpt/bin/python
PYTHONPATH="$PWD:$PWD/tests/elastic" "$PY" -B \
  tests/elastic/test_rail_balance_hybrid_multinode_bench.py
```

This existing test says: the route generator, CPU oracle, public off/force
correctness boundary and watchdog contract were not broken by the benchmark.

```bash
PYTHONPATH="$PWD:$PWD/tests/elastic" "$PY" -B \
  tests/elastic/test_rail_balance_validation_contract.py
```

This check says: both new Python files parse, follow Ruff rules and have no
whitespace-damaged patch.

```bash
"$PY" -m py_compile \
  tests/elastic/bench_rail_balance_hybrid_multinode.py \
  tests/elastic/test_rail_balance_hybrid_multinode_bench.py
/home/chen/.cache/deepep-sjlgpt/bin/ruff check \
  tests/elastic/bench_rail_balance_hybrid_multinode.py \
  tests/elastic/test_rail_balance_hybrid_multinode_bench.py
/home/chen/.cache/deepep-sjlgpt/bin/ruff format --check \
  tests/elastic/bench_rail_balance_hybrid_multinode.py \
  tests/elastic/test_rail_balance_hybrid_multinode_bench.py
git diff --check
```

### 2. Build the exact checkout on every cluster node

This makes the loaded extension match the clean Git commit recorded in the
report.  Run it independently on every node before starting a comparison.

```bash
cd /home/chen/workspace/source_code/DeepEP
PY=/home/chen/.cache/deepep-sjlgpt/bin/python
TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH="$PWD" \
  "$PY" setup.py build_ext --inplace --force
git status --short
sha256sum deep_ep/_C*.so
```

`git status --short` must print nothing.  Benchmark JSON belongs under the
ignored `.cache/` directory so report creation does not dirty the checkout.

### 3. Configure a two-node, eight-GPU-per-node run

Run this block on both nodes.  Change only `RANK`: node 0 uses `0`, node 1
uses `1`.  Replace `10.0.0.10` with the address of node 0.

```bash
cd /home/chen/workspace/source_code/DeepEP
export PY=/home/chen/.cache/deepep-sjlgpt/bin/python
export PYTHONPATH="$PWD:$PWD/tests/elastic"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=10.0.0.10
export OMP_NUM_THREADS=1
export EP_DISABLE_GIN=0

run_ab() {
  case_name=$1
  route=$2
  policy=$3
  threshold=$4
  port=$5
  jit_root="/tmp/deepep-rail-${case_name}"
  if test -e "$jit_root"; then
    echo "refusing non-empty/reused JIT path: $jit_root" >&2
    return 1
  fi
  mkdir -m 700 "$jit_root"
  EP_JIT_CACHE_DIR="$jit_root" MASTER_PORT="$port" \
    "$PY" -B tests/elastic/bench_rail_balance_hybrid_multinode.py \
      --run-id "$case_name" \
      --case "$route" \
      --rail-policy "$policy" \
      --rail-threshold-percent "$threshold" \
      --num-processes 8 \
      --num-tokens 4096 \
      --hidden 7168 \
      --num-topk 8 \
      --num-experts 256 \
      --num-sms 0 \
      --num-allocated-qps 0 \
      --warmup-iters 10 \
      --steady-iters 100 \
      --order-repeats 1 \
      --timeout 300 \
      --watchdog-seconds 7200 \
      --output ".cache/rail_balance/baseline/${case_name}.json"
}
```

The function must be invoked with the same five arguments on both nodes, one
case at a time.

### 4. Formal comparisons

This case asks: when traffic is already spread across all rails, how much
overhead does the new path add compared with DeepEP V2?

```bash
run_ab balanced-all0 balanced all 0 31101
```

This case asks: when one rail owns all remote traffic, does full eight-rail
balancing beat DeepEP V2, which leaves that skew untouched?

```bash
run_ab onehot-all0 one_hot all 0 31102
```

This case asks: can adaptive balancing beat DeepEP V2 while opening fewer
data-bearing rails than unconditional full balancing?

```bash
run_ab onehot-adaptive20 one_hot adaptive 20 31103
```

This negative control asks: if policy forbids expanding beyond the originally
active rail, does it correctly avoid shuffle work, and what fixed overhead
remains relative to DeepEP V2?

```bash
run_ab onehot-active0 one_hot active 0 31104
```

This case asks: with two already-active rails, is widening to every rail worth
the local shuffle cost compared with DeepEP V2?

```bash
run_ab twohot-all0 two_hot all 0 31105
```

For a quick plumbing-only run, change warmup/steady to `2/10`.  That report is
intentionally marked diagnostic-only and cannot be quoted as a speedup.

## Reading the result

The decisive field is:

```text
comparison.roundtrip_cuda_ms.speedup_baseline_over_candidate
```

- greater than `1.0`: rail balance is faster than DeepEP V2 for that route;
- equal to `1.0`: no measured difference;
- less than `1.0`: rail balance loses and should bypass that workload.

Also inspect `eligibility.performance_claim_eligible`.  A numerical speedup in
an ineligible report is diagnostic data, not evidence.  The `*_cuda_ms`
fields are CUDA-event call envelopes: they include GPU idle time while the
host/world gate runs and are not a sum of kernel execution times.
