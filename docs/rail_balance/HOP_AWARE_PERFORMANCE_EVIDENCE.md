# Hop-aware RailBalance local performance evidence

This file is an append-only index of profiler-free and profiler artifacts for
the hop-aware branch. Single-node results are diagnostic and cannot establish
Gin/RDMA or end-to-end multinode speedup.

## HA060-A: eliminate adaptive per-copy rescans

### Identity and environment

```text
base commit: 2605775b79aafc90d00d7598f0e97a71c6f1376a
host: dedicated-developjob-wtl-t1wjo-7c6d5f4d56-qzkfm
target: 8 x H200-class Hopper/NVSwitch
driver-visible name: NVIDIA L20X (GH100)
PyTorch-visible capability: 9.0; 132 SM; 150121021440 bytes/device
nvidia-smi topology: NV18 between every GPU pair
driver: 570.172.08
toolkit/nvcc: CUDA 12.8 / 12.8.61
PyTorch: 2.11.0+cu128
cuDNN: 9.19.0
NCCL: 2.28.9
Nsys: 2024.6.2.225
NCU: 2025.1.1.0
MIG: disabled; compute mode: default; P-state: P0
co-tenants during collection: none
```

The driver string is retained rather than silently normalized. PyTorch and NCU
identify the device as GH100/SM90, while one `nvidia-smi` query printed compute
capability 8.9. This identity discrepancy is missing evidence for formal
hardware naming, but does not alter the local before/after result.

NVIDIA's H200 page specifies 141 GB HBM3e at 4.8 TB/s. The DGX H100/H200 guide
specifies four NVSwitches and 900 GB/s GPU-to-GPU bandwidth. The CUDA 12.8
Hopper tuning guide is the architecture reference for SM90 and TMA:

- https://www.nvidia.com/en-au/data-center/h200/
- https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html
- https://docs.nvidia.com/cuda/archive/12.8.0/hopper-tuning-guide/index.html

### Measurement contract

The first probe calls the existing private planner API on one idle GPU. Each
row fixes `G=8, K=8, D=2`, reuses one input tensor, performs three unreported
warmups, and retains 20 profiler-free steady samples. The API allocates output
tensors, launches the planner, copies status to host, and synchronizes; this is
a diagnostic upper envelope, not production Hybrid latency.

Raw files:

```text
/tmp/ha060-planner-baseline.json
/tmp/ha060-planner-final-guarded-matched.json
```

Reproduce the profiler-free current-tree probe with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/tests/elastic \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hop_plan.py \
  --mode both --tokens 8 32 128 512 --warmup 3 --steady 20 \
  --device 0 --output-json /tmp/rail-hop-plan.json
```

| Mode | N | Before median / p95 / p99 (us) | After median / p95 / p99 (us) | Median speedup |
| --- | ---: | ---: | ---: | ---: |
| one_hop | 128 | 2485.68 / 2488.12 / 2488.30 | 2476.85 / 2479.57 / 2479.77 | 1.004x |
| one_hop | 512 | 13913.18 / 13994.06 / 14008.54 | 13900.84 / 13981.88 / 14067.31 | 1.001x |
| adaptive | 8 | 352.73 / 361.46 / 364.73 | 299.45 / 326.51 / 328.54 | 1.178x |
| adaptive | 32 | 2648.11 / 2654.29 / 2677.91 | 945.48 / 965.63 / 966.56 | 2.801x |
| adaptive | 128 | 33829.91 / 33892.16 / 33896.09 | 3502.81 / 3523.76 / 3539.51 | 9.658x |
| adaptive | 512 | 642349.08 / 642437.54 / 642464.62 | 18885.64 / 18916.24 / 18921.26 | 34.013x |

One-hop is the unchanged control. Its stable result falsifies a global clock or
allocator explanation for the adaptive improvement.

### Nsys critical path

Exact command and installed reports are recorded in `DEVELOPMENT_LOG.md`.
Nsys filters the fourth call with
`rail_balance_hop_plan/adaptive/N128_sample0`.

Current-tree collection command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/tests/elastic \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o .cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hop_plan.py \
  --mode adaptive --tokens 128 --warmup 10 --steady 1 --device 0 --nvtx \
  --output-json /tmp/rail-hop-plan-nsys.json

nsys stats --force-export=true \
  --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum \
  --filter-nvtx rail_balance_hop_plan/adaptive/N128_sample0 \
  .cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded.nsys-rep
```

| Artifact | NVTX range | Planner kernel | Other GPU kernels |
| --- | ---: | ---: | ---: |
| `ha060/adaptive-n128.nsys-rep` | 33.897 ms | 33.749 ms | 11.1 us |
| `ha060/adaptive-n128-final-guarded.nsys-rep` | 3.566 ms | 3.413 ms | 11.4 us |

The planner kernel decreases 9.888x. Allocation/fill kernels are not the root
cause. SQLite exports with the same basename preserve the installed schema.

### NCU kernel dossier

```text
Kernel: rail_balance_hop_plan_impl<0>
Invocation: adaptive, G8/N128/K8/D2, fourth call after three warmups
Nsys exposed time: 33.749 ms before; 3.413 ms after
Launch: grid 1x1x1, block 32x1x1, 80 registers/thread, 0 dynamic smem
Primary limiter: serial algorithm; only thread 0 executes
Secondary limiter: one active warp cannot hide dependency/memory latency
Before metrics: 9,276,790 instructions; 0.03% SM; 0.00% DRAM;
                20.13 MB/s memory; 0.14 eligible warps/scheduler;
                86.17% cycles with no eligible warp; one active thread/warp
Contradicting evidence: 100% branch efficiency and no bandwidth saturation
Source: deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh
        old adaptive loop rescanned all records after every accepted copy
Hypothesis: batch equal-cost moves until a load gap closes or old Rail ceases critical
After metric: 812,377 instructions, 11.419x fewer
Correctness: CPU 12/12, CUDA 9/9, 8-GPU vnode round trip
Sanitizer: memcheck and initcheck, zero errors
Confidence: high for planner kernel and private API; no multinode claim
Next falsification: production prepared-plan timing and source-stage rank-max
```

NCU used kernel replay, cache control `all`, base clock control, exact kernel
regex, three matching-launch skips and one collected invocation. Basic used ten
passes; targeted source/scheduler/memory used fifteen passes. NCU duration is
not used as the performance result. The JIT binary lacked line information, so
NCU imported no source; exact source lines and the installed report are the
correlation evidence.

Minimal current-tree NCU command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/tests/elastic \
ncu --section InstructionStats --section LaunchStats \
  --replay-mode kernel --cache-control all --clock-control base \
  --target-processes all --kernel-name 'regex:rail_balance_hop_plan_impl' \
  --launch-skip 3 --launch-count 1 --force-overwrite \
  -o .cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded-instructions \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hop_plan.py \
  --mode adaptive --tokens 128 --warmup 3 --steady 1 --device 0 \
  --output-json /tmp/rail-hop-plan-ncu.json
```

Artifacts:

```text
.cache/rail_balance/hop-aware/ha060/adaptive-n128.nsys-rep
.cache/rail_balance/hop-aware/ha060/adaptive-n128.sqlite
.cache/rail_balance/hop-aware/ha060/adaptive-n128-basic.ncu-rep
.cache/rail_balance/hop-aware/ha060/adaptive-n128-targeted.ncu-rep
.cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded.nsys-rep
.cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded.sqlite
.cache/rail_balance/hop-aware/ha060/adaptive-n128-final-guarded-instructions.ncu-rep
```

### Missing evidence

- Cold-start/JIT latency was excluded but not separately quantified.
- The private API includes allocation and synchronization; prepared production
  plan, source shuffle, destination forward and complete vnode rank-max raw
  samples remain to be measured.
- No real D>1 Gin/RDMA timing, NIC counters or exposed communication exists.
- No JIT `-lineinfo` source import exists for this report.
- Power and clocks were sampled before collection, not continuously logged.

## HA060-B: parallelize disjoint owner validation

The HA060-A planner still launched one warp but returned 31 lanes immediately.
For one-hop G8/N128/K8, Nsys measured a 2.407 ms kernel and NCU measured 529,602
instructions. The accepted change assigns the fixed validation scan for owner
`o` to lane `o`; initialization is lane-strided, the exact unit count is
warp-reduced, and lane 0 retains the complete greedy/adaptive order.

Matched profiler-free raw files use three warmups and 20 retained samples:

```text
before: /tmp/ha060-planner-final-guarded-matched.json
after:  /tmp/ha060-owner-parallel-validation-matched.json
```

| Mode | N | Before median (us) | After median / p95 / p99 (us) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| one_hop | 128 | 2476.85 | 1805.66 / 1835.95 / 1836.23 | 1.372x |
| one_hop | 512 | 13900.84 | 11084.43 / 11121.65 / 11180.13 | 1.254x |
| adaptive | 128 | 3502.81 | 2816.53 / 2844.12 / 2844.38 | 1.244x |
| adaptive | 512 | 18885.64 | 16092.86 / 16124.97 / 16128.25 | 1.174x |

Matched Nsys reports reduce the one-hop kernel from 2.407 to 1.723 ms
(1.397x). Matched NCU InstructionStats reduce executed instructions from
529,602 to 388,691 (1.363x); launch remains grid 1, block 32, 80 registers per
thread. These reports are diagnostic planner evidence, not network timing:

```text
.cache/rail_balance/hop-aware/ha060/onehop-n128-baseline.nsys-rep
.cache/rail_balance/hop-aware/ha060/onehop-n128-owner-parallel.nsys-rep
.cache/rail_balance/hop-aware/ha060/onehop-n128-baseline-instructions.ncu-rep
.cache/rail_balance/hop-aware/ha060/onehop-n128-owner-parallel-instructions.ncu-rep
```

Correctness evidence is GPU planner 9/9 plus exact eight-GPU one-hop and
adaptive vnode round trips. Compute Sanitizer memcheck, synccheck and initcheck
all report zero errors on the full focused CUDA suite.
