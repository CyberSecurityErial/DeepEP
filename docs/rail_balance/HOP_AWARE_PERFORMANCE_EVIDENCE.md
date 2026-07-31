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

## HA060-C: prepared source/return diagnostic entry

The existing C100 checked-adapter runner now executes `legacy`, `one_hop`, and
`adaptive` through the same prepared plan transaction. Hop source/return JSON
reports deliberately publish no logical byte numerator: the runner's movement
table is the legacy reference, while the runtime hop path counters are not yet
retained. Reporting a GB/s value would therefore attribute the wrong plan.

After fixing a fail-closed retained-staging capacity boundary, four cold
single-sample hop smokes and one legacy control completed:

| Mode | Stage | Cold rank-max stage (us) | Scope |
| --- | --- | ---: | --- |
| legacy | source | 496.658 | checked-adapter diagnostic |
| one_hop | source | 3242.746 | checked-adapter diagnostic |
| adaptive | source | 2838.342 | checked-adapter diagnostic |
| one_hop | return | 269.417 | checked-adapter diagnostic |
| adaptive | return | 416.797 | checked-adapter diagnostic |

Reports are retained under `/tmp/ha060-c100-*-smoke.json` for this workspace.
They are not accepted performance evidence: the tree was dirty, each mode has
one cold sample, the capacity differs between legacy and hop mode, and
`nvidia-smi` reported the devices as L20X. The only justified conclusion is a
next hypothesis: hop source-shuffle has exposed work worth attributing with a
clean profiler-free distribution and Nsys before any kernel edit. Real Gin,
RDMA, NIC load and end-to-end DeepEP speed remain missing evidence.

## HA060-D: precompute retained staging prefixes

The HA060-C clue was reproduced without a profiler over 3 warmups and 20
steady iterations. For the C100 matrix source stage at hidden size 256,
one-hop had a 3,225.273 us median and adaptive had a 2,878.798 us median.
The legacy control was 502.329 us. These are checked-adapter stage timings,
not end-to-end or real-Rail claims.

Nsys then attributed the one-hop critical source kernel on device 0 to
3,081.955 us. Source inspection found that every retained copy recomputed its
slot base by scanning all preceding channel/destination counters. The single
variable edit makes the planner write an exclusive retained prefix into the
existing `owner_channel_prefix` allocation and changes only the hop-aware
source path to load that value once. It adds no output tensor or public ABI.

After the edit and complete vnode validation, matched 3+20 profiler-free runs
reported:

| Mode | Before median (us) | After median (us) | Stage speedup |
| --- | ---: | ---: | ---: |
| one_hop | 3,225.273 | 129.304 | 24.94x |
| adaptive | 2,878.798 | 674.253 | 4.27x |

The post-edit Nsys report measured the same device-0 source kernel at
39.488 us, a 78.05x kernel reduction; the all-device kernel-duration sum fell
from 3,162.436 us to 120.416 us. Reports are retained at:

```text
.cache/rail_balance/hop-aware/ha060/c100-onehop-source-nsys.nsys-rep
.cache/rail_balance/hop-aware/ha060/c100-onehop-source-prefix-nsys.nsys-rep
```

Their SHA-256 values are respectively
`a2466876e83271c8f11a910b0ee9b395075815628977be065877c99ecc591992` and
`f1d3ca0502591f619f0cac728e3093e812020edfb2ef4da25414ab8e04a3ad78`.
Both one-hop off-diagonal and adaptive diagonal eight-GPU vnode round trips
passed after the final pointer fix. The GPU planner passed 10/10 under both
memcheck and initcheck with zero errors.

Rejected observations are preserved rather than counted as evidence:

- The first edit failed to pass the new prefix pointer from the host launcher.
  The kernel fail-closed early return produced a false 113 us result; full
  vnode round-trip correctness caught it, and the result was discarded.
- NCU application replay collected one pass but rejected later passes because
  the multi-process application did not reproduce a consistent profiled
  kernel sequence. No `.ncu-rep` exists and no NCU metric is claimed.
- Full-vnode Compute Sanitizer completed the functional round trip but emitted
  492 `cudaErrorNoKernelImageForDevice` errors from NCCL initialization. This
  is retained as a sanitizer/tool compatibility failure, not reported as a
  RailBalance kernel result.

The current device tools identify the visible accelerator as `NVIDIA L20X`,
CC 9.0 with 132 SMs. Therefore this checkpoint remains
`single_node_diagnostic`; it provides no H200, Gin, RDMA, NIC, or real
multi-node performance claim, and capability remains false.

### Clean HA060-D distribution

Commit `bfffe7c` was measured without a profiler for 10 warmups and 100 steady
iterations; all automatic eligibility gates passed. Legacy, one-hop, and
adaptive source-stage medians were respectively 491.699, 122.753, and
667.974 us. Their p95 values were 503.357, 145.578, and 703.740 us; p99 values
were 505.909, 177.898, and 716.065 us. Raw reports and SHA-256 values are:

```text
ace5da5030fb34601d2c1ab98466fca8225495d3b6d5ce54dd4b666802157b8d  c100-legacy-source-formal.json
7dd28be5c33a966ed2a1d3e9583941b7d3c4099439c8fa47cdffe1630b906f89  c100-onehop-source-prefix-formal.json
de6b977ad02e56c0b412e5df2c4906fc623c7bfd1a6b74b44125af41bea73c37  c100-adaptive-source-prefix-formal.json
```

### HA060-E planner channel selection

The clean one-hop report also exposes a 203.951 ms median `finish` boundary.
The corresponding Nsys report attributes 155.580 ms median to the planner
kernel, 97.7% of captured kernel duration. Replacing its exact 256-counter
channel search with an equivalent eight-word minimum-load bitset reduces a
3+20 diagnostic `finish` median to 44.445 ms and a second Nsys planner median
to 34.928 ms. The kernel reduction is 4.45x; it remains the dominant 92.3% and
therefore is not presented as finished optimization.

The post-edit JSON, Nsys report, and SQLite hashes are:

```text
e219a9e03961646d56a565cc7a0bceee78980f11bedc644f59044613e3495042  c100-onehop-channelmask-diag.json
264b753501895ca97d4e7be73e1c8cd74395637adc99309719d3c87e2613c5a7  c100-onehop-channelmask-nsys.nsys-rep
a146bc749c25d766adb0467dcd3ecfc00614280ea2287e11e4fd0e6812d5c75b  c100-onehop-channelmask-nsys.sqlite
```

The before report is a clean 10+100 distribution and the after diagnostic is
a dirty 3+20 run, so the profiler-free ratio is provisional. The Nsys kernel
comparison uses the same C100 checked-adapter workload and is diagnostic under
profiler perturbation. An exact 256-channel greedy-oracle case, 11/11 GPU
planner tests, two vnode round trips, and all three focused sanitizer tools
pass. Real multi-node capability remains false.

The clean `989c9c7` 10+100 reports close the provisional profiler-free gap:

| Mode | finish median / p95 (ms) | source median / p95 (us) |
| --- | ---: | ---: |
| one-hop | 44.436 / 44.550 | 113.268 / 131.701 |
| adaptive | 416.741 / 416.976 | 681.248 / 727.293 |

Before O096 those finish medians were 203.951 and 575.660 ms. The clean
one-hop reduction is 4.59x and adaptive is 1.38x. Reports passed automatic
baseline eligibility and hash to:

```text
6a16a95b67a6fbd3cac7160098500799c873d6a68d17d8a5af0f9eefc5734b07  c100-onehop-channelmask-formal.json
046d758a895a88cd0e4a7d43821715af5d1fa69013a1488297c3d9eaaf7a7f43  c100-adaptive-channelmask-formal.json
```

Adaptive remains planner-bound because selective residual search is separate
from exact channel materialization. No claim expands beyond the checked
single-node adapter.

### HA060-H packed-record diagnostic

The planner's input producer guarantees a compact per-token destination prefix.
After adding fail-closed validation, all later planner passes skip from the
first unused slot to the next token. Private N=1024/K=8/C=256 one-hop latency
drops from 22.167 to 12.165 ms. C100 3+20 `finish` medians drop from 44.445 to
41.448 ms for one-hop and 416.741 to 399.364 ms for adaptive. The diagnostic
JSON hashes are:

```text
c8ab370db013bafd1124ef175f0b63bd2c4b74ad84367d4406ab774b1b4ff83b  hop-plan-both-n1024-c256-packed.json
178fb1285e990c7650d74ef8fc46835306350ebc0578bc2157d24a9e540d84e8  c100-onehop-packed-diag.json
03a319ed8a0b51da62686d1ec90d0a038889d4c6e573826568c9c99a1f0d2f7e  c100-adaptive-packed-diag.json
```

These three reports were collected from a dirty tree and are not the final
profiler-free claim. Exact planner/vnode correctness and focused sanitizer
evidence pass; public capability remains false.

The clean `eedcdad` 10+100 reports pass eligibility and confirm the finish
boundary:

```text
one-hop  median/p95 finish   41.372 / 41.523 ms
adaptive median/p95 finish  399.294 / 399.510 ms
```

This is a 7.4% one-hop and 4.4% adaptive reduction relative to the clean O096
reports. Source-stage code is identical, so its sample variation is not
claimed as a planner effect. Report hashes are:

```text
1ddce1bff73604a740749b8d053c52c5d2fb3dede555ec82cb14dab604da0588  c100-onehop-packed-formal.json
bcf6cc5c30df701a731e948338dc88d985c7fc01d8957be30a23e2de027a2ae6  c100-adaptive-packed-formal.json
```

### HA060-I adaptive peak diagnostic

Nsys after packed skipping still attributes 306.323 ms median and 98.1% of
captured kernel time to adaptive planning. Computing exact max/second/count
state once per iteration instead of rescanning Rails per candidate reduces a
3+20 C100 `finish` median from 399.294 to 308.661 ms. Exact planner 11/11,
adaptive vnode, memcheck, and initcheck pass. Diagnostic hashes are:

```text
da2ee7d0e79b1eaa0f622ed56f5bd5ebcc8779de0dcf7a81456092bad1d28adc  c100-adaptive-packed-nsys.nsys-rep
b9191ef8b03a5e11a349c4fdedf42c495537838a657fb5b1fb08c0db73625232  hop-plan-both-n1024-c256-peaks.json
eba083d97f7e9fc53082197d9648a549caa9945ada81140a266c0a980a09c5a6  c100-adaptive-peaks-diag.json
```

The after timing is dirty-tree diagnostic evidence; a clean distribution is
required before acceptance. Scope remains single-node checked adapter.

The clean committed-tree 10+100 run closes that requirement:

```text
adaptive finish median / p95       308.736 / 308.977 ms
pre-O100 finish median / p95        399.294 / 399.510 ms
median reduction                                      1.293x
source stage median / p95             666.255 / 698.132 us
```

The result passes baseline eligibility and hashes to:

```text
68d3968621157d97d01fd6deca9d43f64d2fdea80a8591dea0f45faf839f7266  c100-adaptive-peaks-formal.json
```

This accepts the exact adaptive peak cache for the experimental planner. It
does not establish real Gin/RDMA or multi-node Rail performance.

### HA060-J adaptive record-parallel diagnostic

The post-O100 report confirms the residual planner remains the target:

```text
selected rank-0 planner kernel       237.244 ms (99.6% of GPU kernel time)
selected source-shuffle kernel         0.608 ms
all-call planner maximum             308.796 ms
```

Artifacts hash to:

```text
46e4e468c725f403d8383dbd2d9ee636ca6aaaf67adeaeb85ad88281a508a0e0  c100-adaptive-peaks-nsys.nsys-rep
aaa4c744433594fb0d72a37266b58839ac65d86ae16d4de7e09732445b9acb92  c100-adaptive-peaks-nsys.sqlite
60020663c44cb0bc2baf5163b3be5d3225dad6344781898eb9cc52b1b49e3111  c100-adaptive-peaks-nsys.json
```

Warp-parallel packed-row discovery reduces the N1024/C256 adaptive private
median from 17.199 to 11.791 ms. A C100 3+20 diagnostic reduces `finish` from
308.736 to 59.515 ms; its JSON hash is:

```text
9d70c18e9143a796de4d1edbded98a525c58e4c5d8bc789c17a6012ce715052f  c100-adaptive-warp-records-diag.json
```

Correctness and sanitizer gates pass. These are diagnostic results pending a
clean distribution and do not expand claim scope.
