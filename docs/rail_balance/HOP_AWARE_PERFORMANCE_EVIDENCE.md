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

#### 2026-08-01 workload-matching correction

The historical private-planner runner selected different records implicitly:
`one_hop` used the rotating-target pattern, while `adaptive` used the
singleton-target pattern.  Therefore historical rows from the two modes are
**not a matched-workload A/B** and cannot support a one-hop-versus-adaptive
performance conclusion.  Their within-mode before/after comparisons remain
diagnostic evidence only when both revisions used the same implicit pattern.

The current runner makes the pattern explicit and uses one pattern for every
requested mode.  Reproduce the old per-mode inputs, if provenance requires it,
with two separate commands using `--mode one_hop --pattern rotating` and
`--mode adaptive --pattern singleton`.  New matched comparisons must use one
command with `--mode both --pattern <rotating|singleton>` and must not be
joined to the old JSONs as a continuous performance series.

Reproduce a matched profiler-free current-tree probe with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD:$PWD/tests/elastic \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hop_plan.py \
  --mode both --pattern rotating --tokens 8 32 128 512 \
  --warmup 3 --steady 20 \
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

Within each historical mode, one-hop is the unchanged control. Its stable
before/after result weakens a global clock or allocator explanation for the
adaptive within-mode improvement; it is not a matched cross-mode comparison.

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

Clean commit `7b98f83` 10+100 distributions accept the optimization:

| Mode | finish median / p95 (ms) | source median / p95 (us) |
| --- | ---: | ---: |
| one-hop | 41.510 / 41.758 | 119.918 / 152.539 |
| adaptive | 59.493 / 59.689 | 642.279 / 717.373 |

Adaptive finish is 5.19x below the O100 308.736 ms boundary; one-hop changes
only +0.33% from 41.372 ms. Both reports pass baseline eligibility and hash to:

```text
2e92c80f019977837927682e2f9b3ed9a0ef652bb91d09370aa32ba265a1d3e2  c100-onehop-warp-records-formal.json
0de35c1bb93412c15c3df2133dcfd548a75d56e13e6099571333554ad43967f8  c100-adaptive-warp-records-formal.json
```

The evidence remains a checked single-node adapter result, not real Rail/NIC
performance.

### HA060-K singleton endpoint diagnostic

O101's matched Nsys range measures:

```text
adaptive planner kernel              46.378 ms (98.6%)
source-shuffle kernel                 0.525 ms
record materializer                   0.0069 ms
```

The report artifacts hash to:

```text
26e1eadd280566833a08497db7a240691668cb78c80aac95871b68a3b49fbf92  c100-adaptive-warp-records-nsys.nsys-rep
69fdcc7a71a5e8941ac3e3bbe22f2a56547c2816da37b46d906831cd3c7d2598  c100-adaptive-warp-records-nsys.sqlite
65084977f0842b03fa240500bb0d2bc8a950fddf017ecf13a68b2f1ebb0e6bdf  c100-adaptive-warp-records-nsys.json
```

Bypassing the score for one-bit endpoint sets produces 3+20 C100 finish
medians of 34.669 ms one-hop and 52.918 ms adaptive, from clean predecessors
41.510 and 59.493 ms. The diagnostic JSON hashes are:

```text
5a2a3c7fbfc4849b666d2500217ed2a3ae7d95e1ccd40cb7bedd3e38f7abfa8c  c100-onehop-singleton-diag.json
4c3af2bb17bfe53f05a5b498ae1219d7e0632663fc6d0fd7fb452c4009d333e0  c100-adaptive-singleton-diag.json
```

Correctness gates pass; clean distributions are pending and claim scope is
unchanged.

Clean commit `7b0466f` 10+100 distributions accept the singleton bypass:

| Mode | finish median / p95 (ms) | source median / p95 (us) |
| --- | ---: | ---: |
| one-hop | 34.701 / 34.898 | 116.921 / 138.497 |
| adaptive | 52.919 / 52.989 | 665.269 / 744.713 |

The finish boundaries improve 1.196x/1.124x over O101. Reports pass baseline
eligibility and hash to:

```text
55924e573f163b3936faec529df93a951b24b97b65227d02bee75b9772b8dfde  c100-onehop-singleton-formal.json
be007ce3b37100ce4d500de6e3d34b4bc4c78754c6baa587ab9ed1011f96c29c  c100-adaptive-singleton-formal.json
```

Claim scope remains checked single-node adapter only.

### HA060-L multi-candidate peak reuse

The pre-edit rot1 one-hop C100 3+20 `finish` median is 179.096 ms; exact
per-record peak reuse reduces it to 173.004 ms. Diagnostic artifacts hash to:

```text
c38b2391038ae039bd79469a7c6c8253f452b6d39bc16140c92c133b15faa729  c100-rot1-onehop-prepeaks-diag.json
da65b053744e5129dfb6a52fbd1eb60a20f846209119e6194be93996cd55ab0f  c100-rot1-onehop-initial-peaks-diag.json
```

This is a 3.5% diagnostic improvement with exact/vnode/sanitizer evidence.
Clean evidence remains pending; no claim scope changes.

Clean commit `6a6cb2c` records rot1 one-hop `finish` at 173.023 ms median /
173.222 ms p95 and source stage at 284.318 us median. The report passes
eligibility and hashes to:

```text
6a43a1ab01623a469ad59bc26414b71e01f223db61d26c5854f860bd7a34bbfb  c100-rot1-onehop-initial-peaks-formal.json
```

This accepts O103 but falsifies the serial planner as a viable final hot path.

### HA070-E aggregate adaptive quotas

Commit `8c56939` uses the same parallel static-slot materializer for one-hop
and adaptive. Its clean checked-adapter 10+100 distributions are:

| Mode | finish median / p95 (ms) | source median / p95 (us) |
| --- | ---: | ---: |
| one-hop | 24.001 / 24.100 | 115.401 / 160.635 |
| adaptive | 23.116 / 23.221 | 126.176 / 146.244 |

The reports pass eligibility and hash to:

```text
ac866ee5f255d3a6c57a6a70096336254ac2091536918bf8b39102a44d01b882  c100-onehop-10x100.json
c064162bfac9f720f2a5b8065905a998611e486fc5c14c4998149ed2241c483a  c100-adaptive-10x100.json
```

The corresponding eight-rank Nsys trace is the important falsification
experiment. Unlike the singleton-target private planner, C100 produces
four-target records. The aggregate adaptive kernel is 17.387 ms median
(32 invocations) and 68.1% of captured GPU kernel time, while source shuffle
is 25.344 us median. Therefore the current critical path remains the lane-0
multi-target decision loop, not source shuffle and not an inferred host gate.

```text
e264a1352b856b8248a7173cfa763990840a8c993abf57798f87efe841bc38bf  c100-adaptive-0x3.nsys-rep
3ff92e73c7565881379e4fca9be5cb9b8400542690dfa0e26e84a56dcd5f3bb7  c100-adaptive-0x3.sqlite
```

These remain single-node checked-adapter results. They do not establish
Gin/RDMA/NIC speedup or unlock capability.

## HA070-F: sparse groups, fair batches, and bounded adaptive rounds

Scope: uncommitted dirty tree after `ee80a16`; all values in this section are
diagnostic only. They are not eligible formal speedups.

| C100 diagnostic | Adaptive finish |
| --- | ---: |
| dense padded multi-target scan | ~1.586 s |
| sparse active-group scan | ~30.08 ms |
| sparse + tied-hot fair batch | ~12.05 ms |
| singleton `rot1` counterexample to fair batching | ~500.27 ms |
| bounded-round budget, four-target `volume` | ~5.87 ms |
| bounded-round budget, singleton `rot1` | ~6.05 ms |

The sparse version stores only real `(owner,destination,target-mask)` groups,
so the decision loop no longer scans empty padding. Fair batching avoids one
of several tied hot Rails consuming the entire two-hop budget. It was not
general enough: on singleton traffic, more than 95% of selection rounds moved
one unit. The current bounded-round rule derives a required batch from the
remaining two-hop allowance and decision budget, while clipping at real load
and capacity boundaries.

Failures preserved:

- removing one apparent batch-one `gap` condition left singleton latency at
  roughly 500 ms and was reverted;
- using `planner_chunk_size` as the adaptive minimum batch caused a valid
  capacity-edge test to fail; planner materialization granularity is not an
  admissible lower bound for adaptive movement.

Correctness evidence on this tree is 18/18 focused GPU tests, including 81
realizable Top-k structures spanning `G=2..8`, `K=1/2/4/8`, and every target
width from one through `min(K,G)`. One-hop off-diagonal and adaptive diagonal vnode round trips
also pass. The C100 `mesh` checked adapter reports status zero and these rank-0
counters:

| Mode | Path units `(direct,dst,src,2-hop)` | Moved | Source peak |
| --- | ---: | ---: | ---: |
| one-hop | `(1024,3584,3584,0)` | 3584 | 4608 |
| adaptive | `(1024,3104,2016,2048)` | 4064 | 3040 |

The checked hop-aware runtime allocates 4096 proxy slots per egress for
`volume` and 8192 for `rot1`/`mesh`. A separate private call with the legacy
896 moved-copy count as total capacity correctly returned
`CapacityExceeded`; it omitted retained staging and is excluded from all
claims.

Artifacts:

```text
.cache/rail_balance/hop-aware/multi-target/smoke-c100-volume-adaptive-budget-v7.json
.cache/rail_balance/hop-aware/multi-target/smoke-c100-rot1-adaptive-budget-v7.json
.cache/rail_balance/hop-aware/multi-target/smoke-c100-mesh-onehop-diag-v7.json
.cache/rail_balance/hop-aware/multi-target/smoke-c100-mesh-adaptive-diag-v7.json
.cache/rail_balance/hop-aware/multi-target/vnode-onehop-offdiag-budget-v7.json
.cache/rail_balance/hop-aware/multi-target/vnode-adaptive-diag-budget-v7.json
```

The four C100 JSONs above record a dirty Git tree and
`baseline_collection_eligible=false`. The two vnode JSONs record only
`single_node_diagnostic` scope and do not carry the C100 Git/baseline fields.
No Gin/RDMA, NIC, or multinode performance conclusion follows from them;
clean 10+100 collection is pending.

## HA070-G: bounded tiny tail and measured local-forward work

Scope remains an uncommitted, dirty single-node diagnostic. The unrestricted
small-group fallback fixes a real cap-as-target counterexample but is not
acceptable on the main path:

| Adaptive variant | C100 `rot1` finish | Source peak |
| --- | ---: | ---: |
| batch floor only | ~6.05 ms | ~7,305 |
| unrestricted small fallback | 20.77 ms | 7,170 |
| at most G tiny-tail rounds | 7.24 ms | 7,194 |

The bounded tail also passes the minimal G8 regression: equal background load
of 129 units per Rail plus two quota-one diagonal groups starts with source
loads `(131,129,...,129)` and ends with a peak of 130. The 25% two-hop setting
is treated as a limit; the planner does not try to consume its full allowance.
CPU falsification covered 960 broader states and 1,080 ratio/seed cases
without cap, determinism, or single-step peak violations. It also found a
separate full-batch-versus-granular quality counterexample; that remains a
future single-variable experiment and is not hidden by this change.

The cold C100 volume snapshot now measures the shared-payload forwarding
semantics:

```text
path units (direct,dst,src,third) = (0,1687,4457,2048)
source-forward units             = 6505
destination-forward units        = 27287
minimum local-forward units      = 28672
extra local-forward units        = 5120
```

A 2026-08-01 review found that the original dirty diagnostic helper counted
only owner bit zero; its emitted minimum/extra values were 31,744/2,048. The
corrected minimum is 28,672 units; source and destination forwarding remain
6,505 and 27,287 units. The corrected extra work is 5,120 units, or 2,949,120
bytes at 576 bytes per unit. The listed JSON is retained for provenance but
must not be used as corrected accounting evidence.

This makes the claim boundary explicit: `path_units` classify a
`target_mask` payload's egress, while the new counters measure actual local
forwarding to every final target. The corresponding JSON artifacts are:

```text
.cache/rail_balance/hop-aware/multi-target/smoke-c100-volume-adaptive-tinytail-localhops-v8.json
.cache/rail_balance/hop-aware/multi-target/diag-c100-rot1-adaptive-tinytail-v8.json
.cache/rail_balance/hop-aware/multi-target/vnode-onehop-offdiag-tinytail-v8.json
.cache/rail_balance/hop-aware/multi-target/vnode-adaptive-diag-tinytail-v8.json
```

The C100 reports have `baseline_collection_eligible=false`, and the vnode
reports are single-node diagnostics. None establishes real NIC/RDMA speedup
or a clean-tree before/after claim.

## HA070-I: committed functional candidate; no new performance Leader

The sparse multi-target/G2/tiny-tail implementation was committed as
`286da0a`; evidence hardening and supervised CUDA harnesses followed in
`771fcc6` and `0f38688`.  The core source is therefore recoverable and clean,
but source cleanliness alone is not a performance result.  The last accepted
matched profiler-free Leader remains `8c56939`:

| Source | Evidence available | Performance verdict |
| --- | --- | --- |
| `8c56939` | clean checked-adapter 10+100 and matched Nsys | retained Leader for checked single-node adapter only |
| `286da0a` | force build; 20/20 planner; CPU/API/codegen; G2/tiny-tail four-sanitizer gate | functional candidate; no clean matched timing |
| `0f38688` | same core plus benchmark/harness supervision | no source-version performance comparison |

The formal source-round control plane now fixes four shared parent blocks plus
four blocks per each of 2--4 candidates, a global non-overlapping schedule,
paired robust-noise qualification and absolute-latency winner selection.  Its
coordinator remains check-only, while the whole-round executor is now
implemented but defaults to check-only and has no real candidate source
manifest or live raw round.  All planned source rounds therefore remain
`NOT_RUN` and promotion is impossible.  The old
`takeover-scaffold-20260801-01` lacks `FINALIZED.json`; the terminal schema
smoke used `--no-execute`.  Neither is timing evidence.

The machine did not offer eight exclusive GPUs: Qwen occupied GPU 0--1.  No
vnode, benchmark, Nsys or NCU result was collected, and no competitor result
was produced.  This section intentionally records **no speedup** and makes no
claim about NIC/RDMA or any external system.

## HA070-J: committed formal executor; still no new performance evidence

Commit `448a727` adds the fail-closed whole-round source executor that was
missing at HA070-I.  It holds one lease across the exact `4+4N` block order,
executes the frozen campaign runner through an opened and hash-bound file
descriptor, preserves failed-candidate evidence, seals per-block coordinator
records, constructs the formal round manifest, and accepts an evaluator result
only after its source binding and promotion fields agree.  Default invocation
is check-only; live execution additionally requires an exact round-id
confirmation.

This is infrastructure evidence only:

```text
campaign supervisor                              20/20 PASS
source-round coordinator                           9/9 PASS
formal evaluator                                  19/19 PASS
whole-round executor                              16/16 PASS
Ruff / py_compile / independent static audit             PASS
```

There is no fabricated source manifest and no live executor artifact because
new CUDA candidates must come from measured hypotheses, not from a
control-plane smoke.  Qwen occupied GPU 0--1, so no eight-GPU vnode,
benchmark, Nsys, NCU or competitor run was launched.  `8c56939` therefore
remains the last accepted clean matched checked-adapter Leader, while
`286da0a` remains a functional candidate awaiting the same formal
profiler-free comparison.

## HA070-K: source-round authoring path; still no new performance evidence

The CPU-only `prepare_rail_balance_source_round.py` now closes the gap between
a human preregistration and the already-audited coordinator/executor.  From an
owner-controlled strict JSON spec and already-existing clean worktrees, it
derives the parent/candidate commits and trees, shared Git identity, independent
one-commit binary diffs, frozen execution-contract hash, and 20 harness hashes.
It then runs the coordinator preflight twice and no-clobber publishes the
`NOT_RUN` plan before the manifest terminal marker.  It does not create or edit
a candidate, compile code, launch CUDA, benchmark, profile, or choose a winner.

CPU/static evidence for this authoring addition is:

```text
source-round preparer                            10/10 PASS
source-round coordinator                           9/9 PASS
whole-round executor                              16/16 PASS
formal evaluator                                  19/19 PASS
campaign supervisor                               20/20 PASS
Ruff / py_compile / git diff --check                     PASS
```

The last clean finalized resource-gate artifact before this addition is
`takeover-scaffold-20260801-02`, bound to clean commit
`be4a69b9e363614373b1298d94e338a340fdcfa0`.  Its six CPU gates passed, all 19
exclusive-eight-GPU stages were skipped, and no GPU process was started.  Its
`result.json`, `SHA256SUMS`, and `FINALIZED.json` SHA256 values are respectively
`b2c12aba...`, `4b51d963...`, and `aef4a7be...`; the full values are recorded
in the experiment protocol and handoff.  `round_evaluation_allowed=false` is
the required scaffold-only verdict.

Qwen still occupied GPU 0--1, so this checkpoint adds no vnode, source-round,
benchmark, Nsys, NCU, NIC/RDMA, or competitor sample.  It cannot alter the
performance Leader.  A process crash between publishing the plan and manifest
can leave a fail-closed orphan plan; a later failure can leave both names or a
hidden stage file.  This is not a transactional claim: only strict executor
check-only validation can accept a pair, and artifacts must not be manually
repaired into evidence.
