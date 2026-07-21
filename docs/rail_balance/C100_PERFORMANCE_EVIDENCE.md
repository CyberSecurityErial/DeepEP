# C100 local performance evidence

This document is the audit surface for controlled local rail-balance
measurement.  Profiler-free samples are the performance truth; Nsys and NCU
are separate diagnostic experiments.  Raw reports stay under the ignored
`.cache/rail_balance/c100/` directory and are identified by checksum.

## Current state

```text
environment manifest: FROZEN
large source fixture: FUNCTIONAL_PASS, UNPROFILED
large return fixture: FUNCTIONAL_PASS, UNPROFILED
checked-adapter benchmark harness: AUDITED_PASS
profiler-free baseline: PARTIAL_SOURCE_COLLECTED, STABILITY_NOT_ACCEPTED
new Nsys attribution: NOT_COLLECTED
new NCU dossier: NOT_COLLECTED
performance-path change: NONE
```

The workload in this local phase is a synthetic BF16 communication-operator
microbenchmark, not a complete training step.  Its primary metrics will be
transaction latency distribution and effective moved payload bytes/second.
It cannot establish model tokens/s, step time, real Gin/RDMA throughput, QP
utilization, or end-to-end MoE speedup.

## Frozen workload contract

| Field | Value |
| --- | --- |
| Git branch | `feat/rail-balance-prototype` |
| Measurement-code commit | `056ad4d4df7007c3e4927e2f7d096ec5246da571` |
| Python | `/home/chen/.cache/deepep-sjlgpt/bin/python` |
| Topology | one process/GPU, world size 8, local LSA team 8 |
| Scaleout | disabled with `EP_DISABLE_GIN=1`; virtual D=9 plan only |
| Shape | G8/D9/K4/N1024/C256, seed and iteration 100 |
| Routing | owner `g` sends to destination `g+1`; four experts deduplicate to one payload copy |
| Balance | keep 128 and move 896 per owner; receive 896 per egress; 7,168 moved total; Pcap 896 |
| Widths | H256 and H7168 use the same complete schedule |
| Source payload | legacy BF16 `TokenLayout`, 576 or 14,400 bytes/record |
| Return payload | combine BF16 `TokenLayout`, 544 or 14,368 bytes/record |
| Default stream policy | the existing runtime communication stream; no CUDA Graph or `torch.compile` |
| DataLoader/model | none; deterministic synthetic raw records |
| Numerical gate | full raw-byte target comparison, source immutability, poison-preserved non-targets |

Large fixtures are available only by explicit case name:

```text
c100_volume_h256
c100_volume_h7168
```

The default C080 correctness suites do not inherit these allocations or
launches.

Logical payload volumes are fixed before timing:

| Stage | H | Bytes/record | Bytes/rank | 8-rank bytes |
| --- | ---: | ---: | ---: | ---: |
| source shuffle | 256 | 576 | 516,096 | 4,128,768 |
| source shuffle | 7,168 | 14,400 | 12,902,400 | 103,219,200 |
| return unshuffle | 256 | 544 | 487,424 | 3,899,392 |
| return unshuffle | 7,168 | 14,368 | 12,873,728 | 102,989,824 |

These values count user-visible `TokenLayout` bytes assigned to moved records.
They are not hardware NVLink sectors, HBM traffic, or Gin bytes.

## Audited profiler-free harness

`tests/elastic/bench_rail_balance_hybrid_lsa.py` is the committed measurement
surface.  It launches exactly eight local ranks and gives every cold, warm, and
steady sample a fresh invocation/ticket while reusing the ElasticBuffer and
input tensors.  Its timing truth is a same-host monotonic-clock global span:

```text
stage = max(stage_end across ranks) - min(stage_start across ranks)
transaction = max(transaction_end across ranks) - min(transaction_start across ranks)
```

This replaces the rejected `max(per-rank duration)` denominator, which can
hide rank launch skew.  Every report retains per-rank intervals, raw global
samples, median, p95, p99, mean, population standard deviation, CV, and the
logical byte numerator.  Source excludes the later visibility barrier; return
includes the checked wrapper's required B1/B4 path.  Neither is raw-kernel or
public Hybrid/Gin end-to-end latency.

The harness uses a fresh private JIT cache and requires exactly seven target
cubin families.  It hashes the Git/source tree, extension, JIT CU/cubin and
optional PTX/SASS, and the actual loaded NCCL/CUDA-runtime libraries.  It also
records the exact DeepEP-selected CUDA home/nvcc, pre/post GPU state, clocks,
power, throttle reasons, MPS state, and compute-process identities.  A
process-group watchdog performs bounded TERM/KILL/reap cleanup on exceptions,
timeouts, signals, or worker failure.

`baseline_collection_eligible` is only an automatic precondition gate.  A run
is accepted only after correctness evidence and review of the raw timing
distribution, and only when invoked directly without Torch Profiler, Nsys, or
NCU.  Profiler presence is deliberately declared by the run contract rather
than guessed from process names.

The committed harness passed py-compile, three CPU route/oracle suites, the
12-test validation contract, one final-SHA EP8 source report smoke, and an
independent final audit with Blocker 0 / High 0.  One immediately preceding
revision also passed the same report path.  Both used a dirty tree and
insufficient measurement depth, so their JSON correctly rejects baseline
eligibility and neither is a performance result.

## Partial clean source collection — commit `3c54839`

Six direct, unwrapped, profiler-free reports were collected from the clean
commit `3c548390d96c229a318966e987a9a624ba76719d`.  Every report used 10 warmup
and 100 steady iterations, a fresh JIT cache, an empty-GPU preflight, persistent
JSON, and passed `baseline_collection_eligible`.  Raw reports and a verified
checksum manifest remain under the ignored local directory
`.cache/rail_balance/c100/formal/3c54839/`.

These runs are valid raw evidence but are not accepted as a stable baseline:

| Case | Run medians (us) | Median range | Pooled median/p95/p99 (us) | Pooled CV | Min/max (us) |
| --- | --- | ---: | --- | ---: | --- |
| source H256 | 113.745 / 109.794 / 117.764 | 7.26% | 112.700 / 133.650 / 164.241 | 12.15% | 101.394 / 292.757 |
| source H7168 | 130.954 / 157.439 / 143.885 | 20.22% | 144.728 / 169.219 / 271.935 | 82.84% | 121.626 / 2274.190 |

The H7168 maximum is not an all-rank payload slowdown.  In run 1, steady
sample 59 had a 2,253.625 us rank-4 adapter duration while the other rank
durations were 85.855--108.684 us; its global span was 2,274.190 us.  Run 2's
770.897 us maximum combined a 429.110 us start skew with a 344.129 us largest
rank duration.  Run 3 was materially tighter (median 143.885 us, CV 4.73%).
This points to host/rank scheduling or an intermittent synchronization path as
a measurement-stability hypothesis, not yet a kernel root cause.

All pre/post GPU samples were P0 with 3,201 MHz memory clocks and no reported
non-benign throttle.  SM clocks were consistently heterogeneous: GPUs 1, 3,
and 6 reported 1,500 MHz while the other active GPUs reported 1,980 MHz.  The
host load average was approximately 8 on 192 logical CPUs and no dominant CPU
consumer was observed, but pre/post sampling cannot exclude a short scheduler
event.

Report SHA256 values are:

```text
23b145eee50db8c0905bf0fd25c1c849f63d023109353683a45a2ba67348243c  source-h256-r1.json
a41402c620604a2b59988177dd2dc3d8df1e14daed8199254085ad2f7698122a  source-h256-r2.json
8d99ed152a02474b8641592f2ea481ecfd8a31e81d76c2fe2a3f03792bddf1bf  source-h256-r3.json
075b92ecb0ea7312806a12ad44c39fd45c9f1e9de19fcd9e789f97fac3721a91  source-h7168-r1.json
e87d14328577a4e2c6262fb0b043496e1ddff706dbca26b8166a03daafbc9bbf  source-h7168-r2.json
068615503752d68f28bc00603d8dbfd02fb8b95c0a6e1faa41cd7f842ee2d1d0  source-h7168-r3.json
```

Return collection was intentionally stopped at the user's request.  The first
return-H256 process received SIGINT, the watchdog exited 130, all child/GPU
processes were reaped, and no partial report exists.  No Nsys, NCU, or
performance-related source change followed this partial collection.

## Stability falsification E1 — one PyTorch intra-op thread

At clean commit `4211baef530cd94c72eea7dd4a6ab6df1d3d6f29`, the first
single-variable experiment retained the exact source-H7168 workload, timing
boundary, process topology, warmup/steady counts, and report path, and changed
only `OMP_NUM_THREADS` from unset to `1`.  A separate out-of-band probe of the
pinned runtime reports 96 PyTorch intra-op and 96 inter-op threads by default,
and one intra-op plus 96 inter-op threads with the variable set.  The reports
themselves record the environment variable, not these runtime thread counts.
All three direct runs passed the automatic collection gate and retained 10
warmup plus 100 steady samples.

| Run | Median/p95/p99 (us) | Mean/std (us) | CV | Max (us) | Median start skew (us) |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 132.625 / 149.080 / 163.141 | 134.558 / 7.574 | 5.63% | 167.547 | 44.943 |
| 2 | 152.436 / 163.621 / 175.894 | 153.530 / 6.917 | 4.51% | 194.810 | 60.566 |
| 3 | 137.857 / 154.795 / 184.107 | 140.081 / 11.373 | 8.12% | 218.716 | 46.031 |
| pooled | 140.501 / 160.198 / 175.784 | 142.723 / 11.902 | 8.34% | 218.716 | n/a |

Compared with the original source-H7168 collection, this removes the
millisecond-scale tail: pooled CV falls from 82.84% to 8.34% and maximum span
falls from 2,274.190 us to 218.716 us.  This supports host thread pressure as a
cause of the severe tail.  It does not establish a stable global-span
baseline: the three run medians still span 14.94%.

The tracked benchmark, extension, generated source and production sources are
unchanged between the two collections; the six source-shuffle cubins also
produce the same disassembled-instruction hash.  Median rank-local maximum
stage duration changes only from 97.518 to 96.556 us, while its p99 and maximum
collapse.  Therefore E1 is evidence for tail control, not evidence that the
GPU kernel became faster.  The default and OMP runs were collected in two
sequential groups rather than as an interleaved paired A/B, which further
forbids a speedup claim.

The remaining variation is dominated by a repeatable post-Gloo start order,
not a comparable change in rank-local adapter duration.  Rank 1 starts first
in 299/300 steady samples; rank 7 starts last in 295/300.  Median global span
minus start skew is 87.775/93.237/91.610 us across the three runs, while median
start skew itself changes from 44.943 to 60.566 us.  This is evidence that the
current `max(end)-min(start)` truth includes deterministic control-plane
release skew.  It does not yet prove whether the release order originates in
Gloo progress, host scheduling, or both.

The container exposes CPUs 0--191 across NUMA nodes 0--1 but has a cgroup-v1
CPU quota of 9,600,000 us per 100,000 us period, equivalent to 96 CPUs.  Its
historical `cpu.stat` already reports throttled periods.  The benchmark does
not set CPU affinity, so every rank and helper thread inherits CPUs 0--191.
This host evidence explains why limiting intra-op threads is a relevant
falsification, but no cgroup delta or per-thread migration evidence was
captured inside these reports; it is not a complete causal proof.

Raw reports and their verified manifest remain in the ignored local directory
`.cache/rail_balance/c100/stability-omp1/4211bae/`:

```text
93771ef50adc24d7067b56994b96c467931e3213e7b6f83d5923de18720e0b69  source-h7168-r1.json
4e6b064cfea4fe04de6ed8357988a9b3135df6ab5fb1f325e056314e7a39636f  source-h7168-r2.json
6d6a5e5c3093e42cf45b6c6847cb64a4e53fd7ba6aa108b90e25be8cab2ebe9e  source-h7168-r3.json
```

No CUDA/JIT/runtime/hot-path code changed, and no Nsys or NCU was collected.
The next falsification must isolate the rank-release skew before return
baselines or kernel attribution are accepted.

### Pre-registered E2 CPU-only gate criterion

E2 removes CUDA, DeepEP, NCCL and ElasticBuffer entirely, but preserves eight
spawned ranks and a separately created eight-rank Gloo control group.  It
records common-host timestamps immediately before and after
`monitored_barrier(wait_all_ranks=True)`.  Every cold/warm/steady rank interval
is retained by `tests/elastic/bench_rail_balance_gloo_gate.py`; the report
asserts that CUDA was never initialized.

Before formal collection, the falsification criterion is fixed as follows:
the steady median barrier-return span must explain at least 50% of each E1 run's
median stage-start skew, and the dominant earliest and latest return ranks must
each occur in at least 80% of steady samples.  Meeting both supports the claim
that Gloo return plus host wakeup is sufficient to create most of the observed
global-span skew.  Missing either rejects that narrow claim and requires an
Nsys OS-runtime/CUDA timeline.  Even a positive result is not a pure Gloo
protocol latency measurement and cannot describe CUDA kernel performance.

### E2 formal result — criterion passed

Three clean, direct runs at commit `ee41415e86a5b1d220bb2ee2bb119ec5e45566e6`
used `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`, 10 warmup and 100 steady
iterations.  Pre/post Git and source identities match, every rank reports CUDA
uninitialized, and no run increments the cgroup's throttled-period or
throttled-time counters.

| Run | Return-span median/p95/p99 (us) | Mean/std (us) | CV | Min/max (us) | rank 1 first | rank 7 last |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 1 | 38.693 / 50.225 / 133.799 | 41.923 / 15.025 | 35.84% | 32.815 / 147.017 | 98/100 | 98/100 |
| 2 | 36.319 / 40.327 / 42.255 | 36.216 / 2.383 | 6.58% | 27.521 / 47.137 | 100/100 | 100/100 |
| 3 | 38.028 / 44.075 / 45.103 | 38.832 / 2.725 | 7.02% | 31.702 / 45.506 | 98/100 | 98/100 |
| pooled | 37.864 / 44.175 / 57.040 | 38.990 / 9.223 | 23.65% | 27.521 / 147.017 | n/a | n/a |

Run medians span 6.54%.  Even the smallest run median, 36.319 us, explains
80.8%, 60.0%, and 78.9% of the three E1 median stage-start skews.  The dominant
first/last ranks occur at least 98% of the time.  Both pre-registered thresholds
therefore pass: bare Gloo return plus host wakeup is sufficient to create most
of the measured post-gate start skew without any CUDA work.

This result neither assigns E1's former millisecond tail to Gloo nor measures
pure Gloo protocol latency.  It proves only that the artificial pre-stage gate
adds a large, deterministic component to the benchmark's global span.  Future
reports continue to retain that honest global span.  Operator attribution must
also use the already recorded maximum rank-local synchronous adapter duration
and then an Nsys CUDA/OS-runtime timeline; no sample is corrected by subtracting
the gate median.

Verified local artifacts live under the ignored directory
`.cache/rail_balance/c100/gloo-gate/ee41415/`:

```text
9675e24eb7747346e5183397fdb83dcc2ebaa9e5b9df825bff407c316e61bd59  gloo-omp1-r1.json
21d7cb6c4c233b7b16e718c12c8f1afe771b365d3310d6ec554ee903a9ff0d02  gloo-omp1-r2.json
b274444bda0fec04228d4eb9a4c80c34d39888a8a0918a8964cc8f92646dbd5f  gloo-omp1-r3.json
```

## OMP-controlled source-H256 collection — commit `bac80ee`

Three clean source-H256 processes used OMP=1, 10 warmup, 100 steady samples,
fresh JIT caches and empty-GPU preflight.  Every automatic eligibility gate
passes and no report contains an unexpected compute process.  Global span and
the already recorded maximum rank-local synchronous call envelope are reviewed
separately:

| Run | Global median/p95/p99 (us) | Global CV/max | Rank-local-max median/p95/p99 (us) | Rank-local CV/max |
| --- | --- | --- | --- | --- |
| 1 | 130.917 / 151.016 / 173.075 | 39.83% / 678.144 | 76.967 / 91.108 / 171.023 | 49.21% / 472.997 |
| 2 | 126.277 / 140.649 / 157.166 | 9.38% / 224.434 | 75.243 / 84.926 / 105.831 | 20.42% / 224.434 |
| 3 | 116.707 / 140.508 / 208.682 | 44.59% / 665.199 | 79.364 / 96.699 / 158.718 | 63.96% / 630.329 |
| pooled | 126.501 / 145.071 / 204.275 | 35.34% / 678.144 | 76.793 / 92.751 / 168.538 | 49.73% / 630.329 |

Global run medians span 12.18%; their start-skew medians are
63.929/59.297/44.552 us and rank 1 starts first in every steady sample.  The
rank-local envelope medians are materially tighter at 5.48% range, but their
p99/max values retain large intermittent tails.  This group therefore supports
the two-metric interpretation but is not accepted as a fully stable p95/p99
baseline.  OMP=1 is not claimed to improve H256 relative to the original
collection.

Verified local hashes under
`.cache/rail_balance/c100/formal-omp1/bac80ee/` are:

```text
18e32bb9516642e5ef807af879f257aaedc231b8164c8a1eeaf5f5d17f97c178  source-h256-r1.json
c88c67f99b4a4e8af0a9871cab5b444aebd44ccc1935aaaba92532825e0b46bc  source-h256-r2.json
85a02c0354fb583f5da894ff842bdc9834edad3ec4081a4a87ac4ab67d213e65  source-h256-r3.json
```

## Environment manifest — 2026-07-21 UTC

### Host and GPU management view

- Host kernel: Linux `5.10.134-19.103.al8.x86_64`.
- CPU: 2 × Intel Xeon Platinum 8558, 48 cores/socket, 2 threads/core,
  192 logical CPUs, two NUMA nodes.
- GPU 0--3 CPU affinity: NUMA 0; GPU 4--7: NUMA 1.
- User-authorized target hardware: one 8-GPU H200/NVSwitch node.
- `nvidia-smi` management label: 8 × `NVIDIA L20X`, 143,771 MiB each,
  reported compute capability 8.9.
- `ncu --query-metrics` device label: `NVIDIA L20X (GH100)`.
- Executed DeepEP JIT cubins: ELF `sm_90a`, CUDA toolkit 12.8.
- `nvidia-smi topo -m`: every GPU pair is `NV18`.
- `nvidia-smi nvlink --status`: 18 links/GPU at 26.562 GB/s per link.
- `nvidia-smi topo -p2p p`: every cross-GPU pair reports `NS`, despite the
  true-EP8 LSA kernels passing exact peer-memory tests.
- MIG disabled on all devices; no MPS server found.
- Power limit 700 W/device.  At capture, temperatures were 30--35 C; software
  power cap, hardware slowdown, thermal slowdown, power-brake slowdown, and
  sync boost reasons were all inactive.
- No GPU compute process existed at capture.
- The worktree contained only this new manifest and two untracked CPU-only
  C105 contract files; no tracked runtime/CUDA source differed from the base
  commit.  Actual baseline runs require a committed, explicitly recorded tree.

The management label, reported capability, compiled ISA, NVLink topology, and
P2P query are mutually inconsistent.  The user's H200 target remains the
operational contract, but published H200 peak values must not be used as the
denominator for measured efficiency until the management alias/P2P reporting
is explained.  This does not prevent same-machine paired timing or functional
evidence.

### Software and linked libraries

| Component | Exact value |
| --- | --- |
| NVIDIA driver | 570.172.08 |
| CUDA toolkit / nvcc | 12.8 / V12.8.61 |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| cuBLAS package | 12.8.4.1 |
| cuDNN package/runtime | 9.19.0.56 / 91900 |
| CUDA runtime package | 12.8.90 |
| CUDA NVRTC package | 12.8.93 |
| NCCL reported by `torch.cuda.nccl.version()` | 2.28.9 |
| NCCL package linked by `deep_ep._C` | 2.30.4 |
| Transformer Engine | not installed |
| FlashAttention | not installed |
| Nsight Systems | 2024.6.2.225-246235244400v0 |
| Nsight Compute | 2025.1.1.0 build 35528883 |

The NCCL API string and the `libnccl.so.2` selected by `ldd deep_ep/_C*.so`
differ and must remain separate evidence fields.  Real Gin validation must
also report the NCCL library loaded by that exact cluster process.

PyTorch matmul TF32 was false and cuDNN TF32 was true at capture; neither is
used by this byte-copy fixture.  The path is eager custom-extension execution,
with no Inductor, Triton kernel, CUDA Graph, gradient accumulation, optimizer,
or DataLoader.

### Installed profiler capabilities

The local NCU exposes sets `basic`, `detailed`, `full`, `nvlink`, `pmsampling`,
and `roofline`; `--query-metrics` lists 5,364 metrics.  Installed sections
include LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution,
ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SchedulerStats,
WarpStateStats, SourceCounters, InstructionStats, Nvlink, PmSampling, and the
installed roofline charts.

Nsys 2024.6 lists the relevant recipes `cuda_gpu_kern_sum`,
`cuda_gpu_kern_hist`, `cuda_gpu_kern_pace`, `cuda_api_sum`,
`cuda_gpu_mem_{time,size}_sum`, `gpu_gaps`, `gpu_time_util`,
`nvtx_gpu_proj_{sum,trace,pace}`, `osrt_sum`, `nccl_sum`,
`nccl_gpu_proj_sum`, `nccl_gpu_overlap_trace`, `nvlink_sum`,
`network_traffic_map`, and `diff`.  Recipe Python dependencies must be checked
before treating recipe failure as missing trace data.  Every SQLite analysis
must inspect the actual exported schema and time units first.

## Matching official hardware and tool facts

- NVIDIA specifies H200 SXM as Hopper with 141 GB HBM3e, 4.8 TB/s memory
  bandwidth, up to 700 W, and 900 GB/s NVLink.  Tensor-Core figures on that
  page marked with footnote 2 include sparsity and are not dense rooflines:
  <https://www.nvidia.com/en-us/data-center/h200/>.
- CUDA 12.8 documents TMA as a compute-capability-9.0 feature for asynchronous
  bulk/tensor copies between global and shared memory:
  <https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma>.
- NCCL describes LSA as CUDA-P2P load/store-accessible communication and a
  rail team as ranks with the same local LSA index.  GIN Rail puts require the
  peer to belong to that rail team:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html>
  and <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html>.
- Nsys commands/schema are matched to the installed 2024.6 guide:
  <https://docs.nvidia.com/nsight-systems/2024.6/UserGuide/index.html>.
- NCU replay, cache control, clock control, and serialization can perturb the
  measured kernel; NCU duration is not an end-to-end substitute:
  <https://docs.nvidia.com/nsight-compute/2025.1/ProfilingGuide/index.html>.

## Native DeepEP mechanisms to preserve

Any later experiment must first prove why one of these native mechanisms is
insufficient; it must not add a parallel abstraction by default.

- fixed notify/scaleout/forward warp roles and TMA/mbarrier staging:
  `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`;
- one payload stage reused after destination deduplication and aggregated GIN
  requests with interval tail publication: the same dispatch implementation;
- round-robin destination forwarding with TMA receive and LSA peer store: the
  same dispatch implementation;
- linked-list replay, TMA reduction/staging, delayed GIN puts and interval tail
  publication: `rail_balance_hybrid_combine.cuh` reuses these combine rules;
- aligned hidden/scale/metadata/mbarrier layout:
  `deep_ep/include/deep_ep/common/layout.cuh`;
- SM/QP selection based on estimated HBM, NVLink and RDMA traffic:
  `deep_ep/buffers/elastic.py`.

The current rail-balance data path already uses compact plan prefixes,
minimum-move quota, static direct-final proxy slots, moved-only TMA source
shuffle, native retained dispatch, native reduction/combine, and a single
return-unshuffle TMA load/store.  Vnode is a test-only correctness/fault
transport and must not be profiled as the production hot path.

## Existing reports and claim boundary

Six old reports and `SHA256SUMS` are backed up in
`.cache/rail_balance/c100/`.  They describe a tiny source fixture that moved
only three 576-byte user records (1,728 bytes total) and took approximately
125--130 us under the captured tools.  They prove report collection and kernel
identity, not payload-throughput root cause.  The binary-resolver experiment
was slower and was reverted; active-grid truncation remains without a speedup
claim.

No old report may be relabelled as evidence for the new 7,168-record fixture.

## Missing evidence before optimization

1. Explanation for the H200 operational target versus the management labels,
   compute-capability field and P2P `NS` output.
2. Stable, accepted profiler-free source and return samples for H256/H7168.
   Six clean source reports failed the original stability review.  Three
   OMP-one-thread H7168 reports remove the severe tail but retain 14.94%
   cross-run median span from rank-release skew; return remains uncollected.
3. Nsys exposed critical-path attribution for the new large fixture.
4. An exact NCU invocation selected from that Nsys report.
5. Sustained same-machine peer-copy/HBM reference if a bandwidth percentage is
   later needed; published peak alone is insufficient.
6. Real D>1 Gin/RDMA/QP/NIC behavior and network-visible counters.
7. Full-MoE or training-level target metric and compute/communication overlap.

Until items 2--4 exist, standalone TMA waits, segment scans, QP choice, tail
publication cadence, planner launch, and barriers are hypotheses only.  No
performance-path code change is authorized by this manifest.

## Environment collection commands

The manifest was built from the exact installed tools, including:

```bash
nvidia-smi --query-gpu=... --format=csv
nvidia-smi topo -m
nvidia-smi topo -p2p p
nvidia-smi nvlink --status
nvcc --version
nsys --version
nsys stats --help
nsys analyze --help
nsys recipe --help
nsys export --help
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics
ldd deep_ep/_C.cpython-311-x86_64-linux-gnu.so
```

The benchmark boundary has passed correctness and audit.  Formal collection
uses the committed script directly with `EP_DISABLE_GIN=1`, all eight visible
GPUs, a persistent per-run JSON path, at least 10 warmup iterations and 100
steady iterations.  Source/return and H256/H7168 are separate processes with
fresh JIT caches.  Exact commands and report hashes are recorded with the
accepted runs; Nsys/NCU use separate diagnostic invocations and artifacts.
