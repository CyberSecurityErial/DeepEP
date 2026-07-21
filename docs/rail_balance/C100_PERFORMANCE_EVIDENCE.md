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
profiler-free baseline: NOT_COLLECTED
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
2. Stable profiler-free cold and steady-state raw samples for source and
   return, H256 and H7168, with clocks/power/process audit.
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
