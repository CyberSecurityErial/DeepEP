# C100 local performance evidence

This document is the audit surface for controlled local rail-balance
measurement.  Profiler-free samples are the performance truth; Nsys and NCU
are separate diagnostic experiments.  Raw reports stay under the ignored
`.cache/rail_balance/c100/` directory and are identified by checksum.

## Current state

```text
environment manifest: FROZEN
large source fixture: FUNCTIONAL_PASS, NO_DIAGNOSTIC_PROFILE
large return fixture: FUNCTIONAL_PASS, NSYS_ATTRIBUTED
checked-adapter benchmark harness: AUDITED_PASS
profiler-free baseline: SOURCE_AND_RETURN_COLLECTED, TAIL_STABILITY_NOT_ACCEPTED
new Nsys attribution: FORMAL_H7168_PASS, EXACT_NCU_TARGET_SELECTED
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

## OMP-controlled return-H256 collection — commit `d05411a`

Three clean return-H256 reports use OMP=1, 10 warmup and 100 steady samples.
The accepted set is `{r1,r3,r4}`.  All three pass the automatic environment,
identity and persistence gates.  Their global run medians span only 4.44%, and
their rank-local call-envelope medians span 2.76%, but the tail remains
unstable:

| Run | Global median/p95/p99 (us) | Global CV/max | Rank-local-max median/p95/p99 (us) | Rank-local CV/max |
| --- | --- | --- | --- | --- |
| 1 | 225.189 / 274.750 / 398.908 | 20.19% / 613.981 | 218.743 / 257.670 / 377.855 | 16.95% / 518.845 |
| 3 | 215.610 / 232.044 / 289.816 | 52.48% / 1419.432 | 212.878 / 228.404 / 279.921 | 49.36% / 1324.364 |
| 4 | 218.210 / 254.566 / 453.262 | 19.35% / 578.619 | 214.870 / 247.296 / 406.292 | 16.54% / 515.888 |
| pooled | 218.522 / 259.578 / 453.262 | 34.23% / 1419.432 | 215.020 / 249.024 / 406.292 | 31.64% / 1324.364 |

Run 2 completed functional checks but is permanently rejected from every
pooled statistic and comparison.  A separately launched Megatron `restart7`
appeared after preflight; the report's post-measurement gate found worker PIDs
1097155--1097158 and set `baseline_collection_eligible=false`.  The benchmark
therefore failed closed even though the report's timing distribution looked
superficially tighter.  The raw rejected report is retained as evidence that
the co-tenant guard works.

The verified manifest under
`.cache/rail_balance/c100/formal-omp1/d05411a/` contains both accepted and
rejected reports:

```text
906a81597b3afdeb0aadd10cf4d888f563216315eb3c4e597b2a420dc43f17fb  return-h256-r1.json
521dcabfa7afedd0f37ae511c17e9fb5b81c51bb5949e48de0ba891c47758c53  return-h256-r2.json  # rejected: unexpected compute PIDs
719a8451b31baeff82f7f6af00bd7000aefcf3e5bd4204b1c9b64a4fa82fd92d  return-h256-r3.json
17e0b24bd937fc387706b013618ef6440fc2318a85b7be9dc01a68a9db31a2c5  return-h256-r4.json
```

Median repeatability alone does not close the baseline: r3 has a 1.419 ms
global maximum and a 1.324 ms rank-local maximum.  Return-H7168 must be
collected next.  Nsys will then classify the retained tails before any NCU
target or hot-path change is selected.

## OMP-controlled return-H7168 collection — commit `57b69a9`

The final profiler-free group has accepted set `{r1,r3,r4}`.  Each accepted
report uses OMP=1, 10 warmup and 100 steady samples, passes every automatic
gate, and preserves raw per-rank intervals.  Typical latency is repeatable,
but the tails are not:

| Run | Global median/p95/p99 (us) | Global CV/max | Rank-local-max median/p95/p99 (us) | Rank-local CV/max |
| --- | --- | --- | --- | --- |
| 1 | 341.347 / 370.748 / 441.746 | 18.16% / 963.464 | 337.711 / 365.133 / 440.255 | 17.61% / 937.534 |
| 3 | 333.450 / 358.622 / 420.693 | 7.32% / 523.725 | 330.693 / 356.701 / 410.068 | 6.61% / 499.080 |
| 4 | 335.065 / 358.999 / 1753.062 | 59.27% / 2021.557 | 331.282 / 351.044 / 1725.706 | 59.37% / 2008.428 |
| pooled | 337.631 / 363.655 / 528.122 | 37.59% / 2021.557 | 334.003 / 359.490 / 503.465 | 37.49% / 2008.428 |

Global and rank-local run medians span only 2.37% and 2.12%, respectively.
The large events are nevertheless rank-local and retained: r1 steady sample 5
has a 937.534 us rank-1 envelope, while r4 samples 45 and 56 have 1722.850 us
rank-5 and 2008.428 us rank-3 envelopes.  The changing rank and iteration do
not support a fixed-GPU or fixed-input conclusion.

Run 2 is rejected.  Another Codex session launched an eight-rank
`dlb_ep8_dispatch_demo.py --tokens-per-rank 2048 --experts-per-rank 32` after
preflight.  Post-measurement found PIDs 1155497/1155498/1155499/1155500/
1155501/1155504/1155506/1155508 and the report set
`baseline_collection_eligible=false`.  The demo had already exited when
identified, so no unrelated Codex process was killed.  Its 6.938 ms maximum is
not part of any accepted statistic.

The complete verified manifest is local at
`.cache/rail_balance/c100/formal-omp1/57b69a9/`:

```text
995a543a76320cb4de069fc8397def08c558ecaabca253eab144babd4a26e897  return-h7168-r1.json
428947471c62f5ef8ce7c9daa4593f04590be278b53f559e5cd53bd372ed7402  return-h7168-r2.json  # rejected: concurrent DeepEP demo
5c1239ee5cf072728c7aa9c1d4cca77c716d74eecb897658c7251d282fb87a5b  return-h7168-r3.json
ffc08d7a8f623a68814eb874a415cfac7919c6f35e4bdd9294bb80fa82367c69  return-h7168-r4.json
```

Profiler-free source and return collection is now complete for H256 and
H7168.  It establishes repeatable medians, not stable p95/p99 tails or a GPU
kernel root cause.  The next experiment is a separate Nsys run for variability
and exposed-critical-path attribution; NCU and hot-path changes remain blocked.

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

Nsys 2024.6 installs 39 stats reports.  Relevant installed report identifiers
include `cuda_gpu_kern_{sum,gb_sum}`, `cuda_gpu_trace`,
`cuda_kern_exec_{sum,trace}`, `cuda_api_{sum,trace}`,
`cuda_gpu_mem_{time,size}_sum`, `nvtx_gpu_proj_{sum,trace}`, `nvtx_kern_sum`,
`nvtx_sum`, and `osrt_sum`.  Installed analyze rules include `cuda_api_sync`,
`gpu_gaps`, `gpu_time_util`, and synchronous/asynchronous memcpy checks.  The
installed trace collectors include `cuda,nvtx,osrt` but not a separate `nccl`
value; this local fixture uses Gloo control plus CUDA-visible device work.

Thirty-three Nsys recipes are present, but the installed recipe Python
environment currently lacks the common `pandas` dependency; representative
recipe invocations all fail with `No module named 'pandas'`.  This does not
block collection, stats/analyze, SQLite export, or schema-aware custom SQL.
`nsys stats --help-reports` and `nsys analyze --help-rules` also print their
lists while returning status 1, and the correct trace-help form is
`nsys profile --help=trace`, not `--trace=help`.  These tool behaviors are
retained rather than misreported as workload failures.

### Nsys capture contract and falsified trigger topology

The profiler-free harness originally had no NVTX/capture mode and explicitly
marked NVTX disabled.  Commit `feaf03e` added one test-only, default-off
`--nvtx` flag that:

- leaves the default path disabled;
- makes any persistent report automatically ineligible as a profiler-free
  baseline;
- marks target prepare/finish/prerequisite/stage phases and exact steady
  invocation ordinals;
- lets rank 0 delimit a single `c100_nsys_window` around the steady loop so
  post-collection analysis excludes cold JIT and warmup;
- changes no CUDA/JIT/Hybrid production code.

The first diagnostic will target return-H7168 because accepted reports retain
the largest rank-local tails.  It will collect `cuda,nvtx,osrt` separately
from NCU, without GPU metrics or CPU sampling in the initial low-perturbation
pass.  The Nsys report and SQLite schema will determine whether a second pass
needs process-tree sampling/backtraces.  NCU remains forbidden until this
trace proves an exact exposed kernel invocation.

The original range-trigger command was executed and rejected.  Under Nsys's
default `--wait=all`, the launcher reparents a terminated multiprocessing
resource tracker.  It remains a zombie in the worker PGID, so the strict
watchdog correctly refuses to call cleanup complete.  Changing only capture
end from `stop` to `stop-shutdown` reproduces the failure.  Changing only the
Nsys wait policy to `primary` fixes the lifecycle, but neither the default
domain nor `@*` child NVTX trigger produces a report.  A child range is visible
after collection starts but cannot start this launched session.

The working, intentionally simple formal command captures the process tree and
cuts the report afterward by the rank-0 steady range:

```bash
mkdir -p .cache/rail_balance/c100/nsys/formal
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_DISABLE_GIN=1 OMP_NUM_THREADS=1 \
PYTHONPATH=$PWD/tests:$PWD/tests/elastic:$PWD \
/usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --capture-range=none --wait=primary \
  --force-overwrite=true --export=sqlite \
  --output=.cache/rail_balance/c100/nsys/formal/return-h7168-r1 \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
  --stage return --case-name c100_volume_h7168 \
  --warmup-iters 10 --steady-iters 100 --nvtx \
  --json-out .cache/rail_balance/c100/nsys/formal/return-h7168-r1.json \
  --master-port 30246 --timeout 180 --watchdog-seconds 1800
```

The full raw report includes cold JIT and setup, but no custom parent/child
capture controller is added merely to reduce file size.  Analysis reads the
single `c100_nsys_window` start/end from `NVTX_EVENTS` and applies that global
nanosecond interval to all eight devices.  Installed `--filter-nvtx` is not
used for multi-rank totals because it projects only the owning rank here.

The 0+1 proof report is 40,914,531 bytes, its SQLite is 175,443,968 bytes, and
the steady interval contains exactly 11 kernels plus 16 memcpy records on each
device.  Its report SHA-256 is
`0d143ed73025993fcb5f03fba5d5366dd86c947b13f3937e7c070585e2562ac9`;
the SQLite SHA-256 is
`ccd0572ca9ba259ef64b56df8b2cd99a844a415d26bf55aa7a237959f3b3be03`.
It is a collection-topology smoke, not performance evidence or an NCU target.

### Formal return-H7168 Nsys attribution

The command above completed at clean commit
`6339ee2976712a6891d0079a5ca29b2eee87df3f`.  The benchmark declared
`nvtx_diagnostic`, set `baseline_collection_eligible=false`, passed its runtime
and co-tenant gates, and left no worker or GPU process.  The three artifacts
and their verified SHA-256 identities are:

```text
7de4785a92442c202b806984a74f7cccb5fcb713cf7e3a6c20c92504ab918d49  return-h7168-r1.json       1,105,605 bytes
5907e3b2b5223cd4135e90b0dad93ecc2ebcf64691401b3dc6fb9debb0c2dc6f  return-h7168-r1.nsys-rep  60,559,410 bytes
b4e5181d724d3e781841dfde320bf55b89618088fdb923504ebbed9ad1083b97  return-h7168-r1.sqlite    262,561,792 bytes
```

They live under `.cache/rail_balance/c100/nsys/formal/` with a verified
`SHA256SUMS`.  The JSON's profiler-observed global stage median/p95/p99/max is
383.927/434.811/508.700/562.995 us.  It is diagnostic timing, not a new
profiler-free baseline.

The only outer steady range is `[43391267493,44913774211]` ns, or
1.522506718 s.  It contains 100 steady iterations on each of eight ranks,
8,800 kernels, and 12,800 memcpy records.  The exact per-rank target range has
the following invariant stream-26 sequence in all 800 cases:

```text
12,873,728-byte D2D proxy seed
58,851,328-byte D2D reduce seed
B1 local barrier
rail_balance_hybrid_return_unshuffle_impl<7168,4>
B4 local barrier
58,851,328-byte D2D reduce snapshot
4-byte D2H status
```

The first two D2D operations, B1, and the result snapshot belong to
`ElasticBuffer::rail_balance_hybrid_return_unshuffle_test` at commit
`6339ee2` (`csrc/elastic/buffer.hpp:1648`, with the committed sequence at
lines 1731--1780).  They are not submitted by the production combine commit.
The return-unshuffle submission is production-shared, and B4 models the
production unshuffle-to-epilogue visibility barrier.  The 4-byte checked
status read also exists in
`ElasticBuffer::rail_balance_hybrid_combine_commit` (`buffer.hpp:1454`,
sequence lines 1511--1550), but its host API duration is a stream-completion
observation point rather than four bytes of additive copy work.

The target-range attribution is:

| Component | p50 | p95 | p99 | max | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| rank-local NVTX stage | 342.353 us | 388.073 us | 478.908 us | 544.151 us | synchronous diagnostic adapter envelope |
| initial 12.87 MB D2D | 10.720 us | 11.488 us | - | 14.048 us | test seed copy |
| initial 58.85 MB D2D | 30.048 us | 31.008 us | - | 31.520 us | test seed copy |
| B1 | 30.608 us | 75.882 us | 149.934 us | 235.040 us | test-only arrival wait |
| return-unshuffle | 135.168 us | 145.190 us | 149.025 us | 153.472 us | production-shared NCU candidate |
| B4 | 14.688 us | 58.211 us | 62.155 us | 68.321 us | production-relevant visibility wait |
| result 58.85 MB D2D | 28.832 us | 29.792 us | - | 30.208 us | test snapshot |
| 4-byte D2H GPU copy | 2.432 us | 2.624 us | - | 3.488 us | checked status transfer |
| 4-byte D2H host API | 235.220 us | 280.463 us | 349.211 us | 440.639 us | waits for preceding stream work |
| following stream sync API | 3.000 us | 4.196 us | 4.699 us | 23.424 us | most waiting was already paid above |

B1, rather than the return kernel, explains the formal trace's long tail.
This does not retroactively identify every extreme profiler-free sample.
Across 100
global iterations, the maximum B1 duration correlates with the global NVTX
span at 0.883, while the maximum return-kernel duration correlates at only
0.138.  B1 arrival skew reaches 228.657 us, while completion skew never exceeds
0.946 us: the barrier is absorbing late ranks rather than executing slowly.
Iteration 85 is the largest example: rank 7's first GPU work arrives
227.952 us late, another rank waits 235.040 us in B1, all return kernels remain
within 90.208--145.344 us, and the global range reaches 558.044 us.  B1 and
the D2H waiting sink therefore must not be NCU targets for tail diagnosis.
One iteration also retains a post-GPU host/profiler return delay; because this
capture deliberately disabled CPU sampling and context-switch collection,
its scheduler/GIL cause remains missing evidence rather than a guessed root
cause.  The older profiler-free reports contain separate events up to about
2.02 ms.  This 0.56-ms diagnostic trace cannot attribute those historical
events, so they remain explicit missing evidence.

The return kernel is nevertheless materially exposed in the typical stage.
Across each global eight-rank target envelope, the union in which at least one
return kernel is active has a 144.726 us median.  The subset of that union in
which no copy or barrier category is simultaneously active has a 93.871 us
median.  The latter is 24.8% of the 378.525-us global NVTX-envelope median; it
is a diagnostic non-overlap measure, not a production critical-path fraction.
Device medians are not symmetric:

```text
device/rank:   0       1       2       3       4       5       6       7
p50 us:      92.480 137.392 133.584 142.113 132.625 132.577 142.816 129.088
```

Rank 6 is the last target-kernel finisher in 57/100 iterations and rank 3 in
31/100.  The first low-overhead NCU target is therefore fixed before capture:

```text
NVTX range: c100/return/stage/steady/26
kernel: rail_balance_hybrid_return_unshuffle_impl<7168,4>
device/rank: 6
Nsys start/end: 43805472052/43805614804 ns
Nsys duration: 142.752 us
context/stream: 1/26
grid/block: (256,1,1)/(32,1,1)
registers/thread: 60
dynamic shared memory: 14,400 bytes
same-name device-local whole-report ordinal: 37, zero based
```

A later fast-path comparison may use device 0 steady iteration 17, 92.480 us,
with the same launch configuration, but only after clock control or measured
clock parity removes the observed cross-device clock confounder.  Correlation
IDs are report-local and must not be used across the Nsys and NCU runs.
Cross-run identity is the NVTX range, full demangled kernel, device/rank,
launch configuration, workload shape, and invocation ordinal.  NCU may
investigate typical kernel efficiency; it is not being used to explain the
already-attributed formal-trace tail.

### Frozen first-pass NCU safety contract

Default kernel replay is rejected for this target.  The kernel TMA-loads a
local proxy-return record and TMA-stores through an LSA peer pointer into
another process's reduce buffer.  Replaying one rank's launch would repeat a
cross-process side effect while the other seven ranks wait in B4, and this
project has no evidence that NCU checkpoints/restores every peer allocation.
Range replay would additionally capture cross-process barriers without a
provable global restore boundary.

The first pass therefore uses application replay.  Every metric pass relaunches
the complete eight-rank benchmark and reconstructs the process group, arena,
one-shot ticket, and barrier epoch.  Matching is `strict` so a changed kernel
sequence fails instead of being silently dropped; `grid` matches name plus
grid/block without requiring cross-process context and stream IDs to remain
stable.  Only device 6, the exact steady-26 NVTX range, one demangled return
kernel, and the installed `basic` set are eligible.  `--kill 0` is mandatory:
killing rank 6 after collection would strand its peers in B4 or a WORLD gate.

The pre-registered command was tested rather than silently corrected.  NCU
2025.1.1 rejected its `--output` spelling before the target started, and the
first `--export` retry completed functionality but matched no kernel because a
push/pop range needs escaped slashes and a trailing `/`.  Neither failure
authorized a replay or matching fallback.  The accepted command core is:

```bash
/usr/local/cuda/bin/ncu \
  --config-file off --target-processes all --devices 6 \
  --replay-mode application \
  --app-replay-mode strict --app-replay-match grid \
  --nvtx --nvtx-include 'c100\/return\/stage\/steady\/26/' \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*rail_balance_hybrid_return_unshuffle_impl.*' \
  --launch-count 1 --kill 0 --set basic \
  --cache-control none --clock-control none \
  --force-overwrite \
  --log-file .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-basic-r2.log \
  --export .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-basic-r2 \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
  --stage return --case-name c100_volume_h7168 \
  --warmup-iters 10 --steady-iters 27 --nvtx \
  --json-out .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-basic-r2.json \
  --master-port 30248 --timeout 180 --watchdog-seconds 1800
```

The surrounding environment remains the formal Nsys environment.  Cache and
clock control are explicitly `none` to avoid perturbing only rank 6 while its
peers wait; achieved clocks must be read from the report, and the NCU duration
must not be compared with Nsys duration.  A failure is retained as evidence and
does not authorize fallback to kernel replay, relaxed matching, `full`, or
killing the target.  The report is
accepted only if the process exits cleanly, the benchmark passes every replay,
no GPU/process remains, and exactly one device-6 grid-256/block-32 target with
14,400-byte dynamic shared memory is present.  Replay pass count, profiler
warnings, clock/cache state, child injection, metric permission/conflict,
barrier timeout, process cleanup, artifact hash, commit, dirty state, and
co-tenant state are mandatory audit fields.

### Accepted first-pass NCU evidence

Strict multiprocessing application replay succeeds without weakening the
contract.  The accepted run exits zero after ten application-replay passes.
The observed outer console emitted PASS after each replay, no strict-match,
timeout, functional, or cleanup error appeared, and no worker or GPU allocation
remained.  The console stream was not separately redirected; the NCU pass log,
final-pass JSON and report are the persistent raw artifacts.  The report
contains exactly one process, one launch on device 6, and
one `rail_balance_hybrid_return_unshuffle_impl<7168,4>` invocation in the exact
steady-26 push/pop range.  Its identity is stream 26, grid 256, block 32,
60 registers/thread, and 14,400 bytes of dynamic shared memory, matching the
pre-registered Nsys candidate.

The retained artifacts are:

```text
db2f66dfc8edb7a5f9201c5e877976ac3908063e9b314ec134bd697ea69d04b4  return-h7168-rank6-basic-r2.json
e8c6c59d6ae40b84ce8a86a15c5eb87645c2499f2dee065507125de861d33388  return-h7168-rank6-basic-r2.log
878527125f66b9f30cc538f88df674a0fc4a8570148bbf4f121d99247a2e6245  return-h7168-rank6-basic-r2.ncu-rep
```

The JSON records clean commit `4fe7223`, no co-tenant and diagnostic-only
eligibility.  The log preserves the required warnings: caches were uncontrolled
and clocks unmodified.  The report measures 1.50 GHz SM and 3.20 GHz DRAM clocks
for the collected pass.  Its 735.200-us duration is replay-instrumented and is
not compared with the 142.752-us Nsys duration or profiler-free latency.

The installed `basic` set proves launch geometry and broad utilization only:

```text
waves/SM                          0.13
theoretical / achieved occupancy 23.44% / 1.61%
achieved active warps/SM          1.03
SM / DRAM throughput              0.11% / 0.41%
L1/TEX / L2 throughput            1.39% / 0.95%
SM active cycles min/avg/max      9,503 / 270,616.69 / 1,107,335
SMSP active cycles min/avg/max    0 / 71,821.86 / 1,142,220
```

This is not a shared-memory occupancy diagnosis: 256 one-warp blocks across
132 SMs expose only 1.94 blocks/SM even though shared memory permits 15.  A
deterministic CPU-plan audit gives the missing work mapping.  For every egress,
224 target channels have zero moved copies and only 32 target channels each
own 28 copies (four copies for each of seven remote destinations).  The earlier
O059 statement that 224 channels move four records describes source-producer
channels, not the target-channel layout consumed by return-unshuffle.  GPU
prefix materialization uses the same greedy low-channel fill at
`rail_balance_hybrid_plan.cuh:489-510`.  That mapping is consistent with the
large SM/SMSP active-cycle dispersion, but `basic` still cannot distinguish
eligible-warp starvation, the serial TMA wait chain, or peer-link behavior.

The next single-purpose capture therefore adds only `SchedulerStats`,
`WarpStateStats`, and `Nvlink` under the same strict application replay.
`SourceCounters` is deferred because the accepted cubin was compiled without
`EP_JIT_WITH_LINEINFO=1`; a source-correlated rerun must first prove a lineinfo
cubin keeps identical SASS and resources.  Detailed memory sections remain a
conditional experiment.  No hot-path edit is authorized by this first report.

The rejected no-match retry is retained as JSON/log under hashes
`228d8f3b...f1e7` and `00193eac...8d9`; it has no `.ncu-rep`.  The earlier
invalid-`--output` attempt failed before target launch and produced no raw file,
which remains an explicit missing artifact rather than reconstructed evidence.

The exact installed-schema SQL used to prove the multi-rank window is retained
here; formal analysis must first require exactly one outer range:

```sql
SELECT start, end
FROM NVTX_EVENTS
WHERE text = 'c100_nsys_window';

SELECT deviceId, COUNT(*) AS kernels, SUM(end - start) AS kernel_ns,
       MIN(start) AS first_start, MAX(end) AS last_end
FROM CUPTI_ACTIVITY_KIND_KERNEL
WHERE end > :window_start AND start < :window_end
GROUP BY deviceId
ORDER BY deviceId;

SELECT deviceId, COUNT(*) AS memcpys, SUM(end - start) AS memcpy_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE end > :window_start AND start < :window_end
GROUP BY deviceId
ORDER BY deviceId;
```

The first failing outer command is retained in D077 together with its JSON
hash.  Its console showed adapter PASS, then watchdog cleanup failure and Nsys
exit 1; no `.nsys-rep`, SQLite or qdstrm existed.  Console output and the
external process-observer stream were not redirected to standalone files, so
that is an explicit missing raw artifact rather than reconstructed evidence.
The observer did record resource tracker PID 1291724 transitioning to `Z`,
remaining in PGID 1291618 and being reparented to profiler-side PID 1291474;
that parent's executable mapping was not separately persisted.

The default target-stage call and its timer boundary are unchanged.  Moving
phase callables into locals introduces a few disabled Python branches in the
wider transaction envelope, so this is described as minimal host overhead,
not instruction-for-instruction zero overhead.  Profiler-free truth remains
the already frozen pre-instrumentation group and later same-commit A/B tests.

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
2. Accepted profiler-free source and return samples for H256/H7168.  This is
   complete for repeatable typical latency.  p95/p99/max tail stability is
   deliberately not claimed; the retained tail events are inputs to Nsys,
   rather than a circular prerequisite that would forbid profiling them.
3. Nsys exposed critical-path attribution for the new large fixture.  This is
   complete for return-H7168 and selects the exact invocation above.
4. An exact NCU invocation selected from that Nsys report.  The strict `basic`
   capture is complete; scheduler/warp/link attribution is still pending.
5. Sustained same-machine peer-copy/HBM reference if a bandwidth percentage is
   later needed; published peak alone is insufficient.
6. Real D>1 Gin/RDMA/QP/NIC behavior and network-visible counters.
7. Full-MoE or training-level target metric and compute/communication overlap.

Items 2--4 now exist, but the first NCU set only establishes low broad
utilization plus deterministic target-channel packing; it cannot yet name the
dominant wait state or link behavior.  TMA waits, segment scans, QP choice,
tail publication cadence, planner launch, and barriers remain hypotheses.  No
performance-path code change is authorized until the bounded scheduler/warp/
link capture closes that gap.  Unstable tails continue to limit confidence and
must be represented in any later no-profiler validation; they do not prevent
Nsys from determining whether a tail is host-, API-, synchronization-, or
kernel-owned.

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
