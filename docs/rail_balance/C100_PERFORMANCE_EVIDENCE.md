# C100 local performance evidence

This document is the audit surface for controlled local rail-balance
measurement.  Profiler-free samples are the performance truth; Nsys and NCU
are separate diagnostic experiments.  Raw reports stay under the ignored
`.cache/rail_balance/c100/` directory and are identified by checksum.

## Current state

```text
environment manifest: FROZEN
large source fixture: FUNCTIONAL_PASS, POST_O077_NSYS_AND_NCU_ATTRIBUTED
large return fixture: FUNCTIONAL_PASS, PRE_AND_POST_NSYS_ATTRIBUTED
checked-adapter benchmark harness: AUDITED_PASS
profiler-free baseline: SOURCE_AND_RETURN_COLLECTED, TAIL_STABILITY_NOT_ACCEPTED
O077 profiler-free A/B: TWENTY_FOUR_RUNS_GATED_AND_RETAINED, MIXED_RESULT
O078 profiler-free A/B: IDENTITY_MATCHED_TWELVE_RUN_B_SIDE_PASS
new Nsys attribution: O078_SOURCE_SCAN_REMOVAL_CONFIRMED
new NCU dossier: O078_DEVICE6_DYNAMIC_INSTRUCTION_PREDICTION_PASS
performance-path change: O077_CONDITIONAL_PARENT; O078_LOCAL_CHECKED_ADAPTER_ACCEPTED
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

### Accepted directed scheduler, warp and NVLink evidence

The pre-registered follow-up kept the same whole-application replay, exact
device/range/kernel identity, cache/clock policy and complete eight-rank
lifecycle.  It changed only the requested counter sections.  The command core
and its explicit environment were:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_DISABLE_GIN=1 OMP_NUM_THREADS=1 \
PYTHONPATH=$PWD/tests:$PWD/tests/elastic:$PWD \
/usr/local/cuda/bin/ncu \
  --config-file off --target-processes all --devices 6 \
  --replay-mode application --app-replay-mode strict \
  --app-replay-match grid \
  --nvtx --nvtx-include 'c100\/return\/stage\/steady\/26/' \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*rail_balance_hybrid_return_unshuffle_impl.*' \
  --launch-count 1 --kill 0 \
  --section SchedulerStats --section WarpStateStats --section Nvlink \
  --cache-control none --clock-control none --force-overwrite \
  --log-file .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-sched-warp-nvlink.log \
  --export .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-sched-warp-nvlink \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/bench_rail_balance_hybrid_lsa.py \
  --stage return --case-name c100_volume_h7168 \
  --warmup-iters 10 --steady-iters 27 --nvtx \
  --json-out .cache/rail_balance/c100/ncu/formal/return-h7168-rank6-sched-warp-nvlink.json \
  --master-port 30249 --timeout 180 --watchdog-seconds 1800
```

Ten strict application-replay passes completed and each observed outer run
passed.  As with the first capture, that outer console stream was observed but
not independently persisted.  The command exited zero, left no worker or GPU
allocation, and the report again contains exactly one device-6, stream-26,
grid-256/block-32 `return_unshuffle<7168,4>` launch in steady range 26.  The
final-pass JSON is clean commit `e4bd800`, declares no co-tenant, and records
the unchanged G8/D9/K4/H7168/N1024/C256 workload with 896 proxy records per
egress.  Exit and cleanup were session observations rather than a separate
console artifact; the log/report persist replay and target state.  Only the
expected uncontrolled-cache and unmodified-clock warnings appear.  The raw
artifacts are:

```text
3b00510e7d8013a820af498f257d3b3f022e537d298bc52d6e0db605b486ebda  return-h7168-rank6-sched-warp-nvlink.json
7b2848a3075db729211eba058daf3c8d5de17439dbf2c0b85ac67cb490281974  return-h7168-rank6-sched-warp-nvlink.log
e67706da8aa1ccdd29aeb46eff720468c7fc3ee873ea00435c44b5f4a468ffb3  return-h7168-rank6-sched-warp-nvlink.ncu-rep
```

The scheduler is starved rather than throughput-saturated:

```text
active warps / scheduler                         1.0145
eligible warps / scheduler                       0.0176
issued warps / scheduler                         0.02
cycles with one or more eligible warps           1.7611%
cycles with no eligible warp                    98.2389%
warp cycles per issued instruction              57.6083
long-scoreboard stall cycles / issued inst      49.7335
sleeping / wait stall cycles / issued inst       4.9348 / 2.7344
barrier / membar / LG / MIO / TEX throttle       0 / 0 / 0 / 0 / 0
```

Long scoreboard accounts for 86.3% of the reported cycles per issued
instruction.  The source's elected-lane payload loop contains metadata/global
loads, a TMA load and completion wait, an LSA peer TMA store, and a store wait
at `rail_balance_hybrid_unshuffle.cuh:160-253`.  Without PC sampling or source
counters this report cannot identify which LDG/TMA dependency supplies the
scoreboard cycles.  It does reject barrier, membar and execution-pipe throttle
states as the primary explanation for this capture.  Combined with the
deterministic 32-active/224-empty target-channel map, it establishes the
actionable hypothesis: the current launch exposes too little independent
channel work to hide the long-latency memory dependencies in the per-record
payload loop.  O076 must still falsify or confirm the causal benefit.

The link counters provide an independent byte and saturation check.  Device 6
transmitted exactly 12,873,728 user bytes, equal to 896 records times the
14,368-byte TokenLayout stride.  Total transmit traffic was 18,503,952 bytes,
including 5,630,224 protocol bytes, at 26.253 GB/s and 5.73% of the reported
peak.  Receive traffic was 13,334,432 protocol bytes and zero user bytes, which
is NCU's protocol/user classification rather than an endpoint attribution.  The
link counters are device/kernel-window counters and may include peer protocol
traffic; they prove neither a production Gin rate nor a complete fabric
attribution.  They do prove that this local return path is not close to
saturating aggregate NVLink transmit bandwidth while its logical payload byte
count is exact.

NCU reports 761.504 us for this counter pass.  It is replay-instrumented and is
not compared with the prior basic pass, Nsys, or profiler-free latency.
`MemoryWorkloadAnalysis` is not required before the first falsification: the
single variable will change independent channel work while leaving every TMA
instruction, TokenLayout byte, proxy count and link endpoint unchanged.  If
that experiment fails, memory-path attribution becomes the next bounded
section rather than an assumed explanation.

### O077 immediate pre-change profiler-free baseline

Before changing the oracle or GPU materializer, an immediate external preflight
observed no compute application, then the exact frozen benchmark collected
three fresh-JIT eight-rank runs for each source/return and H256/H7168 combination at
clean commit `fc39bf7`.  Persistent pre/post snapshots see only the eight
benchmark workers and no unexpected co-tenant; they cannot exclude a transient
mid-run process, clock, power or thermal event.  Every run uses OMP=1,
10 warmups, 100 steady samples, no NVTX/profiler, and passes the automatic
environment, identity, persistence and co-tenant gates.  These are eligibility-
gated retained A-side reports, not acceptance of tail stability; they do not
replace the earlier historical baseline or erase its tails.

| Stage/shape | run | global median/p95/p99/max (us) | global mean/std/CV | rank-local-max median/p95/p99/max (us) | local mean/std/CV |
| --- | ---: | --- | --- | --- | --- |
| source H256 | 1 | 126.149 / 151.242 / 181.796 / 198.424 | 129.076 / 12.012 / 9.31% | 74.312 / 85.640 / 92.066 / 118.707 | 75.766 / 6.612 / 8.73% |
| source H256 | 2 | 118.382 / 136.386 / 232.119 / 715.141 | 125.999 / 60.891 / 48.33% | 78.809 / 86.830 / 126.632 / 144.039 | 81.237 / 9.142 / 11.25% |
| source H256 | 3 | 113.700 / 134.138 / 262.597 / 713.222 | 122.373 / 61.701 / 50.42% | 77.013 / 88.445 / 187.372 / 673.444 | 85.947 / 60.527 / 70.42% |
| return H256 | 1 | 203.746 / 238.398 / 258.592 / 269.993 | 207.149 / 13.003 / 6.28% | 201.459 / 226.803 / 243.291 / 255.247 | 203.744 / 10.814 / 5.31% |
| return H256 | 2 | 220.534 / 245.205 / 308.324 / 2518.285 | 245.967 / 228.682 / 92.97% | 216.732 / 232.497 / 292.529 / 2463.154 | 239.961 / 223.659 / 93.21% |
| return H256 | 3 | 207.612 / 227.860 / 237.530 / 257.629 | 210.696 / 9.171 / 4.35% | 203.646 / 217.893 / 225.697 / 242.383 | 205.919 / 6.906 / 3.35% |
| source H7168 | 1 | 128.755 / 142.353 / 172.559 / 191.031 | 130.722 / 10.078 / 7.71% | 96.235 / 110.115 / 147.209 / 160.985 | 99.539 / 9.943 / 9.99% |
| source H7168 | 2 | 133.056 / 142.584 / 169.405 / 267.274 | 135.038 / 14.219 / 10.53% | 97.348 / 103.814 / 108.950 / 258.418 | 99.652 / 16.180 / 16.24% |
| source H7168 | 3 | 132.721 / 141.006 / 158.077 / 169.121 | 133.885 / 5.880 / 4.39% | 96.013 / 110.556 / 112.286 / 127.897 | 97.494 / 5.256 / 5.39% |
| return H7168 | 1 | 332.363 / 357.907 / 485.553 / 514.222 | 337.778 / 26.300 / 7.79% | 328.613 / 347.615 / 459.725 / 473.518 | 332.734 / 21.917 / 6.59% |
| return H7168 | 2 | 333.360 / 363.412 / 599.404 / 1580.357 | 353.009 / 128.855 / 36.50% | 329.947 / 359.989 / 599.310 / 1570.964 | 348.493 / 127.277 / 36.52% |
| return H7168 | 3 | 333.805 / 361.237 / 658.128 / 2395.960 | 359.702 / 207.086 / 57.57% | 331.183 / 360.495 / 657.556 / 2387.726 | 356.744 / 206.580 / 57.91% |

H7168 return is the primary A-side anchor: its run medians span only
1.442 us.  H256 and all p99/max columns retain intermittent host/rank-local
tails, so O077 must compare all three distributions rather than a best run.
The retained files are:

```text
f45f6d1cd6d79578f26aeba055f877224b12a83b22caeafd22d363e8820e8db0  source-h256-r1.json
424b46c43791050cf6d1a6f36850ce91f9a36af486becec661fe2eddb293a645  source-h256-r2.json
ac083a153bf2c87e11b65c21f6c8aa1ef7fb1b659bb7399b370ff338b0046529  source-h256-r3.json
3ff1a21fae00aa94a4ac750325010358ce4825c4ab075f42fff396ea954b82ee  return-h256-r1.json
52adb4166ff586f3585a7b54c3c93c85de945a0e15be8fce4b50e51ae870f16f  return-h256-r2.json
871d4622ec42b09c2866aa836a68f019c2a491418ea18abfc844e8043ce1ca4f  return-h256-r3.json
76e1c3b4d623f988db9254155c949b92b1b99d3b8e354f2e1c23f7b4811a97a3  source-h7168-r1.json
f18f248a3c4090fee5e42b0ddd0a18de37de23b37203479e0d22daa12ce122e8  source-h7168-r2.json
6357bcba42b306f25b5347c337b3a16209bf0b63f5421b94a80b36c5e09d26d8  source-h7168-r3.json
21dfec17a8d143e1acb898f4e2afa42c39ffe54709a04080234d880f65d5193f  return-h7168-r1.json
8e1e62be0e33cc35dd21cb015b3b1225ad7578d70fd1e214b24ec14b37aef8a4  return-h7168-r2.json
38cbcc57510f95eae00fca253d2e760e93b6c0fa7114dc0d78bb98c1d00e62ca  return-h7168-r3.json
```

All files live under
`.cache/rail_balance/c100/o077/pre-fc39bf7/`.  They are ignored raw artifacts;
their checksums and statistics above are the tracked audit index.

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
   and directed scheduler/warp/NVLink captures are complete.  They identify
   target-channel packing plus unhidden long-scoreboard latency without a
   saturated NVLink payload path.
5. Sustained same-machine peer-copy/HBM reference if a bandwidth percentage is
   later needed; published peak alone is insufficient.
6. Real D>1 Gin/RDMA/QP/NIC behavior and network-visible counters.
7. Full-MoE or training-level target metric and compute/communication overlap.

Items 2--4 now exist.  The directed NCU capture closes the pre-edit gate:
eligible-warp starvation and long scoreboard dominate active execution, the
logical peer-store bytes are exact, and aggregate NVLink transmit utilization
is low.  This authorizes one target-channel-distribution experiment only.  It
does not authorize changing the TMA chain, proxy layout, barriers, QP choice,
tail publication, public API or production defaults.  Unstable tails continue
to limit confidence and must be represented in the no-profiler A/B validation.

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

## O077 profiler-free A/B result (2026-07-22)

The immediate A side is the clean `fc39bf7` tree under
`.cache/rail_balance/c100/o077/pre-fc39bf7/`; the B side is the clean final
`f895eff` tree under `.cache/rail_balance/c100/o077/post-f895eff/`.  Each side
contains three independent direct runs for source/return at H256/H7168, with
10 warmups and 100 retained steady samples per report.  All identity,
persistence, idle-GPU, co-tenant and MPS gates pass.  An independent audit
recomputed all 2,400 samples exactly.

| Checked adapter | A median of run medians (us) | B (us) | B/A latency |
| --- | ---: | ---: | ---: |
| source H256 | 118.382 | 181.365 | +53.20% |
| source H7168 | 132.721 | 188.542 | +42.06% |
| return H256 | 207.612 | 157.288 | -24.24% |
| return H7168 | 333.360 | 257.583 | -22.73% |

The separately recorded coarse `finish` envelope includes the node-local plan
barrier, plan/prefix work, status D2H, stream synchronization and Python status
materialization:

| Transaction label | A finish (us) | B finish (us) | B/A latency |
| --- | ---: | ---: | ---: |
| source H256 | 368.428 | 367.907 | -0.14% |
| source H7168 | 300.568 | 364.664 | +21.32% |
| return H256 | 324.961 | 337.255 | +3.78% |
| return H7168 | 365.680 | 387.053 | +5.84% |

This is not an additive plan-kernel time and its tails are unstable.  Nsys
later isolates the changed post-source prefix kernel at only about +1.9 us.
The coarse table is retained to satisfy O077's pre-registered plan-cost gate;
it prevents the favorable H7168 adapter-only sum from being presented as an
overall transaction or production win.

For each of the four checked-adapter rows, all three same-label comparisons
agree in direction; pooled medians and ten-percent trimmed means agree as
well.  The coarse finish same-label directions are mixed, consistent with its
classification as a noisy synchronization envelope.  The source and return
fixtures are independent checked adapters.  They cannot be added into a
production Hybrid EP critical path, and this result is not an end-to-end
training or MoE result.  This is a typical-median tradeoff only: p95/p99/max
tails remain unstable, the B-side return-H7168 maximum reaches 2.468 ms, and
pre/post GPU snapshots cannot exclude a transient that begins and ends during
a run.

All twelve B JSON SHA256 values are:

```text
6c121b2b24ee8a252a9bc33387fec5b0deaa8ae167190143ec2b68f16e5a4efa  source-h256-r1.json
fd98db6f830401f85f938d0ba256227cc27bdebe816595e51d800993ad7a9c1a  source-h256-r2.json
724e60fa1eba7179a72fb4d75c8d9f333c4c9273f139aae53f22107251eabf8b  source-h256-r3.json
1be9aeeb46367abc74ad965b2cb9f7add0656d0e5188cf9dbc2b2948c719208f  source-h7168-r1.json
855f91aa5272c2df26c96f4dff6899bcdd145b9cd1d0fa2dcf5deddff2fb9711  source-h7168-r2.json
e5862e3962d532f08157bf6e6a234b24be39fe326ebedfa2b7ae878ac161a64e  source-h7168-r3.json
3fa37bcf2365a7c43427adff5a967f78e23c149d3103c14c55323e87e9f142ee  return-h256-r1.json
061cd10d552ed96f801b124262ee3de02ff9a9b346e774c463db69cd52ce3144  return-h256-r2.json
11bdb59d0c2c74eeb92f2aeb376da3128bafadf9222c736ccd4ba8085074f39d  return-h256-r3.json
4bd8bacabe0e0ceda4e8c68a5345e1a13f04b2a7aa30a383561b8d5fafb8ad11  return-h7168-r1.json
7d37dc7da5c7294f22fcde153ee4b64f370232169ff8f01274ad63371e4edd90  return-h7168-r2.json
a6d6065b2c85e863db642a102a2c059cb1b877a4bf4ae8c52b94af3b3b5cbe2c  return-h7168-r3.json
```

Tracked B-side distribution index, in microseconds except CV:

| Case | median | p95 | p99 | max | mean | std | CV % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source H256 r1 | 177.178 | 194.082 | 207.881 | 298.937 | 179.797 | 14.114 | 7.850 |
| source H256 r2 | 187.892 | 207.294 | 222.793 | 264.494 | 189.986 | 12.022 | 6.328 |
| source H256 r3 | 181.365 | 209.417 | 309.836 | 907.981 | 191.958 | 73.656 | 38.371 |
| source H7168 r1 | 182.757 | 192.928 | 202.955 | 221.828 | 182.811 | 7.281 | 3.983 |
| source H7168 r2 | 188.542 | 204.471 | 277.911 | 377.126 | 191.868 | 21.940 | 11.435 |
| source H7168 r3 | 189.762 | 200.715 | 210.313 | 214.037 | 190.133 | 6.662 | 3.504 |
| return H256 r1 | 157.288 | 182.056 | 325.195 | 829.984 | 167.298 | 71.189 | 42.552 |
| return H256 r2 | 157.845 | 175.673 | 237.026 | 347.665 | 162.346 | 22.735 | 14.004 |
| return H256 r3 | 153.224 | 177.574 | 370.726 | 525.358 | 160.543 | 44.107 | 27.474 |
| return H7168 r1 | 267.719 | 308.695 | 499.650 | 787.065 | 278.488 | 59.510 | 21.369 |
| return H7168 r2 | 257.583 | 288.371 | 353.086 | 700.835 | 264.847 | 46.043 | 17.385 |
| return H7168 r3 | 250.294 | 294.897 | 378.707 | 2468.341 | 278.089 | 220.876 | 79.427 |

The corresponding rank-local maximum adapter distributions are:

| Case | median | p95 | p99 | max | mean | std | CV % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source H256 r1 | 134.670 | 148.497 | 153.266 | 240.974 | 135.587 | 11.707 | 8.634 |
| source H256 r2 | 131.669 | 141.518 | 151.990 | 154.356 | 132.832 | 5.257 | 3.958 |
| source H256 r3 | 132.484 | 142.705 | 211.705 | 850.449 | 140.727 | 71.888 | 51.083 |
| source H7168 r1 | 144.538 | 149.257 | 158.798 | 166.093 | 143.428 | 5.024 | 3.503 |
| source H7168 r2 | 145.312 | 155.286 | 194.782 | 214.701 | 146.521 | 9.795 | 6.685 |
| source H7168 r3 | 140.207 | 147.380 | 153.184 | 160.433 | 141.202 | 3.462 | 2.452 |
| return H256 r1 | 150.418 | 178.018 | 276.022 | 707.096 | 159.736 | 58.371 | 36.542 |
| return H256 r2 | 154.195 | 171.597 | 231.863 | 324.361 | 158.563 | 20.533 | 12.949 |
| return H256 r3 | 148.536 | 172.818 | 357.171 | 453.948 | 154.947 | 37.789 | 24.389 |
| return H7168 r1 | 261.310 | 292.360 | 463.087 | 764.032 | 270.633 | 56.341 | 20.818 |
| return H7168 r2 | 251.339 | 278.803 | 339.854 | 686.271 | 258.165 | 44.919 | 17.399 |
| return H7168 r3 | 247.064 | 278.723 | 371.278 | 2438.569 | 272.882 | 218.286 | 79.993 |

## O077 post-change Nsys attribution

Nsys 2024.6.2 collected separate full-process H7168 post-O077 diagnostics with
`cuda,nvtx,osrt`, no CPU sampling/context-switch sampling, `wait=primary`,
10 warmups and 100 steady iterations.  Diagnostic timing is not relabelled as
the profiler-free baseline.  Each SQLite contains one outer window, 800 exact
target ranges and the expected eight-device launch identity.

The table's comparator is not the immediate `fc39bf7` A-side.  It is the
earlier clean `6339ee2` formal return-H7168 Nsys report, including that report's
identity-matched prerequisite-source range.  This is a cross-run diagnostic
distribution comparison with the same kernel identity, shape and call order,
not a same-session or correlation-ID A/B.

| Diagnostic interval/kernel | formal comparator p50 (us) | O077 post p50 (us) | Interpretation |
| --- | ---: | ---: | --- |
| device-6 source kernel | 53.952 | 105.376 | unchanged SASS/launch; stable scan-work regression |
| per-ordinal maximum source kernel | 55.520 | 105.376 | source critical rank becomes device 6 |
| pooled return kernel | 135.168 | 59.232 | target-channel spreading succeeds |
| eight-device return union | 144.726 | 64.450 | interval union, not summed durations |
| return union excluding simultaneous copy/barrier | 93.871 | 56.470 | diagnostic non-overlap union decreases 39.84% |
| return completion skew | 52.000 | 9.435 | straggler spread contracts |
| prefix kernel | 86.912 | 88.848 | approximately 1.9 us; not source root cause |

The four return-fixture memcpy distributions show no material shift.  Post
source device 6 is longest in 100/100 iterations.  B1 arrival remains the diagnostic tail
source, so the Nsys envelope and NCU result are not used to explain host/rank
arrival outliers.  The verified SQLite joins and interval-union query are
persisted in `sql/o077_nsys_queries.sql` by this evidence checkpoint.
The 140.806-us source and 93.871/56.470-us return values mean only “target
kernel active while no fixture copy/barrier is active” inside these exact
diagnostic ranges.  They are not a production exposed critical-path fraction
and cannot be used as Amdahl's `f`.

Artifact identity:

```text
85f64b34059b1df648de4cc5b31cfae146f68e7a12b2ad45ae06d0a1d395d7dd  source-h7168.json
570dce21d74a7e3a037bb5c852505298839ce150dd169eea2cf88b8f3355b45d  source-h7168.nsys-rep
82bffcad8e4c78ed546b55da15a7155247b60455113b9384a2822ac60f958268  source-h7168.sqlite
9b3247e3ab53e68d708389d21116595a8540781fc70167454875471f9d76e143  return-h7168.json
cc68ff0e4635a722fb22a1d4e911e3541de629ad36e7fe953247cdaa2f804879  return-h7168.nsys-rep
10d0501796e6ae6b5f480b8f47cb2ff86798610fb66cb7ec22f93e036719d4b5  return-h7168.sqlite
76796122cdacb4a790f82f1995620094da6e4f660f1a793c638b3d84e1889b78  source-h7168.console.log
8501e55144288578bdc475ad1adfa40b4a258036b21bae8f6af81b53a3112f9b  return-h7168.console.log
```

The files are under
`.cache/rail_balance/c100/o077/nsys-post-f895eff/`.  The formal pre-return
SQLite SHA256 is
`b4e5181d724d3e781841dfde320bf55b89618088fdb923504ebbed9ad1083b97`.

## O077 source-resolver NCU dossier

```text
Kernel:
  rail_balance_hybrid_source_shuffle_impl<7168,4>
Invocation identity:
  f895eff; BF16 G8/D9/N1024/C256/H7168/K4; 7,168 global moved copies;
  dispatch TokenLayout 14,400 B; 896 records/rank; context 1; stream 26;
  c100/source/stage/steady/26; same-name device-local ordinal 37 zero-based;
  device 6 exposed target and device 0 natural low-scan control; grid 256;
  block 32; REG74; dynamic shared memory 14,432 B; identical code, shape,
  copy count and payload bytes; prefix contents intentionally differ by rank
Nsight Systems exposed time:
  device-6 p50 105.376 us; post source union p50 154.975 us and
  140.806 us after overlapping memcpy removal
Launch configuration:
  256 one-warp CTAs, 0.13 waves/SM, theoretical occupancy 23.44%
Primary limiter:
  data-dependent linear scan of the monotonic moved-channel prefix
Secondary limiter:
  unresolved by the basic set; TMA/global-memory latency is a hypothesis, not
  an established limiter; neither aggregate compute nor DRAM is saturated
Supporting NCU metrics:
  device 0/device 6 dynamic instructions = 906,250/2,971,264;
  LSU pipe = 0.092655%/0.505456% of peak; achieved occupancy =
  2.66%/2.65%; active warps/SM = 1.70/1.70; SM throughput =
  0.29%/1.02%; DRAM throughput = 0.84%/0.71%
Contradicting evidence:
  device 0 is fast with identical code and launch; replay SM clocks differ
  (1.98 versus 1.50 GHz), so NCU durations are not compared; no full/source-PC
  report identifies an exact instruction and none is needed for this variable
Relevant source/PTX/SASS:
  resolve_hybrid_copy in rail_balance_hybrid_plan.cuh; source kernel SASS is
  bit-identical before/after O077; O077 moves average target channel from
  about 15.5 to 111.5 and adds an estimated 688,128 scan steps
Optimization hypothesis:
  bounded upper-bound lookup over the producer-generated monotonic prefix for
  the retained canonical plan; this
  is a new 7,168-copy experiment, not a relabelling of the rejected three-copy
  fixed-overhead binary-resolver report
Expected kernel speedup:
  device-6 source p50 should move materially from about 105 us toward the
  pre-O077 approximately 54-us value without losing the return result
Expected end-to-end speedup ceiling:
  missing evidence: no full MoE/training critical-path fraction exists locally
Confidence:
  high for source-fixture root cause; unknown for production multi-node speedup
Next falsification experiment:
  O078 changes only the resolver lookup, then repeats exact correctness,
  profiler-free source/return A/B and the same source Nsys window
```

NCU 2025.1.1 used `--set basic`, `--target-processes all`, exact device
0-or-6/NVTX steady-26/demangled-kernel filters, `--launch-count 1`, `--kill 0`
and strict whole-application replay for ten passes because the kernel writes
peer proxy memory.  Cache and clock controls were `none`.  Kernel/range replay
and NCU duration comparisons remain forbidden.  Reports:

```text
6ea81dd88f9823b93f071aff46e3bc1e30513cdbc8ffde743b8b667c4de79820  source-h7168-rank0-basic.json
d4c43783c800c85ec37db3d1b1608cae0cd4831daca964991bf01514d1286ba1  source-h7168-rank0-basic.log
83baa42dea96d3e6241565e53f0e5683454ad42cf315f38984c3e024379a8247  source-h7168-rank0-basic.ncu-rep
f3a54903c2c9f50244658d43d235d0d711b8cc449386e3f81afcc29249010ebf  source-h7168-rank0-basic.console.log
d0c4361325f1a27630056e39f6ea6bcd8223d0e0dfae5954c8247285842ee1e7  source-h7168-rank6-basic.json
6617a2c7cc11e3d6fd932fe61139277a17880121c1347c8103a57fec18764d59  source-h7168-rank6-basic.log
dd835e5518cd160304d6a3d8883e9002556feffcfed14cd86df701d380aa0c27  source-h7168-rank6-basic.ncu-rep
edf521c5265b673c7312a6f08beef63de0c41a57d2cdbc7b9980596614d82217  source-h7168-rank6-basic.console.log
```

The benchmark argv is embedded in each JSON.  There is no separate standalone
outer-command artifact, but the exact NCU command is retained in each report's
session metadata and the exact Nsys command/config/cwd is embedded in each raw
report.  O078 and later captures also save the outer command as a visible text
artifact to make this evidence easier to audit without report import tools.

The O077 decision is therefore conditional: keep its auditable checkpoint and
its return-channel result, but do not accept the current combined hot path.
This evidence authorized only O078's single-variable resolver experiment; its
completed result follows.  Real Gin/RDMA, NIC/QP behavior, full-model overlap
and production end-to-end speedup remain missing evidence.

## O078 bounded-resolver result (2026-07-22)

### Identity gate and rejected first collection

The first 12-report O078 collection passed its internal baseline gates and all
numerical thresholds but is not the formal B side.  It used timeout 300 rather
than O077's 180, changing the local-barrier JIT specialization and two SASS
timeout immediates.  The reports remain under
`.cache/rail_balance/c100/o078/post-32b9cfd/` as a failed measurement-contract
record; their hashes are in `DEVELOPMENT_LOG.md` D094.

The accepted collection at clean `32b9cfd` is
`.cache/rail_balance/c100/o078/post-32b9cfd-matched-o077/`.  It repeats O077's
ordering with one empty JIT root per benchmark invocation/report, 10 warmups,
100 steady samples,
absolute `PYTHONPATH`, timeout 180/control 240, watchdog 1800 and OMP=1.
Each same-label semantic config SHA equals O077 exactly.  Baseline eligibility,
clean tree, eight benchmark-only GPU PIDs, no MPS/co-tenant/throttle state,
empty then stable JIT identity, and stable extension/source/library identity
all pass 12/12.  The barrier key/SASS is identical across both sides.

Raw O078 distributions, in microseconds:

| Case | median | p95 | p99 | max | mean | std | CV % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source H256 r1 | 111.626 | 126.301 | 331.930 | 470.046 | 118.391 | 41.971 | 35.451 |
| source H256 r2 | 103.800 | 136.766 | 339.464 | 374.701 | 112.177 | 37.818 | 33.713 |
| source H256 r3 | 105.056 | 130.539 | 206.354 | 234.213 | 110.203 | 18.858 | 17.112 |
| source H7168 r1 | 152.931 | 174.393 | 183.608 | 207.648 | 154.928 | 10.321 | 6.662 |
| source H7168 r2 | 132.736 | 143.556 | 180.992 | 2168.790 | 153.666 | 202.625 | 131.861 |
| source H7168 r3 | 138.450 | 154.512 | 437.819 | 505.701 | 146.916 | 48.356 | 32.914 |
| return H256 r1 | 145.070 | 180.500 | 184.114 | 189.186 | 149.262 | 14.103 | 9.449 |
| return H256 r2 | 144.554 | 168.057 | 229.267 | 303.063 | 149.175 | 21.555 | 14.449 |
| return H256 r3 | 165.619 | 200.708 | 228.716 | 280.967 | 170.834 | 16.929 | 9.910 |
| return H7168 r1 | 258.815 | 303.503 | 396.718 | 554.637 | 264.470 | 35.728 | 13.509 |
| return H7168 r2 | 258.519 | 277.056 | 311.436 | 1010.676 | 267.283 | 75.329 | 28.183 |
| return H7168 r3 | 249.987 | 272.477 | 305.393 | 1248.635 | 262.844 | 99.574 | 37.883 |

The frozen decision statistics are:

| Adapter | O077 to O078 median of medians | change | pooled change | trim-10% change | same-label result |
| --- | ---: | ---: | ---: | ---: | --- |
| source H256 | 181.365 to 105.056 us | -42.07% | -41.10% | -41.00% | all improve 37.00--44.76% |
| source H7168 | 188.542 to 138.450 us | -26.57% | -24.91% | -24.24% | all improve 16.32--29.60% |
| return H256 | 157.288 to 145.070 us | -7.77% | -2.91% | -1.94% | worst regression +8.09% |
| return H7168 | 257.583 to 258.519 us | +0.36% | -1.05% | -1.33% | worst regression +0.36% |

Every pre-registered median/pooled/trimmed and same-label threshold passes.
The 2.169-ms source-H7168 event prevents a tail-latency claim; profiler-free
typical latency is the accepted result.

### O078 source Nsys result

Nsys 2024.6.2 separately captured 100 H7168 source ordinals with the exact
O077 diagnostic semantic SHA
`fd076a98558ec555447e950c3994f29f9dfbaf80461ddd5a0db0329e854735e7`.
The report has exactly 800 context-1/stream-26, grid-256/block-32, REG74,
14,432-byte dynamic-shared target launches.  Diagnostic benchmark timing is
explicitly ineligible as profiler-free truth.

| Metric | O077 | O078 | interpretation |
| --- | ---: | ---: | --- |
| device-6 target p50 | 105.376 us | 52.496 us | high-prefix scan cost removed |
| per-ordinal maximum p50 | 105.376 us | 54.080 us | rank gradient contracts |
| device-6 longest count | 100/100 | 5/100 | device 6 no longer fixed straggler |
| pooled target p50 | not decision input | 51.536 us | all launches near pre-O077 scale |
| eight-device target union p50 | 154.975 us | 109.323 us | interval union, not duration sum |
| copy-excluded diagnostic union | 140.806 us | 95.834 us | fixture diagnostic only |

The existing schema-aware `sql/o077_nsys_queries.sql` produced the interval
rows.  The O078 artifacts are:

```text
c544a54e86957ff7140cc9301b241a1f2734079e44b01d404a459ddd150ef8f9  source-h7168.json
af1f3607b12be0f07aba6529cf3c65692019d6f5596c0aa72186044bf588a489  source-h7168.nsys-rep
824970ee12cfbbfd214c70ed3f685171d7a5436b1a095e668f43e00961c54e27  source-h7168.sqlite
```

### O078 source NCU dossier

```text
Kernel:
  rail_balance_hybrid_source_shuffle_impl<7168,4>
Invocation identity:
  32b9cfd; BF16 G8/D9/N1024/C256/H7168/K4; 7,168 moved copies;
  c100/source/stage/steady/26; device 6; context 1; stream 26; grid 256;
  block 32; REG74; dynamic shared 14,432 B; one launch
Nsight Systems exposed time:
  device-6 p50 52.496 us; target-union p50 109.323 us and 95.834 us after
  overlapping fixture-copy removal
Launch configuration:
  256 one-warp CTAs; 0.13 waves/SM; 23.44% theoretical occupancy
Primary limiter removed:
  O(C) data-dependent scan of the canonical monotonic channel prefix
Secondary limiter:
  not established; broad SM/DRAM/LSU utilization remains low
Supporting NCU metrics:
  dynamic instructions 2,971,264 to 823,680 (-72.28%); SM 0.292%;
  DRAM 0.716%; LSU 0.0774%; achieved occupancy 2.663%;
  active warps/SM 1.704
Contradicting evidence:
  static SASS grows 1,352 to 1,384 instructions and H256 uses two more
  registers; dynamic H7168 work still falls sharply and profiler-free/Nsys
  agree; NCU caches/clocks are uncontrolled
Relevant source/PTX/SASS:
  resolve_hybrid_copy in rail_balance_hybrid_plan.cuh; bounded endpoint,
  upper-bound and final-containment checks; source H7168 REG remains 74
Optimization hypothesis:
  confirmed for the frozen large-volume checked adapter
Expected kernel speedup:
  confirmed by Nsys 105.376 to 52.496 us on device 6
Expected end-to-end speedup ceiling:
  missing evidence: no full MoE/training critical-path fraction or real RDMA
Confidence:
  high locally; unknown for real multi-node speedup
Next falsification experiment:
  local transfer-matrix and compute-interference sweeps, then truthful D>1
  Rail/Gin validation
```

NCU 2025.1.1 used the basic set, exact device/range/demangled kernel,
`--launch-count 1`, strict application replay for ten passing runs, and cache/
clock controls `none`.  Its replay duration is not performance truth.  Report
and exact-command hashes are indexed in `DEVELOPMENT_LOG.md` D095.

The persistent cubin-pair audit index is
`artifacts/o078_sass_audit_manifest.json` (SHA256
`3c82681b26c7be428ff060529de2cb265eaca863fa35c7d92c2b72b8bc3b0c70`).
It validates all 168 report-declared artifacts on each side and records exact
raw-cubin, SASS, instruction-count and cubin-resource evidence for all 84
same-label kernel pairs.  The raw JIT roots remain temporary and are not
treated as repository artifacts.

### Decision and remaining evidence

O078 passes exact function/sanitizer, every profiler-free gate, the source
Nsys prediction and the device-6 dynamic-instruction prediction.  Return SASS
and cubin resource usage are identical in every same-label O077/O078 pair; the
plan SASS difference is only an assertion source-line immediate.  O078 is accepted
as the C100 local checked-adapter hot path and O077 remains its auditable mixed
parent.

This does not establish full-MoE/training step speedup, production overlap,
real Gin/RDMA/QP/NIC behavior or stable tail latency.  Transfer-matrix and
compute-interference experiments remain local C100 work.  C080-H/C110 still
own truthful multi-node activation and performance.

### O079 transfer-matrix contract

O079 prepares the next local C100 evidence without changing production
kernels.  The named matrix fixtures are:

```text
c100_matrix_fanout_h256    c100_matrix_fanout_h7168
c100_matrix_fanin_h256     c100_matrix_fanin_h7168
c100_matrix_mesh_h256      c100_matrix_mesh_h7168
c100_matrix_rot1_h256      c100_matrix_rot1_h7168
c100_matrix_rot4_h256      c100_matrix_rot4_h7168
```

All use G8/D9/N2048-per-rank/K4/C256/Pcap7168 and 7,168 moved records.  The
CPU oracle asserts the exact owner-to-egress moved-copy matrix for each case:
fan-out, fan-in, full mesh, and rotations by 1 and 4.  The benchmark report
schema is version 3 and records:

```text
owner_to_egress_moved_copies
outgoing_moved_copies_per_rank
incoming_moved_copies_per_egress
selected_moved_copies_per_rank
outgoing_logical_bytes_per_rank
incoming_logical_bytes_per_egress
logical_bytes_per_rank
```

For source measurements, `logical_bytes_per_rank` aliases outgoing owner row
sums.  For return measurements, it aliases incoming proxy-egress column sums.
The aggregate numerator still counts every moved TokenLayout record once and
does not claim physical NVLink/HBM transaction bytes.

The validation run includes `py_compile`, C080-D source oracle, C080-F return
oracle on `c100_matrix_rot4_h7168`, normal C080-F return oracle, and `git diff
--check`.  All ten named matrix cases also pass true 8-GPU LSA source and
return functionality gates.  Source validates moved=7168, per-owner/egress=896,
legacy TokenLayout bytes and immutable inputs.  Return validates moved=7168,
exact owner row/token bytes and poison-preserved non-targets.

No O079 no-profiler, Nsys or NCU performance result has been accepted.  The
GPU functionality run is explicitly not performance evidence because
`nvidia-smi` reported visible GPU model `NVIDIA L20X` and GPU0 had a
co-tenant non-megatron Python process using about 15 GiB.

### O080 compute-interference diagnostic harness

O080 extends only the C100 benchmark harness.  It adds:

```text
--interference-mode none|compute-only|concurrent
```

`none` preserves the existing source/return checked-adapter path and is the
only mode that can be baseline-eligible.  Non-`none` modes require an H7168
case and allocate one fixed BF16 GEMM per rank:

```text
[1024,7168] @ [7168,7168] -> [1024,7168]
```

`compute-only` times the GEMM window and records timed logical moved bytes as
zero.  The moved-record byte scope remains available as
`stage_logical_bytes_*` reference fields.  `concurrent` launches the GEMM on an
independent CUDA stream before the checked adapter call and synchronizes the
compute stream before the post-stage WORLD gate.  This is diagnostic until
Nsys proves actual overlap.

Smoke evidence so far: parser/py_compile pass; fan-out/fan-in/mesh/rot1/rot4
benchmark entry accepts non-symmetric matrix cases; source compute-only,
source concurrent, return concurrent, and default `none` source smokes pass
with one steady iteration.  The generated schema-v4 compute-only JSON smoke
has `logical_bytes_aggregate=0`, positive `stage_logical_bytes_aggregate`, and
`baseline_collection_eligible=false`; per-rank raw records include
`compute_stream_id` for future Nsys identity matching.

No O080 performance conclusion is accepted for the same reason as O079: the
run had a GPU0 co-tenant and the visible GPU model string was `NVIDIA L20X`.
