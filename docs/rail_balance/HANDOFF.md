# Rail Balance Session Handoff

Last updated: 2026-07-22 UTC
Branch: `feat/rail-balance-prototype`
Repository: `/home/chen/workspace/source_code/DeepEP`

## Safe pause state

- H6 constructor consensus and the public one-shot force lifecycle are
  implemented, independently audited, and split into production/test commits;
  the real EP8 constructor/public D=1 watchdog also passes. C080-G final-tree
  compatibility and focused sanitizer closure now pass. There is no unfinished
  local Hybrid integration slice or half-written host abstraction.
- C000 through C070 remain complete for the agreed single-node PoC scope.
- C090 exhaustive local correctness is PASS. The final additions are three
  named route distributions plus one g812--g814 true-EP8 corrupt-p/delayed-
  rank/recovery case; no local correctness or sanitizer gap remains.
- C080-A/B/C/D are PASS. C080-D now includes corrupt/missing world-plan
  pre-B0 abort plus same-buffer next-generation recovery; C080-C has the
  fixed-tensor WORLD transaction, public ticket, and truthful D1 watchdog.
- C080-E/F isolated force Hybrid dispatch/combine codegen are complete. Six
  force/legacy combine pairs have identical REG/STACK/SHARED/LOCAL/spill
  resources except the expected eight-byte constant pointer argument.
- Recent code and measurement checkpoints are `1888908` (complete
  profiler-free collection), `f5e7906` (Nsys contract), `feaf03e` (steady-only
  diagnostic NVTX mode), and `bf01ccd` (final clean instrumentation smokes) on
  `fork/feat/rail-balance-prototype`.
- Both capability bits remain false and no result claims real Gin/RDMA
  behavior. The local activation evidence is complete, but the public methods
  deliberately remain unreachable through a normal force construction until a
  truthful D>1 Rail/Gin gate passes.
- Resume from the working tree and the newest entries in `DEVELOPMENT_LOG.md`;
  do not restore the obsolete sidecar/descriptor/ring draft.

## Immutable checkpoints

- C000-C061 Git tree:
  `35e3d8970216bee0547b40e810873a623d677c71`
- C000-C070 Git tree, before this handoff metadata:
  `95cee99ae5f423821ca0eb246bb82487d502d36d`

These are Git tree objects created from the reviewed staged index. They preserve
the exact file snapshots without inventing commit identity.

## Latest accepted evidence

- C061 H256 and H7168 fresh-cache eight-GPU 2x4 runs: PASS.
  - 33 destination copies.
  - Six required per-destination moves despite aggregate 17/16 balance.
  - 72 expert contributions and 282 byte-exact global coverage records.
- C070 H256 and H7168 capture plus two cross-call replays: PASS.
  - Planner manifest and fingerprint inputs were poisoned and deleted before
    replay.
  - Replay uses owning base/expert records, descriptor, route, ready, quota,
    and top-k only.
  - Both replays match an independent CPU expert formula; snapshot inputs are
    byte-immutable; status is zero; the 4 KiB guard is intact.
- One live expert route corruption reports `RouteMismatch=4`, keeps the guard
  intact, and exits collectively: PASS.
- Planner 38/38, C060 compatibility, the original EP8 public path, and the
  in-place extension build pass after C070.
- C070 final independent review: 0 Blocker, 0 High, 1 Medium, 0 Low. The trusted
  snapshot PoC is accepted.
- C080-B1 fresh-cache H200 materializer: PASS.
  - C061 remains 33 destination copies and six moves.
  - Pcap 3 succeeds and Pcap 2 preserves the candidate with capacity status.
  - Zero tokens, 64 random seeds, C1024/D32, signed-int64 seed, invalid routes,
    invalid seeds, and a non-default stream pass exact CPU comparison.
  - Each count/plan/prefix cubin exports exactly one kernel symbol.
  - Independent audit after fixes: 0 Blocker / 0 High.
- C080-E force dispatch codegen: four production-shaped force/legacy pairs,
  zero spill, API 6/6, legacy goldens 4/4, independent review PASS.
- C080-F force combine codegen: six production-shaped force/legacy pairs cover
  all four rank-layout combinations. All are REG216/STACK96/SHARED1024/LOCAL0
  with zero spill; force adds only 8B constant memory. Independent review is
  0 Blocker / 0 High / 0 Medium.
- H1 prepared adapters: root fresh-cache dispatch/combine smoke, prebuilt
  dispatch epilogue source gate, API 6/6, legacy goldens 4/4, and independent
  adapter review pass with Blocker/High 0/0. Public force remains false.
- H1b raw-submit closure: source and return true eight-GPU LSA cases, three
  adapter gates, a real header-triggered extension rebuild, API 6/6, legacy
  goldens 4/4, and independent final review pass. A reviewed N/grid mismatch
  was fixed by making Prepared N the sole owner; final audit is 0/0/0/0.
- H2 force-only sizing: API 9/9, distributed layout 5/5, legacy 4/4, complete
  runtime-argument identity, helper failure ordering, and independent audit
  pass. Runtime receives one `legacy+arena` window; off remains field/helper
  identical and public force remains false.
- H3a fixed gate: one reusable 1 KiB CUDA/pinned pair, one MAX per round,
  exact field mismatch plus deterministic error convergence, true EP8 and
  non-default-stream evidence, and independent audit 0/0/0/0. It is not yet
  connected to plan/dispatch.
- The H3a EP8 command was rerun immediately before the pause on port 29951 and
  passed. H3b later passed its final EP8 run with eighteen fixed MAX gates and
  final audits of 0 Blocker/High/Medium.
- H4a explicit pending states are committed at `52b165e`. After a full rebuild,
  H3b's eighteen-gate EP8 suite (including stale abort), true 8-GPU source and
  return LSA paths, source faults, raw adapters, combine codegen, API 9/9, and
  legacy 4/4 all pass. Independent state audit found no blocking issue.
- H4b owning prepare is committed at `4e48b84` with gates at `7ab382b`.
  Production D/G/C and all round-trip runtime/tensor/raw ownership are frozen
  before Gate1; precommit abort drains in-flight comm work before releasing
  storage. Full build, H4b three-gate EP8 fail-close/recovery, canonical H3b
  eighteen gates, B1/B2/vnode, API/legacy, and focused codegen pass. The local
  machine cannot execute a truthful D>1 production prepare.
- H4c adjacent commit is pushed at `edb1a0e` with its source gate at
  `ba82134`. It pre-poisons ownership, submits source and force main on the
  same stream with no intervening work, then enters DispatchLive. G8xD2/H7168
  codegen, true EP8 H7168 source LSA, 4x2/H7168 vnode, H4b three gates, H3b
  eighteen gates, API/legacy, build, and independent audit pass. This is not
  a real Gin runtime result; low-level partial launch is job-fatal.
- H5a owning finish is committed at `81ecb86` with its source gate at
  `7bd17a4`. It preserves the native 16-item dispatch result order, owns exact
  receive storage before the epilogue, bounds mapped int64 counters before any
  narrowing/allocation, and performs the first safe status readback plus
  `comm_stream` sync. Full build, H5a/H4c/H4b CPU gates, API 9/9, legacy 4/4,
  truthful EP8 D1 fail-close/recovery, 4x2/H7168 vnode, true EP8 return LSA,
  and G8xD2/H7168 dispatch codegen pass. Local production success remains
  impossible at D=1; this is not Gin runtime evidence.
- H5b private combine ownership is committed at `ffe4fd4` with its exact source
  contract at `403e725`. Prepare is retryable before the future WORLD combine
  gate; abort releases only the new combine owner; commit poisons once and
  submits main combine, return-unshuffle, local barrier, and native epilogue
  adjacently before one correctness sync. Adapter, codegen, API/legacy, build,
  truthful D1 fail-close, and independent C++ audit pass. Public `EPHandle`
  identity/consumption was deliberately still closed at that checkpoint.
- H6 constructor consensus is committed at `1e0ea60` with tests at `eb6b834`.
  One universal pre-comm gate rejects mixed mode/geometry before symmetric
  window divergence; force reuses the same 1 KiB CUDA/pinned storage for one
  post-sizing/pre-window gate. Constructor 9/9, API 9/9, legacy 4/4, and
  independent Blocker0/High0 review pass.
- H6 public lifecycle is committed at `741e126` with tests at `cbe7df2`.
  Dispatch performs prepare/Gate1, plan/Gate2, then native commit/finish;
  combine validates one buffer-owned shared ticket, prepares/gates, consumes
  before commit, and returns the native three-item result. Success, shallow-
  copy one-shot, capacity/remote rejection, retry, foreign/off misuse, prepare,
  commit, and finish faults pass the CPU fake runtime. H3-H5 source contracts,
  API 9/9, legacy 4/4, pycompile/diff, and final public audit pass with no
  Blocker/High. This evidence does not execute production D>1 Gin.
- H6 real EP8 watchdog is committed at `0597ed0`. It rejects mixed mode and a
  rank-3 invalid force geometry before communicator creation, creates a real
  unanimous force window, rejects public dispatch twice at the truthful D1
  guard, and collectively destroys. It observes exactly six fixed MAX gates,
  stable accepted-constructor storage, invocation 1→3, no live ticket, and a
  nonterminal retryable buffer. The audit closes Blocker0/High0; production
  capability is restored false and no payload/Gin stage executes.
- C080-D world-plan fault closure is committed at `6377e0c`. Rank 0 corrupts
  one prepared value and rank 1 models one missing tensor; exact tagged
  preflight failures converge before B0, both source/world owners abort, and
  the next generation completes the full 4x2/H256 round trip on the same
  buffers. True EP8 and independent Blocker0/High0 review pass.
- C080-G final-tree closure passes both default-off EP8 modes, the focused
  CPU/source and H3-H6 contracts, 8x1 source/return, 4x2 and D>K 2x4 vnode,
  source/world-plan faults, fresh force/legacy codegen, and the full extension
  build. Compute Sanitizer 2025.1 reports zero errors for memcheck, synccheck,
  initcheck, and focused racecheck on the complete D>K vnode path; racecheck
  also reports zero hazards/warnings. Final audit is Blocker0/High0/Medium0.
  C080 is `LOCAL_COMPLETE / MULTINODE_PENDING`; C080-H remains an environment
  gate, and both production capability bits remain false.
- C090 closes at `5f8eab5`. The B1 matrix is now 70 unchanged cases plus named
  one-hot/Zipf/log-normal, all 73 exact on one GPU. A fresh-cache true-EP8
  4x2/H256 run checks the complete corrupt-p status graph and collective abort
  at generation 812, a rank-2 native comm-stream delay with full success at
  813, and ordinary same-buffer recovery at 814. Full rebuild, API 9/9,
  legacy 4/4, and final C++/protocol/exit audits pass. No CUDA/JIT kernel or
  production hot path changed, so the accepted C080-G sanitizer evidence is
  unchanged.
- C100's audited harness is committed at `056ad4d`; the clean samples use
  commit `3c54839`. Six source reports (three H256, three H7168, each 10+100)
  pass automatic collection gates and are stored with a verified SHA256
  manifest under `.cache/rail_balance/c100/formal/3c54839/`. They are not a
  stable accepted baseline: H256 medians span 7.26%, H7168 medians 20.22%, and
  one rank-4 sample reaches 2.254 ms. No Nsys/NCU or performance-path change
  followed. The first return run was user-interrupted, exited 130 through the
  watchdog, wrote no JSON, and left no worker/GPU process.
- A resumed one-variable H7168 experiment at clean `4211bae` set only
  `OMP_NUM_THREADS=1`.  Three 10+100 reports cut pooled CV from 82.84% to
  8.34% and maximum from 2.274 ms to 218.716 us, but their medians still span
  14.94%.  Rank 1 starts first in 299/300 samples and rank 7 last in 295/300;
  the remaining global-span variability follows post-Gloo release skew.
  Reports plus a verified manifest live under
  `.cache/rail_balance/c100/stability-omp1/4211bae/`.  No Nsys/NCU or hot-path
  change occurred in that experiment.  It led to the now-complete
  release-skew falsification recorded in the next bullet.
- That release-skew falsification is now complete.  The reviewed CPU-only
  probe at `ee41415` produces three clean Gloo return medians of
  38.693/36.319/38.028 us; rank1-first and rank7-last each dominate at least
  98%.  Even the smallest median covers at least 59.97% of every OMP E1 start
  skew, so the pre-registered 50%/80% criterion passes.  Verified artifacts
  are under `.cache/rail_balance/c100/gloo-gate/ee41415/`.  Global spans remain raw, but
  kernel work must be attributed from rank-local adapter envelopes and Nsys,
  never by subtracting the probe median.  Source-H256 and return-H256 OMP=1
  repeated collection are complete as raw evidence; the next bullet records
  the subsequently completed return-H7168 group.
- Return-H256 at clean `d05411a` has accepted reports `{r1,r3,r4}`.  Global
  medians are 225.189/215.610/218.210 us, but r3 retains a 1.419 ms maximum;
  this is not a stable-tail baseline.  `r2` is preserved but rejected because
  the post-run gate detected Megatron restart7 worker PIDs 1097155--1097158.
  Its values must never enter pooled statistics.
- Return-H7168 at clean `57b69a9` also has accepted reports `{r1,r3,r4}`.
  Global medians span 2.37% and rank-local medians 2.12%, but accepted maxima
  reach 2021.557/2008.428 us.  A concurrent eight-GPU DeepEP dispatch demo
  makes r2 permanently rejected.  All four profiler-free source/return groups
  are now collected; retained tail attribution is the remaining baseline task.
- The clean `feaf03e` benchmark has a default-off, baseline-ineligible NVTX
  diagnostic mode.  Final-commit default-off and NVTX-on EP8 smokes pass after
  one separate co-tenant tester report was correctly rejected.
- Nsys's child-owned range trigger is experimentally rejected.  Default
  `wait=all` holds a resource-tracker zombie in the watchdog PGID;
  `--wait=primary` fixes cleanup, but a child range still cannot start
  collection.  The accepted topology is full-process capture plus global-time
  filtering to `c100_nsys_window`.  A 0+1 report/SQLite pair proves all eight
  devices are present; formal return-H7168 10+100 collection is next.

## Mandatory retained boundary

The fixed WORLD consensus and uninterrupted private C++ commit sequences are
connected to public dispatch/combine, and real EP8 covers constructor, window,
D1 fail-close/retry, and destroy. Local regression and focused sanitizer
closure are complete. D=1 still cannot turn the production Hybrid data path
into a success because it has no remote destination server.

Post-publication failure must continue to invalidate the force object;
precommit abort must not destroy an older live dispatch handle. Do not weaken
these rules merely to make a local D=1 test succeed.

No current result claims real Gin/RDMA/QP/NIC behavior or a multi-node speedup.
No C070 timing is performance evidence.

## Resume instructions

1. Read this file plus `CONTEXT.md`, `CHECKPOINTS.md`, the tail of
   `DEVELOPMENT_LOG.md`, and `OPTIMIZATION_LOG.md` decisions O011-O012.
2. Verify branch and state with `git status --short --branch`, then run both
   `git diff --check` and `git diff --cached --check`.
3. Do not redo C061/C070 unless the relevant code changes.
4. Do not rerun C080-G or every historical exhaustive seed unless a relevant
   production core changes. Its final-tree matrix and sanitizer closure pass.
5. Keep both production capability bits false until truthful C080-H D>1
   Rail/Gin correctness passes. D1 evidence is never a D>1 Gin success claim.
6. C090 is complete. Do not rerun its unchanged full suites or sanitizer
   kernels unless production device code changes.
7. The GPU evidence-chain skill is installed and the user has already been
   notified that Nsys started.  Run the pre-registered full-process
   return-H7168 10+100 capture with `--wait=primary`; analyze all ranks through
   the global steady timestamp window.  NCU may target only an exact exposed
   invocation selected by that report, and hot-path optimization remains
   blocked until the evidence exists.  C105 packages the fastest possible
   real-cluster bring-up afterward.

Pinned runtime for accepted commands:

```text
/home/chen/.cache/deepep-sjlgpt/bin/python
PYTHONPATH=$PWD/tests:$PWD
EP_DISABLE_GIN=1
```

Use fresh `EP_JIT_CACHE_DIR` values after CUDA/include changes. Run performance
tests only when GPUs are idle; functional tests may run under contention if
they do not risk OOM.
