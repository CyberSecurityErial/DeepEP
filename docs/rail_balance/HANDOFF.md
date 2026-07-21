# Rail Balance Session Handoff

Last updated: 2026-07-21 UTC
Branch: `feat/rail-balance-prototype`
Repository: `/home/chen/workspace/source_code/DeepEP`

## Safe pause state

- H6 constructor consensus and the public one-shot force lifecycle are
  implemented, independently audited, and split into production/test commits;
  no half-written lifecycle file remains. The next slice is truthful EP8
  constructor/public D=1 fail-close evidence, not another host abstraction.
- C000 through C070 remain complete for the agreed single-node PoC scope.
- C080-A/B are complete; C080-D shared source/return cores and vnode closure
  pass their current local gates. C080-C now has the fixed-tensor WORLD
  transaction and buffer-bound public ticket, but remains open for real local
  watchdog/regression/sanitizer closure.
- C080-E/F isolated force Hybrid dispatch/combine codegen are complete. Six
  force/legacy combine pairs have identical REG/STACK/SHARED/LOCAL/spill
  resources except the expected eight-byte constant pointer argument.
- Latest pushed checkpoints are `741e126` (public Hybrid one-shot lifecycle)
  and `cbe7df2` (public lifecycle/fault contract), following `1e0ea60`/
  `eb6b834` (constructor consensus) on
  `fork/feat/rail-balance-prototype`.
- Both capability bits remain false and no result claims real Gin/RDMA
  behavior. The public methods are wired but unreachable through a normal
  force construction until the local activation evidence is complete.
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

## Mandatory retained boundary

The fixed WORLD consensus and uninterrupted private C++ commit sequences are
now connected to public dispatch/combine, but only fake/source evidence covers
that final Python lifecycle. The next local proof must exercise the real EP8
constructor and public D=1 failure path under a watchdog: every rank must reject
before publication, abort the same invocation, remain retryable where allowed,
and destroy cleanly. D=1 cannot turn the production Hybrid data path into a
success because it has no remote destination server.

Post-publication failure must continue to invalidate the force object;
precommit abort must not destroy an older live dispatch handle. Do not weaken
these rules merely to make a local D=1 test succeed.

No current result claims real Gin/RDMA/QP/NIC behavior or a multi-node speedup.
No C070 timing is performance evidence.

## Resume instructions

1. Read this file plus `CONTEXT.md`, `CHECKPOINTS.md`, the tail of
   `DEVELOPMENT_LOG.md`, and `OPTIMIZATION_LOG.md` decisions O011-O012.
2. Verify branch and state with `git status --short --branch` and run
   `git diff --cached --check`.
3. Do not redo C061/C070 unless the relevant code changes.
4. Add the smallest real EP8 watchdog around existing public APIs. Temporarily
   expose the already compiled private capability only inside each test process;
   cover mixed off/force Gate0, asymmetric invalid force config, unanimous
   force construction/destruction, and public D=1 dispatch fail-close plus a
   second retry. Use small H256/K2/M/Pcap, `EP_DISABLE_GIN=1`, explicit process-
   group teardown, and a hard timeout. Do not add vnode or a second runtime.
5. Keep both production capability bits false until that watchdog, default-off
   regression, focused C080-G/C090 sanitizer, and an independent boundary audit
   pass. D=1 evidence may justify a test-only activation hook, never a D>1 Gin
   success claim.
6. After local correctness closure, notify the user before entering controlled
   C100 NCU/Nsys work and wait for the promised profiling procedure. Run
   performance tests only with idle GPUs; retain negative/tool-failure evidence.

Pinned runtime for accepted commands:

```text
/home/chen/.cache/deepep-sjlgpt/bin/python
PYTHONPATH=$PWD/tests:$PWD
EP_DISABLE_GIN=1
```

Use fresh `EP_JIT_CACHE_DIR` values after CUDA/include changes. Run performance
tests only when GPUs are idle; functional tests may run under contention if
they do not risk OOM.
