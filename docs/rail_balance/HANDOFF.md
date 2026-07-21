# Rail Balance Session Handoff

Last updated: 2026-07-21 UTC
Branch: `feat/rail-balance-prototype`
Repository: `/home/chen/workspace/source_code/DeepEP`

## Safe pause state

- H5a dispatch completion and H5b private combine completion are implemented,
  independently audited, and split into implementation/test commits; no
  half-written C++ lifecycle file remains. The next slice is the minimal public
  Python `EPHandle` ownership/consumption and WORLD-gate wiring.
- C000 through C070 remain complete for the agreed single-node PoC scope.
- C080-A/B are complete; C080-D shared source/return cores and vnode closure
  pass their current local gates. C080-C remains open only for the production
  fixed-tensor WORLD transaction/state machine.
- C080-E/F isolated force Hybrid dispatch/combine codegen are complete. Six
  force/legacy combine pairs have identical REG/STACK/SHARED/LOCAL/spill
  resources except the expected eight-byte constant pointer argument.
- Latest local implementation checkpoints are `ffe4fd4` (owning combine
  lifecycle) and `403e725` (H5b source gate), following `81ecb86`/`7bd17a4`
  (owning dispatch finish), `edb1a0e`/`ba82134` (adjacent Hybrid
  dispatch commit) and `4e48b84`/`7ab382b` (owning prepare), on
  `fork/feat/rail-balance-prototype`.
- Public force still fails closed and no result claims real Gin/RDMA behavior.
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
  identity/consumption and force capability are deliberately still closed.

## Mandatory retained boundary

The remaining boundary is production reliability, not CUDA codegen: force must
use fixed CUDA tensor WORLD consensus before publication and one uninterrupted
C++ committed sequence on every rank. No rank may throw between source shuffle
and Hybrid dispatch Tag0, or between main combine and the post-unshuffle local
barrier/legacy epilogue. Post-publication failure invalidates the force object;
precommit abort must not destroy an older live dispatch handle.

No current result claims real Gin/RDMA/QP/NIC behavior or a multi-node speedup.
No C070 timing is performance evidence.

## Resume instructions

1. Read this file plus `CONTEXT.md`, `CHECKPOINTS.md`, the tail of
   `DEVELOPMENT_LOG.md`, and `OPTIMIZATION_LOG.md` decisions O011-O012.
2. Verify branch and state with `git status --short --branch` and run
   `git diff --cached --check`.
3. Do not redo C061/C070 unless the relevant code changes.
4. H3/H4/H5a/H5b now reach a synchronized private DispatchLive result and a
   complete private combine transaction. Wire the smallest public force path:
   dispatch prepare → fixed WORLD Gate1 → finish/Gate2 → commit/finish → exact
   native tuple; bind the result to a buffer/invocation-owned, one-shot,
   non-cached handle. Combine must validate that handle, prepare before its
   WORLD gate, abort retryably on gate rejection, atomically consume on gate
   acceptance, then call the existing private commit. Do not return to vnode
   or add a descriptor/ring/coordinator. Preserve the distinction between host
   launch acceptance and real Gin completion.
   Its accepted evidence is in
   `tests/elastic/test_rail_balance_hybrid_plan_world_gate.py`: CPU and EP8
   watchdog, four abort/retry states, capacity/asymmetric fail-close,
   fourteen-output oracle, stable storage, actual-K mismatch, non-default
   compatibility, and exactly two MAX calls per completed transaction.
5. Keep public force unavailable until force combine is connected as one
   main-combine→return-unshuffle→local-barrier→legacy-epilogue commit and handle
   ownership/consumption is fail-closed.
6. After host closure, run C080-G/C090 compatibility and sanitizer gates. Run
   controlled C100 NCU/Nsys only while GPUs are idle and retain negative/tool-
   failure evidence.

Pinned runtime for accepted commands:

```text
/home/chen/.cache/deepep-sjlgpt/bin/python
PYTHONPATH=$PWD/tests:$PWD
EP_DISABLE_GIN=1
```

Use fresh `EP_JIT_CACHE_DIR` values after CUDA/include changes. Run performance
tests only when GPUs are idle; functional tests may run under contention if
they do not risk OOM.
