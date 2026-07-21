# Rail Balance Session Handoff

Last updated: 2026-07-21 UTC
Branch: `feat/rail-balance-prototype`
Repository: `/home/chen/workspace/source_code/DeepEP`

## Safe pause state

- The lunch archive was resumed. H3b is now implemented, audited, committed,
  and pushed; no half-written test remains.
- C000 through C070 remain complete for the agreed single-node PoC scope.
- C080-A/B are complete; C080-D shared source/return cores and vnode closure
  pass their current local gates. C080-C remains open only for the production
  fixed-tensor WORLD transaction/state machine.
- C080-E/F isolated force Hybrid dispatch/combine codegen are complete. Six
  force/legacy combine pairs have identical REG/STACK/SHARED/LOCAL/spill
  resources except the expected eight-byte constant pointer argument.
- Latest pushed implementation checkpoint is `878b0db` (real planner WORLD
  gate test), following `2f39a46` (prevalidated Gate2 word patch), `dc2b42c`
  (H3a record), and the earlier H2/H1b closures, on
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
4. H3b already connects H3a storage to private plan Gate1/Gate2. Implement the
   production C080 dispatch transaction next; do not return to vnode or the
   obsolete CPU-manifest/sidecar path. Keep the pre-window mixed-mode boundary
   explicit, derive C/topology in C++, and launch shuffle→force dispatch in one
   C++ commit call.
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
