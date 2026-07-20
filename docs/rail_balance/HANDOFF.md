# Rail Balance Session Handoff

Last updated: 2026-07-21 UTC
Branch: `feat/rail-balance-prototype`
Repository: `/home/chen/workspace/source_code/DeepEP`

## Safe pause state

- C000 through C070 remain complete for the agreed single-node PoC scope.
- C080-A is complete: default-off identity, strict constructor config, disabled
  capability gate, checked force arena ABI, and EP8 size agreement pass.
- C080-B1 is complete: the private stacked-owner GPU count/plan/prefix
  materializer matches 69 exact H200 cases and final independent review reports
  0 Blocker/High. C080-B2 still needs the per-rank LSA count snapshot; C080-C
  still needs cross-rank preflight/status.
- Latest pushed checkpoints are `87cfcc5` (GPU materializer) and `e46b701`
  (strict CUDA tests) on `fork/feat/rail-balance-prototype`.
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

## Mandatory retained boundary

The remaining Medium is production reliability, not valid-snapshot
correctness: `_rail_balance_vnode_replay` performs rank-local host validation
before the first collective. A malformed input on only one rank can throw while
peers enter a barrier. C080 must add bounded cross-rank preflight consensus or a
common abort protocol before exposing this path.

No current result claims real Gin/RDMA/QP/NIC behavior or a multi-node speedup.
No C070 timing is performance evidence.

## Resume instructions

1. Read this file plus `CONTEXT.md`, `CHECKPOINTS.md`, the tail of
   `DEVELOPMENT_LOG.md`, and `OPTIMIZATION_LOG.md` decisions O011-O012.
2. Verify branch and state with `git status --short --branch` and run
   `git diff --cached --check`.
3. Do not redo C061/C070 unless the relevant code changes.
4. Resume at C080-B2/C, not at the obsolete CPU-manifest or sidecar paths:
   - count one local owner per rank into the force arena;
   - establish a real LSA-team barrier and compact `[G,C,D]` snapshot;
   - feed the existing B1 plan/prefix semantics unchanged.
5. Keep public force unavailable until dispatch and combine both exist. Add
   world-consensed Gate1 before any local barrier and Gate2 after capacity
   status but before payload publication; do not let a rank-local throw strand
   peers.
6. Prioritize the direct-final 8x1 skewed source-shuffle path, then 4x2/2x4
   closure and isolated Hybrid codegen. Performance runs require idle GPUs;
   controlled C100 profiling uses NCU/Nsys and retains negative/tool-failure
   evidence.

Pinned runtime for accepted commands:

```text
/home/chen/.cache/deepep-sjlgpt/bin/python
PYTHONPATH=$PWD/tests:$PWD
EP_DISABLE_GIN=1
```

Use fresh `EP_JIT_CACHE_DIR` values after CUDA/include changes. Run performance
tests only when GPUs are idle; functional tests may run under contention if
they do not risk OOM.
