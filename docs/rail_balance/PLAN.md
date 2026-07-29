# DeepEP Source-Side Rail Balance Plan

## Objective

Prototype and validate a source-local, per-destination rail-balancing path for
DeepEP V2 Hybrid dispatch/combine. The implementation must redistribute only
the traffic needed to reduce rail tail, preserve DeepEP's destination-side
forwarding, and keep the existing public API and default behavior compatible.

The target data path is:

```text
dispatch:
owner GPU -> optional local shuffle -> egress GPU/rail -> destination ingress
          -> existing destination forwarding -> expert GPU

combine:
expert GPU -> destination ingress -> source proxy GPU
           -> optional local return unshuffle -> original owner GPU
```

## Non-negotiable design constraints

1. Balance independently for every `(source server, destination server)` pair.
2. Count one hidden-payload copy per distinct destination server for each token;
   do not sum expert/rank counts when top-k entries share a server.
   Exclude the source-local destination because it does not consume a NIC rail.
3. Preserve the current fast path. Production default remains `off`, and a
   balanced workload must be able to bypass the new data path.
4. Use move-only assignment. Retained traffic stays on its owner rail.
5. Precompute proxy slots and offsets; no per-copy peer-hotspot `atomicAdd` in
   the intended production data path.
   Retained and incoming proxy copies share one egress/destination slot
   namespace, and every per-channel capacity bound must be proven.
6. Write moved payloads directly into the egress GPU's final send slots whenever
   possible; avoid a temporary shuffle buffer followed by an HBM copy.
7. Dispatch and combine are symmetric. Every proxy copy must carry enough route
   information to return to the original owner without slot collisions.
8. Bound payload duplication and proxy capacity. In experimental `force`,
   capacity exhaustion is a collective pre-publication error; partial balancing
   or original-path fallback belongs to the later `auto` policy.
   Production proxy payload space is reserved in the constructor's symmetric
   communication window; dispatch-time dynamic tensors are not Gin source
   buffers.
9. Prototype with internal launchers/JIT specializations; do not add a new
   public PyTorch operator unless later evidence requires one.
10. Keep unsuccessful designs, negative performance results, and environment
    blockers in the logs.

## Work packages and exit gates

### WP0 — Reproducible baseline and project records

- Create an isolated branch and persistent context/development/optimization
  logs.
- Record the exact upstream commit, build/JIT state, runtime topology, and test
  resource policy.
- Run the smallest safe existing EP8 correctness smoke after confirming buffer
  memory fits.
- Build or link the in-place `_C` extension and run with `PYTHONPATH=$PWD`, so
  JIT include roots observe edits in this checkout rather than a stale copied
  `build/lib` tree.
- Explicitly document that a one-node run selects the direct path and therefore
  does not by itself cover `hybrid_dispatch`/`hybrid_combine`.

Exit gate: logs exist, baseline command/result is recorded, and any baseline
failure is classified as code, build, dependency, or transient resource state.

### WP1 — Planner contract and CPU oracle

- Define normalized inputs as destination-copy records produced after per-token
  destination-server deduplication.
- Define a stable owner/channel/destination copy ordinal before using
  `owner_begin` or channel prefixes.
- Implement a deterministic CPU oracle for quota, keep counts, surplus/deficit
  segments, channel prefixes, static proxy slots, moved bytes, duplication, and
  bypass reasons.
- For `total = base * G + remainder`, assign the `base + 1` quota first to
  rails with `count > base`, using a canonical destination/seed-derived ring
  order as the tie-break. A fixed count-blind remainder rotation can move data
  unnecessarily and can even break identity on an already balanced input.
- Cover exact-copy, token-primary, and bounded-split policies at the model level;
  use exact-copy first to establish the balancing upper bound.

Exit gate: exhaustive small cases and seeded randomized cases satisfy all
conservation, balance, uniqueness, capacity, and determinism invariants.

### WP2 — Internal GPU planner prototype

- Add an internal JIT launcher and a small CUDA planner kernel.
- Start with an explicit plan stage; fusion into notify warps is deferred.
- Compare every output field against the CPU oracle.
- Instrument original/balanced bytes, moved copies/bytes, segment count,
  capacity/high-water mark, and bypass reason.

Exit gate: CPU/GPU plan equality across boundary, skewed, masked, multi-
destination, and random distributions.

### WP3 — 8x1 one-shot source-shuffle prototype

- Add a standalone internal LSA launcher with an explicit virtual topology;
  do not first refactor all existing Gin count/tail/signal operations behind a
  transport abstraction.
- Implement owner-to-egress direct-final-slot writes using the same payload,
  scale-factor, and metadata layout as production.
- Implement only move-only exact-copy in this checkpoint. Token-primary and
  bounded split are deferred until the upper-bound path is proven.
- Use one-shot static slots. Ring reuse, backpressure, and `auto` are not part of
  this MVP.
- Use fingerprints, byte checks, canaries, and unique slot checks.

Exit gate: no lost, duplicate, misrouted, or corrupted copies for all eight
concurrent source GPUs, including capacity boundary cases.

### WP4 — One-shot proxy protocol hardening (completed PoC evidence)

- The C050 standalone overlapping PoC uses generation/sequence values rather
  than a reusable boolean-only ready flag.
- Define payload publication and descriptor acquire/release ordering.
- If readiness may be out of order, publish only a contiguous-ready tail; never
  advance a monotonic consumer tail across a hole.
- Test out-of-order readiness, random producer delay, slow consumers, and
  whole-plan capacity bypass before adding slot reuse.
- Batch readiness publication only after the scalar-correct protocol is proven.

Exit gate: long repeated one-shot runs and injected scheduling variation
complete without deadlock, overwrite, or stale-read failures. Small-ring
wraparound/ABA testing is unlocked only after the first 4x2 loop works.

### WP5 — Virtual multi-node correctness loop

- `4x2`: source GPUs 0–3 and destination GPUs 4–7, four emulated rails, full
  dispatch -> forwarding -> synthetic expert -> combine -> return-unshuffle.
- `2x4`: four logical nodes with two rails, emphasizing per-destination balance
  and multi-destination token copies.
- Compare results after sorting by stable source/destination/expert identifiers;
  never rely on physical arrival order.

Start with BF16, non-expanded data, and one fixed reduction mode. Exit gate for
this package is an exact BF16 round trip including original-owner restoration;
FP8 and the remaining semantic matrix are unlocked in WP6.

### WP6 — Handle, replay, and training semantics

- Keep the legacy transmitted `TokenLayout` byte-identical. In restricted
  non-cached force-v1, carry only compact proxy slot `p` in the otherwise
  unused transit lifetime of `linked_list_idx[0]`; destination forwarding
  snapshots it before restoring the normal linked-list fields. Do not append
  it to `recv_src_metadata`, whose columns after index 2 have expanded-layout
  meaning.
- Group `p` by `(egress, channel, destination)` so destination, channel,
  remote slot, original owner/token, and final reduce row are derived from
  prefixes plus the retained dispatch payload. The staged epoch barrier removes
  per-copy descriptor, ready, route-sidecar, and return-header requirements.
- First prove that a matching combine can consume immutable route state from
  its non-cached dispatch. Extend cached replay only after fresh-generation
  child-handle semantics are defined; cached dispatch must replay the original
  decision and must not rerun `auto` planning.
- Treat expanded, cached, deterministic, masked, alignment, and
  multiple-reduction matrices as incremental semantic gates rather than
  prerequisites for the first isolated Hybrid code-generation proof.
- Exercise dispatch/combine symmetry with explicit replay tests. This repository
  does not provide an `autograd.Function`; backward coverage is an external
  integration contract and must not be inferred from `test_ep.py`.

Exit gate: minimum non-cached BF16 dispatch/combine route replay passes; every
unsupported semantic combination fails collectively before publication. Full
semantic coverage remains required before production enablement.

### WP7 — Optional Hybrid integration

- Add an internal JIT specialization and a compatible Python control surface,
  initially constructor-fixed `off|force`; enable `auto` only after the cost
  model is calibrated.
- Extend workspace/buffer sizing and JIT keys without stealing bytes from
  existing counters with incompatible lifetimes.
- Reserve proxy payload space at buffer construction and update both dispatch
  and combine buffer-size calculations; workspace bytes alone are insufficient.
- Keep destination forwarding changes limited to added route metadata.
- Keep the original implementation bit-for-bit selectable.

Exit gate: existing regression coverage passes with default `off`; forced mode
passes virtual-hybrid correctness, fail-closed capacity tests, and isolated
Hybrid code generation.

The detailed C080 integration sequence, supported matrix, evidence labels, and
stop conditions live in `C080_PLAN.md`. It is rereviewed before source edits.
Fallback remains future `auto` work.

### WP8 — Validation and performance evidence

- Run memcheck, synccheck, initcheck, and a focused racecheck suite on small
  shapes.
- When GPUs are shared: run only safe correctness tests; do not interpret timing.
- When GPUs are idle/available for exclusive measurement: benchmark planner,
  direct-final source shuffle, transfer matrices, segment/publication sizes, and
  compute overlap on the local eight-GPU machine.
- On a real multi-node rail cluster: validate Gin/QP/NIC behavior, RDMA tails,
  signals, fabric effects, and end-to-end speedup.

Exit gate: correctness and sanitizer results are clean; performance conclusions
are tagged with the environment in which they were obtained.

Before real networking, package one-command build/JIT warmup, off/force A/B
correctness, stable JSON protocol/traffic counters, canonical skew/capacity
cases, and explicit evidence labels. The active Goal includes exhaustive local
validation and locally measurable operator optimization; it does not stop at
the first Hybrid code-generation pass.

### WP9 — Production fusion and optimization

- Fold destination-server counting into existing notify scanning.
- Reuse completed notify warps for shuffle production instead of increasing the
  permanent warp count.
- Let existing scaleout warps consume retained and proxy-ready work.
- Fuse combine return-unshuffle only after the standalone implementation is
  correct and measurable.
- Update SM/QP/traffic estimates and the `auto` bypass model using measured
  bytes and timings.

Exit gate: fused code preserves correctness, improves or matches the standalone
prototype in target skewed cases, and bypass overhead is acceptable on balanced
cases.

## Correctness invariants

For destination `d`, owner rail `g`, original count `c[g,d]`, and quota
`q[g,d]`:

```text
sum_g q[g,d] == sum_g c[g,d]
max_g q[g,d] - min_g q[g,d] <= 1          # exact-copy policy
moved[d] == sum_g max(c[g,d] - q[g,d], 0)
each input destination-copy is retained or moved exactly once
each proxy slot is unique within the staged invocation
each return route resolves to exactly one original owner/token
retained and moved copies never collide in a remote slot
per-channel assigned copies never exceed the current channel capacity
an already exact-balanced count vector produces quota == count and moved == 0
```

Every bypass must also satisfy identity semantics: no moved payload, no proxy
route dependency, and the same externally visible output as the original path.

## Decision gates

- If per-token destination deduplication cannot be represented without breaking
  the existing send-buffer reuse contract, stop before data-plane integration
  and revise the planner input model.
- If direct-final peer writes cannot meet correctness or ordering requirements,
  retain the two-stage buffer only as a diagnostic baseline and record the cost.
- If bounded split cannot materially approach exact balance at controlled copy
  amplification, keep token-primary as the production candidate.
- If source shuffle plus destination forwarding creates a larger local bottleneck
  than the projected RDMA tail saving, `auto` must bypass that workload.
- If any GPU cannot form the same deterministic node-wide plan before payload
  publication, fail the experimental `force` call collectively. A later `auto`
  mode may bypass the entire call; per-GPU disagreement is never a valid
  partial plan.

## Checkpoint discipline

- Every completed work package or material design decision receives a checkpoint
  entry in `CHECKPOINTS.md`.
- `DEVELOPMENT_LOG.md` records commands, results, failures, and next actions.
- `OPTIMIZATION_LOG.md` records hypotheses, metrics, negative results, and
  retained/rejected variants.
- Local commits are made at coherent, reviewable checkpoints; experimental
  failures remain documented even if their code is later removed.
