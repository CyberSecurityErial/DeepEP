# RailBalance Hop-Aware Implementation Audit

Audit base: `5199a04bea86cae03f52e1316672d7b092b62dd5`  
Audit branch: `feat/rail-balance-hop-aware`  
Scope: read-only audit before changing planner or CUDA data paths.

## Executive result

The prototype already has the difficult transport machinery: deterministic static proxy slots, source-side LSA/TMA shuffle, egress-side Rail/Gin send, destination forwarding, inverse combine return, one-shot tickets, vnode coverage, and fail-closed public capability gates.

The missing fact is the destination endpoint at **planning time**. The current planner balances one deduplicated `(owner, destination node)` payload over arbitrary local Rails. It therefore implements `legacy_exact`, not endpoint-aware one-hop.

There is one important refinement to the proposed `(o,d,t)` input model. DeepEP sends one hidden payload per `(token, destination node)`, while that payload can target several local expert ranks on the destination node. Splitting it into one network copy per `t` would destroy the existing payload reuse and can multiply RDMA bytes. The exact transport identity is therefore:

```text
(o, d, T)

o = source owner local rank
d = destination node
T = non-empty set of destination local ranks reached by this payload
e = selected egress/ingress Rail
```

For the common single-target case, `T={t}` and this reduces exactly to the requested `(o,d,t,e)` semantics. The implementation must retain `T` (a small rank bit mask on the current supported topology) and may expose `count[o][d][t]` as a diagnostic projection, but it must not create separate scale-out copies solely to obtain that tensor.

This is the only material mismatch between the supplied design and the current DeepEP transport unit. It changes the planner record, not the project goal.

## 1. Current planner input and output

### Python reference

`tests/elastic/rail_balance_hybrid_reference.py` currently:

- validates full expert IDs, then maps every expert to `destination = expert // experts_per_scaleout`;
- converts the lanes of one token to a set of remote destination nodes;
- drops destination local-rank identity before quota construction;
- counts `[owner][channel][destination]` and reduces it to `[owner][destination]`;
- creates at most `G-1` five-int segments per destination:

```text
(owner, egress, owner_begin, count, egress_begin)
```

The relevant structures are `HybridTransferSegment` and `HybridRailSchedule` at lines 60-114. The endpoint loss occurs in `map_topk_experts_to_destinations()` at lines 242-270. `_build_quota_and_segments()` at lines 310-417 balances every destination column across policy-selected Rails without access to any `t` or target set.

### GPU planner

`deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh` mirrors the reference:

- `rail_balance_hybrid_count_impl` derives only the destination node and emits one bit/copy per token and destination (`243-292`);
- the compact snapshot is `[G,C,D]` (`296-299`);
- `resolve_hybrid_copy()` resolves a deduplicated `(owner, source channel, destination)` copy and returns an arbitrary planned egress (`77-163`);
- the output segment retains no target local rank or path kind.

The C++ launcher in `csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp` owns the same fourteen destination-only plan tensors. There is no per-copy global atomic or manifest in the hot path.

### Correct mode name

The current `all`, `active`, and `adaptive` policies only control which Rails participate in the destination-column quota. They do not impose an endpoint constraint. A moved copy may select `e != o` and `e` may be unrelated to every target local rank.

Therefore the current force planner is accurately named `legacy_exact` in the new mode model. It must not be renamed to `one_hop`.

## 2. Where destination local rank is lost

Destination rank information is not destroyed from the payload. It is lost only at the planner boundary:

```text
full top-k expert IDs
  -> deduplicate by destination node
  -> planner sees (o,d), not target rank(s)
  -> Rail e is selected
  -> full top-k expert IDs still travel with the payload
  -> destination forwarder reconstructs target rank(s)
```

The source shuffle copies the original top-k IDs and weights into `TokenLayout` (`rail_balance_hybrid_shuffle.cuh:140-149`). The destination forwarder later derives:

```cpp
dst_local_rank = local_expert / experts_per_rank;
```

and deduplicates target local ranks before forwarding (`rail_balance_hybrid_dispatch.cuh:845-899`). It also records each target local rank and destination slot for combine replay (`903-927`).

Consequently, the smallest correct extension is to retain the target-rank mask next to the planner's transport copy. Replacing the existing destination-node copy with independent `(o,d,t)` network copies would be a semantic and bandwidth regression.

## 3. Exact hop semantics for the real payload unit

For `T={t}`, the supplied definitions remain unchanged:

```text
e=o=t       direct, 0 local forwards
e=o!=t      destination-forward, 1 local forward
e=t!=o      source-forward, 1 local forward
e not o,t   two-hop, 2 local forwards
```

For a payload with several target ranks, the exact number of cross-GPU local forwards is:

```text
source_forwards      = [e != o]
destination_forwards = |T - {e}|
total_local_forwards = [e != o] + |T - {e}|
```

The endpoint-aware one-hop candidate set generalizes without adding network copies:

```text
E_endpoint(o,T) = {o} union T
```

Choosing `e=o` is the owner endpoint path. Choosing `e in T, e!=o` is a target endpoint path and removes one destination cross-GPU forwarding operation. Choosing `e not in ({o} union T)` is third-Rail/two-sided forwarding. In tests where `|T|=1`, path names and hop counts are exactly the manual's direct/destination-forward/source-forward/two-hop classes.

The new reference should make the single-target contract primary and retain at least one multi-target regression so that optimization never duplicates the shared payload accidentally.

## 4. Current source shuffle and destination forwarding

### Retained path (`e=o`)

`rail_balance_hybrid_shuffle.cuh` resolves each deduplicated destination copy. A retained copy is packed into the owner Rail's local staging area (`182-185`, `230-267`) and sent by that Rail's persistent scale-out warp.

At the destination, the ingress GPU reads the original top-k expert IDs, derives every final target local rank, allocates/deduplicates its destination slots, and TMA-stores the payload through LSA to those ranks (`rail_balance_hybrid_dispatch.cuh:845-899`).

### Moved path (`e!=o`)

The source owner writes the complete token record directly into egress `e`'s final proxy-dispatch slot using LSA/TMA. It waits for TMA completion, performs a global visibility fence, then publishes an invocation-specific ready word with system release semantics (`rail_balance_hybrid_shuffle.cuh:269-304`).

The egress persistent scale-out warp waits for that ready word, stages the record, and issues Gin on its own Rail (`rail_balance_hybrid_dispatch.cuh:601-660`). No per-token proxy-tail atomic is used; slot identity comes from the compact static plan.

### Destination behavior and the one-hop gap

The destination forwarder currently handles every received record through the same forwarding loop. When `e` is one of the final target ranks, its LSA target for that rank is self, so no cross-GPU transfer is required for that target. However, the record still incurs the general forwarding control/metadata path.

True `source-forward` therefore needs a small semantic flag or target-mask-aware branch so the already-final target can use the direct receive layout without a duplicate forwarding operation. It does **not** require another transport stack.

Third-Rail records already have the required physical shape: source shuffle to `e`, Gin on `e`, then destination forwarding from ingress `e` to every rank in `T`.

## 5. Dispatch/combine ticket and route lifetime

The current one-shot protocol is reusable:

1. Python validates constructor-fixed mode/config and reserves a monotonically increasing invocation ID.
2. Dispatch performs WORLD prepare/plan agreement. If no rank has moves, it aborts the temporary plan and runs the native off path.
3. A moved dispatch commits the C++ pending transaction, launches the force path, and attaches one `_RailBalanceForceTicket` to the returned `EPHandle`.
4. Destination forwarding records source token, last-token marker, proxy slot, destination local ranks, and destination slots in `token_metadata_at_forward`.
5. Combine requires the same live ticket. It replays the dispatch metadata rather than running a second planner.
6. A moved result is returned over ingress Rail `e` into source proxy-return slot `p` (`rail_balance_hybrid_combine.cuh:420-458,585-630`).
7. Return-unshuffle reads proxy-dispatch slot `p` to recover the original source token/owner and TMA-copies the result to the original reduce row.
8. Successful combine consumes the ticket exactly once; recoverable pre-commit failure restores it to live.

C++ plan state progresses `Preparing -> PlanReady -> DispatchLive`; owned plan tensors and prepared runtimes remain alive until abort or combine completion. This lifecycle is sufficient for inverse hop-aware combine. The first implementation should reuse the dispatch plan/ticket and must not add an independent combine planner.

## 6. Capability and fail-closed behavior

Public availability is currently dual-gated:

- Python `_RAIL_BALANCE_FORCE_HOST_AVAILABLE` is false;
- compiled `_rail_balance_force_available()` is false;
- `_rail_balance_force_available()` returns true only if both gates are explicitly true.

Ordinary public `force` construction therefore fails closed. The real-D benchmark runners temporarily replace both private gates inside their worker process to exercise the experimental implementation, assert and restore the original false values, and label results according to environment eligibility. This is a deliberate test-only exception, not an enabled production capability.

Hop-aware modes must remain behind the same false production capability until truthful multi-node Rail/Gin correctness is complete. Vnode success must not change those bits.

## 7. Reusable test and benchmark components

The following should be extended rather than copied:

- `rail_balance_hybrid_reference.py`: compact deterministic schedule and static-slot resolution;
- `rail_balance_hybrid_vnode_reference.py`: full top-k endpoint knowledge, source/proxy/ingress/return identities, synthetic expert round trip;
- `rail_balance_validation_common.py`: deterministic workload generation, remote-copy oracle, JSON-safe result skeleton;
- `run_rail_balance_hybrid_multinode.py`: launcher checks, environment identity, watchdog, official-reference correctness, atomic result handling;
- `bench_rail_balance_hybrid_multinode.py`: correctness outside timing, rank-max statistics, ABBA/BAAB paired blocks, off baseline, profiler/contended-run labeling.

The unified `bench_rail_balance_hop.py` should be a thin frontend over these pieces. It should not reproduce their process management or benchmark statistics.

The current vnode route already retains `expert_physicals` and `topk_lanes` for every destination-node payload (`rail_balance_hybrid_vnode_reference.py:97-118,407-476`). Those fields can directly derive `T` and detect an accidental destination double-forward.

## 8. Assumptions from the task that differ from the tree

1. **Planner atom:** the tree sends a deduplicated token-to-node payload, not necessarily one payload per destination local rank. The design must retain target set `T` to preserve RDMA reuse.
2. **Existing source-forward:** choosing `e=t` already lands on the target GPU physically, but it is not yet a distinct direct-receive path; the generic destination forwarder still processes it.
3. **Existing adaptive:** the current `adaptive` policy selects how many arbitrary Rails join a quota. It is unrelated to the requested “one-hop first, then selective 2-hop” algorithm.
4. **Benchmark modes:** current public/test interfaces expose only `off` and `force`; the new semantic modes do not exist yet.
5. **Claim scope:** the tree has real-D launchers and paired benchmark machinery, but the production capability is intentionally false and the checkpoint remains `LOCAL_COMPLETE / MULTINODE_PENDING`.
6. **Baseline debt:** the prototype's existing logs record local optimization evidence and deferred network evidence. Hop-aware work must not reinterpret those results as evidence for the new path selection.

## 9. Minimal implementation boundary after this audit

The next work should remain deliberately small:

1. Add a pure Python endpoint-aware oracle whose record is `(o,d,target_mask,count)` and whose single-target behavior is exactly `(o,d,t,e)`.
2. Add C0-C7 plus a multi-target payload-reuse regression before touching CUDA.
3. Extend compact GPU plan metadata with target mask and selected egress/path diagnostics while leaving `off` and `legacy_exact` byte-identical.
4. Reuse the existing source proxy for `e!=o`; add only the destination direct-receive distinction needed when `e in T`.
5. Reuse the existing ticket and inverse proxy slot for combine.
6. Add selective third-Rail moves only after endpoint-only vnode round trip passes.
7. Profile and tune only after profiler-free correctness and timing boundaries exist.

No new descriptor ring, fallback planner, second combine planner, or duplicated benchmark harness is justified by the current code.
