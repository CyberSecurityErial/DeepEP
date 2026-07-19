# Rail Balance Optimization Log

This log separates performance hypotheses from established results. A hypothesis
is not promoted to a conclusion until the relevant environment is idle and the
measurement protocol is recorded.

## Measurement policy

- Correctness can be tested on shared GPUs when memory safely fits.
- Shared-GPU timing is diagnostic and is tagged `CONTENDED`; it cannot accept or
  reject an optimization.
- Local performance results require an idle/controlled eight-GPU window and are
  tagged `LOCAL_CONTROLLED`.
- GIN/QP/NIC/fabric and end-to-end multi-node results require the target rail
  cluster and are tagged `CLUSTER_CONTROLLED`.
- Every comparison reports shape, dtype, topology mode, policy, moved bytes,
  segment count, warmup/iteration count, and observed GPU occupancy state.

## Initial hypotheses

| ID | Hypothesis | Evidence required | Status |
| --- | --- | --- | --- |
| H01 | A count-aware exact quota minimizes moved destination copies; count-blind remainder rotation does not. | CPU proof/invariants and exhaustive enumeration of all base/base+1 quota choices. | VALIDATED_CPU |
| H02 | Static plan slots outperform peer-hotspot atomics for one-hot destination skew. | Controlled local microbenchmark. | UNVALIDATED |
| H03 | Owner-to-final-proxy writes materially outperform temporary-buffer plus egress-copy staging. | Controlled local TMA/LSA comparison with equal bytes. | UNVALIDATED |
| H04 | Bounded split with at most two egress GPUs per token captures most exact-balance tail reduction without large payload duplication. | Synthetic and real routing Pareto curves. | UNVALIDATED |
| H05 | Batched ready publication improves throughput without harming liveness or wraparound correctness. | Protocol tests, then controlled batch-size sweep. | UNVALIDATED |
| H06 | Existing notify warps can perform plan/shuffle production without increasing permanent warp count. | Register/occupancy profile and fused-vs-standalone comparison. | UNVALIDATED |
| H07 | Balanced inputs can bypass with only small overhead. | Default-off identity plus controlled balanced-case timing. | UNVALIDATED |
| H08 | Source shuffle cost is justified only when projected RDMA tail saving exceeds local move cost plus fixed synchronization. | Local move measurements plus cluster RDMA measurements. | UNVALIDATED |

## Candidate policies

## Planner finding O001 — remainder placement is part of the optimization

For `counts = [4, 0, 0]`, `G = 3`, the total is 4, so every exact quota is a
permutation of `[2, 1, 1]`. Assigning the extra unit to the hot rail moves two
copies; assigning it to an empty rail moves three. Therefore a purely rotating,
count-blind remainder rule is not move-minimal.

The marginal retained-copy gain of raising rail `g` from `base` to `base + 1`
is `1[count[g] > base]`. The oracle therefore prioritizes those rails and uses
a canonical destination/seed-derived ring order only as a deterministic
tie-break. This also guarantees that an already balanced vector is an identity
plan.

Validation on 2026-07-19 enumerated all quota winner sets for six remainder
seeds, one through five rails, and per-rail counts zero through three (8,184
count-vector/seed cases). The count-aware oracle matched the minimum moved-copy
count in every case. Sixty-four seeded multi-destination route/slot cases also
preserved the committed quota and unified-slot uniqueness. This validates the
CPU scheduling result only; it does not yet establish GPU planner cost or
end-to-end performance.

## Planner finding O002 — balance within each rail's channels

A first-nonfull channel policy does not stripe a segment while the preferred
channel has spare capacity. In the review counterexample, four proxy copies and
four high-capacity channels all selected channel 0. The CPU contract now selects
the lowest current occupancy for `(egress, destination)`, with the canonical
ring order as a deterministic tie-break, producing one copy per channel in that
case.

This is a correctness-of-plan/layout decision, not a measured throughput claim.
The expected QP/channel utilization benefit remains unvalidated until the local
controlled performance checkpoint.

## Planner hardening note O003 — extreme reduction domain

The C030 host preflight converts the count matrix to `int64` and reduces it
before enforcing the public planner bound `total_copies <= INT_MAX`. An input
with more than roughly 4.3 billion maximum-valued `int32` elements could
theoretically overflow that intermediate sum. Such an input needs on the order
of 120 GiB once source and converted tensors are both resident, while the
intended planner domain has at most tens to hundreds of destinations. This is a
retained Low hardening item, not evidence against the current algorithm or a
performance result.

## Data-plane decision O004 — separate policy budget from physical arena

The C040 contract distinguishes a global moved-copy budget `B` from the
per-egress physical capacity `P`. `B` is an enable/bypass policy and consumes no
memory by itself. `P` reserves exactly `P` compact moved-only records in every
GPU's symmetric arena. An egress's compact slot is already the final
Gin-readable source slot; the descriptor carries the unified logical remote
slot, so no temporary egress HBM copy is required.

This decision is validated for correctness by the 28-test CPU contract and two
8-GPU LSA/TMA functional runs. It is not yet a performance conclusion. The
direct-final advantage over two-stage staging remains hypothesis H03 until a
controlled local comparison is run.

## Correctness finding O005 — exact capacity must exercise every owner

An 8-process test is not evidence for eight concurrent producers. The first
passing fixture moved 64 records but only owners 0 and 1 launched producer
kernels, and no egress came close to its allocation boundary. The final C040
fixture makes owner participation and exact per-egress capacity explicit
invariants: every owner sends 32 records and every egress consumes all 32
slots. It also uses 14,528-byte records, checks one unused headroom slot in a
separate valid run, and verifies a caller-side capacity bypass with poisoned
payload memory. Future data-plane checkpoints must report producer and
consumer distributions, not only world size and total records.

### Exact copy-level

- Establishes the per-destination balance upper bound.
- May duplicate one token payload across several egress GPUs.
- Prototype/reference policy, not presumed to be the production winner.

### Token-primary

- One egress for all remote destinations of a token.
- Preserves send-buffer reuse and minimizes staging duplication.
- Does not guarantee exact per-destination balance.

### Bounded split

- One primary egress and at most one secondary egress per token.
- Current leading production candidate.
- Must be evaluated on balance, moved bytes, duplicated bytes, proxy capacity,
  and segment count together.

## Required counters

```text
original_bytes[rail][dst]
balanced_bytes[rail][dst]
quota[rail][dst]
moved_copies
moved_bytes
duplicated_bytes
move_ratio
payload_duplication_factor
segment_count
max_segments_per_dst
proxy_capacity
proxy_high_watermark
plan_cycles
shuffle_cycles
publish_count
producer_wait_cycles
consumer_idle_cycles
return_unshuffle_cycles
bypass_reason
environment_tag
```

## Negative-result template

When a variant fails or regresses, append:

```text
Variant:
Environment/shape:
Expected mechanism:
Observed result:
Correctness status:
Likely cause:
Evidence/log path:
Decision: retain / revise / reject
Reusable lesson:
```

## Protocol decision O006 — publish only a contiguous generation tail

Per-slot readiness remains an exact generation value. A designated scanner
acquire-loads slots in physical order and release-publishes a packed
`(generation, exclusive_tail)` only after the complete prefix is ready. The
downstream consumer acquire-loads that tail, which creates the transitive
visibility chain:

```text
producer payload write
→ producer release ready/sequence
→ scanner acquire ready/sequence
→ scanner release contiguous tail
→ consumer acquire tail
→ consumer payload read
```

Per-producer `atomicAdd(tail, 1)`, `atomicMax`, boolean readiness, and
`ready >= generation` are rejected: each can publish across a real hole or
accept stale/future state. C050 is intentionally a one-shot protocol with a
positive non-wrapping signed-int32 generation. Ring reuse, ABA prevention, and
backpressure are not claimed by this checkpoint.

## Protocol decision O007 — schedule dependency classes, not mixed producer work

The first producer schedule used one launch, then two launches. Both retained a
legal residency deadlock: CTAs waiting for a hole or consumed tail could occupy
the machine while CTAs that create that progress remained queued. The accepted
one-shot schedule has three dependency classes launched serially on the same
producer stream:

```text
phase 0: no protocol wait
phase 1: waits for a scanner-observed hole, never for consumption
phase 2: waits for downstream consumed_tail
```

This is a liveness decision, not a throughput conclusion. Production fusion may
replace kernel boundaries only if it preserves the same partial order without
adding permanently resident warps.

Timeouts are progress leases rather than wall-clock deadlines. A lease renews
only when a finite monotonic state strictly advances: contiguous published
tail, consumed tail, or a newly observed hole state. A fixed absolute deadline
rejects legitimate slow pipelines; renewing on identical observations hides
deadlocks. The C050 regression makes this distinction executable by requiring
eight consumer steps whose aggregate service time exceeds one timeout while
every individual step remains below it.

The protocol uses system-scope publication and error arbitration for LSA peer
pointers. This correctness requirement is independent of whether a local-only
microbenchmark appears to work with device-scope atomics. Batching publication
remains hypothesis H05 and is not inferred from C050.

## Architecture decision O008 — finite vnode stages before Hybrid fusion

C060 uses seven prepared cubins and explicit all-rank barriers rather than
forcing the existing Hybrid kernels to impersonate a 2x4 physical topology.
The live NCCL context is pure single-node (`scaleout=1`, `scaleup=8`); Hybrid
barrier teams, Rail Gin teams, and rank decoding would be false under a virtual
split. The accepted validation pipeline is therefore:

```text
full retained+moved pack
→ source egress to paired destination ingress
→ one contribution per top-k lane to its expert GPU
→ weighted synthetic expert result back to its ingress
→ same-rail return to source egress
→ owner partial LSA unshuffle
→ fixed-order local FP32 reduction and BF16 cast
```

All cubins are compiled before the first barrier, all eight ranks execute the
same barrier count, and role-specific arenas overlap only on physically
disjoint source/destination ranks. This increases launch and synchronization
cost and is not a production performance candidate. It is intentionally the
smallest structure that can prove payload, route, combine, and original-owner
semantics before persistent fusion.

## Scope decision O009 — PoC exit before exhaustive local tuning

The immediate objective is to decide whether rail balancing works, not to
finish every production tuning sweep before Hybrid integration. The shortened
single-node decision path is:

```text
C061 per-destination correctness
→ C070 minimum replay/combine symmetry
→ C080 optional Hybrid integration
→ critical C090 memory-order/sanitizer gates
→ narrow C100 planner + source-shuffle viability measurement
→ C110 real Gin/RDMA validation
```

The narrow C100 gate retains production-relevant planner, static-slot,
direct-final LSA, moved-byte, and peer-copy measurements plus an RDMA bandwidth
sweep. Exhaustive segment/QP/GEMM-overlap tuning is deferred until the narrow
gate finds a credible positive region. Local LSA absolute timings are never
substituted for real Gin/RDMA evidence.

## Architecture decision O010 — preserve destination namespaces physically

C061 does not compact all destinations into one aggregate egress prefix. Each
destination owns a fixed six-slot segment and only its quota-sized prefix is
valid. This leaves deliberate holes but makes three properties executable:

- an aggregate-balanced 17/16 matrix still performs six required moves;
- a stale or unexpected record in a quota hole is an error, not useful work;
- destination identity is recoverable from the physical base slot before any
  route sidecar is trusted.

Expert storage remains compact per physical destination rank: its slot uses
`(source_egress, destination-local slot, lane)`, while `VNodeRoute` retains the
global ingress slot. Return contributions use the global base namespace so the
source egress can distinguish multiple copies of the same original token.

## Architecture decision O011 — replay route and descriptor together

The minimum C070 proof will test cross-call route replay rather than expanding
the public EPHandle or entering Hybrid prematurely. A 32-byte `VNodeRoute`
does not contain the original owner and token; those fields live in the
record's 64-byte `ProxyDescriptor`. A valid replay snapshot must therefore
retain both the route sidecar and the record descriptor.

The proposed private replay clears the symmetric arena, restores owning local
snapshots, and executes only `expert -> return -> unshuffle -> reduce`. It must
not accept the original manifest or fingerprints and must not rerun the
planner. Replaying the same snapshot twice, then replaying changed expert
payload with unchanged route metadata, is the smallest evidence that combine
can consume persisted routing state across calls.

This full-record snapshot is intentionally a PoC artifact, not a production
handle design. Compact metadata, public cached/expanded/backward semantics,
and the Hybrid specialization remain C080 work.

## Reliability decision O012 — collective preflight before exposure

The C070 private replay is accepted only for owning snapshots produced by the
same verified C061 call. Its host method performs shape, dtype, generation,
quota, descriptor, and route checks before the first device collective. That
is useful fail-closed validation, but a failure on only one rank can throw
while valid peers proceed to the first barrier.

C080 must not copy this trust assumption into a public `rail_balance` path.
All rank-local validation results and geometry must first participate in a
bounded cross-rank consensus (or a common abort protocol), and no rank may
enter a replay/Hybrid barrier unless every peer agreed to proceed. The C070
route corruption test intentionally mutates a field that passes safe host
bounds and is then rejected by the device protocol; it does not replace this
host-preflight consensus requirement.
