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

## Architecture decision O013 — isolate force from the old Hybrid JIT graph

The first Hybrid integration will not add an enable template parameter to the
existing dispatch/combine kernels and will not modify their transitive layout
header. The compiler cache key includes generated code plus recursive DeepEP
include hashes, and each `LaunchRuntime` derived type caches one include hash.
Consequently, even a compile-time false branch in an edited old header would
change the default path's cache identity.

C080 will use independent rail-balance layout, dispatch, combine, launcher, and
runtime files. `off` continues to call the original runtime directly. The
immutable set is the complete recursive include closure, not only the two root
kernels and `layout.cuh`; it also includes `combine_utils.cuh` and common
comm/handle/exception/ptx/compiled/math headers. The force-only registered
arena is appended after the aligned legacy maximum; `WorkspaceLayout` is not
expanded. This makes unchanged default buffer bytes, old JIT source/hash, old
kernel arguments, and old result semantics executable acceptance criteria
rather than assumptions.

Identity tooling must also avoid mutating the state it measures. The old
`LaunchRuntime` stores one static include hash per derived class, so a process
that first calls `generate()` for a synthetic Hybrid geometry can poison later
direct generation. Baselines use `generate_impl()` with explicit include
parsing, or one-mode subprocesses that exit immediately.

## Architecture decision O014 — materialize production assignments on GPU

C030 proves quota and segment planning, but C040-C070 receive a per-copy
manifest built by the test oracle. Hybrid dispatch starts with `topk_idx`, so a
production-shaped path still needs to deduplicate destinations, count by source
channel, assign a stable owner ordinal, and materialize static egress/channel/
remote/proxy slots.

C080 makes this a separate gate before persistent-kernel work. The ordinal is
channel-major because it matches the current scaleout warp traversal. Retained
and moved copies share the destination forwarder's dense remote prefix; moved
copies occupy only the remaining per-channel capacity and use compact proxy
slots. No peer-hotspot per-copy allocation atomic is allowed.

## Reliability decision O015 — force is fail-closed before publication

The experimental API is constructor-fixed `off|force`. Construction performs a
bounded configuration consensus before symmetric-window creation. Each force
invocation then has two collective phases: prepare/JIT agreement, followed by
plan-status/capacity commit. The planner is not allowed to publish payload or
remote tail state.

`force` is an observability mode: when a move is required it must run the new
path. Insufficient capacity or an unsupported semantic combination produces the
same bounded error on every rank before publication. Silent fallback and
partial balancing are reserved for future `auto`; otherwise a test could pass
without proving that rail balancing executed.

## Evidence decision O016 — vnode, codegen, and real Hybrid are distinct

The current 8xH200 node has a real `scaleout=1` logical topology. It can validate
the GPU materializer, LSA source shuffle, proxy protocol, route persistence,
and return-unshuffle, and it can compile synthetic 2x4/4x2 Hybrid cubins. It
cannot truthfully execute Rail Gin teams.

C080 local results must use separate labels:

```text
DEFAULT_OFF_REGRESSION_PASS
VNODE_FUNCTIONAL_PASS
HYBRID_CODEGEN_PASS
REAL_HYBRID_RUNTIME_UNTESTED
```

Only a real multi-node run can replace the last label with a runtime pass. A
single-node launcher must reject public force rather than invent scaleout ranks.

## Rejected candidate O017 — preserve TokenLayout with a slot sidecar

The initial plan proposed a force-specific TokenLayout containing two proxy
fields. Independent review found that this would require a complete parallel
dispatch buffer calculator and copy epilogue: the additional bytes cross the
32-byte component and 128-byte metadata boundaries for some top-k values, so
they are not reliably free padding.

C080 still keeps the legacy transmitted TokenLayout byte-identical. The first
corrected draft proposed one registered same-slot 32-byte route sidecar per
remote copy. It was semantically valid, but a second hot-path review rejected
it before implementation: it adds wire bytes, a second Gin request, registered
send/receive arrays, more flush/QP pressure, and a return-header staging problem
for data that the restricted staged path can derive.

The failed draft remains recorded because it is a valid fallback if future
cached/ring overlap removes the free metadata lifetime used by O020.

## Superseded reliability draft O018 — full slot-generation state machine

The sidecar/ready draft proposed instance cookies, arena epochs, uint64 slot
generations, and a multi-state publication machine. Those mechanisms are
appropriate for a reusable overlapping ring but redundant for force-v1's
one-shot staged epoch. O021 replaces them with one live-handle id, two barriers,
and one aggregate device status. The public capability still remains disabled
until dispatch and combine both exist.

## Validation decision O019 — exhaust and optimize locally before networking

The active Goal includes C090 exhaustive correctness/fault/sanitizer coverage,
C100 controlled local performance optimization, and C105 network-ready scripts
and metrics after C080 closes locally. Shared-core planner, assignment, LSA
shuffle, publication, and unshuffle decisions will be selected from idle
8xH200 measurements. Gin-specific fusion is not selected from vnode timing;
the local output is instead compiled and instrumented so real-network testing
can begin with A/B correctness and useful counters immediately.

## Hot-path decision O020 — reuse four transit bytes, not a network sidecar

Source Hybrid dispatch defines top-k, weights, and src_token_global_idx but does
not assign business meaning to the transported linked-list array. Destination
forwarding later overwrites every linked-list entry before the existing copy
epilogue reads it. Two independent source audits verified this exact lifetime.

Restricted non-cached force-v1 therefore stores compact proxy slot p in
linked_list_idx[0] only for moved copies. The destination forwarder derives
moved state by comparing the original owner local rank from src_token_global_idx
with its current ingress local rank, snapshots p before the overwrite, and adds
one int to force-only forward metadata.

Compared with O017 this removes, per moved copy:

    32 route bytes on the network
    one Gin request
    one registered send sidecar slot
    one registered receive sidecar slot
    one return header and its registered staging

The TokenLayout stride and legacy copy epilogue remain unchanged. The decision
is valid only for non-cached force-v1; cached replay may require another design.

## Hot-path decision O021 — grouped static epoch removes per-copy protocol

Proxy slot p is contiguous within an (egress, channel, destination) group.
Group prefix/count implies channel, destination, remote slot, and the egress
that owns p. The retained dispatch payload kept through combine contains
src_token_global_idx, so unshuffle also derives owner and token without a route
table. Force-v1 restricts num_scaleout_ranks<=num_topk with multiple reduction,
making the final reduce row equal to destination.

The staged schedule is:

    source shuffle and TMA completion
    -> LSA barrier
    -> Hybrid dispatch/forward
    -> Hybrid combine and Gin completion
    -> return unshuffle and TMA completion
    -> LSA barrier
    -> legacy reduce epilogue

Because every producer completes before its consumer epoch starts, the first
force path needs no per-slot descriptor, ready flag, generation, or return
header. It reserves only dispatch/return payload arenas plus O(G*C*D) count and
prefix state. If later measurement requires producer/consumer overlap, C050's
generation protocol remains a tested candidate rather than being paid
unconditionally now.

## Hot-path decision O022 — reuse Hybrid dispatch Tag0 as the epoch barrier

The first minimal draft placed a standalone scaleup barrier after source
shuffle and then entered force Hybrid dispatch. Source audit showed that the
legacy Hybrid dispatch already begins with Tag0, a combined scaleup/scaleout
release-acquire barrier. With source TMA stores waited before shuffle-kernel
completion and both kernels ordered on the same comm stream, Tag0 is the needed
proxy epoch boundary. A separate barrier would synchronize the node twice and
was removed before implementation.

The post-unshuffle scaleup barrier is not redundant. The legacy reduce epilogue
only has a same-GPU programmatic dependency wait and cannot observe completion
of peer egress writes by itself.

## Reliability decision O023 — allocate combine outputs before commit

Legacy host flow allocates final combine outputs after launching the main
combine. Force-v1 inserts peer return-unshuffle plus a local-team barrier, so an
OOM in that gap could strand other local GPUs. Force combine therefore catches
and world-consenses handle/output allocation before launching its uninterrupted
combine -> unshuffle -> barrier -> epilogue sequence. This changes only the
correctness-first force control path; default off retains legacy overlap.

## ABI decision O024 — inherit the 1024-channel legacy ceiling

The generic host heuristic can select up to 1280 channels, but the immutable
Hybrid workspace arrays are compiled for 1024. Enlarging the new count arena
does not make the old tails/counters safe. Force-v1 therefore uses 1024 as a
hard preflight limit instead of enlarging or rewriting legacy workspace. This
keeps default-off identity and prevents a hidden out-of-bounds path.

## Hot-path decision O025 — destination-bucketed five-int segments

A correct CPU draft flattened destination into every transfer segment. That
would spend one redundant integer per segment and diverge from the frozen GPU
shape. Destination is now implied by the outer bucket:

    segments[d][s] = owner, egress, owner_begin, count, egress_begin

The production resolver scans only one destination bucket. No per-copy
assignment is materialized. Strict expert validation and server mapping stay
outside the hot resolver, before the first collective.

## Hot-path decision O026 — materialize compact prefixes, not copy assignments

C080-B1 proves the production schedule can be generated entirely on device as
three small stages: deduplicated channel counts, destination quota/segments,
and inverse grouped prefixes. The data path resolves one copy with a scan of at
most `G-1` segments and the target channel prefix; it performs no global atomic
and reads no full copy manifest.

The first implementation intentionally keeps the per-destination plan serial
and the ABI explicit. This is the smallest correctness baseline and avoids
optimizing metadata work before source-shuffle cost is known. C100 must profile
with NCU/Nsys on idle GPUs before parallelizing/fusing it. Negative or noisy
profiling results, register/spill changes, launch gaps, and rejected variants
remain part of this log rather than being discarded.

## Hot-path decision O027 — one local barrier and G prefix copies for B2

The first production-shaped count snapshot reuses NCCL symmetric memory and the
existing barrier algorithm but not the actual Hybrid topology. A force-only
`(1,G)` specialization avoids an unnecessary Rail barrier. After it completes,
G `cudaMemcpyAsync` D2D copies gather only active `C*D*sizeof(int)` prefixes.

This deliberately favors a transparent correctness baseline over an extra
gather kernel. C100 will compare the copies with a batched copy or acquire-load
gather only on idle GPUs and only if NCU/Nsys attributes meaningful time or
visibility cost to this stage.

## Reliability decision O028 — publish compact counts, keep barrier state separate

The count kernel now ends each active compact count with `st.release.sys`.
Peers do not poll those values: they first complete the existing monotonic
NVLink barrier, then copy only `C*D*sizeof(int)` bytes from each symmetric LSA
pointer on the same communication stream. The force arena's control header is
never reused as barrier state; the barrier continues to use the first legacy
workspace words whose phase protocol is already exercised by DeepEP.

This adds no descriptor, ready flag, ring, per-copy atomic, or full-stride count
padding. The initial implementation intentionally launches eight transparent
D2D copies. Nsys must first attribute a material launch/copy gap and NCU must
show that a replacement gather is worthwhile before introducing another
kernel. No B2 timing is treated as performance evidence yet.

## Hot-path decision O029 — stage each token once, serialize only its moved copies

The manifest-free source kernel is one warp per source channel. It first
resolves all destination copies for the current token. If none move, it performs
no hidden load and no LSA write. If one or more move, hidden and ordinary
metadata enter shared memory once; only the four-byte transit key changes
between destination copies. Each key change is followed by a full TokenLayout
TMA store/commit/wait before the key can change again.

Serializing a token's moved stores is the smallest provably correct baseline:
an asynchronous TMA may otherwise observe the next destination's key. It also
preserves payload reuse while limiting replication exactly to the planner's
moved-copy count. C100 will use NCU/Nsys to decide whether safe batching or
another staging form is worthwhile; the initial implementation will not add a
second scratch buffer or a per-copy work list without measured evidence.
