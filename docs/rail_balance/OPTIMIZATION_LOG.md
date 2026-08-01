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
and return-unshuffle, and it can compile synthetic 2x4/4x2 plus production-
target 8x2 Hybrid cubins. It cannot truthfully execute Rail Gin teams.

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
table. The initial force-v1 restricted num_scaleout_ranks<=num_topk with
multiple reduction, making the final reduce row equal to destination. O030
later removes that artificial planner/source restriction and derives the
alternate legacy row locally without adding route metadata.

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

## Layout decision O030 — support D greater than K without route metadata

An independent C080-D audit found that the private host gate unnecessarily
required `num_scaleout_ranks<=num_topk`. The planner and source kernel use a
32-bit destination mask and lane-owned destination ordinals, so their real
bound is `D<=32`; each token need only touch at most K of those destinations.
The restriction was removed and true 8-GPU C1024/D32/K4 execution now passes
with all eight owners active.

This does not require a sidecar for combine. Legacy multiple-reduction already
selects its receive row statically: destination rank when `D<=K`, otherwise the
highest top-k lane targeting that destination. The preserved proxy-dispatch
payload contains those top-k ids, so standalone return-unshuffle can derive the
same row locally. This adds no Gin bytes and no branch to the persistent
dispatch/combine hot path.

## Profiling decision O031 — peer-TMA latency dominates the small source fixture

Nsys first showed that shrinking the launch from 1024 blocks to four removed
empty CTAs but did not reduce moved-owner latency: approximately 125 us before
and 127 us after. The smaller grid is retained because the proof is exact and
it avoids useless work, but it is not claimed as a speedup.

NCU then measured an all-owner C1024/D32/K4 case. One profiled owner moves
three complete 576-byte records. NVLink reports exactly 1728 user bytes plus
1248 overhead bytes and only 0.02% peak utilization. The kernel has 74
registers, 608 bytes dynamic shared memory, 1.56% achieved occupancy, 92.07%
no-eligible scheduler cycles, and long-scoreboard as the dominant sampled
stall. The result is a fixed-latency/TMA-completion workload, not a compute,
HBM, occupancy, or link-bandwidth workload.

Consequently the <=7-entry linear segment resolver remains. A binary-search
candidate was correct but measured about 134 us and was reverted. Metadata
initialization is also left simple despite racecheck's deterministic WAW
warnings: it is warp-ordered, contributes no reported error, and NCU gives no
evidence that adding ownership branches or another staging form would matter.
The next meaningful performance comparison must use many moved H7168 records
or the integrated pipeline; this tiny correctness fixture cannot select a
payload-throughput optimization.

## Hot-path decision O032 — derive return routing from frozen payload state

Return-unshuffle reuses the compact group prefix and the dispatch TokenLayout
that must remain alive through combine. `linked_list_idx[0]` verifies p, while
`src_token_global_idx` derives source node, owner, and token. D>K scans only the
K preserved experts because the legacy row cannot otherwise be known; D<=K
uses destination directly and avoids that scan. A moved record always has
owner!=egress, so the dead local/peer branch was removed and the LSA symmetric
pointer is formed unconditionally.

An independent audit considered a separate metadata-only validation phase to
guarantee zero writes under a deliberately corrupted post-Gate2 plan. It would
add a launch/global pass solely for an impossible production transition. The
fused kernel instead publishes sticky INVALID and discards the transaction;
partial scratch writes are permitted on that failure path. Two cheap semantic
checks—source node in range and local-destination moved count equal to zero—were
retained without adding per-record metadata or another synchronization epoch.

Production-like sm_90a cubins for H256/K4 and H7168/K8 both use 60 registers,
a 72-byte stack frame, zero local memory, and zero spills. Functional GPU runs
under unrelated load are not timing evidence. Return-unshuffle Nsys/NCU work is
deferred until GPUs are idle and a larger moved-volume fixture can separate TMA
payload throughput from fixed barrier/JIT/test-adapter latency.

## Hot-path decision O033 — bridge with existing prefixes, not another schedule

The vnode adapter does not add a destination-wide prefix tensor or a per-copy
work list. For channel c, the retained prefix is
`min(owner_channel_prefix, keep_count)` and the incoming moved prefix is the
existing `moved_channel_prefix`; their sum is exactly the number of vnode base
records in earlier channels. An independent 1,920-schedule exhaustion proves
that these intervals concatenate to `[0,quota)` for every egress/destination.

This matters beyond test code: the same algebra is what the future force
Hybrid consumer needs to publish dense channel tails. Adding another plan
array would consume memory bandwidth and enlarge the correctness surface
without reducing a measured hot-path operation. It remains rejected unless
NCU later attributes material cost to the small existing prefix lookup.

The two-buffer snapshot copies are intentionally outside this decision. They
exist only because separate virtual communicators own separate symmetric
windows; they are never candidates for the production data path and no Nsys or
NCU number containing them will be reported as operator performance.

## Reliability/performance decision O034 — keep strict checks outside the hot path

The vnode adapter deliberately revalidates descriptor, canary, ready, route,
dispatch metadata, contribution metadata, and every empty nonmatching lane.
That work is too expensive for the eventual persistent Hybrid kernel, but this
adapter is a single-node correctness oracle rather than a production path.
Deleting its checks would weaken the exact evidence while producing no user
performance gain, so they remain.

The accepted data mapping still adds no prefix array, copy manifest, queue,
ring, or success-path global atomic. Its only slot arithmetic is the existing
retained prefix plus incoming moved prefix. Post-audit safety checks increased
the H7168 pack instance from 90 to 92 registers; all four static instances
remain spill-free. No speedup is inferred from those compiler resources.

NCU/Nsys profiling is intentionally deferred until the two-buffer functional
round trip passes and the GPUs are idle. Profiles containing the owning tensor
copies between independent NCCL windows are evidence about test scaffolding,
not production operator performance. Later profiling must isolate adapter,
source shuffle, return-unshuffle, and legacy epilogue ranges and retain noisy
or negative results in this log.

## Identity decision O035 — reuse the exact legacy epilogue cache entry

The force-only source hook does not add a new reduction kernel. Its prepared
runtime generates the same `CombineReduceEpilogueRuntime` source under the same
`combine_reduce_epilogue` key and keeps the original rank/top-k layout choice,
SM count, shared-memory-derived warp count, and PDL launch flag. This makes the
single-node bridge exercise the production reduction semantics without
forking another implementation or changing `combine.hpp`.

Preparation happens once before the private transaction publishes count state;
the committed return-to-epilogue interval performs no JIT build or allocation.
The explicit comm-stream synchronize is correctness-test overhead and is not a
production optimization. NCU/Nsys will profile the epilogue only after the
full vnode loop passes; no standalone timing claim is made here.

## Profiling decision O036 — isolate one warmed kernel before optimizing

The installed tools support the required focused workflow: NCU 2025.1.1,
Nsight Systems 2024.6.2, and Compute Sanitizer 2025.1. Production kernels will
not gain NVTX or profiling branches. The new test harness may place one outer
`c080_profile` range around a warmed transaction and nested host-phase ranges;
all ranks use control-plane gates around that range so rank-zero capture does
not silently omit peer work.

The first Nsys pass uses CUDA/NVTX/OSRT without GPU metrics; metrics are a
separate run so sampling does not perturb the timeline used for launch-gap and
overlap decisions. The first NCU pass selects one device, one named kernel,
and one launch with application replay. It starts with `--set basic`; only a
measured bottleneck justifies LaunchStats, Occupancy, SpeedOfLight,
MemoryWorkloadAnalysis, SchedulerStats, WarpStateStats, or the dedicated
NVLink set. `--set full` is not an acceptable discovery pass.

Canonical filters are:

```text
rail_balance_hybrid_count_impl
rail_balance_hybrid_plan_impl
rail_balance_hybrid_prefix_impl
rail_balance_hybrid_source_shuffle_impl
rail_balance_hybrid_pack_vnode_base_impl
rail_balance_vnode_scaleout_impl
rail_balance_vnode_forward_impl
rail_balance_vnode_expert_impl
rail_balance_vnode_return_impl
rail_balance_hybrid_return_demux_impl
rail_balance_hybrid_return_unshuffle_impl
combine_reduce_epilogue_impl
```

Sanitizer order is memcheck, initcheck, synccheck, then focused racecheck.
Racecheck is evidence for shared-memory/TMA hazards only and cannot prove LSA
system-scope ordering. During unrelated GPU occupancy, ordinary non-OOM runs
remain useful for correctness but no timing or bandwidth number is accepted.

## Scope decision O037 — keep the two-window bridge outside the hot path

The world vnode bridge adds substantial host-side validation and owning
snapshots because it is a correctness oracle spanning two independent NCCL
symmetric windows. None of that code is reachable from public dispatch or
combine, and none of its cross-object copies may be included in an operator
speedup claim. Refactoring the older vnode emulator or adding a general
transport abstraction merely to shorten this private method was rejected: it
would enlarge the changed production-adjacent surface without removing a
single operation from the eventual Hybrid kernel.

The device mapping remains the two audited one-warp adapters. It still uses
the existing retained and moved prefixes, no destination-wide prefix, no copy
manifest, no queue/ring, and no success-path global atomic. The host bridge
rebuilds the production-shaped plan once before commit and compacts quota with
one 2D D2D copy. These costs establish functional equivalence only; C100 will
profile source shuffle, demux, return-unshuffle, and the legacy epilogue in
isolation after the GPU is idle.

Returning live owning plan references avoids fifteen redundant tensor clones.
The tradeoff is an explicit private-harness read-only contract. Since the
harness controls every call and compares digests immediately before B0, the
extra copies would provide no production safety or performance evidence and
are not justified.

## Test-cost decision O038 — tile a compact exact oracle, keep full GPU bytes

The first Hybrid vnode harness expanded Python `Fraction` vectors to all 7,168
columns in every spawned worker. Routing and the synthetic expert transform
are column-independent, and the source pattern has period eight. The accepted
harness therefore computes an exact eight-column oracle (three columns for the
rounding counterexample) and tiles the resulting BF16 pattern to H256/H7168
when constructing expected payloads and outputs.

Measured per-worker CPU oracle cost fell from about 4.408 s to 0.0054 s for
4x2/H7168 and from 2.954 s to 0.0049 s for 2x4/H7168. The full H7168 CUDA
payload, record bytes, reduce buffer, and final output are still compared, so
this removes Python object overhead rather than weakening device evidence.

Three redundant host checks were removed after their stronger supersets had
already passed: global coverage tuples after byte-exact proxy comparison,
global coverage tuples after complete world-array comparison, and a separate
channel-count hash after the full fourteen-tensor plan digest. Manual layout,
descriptor, route, and expected-record packing remain because they are the
independent evidence that prevents a C++ getter from validating itself.

## Scope decision O039 — vnode is disposable validation scaffolding

The vnode transport, synthetic expert, descriptors, ready words, canaries,
and two-window host bridge are not a planned public feature and must never be
copied into the real Hybrid dispatch/combine hot path. They exist only to
exercise the complete route and inverse route on one eight-GPU NVSwitch node
while real Rail/Gin hardware is unavailable.

They remain necessary until the isolated force dispatch/combine codegen and a
truthful multi-node Gin run cover the same plan, proxy-slot, forward-metadata,
return-unshuffle, and legacy-epilogue contracts. After that gate, perform an
explicit deletion audit: remove C080 vnode adapters and host wrappers whose
coverage is fully superseded, retain only a compact private regression/oracle
when it still catches failures that the real-cluster suite does not, and
delete that remainder as well if it is redundant. Vnode-only bytes and
branches are therefore forbidden from becoming production compatibility
surface.

## Hot-path decision O040 — compact retained/moved loops, one final tail

The first force dispatch specialization makes no attempt to preserve legacy
six-token interval tail publication. The owner scan issues only retained puts;
the egress then consumes one descriptor-free static proxy group per
destination and publishes one final dense tail after both classes of put. This
is the smallest schedule whose remote namespace is exactly
`[0,retained+moved)` and whose success path adds no atomic slot allocation,
ready polling, or route load.

The cost is reduced network/forward overlap within one channel. That tradeoff
is intentional for correctness-first codegen and remains a C100 measurement,
not an invitation to add a queue preemptively. A later candidate may publish
one retained prefix and batched moved suffix only if Nsys/real Gin evidence
shows the final-tail delay is material. It must keep dense slots and prove that
every publication is ordered after its corresponding put.

Dispatch-payload instrumentation is derived from immutable plan tensors
instead of a per-put counter. It deliberately excludes notify and packed-tail
control Gin operations:

```text
retained_puts = sum(retained)
moved_puts    = sum(moved) = sum(proxy_required)
payload_puts      = retained_puts + moved_puts
payload_gin_bytes = payload_puts * dispatch_token_bytes
```

The codegen test freezes exactly two payload-put sites in the scaleout role and
the retained threshold, `p=group_prefix+u`,
`remote_slot=retained+u`, proxy-put-before-tail, and transit-snapshot ordering.
`proxy_required` stays ABI-visible but is not read after Tag0; Gate2 validation
is the liveness boundary. The H256 specialization costs two registers versus
legacy (69 versus 67), while H7168 saves three to four; all retain the same
96-byte stack, zero spill, 384-thread/full-smem launch class. No optimization
is selected from these compiler counts alone.

## Hot-path decision O041 — change only the combine return destination

Force combine does not add a proxy consumer, descriptor, ready flag, queue, or
plan lookup. Dispatch already transported the immutable proxy slot `p`, and the
current destination ingress GPU is the same rail identity as the source
egress. Therefore the persistent combine kernel needs only one pointer select:

```text
p < 0  -> legacy owner/token receive slot
p >= 0 -> proxy_return_base + p * combine_token_bytes
```

The existing aggregated `gin.put`, final `gin.flush`, and rail completion cover
both targets. After it completes, the already validated static
return-unshuffle maps proxy rows back to `(owner,reduce-row,token)` and the
unchanged local reduction epilogue finishes the result. `D>K` row selection
stays in unshuffle, where the preserved proxy-dispatch record can choose the
highest matching top-k lane; duplicating that logic in persistent combine was
rejected.

All six force cubins have the same 216-register, 96-byte-stack,
1,024-byte-static-shared, zero-local, zero-spill resource profile as their
legacy twins. The only binary resource increase is eight bytes in constant
bank 0 for `proxy_return_base`. This establishes that the selected design adds
no occupancy cost at codegen. It does not establish network latency or
throughput; those remain real Rail/Gin measurements.

## Host-path decision O042 — prebuild once, submit raw state after the gate

The persistent kernels were already isolated, but calling the old launch
helpers would still generate/build JIT code in the middle of a force
transaction. The accepted adapter boundary stores runtime, specialization, and
LaunchArgs during prepare. A committed submit constructs only the small stack
Args object and launches on the existing comm stream.

The same rule now covers the dispatch copy epilogue and the combine reduce-base
offset. It deliberately does not move ordinary dispatch CPU count polling and
exact output allocation onto the GPU: Hybrid dispatch has already completed its
collective epoch before that phase, so preserving the legacy control flow is
simpler and safer. Combine is different; its return-unshuffle, local barrier,
and epilogue still form one liveness-sensitive chain and will receive unchecked
submit layers before production connection.

No queue, callback, event graph, generic transaction framework, or new public
operator was introduced. The adapter code is control-plane only and produces
the same CUDA specializations and resource counts as C080-E/F.

## Host-path decision O043 — one frozen geometry owner per raw launch

Every liveness-sensitive stage now follows the native DeepEP prepare/launch
split. TokenLayout construction, shared-memory arithmetic, active-grid
selection, JIT compilation, and all Tensor pointer extraction happen before a
collective epoch. The committed submit constructs only the existing runtime's
Args and launches it on `comm_stream`; checked wrappers exist only for private
standalone tests.

Source N is deliberately stored once in the Prepared specialization and reused
for both active-grid selection and the kernel Args. Passing a second dynamic N
was rejected because it enlarged the ABI and could make the grid and traversal
bound disagree. Return C follows the same single-owner rule. This is a host
reliability optimization, not a kernel speed claim: no CUDA loop, persistent
warp allocation, JIT key, queue, event graph, public operator, or default-off
path changed.

The production-integration style is now explicit: extend `ElasticBuffer`, the
existing symmetric buffer, TokenLayout, JIT runtime, streams, and handle
lifetime in place. Vnode remains disposable validation scaffolding; it must not
become a second transport/runtime framework or contribute a branch to the
production Hybrid hot path.

## Host-path decision O044 — extend the registered buffer, do not add storage

The force arena is a checked, aligned tail of the exact legacy Hybrid buffer.
It is not a second allocation, workspace extension, side window, or Python
tensor owner. This preserves the native NCCL symmetric-memory lifecycle and
lets every raw kernel derive peer addresses from the same registered base.

Force construction deliberately computes both the legacy formula and the
independent combined C++ helper, then requires the combined result to equal
`legacy + arena`. The redundancy is outside every operation hot path and
guards Python/C++ layout drift. No analogous call or field is allowed on off;
the capability remains false while pre-window cross-rank configuration safety
is unresolved.

## Host-path decision O045 — one MAX recovers exact min and max

The WORLD gate uses fixed pair encoding `(v, -v)` and a single signed-int64 MAX
reduction. Compared with an all-gather it keeps storage constant at 1 KiB;
compared with separate MIN/MAX it removes one collective; compared with a hash
it has no collision argument. The fields are nonnegative int64 values, so
INT64_MAX and its negation remain representable.

This helper is not a transaction framework. It has no class, callback, event
graph, dynamic tensor, public operator, or semantic manifest of its own. H4
will supply the small force-v1 field tuple. Per-rank values such as N and local
rail identity stay out of the equality set and are converted to the fixed
error key only when local invariants fail.

## Host-path decision O046 — test the real planner transaction before wiring dispatch

H3b remains a focused liveness test, not another runtime abstraction. It will
reuse the existing private planner, fixed WORLD-gate storage, local LSA barrier,
and CPU oracle. No generic coordinator, manifest class, callback graph, queue,
or public operator is justified.

The critical optimization is negative: prevent rank-local work between a
successful Gate1 and the one required `finish` call. A rank-local assertion or
allocation there can strand peers in the LSA barrier. Likewise, Gate2 must be
host-observed before any source payload publication. This ordering is a
correctness prerequisite for the later zero-extra-barrier hot path; it is not
a timing claim.

Local N and local rail identity are intentionally absent from WORLD-equal
fields. Comparing them would reject valid variable-token ranks and real
multi-node topology. Fixed fields include only operation/version/phase,
invocation, world topology, H/K/C/M/E/Pcap, arena identity, seed, and flags.

A local EP8 capacity failure is naturally identical on every rank after the
shared LSA snapshot. It proves real `finish` integration and recovery, but not
an asymmetric multi-node Gate2 failure by itself. H3b may additionally inject
one rank's post-finish Gate2 error before publication, then require WORLD-wide
rejection and abort/retry; the real capacity case remains mandatory.

## Host-path decision O047 — advance Gate2 by patching three prepared words

Gate1 and Gate2 use the same fixed 19-field manifest; only the phase changes.
After a successful Gate1, its D2H result is already a complete agreed manifest.
Re-encoding all fields after `finish` was rejected because the checked encoder
builds a Python list and repeats per-field validation after the local barrier.

The accepted path reuses that same pinned storage and patches only:

```text
word[0] = local error key
word[8] = phase 2
word[9] = -phase 2
```

This is smaller than owning a second manifest/template and keeps the existing
one-device/one-pinned-buffer identity. The private helper performs no field
iteration or dynamic validation. All storage, field, and value checks remain
before prepare.

Manifest K is taken from the actual `topk_idx.shape[1]`, not a nominal config.
This matters for zero-token ranks: shape `(0,K+1)` is locally valid and can
otherwise prepare a different JIT/layout while contributing no route data. A
focused EP8 fault now proves Gate1 rejects this difference before the LSA
barrier. N and local rail identities remain deliberately absent.

## Host-path decision O048 — use four explicit states, not a coordinator

The pending force transaction needs to distinguish a prepared plan from a
published dispatch. A second Boolean would permit invalid combinations, while
a generic transaction/coordinator class would add indirection and lifecycle
surface without helping the CUDA hot path. The accepted representation is one
host-only `uint8_t` enum: `Preparing`, `PlanReady`, `DispatchLive`, `Invalid`.

This change adds no device field, Tensor, collective, allocation, launch,
branch, or JIT specialization. Existing one-shot stage flags stay local to the
private validation path. A stale abort compares only the invocation ID and
does nothing; precommit abort releases ownership; post-publication abort poisons
the object instead of pretending published network work can be rolled back.
The permanent invalid state is deliberate fail-closed behavior.

`DispatchLive` is reserved until the real shuffle-to-dispatch commit exists.
Making it reachable early was rejected because it would make state evidence
stronger than the implementation. H4c must set it only after the adjacent raw
source and main-dispatch launches have been committed successfully.

## Host-path decision O049 — freeze the round trip once before Gate1

The first production force prepare owns the existing native pieces directly:
plan, local barrier, source shuffle, main dispatch, dispatch epilogue, main
combine, return unshuffle, combine epilogue, handle tensors, and raw pointers.
It does not introduce a generic coordinator, callback graph, second buffer, or
new public operator.  Topology, channel geometry, metadata width, and legacy
buffer bounds have one C++ owner and are returned only as a fixed Gate1 tuple.

The performance-relevant choice is negative: no allocation, JIT build,
`data_ptr`, peer-pointer lookup, or LaunchArgs derivation is allowed after
Gate1.  This removes host variance from the future publication interval while
adding no device instruction or default-off branch.  Receive tensors whose
leading dimension depends on completed dispatch counts remain a documented
post-dispatch allocation; pretending to size them early would waste memory and
diverge from native DeepEP.

Precommit abort drains the comm stream before releasing ownership.  That sync
is failure-only and closes an asynchronous lifetime hole without taxing the
hot path.  Frozen LaunchArgs do not make `LaunchRuntime::launch` infallible;
H4c still poisons state before the adjacent source/main submissions.  It must
also pass the dispatch prefix storage base to the persistent kernel but the
inclusive `base+1` view to the non-expanded epilogue.  These are correctness
constraints, not reasons to add another abstraction.

## Host-path decision O050 — poison once, then submit exactly twice

H4c does not add a coordinator, CUDA graph, callback, extra gate, or status
round trip.  After all fallible validation and raw capture, it marks the
transaction Invalid and issues source shuffle followed immediately by the
existing force Hybrid dispatch on the same comm stream.  Only two accepted
host submissions promote the owner to DispatchLive.  This preserves stream
ordering from final proxy writes into Tag0 without an extra local barrier or
device-wide synchronization.

The remaining low-level launch calls are checked and may fail asymmetrically.
Preconstructing launch config could remove `FuncSetAttribute` from the window
but cannot make `cuLaunchKernelEx` infallible; a graph would merely move the
same fatal boundary while adding machinery.  The private proof therefore
records partial submission as job-fatal and permanently poisoned.  Production
fusion remains the only way to remove the between-kernel host boundary if real
cluster evidence requires it.

Expert-prefix storage has two frozen views rather than post-gate pointer
arithmetic hidden in H5: base for persistent dispatch writes and base+1 for
the inclusive non-expanded epilogue.  The extra host pointer has no device
traffic, allocation, JIT key, or hot-path branch.

## Host-path decision O051 — exact post-main allocation, one correctness sync

H5a does not preallocate worst-case receive payloads merely to make dispatch a
single host call.  Main Hybrid dispatch already publishes exact mapped rank and
expert counts; allocating after those counts preserves native DeepEP memory
behavior and avoids a potentially large permanent force-only buffer.  The four
exact outputs are installed in the pending owner before the epilogue sees a raw
pointer, so no launch or asynchronous failure can release in-flight storage.

The correctness version uses one status D2H followed by one `comm_stream`
synchronization after the native epilogue.  This is intentionally not presented
as the final performance shape: it is the first point that can jointly observe
source shuffle, main Hybrid dispatch, and epilogue failure without inserting a
barrier into the critical source→Tag0 path.  C100 may replace it with an event-
owned asynchronous completion only after NCU/Nsys evidence and the user's
profiling procedure are available.  H5a adds no CUDA kernel, device branch,
metadata field, JIT specialization, arena byte, or default-off work.

Mapped counter hardening stays on the CPU polling path.  Int64 checked totals
were preferred over trusting the legal-kernel bound because H5a runs before its
first safe status read and must remain defined under partial/asynchronous
failure.  The checks have no effect on the communication kernel or successful
GPU schedule.

## Host-path decision O052 — make ticket transfer metadata-only

H5b does not copy the dispatch handle or send it through NVLink/RDMA. The
future public `EPHandle` will carry only buffer identity, invocation identity,
and one-shot consumption state while C++ retains the owning tensors. Handing
the ticket back to combine is therefore a host lookup/validation operation;
the payload stays where dispatch placed it.

The nonzero control-plane cost is the WORLD agreement before publication and
before irreversible combine submission. Each gate reduces a fixed 1 KiB
buffer with one MAX; its byte volume is negligible, but rank rendezvous waits
for the slowest participant. It is retained for force-v1 correctness and must
be timed separately from prepare, ticket validation, stream dependency, and
the four device stages. Removing or overlapping a gate is C100 work and
requires the user's NCU/Nsys procedure plus real measurements; no extra
collective or generic transaction layer is added speculatively.

## Host-path decision O053 — retain three safety gates, expose their cost

Dispatch Gate1 protects entry into the source-node LSA barrier. Gate2 depends
on the plan/capacity result produced after that barrier and protects payload
publication. Combine Gate1 depends on expert outputs and exact final-output
ownership that do not exist at dispatch time. The three reductions therefore
cannot be merged without either moving a fallible operation past publication
or making buffers permanently worst-case sized.

The fixed payload is 1 KiB and bandwidth is irrelevant; cost is collective
startup plus waiting for the slowest rank. Constructor mixed-mode consensus is
separate and paid once per buffer. Public ticket lookup is host-only and must
remain synchronization-free.

Two possible redundancies are recorded, not changed during H6 correctness:

- Python currently observes Gate2 plan status through `outputs[-1].item()`
  even though C++ plan finish has already synchronized and saved host status;
- combine prepare records compute→comm ordering even though the current fixed
  WORLD gate synchronizes the caller stream before commit.

C100 must time gate, `.item()`, prepare, ticket validation, status D2H, and
comm-stream synchronization separately. Removing either dependency without
that evidence would mix correctness work with speculative optimization.

## Host-path decision O054 — pay a second constructor gate only for force

Mixed off/force ranks must agree before registering differently sized
symmetric windows, so one constructor rendezvous is unavoidable when the host
protocol is compiled in. Resolved force bytes and QP/runtime settings are not
known until after a common NCCL communicator exists. Folding both facts into a
single gate would either compare guesses or move fallible sizing past window
creation. The minimal protocol is therefore one universal Gate0 and one
force-only Gate1, both reusing the same fixed 1 KiB CUDA/pinned storage.

The second gate adds no device kernel, buffer allocation, JIT specialization,
payload movement, NVLink traffic, or RDMA traffic. Its cost is one-time
collective startup and slowest-rank rendezvous during force buffer creation.
Default off pays only Gate0, retains no gate storage, and follows the exact
legacy hot path after construction. This keeps the per-operation cost model
unchanged while making the experimental window size deterministic across
ranks.

The legacy Python NVLink heuristic is intentionally not reused by force. Its
PCIe fallback contains an object collective after fallible local work, which
cannot safely live between two fixed MAX gates. Splitting it into a local
probe, another consensus, and a coordinated object collective would add
machinery for a topology outside force-v1's H200/NVSwitch target. Relying on
the existing NCCL/C++ topology boundary is smaller and avoids a new control
path. Constructor-gate timing belongs in later host profiling; no performance
claim is made from the CPU/fake correctness tests.

## Host-path decision O055 — share one ticket and consume before commit

An ordinary `EPHandle` receives one dynamic three-field ticket rather than a
second handle hierarchy or copied route. Its mutable state is shared by shallow
copies, so one copy consuming combine invalidates every alias. `PREPARING`
separates a retryable gate interval from `LIVE`; after gate acceptance Python
first sets `CONSUMED` and clears the buffer's live pointer, then enters the
irreversible C++ commit. A later failure is terminal and cannot recreate a
route whose device work may already have been submitted.

Combine acquires cleanup ownership before calling private prepare. Abort is
therefore attempted only for this buffer's matching invocation and is
idempotent even if prepare threw before installing completion. Foreign handles
are rejected by owner identity before they can mutate state or collide with an
invocation number. Dispatch uses the same attempt-before-call rule for its
stale-safe plan abort.

This public closure adds no kernel, arena byte, JIT specialization, payload
copy, NVLink/RDMA transfer, or device hot-path branch. It retains exactly two
dispatch WORLD gates and one combine WORLD gate. Replacing the explicit code
with a generic coordinator/state machine was rejected: it would hide the
prepare/gate/abort/publication boundaries while saving no device work.

Two host-cost candidates are recorded for C100, not changed in the correctness
checkpoint:

- production currently asks `plan_finish` for fourteen CUDA tensors, discards
  thirteen, then calls `.item()` on status even though C++ already has a
  synchronized host status. A status-only production result could remove both
  Python tensor materialization and the redundant readback;
- an exploratory CPU-tensor microbenchmark measured successful gate decode at
  roughly 161 us when indexing/`.item()` is repeated for 127 words versus
  roughly 8.6 us after one `tolist()`. This is only a host candidate; pinned
  storage and real collectives must be measured before adopting it. Encode has
  similar vectorization potential but requires a fail-closed fallback.

No new NCU/Nsys optimization was started in this host-path checkpoint. Per the
agreed workflow, the next device/end-to-end tuning run waits for the user's
profiling procedure and an idle GPU window; these CPU observations carry no
network or multi-node speedup claim.

## Validation decision O056 — sanitize the smallest complete changed path

C080-G does not rerun every historical sanitizer seed. The production source,
return, planner, and codegen cores were unchanged by H6 and the final vnode
fault test. Repeating all old cases would consume eight H200s without exposing
a new memory-ordering boundary.

The selected final target is the `2x4`, H256 vnode loop. It is the smallest
case that still has multiple destination servers, satisfies `D>K`, moves real
records between source rails, packs the world view, demultiplexes the return,
and invokes the same production return-unshuffle kernel used by force combine.
One dedicated warm cache then supports four complementary checks:

- memcheck for bounds, alignment, and hardware access faults;
- synccheck for CUDA synchronization misuse;
- unfiltered initcheck so writes from predecessor kernels remain visible;
- racecheck filtered to pack, demux, and return-unshuffle, where the new shared
  concurrency contract lives.

All four checks pass with zero errors; the focused racecheck also reports zero
warnings. This result complements rather than replaces the earlier H7168,
D32>K4, non-default-stream, all-zero, capacity, stale/corrupt route, and
delayed/fault evidence. It is correctness evidence only.

The next action is the two-case C090 correctness gap, not speculative code
cleanup. Once it closes, controlled C100 profiling begins: obtain the user's
promised NCU/Nsys workflow and verify that no unrelated GPU workload would
contaminate timing. Candidate host cost reductions from O055 remain hypotheses
until separately measured.

Those two correctness cases are closed in O057/D060; this O056 sentence is the
recorded decision at the sanitizer checkpoint, not the current task state.

## Validation decision O057 — inject state, not a fault framework

The C090 transit-key test mutates the exact four-byte `p` word already consumed
by the descriptor-free pack/demux path. It does not add a fault kernel, device
branch, delay argument, descriptor, or ring. Exact diagnostics reuse the host
status array that finish already copies after B5; one readiness bit prevents a
snapshot before that copy completes. This is smaller than encoding error values
in exception text and stronger because all six stage rows remain observable.

Rank delay must model late GPU work rather than a slow Python participant. The
selected sequence first agrees which rank is delayed, then queues `_sleep` on
that rank's existing comm stream and enters finish with no intervening Gloo or
CUDA synchronization. Thus peer B0 arrivals can precede the delayed producer,
while ordering stays native to the same stream used by the protocol. The sleep
duration is a liveness perturbation only and carries no latency claim.

Because this slice changes host/test code only, the correct validation is a
full extension rebuild, exact EP8 fault/status/recovery, API/legacy identity,
and the existing B1 materializer. Repeating Compute Sanitizer or codegen would
not exercise a changed device instruction. Resume the next C100 NCU/Nsys and
controlled-timing run only after the user provides the profiling workflow.

## Profiling decision O058 — measure payload volume before control-plane polish

The post-C090 inventory does not promote the 127-word gate decode or ticket
checks into C100 blockers.  Neither is on the device payload loop, and the only
accepted runtime profile moved three 576-byte records.  That fixture showed
fixed peer-TMA completion latency and cannot distinguish a bandwidth
optimization.

The next measurement therefore constructs H256 and H7168 cases with identical
top-k routes, remainder seed, plan/egress assignment, and moved-record count,
using enough moved records to expose payload scaling.  It then isolates source-
shuffle and return-unshuffle.  Nsys first establishes launch, barrier, copy,
and overlap boundaries; NCU then pins one device, one named kernel, and one
launch.  Planner/count/prefix, alternate slot allocation, and host decode remain
untouched unless this attribution makes them material.  No new profile was run
while recording this decision; execution waits for the user's supplied
workflow and idle GPUs.

C105 likewise uses a dedicated narrow force-v1 harness instead of adding
conditionals to the broad `test_ep.py` matrix.  Planner-derived puts and bytes
may be reported as derived values, but missing physical Gin wait/QP counters
must remain explicitly unavailable until the real runtime exposes them.

## Fixture decision O059 — fill the source kernel without profiling vnode

The retained source-scaling pair fixes G8/D9/K4/N1024/C256 and seed 100.  Each
owner is hot for a different remote destination, so all eight ranks are both
producers and consumers: each keeps 128 records, moves 896, receives 896, and
uses exactly Pcap 896.  Four duplicate-server top-k lanes remain distinct
experts but produce one deduplicated payload copy.  H256 and H7168 have equal
plans and differ only in record width (576 versus 14,400 bytes).

C=8 was functionally correct but left only seven moved CTAs per rank.  It was
rejected before profiling because such a grid would repeat the old fixed-
latency mistake.  C=128 also passed functionality during review.  C=256 is
retained because a normal non-overlap force configuration caps at four channels
per SM and commonly selects at least 64 SMs; here 224 channels per rank move
four records each.  This is a controlled microbenchmark shape, not a claim that
every production launch always uses exactly 256 channels.

The fixture stays in the existing source LSA harness and is named-only, which
keeps default regression cost unchanged.  Vnode is rejected for this purpose:
its pack, synthetic transport/expert, demux, two buffers, Gloo gates, and strict
snapshots would contaminate timeline and replay attribution.  Return-unshuffle
will instead import the same plan into its existing standalone harness.

Both final widths pass full byte correctness.  No performance tool was invoked
and no duration was recorded.  The next accepted evidence requires the user's
workflow, an idle-GPU audit, a separate warmup, and one pinned target launch.

## Fixture decision O060 — measure both directions with one immutable plan

Source-shuffle and return-unshuffle now consume the same named C100 plan.  This
removes route variance from their later comparison: H256/H7168 differ only in
TokenLayout width, and both directions see the same 7,168 moved copies and 896
slots per egress.  A separate vnode, planner variant, descriptor queue, timing
branch, or public API was not added.

The return harness derives allocation limits from the selected case and keeps
the two large fixtures behind explicit `--case-name`.  Default correctness
work therefore neither allocates nor launches the large profile shapes.  Its
CPU identity check computes all fingerprints without materializing every wide
row, while the actual GPU transaction still performs complete raw-byte and
poison comparison; this is test-overhead removal, not a device optimization.

Both widths pass true EP8 LSA functionality.  That result establishes a valid
measurement fixture but no performance finding.  Under the supplied evidence
contract, the next sequence is: record the exact environment and tool
capabilities, collect repeated profiler-free H256/H7168 samples, use Nsys to
prove the exposed timeline contribution, and only then pin an invocation in
NCU.  No performance-path code may change before that causal evidence exists.

## Measurement decision O061 — call the first baseline an adapter baseline

The private C100 source and return APIs are correctness-first synchronous
adapters.  They cannot be repeatedly launched against one prepared plan:
source consumes the shuffle right once, return consumes its own test right
once, and abort releases PlanReady.  Every timing sample must therefore use a
new monotonic invocation and the fixed prepare/finish/gate/stage/abort
lifecycle while reusing the Buffer and input tensors.

The primary no-profiler timer is host `perf_counter_ns`, because each adapter
already synchronizes internally.  CUDA events queued after the synchronous
call would not cleanly surround the kernel.  Per-rank raw samples are retained.
This checkpoint initially proposed the maximum per-rank duration as the
eight-rank value; O063 supersedes that denominator after proving it can hide
launch skew.  Cold, warmup, and steady samples remain separate, and logical
TokenLayout bytes are never relabelled as hardware NVLink/HBM traffic.

Source timing excludes the post-call visibility/recovery barrier.  Return
timing includes its mandatory test-adapter B1/B4 barriers and device snapshot/
status path, but excludes Python `.cpu()` correctness verification.  Nsys is
required before attributing any part of these wall times to the two kernels.
This deliberately avoids a speculative CUDA/runtime timer branch or a second
benchmark-only device path.

## Validation-package decision O062 — make unavailable evidence impossible to fake

The C105 skeleton fixes its evidence label in code and leaves physical Gin,
QP, NIC, wait, plan, and traffic fields unavailable until a real runner fills
them from the proper source.  Canonical traffic cases intentionally carry
different payload volumes, so later comparisons must report volume and cannot
present their absolute latency as an equal-work A/B.

The route/config bounds mirror the compiled force Hybrid workspace and index
domain.  Supporting configurations that the real kernel rejects would only
move bring-up failure from the local contract to the expensive network run.
Conversely, no sampler, performance policy, buffer planner, or runtime
instrumentation is added at this stage; the smallest useful artifact is a
strict deterministic schema plus an independent CPU oracle.

## Measurement decision O063 — fix the clock boundary before profiling

The accepted eight-rank latency is now
`max(end across ranks) - min(start across ranks)` on the common same-host
monotonic clock.  The rejected `max(end-start per rank)` formulation omitted
launch skew: one H256 source smoke measured 79.335 us by rank maximum but
141.075 us by the global stage span.  This is a measurement correction, not a
device optimization.

Logical bandwidth divides the fixed aggregate moved `TokenLayout` bytes by
that global stage span.  It is labelled aggregate checked-adapter rate and is
never called physical NVLink, HBM, Gin, or production throughput.  Return also
reports its wrapper-only D2D copies separately so the logical numerator cannot
silently include reduce seed/snapshot traffic.

No CUDA, JIT kernel, buffer layout, public Hybrid path, capability bit, or
production hot-path branch changed.  The only accepted code is an external
measurement harness with fail-closed cleanup and exact artifact identity.
Dirty-tree one-sample runs are retained solely as functionality evidence.  The
next optimization decision requires clean direct profiler-free distributions,
then Nsys exposed-time attribution; NCU and hot-path edits remain forbidden
until that evidence exists.

## Measurement decision O064 — retain clean samples but reject stability

The first clean source collection passed every automatic validity gate, so its
six JSON reports are preserved with hashes.  It nevertheless fails the manual
stability gate: H256 run medians span 7.26%, H7168 medians span 20.22%, and one
H7168 rank-4 call reaches 2.254 ms while peers remain near 0.1 ms.  Removing
that sample or reporting only the best run would violate the measurement
contract.

The isolated-rank shape contradicts a simple claim that all eight source TMA
payload loops became slow.  Candidate explanations are host descheduling
around the synchronous adapter boundary, an intermittent stream/barrier wait,
or per-device clock asymmetry; none is proven.  GPU pre/post state alone cannot
resolve a transient event.  Therefore no kernel, launch geometry, plan, or
buffer optimization is authorized from these numbers.

When work resumes, measurement stability is the first falsification target.
Use a single controlled variable—such as explicit rank CPU affinity or an Nsys
OS-runtime timeline chosen to explain the outlier—while retaining the original
unbound reports.  Only after a repeatable boundary exists should return
baseline collection and target-kernel NCU proceed.

## Measurement decision O065 — keep OMP=1 for tail control, reject it as a full fix

Setting only `OMP_NUM_THREADS=1` reduces the source-H7168 pooled CV from
82.84% to 8.34% and its maximum from 2,274.190 us to 218.716 us.  The result is
large, repeatable across three independent 10+100 processes, and consistent
with the container's 96-CPU quota versus default 96 intra-op threads per each
of eight ranks.  It is accepted as a measurement-environment control for the
next falsification, not as an operator optimization or throughput claim.
The rank-local maximum-stage median is essentially unchanged
(97.518→96.556 us), while its p99/max collapse; unchanged disassembled kernel
instructions reinforce the same interpretation.  The sequential, unpaired
collection order also prevents a causal speedup percentage claim.

The experiment does not stabilize the primary global-span median: its three
medians span 14.94%.  Raw timestamps show a nearly fixed rank 1→rank 7 launch
order after the pre-stage Gloo gate; the run with the largest median also has
the largest median start skew.  Rank-local stage-duration medians and
global-minus-start-skew medians are much tighter.  Therefore changing source
shuffle instructions, launch geometry, TMA cadence, or plan layout would not
address the currently exposed variance and remains prohibited.

The next experiment holds OMP=1 and changes only the release measurement or
main-thread scheduling condition needed to falsify this control-plane skew.
It must stay default-off or benchmark-only, preserve all raw per-rank times,
and be removed/rejected if it merely hides real work.  Nsys remains the next
diagnostic layer only if this minimal experiment cannot distinguish host
release from device execution; NCU still has no justified invocation.

## Measurement decision O066 — prove Gloo/host release before changing the timer

Do not subtract start skew from the accepted C100 timer or add scheduled launch
alignment yet.  Either could make distributions look stable by redefining the
measurement before proving that the removed interval is control-plane noise.
The smaller falsification is a separate CPU-only eight-rank Gloo probe with
identical post-barrier timestamp semantics.

The decision threshold is registered before the formal run: median CPU-only
return span must cover at least half of every OMP E1 median start skew, and the
dominant first/last return ranks must each occur at least 80% of the time.
Passing shows that the control plane alone is sufficient to create most of the
skew; failing sends the problem to Nsys OSRT+CUDA rather than to a speculative
CUDA edit.  This probe cannot prove pure Gloo protocol cost, production Hybrid
latency, or a kernel speedup because host scheduling is inseparable from the
observed barrier return.

## Measurement decision O067 — separate gate skew from adapter attribution without subtraction

E2 passes its registered criterion: all three bare-Gloo medians cover at least
59.97% of every E1 median start skew, and rank1-first/rank7-last each reach 98%.
The artificial pre-stage control gate is therefore a proven major component
of the global span's run-to-run median variance.

The response is not to subtract 36--39 us, cherry-pick a rank, or replace the
primary report with a corrected number.  Every future C100 report retains the
same common-host global span and all raw rank intervals.  For operator work it
also treats the existing per-iteration maximum rank-local synchronous adapter
duration as a separate call-envelope metric; its source-H7168 run medians were
already much tighter than global spans.  Nsys must then split that envelope
into CPU launch/wakeup, CUDA API/synchronization and exposed GPU kernels.

This two-metric interpretation supersedes the earlier attempt to make one
number serve both end-to-end rank arrival and kernel attribution.  It changes
no timer, benchmark code or CUDA hot path and can be falsified by the upcoming
source/return distributions and Nsys timeline.

## Measurement decision O068 — do not let a stable median hide rank-local tails

Source-H256 supports separating the Gloo-contaminated global span from the
rank-local adapter envelope: run-median range improves from 12.18% to 5.48%.
It also falsifies the stronger hope that OMP=1 makes the whole distribution
stable.  Two rank-local samples still exceed 470 us and pooled p99 is
168.538 us versus a 76.793 us median.

Therefore median stability is not sufficient to enter NCU or modify the
payload kernel.  Continue the unchanged return collection, then use Nsys to
classify representative and tail calls as host deschedule, CUDA API/sync wait,
or exposed GPU execution.  No outlier removal, percentile replacement, CPU
pinning, or timer correction is introduced in this checkpoint.

## Measurement decision O069 — reject co-tenant runs and let Nsys classify tails

Return-H256 run 2 is permanently rejected even though its observed CV is
smaller than some accepted runs.  The automatic post-run gate identified four
Megatron workers that were absent at preflight.  Process identity is part of
the measurement contract; visually plausible timing cannot override it.

Accepted runs r1/r3/r4 have close medians but large p99/max tails, including a
1.324 ms rank-local call envelope.  Median repeatability therefore does not
authorize kernel optimization.  Finish the unchanged return-H7168 collection,
then use Nsys to distinguish host descheduling, CUDA API/synchronization wait,
and exposed GPU execution.  NCU is allowed only for an exact invocation whose
critical-path exposure is established by Nsys.  No outlier trimming, affinity
change, hot-path edit, or speculative TMA rewrite is introduced here.

## Measurement decision O070 — freeze typical latency and profile the tails

The final return-H7168 group closes raw profiler-free collection, not tail
causality.  Accepted global medians span only 2.37% and rank-local medians
2.12%, while two accepted rank-local calls exceed 1.7 ms and the maximum is
2.008 ms.  Automatic eligibility proves the sampled environment contract, not
distribution stability.

The next Nsys capture is therefore a variability-diagnosis experiment.  It
must preserve representative and tail invocations and separate CPU release,
CUDA API/synchronization, and GPU-kernel intervals.  We will not optimize from
the stable median alone, and will not profile the rejected co-tenant run as a
kernel workload.  NCU remains conditional on Nsys identifying an exact
exposed invocation with end-to-end impact.

## Measurement decision O071 — add only a diagnostic NVTX window

Full-process Nsys collection would include cold JIT, initialization and report
generation, while the current harness has no valid capture trigger.  Reusing
the vnode profiler ranges would profile a test-only transport rather than the
checked source/return adapter under study.  The minimum natural extension is
one default-off `--nvtx` flag in the C100 benchmark itself.

Rank 0 will own one steady-only capture range; every rank will expose sparse
phase and exact-invocation ranges.  Diagnostic reports must set automatic
baseline eligibility false.  The first Nsys pass uses no CPU sampling, GPU
metrics or NCU counters; it asks only whether host API/synchronization or GPU
execution owns the exposed stage and tail.  This instrumentation is not a
performance optimization and cannot create a speedup claim.

The default target-stage timer still surrounds the same checked adapter call.
The wider transaction now constructs local callables and evaluates a handful
of false branches, so the change is minimal-overhead rather than literal
zero-overhead.  No before/after claim may mix the old and new harness commits.
The pre-registered Nsys command used `--capture-range-end=stop --kill=none`;
O072 below preserves this falsified decision as history and supersedes the
actual collection command.

## Measurement decision O072 — profile the full process tree, then cut by the steady timestamp window

The pre-registered child-owned range-trigger topology is rejected by direct
experiment.  With Nsys's default `wait=all`, an orphaned resource-tracker
zombie is held by the launcher and the benchmark correctly refuses to declare
its process group clean.  `stop-shutdown` does not change this.  The installed
`--wait=primary` mode fixes lifecycle without weakening the watchdog, but a
range pushed by rank 0 still does not start collection: both default-domain
and any-domain trigger runs produce no report.

Adding an interprocess start/stop controller solely to obtain a smaller trace
would be over-designed test infrastructure.  The smallest working experiment
is therefore `--capture-range=none --wait=primary`, followed by schema-aware
filtering to the already existing `c100_nsys_window`.  The 0+1 smoke proves
that this full capture sees all eight devices.  Because installed
`--filter-nvtx` projects only the process that owns the range, multi-rank
statistics will use the range's global nanosecond start/end as a time filter.

This decision accepts a larger diagnostic report and profiler perturbation;
it does not alter profiler-free truth.  Cold JIT and setup records remain in
the raw file for audit but are excluded from steady-window SQL and stats.
No kernel conclusion may be drawn from the one-sample smoke.  Formal H7168
10+100 Nsys data must identify representative and tail invocations before any
NCU capture or performance-path edit.

## Measurement decision O073 — separate test arrival tails from the exposed return kernel

The formal H7168 Nsys trace falsifies “return-unshuffle variation causes this
trace's long tail.”  It does not attribute the older profiler-free millisecond
events.  The production-shared return kernel is stable enough to profile:
pooled p50/p95/max are 135.168/145.190/153.472 us and its exclusive-union p50
is 93.871 us.  But its per-iteration maximum correlates only 0.138 with global
stage span.  The test-only pre-kernel B1 arrival wait correlates 0.883, reaches
235.040 us, and converges all ranks to sub-microsecond completion skew.  The
large 4-byte D2H host duration is the downstream waiting sink for that stream,
not four bytes of expensive transfer.

Therefore:

1. Do not NCU-profile B1 or the status D2H to explain tails; replay would alter
   the cross-rank timing that they observe.
2. Do not optimize the adapter's two seed copies or result snapshot; they are
   absent from the production combine commit.
3. Use NCU only to test a typical-kernel hypothesis on the exact slow-path
   rank-6 invocation selected in D078.  Use strict whole-application replay,
   not default kernel replay, because the kernel makes LSA peer stores.  Start
   with the installed `basic` set, record replay/cache/clock warnings, and add
   only evidence-selected sections.
4. Compare device 6 with device 0 only if the first report cannot distinguish
   work/path asymmetry from memory/TMA/LSA limitation.
5. Permit one hot-path variable to change only after NCU identifies a source-
   mapped limiter.  Retest correctness and profiler-free distributions; revert
   complexity if the no-profiler result does not improve.

This decision opens targeted NCU for the return kernel.  It does not authorize
a barrier rewrite, CPU-affinity workaround, status-contract weakening, or
production performance claim.

## Measurement decision O074 — reject replay that duplicates peer side effects

The target kernel is not a pure single-device function: one profiled rank reads
its local proxy-return arena and writes another process's reduce buffer through
LSA.  Default kernel replay and range replay have no proved cross-process peer-
allocation restore contract here.  They can repeat remote stores while peers
wait in B4, perturb barrier timing, or time out the transaction.

Use `--replay-mode application`, `--app-replay-mode strict`, and
`--app-replay-match grid`.  Filter device 6 plus exact NVTX steady-26 and one
demangled kernel; keep `--launch-count 1 --kill 0 --set basic`.  Each pass must
rebuild all distributed state.  Explicit `cache-control none` and
`clock-control none` minimize single-rank control perturbation but prevent an
unqualified cross-device clock comparison.  Application-replay mismatch,
timeout, permission error, or cleanup failure is a retained falsification, not
permission to relax matching or replay the peer-writing kernel.

## Measurement decision O075 — separate target-channel packing from warp wait

The exact `basic` capture passes identity and lifecycle gates after ten strict
whole-application replays.  It shows 0.13 waves/SM, 1.61% achieved occupancy,
1.03 active warps/SM, 0.11% SM throughput and 0.41% DRAM throughput.  These
values reject a saturated-SM or saturated-HBM explanation for the instrumented
pass, but do not identify why one-warp CTAs are not issuing.  Shared memory is
not the first fix: only 256 blocks exist for 132 SMs, far below its 15-block/SM
resident limit.

The deterministic plan adds a concrete imbalance hypothesis.  Return-unshuffle
sees 32 target channels with 28 records each and 224 empty channels per egress,
because prefix materialization greedily fills each destination from channel
zero.  The source-side 224-by-four producer distribution recorded in O059 is a
different mapping and remains valid.  Empty CTAs explain some distribution
skew; only scheduler/warp evidence can show whether the active CTAs are then
dominated by serial TMA waits, scoreboard, membar, or another dependency.

Do not increase occupancy, change channel count, stripe the plan, batch TMA, or
rewrite the proxy layout yet.  The smallest next falsification keeps the same
device-6 steady-26 identity and adds only SchedulerStats, WarpStateStats and
Nvlink.  Nvlink tests whether transmitted user bytes match the expected
12,873,728-byte record volume and whether peer transport is saturated.
SourceCounters is not collected from the current no-lineinfo cubin; a later
lineinfo build must first prove identical SASS/resources.  MemoryWorkloadAnalysis
is conditional, and `full` remains rejected.  A hot-path single-variable
experiment is still blocked.

## Measurement decision O076 — authorize only target-channel distribution

The directed report closes O075's attribution gate.  Deterministic schedule
reconstruction gives 32 payload-bearing and 224 empty one-warp CTAs per egress.
NCU simultaneously reports 98.238888% no-eligible scheduler cycles,
0.017611 eligible warps/scheduler, and 49.733468 long-scoreboard cycles per
issued instruction, 86.3304% of the 57.608302-cycle interval.  Barrier,
membar, and execution-pipe throttle states are zero.  Exact logical payload
bytes appear as 12,873,728 NVLink TX user bytes, while aggregate TX peak
utilization is only 5.730689%.

This evidence authorizes one variable: change how the existing incoming copies
are assigned to target channels so more existing return CTAs carry work.  Keep
quota, minimum moved-copy count, segment ownership, proxy count/capacity,
TokenLayout, group-major static slots, TMA payload loop, barriers, public API,
JIT specialization and default-off behavior unchanged.  Predict unchanged
logical bytes and correctness, more nonempty target channels, lower no-profiler
return latency, and a shorter Nsys return kernel.  A later NCU rerun is needed
only if the profiler-free and Nsys result disagree with the prediction.

Do not pipeline TMA, alter shared memory, raise channel count, change block
geometry, or add a work queue in this experiment.  Long scoreboard proves a
long-latency memory dependency, not the exact responsible instruction; if
channel distribution fails to improve the no-profiler result, collect bounded
memory/source evidence before touching that loop.  This is deliberately a
simple plan-layout experiment, not a new runtime abstraction.

## Optimization experiment O077 — rotate destination channel windows

The first O076 experiment uses destination-window rotation, not balanced
waterfill and not a `channels * destinations` return grid.  For one egress and
destination, define:

```text
incoming[d] = quota[e,d] - keep[e,d]
rotate[d]   = (incoming[d] > 0 and keep[e,d] == 0)
span[d]     = rotate[d] ? ceil(incoming[d] / channel_capacity) : 0
start[d]    = rotate[d] ? exclusive_prefix_sum(span)[d] mod num_channels : 0
```

Retained counts remain the native source-channel prefixes.  Only a pure-deficit
group (`keep == 0`) rotates: all of its spare capacities are identical, so its
nominal span is exact and safe to contribute to the cursor.  A partial-deficit
group preserves the old physical start at channel zero and contributes no span;
this prevents retained placement from turning a nominal span into a misleading
cursor increment.  Starting at the selected channel, one cyclic O(C) pass
greedily places the unchanged `incoming[d]` copies into each channel's
`channel_capacity - retained` spare space.  A final physical-channel pass
rebuilds the existing monotonic moved prefix; the channel-major/destination-
minor group prefix and dense proxy slots are unchanged.  The capacity proof
remains `sum(spare) >= quota - keep`, so a complete cyclic pass must consume
every incoming copy.

This is the smallest natural change because each destination lane can reuse the
current warp scan to obtain `start`; no field, allocation, atomic, queue,
manifest, kernel argument, public option or return-kernel instruction changes.
For the frozen C100 G8/D9/N1024/C256 case, the exact prediction per egress is:

```text
before: 32 channels * 28 records, 224 empty
after:  224 channels * 4 records, 32 empty
```

The following facts must remain bit-exact: count/quota/keep/segments,
`proxy_required=896`, global `moved_copies=7168`, proxy slot density, payload,
metadata, owner/egress/destination identity and final combine output.  Add
explicit zero-move, pure-deficit cyclic-wrap, non-divisible, partial-deficit
fallback and exact C100 distribution checks.  CPU oracle and validator change
first; the GPU prefix materializer then mirrors them and must pass exact
CPU/GPU parity before any performance run.

Two costs are pre-registered rather than hidden.  The original cyclic GPU
translation would have added one channel pass.  The pure-deficit gate makes a
smaller materialization possible: `full=incoming/capacity`, `tail=incoming %
capacity`, and a physical channel's cyclic distance from `start` determine its
moved count in closed form.  The final GPU kernel therefore retains its two
existing channel passes and adds one warp prefix scan plus integer mapping;
this implementation refinement does not change the O077 scheduling variable.
Static `cuobjdump --dump-resource-usage` reports REG 62 before versus REG 64
after, with STACK/SHARED/LOCAL all still zero; this is a measured codegen cost
to test, not a performance conclusion.
More importantly, the current source resolver linearly scans physical target-
channel prefixes; the C100 average resolved channel is expected to move from
about 15.5 to about 111.5.  Therefore source, return and plan are all measured
at H256 and H7168.  A return-only win is rejected if source/plan cost absorbs
it.  Binary search, waterfill, TMA pipelining and a compact work queue are
separate future hypotheses and must not be bundled into O077.

The pure-deficit gate removes a real rotate-all regression but is not a global
minimax scheduler.  A retained audit counterexample with G2/D5/C8/T5 produces
per-channel target loads `[2, 2, 0, ...]` under the legacy fill and
`[1, 3, 0, ...]` under O077 when partial groups occupy a channel later selected
by a pure-deficit window.  Its exact destination routes are
`(((0,), (1,3,4), (0,1,2,4), (0,1,2,4)), ((0,1,3),))`, with count
`((3,3,2,1,3), (1,1,0,1,0))`; the affected egress is one.  Exhaustive
G2/D2/C2/T4 enumeration and 20,000 random cases found no active-channel-count
regression.  The random audit used `random.Random(0x077)`, G in [2,5], D in
[1,6], C in [1,8], T in [1,16], uniform 0..T tokens per owner and independent
destination probability 0.35; the peak-load counterexample appeared at
iteration 9118.  This means
no generic dominance is claimed.  This is a known performance risk, not a
correctness failure.  B-side reports must retain the full channel distribution
and reject O077 if the target workload loses overall source/return performance;
adding load-aware placement now would violate the one-variable experiment.

The falsification order is fixed:

1. Capture a clean, immediate profiler-free pre-change source/return baseline
   if all eight GPUs are idle; otherwise preserve the prior accepted baseline
   and defer timing without blocking functional work.
2. Change CPU oracle/validator and focused goldens; run CPU correctness.
3. Change only the GPU target-channel materializer; run exact GPU parity and
   all affected source/return/vnode/default-off correctness.
4. On idle GPUs run three profiler-free H256/H7168 source and return trials with
   the frozen warmup/sample contract, retaining median/p95/p99/raw samples.
5. Retain O077 only if the combined evidence is favorable, then collect the
   same Nsys window.  Otherwise revert the production complexity while keeping
   the failed experiment and artifacts in the logs.

Implementation status: the builder and strict validator implement the pure-
deficit gate.  Direct CPU schedule tests pass 14/14 and vnode round-trip tests
pass 7/7.  The final closed-form GPU materializer passes 76 exact single-GPU
CPU/GPU plan cases and the fresh-JIT eight-GPU LSA plan transaction.  These
include the audit partial-deficit counterexample, pure-deficit cyclic wrap,
C1024/D32 and exact C100 224-by-4 distribution.  The missing-pytest, both stale-
golden failures, rejected rotate-all CPU prototype and rejected extra-GPU-pass
prototype are retained in D084/D085.  Full source/return/vnode consumer-matrix
correctness and all performance conclusions remain pending.

Consumer status: fresh-JIT C100 H256 source shuffle passes exact legacy
TokenLayout bytes for 7168 moved copies, and return-unshuffle passes exact
owner row/token bytes with poison-preserved non-targets.  A fresh-JIT
2x4/H256 vnode passes the full dispatch, forwarding, synthetic expert, combine
return and original-owner unshuffle round trip.  This establishes the minimal
local semantic data path; it does not replace H7168/full-matrix/fault/default-
off regression and carries no performance conclusion.

Full functional status: the default source suite, default return suite, named
C100 H7168 source/return, all nine vnode cases, compact-plan faults, fixed
WORLD gates, default-off native EP8 smoke, and force dispatch/combine codegen
all pass from fresh JIT caches.  This closes the ordinary local functional
matrix for O077.  The changed prefix kernel still requires a focused sanitizer
rerun before profiler-free B-side collection.

Sanitizer status: Compute Sanitizer 2025.1.0.0 ran the 13-case focused B1
matrix against four fresh JIT caches.  Prefix-filtered memcheck and synccheck,
unfiltered device initcheck with only API-memory shadow checking disabled, and
prefix-filtered racecheck all pass.  The first three report zero errors;
racecheck reports zero hazards, zero errors and zero warnings.  This is the
final correctness gate for O077; it is not performance evidence and does not
claim that racecheck proves cross-GPU system-scope ordering.

## Measurement decision O077 — retain the result, not the current hot-path cost

The complete profiler-free B side, identity-matched cross-run Nsys diagnostics
and exact NCU controls make O077 a mixed result rather than an unconditional
optimization.  The three run-median aggregates show source H256/H7168
regressions of 53.20%/42.06% and return H256/H7168 improvements of
24.24%/22.73%.  These are independent checked adapters, not additive
production-EP phases, so neither their sums nor their diagnostic NVTX envelopes
are called an end-to-end speedup.

Nsys isolates both directions in unchanged-launch, unchanged-SASS kernels.  At
H7168 the pooled return-kernel p50 falls from 135.168 to 59.232 us and its
copy/barrier-excluded union falls from 93.871 to 56.470 us.  Conversely, the
unchanged source kernel on device 6 rises from a 53.952-us pre-O077 p50 to
105.376 us and is longest in 100/100 post iterations.  Prefix materialization
itself rises only about 1.9 us.  Exact NCU application-replay controls then show
2,971,264 dynamic instructions on the high-channel device 6 versus 906,250 on
the same-code low-channel device 0, with neither SM nor DRAM saturated.  NCU
durations are not compared because replay clocks differ.

This proves target-channel packing is a return bottleneck for the frozen C100
checked adapter only.  It does not prove a production Hybrid critical-path
fraction.  Typical medians are repeatable, but p95/p99/max tails are not
accepted as stable; post return-H7168 retains a 2.468-ms maximum.

The causal chain is therefore:

```text
O077 spreads return work from 32 to 224 channels
-> return-unshuffle parallelism and finish skew improve
-> moved target ordinals now resolve around later physical channels
-> source's unchanged linear channel-prefix scan executes much more work
-> source latency regresses in every profiler-free run
```

O077 remains committed as an auditable parent and proves that target-channel
packing is a real return bottleneck for the frozen checked adapter.  It is not
accepted as the final/default hot path by itself.  It may be retained only if
the following independent resolver experiment removes the source cost without
losing its return result; otherwise the production complexity is reverted
while all measurements and failed-experiment logs stay retained.

## Optimization experiment O078 — bounded monotonic prefix lookup

O078 changes one variable inside `resolve_hybrid_copy`: replace the linear
`channel=0..C-1` scan of `moved_channel_prefix` with a bounded upper-bound
lookup for the first channel whose exclusive end is greater than the incoming
ordinal.  With an endpoint check, C=256 needs eight bounded search iterations
rather than a data-dependent average near one hundred.  After selection,
the existing containment condition remains mandatory:

```text
prefix[channel] <= incoming_ordinal < prefix[channel + 1]
```

This deliberately reopens, rather than erases, an older rejected binary-search
result.  The old tiny source fixture moved only three 576-byte records and was
fixed-overhead dominated; its binary resolver was slower and remains rejected.
O078 is a new large-volume hypothesis authorized by 7,168 copies, a stable
rank/channel latency gradient and a 2.065-million-instruction device control.
The new result must stand on its own no-profiler A/B data.

The plan producer supplies a nondecreasing prefix and the runtime contract
treats its tensors as immutable after Gate2.  Gate2 propagates plan status; it
does not rescan every prefix value.  O078 therefore preserves explicit row
endpoint, candidate containment, channel/slot and caller capacity checks, but
does not claim to detect every arbitrary post-Gate2 nonmonotonic memory
mutation.  The old linear scan cannot guarantee that either when a corrupt row
still contains a plausible overlapping interval.  A full O(C) integrity scan
would erase the measured resolver objective and is not bundled into O078.

No other variable may change: ABI, plan/quota/keep/segments, channel rotation,
proxy count or slots, allocation, queue/atomic behavior, TokenLayout, payload
bytes, TMA issue/wait/fence order, barriers, launch geometry, return kernel,
public API, capability bits and default-off behavior remain identical.  No
inverse manifest, per-copy cache or new workspace is added.

Pre-registered predictions are:

1. CPU/GPU plan bytes, 7,168 moved copies, proxy slots and all owner/egress/
   destination mappings stay bit-exact.
2. Device-6 source H7168 kernel p50 falls materially from about 105 us toward
   the pre-O077 approximately 54-us value, and the rank-dependent duration
   gradient contracts.
3. Source dynamic instructions on the high-channel target fall materially
   from 2.971 million toward the low-channel control's 0.906 million.
4. Plan/prefix producer algorithms and exact outputs remain unchanged.  Return must satisfy
   the quantified no-regression gates below; in particular O077's approximately
   59-us pooled return-kernel p50 should be retained.
5. Profiler-free source H256/H7168 improves in three clean runs without a
   corresponding return regression.

Falsification order:

1. Audit valid-prefix equivalence, endpoint/out-of-range rejection and the
   pre-existing immutable-plan/fault boundary.
2. Implement only the bounded lookup and run focused CPU/unit checks.
3. Use fresh JIT caches for exact source, return, vnode, fault/WORLD and
   default-off/codegen regressions affected by the shared resolver.
4. Run focused memcheck/synccheck/initcheck/racecheck for the changed source
   resolver; do not claim racecheck proves inter-GPU ordering.
5. On an idle eight-GPU window, repeat the same three profiler-free source and
   return H256/H7168 runs.  If positive, repeat the same H7168 source Nsys
   window.  Repeat NCU only when needed to confirm the instruction prediction
   or resolve an Nsys/no-profiler disagreement.

The no-profiler acceptance rule is fixed before editing.  For both H256 and
H7168, every one of the three same-label O078 source medians must beat O077,
the median of run medians must improve by at least 25%, and pooled median plus
ten-percent trimmed mean must each improve by at least 20%.  For each return
width, the median of run medians, pooled median and trimmed mean may regress no
more than 5%, and no same-label median may regress more than 10%.  Raw
p95/p99/max remain reported but are not a primary accept gate because the
frozen baseline already rejects their stability.  Nsys must show the device-6
source-kernel p50 and rank gradient contract while return SASS/cubin resources remain
unchanged; no production end-to-end claim is inferred.

Reject O078 if exact semantics change or any quantified source/return gate
fails.  In that case revert the resolver complexity and reassess whether O077
itself should leave the production hot path.  Do not escalate to TMA
pipelining, queues, block-geometry changes or a work manifest without a new
evidence chain.

### O078 result and decision

Status: **ACCEPTED_FOR_LOCAL_CHECKED_ADAPTER** at clean `32b9cfd`.

The first numerical B side is retained but rejected because timeout 300 rather
than 180 changed the local-barrier JIT specialization.  The formal B side is
the independently audited, identity-matched 12-report collection under
`.cache/rail_balance/c100/o078/post-32b9cfd-matched-o077/`.  Every same-label
semantic SHA equals O077 and all code/JIT/co-tenant/baseline gates pass.

| Gate | Required | Observed | Result |
| --- | --- | --- | --- |
| source H256 same-label | all improve | 37.00%, 44.76%, 42.07% | PASS |
| source H256 aggregate | MoM >=25%, pooled/trim >=20% | 42.07%, 41.10%, 41.00% | PASS |
| source H7168 same-label | all improve | 16.32%, 29.60%, 27.04% | PASS |
| source H7168 aggregate | MoM >=25%, pooled/trim >=20% | 26.57%, 24.91%, 24.24% | PASS |
| return H256 aggregate/run | <=5% / <=10% regression | all aggregates improve; worst run +8.09% | PASS |
| return H7168 aggregate/run | <=5% / <=10% regression | MoM +0.36%; pooled/trim improve; worst run +0.36% | PASS |
| device-6 source p50 | materially toward ~54 us | 105.376 to 52.496 us | PASS |
| per-ordinal max p50 | rank gradient contracts | 105.376 to 54.080 us | PASS |
| device-6 dynamic instructions | toward 0.906M control | 2.971M to 0.824M | PASS |
| static return geometry/SASS | unchanged | launcher geometry and pairwise SASS/cubin resource usage unchanged | PASS |

The matched cubin audit finds successful hot-path SASS changes only in source.
H7168 keeps REG74; H256 rises REG74 to REG76.  Return/count/prefix/combine are
strictly identical.  Plan changes one assertion line-number immediate only;
barrier SASS is identical after the timeout fix.  Static source code grows by
32 instructions, but dynamic device-6 instructions fall 72.28%, while SM,
DRAM and LSU remain unsaturated.  This is the intended algorithmic removal of
redundant scan work, not an occupancy or bandwidth optimization.

No tail claim is accepted because the formal B side contains a 2.169-ms
source-H7168 outlier.  No source/return fixture sum is called an EP or model
step.  O078 does not authorize a TMA pipeline, queue, block geometry, ABI,
buffer or fusion change.  The next local work must first freeze a transfer-
matrix/compute-interference measurement contract; real D>1 Gin/RDMA remains
an external gate.

## Measurement preparation O079 — transfer-matrix fixtures and source/return byte scopes

Status: **FUNCTIONAL_FIXTURE_PASS / PERFORMANCE_PENDING**.

O079 is not a new device-kernel optimization.  It makes the next C100
measurements falsifiable by separating four transfer shapes that the old
symmetric volume fixture could not distinguish:

- `fanout`: owner 0 sends 1,024 moved records to each other egress.
- `fanin`: owners 1--7 each send 1,024 moved records to egress 0.
- `mesh`: every owner sends 128 moved records to every non-owner egress.
- `rot1` and `rot4`: every owner sends 896 moved records to one rotated
  egress.

Each fixture is fixed at G8/D9/N2048-per-rank/K4/C256/Pcap7168 and 7,168
moved records.  This keeps scan count, destination count, top-k, channel
count, capacity and arena geometry fixed while changing only the owner-to-
egress matrix.  The benchmark schema now records the full matrix plus
outgoing and incoming sums.  For source shuffle, rank-local logical bytes are
the outgoing row sums because the source owner performs those moves.  For
return-unshuffle, rank-local logical bytes are the incoming column sums because
the proxy egress receives and returns those records.

All ten named matrix fixtures pass true 8-GPU LSA source and return
functionality.  No performance result is accepted from O079 yet: the run had a
GPU0 co-tenant and the visible GPU model string was `NVIDIA L20X`, so the data
cannot satisfy the no-profiler contract.  Ruff and pyrefly are missing in the
current environment; their absence is a tooling gap, not a pass.  Full-repo
formatting is deferred to a separate mechanical commit if it is needed.

## Measurement preparation O080 — compute interference harness

Status: **CODE_SMOKE_PASS / PERFORMANCE_PENDING**.

Hypothesis: source-shuffle or return-unshuffle may look acceptable in
isolation but lose usefulness if it competes with MoE GEMM for HBM/NVLink/L2
or SM issue resources.  O080 measures that interaction without changing
production kernels.

Single variable: add a default-off diagnostic mode to the existing C100
benchmark.  The production CUDA kernels, plan, source, return, buffer layout,
JIT specialization, capability bits and public API remain unchanged.

Modes:

- `none`: existing checked-adapter measurement; baseline-eligible only under
  the existing no-profiler and no-co-tenant rules.
- `compute-only`: same lifecycle and preallocated tensors, but the target
  window is only one BF16 GEMM.
- `concurrent`: launch the BF16 GEMM on an independent CUDA stream, then call
  the checked adapter, then synchronize the compute stream.

Fixed GEMM:

```text
[1024,7168] BF16 @ [7168,7168] BF16 -> [1024,7168] BF16
```

Predicted observations if there is harmful interference:

1. The concurrent window exceeds `max(stage-only, compute-only)` by a material
   margin after start-skew checks.
2. Nsys shows actual temporal overlap rather than launch serialization.
3. If NCU is later justified by exposed critical-path evidence, the affected
   kernel dossier must show a concrete resource limiter; no inference from
   utilization alone is accepted.

Reject the experiment as performance evidence if GPUs are shared, clocks are
unstable, the visible hardware does not match the target environment, or Nsys
cannot prove overlap.  In those cases the code may remain as a diagnostic
harness only if it is default-off and does not perturb `none`.

Implementation result: the benchmark now has `--interference-mode`, fixed
H7168 GEMM allocation, schema v4, baseline ineligibility for diagnostic modes,
and compute-only logical bytes set to zero.  Source compute-only, source
concurrent, return concurrent, and default `none` source matrix smokes pass
with one steady iteration.  These are path/sanity checks only.  The first
compute-only smoke exposed a misleading logical-bandwidth print based on moved
stage bytes; that was fixed before acceptance by adding separate
`stage_logical_bytes_*` reference fields and printing `n/a` for compute-only.
Per-rank raw records include `compute_stream_id` so a later Nsys run can match
the diagnostic GEMM stream explicitly.

## Scope decision O081 — C105 bundle is evidence packaging, not runtime proof

C105 now emits a real-cluster validation bundle from the existing Hybrid CPU
oracle.  This is intentionally not a profiler or performance artifact.

The single source of truth for scheduling remains
`rail_balance_hybrid_reference.build_hybrid_rail_schedule`.  The bundle records
count, quota, moved bytes, group counts, proxy requirements, expected Gin puts,
and expected payload bytes.  Runtime fields stay unavailable and the evidence
label remains `REAL_HYBRID_RUNTIME_UNTESTED`.

The capacity case is deliberately fail-closed.  It is useful because it proves
the network package will carry an expected capacity rejection case, but it does
not exercise a live multi-node barrier or Gin path yet.

## Scope decision O082 — live C105 runner is correctness packaging, not an optimization

The new D>1 runner changes no production kernel or runtime path.  Its purpose is
to make the first Rail/Gin correctness experiment reproducible and falsifiable,
not to improve or measure speed.

Off and force channel geometry are not forced equal: the off handle is observed
as its own baseline, constructor capacity uses the channel-invariant C=1 oracle,
and successful force evidence records the force handle's actual channel count.
This avoids accidentally changing the operator merely to simplify a test.

The output intentionally records null QP utilization, NIC bytes, and wait
cycles because the current implementation exposes no trustworthy runtime
counters.  No profiler was run and no local timing was collected in this
checkpoint.  Performance work starts only after a live correctness result and
a separately defined no-profiler/nsys/ncu evidence chain.

## Scope decision O083 — build/codegen preflight is not runtime warmup or performance

The C105 preparation entry rebuilds the current extension and compile-checks
one representative main dispatch/combine force/legacy pair in an empty cache.
It changes no production source and measures no performance.

The rejected first draft reran the full C080 compile matrix and repeated its
case counts.  Those extra cubins could not warm the exact production keys and
would create a second maintenance source, so the accepted version keeps one
D2xG8/H7168/K8 pair.  Full codegen coverage remains with the original tests.

The resulting manifest says `full_force_runtime_cache=false`.  Production keys
also include the final token capacity, SM/QP count, timeout and other geometry;
the remaining planner, local barrier, source shuffle, epilogues and return-
unshuffle are prepared only by the live force buffer.  Therefore the real
cluster sequence is fixed: run `balanced` first with final parameters, then run
the skew and capacity cases.  No build time, compile time, register count, or
cache hit is interpreted as an end-to-end speedup.

## Scope decision O084 — remove duplicated ABI facts before performance edits

The force arena prefix and forward-metadata layout were repeated across host,
dispatch, and combine.  This was a correctness and auditability risk, not a
measured bottleneck.  Commit `8e8df41` centralizes them as compile-time facts;
the generated representative kernels retain their prior register, stack, and
zero-spill resources.

No timing claim is made.  The remaining hot-path candidates are deliberately
separate experiments:

1. cache per-geometry prepared kernels and fixed plan storage;
2. fuse moved-token staging into the owner pass so hidden data is read once and
   proxy Gin requests can begin earlier;
3. merge or hoist dynamic gates without weakening collective fail-close.

These changes can alter overlap, lifetime, and failure semantics.  Each must
start from an unprofiled baseline and an Nsys critical path; NCU is used only
for a kernel already shown to contribute exposed time.

## Scope decision O085 — policy selection belongs only in the planner

The accepted extension exposes three policy choices without multiplying the
data path:

- `all`: exact balance across all rails, preserving the old force plan when
  threshold is zero;
- `active`: balance only among rails already carrying that destination;
- `adaptive`: begin with active rails and recruit an inactive rail only when
  the current discrete peak exceeds the next-rail target by more than the
  threshold.

The same threshold also bypasses movement when the observed peak is no more
than that percentage above the selected-set target. This is an integer count model, not
a measured time model; it intentionally adds no bandwidth constants, device
telemetry, heap object, or fallback to the kernel. A later `auto` policy may
replace the count threshold only after real Gin/RDMA evidence.

No performance comparison was accepted in this checkpoint because GPUs 0/1
had co-tenants. Functional CUDA/LSA/vnode and spill-free codegen results prove
semantic viability only. The first useful comparison matrix on idle hardware
is `all/0`, `active/0`, `active/20`, and `adaptive/20`, with profiler-free
latency and moved bytes recorded before Nsys/NCU attribution.

## Scope decision O086 — DeepEP V2 off is the only production baseline

The final optimization question is intentionally narrower than the local
policy microbenchmarks: does the complete public force dispatch/combine path
beat native DeepEP V2 Hybrid with `rail_balance='off'`?  Local `all/0`, vnode,
and checked-LSA measurements remain diagnostic controls and cannot answer that
question because they omit real Gin, QP, NIC, and fabric behavior.

The new D>1 benchmark changes only the constructor mode between alternating
fresh-buffer blocks.  Input, topology, precision, SM/QP selection, route,
source, extension and runtime remain fixed.  `balanced` measures fixed
overhead, `one_hot` measures maximum redistribution opportunity, `two_hot`
measures a narrower active set, and separate all/active/adaptive runs expose
the connection-spread versus tail tradeoff.  Every policy job contains its own
DeepEP V2 off blocks, so results never compare different machines or commits.

The decisive ratio is baseline median divided by candidate median.  A value
above one is only accepted when correctness, identity, environment, sample
depth and direct-unwrapped gates all pass.  A losing balanced or low-skew case
is not hidden: it becomes evidence for bypass/auto policy.  Nsys and NCU may
explain a result later, but cannot replace this profiler-free A/B truth.

No kernel optimization or speedup claim was made in O086.  Real numbers remain
blocked on C080-H/C110 hardware.

## Scope decision O087 — production wiring is not a tuning result

Endpoint one-hop now reaches the production-shaped prepare/finish/commit
transaction, but this checkpoint changes selection and ownership only.  The
persistent Hybrid dispatch kernel, warp split, TMA tile, QP count, and receiver
forwarder are unchanged.  Eight-GPU LSA correctness was rerun while unrelated
`mmunlearner` processes occupied the GPUs, so no timing sample was retained.

The next accepted optimization experiment starts only after a full hop-aware
dispatch/combine vnode round trip.  It will compare profiler-free native
`off`, `legacy_exact`, and `one_hop` first; Nsys/NCU will then target exposed
source-shuffle or receiver-forward time rather than this control-plane wiring.

## Scope decision O088 — vnode closure proves semantics, not speed

The endpoint-aware vnode reuses the existing scale-out, destination expert,
return and combine stages.  Only route materialization at pack/demux differs,
so the passing 4x2 run proves that source-forward reaches its final target
without a second destination forwarding step and that combine follows the
inverse route.  It is still an LSA-backed virtual network and contributes no
accepted latency or bandwidth claim.

The hop pack currently scans the fixed-capacity record table.  That is a
deliberately simple correctness implementation, not a proposed production hot
path.  It will not be tuned in isolation: selective two-hop must first share
the same resolution ABI, after which profiling can decide whether compaction,
segment merging, or a fused producer is justified.

## Scope decision O089 — adaptive correctness precedes planner parallelism

Selective two-hop now shares the one-hop record/resolution ABI and completes an
eight-H200 vnode round trip.  The current planner uses one CUDA lane and
rescans records and Rails after each accepted move.  This makes tie-breaking
deterministic and the cap easy to audit, but it is not assumed fast.

No latency result was accepted in this checkpoint.  The observed reduction of
the synthetic diagonal pair peak from 8 to 4 is a scheduling invariant, not an
end-to-end speedup.  Before changing the algorithm, the unified benchmark must
measure exposed planner time relative to source shuffle and forwarding.  Only
if planner time is material will the next experiment parallelize candidate
evaluation or batch third-Rail moves; otherwise the simpler implementation is
retained.

## Scope decision O090 — unified tests are not an optimization result

The reference/vnode/multinode wrapper changes measurement organization only.
Its vnode diagonal case proves that the bounded plan and inverse data path
agree: 16 of 32 payload units use two-hop, pair peak falls from 16 to 8, and
the report records 16,384 extra local-hop bytes. These are oracle/diagnostic
quantities, not measured network speed.

The multinode backend still compares every candidate with native DeepEP V2
`off` using the existing ABBA+BAAB, rank-max profiler-free protocol. Until a
real D>1 run exports actual path counters, the report labels its path split as
the Python endpoint-oracle expectation. No kernel tuning is accepted from
this checkpoint.

## O091 — batch adaptive moves while the old Rail remains critical

Hypothesis: adaptive is dominated by an avoidable full-record-table rescan for
every accepted third-Rail copy. The prediction was linear growth in accepted
moves on top of each table scan, low SM/DRAM utilization, and large instruction
count. Nsys and NCU confirmed all three: the N128 kernel was 33.749 ms, executed
9.28M instructions, used 0.03% SM throughput and did not saturate memory.

Single variable: after selecting the best `(old egress, new egress,
destination, hop cost)`, migrate the deterministic record prefix while its
pair/source relief remains admissible. The batch stops when the first relief
gap closes or the old Rail drops below another hotspot, and is also bounded by
the remaining two-hop cap and proxy capacity. Endpoint constraints, threshold,
penalty, dense slot materialization and one-hop are unchanged.

Result: N128/N512 adaptive private-API medians improve 9.658x/34.013x;
unchanged one-hop is 1.004x/1.001x. Matched Nsys kernel time improves 9.888x
and targeted NCU instructions fall 11.419x. Correctness and sanitizer gates
pass. This is accepted as a planner optimization but not a multinode speedup.

## O092 — reject per-record peak caching

Hypothesis: one-hop repeatedly scans all Rails for each endpoint candidate.
Hoisting the current pair/source peak outside the candidate loop should reduce
work without changing the score. Profiler-free N128/N512 one-hop remained
2.476/13.832 ms and adaptive remained 3.496/18.889 ms, within control noise.
The source edit was reverted; no complexity was retained.

## O093 — validate fixed records by owner lane

Hypothesis: a material fraction of the serial kernel initializes state and
scans the fixed `[owner, token, topk]` table before greedy assignment. Owners
have disjoint output rows, so one warp lane can validate and count each owner
without atomics or changing assignment order.

Single variable: use the already-launched warp to initialize arrays and scan
one owner's records per active lane; reduce only the total unit count, then let
lane 0 execute the unchanged planner. Matched N128/N512 one-hop medians improve
1.372x/1.254x and adaptive 1.244x/1.174x. Nsys one-hop kernel time improves
1.397x and NCU instructions 1.363x. Both vnode paths and all three focused
sanitizers pass. The optimization is accepted for the planner only.

## O094 — retain the source-shuffle gap as an open measurement target

The first full C100 hop-aware source run exposed a correctness boundary before
it exposed a tuning target: retained destination copies are densely staged,
so their count can exceed the legacy moved-copy proxy capacity. The planner
now fails closed before shuffle when either moved or retained staging exceeds
capacity; the benchmark reserves the simple worst-case `N*K`. This is a
capacity contract, not an optimization, and no second buffer-size field was
added before measurements justify it.

After that fix, one cold source smoke measured 496.658 us for legacy,
3,242.746 us for one-hop, and 2,838.342 us for adaptive. These are dirty-tree,
single-sample diagnostic numbers on devices which `nvidia-smi` identified as
L20X, so none is accepted as a performance claim. They do identify the next
falsifiable target: determine whether hop source-shuffle scans the dense record
table or performs avoidable per-copy lookup work. The next edit is forbidden
until a clean profiler-free distribution plus Nsys and, if needed, NCU locate
the exposed time. Return smokes (269.417/416.797 us) are likewise diagnostic.

## O095 — precompute retained staging prefixes

Hypothesis: the hop source-shuffle long tail is integer/address work, not TMA
payload bandwidth. Nsys isolated device 0 at 3,081.955 us while the other seven
devices took 10–13 us. The hot source line recomputed the retained staging
offset for every copy by summing all preceding channel/destination counts.

Single variable: the hop planner now writes the exclusive retained prefix for
each `(egress, channel, destination)` group into the existing, otherwise-unused
hop specialization of `owner_channel_prefix`. Source shuffle replaces the
nested rescan with one indexed load. The legacy specialization is unchanged;
no output tensor, descriptor, resolution field, or persistent buffer byte was
added.

Matched 3-warmup/20-sample checked-source diagnostics:

| Mode | Before median / p95 (us) | After median / p95 (us) | Speedup |
| --- | ---: | ---: | ---: |
| one-hop | 3225.273 / 3246.491 | 129.304 / 148.484 | 24.943x |
| adaptive | 2878.798 / 2918.892 | 674.253 / 704.638 | 4.270x |

Matched Nsys one-hop device-0 source kernel time falls from 3,081.955 to
39.488 us (78.047x); aggregate duration across all eight device invocations
falls from 3,162.436 to 120.416 us (26.262x). The stage is now dominated more
by fixed launch/gate overhead. These are single-node checked-adapter results,
not Gin/RDMA or end-to-end speedup.

Rejected evidence and failures are material:

- the first edited launcher still passed a null retained-prefix pointer. The
  strict kernel returned immediately, producing a false 0.113 ms median; C100
  status/guard checks passed, but full vnode correctly failed. The pointer was
  wired through both submit paths, the false samples were rejected, and both
  one-hop/adaptive full round trips then passed;
- NCU application replay completed pass 1 but could not reproduce the exact
  kernel set for pass 2. No `.ncu-rep` was generated and no metric claimed;
- full-vnode Compute Sanitizer ran the functional round trip to PASS but its
  NCCL initialization produced `NoKernelImageForDevice` under child-process
  instrumentation. That run is rejected as tool/environment incompatible.
Focused planner memcheck and initcheck remain zero-error.

The committed `bfffe7c` tree was then measured with the profiler disabled for
10 warmups and 100 steady iterations. All three reports passed their automatic
baseline-eligibility gates:

| Mode | median / p95 / p99 (us) | population stddev (us) |
| --- | ---: | ---: |
| legacy | 491.699 / 503.357 / 505.909 | 6.453 |
| one-hop | 122.753 / 145.578 / 177.898 | 13.187 |
| adaptive | 667.974 / 703.740 / 716.065 | 36.323 |

This closes O095. The source stage alone is 4.01x lower for one-hop than the
legacy checked adapter in this fanout case. Adaptive is slower because it
executes actual extra peer movement. Neither result includes plan/finish time,
real RDMA, or the public Hybrid end-to-end path.

## O096 — replace dense channel scan with exact minimum-load bitsets

Before editing, the clean one-hop report measured 203.951 ms median in
`finish`, and Nsys measured `rail_balance_hop_plan_impl` at 155.580 ms median
over 16 invocations, 97.7% of captured kernel time. Source correlation found
the planner scanning all 256 channel counters for every active copy.

The original choice is lexicographic `(group load, circular distance from
source channel, channel)`. Because the algorithm always increments a
minimum-load channel, loads for one `(egress,destination)` differ by at most
one. The exact candidate set is therefore represented by eight 32-bit words
for 256 channels. `owner_remaining`, dead after endpoint assignment, holds the
number of channels still at the current minimum; `group_prefix`, not consumed
until the final prefix pass, temporarily holds the masks. No allocation,
output field, public ABI, or approximate choice was introduced.

The first diagnostic after the single edit reports 44.445 ms median `finish`
(3+20, dirty-tree diagnostic), 4.59x below the clean pre-edit boundary. Nsys
measures the planner kernel at 34.928 ms median, 4.45x below the matched
155.580 ms kernel and still 92.3% of captured kernel time. The source stage
remains unchanged within noise at 125.733 us median.

Correctness is exact rather than invariant-only: the existing greedy oracle
passes, plus a new 256-channel/513-token case crosses all eight words and
forces one complete minimum-level reset. GPU planner 11/11, both one-hop and
adaptive vnode round trips, and planner memcheck/initcheck/synccheck pass with
zero errors. This checkpoint is useful but not a final planner design; the
remaining sequential endpoint assignment is the next measured target.

Commit `989c9c7` then passed clean-tree 10+100 eligibility. One-hop `finish`
is 44.436 ms median (p95 44.550 ms), confirming the 4.59x reduction from the
203.951 ms pre-edit boundary; its source stage is 113.268 us median. Adaptive
`finish` falls from 575.660 to 416.741 ms (1.38x), while its source stage is
681.248 us. The adaptive residual search is therefore a separate dominant
algorithmic cost and is not hidden by calling O096 complete for all modes.

## O097 — rejected warp-parallel endpoint scoring

NCU basic on one stable N=1024/K=8/C=256 invocation used kernel replay for ten
passes. It reported grid 1, block 32, 80 registers/thread, 27.97 ms replay
duration, 1.56% achieved occupancy, and about 0.01% SM and memory throughput.
The report hash is
`e21ea258d194135e33d99133b2a994a2a6c9f9b433047a8fd765a7b6000a002c`.
NCU duration is not compared with the profiler-free result.

The tested single variable kept record order but assigned one candidate Rail
to each warp lane and reduced the existing five-field lexicographic score.
GPU planner 11/11 stayed exact. The private API median improved only from
22.167 to 20.950 ms (5.8%), while the more representative C100 `finish`
boundary regressed from 44.445 to 48.860 ms (9.9%). Twenty-five shuffle/reduce
operations per record cost more than evaluating at most eight Rails serially.
The production edit was fully reverted; only the reusable benchmark channel
parameter and these negative results remain.

## O098 — rejected shared load-state cache

Targeted NCU on the same stable invocation reports 2.91M executed
instructions, 93.03% scheduler cycles with no eligible warp, 14.35 cycles per
issued instruction, and 7.3 cycles (51.08%) in long-scoreboard stalls. L1/TEX
and L2 hit rates are already 85.88% and 87.98%; branch efficiency is 99.76%.
The report hash is
`fe619582250b25cf23f660aeb8e927a4001afdfd1b23f89ce44a32595b3ca1e4`.

The tested edit cached the at-most 1,056 pair/source counters in 4.125 KiB of
shared memory while continuing to update the original outputs. Exact tests
passed 11/11, but the N=1024/C=256 private median changed from 22.167 to
22.257 ms and p95 regressed from 22.199 to 22.935 ms. Existing cache locality
is sufficient; shared addressing plus duplicate stores add more work than
they hide. The production edit was fully reverted.

## O099 — skip validated packed-record padding

The record materializer already writes each token's remote destinations into a
dense prefix and leaves trailing `target_mask=0` slots. The planner nevertheless
visited every K slot in its endpoint, adaptive-candidate, adaptive-migration,
channel, and proxy passes. The edit validates the packed-prefix invariant once;
an active record after padding fails closed. Each later scan then jumps from
the first padding slot directly to the next token. Valid-record order, scores,
tie-breaking, output layout, and buffer ownership are unchanged.

The N=1024/K=8/C=256 private one-hop median falls from 22.167 to 12.165 ms
(1.82x); the same post-edit adaptive median is 18.540 ms. C100 3+20 diagnostics
reduce `finish` from 44.445 to 41.448 ms for one-hop and from 416.741 to
399.364 ms for adaptive. Source-stage timing stays within its prior range.

The patch adds no allocation or metadata. GPU exact tests pass 11/11 including
explicit gap rejection, both full vnode round trips pass, and focused memcheck
and initcheck report zero errors. Clean 10+100 distributions follow the code
commit before the ratios are treated as accepted profiler-free evidence.

Commit `eedcdad` passed clean 10+100 eligibility. One-hop `finish` is
41.372 ms median (p95 41.523 ms), 7.4% below the clean 44.436 ms predecessor.
Adaptive is 399.294 ms (p95 399.510 ms), 4.4% below 416.741 ms. Source-stage
code did not change; its observed 134.296/671.194 us medians are recorded as
run-to-run context rather than attributed to the planner edit.

## O100 — cache exact adaptive load peaks per iteration

Post-O099 Nsys still measures the adaptive planner at 306.323 ms median and
98.1% of captured kernel duration. Its report hash is
`da2ee7d0e79b1eaa0f622ed56f5bd5ebcc8779de0dcf7a81456092bad1d28adc`.
Source inspection shows each candidate copy and each possible third Rail
rescanning all Rails to reconstruct pair/source peaks before and after one
move.

Each adaptive iteration now computes `(maximum, second maximum, maximum
count)` once for source load and each destination. The maximum after an exact
old→new unit move is then O(1), including tied maxima. Dead
`owner_remaining` workspace holds destination peak triples until channel
materialization reuses it. Candidate order, net gain, threshold, cap, batch
selection, and tie-breaking are unchanged; G<3 skips an impossible third-Rail
search.

Private N=1024/C=256 adaptive median improves from 18.540 to 17.199 ms; one-hop
stays at 12.13 ms. C100 adaptive 3+20 `finish` improves from the clean
399.294 ms predecessor to 308.661 ms (1.29x). GPU exact tests pass 11/11, the
adaptive vnode round trip passes, and focused memcheck/initcheck report zero
errors. Clean 10+100 evidence follows the code commit.

Clean commit `1a01f44` 10+100 evidence accepts O100:

```text
adaptive finish median / p95   399.294 / 399.510 -> 308.736 / 308.977 ms
median speedup                                           1.293x
population CV                                             0.038%
source median                                            666.255 us
baseline collection eligible                                  true
```

The source stage is reported but not attributed to this planner-only edit.
Report SHA-256 is
`68d3968621157d97d01fd6deca9d43f64d2fdea80a8591dea0f45faf839f7266`.

## O101 — warp-parallel adaptive record discovery

Post-O100 Nsys leaves one planner invocation at 237.244 ms and 99.6% of the
selected rank-0 GPU time. The C100 cap case runs roughly eight residual
iterations; each iteration searched all packed records on lane 0 and then
searched them again to migrate a prefix.

Only read-only candidate discovery is now distributed: each warp lane scans
packed token rows, then one exact lexicographic reduction chooses the same
global candidate. Lane 0 retains endpoint assignment, load mutation, batch
migration, channel assignment, and output writes. No allocation, metadata,
public API, score, cap, threshold, or tie rule changes.

```text
private adaptive median                 17.199 -> 11.791 ms
C100 adaptive finish diagnostic        308.736 -> 59.515 ms
diagnostic speedup                                  5.19x
one-hop microbenchmark                           unchanged
GPU/vnode/sanitizer gates                              PASS
```

This is preliminary dirty-tree evidence. A clean 10+100 run after the code
checkpoint decides acceptance.

Clean commit `7b98f83` accepts O101. One-hop is a stable control at 41.510 ms
median versus 41.372 ms before the edit. Adaptive `finish` is 59.493 ms median
/ 59.689 ms p95 versus 308.736 / 308.977 ms at O100, a 5.19x median speedup.
Source-stage medians are 119.918 us and 642.279 us respectively and remain a
separate data-plane boundary. Both eligibility gates pass.

## O102 — exact singleton endpoint bypass

The C100 channel sweep shows that C=1 through C=256 changes private one-hop
time by less than 1 ms; endpoint assignment dominates. A record whose
`target_mask | owner_bit` has one bit has no Rail choice. O102 takes that bit
directly after the same reservation-aware capacity check and leaves all
multi-candidate scoring untouched.

```text
C100 one-hop finish diagnostic    41.510 -> 34.669 ms (1.197x)
C100 adaptive finish diagnostic   59.493 -> 52.918 ms (1.124x)
off-diagonal control              12.143 -> 12.177 ms (noise)
```

Exact tests, both vnode round trips, and focused memcheck/initcheck/synccheck
pass. The dirty-tree diagnostics decide to keep the candidate pending clean
10+100 evidence.

Clean commit `7b0466f` accepts O102. One-hop `finish` is 34.701 ms median /
34.898 ms p95 (1.196x below O101); adaptive is 52.919 / 52.989 ms (1.124x).
Source-stage medians remain separate at 116.921/665.269 us. Both reports pass
baseline eligibility.

## O103 — reuse pair/source peaks per multi-candidate record

For a fixed record, current pair/source loads do not change while its endpoint
candidates are compared. O103 scans each load vector once and evaluates each
candidate as `max(current_peak, candidate_load + 1)`. This is algebraically
identical to rescanning all Rails per candidate.

```text
C100 rot1 one-hop diagnostic    179.096 -> 173.004 ms (1.035x)
private one-hop control          12.177 -> 12.078 ms
```

Exact, vnode, and focused sanitizer gates pass. The modest result is retained
because the code is smaller and exact. It also closes this line of tuning: the
remaining rot1 cost is sequential per-copy assignment and requires chunk/group
planning, not another local score cache.

Clean commit `6a6cb2c` closes O103 at 173.023 ms median / 173.222 ms p95 on
rot1 C100 10+100. Population CV is 0.068% and eligibility passes. The source
stage is 284.318 us median, so further serial-score tuning is explicitly
stopped in favor of group/chunk planning.

## O104 — measured chunk/group planner direction

This checkpoint changes no CUDA. It tests whether endpoint decisions can be
coarsened before replacing the serial exact loop. On the C100 rot1 endpoint
matrix, Python `chunk_size=1/8/32` takes 608.358/74.607/17.038 ms. All three
produce pair peak 1792; source peaks are 11193/11192/11168 and local-forward
units stay 50176.

A separate 128-seed small random sweep prevents accepting only the favorable
large case. Relative to chunk 1, chunk 8 has worst pair/source peak changes of
4.37%/3.46%; chunk 32 reaches 12.66%/13.88%. The accepted design therefore
keeps exact chunk 1 as oracle, uses a conservative minimum chunk, and bounds
decisions per large endpoint group instead of applying chunk 32 globally.

Full CPU offload is rejected: every fresh route would require D2H, host work,
H2D, and a synchronization before dispatch. CPU/background work is limited to
low-frequency policy updates; per-route histograms, quotas, and slot
materialization remain GPU-resident and may overlap the previous microbatch.

### O104 GPU prototype

The private API keeps `planner_chunk_size=1` exact. With chunk 8 it aggregates
one-hot `(owner,destination,target)` records, limits large groups to 32
decisions, and uses a deterministic per-worker channel prefix. Multi-target
payloads retain exact endpoint assignment.

Profiler-free N=8192/C=256 10+100 results:

```text
                                      one-hop median / p95
exact chunk 1                          92.678 / 92.801 ms
chunk decisions only                   88.633 / 89.315 ms
direct channel mapping                 87.281 / 87.993 ms
8-owner deterministic prefix           63.918 / 64.045 ms
32-lane deterministic prefix           14.483 / 14.599 ms
32-decision large-group budget          13.586 / 13.597 ms

adaptive chunk 1                       89.563 / 89.639 ms
adaptive chunk 8                       15.608 / 15.739 ms
```

Thus the accepted private prototype is 6.82x faster for one-hop and 5.74x for
adaptive. It does not use nondeterministic slot atomics. GPU tests pass 12/12;
focused memcheck, initcheck, and synccheck each report zero errors.

Nsys measures the pre-prefix chunk-8 planner at 88.518 ms and the 32-lane
version at 14.407 ms. NCU basic on the former records a one-block/one-warp
launch, 1.56% achieved occupancy, and roughly 0.02% SM/memory throughput.
NCU replay duration is not used as an end-to-end number. Report hashes:

```text
pre Nsys   ac022a735299773557f43cf5587d0aa2ea12c97da73deef5b20bb6dfac4b18c1
NCU basic  5726f9268a8ed6119c19dcb28838ea3404d014b02d339a0ac0cd148be0659514
post Nsys  b067a076a93e3125a29c6bb2710bb28f635e7210b618cbc625fdf6d912016d2b
final A/B  f6e0e4bc6dd5de8b8357da44e3502c5018a01a23eca39eb86d81ec2e857ad954
```

An initial Nsys command included a disallowed `rm -f` cleanup and was rejected
before execution. The retry used a unique report name; this is a command-safety
failure, not a profiler or kernel failure.

## O105 — asynchronous record boundary and multi-block evidence

The first production integration removes the eager status readback and stream
synchronization after endpoint record extraction. This does not pretend that a
fresh plan can finish after its dispatch: it moves the already-independent
record work across Gate #1 and leaves one true dependency at `finish`/Gate #2.
Full per-route CPU offload remains rejected because it adds D2H, CPU planning,
H2D, and a required synchronization.

Targeted NCU on the accepted chunk-8 planner records one block of 32 threads,
2,035,063 executed instructions, 0.08 eligible and issued warps per scheduler,
92.22% cycles with no eligible warp, and 57.3% long-scoreboard warp cycles.
Memory throughput is only 266.87 MB/s (0.18% of peak), L2 hit rate is 98.18%,
and NCU estimates only about 3.224% local opportunity from coalescing. The
remaining 13--16 ms is therefore a latency-bound single-warp dependency chain,
not HBM bandwidth pressure.

A lane-private endpoint histogram was tested as a single-variable experiment.
It passed all 12 GPU cases but measured 13.648 ms one-hop and 15.712 ms
adaptive, versus 13.586/15.608 ms before; it was fully reverted. This falsifies
another histogram-local tweak and commits the next experiment to the explicit
multi-block assignment/count/prefix/finalize split documented in the design.

Evidence:

```text
.cache/rail_balance/hop-aware/ha070-onehop-n8192-fast-targeted.ncu-rep
/tmp/rail-hop-ha070-private-hist.json
private-hist SHA256 246fdf215f9e08ff2b0367b90b6b94505298da002895fcf0dbc9e6176477ae
```

## O106 — parallel record pre-count

Hypothesis: the first of several full record-table passes is a material part of
the one-warp latency chain. Only that pass is moved to `G*C` one-warp blocks;
endpoint decisions and final slots remain byte-for-byte on the prior path.

```text
                                  before      after       speedup
one-hop N8192/C256 median         13.586 ms    9.198 ms      1.477x
adaptive N8192/C256 median        15.608 ms   11.177 ms      1.396x
one-hop after p95                              9.206 ms
adaptive after p95                            11.238 ms
```

Matched Nsys attributes 16.672 us to the new pre-count and 9.098 ms to the
remaining precounted planner. Thus the experiment replaces roughly 4.4 ms of
serialized work with 0.017 ms and is accepted; it also falsifies pre-count as
the remaining bottleneck.

```text
.cache/rail_balance/hop-aware/ha070-onehop-n8192-precount.nsys-rep
/tmp/rail-hop-ha070-precount-onehop.json
/tmp/rail-hop-ha070-precount-adaptive.json
one-hop SHA256 d3fbdcc9451e1a322b907710dd1fcc3c6913e01aad80c68ac2f47120da32476c
adaptive SHA256 dc12f779b42fd4ca48ba3a5280dad4b7db0a687b4d59ec0a3180c049fa7432c4
```

## O107 — parallel one-hop assignment and static slots

Hypothesis: after O106, repeated per-record assignment and slot materialization
inside one warp dominate the remaining 9.098 ms. The implementation preserves
one ordered coarse decision but assigns dense work to the predeclared CUDA
ownership map. Five launch boundaries replace unsafe cross-block barriers.

Profiler-free N8192/C256 one-hop 10+100:

```text
O106 precounted median / p95       9.198 / 9.206 ms
O107 final median / p95            0.429 / 0.435 ms
incremental speedup                           21.45x
O104 accepted chunk-8 median                  13.586 ms
cumulative speedup                            31.68x
```

Matched Nsys measures a 0.489 ms projected GPU range. The remaining coarse
decision is 251.072 us; pre-count is 15.392 us; endpoint prefix, assignment,
group count, group prefix, and finalize are 4.576, 14.592, 17.472, 20.320, and
19.104 us. Allocator fill kernels account for 16.832 us. This is no longer a
single-warp multi-millisecond data-materialization path.

The exact chunk-1 control measured 96.034 ms in the final session versus
92.678 ms in the older O104 session. Compile-time specialization did not change
that value, so the cross-session 3.6% difference is not attributed to this
optimization. Exact remains a correctness oracle, and O107's accepted evidence
is the paired O106-to-O107 fast path plus downstream LSA/vnode correctness.

```text
/tmp/rail-hop-ha070-parallel-specialized.json
SHA256 7d656074e99cd371965cde9b3f1cc1b1eee8af096d299079c1f199c7baf05bba
.cache/rail_balance/hop-aware/ha070-onehop-n8192-parallel-materialize.nsys-rep
SHA256 db90bcaae747593200d2231f1bf43df86781eb574a0031f6d370e7e8fc9e941a
```

## O108 — close the planner-to-source-shuffle capacity contract

The first C100 run that forced the private transaction onto chunk 8 failed in
the checked source shuffle with status 4. This falsified the assumption that a
deterministic rotated source channel was sufficient. Several owners may share
an egress and destination, so their individually legal stripes can overflow a
single physical channel.

The first replacement used separate retained and moved path-local ordinals.
Review falsified that repair with G2/C2/N2: one retained and one moved copy
both mapped to channel zero, so their shared final tail was 2 while physical
capacity was 1. That failed intermediate design is retained here because the
old oracle checked only each path-local slot.

Accepted change: one egress warp prefixes retained copies first and then moved
copies in one combined `(egress,destination)` sequence. The finalizer maps
combined ordinal `u` to `channel=u%C`; a moved copy uses
`remote_slot=u/C-retained[channel]`. This bounds the actual shared
`retained+moved` tail by `ceil(N/C)`. The obsolete adaptive rotated
materializer was deleted, removing roughly 130 lines and leaving one slot
mapping fact source.

The regression oracle now asserts both each remote slot and the combined
channel tail. GPU planner tests pass 13/13 with mixed chunk-1/chunk-8 random
coverage; focused memcheck/initcheck/synccheck report zero errors. C100
one-hop/adaptive one-iteration source smokes pass at 157.811/366.901 us, and
true EP8 LSA plus both vnode round trips pass. These zero-warmup samples are
correctness evidence, not the forthcoming committed-tree 10+100 performance
claim.

The first vnode run after O108 timed out because its path-distribution oracle
still used chunk 1 while the transaction used chunk 8. A single-rank assertion
failed and peers waited in the next collective. The planner itself had already
completed. Aligning the test oracle to chunk 8 is required before the vnode
round-trip result can be accepted.

After the oracle was aligned, vnode reached return demux and reported
`stage=5, code=36`: its internal GPU planner still rebuilt chunk-1
resolutions, which cannot consume proxy payload laid out by the chunk-8 source
plan. The adapter now uses the shared production chunk and parallel workspace.
This is a correctness alignment, not a claimed speedup.

## O109 — aggregate adaptive quota materialization

Hypothesis: after O107, adaptive remained slow because it performed residual
selection and final record/slot materialization in the old monolithic kernel.
Only the adaptive representation changed: one warp mutates the small
`[owner,destination,target,egress]` quota table and the accepted one-hop
materializer consumes it.

Profiler-free N8192/C256 chunk-8 medians:

```text
O106 adaptive predecessor             11.177 ms
O109 adaptive                          0.253 ms
speedup                                44.2x
O109 one-hop control                   0.463 ms
```

Clean C100 checked-adapter 10+100:

```text
                            finish median / p95   source median / p95
one-hop                     24.001 / 24.100 ms    115.401 / 160.635 us
adaptive                    23.116 / 23.221 ms    126.176 / 146.244 us
```

The private result is accepted only for its singleton-target workload. A
matched eight-rank C100 Nsys run contradicts extrapolation to shared
multi-target payloads:

```text
aggregate adaptive decision median      17.387 ms
aggregate adaptive decision share         68.1 %
source shuffle median                    25.344 us
```

C100's repeated destination experts become one payload record with target mask
`0b1111`. The aggregate quota path represents one target bit, so multi-target
records deliberately remain on the exact lane-0 path. This proves the next
single variable: aggregate equal `(owner,destination,target_mask)` payload
groups while preserving endpoint candidates and static-slot materialization.
NCU is deferred until Nsys shows a smaller kernel remains materially exposed.

Artifacts:

```text
ac866ee5f255d3a6c57a6a70096336254ac2091536918bf8b39102a44d01b882  c100-onehop-10x100.json
c064162bfac9f720f2a5b8065905a998611e486fc5c14c4998149ed2241c483a  c100-adaptive-10x100.json
80c991b7394938193324cf983f1bbfb64790db676f2c1bfe18ff1471fd7f6bd8  planner-adaptive.nsys-rep
cf22fd3fedf550c5cdeaa0b9b039bef00fe03d91df597530c4cb546e4eddf309  planner-adaptive.sqlite
e264a1352b856b8248a7173cfa763990840a8c993abf57798f87efe841bc38bf  c100-adaptive-0x3.nsys-rep
3ff92e73c7565881379e4fca9be5cb9b8400542690dfa0e26e84a56dcd5f3bb7  c100-adaptive-0x3.sqlite
```

The earlier explanation that C100 `finish` was probably host-gate dominated is
rejected and retained here. Nsys locates the exposed cost in the GPU planner.

## O110 — bound low-value adaptive rounds

Current-HEAD Nsys isolated the remaining adaptive cost in
`rail_balance_hop_plan_impl<0,true,true>`: one block, one warp, 100 registers
per thread, and 4.20--5.52 ms per C100 invocation. It represented 69.5% of
kernel time while source shuffle was 24.56 us median. A targeted private NCU
replay confirmed the structural limit: one active warp on one of 132 SMs,
1.56% achieved occupancy, only 14.56% scheduler cycles with an eligible warp,
78,184 executed instructions, 0.03% SM throughput and 0.02% memory throughput.
The kernel is a serial, dependency-heavy search rather than a bandwidth-bound
copy kernel.

The single variable was the maximum adaptive search budget per Rail. The old
`G*32` bound permits 256 full candidate rescans on G8. The accepted `G*8`
bound permits 64 rescans and raises the deterministic batch floor; candidate
scoring, threshold, two-hop cap, quota materialization and data movement are
unchanged. This is an algorithm-budget optimization, not a byte-for-byte
equivalent kernel rewrite, so plan quality is part of its acceptance contract.

Profiler-free 10+100 diagnostics:

```text
case          rounds/Rail  finish median  transaction median  source peak  pair peak  two-hop
volume                 32       6.080 ms          15.684 ms         1412        256     2048
volume                  8       2.778 ms          12.465 ms         1405        256     2048
rot1                   32       7.394 ms          16.901 ms         7285       1382     5872
rot1                    8       5.078 ms          14.264 ms         7194       1344     5236
```

The volume run repeated at 2.775 ms finish with an identical plan. After the
JIT identity was bumped to `rail_balance_hop_adaptive_decision_v8`, the same
source produced 2.812 ms finish and the identical 2048 two-hop / 1405 source
peak plan. These are local checked-adapter diagnostics, not NIC/RDMA claims.

The more aggressive `G*4` experiment passed correctness and reduced volume /
rot1 finish to 2.260 / 4.231 ms, but volume source peak regressed from 1405 to
1484 (+5.6%). It was rejected and reverted. This establishes 8 rounds/Rail as
the measured latency/plan-quality knee rather than selecting the fastest
synthetic point.

Validation after restoring 8:

```text
GPU endpoint/adaptive invariants, capacity and determinism   20/20 PASS
CPU reference hop semantics and randomized properties       12/12 PASS
random destination widths extended through D=32                  PASS
volume plan repeated identically                                  PASS
```

One failed C100 attempt reused a populated JIT directory and was correctly
rejected by the report builder's empty-pre-JIT identity assertion. It was not
counted; the rerun used a new empty cache. The first clean v8 report then
rejected its stale v7 JIT-name whitelist after the kernel had run; the
whitelist was updated without changing the measurement path. The next clean checkpoint must
repeat profiler-free volume/rot1 and then collect matched Nsys. A multi-warp
scan that preserves all 256 rounds remains a separate future experiment; it
is not mixed into O110.

Evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/current-adaptive-volume.nsys-rep
.cache/rail_balance/hop-aware/takeover-20260801/current-adaptive-multitarget-decision-basic.ncu-rep
.cache/rail_balance/hop-aware/takeover-20260801/current-adaptive-multitarget-decision-directed.ncu-rep
.cache/rail_balance/hop-aware/takeover-20260801/round-budget-8-adaptive-volume-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/round-budget-8-adaptive-volume-r2-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/round-budget-8-adaptive-rot1-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/round-budget-4-adaptive-volume-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/round-budget-4-adaptive-rot1-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/final-v8-adaptive-volume-10x100.json
```

## O111 — tree-reduce the adaptive candidate

Matched post-O110 Nsys still attributes a 1.706 ms median to the adaptive
decision kernel, about 61% of the 2.776 ms profiler-free `finish` median.
Source shuffle is only about 25 us.  The next single variable is therefore
the warp candidate reduction: the old loop made every lane exchange and
compare all 32 nine-field candidates.  A five-level tree now elects the same
strictly ordered winner and broadcasts lane 0.  Search rounds, scan order,
candidate scoring, cap, threshold, quota updates and materialization remain
unchanged.  Acceptance requires byte-identical plan tensors and lower matched
profiler-free/Nsys time; otherwise this patch is reverted before trying a
multi-warp scan.

Pre-change evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/clean-a9bd59a-adaptive-volume-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/clean-a9bd59a-adaptive-rot1-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/post-a9bd59a-volume.nsys-rep
.cache/rail_balance/hop-aware/takeover-20260801/post-a9bd59a-volume.sqlite
```

The dirty-tree 3+20 acceptance diagnostic kept both measured plans identical
to v8 and reduced the exposed phase without changing the policy:

```text
case       v8 clean finish   tree finish   v8/tree source peak   pair peak   two-hop
volume            2.776 ms      2.489 ms             1405/1405         256      2048
rot1              5.065 ms      4.827 ms             7194/7194        1344      5236
```

All path units, moved copies and extra local-forward units also match.  The
focused CUDA planner passed 20/20, the CPU oracle passed 12/12, and the 8-rank
plan transaction passed.  A first test command tried `pytest`, which is not
installed in the frozen runtime; the repository's direct runners were then
used without changing the tests.  A rot1 command typo (`c100_rot1_h256`) was
rejected by argparse before CUDA initialization and rerun with the actual
`c100_matrix_rot1_h256` case.  Neither failed command contributed timing.

Post-change diagnostic evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/o111-tree-adaptive-volume-3x20.json
.cache/rail_balance/hop-aware/takeover-20260801/o111-tree-adaptive-rot1-3x20.json
```

The clean 10+100 checkpoint measured 2.450 / 4.898 ms volume/rot1 `finish`
with the same plans.  Matched Nsys reduced the decision median from 1.706 to
1.472 ms.  Targeted NCU reduced executed instructions from 78,184 to 67,341
(-13.9%) and registers/thread from 100 to 96; NCU replay duration is not used
as endpoint truth.

## O112 — parallelize the per-round peak refresh

Before considering a 128-thread kernel, one final small exact experiment uses
the existing warp more evenly.  The old code made lane 0 recompute all D pair
peaks while the other 31 lanes waited, then made every lane redundantly scan
the same G source loads.  The candidate scan and serial round mutation remain
unchanged; destinations are now striped across lanes and lane 0 broadcasts
one source peak.  The v9 plan must remain identical.  A result within normal
run noise is rejected rather than followed by a larger multi-warp refactor.

Clean O111 evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/clean-28791cd-adaptive-volume-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/clean-28791cd-adaptive-rot1-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/post-28791cd-volume.nsys-rep
.cache/rail_balance/hop-aware/takeover-20260801/post-28791cd-adaptive-multitarget-directed.ncu-rep
```

The dirty-tree 3+20 result passed the exact-plan gate and improved both
structures:

```text
case       O111 tree finish   O112 peak finish   source peak   pair peak   two-hop
volume              2.489 ms             2.215 ms          1405         256      2048
rot1                4.827 ms             4.694 ms          7194        1344      5236
```

Path units and extra local-forward units remain identical.  The focused CUDA
planner passed 20/20.  O112 is retained for a clean 10+100 checkpoint; a
larger multi-warp rewrite remains deliberately out of tree.

Diagnostic evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/o112-peaks-adaptive-volume-3x20.json
.cache/rail_balance/hop-aware/takeover-20260801/o112-peaks-adaptive-rot1-3x20.json
```

The clean `cd356df` 10+100 checkpoint accepted O112:

```text
case       initial 32-round finish   final finish   speedup   final decision (Nsys)
volume                    6.080 ms       2.207 ms     2.75x                 1.300 ms
rot1                      7.394 ms       4.697 ms     1.57x              not reprofiled
```

The final volume/rot1 plans remain `[0,1605,4539,2048]` and
`[6142,44972,994,5236]`; source peaks remain 1405/7194 and pair peaks remain
256/1344.  Final source-shuffle medians are about 116/140 us, so data movement
is not the local bottleneck.  The reports are profiler-free and start from
empty JIT caches with a clean Git tree, but the conservative report flag is
false because an idle VNC `xterm` title contains the text `ncu-ui`; the raw
process row is retained and no profiler or foreign GPU process was active.

Matched Nsys places the final adaptive decision median at 1.300 ms, below the
pre-registered 1.5 ms stop line.  A four-warp rewrite was rejected before
implementation: it must restructure all block-uniform early exits and the
single-warp setup, while its remaining upper bound is only a few hundred
microseconds.  Final direct validation is 20/20 focused CUDA, 12/12 reference,
and the exact 8-rank LSA transaction PASS.

Final evidence:

```text
.cache/rail_balance/hop-aware/takeover-20260801/clean-cd356df-adaptive-volume-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/clean-cd356df-adaptive-rot1-10x100.json
.cache/rail_balance/hop-aware/takeover-20260801/post-cd356df-volume.nsys-rep
.cache/rail_balance/hop-aware/takeover-20260801/post-cd356df-volume.sqlite
```

## AT001 — preserve V2 auto as the autotune execution baseline

No performance implementation changed.  The first offline tuning space keeps
one execution preset only:

```text
strategy=v2_auto, num_sms=0, num_allocated_qps=0
```

This delegates SM estimation and QP selection to the existing DeepEP V2 model.
Explicit SM/QP presets are schema-supported but absent from the default search;
they may be added only after Nsys identifies exposed resource imbalance and
must still win profiler-free multinode confirmation.  Planner chunk size 8 and
seed 0 are recorded as frozen identity fields, not search axes.  No CPU timing
from AT001 is an operator-performance claim.
