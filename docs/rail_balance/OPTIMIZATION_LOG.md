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
