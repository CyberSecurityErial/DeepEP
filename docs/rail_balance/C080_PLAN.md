# C080 Hybrid Rail-Balance Integration Plan

Status: PLAN_FROZEN / LOCAL_IMPLEMENTATION_COMPLETE / MULTINODE_PENDING

This is the correctness-first plan for connecting the proved C030-C070
components to DeepEP Hybrid dispatch/combine.  It is intentionally narrow.
The first goal is to decide whether source-side rail balancing works, with the
smallest hot-path ABI that can prove the complete round trip.

## 1. Evidence boundary

The intended production-shaped path is:

    topk_idx
      -> destination dedup and per-channel count
      -> minimum-move quota and segments
      -> moved copies written directly to egress proxy payload slots over LSA
      -> retained copies use the owner rail
      -> moved copies use the egress rail
      -> existing destination forwarding semantics
      -> combine returns moved copies to source-egress proxy slots
      -> LSA unshuffle to the original owner's legacy reduce row
      -> existing combine epilogue

This machine has eight H200 GPUs but a truthful logical scaleout size of one.
It can validate planning, LSA movement, metadata lifetime, full vnode
dispatch/combine, code generation, sanitizers, and local performance.  It
cannot validate Rail Gin/QP/NIC behavior.  Evidence labels remain separate:

    DEFAULT_OFF_REGRESSION_PASS
    VNODE_FUNCTIONAL_PASS
    HYBRID_CODEGEN_PASS
    REAL_HYBRID_RUNTIME_UNTESTED

Only a real multi-node run may replace the final label.

## 2. Simplicity rules

The force-v1 operator follows these rules:

- no change to the legacy TokenLayout stride;
- no network route sidecar;
- no per-copy descriptor;
- no per-slot ready or generation value;
- no return header;
- no ring, queue tail, or per-copy peer atomic;
- no new permanent warp group;
- no extra pre-dispatch barrier; reuse Hybrid dispatch's existing Tag0 barrier;
- no silent fallback in force mode;
- no fusion until the staged path is correct and measured.

The host control path may validate strictly.  The GPU hot path keeps only data
that cannot be derived from static prefixes or the existing payload.

## 3. Public and semantic scope

The constructor adds experimental configuration:

    rail_balance = "off" | "force"
    rail_balance_proxy_slots_per_rank = 0

Defaults are off and zero.  Auto is deferred until measured costs exist.
Force-v1 accepts only:

    dispatch dtype                 BF16
    routing                       exact copy-level, move-only
    cached dispatch               false
    expanded dispatch             false
    deterministic                 false
    masked top-k                  false
    topk weights                  required FP32
    expert alignment              1
    allow_multiple_reduction      true
    do_cpu_sync                   true
    do_handle_copy                true
    previous events               None
    async_with_compute_stream     false
    allocate_on_comm_stream       false
    combine bias                  None
    concurrent force handles      one
    num_cpu_bytes                 0
    buffer sizing                 constructor shape, not manual num_bytes

The first numeric domain is:

    1 <= num_topk <= 32
    2 <= num_scaleout_ranks <= 32
    2 <= num_scaleup_ranks <= 32
    0 < num_experts <= 2048
    num_experts % (num_scaleout_ranks * num_scaleup_ranks) == 0
    num_experts / world_size <= 256
    0 <= num_tokens <= num_max_tokens_per_rank
    world_size * num_max_tokens_per_rank <= INT_MAX
    world_size * num_max_tokens_per_rank
        * min(num_topk, num_experts / world_size) <= INT_MAX
    hidden % 256 == 0
    every top-k expert is in [0, num_experts)
    no top-k lane is -1 and expert ids within one token are distinct
    proxy capacity is positive and fits signed 32-bit indexing

All C/D/prefix products and Pcap times either token stride use checked int64
arithmetic. Their aligned sum must fit size_t and the registered symmetric
window before any pointer is formed.

Unsupported combinations fail before a rail-balance device barrier.  A
balanced input with zero moves is a valid forced no-op plan.  If moves are
required and capacity is insufficient, force reports CapacityExceeded; it
does not execute the legacy path.

Force-v1 requires nonzero constructor num_max_tokens_per_rank, hidden, and
num_topk so both payload strides and the appended arena are fixed before the
symmetric window is created. Manual num_bytes construction is rejected for
force-v1 rather than guessing whether its tail is reserved.

## 4. Default-off identity

Force uses new headers, runtimes, kernel names, and JIT cache keys.  It does not
edit the legacy JIT roots or their recursive include closure:

    deep_ep/include/deep_ep/impls/dispatch.cuh
    deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh
    deep_ep/include/deep_ep/impls/combine.cuh
    deep_ep/include/deep_ep/impls/hybrid_combine.cuh
    deep_ep/include/deep_ep/impls/dispatch_copy_epilogue.cuh
    deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh
    deep_ep/include/deep_ep/impls/combine_utils.cuh
    deep_ep/include/deep_ep/common/comm.cuh
    deep_ep/include/deep_ep/common/handle.cuh
    deep_ep/include/deep_ep/common/exception.cuh
    deep_ep/include/deep_ep/common/ptx.cuh
    deep_ep/include/deep_ep/common/compiled.cuh
    deep_ep/include/deep_ep/common/layout.cuh
    deep_ep/include/deep_ep/common/math.cuh
    csrc/kernels/elastic/dispatch.hpp
    csrc/kernels/elastic/combine.hpp

With off and zero capacity:

- legacy buffer bytes and offsets are unchanged;
- the legacy runtimes and arguments are selected directly;
- no force tensor or arena is allocated;
- EPHandle construction and instance fields are unchanged;
- generated-source and recursive-include goldens remain exact.

The old LaunchRuntime public generate method is not used for mixed synthetic
Hybrid/direct probes in one process because it caches the first include hash.
Identity tests use generate_impl plus explicit include parsing or isolated
one-mode subprocesses.

## 5. Minimal planner and schedule

### 5.1 Count

One warp owns one source channel and scans tokens in the same order as the
Hybrid scaleout warp:

    token = channel, channel + C, channel + 2*C, ...

For each token it maps valid experts to destination servers, deduplicates
destinations within that token, and excludes the source-local server.  With at
most 32 destinations, a warp mask gives:

    channel_count[owner][channel][destination]

without a hot global atomic.

The stable owner prefix is an explicit plan output:

    owner_channel_prefix[o,c,d] =
        sum over h<c of channel_count[o,h,d]

During another channel-major scan, local ordinal l resolves to:

    owner_ordinal x = owner_channel_prefix[o,c,d] + l

Counts are written into a small symmetric control region.  After one LSA-team
barrier every local GPU can read the G by C by D matrix and build the same
deterministic plan.

### 5.2 Quota and segments

C030's minimum-move rule remains canonical:

    sum_g quota[g,d] == sum_g count[g,d]
    max_g quota[g,d] - min_g quota[g,d] <= 1
    keep[g,d] = min(count[g,d], quota[g,d])
    moved[d] = sum_g max(count[g,d] - quota[g,d], 0)

Each segment stores only:

    owner
    egress
    owner_begin
    count
    egress_begin

egress_begin is the canonical incoming ordinal for that egress/destination.
It is assigned from an independent incoming_cursor[egress][destination], so
segments targeting the same pair occupy adjacent non-overlapping intervals.
There are at most G-1 segments per destination.

### 5.3 Channel placement and compact proxy namespace

Let:

    M = ceil(num_max_tokens_per_rank / num_channels)

Retained copies occupy the prefix of their original source channel:

    retained[e,c,d] =
        intersection(channel owner-ordinal interval, [0, keep[e,d]))

Moved incoming copies fill only the remaining channel capacity, in canonical
channel order:

    spare[e,c,d] = M - retained[e,c,d]
    retained[e,c,d] + moved[e,c,d] <= M

This placement cannot run out of channel space for a valid quota:

    C * M >= num_max_tokens_per_rank
    quota[e,d] <= num_max_tokens_per_rank
    sum_c spare[e,c,d]
        = C*M - keep[e,d]
        >= quota[e,d] - keep[e,d]

The right-hand side is exactly the number of incoming moved copies.

The exact fixed-size outputs are:

    retained_count[G,C,D]
    moved_count[G,C,D]
    moved_channel_prefix[G,D,C+1]
    group_prefix[G,C,D]
    proxy_required[G]

For each egress/destination:

    moved_channel_prefix[e,d,0] = 0
    moved_channel_prefix[e,d,c+1] =
        moved_channel_prefix[e,d,c] + moved_count[e,c,d]

group_prefix is the exclusive sum of moved_count in channel-major,
destination-minor order:

    group_prefix[e,c,d] =
        sum over h<c and all destinations of moved_count[e,h,*]
        + sum over j<d of moved_count[e,c,j]

Proxy slots on each egress are grouped contiguously by channel and destination:

    p = group_prefix[e,c,d] + group_local_ordinal

The source-shuffle kernel recomputes the channel-major owner ordinal while
scanning topk_idx.  For a moved copy it finds its segment, target channel, and
p from the small segment/prefix tables, then writes the full payload directly
to peer egress proxy_dispatch[p].

The shared resolver is:

    if x < keep[owner,d]:
        retained, remote_slot = channel_local_ordinal
    else:
        s = unique segment containing (owner,d,x)
        j = s.egress_begin + x - s.owner_begin
        c = unique interval with
            moved_channel_prefix[s.egress,d,c] <= j
            < moved_channel_prefix[s.egress,d,c+1]
        u = j - moved_channel_prefix[s.egress,d,c]
        p = group_prefix[s.egress,c,d] + u
        remote_slot = retained_count[s.egress,c,d] + u

The first resolver uses a linear channel scan. Binary search or a moved
work-list is considered only if C100 measures this lookup as material.

There is no full per-copy assignment table and no moved work-list in the first
implementation.  If measurement later shows prefix lookup material, a compact
work-list is an optimization candidate, not a correctness dependency.

The egress consumes groups directly:

    for channel c:
        send retained prefixes
        for destination d:
            for p in group_prefix[e,c,d] .. + moved[e,c,d]:
                remote_slot =
                    retained[e,c,d] + p - group_prefix[e,c,d]
                send proxy_dispatch[p] through this GPU's Rail context

Thus destination, channel, remote slot, and proxy slot are all implied by the
loop and prefix.  No per-copy descriptor is needed.

The required reservation on one egress is:

    P_required[e] =
        sum over remote d of max(quota[e,d] - count[e,d], 0)

and the invocation requirement is max_e P_required[e] <= Pcap.

Before publication the plan verifies:

- every deduplicated remote destination-copy is retained or moved once;
- all quota, keep, and segment conservation equations hold;
- each proxy slot is unique;
- each remote slot is unique and belongs to a dense prefix;
- every channel count is at most M;
- required moved slots on each egress fit the reserved capacity.

## 6. Four-byte transit key

The existing dispatch TokenLayout already transports K linked-list integers.
On the source-to-destination leg they have no business meaning.  The
destination forwarder later overwrites all K values before the copy epilogue
uses them.

Force-v1 reuses only linked_list_idx[0] during that free lifetime:

    retained record: value ignored
    moved record:    linked_list_idx[0] = p

At destination ingress the force forwarder:

1. reads src_token_global_idx;
2. derives original_owner_local_rank;
3. determines moved by comparing owner local rank with current ingress local
   rank;
4. if moved, snapshots linked_list_idx[0] before any overwrite;
5. runs the original linked-list overwrite logic;
6. stores proxy_slot or -1 in one added force-forward-metadata column.

The destination ingress local rank is already the source egress rail identity,
so egress is not transported.  Lane zero need not target the current
destination: this field is transit scratch, not a top-k-lane value.

Force forward metadata has:

    [0] src_token_global_idx
    [1] last_in_chunk
    [2] proxy_slot, or -1
    [3 : 3+K] source scaleup ranks
    [3+K : 3+2K] destination slots

The legacy TokenLayout, dispatch copy epilogue, and default forward metadata
remain unchanged.

The source-shuffle kernel initializes the small aligned metadata region
deterministically before filling it, including every linked-list word and
metadata padding. Hidden bytes arrive through TMA. This avoids publishing
uninitialized scratch without zeroing or copying the large hidden payload twice.

## 7. Staged dispatch

The first dispatch schedule is:

    prepare/build/allocate all round-trip force resources
    -> route-validation plus local [C,D] count writes symmetric arena
    -> WORLD GATE #1: every EP rank valid and ready
    -> source-node-only LSA barrier and [G,C,D] count snapshot
    -> plan/prefix build
    -> WORLD GATE #2: every EP rank plan/capacity valid
    -> source shuffle of moved TokenLayouts
    -> TMA peer stores complete
    -> force Hybrid dispatch starts with its existing Tag0 barrier
    -> original destination-forward semantics
    -> original copy epilogue semantics

The force Hybrid dispatch preserves the original opening Tag0 barrier, which
already synchronizes the scaleup and scaleout teams with system release/acquire
semantics. That barrier is the proxy epoch boundary. Adding a separate
source-node barrier immediately before it would be redundant. Because all moved
payloads exist before the Hybrid kernel passes Tag0, C080 needs neither ready
flags nor generation values. Egress warps know exact group counts and never
wait on unknown work.

Retained sends compact their own per-channel destination counters instead of
leaving holes for skipped moved copies.  Each egress sends its retained prefix
first and the moved group immediately after it.  It publishes a final dense
tail only after all puts for that channel/destination have been issued.

Source-local destination copies never enter the rail plan and keep the original
LSA bypass.

Source shuffle and force Hybrid dispatch are launched in that exact order on
the same DeepEP comm stream on every world rank. Each producer executes TMA
store wait before kernel completion; the dispatch Tag0 release/acquire barrier
then establishes peer visibility. No rank may allocate, compile, validate, or
throw between the committed source shuffle and force dispatch.

## 8. Staged combine and unshuffle

At combine entry, handle validation and allocation of combined_x and optional
combined_topk_weights occur before any force combine kernel. Their shapes are
already known from num_combined_tokens, hidden, and top-k. Allocation exceptions
are caught into status, and WORLD COMBINE GATE #1 requires every EP rank to have
a valid live handle, all outputs, and all prebuilt kernels.

The force combine reads proxy_slot from force forward metadata.

- proxy_slot == -1: use the existing return to the original owner/token slot;
- proxy_slot >= 0: return the complete combine TokenLayout to the symmetric
  source-egress proxy_return[p] slot.

The complete record includes hidden data and top-k weights.  The main combine
must finish its Gin flush and rail completion protocol before unshuffle starts.

The standalone unshuffle iterates the same local groups.  For each p it reads:

    destination = current group destination
    src_global = proxy_dispatch[p].src_token_global_idx
    owner_local =
        (src_global / num_max_tokens_per_rank) % num_scaleup_ranks
    owner_token = src_global % num_max_tokens_per_rank
    final_reduce_row = destination

The final row follows the legacy compile-time layout choice:

    if num_scaleout_ranks <= num_topk:
        final_reduce_row = destination
    else:
        final_reduce_row = highest top-k lane targeting destination

The second value is derived by scanning the preserved dispatch top-k ids in
the standalone unshuffle.  It adds no transmitted route state and no branch to
the persistent dispatch/combine hot path.

Unshuffle copies proxy_return[p] over LSA into:

    owner legacy_reduce_buffer[final_reduce_row][owner_token]

Then every peer TMA store completes, an LSA-team barrier runs, and only then is
the existing combine reduce epilogue launched.  Same-stream ordering on one GPU
is insufficient because another source egress may be writing this GPU.

proxy_dispatch and proxy_return are separate bounded arenas.  Dispatch payloads
stay alive until unshuffle finishes because they contain the route derivation.
No return header or local route table is needed.

Force Hybrid combine, return unshuffle, the unshuffle's final scaleup-team
barrier, and the legacy epilogue also use the same comm stream and identical
launch order on every world rank. No host allocation, JIT build, validation, or
exception point is allowed inside this committed sequence. The epilogue is
stream-dependent on the barrier kernel; local stream ordering without that
cross-GPU barrier is not an accepted substitute.

## 9. Memory, handle, and failure model

Force memory is appended after the aligned legacy region:

    symmetric count/control
    proxy_dispatch[Pcap] using the legacy dispatch TokenLayout
    proxy_return[Pcap] using the legacy combine TokenLayout

Plan, segments, retained/moved counts, and group prefixes are local tensors.
Every local GPU builds the same values from symmetric counts, so they do not
need to be registered or sent over Gin.

The force handle adds one private aggregate state containing:

    invocation id
    owning buffer id
    plan/prefix tensors
    force forward metadata
    moved slot count

Default EPHandle objects are unchanged.  Force handles cannot be used for
cached dispatch and can be consumed by combine once. The normal state is:

    IDLE -> PREPARING -> PLAN_READY -> DISPATCH_LIVE -> IDLE

Prepare is a transaction that records its entry state. Only IDLE may enter
PREPARING. A second dispatch while DISPATCH_LIVE reports Busy in Gate #1 without
writing `HybridControl`, counts, or proxy payload; abort preserves the original
live handle and arena. A failure from PREPARING restores IDLE, while a combine
entry-gate failure preserves DISPATCH_LIVE.

One terminal invalid bit handles the exceptional path:

    any post-publication device error -> INVALID

INVALID cannot return to IDLE or reuse the arena; destruction/reconstruction is
required. A pre-publication dispatch transaction restores its recorded entry
state; only a transaction that entered from IDLE returns there. A combine-entry
gate failure leaves the same DISPATCH_LIVE handle unconsumed so all ranks may
retry or destroy it consistently.

A local validation/JIT/allocation error is converted to a fixed status before
any new device collective. The first force call generates and builds every
shape-dependent kernel needed by the whole round trip before that point:

    count
    plan/prefix
    source shuffle
    force Hybrid dispatch with its Tag0 barrier, and copy epilogue
    force Hybrid combine
    return unshuffle and barrier
    combine reduce epilogue

It also allocates every force-internal tensor needed by dispatch before the next
collective. Dispatch receive-output size is known only after its completed
count/communication stage; its copy epilogue is local and contains no later
force barrier in that call. Combine outputs are different: they are known at
combine entry and must be allocated, caught, and accepted by WORLD COMBINE GATE
#1 before the main combine is launched. There is no public prepare API.
Internally, however, force dispatch is split at WORLD GATE #1 because the C++
buffer does not own a PyTorch ProcessGroup:

    private prepare
        host validation, all allocations/JIT, local route count/status
    Python WORLD GATE #1
    private finish-plan
        prebuilt local LSA barrier, G peer-prefix copies, plan/prefix/status
    Python WORLD GATE #2

The user still makes one dispatch call. Python catches any prepare exception,
participates in Gate #1 with a fixed error code, and invokes an idempotent abort
on every rank if consensus fails. `finish-plan` contains no allocation, JIT,
shape validation, or callback. Force branches into its private Python path
before legacy dispatch performs rank-local shape/default/handle/SM/QP work; all
force normalization plus C++ prepare is inside one catch-to-Gate #1 region.

Gate #1 does not merely reduce an error code. Its fixed payload identifies the
operation and phase, invocation epoch and one-live state, arena ABI, topology
and rank mapping, `G/D/C/E/K/hidden/M/Pcap`, SM/QP choices, seed, and force
flags. Every fixed field must agree world-wide; local token count N may differ.
Gate #2 carries the operation/phase/epoch plus finish status and capacity
decision. This prevents valid-but-different calls from entering the same
collective sequence with different shapes or meanings.

Production gates use constructor/warmup-preallocated fixed-shape input/output
tensors and a warmed fixed-field tensor collective. The catch path must not
allocate a tensor, serialize an object, or cold-start a ProcessGroup operation;
otherwise an allocation failure could prevent the failure consensus itself.
Private single-node harnesses may use a short-timeout Gloo object gate, but that
is test scaffolding and not the force runtime protocol. Failure of the warmed
gate collective itself is process-fatal.

The LSA barrier uses an independent force-only runtime specialized as a
synthetic `(scaleout=1, scaleup=G)` topology. Calling the full actual Hybrid
barrier here would also enter Rail on a real multi-node job and is forbidden.
It receives the existing `ElasticBuffer::workspace`, whose first 16 bytes are
the shared `WorkspaceLayout` NVLink barrier phase/signals; it must never receive
the force arena base, where offset zero is `HybridControl`. The exact runtime is
`is_scaleup_nvlink=true`, `sequential=true`, scaleout rank zero, and physical
NCCL LSA rank as scaleup rank. No force operation may concurrently reuse the
workspace barrier state. More strongly, the same `ElasticBuffer` may not run
any other EP operation that uses this workspace barrier concurrently.

The legacy 16-byte barrier state is a monotonic cross-operation epoch and is
never cleared or rolled back by force prepare/abort. Gate #1 failure consumes
no local barrier. Gate #1 success consumes exactly one barrier on every local
rank even if Gate #2 later rejects capacity; abort then drops pending tensors
and force control only. Failure inside that barrier is fatal/INVALID.
After that barrier, the correctness-first implementation forms the local
snapshot with G active-prefix peer D2D copies on the DeepEP comm stream. For
LSA-local owner `o`, source is
`nccl_context->get_sym_ptr(local_count_base, o)` and destination is the local
snapshot offset `o*C*D`; the checked byte count is `C*D*sizeof(int)` and the
kind is `cudaMemcpyDeviceToDevice`. An LSA rank is never passed as a CUDA device
or global EP rank. The snapshot and every plan output are allocated before
Gate #1. C100 may replace those copies
only after NCU/Nsys evidence; B2 does not add a gather protocol. Tests and
cluster scripts provide explicit subprocess warmup and timeout.

Every consensus in force-v1 is world-wide over the full EP process group, even
though count exchange and the two data-visibility barriers are local
scaleup-team operations. One server may not continue because its own node plan
is valid while another server has failed.

Within the local count pass, every token range-checks all top-k entries before
deriving a destination index or bit shift. The pass rejects masking and
duplicate expert ids and writes one aggregate status. Every rank completes this
noncollective pass, then WORLD GATE #1 decides whether any rank may enter its
source-node barrier. After GPU
planning, WORLD GATE #2 performs the status/capacity decision before source
shuffle. A failed gate makes every rank return the same error; no rank skips
into or around a device barrier.

An unexpected returnable D2D/plan/prefix launch error after the local barrier is
caught into the fixed finish status so healthy ranks still reach Gate #2. A
CUDA-context failure, process crash, or failure inside the committed LSA
barrier is fatal and relies on GPU/ProcessGroup timeout; force does not pretend
it can recover a poisoned context.

A post-publication device error marks the force arena invalid; it is not reused
silently.  This is one status word and one host state, not a per-slot protocol.
The first correctness implementation may synchronize at force boundaries to
surface this status.  Private benchmarks time kernels with CUDA events and do
not include host consensus/synchronization in data-plane measurements.

## 10. Planned files

Expected force-only files are deliberately consolidated:

    deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh
    deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh
    deep_ep/include/deep_ep/impls/rail_balance_hybrid_dispatch.cuh
    deep_ep/include/deep_ep/impls/rail_balance_hybrid_combine.cuh
    csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp
    csrc/kernels/elastic/rail_balance_hybrid_combine.hpp

Host integration changes are limited to buffer sizing/selection, bindings, and
the Python constructor/handle.  Legacy kernel/runtime files listed in section 4
stay untouched.

## 11. Checkpoints and stop/go gates

### C080-A — identity and contract

- Keep the read-only SHA/include/layout goldens.
- Add backward-compatible constructor parsing and a disabled force capability.
- Freeze exact force errors and buffer formula.
- Run BF16 and FP8 default-off EP8 regressions.

Go: all old fingerprints, bytes, handle fields, runtime selection, and results
are unchanged.  Stop on any legacy identity change.

### C080-B — GPU count, plan, and inverse schedule

- Implement destination dedup/count from real topk_idx.
- Extend segments with egress_begin.
- Materialize retained/moved group counts and prefixes.
- Directly compute moved p from owner ordinal; do not consume a test manifest.
- Compare every value with an independent CPU oracle.

Go: boundary/random/rotating-hotspot cases, a new channel-major form of C061's
33-copy/six-move fixture, slot uniqueness, dense prefixes, and capacity failures
all match. C061's old token-major moved-token/channel golden remains a C061
compatibility test and is not reused as the production schedule golden.

### C080-C — minimal arena and collective gate

- Append the two payload arenas and symmetric count/control.
- Add private prepare/finish-plan/abort around the two pre-publication status
  consensuses and one-live-handle gate.
- Prebuild the force-only local LSA barrier before Gate #1; after Gate #1 it is
  the only barrier used for the compact count snapshot.
- Reject physical scaleout=1 in the public force path.

Go: per-call mismatched config, unsupported flags, JIT failure, and capacity
failure all exit every rank before publication; default off bytes remain exact.
Mixed constructor off/force configuration must be closed separately before the
public capability becomes true.

### C080-D — shared-core single-node path

- Replace C061/C070's CPU manifest with C080-B's GPU plan/prefix path.
- Use the four-byte transit key and descriptor-free grouped schedule.
- Validate 8x1 source shuffle plus 4x2 and 2x4 full round trips.

#### Frozen vnode bridge

The vnode bridge is test-only evidence plumbing.  It must consume the actual
GPU plan and source-shuffle bytes while leaving both the production Hybrid ABI
and the older C060/C061 emulator entry point unchanged.

Virtual 4x2 and 2x4 topologies use two independent buffers:

    source ranks [0,G)       source-subgroup ElasticBuffer
        physical topology   (scaleout=1, scaleup=G)
        work                B2 count/plan/prefix and direct-final shuffle

    all eight ranks          world ElasticBuffer
        physical topology   (scaleout=1, scaleup=8)
        work                vnode scaleout/forward/expert/return only

The two symmetric windows never expose pointers to one another.  The existing
owning CUDA proxy-dispatch snapshot crosses from the source buffer to the world
adapter.  The world adapter returns owning CUDA proxy-return and retained-seed
tensors to the source buffer.  These extra local copies are correctness-test
costs and are excluded from production performance evidence.  Ordinary local
inputs remain normal device allocations.  Explicit producer-to-consumer stream
edges are required even though the two objects currently happen to share a
process-global communication stream.

The first full-roundtrip fixtures contain remote-only top-k experts.  The old
vnode forwarder decodes every lane before deciding whether it belongs to the
current remote destination, so a local expert lane is an error in that
emulator.  Production Hybrid continues to support its normal local route; the
emulator is not generalized merely to test it.  Variable source token counts
are kept as their real N on each source rank; M is only the route and buffer
capacity.  No masked padding is fed into the production count kernel.

The source subgroup produces the unchanged fourteen B2 tensors.  Only its
`channel_count[G,C,D]` crosses the host-object boundary: source ranks first
compare their complete B2 digests, source rank zero broadcasts that one count
tensor through the test control plane, and every world rank runs the unchanged
plan/prefix kernels locally.  This keeps the private API small and makes the
world adapter's plan independently comparable with the source plan that made
the proxy bytes.  It also handles variable N without padding or changing the
production count kernel.  The resulting quota is compacted from `[G,D]` to
contiguous `[G,D-1]` before the old vnode kernels see it; taking
`quota.data()+1` is invalid because each source row still has pitch D.  These
test-only transfers add no production plan field.

Two new one-warp adapter kernels are sufficient.

`pack_vnode_base` runs only on source ranks, with one block per source channel.
For absolute remote destination `d`, channel `c`, and egress `e`, it derives
the destination-wide vnode prefix without another O(G*C*D) tensor:

    channel_base(e,c,d) =
        min(owner_channel_prefix[e,c,d], keep_count[e,d])
        + moved_channel_prefix[e,d,c]

    dense_slot = channel_base + remote_slot
    vnode_destination = d - 1
    vnode_physical_slot = vnode_destination * M + dense_slot

The identity

    channel_base == sum over h<c of
        (retained[e,h,d] + moved[e,h,d])

makes each `(egress,destination)` image exactly `[0,quota)`, with no atomics,
holes, or collisions.  Retained records are reconstructed from local
`x/topk_idx/topk_weights`; moved records are copied byte-for-byte from the
actual proxy-dispatch snapshot and retain `linked_list_idx[0] == p`.  Only the
test vnode descriptor, canaries, and ready word are synthesized.  A
1920-schedule independent CPU exhaustion over G={2,3,4,8}, varied C, and varied
D validated this mapping.

`return_demux` runs only on source ranks, with one block for each old vnode base
slot.  With `old_d=d-1`, `P=(D-1)*M`, and dense slot `s`, the old vnode return
record for top-k lane `l` is:

    base slot          b = old_d*M + s
    contribution slot u = P + b*K + l

It validates the base record, every matching contribution and route, and every
nonmatching empty lane.  It then uses the existing `compute_topk_slots` and
`combine_reduce` semantics to form one complete combine TokenLayout for that
destination.  Dispatch and combine records are not layout-compatible and are
never copied as if they were.  All K expert ids and all K weights are copied
from the base record; aligned metadata padding is deterministic zero.

The demux scans the small retained/moved channel intervals to invert `s`:

    local = s - channel_base
    local < retained[e,c,d]       -> retained
    otherwise p = group_prefix[e,c,d]
                  + local - retained[e,c,d]

Retained output is written to the local legacy-reduce seed.  Moved output is
written to local `proxy_return[p]` after verifying the preserved dispatch
record, source owner/token, and transit key.  Static plan injectivity proves
both destinations are collision-free, so neither path needs an atomic.

The combine row is exactly the legacy choice:

    D <= K    row = absolute destination d
    D > K     row = highest top-k lane whose expert targets d

After vnode demux, the source buffer copies the two owning tensors into its
own symmetric reduce/proxy-return areas, launches the production-shared return
unshuffle, performs the source-G LSA visibility barrier, and invokes the exact
legacy combine reduce epilogue.  A force-only wrapper may prepare and launch
`CombineReduceEpilogueRuntime`, but it must retain the name
`combine_reduce_epilogue`, generated specialization, PDL launch configuration,
and cache identity.  `csrc/kernels/elastic/combine.hpp` remains unchanged.

All eight processes create the NCCL source subgroup in the same order, but
only prefix world ranks `[0,G)` construct its ElasticBuffer; nonmembers never
enter the constructor.  The prefix restriction is part of this harness because
the production source payload encodes subgroup-local rank and the emulator
interprets it as world-local owner.  Runtime assertions require source topology
`(scaleout=1,scaleup=G)` and world topology `(1,8)`.

All adapter JIT builds, allocations, input validation, quota compaction, and
cross-object stream edges precede the first committed emulator barrier.  The
world vnode keeps its existing all-rank stage barriers.  Source return
unshuffle then uses the already proved source-team barrier before the local
epilogue.  Source-group and world-group NCCL operations are serialized rather
than overlapped.  Any device error is sticky; all participating ranks finish
the current fixed barrier sequence before host inspection.  No rollback of
test-only scratch is promised after publication.

Go: H256/H7168, exact payload/weight/route/output, zero/one/capacity edges,
repeated reuse, and injected corrupt/missing plan state pass.  Label only
VNODE_FUNCTIONAL_PASS.

### C080-E — isolated Hybrid dispatch codegen

- Implement the force dispatch specialization.
- Snapshot p before linked-list overwrite.
- Keep static retained+moved dense slots and final-tail publication.
- Compile representative 2x4 and 4x2 specializations in timed subprocesses.

Go: cubins compile, ptxas reports no spill/regression outside frozen gates,
instrumentation counts exact moved/retained/Gin bytes, and old JIT hashes stay
unchanged. Counters are derived from the static plan or enabled only in a debug
specialization; the release hot path does not add per-put atomics. Local label
is HYBRID_CODEGEN_PASS only.

### C080-F — isolated Hybrid combine and unshuffle

- Return moved records to proxy_return[p].
- Unshuffle complete records to the legacy layout-selected owner reduce row.
- Place the mandatory post-unshuffle LSA barrier before the legacy epilogue.

Go: shared-core full loop and matching combine/unshuffle codegen pass; hidden
and weights are independently exact; owner-token collision cases pass.

### C080-G — local closure

- Run default-off regression, vnode, codegen, fault, sanitizer, and C000-C070
  compatibility suites.
- Perform independent correctness and hot-path-minimality review.

Go: no unresolved Blocker/High issue.  State becomes:

    LOCAL_COMPLETE / MULTINODE_PENDING

### C080-H — real Rail/Gin gate

- Run a truthful scaleout-greater-than-one topology.
- Check real Gin destinations, per-rail bytes, tails, QP/channel usage, return
  completion, and off/force numerical equality.

This gate is DEFERRED_ENVIRONMENT locally.

## 12. C090, C100, and C105 before networking

C090 exhausts locally meaningful correctness:

- zero, one, boundary, and capacity-plus-one token/copy cases;
- uniform, one-hot, rotating, Zipf, log-normal, and seeded random routing;
- 8x1, 4x2, and 2x4 paths;
- duplicate destination lanes and multi-destination tokens;
- repeated arena reuse, stale/corrupt p, plan corruption, delayed ranks;
- memcheck, synccheck, initcheck, and focused racecheck where supported.

C100 measures and optimizes only after correctness:

- count, plan/prefix, source shuffle, dispatch proxy consumption, and unshuffle
  separately;
- direct-final LSA against cudaMemcpyPeerAsync and vector peer-copy baselines;
- sequential channel fill versus one measured alternative;
- prefix lookup versus a compact moved work-list only if lookup is measurable;
- static slots against an atomic diagnostic;
- all-to-one, one-to-all, all-to-all, pairwise, and rotating transfers;
- H256 and H7168, controlled GPU-idle runs, repetitions, medians, and spread;
- compute overlap and moved-byte RDMA bandwidth sweeps.

Every candidate keeps a baseline, correctness result, GPU state, commands,
metrics, decision, and rollback point.  Negative and noisy results stay in the
optimization log.  No local LSA number is presented as real RDMA speedup.

C105 packages:

- one-command build and JIT warmup;
- one-command off/force correctness A/B;
- canonical balanced, two-hot, one-hot, capacity, and real-route cases;
- stable JSON counts for original/balanced rail bytes, moved bytes, group
  counts, Gin puts/bytes, waits, and error status;
- explicit local/codegen/real-runtime evidence labels.

Current state: the validation-only JSON bundle generator covers the canonical
balanced/two-hot/one-hot/capacity route cases and stable CPU-oracle traffic
fields.  `run_rail_balance_hybrid_multinode.py` now adds the truthful live
D>1 off/force runner: it verifies identical Git/extension/environment identity,
executes public dispatch/combine against the official DeepEP references, checks
off/force equality, validates the capacity plan-gate rejection twice, restores
the temporary validation capability, and writes one collective JSON result.
The runner deliberately leaves QP/NIC/wait fields unavailable.  The one-command
build plus representative main-kernel codegen preflight is now available as
`run_rail_balance_build_warmup.py`; it explicitly does not claim the exact ten-
runtime production cache is warm.  A real-cluster `balanced` round trip must
warm that exact cache before the skew cases.  Real execution, real-route
ingestion, hardware counters, and any physical-runtime evidence-label upgrade
remain pending.

## 13. Explicitly deferred

The following are not required to prove the first idea:

- auto policy;
- cached/replay dispatch;
- FP8 rail-balance transport;
- expanded, deterministic, masked, bias, and single-reduction modes;
- token-primary and bounded-split policies;
- ring reuse and source/egress producer-consumer overlap;
- per-slot generation, descriptors, sidecars, or return headers;
- notify-warp fusion;
- batched incremental tail publication;
- QP-aware channel striping;
- fused return-unshuffle;
- real-network tuning.

Any deferred mechanism is added only if correctness or measurement creates a
specific need for it.

## 14. Checkpoint discipline

Each checkpoint records:

- base commit and diff;
- commands and exact result;
- successful and failed attempts;
- correctness and performance evidence;
- unresolved environment limits;
- rollback point and next action.

Coherent checkpoints are committed and pushed to
fork/feat/rail-balance-prototype.  Failed designs remain documented even when
their code is removed.
