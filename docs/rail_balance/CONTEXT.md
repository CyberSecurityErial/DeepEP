# Rail Balance Context

## User objective

Propose and implement a new DeepEP optimization for an eight-GPU Hopper rail
topology: redistribute source-side remote traffic across local egress GPUs so
that each `(source server, destination server)` pair uses the rails more evenly,
then reuse DeepEP V2's destination-side forwarding. Combine must return through
the selected proxy rail and unshuffle to the original owner.

The active long-running Goal additionally requires exhausting the correctness,
fault, sanitizer, concurrency, and API cases that can be tested on the local
8xH200 node, then measuring and optimizing every locally observable planner,
shuffle, publication, and return-unshuffle component while GPUs are idle. A
network-ready one-command test/metrics package must minimize cluster bring-up
time. Only Gin/QP/NIC/fabric behavior that cannot exist locally may remain for
the network phase.

## Source context reviewed

- Proposal evaluation:
  `https://chatgpt.com/s/t_6a5bc6be71cc8191903e932d7e70f2a7`
- Prior-art/mainline scan:
  `https://chatgpt.com/s/t_6a5bc6e8170881918e86c7ff741015c1`
- The detailed implementation and single-node validation designs supplied in
  the task prompt are authoritative working context for this branch.

Both shared chats were fully decoded from their public share pages on
2026-07-19 after the generic web reader returned a cache miss.

## Distilled technical conclusions

1. DeepEP V2 Hybrid fixes a source GPU to its own `ncclTeamTagRail` context.
   Changing only the remote destination of `gin.put()` cannot borrow another
   local GPU's rail.
2. V2 already provides destination ingress forwarding through LSA/TMA. The new
   core is source-local staging/proxying plus the symmetric combine return path.
3. Balancing must be per destination server. Balancing only aggregate source
   bytes does not guarantee balanced ingress at each destination.
4. A token with top-k experts on the same remote server produces one cross-node
   hidden-payload copy. Planner counts must deduplicate destination servers per
   token before constructing `c[g,d]`.
5. Exact copy-level balancing may destroy the current one-token/one-send-buffer
   reuse when a token reaches several destination servers through different
   egress GPUs. This is the dominant algorithm/data-layout risk.
6. The leading production candidate is move-only, bounded split (normally one
   egress, at most two), static slots, direct-final peer writes, symmetric
   dispatch/combine routing, and dynamic bypass.
7. FAST is related but broader: this proposal targets node-local rail balance
   and does not solve global server incast, expert compute imbalance, or fabric
   congestion by itself. RailS is close prior art for local rail scheduling, so
   novelty must come from the DeepEP-specific fused protocol, reuse semantics,
   cost-aware bounded split, and measured behavior.
8. DeepEP naming is easy to invert: `scaleout_rank_idx` denotes the server/node;
   `scaleup_rank_idx` denotes the local GPU/rail within that server. A global
   rank is `scaleout * num_scaleup + scaleup`.
9. Local-destination traffic must be excluded from rail planning.

## Repository baseline

```text
repository: /home/chen/workspace/source_code/DeepEP
base branch: main
base commit: dd758caf451848bd150e1046af3d0a73e5fff38d
base subject: NCCL Device API compat: use runtime version for ncclDevCommCreate (#688)
initial worktree: clean
development branch: feat/rail-balance-prototype
checkpoint remote: https://github.com/CyberSecurityErial/DeepEP
```

Checkpoint branches may be pushed to the user-provided fork. Do not modify or
force-push its `main` branch.

No repository-local `AGENTS.md` or `CLAUDE.md` was found in the first audit.

## Confirmed code anchors

- `deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`
  - Creates a metadata-bearing `TokenLayout`.
  - Deduplicates `dst_scaleout_rank_idx` across top-k lanes.
  - Stores `src_token_global_idx`.
  - Uses the current GPU's `gin.put<ncclTeamTagRail>` for scaleout.
- `deep_ep/include/deep_ep/impls/hybrid_combine.cuh`
  - Recovers source rank/server/token from `src_token_global_idx`.
  - Returns through the matching rail and uses forward metadata/channel lists.
  - The existing source token index alone is not collision-free for several
    owners sharing one new proxy GPU.
  - Multiple-reduction modes can require a contribution/layout row in addition
    to a proxy token slot.
- `deep_ep/include/deep_ep/common/layout.cuh`
  - `WorkspaceLayout`, `TokenLayout`, and `BufferLayout` define the lifetimes and
    TMA-aligned metadata boundaries that the prototype must preserve.
- `deep_ep/buffers/elastic.py`
  - `EPHandle` carries receive metadata, destination slots, forward metadata,
    and linked lists across dispatch/combine/replay.
  - Buffer-size, SM, QP, dispatch, and combine calculations are integration
    points; public compatibility must be preserved.
- `csrc/elastic/buffer.hpp`
  - Allocates handle tensors, checks buffer sizes, and launches communication
    and epilogue kernels. Cached-handle tuple and lifetime changes must be made
    here together with Python changes.
- `csrc/kernels/elastic/dispatch.hpp` and `combine.hpp`
  - These are the actual JIT launcher locations. There is no
    `deep_ep/jit/kernels/*.py` launcher tree in this checkout.
  - The generated specialization and include graph form the JIT cache key.
- `tests/elastic/test_ep.py`
  - Existing coverage includes normal/expanded/cached dispatch, combine,
    deterministic behavior, masks, imbalance controls, and profiling.
  - A physical single-node run selects the direct implementation because
    `num_scaleout_ranks == 1`; it does not execute Hybrid kernels.
  - `--test-first-only` covers the first FP8/alignment-128 case; it is not a BF16
    baseline. The full current matrix is large and runs under inference mode,
    so it does not supply backward coverage.

## Buffer and protocol constraints confirmed by audit

- Current Hybrid scaleout source payload capacity is one slot per local token,
  reused for multiple destination servers. Exact-copy balancing needs separate
  moved-copy payload capacity.
- Current channel capacity is approximately
  `ceil(num_max_tokens_per_rank / num_channels)`. Rail quotas alone do not prove
  channel safety; planner output must include capacity-aware channel prefixes.
- Retained and proxy traffic cannot allocate remote slots independently from
  zero. They share one destination queue/slot namespace.
- `recv_src_metadata[:, 2:]` has existing expanded-path semantics. A proxy ID
  cannot simply be appended there.
- The production communication window is symmetric and registered at buffer
  construction. Proxy payload capacity cannot be added as an arbitrary dynamic
  dispatch tensor.
- A virtual single-machine topology requires an independent launcher with an
  explicit physical LSA peer map. Replacing `gin.put()` alone cannot make the
  existing one-node runtime instantiate Hybrid kernels.

## Target and runtime assumptions

- Target platform is fixed by the user as a single machine with 8 x H200 and an
  H200/NVSwitch production goal. Do not reopen that product requirement.
- The current container-visible device label may differ from the target label;
  performance decisions must rely on measured peer connectivity, available
  capabilities, clocks, topology, and an idle measurement window rather than a
  label alone.
- There is no usable NIC path for the local prototype. LSA emulation validates
  planner/proxy/metadata/return behavior, not Gin/QP/NIC/fabric behavior.
- Functional tests may run while other jobs occupy GPUs if the estimated buffer
  budget safely fits and no OOM is expected.
- Local performance tests are required when GPUs are idle/available for clean
  measurement. Timing collected while GPUs are shared is diagnostic only and
  must not be used as an optimization conclusion.
- On 2026-07-20 the user explicitly authorized stopping GPU jobs that are
  positively identified as Megatron/MGT after coordinating with their owner.
  Always inspect the PID and full command first; this authorization does not
  extend to unrelated GPU processes.
- The first audit found no built `_C` artifact and an uninitialized `fmt`
  submodule. Build/JIT setup is therefore a real WP0 prerequisite.
- In-place development must ensure `_C` resolves from this checkout and JIT
  `library_root` points at the live source include tree; otherwise `.cuh` edits
  can silently compile from an old copied `build/lib` package.

## Scope boundaries

In scope:

- planner, static slot assignment, source shuffle, proxy queue/protocol;
- virtual 8x1, 4x2, and 2x4 layouts;
- dispatch/combine symmetry and autograd replay;
- BF16/FP8 payload, scale, weights, masks, alignment, cached/expanded paths;
- conditional local performance tests and later real-cluster Gin validation.

Out of scope until evidence expands the project:

- global FAST/Birkhoff server scheduling;
- router/expert placement or replication;
- solving hot-expert compute imbalance;
- asserting multi-node speedup from LSA emulation;
- changing the default DeepEP public behavior before forced-mode correctness.

## Remaining questions to resolve with evidence

1. Where is the narrowest internal launcher boundary that shares production
   `TokenLayout` without prematurely changing `ElasticBuffer.dispatch()`?
2. Should plan ownership be replicated per GPU or computed by a designated
   local warp after count publication?
3. Can a bounded two-egress policy approach exact per-destination balance on
   real routing traces at acceptable payload duplication?
4. Which future `auto` capacity policy is best: trim lowest-value moves,
   balance only the heaviest `(rail,dst)` pairs, or bypass the whole plan?
   Experimental `force` is already resolved as collective fail-before-publish.
5. Which TMA/LSA ordering primitive is required for direct peer writes before a
   descriptor is made visible to the egress consumer?
6. Does the selected 32-byte same-slot route sidecar remain worthwhile after
   real Gin measurement, or should the isolated force layout later pack a
   smaller key once all top-k alignment cases have measured evidence?

## Communication protocol terminology

- **owner**: GPU that originally owns the source token.
- **egress/proxy**: local GPU whose rail sends a retained or moved destination
  copy.
- **destination copy**: one hidden-payload transmission for one distinct remote
  destination server after top-k deduplication.
- **proxy slot**: collision-free egress send/return location assigned by the
  plan.
- **return route**: mapping from proxy slot/generation back to original owner
  and source token.
- **retained copy**: destination copy sent on the owner's original rail.
- **moved copy**: destination copy staged to another local egress rail.

## Current implementation checkpoint (2026-07-19)

- C000 through C050 are complete on branch `feat/rail-balance-prototype`.
- The private C040 path is a one-shot 8x1 LSA/TMA manifest executor. It writes
  BF16 `TokenLayout` records directly into compact final egress slots and
  release-publishes a positive generation. It is not yet a queue consumer or
  a public `ElasticBuffer.dispatch()` option.
- The final C040 case uses hidden 7168, four top-k lanes, four logical
  destinations, eight concurrent owners, and `P=32`. It moves 256 records and
  exactly fills every egress. A second valid call with `P=33` proves the unused
  record stays poisoned; a caller-side physical-capacity rejection at `P=31`
  proves that an empty atomic manifest launches no producer.
- `StaticSlotPlan` now freezes its channel count/capacity and is validated
  fail-closed before physical packing. Corrupted destination, egress, channel,
  slot, generation, moved state, coverage, or committed plan fields are
  rejected by persistent mutation tests.
- C050 adds a separate one-shot producer/scanner/downstream protocol while
  preserving C040's record and ready offsets. A release/acquire chain publishes
  only a contiguous current-generation prefix. Producers launch in three
  dependency-ordered phases (unconditional, hole-gated, consumption-gated),
  and both producer and scanner watchdogs renew only on strictly advancing
  finite state. Error arbitration uses system-scope CAS because the control
  pointer may be an LSA peer pointer.
- C050 is deliberately not a reusable ring. It proves one-shot ordering,
  payload visibility, live overlap, slow-progress liveness, and bounded fault
  convergence across eight GPUs. Generation wraparound, ABA protection,
  head/tail reuse, and backpressure remain future work after the 4x2 loop.
- C060 is complete: GPUs 0–3 form a virtual source node, GPUs 4–7 form a
  virtual destination node, and the restricted BF16 non-expanded dispatch →
  synthetic expert → combine → original-owner round trip passes exactly.
- C060 implementation is present behind a private API. It uses one eight-rank
  LSA window, compact full-egress prefixes, a frozen record plus 32-byte route
  sidecar, finite stage kernels, and eight identical barriers. Fresh-cache
  H256 and H7168 eight-process tests pass byte-for-byte; C040, C050, and the
  original EP8 correctness gate also pass after the change.
- The C060 main fixture is frozen to source token prefixes `(32,24,16,8)`, one
  remote destination, quota `(20,20,20,20)`, physical base capacity 20,
  generation 61, four lanes rotated across destination GPUs by owner and token,
  and exact binary weights `(1/2,1/4,1/8,1/8)`.
- `VNodeRoundTripLayout` uses rail capacity `P*(K+1)`: the first `P` slots are
  base dispatch records and the remaining `P*K` slots are returned expert
  contributions. The expert and source-owner regions share the same offset
  only because those roles occupy disjoint physical ranks.
- The current private bridge returns twelve owning tensors. The final snapshot
  is a 4 KiB `0xA5` guard immediately after the independently derived vnode
  arena; it remained intact in the H7168 fresh-cache run.
- C061 is active. It must generalize the emulator to the 2x4 virtual layout,
  prove per-destination rather than aggregate balance, and replace C060's
  fixed `P` base-prefix assumption with explicit per-rail counts. C060's
  restricted bridge is not represented as a generic variable-prefix API.
- The frozen C061 topology is source ranks `0,1`, destination 0 ranks `2,3`,
  destination 1 ranks `4,5`, and destination 2 ranks `6,7`. Counts are
  `((9,2,6),(2,9,5))`: aggregate source totals are already `17/16`, but the
  exact per-destination quota `((6,5,6),(5,6,5))` still moves six copies.
- C061 uses destination capacity 6, flattened base capacity 18, rail capacity
  90, compact per-destination expert capacity 48, and status stride 72. The
  intentional base holes are `{11}` on egress 0 and `{5,17}` on egress 1.
  Thirty-three destination copies produce exactly 72 contributions, not
  `33*4`, because each destination forwards only its matching expert lanes.
- C061 H256 and H7168 fresh-cache eight-GPU runs now pass all 282 exact global
  coverage records. The 38-test planner suite, C060 single-destination vnode,
  and the original EP8 public path also pass after the generalization. An
  independent final review found 0 Blocker, High, or Medium issues; C061 is
  closed `PASS`.
- C070 is intentionally narrow: persist the record descriptor and 32-byte
  route together, clear and restore the private arena across a second call,
  and replay only expert, return, owner unshuffle, and deterministic reduce.
  It will not yet modify public EPHandle or claim Hybrid integration.
- C070 is now `PASS` for the frozen trusted-snapshot PoC. H256 and H7168 both
  survive two byte-identical cross-call replays after manifest/fingerprint
  poison and deletion; a corrupted live ingress slot reports RouteMismatch=4
  without deadlock. Before C080 can expose this through a less trusted path,
  rank-local host validation must be converted into a cross-rank preflight
  consensus so one early throw cannot strand peers at the first barrier.

## C080 frozen minimal integration context (2026-07-20)

- The branch resumed cleanly at
  `cdac8a3fe218a2c62a1d21aed04c467fff7b65b1`. C080 began with a read-only audit;
  no Hybrid source change preceded the plan freeze.
- The old Hybrid JIT graph is immutable for this checkpoint. In particular,
  the complete recursive include closure rooted at `hybrid_dispatch.cuh` and
  `hybrid_combine.cuh` (including `combine_utils.cuh` and common comm/handle/
  exception/ptx/compiled/layout/math headers), plus the old dispatch/combine
  JIT launchers, cannot be edited or reused by force mode. Force receives
  independent headers, runtime classes, generated code, and cache keys so
  default-off identity is testable.
- The old `LaunchRuntime` caches one include hash per derived class. A baseline
  probe must not call its public `generate()` first with synthetic Hybrid args
  and later with direct args in the same process. Use `generate_impl()` plus an
  explicit include parse, or an isolated one-mode subprocess.
- The main missing production component is not quota planning. It is a GPU path
  that starts from real `topk_idx`, deduplicates remote destination servers,
  counts by source channel, and derives static egress/channel/remote/proxy slots.
  The simplified plan derives these from segments and grouped prefixes instead
  of storing a full per-copy assignment table. C040-C070 currently receive this
  information from a test/CPU manifest.
- Production owner ordinals will be channel-major, matching the existing
  one-warp-per-channel Hybrid traversal. Retained and moved copies share one
  dense remote-slot prefix; a published tail may never cross a hole.
- The experimental API is constructor-fixed `rail_balance="off|force"`, default
  `off`, with explicit per-rank proxy capacity. `auto`, per-call switching, and
  cached replay are deferred. First force support is BF16, non-expanded,
  non-deterministic, synchronous, exact-copy, one-shot, and one-in-flight.
- Force is fail-closed: constructor configuration, invocation/JIT preparation,
  and GPU plan/capacity status reach consensus before any payload or remote tail
  is published. Capacity exhaustion is a collective force error; fallback
  belongs to future `auto`.
- The first sidecar-heavy C080 draft is rejected. In non-cached force-v1,
  `linked_list_idx[0]` has a verified free transit lifetime and carries only
  compact proxy slot `p`. Static `(egress,channel,destination)` group prefixes
  remove per-copy descriptors, ready/generation values, network route sidecars,
  and return headers. Dispatch payloads remain alive through combine so
  unshuffle derives owner/token from `src_token_global_idx`.
- Force-v1 requires `allow_multiple_reduction=true`. The planner/source path
  accepts up to 32 destinations even when `D>K`. Return unshuffle follows the
  legacy layout: row `destination` when `D<=K`, otherwise the highest top-k lane
  targeting that destination, derived from the preserved dispatch payload.
  No extra transmitted route state is needed.
- A moved combine result must return to its source egress's unique proxy slot
  and then be written over LSA to the original owner's existing reduction row.
  `src_token_idx` alone is collision-prone, and `recv_src_metadata[:,2:]` cannot
  hold the key because those columns already belong to expanded mode.
- The current node is truthfully `scaleout=1`. Local C080 evidence is limited to
  default-off regression, shared planner/LSA/vnode functionality, and synthetic
  Hybrid code generation. A public force call must reject this topology; only
  real multi-node Rail execution can establish Hybrid runtime correctness.
- The authoritative implementation order, support matrix, stop conditions, and
  evidence labels are in `C080_PLAN.md`, checkpoints C080-A through C080-H,
  followed by C090 exhaustive local validation, C100 local optimization, and
  C105 network-readiness packaging.
- Standing resource authorization: when a process is positively identified as
  Megatron/MGT training, it may be terminated without another prompt. Other GPU
  processes remain out of scope.

## C080 implementation context (2026-07-21)

- Remote branch checkpoints now include `bb47ad0` (force-only Hybrid arena ABI)
  and `3676193` (strict compact CPU schedule oracle).
- Public force remains intentionally unavailable. Default off has exact API,
  buffer, handle, runtime, and recursive-JIT identity coverage.
- The force arena is tail-only and contains control/count plus dispatch/return
  payloads. It does not contain the full plan, a manifest, descriptors, ready
  words, generations, a route sidecar, or a ring.
- The immutable legacy Hybrid workspace ceiling is 1024 channels even though
  the generic host heuristic can calculate 1280. Every force entry must reject
  C > 1024 before an arena write or device collective.
- The schedule ABI is channel-major with destination-bucketed five-int
  segments. A strict test adapter maps real expert ids to scaleout servers and
  enforces the first force support matrix. It passes 11/11 and independently
  cross-checks the established C030 planner.
- The next production unit is an isolated GPU count/plan/prefix materializer.
  C040-C070 may supply algorithmic and vnode evidence, but their CPU manifest,
  descriptor/ready protocol, and replay snapshot are not production C080
  dependencies.
- C080-B1 now has that isolated materializer. Fresh-cache H200 execution matches
  the CPU oracle in 70 exact cases, including zero tokens, signed-int64 seed
  normalization, capacity failure, strict route rejection, maximum C/D, and a
  non-default stream. The three JIT cubins each export one kernel symbol.
- B1 still uses a stacked single-GPU `[G,N,K]` fixture. Production B2 must run
  the same count core with one local owner per rank, publish compact counts into
  the force arena, establish the node-consistent snapshot with the real LSA
  team/barrier, and only then launch the unchanged plan/prefix semantics.
- Performance work is conditional on idle GPUs. Functional work may continue
  under non-OOM contention, but timing under other workloads is not evidence.
  Controlled C100 work must use NCU, Nsys, and byte/cycle counters and retain
  tool failures and negative variants.
- The reviewed B2/C bridge is private prepare, fixed Gate1, local-only
  `(scaleout=1, scaleup=G)` LSA barrier over the legacy workspace state, G
  active-prefix peer D2D copies, unchanged B1 plan/prefix, then fixed Gate2.
  Every output/runtime exists before Gate1; barrier epochs never roll back; the
  force arena is never passed as barrier workspace. Final plan review is
  0 Blocker / 0 High.
- Production gate tensors/collectives must be preallocated and warmed. Private
  local tests may use a short-timeout Gloo control group but cannot claim that
  object exchange is the final error path. Mixed off/force construction remains
  an explicit public-enable blocker, not a private B2 blocker.
- C080-B2 now executes that private transaction on all eight local GPUs. Each
  rank publishes only its active `[C,D]` count prefix into the symmetric force
  arena; the synthetic `(1,8)` barrier over the legacy workspace precedes eight
  peer D2D snapshot copies and the unchanged B1 plan/prefix kernels. Exact
  comparison of all fourteen outputs passes for variable/zero local N, capacity
  failure, route/config Gate1 aborts, maximum C/D, int64 seed, non-default
  streams, and repeated monotonic barrier phases.
- The B2 test uses Gloo object exchange only as an explicitly private watchdog
  and consensus oracle. It is not evidence that production WORLD Gate1/Gate2
  exists; public `force` remains disabled until the fixed-tensor gate and the
  dispatch/combine transaction are integrated.
- C080-D now has a descriptor-free direct-final source kernel. One warp owns a
  source channel, resolves only compact plan state, stages each moved token
  once, and TMA-stores complete legacy TokenLayouts into peer proxy slots. True
  8-GPU evidence covers multi-destination copies, zero-N owners, exact Pcap,
  all eight owners, C=1024/D=32/K=4, non-default streams, and byte-unchanged
  inactive slots. H7168/reverse arena reuse, fail-closed plan corruption, and
  focused mem/sync/init/race tool runs now pass their stated gates. NCU/Nsys
  show the tiny moved fixture is peer-TMA latency dominated, so no extra hot
  path mechanism was added.
- The production-shared return-unshuffle now consumes the same static
  `(egress,channel,destination)` groups and preserved dispatch slot `p`.
  It copies the complete combine TokenLayout from `proxy_return[p]` over LSA to
  the original owner's legacy reduction row, using row=destination for D<=K
  and the highest matching top-k lane for D>K. True 8-GPU H256/H7168 and
  D32>K4 byte-exact tests pass without a return descriptor, ready flag, route
  sidecar, or success-path atomic. Full 4x2/2x4 vnode round trip and the
  production-shaped vnode round trip now pass; destination-side force Hybrid
  combine specialization remains open.
- C080-E now has an isolated force Hybrid dispatch header/runtime. It preserves
  legacy notify/local-forward behavior, sends retained owner prefixes plus
  descriptor-free moved proxy groups, publishes one final dense tail, and
  snapshots only four-byte `p` before linked-list overwrite into `3+2K`
  forward metadata. Production-shaped G8xD2/H7168/K8, G4xD2/H7168/K4, and
  G2xD4 H256/H7168 (including D>K) force cubins plus matched legacy baselines
  compile with zero spill. The H256 force instance is two registers above
  legacy but remains in the same launch/occupancy class. `proxy_required` is
  ABI-only after trusted Gate2; dispatch-payload counters are derived from
  existing plan tensors. Each cold-compile case runs under its own process-
  group watchdog. Public force remains disabled, and no local result claims
  real Gin/RDMA execution.
- C080-F now has the symmetric isolated force Hybrid combine specialization.
  Replay metadata is `3+2K`; moved records select the registered symmetric
  `proxy_return_base+p*stride`, while retained records and all reduction,
  staging, aggregation, flush, and completion behavior stay legacy. The
  sentinel is checked before `p`/2K reads, and dispatch/combine both reject
  fixed WorkspaceLayout geometry on the host before JIT plus statically before
  layout construction. Six force/legacy pairs cover the full
  `(D<=K,G<=K)` truth table with identical REG216/STACK96/SHARED1024/LOCAL0 and
  zero spill; only the new pointer adds 8B constant memory. Public force is
  still false. The next essential unit is the production host transaction:
  fixed-tensor WORLD gates, owning pending/handle state, exact legacy/arena
  offsets, uninterrupted shuffle→dispatch and combine→unshuffle→barrier→
  epilogue commit calls, then local sanitizer/regression/performance closure.
- Host integration H1 is now a pushed, default-inactive adapter layer. Main
  force dispatch/combine own their prebuilt runtime/spec/LaunchArgs and expose
  launch-only raw argument binders. Combine freezes the checked legacy reduce
  offset `min(G,K)*D*M*stride`; dispatch copy epilogue prebuilds the exact
  legacy BF16/noncached/nonexpanded specialization on all physical SMs. No
  public behavior changed. H1b now also freezes source H/K/C/N and return H/K/C
  geometry, exposes assertion-free raw source/return/combine-epilogue submits,
  and preserves the existing launch-only local barrier. Grid and source token
  count have one Prepared owner. H4 must still connect fixed-tensor WORLD gates
  plus owning transaction state before capability activation.
- Host integration H2 keeps the native single registered window. Force-only
  construction proves `total_bytes == legacy_bytes + arena_bytes`, passes the
  total to the unchanged C++ runtime, and records the arena at exact offset
  `legacy_bytes`. Off invokes no force helper and owns no rail-balance field.
  Both capabilities remain false until H3/H4 close cross-rank consensus and
  the production transaction.
- H3a now provides the private fixed CUDA WORLD primitive: caller-owned
  `int64[128]` CUDA/pinned storage, exact `(v,-v)` min/max recovery through one
  MAX reduction, deterministic error arbitration, and caller-stream-only
  synchronization. Real EP8 fault/stream/reuse tests pass. It is deliberately
  not yet wired into constructor or dispatch, so off and both capability bits
  remain unchanged.
- Pause boundary 2026-07-21: the branch was clean at `dc2b42c`, the fork
  matched, all child tasks were stopped, and H3a was freshly rerun on EP8. H3b
  had reached design review only and left no file on disk. Resume by adding the
  focused real-planner Gate1/Gate2 test described in `HANDOFF.md`; do not infer
  that public force or production dispatch was connected.
- H3b subsequently closed the private planner protocol: checked Gate1 state is
  prepared before transaction ownership, `finish` consumes exactly one local
  barrier after WORLD acceptance, and Gate2 reuses the same pinned manifest by
  patching only error and phase words. True EP8 covers actual tensor-K mismatch
  on a zero-N rank, invalid route, real capacity failure, asymmetric Gate2
  error, four abort/retry states, fourteen exact outputs, and eighteen MAX
  calls. This is the H4 transaction substrate; public force remains disabled.
- H4a now gives that substrate explicit host-only ownership states:
  `Preparing`, `PlanReady`, reserved `DispatchLive`, and permanent `Invalid`.
  Stale abort is a no-op; matching precommit abort releases state; matching
  live abort poisons it. A true EP8 stale-ID case plus source/return/fault
  regressions pass after the extension rebuild. No CUDA ABI or hot path changed,
  and `DispatchLive` must remain unreachable until H4c commits real dispatch.
- H4b adds the private production dispatch prepare boundary. It derives
  D/G/local ranks/H/K/C in C++, checks legacy bytes only before `arena_offset`,
  owns all pre-dispatch handle tensors and the complete prepared round trip,
  and freezes every plan/peer/main-dispatch pointer before Gate1. The local
  truthful D=1 topology fails before publication, aborts cleanly, and recovers
  the old planner on EP8; B1/B2/vnode and canonical eighteen-gate H3b remain
  green. No source/main launch or `DispatchLive` transition exists yet. H4c
  must submit those two stages adjacently, poison before the still-fallible
  launch boundary, and preserve the expert-prefix base versus `base+1` views.
- H4c now implements that private adjacent commit. All checks, device guard,
  raw captures, and prefix-view separation precede `Invalid`; the critical
  window is source submit, main force dispatch, shuffled, then DispatchLive.
  There is no epilogue, handle publication, combine, public force, or truthful
  local D>1 execution yet. A low-level one-rank launch failure is explicitly
  job-fatal, and H5 must observe asynchronous device faults at its first safe
  synchronization while keeping the object poisoned.
- H5a now owns that first safe completion boundary. It reads only the mapped
  int64 counter pointers frozen before Gate1, bounds every decoded addition by
  the topology-derived receive/expanded maxima, allocates exact BF16/top-k/src
  outputs, installs their owner, launches the unchanged non-expanded dispatch
  epilogue with the inclusive expert-prefix view, then copies status and
  synchronizes `comm_stream`. Only a zero status and a fully constructed native
  16-item result restore private `DispatchLive`; every post-publication failure
  remains `Invalid`. The synchronized force-v1 returns no event. Public Python
  force and one-shot combine ownership remain H5b/public work, and the local
  D=1 machine still cannot execute a truthful production Hybrid success path.
- H5b now closes the private symmetric combine transaction. Prepare validates
  the synchronized H5a `DispatchLive` owner, requires native force-v1 FP32
  weights, allocates exact owner-shaped outputs, freezes every raw dependency,
  and installs one owning completion before the future WORLD combine gate.
  Gate rejection can call an idempotent abort that releases only this new
  completion and preserves the dispatch handle. Commit first poisons the
  transaction, then submits main combine, return-unshuffle, the existing local
  barrier, and the native epilogue adjacently on `comm_stream`; only one final
  status read/sync can reset the entire transaction and return the native
  three-item ABI. Public Python still does not call these methods, so buffer
  identity, cached rejection, atomic handle consumption, and capability
  opening remain the next host slice. Local D=1 still supplies no successful
  production Gin evidence.
- H6 constructor consensus is now committed at `1e0ea60` with tests at
  `eb6b834`. With the host protocol enabled, off and force share one fixed
  pre-comm Gate0; force alone reuses the same CUDA/pinned `int64[128]` storage
  for a post-sizing/pre-window Gate1. The second manifest freezes exact bytes,
  resolved SL/QP/timeouts, and runtime flags; local helper/config/alignment
  failures reject every rank before window registration. Force deliberately
  skips the legacy PCIe/object-collective topology helper to avoid collective
  reordering. Constructor 9/9, API 9/9, legacy goldens 4/4, and an independent
  Blocker0/High0 audit pass. Public dispatch/combine and the shared one-shot
  ticket remain unimplemented, `_RAIL_BALANCE_FORCE_HOST_AVAILABLE` remains
  false, and no local evidence claims a truthful D>1 Gin/RDMA execution.
- H6 public lifecycle is now committed at `741e126` with its focused contract
  at `cbe7df2`. It connects the private H3b/H4/H5 transaction through two
  dispatch WORLD gates and one combine WORLD gate, then attaches one dynamic
  owner/invocation/shared-state ticket to the ordinary `EPHandle`. Validation
  and rejected gates are retryable with stale-safe/idempotent abort; accepted
  dispatch Gate2 or combine-gate failures are terminal. Combine marks the
  shared ticket `PREPARING`, then consumes it and clears the buffer live pointer
  before irreversible commit, so shallow handle copies cannot replay. Off
  keeps the legacy body and rejects only an exact force ticket before legacy
  metadata unpacking. Focused fake/source, constructor 9/9, API 9/9, legacy
  4/4, and independent Blocker0/High0 evidence pass. Both capability bits stay
  false: this proves host lifecycle ordering, not truthful D>1 Rail/Gin/RDMA.
- H6 real EP8 activation evidence is committed at `0597ed0`. A test-only
  per-worker patch opens the already installed private path while leaving both
  production capability bits false. Eight H200 ranks reject mixed mode and a
  rank-local invalid force geometry before communicator creation, unanimously
  create one real force window, reject public dispatch twice at the truthful
  `D=1` topology guard, and collectively destroy. Exactly six fixed MAX gates
  run; accepted-constructor storage stays pointer-stable; invocation advances
  1→3 with no live ticket or terminal state. Watchdog review is Blocker0/High0.
  C080-C is locally complete, but no payload/JIT/Gin stage executes at D=1;
  C080-G compatibility/sanitizer and C080-H real D>1 remain separate gates.
- C080-D's retained world-plan fault gap is closed at `6377e0c`. The existing
  4x2/H256 two-window harness now corrupts one prepared plan value on rank 0,
  models a missing tensor on rank 1, and requires exact value/count mismatch
  tags before any B0 entry. Each generation aborts both matching owners
  idempotently; generation 811 then completes the full round trip on the same
  buffers. True EP8 and an independent Blocker0/High0 audit pass. This is test
  preflight/recovery coverage and adds no production branch or device work.

## C080 local closure context (2026-07-21)

- C080-G is locally complete. The final tree passes default-off EP8 in both
  direct and truthful D1 Hybrid configurations, the complete focused host and
  source-contract matrix, real EP8 H3/H3b/H4b gates, true-LSA source/return,
  4x2 and D>K 2x4 vnode loops, source/world-plan faults, fresh force/legacy
  dispatch and combine codegen, and the in-place extension build.
- Compute Sanitizer 2025.1 runs the smallest complete changed data path,
  `hybrid_vnode_2x4_h256`, from one dedicated warm cache. Memcheck, synccheck,
  unfiltered initcheck, and focused racecheck all report zero errors; focused
  racecheck reports zero hazards and zero warnings.
- Final independent boundary review is Blocker 0 / High 0 / Medium 0. Python
  and C++ capability bits remain false. Since the last CUDA/header checkpoint,
  the public watchdog and plan faults changed only Python/tests/docs and did
  not add a device hot-path branch.
- Default-off identity means native sizing/runtime arguments, GPU buffer, JIT,
  handle, and result identity. Constructor mode parsing, a local ticket guard,
  and device discovery do exist and must not be described as zero host work.
- C080 is now `LOCAL_COMPLETE / MULTINODE_PENDING`, represented by the allowed
  checkpoint state `DEFERRED_ENVIRONMENT`. D1 control-plane evidence and vnode
  LSA execution are not real D>1 Gin/RDMA/QP/NIC evidence. C080-H remains the
  only gate that can establish multi-node correctness.
- At the C080 closure boundary, C090 still needed one small named-distribution
  matrix and one EP8 corrupt-p/delay/recovery case. The section below records
  their completion; this bullet is retained only as the transition rationale.
  No new NCU/Nsys run began in that closure, and unchanged historical suites
  or sanitizer kernels did not need reruns.

## C090 exhaustive-local closure context (2026-07-21)

- C090 is PASS after commits `ee827e8` and `5f8eab5`. The former adds named
  route-level one-hot, Zipf, and log-normal fixtures to the existing B1 path;
  70 unchanged baseline cases plus three additions pass exact CPU and single-
  GPU materializer comparison. It is deterministic distribution correctness,
  not a statistical sampler or throughput test.
- One 4x2/H256 true-EP8 case uses generations 812--814 on the same buffers.
  It corrupts a live moved record's four-byte `p`, checks the complete sticky
  status matrix, converges all ranks before abort, queues a rank-2 delay on the
  native world comm stream with no intervening synchronization, then completes
  both delayed and ordinary recovery round trips.
- Exact failed-finish diagnostics reuse the vnode bridge's existing D2H host
  status. The additive private getter has a readiness bit and changes no CUDA
  kernel, JIT source, layout, launch ABI, public Python API, or production
  dispatch/combine hot path.
- Initial reviews rejected a delay followed by Gloo synchronization and an
  unguarded status getter; both could invalidate the evidence or split ranks.
  The accepted ordering closes them. Final C++ and protocol reviews are each
  Blocker/High/Medium/Low 0/0/0/0; C090 exit review is 0/0/0 with one corrected
  documentation Low.
- At that checkpoint the next local gate was C100 after the user's profiling
  workflow arrived; that workflow is now installed and the later C100 sections
  supersede this historical resume point.  C080-H/Gin remains deferred to a
  truthful D>1 environment.

## Post-C090 Goal and readiness audit (2026-07-21)

- The C080 integration slice of the active long-running Goal is locally
  satisfied with Blocker 0 / High 0.  Its only C080 hard gap is the explicitly
  allowed truthful D>1 Rail/Gin environment gate; C100 remains in progress and
  C105 remains planned.  Both production capability bits therefore stay false.
- Each process with the host capability enabled enters the universal mixed-
  mode constructor gate; the gate does not first prove that all processes made
  the same activation choice.  The hard-false default-off branch deliberately
  avoids that new collective.  Collective safety therefore requires unanimous
  activation, and the disabled branch is not mixed-mode consensus evidence.
- Stale public wording was corrected: off preserves native sizing/runtime/JIT/
  handle/result identity but still has local configuration parsing and mode
  guards, and force is gated by D>1 correctness rather than missing dispatch/
  combine code.  API 9/9 and legacy identity 4/4 pass after updating the exact
  error-string test; the first stale-string assertion failure is retained in
  D061.
- C100's next meaningful target is a large moved-volume H7168 source and return
  profile.  Tiny three-record source evidence only measured peer-TMA fixed
  latency.  Six accepted raw reports are backed up under the ignored
  `.cache/rail_balance/c100/` with a SHA256 manifest.  Do not optimize host
  ticket/gate micro-cost or planner structure before the user's profiler
  workflow and attribution justify it.
- C105 still needs a dedicated narrow multi-node harness; the broad official
  `test_ep.py` matrix is not force-v1-compatible.  A validation-only unanimous
  capability override can exercise public D>1 code while leaving production
  defaults false, but it must restore both values and label unavailable Gin/QP
  counters honestly.

## C100 controlled source-fixture context (2026-07-21)

- Named-only `c100_volume_h256` and `c100_volume_h7168` cases use the same
  G8/D9/K4/N1024/C256/seed100 schedule.  Each owner keeps 128 and moves 896;
  every egress receives 896, total moved is 7,168, and 224 source channels per
  rank perform four records each.  H256 and H7168 differ only in payload width.
- Both final C256 cases pass true EP8 LSA exact raw TokenLayout, dense slot,
  immutable input, CPU oracle, and capacity checks.  They are excluded from the
  default correctness suite and are functionality-only evidence.
- C8 passed both widths but was rejected for profiling because seven moved CTAs
  underfill H200.  C128 passed H256 during the review transition; C256 is the
  retained production-scale fixture.  No timing was accepted from any version.
- A definite Megatron PGID 173641 with four approximately 36 GiB workers was
  terminated under the standing authorization; all GPUs were empty before the
  final functional runs.
- At that checkpoint the next NCU/Nsys invocation still waited for the user's
  workflow.  The workflow and standalone return harness are now complete; the
  later C100 Nsys sections supersede this historical state.  Vnode remains
  correctness scaffolding, not a performance fixture.

## C100 controlled return-fixture context (2026-07-21)

- The standalone production-shared return-unshuffle harness now imports the
  exact C100 source plan.  Named H256 and H7168 runs each consume 7,168 moved
  records with Pcap 896 and differ only in complete TokenLayout width.
- Both named cases pass true EP8 LSA exact owner/row/token bytes and prove all
  non-target bytes remain poisoned.  CPU fingerprint enumeration is cheaper,
  but full GPU proxy-return rows and byte comparisons are unchanged.
- Large cases remain named-only; the default C080-F GPU suite retains its three
  historical cases and allocation sizes and passes a fresh true-EP8 regression.
  No public API, JIT kernel, device source, planner, layout, or capability bit
  changed.  Independent review is Blocker0/High0/Medium0.
- The first named CPU run failed because an assertion still described exactly
  three default cases.  It was split into base and profile invariants without
  weakening either and is retained in D063.
- The user supplied an auditable profiler-free → Nsys → targeted NCU →
  one-variable change → profiler-free retest contract.  Fixture correctness is
  now ready for the environment manifest and baseline; no new performance
  evidence has yet been collected.

## C100 measurement-contract context (2026-07-21)

- `C100_PERFORMANCE_EVIDENCE.md` freezes the local workload, logical bytes,
  environment, installed tool capabilities, official-source boundaries,
  native DeepEP mechanisms, old-report scope, and missing evidence.  Every new
  baseline/profiler field is still uncollected.
- The operational target remains the user-declared H200/NVSwitch node.  The
  inconsistent management labels, CC field, sm90a cubin, NV18 topology and P2P
  output are retained as missing evidence; they do not block same-machine
  relative timing but forbid an unverified peak-efficiency denominator.
- The first baseline measures synchronous checked adapters, not raw kernels.
  Each sample needs a fresh one-shot transaction; the same-host eight-rank
  duration is `max(end)-min(start)`, with every per-rank interval retained.
  Nsys must expose kernel/copy/barrier/host contributions before an NCU target
  or performance-path change is selected.
- The standalone audited harness is committed at `056ad4d`.  Its dirty-tree,
  one-sample source runs prove report generation only and are not accepted
  performance evidence.  Clean 10+100 source/return × H256/H7168 baselines are
  next; no new Nsys/NCU or performance-path modification has started.

## C105 contract-only context (2026-07-21)

- Deterministic balanced/two-hot/one-hot routes and a stable JSON result
  skeleton now exist as CPU-only runtime-independent contracts.
  The evidence label is fixed to `REAL_HYBRID_RUNTIME_UNTESTED` and physical
  counters remain unavailable.
- Numeric bounds now match executable force Hybrid: D/G 2--32, K 1--32,
  E<=2048, E/world<=256, and int32 token/contribution products.  The stale G1
  and missing expert bounds in C080_PLAN were corrected.
- System and fixed ABI Python pass 12/12 after a retained handwritten-oracle
  failure.  A dedicated public multi-node runner, capability override,
  correctness execution, off/force A/B, and real counters remain C105 work.

## C100 audited-baseline harness context (2026-07-21)

- `tests/elastic/bench_rail_balance_hybrid_lsa.py` measures only the synchronous
  private checked source/return adapters.  It does not measure public Hybrid,
  Gin/RDMA/NIC, a raw kernel, or model-level speedup.
- Every sample owns a new invocation and preserves cold/warm/steady raw data.
  The global stage and transaction spans use the common single-host monotonic
  clock; the earlier rank-maximum denominator is rejected and retained in the
  development/optimization logs.
- Reports hash Git/source, `_C.so`, seven exact JIT cubin families, loaded
  NCCL/CUDART libraries, and the DeepEP-selected nvcc.  They record pre/post
  GPU/process/MPS/throttle state and explicitly disclose that historical
  extension build flags cannot be recovered from a binary hash.
- The watchdog uses a dedicated process group and bounded TERM/KILL/reap.  A
  real SIGTERM/SIGINT injection left no workers or GPU processes after fixes.
- Final frozen benchmark SHA is
  `1743adfca7edce6ac224b0d842d30b0fabcb6e0b5be2e3ec77d60776d016799b`;
  independent review is Blocker0/High0.  Benchmark commit is `056ad4d`.
- Formal acceptance additionally requires a clean commit, direct unwrapped
  execution, fresh cache, persistent JSON, at least 10 warmup and 100 steady
  samples, same-commit correctness, and stable raw distributions.

## C100 partial clean-baseline context (2026-07-21)

- Clean commit `3c54839` produced three source-H256 and three source-H7168
  direct profiler-free reports, each with 10 warmup plus 100 steady samples.
  All six pass the automatic collection gate and are checksum-preserved under
  `.cache/rail_balance/c100/formal/3c54839/`.
- They fail manual stability acceptance.  Pooled H256 median/p95/p99 are
  112.700/133.650/164.241 us with 12.15% CV.  Pooled H7168 values are
  144.728/169.219/271.935 us with 82.84% CV because one rank-4 call reaches
  2.254 ms; run medians themselves span 20.22%.
- The outlier is rank-local rather than a uniform eight-rank slowdown.  P0,
  memory clocks and throttle state remained valid, but SM clocks were
  consistently 1,500 MHz on GPUs 1/3/6 and 1,980 MHz on the others.  This is a
  host scheduling/synchronization/clock hypothesis, not a kernel conclusion.
- A return-H256 run was deliberately interrupted when the user requested a
  stop.  Watchdog exit was 130, no partial JSON was written, and no worker or
  GPU process remained.
- No Nsys/NCU or performance-path change occurred.  Resume at measurement
  stability analysis, then finish return baselines; do not discard outliers or
  proceed directly to kernel optimization.

## C100 OMP stability falsification context (2026-07-21)

- Clean commit `4211bae` ran source-H7168 three times with the sole workload
  change `OMP_NUM_THREADS=1`; each report contains 10 warmup and 100 steady
  samples and passes the automatic collection gate.
- Pooled median/p95/p99 are 140.501/160.198/175.784 us, pooled CV is 8.34%,
  and maximum is 218.716 us.  This removes the original 2.274 ms severe tail
  and materially supports host thread pressure as one cause.
- It is not yet an accepted stable baseline: run medians
  132.625/152.436/137.857 us span 14.94%.  Rank 1 starts first in 299/300
  samples and rank 7 starts last in 295/300; median post-barrier start skew is
  44.943/60.566/46.031 us while global-minus-skew medians are much tighter at
  87.775/93.237/91.610 us.
- The process inherits CPUs 0--191, while its cgroup-v1 quota is equivalent to
  96 CPUs.  Default PyTorch exposes 96 intra-op and 96 inter-op threads per
  rank; OMP=1 changes intra-op only.  The reports do not contain per-iteration
  cgroup/migration evidence, so scheduler/Gloo causality remains open.
- Original and OMP experiments are both retained.  No source/return CUDA path,
  JIT, public Hybrid path, or production code changed, and Nsys/NCU have not
  started.  Isolate rank-release skew next; only then accept source/return
  baseline distributions.

## C100 CPU-only Gloo release context (2026-07-22)

- Commit `ee41415` adds a standalone eight-rank bare-Gloo probe that never
  imports DeepEP or initializes CUDA.  Its watchdog, atomic report, raw timing,
  identity and claim boundaries pass final independent review with
  Blocker0/High0/Medium0/Low0.
- The formal three 10+100 OMP=1 runs have return-span medians
  38.693/36.319/38.028 us and a 6.54% run-median range.  rank 1 returns first
  in 98/100, 100/100 and 98/100 samples; rank 7 returns last at the same rates.
- The smallest probe median explains 80.8%/60.0%/78.9% of the three E1 median
  stage-start skews, so the registered 50% coverage and 80% ordering criteria
  both pass.  No cgroup throttle counter changes and all rank manifests remain
  CUDA-uninitialized.
- The result proves the artificial Gloo gate plus host wakeup is sufficient to
  create most global-span start skew.  It does not prove pure Gloo protocol
  cost or explain every rank-local tail.  Keep global spans raw; use the
  already reported rank-local synchronous envelope plus Nsys for operator
  attribution.  Do not algebraically subtract a median.

## C100 OMP-controlled return-H256 context (2026-07-22)

- Clean commit `d05411a` now has three accepted OMP=1 return-H256 reports,
  each with 10 warmup plus 100 steady samples: r1, r3 and r4.  Their global
  medians are 225.189/215.610/218.210 us and rank-local maximum-envelope
  medians are 218.743/212.878/214.870 us.
- Median ranges tighten to 4.44% globally and 2.76% rank-locally, but this is
  not tail stability.  Pooled global p99/max are 453.262/1419.432 us and
  rank-local p99/max are 406.292/1324.364 us.
- A separately launched Megatron `restart7` appeared during r2.  The exact
  outer PGID was 1097012; post-measurement discovered worker PIDs
  1097155--1097158.  The report passed functionality but correctly set
  `baseline_collection_eligible=false`.  It is checksum-preserved and never
  participates in accepted statistics.
- Only PGID 1097012 was terminated under standing authorization; post-kill
  process and `nvidia-smi` checks were empty.  The accepted replacement runs
  then passed every automatic gate.
- No benchmark, CUDA/JIT, runtime, Hybrid hot path, Nsys or NCU code changed at
  that checkpoint.  Return-H7168 and the Nsys notification are now complete;
  the following sections supersede that historical resume instruction.  NCU
  still requires an exact exposed invocation from the formal Nsys report.

## C100 OMP-controlled return-H7168 context (2026-07-22)

- Clean commit `57b69a9` has accepted return-H7168 reports r1/r3/r4, each
  OMP=1 with 10 warmup and 100 steady samples.  Global medians are
  341.347/333.450/335.065 us (2.37% range); rank-local maximum-envelope
  medians are 337.711/330.693/331.282 us (2.12% range).
- Typical latency is reproducible, but tails remain open.  Pooled global
  p99/max are 528.122/2021.557 us and rank-local p99/max are
  503.465/2008.428 us.  The largest accepted events occur on changing ranks
  and iterations, so no fixed-GPU or fixed-input cause is accepted.
- r2 was correctly rejected after another Codex session launched an eight-GPU
  `dlb_ep8_dispatch_demo.py` during measurement.  Its eight worker PIDs were
  absent at preflight and present in the post-run compute-app snapshot.  The
  demo had exited by diagnosis time; no unrelated process was killed.
- All four profiler-free source/return stage-width groups are now collected
  with checksums.  They support a repeatable-median claim only.  Nsys is next
  for variability and exposed-critical-path attribution; NCU and hot-path
  edits remain blocked until that evidence exists.

## C100 Nsys diagnostic instrumentation context (2026-07-22)

- The C100 benchmark owns a default-off `--nvtx` mode; production Hybrid and
  every CUDA/JIT kernel remain unchanged.  Diagnostic mode is always excluded
  from profiler-free baseline eligibility.
- Rank 0's `c100_nsys_window` delimits the steady loop.  All ranks provide
  coarse phase ranges and exact target-stage ordinals.  Start/stop range errors
  converge through WORLD gates; exceptional cleanup is rank0-only and never
  starts a new collective.
- True EP8 NVTX-on and default-off 0+1 smokes pass the report contract and
  leave GPUs empty.  Default target-stage timing semantics are unchanged; the
  broader host envelope has minimal added Python branching and is not claimed
  bit-for-bit overhead-free.
- The first pre-registered command selected
  `--capture-range-end=stop --kill=none`; the experiment below supersedes that
  command without deleting its audit history.  NCU remains blocked until Nsys
  proves an exposed exact invocation.

## C100 Nsys collection topology context (2026-07-22)

- The earlier range-trigger command is superseded by experiment.  Nsys
  2024.6's default `wait=all` reparents the multiprocessing resource tracker;
  after it becomes a zombie the inner watchdog still sees the PGID, so Nsys
  and the target wait on each other.  `stop-shutdown` does not fix this.
- `--wait=primary` preserves the benchmark's own strict process-group cleanup
  and lets Nsys exit normally.  A child-rank NVTX range is visible in a full
  report but cannot start this launched collection, even with `@*` domain
  matching.  Empty range-trigger runs are retained as failures, not reports.
- The accepted topology is full-process `cuda,nvtx,osrt` collection with
  `--capture-range=none --wait=primary`, then global-time filtering to rank 0's
  `c100_nsys_window`.  The 0+1 proof report contains 11 kernels and 16 memcpy
  records on each of all eight devices inside that interval.
- Nsys `--filter-nvtx` is process-scoped here and shows rank 0 only.  Formal
  multi-rank analysis must read `NVTX_EVENTS.start/end` and filter every CUDA
  process by the same timestamp window.  NCU and hot-path edits remain blocked
  until the formal H7168 trace establishes an exposed exact invocation.

## C100 formal return-H7168 attribution context (2026-07-22)

- The formal full-process trace is complete at clean `6339ee2`.  JSON, report,
  SQLite and verified hashes live under
  `.cache/rail_balance/c100/nsys/formal/`; diagnostic timing is never relabelled
  as profiler-free truth.
- All 800 target-stage ranges contain two adapter input D2D copies, test-only
  B1, the production-shared return kernel, production-relevant B4, one adapter
  snapshot D2D, and checked status D2H.  The longest tails are late-rank arrival
  absorbed by B1, not return-kernel variation.
- The return kernel still has material diagnostic non-overlap: pooled p50 is
  135.168 us; the cross-device interval with a return kernel but no copy or
  barrier has p50 93.871 us, 24.8% of the 378.525-us global NVTX-envelope
  median.  This is not a production critical-path fraction.  Device medians
  range from 92.480 us on rank 0 to 142.816 us on rank 6, but their observed
  clocks are a confounder until explicitly controlled or measured equal.
- Targeted NCU is now allowed only for rank/device 6,
  `c100/return/stage/steady/26`, kernel
  `rail_balance_hybrid_return_unshuffle_impl<7168,4>`, grid 256, block 32,
  60 registers/thread, 14,400-byte dynamic shared memory, Nsys duration
  142.752 us and same-name device-local ordinal 37.  B1/status are not NCU
  targets.  Default kernel/range replay is unsafe for this peer-writing kernel;
  use strict application replay, `basic`, device 6, `kill 0`, and stop rather
  than relaxing the contract if matching fails.  No performance-path edit has
  yet begun.  Older profiler-free millisecond events remain unattributed.

## C100 exact NCU basic context (2026-07-22)

- Goal remains active.  NCU 2025.1.1 matched the cross-process peer-writing
  return kernel only after retaining two command failures: invalid `--output`,
  then an `--export` run whose NVTX expression matched no push/pop range.  The
  accepted expression escapes each `/` and ends in `/`.
- The accepted report at clean `4fe7223` contains exactly one device-6,
  steady-26 `return_unshuffle<7168,4>` launch.  Ten strict application replays
  pass full EP8 functionality and leave no process/GPU allocation.  Kernel or
  range replay, relaxed matching and target killing remain forbidden.
- Launch identity is stream 26, grid 256, block 32, REG60 and 14,400-byte
  dynamic shared memory.  Observed clocks are 1.50 GHz SM / 3.20 GHz DRAM;
  cache and clocks were intentionally uncontrolled.  NCU duration is diagnostic
  replay time and cannot be compared with Nsys or profiler-free latency.
- Basic metrics show low broad utilization.  CPU/GPU-plan source review proves
  return target work is packed into 32 channels with 28 copies each while 224
  channels are empty.  This is distinct from the source-producer distribution.
  Scheduler/warp/link attribution is next; no performance-path edit is yet
  authorized.

## C100 directed NCU hypothesis context (2026-07-22)

- The same device-6 steady-26 return target completed ten strict application
  replays at clean `e4bd800` with SchedulerStats, WarpStateStats and Nvlink.
  Exactly one grid-256/block-32 launch is retained; the artifact hashes are in
  `C100_PERFORMANCE_EVIDENCE.md`.
- Every egress still has 896 records packed into 32 one-warp target channels;
  224 channels carry no payload.  NCU observes 98.238888% no-eligible cycles,
  0.017611 eligible warps/scheduler, and 49.733468 long-scoreboard cycles per
  issued instruction.  The report cannot identify the exact LDG/TMA source of
  those dependencies without PC/source counters.
- TX user bytes are exactly 12,873,728 and aggregate TX peak utilization is
  5.730689%.  Therefore local NVLink saturation is not the captured limiter.
  This remains LSA/TMA evidence, not real Gin/RDMA evidence.
- O076 authorizes only a target-channel-distribution experiment.  Quotas,
  segments, proxy slots/count, payload bytes, TMA loop, synchronization,
  default-off behavior and public API must stay unchanged.  Preserve a failed
  experiment and collect memory/source counters before any TMA-loop rewrite.

## C100 O078 accepted local result and resume context (2026-07-22)

- O077 at clean `f895eff` is functionally and sanitizer complete.  It rotates
  only pure-deficit target windows and changes the frozen C100 return layout
  from 32 channels x 28 records to 224 channels x 4 records, with 7,168 moved
  copies.  Plan schema, quota/keep/segments, proxy count/dense slots, payload
  bytes and owner/egress/destination mapping stay unchanged; `moved` and
  `moved_channel_prefix` values intentionally change physical channels.
- Its complete no-profiler A/B is mixed: source H256/H7168 median-of-run-medians
  regress 53.20%/42.06%, while return H256/H7168 improve 24.24%/22.73%.
  Source and return are independent adapters and cannot be summed into a model
  step or claimed as production end-to-end performance.
- Identity-matched cross-run Nsys diagnostics prove the tradeoff is inside the
  device kernels: pooled return
  p50 falls 135.168 to 59.232 us, while unchanged-SASS device-6 source p50
  rises 53.952 to 105.376 us.  Exact NCU controls show 2,971,264 dynamic
  instructions on device 6 versus 906,250 on low-channel device 0, without SM
  or DRAM saturation.  The root cause is the source resolver's linear scan over
  later O077 channel-prefix positions, not prefix materialization itself.
- Raw A/B, Nsys and NCU reports live under
  `.cache/rail_balance/c100/o077/`; immutable hashes and query boundaries are
  in `DEVELOPMENT_LOG.md` D089-D091 and `C100_PERFORMANCE_EVIDENCE.md`.
- O077 is retained as an auditable parent and return-parallelism proof, not an
  accepted final hot path.  O078 replaced only the canonical monotonic-prefix
  scan in `resolve_hybrid_copy` with an endpoint-checked bounded upper-bound
  lookup and final containment check.  Valid-prefix equivalence, explicit
  origin/terminal faults and the pre-existing immutable-plan boundary are
  tested.  It does not claim arbitrary post-Gate2 nonmonotonic-memory detection.
- O078's CPU/static, source/return/vnode/WORLD/default-off/codegen, true EP8 and
  focused sanitizer matrix passes.  ABI, plan, dense proxy slots, layout, TMA,
  barriers, launch, return, public API, capability bits and default-off identity
  remain unchanged at the algorithm/output/resource level; plan SASS moves one
  failure-only assertion line-number immediate and no successful-path logic.
- The first O078 no-profiler collection is retained but rejected because
  timeout 300 versus 180 changed the barrier specialization.  The formal clean
  `32b9cfd` B side repeats 12 runs with exact O077 semantic hashes.  Source
  H256/H7168 median-of-run-medians improve 42.07%/26.57%; return H256 improves
  7.77% and H7168 regresses 0.36%.  Every frozen same-label, pooled and trimmed
  gate passes; a 2.169-ms outlier still blocks tail claims.
- Matched source Nsys moves device-6 p50 105.376 to 52.496 us and the
  per-ordinal maximum 105.376 to 54.080 us.  Strict device-6 basic NCU moves
  dynamic instructions 2.971M to 0.824M with low SM/DRAM utilization.  Return
  SASS/cubin resources remain identical; the 84-pair audit manifest is tracked
  under `artifacts/`.  O078 is accepted for the local checked
  adapter, not as a model-step or real-network speedup.
- Resume C100 by freezing the smallest transfer-matrix and compute-interference
  experiments; do not reopen resolver/TMA/ABI design without new evidence.
  Real D>1 Rail/Gin runtime remains C080-H/C110's external environment gate and
  is not inferred from LSA/vnode results.

## C100 O079 transfer-matrix fixture context (2026-07-26)

- O079 stays intentionally narrow: Python C100 fixture/report instrumentation
  only, no production kernel, JIT ABI, buffer identity, capability bit or
  public API change.
- The transfer-matrix family is fixed at G8/D9/N2048-per-rank/K4/C256/Pcap7168,
  moved=7168, H256/H7168.  Named cases are fan-out, fan-in, full mesh, rot1
  and rot4.  These are matrix-shape controls; they are not new performance
  claims.
- The CPU oracle asserts exact owner-to-egress matrices.  Source reports now
  use outgoing owner row sums for rank-local logical bytes; return reports use
  incoming proxy-egress column sums.  The aggregate moved-record numerator is
  unchanged and still excludes physical transaction amplification.
- Retained mistakes: an early row builder repeated a destination within one
  token, which destination deduplication collapsed; it now selects distinct
  destinations per token.  A return highest-lane assertion was too broad and
  is restricted to the original wide-destination oracle.
- Current local evidence: `py_compile`, source oracle, return oracle on
  `c100_matrix_rot4_h7168`, normal return oracle, and `git diff --check` pass.
  All ten named matrix cases also pass true 8-GPU LSA source and return
  functionality gates.  These runs validate payload/route semantics only.
  They are not performance evidence because the visible GPUs were reported as
  `NVIDIA L20X` and GPU0 had a non-megatron Python co-tenant using about
  15 GiB.  Ruff and pyrefly are missing from the active/pinned environments
  and are recorded as missing tool evidence.  Next work is O080 compute
  interference or a separate lint-tool setup/formatting slice, not another
  O079 fixture change.

## C100 O080 compute-interference harness context (2026-07-26)

- O080 adds only a default-off diagnostic mode to
  `tests/elastic/bench_rail_balance_hybrid_lsa.py`; production kernels, JIT
  ABI, buffer identity, capability bits and public API are unchanged.
- CLI: `--interference-mode none|compute-only|concurrent`.  `none` preserves
  the existing checked-adapter benchmark and remains the only baseline-
  eligible mode.  Non-`none` modes require H7168 and directly reject H256.
- The fixed compute workload is BF16
  `[1024,7168] @ [7168,7168] -> [1024,7168]`, allocated once per rank before
  cold iteration and reused across warmup/steady.
- `compute-only` times only the GEMM window and reports timed logical moved
  bytes as zero.  The moved-record byte scope remains in
  `stage_logical_bytes_*` reference fields.  `concurrent` launches the GEMM on
  an independent stream before the checked adapter call, then synchronizes the
  compute stream before the post-stage WORLD gate.
- Smokes passing: py_compile, parser H7168 accept/H256 reject,
  non-symmetric matrix `_selected_case`, source compute-only, source
  concurrent, return concurrent, default `none` source, and schema-v4 JSON
  compute-only checks with per-rank `compute_stream_id`.  All GPU smokes are
  path checks only; no performance conclusion is accepted until the target
  machine is idle and Nsys proves actual overlap.

## C105 real-cluster runner context (2026-07-26)

- `tests/elastic/run_rail_balance_hybrid_multinode.py` is now the truthful D>1
  public off/force correctness entry point.  It is launched once per node;
  DeepEP `WORLD_SIZE`/`RANK` mean node count/node index, and the runner spawns
  local GPU ranks itself.
- It requires a clean identical Git tree and matching loaded extension SHA256,
  performs official-reference dispatch/combine checks, exact off/force A/B,
  and a two-invocation capacity fail-close case.  A process-group watchdog is
  the final liveness boundary for faults inside NCCL/Gin.
- Temporary force capability enablement exists only inside the validation
  worker and is explicitly restored before evidence is written.  Production
  capability bits remain false and production code is unchanged.
- Local contract/static evidence is 23/23 plus Ruff, Pyrefly, py_compile and D1
  rejection.  The runner cannot be truthfully executed on this single-node
  environment.  QP/NIC/wait counters and performance remain unavailable, not
  inferred.
- Tests may retain the control/error-handling structure needed to expose
  distributed failures.  Concision and branch minimization remain strict for
  production CUDA/Hybrid hot paths, not an arbitrary line-count target for the
  validation harness.

## C105 build/codegen preparation context (2026-07-26)

- `run_rail_balance_build_warmup.py` is the single local preparation command.
  It uses the existing `setup.py build_ext --inplace --force` and existing
  compile-only dispatch/combine codegen tests; it introduces no second CUDA or
  JIT generation path.
- An early draft repeated the complete four-dispatch/six-combine C080 matrix and
  duplicated its 4/4/6/6 key counts.  Simplicity review rejected that as a
  second fact source.  The accepted script compiles only the representative
  D2xG8/H7168/K8 force/legacy pair and records whatever fresh cache it produces.
- Clean `a4a97ab` execution on logical CUDA device 1 rebuilt `_C.so` to SHA256
  `9e5415128d9a3e4e926c5da4f9302a32b64ccc222cf83b994b9953a4e806dcca`.
  Four cache directories each contain nonempty CU/CUBIN/PTX/SASS files; their
  combined manifest tree hash is
  `53738af2b8dcb30957353b527e05731da399159c4dabb3ee2e2c1f952ce82d0d`.
- This preflight is deliberately labelled `HYBRID_CODEGEN_WARMUP_ONLY` with
  `real_gin_runtime=false` and `full_force_runtime_cache=false`.  The exact
  count/plan/prefix/barrier/shuffle/dispatch/epilogue/combine/unshuffle cache is
  first prepared only by a live D>1 `balanced` round trip with final parameters.

## Post-C105 hot-path simplicity audit (2026-07-26)

- The audit found no silent fallback and confirmed both force capabilities are
  still disabled.  It did find one correctness-maintenance risk: the proxy
  arena prefix and force forward-metadata width were repeated in host and
  device code.
- Commit `8e8df41` makes `rail_balance_hybrid_layout.cuh` the only fact source
  for the arena prefix and the force metadata fields/width.  Host allocation,
  manifests, dispatch writes, combine reads, and codegen probes now share it.
- Three performance changes remain hypotheses, not permission to edit: reuse
  prepared kernels and fixed plan storage, avoid reading moved payload twice
  while issuing proxy traffic earlier, and reduce per-round synchronization.
  Each needs an isolated profiler-free/nsys/ncu experiment before production
  complexity is accepted.

## C106 planner policy context (2026-07-26)

- Force mode now fixes two additional constructor inputs: policy
  `all|active|adaptive` and integer threshold percent `0..3100`.  They join
  both constructor and dispatch consensus manifests, so rank-local drift fails
  before the owning Hybrid transaction commits.
- Selection is per destination. `all` targets every local rail; `active`
  targets only rails whose original count for that destination is nonzero;
  `adaptive` begins with that active set and recruits deterministic inactive
  rails only while the current discrete peak exceeds the next-rail target by
  more than the configured percentage. Threshold zero disables the final
  movement gate and preserves the historical exact `all` plan.
- This is planner-only state. Quota, compact segments, grouped static slots,
  source shuffle, persistent dispatch, combine, and return-unshuffle remain
  one shared implementation. No policy-specific buffer, descriptor, JIT
  specialization, persistent-kernel branch, or fallback was added.
- `active` reduces the set of rails that carry payload for a destination. It
  does not claim to destroy NCCL/Gin teams or pre-created QPs. Real connection
  setup and network benefit remain C080-H/C110 evidence.

## C107 DeepEP V2 baseline harness context (2026-07-26)

- The production baseline is now unambiguous: public Hybrid
  `rail_balance='off'`, not local vnode, checked LSA, or force `all/0`.
- `tests/elastic/bench_rail_balance_hybrid_multinode.py` runs one real D>1 job
  containing symmetric off/force blocks.  Each force dispatch ticket is
  consumed by the matching combine before the next iteration.
- The primary metric is the per-ordinal maximum rank-local current-stream CUDA
  event envelope for public dispatch+combine.  It includes host/world-gate
  idle gaps and is not a kernel-time sum.  Cross-node absolute clocks are never
  subtracted.
- One official-reference probe per mode and a second same-input timed-path
  probe are untimed.  Every measured block's last rank digest must equal its
  corresponding timed-path probe.
- WORLD consensus now covers warmup, steady count, symmetric order, timeout,
  profiler declaration, co-tenant override, run ID and output identity in
  addition to the C105 source/extension/topology/config identity.
- Formal reports require clean identical pre/post source and extension,
  10+100 samples per block, direct unwrapped execution, stable GPU identity,
  P0/no throttle, MIG/MPS off, default compute mode, and no unexpected GPU
  process.  A diagnostic override preserves samples but forbids a claim.
- CPU evidence is 5/5 new benchmark contracts plus the unchanged C105 23/23
  suite.  The current workspace cannot execute real D>1 Gin, so the report
  schema and commands are ready but no DeepEP V2 speedup number exists yet.

## Hop-aware branch context (2026-07-31)

- Branch `feat/rail-balance-hop-aware` starts from the existing prototype and
  keeps the native public capability closed.  The semantic source of truth is
  the endpoint-aware Python reference plus
  `docs/rail_balance_hop_aware_audit.md` and
  `docs/rail_balance_hop_aware_design.md`.
- DeepEP shares one payload across the target local-rank mask `T`; therefore
  the implementation retains `(owner, destination node, target mask, egress)`.
  The literal per-copy rule remains: endpoint Rails are one-hop candidates and
  any Rail outside owner plus targets is two-hop.
- One-hop and adaptive use one record/resolution ABI, one source shuffle, one
  vnode pack/demux specialization, and the existing destination/combine path.
  Adaptive first completes one-hop and then performs bounded, profitable
  third-Rail moves.  No expert endpoint changes.
- Current local evidence: planner CUDA 8/8; one-hop and adaptive 4x2 round trips
  pass on eight H200s; API and constructor preflight are 9/9; public ticket and
  H4b/H4c source contracts pass.  Adaptive's synthetic diagonal proof moves 8
  of 16 copies under a 50% cap and returns exact BF16 values and weights.
- Current planner selection is deterministic single-lane code.  It is a
  correctness implementation pending profiler-free timing and Nsys/NCU
  attribution, not an accepted hot-path optimization.
- Single-node vnode results have diagnostic scope only.  Real Gin/RDMA,
  receiver NIC load, rank-max latency, and capability enablement remain
  deferred to truthful D>1 hardware.
- HA050 adds one `bench_rail_balance_hop.py` entry for reference, vnode and
  multinode. It reuses the existing strict DeepEP-V2-off A/B runner rather
  than creating another distributed lifecycle. Endpoint-count manifests,
  planner/path/load fields and claim scopes are shared; real runtime path
  counters remain missing evidence until the kernel exports them.
