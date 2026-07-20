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
  legacy layout: row `destination` when `D<=K`, otherwise the first top-k lane
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
  inactive slots. Full vnode round trip and return unshuffle remain open.
