# Rail Balance Checkpoints

Checkpoint states: `PLANNED`, `IN_PROGRESS`, `PASS`, `FAIL`, `BLOCKED`,
`DEFERRED_ENVIRONMENT`, or `SUPERSEDED`.

| ID | Scope | State | Evidence / exit condition |
| --- | --- | --- | --- |
| C000 | Read all supplied context and audit the current implementation | PASS | Both shared chats decoded/read; prompt design reviewed; code anchors and base commit recorded in `CONTEXT.md`. |
| C001 | Create isolated branch and persistent records | PASS | Verified branch `feat/rail-balance-prototype`; plan, context, development, optimization, and checkpoint records are present under `docs/rail_balance/`. |
| C010 | Existing-build and minimal EP8 correctness baseline | PASS | In-place SM90 build/import passed; EP8 direct-path first FP8/alignment-128 case passed at hidden 256 with performance disabled. The invalid hidden-128 attempt is retained in `DEVELOPMENT_LOG.md`. |
| C020 | CPU planner contract and deterministic oracle | PASS | Independent review passed; the 28-test direct runner covers 8,184 exhaustive quota cases, 64 seeded route/slot cases, strict logical-plan mutation validation, channel striping, and atomic capacity bypass. |
| C030 | GPU planner equals CPU oracle | PASS | Fresh-cache private JIT matches the canonical CPU oracle in 876 device-cases across isolated devices 0 and 7; invalid ABI, non-default stream, int64 offsets, and stable same-process cross-device rejection pass independent final review. |
| C040 | 8x1 copy-exact, one-shot direct-final source shuffle | PASS | Fresh-cache 8-GPU H7168 LSA/TMA run moves 256 records from all eight owners and exactly fills all eight 32-record egress arenas; full-record bytes, headroom poison, generation, physical-capacity caller bypass, source immutability, global coverage, independent review, and original EP8 direct regression pass. |
| C050 | One-shot proxy publication ordering and liveness | PASS | CPU oracle passes 34/34 and an independent 167,481-state exhaustion. Fresh-cache 8-GPU SM90 v6 passes 16 generations plus forced holes, stale values, byte-exact live payload consumption, a slow consumer, progress-aware late gating whose aggregate wait exceeds its watchdog, bounded missing-slot failure, and system-scope error arbitration. C040 and original EP8 regressions pass; two independent reviews report no Blocker/High/Medium. Ring reuse remains deferred. |
| C060 | Restricted 4x2 full dispatch/combine round trip | PASS | CPU full-egress oracle passes 36/36; six H7168/K4 SM90 kernels compile with 0 spill; fresh-cache H256 and H7168 8-GPU loops pass exact payload/route/combine checks, an independent layout formula, and an intact 4 KiB tail guard. A post-review fresh-cache run also passes token-rotated lane-to-expert routing. C040, C050, original EP8, core rebuild, and two independent final reviews pass with no unresolved Blocker/High/Medium. |
| C061 | 2x4 per-destination multi-node emulation | PASS | Fresh-cache H256 and H7168 eight-GPU runs pass 33 destination copies, six required per-destination moves, three quota holes, 72 expert contributions, and 282 byte-exact global coverage records. Planner 38/38, C060, original EP8, full build, and independent review pass with 0 Blocker/High/Medium. |
| C070 | Minimum cross-call route replay | PASS | H256/H7168 eight-GPU capture plus two manifest-free replays pass exact independent partial/output, immutable owning snapshots, zero status, and intact guard. One corrupt live route reports RouteMismatch=4 and exits collectively. Review closes 0 Blocker/High/Low; one production-only Medium requires cross-rank preflight consensus before C080 exposes the path. |
| C080 | Optional Hybrid integration with default off | IN_PROGRESS | Minimal staged plan is frozen after independent review with 0 Blocker/High. The sidecar/descriptor/ready/header draft was rejected; the accepted force-v1 uses grouped static slots, a 4-byte transit key, existing Tag0, and one post-unshuffle barrier. GPU schedule materialization exists, but force dispatch/combine remain untouched. |
| C080-A | Contract, baseline fingerprints, and off identity | PASS | Keyword-only off/force config and a disabled dual capability gate are committed. Off preserves old constructor/calculator/runtime/JIT/handle identity; API 6/6, legacy goldens 4/4, default-off EP8 smoke, and independent review pass. |
| C080-B | GPU count, plan, and grouped proxy schedule | PASS | B1 matches 69 exact H200 materializer cases. B2 then publishes per-rank compact counts and passes a fresh-cache true 8-GPU LSA snapshot: all fourteen outputs equal the CPU oracle for variable/zero N, phase reuse, Pcap 5/4, route/config recovery, C1024/D32, int64 seed, and non-default stream. No manifest/descriptor/ring exists. |
| C080-C | Minimal force arena and collective preflight | IN_PROGRESS | The checked 32B/2MiB force arena formula, EP8 size agreement, private prepare/finish/abort transaction, local-only LSA barrier, and Gloo-scaffolded Gate1/Gate2 fault convergence pass. Production fixed-tensor WORLD gates and the full dispatch state machine remain; public force stays unavailable. |
| C080-D | Shared-core LSA data path without a supplied manifest | PLANNED | 8x1 shuffle and 4x2/2x4 H256/H7168 vnode loops consume the GPU materializer and pass exact route/combine/fault checks. Evidence is labeled `VNODE_FUNCTIONAL_PASS`. |
| C080-E | Isolated Hybrid dispatch specialization | PLANNED | New force-only dispatch header/runtime compiles synthetic 2x4/4x2 instances, carries only compact `proxy_slot` in transit `linked_list_idx[0]`, consumes descriptor-free grouped slots, exposes counters, and leaves old JIT inputs untouched. Evidence is codegen only locally. |
| C080-F | Isolated Hybrid combine plus return-unshuffle | PLANNED | Moved results and weights land in collision-free `proxy_return[p]`, group/preserved-dispatch state derives the owner and reduce row without a return header, the mandatory post-unshuffle LSA barrier precedes the legacy epilogue, and matching synthetic specializations compile. |
| C080-G | Local closure and independent review | PLANNED | Default-off regression, vnode shared-core, codegen, fault, focused sanitizer, and C000-C070 compatibility gates pass with no unresolved Blocker/High; state becomes `LOCAL_COMPLETE / MULTINODE_PENDING`. |
| C080-H | Real multi-node Rail/Gin runtime gate | DEFERRED_ENVIRONMENT | Requires truthful scaleout>1 Rail teams; only this gate can mark C080 `MULTINODE_PASS`. |
| C090 | Exhaustive local correctness, sanitizer, and fault campaign | IN_PROGRESS | Compute Sanitizer 2025.1 memcheck and synccheck report zero errors on the fresh-cache minimal B1 materializer matrix. True 8-GPU LSA functional/fault tests pass, while multi-process sanitizer, source shuffle/combine, race/init checks, and the broader matrix remain. |
| C100 | Controlled local performance and operator optimization | PLANNED | All eight GPUs are currently idle. Benchmark and iterate planner, assignment, direct-final LSA shuffle, publication, channel policy, return-unshuffle, transfer matrices, and compute interference; retain negative/noisy results. |
| C105 | Network-ready validation package | PLANNED | One-command build/warmup and off/force A/B runs, stable JSON counters/errors, canonical cases, and evidence labeling minimize real-cluster bring-up time. |
| C110 | Real-cluster Gin/RDMA scale and performance validation | DEFERRED_ENVIRONMENT | Requires multi-node rail/NIC environment; C080-H owns first correctness, while C110 owns repeated scale, performance, and fabric characterization. |
| C120 | Production fusion and auto policy | PLANNED | Fused implementation passes correctness and evidence-based performance gates. |

## Checkpoint update template

```text
ID / state / date:
Base commit and diff summary:
Commands/tests:
Results:
Failures retained:
Artifacts:
Decision:
Next checkpoint:
```
