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
| C080 | Optional Hybrid integration with default off | PLANNED | Regression identity and forced virtual-hybrid correctness pass. |
| C090 | Sanitizer and fault-injection campaign | PLANNED | Required tools clean or every limitation explicitly classified. |
| C100 | Controlled local performance campaign | DEFERRED_ENVIRONMENT | Run when GPUs are idle; contended timings are not acceptance evidence. |
| C110 | Real-cluster Gin/RDMA validation | DEFERRED_ENVIRONMENT | Requires multi-node rail/NIC environment. |
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
