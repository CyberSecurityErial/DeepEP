# Rail-balance review guide

## What is being reviewed

This branch adds an opt-in source-side rail balancer to DeepEP V2 Hybrid.
The native path remains the production baseline:

```python
rail_balance="off"
```

The candidate path is selected at buffer construction:

```python
rail_balance="force"
rail_balance_policy="all"       # all | active | adaptive
rail_balance_threshold_percent=0
```

The review branch was rebuilt from 238 development commits into a short,
dependency-ordered series. Before this guide was added, its tree object was
identical to `5199a04bea86cae03f52e1316672d7b092b62dd5`. The original history,
including rejected experiments and reversions, is preserved at:

```text
archive/rail-balance-prototype-pre-review-20260729
```

Base commit: `dd758caf451848bd150e1046af3d0a73e5fff38d` (`origin/main`).

## The idea in one minute

The balancing happens on the source node immediately before data is submitted
to a NIC rail.

```text
dispatch

owner GPU i
  | keep: native send slot on rail i
  | move: LSA/TMA copy to the final proxy slot on egress GPU k
  v
egress GPU k -- Gin put on rail k --> destination ingress GPU k
                                      |
                                      v
                               native-style forwarding

combine

destination expert
  |
  v
destination ingress GPU k -- Gin return on rail k --> source proxy GPU k
                                                       |
                                                       v
                                          LSA/TMA return-unshuffle
                                                       |
                                                       v
                                                original owner GPU i
```

The planner counts token copies per `(source rail, destination server)`, chooses
quotas, and emits compact retained/moved prefixes. Proxy slots are assigned
statically; the hot path does not allocate a slot with a per-copy peer atomic.
Dispatch metadata carries the proxy slot and original source identity, so
combine can return through the chosen rail and then restore the result to the
original owner/token.

Three policies share the same data path:

- `all`: balance across every local rail.
- `active`: use only rails that already send to that destination; this limits
  connection spread and local shuffling.
- `adaptive`: start with active rails and recruit inactive rails only while the
  discrete peak exceeds the next-rail target by more than the threshold.

The threshold is a count-based planner rule, not a bandwidth model. When the
world-wide plan contains no moves, force mode aborts the private plan and calls
the native Hybrid path. The no-move decision is cached for 31 subsequent calls
with the same checked geometry, then re-evaluated.

## Recommended review order

Do not start with the tests or the historical logs. Review the first five
production layers, then the final two-node performance patch. Use the test,
benchmark, and documentation commits only to challenge an invariant.

```bash
git log --reverse --oneline origin/main..HEAD
```

| Commit | Purpose | Main review question |
|---|---|---|
| `6cbb6a2` | Planner, layouts, protocol primitives | Are counts, quotas, prefixes, capacities, and errors defined once and bounded? |
| `b2861d9` | CUDA dispatch/combine/shuffle data path | Are slots unique, payload publication ordered, and routes invertible? |
| `61ff785` | JIT specs and prepared launchers | Are specializations validated before launch and keyed by every compile-time fact? |
| `30a2a99` | C++ `ElasticBuffer` transaction owner | Can prepare/plan/commit/finish be used only in a legal order? |
| `9b94650` | Python public lifecycle and world gates | Do all ranks make the same fail-closed decision, and is a handle consumed once? |
| `b52e483` | CPU/GPU/reference tests | Does each production invariant have an independent oracle or fault test? |
| `a6d5f56` | Local and multi-node runners | Is DeepEP V2 `off` the only claimed production baseline? |
| `05015b4` | Context, checkpoints, failures, evidence | Are rejected results retained and claims kept inside their evidence boundary? |
| `e08d73e` | Latest two-node correctness/performance patch | Does the optimized moved path preserve ordering and native no-move behavior? |

Useful commands:

```bash
git show --stat 6cbb6a2
git show --stat b2861d9
git show --stat 61ff785
git show --stat 30a2a99
git show --stat 9b94650
git show --stat e08d73e
```

## Production source reading map

### 1. Facts and memory layout

Read these first:

- `deep_ep/include/deep_ep/common/rail_balance_layout.cuh`
- `deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh`
- `deep_ep/include/deep_ep/common/rail_balance_protocol_layout.cuh`

Check that one layout object owns every byte offset, arithmetic is checked,
arena alignment is explicit, and proxy capacity is constructor-fixed. The
forward metadata width is `3 + 2 * topk`: source token, linked-list predecessor,
proxy slot, followed by the original route and slot for each top-k entry.

### 2. Planner

Read:

- `deep_ep/include/deep_ep/impls/rail_balance_hybrid_plan.cuh`
- planner runtimes at the top of
  `csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp`

For every destination, verify:

```text
sum(quota) == sum(count)
max(quota) - min(quota) <= 1          # for the selected rail set
moved == sum(surplus) == sum(deficit)
proxy_required[egress] <= proxy_capacity
```

Also verify token destinations are deduplicated before counting. Several top-k
experts on the same remote server require one scaleout payload copy, not one
copy per expert.

### 3. Dispatch data path

Read:

- `deep_ep/include/deep_ep/impls/rail_balance_hybrid_dispatch.cuh`
- `deep_ep/include/deep_ep/impls/rail_balance_hybrid_shuffle.cuh`
- dispatch launch adapters in
  `csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp`

The important split is retained versus moved copies. Retained data stays on
the owner's rail. Moved data is written directly into the egress GPU's final
proxy send slot, then that egress submits Gin. There must be no hidden second
payload copy or data-dependent heap allocation.

Review publication in this order:

1. payload and metadata reach the proxy slot;
2. TMA writes receive global visibility;
3. the egress observes a ready slot only after step 2;
4. Gin completion is ordered before the final remote tail is published;
5. destination rank counts, linked lists, and tails include retained and moved
   records exactly once.

### 4. Combine inverse

Read:

- `deep_ep/include/deep_ep/impls/rail_balance_hybrid_combine.cuh`
- combine and return-unshuffle launch adapters under
  `csrc/kernels/elastic/`

A moved result returns to its source proxy slot, not directly to its original
token slot. The proxy route must recover both original owner GPU and original
token index. Two owners with the same local token index must never collide on
one proxy GPU.

### 5. Runtime lifecycle

Read the rail-balance methods in `csrc/elastic/buffer.hpp` in this order:

```text
rail_balance_hybrid_dispatch_prepare
rail_balance_hybrid_plan_finish
rail_balance_hybrid_dispatch_commit
rail_balance_hybrid_dispatch_finish
rail_balance_hybrid_combine_prepare
rail_balance_hybrid_combine_commit
rail_balance_hybrid_plan_abort / combine_abort
```

Then read `_dispatch_rail_balance_force` and
`_combine_rail_balance_force` in `deep_ep/buffers/elastic.py`.

The Python world gates are deliberate. A local validation, plan, or combine
failure cannot let one rank enter a collective data path while another rank
raises. `_RailBalanceForceTicket` is shared by copied handles; exactly one
combine changes it from live to consumed.

## Focused review of the latest optimization (`e08d73e`)

This is the highest-value diff to review after understanding the lifecycle:

```bash
git show --stat e08d73e
git diff e08d73e^ e08d73e -- \
  deep_ep/include/deep_ep/impls/rail_balance_hybrid_dispatch.cuh \
  deep_ep/include/deep_ep/impls/rail_balance_hybrid_shuffle.cuh \
  deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh \
  csrc/elastic/buffer.hpp deep_ep/buffers/elastic.py
```

Review it as five independent claims:

1. proxy capacity is at least the source token capacity, so a dense static slot
   assignment cannot overlap;
2. cross-GPU source publication has a local-rank barrier before egress
   consumption;
3. retained and moved records rebuild final rank metadata and remote tails in
   one consistent index space;
4. per-channel/per-destination Gin requests delay tail publication until all
   payload puts complete;
5. a zero-move world plan falls back to native DeepEP V2 and periodically
   rechecks instead of paying the moved path forever.

The patch intentionally touches two upstream common files:

- `common/ptx.cuh` adds `fence.proxy.async.global` for TMA visibility;
- `common/layout.cuh` reserves per-channel/per-scaleout Gin request storage.

These changes are the source of the legacy-golden issue described below and
deserve an explicit accept-or-refactor decision.

## Review invariants

### Native path

- `rail_balance="off"` selects the original dispatch/combine implementation.
- Force-only tensors, prepared runtimes, tickets, and world gates do not run in
  off mode.
- No speedup is claimed against vnode, a local adapter, or an earlier force
  implementation; the baseline is native DeepEP V2 Hybrid in the same binary.

### Planner and policy

- Every source token is counted once per distinct remote destination server.
- Policy changes only planner output; it does not fork the data-path kernels.
- `active` never introduces a previously inactive rail.
- `adaptive` recruits rails monotonically and respects the integer threshold.
- Capacity failure is explicit and world-consistent; there is no fallback to a
  partially balanced schedule.

### Data and synchronization

- A static proxy slot has one writer and one lifecycle generation.
- Payload becomes visible before ready/tail publication.
- A remote tail never advertises an unfinished Gin payload put.
- Final linked lists and counts describe the physical receive order.
- Return unshuffle is the exact inverse of the proxy route.

### Lifecycle

- A dispatch invocation has one owner buffer and monotonically increasing ID.
- All participating ranks pass the same manifest and phase.
- Failure before commit is abortable; failure after an unsafe partial
  collective makes the buffer terminal.
- A force handle cannot be cached for another dispatch and can be combined
  exactly once, including when the Python handle was copied.

## Tests versus production code

The following are validation tools, not public runtime features:

- `rail_balance_reference.py`: slow CPU oracle;
- `rail_balance_hybrid_vnode_reference.py`: virtual-node closed-loop oracle;
- `rail_balance_vnode*.cuh` and vnode bindings: single-node emulation of a
  multi-node route;
- `*_lsa.py`: real local peer-access tests without claiming NIC behavior;
- benchmark/report scripts: measurement and evidence packaging.

VNode code may be removed from a final minimal production patch after the real
Gin path, route replay, and faults have equivalent coverage. It must not be
mistaken for the production transport.

## Current verification status

The rewritten commits preserve the exact source tree. On the review host:

```text
PASS  python3 -m compileall -q deep_ep tests/elastic
PASS  test_rail_balance_plan.py                       38 tests
PASS  test_rail_balance_hybrid_reference.py           15 tests
PASS  test_rail_balance_hybrid_policy.py                8 tests
PASS  test_rail_balance_hybrid_vnode_reference.py       7 tests
PASS  public lifecycle fake-runtime suite
PASS  multi-node benchmark contract                    5 tests
PASS  constructor fail-closed preflight                 9 tests
PASS  git diff --check
BLOCK the review worktree has no locally built `deep_ep._C`; the matching
      feature worktree extension was used for import-dependent host suites
BLOCK pytest is absent; direct script entry points were used
WARN  targeted Ruff check reports 3 findings inherited from `origin/main`
      and 7 style-only findings in rail-balance tests
```

Re-run the pure CPU checks with:

```bash
PYTHONPATH="$PWD/tests/elastic" python3 -B \
  tests/elastic/test_rail_balance_plan.py
PYTHONPATH="$PWD/tests/elastic" python3 -B \
  tests/elastic/test_rail_balance_hybrid_reference.py
PYTHONPATH="$PWD/tests/elastic" python3 -B \
  tests/elastic/test_rail_balance_hybrid_policy.py
```

### Known review blocker: legacy identity golden

`tests/elastic/test_rail_balance_hybrid_legacy_golden.py` currently fails in
the latest feature tree. This predates history reconstruction; tree identity
proves it was not introduced by commit regrouping.

Direct SHA mismatches:

```text
deep_ep/include/deep_ep/common/ptx.cuh
deep_ep/include/deep_ep/common/layout.cuh
```

All six recursive dispatch/combine include hashes consequently differ too.
Run independently, the canonical layout and source-order assertions pass.

Resolve this explicitly before merge:

1. if upstream Hybrid files must remain byte-identical, move the visibility
   helper and request workspace into rail-balance-specific storage; or
2. if these two minimal common changes are accepted, update the golden and
   rename/document the test so it no longer promises byte identity.

Do not merely refresh the golden without choosing which contract is intended.

`test_rail_balance_hybrid_api.py` also needs one expectation update: the final
zero-move optimization adds `_rail_balance_zero_move_bypass_budget` and
`_rail_balance_zero_move_common_fields` to the force buffer, while the test's
exact private-attribute set still expects the pre-optimization set. The public
lifecycle tests for the optimized behavior pass; this is a stale API contract
expectation, not a history-rewrite difference.

## Two-node baseline command

The complete, auditable command sequence is in
`docs/rail_balance/BASELINE_BENCHMARK.md`. Its decisive A/B alternates fresh
buffers using identical inputs and changes only `off` versus `force`.

After defining `run_ab` on both nodes exactly as documented:

```bash
run_ab balanced-all0 balanced all 0 31101
run_ab onehot-all0 one_hot all 0 31102
run_ab onehot-adaptive20 one_hot adaptive 20 31103
run_ab onehot-active0 one_hot active 0 31104
run_ab twohot-all0 two_hot all 0 31105
```

Read:

```text
comparison.roundtrip_cuda_ms.speedup_baseline_over_candidate
eligibility.performance_claim_eligible
```

Only an eligible, profiler-free ratio above one is a speedup over DeepEP V2.
Nsys/NCU reports explain a result; they do not replace the A/B result.

## Reviewer sign-off

Before merging, record answers to these questions:

1. Is the common `ptx.cuh`/`layout.cuh` change accepted, or must it be isolated?
2. Does every policy preserve planner conservation and capacity bounds?
3. Is cross-GPU payload publication ordered before proxy consumption?
4. Are Gin payload puts complete before a destination tail becomes visible?
5. Does moved combine return exactly once to the original owner/token?
6. Can any rank fail while a peer proceeds into a collective stage?
7. Does no-move traffic genuinely execute native DeepEP V2?
8. Does the reported two-node gain use the same checkout, route, geometry,
   precision, inputs, and clean profiler-free measurement window?

Use the archive branch for forensic history. Keep review comments against this
layered branch so production design feedback is not mixed with abandoned
experiments.
