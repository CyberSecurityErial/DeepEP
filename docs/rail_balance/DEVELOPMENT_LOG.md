# Rail Balance Development Log

This is an append-oriented engineering record. Failed attempts remain here even
if their code is reverted. Each entry records evidence, interpretation, and the
next safe action.

## 2026-07-19 — D000: Context ingestion and initial audit

### Completed

- Read the full task prompt, including production and single-node designs.
- Attempted to open both ChatGPT share links with the generic web reader.
- The reader returned `Cache miss` for both links.
- Retried through the public share pages, decoded the embedded serialized
  messages, and read both complete shared answers.
- Audited the current Git baseline: clean `main` at
  `dd758caf451848bd150e1046af3d0a73e5fff38d`.
- Confirmed the key Hybrid dispatch/combine/layout/handle/test anchors.
- Confirmed an important coverage limitation: physical one-node EP8 selects the
  direct kernels, so existing single-node smoke tests cannot prove the new
  Hybrid route.

### Failures/blockers retained

- Generic share-link fetch failed with a cache miss. Workaround: decode the
  public page payload directly; no context was omitted.
- Sandboxed shell startup repeatedly failed with:
  `bwrap: Creating new namespace failed: nesting depth or
  /proc/sys/user/max_*_namespaces exceeded (ENOSPC)`.
  This is an execution-environment limitation, not a repository failure.
  Commands that require the shell may need the already-approved/full-access
  execution path until the namespace condition changes.
- No built `_C` extension or repository build artifacts were visible in the
  first test audit. Baseline execution must begin with build/JIT verification,
  not assume the Python test is immediately runnable.
- The `fmt` submodule was not initialized in the first audit.

### Resource policy clarified with the user

- Other GPU jobs do not automatically block correctness work.
- Run small functional shapes when the memory budget safely fits.
- Do not use shared-GPU timings as performance evidence.
- Run local planner/shuffle/overlap benchmarks when GPUs become idle.
- Defer only true Gin/RDMA/NIC/fabric conclusions to a multi-node rail cluster.

### Decision

Use a staged internal prototype. Do not begin by modifying the persistent Hybrid
kernel or widening the public API.

### Audit corrections incorporated into the frozen plan

- `scaleout_rank` is the server dimension; `scaleup_rank` is the local GPU/rail.
- Exclude local-destination copies from rail counts.
- The standalone launcher belongs under `csrc/kernels/elastic/`, not a
  non-existent Python JIT-kernel tree.
- First MVP is only 8x1, copy-exact, move-only, one-shot static slots, and BF16
  source shuffle. Full transport abstraction, rings, auto policy, other
  strategies, and Hybrid fusion are later checkpoints.
- Proxy slot allocation must be capacity-aware per channel and unified across
  retained/proxy traffic.
- Proxy payload buffers are constructor-reserved symmetric memory.
- Proxy replay metadata must not be appended to `recv_src_metadata`.
- Existing tests do not implement autograd/backward coverage.

### Next action

Establish branch/log checkpoint, then verify the build/JIT baseline and implement
the CPU planner contract.

## 2026-07-19 — D001: Branch and record structure

### Completed

- Created branch `feat/rail-balance-prototype` from the clean base commit.
- Added the persistent plan, context, development, optimization, and checkpoint
  records under `docs/rail_balance/`.

### Decision

Checkpoint status is tracked in `CHECKPOINTS.md`; coherent code milestones will
also receive local Git commits. Logs are updated before moving to the next work
package so context survives long runs and conversation compaction.

### Next action

Verify branch state and record C001, then start WP0 baseline/build inspection.

## 2026-07-19 — D002: C001 verified and fork target accepted

### Completed

- Verified the active branch is `feat/rail-balance-prototype`.
- Verified all five persistent records are present under `docs/rail_balance/`.
- Marked C001 `PASS` and C010 `IN_PROGRESS`.
- Recorded `https://github.com/CyberSecurityErial/DeepEP` as the checkpoint fork.

### Remote policy

- Push feature/checkpoint branches only.
- Do not modify or force-push the fork's `main` branch.
- Push after a coherent, tested checkpoint rather than every partial edit.

### Next action

Complete the build/import baseline while implementing the CPU planner oracle in
parallel with environment setup.

## 2026-07-19 — D003: C001 commit attempt blocked by missing identity

### Completed

- Added the user-provided fork as remote `fork`.
- Staged all five C001 records.
- `git diff --cached --check` passed.
- Staged scope: five files, 672 inserted lines.

### Failure retained

The checkpoint commit was rejected because this repository has no configured
`user.name` or `user.email`. No author identity was inferred or fabricated.
The staged files remain intact and development can continue.

### Next action

Continue C020. Create the local checkpoint commit after an explicit repository
author identity is available.

## 2026-07-19 — D004: In-place build/import baseline passed

### Completed

- Initialized `third-party/fmt` at the repository-locked commit
  `a4c7e17133ee9cb6a2f45545f6e974dd3c393efa`.
- Built `deep_ep._C` with the designated Python environment using
  `setup.py build_ext --inplace`.
- Verified both `deep_ep` and `deep_ep._C` import from this source checkout.
- Build target was SM90 and linked the environment's NCCL/NVSHMEM libraries.

### Resource observation

At the post-build check, all GPUs had enough free memory for a very small
correctness smoke, but GPUs 0-3 were near full compute utilization and GPU 6 had
an unrelated large allocation. Correctness may proceed with conservative shapes
and generous timeouts; no timing from this window is performance evidence.

### Planner correction discovered

A count-blind rotating remainder is not move-minimal. The CPU oracle will first
assign `base + 1` quotas to rails whose counts exceed `base`, using a canonical
ring order only for ties. The counterexample `[4, 0, 0]` is a required test.

### Next action

Finish and exhaustively validate C020, then run the smallest safe direct EP8
smoke for C010.

## 2026-07-19 — D005: CPU planner candidate passes exhaustive and randomized tests

### Implemented

- Added `tests/elastic/rail_balance_reference.py`, a CUDA-independent exact-copy
  planner contract covering normalized destination copies, count-aware exact
  quota, stable surplus/deficit segments, unified retained/proxy static slots,
  deterministic channel re-striping, generation-aware slot keys, and atomic
  whole-plan capacity bypass.
- Added `tests/elastic/test_rail_balance_plan.py` with a dependency-free direct
  runner so the oracle can be validated before the GPU launcher exists.
- Exhaustively compared the oracle's moved-copy count with every base/base+1
  quota choice for six remainder seeds, one through five rails, and counts zero
  through three: 8,184 count-vector/seed cases.
- Added 64 fixed-seed multi-destination route cases covering masks, duplicate
  top-k destinations, local-destination exclusion, zero-token owners, one to
  eight rails, one to four channels, deterministic input reordering, copy
  conservation, exact egress quota, and slot uniqueness.

### Failures retained

1. The first test command used `python -m pytest`, but the designated DeepEP
   environment does not contain pytest. No dependency was installed merely for
   this checkpoint; the test module now also has a direct runner.
2. The first direct run failed one capacity-bypass assertion. The fixture used
   counts `[[2, 2], [1, 1]]`, which are already exact-balanced per destination,
   so the correct minimum moved count is zero. The fixture was corrected to
   `[[3, 3], [1, 1]]`, which genuinely requires two proxy copies. The planner
   behavior was unchanged.

### Verification

```text
/home/chen/.cache/deepep-sjlgpt/bin/python tests/elastic/test_rail_balance_plan.py
PASS 19 rail-balance planner tests

/home/chen/.cache/deepep-sjlgpt/bin/python -m py_compile \
  tests/elastic/rail_balance_reference.py \
  tests/elastic/test_rail_balance_plan.py
PASS

git diff --check
PASS
```

These checks are CPU-only and consumed no GPU memory.

### Next action

Resolve independent review findings, mark C020, then start the internal GPU
planner contract while continuing the C010 minimal correctness baseline.

## 2026-07-19 — D006: C020 independent review closed

### Findings fixed before checkpoint

- Replaced first-nonfull proxy channel selection with lowest-occupancy selection
  and canonical ring-order tie breaking. With four moved copies and four
  unconstrained channels, the old policy produced `0/0/0/0`; the corrected
  policy and discriminating regression produce `0/1/2/3`.
- Made `validate_count_plan` recompute the count-aware canonical minimum-move
  quota and made static slot materialization validate its candidate before any
  staging.
- Added strict integer validation for seeds, capacities, generation, copy
  fields, and shape parameters; Python `bool`/`float` values no longer silently
  enter slot keys or occupancy maps.
- Added real destination-dependent remainder rotation, independent expansion of
  segment suffix ordinals, corrupted-plan rejection, metadata-capacity bypass,
  and full identity assertions after a capacity failure that occurs after
  temporary retained assignments were staged.

### Independent result

The read-only reviewer reran the direct suite and reported C020 `PASS` with no
blocker, high, or medium correctness issue. Two non-blocking follow-ups are
carried forward:

- C020's scalar `proxy_payload_capacity` is a global policy budget. C040 must
  additionally model and enforce physical per-egress payload capacity.
- The validator accepts any semantically valid continuous suffix pairing, while
  the builder emits one stable canonical owner-to-egress order. C030 equality
  tests must compare the GPU output to that exact builder order.

### Decision

Mark C020 `PASS` and use the frozen CPU oracle as the source of truth for C030.

## 2026-07-19 — D007: C010 minimal EP8 direct-path baseline passed

### Device and resource gate

- PyTorch reported eight visible CUDA devices with compute capability 9.0 and
  approximately 140 GiB each.
- Before the smoke, all devices reported zero utilization; the minimum free
  memory was approximately 32 GiB. The run was correctness-only and no timing
  is accepted as performance evidence.

### Failure retained

The first smoke used `hidden=128`. Existing `combine.cuh` rejected that JIT
specialization with its compile-time `Invalid hidden` assertion because the
BF16 combine vector width requires `hidden % 256 == 0`. This was a bad test
shape, not a rail-balance change. Production code was not modified.

### Passing command and scope

The same EP8 command was rerun with the minimum legal `hidden=256`, 32 tokens,
top-k 1, eight experts, `allow_hybrid_mode=0`, `EP_DISABLE_GIN=1`,
`--test-first-only`, and `--skip-perf-test`. It completed with exit code 0.

This establishes the in-place build and the first direct-path FP8,
alignment-128 dispatch/combine case on eight GPUs. It does not cover Hybrid,
Gin/RDMA, BF16, the full semantic matrix, or performance.

### Decision

Mark C010 `PASS`. Start C030 as a private counts-to-plan GPU JIT kernel; do not
couple its algorithm check to LSA gather or virtual-node transport yet.

## 2026-07-19 — D008: C030 GPU count planner candidate passes two-device equality

### Implemented

- Added a private `_C._build_rail_balance_plan` API; no public `ElasticBuffer`
  method or PyTorch operator was added.
- The input is one complete contiguous CUDA `int32[G,D]` count matrix. C030 does
  not scan top-k, gather through LSA, construct an NCCL communicator, or touch a
  symmetric communication buffer.
- Added a one-warp-per-destination JIT kernel. Lane 0 intentionally runs the
  small `G <= 32` canonical algorithm serially so quota winner and
  owner-to-egress segment order match the CPU oracle exactly.
- Fixed output ABI: quota and keep matrices, destination-major fixed segment
  storage, per-destination segment counts, and total moved copies. Unused
  segment cells are initialized to `-1`.
- Added a dependency-free CUDA direct runner with exact shape/dtype/device/value
  comparison and an invalid-input contract.

### Failures retained

1. An early fixed case passed `1 << 70` as a seed while the host API converted
   the value to a C++ integer. The call failed before the GPU kernel. The private
   ABI is now explicit: seed must be an exact Python integer in signed-int64
   range; negative values are normalized with Python-equivalent nonnegative
   modulo. The out-of-range seed is an invalid-input test.
2. Device 0 passed all 438 matrices, but switching the same process to device 7
   reused a JIT kernel handle loaded in device 0's CUDA context and raised
   `CUDA_ERROR_INVALID_HANDLE`. DeepEP's execution model is one process per GPU,
   so the test now launches each target GPU in an isolated child process instead
   of broadening the global JIT cache in this checkpoint.

### Verification

```text
setup.py build_ext --inplace
PASS

tests/elastic/test_rail_balance_plan_cuda.py
PASS 438 exact planner cases on source CUDA device 0
PASS invalid-input contract on source CUDA device 0
PASS 438 exact planner cases on source CUDA device 7
PASS invalid-input contract on source CUDA device 7
PASS 876 device-cases across 2 isolated processes

python -m py_compile tests/elastic/test_rail_balance_plan_cuda.py
PASS

git diff --check
PASS
```

The 438 cases per device include `G=1,D=0`, the `[4,0,0]` remainder
counterexample, balanced identity, destination rotation, a negative seed,
`G=8,D=256`, and 432 fixed-seed random matrices. This is correctness evidence,
not a planner latency result.

### Next action

Resolve independent final review findings, mark C030, then define C040's
per-egress physical proxy capacity before implementing direct-final source
shuffle.

## 2026-07-19 — D009: C030 independent final review closed

### Review findings fixed

- Changed every planner matrix and segment linear offset to `int64_t`; the
  planner still emits the frozen `int32` ABI after enforcing total copies no
  greater than `INT_MAX`.
- Made the process-wide JIT context limitation explicit. The private API now
  accepts repeated calls and non-default streams on its first CUDA device but
  rejects a different device in the same process with the stable message
  `rail-balance planner supports one CUDA device per process`.
- Strengthened the cross-device regression so an old
  `CUDA_ERROR_INVALID_HANDLE` failure cannot produce a false PASS.
- Added an automated non-default-stream case and corrected the runner's
  documented `PYTHONPATH` command.

### Final verification

```text
setup.py build_ext --inplace
PASS

EP_JIT_CACHE_DIR=/tmp/deepep-c030-jit-dd758caf-final \
  tests/elastic/test_rail_balance_plan_cuda.py
PASS 438 exact planner cases on source CUDA device 0
PASS invalid-input contract on source CUDA device 0
PASS non-default stream on source CUDA device 0
PASS 438 exact planner cases on source CUDA device 7
PASS invalid-input contract on source CUDA device 7
PASS non-default stream on source CUDA device 7
PASS explicit same-process cross-device rejection
PASS 876 device-cases across 2 isolated processes
```

The independent reviewer reported no remaining blocker or Medium issue and
marked C030 `PASS`.

### Retained low-priority hardening item

The host-side `int64` reduction used to enforce `total_copies <= INT_MAX`
could itself overflow only for an unrealistically huge count matrix (roughly
more than 4.3 billion maximum-valued elements and about 120 GiB of planner
working data). This is outside the planner domain but remains recorded rather
than silently discarded.

### Decision

Mark C030 `PASS`. Freeze the private counts-to-plan ABI and begin C040 with a
separate global moved-copy policy budget and per-egress physical arena
capacity.

## 2026-07-19 — D010: C040 capacity and physical-slot contract frozen

### Contract

- The existing `proxy_payload_capacity` remains a node-wide policy budget
  `B`: the plan may commit only when total moved copies `M <= B`.
- A separate symmetric-arena capacity `P` applies independently to every
  egress GPU: incoming moved records `I[e] <= P` for every `e`.
- Unified logical `(egress, destination, channel, slot, generation)` remains
  the future remote receive namespace for retained and moved copies.
- Only moved copies receive a compact egress-local `physical_slot` in
  `[0, I[e])`; that slot is the final registered source location for future
  Gin reads.
- Either capacity failure returns a complete identity bypass. No physical
  assignment or partially committed logical assignment escapes.

### Verification

The dependency-free CPU suite now has 23 tests. It distinguishes `M > B` while
all egresses fit, one egress overflowing while `M <= B`, a distributed total
larger than `P` that still fits, and exact boundaries. It also checks stable
packing, physical uniqueness, moved-only assignment, strict scalar types, and
atomic bypass. All 23 tests pass.

## 2026-07-19 — D011: C040 first 8-GPU direct-final data path passes

### Implemented

- Added a private one-shot method on `ElasticBuffer`, so its arena is a suffix
  of the constructor-registered symmetric GPU buffer rather than a dynamic
  PyTorch allocation.
- Defined a stable record ABI:

  ```text
  32-byte head canary
  TokenLayout(BF16 hidden, top-k indices/weights/source metadata)
  64-byte ProxyDescriptor
  32-byte tail canary
  ```

- Every owner warp stages one deterministic record, TMA-loads hidden data,
  TMA-stores directly to the peer egress's final physical slot through the LSA
  pointer, waits for the store, then release-publishes the generation.
- C040 uses a collective clear barrier and arrival barrier. The egress
  readback uses acquire loads for every ready word. Queue spinning, reorder,
  and ring reuse remain C050.
- Added all-rank preflight for identical byte/config ABI, exact manifest
  coverage, global physical-slot uniqueness, compact per-egress prefixes, and
  move-only `owner != egress`.

### Failures retained

1. The first host build failed because `buffer.hpp` used `CUDAGuard` without
   including `c10/cuda/CUDAGuard.h`. The include was added; no algorithm code
   changed.
2. Static review caught a JIT-only compile blocker before execution:
   `const void* x` was passed to the existing mutable `advance_ptr` helper.
   The source pointer is now consistently `void*`, matching existing DeepEP
   kernels. The review also caused deterministic shared-record zeroing,
   explicit buffer-device ownership, int64 source-index preflight, stable
   descriptor offsets, and a clear collective-call contract.
3. The initial runner confused DeepEP's 32-byte TMA layout alignment with an
   unrelated 128 alignment parameter. The byte-layout oracle now uses 32.
4. Fresh-cache run `v1` stopped safely in the all-rank GPU-plan verification:
   `init_dist` changes the default device to CUDA, while two expected tensors
   omitted `device="cpu"`. All eight ranks reported the same fixture failure;
   the source-shuffle collective was never entered.
5. Fresh-cache run `v2` compiled and ran the new JIT/LSA path, then the local
   verifier hit the same fixture class for expected ready tensors. Explicit
   CPU devices were added. This was not a ready-value mismatch.

### Passing evidence

```text
EP_JIT_CACHE_DIR=/tmp/deepep-c040-functional-v3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EP_DISABLE_GIN=1
tests/elastic/test_rail_balance_shuffle.py --num-processes 8 --timeout 90

PASS C040 8x1 source shuffle: 64 moved records,
incoming=(0, 0, 10, 11, 12, 12, 10, 9)
exit code 0
```

Every process independently matched C030 to the CPU oracle before entering the
data path. The test then checked byte-exact BF16 hidden values, int32 top-k
metadata, float32 weights, source-global index, linked-list sentinels, the
complete descriptor, both canaries, ready generations, unused ready zeros,
and exact global moved-copy coverage. This is functional evidence only; no
timing from this run is retained as a performance result.

### Next action

Run independent C040 final review and the original EP8 direct-path regression.
If both pass, mark C040 and begin C050's generation-aware publication protocol.

## 2026-07-19 — D012: C040 exit gate closed after adversarial review

### Review findings that prevented premature closure

The first passing 8-GPU run was only a conditional pass. Its 64 moved records
came from owners 0 and 1; owners 2 through 7 only participated in barriers and
readback. Its largest egress occupancy was 12 while physical capacity was 128.
An independent audit therefore correctly rejected the claim that the WP3
all-owner and exact-capacity exit gate had passed.

The same audit reproduced a fail-open CPU-oracle bug by mutating a logical
assignment to an owner egress with an invalid destination, channel, and slot.
`materialize_physical_proxy_slots()` accepted that damaged trusted input. The
CUDA runner's all-rank preflight would have rejected the resulting manifest,
so the earlier transport evidence was real, but the physical oracle contract
was not yet safe to freeze.

### Hardening implemented

- Replaced the final fixture with a deterministic all-owner, multi-destination
  route. Every owner moves 32 copies and every egress receives exactly 32.
- Raised hidden size to 7168, making each published record 14,528 bytes and
  exercising multi-iteration staging/copy loops rather than a tiny payload.
- Added a segment-derived moved-set/egress checker independent of the physical
  helper that creates the manifest.
- Compare every byte of each record, including all `TokenLayout` and descriptor
  padding, against a zero-initialized byte oracle.
- Added a valid `P=33` run that reads the entire arena and requires unused slot
  32 to remain `0xA5` with ready zero.
- Added a `P=31` caller-side atomic capacity rejection. All ranks preflight an
  empty manifest, the private launcher runs no producer, every returned record
  remains `0xA5`, and every ready value remains zero. This is caller-side
  enforcement; the C++ kernel does not calculate capacity policy.
- Check output dtype/device/contiguity, source tensor immutability, all-owner
  production, exact physical prefixes, full global coverage, and refuse
  optimized Python where assertions would be disabled.
- Added `validate_static_slot_plan()`, stored `num_channels` and
  `channel_capacity` in the plan, and made both static construction and
  physical packing validate fail-closed. The CPU suite now has 28 tests,
  including field, slot collision, missing-copy, enabled commit, disabled
  partial-commit, and physical-entry mutations.

### Failure retained

The first 28-test run failed one new mutation assertion. Changing a moved
assignment to `moved=False` caused the stricter validator to report
`retained copy must stay on its owner egress` before reaching the test's
expected later `segment copy was marked retained` check. The test accepted
both valid fail-closed error orders; no validator logic was weakened. The next
run passed 28/28.

### Final functional evidence

```text
setup.py build_ext --inplace
PASS

tests/elastic/test_rail_balance_plan.py
PASS 28 rail-balance planner tests

EP_JIT_CACHE_DIR=/tmp/deepep-c040-functional-v4
tests/elastic/test_rail_balance_shuffle.py --num-processes 8 --timeout 120
PASS C040 8x1 source shuffle: 256 moved records from all owners,
incoming=(32, 32, 32, 32, 32, 32, 32, 32), exact-capacity and GPU bypass verified
exit code 0

EP_JIT_CACHE_DIR=/tmp/deepep-c040-regression-final
tests/elastic/test_ep.py --num-processes 8 --num-tokens 32 --hidden 256
  --num-topk 1 --num-experts 8 --allow-hybrid-mode 0
  --test-first-only --skip-perf-test
exit code 0
```

The independent final reviewer reported no Blocker or High issue and approved
the fresh 8-GPU run. Old `v3` evidence remains useful because it covered empty
sender ranks 2 through 7 and empty receiver ranks 0 and 1; the final `v4` run
closes all-owner, exact-boundary, long-record, unused-record, and capacity
bypass gates. Neither run is retained as a performance measurement.

### Decision

Mark C040 `PASS`. Freeze the one-shot direct-final record ABI and begin C050.
Ring reuse and virtual multi-node forwarding remain explicitly unproven.

## 2026-07-19 — D013: C050 contiguous-ready CPU contract

### Completed

- Added a dependency-free one-shot ready-publication oracle. The public tail
  advances only across an exact current-generation prefix; stale, future, and
  nonzero-but-wrong values remain holes.
- Exhausted all 720 publication orders for six slots. Every intermediate tail
  is checked against an independent first-unpublished-slot calculation.
- Added a forced odd-before-even stale-generation trace, unsafe boolean/`>=`/
  max-ready shortcut discriminators, empty queues, partial/stuck publication,
  and signed-int32 ABI boundaries.
- Full direct suite passes 34/34. Independent review also exhaustively checked
  167,481 combinations for N=0..7, ready values in {0, 17, 29, 30}, and every
  possible `start_tail`.

### Failure retained

The first oracle accepted a caller-provided nonzero `start_tail` without
proving that `[0, start_tail)` belonged to the current generation. Concrete
counterexamples such as ready `(17, 17)`, generation 29, and `start_tail=2`
therefore crossed stale data. Review caught this before any GPU protocol used
the helper. The contract now rejects every unproven starting prefix and rejects
generation values outside the positive signed-int32 CUDA ABI.

### Decision and limitation

The CPU contiguous-tail contract is `PASS`, but C050 remains `IN_PROGRESS`.
Scalar state safety does not prove GPU release/acquire visibility, overlap
liveness, timeout behavior, or that a consumer reads payload while producers
are still running. Those require the 8-GPU forced-hole protocol fixture.

### Next action

Implement the isolated one-shot protocol layout and producer/scanner/consumer
kernels without changing C040's frozen record ABI.

## 2026-07-19 — D014: C050 GPU protocol and liveness exit gate closed

### Implemented

- Added `OneShotProtocolLayout` around the frozen C040 source-shuffle layout.
  Record and ready offsets are unchanged; generation-aware publish sequences
  and a 64-byte, 64-byte-aligned control block occupy the remaining symmetric
  arena suffix.
- Added separate prepared JIT producer and consumer cubins. All compilation is
  complete before the consumer begins polling.
- Producers TMA-store directly into final peer records, wait for the complete
  store, release-publish the ready generation, then release-publish the exact
  `(generation, slot)` key.
- A scanner acquire-validates contiguous records and release-publishes only the
  packed `(generation, exclusive_tail)`. A second warp acquire-loads that tail,
  copies the entire record while production is still live, and release-publishes
  `consumed_tail`.
- Producers execute in three dependency-ordered launches on one stream:
  unconditional records, records waiting only for a forced hole observation,
  then records waiting for downstream consumption. This removes CTA residency
  dependencies between work that creates and releases progress.
- Producer and scanner watchdogs renew only for a strictly larger
  current-generation consumed count, a new finite hole state, or a larger
  contiguous tail. Repeated observations cannot keep a broken protocol alive.
- Error claiming uses `atomicCAS_system`, followed by `error_slot` and a
  system-release final error code, because a producer can address peer control
  through LSA.
- Added all-rank ABI/manifest/call preflight, deterministic nonzero producer
  jitter, stale-generation seeding, forced odd/even holes, payload canaries,
  exact full-record comparison, per-slot consume counts, trace validation,
  capacity bypass, and bounded dropped-slot fault injection.

### Review findings and failed attempts retained

1. The first protocol layout left an unnecessary 4 MiB arena gap. Host layout
   inspection caught it and the layout was tightened to the existing 2 MiB
   aligned arena without moving C040 record/ready offsets.
2. The first JIT header omitted `handle.cuh`; static compile review caught the
   missing `NCCLGin` definition before GPU execution.
3. The first downstream warp let lanes poll independently. Divergent break/copy
   decisions could deadlock at `__syncwarp`; lane 0 now chooses a uniform action
   and broadcasts it before every full-mask barrier.
4. The first scanner watchdog renewed on every observation of the same hole,
   allowing an infinite wait. It now renews once per distinct
   `(tail, later_ready_slot)` state.
5. A two-phase producer split still mixed hole-only work with
   consumption-gated work. Adversarial review found a legal CTA-starvation
   schedule; the final implementation uses three phases.
6. The next watchdog used one absolute start time for a multi-step consumption
   gate. A legal slow consumer could make continuous progress and still be
   rejected. Both producer and scanner now track strict consumed-tail progress;
   a regression deliberately waits for eight slow steps whose aggregate delay
   exceeds the watchdog interval.
7. The first error claim used device-scope `atomicCAS` on a possible peer LSA
   pointer. Final review required and received `atomicCAS_system`.
8. The first test invocation omitted `PYTHONPATH` and failed before GPU startup
   with `ModuleNotFoundError`. The documented command now pins this checkout.
9. Fresh-cache v1 successfully reached the protocol but a verifier constructed
   an expected tensor on the CUDA default device instead of CPU. Explicit CPU
   placement fixed the fixture; no protocol output was relaxed.
10. A generic older-architecture layout compile hit existing repository PTX
    support limits. The production target is H200/SM90 and the accepted JIT
    cubins target SM90(a). Functional runs in the current environment are
    eight-GPU LSA correctness evidence, not independent H200 hardware or
    performance proof.

### Final verification

```text
tests/elastic/test_rail_balance_plan.py
PASS 34 rail-balance planner tests

independent contiguous-tail exhaustion
PASS 167,481 ready/start-tail states

EP_JIT_CACHE_DIR=/tmp/deepep-c050-functional-v6
tests/elastic/test_rail_balance_protocol.py --num-processes 8
  --gpu-timeout-cycles 4000000000 --repeat-generations 16
PASS forced hole, stale generation, live payload overlap, slow consumer,
  progress-aware late gating, and bounded missing-slot timeout

EP_JIT_CACHE_DIR=/tmp/deepep-c040-regression-after-c050
tests/elastic/test_rail_balance_shuffle.py --num-processes 8
PASS 256 moved records, all owners, exact capacity and bypass

tests/elastic/test_ep.py --num-processes 8 --num-tokens 128 --hidden 1024
  --num-topk 2 --num-experts 64 --allow-hybrid-mode 0
  --test-first-only --skip-perf-test
exit code 0

git diff --check
PASS
```

The kernel/protocol reviewer and host/API/test reviewer independently report
zero Blocker, High, or Medium findings. These are correctness results; elapsed
times from the functional runs are not performance evidence.

### Decision and boundary

Mark C050 `PASS` and begin C060. The frozen evidence is for a one-shot queue on
SM90/LSA. It does not claim ring reuse, generation wraparound, Gin/QP/NIC
behavior, or multi-node speedup.

## 2026-07-19 — D015: C060 finite 4x2 round-trip implementation

### Frozen contract

- Keep one physical eight-rank NCCL communicator and symmetric window. Ranks
  0–3 are source owner/egress roles; ranks 4–7 are destination ingress/expert
  roles. Virtual roles never create a second four-rank communicator.
- Preserve the C040/C050 record ABI byte-for-byte. A separate aligned 32-byte
  `VNodeRoute` sidecar carries per-contribution ingress, lane, expert, and slot
  state.
- Pack retained and moved destination copies together. The CPU full-egress
  materializer produces compact per-egress prefixes and atomically rejects a
  capacity smaller than any full rail load. The CPU suite is now 36/36.
- Keep every top-k contribution independent through return. Owner ranks reduce
  BF16 partials in fixed lane order through FP32 and cast once; ordinary torch
  outputs are never peer LSA write targets.
- Use finite stage kernels and eight identical all-rank barriers: clear,
  complete source pack, scaleout, forward, expert, return, unshuffle, owner
  reduction. This checkpoint proves function, not persistent-pipeline speed.

### Implemented

- Added `rail_balance_vnode_layout.cuh`: a base/contribution rail arena, an
  expert arena, and a role-overlaid source-owner partial arena. H7168/K4/P20
  requires exactly 8 MiB per rank.
- Added six standalone SM90 kernels for fake scaleout, destination forwarding,
  synthetic weighted expert work, return, source unshuffle, and owner reduce.
- Added prepared JIT runtimes for all six kernels and refactored the source
  packer so its cubin can also be built before the first collective.
- Added the private `_rail_balance_vnode_roundtrip` host bridge and an absolute
  offset layout getter. The bridge returns twelve owning debug tensors rather
  than views into the one-shot symmetric arena; the twelfth is a 4 KiB `0xA5`
  tail guard immediately beyond the computed vnode arena.
- Full source packing intentionally passes `P*(K+1)` as the record-layout
  capacity while restricting manifests to base slots `<P`; using `P` would
  move the ready array and corrupt the vnode layout.
- Self-egress retained records use the local symmetric pointer. Moved records
  continue to resolve a peer LSA pointer. The older C040 moved-only public
  prototype retains its host-side self-egress rejection.
- Route validation now derives lane/ingress/egress/expert slots from fixed work
  indices before indexing metadata. Unshuffle also verifies the canonical
  `src_token_global_idx` before claiming an owner partial with system CAS.

### Verification before final review

```text
core extension incremental build and link
PASS

source-package in-place extension import
PASS private roundtrip binding and vnode layout getter

manual NVCC 12.8 / sm_90, H7168 K4
PASS all six kernels, registers 38/42/44/42/45/46, zero spill,
  dynamic record shared memory 14,528 bytes

tests/elastic/test_rail_balance_plan.py
PASS 36/36

git diff --check
PASS

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c060-h256-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode.py --num-processes 8 \
  --hidden 256 --timeout 600 --master-port 29661
PASS 8-GPU 4x2 loop, 80 base records, 320 independent contributions,
  exact owner combine

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c060-h7168-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode.py --num-processes 8 \
  --hidden 7168 --timeout 600 --master-port 29662
PASS fresh-cache full records, exact payload/metadata/routes/readiness,
  independent 8 MiB layout formula, and intact 4 KiB tail guard

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c040-after-c060-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_shuffle.py --num-processes 8 \
  --timeout 600 --master-port 29663
PASS 256 moved records, exact capacity, and GPU bypass

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c050-after-c060-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_protocol.py --num-processes 8 \
  --timeout 600 --repeat-generations 16 \
  --gpu-timeout-cycles 4000000000 --master-port 29664
PASS all C050 liveness and bounded-failure cases

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B tests/elastic/test_ep.py \
  --num-processes 8 --num-tokens 128 --hidden 1024 --num-topk 2 \
  --num-experts 64 --allow-hybrid-mode 0 --test-first-only \
  --skip-perf-test
PASS exit code 0
```

### Failed attempts retained

1. The first vnode launcher called inherited `generate(spec, kind)`, but the
   JIT base accepts one spec. Host compilation failed deterministically. Kernel
   kind now lives inside dedicated peer/expert specs; the next build linked.
2. The initial reduce launcher used `NoRefPtr` for a normal `const void*`
   arena. Static review caught the kernel-argument ABI error before execution.
3. Expert route validation originally compared several route fields to
   themselves, allowing a corrupted lane to become an OOB index. Fixed work
   indices now derive every expected field before any metadata access.
4. A destination-equals-one assertion briefly matched the earlier C050 method
   during patching and also used the physical node label instead of C060's
   zero-based single-destination namespace. Immediate scoped review caught both
   mistakes before GPU execution; only C060 now requires destination zero.
5. The dedicated environment has no `pytest`, so a pytest invocation exited
   before collection. No dependency was installed; the direct test runner
   remains the project convention and passed all 36 CPU cases.
6. Invoking `test_rail_balance_vnode.py` without the documented repository
   `PYTHONPATH` failed at import. No GPU work started and no vnode evidence was
   changed.
7. A review note initially stated that the live TMA alignment was 16 bytes.
   Direct inspection of `ptx::kNumTMAAlignBytes` showed the branch uses 32;
   the independent Python layout formula is pinned to the actual 32-byte ABI.
8. One combined `git diff --stat`/status inspection took roughly eleven minutes
   despite being read-only, then returned normally. Subsequent focused
   `git diff --check` and per-new-file checks completed normally; no data or
   test process was active during the delay.

### Decision and next action

C060 runtime and regression evidence now passes. Keep the checkpoint
`IN_PROGRESS` only until the final diff/log reviewers close with no unresolved
Blocker/High/Medium; then mark the restricted main path `PASS` and start C061.

## 2026-07-19 — D016: C060 final audit closed

### Final fixes and hardening

- Fixed formal C++ undefined behavior in size-only `VNodeRoundTripLayout`
  construction. A null arena base is no longer passed through pointer
  arithmetic when constructing the overlaid expert and owner layouts.
- Changed the deterministic routing fixture so lane ownership rotates by
  `(owner + token + lane) % 4`. Every fixed lane now reaches every destination
  expert rank; a kernel that incorrectly equates top-k lane with physical GPU
  can no longer pass by common-mode expectation.
- Rebuilt the core extension after the layout-header change, then used a new
  JIT cache so every vnode cubin was regenerated from the corrected header.

### Final verification

```text
/home/chen/.cache/deepep-sjlgpt/bin/python setup.py build_ext --inplace
PASS compile, device link, host link, and in-place copy; exit code 0

/home/chen/.cache/deepep-sjlgpt/bin/python -m py_compile \
  tests/elastic/test_rail_balance_vnode.py
PASS

PYTHONPATH=$PWD EP_DISABLE_GIN=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c060-rotated-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode.py --num-processes 8 \
  --hidden 256 --timeout 600 --master-port 29665
PASS 80 full-egress records, 320 independently routed expert contributions,
  rotated lane-to-expert coverage, and exact combine; exit code 0

git diff --check
PASS
```

The independent kernel/API review and independent evidence-log review both
closed with zero unresolved Blocker, High, or Medium findings.

### Failure and restricted boundary retained

- The first attempt to apply the audit patch hit a transient nested `bwrap`
  namespace `ENOSPC` before any file changed. Repeating the same reviewed patch
  through the approved workspace path succeeded.
- The C060 bridge launches exactly `P` base slots on every virtual rail. This
  is correct for the frozen equal-prefix fixture, but it is not a generic
  non-divisible-quota contract. Explicit per-rail prefix counts are a C061
  requirement rather than an undocumented C060 capability.
- These are functional LSA results. No local performance, Gin/QP/NIC, or
  multi-node speedup claim is made.

### Decision

Mark the restricted C060 checkpoint `PASS`. Start C061 with a 2x4 virtual
layout, multiple destinations, variable prefixes, and order-independent exact
round-trip coverage.

## 2026-07-19 — D017: C061 2x4 contract and CPU oracle frozen

### Discriminating topology and traffic

- Ranks 0–1 are the only source owners/egresses. Destination pairs are 2–3,
  4–5, and 6–7; `ingress(d,e) = 2 + 2*d + e`.
- Ten tokens, four lanes, three destinations, generation 62, and two logical
  channels produce counts `((9,2,6),(2,9,5))`. The aggregate rail totals are
  already 17/16, so an aggregate-only policy would incorrectly bypass.
- The canonical per-destination quota is `((6,5,6),(5,6,5))`. It moves three
  destination-0 copies from owner 0 to egress 1 and three destination-1 copies
  in the opposite direction; destination 2 is identity.
- A destination-segmented physical pack reserves six slots per destination.
  Flattened base capacity is 18. Egress 0 must leave slot 11 empty; egress 1
  must leave slots 5 and 17 empty.
- Thirty-three deduplicated destination copies yield only 72 valid expert
  contributions because each copy forwards the lanes whose decoded expert
  belongs to that destination. Fingerprints include destination and remain
  unique when one token has multiple copies.

### Implemented and verified

- Added an atomic destination-segmented CPU physical pack. One overflowing
  destination rejects the whole pack without returning partial assignments.
- Added two permanent planner tests for successful segmented holes and strict
  capacity failure. The complete direct suite passes 38/38.
- Added the route-derived C061 fixture with two-channel retained/proxy
  collision checks, exact moved keys, expert mapping, and expected BF16
  results. Its CPU-only entry point passes 33 copies, six moves, and 72 unique
  contributions; per-expert-rank counts are `(14,11,11,17,11,8)`.

```text
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_plan.py
PASS 38 rail-balance planner tests

PYTHONPATH=$PWD/tests:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_multidst.py --cpu-only
PASS C061 CPU oracle: 33 destination copies, 6 exact moves,
  72 expert contributions
```

### Failed attempt retained

The first combined reference/test patch used a stale test anchor (`False`
instead of the live strict-boolean `True` case). `apply_patch` updated the
reference file before rejecting the second file. A focused inspection proved
the reference half complete and the test half untouched; the corrected test
patch then applied and both suites passed. No half-written runtime path was
executed.

### Next action

Finish the destination/count-aware CUDA stages, host preflight, compact expert
layout, and the eight-process byte-exact 2x4 round trip. Do not claim Gin/RDMA
or performance from this emulator.

## 2026-07-19 — D018: C061 multi-destination CUDA round trip closed

### Implementation

- Generalized the private vnode layout from C060's single destination to
  explicit source-rail count `G`, destination count `D`, per-destination
  capacity `C`, and token count `T`. C060 retains its compatibility
  constructor.
- The flattened base namespace is `B=D*C`; rail contribution slots use
  `B + base_slot*K + lane`, while destination-local expert slots use
  `(egress*C + local_slot)*K + lane`.
- The host bridge accepts a typed contiguous CUDA `quota[G,D]`, validates the
  2x4 topology and every manifest field before the first CUDA barrier, and
  sizes the status area with `max(T, B*K, G*C*K)`.
- Scaleout validates every quota hole as empty, forwarding emits only lanes
  whose decoded expert belongs to the record destination, and return/unshuffle
  preserves the global destination-bearing ingress slot.
- Added an eight-process verifier with an independent 13-field arena formula,
  exact 32-byte routes, full record bytes and canaries, sparse holes, BF16
  partials and fixed-lane combine, 4 KiB guard, source immutability, and 282
  globally unique coverage records.

### Functional evidence

```text
PYTHONPATH=$PWD/tests:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_multidst.py --cpu-only
PASS C061 CPU oracle: 33 destination copies, 6 exact moves,
  72 expert contributions

PYTHONPATH=$PWD/tests:$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c061-h256-v2 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_multidst.py --num-processes 8 \
  --hidden 256 --timeout 600 --master-port 29669
PASS 33 copies, 72 contributions, 282 exact coverage records; exit code 0

PYTHONPATH=$PWD/tests:$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c061-h7168-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_multidst.py --num-processes 8 \
  --hidden 7168 --timeout 600 --master-port 29670
PASS 33 copies, 72 contributions, 282 exact coverage records; exit code 0

/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_plan.py
PASS 38/38

PYTHONPATH=$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c060-after-c061-final-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode.py --num-processes 8 \
  --hidden 256 --timeout 600 --master-port 29671
PASS C060 80 records, 320 contributions, exact combine; exit code 0

PYTHONPATH=$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B tests/elastic/test_ep.py \
  --num-processes 8 --num-tokens 128 --hidden 1024 --num-topk 2 \
  --num-experts 64 --allow-hybrid-mode 0 --test-first-only \
  --skip-perf-test
PASS original EP8 public path; exit code 0

/home/chen/.cache/deepep-sjlgpt/bin/python setup.py build_ext --inplace
PASS current extension link and in-place copy; exit code 0
```

Independent final review closed with 0 Blocker, 0 High, and 0 Medium. It found
two Low PoC boundaries: the private C++ entry trusts the test's global
manifest uniqueness/prefix proof, and logical channels are preserved as
planner/descriptor metadata while the vnode data plane consumes prepacked
physical slots. Both contracts must be internalized when C080 connects an
untrusted production path or real channel/QP lookup.

### Failed attempts retained

1. Four initial read-only checks failed before execution because nested bwrap
   namespaces hit `ENOSPC`. Re-running through the approved host path worked;
   no repository state changed in the failed calls.
2. The approved shell did not expose a `python` alias. System `python3` lacked
   PyTorch, and the raw `sjlgpt` Conda interpreter loaded an incompatible NCCL
   at runtime (`ncclCommQueryProperties`). The project runtime
   `/home/chen/.cache/deepep-sjlgpt/bin/python` is the verified ABI/library
   environment and was pinned for every acceptance run.
3. The first H256 GPU run reached all CUDA stages, then the verifier tried to
   call NumPy on expected tensors implicitly placed on CUDA by
   `init_dist()`'s default-device setting. All eight ranks failed at the same
   byte-check location. The CPU oracle constructors now set `device="cpu"`
   explicitly; fresh-cache v2 passed.
4. A temporary hardening attempt required `channel==0` and
   `logical_slot==destination_slot`. That was incompatible with the frozen
   two-channel compact pack. It was caught during contract review and reverted
   before a GPU acceptance run; the host still validates both fields without
   imposing the false equality.
5. A multi-file log patch used an imprecise O010 tail anchor. The development
   log half applied before the optimization/context halves were rejected.
   Inspection proved the exact partial state; a second focused patch completed
   the other files without duplicating D018.
6. The independent reviewer also first tried a missing `python` alias and the
   torch-less system `python3`; its verified reruns used the pinned project
   runtime. These duplicate environment failures are retained because they
   explain why bare interpreter commands are not accepted evidence here.

### Boundary

This closes the single-source, three-destination LSA functional emulator. It
does not establish public Hybrid integration, cached replay, reusable rings,
Gin/QP/NIC behavior, or speedup. No performance samples were collected for
this checkpoint.

### Decision

Mark C061 `PASS` and checkpoint the reviewed files. Begin the minimal C070
cross-call replay proof; do not expand it into public Hybrid/EPHandle work.

## 2026-07-19 — D019: C070 snapshot oracle and private replay entry

### CPU replay oracle

- Added a self-contained frozen C061 protocol snapshot with 33 exact
  descriptor byte strings, 72 contribution routes, ready generations, quota,
  top-k indices, and record payloads.
- `replay_c061_snapshot()` accepts no manifest, planner segment, or source
  routing table. It reconstructs owner/token/lane/destination/egress/expert
  identity from persisted descriptor and route bytes, verifies every quota
  prefix/hole, and performs fixed-lane FP32 accumulation followed by BF16 cast.

```text
PYTHONPATH=$PWD/tests:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_replay.py --cpu-only --hidden 128
PASS 33 descriptors, 72 routes, 6 moved copies, fixed-lane BF16 combine

PYTHONPATH=$PWD/tests:$PWD \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_replay.py --cpu-only --hidden 7168
PASS same replay contract at H7168
```

### Private host entry

- Added `_rail_balance_vnode_replay` for the frozen `G=2,D=3,C=6` PoC. It
  accepts base rail records/readiness plus expert records/routes/readiness,
  quota and source top-k. It cannot accept manifest or fingerprints.
- The entry clears the symmetric arena, restores only base and expert state in
  `record -> route -> ready` order, leaves prior contribution slots zero, then
  launches only `expert -> return -> unshuffle -> reduce`.
- Source ranks do not restore the expert region because it overlays owner
  partial storage. Snapshot inputs are owning CUDA tensors and are rejected if
  they alias the symmetric arena. The existing 12-tensor debug return contract
  is reused with a four-row status array.

```text
/home/chen/.cache/deepep-sjlgpt/bin/python setup.py build_ext --inplace
PASS current object/link/copy; extension contains the replay binding

/home/chen/.cache/deepep-sjlgpt/bin/python -c \
  "import deep_ep._C as C; print(hasattr(\
      C.ElasticBuffer, '_rail_balance_vnode_replay'))"
True
```

### Failed attempts retained

- The first host build used the raw `sjlgpt` Conda interpreter. Its older NCCL
  headers lack current GIN types and failed in pre-existing NCCL/Engram code
  before reaching the replay entry. Rebuilding with the pinned project runtime
  succeeded.
- A main-thread visibility probe checked `deep_ep._C.Buffer`, the wrong
  extension class, and returned `False`. The registered class is
  `deep_ep._C.ElasticBuffer`; the corrected probe returned `True`.

### Next gate

Capture a passing C061 round trip, poison/delete its manifest and fingerprints,
replay the owning snapshot twice, and compare both GPU results with the
snapshot-only CPU oracle. Do not mark C070 PASS until the eight-rank collective
shutdown and guard checks succeed.

### Eight-rank GPU evidence

The GPU test first executes and byte-checks the C061 capture. It clones owning
base/expert snapshot tensors, fills planner-only manifest/fingerprint tensors
with poison, deletes them, and calls the replay entry twice. Both calls must
match the snapshot-only oracle, preserve every input byte, clear old
contributions, report a zero `[4,72]` status matrix, preserve the 4 KiB guard,
and collectively destroy the buffer.

```text
PYTHONPATH=$PWD/tests:$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c070-h256-v2 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_replay_gpu.py \
  --num-processes 8 --hidden 256 --timeout 600 --master-port 29673
PASS manifest-free exact replay and byte-identical second replay; exit code 0

PYTHONPATH=$PWD/tests:$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c070-h7168-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_replay_gpu.py \
  --num-processes 8 --hidden 7168 --timeout 600 --master-port 29674
PASS manifest-free exact replay and byte-identical second replay; exit code 0

PYTHONPATH=$PWD/tests:$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c070-h256-v2 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode_replay_gpu.py \
  --num-processes 8 --hidden 256 --timeout 600 --master-port 29675 \
  --check-route-corruption
PASS one live ingress-slot corruption reports RouteMismatch=4, leaves the
  guard intact, and all eight ranks terminate collectively; exit code 0

/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_plan.py
PASS 38/38

PYTHONPATH=$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
EP_JIT_CACHE_DIR=/tmp/deepep-c060-after-c070-v1 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_vnode.py --num-processes 8 \
  --hidden 256 --timeout 600 --master-port 29676
PASS C060 exact round trip; exit code 0

PYTHONPATH=$PWD EP_DISABLE_GIN=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B tests/elastic/test_ep.py \
  --num-processes 8 --num-tokens 128 --hidden 1024 --num-topk 2 \
  --num-experts 64 --allow-hybrid-mode 0 --test-first-only \
  --skip-perf-test
PASS original EP8 public path; exit code 0
```

### Additional failure retained

The first H256 GPU replay executed all four CUDA stages, then the CPU oracle
rejected the gathered source top-k tensor because its frozen fixture used
int32 while DeepEP's public `topk_idx_t` is int64. Ready/quota remain protocol
int32; the GPU gather now explicitly converts top-k to TokenLayout's persisted
int32 representation, and the oracle accepts either integer source form. A
fresh-cache v2 rerun passed. The longer v2 startup was observed with active
parallel NVCC processes and zero GPU utilization, proving cold compilation
rather than a replay barrier hang.

### Independent review and decision

The first independent review reported 0 Blocker, 0 High, 1 Medium, and 1 Low.
The Low noted that the snapshot oracle consumes GPU-produced contribution
payload and could therefore self-confirm a consistently wrong expert numeric
transform. The test now additionally compares both owner partials and combined
output against C061's independent CPU expert formula. A cached H256 eight-GPU
rerun passed, and review confirmed that Low closed.

Final review state is 0 Blocker, 0 High, 1 Medium, and 0 Low. The remaining
Medium concerns invalid/mismatched inputs: rank-local host assertions occur
before the first collective, so one rank could throw while its peers enter a
barrier. The frozen test supplies a trusted snapshot and proves its all-rank
configuration before replay, so this does not invalidate the C070 PoC. It is a
mandatory C080 reliability gate before any external or production path uses
the method.

The exact C000-C061 staged snapshot was preserved before adding C070 as Git
tree `35e3d8970216bee0547b40e810873a623d677c71`; this avoids losing the
checkpoint even though repository commit identity is still unset.

Decision: mark C070 `PASS` for the minimum trusted-snapshot replay proof. Move
to C080 optional Hybrid integration with default off; do not claim that full
public cached/expanded/backward semantics are complete.

## 2026-07-20 — D020: C080 read-only Hybrid integration audit and plan freeze

### Repository state restored

- Resumed `feat/rail-balance-prototype` at
  `cdac8a3fe218a2c62a1d21aed04c467fff7b65b1`, tracking the fork branch with a
  clean worktree.
- Re-read the C070 handoff, remaining Medium, checkpoint table, and architecture
  decisions O011-O012 before inspecting C080 code anchors.
- No Hybrid source file was edited during the audit.

### Dispatch audit

- The existing Hybrid source warp assigns destination slots dynamically within
  each channel and sends with the current GPU's Rail Gin context. Changing the
  destination pointer cannot select another local rail.
- The destination forwarder consumes dense channel prefixes. Retained and
  moved traffic therefore need a single static remote-slot namespace; separate
  prefixes would collide or expose holes.
- C030 stops at quota/segments. C040-C070 receive a CPU/test-generated per-copy
  manifest. The missing production stage is a GPU materializer from real
  `topk_idx` through destination dedup and channel counts to static egress,
  channel, remote-slot, and proxy-slot assignments.
- A channel-major owner ordinal matches the existing one-warp-per-channel token
  walk and avoids a cross-channel data-path atomic.

### Combine audit

- Existing Hybrid combine derives source rail and receive slot from
  `src_token_global_idx`; it assumes the dispatch egress rail is the original
  owner's local rank.
- A moved copy returns along the proxy egress rail. Different owners can share
  the same local token index, so `src_token_idx` is not a valid proxy namespace.
- Force metadata needs a unique per-destination-copy proxy slot plus egress
  identity. Moved results first land on the source egress, then a standalone
  LSA unshuffle writes the original owner's existing reduction row before the
  existing epilogue.
- Proxy fields cannot be appended to `recv_src_metadata`, whose later columns
  have expanded-layout meaning. A force-only TokenLayout and forward-metadata
  shape are required.

### JIT/default-off audit

- `LaunchRuntime` caches one transitive include hash per derived runtime, while
  the compiler key includes generated source and those include hashes.
- Modifying the old Hybrid headers, `common/layout.cuh`, or sharing the old
  runtime would change default-off cache identity. C080 will add independent
  force headers/runtimes and will leave the old transitive JIT inputs unchanged.
- Force-only symmetric storage will be appended after the aligned legacy
  dispatch/combine maximum. Expanding `WorkspaceLayout` would shift old offsets
  and is rejected.
- A follow-up baseline audit expanded the immutable old-Hybrid closure to
  include `combine_utils.cuh` and common comm/handle/exception/ptx/compiled/
  layout/math headers. Hashing only the two root kernels and `layout.cuh` would
  miss real cache-key changes.
- `LaunchRuntime` keeps one static include hash per derived runtime. Calling its
  public `generate()` with synthetic Hybrid args and then generating a direct
  kernel in the same process would reuse the wrong first hash. C080-A probes
  must use `generate_impl()` plus explicit include parsing or isolated
  subprocesses.

### Topology/evidence boundary

- This node's real logical topology has one scaleout rank, so the launcher does
  not select Hybrid. Faking scaleout ranks would also fake Rail teams, barrier
  teams, and rank decoding.
- Local evidence will be reported separately as default-off regression, vnode
  shared-core functionality, and synthetic Hybrid code generation. Real Hybrid
  runtime remains pending until a multi-node Rail environment is available.

### Frozen scope and decision

- First mode: constructor-fixed `off|force`, default `off`; `auto` deferred.
- First force matrix: BF16, non-expanded, non-cached, non-deterministic,
  synchronous, exact copy-level, one-shot slots, one reduction configuration.
- `force` capacity exhaustion is a collective error before publication; it does
  not silently fallback.
- Two-phase preflight is mandatory: all-rank prepare, GPU plan without
  publication, then all-rank commit before any payload/ready/tail exposure.
- The detailed sequence and stop conditions are frozen in `C080_PLAN.md`.

Decision: begin C080-A only after this plan passes diff/consistency review.

### Resource release authorized by the user

- Four GPU processes were positively identified as children of Megatron
  `torchrun` PID 755241, running
  `records/track_3_optimization/train_gpt_layer_cosine.py` from the
  `megatron-lm-gpu` environment on GPUs 0-3.
- After explicit user authorization, sent SIGTERM to the torchrun parent. Its
  four workers, inductor workers, and W&B core exited; the one orphaned W&B XPU
  helper from the same process tree was also terminated.
- No file or checkpoint was deleted. Final `nvidia-smi` showed all eight GPUs at
  0 MiB allocated and 0% utilization, enabling correctness and controlled local
  performance work.

### Goal activation and expanded local exit condition

- Activated a persistent Goal for C080 with no token budget. The user clarified
  that local completion must go beyond the first functional path: exhaust local
  correctness/fault/sanitizer coverage, optimize all locally observable
  kernels while GPUs are idle, and prepare the fastest possible multi-node
  bring-up package.
- C080 local closure therefore feeds C090, C100, and C105 before the Goal can be
  considered complete. Real Gin/QP/NIC/fabric behavior remains an explicit
  environment gate rather than a hidden blocker.

### Independent plan review: blocker and corrections

The first C080 plan review reported one Blocker, five High, and several Medium
findings. No constructor or Hybrid source implementation had started.

- Blocker: adding proxy fields to the transported TokenLayout would also require
  a force-only buffer calculator and dispatch copy epilogue. Two `int32` fields
  can cross alignment boundaries (including top-k 10 and 21), so padding cannot
  be assumed. Resolution: preserve legacy TokenLayout bytes and carry the key
  in a registered same-slot route sidecar consumed by the new forwarder.
- Added an executable `IDLE -> PREPARING -> DISPATCH_RUNNING -> DISPATCH_LIVE ->
  COMBINE_RUNNING -> IDLE` state machine, with instance cookie, arena epoch,
  one-shot handle consumption, and `POISONED` recovery after post-publication
  failure.
- Added a capability gate so public force remains collectively unavailable
  until both dispatch and combine exist; partial checkpoints cannot expose a
  half pipeline.
- Replaced the nonexistent `async_finish` shorthand with the real Python
  stream/event/copy flags and froze the entire first support matrix.
- Narrowed JIT liveness claims: returned/thrown build errors enter consensus;
  compile-only tests use a subprocess watchdog, while an OS-level compiler hang
  is recorded as an environment failure.
- Added focused sanitizer prerequisites, instrumentation ABI, local symmetric
  count placement, uint64 generation/wrap behavior, objective ptxas gates, and
  an explicit C080-H correctness versus C110 performance split.

Decision: request a second independent plan review. C080-A may begin only with
the read-only golden snapshot until that review closes.

### C080-A read-only golden first attempt

- Added a CPU-only golden for immutable launcher/include SHA256 values, DeepEP's
  recursive include hash algorithm, representative direct/Hybrid buffer bytes,
  token strides, and workspace bytes. It does not construct a fake topology or
  call `LaunchRuntime::generate()`.
- The first run failed immediately because the manually transcribed
  `hybrid_combine.cuh` SHA256 had one extra character sequence (`...b5a6c...`
  instead of `...b5a6b...`). The verifier correctly reported the observed and
  expected path/hash before evaluating later fields.
- Corrected only the golden transcription. This is retained as evidence that
  the identity check fails closed rather than accepting a near-match.
- Second plan review noted that default EP8 direct roots and both copy/reduce
  epilogues were missing from the immutable closure. Added `dispatch.cuh`,
  `combine.cuh`, `dispatch_copy_epilogue.cuh`, and
  `combine_reduce_epilogue.cuh` SHA256 plus recursive include hashes before any
  source integration change.

## 2026-07-20 — D021: reject the sidecar draft and minimize force-v1

The user made hot-path simplicity an explicit requirement. Before starting a
Hybrid source edit, two independent read-only audits rechecked every proposed
route field against the current dispatch/combine lifetime.

### Verified free transit lifetime

- Source Hybrid dispatch writes top-k, weights, and src_token_global_idx but
  gives no business meaning to linked_list_idx before network transit.
- Destination forwarding overwrites every linked-list entry before the legacy
  copy epilogue reads it.
- Restricted non-cached force-v1 can therefore write compact proxy slot p into
  linked_list_idx[0], snapshot it at destination before overwrite, and add only
  one int to force-only forward metadata.
- Moved state and egress identity are derived: original owner local rank comes
  from src_token_global_idx and current ingress local rank is the return rail.

### Removed from the first implementation

The prior 32-byte same-slot route-sidecar design was semantically viable but
rejected before code was written. The staged force-v1 path no longer contains:

    network route sidecar
    extra route Gin put
    per-copy descriptor
    per-slot ready/generation
    return header
    full per-copy assignment table
    ring or producer/consumer queue

Proxy slots are grouped by (egress, channel, destination). Group prefix/count
derives destination, channel, remote slot, and p. The preserved dispatch
payload derives original owner/token during unshuffle. Force-v1 is narrowed to
allow_multiple_reduction=true and num_scaleout_ranks<=num_topk, so the final
legacy reduce row is the destination and no alternate lane rule or route field
is needed.

### Required synchronization retained

- Source peer TMA writes complete before an LSA-team barrier; only then may the
  force Hybrid kernel read proxy payloads.
- Hybrid combine completes Gin flush/rail signals before return unshuffle.
- All unshuffle peer TMA writes complete before another LSA-team barrier; only
  then may the legacy reduce epilogue run. Per-GPU stream order alone does not
  order writes from other local GPUs.

The plan state returned from frozen to rereview while this simplification was
checked. C080_PLAN.md has been rewritten around the minimal staged ABI. No
Hybrid source, constructor, binding, or buffer layout has been edited yet.

### Recurrent Megatron resource release

The user gave standing authorization to terminate any process positively
identified as Megatron/MGT training. A new torchrun parent PID 843160 and four
train_gpt_layer_cosine.py workers (PIDs 843238-843241) occupied about 36 GiB on
each of GPUs 0-3. SIGTERM was sent to the parent; its workers and helpers exited.
No unrelated process or file was touched, and nvidia-smi then reported no
compute process.

Next: run consistency/diff checks, obtain an independent review of the rewritten
minimal plan, then commit and push the C080-A plan/golden checkpoint before any
source integration.

## 2026-07-20 — D022: minimal-plan executable guards and off smoke

- Extended the legacy Hybrid identity runner with a semantic lifetime guard:
  source scaleout must not use linked_list_idx before transmission, destination
  linked-list overwrite remains after the token load and before forward
  metadata, and the legacy rank-layout selector remains ranks <= top-k.
- The identity runner now passes 4/4 checks.
- The existing CPU planner/invariant runner passes all 38 tests after the
  plan rewrite.
- The first EP8 smoke invocation omitted PYTHONPATH and failed at
  ModuleNotFoundError before any GPU process started. This repeats a known
  command-line pitfall and is retained.
- Corrected command used PYTHONPATH=$PWD, EP_DISABLE_GIN=1, all eight H200s,
  128 tokens, hidden 1024, top-k 2, 64 experts, direct mode, first case only,
  and performance disabled. The FP8/alignment-128 dispatch/combine case exited
  zero on all ranks.

No production or Hybrid source was modified by this checkpoint.

## 2026-07-20 — D023: minimal plan final review and freeze

The rewritten staged plan received a final independent review. The first pass
found no data-path architecture blocker but identified four control-contract
gaps. All were closed before freeze:

- Every round-trip kernel is generated/built and every force-internal tensor is
  allocated before WORLD GATE #1 and the first device collective.
- The route-validation kernel checks all expert ids without derived indexing;
  WORLD GATE #1 precedes count/node barriers and WORLD GATE #2 precedes shuffle.
- Combine outputs are allocated and world-consensed before the uninterrupted
  force combine -> unshuffle -> scaleup barrier -> epilogue sequence.
- Source shuffle does not add a second node barrier; force Hybrid dispatch
  reuses its existing Tag0 scaleup/scaleout release-acquire barrier.

A separate minimal-schedule audit then froze exact owner_channel_prefix,
moved_channel_prefix, channel-major/destination-minor group_prefix, and resolver
formulas. It proved sequential fill capacity and dense remote/proxy namespaces.
It also found that C061's old token-major moved-token golden must remain a
compatibility test while C080-B gains a new channel-major 33-copy/six-move
golden.

Final review after checked arithmetic, terminal INVALID state, debug/static
instrumentation, and file consolidation reports:

    0 Blocker
    0 High
    PLAN_FROZEN / IMPLEMENTATION_NOT_STARTED

Executable evidence at freeze:

    PASS 4/4 legacy Hybrid identity goldens
    PASS 38/38 rail-balance planner tests
    PASS default-off EP8 FP8/alignment-128 smoke

Decision: commit and push this plan/golden checkpoint, then begin C080-A
constructor/config/capability integration. No Hybrid source edit preceded the
freeze.

## 2026-07-21 — D024: C080-A fail-closed API and arena ABI

- Added keyword-only `rail_balance="off|force"` and
  `rail_balance_proxy_slots_per_rank` constructor configuration. Off/zero takes
  the exact old constructor, size calculator, runtime, JIT, instance-field, and
  EPHandle paths.
- Added a dual host/extension capability check. It is deliberately false until
  both force dispatch and combine exist; missing, false, or throwing compiled
  capability all fail closed before touching process groups, CUDA, or NCCL.
- Froze a force-only tail arena containing one 32-byte control block, one fixed
  channel-count array, and exactly `Pcap` legacy BF16 dispatch plus `Pcap`
  legacy combine payload slots. Every section is 32-byte aligned and the arena
  is rounded to 2 MiB. No descriptor, ready/generation, route sidecar, or ring
  was added.
- Independent audit found a real High before commit: the host channel heuristic
  can calculate 1280 channels, while the immutable legacy Hybrid workspace is
  compiled for 1024. The first arena draft copied 1280 and would have permitted
  legacy OOB access. Force-v1 now inherits `deep_ep::kNumMaxChannels == 1024`;
  runtime preflight must reject larger C before any arena write or collective.
- Rebuilt the extension and passed layout 5/5, API 6/6, legacy identity 4/4,
  and the 8-rank NCCL/all-gather buffer formula.

Failures retained:

- A read-only audit ran the updated 1024 golden while `_C.so` still contained
  the earlier 1280 build and correctly failed. Rebuilding closed it; this was a
  stale-extension failure, not a formula failure.
- Direct test commands without `PYTHONPATH=.` failed with
  `ModuleNotFoundError`. A guessed legacy identity filename was also wrong.
  Both commands were corrected without changing code.
- The first EP8 formula run used default port 8361 and failed `EADDRINUSE`.
  Re-running with `MASTER_PORT=29881` passed on all eight H200s.

Checkpoint commits, both pushed to the fork branch:

    85b6b48 feat: add fail-closed rail balance config gate
    9ba9ae8 feat: expose disabled hybrid rail capability
    bb47ad0 feat: freeze hybrid rail balance arena layout

## 2026-07-21 — D025: C080-B compact CPU oracle freeze

- Added a deliberately slow CPU standard answer for channel-major count,
  minimum-move quota, retained/moved placement, owner and moved prefixes,
  channel-major group prefixes, and capacity fail-closed behavior. Exhaustive
  copy enumeration exists only in tests; it is not stored in the schedule and
  is not a production manifest.
- The first review found a Blocker in an otherwise correct result: segments
  were flattened and included destination, making six integers. The frozen ABI
  already buckets by destination. It is now `segments[D][<=G-1]`, with each
  record exactly `(owner, egress, owner_begin, count, egress_begin)`.
- Added the strict real-input adapter from rectangular top-k expert ids to
  deduplicated remote destination servers. Force input rejects masks, ragged K,
  duplicate experts, invalid expert ids, K/D/G/C limit violations, invalid
  expert divisibility, nonpositive Pcap, and `G*D*T`/prefix int32 overflow.
- Every schedule test cross-checks count/quota/keep and the first four segment
  fields against the established C030 oracle; egress_begin and grouped slot
  namespaces are checked independently.
- C061 remains the fixed channel-major golden: 33 destination copies, six
  moved copies, Pcap 3 succeeds and Pcap 2 disables the candidate atomically.

Evidence:

    PASS 11/11 C080-B Hybrid CPU reference tests
    PASS 38/38 existing rail-balance planner tests
    PASS py_compile and git diff --check
    independent rereview: 0 unresolved Blocker / 0 High

Checkpoint `3676193 test: freeze hybrid rail schedule oracle` was pushed.
No force Hybrid data-path source is enabled at this point.

## 2026-07-21 — D026: C080-B1 GPU schedule materializer

- Added a private, single-process CUDA proof entry from stacked real
  `topk_idx[G,N,K]` to the compact force-v1 schedule. Three isolated JIT
  kernels perform strict route validation and destination-deduplicated
  `[G,C,D]` counting, canonical minimum-move quota/segment construction, and
  retained/moved/group prefix materialization.
- The returned ABI is fourteen owning, contiguous CUDA int32 tensors. Transfer
  segments remain destination-bucketed five-int records; no per-copy manifest,
  descriptor, ready word, ring, or hot-path per-copy atomic was introduced.
- Capacity exhaustion is the only non-throwing status and preserves the full
  candidate schedule. Invalid/masked or duplicate expert routes fail before
  planning. Exact Python integer seeds use the nonnegative signed-int64 host
  ABI and normalize to the local rail ring.
- The shared device resolver maps a channel-local destination copy to retained
  or moved egress/channel/remote/proxy slots from compact prefixes. B1 does not
  claim LSA count exchange or Hybrid dispatch/combine integration; those remain
  B2/C080-C onward.

Failures found and retained before checkpoint:

- The first segment two-pointer compared full surplus/deficit after partial
  consumption. A split segment could therefore compute zero repeatedly and
  never advance. Remaining counts now subtract per-side consumed cursors and
  assert every emitted amount is positive.
- The first JIT form included three non-template `__global__` definitions in
  every generated cubin. DeepEP correctly rejected the cache because each
  runtime requires exactly one kernel symbol. Each kernel is now a template
  and the corresponding runtime explicitly instantiates only `<0>`.
- Independent audit found that a legal `[G,0,K]` CUDA tensor can have a null
  data pointer. The count kernel used to return without zeroing
  `channel_count`; it now requires a non-null input only when `N>0` and writes
  every compact count as zero for the empty case.
- Independent audit also found that the CPU oracle rejected seeds above int32
  while the private host binding deliberately accepted signed int64. The
  oracle now matches the host ABI; `1<<40` compares exactly and `1<<70` is
  rejected.

Evidence after these fixes:

    PASS 69 exact H200 CUDA cases
    PASS C061 33 copies / six moves / Pcap 3 success / Pcap 2 failure
    PASS zero-token, 64 seeded random, C=1024/D=32 boundary
    PASS masked/out-of-range/duplicate route and invalid seed rejection
    PASS non-default CUDA stream
    PASS 11/11 Hybrid CPU oracle, 38/38 old planner, 4/4 legacy identity
    PASS one exported kernel symbol in each of the three fresh-cache cubins
    independent final review: 0 Blocker / 0 High

Performance conclusions are deliberately deferred. Current GPU occupancy and
timing are correctness evidence only; C100 will use idle GPUs plus NCU/Nsys to
measure the serial plan CTA, prefix kernel, launch gaps, memory traffic, and
later LSA shuffle/unshuffle before changing the simple implementation.

## 2026-07-21 — D027: C080-B2/C local snapshot and gate plan freeze

A read-only audit froze the smallest safe bridge from B1 to the real symmetric
arena. Each rank runs the existing count core with one owner and writes compact
`[C,D]` into its own registered force arena. After WORLD Gate #1, an independent
force-only barrier specialized as `(scaleout=1, scaleup=G)` performs only LSA
synchronization; G active-prefix peer D2D copies form local `[G,C,D]`, which is
fed to the unchanged B1 plan and prefix kernels. No sidecar, descriptor, ready
state, queue, fixed-stride count layout, or gather protocol is added.

The audit caught three control boundaries before implementation:

- C++ `ElasticBuffer` has no ProcessGroup, so a monolithic method cannot safely
  put WORLD Gate #1 between local validation and the device LSA barrier. Force
  needs private prepare/finish-plan/abort phases behind one Python dispatch.
- The local barrier cubin must be built before Gate #1. A cold-JIT failure after
  another rank enters the barrier would deadlock.
- Calling the existing full Hybrid barrier for the count snapshot would enter
  Rail on a real multi-node topology. The new barrier runtime must compile a
  local-only synthetic topology and use the physical LSA rank.
- The synthetic barrier receives the existing legacy `workspace` barrier state,
  not the force arena base. Passing the force arena would reinterpret and
  overwrite the first four `HybridControl` fields as barrier counters/signals.
  Its exact specialization is scaleup-NVLink, sequential, `(1,G)`, rank
  `(0,nvl_rank)`, with no concurrent force operation using the workspace.
- The shared legacy barrier phase is never cleared or rolled back. Gate1
  failure consumes no phase; Gate1 success consumes one phase on all local
  ranks even when Gate2 fails, and abort clears only pending force state. The
  concurrency ban covers every operation on the same buffer that uses the
  workspace barrier, not only another force call.

A second review found four protocol requirements and the plan now freezes them:

- Gate payloads include operation/phase/epoch/state plus a fixed configuration,
  topology/rank, arena, SM/QP, and force-flag fingerprint. They do not require
  per-rank token count N to match.
- Force branches before legacy Python dispatch performs rank-local argument or
  handle work; the entire force normalization and C++ prepare region converts
  local exceptions into Gate1 status.
- All output tensors and the barrier/count/plan/prefix runtimes exist before
  Gate1. Returnable errors after the local barrier convert into Gate2 status;
  context/barrier failure is explicitly fatal rather than falsely recoverable.
- Each peer copy uses an LSA-local symmetric pointer, a local snapshot
  destination, checked active-prefix bytes, `cudaMemcpyDeviceToDevice`, and the
  DeepEP comm stream. LSA rank is never interpreted as CUDA device/global rank.
- Production Gate1/Gate2 use fixed tensors allocated and collectively warmed
  before a dispatch can fail. The error path cannot allocate or call an object
  collective. Short-timeout Gloo object exchange remains private test
  scaffolding only; failure of the warmed production gate is fatal.
- Prepare is transactional and only IDLE enters PREPARING. Busy on a second
  dispatch does not touch the old DISPATCH_LIVE handle or arena; abort restores
  the transaction's recorded entry state rather than unconditionally clearing
  the buffer to IDLE.

An additional constructor contract conflict remains explicit: detecting a
mixed off/force configuration before NCCL symmetric-window creation requires
all modes to join one initialization consensus, while absolute off-mode
zero-new-collective identity forbids it. Public capability remains false while
B2 is private. Before force is enabled, safety wins: either a one-time
constructor control consensus is accepted and documented, or the public
supported contract is narrowed and rereviewed. It will not be hidden in the
data path.

The local machine can prove LSA addresses, barrier reuse, snapshot equality,
zero/different token counts, Gate1/Gate2 fault convergence, and prefix
correctness with virtual destinations. It still cannot label this real Hybrid
runtime because physical scaleout is one.

Final independent plan review reports 0 Blocker / 0 High. Local profiling tool
inventory is available for later controlled stages: NCU 2025.1.1, Nsys
2024.6.2, Compute Sanitizer 2025.1, and CUDA/NVCC 12.8. No timing was taken as
evidence during this plan checkpoint.

## 2026-07-21 — D028: C080-B2 real local-LSA count snapshot

The private Hybrid planner now has a production-shaped, two-phase C++
transaction. `prepare` validates the local route, allocates all fourteen plan
outputs, builds count/plan/prefix plus a uniquely keyed local barrier cubin,
orders the caller stream before the communication stream, and publishes the
compact local `[C,D]` count into the symmetric force arena. Only status zero
creates pending state. `finish` consumes one synthetic `(scaleout=1,
scaleup=G)` LSA barrier epoch over the legacy workspace, copies each peer's
active prefix into local `[G,C,D]`, and runs the unchanged B1 plan/prefix.
Matching explicit `abort` releases the pending transaction without clearing or
rolling back the shared barrier phase. Public force capability remains false.

The final count store uses system-scope release semantics. The barrier cubin is
specialized exactly as
`barrier_impl<true,1,512,1,8,356400000000,true>` and exports one kernel symbol.
Arena bounds are checked before pointer arithmetic; the runtime asserts a real
NVLink scaleup team and treats peer indices only as LSA-local ranks. All device
work after prepare is on the DeepEP communication stream, and the final status
copy synchronizes it before host inspection.

Strict fresh-cache 8-GPU evidence:

```text
cache: /tmp/deepep-c080b2-lsa.HjWFtF
port:  29884
PASS C080-B2 CPU oracle: C061-like 8-rail skew, moved=21, Pcap=5/4,
     boundary C=1024/D=32
PASS C080-B2 local LSA plan transaction: variable/zero N, exact fourteen
     tensors, identical eight-rank snapshot digest, reusable barrier phase,
     Gate1 route/config abort recovery, Gate2 capacity recovery,
     C1024/D32, int64 seed, non-default stream
PASS independent fresh-cache rerun at /tmp/deepep-c080b2-rerun.TqGFH7,
     port 29885
PASS fresh-cache B1 full matrix: 69 exact CUDA cases at
     /tmp/deepep-c080b2-b1full.HJikws
PASS extension build, CPU/reference/API/layout/legacy gates, diff check
```

The exact accepted launch was:

```bash
EP_JIT_CACHE_DIR=/tmp/deepep-c080b2-lsa.HjWFtF \
EP_DISABLE_GIN=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_hybrid_plan_lsa.py \
  --timeout 180 --watchdog-seconds 900 --master-port 29884
```

### Failed attempts retained

1. The first 8-GPU harness run completed the first LSA transaction and abort,
   then its Gloo helper rejected a gathered `None`. `None` was the legitimate
   no-error payload, not an unfilled slot. The invalid non-`None` assertion was
   removed; the outer watchdog contained the failure and the expected explicit
   buffer-leak warnings were limited to those aborted workers.
2. The second run reached the exact comparison, but `init_dist()` had installed
   the rank-local CUDA device as PyTorch's default. Oracle constructors that did
   not name a device therefore created CUDA expected tensors. Every oracle
   tensor now explicitly uses CPU; the next fresh-cache run passed.
3. Bare `python` remains absent and system `python3` remains outside the verified
   PyTorch/NCCL environment. All accepted evidence uses the pinned project
   interpreter shown above.

This checkpoint proves local LSA visibility, phase reuse, compact snapshot
identity, and private fault convergence. It does not prove production WORLD
gate behavior, source payload shuffle, Hybrid dispatch/combine, Gin, QP, NIC,
fabric, or speedup. Those labels remain reserved for C080-C/D onward and the
real multi-node gate.

### Focused Compute Sanitizer evidence

With all GPUs idle and no Megatron/MGT process present, Compute Sanitizer
2025.1 ran the smallest B1 materializer matrix against fresh JIT caches:

```text
memcheck cache /tmp/deepep-c080-b2-memcheck.t0zGypsK
  exit 0; ERROR SUMMARY: 0 errors
synccheck cache /tmp/deepep-c080-b2-synccheck.VQZ3wIrz
  exit 0; ERROR SUMMARY: 0 errors
```

Both runs passed six exact CUDA cases, route/seed/C1025/D33 rejection, and the
non-default-stream case. This evidence covers the single-GPU count/plan/prefix
kernels only. It does not validate cross-GPU system-scope ordering, the B2 LSA
barrier, or peer D2D copies; those require a separately bounded multi-process
tool run after the shared-core data path exists.

### Independent implementation audit

Final result is 0 Blocker / 0 High. Three Medium boundaries are retained for
the next state-machine checkpoint:

1. Private pending state rejects a second private prepare but does not yet gate
   unrelated legacy dispatch/combine/barrier calls on the same `ElasticBuffer`.
   Rank-inconsistent interleaving could advance the shared workspace phase
   differently. The private harness is serial and public force is disabled;
   the public transaction must install one unified buffer state machine.
2. The true-LSA harness covers route status 2, capacity status 1, configuration
   mismatch, and recovery, but not duplicate-route status 3, busy prepare,
   stale abort, double finish, or a single-rank host-prepare exception. These
   fault cases are required before C080-D/public state integration closes.
3. Gloo object collectives are test scaffolding and the physical topology is
   one node. They prove local barrier/snapshot convergence, not production
   fixed-tensor WORLD gates or rank-aware multi-node consensus.

## 2026-07-21 — D030: C080-D direct-final shuffle implementation freeze

Two independent read-only audits reduced the next shared-core slice to one
force-only kernel. A source channel owns one warp and visits tokens in the same
`channel, channel+C, ...` order as Hybrid dispatch. Destination lanes keep
their own channel-local ordinals, deduplicate a token's remote servers, and call
the existing compact `resolve_hybrid_copy`. Retained copies stay on the legacy
path; only moved copies enter this kernel.

Each moved token is staged once as the unmodified legacy BF16 dispatch
`TokenLayout`: hidden, top-k ids, weights, original global source token, and
linked-list scratch. For each moved destination, the kernel sets only
`linked_list_idx[0]=p`, leaves the remaining linked-list entries at `-1`, and
performs a complete TMA store plus wait directly into the selected egress's
symmetric `HybridArenaLayout::proxy_dispatch[p]`. The wait precedes changing
`p` for another destination of the same token. Normal execution has no CPU or
GPU manifest, `ProxyDescriptor`, ready/generation state, queue, ring, route
sidecar, or data-path atomic.

The production sequence adds no standalone post-shuffle barrier. Source
shuffle and the future force Hybrid dispatch are ordered on the same DeepEP
communication stream, and the existing Hybrid Tag0 barrier is the cross-GPU
visibility epoch. The isolated 8x1 verification may use one all-rank local
barrier before readback; that barrier is test scaffolding and is not part of
the production source-shuffle kernel.

Runtime defenses still check `p < proxy_required[egress] <= Pcap`, target
channel/remote-slot capacity, owner/egress ranges, and resolver success before
forming a peer address. A rare error may atomically publish status; successful
copies perform no atomic. All ranks launch the same prebuilt kernel even when
their local N or moved count is zero. JIT build and every allocation remain on
the pre-Gate1 side of the pending transaction.

The first acceptance slice is true 8-GPU 8x1: reconstruct expected moved copies
only in the test oracle, parse raw pure-`TokenLayout` proxy slots, and compare
BF16 hidden bytes, FP32 weights, top-k ids, original source token, four-byte
transit key, slot uniqueness, byte-unchanged unused slots, and repeated reuse.
The test snapshots the arena before shuffle; production does not pay a full
Pcap poison or memset. Only after
this passes will a test-only adapter feed the same descriptor-free payloads
into the existing 4x2/2x4 functional emulator; old vnode record/protocol ABIs
are not production dependencies.

## 2026-07-21 — D031: C080-D true-LSA source shuffle checkpoint

The frozen direct-final source slice is now implemented behind private APIs and
public force remains disabled. One 32-thread block owns each source channel,
reconstructs a deduplicated destination ordinal, resolves the compact
segments/prefixes, and skips retained copies. A moved token is loaded into one
shared legacy BF16 `TokenLayout`; each moved destination changes only
`linked_list_idx[0]=p` and receives a complete TMA store directly into the
selected peer's final proxy-dispatch slot. Every store commits and waits before
the next p is written. The success path has no manifest, descriptor, ready
word, queue, ring, route sidecar, or global atomic.

Two independent audits found 0 Blocker and no confirmed data-path High. One
audit exposed a non-default-stream evidence race in the test: inputs created on
the caller stream lacked an explicit edge to the alternate stream, and the
snapshot was consumed from the default stream before an explicit wait. The
test now installs both dependencies. It also verifies zeroed metadata padding;
unknown device status is converted into sticky InvalidSchedule instead of
asserting before poisoning the transaction.

The second audit found an artificial `D<=K` host restriction. It was inherited
from the first combine simplification, not from planner/source mechanics. The
restriction was removed from the private planner, CPU oracle, and source path.
For future unshuffle, legacy layout selection is still derivable without new
metadata: row=destination for D<=K, otherwise row=the highest top-k lane targeting
that destination. The preserved dispatch TokenLayout contains the required
expert ids.

Accepted local evidence on idle 8xH200:

```text
PASS source oracle: C061 moved=21, Pcap=6, seeds 62/63;
     all-owner C1024/D32/K4
PASS true 8-GPU LSA source shuffle: C061 multi-destination/multi-copy,
     zero-N owners, all eight owners active, D32>K4, exact Pcap,
     repeated transaction, non-default stream, exact legacy TokenLayout bytes,
     source immutability, and byte-unchanged inactive slots
PASS B1: 70 exact CUDA plans including D32>K4, plus route/seed/C1025/D33
     rejection and non-default stream
PASS 11/11 CPU reference, 6/6 API, 5/5 layout, 4/4 legacy goldens,
     extension rebuild, py_compile, and diff check
```

Static cubin inspection for H256/K4 reports 74 registers, 1024 shared bytes,
0 local bytes, and a 72-byte stack/parameter allocation. This is only a
profiling target, not a performance regression claim; NCU/Nsys evidence is
deferred until the correctness boundary matrix is stronger.

### Failed attempts retained

1. The first local API/layout rerun omitted `PYTHONPATH=.` and both commands
   failed with `ModuleNotFoundError: deep_ep`; the corrected pinned-interpreter
   commands passed 6/6 and 5/5.
2. The first host rebuild after moving `HybridPlanError` into the shared layout
   failed because the enum was not visible to host code. Moving the common
   status definition out of the JIT-only plan header fixed the build.
3. A test worker invoked the raw Conda interpreter and failed on the known
   `undefined symbol: ncclCommQueryProperties`; all accepted evidence uses
   `/home/chen/.cache/deepep-sjlgpt/bin/python`.
4. The first D32>K4 oracle patch forgot to import
   `_encode_destination_rows`, causing `NameError`; after the import, its first
   expected proxy distribution assumed uniform `(3,...,3)` but the rotating
   remainder seed correctly produced `(4,1,4,4,3,2,3,3)`. The independent CPU
   schedule supplied the corrected exact golden.

This checkpoint is only the 8x1 source data slice. It does not prove H7168,
corrupt-plan fail-closed behavior, 4x2/2x4 round trip, return-unshuffle, force
Hybrid dispatch/combine, Gin/QP/NIC behavior, or speedup. C080-D remains
IN_PROGRESS.

## 2026-07-21 — D032: source-shuffle boundary, sanitizer, and profiler closure

The descriptor-free source slice now passes the remaining local 8x1 boundary
matrix. A single max-hidden arena was reused in both directions across
H256 -> H7168 -> H256 transactions. The H7168 case publishes 21 complete
legacy TokenLayouts; the C1024/D32/K4 case gives every owner four input tokens
and moves three copies per owner. Balanced input and a single global token both
publish zero bytes. Exact-capacity, source immutability, metadata padding,
inactive-slot poison, zero-N owners, and a delayed non-default producer stream
remain checked.

Three corrupt compact-plan variants (`num_segments`, `group_prefix`, and
`proxy_required`) report sticky status 4 on every moved owner before an invalid
peer write. A capacity-disabled plan reports status 1, a duplicate expert
reports status 3, every second source-shuffle call is rejected, and a complete
transaction after every abort proves barrier-phase recovery. The host launch
now uses `max(1,min(C,N))` blocks: channel c owns only tokens
`c,c+C,...`, so every omitted c>=N is empty; N=0 keeps one validation block.

Focused tool evidence on otherwise idle 8xH200:

```text
Nsys /tmp/deepep-c080d-final-T2v33l/c080d_final.nsys-rep
  C1024/N4 launches grid=4 on all ranks; moved-owner source kernels remain
  about 127 us, statistically unchanged from the earlier grid=1024 baseline
  at /tmp/deepep-c080d-nsys-baseline.nsys-rep (about 125 us).

NCU /tmp/deepep-c080d-ncu-basic-moved.ncu-rep
  grid=4, block=32, REG=74, dynamic shared=608 B, duration=130.3 us;
  achieved occupancy=1.56%, DRAM throughput rounds to 0%, L1=2.73%.
NCU /tmp/deepep-c080d-ncu-nvlink-moved.ncu-rep
  transmitted user bytes=1728 B, exactly 3 * 576-B dispatch records;
  transmitted overhead=1248 B, peak link utilization=0.02%.
NCU /tmp/deepep-c080d-ncu-stalls-moved.ncu-rep
  scheduler no-eligible=92.07%, active warps/scheduler=0.99.
NCU /tmp/deepep-c080d-ncu-pmsampling-moved.ncu-rep
  dominant sampled state is long scoreboard (33.58 average warps), followed
  by wait (13.75) and short scoreboard (6.85); no samples were dropped.

Compute Sanitizer source/fault transaction:
  filtered memcheck: ERROR SUMMARY 0
  filtered synccheck: ERROR SUMMARY 0
  device-side initcheck with API-memory checking disabled: ERROR SUMMARY 0
  filtered racecheck: 0 errors, 6 WAW warnings
```

The racecheck warnings map to the intentional shared-metadata sequence: SASS
`+0xef0` vector-clears aligned metadata, a warp sync separates it from scalar
top-k/weight/src/linked writes at `+0x1710..+0x1780`. They are retained rather
than hidden. The unfiltered initcheck also retained five 32-byte host-API
warnings at the peer-count D2D snapshot. Here C=2,D=4 and the count kernel
unconditionally release-stores all eight ints; the warnings occur only through
NCCL imported LSA peer aliases. Compute Sanitizer 2025.1's bundled release
notes state that initcheck does not support IPC allocations and produces false
positives. Disabling only API-memory checking leaves device reads checked and
passes with zero errors; no redundant production memset was added.

### Failed or rejected attempts retained

1. Unfiltered memcheck with API reporting enabled emitted 328
   `cudaErrorNoKernelImageForDevice` reports from NCCL initialization probes
   before the JIT kernel, although the full application passed. Filtering the
   source kernel and disabling API reports produced the valid zero-error run.
2. Filtering initcheck to the source kernel produced 3701 false uninitialized
   reports because the excluded planner/fill kernels' writes were not tracked.
   The valid device-side run includes every kernel and disables only IPC host
   API shadow checking.
3. The first NCU launch-count sample selected a zero-token rank and measured
   only the 3.328-us empty path. A default-preserving `--case-name` test option
   selected the all-owner moved case for every accepted NCU report above.
4. Replacing the compact resolver's <=7-entry linear segment scan with binary
   search passed correctness but measured about 134 us versus about 127 us for
   the retained linear version. The candidate was reverted before commit.
5. An audit disproved the planned D>K "first matching top-k lane" rule. Legacy
   uses `bfind/get_master_lane_idx`, hence the highest matching lane. The CPU
   oracle, plan, context, and return-unshuffle design now use the legacy rule;
   `[2,1,2]` for destination 2 is the explicit row-2 counterexample.

This closes the 8x1 source-only evidence. It still does not prove the 4x2/2x4
round trip, return-unshuffle GPU ordering, force Hybrid dispatch/combine, or
Gin/QP/NIC behavior. Public force remains disabled.

## 2026-07-21 — D033: production-shared return-unshuffle checkpoint

The combine return leg now has a production-shared, descriptor-free kernel.
One warp owns each `(egress,channel)` sequence, validates its dense static group
prefix, and visits every local `proxy_return[p]` exactly once. The preserved
dispatch record supplies `src_token_global_idx` and the four-byte transit key;
owner, token, source node, and reduce row are derived locally. The kernel TMA
loads one complete combine TokenLayout into shared memory and TMA stores it
through LSA to the original owner's existing legacy reduce buffer. It adds no
return header, route array, ready/generation word, ring, or success-path atomic.

For D<=K the reduce row is the destination. For D>K the kernel validates all K
preserved experts and uses `get_master_lane_idx`, matching legacy Hybrid's
highest matching lane. It rejects an out-of-domain source node, a moved record
for the local destination, a non-moved owner/egress identity, an invalid p key,
and malformed group/capacity state. An independent audit reports 0 Blocker and
0 High under the frozen Gate-2 plan contract.

The audit retained an important corruption boundary: validation and movement
are fused. If a published plan is artificially mutated after Gate 2, another
block may have completed a peer store before the sticky error is observed.
Such an error invalidates the whole transaction; it does not promise rollback
or an unchanged scratch reduce buffer. A metadata-only validation kernel would
be required for that stronger property and was rejected as unnecessary hot
path work for the immutable-plan contract.

Accepted evidence:

```text
PASS production-JIT-like sm_90a static cubin:
     H256/K4 and H7168/K8, REG=60, STACK=72 B, LOCAL=0, spill=0
PASS extension build_ext --inplace
PASS C080-F CPU oracle
PASS true 8-GPU return-unshuffle:
     C061 H256 moved=21
     all-owner C1024/D32>K4 H256 moved=24, legacy row=3
     C061 H7168 moved=21
PASS complete raw combine TokenLayout equality, unique fingerprints,
     collision-free (owner,row,token), all non-target bytes remain 0xA7,
     one-shot rejection, idempotent abort, and next-transaction recovery
```

The accepted GPU run completed while unrelated vLLM work occupied GPUs 4-7.
There was enough memory and functionality passed, but no timing from this run
is performance evidence. The processes were not Megatron/MGT and were not
terminated.

### Failed attempts retained

1. The first local command used bare `python`, which is absent; the next command
   used the pinned interpreter but omitted `PYTHONPATH=.`, producing
   `ModuleNotFoundError: deep_ep`. The corrected pinned command passed.
2. The first 8-GPU test already passed the full reduce-snapshot equality, then
   failed in a redundant poison-mask assertion because `init_dist` had changed
   PyTorch's default device to CUDA while the snapshot was on CPU. Explicitly
   constructing the mask and index on CPU fixed only the test; the complete
   second run passed.
3. Earlier RDC-object inspection reported STACK=0. Recompiling with the actual
   non-RDC `sm_90a` production-JIT flags shows the authoritative 72-byte stack
   frame. The old observation was a relocatable-fatbin reporting limitation,
   not a resource improvement.

This closes the standalone GPU ordering and layout proof for return-unshuffle.
It does not yet prove destination-side Hybrid forwarding/combine, the full
4x2/2x4 round trip, real Rail/Gin completion, or a performance gain. Public
force remains disabled.

## 2026-07-21 — D034: freeze the minimal C080-D vnode bridge

Three independent read-only audits froze the bridge between the proved Hybrid
source/return kernels and the older 8-GPU vnode transport. The bridge uses a
source-subgroup ElasticBuffer of G ranks and a separate eight-rank world
ElasticBuffer. Each object owns its own NCCL device communicator, symmetric
allocation, and window. Passing a symmetric pointer between them is invalid:
`get_sym_ptr` interprets an address relative to the receiving object's window.
The accepted bridge therefore uses owning local CUDA snapshots in both
directions. Those copies are test scaffolding and cannot be included in a
production data-path timing.

Only `channel_count[G,C,D]` crosses from the source plan to the world adapter.
Every world rank reuses the existing plan/prefix kernels to reconstruct the
same quota, segments, retained/moved counts, and compact group prefixes. This
avoids a fourteen-tensor private ABI, avoids padding variable source N, and
keeps the production count kernel unchanged. The source and world plan digests
must match before the vnode stages commit. VNode receives a real contiguous
remote quota `[G,D-1]`; a pointer offset into `[G,D]` would preserve the wrong
row pitch.

The accepted adapter has only two one-warp kernels. Pack derives its
destination-wide dense prefix as

```text
min(owner_channel_prefix[e,c,d], keep_count[e,d])
    + moved_channel_prefix[e,d,c]
```

and materializes retained local records plus actual moved proxy-dispatch
records into the old vnode base namespace. Return demux validates the old
base/contribution/route records, reduces matching expert lanes with the legacy
combine helpers, and reconstructs a complete combine TokenLayout. Retained
records become a local reduce seed; moved records become proxy-return p. The
already proved source-buffer return-unshuffle performs the only peer writes,
then the unchanged legacy epilogue produces the final result.

Independent schedule exhaustion covered 1,920 combinations of G, C, and D and
found no dense-slot hole, collision, or overflow. The return audit proved the
old vnode physical formulas, the moved/retained inverse, and the D>K highest
matching-lane row rule. No Blocker was found in either adapter kernel.

### Rejected alternatives and retained boundaries

1. A new `O(G*C*D)` destination prefix was rejected: existing retained and
   moved prefixes already give the exact dense base in O(1).
2. Cross-object raw symmetric pointers were rejected because they belong to
   different NCCL windows. Exposing uintptr values or adding friend plumbing
   would make an unsafe test abstraction.
3. Passing all fourteen plan tensors was rejected. Passing one actual GPU
   channel-count tensor and rebuilding with shared kernels is smaller and
   independently checkable.
4. The old vnode emulator is not generalized for local expert lanes. Its
   forwarder decodes every lane in a remote-only namespace, so the first GPU
   fixtures are explicitly remote-only. This is an emulator boundary, not a
   production Hybrid limitation.
5. Dispatch records are not copied as combine records. Their metadata layouts
   differ; demux reconstructs combine metadata and zeroes padding.
6. `csrc/kernels/elastic/combine.hpp` remains untouched. A force-only wrapper
   may prepare the exact existing epilogue specialization and JIT key.
7. The source ranks must be the world prefix `[0,G)`, so subgroup-local source
   ids equal the emulator's world-local owner ids. A non-prefix virtual source
   topology is outside this proof.

The only environment fact still unproved is that a newly created source NCCL
subgroup exposes the expected `(1,G)` LSA topology on this installation. The
first GPU harness asserts it before publication. Public force remains
disabled.

## 2026-07-21 — D035: implement and audit the two-kernel vnode adapter

The frozen bridge now has concrete device and JIT wrappers. `pack_vnode_base`
materializes retained source records and actual moved proxy-dispatch records
into one dense old-vnode prefix per `(egress,destination)`. `return_demux`
validates the returned base/contribution/route state, applies the existing
`compute_topk_slots` plus `combine_reduce` semantics, and emits either a
retained legacy-reduce seed or a moved proxy-return record. A separate private
wrapper prepares the exact unchanged legacy combine epilogue specialization;
`combine.hpp` remains untouched.

An independent device audit initially found one High retained-path identity
hole. The demux read `owner_token` from the vnode descriptor and validated the
descriptor's `src_token_global_idx` against that same field. Corrupting both to
another token in the same channel could therefore redirect a correct payload
to the wrong reduce slot without setting status. The fix independently reads
the base TokenLayout `src_token_global_idx` and compares it, with int64
arithmetic, against `expected_owner*M+owner_token`. Moved records already had
an independent proxy-dispatch identity check.

The same audit found that an accidentally oversized launch grid could report
an argument error through `status[blockIdx.x]` after the valid status extent.
Both adapters now return without writing for an out-of-range block or extra
warp. The real launchers still use the exact `C` and `(D-1)*M` grids. The host
bridge must allocate and zero those two status ranges separately; it may not
reuse the plan's one-element status tensor.

Post-fix production-JIT-like non-RDC `sm_90a` compilation passed:

```text
pack  H256/K4:   REG=90,  STACK=72 B, LOCAL=0, spill=0
pack  H7168/K8:  REG=92,  STACK=72 B, LOCAL=0, spill=0
demux H256/K4:   REG=64,  STACK=72 B, LOCAL=0, spill=0
demux H7168/K8:  REG=163, STACK=72 B, LOCAL=0, spill=0
SASS: no global ATOM. or RED. instruction
```

`REDUX.OR` remains, correctly, for warp voting; it is not a global atomic and
must not be filtered as one. The temporary static compile source was removed;
the cubin is `/tmp/c080_vnode_post_audit.cubin`. The H7168 demux register count
is accepted only because this is a correctness adapter outside the production
Hybrid hot path.

The remote-only CPU structural fixtures now pass 6/6 and the legacy Hybrid
identity goldens pass 4/4. They cover 4x2 `D<=K`, 2x4 `D>K`, variable source N,
retained/moved routes, dense old-vnode slots, p0/p1, and the highest matching
lane. A second independent fixture audit retained two evidence gaps before GPU
acceptance: the chosen numeric values are not yet rounding-sensitive, and all
moved target-channel/group prefixes are still zero. Those are test gaps, not
accepted CUDA evidence, and the next checkpoint strengthens the fixtures
before the 8-GPU round trip.

### Failed attempts retained

1. A baseline command named the nonexistent
   `tests/elastic/test_rail_balance_hybrid_jit_identity.py`; the correct legacy
   identity test is `test_rail_balance_hybrid_legacy_golden.py` and passes.
2. An environment-discovery snippet tried `from deep_ep import envs`, but this
   source-tree package does not export generated build-time env metadata. The
   static compiler instead used the pinned environment's NCCL include path.
3. A broad SASS search for `ATOM|RED` falsely matched `REDUX.OR`. The accepted
   check searches only global `ATOM.` and `RED.` opcodes.

No GPU execution is claimed at this checkpoint. Public force remains disabled.
