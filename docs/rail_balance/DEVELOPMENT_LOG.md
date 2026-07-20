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
