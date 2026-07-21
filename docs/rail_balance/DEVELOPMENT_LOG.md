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

## 2026-07-21 — D036: preserve the legacy epilogue and strengthen its oracle

The source-side test transaction now prebuilds the exact existing
`combine_reduce_epilogue` specialization before its first arena write. After
the proved return-unshuffle has populated the legacy reduce buffer, a private
one-shot hook launches that same runtime with the original JIT key, generated
template arguments, shared-memory/thread configuration, and PDL setting. It
uses the pending original top-k ids as the epilogue's routing input and returns
owning BF16 output plus FP32 combined weights. The public dispatch/combine API,
default capability gates, and `combine.hpp` remain unchanged.

An independent host-state audit found a pre-existing handoff hole while
reviewing the new epilogue guard. `return_unshuffle_tested` was set before its
committed copies and barriers, but an exception in that interval could leave
`plan_status==0`; a caller that caught the exception could then invoke the new
epilogue on partial data. The complete committed return interval is now inside
a try/catch that makes `InvalidSchedule` sticky. The epilogue launch/sync has
the same one-shot sticky failure rule, so neither partial stage can be retried
or consumed.

The CPU vnode oracle was also strengthened in commit `27dcb4c`. A full
roundtrip now distinguishes the correct two-level legacy BF16 result `1.0`
(`0x3f80`) from an incorrect flat three-slot result `1.0078125` (`0x3f81`).
The 2x4 fixture now exercises target channel one, nonzero incoming moved
prefix, nonzero group prefix, and dense proxy slots p0 through p3. Flat
reduction remains diagnostic only; the destination reduction followed by the
legacy epilogue is authoritative.

Accepted static/CPU evidence:

```text
PASS full extension build_ext --inplace after the final sticky-state fix
PASS C080-A private API/off checks 6/6
PASS legacy Hybrid identity goldens 4/4
PASS private hook is bound on deep_ep._C.ElasticBuffer
PASS vnode CPU oracle 7/7 plus Torch BF16 bit cross-check
PASS independent epilogue host audit: Blocker/High/Medium = 0/0/0
```

### Failed attempts retained

1. A binding check incorrectly looked on `deep_ep.Buffer`, which is the legacy
   Python wrapper and intentionally does not expose Elastic private hooks. The
   correct object is `deep_ep._C.ElasticBuffer`; the rebuilt binding is present.
2. An optional `python -m pytest` run failed because the pinned environment has
   no pytest package. The dependency-free direct runner is canonical and
   passes 7/7; this was an environment failure, not a test failure.

Runtime PDL behavior for `D<=K`, `D>K`, and `N=0` remains part of the upcoming
8-GPU bridge acceptance. Public force remains disabled.

## 2026-07-21 — D037: freeze the local profiling toolchain and evidence policy

The local profiling tools were audited before the full Hybrid vnode harness
was launched. The installed versions are NCU 2025.1.1, Nsight Systems 2024.6.2,
and Compute Sanitizer 2025.1. All required multi-process, kernel-filter, NVTX,
and focused sanitizer controls are available. The source plan/shuffle,
pack/demux, vnode transport stages, return-unshuffle, and unchanged combine
epilogue can each be selected by their concrete kernel names.

The current GPU processes are `mmunlearner_merger` and vLLM workloads. No
process was positively identified as Megatron or standalone MGT, so nothing
was terminated. While those workloads remain, a non-OOM run may count as
functional evidence only; latency, bandwidth, overlap, sanitizer memory
pressure, and throughput measurements are deferred to an idle window.

The accepted profiling order is intentionally narrow:

1. run ordinary H256 correctness without timing claims;
2. run focused memcheck, initcheck, synccheck, then racecheck, one new kernel
   at a time;
3. add NVTX ranges only to the test harness and capture one warmed transaction
   with Nsys;
4. use NCU on one GPU, one kernel, and one launch, starting with the basic set;
5. request detailed scheduler/memory/NVLink sections only after the first
   profile identifies a real bottleneck.

Application replay is preferred for peer-memory kernels. `--kill 1` is
forbidden because terminating one profiled child would strand the other seven
distributed ranks. A full NCU metric set is also rejected as a first pass: it
would replay thousands of counters before a bottleneck has been localized.
The exact commands and kernel filters are retained in the optimization log.

## 2026-07-21 — D038: complete the fail-closed two-window vnode bridge

The private world-side half of the C080-D harness now owns a strict
`prepare/finish/abort` transaction. It accepts only the single-machine world
topology `(1,8)`, a prefix source node `[0,G)`, and `G*D=8`. No raw pointer is
allowed to cross from the source-subgroup symmetric window into the world
window; source payload and count state arrive only as ordinary owning CUDA
tensors.

`prepare` performs every validation, allocation, JIT build, stream edge, GPU
plan/prefix rebuild, device-status observation, and arena-bound check before a
world barrier can be entered. The source plan's `[G,C,D]` count tensor is the
only schedule input crossing the object boundary. The rebuilt quota is copied
with `cudaMemcpy2DAsync` into a genuinely contiguous `[G,D-1]` tensor; a
pointer offset into the pitched `[G,D]` matrix is never exposed to the old
vnode kernels. `prepare` returns status, the complete fourteen-tensor world
plan, and compact quota so Python/Gloo can compare them with the source plan
before B0.

After that gate, `finish` performs no allocation, JIT, host validation, or
ordinary `ElasticBuffer::barrier()` call. Its fixed sequence is:

```text
clear + pack -> B0 -> scaleout -> B1 -> forward -> B2
             -> expert -> B3 -> return -> B4 -> demux -> B5
```

Six independent zeroed status rows cover the exact launch extents `C`, `P`,
`P*K`, `G*M*K`, `P*K`, and `P`. Returned proxy data, reduce seeds,
record/ready/route snapshots, status, guard, compact quota, and nested plan are
all owning CUDA tensors. The transaction becomes one-shot as soon as `finish`
may enter B0; `abort(invocation_id)` is idempotent and only releases the
matching pending state.

Accepted evidence:

```text
PASS clean python_api.o compile and full device-link/shared-library link
PASS import and nested pybind ABI for prepare/finish/abort
PASS C080-A private API/default-off tests 6/6
PASS legacy Hybrid identity goldens 4/4
PASS C080 vnode CPU oracle 7/7
PASS git diff --check
PASS independent world-bridge audit: Blocker/High = 0/0
```

The private bridge intentionally returns live owning plan/quota handles rather
than cloning fifteen tensors. The controlled harness must treat them as
strictly read-only between prepare and finish and immediately perform the
cross-rank digest comparison. This is an accepted test-contract boundary, not
a public API. A failure after B0 remains process-level: all ranks must converge
before pending state is released. No 8-GPU vnode runtime result is claimed at
this checkpoint, and public force remains disabled.

## 2026-07-21 — D039: pass the first production-shaped Hybrid vnode loop

The new `test_rail_balance_hybrid_vnode.py` creates two real NCCL
communicators and two independent `ElasticBuffer` windows in every worker.
The prefix source group runs the production-shaped GPU plan and direct-final
LSA source shuffle. Only owning proxy/count tensors cross to the eight-rank
world buffer, which independently rebuilds the same fourteen-tensor plan
before entering the fixed vnode stages. The returned owning tensors then feed
the production-shared return-unshuffle and the exact legacy combine epilogue
on the source group.

Five separate watchdog subprocesses passed on eight GPUs:

```text
4x2 H256:   G=4 D=2 K=4 N=(4,2,1,1) moved=2
4x2 H7168:  G=4 D=2 K=4 N=(4,2,1,1) moved=2
2x4 H256:   G=2 D=4 K=2 N=(6,6)     moved=6
2x4 H7168:  G=2 D=4 K=2 N=(6,6)     moved=6
2x4 H256:   G=2 D=4 K=3 N=(1,0)     moved=0, rounding-sensitive
```

The 2x4 case exercises `D>K`, target channel one, nonzero incoming-moved and
group prefixes, repeated-destination lanes, and proxy p0 through p3. Both
H7168 cases compare complete raw TokenLayout/record/route bytes. The rounding
case proves the authoritative two-level legacy BF16 result is `1.0`
(`0x3f80`); an incorrectly flattened reduction would produce `129/128`
(`0x3f81`). The zero-token source rank returns exact empty `[0,H]` and `[0,K]`
outputs.

The harness also verifies all fourteen source/world plan tensors, compact
quota, active proxy records, unchanged proxy tail, vnode base/contribution and
expert records, route sidecars, ready values, six exact status extents, the
4-KiB world guard, the complete legacy reduce buffer, immutable source inputs,
combined BF16 output, and combined FP32 weights. The main thread independently
reran 4x2/H256 after all fixes and observed exit zero.

### Audit findings fixed before acceptance

1. The first harness draft performed rank-zero count D2H, Gloo broadcast, and
   per-rank CUDA materialization outside fail-close phases. A one-rank OOM
   could therefore send that rank to abort while peers entered world prepare
   or B0. The three operations and world status/type validation now each
   converge through the control group before the next phase.
2. The first post-shuffle generic source barrier could cold-JIT after proxy
   publication. It is now warmed before any transaction, then retained after
   shuffle as the explicit device/system-scope LSA visibility boundary.
3. Initial exact-proxy validation omitted unused `[required,Pcap)` rows. A
   pre-shuffle owning snapshot now proves those bytes remain unchanged.
4. A minimality refactor temporarily left two deleted coverage return values
   and three new `hidden` parameters at stale call sites. Static review caught
   them before acceptance; all were removed/fixed and GPU regressions reran.

Final evidence is py-compile PASS, direct five-case CPU oracle PASS, all five
GPU cases PASS, main-thread 4x2/H256 PASS, diff check PASS, and independent
harness audit Blocker/High = 0/0. The audited file SHA256 is
`955b1c27d733e62360bf94f943a07c7776c1de2f486b415040b54c688f7de0de`.

Two nonblocking local gaps remain explicit: a zero-token owner has not yet
acted as a deficit egress receiving moved copies, and the complete two-buffer
loop has not yet produced its inputs/calls on a non-default stream. These move
to the boundary/reuse checkpoint together with an all-zero fixture. Public
force remains disabled; real Gin/RDMA execution remains unclaimed.

## 2026-07-21 — D040: close vnode zero-work, stream, and reuse boundaries

The production-shaped two-window harness now adds two deliberately small
fixtures rather than another transport or test framework.

The first is 4x2/H256 with `N=(4,0,0,0)`. Its exact CPU schedule has quota
`((0,1),(0,1),(0,1),(0,1))`, `proxy_required=(0,1,1,1)`, and three moved
copies. Source rails 1, 2, and 3 own no tokens yet each receive one direct-final
proxy record from owner zero and complete the entire B0..B5, return-unshuffle,
and legacy-epilogue path. This closes the earlier zero-token-owner gap rather
than merely testing an empty output rank with no incoming proxy work.

That same fixture runs every CUDA input creation and complete transaction on
one independently created non-default caller stream per rank. The harness
asserts the current stream identity at entry. It then reuses the same source
and world `ElasticBuffer` objects and the same arena offsets for two complete
transactions with generations 806 and 807 and distinct invocation IDs. Every
generation independently verifies the plan, active and unused proxy bytes,
world records/routes/readies, guard, complete reduce buffer, and final output,
then performs the idempotent abort cleanup. No ready, slot, or guard state
leaks across generations.

The second fixture is a harness-local 2x4/K2/C1/M1/Pcap1 global-zero case with
`N=(0,0)`. Counts, quota, proxy requirements, routes, and contributions are
all exactly empty, but the test still executes prepare, all six fixed vnode
stages, return-unshuffle, and the legacy epilogue. This proves the forced
zero-work transaction exits every barrier and does not rely on bypassing the
protocol.

Accepted implementation evidence before final independent audit:

```text
PASS py_compile and git diff --check
PASS 7/7 direct CPU oracle cases
PASS 7/7 separate eight-GPU watchdog cases
PASS original 4x2/2x4 H256 and H7168 plus BF16 rounding case
PASS empty-egress H256, non-default stream, two arena generations
PASS all-zero H256 complete fixed-stage loop
SHA256 1bc2ca73cf19e44e59388eecac5b28cbcbc0a7346f1c5d944c84f7c2e853cef1
```

Unrelated MMUnlearner/vLLM jobs occupied parts of GPUs 4 through 7 during
these runs. They were not Megatron and were not killed. The non-OOM results
are accepted as correctness evidence only; no timing or bandwidth number is
recorded.

One documentation-only `apply_patch` attempt used a line-wrapped context that
did not match `OPTIMIZATION_LOG.md`; it changed no file and was immediately
retried with the exact context. The successful entry records vnode as
disposable validation scaffolding, with an explicit post-Gin deletion audit.

Independent final review reports Blocker 0, High 0, Medium 1, and Low 2 and
accepts the checkpoint. It independently repeated the seven CPU cases and the
empty-egress/non-default/two-generation GPU case. The Medium records that this
is a non-default-stream compatibility smoke, not a strict asynchronous
ordering proof: input copies and private snapshots contain synchronization.
The real Hybrid caller therefore still needs a delayed producer/event test
without a global synchronization point. The two Low findings are that both
reuse generations intentionally use identical payload bytes, so unchanged
active source slots cannot prove a physical rewrite, and that a one-rank
constructor/setup failure still relies on the outer watchdog. Neither finding
is a numerical error or a reason to expand the vnode scaffold before force
dispatch codegen.

## 2026-07-21 — D041: compile the isolated force Hybrid dispatch

C080-E now has a force-only Hybrid dispatch implementation and JIT runtime in
new files/paths; the legacy `hybrid_dispatch.cuh`, `dispatch.hpp`, and
`combine.hpp` remain byte-identical. The new kernel retains the original
notify, local-destination bypass, destination forwarding, Tag0/Tag1 barriers,
and copy-epilogue contract. Its source-side difference is deliberately narrow:

```text
owner token scan:
    remote ordinal < retained[g,c,d] -> legacy owner Gin put at slot ordinal
    otherwise                         -> no owner put; source shuffle owns it

egress grouped scan:
    p = group_prefix[g,c,d] + u
    remote slot = retained[g,c,d] + u
    Gin put proxy_dispatch[p]

after both scans:
    publish one packed final tail = retained + moved
```

The local destination never enters this plan and still uses the original TMA
bypass. There is no descriptor, ready word, queue, ring, per-put atomic, or
success-path status check. `proxy_required` remains exactly one ABI declaration
for the eventual transaction/counter surface, but the committed post-Tag0
kernel does not read it. Gate2 owns validation; adding a device assert, trap,
data-dependent return, or clamp after Tag0 would be a liveness bug.

At destination ingress, the forwarder derives original owner-local rank from
`src_token_global_idx`, compares it with the ingress `scaleup_rank_idx`, and
snapshots `linked_list_idx[0]` only for a moved record before the existing
linked-list overwrite. Force forward metadata is exactly
`[src,last,p,src_scaleup[K],dst_slot[K]]`, or `3+2K` integers. Cached and FP8
specializations are compile-time rejected; the first force specialization is
BF16, non-cached, multiple-reduction, expert alignment one.

The private compile probe uses production-shaped values: 64 SMs, four
scaleout and four forward warps per SM, 384 threads, C=256, M=8192, E=256,
and nine QPs. Both force and legacy baselines use independent LaunchRuntime
types with recursive include hashes; the probe never initializes the legacy
`DispatchRuntime` static hash with synthetic geometry. Fresh-cache SM90a
results were:

```text
case                         force                 legacy
G8 D2 H7168 K8              REG64 STACK96 S0/L0   REG68 STACK96 S0/L0
G4 D2 H7168 K4              REG64 STACK96 S0/L0   REG68 STACK96 S0/L0
G2 D4 H256  K4              REG69 STACK96 S0/L0   REG67 STACK96 S0/L0
G2 D4 H7168 K2 (D>K)        REG64 STACK96 S0/L0   REG67 STACK96 S0/L0
```

All eight cubins passed `EP_JIT_PTXAS_CHECK=1`; PTX and SASS were emitted in
fresh cache `/tmp/deepep-c080-e-8rail.2Hhgdd`. The H256 force kernel
uses two more registers than legacy, while both retain the same 384-thread,
full-dynamic-smem one-block launch class and zero spill. This small codegen
cost is retained rather than hidden or optimized before measurement.

The first fresh H7168 attempt compiled successfully at REG64/SPILL0, then the
loader reported `CUDA_ERROR_INVALID_CONTEXT`: a cold test process had not
created its current CUDA driver context. The test now explicitly selects the
device and allocates one CUDA scalar before calling the C++ probe; the same
cubin then loaded and passed. A later audit found the first legacy comparison
called the compiler directly, so its key did not include the recursive include
hash. It was replaced by a dedicated probe LaunchRuntime and the entire matrix
was rebuilt from a fresh cache.

Each valid codegen case is launched in a separate process group by the Python
harness. Its internal watchdog terminates the whole NVCC/cuobjdump tree with
TERM followed by KILL on timeout, so compiler hangs cannot leave an orphan to
race a later cache rename.

Accepted local evidence is main extension build PASS, codegen matrix 4/4,
constraint rejection 6/6, static retained/moved/remote-slot/tail ordering,
exact two payload-put sites, plan-derived dispatch-payload counter identity
`payload_puts=sum(retained)+sum(moved)` and
`payload_gin_bytes=payload_puts*dispatch_token_bytes`, post-Tag0 control-flow
parity with legacy, API 6/6, recursive legacy goldens 4/4, and diff check PASS.
This is
`HYBRID_CODEGEN_PASS`, not Hybrid runtime evidence: the single-node machine
still cannot execute truthful Rail/Gin peers, and public force remains disabled.

The primary thread then repeated the acceptance independently after the source
was frozen. The in-place extension build, API 6/6, and recursive legacy
goldens 4/4 passed. A second empty cache,
`/tmp/deepep-c080-e-root-final.zaH9ua`, rebuilt all four force/legacy pairs
under the harness-owned process-group watchdog with PTX/SASS dumps and the
same register/stack/spill results. Final read-only review reports Blocker 0,
High 0, and Medium implementation defects 0. It keeps two environment/host
boundaries explicit: this machine cannot launch truthful Rail/Gin peers, and
the production WORLD transaction, force buffer/metadata ownership, and
combine path are not connected yet.

## 2026-07-21 — D042: compile the isolated force Hybrid combine

C080-F now has a separate BF16, non-expanded, multiple-reduction Hybrid
combine specialization. The immutable legacy `hybrid_combine.cuh` and
`combine.hpp` remain unchanged. Force dispatch carries one additional transit
integer `p`, so the replay metadata is exactly
`[src,last,p,src_scaleup[K],dst_slot[K]]` (`3+2K`). A retained record keeps the
legacy owner/token destination. A moved record changes only the remote receive
pointer to the symmetric source-egress address
`proxy_return_base + p * combine_token_bytes`; reduction, TMA staging,
aggregated Gin issue, final flush, rail completion, and tail protocol remain
legacy behavior.

The sentinel row initializes only `src`. An early draft read `p` and the 2K
replay fields before checking `src < 0`; review caught this sanitizer-visible
uninitialized read before acceptance. The uniform sentinel break now precedes
every force-only metadata read. A second review found that `WorkspaceLayout`
could assert while being constructed if rank/expert geometry exceeded its
fixed maxima. Both force dispatch and combine now reject total ranks, total
experts, and experts per rank on the host before JIT/launch and repeat those
limits as static kernel gates before either workspace layout construction.
The total-expert rejection was made independent with `D=2,G=32,E=2112`
(33 experts per rank), while `D=2,G=2,E=2048` independently exercises the
per-rank ceiling. Dispatch and combine constraint matrices both pass 8/8.

The final build and compile commands were:

```text
TORCH_CUDA_ARCH_LIST=9.0 PYTHONPATH=. \
  /home/chen/.cache/deepep-sjlgpt/bin/python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=0 \
EP_JIT_CACHE_DIR=/tmp/deepep-c080f-combine-final.eSgcac \
EP_JIT_PTXAS_VERBOSE=1 EP_JIT_PTXAS_CHECK=1 \
EP_JIT_DUMP_PTX=1 EP_JIT_DUMP_SASS=1 PYTHONPATH=. \
  /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_hybrid_combine_codegen.py \
  --watchdog-seconds 900
```

The extension build passed. Six force and six independently cached legacy
cubins, PTX files, and SASS files were emitted. The first four cases cover the
complete `(D<=K,G<=K)` truth table; the remaining two match dispatch codegen
geometries. Every shape reports `REG216 / STACK96 / SHARED1024 / LOCAL0` and
zero spill for both force and legacy. Force uses 860 bytes of constant bank 0
versus legacy 852, exactly the added eight-byte proxy-base ABI argument; no
register, stack, shared-memory, local-memory, or spill regression is hidden.
API 6/6, recursive legacy goldens 4/4, exact force-to-legacy normalization,
post-Tag0 control checks, and `git diff --check` pass. Independent final review
reports Blocker 0, High 0, and Medium 0.

Failures and corrections are retained:

- a nonexistent `/home/chen/miniconda3/envs/py311_cuda129/bin/python` and then
  the incompatible generic sjlgpt environment were tried for syntax/import;
  only `/home/chen/.cache/deepep-sjlgpt/bin/python` is accepted evidence;
- the first exact normalizer differed from legacy by one blank line, and the
  immutable legacy root contains two whitespace-only lines; the new file stays
  diff-clean and the test restores those bytes only for the exact golden;
- the initial layout-coverage assertion was tautological and was replaced by
  explicit TT/FT/TF/FF expectations plus a complete-set assertion;
- an include-cache guard committed with an over-broad substring also matched a
  comment and failed one cached smoke; commit `2ec61c6` replaced it with a
  token-boundary regex and the same H7168/K8 case passed;
- one attempted nested audit worker hit the environment thread limit; the
  primary and independent audit agents instead inspected the complete 12-cubin
  cache and repeated the critical cold cases.

Commits `9f22083`, `8abc90e`, and `2ec61c6` are pushed to the fork. This is
`HYBRID_CODEGEN_PASS`, not a truthful multi-node Rail/Gin runtime result. The
public force capability remains false until the production WORLD transaction,
owning handle state, symmetric arena offsets, and uninterrupted committed
dispatch/combine sequences are connected.

## 2026-07-21 — D043: freeze prepared Hybrid launch adapters

The first host-integration slice adds no public path. Force main dispatch and
combine now prepare and own their compiled runtime, specialization spec, and
launch geometry. Their `launch_prepared_*` adapters only bind invocation raw
pointers/counts and submit to the caller stream. Source gates reject validation,
JIT, allocation, tensor `.data_ptr()`, D2H reads, status checks, and stream or
device synchronization in these committed adapter bodies.

Force combine additionally freezes the legacy reduce-buffer offset during
prepare with checked 64-bit arithmetic:

```text
rows              = min(G, K)
tokens per row    = D * M
reduce offset     = rows * tokens_per_row * combine_token_bytes
legacy reduce ptr = legacy_buffer_base + reduce_offset
```

The final pointer helper performs only the last `advance_ptr`; its input is the
legacy Hybrid buffer base, never workspace or the force arena. The proxy return
base remains the separately checked `arena + proxy_return_offset` inside the
same registered symmetric window.

The ordinary dispatch copy epilogue was also converted into a private prepared
wrapper. Its force-v1 specialization is the unchanged BF16, non-cached,
non-expanded, no-SF, no-zero-padding, alignment-one production kernel. It uses
the same `DispatchCopyEpilogueRuntime::generate` source, the same
`dispatch_copy_epilogue` cache key, the same TokenLayout/warp formula, and the
same PDL launch geometry. The actual legacy callsite gives this epilogue every
physical SM, independently of the main dispatch SM count; prepare therefore
requires exactly the physical SM count and freezes that callsite contract.

Accepted evidence:

```text
PASS C080-H1 prepared dispatch-copy epilogue adapter
PASS 6/6 C080-A Hybrid API tests
PASS 4/4 legacy Hybrid identity goldens
PASS dispatch G8/D2/H7168/K8 fresh force+legacy codegen
PASS combine D4/G2/H7168/K2 fresh force+legacy codegen
```

Fresh cache `/tmp/deepep-c080-h1-root.OcEXpf` contains four cubins plus PTX and
SASS. Dispatch force/legacy remain REG64/68, STACK96, zero spill. Combine
force/legacy remain REG216, STACK96, zero spill. Both codegen constraint
matrices, the full six-case combine adapter run, Python compilation,
`git diff --check`, extension builds, and an independent source/ABI audit pass.
The final audit reports Blocker 0 and High 0. Immutable legacy dispatch,
combine, and their CUDA roots retain their recorded SHA values.

Failures and corrections retained in this slice:

- the first review incorrectly inferred that dispatch copy epilogue follows
  the main dispatch's selected SM count; `buffer.hpp` proves it always receives
  `device_runtime->get_num_sms()`. Allowing a partial-SM specialization under
  the legacy cache key was classified High and corrected before commit;
- one syntax command used a nonexistent Python environment and another import
  used the ABI-incompatible generic sjlgpt environment; accepted commands use
  `/home/chen/.cache/deepep-sjlgpt/bin/python` only;
- one API invocation omitted `PYTHONPATH=.` and failed before testing code; it
  was rerun with the pinned environment and passed 6/6;
- `setup.py --force` still let Ninja report no work after agent builds had
  already refreshed the dependency object. Acceptance therefore also includes
  the agents' successful header-triggered extension builds and independent
  fresh JIT codegen, not an unsupported claim that `--force` rebuilt every
  translation unit.

Commits `7a5bd9c`, `5b9df10`, and `e2ee1fa` are pushed. H1 is adapter closure,
not the production transaction. Before H4 can commit a collective epoch, the
source shuffle and return-unshuffle need frozen raw-pointer submit layers, and
the existing combine epilogue's launch-time assertions must move into
precommit validation. Public force remains false.

## 2026-07-21 — D044: close the H1b raw-submit boundary

H1b completes the launch boundary required before the production WORLD
transaction can be connected. Source shuffle now prepares and owns its JIT
runtime, H/K/C/N specialization, TokenLayout shared-memory size, active grid,
and LaunchArgs. Return-unshuffle similarly freezes H/K/C, shared memory, grid,
and runtime. Their raw submit functions accept only previously captured
pointers and validated dynamic topology values, construct the runtime Args,
and launch on the existing communication stream. Checked wrappers remain for
the private LSA tests and unwrap owning plan tensors before entering the raw
path.

The combine reduction epilogue now has the same split: its checked private test
wrapper retains all six host checks, while the committed submit is only an
Args bind plus launch. The existing local LSA barrier was audited and already
had that property, so no redundant wrapper was added. The return-unshuffle
private transaction captures the arena, plan, seed, and snapshot pointers
before B1; B1 through B4 contain only the two prebound D2D copies, prepared
barriers, and the raw unshuffle submit. Status readback and synchronization
remain after B4.

Independent review found one High before acceptance: source prepare froze an
active grid from N, but the first raw ABI also accepted a second dynamic N. A
mismatch could make the kernel loop bound exceed the frozen grid and omit
tokens. The dynamic argument was removed; both grid and kernel N now come only
from `prepared.spec.num_tokens`. A persistent source adapter gate freezes this
invariant. The same review found one Low wording mismatch around the one-shot
state bit; the comment now accurately states that replay is disabled before
either copy is queued. Final review is Blocker 0 / High 0 / Medium 0 / Low 0.

Accepted evidence on the final tree:

```text
PASS true header-triggered python_api.o rebuild, device link, and shared link
PASS C080-H1b prepared source-shuffle raw submit
PASS C080-H1 prepared return-unshuffle raw submit
PASS C080-H1b prepared epilogue/barrier submit adapters
PASS 6/6 C080-A Hybrid API tests
PASS 4/4 legacy Hybrid identity goldens
PASS true 8-GPU source LSA, 21 moved copies, non-default stream
PASS true 8-GPU return-unshuffle C061 H256 and abort/recovery
PASS py_compile and git diff --check
```

Failures and corrections retained:

- the first source GPU command selected `c061_source_shuffle_seed62`; that
  name occurs twice in the existing source fixture table, so the selector's
  `len == 1` assertion failed before any GPU stage. The uniquely named
  `c061_source_shuffle_reuse_seed63` case then passed the real eight-GPU path;
- one return-unshuffle `--help` command omitted `PYTHONPATH=.` and failed at
  import. It was rerun with the pinned environment before the real GPU case;
- the first main-thread build after agent builds reported Ninja no work and
  only relinked. After the N/grid correction changed the header, the accepted
  build visibly recompiled `csrc/python_api.o`, device-linked, and copied the
  extension;
- the N/grid High and the one-shot comment Low were fixed before commit rather
  than waived.

Commits `6a6dc02` and `d5a60a9` are pushed to the fork. They do not modify a
CUDA hot loop, public operator, Python dispatch/combine, capability bit, or
legacy JIT identity. H2 is the minimal force-only constructor sizing/ownership
slice; H3/H4 still own the pre-window/fixed-tensor WORLD decisions and the
actual uninterrupted dispatch/combine transaction.

## 2026-07-21 — D045: own the force arena as a legacy buffer tail

H2 connects the existing checked arena calculator to construction without
changing the C++ runtime signature or adding a second allocation. Only after
the host and compiled capability gates both report true, force construction
now performs:

```text
legacy_bytes = calculate_elastic_buffer_size(..., BF16, Hybrid, multi-reduce)
arena_bytes  = HybridArenaLayout(H, K, Pcap).arena_bytes
total_bytes  = calculate_rail_balance_hybrid_buffer_size(..., Pcap)
require total_bytes == legacy_bytes + arena_bytes
arena_offset = legacy_bytes
```

The unchanged `ElasticBuffer` runtime receives `total_bytes`, so NCCL registers
one symmetric window containing the original DeepEP buffer followed by the
2 MiB-aligned force tail. Four private force-only scalar fields retain mode,
Pcap, arena offset, and arena bytes for H4. They are published only after the
three formulas agree. The normal off branch does not call either force helper,
does not gain a field, and passes the exact original runtime argument tuple.
The public size hint, public OP set, runtime signature, and both capability
defaults remain unchanged.

Accepted evidence:

```text
PASS 9/9 C080-A Hybrid API tests
PASS 5/5 C080-A Hybrid layout tests
PASS distributed Hybrid buffer formula on eight H200 ranks
PASS 4/4 legacy Hybrid identity goldens
PASS py_compile and git diff --check
PASS independent H2 audit: Blocker/High/Medium/Low = 0/0/0/0
```

The API tests monkeypatch both capabilities true only inside the force fixture.
They freeze helper order, exact arguments, complete runtime tuple, one barrier,
and the four-field ownership set. A stale total and exceptions from each of the
legacy/layout/total helpers all fail before runtime construction. The off test
installs tripwires on both force helpers and rejects any `_rail_balance_*`
instance field.

One command failure is retained: the main thread initially passed unsupported
`--oracle-only` to `test_rail_balance_hybrid_layout.py`; argparse exited before
testing. The corrected `--num-processes 8` command passed all five layout cases
and the distributed formula.

Commit `c05ff0d` is pushed. Public force remains unavailable. H3 must settle
the constructor pre-window consensus boundary and implement fixed CUDA tensor
WORLD gates; H4 then connects the prepared raw stages and owning handle state.

## 2026-07-21 — D046: prove one fixed CUDA WORLD consensus primitive

H3a adds a private fixed-storage WORLD gate without connecting it to the
constructor or operation path. The caller owns one reusable CUDA `int64[128]`
and one pinned CPU mirror. Word zero carries a deterministic nonnegative error
key, word one is reserved zero, and up to 63 common fields occupy exact pairs:

```text
word[2 + 2*i] = value
word[3 + 2*i] = -value
```

One signed-int64 MAX all-reduce therefore produces `max(value)` and
`-min(value)` without a hash or second collective. The decoder returns the
first unequal field. A nonzero error key wins before manifest mismatch; its
high 31 bits select priority and its low rank complement selects the lowest
rank within that priority. The largest legal key is exactly INT64_MAX.

The raw runner is intentionally short:

```text
pinned H2D -> one MAX all_reduce -> pinned D2H
           -> synchronize caller current stream -> CPU decode
```

Storage validation and checked encoding are separate preparation functions.
The runner has no tensor allocation, object/gather collective, rank-local
shape assertion, JIT, or whole-device synchronization. Public API, constructor,
off mode, capability bits, and dispatch/combine remain unchanged.

Accepted evidence:

```text
PASS C080-H3 fixed WORLD gate CPU reference
PASS true EP8 equal/mismatch/error-first and lowest-rank arbitration
PASS variable local N omitted from common manifest
PASS stable CUDA/pinned data_ptr over repeated rounds
PASS non-default caller stream
PASS exactly one int64[128] MAX all_reduce per round
PASS gather/object-collective and cuda-device-sync tripwires
PASS API 9/9 and legacy Hybrid identity 4/4
PASS py_compile and git diff --check
PASS independent audit plus 100 random/INT64_MAX properties: 0/0/0/0
```

Two design hazards were found before production connection:

- pre-window mixed off/force detection and a promise that off participates in
  no new constructor collective are logically incompatible because off and
  force register different byte counts and `ncclCommWindowRegister` is itself
  collective. Capability remains false. Final activation must either make all
  modes pay one fixed pre-window gate or leave force unavailable; it must not
  use window registration as a mismatch probe;
- the old private Gloo signature includes `local_scaleout_rank` in WORLD-equal
  data. That happens to work in the local fixture but is wrong on real nodes,
  where source servers legitimately have different scaleout ranks. Production
  manifests compare D/G/world and validate each local rank equation separately.
  Local N and local scaleout/scaleup indices are never WORLD-equal fields.

Commit `63379bb` is pushed. H3b next reuses these same stable tensors around the
private C++ plan prepare/finish as Gate1/Gate2. H4 then consumes successful
Gate2 state in one C++ source-shuffle to force-dispatch commit.

## 2026-07-21 — D047: lunch-pause archive before H3b lands

The user requested a recoverable pause while H3b was still being designed.
All child work was stopped before exit. The proposed single new test file,
`tests/elastic/test_rail_balance_hybrid_plan_world_gate.py`, had not been
written to the shared worktree, so no incomplete test was staged or committed.
At the pause boundary the branch and fork were identical at `dc2b42c`, and the
worktree was clean.

Immediately before the pause, the accepted H3a gate was rerun on all eight idle
H200 GPUs:

```text
PYTHONPATH=. /home/chen/.cache/deepep-sjlgpt/bin/python -B \
  tests/elastic/test_rail_balance_hybrid_world_gate.py \
  --num-processes 8 --master-port 29951 --watchdog-seconds 300

PASS C080-H3 fixed WORLD gate CPU reference
PASS C080-H3 fixed WORLD gate: equal/mismatch/error-first, variable N,
     stable storage, non-default stream, one MAX/round
```

The interrupted H3b task was approximately 35% complete in design only. Its
reviewed restart contract is:

- wrap the real private C++ planner in
  `prepare -> WORLD Gate1 -> finish -> WORLD Gate2`;
- use the NCCL EP group for each fixed CUDA gate; Gloo may only supervise the
  process watchdog and final error reporting;
- keep one CUDA `int64[128]` tensor and one pinned mirror for every round, and
  prove exactly one MAX per gate;
- omit local token count and local scaleout/scaleup identities from the WORLD
  equality manifest;
- cover variable/zero local N, a one-rank common-field mismatch, a one-rank
  invalid route returned by prepare, collective capacity failure after exactly
  one local barrier, abort/retry after both gates, and exact comparison of all
  fourteen plan outputs with the CPU oracle;
- never enter `finish` after Gate1 failure and never publish source payload
  after Gate2 failure.

The next implementation step is to create only that focused test, run its CPU
suite and true EP8 watchdog, independently audit its deadlock behavior, then
commit and push it before starting H4a. No production code is partially edited
at this checkpoint.

## 2026-07-21 — D048: close H3b around the real planner

H3b now wraps the existing private C++ planner in two real fixed-tensor NCCL
WORLD gates:

```text
checked manifest encode before prepare
-> prepare
-> raw error-word patch
-> Gate1: one int64[128] MAX
-> finish: exactly one local LSA barrier
-> raw error/phase-word patch
-> Gate2: one int64[128] MAX
-> abort or later commit boundary
```

The Gate2 patch reuses the exact pinned tensor copied back by Gate1. Since the
two manifests differ only at phase field three, it changes only word zero and
the phase pair at words eight/nine. It does not rebuild a field list or repeat
dynamic manifest validation after the local barrier. The non-default-stream
case enters one caller-stream context before prepare and leaves only after
Gate2.

The final true EP8 suite executes ten transactions and exactly eighteen MAX
collectives. It covers:

- variable/zero local N and exact comparison of all fourteen plan tensors;
- a zero-N rank whose actual `topk_idx.shape[1]` is K+1, rejected at manifest
  field ten before finish, followed by a successful retry;
- one rank returning invalid-route status two while peers own pending state,
  followed by a successful retry;
- real all-rank capacity status one after finish, followed by a successful
  retry;
- an injected one-rank Gate2 error after successful finish, proving asymmetric
  WORLD convergence, followed by a successful retry;
- C1024/D32 and non-default-stream compatibility.

Accepted evidence:

```text
PASS C080-B2 local LSA plan transaction (fresh resume baseline)
PASS C080-H3 fixed WORLD gate on EP8
PASS C080-H3b CPU contract
PASS C080-H3b real planner WORLD gates ... 18 fixed MAX gates
PASS 9/9 C080-A Hybrid API tests
PASS 4/4 legacy Hybrid identity goldens
PASS py_compile and git diff --cached --check
PASS final transaction audit: Blocker/High/Medium/Low = 0/0/0/0
PASS final minimality audit: Blocker/High/Medium = 0/0/0
```

Failures and review fixes retained:

- the first implementation delegate remained at design state without writing
  a file and was interrupted; the root implementation then created the single
  focused test and checkpointed it;
- the first EP8 run used remainder seed 71 for an intended success retry, but
  the CPU oracle correctly showed that seed needs six proxy slots while Pcap
  was five. The device returned capacity status one. The fixture now uses
  oracle-enabled seed 66 and asserts schedule intent before every transaction;
- an attempted legacy regression command named nonexistent
  `test_rail_balance_hybrid_legacy_identity.py` and exited before testing. The
  actual `test_rail_balance_hybrid_legacy_golden.py` passed 4/4;
- audit found a second stream-context entry between Gate1 and finish, checked
  list construction between finish and Gate2, and a fail-open manifest K taken
  from PlanCase instead of the actual tensor. All three were fixed before the
  accepted EP8 run;
- an initially added C mismatch duplicated the stronger actual-K mismatch. It
  was removed, restoring the planned eighteen gates rather than expanding the
  test surface.

Evidence boundaries remain explicit. The non-default case is compatibility,
not asynchronous producer-ordering proof. The direct barrier/no-payload checks
are source guards, not binary counters. H3b has not enabled public force and
does not claim real Gin/RDMA behavior.

Commits `2f39a46` and `878b0db` are pushed to the fork. H4a may now connect the
same fixed Gate1/Gate2 protocol to the force dispatch transaction while both
capability bits remain false.

## 2026-07-21 — D049: make pending-plan state transitions explicit

H4a replaces the pending plan's ambiguous `finished` bit with the smallest
state machine needed by the production transaction:

```text
Preparing -> PlanReady -> DispatchLive
                              |
                              v
                           Invalid
```

`finish` accepts only `Preparing` or an idempotent `PlanReady`. Source shuffle,
return unshuffle, combine epilogue, and snapshots require `PlanReady`; their
existing one-shot flags remain unchanged. Abort is invocation-scoped: a stale
ID is a no-op, a matching precommit transaction is released, a matching live
dispatch becomes permanently `Invalid`, and an already invalid transaction is
left unchanged. `DispatchLive` is intentionally unreachable until H4c owns the
real dispatch commit.

The H3b EP8 test now calls abort with stale invocation 8300 while invocation
8301 is pending. It still completes the same eighteen fixed MAX gates, so this
proves stale cleanup cannot erase a newer plan without adding another
collective or test-only runtime path.

Accepted evidence after rebuilding the extension:

```text
PASS C080-H3b real planner WORLD gates ... 18 fixed MAX gates
PASS C080-D true 8-GPU source shuffle, including H256/H7168 arena reuse
PASS C080-F true 8-GPU return unshuffle, one-shot rejection, abort/recovery
PASS C080-D source-shuffle faults: sticky status, capacity, duplicate, recovery
PASS prepared source and return raw-submit adapter gates
PASS six Hybrid combine codegen cases plus constraints 8/8
PASS 9/9 C080-A Hybrid API tests
PASS 4/4 legacy Hybrid identity goldens
PASS full extension build, py_compile, and git diff --check
PASS independent state audit: no blocking issue
```

No performance claim is made and no NCU/Nsys run was started. Public `force`
and both capability bits remain disabled. Commit `52b165e` is the recoverable
H4a code checkpoint; H4b next prepares the complete owning dispatch bundle
before Gate1, and H4c performs the uninterrupted raw commit and enters
`DispatchLive`.

## 2026-07-21 — D050: own the complete Hybrid dispatch prepare transaction

H4b adds one private production-shaped
`_rail_balance_hybrid_dispatch_prepare` without connecting public force.  The
method derives physical D/G/local identities, H/K, and the exact legacy Hybrid
channel formula inside C++; checks both legacy layouts against `arena_offset`;
allocates the non-cached BF16 handle tensors with force metadata width `3+2K`;
and prebuilds plan, barrier, source shuffle, main dispatch, dispatch epilogue,
main combine, return unshuffle, and combine epilogue before WORLD Gate1.

The pending object owns every Tensor and prepared runtime whose address will
survive the gate.  All fourteen plan pointers, all LSA peer count pointers,
the main dispatch ABI pointers, and LaunchArgs are frozen before the first
arena count launch.  `finish` now consumes those frozen values rather than
calling `data_ptr`, `get_sym_ptr`, or JIT preparation after Gate1.  H4b stops
at ownership: it launches neither source shuffle nor main Hybrid dispatch and
cannot enter `DispatchLive`.

An independent audit found one precommit lifetime window: if count had been
queued and its following D2H enqueue failed, abort could release backing
tensors while the comm stream still referenced them.  The accepted failure-
path fix synchronizes `comm_stream`, resets ownership even when CUDA reports an
asynchronous error, and only then rethrows.  This adds no success-path work.
The focused audit and re-audit close Blocker/High at 0/0.

Accepted evidence after rebuilding the extension:

```text
PASS H4b CPU/source ownership and ordering contract
PASS H4b truthful D=1 fail-close plus exact H3b recovery on EP8: 3 MAX gates
PASS canonical H3b planner WORLD suite on EP8: 18 MAX gates
PASS B1 GPU materializer: 70 cases
PASS B2 local LSA planner transaction
PASS vnode 4x2 H256 and H7168 round trips
PASS H3a fixed WORLD gate and source/return raw-submit adapters
PASS prepared epilogue/barrier adapter
PASS API 9/9 and legacy Hybrid identity 4/4
PASS one production-shaped 4x2/H256 force dispatch codegen case
PASS full extension build, py_compile, and git diff --check
PASS independent H4b audit and lifetime-fix re-audit: Blocker/High = 0/0
```

Failures and review fixes retained:

- the first compile used `int*` for two native mapped-host `int64_t*`
  counters; the types were corrected and the next full build passed;
- H3b and the epilogue adapter initially matched the old checked barrier helper
  name.  Their source contracts now separately require one frozen raw submit
  and one checked-wrapper delegation; CPU and EP8 reruns pass;
- one H3a regression command included its unsupported `--timeout` option and
  exited before testing; the corrected command passed;
- the abort lifetime issue above was found after the first green EP8 run,
  fixed, rebuilt, and followed by fresh H4b three-gate and H3b eighteen-gate
  runs.

This machine's truthful physical Hybrid topology is D=1, so it can prove only
production prepare fail-close, state cleanup, shared-core/vnode behavior, and
synthetic D>1 code generation.  A successful production prepare and all
Gin/RDMA claims remain multi-node gates.  H4c next performs only the adjacent
source-shuffle/main-dispatch commit.  It must poison state before submission,
account for the still-fallible runtime launch-config construction, and keep the
expert prefix storage base distinct from the non-expanded epilogue's
inclusive `base+1` view.

Implementation commit `4e48b84` and gate commit `7ab382b` are pushed to
`fork/feat/rail-balance-prototype`.

## 2026-07-21 — D051: commit source shuffle directly into Hybrid dispatch

H4c adds the smallest private post-Gate2 commit.  All invocation/state/status,
runtime-owner, and device checks plus `CUDAGuard` execute while the transaction
is still `PlanReady`.  The method then snapshots only prepared references,
raw pointers, NCCL handles, and POD geometry.  Its committed interval is:

```text
state = Invalid
submit prepared source shuffle on comm_stream
launch prepared force Hybrid dispatch on comm_stream
shuffled = true
state = DispatchLive
```

There is no branch, Tensor accessor, JIT, allocation, peer lookup, barrier,
D2H read, event, or synchronization in that interval.  Poisoning first makes
either synchronous launch failure non-replayable.  Success means both kernels
were accepted by the host stream; it does not claim GPU, Gin, or remote
completion.

The expert-prefix address was also made explicit before Gate1.  Main dispatch
receives `psum_num_recv_tokens_per_expert_storage` at the allocation base,
while H5's non-expanded epilogue will consume the separately frozen
`psum_num_recv_tokens_per_expert_inclusive = base + 1`.  H5 is forbidden from
recovering that view through a post-gate Tensor accessor.

Accepted evidence:

```text
PASS H4c source/ordering/raw-ABI contract
PASS full extension rebuild after the H4c C++ change
PASS production-shaped G8xD2/H7168/K8 force dispatch codegen
PASS true EP8 H7168 direct-final source LSA shuffle
PASS vnode 4x2/H7168 full dispatch/expert/combine round trip
PASS H4b truthful D=1 fail-close/recovery: 3 fixed MAX gates
PASS canonical H3b planner transaction: 18 fixed MAX gates
PASS source and epilogue raw-submit adapter gates
PASS API 9/9 and legacy Hybrid identity 4/4
PASS py_compile and git diff --check
PASS independent H4c audit: Blocker = 0
```

Failures, resource actions, and audit boundaries retained:

- four positively identified `megatron-lm-gpu/bin/python` processes occupied
  GPUs 0-3 at roughly 36 GiB each.  They were terminated under the user's
  standing authorization; all eight GPUs returned to zero MiB before tests;
- one direct H4b CPU invocation omitted the required checkout `PYTHONPATH` and
  failed with `ModuleNotFoundError`; the pinned corrected command passed;
- H3b's source guard originally delimited `finish` by the later source-test
  method.  Inserting H4c between them made the static test see H4c's submit and
  fail before GPU spawn.  Its endpoint now names H4c itself; CPU and eighteen-
  gate EP8 reruns pass;
- generic `LaunchRuntime::launch` still constructs a config and performs a
  checked kernel launch after state is Invalid.  A one-rank source-success/
  main-launch failure can strand peers at Tag0 until the job timeout; this is a
  job-fatal boundary, not recoverable abort.  An overstrong header comment was
  corrected;
- source device status is intentionally not synchronized before main dispatch.
  Gate2 inputs/plan must remain immutable through commit, and H5's first safe
  synchronization must turn any asynchronous source/main failure into a
  permanently poisoned handle.

No virtual production topology, empty Gin launch, loopback main kernel, public
force call, epilogue, or combine path was added.  Local evidence labels remain
`H4C_COMMIT_SOURCE_PASS`, `HYBRID_CODEGEN_PASS`, and
`VNODE_FUNCTIONAL_PASS`; real Hybrid runtime remains untested.  Implementation
commit `edb1a0e` and gate commit `ba82134` are pushed to the fork.  H5 next
owns dispatch CPU-count completion, exact receive allocation, dispatch
epilogue, handle publication, and the symmetric combine lifecycle.

## 2026-07-21 — D052: own and finish the committed Hybrid dispatch

H5a adds the private `_rail_balance_hybrid_dispatch_finish` boundary without
opening public force or touching combine.  The main dispatch has already been
published by H4c, so only invocation/state/owner checks execute while the
transaction is retryable.  Finish then poisons the state, reads the mapped
rank/expert counters frozen before Gate1, allocates the exact receive shape,
installs `RailBalanceHybridDispatchCompletion`, and launches the prebuilt
native BF16 non-expanded copy epilogue.  The epilogue receives expert prefix
`storage + 1`; main dispatch continues to own the allocation base.

The first safe post-H4c status read is deliberately after the epilogue on the
same `comm_stream`.  One D2H status copy and one host stream synchronization
therefore expose asynchronous source, main-dispatch, and epilogue failures
before the method restores `DispatchLive`.  Force-v1 is fixed synchronous for
correctness and returns `event=None`; removing that sync belongs to measured
C100 work, not H5a.  The returned tuple is exactly the native sixteen-item
dispatch ABI, including the inclusive expert prefix, K+2 source metadata,
3+2K forward metadata, and linked-list handle fields.

Independent review found that the first draft narrowed mapped int64 counters
and accumulated in signed int before the safe status sync.  A ready-looking
corrupt/partial value could therefore overflow or request an invalid large
allocation before the device error was observed.  The final code uses int64
decode/accumulation, checks each addition against `D*G*M` and
`D*G*M*min(K,local_experts)`, and narrows only after all totals are bounded.
This work is host-only and adds no CUDA hot-path instruction.

Accepted evidence:

```text
PASS full extension rebuild and pybind finish symbol
PASS C080-H5a exact ownership/order/16-item source contract
PASS C080-H4c adjacent-commit source contract
PASS C080-H4b owning-prepare CPU contract
PASS C080-H4b truthful D1 fail-close/recovery on EP8: 3 fixed MAX gates
PASS C080-D true eight-GPU source-shuffle regression
PASS C080-D 4x2/H7168 vnode full round trip
PASS C080-F true eight-GPU return-unshuffle, H256/H7168 and D32>K4
PASS C080-E G8xD2/H7168/K8 dispatch codegen
PASS API 9/9 and legacy Hybrid identity 4/4
PASS independent H5a ABI/lifetime audit: no blocker
```

Failures and resource actions retained:

- the first H5a source-gate run looked for the result alias in the pending
  header although the `EventHandle`-bearing host alias correctly lives in
  `buffer.hpp`; a second run matched the completion raw-pointer prefix, and a
  third assumed the balanced-section helper included the return type.  These
  were test-only format defects; each was corrected before the gate passed;
- review then found two genuine gate blind spots: adjacent Tensor result items
  and adjacent pointer epilogue arguments could be interchanged while a type-
  only/contains test stayed green.  The final gate freezes all sixteen result
  expressions and all fourteen epilogue arguments in exact order;
- one initial review suggestion placed `CUDAGuard` before poison.  Rechecking
  the already-published H4c boundary showed that only pure caller validation
  is retryable; guard/device/allocation failure must remain fail-closed.  The
  accepted order is `Invalid` then guard;
- four positively identified `megatron-lm-gpu/bin/python` processes, PIDs
  3697293-3697296, each held about 36 GiB on GPUs 0-3.  They were terminated
  under the user's standing authorization before EP8 tests; all GPU process
  allocations cleared;
- the first vnode subprocess completed while its polling session handle was
  already closed, so its exit status was not accepted as evidence.  The same
  4x2/H7168 case was rerun and printed an explicit PASS with exit code zero.

Implementation commit `81ecb86` and source-gate commit `7bd17a4` are the H5a
checkpoint.  H5b next owns buffer/invocation-bound one-shot handle publication
and the uninterrupted main-combine → return-unshuffle → local barrier → native
epilogue transaction.  Public force remains disabled and no local result
claims a truthful D>1 Gin runtime.

## 2026-07-21 — D053: close the private one-shot Hybrid combine transaction

H5b adds three private C++ boundaries without opening the public force path:
`_rail_balance_hybrid_combine_prepare`, `_abort`, and `_commit`. Prepare checks
the live H5a owner and every prebuilt stage/raw dependency, requires the
force-v1 BF16 expert payload plus FP32 top-k weights, allocates exact
owner-token outputs, freezes their pointers, orders compute before comm, and
installs the completion owner as its final operation. A future WORLD-gate
failure can call the idempotent abort; it releases only that new owner and
leaves the older synchronized dispatch handle retryable.

Commit captures all references, pointers, handles, and POD geometry while the
transaction is still live. Its committed interval contains exactly:

```text
state = Invalid
main Hybrid combine
return-unshuffle
local LSA barrier
native non-expanded combine epilogue
```

All four submissions use the same `comm_stream` with no intervening allocation,
JIT, Tensor accessor, synchronization, or branch. The epilogue replays H5a's
owned `copied_topk_idx`, never the caller's mutable original. One status D2H
and one stream synchronization observe the complete chain. Only success first
constructs the native three-item result and then resets the whole pending
transaction; any post-submit failure retains `Invalid` plus every owner.

Accepted evidence:

```text
PASS full extension rebuild and all three pybind symbols
PASS H5b exact owner/prepare/abort/four-call commit source contract
PASS H5a, H4c, and H4b source contracts
PASS prepared epilogue/barrier and return-unshuffle adapter regressions
PASS production-shaped G8xD2/H7168/K8 combine codegen
PASS truthful EP8 D1 H4b fail-close/recovery: 3 fixed MAX gates
PASS API 9/9 and legacy Hybrid identity 4/4
PASS independent H5b ABI/lifetime audit: no C++ blocker
PASS git diff --check
```

Failures, fixes, and resource decisions retained:

- the first prepare draft accepted missing top-k weights although force-v1's
  combine specialization requires FP32 weights; prepare now rejects `None`;
- the first epilogue draft replayed the mutable original top-k tensor instead
  of H5a's owned copy; the raw ABI and source gate now freeze
  `copied_topk_idx`;
- the first abort draft asserted that every rank had installed an owner. A
  rank can fail prepare earlier than its peers, so abort is now stale-safe and
  idempotent while preserving `DispatchLive`;
- the first focused codegen command used the nonexistent shorthand case name
  `8x2_h7168_k8` and exited in argparse. The canonical
  `8x2_h7168_k8_rank_tt` case passed;
- positively identified Megatron PIDs 3806509-3806512, each holding roughly
  36 GiB, were terminated under standing authorization. The unrelated
  `/home/w00809645/repos/CUDA_Kernel_Samples/elementwise/add` process was left
  running because it uses only about 528 MiB and is not in the authorized
  Megatron scope.

Implementation commit `ffe4fd4` and source-gate commit `403e725` form the H5b
checkpoint. This is private host/C++ closure, not public `EPHandle` closure:
Python buffer identity, cached-handle rejection, consumed-state atomicity,
constructor/dispatch WORLD consensus, and capability activation remain next.
No local result claims a truthful D>1 Gin/RDMA run.

## 2026-07-21 — D054: freeze the minimal public force lifecycle

The public H6 slice is deliberately three small changes rather than a new
transaction framework:

1. one pre-window constructor consensus shared by off and force, with the
   existing fixed force gate storage retained only by force;
2. an early force branch in `dispatch` that executes prepare → WORLD Gate1 →
   finish-plan → WORLD Gate2 → adjacent commit/finish, then publishes an
   ordinary `EPHandle` with private owner/invocation/state fields;
3. an early force branch in `combine` that validates the exact live ticket,
   prepares all output ownership, executes one WORLD gate, marks the ticket
   consumed, and invokes H5b commit.

The legacy `dispatch`, `_unpack_handle`, `combine`, default `EPHandle`
constructor, runtime ABI, buffer bytes, and JIT roots remain untouched when
off. Force rejects cached dispatch, expansion, FP8, masking, missing weights,
events/async allocation, bias, deterministic mode, and non-unit alignment.
Every force attempt reserves a monotonic invocation before local validation so
rank-local errors still enter the same fixed collective.

Failure ownership is frozen as follows:

- dispatch prepare/Gate1 or plan/Gate2 rejection invokes the stale-safe plan
  abort and is retryable; a Busy attempt cannot destroy the older live handle;
- any dispatch failure after Gate2 acceptance calls plan abort only to poison a
  possible `DispatchLive` owner, marks the Python buffer terminal, and is not
  retried;
- combine validation/prepare/gate rejection invokes combine abort only and
  leaves the exact ticket live;
- combine gate acceptance changes the shared ticket state to consumed before
  commit; every later failure is terminal and cannot resurrect the ticket;
- a WORLD collective failure has unknown distributed state and is job-fatal,
  never reported as a successful local abort.

One full round trip therefore has exactly three per-call WORLD reductions:
two before dispatch publication and one before combine publication. Ticket
issuance and return are metadata-only and add no collective, payload copy,
NVLink, or RDMA operation. H5a/H5b additionally retain one local comm-stream
completion sync each in correctness-v1.

The constructor must close mixed off/force before creating differently sized
symmetric windows. The accepted minimal tradeoff is one fixed-tensor
preflight for every buffer construction when the compiled feature is present.
Off keeps no gate tensor or rail-balance field afterwards. The extra
constructor rendezvous is one-time; claiming zero off construction overhead
would be incompatible with detecting a valid off rank mixed with valid force
ranks. Dispatch gates are too late to repair that mismatch.

No capability bit opens until all three pieces and their default-off, fake
lifecycle, and truthful EP8 D1 fail-close tests pass together.

## 2026-07-21 — D055: close the public constructor consensus boundary

H6 construction now has two deliberately different costs. When the host
protocol is enabled, every rank first enters one fixed CUDA `int64[128]` MAX
gate before communicator/window creation; this rejects mixed `off|force`,
force geometry, capability, and ten-field arena-layout disagreements. A
unanimous off construction then follows the unchanged legacy sizing, topology,
SL/QP, CPU-communicator, runtime-argument, and field path and retains no
`_rail_balance_*` state.

Unanimous force reuses the same CUDA tensor and pinned mirror for a second MAX
after the common NCCL communicator exists but before the symmetric window is
registered. The second 20-field manifest freezes legacy/arena/total bytes,
resolved SL/QP, CPU/GPU timeouts, overlap/destroy flags, and repeats the force
geometry. Local sizing, environment parsing, QP probing, int32/int64, exact
bool, equality, and fixed 2 MiB alignment failures are encoded into this gate,
so a healthy peer cannot enter window creation alone. Runtime arguments are
fully built before the gate; after acceptance the next substantive call is
directly `_C.ElasticBuffer(*force_runtime_args)`.

Accepted evidence:

```text
PASS H6 constructor preflight 9/9
PASS C080-A Hybrid API 9/9
PASS legacy Hybrid identity goldens 4/4
PASS py_compile and git diff --check
PASS independent final review: Blocker 0 / High 0
```

Failures and rejected designs retained:

- the first Gate0 encoder validated layout elements only as nonnegative Python
  integers; an element or `M` above signed-int64 could throw during encode and
  strand peers. The complete manifest now validates the signed-int64 envelope
  before the collective and uses a caller-independent fallback;
- the first draft stopped after Gate0, leaving legacy sizing, force sizing,
  SL/QP parsing, and Python-to-C++ conversion able to diverge before window
  registration. The force-only sizing gate closes that interval without
  charging off a second rendezvous;
- a proposed force call to legacy `check_nvlink_connections()` was removed.
  Its PCIe branch performs rank-local NVML/import/parsing before an object
  collective, so catching a local failure and proceeding to the MAX gate could
  create a collective-order deadlock. Force-v1 targets H200/NVSwitch and relies
  on NCCL/C++ topology validation instead; off keeps the exact old helper;
- the initial Gate1 placement still performed Python field publication and
  argument construction after acceptance. Arguments are now frozen before the
  gate and C++ window construction is the immediate accepted operation;
- partial/broken sizing helpers could return mutually consistent but
  unaligned bytes. A fixed 2 MiB contract check now rejects them uniformly.

Implementation commit `1e0ea60` and test commit `eb6b834` are the recoverable
constructor checkpoint. This does not yet expose public force dispatch or
combine: the capability remains false, and the next slice is the buffer-bound
one-shot `EPHandle` lifecycle plus its three per-call WORLD gates. A truthful
EP8 constructor watchdog remains activation evidence; communicator creation
and the accepted C++ window call are explicitly job-fatal distributed/resource
boundaries rather than locally recoverable protocol errors.

## 2026-07-21 — D056: close the public one-shot Hybrid lifecycle

The narrow force-v1 public path now delegates to the private H3b/H4/H5
transaction without changing the ordinary `EPHandle` constructor or the
legacy off body. Every dispatch attempt reserves a monotonic invocation before
local validation, prepares the complete private transaction, enters WORLD
Gate1, finishes the plan, enters WORLD Gate2, then calls the adjacent private
commit/finish and preserves the native sixteen-item dispatch ABI internally.
Only after success does Python publish an ordinary `EPHandle` with one dynamic
private ticket containing buffer identity, invocation identity, and shared
one-shot state.

Combine accepts only that buffer's exact live ticket. It changes the shared
state from `LIVE` to `PREPARING` before private prepare, enters one WORLD gate,
then changes it to `CONSUMED` and clears the buffer's live pointer before the
irreversible private commit. Shallow `EPHandle` copies share the same ticket,
so only one copy can consume the route. A prepare/gate rejection executes the
idempotent abort and restores `LIVE`; any failure after an accepted dispatch
Gate2 or combine gate makes the buffer terminal and never resurrects the
ticket. A collective exception is also terminal because distributed state is
unknown.

The off path remains the original runtime/JIT/result path. It now rejects an
exact force ticket before any legacy handle metadata is unpacked; this is a
safety check only and prevents one buffer from replaying another force
window's routing state. Cached force dispatch, FP8, expansion, masking, bias,
events/async mode, deterministic mode, and non-unit alignment remain outside
the deliberately small v1 contract.

Accepted evidence at commits `741e126` and `cbe7df2`:

```text
PASS H3/H3b/H4b/H4c/H5a/H5b focused source/CPU contracts
PASS H6 public success, one-shot/copy, retry, foreign/off misuse, and terminal faults
PASS H6 constructor preflight 9/9
PASS C080-A Hybrid API 9/9 and legacy identity 4/4
PASS py_compile and git diff --check
PASS independent public audit: Blocker 0 / High 0
```

Failures and corrections retained:

- the old H4b source test sliced from public `dispatch` through public
  `combine`, so the newly inserted private combine helper was accidentally
  included in a dispatch-only negative assertion. The test now ends at the
  stable `_unpack_bias` boundary and still checks that public dispatch only
  delegates;
- independent review found that an off buffer could receive a force ticket and
  dereference its foreign routing metadata through the legacy path. Both off
  entry points now reject the exact ticket before legacy unpacking, with a
  focused regression;
- combine originally acquired abort ownership only after C++ prepare returned.
  It now marks `prepare_attempted` before the call, so an exception after C++
  installs completion cannot leave a hidden owner. The injected prepare-fault
  test initially retained the old no-abort trace and was updated to require
  the idempotent cleanup plus successful retry;
- command-only failures were retained rather than mistaken for code failures:
  one audit used a nonexistent legacy-test filename, `/usr/bin/time` was not
  installed, three reruns initially omitted `PYTHONPATH`, and the first log
  inspection assumed these records lived at the repository root. Corrected
  commands passed the complete focused matrix;
- a 552-line-control-plane concern was reviewed explicitly. The length comes
  from fixed fail-closed validation/gate/abort boundaries, not CUDA data-path
  branches. A generic transaction framework was rejected because it would
  hide those ownership edges without reducing device work.

Both Python and C++ capability bits remain false. CPU fake-runtime and source
contracts prove host ordering and failure ownership only; they are not a
truthful D>1 Rail/Gin/RDMA execution. The next local evidence is a real EP8
constructor/public D=1 watchdog followed by focused regression and sanitizer
closure.

## 2026-07-21 — D057: pass the real EP8 public D=1 watchdog

One isolated test now activates the already compiled private capability only
inside each of eight spawned worker processes. Production Python and C++
capability bits remain false before and after the run; no environment switch,
second runtime, vnode path, or public API was added.

The real H200/NCCL sequence performs exactly six fixed MAX gates:

```text
1  mixed off/force constructor Gate0 rejects before communicator creation
1  rank-3 invalid force geometry rejects before communicator creation
2  unanimous force Gate0 + sizing Gate1 create the registered window
1  first public dispatch rejects truthful D=1 prepare
1  second public dispatch retries and rejects the same truthful D=1 prepare
```

The successful force constructor reports logical `D=1,G=8`, exact
`legacy+arena` bytes, one allocated QP, and live registered runtime ownership.
Both public dispatch attempts converge on error priority 31/rank 0, advance the
invocation from 1 to 3, retain no live ticket, and keep the buffer nonterminal.
The accepted constructor's CUDA/pinned gate storage is pointer-stable across
its two constructor gates and both dispatch attempts. All ranks then call the
real collective destroy in the same order and restore both capability gates to
false.

Accepted evidence at commit `0597ed0`:

```text
PASS CPU/source default-capability contract
PASS real EP8 mixed-mode and asymmetric-invalid constructor fail-close
PASS real EP8 force constructor/window and collective destroy
PASS real EP8 public D1 dispatch fail-close twice with six stable MAX gates
PASS py_compile and git diff --check
PASS watchdog audit: Blocker 0 / High 0
```

Failures and resource decisions retained:

- the first new file inherited mode 0600 from the local umask; it was corrected
  to the repository's normal 0644 before commit;
- pre-run review found three watchdog weaknesses: rejection cases only had an
  aggregate gate-count check, `EP_DISABLE_GIN` used `setdefault`, and nonzero
  subprocess exit relied on spawn cleanup. Each case now asserts one Gate0,
  GIN disable is forced inside the isolated subprocess, and nonzero/timeout
  exits clean the dedicated process group;
- Megatron was restarted by a higher-level `launch.sh` after its first
  torchrun was terminated. The second cleanup killed the identified launcher,
  torchrun, and four training children under standing authorization. The
  unrelated 528 MiB CUDA elementwise sample remained untouched.

This closes the local public-control-plane activation evidence only. The D=1
guard fires before JIT, arena stores, source shuffle, or payload publication,
so the run is deliberately not D>1 Gin/RDMA or performance evidence. C080-G
now owns default-off, compatibility, fault, and sanitizer closure.

## 2026-07-21 — D058: close corrupt/missing world-plan fail-close

The final C080-D acceptance gap is covered inside the existing two-window
vnode harness, without a new kernel, runtime, transport, or public API. One
small 4x2/H256 case performs three generations on the same source/world
buffers:

1. rank 0 changes one value in world `channel_count` after prepare;
2. rank 1 removes one tensor from its local owning world-plan tuple;
3. the next generation executes the complete normal round trip.

Both injected generations gather the exact preflight result over Gloo before
the first `vnode_finish`/B0 entry. Corruption must report
`world plan tensor 0 value mismatch`; missing state must report
`world plan tensor count mismatch`. Every rank then returns through the same
finally block, calls idempotent world/source abort for that invocation, and
enters a monitored barrier. Generations 809/810/811 give distinct source/world
invocations, so the final byte-exact dispatch→vnode→return→combine success is
also a real recovery proof.

Accepted evidence at commit `6377e0c`:

```text
PASS focused CPU oracle and py_compile/diff check
PASS true EP8 corrupt world-plan pre-B0 reject and abort
PASS true EP8 missing world-plan pre-B0 reject and abort
PASS same-buffer next-generation complete 4x2/H256 round trip
PASS independent fault audit: Blocker 0 / High 0
```

The first review found one Medium in the test: matching only the generic word
`AssertionError` could accept an unrelated rank-local failure. Plan validation
now labels tensor count/device/layout/shape/value separately, and each fault
must match its specific tag. No production source changed. This closes C080-D;
C080-G still owns the final compatibility and focused sanitizer matrix.

## 2026-07-21 — D059: C080-G final-tree local closure matrix

The final tree was exercised as one compatibility manifest rather than by
adding another runtime or test-only data path. Both production capability bits
remained false throughout. The public force lifecycle was activated only by
its isolated worker-local watchdog; the ordinary product configuration stayed
default-off.

Fresh final-tree evidence:

```text
PASS original EP8 default-off smoke with allow_hybrid_mode=0
PASS original EP8 default-off smoke with allow_hybrid_mode=1 at truthful D=1
PASS planner 38/38, API 9/9, layout 5/5, legacy identity 4/4
PASS public one-shot fake-runtime lifecycle and H4b/H4c/H5a/H5b source contracts
PASS H3 fixed WORLD gate, H3b eighteen-gate planner, H4b D1 fail-close/recovery
PASS true EP8 H7168 source shuffle and return-unshuffle, including D32>K4
PASS 4x2/H7168 and 2x4/H7168 vnode full loops
PASS source faults and corrupt/missing world-plan abort plus recovery
PASS fresh-cache dispatch 4/4 and combine 6/6 force/legacy codegen pairs
PASS dispatch/combine constraint suites 8/8 and unchanged recursive legacy hashes
PASS setup.py build_ext --inplace
```

The focused sanitizer target was `hybrid_vnode_2x4_h256` from a dedicated
warm JIT cache. It deliberately uses `D=4>K=2` and executes the shared
pack-vnode, return-demux, and production return-unshuffle kernels. With Compute
Sanitizer 2025.1.0.0:

```text
memcheck:  PASS, ERROR SUMMARY: 0 errors
synccheck: PASS, ERROR SUMMARY: 0 errors
initcheck: PASS, ERROR SUMMARY: 0 errors
racecheck: PASS, 0 hazards displayed (0 errors, 0 warnings)
```

Initcheck ran without a kernel filter so predecessor writes were visible;
API-memory checking was disabled to avoid allocator-only noise. Racecheck was
filtered to the three new vnode/return kernels. Earlier unchanged source-core
evidence remains valid: its racecheck has zero errors and six documented
shared-metadata WAW warnings. The H6 delta itself is Python/test/documentation
only and truthful D1 rejects before per-call arena use, JIT, or data kernels,
so a public-D1 sanitizer rerun would add no Hybrid data-plane coverage.

Failures and resource decisions retained:

- `pstree` was unavailable on this host. Process ownership was verified with
  `ps` and `nvidia-smi` instead; this was a diagnostic-command failure, not a
  test failure.
- The independent final audit's first API invocation omitted `PYTHONPATH` and
  failed with `ModuleNotFoundError`. Re-running with the pinned repository
  environment passed all 9/9 API cases; no product change was made for this
  command error.
- An unrelated small CUDA sample was left alone as requested. During the
  sanitizer run, the eight approximately 1.3 GiB GPU allocations were verified
  to belong to this test's worker processes.
- No NCU, Nsys, throughput, or speedup result was produced. Those runs remain
  behind the user's profiling procedure and an idle-GPU check.

No source or CUDA change was needed in this closure slice. The local evidence
is sufficient for `C080-G`. Independent final review reports Blocker 0, High 0,
Medium 0, and two documentation-only Low findings that are corrected in the
closure checkpoint: off identity is scoped to native sizing/runtime arguments,
GPU buffer, JIT and result identity because local mode/ticket guards exist; the
handoff is advanced beyond its old C080-G resume point. This still cannot
establish real D>1 Gin destination, QP/NIC ordering, RDMA completion, or
multi-node speedup. Those remain C080-H.

A separate C090 exit audit prevents this checkpoint from being overextended.
It reports Blocker 0, High 1, and Medium 2 for the broader exhaustive-local
campaign: no explicitly named Zipf/log-normal routing fixtures, no GPU
stale/corrupt transit-key `p` injection on the current descriptor-free path,
and no rank-selective delayed-rank/liveness case on that path. Earlier queue
slow-consumer evidence belongs to a superseded protocol and cannot close the
last item. C080-G therefore passes independently while C090 remains in
progress; these three minimal cases precede C100 profiling.
This paragraph records the D059-time gap; all three items are closed by D060.

## 2026-07-21 — D060: close the three retained C090 local gaps

The route-distribution gap is closed inside the existing B1 strict runner,
without a sampler, new planner, or production path. Three fixed, explicitly
named token-level expert-route fixtures now pass the same destination
deduplication and fourteen-tensor GPU materializer as the original cases:

```text
route_level_one_hot:    moved=18, proxy_required=(0, 6, 6, 6)
route_level_zipf:       moved=21, proxy_required=(7, 4, 3, 7)
route_level_log_normal: moved=33, proxy_required=(11, 6, 5, 11)
```

Destination zero is local padding and remains zero in the counted matrix. The
original seventy cases are unchanged; the three C090 cases raise the strict
single-GPU total to 73. CPU oracle, legacy 4/4, `py_compile`, and CUDA exact
comparison on device zero pass. Independent review manually recomputed every
move/proxy total and reports Blocker/High/Medium/Low 0/0/0/0. This is fixed
distribution correctness, not statistical-fit or performance evidence.

The final protocol gap uses one existing 4x2/H256 vnode case over three new
generations on the same source/world buffers:

1. generation 812 changes the already prepared moved record's four-byte
   linked transit key `p` from 0 to 1 on source egress rank 2;
2. generation 813 queues a two-million-cycle sleep only on rank 2's existing
   world comm stream, then immediately enters the fixed B0--B5 protocol;
3. generation 814 runs an ordinary byte-exact round trip with no injection.

The fault is a sticky status, never a trap. Every rank calls finish exactly
once. After B5 and the existing stream synchronization, a new private read-only
host getter exposes the status vector already copied by the vnode bridge. The
exact nonzero graph is frozen: rank 2 reports `InvalidProxy=35` at pack plus
two downstream `ReadyMismatch=1` values; rank 6 reports the corresponding
forward/return mismatch lanes; all other entries are zero. Ranks 2 and 6 throw
only after the synchronized snapshot, all ranks gather diagnostics, then the
existing idempotent world/source abort runs. Generations 813 and 814 both
complete exact dispatch, return, unshuffle, epilogue, and immutable-input
checks, proving late-rank liveness and same-buffer recovery.

The private getter adds one host boolean and one additive underscored pybind.
It is compiled into the extension rather than hidden by a test build flag, but
only serves the existing private vnode proof bridge; it changes no CUDA/JIT
kernel, layout, launch ABI, public Python API, or production dispatch/combine
hot path. Final-tree evidence:

```text
PASS build_ext --inplace after the host-only change
PASS true EP8 fresh-cache g812 fault, g813 delayed success, g814 recovery
PASS C080-A API 9/9 and legacy identity 4/4
PASS B1 CPU oracle plus single-GPU 73/73 exact materializer cases
PASS C++ final audit: Blocker/High/Medium/Low 0/0/0/0
PASS protocol final audit: Blocker/High/Medium/Low 0/0/0/0
```

The true-EP8 run used dedicated cache
`/tmp/deepep-c090-transit.YIOYYv` and master port 30006. The cache is an
ephemeral build artifact, not a checked-in dependency.

Failed designs and retained boundaries:

- The first delay draft enqueued `_sleep` and then performed two Gloo gathers.
  Review correctly rejected that as liveness evidence because the delay could
  finish before B0. The accepted order gathers the rank marker first, enqueues
  the delay second, and performs no host/CUDA rendezvous before finish.
- The first fault draft called the status getter outside an exception guard.
  If finish failed before the D2H snapshot, one rank could skip the diagnostic
  gather and deadlock peers. Getter errors now join the same all-rank diagnostic
  gather before any assertion or abort.
- A size check initially multiplied `6 * status_stride` in `int` before its
  cast. Review changed it to `size_t(6) * status_stride` before commit.
- The pre-existing vnode abort does not promise in-process recovery from an
  arbitrary CUDA failure before stream synchronization. The accepted corrupt-p
  case reaches the synchronized sticky-status boundary; it must not be cited
  as proof for traps or pre-sync asynchronous failures.

No sanitizer rerun is required because no device/JIT code changed and C080-G
already sanitized these kernels. No new NCU/Nsys or performance measurement
was run in this slice. Real D>1 Gin/RDMA remains outside this evidence.

The final C090 exit audit maps every item in `C080_PLAN.md` section 12 to the
unchanged historical matrix or the two new checkpoints. It reports Blocker 0,
High 0, Medium 0; its sole Low was the stale checkpoint text corrected here.
C090 is therefore PASS. No additional CPU, GPU, full-suite, or sanitizer rerun
is required before the next controlled C100 profiling run.

## 2026-07-21 — D061: audit the Goal boundary and network-readiness gap

A fresh requirement-to-evidence audit maps every C080 integration requirement
in the active long-running Goal to the final tree.  GPU assignment
materialization, bounded static proxy slots,
direct-final source shuffle, proxy-return combine, owner return-unshuffle,
fixed WORLD preflight, the shared-core vnode oracle, isolated Hybrid codegen,
and recoverable Git/log checkpoints are present.  The audit reports zero
Blocker and zero High findings.  C080 remains accurately labelled
`LOCAL_COMPLETE / MULTINODE_PENDING`: truthful D>1 Rail/Gin execution, QP
behaviour, and RDMA completion are the deliberate environment gate, not local
evidence.

Two activation boundaries are retained rather than hidden.  Both production
capability bits remain hard false.  While the host bit is false, an ordinary
off rank does not join a new constructor collective merely because another
rank incorrectly asks for the unavailable force mode; this preserves the
default-off contract.  When every rank test-enables or eventually production-
enables the host capability, the already implemented universal constructor
gate runs before communicator/window creation and owns mixed off/force
consensus.  Therefore the hard-false development branch must not itself be
described as mixed-mode-safe.  The gate does not first prove that all processes
enabled the host bit: unanimous capability activation is an external safety
precondition for every validation or production run.

The same audit found two stale user-facing descriptions.  The constructor
docstring said off was fully identical even though local configuration parsing
and mode guards exist, and `FeatureUnavailable` still claimed that dispatch
and combine had not been installed.  They now state the narrower frozen
identity and the actual D>1 validation gate.  The first API run failed because
its exact error-string assertion still expected the obsolete message;
updating that assertion produced:

```text
PASS 9/9 C080-A Hybrid API tests
PASS 4/4 legacy Hybrid identity goldens
PASS 9/9 H6 constructor preflight tests
PASS C080-H6 public lifecycle CPU fake runtime
PASS git diff --check
```

This exact-string failure is a test maintenance failure, not a product or GPU
failure.  No CUDA source, JIT specialization, ABI, layout, capability value,
or hot-path instruction changed.

The C100 inventory confirms that the only accepted runtime profile so far is
the tiny three-record source-shuffle fixture.  It established peer-TMA fixed
latency and rejected a slower binary resolver, but cannot select a throughput
optimization.  The next controlled experiment must use large moved-volume
H7168 source-shuffle and return-unshuffle cases, pin one rank/kernel/launch,
and exclude vnode validation scaffolding.  Existing raw NCU/Nsys reports still
live under `/tmp`; their reported metrics are already preserved in D032/O031,
but the binary reports are not a checked-in dependency.  Six accepted reports
were also copied, without modification, to the ignored persistent workspace
directory `.cache/rail_balance/c100/`; `SHA256SUMS` records their checksums.
This local backup prevents a `/tmp` cleanup from erasing the raw evidence while
keeping 74 MiB of profiler binaries out of Git history.

C105 has no one-command multi-node runner yet.  `test_ep.py` cannot simply be
reused because its expanded, cached, FP8, event, and alignment matrix exceeds
force-v1's deliberately narrow contract.  The smallest future package is a
dedicated BF16/noncached/synchronous validation harness.  It may test-enable
both hard-false capabilities only inside that process, assert and restore the
original values, and emit an explicit unreleased-validation evidence label;
production defaults remain closed until C080-H passes.  Real Gin wait/QP
counters do not exist yet and must be emitted as unavailable rather than
invented from planner counts.

No new NCU, Nsys, throughput, or GPU correctness run was started in this
checkpoint.  The next controlled profiler invocation still waits for the
user's promised workflow and an idle-GPU check.  The unrelated light CUDA
sample remains untouched as requested.  C100 is still `IN_PROGRESS` and C105
is still `PLANNED`; this audit closes only the C080 local-integration slice, not
the full long-running Goal.

## 2026-07-21 — D062: build the controlled large-volume source fixture

C100 now has a fixed source-only fixture large enough for later payload-scaling
attribution without pulling vnode B0--B5, synthetic expert, demux, or two-window
validation work into the profile.  Eight owners each route 1,024 tokens to a
different remote destination in a nine-destination namespace (destination zero
is local padding).  The four distinct top-k experts for each token belong to
the same server, so production destination deduplication emits one payload copy
per token.  The exact schedule is:

```text
G=8, D=9, K=4, N=1024/rank, C=256, seed=100
keep=128 per hot owner/destination
moved=896 per owner and 896 per egress
total moved=7168, Pcap=896 exactly
moved-active source channels=224/rank, four records/channel
```

`c100_volume_h256` and `c100_volume_h7168` differ only in hidden width.  Their
top-k routes, token count, iteration, remainder seed, count/quota/segments,
source channels, egresses, proxy slots, and complete CPU schedule are equal.
Explicit owner and egress counters each equal `(896,) * 8`.  The two cases are
available only through `--case-name`; the default exhaustive C080-D suite does
not inherit their larger cost.  The raw verifier caches each remote owner's
expected source once instead of rebuilding a full N×H tensor for every proxy
slot; this changes only test cost, not the comparison or device path.

The first implementation used C=8 and passed both H256 and H7168 functionality,
but independent review rejected it for profiling because only seven source
CTAs per rank moved data.  C=128 also passed H256 after review requested higher
production-like concurrency.  The final C=256 matches the normal
`num_sms * four-channels-per-SM` scale for a non-overlap force configuration;
224 CTAs move payload on every rank.  No C8/C128 result is retained as a timing
or optimization claim.

Final C256 evidence uses the existing warmed cache
`/tmp/deepep-c100-fixture.heZiRA` and ports 30010/30011:

```text
PASS C100 H256: true EP8 LSA, moved=7168, per-owner/egress=896,
     exact raw legacy TokenLayout and immutable source inputs
PASS C100 H7168: same plan and the same exact checks
PASS CPU oracle and py_compile
PASS git diff --check
```

The Hybrid arena is 2 MiB/GPU for H256 and 26 MiB/GPU for H7168; the complete
H7168 symmetric test allocation is approximately 140 MiB/GPU, safely below an
H200 OOM boundary.  These runs are functionality evidence only: no duration,
bandwidth, NCU, Nsys, or speedup result was collected.

Before these functional runs, process inspection found one definite Megatron
torchrun group, PGID 173641, with four 36,212 MiB workers.  Under the user's
standing authorization, the exact launch group was terminated and GPU compute
processes were verified empty; no unrelated process was touched.

The next source measurement must use the user's profiling workflow.  It will
warm separately, pin one device/kernel/launch, and compare these H256/H7168
cases rather than reusing the old three-record fixture.  Return-unshuffle still
needs the same large plan wired into its existing standalone harness before its
own runtime attribution.

## 2026-07-21 — D063: reuse the controlled plan in return-unshuffle

The standalone production-shared return-unshuffle harness now accepts the two
named-only C100 source fixtures.  It imports the exact G8/D9/K4/N1024/C256 plan
rather than rebuilding a second workload: H256 and H7168 therefore retain the
same routes, counts, quota, segment assignment, source channels, egresses,
proxy slots, total 7,168 moved records, and Pcap 896.  Only the raw combine
TokenLayout width changes.

The first CPU-oracle run failed before CUDA because `_selected_cases()` still
asserted that its complete tuple contained exactly the three default C080
cases.  Appending a fourth named C100 case correctly tripped both the hidden-
width and moved-count tuple assertions.  The fix does not weaken those
regression checks: they now apply to an explicit `base_cases` tuple, while the
optional named profile case has independent `moved == 7168` and `Pcap == 896`
assertions.  This failed attempt is retained because it exposed a test
assumption rather than a device or product failure.

Enumerating CPU fingerprints no longer constructs all 896 full H7168 return
rows for every egress.  The oracle checks the exact deterministic fingerprint
formula for every slot and constructs the last raw row as a formula probe.
The GPU test remains strict: it creates every full proxy-return row, invokes
the unchanged production-shared return-unshuffle kernel, compares every target
row byte-for-byte, and proves every non-target byte retains its poison value.
Buffer sizing is derived from the selected plan's maximum token count, top-k,
hidden width, and arena layout; the default three-case GPU suite is unchanged.

Accepted checks use the pinned ABI Python and ports 30012/30013:

```text
PASS py_compile
PASS default C080-F CPU oracle
PASS named C100 H256 CPU oracle: moved=7168, Pcap=896
PASS named C100 H7168 CPU oracle: moved=7168, Pcap=896
PASS C100 H256 true EP8 LSA return-unshuffle, exact targets/non-targets
PASS C100 H7168 same plan and exact checks
PASS default three-case C080-F true EP8 regression, one-shot and recovery
PASS git diff --check
```

GPU compute-process inspection was empty before these runs; no process was
terminated in this checkpoint.  Neither run recorded duration or bandwidth,
and no Nsys/NCU command was invoked.  The user then supplied the profiling
measurement contract, which is saved in the active Codex profile as
`gpu-performance-evidence-chain`; its required order is profiler-free baseline,
Nsys critical-path attribution, targeted NCU, one-variable change, and
profiler-free retest.

Independent final review reports Blocker 0 / High 0 / Medium 0.  Its three
Low observations are either resolved by documenting the two named commands or
explicitly accepted test structure: the CPU-only probe is intentionally
lighter while GPU bytes remain exhaustive, and the return harness imports the
already committed source fixture instead of duplicating its private schedule.

## 2026-07-21 — D064: freeze the C100 measurement environment and claim scope

The user supplied a profiler-free → Nsys → targeted NCU → single-variable
change → profiler-free retest contract.  It is saved in the active Codex
profile as the validated `gpu-performance-evidence-chain` skill: a 61-line
core and a 325-line detailed reference.  The project-facing audit surface is
`C100_PERFORMANCE_EVIDENCE.md`; it starts with every new performance field
explicitly `NOT_COLLECTED` rather than backfilling old tiny-fixture results.

The exact host, package, linked-library, topology, profiler-version, section,
recipe, clock, power, temperature, MIG/MPS, and process state were read from
the installed environment.  The operational target remains the user's 8-GPU
H200/NVSwitch node.  A management-plane discrepancy is retained as missing
evidence: `nvidia-smi` labels the devices L20X/CC8.9, NCU labels them
L20X(GH100), executed cubins are sm90a, topology reports NV18, and the P2P
query says `NS` even though exact true-EP8 LSA kernels pass.  This does not
block same-machine paired timing, but published H200 peaks will not be used as
an efficiency denominator until the alias/P2P reporting is explained.

The manifest also separates PyTorch's NCCL API version 2.28.9 from the pip
NCCL 2.30.4 `libnccl.so.2` actually selected by the DeepEP extension.  Nsys is
2024.6.2 and NCU is 2025.1.1; installed help, recipes, sets, sections, and the
5,364-metric query were inspected before constructing future commands.
Official NVIDIA H200, CUDA 12.8 TMA, NCCL Device API/GIN, Nsys 2024.6, and NCU
2025.1 references are linked directly in the evidence file.

A source/runtime audit establishes the next measurement boundary.  Each
private transaction is one-shot: prepare synchronizes, finish synchronizes and
publishes PlanReady, source synchronizes after status readback, return includes
test B1/B4 barriers and snapshot/status copies, and abort synchronizes before
releasing the transaction.  Therefore the first no-profiler result must be
named `source_shuffle_checked_adapter` or `return_unshuffle_test_adapter`.
Calling either result raw production-kernel latency would be false.  Nsys will
later split launch, kernel, barrier, copies and host gaps.

No profiler, timing loop, CUDA/runtime edit, or performance claim was made in
this checkpoint.  The old six reports remain tiny-fixture evidence only.  The
next code change is a standalone measurement harness that reuses the exact
C100 schedule and leaves production source untouched.

## 2026-07-21 — D065: freeze the C105 validation contract before a runner

C105 now has a CPU-only contract module and test, not yet a real multi-node
runner.  It generates deterministic `balanced`, `two_hot`, and `one_hot`
routes, counts distinct remote destination servers, and constructs a stable
JSON result skeleton whose label is fixed to
`REAL_HYBRID_RUNTIME_UNTESTED`.  Missing QP, NIC, wait, plan, and traffic
evidence stays null/not-available; callers cannot supply a stronger label.

Two independent reviews forced the helper to match the executable force-v1
domain rather than a convenient test domain: D/G are 2--32, K is 1--32,
E is at most 2,048, E/world is at most 256, and both token and contribution
flattened products fit int32.  `C080_PLAN.md` had incorrectly said G could be
one and omitted the workspace expert/contribution bounds; it now matches the
dispatch/combine/epilogue host and CUDA asserts.  `two_hot` additionally needs
G>=3 only so it differs from balanced.

The destination counter is checked by a handwritten non-square D3/G2 oracle,
while builder output is independently checked for its expert server.  JSON
input is deep-copied, finite, recursively string-keyed even through list/dict
subclasses, and stable under caller mutation.  Canonical remote volumes are
deliberately different (36/18/9) and must be reported rather than normalized.

Retained failures:

- an initial bare `python` command did not exist in the environment;
- one review command omitted `tests/elastic` from `PYTHONPATH` and raised
  ImportError;
- the first non-square hand oracle expected `[1,0,1]` where both source tokens
  actually target server zero; both Python interpreters reported the same
  failure, and the manually recomputed expectation is `[2,0,0]`.

After those corrections, system Python and the fixed ABI Python each pass
12/12 tests plus `py_compile`.  This checkpoint proves only route/schema
contracts; capability bits remain false and real runtime execution remains
untested.

## 2026-07-21 — D066: commit the auditable C100 adapter benchmark

Commit `056ad4d4df7007c3e4927e2f7d096ec5246da571` adds
`tests/elastic/bench_rail_balance_hybrid_lsa.py` without changing CUDA, JIT,
buffer layout, capability bits, or the public Hybrid path.  It reuses the exact
G8/D9/K4/N1024/C256 schedule for source and production-shared return, creates a
fresh invocation/ticket for every cold/warm/steady sample, and retains all raw
rank intervals plus same-host global stage/transaction spans.

The report is deliberately fail-closed.  It requires an initially empty private
JIT cache and exactly seven target cubin families; hashes Git/source, `_C.so`,
generated CU/cubin and optional PTX/SASS, and the actually loaded NCCL/CUDART;
records DeepEP's selected CUDA home/nvcc; and checks pre/post GPU UUID, driver,
MIG, compute mode, throttle, MPS, and compute-process state.  Its watchdog owns
a separate process group and performs TERM, bounded polling, KILL if required,
and leader reap on normal failure, timeout, SIGINT, or SIGTERM.

Retained development failures and rejected evidence:

- the first large `apply_patch` attempt was rejected atomically, so no partial
  file entered the tree; the harness was then added in reviewable increments;
- the first skill/CLI validation used bare `python`, which is absent; all
  accepted checks use `/home/chen/.cache/deepep-sjlgpt/bin/python`;
- an early percentile self-check compared an interpolated p99 as an exact
  convenient decimal; it was corrected before any report was accepted;
- v1 used `max(per-rank duration)` and could hide launch skew.  A v2 smoke
  exposed 79.335 us rank-max versus 141.075 us global span, so v3/v4 use only
  `max(end)-min(start)` as timing truth;
- the first signal test fired before workers existed and was rejected as
  watchdog evidence.  The accepted injection waited for `spawn_main`, sent
  SIGTERM then SIGINT, returned 130, and proved the child process group and GPU
  processes disappeared;
- successive audits found orphan process-group liveness, repeated-signal,
  spawn-window, report-before-destroy, equal data/control timeout, JIT identity,
  byte-scope, runtime-state, misleading formal-eligibility, nvcc-selection, and
  profiler-declaration issues.  Each was fixed or made an explicit manual
  acceptance boundary; no issue was hidden by weakening the gate;
- `git diff --no-index --check /dev/null <new-file>` returned 1 because the new
  file differs, with no whitespace diagnostic; py-compile and staged
  `git diff --check` are the accepted syntax/whitespace evidence;
- all v1--v4 timing runs used a dirty worktree and/or one steady sample.  The
  harness correctly marked every one ineligible, so none is a baseline;
- during watchdog work, a definite Megatron launch/session PGID 459498,
  torchrun 459501, worker groups 459530--459533, and wandb group 459924 were
  terminated under the user's standing authorization.  Exact groups exited
  and all GPUs were empty; no unrelated process was touched.

Final validation on benchmark SHA
`1743adfca7edce6ac224b0d842d30b0fabcb6e0b5be2e3ec77d60776d016799b`:

```text
PASS py_compile and CLI help
PASS C080-B2 plan CPU oracle
PASS C080-D source CPU oracle, including both C100 cases
PASS C080-F return CPU oracle
PASS C105 validation contract 12/12
PASS one final-SHA EP8 H256 source report smoke
PASS one immediately preceding revision of the same report path
PASS actual loaded libnccl/libcudart identity and DeepEP-selected nvcc 12.8
PASS exact seven JIT cubin families and stable pre/post identities
PASS independent final audit: Blocker 0 / High 0
```

The last smoke reported 146.117 us only to prove the report path; it is dirty,
one-sample, and explicitly not a performance result.  No Nsys, NCU, or
performance-path edit was made.  The next checkpoint is clean direct 10+100
source/return × H256/H7168 collection with repeated independent runs.

## 2026-07-21 — D067: preserve partial clean source baselines and stop safely

At clean commit `3c548390d96c229a318966e987a9a624ba76719d`, the current
HEAD passed the plan/source/return CPU oracles and the 12-test C105 validation
contract.  With GPUs empty, six direct profiler-free runs then completed:
source H256 and H7168, three independent processes each, every process using
10 warmup and 100 steady samples, a fresh JIT cache, persistent JSON, and a
unique master port.  All six reports pass `baseline_collection_eligible`, have
clean/stable code identities, and report no unexpected GPU process.

The raw results are preserved but rejected as a stable baseline.  H256 run
medians are 113.745, 109.794, and 117.764 us; pooled median/p95/p99 are
112.700/133.650/164.241 us, with 12.15% CV and a 292.757 us maximum.  H7168
medians are 130.954, 157.439, and 143.885 us; pooled median/p95/p99 are
144.728/169.219/271.935 us, with 82.84% CV and a 2,274.190 us maximum.

The largest H7168 sample is specifically rank-local: rank 4 spends 2,253.625
us in the synchronous adapter while the seven peers spend 85.855--108.684 us.
Its start skew is only 38.153 us and end skew is 2,180.359 us.  A second run's
770.897 us maximum instead combines a 429.110 us start skew and 344.129 us
rank maximum.  P0, 3,201 MHz memory clocks, and non-benign throttle checks stay
valid; GPUs 1/3/6 consistently report 1,500 MHz SM clocks while peers report
1,980 MHz during pre/post samples.  No causal claim is made from snapshots.

One aggregation command initially assumed a nonexistent `steady_raw` report
key and failed with `KeyError`.  Reading the committed schema showed raw
records live in `steady`; the corrected aggregation used all 300 records per
case and retained the failure instead of editing reports.

The user requested a stop just after the first return-H256 process launched.
SIGINT was sent to the benchmark watchdog; it exited 130, reaped its process
group, wrote no partial return JSON, and left no benchmark, `spawn_main`, or
GPU compute process.  Return H256/H7168 remain uncollected.  The six source
JSON files and a verified `SHA256SUMS` live at
`.cache/rail_balance/c100/formal/3c54839/`; their hashes are also recorded in
`C100_PERFORMANCE_EVIDENCE.md` so the checkpoint survives remotely.

No Nsys, NCU, CUDA/JIT/runtime edit, or optimization followed these unstable
samples.  Resume first at measurement-stability falsification, then finish
return baselines; only afterward select Nsys/NCU targets.

## 2026-07-21 — D068: limit PyTorch intra-op threads and localize remaining skew

Work resumed from clean commit `4211bae`.  Recreating the Goal through the
goal API failed with `cannot create a new goal because this thread has an
unfinished goal`: the previously paused Goal still appears blocked to the
tool.  The failure changes neither repository state nor the user-authorized
objective, so execution continued under the existing checkpoint plan.

The first stability experiment changed exactly one external variable:
`OMP_NUM_THREADS=1`.  Source-H7168, G8/D9/K4/N1024/C256, 7,168 moved records,
10 warmup, 100 steady iterations, direct watchdog execution, fresh JIT cache,
and the same `max(end)-min(start)` truth were unchanged.  All GPUs were empty
before each process, all three reports pass the automatic gate, and no worker
or GPU process leaked.

Run medians are 132.625, 152.436, and 137.857 us; pooled median/p95/p99 are
140.501/160.198/175.784 us, CV is 8.34%, and maximum is 218.716 us.  Relative
to the original H7168 reports, pooled CV falls from 82.84% and maximum from
2,274.190 us.  The cgroup exposes 192 logical CPUs but grants a 96-CPU quota;
default PyTorch reports 96 intra-op and 96 inter-op threads per rank, whereas
OMP=1 reduces only intra-op to one.  This supports thread pressure as the
severe-tail cause.

An independent read-only audit verified all report statistics and hashes.
Tracked benchmark/extension/generated/production source is identical across
the original and OMP collections, and all six source-shuffle cubins have the
same disassembled-instruction hash.  Rank-local maximum-stage median changes
only 97.518→96.556 us while its p99/max collapse.  Because the two conditions
were collected sequentially rather than interleaved, this remains tail-control
evidence and is not called a kernel or end-to-end speedup.

It does not close stability.  Rank 1 starts first in 299/300 samples and rank
7 last in 295/300.  Run median start skews are 44.943/60.566/46.031 us, while
the median global span after subtracting that skew is
87.775/93.237/91.610 us.  The three global medians still span 14.94%, so the
next single-variable experiment must isolate the Gloo/post-barrier release
order rather than edit a CUDA kernel.

Two exploratory aggregation commands failed with `KeyError`: the first
assumed top-level `baseline_collection_eligible`, and the second assumed a
`stages` dictionary.  The committed schema instead stores the former under
`measurement` and raw samples directly under `steady`; the corrected reader
used all 300 records.  Both mistakes are retained here and no report was
modified.  One checksum verification was also launched from the repository
root even though the manifest stores relative basenames; it failed to find the
three files.  Rerunning from the artifact directory verified all three as OK.

The verified reports are local ignored artifacts under
`.cache/rail_balance/c100/stability-omp1/4211bae/`.  No Nsys, NCU, production
code, CUDA/JIT kernel, or public Hybrid path changed.

## 2026-07-22 — D069: add a CPU-only Gloo release probe

The next diagnostic is isolated in
`tests/elastic/bench_rail_balance_gloo_gate.py`; it does not import DeepEP,
call `init_dist`, set a CUDA device, or create NCCL.  Eight spawned processes
create a default Gloo rendezvous group and a separate all-rank Gloo control
group matching the benchmark's control-plane shape.  The only timed window is
the interval around `monitored_barrier`; all-gather and JSON work occur after
the exit timestamp.

The report preserves cold/warm/steady raw intervals, entry and return spans,
the complete rank order, last-arrival-to-first-exit time, host/thread affinity,
context-switch/scheduler snapshots, cgroup state, Git/source identity, and the
full command.  Both pre/post rank manifests must report CUDA uninitialized.
The parent watchdog owns a new process group, handles HUP/INT/TERM/QUIT,
performs TERM/KILL cleanup, and writes JSON through fsync plus atomic replace.

Two dirty-tree 0+3 smokes pass.  The second uses
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1`, reports 43.624 us median Gloo
return span, and returns the exact order `[1,2,3,4,5,6,0,7]` in all three
samples.  A standalone reader recomputed every timestamp/span/permutation and
verified all eight ranks remained CUDA-uninitialized.  These smokes validate
the instrument only; they are not formal evidence.

A forced SIGTERM during the spawn window exits 124 through the outer timeout,
leaves no JSON, and leaves no probe/spawn/GPU process.  The visible traceback
is retained as expected interruption evidence rather than hidden as success.

The first independent code audit found one High watchdog race: signals were
handled but not blocked across `Popen` and PGID assignment, and cleanup itself
could be interrupted.  Before commit, the probe adopted the already audited
C100 signal-mask/two-phase TERM→KILL/PGID-disappearance sequence.  The same
audit's three Medium findings were also closed: report build/write errors now
converge across ranks before PASS, the actual `se.nr_migrations` field is
captured, and pre/post Git/source identity plus run/config hashes must match.
The cgroup snapshot now discloses `/proc/self/cgroup`, and raw JSON invariants
are asserted in the producer.  No finding was waived to obtain a clean audit.
The pinned environment has neither `black` nor `ruff`; both version/check
attempts failed with `No module named ...`.  Validation therefore uses
`py_compile`, `git diff --check`, the explicit JSON invariant reader, watchdog
fault injection, and independent source review rather than silently switching
Python environments.
Final read-only re-audit on source SHA
`e0f68b42380a415708c97f3f855efeedb014665374dc2c2ba421782e48d8f45b`
reports Blocker0/High0/Medium0/Low0 and confirms the only `torch.cuda` calls
are two non-initializing `is_initialized()` assertions.

During this work, a positively identified training launch using the
`megatron-lm-gpu` environment occupied four GPUs.  Under the user's explicit
standing authorization, only its resolved process group 983614 (launcher,
torchrun, four workers, compiler helpers and W&B children) received TERM and
fully exited.  A separate DLB/NVSHMEM test briefly appeared afterward; it was
not Megatron, was left untouched, and exited on its own.  All GPUs were empty
before the second smoke.

## 2026-07-22 — D070: formally prove bare-Gloo release skew

After committing and pushing the audited probe as `ee41415`, three independent
clean processes ran with an empty CUDA visibility mask, OMP=1, 10 warmup and
100 steady samples.  Their Gloo return-span medians are
38.693/36.319/38.028 us; run medians span 6.54%.  The first run retains two
host-tail samples and reaches 147.017 us, so pooled CV is 23.65%; no sample was
deleted.  Runs two and three have CV 6.58% and 7.02%.

rank 1 returns first in 98/100, 100/100 and 98/100 samples; rank 7 returns last
at the same rates.  The smallest formal median covers
80.8%/60.0%/78.9% of E1's three median stage-start skews.  This passes both
pre-registered thresholds without reinterpretation.  Each report is clean,
pre/post source and Git identities match, every rank remains
CUDA-uninitialized, and cgroup throttled count/time have zero delta.

Before collection, the earlier Megatron-based workload restarted as a new
explicit `restart5` launch and consumed four GPUs plus many CPU compiler
workers.  Exact PGID 1046676 was resolved and terminated under standing user
authorization.  It did not restart again, the host's active CPU consumers
dropped below 3%, and the formal runs began only afterward.  No unrelated
process was terminated.

The verified ignored artifacts and manifest live under
`.cache/rail_balance/c100/gloo-gate/ee41415/`.  This closes the narrow Gloo
release falsification, not measurement work as a whole.  Source-H256 and both
return widths remain, followed by Nsys; NCU still has no selected invocation.

## 2026-07-22 — D071: collect source-H256 and retain rank-local tails

Before GPU collection, another independently launched Megatron-environment
`restart6` occupied four GPUs and spawned many Inductor compiler workers.  Its
resolved PGID 1066956 was terminated under standing authorization.  Three
seconds later GPUs were still empty and no high-CPU training process remained;
the source-H256 group began only then.

Three clean OMP=1, 10+100 processes at `bac80ee` pass every automatic gate,
use fresh JIT caches, and list no unexpected GPU process.  Global medians are
130.917/126.277/116.707 us and their start-skew medians are
63.929/59.297/44.552 us.  rank 1 starts first in all 300 samples, preserving
the E2 control-plane signature.

Maximum rank-local call-envelope medians are much tighter at
76.967/75.243/79.364 us, a 5.48% range.  However, their run maxima are
472.997/224.434/630.329 us and pooled p99 is 168.538 us.  These tails are
retained, not trimmed.  Automatic eligibility is necessary but does not make
the manual p95/p99 stability gate pass.

Reports and a verified partial manifest live under
`.cache/rail_balance/c100/formal-omp1/bac80ee/`.  No benchmark, CUDA/JIT,
runtime or production hot-path code changed.  Return-H256 is the next group.

## 2026-07-22 — D072: return-H256 collection, co-tenant rejection, and recovery

The first clean return-H256 10+100 run at `d05411a` passed.  During the next
run, an independently launched Megatron `restart7` appeared after preflight:
outer launcher PGID 1097012, torchrun PID 1097020, and GPU worker PIDs
1097155--1097158.  The benchmark completed functionality but its post-run
identity gate found all four unexpected workers and set
`baseline_collection_eligible=false`.  That r2 report is retained with hash
`521dcabfa7afedd0f37ae511c17e9fb5b81c51bb5949e48de0ba891c47758c53`;
it is not renamed, deleted, pooled, or used in comparisons.

Under the standing user authorization, only exact PGID 1097012 received
TERM/KILL cleanup.  A following process-group query and `nvidia-smi`
compute-app query were empty.  Two replacement runs then passed every gate.
The accepted set and hashes are:

```text
906a81597b3afdeb0aadd10cf4d888f563216315eb3c4e597b2a420dc43f17fb  return-h256-r1.json
719a8451b31baeff82f7f6af00bd7000aefcf3e5bd4204b1c9b64a4fa82fd92d  return-h256-r3.json
17e0b24bd937fc387706b013618ef6440fc2318a85b7be9dc01a68a9db31a2c5  return-h256-r4.json
```

Their pooled global median/p95/p99 are 218.522/259.578/453.262 us; the
rank-local maximum-envelope values are 215.020/249.024/406.292 us.  Run
medians are comparatively close, but r3 reaches 1419.432 us globally and
1324.364 us rank-locally, so manual tail stability remains open.

One checksum verification command was first run from the repository root even
though the manifest contains directory-relative names; it failed only with
four `No such file or directory` messages.  Re-running from the manifest
directory verified all four reports `OK`.  This path-resolution failure is
retained to prevent confusing it with data corruption.

A first multi-file documentation patch also failed atomically because the
expected final Optimization Log line wrapped differently from the patch
context.  `git status` and a marker search proved that no tracked file changed;
the updates were then split per file.  No partial documentation state was
silently accepted.

No CUDA/JIT/runtime/Hybrid hot-path code and no profiler configuration changed.
Return-H7168 is next; Nsys remains the first diagnostic profiler.

Post-commit independent review reports Blocker0/High0/Medium0/Low1.  The only
Low was an ambiguous C100 checkpoint label: 406.292/1324.364 us are the pooled
rank-local-envelope p99/max, whereas global values are 453.262/1419.432 us.
The follow-up documentation commit makes both scopes explicit.

## 2026-07-22 — D073: complete return-H7168 profiler-free collection

At clean `57b69a9`, return-H7168 r1 passed with global median/p95
341.347/370.748 us.  The next process completed functionality but the automatic
post-run gate found eight new compute applications.  The retained process
table resolves them to a separate
`torchrun --standalone --nproc_per_node=8 dlb_ep8_dispatch_demo.py
--tokens-per-rank 2048 --experts-per-rank 32` invocation.  Its worker PIDs are
1155497/1155498/1155499/1155500/1155501/1155504/1155506/1155508.  r2 is
therefore permanently rejected despite otherwise valid Git, JIT, GPU, MPS and
measurement-depth gates.

The demo was already gone when diagnosed.  Its parent traced to another Codex
app-server session, not Megatron, so no kill authorization was inferred and no
unrelated process was terminated.  Empty-GPU preflights then preceded clean r3
and r4.  The accepted set `{r1,r3,r4}` has pooled global
median/p95/p99/max 337.631/363.655/528.122/2021.557 us and pooled rank-local
334.003/359.490/503.465/2008.428 us.  Global and rank-local run-median ranges
are 2.37% and 2.12%, while the retained tails remain large.

Accepted hashes are:

```text
995a543a76320cb4de069fc8397def08c558ecaabca253eab144babd4a26e897  return-h7168-r1.json
5c1239ee5cf072728c7aa9c1d4cca77c716d74eecb897658c7251d282fb87a5b  return-h7168-r3.json
ffc08d7a8f623a68814eb874a415cfac7919c6f35e4bdd9294bb80fa82367c69  return-h7168-r4.json
```

Rejected r2 remains under hash
`428947471c62f5ef8ce7c9daa4593f04590be278b53f559e5cd53bd372ed7402`.
The manifest verifies all four files.  No benchmark, CUDA/JIT/runtime/Hybrid
hot path, Nsys or NCU configuration changed.

Independent documentation review reports Blocker0/High0/Medium1/Low1.  The
Medium was an active HANDOFF sentence that still said return-H7168 remained;
the Low was a PID range that could imply nonexistent intermediate PIDs.  The
handoff now points to the completed group and the evidence file lists all eight
actual PIDs explicitly.

## 2026-07-22 — D074: freeze the installed Nsys contract before instrumentation

No profile was started in this slice.  The installed CLI is Nsys
2024.6.2.225.  Read-only help and installation inspection prove that
`cuda,nvtx,osrt`, 39 stats reports and seven analyze rules are available.  The
33 installed recipes are not currently runnable because their Python
environment lacks `pandas`; stats/analyze and SQLite export remain usable.
There is no separate `nccl` trace collector in this version.

Retained command/tool failures are: the first benchmark `--help` invocation
lacked the frozen `PYTHONPATH` and raised `ModuleNotFoundError: deep_ep`; the
corrected command passed.  `nsys stats --help-reports` and
`nsys analyze --help-rules` print valid lists but return 1.  The attempted
`nsys profile --trace=help` form is invalid; `--help=trace` is the installed
syntax.  Representative recipe probes all fail with the same missing-pandas
error, not missing trace support.

Static audit confirms the benchmark has no profiler CLI or NVTX range and its
report intentionally declares profiling disabled.  A minimal benchmark-only
`--nvtx` mode is therefore the next code slice.  It will fail closed for
baseline eligibility, delimit only the steady window, and add sparse phase and
invocation ranges.  No device kernel or production runtime edit is authorized.

## 2026-07-22 — D075: add and harden the steady-only NVTX diagnostic mode

The benchmark now accepts a default-off `--nvtx` flag.  Rank 0 owns one
`c100_nsys_window` around all steady iterations; all ranks expose coarse
prepare/finish/prerequisite/stage/visibility/abort ranges, and target-stage
ranges include category plus ordinal.  Cold JIT and warmup remain outside the
capture trigger.  The watchdog forwards the flag, semantic identity includes
it, and every diagnostic report is automatically baseline-ineligible.

Independent review found Blocker0/High1/Medium1/Low1.  The High is an Nsys
command contract: this installed version defaults capture end to
`stop-shutdown` and kill to `sigterm`, so the accepted command explicitly uses
`--capture-range-end=stop --kill=none`.  The Medium was unmatched outer-range
cleanup on an exceptional steady iteration.  Start/stop now use converged
WORLD gates, and worker finally performs rank0-only best-effort pop without a
collective.  The Low correctly rejects a strict zero-overhead claim: the
target-stage callable/timer boundary is unchanged, but the wider Python
transaction executes minimal disabled branches.

In-memory compile, CLI help and `git diff --check` pass.  Two true EP8 H256
return smokes pass: one with NVTX and one default-off.  The NVTX report records
`nvtx_diagnostic`, semantic flag true, one forwarded `--nvtx`, and automatic
eligibility false.  The default report records `disabled`, semantic flag
false, no forwarded flag, and is ineligible only because the tree is dirty and
the 0+1 smoke lacks formal depth.  Both leave no GPU process.  No Nsys profile
has yet started and no device/production hot path changed.

## 2026-07-22 — D076: final-commit smoke and another co-tenant rejection

Commit `feaf03e` is pushed and clean.  Its first default-off 0+1 smoke passed
functionality but the report gate found eight processes from a concurrent
`torchrun --standalone --nproc-per-node=8 dlb_nvshmem_tester.py` that appeared
after the shell preflight.  They were present in the benchmark pre-snapshot and
gone at post-snapshot.  The report is retained and rejected; the tester had
already exited and no unrelated Codex process was killed.  A validation script
that required an empty unexpected-PID list failed at this report as intended.

The clean retry and clean NVTX smoke both pass with no unexpected compute
process.  The retry records profiler mode disabled, semantic NVTX false and no
forwarded flag.  The NVTX smoke records `nvtx_diagnostic`, semantic NVTX true
and exactly one forwarded flag.  Both are intentionally non-baseline 0+1
reports.  Verified hashes are:

```text
82b58266c5a508c6f39f8fffc77ca8e9620e64cc847f82dcc6daabf7bdeeae39  default-off-smoke.json  # rejected co-tenant
b99c7991334314aa96f5c222e29f539e60b8bd64c665a01559890ccd14134b43  default-off-smoke-r2.json
2445b8467fed377cfacfe4ebbc46038082cd7752f25c1b9274d0ef8f5d5d7b94  nvtx-smoke.json
```

The local manifest verifies all three.  GPUs are empty and the first actual
Nsys range-trigger smoke is next.
