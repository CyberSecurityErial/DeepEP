"""C080-D: strict 8-GPU Hybrid source-shuffle correctness test.

This test deliberately exercises only the new node-local data path.  Eight
physical GPUs form one source node; destination servers remain a logical
namespace.  A compact B2 plan is built from real expert IDs, then every moved
``(token, destination)`` copy is written directly into its target egress
GPU's legacy Hybrid proxy-dispatch slot through LSA.

There is no manifest, descriptor, ready flag, or ring in this contract.  The
test parses the raw legacy BF16 ``TokenLayout`` bytes and compares them with
``enumerate_resolved_copies``.  A snapshot taken before the shuffle also
proves that slots outside the dense active prefix remain byte-for-byte
unchanged without imposing a production memset.

CPU contract::

    PYTHONPATH=. python -B \
      tests/elastic/test_rail_balance_hybrid_shuffle_lsa.py --oracle-only

Strict H200 path::

    PYTHONPATH=. EP_DISABLE_GIN=1 \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      python -B tests/elastic/test_rail_balance_hybrid_shuffle_lsa.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from rail_balance_hybrid_reference import (
    HybridRailSchedule,
    ResolvedHybridDestinationCopy,
    enumerate_resolved_copies,
    map_topk_experts_to_destinations,
)
from test_rail_balance_hybrid_plan_lsa import (
    PlanCase,
    _abort,
    _finish,
    _gate_signature,
    _gather_objects,
    _local_topk,
    _monitored_barrier,
    _prepare,
    _schedule,
    _skew_case,
    _verify_outputs,
)


_WORLD_SIZE = 8
_HIDDEN = 256
_NUM_TOPK = 4
_MAX_TOKENS = 10
_PROXY_CAPACITY = 8
_TMA_ALIGNMENT = 32
_OPERATION = "rail_balance_hybrid_source_shuffle"
_TYPE = TypeVar("_TYPE")


def _align(value: int, alignment: int = _TMA_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def _mapped_destinations(
    case: PlanCase,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return map_topk_experts_to_destinations(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=case.num_scaleout_ranks,
        local_scaleout_rank=case.local_scaleout_rank,
    )


def _moved_copies(
    case: PlanCase,
    schedule: HybridRailSchedule,
) -> tuple[ResolvedHybridDestinationCopy, ...]:
    return tuple(
        record
        for record in enumerate_resolved_copies(
            _mapped_destinations(case), schedule)
        if record.resolution.moved
    )


def _copies_by_egress(
    records: Sequence[ResolvedHybridDestinationCopy],
) -> tuple[dict[int, ResolvedHybridDestinationCopy], ...]:
    result: list[dict[int, ResolvedHybridDestinationCopy]] = [
        {} for _ in range(_WORLD_SIZE)
    ]
    for record in records:
        resolution = record.resolution
        slots = result[resolution.egress]
        assert resolution.proxy_slot not in slots
        slots[resolution.proxy_slot] = record
    return tuple(result)


def _assert_shuffle_oracle() -> tuple[PlanCase, PlanCase]:
    first = _skew_case(
        capacity=_PROXY_CAPACITY,
        seed=62,
        name="c061_source_shuffle_seed62",
    )
    second = replace(
        first,
        name="c061_source_shuffle_reuse_seed63",
        remainder_seed=63,
    )
    assert first.num_topk == second.num_topk == _NUM_TOPK
    assert first.num_max_tokens_per_rank == _MAX_TOKENS
    assert first.num_tokens_per_rank == (9, 9, 0, 0, 0, 0, 0, 0)

    for case in (first, second):
        schedule = _schedule(case)
        records = _moved_copies(case, schedule)
        by_egress = _copies_by_egress(records)
        assert schedule.enabled and schedule.moved_copies == len(records)
        assert schedule.moved_copies > 0
        assert max(schedule.proxy_required) < case.proxy_capacity_per_egress
        for egress, slots in enumerate(by_egress):
            required = schedule.proxy_required[egress]
            assert set(slots) == set(range(required))
            for proxy_slot, record in slots.items():
                resolution = record.resolution
                assert resolution.egress == egress
                assert resolution.proxy_slot == proxy_slot
                assert proxy_slot == (
                    schedule.group_prefix[egress][resolution.channel][
                        record.copy.destination]
                    + resolution.incoming_ordinal
                    - schedule.moved_channel_prefix[egress][
                        record.copy.destination][resolution.channel]
                )

        # The C061 route is not merely one-hot: one physical owner token can
        # contribute multiple independently balanced destination copies.
        multiplicity = Counter(
            (record.copy.owner, record.copy.token) for record in records)
        assert max(multiplicity.values()) >= 2
        multi = [
            key for key, count in multiplicity.items() if count >= 2
        ]
        assert any(
            len({
                record.copy.destination
                for record in records
                if (record.copy.owner, record.copy.token) == key
            }) >= 2
            for key in multi
        )
    return first, second


def _cpu_source_inputs(
    case: PlanCase,
    owner: int,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = len(case.topk_idx[owner])
    linear = torch.arange(
        num_tokens * _HIDDEN, dtype=torch.int32, device="cpu")
    x = (
        linear.reshape(num_tokens, _HIDDEN) % 29
        + owner * 31
        + iteration * 3
    ).to(torch.bfloat16).contiguous()
    token = torch.arange(
        num_tokens, dtype=torch.float32, device="cpu")[:, None]
    lane = torch.arange(
        case.num_topk, dtype=torch.float32, device="cpu")[None, :]
    weights = (
        iteration * 10000.0 + owner * 1000.0 + token * 32.0 + lane
    ).contiguous()
    assert x.shape == (num_tokens, _HIDDEN)
    assert weights.shape == (num_tokens, case.num_topk)
    return x, weights


def _local_source_inputs(
    case: PlanCase,
    rank: int,
    iteration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, weights = _cpu_source_inputs(case, rank, iteration)
    device = torch.device("cuda", rank)
    return x.to(device), weights.to(device)


def _token_layout(case: PlanCase) -> tuple[int, int, int, int, int]:
    hidden_bytes = _HIDDEN * torch.bfloat16.itemsize
    metadata_offset = _align(hidden_bytes)
    topk_offset = metadata_offset
    weights_offset = topk_offset + case.num_topk * 4
    src_global_offset = weights_offset + case.num_topk * 4
    linked_offset = src_global_offset + 4
    metadata_bytes = case.num_topk * (4 + 4) + (1 + case.num_topk) * 4
    token_bytes = _align(hidden_bytes) + _align(metadata_bytes)
    assert linked_offset + case.num_topk * 4 <= token_bytes
    return (
        topk_offset,
        weights_offset,
        src_global_offset,
        linked_offset,
        token_bytes,
    )


def _reinterpret(
    row: torch.Tensor,
    offset: int,
    count: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_bytes = count * torch.empty((), dtype=dtype).element_size()
    return row.narrow(0, offset, num_bytes).contiguous().view(dtype)


def _verify_local_snapshot(
    *,
    case: PlanCase,
    schedule: HybridRailSchedule,
    records: Sequence[ResolvedHybridDestinationCopy],
    rank: int,
    baseline: torch.Tensor,
    snapshot: torch.Tensor,
    iteration: int,
) -> tuple[tuple[int, ...], ...]:
    assert baseline.device.type == snapshot.device.type == "cpu"
    assert baseline.dtype == snapshot.dtype == torch.uint8
    assert baseline.is_contiguous() and snapshot.is_contiguous()
    topk_offset, weights_offset, src_offset, linked_offset, token_bytes = \
        _token_layout(case)
    assert tuple(baseline.shape) == tuple(snapshot.shape) == (
        case.proxy_capacity_per_egress, token_bytes)

    by_slot = _copies_by_egress(records)[rank]
    required = schedule.proxy_required[rank]
    assert set(by_slot) == set(range(required))
    assert required < case.proxy_capacity_per_egress
    assert torch.equal(snapshot[required:], baseline[required:])

    coverage = []
    for proxy_slot in range(required):
        record = by_slot[proxy_slot]
        copy = record.copy
        resolution = record.resolution
        assert resolution.egress == rank
        assert resolution.proxy_slot == proxy_slot
        assert proxy_slot < schedule.proxy_required[rank]
        assert proxy_slot == (
            schedule.group_prefix[rank][resolution.channel][copy.destination]
            + resolution.incoming_ordinal
            - schedule.moved_channel_prefix[rank][copy.destination][
                resolution.channel]
        )

        row = snapshot[proxy_slot]
        expected_x, expected_weights = _cpu_source_inputs(
            case, copy.owner, iteration)
        hidden = _reinterpret(row, 0, _HIDDEN, torch.bfloat16)
        topk_idx = _reinterpret(
            row, topk_offset, case.num_topk, torch.int32)
        topk_weights = _reinterpret(
            row, weights_offset, case.num_topk, torch.float32)
        src_global = int(_reinterpret(
            row, src_offset, 1, torch.int32)[0].item())
        linked = _reinterpret(
            row, linked_offset, case.num_topk, torch.int32)

        assert torch.equal(hidden, expected_x[copy.token])
        assert topk_idx.tolist() == list(case.topk_idx[copy.owner][copy.token])
        assert torch.equal(topk_weights, expected_weights[copy.token])
        assert src_global == copy.owner * case.num_max_tokens_per_rank + copy.token
        assert linked.tolist() == [proxy_slot] + [-1] * (case.num_topk - 1)
        metadata_end = linked_offset + case.num_topk * 4
        assert torch.count_nonzero(row[metadata_end:token_bytes]).item() == 0
        coverage.append((
            copy.owner,
            copy.token,
            copy.destination,
            resolution.source_channel,
            resolution.owner_ordinal,
            rank,
            resolution.channel,
            resolution.remote_slot,
            proxy_slot,
        ))
    return tuple(coverage)


def _checked_phase(
    label: str,
    control_group: dist.ProcessGroup,
    function: Callable[[], _TYPE],
) -> _TYPE:
    value = None
    error = None
    try:
        value = function()
    except BaseException:
        error = traceback.format_exc()
    errors = _gather_objects(error, control_group)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(errors)
        if message is not None
    ]
    if messages:
        raise AssertionError(f"{label} failed:\n" + "\n".join(messages))
    return value  # type: ignore[return-value]


def _snapshot(
    runtime: object,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    def call() -> torch.Tensor:
        value = runtime._rail_balance_hybrid_proxy_dispatch_snapshot(  # type: ignore[attr-defined]
            invocation_id)
        assert isinstance(value, torch.Tensor)
        return value

    if stream is None:
        value = call()
    else:
        with torch.cuda.stream(stream):
            value = call()
        # The snapshot copy is queued on ``stream``.  Do not consume the CUDA
        # tensor from the caller's default stream without an explicit edge.
        stream.synchronize()
    assert value.is_cuda and value.dtype == torch.uint8
    assert value.is_contiguous() and value.ndim == 2
    return value.cpu().contiguous()


def _source_shuffle(
    runtime: object,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    invocation_id: int,
    stream: torch.cuda.Stream | None,
) -> None:
    def call() -> None:
        result = runtime._rail_balance_hybrid_source_shuffle(  # type: ignore[attr-defined]
            x, topk_weights, invocation_id)
        assert result is None

    if stream is None:
        call()
    else:
        with torch.cuda.stream(stream):
            call()


def _run_shuffle_transaction(
    *,
    runtime: object,
    rank: int,
    control_group: dist.ProcessGroup,
    timeout: int,
    case: PlanCase,
    arena_offset: int,
    arena_bytes: int,
    invocation_id: int,
    iteration: int,
    nondefault_stream: bool,
) -> None:
    schedule = _schedule(case)
    records = _moved_copies(case, schedule)
    topk_idx = _local_topk(case, rank)
    x, topk_weights = _local_source_inputs(case, rank, iteration)
    source_topk = topk_idx.cpu()
    source_x = x.cpu()
    source_weights = topk_weights.cpu()
    stream = torch.cuda.Stream(device=rank) if nondefault_stream else None
    if stream is not None:
        # Inputs were constructed on the caller's current stream.  Make the
        # private transaction's stream dependency explicit instead of relying
        # on small fixtures completing before the alternate stream starts.
        stream.wait_stream(torch.cuda.current_stream(rank))

    try:
        status, prepare_error = _prepare(
            runtime,
            topk_idx,
            case,
            arena_offset=arena_offset,
            invocation_id=invocation_id,
            stream=stream,
        )
        gate1 = _gather_objects((
            _OPERATION,
            1,
            invocation_id,
            status,
            _gate_signature(
                case, arena_offset=arena_offset, arena_bytes=arena_bytes),
            int(topk_idx.shape[0]),
            prepare_error,
        ), control_group)
        assert tuple(int(item[3]) for item in gate1) == (0,) * _WORLD_SIZE
        assert len({item[4] for item in gate1}) == 1
        assert tuple(int(item[5]) for item in gate1) == \
            case.num_tokens_per_rank

        outputs, finish_status, finish_error = _finish(
            runtime, invocation_id, stream)
        gate2 = _gather_objects((
            _OPERATION,
            2,
            invocation_id,
            finish_status,
            finish_error,
        ), control_group)
        assert tuple(int(item[3]) for item in gate2) == (0,) * _WORLD_SIZE
        assert outputs is not None
        _checked_phase(
            "exact compact plan",
            control_group,
            lambda: _verify_outputs(outputs, schedule, rank),
        )

        baseline = _checked_phase(
            "pre-shuffle proxy snapshot",
            control_group,
            lambda: _snapshot(runtime, invocation_id, stream),
        )
        _checked_phase(
            "direct-final LSA source shuffle",
            control_group,
            lambda: _source_shuffle(
                runtime, x, topk_weights, invocation_id, stream),
        )

        # The preceding checked phase is the host WORLD agreement that every
        # producer returned.  Enter the existing device barrier on every rank
        # in the same order to make peer LSA publication visible before any
        # egress snapshots its proxy slots.  This is test-only synchronization;
        # the production shuffle core deliberately contains no extra barrier.
        _checked_phase(
            "post-shuffle LSA visibility barrier",
            control_group,
            lambda: runtime.barrier(True, True, True),  # type: ignore[attr-defined]
        )
        snapshot = _checked_phase(
            "post-shuffle proxy snapshot",
            control_group,
            lambda: _snapshot(runtime, invocation_id, stream),
        )
        local_coverage = _checked_phase(
            "raw legacy TokenLayout comparison",
            control_group,
            lambda: _verify_local_snapshot(
                case=case,
                schedule=schedule,
                records=records,
                rank=rank,
                baseline=baseline,
                snapshot=snapshot,
                iteration=iteration,
            ),
        )
        coverage = _gather_objects(local_coverage, control_group)

        def verify_global_coverage() -> None:
            observed = sorted(
                item for rank_items in coverage for item in rank_items)
            expected = sorted((
                record.copy.owner,
                record.copy.token,
                record.copy.destination,
                record.resolution.source_channel,
                record.resolution.owner_ordinal,
                record.resolution.egress,
                record.resolution.channel,
                record.resolution.remote_slot,
                record.resolution.proxy_slot,
            ) for record in records)
            assert observed == expected
            assert len(observed) == len(set(observed)) == schedule.moved_copies

        _checked_phase(
            "global moved-copy coverage", control_group,
            verify_global_coverage)

        def verify_sources_unchanged() -> None:
            assert torch.equal(topk_idx.cpu(), source_topk)
            assert torch.equal(x.cpu(), source_x)
            assert torch.equal(topk_weights.cpu(), source_weights)

        _checked_phase(
            "source tensors remain immutable",
            control_group,
            verify_sources_unchanged,
        )
    finally:
        _abort(runtime, invocation_id, control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int,
            args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=81)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    buffer = None
    clean_shutdown = False
    try:
        first, second = _assert_shuffle_oracle()
        layout = tuple(int(value) for value in
                       _C._get_rail_balance_hybrid_layout(
                           _HIDDEN, _NUM_TOPK, _PROXY_CAPACITY))
        assert len(layout) == 10
        assert layout[5] == _token_layout(first)[-1]
        arena_bytes = layout[-1]
        alignment = int(_C.get_elastic_buffer_alignment())
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=_MAX_TOKENS,
            hidden=_HIDDEN,
            num_topk=_NUM_TOPK,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
        )
        arena_offset = align(base_bytes, alignment)
        assert arena_offset == base_bytes
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=arena_offset + arena_bytes,
            num_max_tokens_per_rank=_MAX_TOKENS,
            hidden=_HIDDEN,
            num_topk=_NUM_TOPK,
            allow_hybrid_mode=False,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=args.timeout,
            num_cpu_timeout_secs=args.timeout,
        )
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
        assert buffer.scaleout_rank_idx == 0
        assert buffer.scaleup_rank_idx == rank

        # Warm the same barrier before any proxy publication.  Its shared
        # workspace phase is monotonic, so every later transaction must enter
        # it exactly once and in the same order on all eight ranks.
        _checked_phase(
            "warm LSA visibility barrier",
            control_group,
            lambda: buffer.runtime.barrier(  # type: ignore[attr-defined]
                True, True, True),
        )

        _run_shuffle_transaction(
            runtime=buffer.runtime,
            rank=rank,
            control_group=control_group,
            timeout=args.timeout,
            case=first,
            arena_offset=arena_offset,
            arena_bytes=arena_bytes,
            invocation_id=881,
            iteration=1,
            nondefault_stream=False,
        )
        _run_shuffle_transaction(
            runtime=buffer.runtime,
            rank=rank,
            control_group=control_group,
            timeout=args.timeout,
            case=second,
            arena_offset=arena_offset,
            arena_bytes=arena_bytes,
            invocation_id=882,
            iteration=2,
            nondefault_stream=True,
        )

        _monitored_barrier(control_group, args.timeout)
        clean_shutdown = True
    finally:
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            _monitored_barrier(control_group, args.timeout)
            if rank == 0:
                print(
                    "PASS C080-D Hybrid source shuffle: true 8-GPU LSA, "
                    "C061 skew/zero-N/multi-destination/multi-copy, exact "
                    "legacy TokenLayout bytes, unused unchanged, reusable "
                    "transaction, non-default stream",
                    flush=True,
                )
            dist.destroy_process_group()


def _run_watchdog(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        "--num-processes", str(arguments.num_processes),
        "--timeout", str(arguments.timeout),
        "--master-port", str(arguments.master_port),
        "--watchdog-seconds", str(arguments.watchdog_seconds),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=arguments.watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "C080-D subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-D strict 8-GPU Hybrid source-shuffle test")
    if not __debug__:
        parser.error("C080-D correctness checks require Python assertions")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29883)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-D requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    first, second = _assert_shuffle_oracle()
    assert _schedule(first).moved_copies == 21
    print(
        "PASS C080-D CPU oracle: C061-like 8-rail source shuffle, "
        "zero-N owners, multi-destination/multi-copy, moved=21, "
        f"Pcap={_PROXY_CAPACITY}, seeds="
        f"{first.remainder_seed}/{second.remainder_seed}",
        flush=True,
    )
    if arguments.oracle_only:
        return

    required = (
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
        "_rail_balance_hybrid_source_shuffle",
        "_rail_balance_hybrid_proxy_dispatch_snapshot",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        parser.error(
            "rebuild the extension with the C080-D private API; missing "
            + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-D requires at least 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(arguments.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    if not arguments.worker_suite:
        _run_watchdog(arguments)
        return
    torch.multiprocessing.spawn(
        _worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
