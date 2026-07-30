"""True 8-GPU LSA smoke test for endpoint-aware source shuffle."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.envs import init_dist
from deep_ep.utils.math import align
from test_rail_balance_hybrid_plan_lsa import (
    PlanCase,
    _abort,
    _finish,
    _gather_objects,
    _local_topk,
    _monitored_barrier,
)
from test_rail_balance_hybrid_shuffle_lsa import (
    _checked_phase,
    _cpu_source_inputs,
    _local_source_inputs,
    _reinterpret,
    _snapshot,
    _source_shuffle,
    _token_layout,
)


_WORLD_SIZE = 8
_HIDDEN = 256
_NUM_TOPK = 4
_NUM_TOKENS = 8
_NUM_CHANNELS = 2
_PROXY_CAPACITY = 8


def _case() -> PlanCase:
    rows = tuple(
        tuple(8 + (token + lane + 1) % _WORLD_SIZE
              for lane in range(_NUM_TOPK))
        for token in range(_NUM_TOKENS)
    )
    return PlanCase(
        name="hop_one_hop_owner0_hot",
        topk_idx=(rows,) + ((),) * (_WORLD_SIZE - 1),
        num_topk=_NUM_TOPK,
        num_channels=_NUM_CHANNELS,
        num_max_tokens_per_rank=_NUM_TOKENS,
        num_experts=2 * _WORLD_SIZE,
        num_scaleout_ranks=2,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=_PROXY_CAPACITY,
        remainder_seed=0,
    )


def _decode_record(word: int) -> tuple[int, int]:
    mask = word & 0xffffffff
    destination = (word >> 32) & 0xffffffff
    if destination & 0x80000000:
        destination -= 1 << 32
    return destination, mask


def _verify_snapshot(
    case: PlanCase,
    rank: int,
    plan: tuple[torch.Tensor, ...],
    before: torch.Tensor,
    after: torch.Tensor,
) -> tuple[tuple[int, int, int, int], ...]:
    records, resolutions = plan[:2]
    proxy_required = plan[7]
    moved_copies, status = plan[9:11]
    assert status.tolist() == [0]
    assert moved_copies.item() > 0
    required = int(proxy_required[rank].item())
    assert torch.equal(after[required:], before[required:])

    expected = {}
    for owner in range(_WORLD_SIZE):
        for token in range(_NUM_TOKENS):
            for slot in range(_NUM_TOPK):
                destination, mask = _decode_record(
                    int(records[owner, token, slot].item())
                )
                if mask == 0:
                    continue
                egress, _channel, _remote, proxy = (
                    int(value) for value in resolutions[owner, token, slot]
                )
                assert egress in {owner, *(
                    target for target in range(_WORLD_SIZE)
                    if mask & (1 << target)
                )}
                if egress == rank and egress != owner:
                    assert proxy >= 0
                    expected[proxy] = (owner, token, destination, mask)
    assert set(expected) == set(range(required))

    topk_offset, weights_offset, src_offset, linked_offset, token_bytes = \
        _token_layout(case, _HIDDEN)
    assert after.shape == (_PROXY_CAPACITY, token_bytes)
    sources = {}
    coverage = []
    for proxy in range(required):
        owner, token, destination, _mask = expected[proxy]
        if owner not in sources:
            sources[owner] = _cpu_source_inputs(case, owner, 1, _HIDDEN)
        expected_x, expected_weights = sources[owner]
        row = after[proxy]
        assert torch.equal(
            _reinterpret(row, 0, _HIDDEN, torch.bfloat16),
            expected_x[token],
        )
        assert _reinterpret(
            row, topk_offset, _NUM_TOPK, torch.int32
        ).tolist() == list(case.topk_idx[owner][token])
        assert torch.equal(
            _reinterpret(row, weights_offset, _NUM_TOPK, torch.float32),
            expected_weights[token],
        )
        assert int(_reinterpret(row, src_offset, 1, torch.int32)[0]) == \
            owner * _NUM_TOKENS + token
        assert _reinterpret(
            row, linked_offset, _NUM_TOPK, torch.int32
        ).tolist() == [proxy, -1, -1, -1]
        coverage.append((owner, token, destination, rank))
    return tuple(coverage)


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_local_ranks, seed=93
    )
    control_group = dist.new_group(
        ranks=list(range(world_size)), backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE
    case = _case()
    layout = tuple(int(value) for value in
                   _C._get_rail_balance_hybrid_layout(
                       _HIDDEN, _NUM_TOPK, _PROXY_CAPACITY))
    alignment = int(_C.get_elastic_buffer_alignment())
    base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
        ep_group,
        num_max_tokens_per_rank=_NUM_TOKENS,
        hidden=_HIDDEN,
        num_topk=_NUM_TOPK,
        use_fp8_dispatch=False,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
    )
    arena_offset = align(base_bytes, alignment)
    buffer = deep_ep.ElasticBuffer(
        ep_group,
        num_bytes=arena_offset + layout[-1],
        num_max_tokens_per_rank=_NUM_TOKENS,
        hidden=_HIDDEN,
        num_topk=_NUM_TOPK,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.timeout,
        num_cpu_timeout_secs=args.timeout,
    )
    invocation_id = 971
    try:
        topk_idx = _local_topk(case, rank)
        x, weights = _local_source_inputs(case, rank, 1, _HIDDEN)
        status = int(buffer.runtime._rail_balance_hybrid_plan_prepare(
            topk_idx, _HIDDEN, _NUM_CHANNELS, _NUM_TOKENS,
            case.num_experts, case.num_scaleout_ranks,
            case.local_scaleout_rank, _PROXY_CAPACITY,
            arena_offset, invocation_id, 0, 0, 0, True,
        ))
        assert _gather_objects(status, control_group) == [0] * _WORLD_SIZE
        _outputs, finish_status, finish_error = _finish(
            buffer.runtime, invocation_id, None
        )
        assert _gather_objects(
            (finish_status, finish_error), control_group
        ) == [(0, None)] * _WORLD_SIZE
        plan = tuple(
            tensor.cpu() for tensor in
            buffer.runtime._rail_balance_hop_plan_snapshot(invocation_id)
        )
        before = _snapshot(buffer.runtime, invocation_id, None)
        _checked_phase(
            "hop source shuffle", control_group,
            lambda: _source_shuffle(
                buffer.runtime, x, weights, invocation_id, None),
        )
        _checked_phase(
            "hop LSA visibility", control_group,
            lambda: buffer.runtime.barrier(True, True, True),
        )
        after = _snapshot(buffer.runtime, invocation_id, None)
        local = _verify_snapshot(case, rank, plan, before, after)
        observed = sorted(
            item for rank_items in _gather_objects(local, control_group)
            for item in rank_items
        )
        assert len(observed) == len(set(observed)) == int(plan[9].item())
    finally:
        _abort(buffer.runtime, invocation_id, control_group, args.timeout)
        buffer.destroy()
        _monitored_barrier(control_group, args.timeout)
        if rank == 0:
            print(
                "PASS hop-aware source shuffle: one-hop endpoint plan, "
                "true 8-GPU LSA, exact static proxy payload bytes",
                flush=True,
            )
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--watchdog-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.worker:
        torch.multiprocessing.spawn(
            _worker, args=(_WORLD_SIZE, args), nprocs=_WORLD_SIZE, join=True
        )
        return

    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
               "--timeout", str(args.timeout),
               "--watchdog-seconds", str(args.watchdog_seconds)]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.watchdog_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        raise RuntimeError("hop-aware LSA watchdog expired")
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
