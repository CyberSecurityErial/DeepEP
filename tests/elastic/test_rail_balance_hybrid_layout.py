"""Independent C080-A golden for the force-only Hybrid arena formula."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.utils.comm import get_nccl_comm_handle
from deep_ep.utils.envs import init_dist


REPO_ROOT = Path(__file__).resolve().parents[2]
LAYOUT_HEADER = (
    REPO_ROOT /
    'deep_ep/include/deep_ep/common/rail_balance_hybrid_layout.cuh')
TMA_ALIGNMENT = 32
BUFFER_ALIGNMENT = 2 * 1024 * 1024
MAX_CHANNELS = 1024
MAX_DESTINATIONS = 32
CONTROL_BYTES = 32


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _token_bytes(hidden: int, num_topk: int, with_metadata: bool) -> int:
    metadata_bytes = num_topk * (4 + 4)
    if with_metadata:
        metadata_bytes += (1 + num_topk) * 4
    return _align(hidden * 2, TMA_ALIGNMENT) + _align(
        metadata_bytes, TMA_ALIGNMENT)


def _reference_layout(hidden: int, num_topk: int,
                      proxy_capacity: int) -> tuple[int, ...]:
    channel_count_offset = _align(CONTROL_BYTES, TMA_ALIGNMENT)
    channel_count_bytes = MAX_CHANNELS * MAX_DESTINATIONS * 4
    proxy_ready_offset = _align(
        channel_count_offset + channel_count_bytes, TMA_ALIGNMENT)
    proxy_dispatch_offset = _align(
        proxy_ready_offset + proxy_capacity * 4, TMA_ALIGNMENT)
    dispatch_token_bytes = _token_bytes(hidden, num_topk, True)
    proxy_return_offset = _align(
        proxy_dispatch_offset + proxy_capacity * dispatch_token_bytes,
        TMA_ALIGNMENT)
    combine_token_bytes = _token_bytes(hidden, num_topk, False)
    raw_bytes = proxy_return_offset + proxy_capacity * combine_token_bytes
    arena_bytes = _align(raw_bytes, BUFFER_ALIGNMENT)
    return (
        0,
        CONTROL_BYTES,
        channel_count_offset,
        channel_count_bytes,
        proxy_dispatch_offset,
        dispatch_token_bytes,
        proxy_return_offset,
        combine_token_bytes,
        raw_bytes,
        arena_bytes,
    )


def _device_layout(hidden: int, num_topk: int,
                   proxy_capacity: int) -> tuple[int, ...]:
    return tuple(int(value) for value in
                 _C._get_rail_balance_hybrid_layout(
                     hidden, num_topk, proxy_capacity))


def _expect_failure(function) -> None:
    try:
        function()
    except Exception:
        return
    raise AssertionError('expected force Hybrid layout validation failure')


def test_control_block_fields_and_fixed_capacity_are_frozen():
    source = LAYOUT_HEADER.read_text()
    control = re.search(
        r'struct alignas\([^)]*\) HybridControl \{(.*?)\};',
        source,
        flags=re.DOTALL)
    assert control is not None
    fields = re.findall(r'int32_t\s+(\w+)\s*;', control.group(1))
    assert fields == [
        'invocation_id',
        'state',
        'error_code',
        'error_detail',
        'num_channels',
        'num_destinations',
        'proxy_required',
        'reserved',
    ]
    assert 'kNumHybridMaxChannels = deep_ep::kNumMaxChannels' in source
    assert 'kNumHybridMaxDestinations = 32' in source


def test_cpp_layout_matches_independent_formula_and_goldens():
    goldens = {
        (256, 1, 1):
            (0, 32, 32, 131072, 131136, 544,
             131680, 544, 132224, 2097152),
        (1024, 4, 32):
            (0, 32, 32, 131072, 131232, 2112,
             198816, 2080, 265376, 2097152),
        (7168, 8, 32):
            (0, 32, 32, 131072, 131232, 14464,
             594080, 14400, 1054880, 2097152),
    }
    for arguments, expected in goldens.items():
        assert _reference_layout(*arguments) == expected
        assert _device_layout(*arguments) == expected


def test_proxy_capacity_is_per_egress_and_each_arena_has_pcap_slots():
    hidden, num_topk = 7168, 8
    one = _device_layout(hidden, num_topk, 1)
    thirty_two = _device_layout(hidden, num_topk, 32)
    ready_growth = thirty_two[4] - one[4]
    assert thirty_two[6] - one[6] == ready_growth + 31 * one[5]
    assert thirty_two[8] - one[8] == (
        ready_growth + 31 * (one[5] + one[7]))
    assert thirty_two == _reference_layout(hidden, num_topk, 32)


def test_alignment_and_checked_large_capacity_contract():
    values = _device_layout(7168, 32, (1 << 31) - 1)
    assert values == _reference_layout(7168, 32, (1 << 31) - 1)
    assert values[2] % TMA_ALIGNMENT == 0
    assert values[4] % TMA_ALIGNMENT == 0
    assert values[6] % TMA_ALIGNMENT == 0
    assert values[9] % BUFFER_ALIGNMENT == 0
    assert values[8] <= values[9] < values[8] + BUFFER_ALIGNMENT
    assert values[9] < (1 << 63)


def test_invalid_layout_domains_fail_before_pointer_use():
    invalid = (
        (0, 1, 1),
        (255, 1, 1),
        (256, 0, 1),
        (256, 33, 1),
        (256, 1, 0),
        (256, 1, -1),
    )
    for arguments in invalid:
        _expect_failure(
            lambda arguments=arguments: _device_layout(*arguments))


def _distributed_buffer_size_loop(local_rank: int, num_processes: int) -> None:
    _rank, world_size, group = init_dist(local_rank, num_processes)
    nccl_comm = get_nccl_comm_handle(group).get()
    arguments = (128, 7168, 8)
    proxy_capacity = 32
    legacy_bytes = int(_C.calculate_elastic_buffer_size(
        nccl_comm, *arguments, False, True, True))
    force_bytes = int(_C._calculate_rail_balance_hybrid_buffer_size(
        nccl_comm, *arguments, proxy_capacity))
    arena_bytes = _device_layout(
        arguments[1], arguments[2], proxy_capacity)[-1]
    public_off_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
        group,
        num_max_tokens_per_rank=arguments[0],
        hidden=arguments[1],
        num_topk=arguments[2],
        use_fp8_dispatch=False,
        allow_hybrid_mode=True,
        allow_multiple_reduction=True)
    assert public_off_bytes == legacy_bytes
    assert force_bytes == legacy_bytes + arena_bytes
    assert legacy_bytes % BUFFER_ALIGNMENT == 0
    assert force_bytes % BUFFER_ALIGNMENT == 0

    gathered = [None] * world_size
    dist.all_gather_object(
        gathered, (legacy_bytes, arena_bytes, force_bytes), group=group)
    assert all(value == gathered[0] for value in gathered)
    dist.destroy_process_group()


def run_all():
    tests = (
        test_control_block_fields_and_fixed_capacity_are_frozen,
        test_cpp_layout_matches_independent_formula_and_goldens,
        test_proxy_capacity_is_per_egress_and_each_arena_has_pcap_slots,
        test_alignment_and_checked_large_capacity_contract,
        test_invalid_layout_domains_fail_before_pointer_use,
    )
    for test in tests:
        test()
    print(f'PASS {len(tests)}/{len(tests)} C080-A Hybrid layout tests')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=0)
    arguments = parser.parse_args()
    run_all()
    if arguments.num_processes:
        torch.multiprocessing.spawn(
            _distributed_buffer_size_loop,
            args=(arguments.num_processes,),
            nprocs=arguments.num_processes)
        print('PASS distributed Hybrid buffer formula')
