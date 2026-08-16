"""UltraEP planner + EchoP token-communication integration smoke.

UltraEP owns placement, replicas, weight synchronization, and logical-to-
physical rerouting. EchoP consumes only the final physical expert IDs and may
change the source-node Rail path; it never changes UltraEP's expert decision.
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist


def _import_ultra_ep():
    root = os.environ.get('ULTRA_EP_ROOT')
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        import ultra_ep
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'UltraEP is not installed; set ULTRA_EP_ROOT to its repository '
            'directory before launching this benchmark') from error
    return ultra_ep


def _zipf_counts(total: int, alpha: float, size: int) -> list[int]:
    weights = [(index + 1) ** (-alpha) for index in range(size)]
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(size), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _active_sources(num_nodes: int, sink: int, fan_in: int) -> list[int]:
    return [
        (sink - offset) % num_nodes for offset in range(1, fan_in + 1)
    ]


def _source_totals(
        num_nodes: int, total: int, sink: int, fan_in: int,
        source_alpha: float) -> list[int]:
    result = [0] * num_nodes
    for source, count in zip(
            _active_sources(num_nodes, sink, fan_in),
            _zipf_counts(total, source_alpha, fan_in)):
        result[source] = count
    return result


def _logical_targets_for_rank(
        num_nodes: int, num_rails: int, source_node: int, source_rail: int,
        source_totals: list[int], sink: int, rail_alpha: float,
        expert_alpha: float, seed: int, rail_phase: str) -> list[int]:
    source_total = source_totals[source_node]
    if source_total == 0:
        # Keep every rank on the same API path without adding scale-out traffic.
        return [source_node * num_rails + source_rail]

    rail_counts = _zipf_counts(source_total, rail_alpha, num_rails)
    if rail_phase == 'rotated':
        offset = (source_node * 3 + sink) % num_rails
        rail_counts = [
            rail_counts[(rail - offset) % num_rails]
            for rail in range(num_rails)
        ]
    local_count = rail_counts[source_rail]
    expert_counts = _zipf_counts(source_total, expert_alpha, num_rails)
    targets = [
        sink * num_rails + expert
        for expert, count in enumerate(expert_counts)
        for _ in range(count)
    ]
    random.Random(seed + source_node * 1009 + source_total * 17).shuffle(targets)
    begin = sum(rail_counts[:source_rail])
    return targets[begin:begin + local_count]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _measure_global_us(fn, warmups: int, iterations: int, group) -> dict[str, float]:
    for _ in range(warmups):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        result = fn()
        torch.cuda.synchronize()
        del result

    samples = []
    for _ in range(iterations):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            (time.perf_counter() - start) * 1e6,
            dtype=torch.float64, device='cuda')
        del result
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        samples.append(elapsed.item())
    return {
        'median_us': statistics.median(samples),
        'p90_us': _percentile(samples, 0.90),
        'p99_us': _percentile(samples, 0.99),
        'samples_us': samples,
    }


def _max_mean(values: torch.Tensor) -> float:
    values = values.to(torch.float64)
    mean = values.mean().item()
    return values.max().item() / mean if mean else 0.0


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    if world_size % num_local_ranks:
        raise ValueError('world size must be divisible by local process count')
    num_nodes = world_size // num_local_ranks
    if num_nodes < 2:
        raise ValueError('UltraEP + EchoP smoke requires multiple nodes')
    sink = args.sink if args.sink >= 0 else num_nodes - 1
    if not 0 <= sink < num_nodes:
        raise ValueError(f'sink must be in [0, {num_nodes})')
    if not 1 <= args.fan_in < num_nodes:
        raise ValueError(f'fan-in must be in [1, {num_nodes})')

    node = rank // num_local_ranks
    rail = rank % num_local_ranks
    source_totals = _source_totals(
        num_nodes, args.total, sink, args.fan_in, args.source_alpha)
    logical_targets = _logical_targets_for_rank(
        num_nodes, num_local_ranks, node, rail, source_totals, sink,
        args.rail_alpha, args.expert_alpha, args.seed, args.rail_phase)
    num_tokens = len(logical_targets)
    num_max_tokens_per_rank = max(
        1,
        max(
            max(_zipf_counts(count, args.rail_alpha, num_local_ranks))
            for count in source_totals if count > 0),
    )

    logical_topk = torch.tensor(
        logical_targets, dtype=torch.int64, device='cuda')[:, None]
    routing_map = torch.zeros(
        (num_tokens, world_size), dtype=torch.bool, device='cuda')
    routing_map.scatter_(1, logical_topk, True)
    logical_probs = routing_map.to(torch.float32)

    ultra_ep = _import_ultra_ep()
    manager = ultra_ep.Manager(
        group=group,
        num_layers=1,
        num_local_master_experts=1,
        num_local_redundant_experts=args.redundant_experts_per_rank,
        expert_fc1_numel=args.expert_weight_numel,
        expert_fc2_numel=args.expert_weight_numel,
        is_train=False,
        explicitly_destroy=True,
    )
    num_local_physical = manager.num_local_physical_experts
    num_physical = manager.num_global_physical_experts

    # Register real master buffers so measured weight_sync is not omitted.
    master_id = rank * num_local_physical
    weight_index = torch.arange(
        args.expert_weight_numel, dtype=torch.int64, device='cuda')
    fc1 = ((weight_index * 131 + master_id * 17) % 2048).to(torch.bfloat16)
    fc2 = ((weight_index * 137 + master_id * 19) % 2048).to(torch.bfloat16)
    manager.construct_local_master_ptr_pool(0, [fc1], [fc2])

    echo_buffer = None
    try:
        def update_placement():
            return manager.update_placement(0, routing_map)

        update_placement()
        placement_stats = _measure_global_us(
            update_placement, args.warmups, args.iterations, group)
        update_placement()

        def weight_sync():
            return manager.weight_sync(0)

        weight_sync()
        weight_sync_stats = _measure_global_us(
            weight_sync, args.warmups, args.iterations, group)

        def reroute():
            return manager.reroute(0, logical_probs, routing_map)

        expanded_probs, expanded_routing = reroute()
        assert bool((expanded_routing.sum(dim=1) == 1).all().item())
        reroute_stats = _measure_global_us(
            reroute, args.warmups, args.iterations, group)
        expanded_probs, expanded_routing = reroute()

        physical_idx64 = expanded_routing.to(torch.int64).argmax(dim=1)
        physical_topk = physical_idx64.to(deep_ep.topk_idx_t)[:, None]
        physical_weights = expanded_probs.gather(
            1, physical_idx64[:, None]).to(torch.float32)

        physical_counts = torch.bincount(
            physical_idx64, minlength=num_physical).to(torch.int64)
        dist.all_reduce(physical_counts, group=group)
        logical_counts = routing_map.sum(dim=0, dtype=torch.int64)
        dist.all_reduce(logical_counts, group=group)
        physical_rank_load = physical_counts.view(
            world_size, num_local_physical).sum(dim=1)

        echo_buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=args.hidden,
            num_topk=1,
            allow_hybrid_mode=True,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=True,
            num_allocated_qps=129,
            num_gpu_timeout_secs=120,
            explicitly_destroy=True,
            rail_balance=args.rail_balance,
        )

        target_rank = torch.div(
            physical_idx64, num_local_physical, rounding_mode='floor')
        target_node = torch.div(
            target_rank, num_local_ranks, rounding_mode='floor')
        local_demand = torch.bincount(
            target_node, minlength=num_nodes).to(torch.float64)
        control_stats = {'median_us': 0.0, 'p90_us': 0.0}
        plan = None
        if args.rail_balance == 'incast':
            plan, _ = echo_buffer.update_incast_rail_masks(local_demand)
            control_stats = _measure_global_us(
                lambda: echo_buffer.update_incast_rail_masks(local_demand),
                args.warmups, args.iterations, group)
            plan, demand = echo_buffer.update_incast_rail_masks(local_demand)
        else:
            demand_cuda = torch.zeros(
                (num_nodes, num_nodes), dtype=torch.float64, device='cuda')
            demand_cuda[node].copy_(local_demand)
            dist.all_reduce(demand_cuda, group=group)
            demand = demand_cuda.cpu()

        token = torch.arange(num_tokens, dtype=torch.int64, device='cuda')
        src_global = rank * num_max_tokens_per_rank + token
        x_value = (src_global % 127 + 1).to(torch.bfloat16)
        x = x_value[:, None].expand(num_tokens, args.hidden).contiguous()

        dispatch_args = dict(
            x=x,
            topk_idx=physical_topk,
            topk_weights=physical_weights,
            num_experts=num_physical,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            expert_alignment=1,
            num_sms=args.num_sms,
            do_cpu_sync=True,
        )

        def dispatch_once():
            return echo_buffer.dispatch(**dispatch_args)

        recv_x, recv_topk, recv_weights, handle, _ = dispatch_once()
        num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
        expected_local_counts = physical_counts[
            rank * num_local_physical:(rank + 1) * num_local_physical]
        assert num_recv == expected_local_counts.sum().item()
        assert torch.equal(
            torch.bincount(
                recv_topk[:num_recv, 0].to(torch.int64),
                minlength=num_local_physical),
            expected_local_counts)
        src_ids = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)
        expected_x = (src_ids % 127 + 1).to(torch.bfloat16)
        assert torch.equal(
            recv_x[:num_recv],
            expected_x[:, None].expand(num_recv, args.hidden))

        combined_x, combined_weights, _ = echo_buffer.combine(
            recv_x, handle=handle, topk_weights=recv_weights,
            num_sms=args.num_sms)
        assert torch.equal(combined_x, x)
        assert torch.equal(combined_weights, physical_weights)

        dispatch_stats = _measure_global_us(
            dispatch_once, args.warmups, args.iterations, group)
        recv_x, _, recv_weights, handle, _ = dispatch_once()

        def combine_once():
            return echo_buffer.combine(
                recv_x, handle=handle, topk_weights=recv_weights,
                num_sms=args.num_sms)

        combine_stats = _measure_global_us(
            combine_once, args.warmups, args.iterations, group)

        def roundtrip_once():
            rt_recv_x, _, rt_recv_weights, rt_handle, _ = dispatch_once()
            return echo_buffer.combine(
                rt_recv_x, handle=rt_handle, topk_weights=rt_recv_weights,
                num_sms=args.num_sms)

        roundtrip_stats = _measure_global_us(
            roundtrip_once, args.warmups, args.iterations, group)

        if rank == 0:
            ep_us = dispatch_stats['median_us'] + combine_stats['median_us']
            per_step_us = reroute_stats['median_us'] + ep_us
            amortized_control_us = (
                placement_stats['median_us'] +
                weight_sync_stats['median_us'] +
                control_stats['median_us']
            ) / args.control_interval
            print('RESULT ' + json.dumps({
                'stack': 'UltraEP+EchoP',
                'rail_balance': args.rail_balance,
                'num_nodes': num_nodes,
                'num_rails_per_node': num_local_ranks,
                'total_remote_tokens_requested': args.total,
                'fan_in': args.fan_in,
                'sink': sink,
                'source_alpha': args.source_alpha,
                'rail_alpha': args.rail_alpha,
                'rail_phase': args.rail_phase,
                'expert_alpha': args.expert_alpha,
                'redundant_experts_per_rank': args.redundant_experts_per_rank,
                'logical_rank_load_max_over_mean': _max_mean(logical_counts),
                'physical_rank_load_max_over_mean': _max_mean(physical_rank_load),
                'physical_node_demand': demand.to(torch.int64).tolist(),
                'rail_masks': (
                    [[int(value) & 0xFFFFFFFF for value in row]
                     for row in plan.tolist()]
                    if plan is not None else None),
                'ultra_placement_median_us': placement_stats['median_us'],
                'ultra_weight_sync_median_us': weight_sync_stats['median_us'],
                'ultra_reroute_median_us': reroute_stats['median_us'],
                'echop_control_update_median_us': control_stats['median_us'],
                'dispatch_median_us': dispatch_stats['median_us'],
                'dispatch_p99_us': dispatch_stats['p99_us'],
                'combine_median_us': combine_stats['median_us'],
                'combine_p99_us': combine_stats['p99_us'],
                'ep_median_us': ep_us,
                'roundtrip_median_us': roundtrip_stats['median_us'],
                'roundtrip_p99_us': roundtrip_stats['p99_us'],
                'roundtrip_samples_us': roundtrip_stats['samples_us'],
                'steady_step_median_us': per_step_us,
                'control_interval_steps': args.control_interval,
                'amortized_control_us_per_step': amortized_control_us,
                'accounted_total_us': per_step_us + amortized_control_us,
                'roundtrip_accounted_total_us': (
                    roundtrip_stats['median_us'] +
                    reroute_stats['median_us'] + amortized_control_us),
                'weight_sync_included': True,
                'grad_reduce_included': False,
                'correctness': 'PASS',
            }), flush=True)
    finally:
        if echo_buffer is not None:
            echo_buffer.destroy()
        manager.destroy()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--total', type=int, default=256)
    parser.add_argument('--fan-in', type=int, default=1)
    parser.add_argument('--sink', type=int, default=-1)
    parser.add_argument('--source-alpha', type=float, default=0.0)
    parser.add_argument('--rail-alpha', type=float, default=1.5)
    parser.add_argument(
        '--rail-phase', choices=('aligned', 'rotated'), default='aligned')
    parser.add_argument('--expert-alpha', type=float, default=2.0)
    parser.add_argument('--redundant-experts-per-rank', type=int, default=1)
    parser.add_argument('--expert-weight-numel', type=int, default=1024)
    parser.add_argument(
        '--rail-balance', choices=('off', 'active', 'incast'), required=True)
    parser.add_argument('--num-sms', type=int, default=16)
    parser.add_argument('--control-interval', type=int, default=32)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--seed', type=int, default=37)
    args = parser.parse_args()
    if args.total <= 0 or args.hidden <= 0 or args.control_interval <= 0:
        parser.error('total, hidden, and control interval must be positive')
    if min(args.source_alpha, args.rail_alpha, args.expert_alpha) < 0:
        parser.error('all alpha values must be non-negative')
    if args.redundant_experts_per_rank <= 0:
        parser.error('at least one redundant expert per rank is required')
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
