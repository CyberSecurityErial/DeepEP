"""Generic multi-node EchoP Rail-balance and incast benchmark.

The benchmark deliberately keeps expert placement fixed. It changes only the
source-Rail traffic histogram, expert histogram, and node-pair fan-in so the
network-layer ablations remain comparable across DeepEP/Ultra/Moon-style plans.
"""

import argparse
import itertools
import json
import math
import os
import random
import statistics
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.buffers.elastic import _pack_incast_weighted_quotas
from deep_ep.utils.envs import init_dist
from fast_style_planner import (
    plan_cyclic_source_local_waves,
    plan_equal_chunk_interleaved_waves,
    plan_fast_style_waves,
    reconstruct_demand,
)


def _parse_int_csv(value: str) -> list[int]:
    result = [int(item) for item in value.split(',')]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError('expected comma-separated positive integers')
    return result


def _parse_float_csv(value: str) -> list[float]:
    result = [float(item) for item in value.split(',')]
    if not result or any(not math.isfinite(item) or item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            'expected comma-separated finite non-negative floats')
    return result


def _policy_variant(
        policy: str, weighted_quotas: bool,
        pairwise_peer_budget: int = 0,
        pairwise_planner: str = 'round-robin') -> str:
    if pairwise_peer_budget > 0:
        if pairwise_planner == 'fast-global':
            return (
                'fast_style_global'
                if policy in ('active', 'all') else
                'fast_style_global_no_local_balance')
        if pairwise_planner == 'cyclic-local':
            return (
                'echop_local_matching'
                if policy in ('active', 'all') else
                'echop_local_matching_no_rail_balance')
        if pairwise_planner == 'chunked-global':
            return (
                'equal_chunk_interleave_joint'
                if policy in ('active', 'all') else
                'equal_chunk_interleave_matching_only')
        if policy == 'off':
            return 'fanin_only'
        if policy in ('active', 'all'):
            return 'joint'
    if policy == 'off':
        return 'native'
    if policy == 'active':
        return 'rail_only'
    if policy == 'incast':
        return 'joint' if weighted_quotas else 'fanin_only'
    return policy


def _proportional_counts(total: int, weights: list[float]) -> list[int]:
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    assert sum(counts) == total
    return counts


def _zipf_counts(total: int, alpha: float, size: int) -> list[int]:
    return _proportional_counts(
        total, [(index + 1) ** (-alpha) for index in range(size)])


def _rail_counts(
        total: int, alpha: float, num_rails: int, source_node: int,
        destination: int, phase: str) -> list[int]:
    """Return a Rail histogram with controlled cross-source hotspot phase."""
    counts = _zipf_counts(total, alpha, num_rails)
    if phase == 'aligned' or total == 0:
        return counts
    if phase != 'rotated':
        raise ValueError(f'unsupported Rail phase: {phase}')
    offset = (source_node * 3 + destination) % num_rails
    rotated = [0] * num_rails
    for rail, count in enumerate(counts):
        rotated[(rail + offset) % num_rails] = count
    return rotated


def _active_incast_sources(num_nodes: int, sink: int, fan_in: int) -> list[int]:
    candidates = [
        (sink - offset) % num_nodes for offset in range(1, num_nodes)
    ]
    return candidates[:fan_in]


def _balanced_destinations(
        num_nodes: int, source: int, total: int, fan_in: int,
        flow_alpha: float,
        flow_shape: str = 'directed-zipf') -> list[tuple[int, int]]:
    """Build a regular directed graph with equal send/receive node totals."""
    if flow_shape == 'directed-zipf':
        counts = _zipf_counts(total, flow_alpha, fan_in)
    elif flow_shape == 'permuted-zipf':
        if num_nodes != 4 or fan_in != 3:
            raise ValueError(
                'permuted-zipf currently requires four nodes and fan-in 3')
        counts = _zipf_counts(total, flow_alpha, fan_in)
        # Three non-cyclic perfect matchings.  Every row and column sees one
        # hot, one medium, and one cold edge, while a fixed cyclic wave mixes
        # edge sizes and exposes its hot/cold straggler problem.
        edge_class = {}
        for index, pairs in enumerate((
                ((0, 1), (2, 3)),
                ((0, 2), (1, 3)),
                ((0, 3), (1, 2)))):
            for lhs, rhs in pairs:
                edge_class[lhs, rhs] = index
                edge_class[rhs, lhs] = index
        return [
            (destination, counts[edge_class[source, destination]])
            for destination in range(num_nodes) if destination != source
        ]
    elif flow_shape == 'symmetric-distance':
        # Equal weights for opposite directions make every physical node-pair
        # bidirectionally symmetric.  For four nodes this produces two hot
        # ring neighbors and one cold diagonal while preserving exactly equal
        # send totals, receive totals, and aggregate expert load.
        counts = _proportional_counts(total, [
            min(offset, num_nodes - offset) ** (-flow_alpha)
            for offset in range(1, fan_in + 1)
        ])
    else:
        raise ValueError(f'unsupported All-to-All flow shape: {flow_shape}')
    return [
        ((source + offset) % num_nodes, count)
        for offset, count in enumerate(counts, start=1)
    ]


def _balanced_flow_expert_counts(
        flow_totals: list[int], flow_index: int,
        num_experts: int) -> list[int]:
    """Split one flow while balancing remainders across all peer flows.

    Every destination in ``balanced-alltoall`` receives one flow at every
    relative offset. Packing each flow's remainder consecutively after all
    preceding remainders therefore makes the aggregate expert histogram
    exactly uniform whenever the per-node total is divisible by the number
    of local experts (as all frozen matrix sizes are).
    """
    total = flow_totals[flow_index]
    base, remainder = divmod(total, num_experts)
    counts = [base] * num_experts
    start = sum(value % num_experts for value in flow_totals[:flow_index])
    for offset in range(remainder):
        counts[(start + offset) % num_experts] += 1
    assert sum(counts) == total
    return counts


def _plan_scenario(
        case: str, num_nodes: int, sink: int, fan_in: int,
        plan_state: str) -> tuple[int, int]:
    if case in ('rail-ring', 'balanced-alltoall') or plan_state == 'fresh':
        return sink, fan_in
    if plan_state == 'previous-sink':
        return (sink - 1) % num_nodes, fan_in
    if plan_state == 'lower-fan-in':
        return sink, max(1, fan_in - 1)
    raise ValueError(f'unsupported plan state: {plan_state}')


def _source_node_totals(
        case: str, num_nodes: int, total: int, sink: int, fan_in: int,
        source_alpha: float, incast_total_mode: str) -> list[int]:
    """Return exact remote-token totals for every source node.

    ``aggregate`` keeps the total incast bytes fixed while fan-in changes, so
    it isolates concurrency. ``per-source`` keeps the mean offered load per
    source fixed, so aggregate bytes grow with fan-in and model overload.
    """
    if case in ('rail-ring', 'balanced-alltoall'):
        return [total] * num_nodes

    active_sources = _active_incast_sources(num_nodes, sink, fan_in)
    aggregate = total if incast_total_mode == 'aggregate' else total * fan_in
    active_totals = _zipf_counts(aggregate, source_alpha, fan_in)
    result = [0] * num_nodes
    for source, source_total in zip(active_sources, active_totals):
        result[source] = source_total
    return result


def _traffic_demand(
        case: str, num_nodes: int, total: int, sink: int,
        fan_in: int, source_alpha: float,
        incast_total_mode: str,
        alltoall_flow_shape: str = 'directed-zipf') -> torch.Tensor:
    demand = torch.zeros(
        (num_nodes, num_nodes), dtype=torch.float64, device='cpu')
    source_totals = _source_node_totals(
        case, num_nodes, total, sink, fan_in, source_alpha,
        incast_total_mode)
    if case == 'rail-ring':
        for source in range(num_nodes):
            demand[source, (source + 1) % num_nodes] = source_totals[source]
    elif case == 'balanced-alltoall':
        for source in range(num_nodes):
            for destination, count in _balanced_destinations(
                    num_nodes, source, total, fan_in, source_alpha,
                    alltoall_flow_shape):
                demand[source, destination] = count
    else:
        for source, source_total in enumerate(source_totals):
            demand[source, sink] = source_total
    return demand


def _rank_targets(
        case: str, num_nodes: int, num_rails: int, source_node: int,
        source_rail: int, total: int, rail_alpha: float,
        expert_alpha: float, source_alpha: float, sink: int, fan_in: int,
        incast_total_mode: str, rail_phase: str,
        alltoall_flow_shape: str = 'directed-zipf') -> list[int]:
    """Return deterministic expert targets for one physical source rank."""
    if case == 'rail-ring':
        destination = (source_node + 1) % num_nodes
        active = True
    elif case == 'balanced-alltoall':
        destination = 0
        active = True
    else:
        destination = sink
        active = source_node in _active_incast_sources(
            num_nodes, sink, fan_in)

    source_total = _source_node_totals(
        case, num_nodes, total, sink, fan_in, source_alpha,
        incast_total_mode)[source_node]

    if not active or source_total == 0:
        # A one-token local route keeps every process on the same tested API
        # path without adding any scale-out traffic.
        return [source_node * num_rails + source_rail]

    source_counts = _rail_counts(
        source_total, rail_alpha, num_rails, source_node, destination,
        rail_phase)
    local_count = source_counts[source_rail]
    if case == 'balanced-alltoall':
        targets = []
        flows = _balanced_destinations(
            num_nodes, source_node, total, fan_in, source_alpha,
            alltoall_flow_shape)
        flow_totals = [flow_total for _, flow_total in flows]
        for flow_index, (target_node, flow_total) in enumerate(flows):
            expert_counts = (
                _balanced_flow_expert_counts(
                    flow_totals, flow_index, num_rails)
                if expert_alpha == 0 else
                _zipf_counts(flow_total, expert_alpha, num_rails)
            )
            targets.extend(
                target_node * num_rails + expert
                for expert, count in enumerate(expert_counts)
                for _ in range(count)
            )
        shuffle_seed = (
            104729 + source_node * 1009 + total * 17 +
            int(round(rail_alpha * 1000)) * 31 +
            int(round(expert_alpha * 1000)) * 43 +
            int(round(source_alpha * 1000)) * 47)
        random.Random(shuffle_seed).shuffle(targets)
        begin = sum(source_counts[:source_rail])
        return targets[begin:begin + local_count]
    if case != 'expert-rail-incast':
        return [destination * num_rails + source_rail] * local_count

    expert_counts = _zipf_counts(source_total, expert_alpha, num_rails)
    all_experts = [
        destination * num_rails + expert
        for expert, count in enumerate(expert_counts)
        for _ in range(count)
    ]
    # Fixed shuffle removes an accidental block correlation between hot source
    # Rails and hot experts while preserving both exact marginal histograms.
    seed = (
        104729 + source_node * 1009 + int(round(rail_alpha * 1000)) * 17 +
        int(round(expert_alpha * 1000)) * 31 +
        int(round(source_alpha * 1000)) * 43 + total)
    random.Random(seed).shuffle(all_experts)
    begin = sum(source_counts[:source_rail])
    return all_experts[begin:begin + local_count]


def _expected_source_ids(
        target_rank: int, case: str, num_nodes: int, num_rails: int,
        total: int, rail_alpha: float, expert_alpha: float, sink: int,
        source_alpha: float, fan_in: int, incast_total_mode: str,
        num_max_tokens_per_rank: int, rail_phase: str,
        alltoall_flow_shape: str = 'directed-zipf') -> list[int]:
    result = []
    for source_node in range(num_nodes):
        for source_rail in range(num_rails):
            targets = _rank_targets(
                case, num_nodes, num_rails, source_node, source_rail,
                total, rail_alpha, expert_alpha, source_alpha, sink, fan_in,
                incast_total_mode, rail_phase, alltoall_flow_shape)
            source_rank = source_node * num_rails + source_rail
            result.extend(
                source_rank * num_max_tokens_per_rank + token
                for token, target in enumerate(targets)
                if target == target_rank
            )
    return sorted(result)


def _target_expert_counts(
        case: str, num_nodes: int, num_rails: int, total: int,
        rail_alpha: float, expert_alpha: float, source_alpha: float,
        sink: int, fan_in: int, incast_total_mode: str,
        alltoall_flow_shape: str = 'directed-zipf') -> list[int]:
    if case == 'balanced-alltoall':
        counts = [0] * (num_nodes * num_rails)
        for source in range(num_nodes):
            flows = _balanced_destinations(
                num_nodes, source, total, fan_in, source_alpha,
                alltoall_flow_shape)
            flow_totals = [flow_total for _, flow_total in flows]
            for flow_index, (destination, flow_total) in enumerate(flows):
                expert_counts = (
                    _balanced_flow_expert_counts(
                        flow_totals, flow_index, num_rails)
                    if expert_alpha == 0 else
                    _zipf_counts(flow_total, expert_alpha, num_rails)
                )
                for expert, count in enumerate(expert_counts):
                    counts[destination * num_rails + expert] += count
        return counts
    source_totals = _source_node_totals(
        case, num_nodes, total, sink, fan_in, source_alpha,
        incast_total_mode)
    counts = [0] * num_rails
    for source_total in source_totals:
        if source_total == 0:
            continue
        alpha = expert_alpha if case == 'expert-rail-incast' else rail_alpha
        for expert, count in enumerate(
                _zipf_counts(source_total, alpha, num_rails)):
            counts[expert] += count
    return counts


def _source_to_sink_rail_counts(
        case: str, num_nodes: int, num_rails: int, total: int,
        rail_alpha: float, expert_alpha: float, source_alpha: float,
        sink: int, fan_in: int, incast_total_mode: str,
        rail_phase: str,
        alltoall_flow_shape: str = 'directed-zipf') -> list[list[int]]:
    counts = [[0] * num_rails for _ in range(num_nodes)]
    for source in range(num_nodes):
        for rail in range(num_rails):
            targets = _rank_targets(
                case, num_nodes, num_rails, source, rail, total,
                rail_alpha, expert_alpha, source_alpha, sink, fan_in,
                incast_total_mode, rail_phase, alltoall_flow_shape)
            counts[source][rail] = sum(
                target // num_rails == sink for target in targets)
    return counts


def _max_input_rank_tokens(num_nodes: int, num_rails: int, args) -> int:
    result = 1
    for fan_in in args.fan_ins:
        if fan_in >= num_nodes:
            continue
        source_alphas = (
            args.source_alphas if args.case != 'rail-ring'
            else args.source_alphas[:1])
        for total in args.totals:
            for source_alpha in source_alphas:
                source_totals = _source_node_totals(
                    args.case, num_nodes, total,
                    args.sink if args.sink >= 0 else num_nodes - 1,
                    fan_in, source_alpha, args.incast_total_mode)
                for rail_alpha in args.rail_alphas:
                    for source_total in source_totals:
                        if source_total:
                            result = max(
                                result,
                                max(_zipf_counts(
                                    source_total, rail_alpha, num_rails)))
        if args.case == 'rail-ring':
            break
    return result


def _max_source_node_tokens(
        num_nodes: int, args: argparse.Namespace) -> int:
    """Return the largest exact source-node total in the requested matrix."""
    result = 1
    sink = args.sink if args.sink >= 0 else num_nodes - 1
    for fan_in in args.fan_ins:
        if fan_in >= num_nodes:
            continue
        source_alphas = (
            args.source_alphas if args.case != 'rail-ring'
            else args.source_alphas[:1])
        for total in args.totals:
            for source_alpha in source_alphas:
                source_totals = _source_node_totals(
                    args.case, num_nodes, total, sink, fan_in,
                    source_alpha, args.incast_total_mode)
                result = max(result, max(source_totals))
    return result


def _buffer_token_capacity(
        max_input_tokens: int, max_source_node_tokens: int, num_rails: int,
        args: argparse.Namespace) -> int:
    """Reserve enough per-channel slots for Rail-subset concentration.

    In one channel, a destination can receive at most all source-node tokens
    assigned to that channel.  Tokens are striped independently by each owner
    Rail, so summing their rounded-up channel counts adds at most one tail per
    owner.  Therefore ``source_node_total + num_rails * num_channels`` is a
    deterministic bound even for an adversarial destination ordering or a
    strict one-Rail mask.  This is much tighter than pretending every owner
    Rail simultaneously contains the hottest rank's token count.  Every
    policy receives the same capacity.
    """
    if args.case == 'rail-ring':
        return max_input_tokens
    max_runtime_channels = args.num_sms * 8
    proxy_capacity = (
        max_source_node_tokens + num_rails * max_runtime_channels)
    return max(max_input_tokens, proxy_capacity)


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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start_event.record()
        result = fn()
        end_event.record()
        end_event.synchronize()
        elapsed_us = start_event.elapsed_time(end_event) * 1e3
        del result

        global_us = torch.tensor(elapsed_us, dtype=torch.float64, device='cuda')
        dist.all_reduce(global_us, op=dist.ReduceOp.MAX, group=group)
        if dist.get_rank(group=group) == 0:
            samples.append(global_us.item())

    if dist.get_rank(group=group) != 0:
        return {}
    return {
        'median_us': statistics.median(samples),
        'p90_us': _percentile(samples, 0.90),
        'p99_us': _percentile(samples, 0.99),
        'min_us': min(samples),
        'samples_us': samples,
    }


def _profile_global_once(fn, trace_path: str, group, rank: int) -> None:
    """Capture one correctness-equivalent roundtrip for kernel attribution."""
    dist.barrier(group=group)
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        result = fn()
        torch.cuda.synchronize()
        del result
    if rank == 0:
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        profiler.export_chrome_trace(trace_path)
        print('PROFILE_TABLE_BEGIN ' + trace_path, flush=True)
        print(profiler.key_averages().table(
            sort_by='cuda_time_total', max_name_column_width=120), flush=True)
        print('PROFILE_TABLE_END ' + trace_path, flush=True)
    dist.barrier(group=group)


def _measure_control_us(
        fn, group, warmups: int = 1,
        iterations: int = 5) -> tuple[object, float]:
    result = None
    for _ in range(warmups):
        result = fn()
    torch.cuda.synchronize()

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
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        samples.append(elapsed.item())
    return result, statistics.median(samples)


def _unsigned_plan(plan: torch.Tensor) -> list[list[int]]:
    return [
        [int(value) & 0xFFFFFFFF for value in row]
        for row in plan.tolist()
    ]


def _balanced_counts(total: int, selected: list[int], num_rails: int) -> list[int]:
    result = [0] * num_rails
    if not selected:
        return result
    base, remainder = divmod(total, len(selected))
    for position, rail in enumerate(selected):
        result[rail] = base + int(position < remainder)
    return result


def _weighted_counts(total: int, control: int, num_rails: int) -> list[int]:
    weights = [(control >> (4 * rail)) & 0xF for rail in range(num_rails)]
    weight_sum = sum(weights)
    if weight_sum == 0:
        return [0] * num_rails
    result = [total * weight // weight_sum for weight in weights]
    remainder = total - sum(result)
    order = sorted(
        range(num_rails),
        key=lambda rail: (-(total * weights[rail] % weight_sum), rail))
    for rail in order[:remainder]:
        result[rail] += 1
    return result


def _planned_sink_metrics(
        policy: str, source_counts: list[list[int]], demand: torch.Tensor,
        plan: torch.Tensor | None, sink: int, num_rails: int,
        weighted_quotas: bool = False) -> dict[str, object]:
    """Predict aggregate sink Rail pressure from the exact quota policy."""
    sink_tokens = [0] * num_rails
    sink_fan_in = [0] * num_rails
    moved = 0
    remote_total = 0
    for source, original in enumerate(source_counts):
        if source == sink or demand[source, sink].item() <= 0:
            continue
        total = sum(original)
        remote_total += total
        if policy == 'off':
            target = original
        else:
            selected = [rail for rail, count in enumerate(original) if count]
            if policy == 'all':
                selected = list(range(num_rails))
            elif policy == 'incast' and plan is not None:
                mask = int(plan[source, sink].item()) & 0xFFFFFFFF
                masked = [
                    rail for rail in range(num_rails) if mask & (1 << rail)
                ]
                if masked:
                    selected = masked
            if weighted_quotas and policy == 'incast' and plan is not None:
                control = int(_pack_incast_weighted_quotas(
                    plan, num_rails, source)[sink].item()) & 0xFFFFFFFF
                target = _weighted_counts(total, control, num_rails)
            else:
                target = _balanced_counts(total, selected, num_rails)
        moved += sum(max(before - after, 0) for before, after in zip(original, target))
        for rail, count in enumerate(target):
            sink_tokens[rail] += count
            sink_fan_in[rail] += int(count > 0)

    mean = remote_total / num_rails if num_rails else 0.0
    return {
        'sink_rail_tokens_after_policy': sink_tokens,
        'sink_rail_source_fan_in_after_policy': sink_fan_in,
        'sink_rail_max_over_mean_after_policy': (
            max(sink_tokens) / mean if mean > 0 else 0.0),
        'predicted_moved_tokens': moved,
        'predicted_moved_ratio': (
            moved / remote_total if remote_total else 0.0),
    }


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=23)
    assert world_size % num_local_ranks == 0
    num_nodes = world_size // num_local_ranks
    assert num_nodes >= 2
    node = rank // num_local_ranks
    rail = rank % num_local_ranks
    sink = args.sink if args.sink >= 0 else num_nodes - 1
    if not 0 <= sink < num_nodes:
        raise ValueError(f'sink must be in [0, {num_nodes})')
    if args.case == 'balanced-alltoall' and args.plan_state != 'fresh':
        raise ValueError('balanced-alltoall currently requires plan_state=fresh')
    if args.pairwise_peer_budget > 0:
        if args.case != 'balanced-alltoall':
            raise ValueError(
                'pairwise waves currently require case=balanced-alltoall')
        if args.rail_balance not in ('off', 'active', 'all'):
            raise ValueError(
                'pairwise waves require rail_balance=off/active/all')
        if (args.pairwise_planner in (
                'cyclic-local', 'fast-global', 'chunked-global') and
                args.pairwise_peer_budget != 1):
            raise ValueError(
                'directed permutations require peer budget 1')
        if (args.pairwise_planner == 'chunked-global' and
                args.pairwise_chunk_tokens <= 0 and
                args.pairwise_chunk_divisor <= 0):
            raise ValueError(
                'chunked-global requires a positive chunk size or divisor')
    num_experts = world_size
    max_input_tokens_per_rank = _max_input_rank_tokens(
        num_nodes, num_local_ranks, args)
    max_source_node_tokens = _max_source_node_tokens(
        num_nodes, args)
    num_max_tokens_per_rank = _buffer_token_capacity(
        max_input_tokens_per_rank, max_source_node_tokens,
        num_local_ranks, args)
    buffer = deep_ep.ElasticBuffer(
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
    assert buffer.get_logical_domain_size() == (num_nodes, num_local_ranks)

    if rank == 0:
        print('CONFIG ' + json.dumps({
            'case': args.case,
            'policy': args.rail_balance,
            'variant': _policy_variant(
                args.rail_balance, args.incast_weighted_quotas,
                args.pairwise_peer_budget, args.pairwise_planner),
            'num_nodes': num_nodes,
            'num_rails_per_node': num_local_ranks,
            'totals_parameter': args.totals,
            'incast_total_mode': (
                None if args.case == 'rail-ring' else
                'per-node-fixed' if args.case == 'balanced-alltoall' else
                args.incast_total_mode),
            'alltoall_flow_shape': args.alltoall_flow_shape,
            'source_alphas': args.source_alphas,
            'rail_alphas': args.rail_alphas,
            'rail_phase': args.rail_phase,
            'expert_alphas': args.expert_alphas,
            'fan_ins': args.fan_ins,
            'sink': sink,
            'plan_state': args.plan_state,
            'incast_rail_overlap': args.incast_rail_overlap,
            'incast_weighted_quotas': args.incast_weighted_quotas,
            'pairwise_peer_budget': args.pairwise_peer_budget,
            'pairwise_planner': args.pairwise_planner,
            'pairwise_chunk_tokens': args.pairwise_chunk_tokens,
            'pairwise_chunk_divisor': args.pairwise_chunk_divisor,
            'pairwise_execution': args.pairwise_execution,
            'hidden': args.hidden,
            'dtype': 'bfloat16',
            'topk': 1,
            'num_qps': args.num_qps,
            'warmups': args.warmups,
            'iterations': args.iterations,
            'control_interval_steps': args.control_interval,
            'max_input_tokens_per_rank': max_input_tokens_per_rank,
            'max_source_node_tokens': max_source_node_tokens,
            'num_max_tokens_per_rank': num_max_tokens_per_rank,
            'buffer_gib_per_rank': buffer.num_bytes / (1024 ** 3),
        }), flush=True)

    for fan_in in args.fan_ins:
        if fan_in >= num_nodes:
            if rank == 0:
                print('SKIP ' + json.dumps({
                    'fan_in': fan_in,
                    'reason': f'requires at least {fan_in + 1} nodes',
                }), flush=True)
            continue
        if args.case == 'rail-ring' and fan_in != args.fan_ins[0]:
            continue
        plan_sink, plan_fan_in = _plan_scenario(
            args.case, num_nodes, sink, fan_in, args.plan_state)

        expert_alphas = (
            args.expert_alphas
            if args.case in ('expert-rail-incast', 'balanced-alltoall')
            else args.expert_alphas[:1]
        )
        source_alphas = (
            args.source_alphas
            if args.case != 'rail-ring'
            else args.source_alphas[:1]
        )
        for total in args.totals:
            for source_alpha in source_alphas:
                for rail_alpha, expert_alpha in itertools.product(
                        args.rail_alphas, expert_alphas):
                    demand = _traffic_demand(
                        args.case, num_nodes, total, sink, fan_in,
                        source_alpha, args.incast_total_mode,
                        args.alltoall_flow_shape)
                    plan_demand = _traffic_demand(
                        args.case, num_nodes, total, plan_sink, plan_fan_in,
                        source_alpha, args.incast_total_mode,
                        args.alltoall_flow_shape)
                    local_targets = _rank_targets(
                        args.case, num_nodes, num_local_ranks, node, rail,
                        total, rail_alpha, expert_alpha, source_alpha, sink,
                        fan_in, args.incast_total_mode, args.rail_phase,
                        args.alltoall_flow_shape)
                    plan = None
                    plan_us = 0.0
                    install_us = 0.0
                    update_us = 0.0
                    if args.rail_balance == 'incast':
                        plan_local_targets = _rank_targets(
                            args.case, num_nodes, num_local_ranks, node, rail,
                            total, rail_alpha, expert_alpha, source_alpha,
                            plan_sink, plan_fan_in, args.incast_total_mode,
                            args.rail_phase, args.alltoall_flow_shape)
                        plan_local_demand = torch.bincount(
                            torch.tensor(
                                plan_local_targets, dtype=torch.int64) //
                            num_local_ranks,
                            minlength=num_nodes,
                        ).to(torch.float64)
                        plan, plan_us = _measure_control_us(
                            lambda: deep_ep.utils.plan_incast_rail_masks(
                                plan_demand, num_local_ranks,
                                epoch=args.plan_epoch,
                                rail_overlap=args.incast_rail_overlap),
                            group,
                        )
                        _, install_us = _measure_control_us(
                            lambda: buffer.set_incast_rail_masks(
                                plan,
                                weighted_quotas=args.incast_weighted_quotas),
                            group)
                        update_result, update_us = _measure_control_us(
                            lambda: buffer.update_incast_rail_masks(
                                plan_local_demand, epoch=args.plan_epoch,
                                rail_overlap=args.incast_rail_overlap,
                                weighted_quotas=args.incast_weighted_quotas),
                            group,
                        )
                        collective_plan, observed_demand = update_result
                        off_diagonal = ~torch.eye(
                            num_nodes, dtype=torch.bool, device='cpu')
                        assert torch.equal(
                            observed_demand[off_diagonal],
                            plan_demand[off_diagonal])
                        assert torch.equal(collective_plan, plan)
                        assert torch.equal(
                            buffer.get_incast_rail_masks().cpu(), plan[node])

                    num_tokens = len(local_targets)
                    topk_idx = torch.tensor(
                        local_targets, dtype=deep_ep.topk_idx_t,
                        device='cuda')[:, None]
                    token = torch.arange(
                        num_tokens, dtype=torch.int64, device='cuda')
                    src_global = rank * num_max_tokens_per_rank + token
                    x_value = (src_global % 127 + 1).to(torch.bfloat16)
                    x = x_value[:, None].expand(
                        num_tokens, args.hidden).contiguous()
                    topk_weights = torch.ones(
                        (num_tokens, 1), dtype=torch.float, device='cuda')

                    if args.pairwise_peer_budget > 0:
                        effective_chunk_tokens = 0
                        if args.pairwise_planner == 'chunked-global':
                            effective_chunk_tokens = (
                                args.pairwise_chunk_tokens
                                if args.pairwise_chunk_tokens > 0 else
                                max(1, total // args.pairwise_chunk_divisor))

                        def make_pairwise_waves():
                            if args.pairwise_planner == 'fast-global':
                                integer_demand = demand.to(
                                    torch.int64).tolist()
                                waves = plan_fast_style_waves(integer_demand)
                                assert reconstruct_demand(
                                    num_nodes, waves) == integer_demand
                                return waves
                            if args.pairwise_planner == 'cyclic-local':
                                integer_demand = demand.to(
                                    torch.int64).tolist()
                                return plan_cyclic_source_local_waves(
                                    integer_demand)
                            if args.pairwise_planner == 'chunked-global':
                                integer_demand = demand.to(
                                    torch.int64).tolist()
                                waves = plan_equal_chunk_interleaved_waves(
                                    integer_demand,
                                    effective_chunk_tokens)
                                assert reconstruct_demand(
                                    num_nodes, waves) == integer_demand
                                return waves
                            plain_waves = (
                                deep_ep.utils.plan_pairwise_incast_waves(
                                    num_nodes,
                                    max_peers_per_wave=(
                                        args.pairwise_peer_budget),
                                    epoch=args.plan_epoch))
                            return tuple(tuple(
                                (lhs, rhs, 0, 1) for lhs, rhs in wave)
                                for wave in plain_waves)

                        pairwise_waves, pairwise_plan_us = _measure_control_us(
                            make_pairwise_waves, group)
                        pairwise_wave_edge_tokens = []
                        for wave in pairwise_waves:
                            edge_tokens = []
                            if args.pairwise_planner in (
                                    'cyclic-local', 'fast-global',
                                    'chunked-global'):
                                edge_tokens.extend(
                                    int(transfer[3]) for transfer in wave)
                            else:
                                for (lhs, rhs, chunk_index,
                                     num_chunks) in wave:
                                    for source, destination in (
                                            (lhs, rhs), (rhs, lhs)):
                                        edge_total = int(
                                            demand[source, destination].item())
                                        edge_tokens.append(max(
                                            0,
                                            (edge_total + num_chunks - 1 -
                                             chunk_index) // num_chunks))
                            pairwise_wave_edge_tokens.append(edge_tokens)
                        covered_positions = []
                        timed_wave_dispatch_args = []
                        timed_wave_indices = []
                        timed_wave_num_recv = []
                        reconstructed_x = torch.empty_like(x)
                        reconstructed_weights = torch.empty_like(topk_weights)
                        max_wave_fan_in = 0

                        for wave in pairwise_waves:
                            positions = []
                            if args.pairwise_planner in (
                                    'cyclic-local', 'fast-global',
                                    'chunked-global'):
                                incoming = [0] * num_nodes
                                outgoing = [0] * num_nodes
                                local_transfers = []
                                for (source, destination, offset, count,
                                     edge_total) in wave:
                                    outgoing[source] += 1
                                    incoming[destination] += 1
                                    if source == node:
                                        local_transfers.append((
                                            destination, offset, count,
                                            edge_total))
                                max_wave_fan_in = max(
                                    max_wave_fan_in, max(incoming))
                                assert max(incoming) <= 1
                                assert max(outgoing) <= 1
                                assert len(local_transfers) == 1
                                for (peer, offset, count,
                                     edge_total) in local_transfers:
                                    peer_positions = [
                                        position
                                        for position, target in enumerate(
                                            local_targets)
                                        if (target // num_local_ranks == peer)
                                    ]
                                    begin = offset * len(
                                        peer_positions) // edge_total
                                    end = (offset + count) * len(
                                        peer_positions) // edge_total
                                    positions.extend(
                                        peer_positions[begin:end])
                            else:
                                degrees = [0] * num_nodes
                                peer_chunks = []
                                for (lhs, rhs, chunk_index,
                                     num_chunks) in wave:
                                    degrees[lhs] += 1
                                    degrees[rhs] += 1
                                    if lhs == node:
                                        peer_chunks.append(
                                            (rhs, chunk_index, num_chunks))
                                    elif rhs == node:
                                        peer_chunks.append(
                                            (lhs, chunk_index, num_chunks))
                                max_wave_fan_in = max(
                                    max_wave_fan_in, max(degrees))
                                assert (1 <= len(peer_chunks) <=
                                        args.pairwise_peer_budget)
                                for (peer, chunk_index,
                                     num_chunks) in peer_chunks:
                                    peer_positions = [
                                        position
                                        for position, target in enumerate(
                                            local_targets)
                                        if (target // num_local_ranks == peer)
                                    ]
                                    positions.extend(peer_positions[
                                        chunk_index::num_chunks])
                            positions.sort()
                            # The four-node hardware experiment uses large,
                            # expert-balanced flows, so every physical source
                            # rank contributes to its paired peer in every wave.
                            assert positions
                            covered_positions.extend(positions)
                            index = torch.tensor(
                                positions, dtype=torch.int64, device='cuda')
                            wave_x = x.index_select(0, index)
                            wave_topk_idx = topk_idx.index_select(0, index)
                            wave_topk_weights = topk_weights.index_select(
                                0, index)
                            wave_dispatch_args = dict(
                                x=wave_x,
                                topk_idx=wave_topk_idx,
                                topk_weights=wave_topk_weights,
                                num_experts=num_experts,
                                num_max_tokens_per_rank=(
                                    num_max_tokens_per_rank),
                                expert_alignment=1,
                                num_sms=args.num_sms,
                                num_qps=args.num_qps,
                                do_cpu_sync=True,
                            )
                            (wave_recv_x, _, wave_recv_weights,
                             wave_handle, _) = buffer.dispatch(
                                 **wave_dispatch_args)
                            wave_num_recv = (
                                wave_handle.
                                psum_num_recv_tokens_per_scaleup_rank[-1].
                                item())
                            (wave_combined_x, wave_combined_weights,
                             _) = buffer.combine(
                                 wave_recv_x, handle=wave_handle,
                                 topk_weights=wave_recv_weights,
                                 num_sms=args.num_sms,
                                 num_qps=args.num_qps)
                            assert torch.equal(wave_combined_x, wave_x)
                            assert torch.equal(
                                wave_combined_weights, wave_topk_weights)
                            reconstructed_x.index_copy_(
                                0, index, wave_combined_x)
                            reconstructed_weights.index_copy_(
                                0, index, wave_combined_weights)

                            timed_wave_args = dict(wave_dispatch_args)
                            timed_wave_args['do_cpu_sync'] = False
                            timed_wave_args['num_recv_tokens_hint'] = (
                                wave_num_recv)
                            (async_wave_x, _, async_wave_weights,
                             async_wave_handle, _) = buffer.dispatch(
                                 **timed_wave_args)
                            (async_wave_combined_x,
                             async_wave_combined_weights, _) = buffer.combine(
                                 async_wave_x, handle=async_wave_handle,
                                 topk_weights=async_wave_weights,
                                 num_sms=args.num_sms,
                                 num_qps=args.num_qps)
                            assert torch.equal(
                                async_wave_combined_x, wave_x)
                            assert torch.equal(
                                async_wave_combined_weights,
                                wave_topk_weights)
                            timed_wave_dispatch_args.append(timed_wave_args)
                            timed_wave_indices.append(index)
                            timed_wave_num_recv.append(wave_num_recv)
                            del wave_recv_x, wave_recv_weights
                            del wave_combined_x, wave_combined_weights
                            del async_wave_x, async_wave_weights
                            del async_wave_combined_x
                            del async_wave_combined_weights

                        assert sorted(covered_positions) == list(
                            range(num_tokens))
                        assert torch.equal(reconstructed_x, x)
                        assert torch.equal(
                            reconstructed_weights, topk_weights)

                        def pairwise_roundtrip_once():
                            outputs = []
                            for timed_wave_args in timed_wave_dispatch_args:
                                (wave_recv_x, _, wave_recv_weights,
                                 wave_handle, _) = buffer.dispatch(
                                     **timed_wave_args)
                                outputs.append(buffer.combine(
                                    wave_recv_x, handle=wave_handle,
                                    topk_weights=wave_recv_weights,
                                    num_sms=args.num_sms,
                                    num_qps=args.num_qps))
                            return outputs

                        def materialized_roundtrip_once(pipelined):
                            outputs = []
                            for index, wave_num_recv in zip(
                                    timed_wave_indices,
                                    timed_wave_num_recv):
                                wave_x = x.index_select(0, index)
                                wave_topk_idx = topk_idx.index_select(0, index)
                                wave_topk_weights = topk_weights.index_select(
                                    0, index)
                                dispatch_args = dict(
                                    x=wave_x,
                                    topk_idx=wave_topk_idx,
                                    topk_weights=wave_topk_weights,
                                    num_experts=num_experts,
                                    num_max_tokens_per_rank=(
                                        num_max_tokens_per_rank),
                                    expert_alignment=1,
                                    num_sms=args.num_sms,
                                    num_qps=args.num_qps,
                                    do_cpu_sync=False,
                                    num_recv_tokens_hint=wave_num_recv,
                                )
                                if pipelined:
                                    ready = buffer.capture()
                                    dispatch_args.update(
                                        previous_event=ready,
                                        async_with_compute_stream=True,
                                        allocate_on_comm_stream=True)
                                (wave_recv_x, _, wave_recv_weights,
                                 wave_handle,
                                 dispatch_event) = buffer.dispatch(
                                     **dispatch_args)
                                combine_args = dict(
                                    x=wave_recv_x,
                                    handle=wave_handle,
                                    topk_weights=wave_recv_weights,
                                    num_sms=args.num_sms,
                                    num_qps=args.num_qps)
                                if pipelined:
                                    combine_args.update(
                                        previous_event=dispatch_event.event,
                                        async_with_compute_stream=True,
                                        allocate_on_comm_stream=True)
                                (combined_x, combined_weights,
                                 combine_event) = buffer.combine(
                                     **combine_args)
                                outputs.append((
                                    wave_x, wave_topk_idx,
                                    wave_topk_weights, wave_recv_x,
                                    wave_recv_weights, wave_handle,
                                    combined_x, combined_weights,
                                    dispatch_event, combine_event))
                            if pipelined:
                                for output in outputs:
                                    output[-1].current_stream_wait()
                            return outputs

                        measured_roundtrip = pairwise_roundtrip_once
                        if args.pairwise_execution == 'sequential-pack':
                            measured_roundtrip = lambda: (
                                materialized_roundtrip_once(False))
                        elif args.pairwise_execution == 'pipelined-pack':
                            measured_roundtrip = lambda: (
                                materialized_roundtrip_once(True))

                        roundtrip_stats = _measure_global_us(
                            measured_roundtrip, args.warmups,
                            args.iterations, group)

                        pairwise_wave_stats = []
                        if args.profile_pairwise_waves:
                            def one_wave_roundtrip_once(timed_wave_args):
                                (wave_recv_x, _, wave_recv_weights,
                                 wave_handle, _) = buffer.dispatch(
                                     **timed_wave_args)
                                return buffer.combine(
                                    wave_recv_x, handle=wave_handle,
                                    topk_weights=wave_recv_weights,
                                    num_sms=args.num_sms,
                                    num_qps=args.num_qps)

                            for timed_wave_args in timed_wave_dispatch_args:
                                pairwise_wave_stats.append(
                                    _measure_global_us(
                                        lambda wave_args=timed_wave_args:
                                        one_wave_roundtrip_once(wave_args),
                                        args.warmups, args.iterations, group))

                        if rank == 0:
                            remote_tokens = int(demand.sum().item())
                            source_node_counts = demand.sum(
                                dim=1).to(torch.int64).tolist()
                            source_rail_counts_by_node = [
                                _rail_counts(
                                    count, rail_alpha, num_local_ranks,
                                    source, 0, args.rail_phase)
                                if count else [0] * num_local_ranks
                                for source, count in enumerate(
                                    source_node_counts)
                            ]
                            source_to_sink_rail_counts = (
                                _source_to_sink_rail_counts(
                                    args.case, num_nodes, num_local_ranks,
                                    total, rail_alpha, expert_alpha,
                                    source_alpha, sink, fan_in,
                                    args.incast_total_mode,
                                    args.rail_phase,
                                    args.alltoall_flow_shape))
                            base_sink_metrics = _planned_sink_metrics(
                                args.rail_balance,
                                source_to_sink_rail_counts, demand, None,
                                sink, num_local_ranks, False)
                            phase_max_over_mean = 1.0
                            if args.rail_balance == 'off':
                                phase_max_over_mean = max(
                                    max(row) / (sum(row) / num_local_ranks)
                                    for source, row in enumerate(
                                        source_to_sink_rail_counts)
                                    if source != sink and sum(row) > 0)
                            payload_bytes = (
                                remote_tokens * args.hidden * 2)
                            result = {
                                'case': args.case,
                                'policy': args.rail_balance,
                                'variant': _policy_variant(
                                    args.rail_balance,
                                    args.incast_weighted_quotas,
                                    args.pairwise_peer_budget,
                                    args.pairwise_planner),
                                'fanin_strategy': (
                                    'fast-style-global-weighted-permutations'
                                    if args.pairwise_planner == 'fast-global'
                                    else (
                                        'equal-chunk-interleaved-permutations'
                                        if args.pairwise_planner ==
                                        'chunked-global'
                                        else (
                                        'echop-cyclic-source-local-permutations'
                                        if args.pairwise_planner ==
                                        'cyclic-local'
                                        else 'pairwise-waves'))),
                                'latency_timing': 'gpu-event',
                                'component_timing': (
                                    'all-waves-roundtrip-total'),
                                'tokens_parameter': total,
                                'incast_total_mode': 'per-node-fixed',
                                'tokens_per_active_source_node': total,
                                'remote_tokens_global': remote_tokens,
                                'payload_mib_global': (
                                    payload_bytes / (1024 ** 2)),
                                'source_alpha': source_alpha,
                                'alltoall_flow_shape': (
                                    args.alltoall_flow_shape),
                                'rail_alpha': rail_alpha,
                                'rail_phase': args.rail_phase,
                                'expert_alpha': expert_alpha,
                                'fan_in': fan_in,
                                'plan_state': args.plan_state,
                                'pairwise_peer_budget': (
                                    args.pairwise_peer_budget),
                                'pairwise_planner': args.pairwise_planner,
                                'pairwise_chunk_tokens': (
                                    effective_chunk_tokens),
                                'pairwise_chunk_divisor': (
                                    args.pairwise_chunk_divisor),
                                'pairwise_execution': (
                                    args.pairwise_execution),
                                'planner_input_scope': (
                                    'complete-global-demand-matrix'
                                    if args.pairwise_planner in (
                                        'fast-global', 'chunked-global')
                                    else (
                                        'source-local-outgoing-row+topology'
                                        if args.pairwise_planner ==
                                        'cyclic-local'
                                        else 'topology-only')),
                                'global_demand_cells': (
                                    world_size * world_size
                                    if args.pairwise_planner in (
                                        'fast-global', 'chunked-global')
                                    else 0),
                                'global_demand_bytes_u32': (
                                    world_size * world_size * 4
                                    if args.pairwise_planner in (
                                        'fast-global', 'chunked-global')
                                    else 0),
                                'local_demand_cells_per_node': (
                                    num_nodes
                                    if args.pairwise_planner == 'cyclic-local'
                                    else 0),
                                'num_qps': args.num_qps,
                                'pairwise_num_waves': len(pairwise_waves),
                                'pairwise_waves': pairwise_waves,
                                'pairwise_wave_edge_tokens': (
                                    pairwise_wave_edge_tokens),
                                'pairwise_wave_edge_max_over_mean': [
                                    (max(loads) /
                                     (sum(loads) / len(loads)))
                                    if loads and sum(loads) else 0.0
                                    for loads in pairwise_wave_edge_tokens
                                ],
                                'pairwise_wave_roundtrip_us': [
                                    stats['median_us']
                                    for stats in pairwise_wave_stats
                                ],
                                'pairwise_wave_roundtrip_p99_us': [
                                    stats['p99_us']
                                    for stats in pairwise_wave_stats
                                ],
                                'node_fan_in_after_policy': (
                                    max_wave_fan_in),
                                'source_node_counts': source_node_counts,
                                'source_rail_counts_by_node': (
                                    source_rail_counts_by_node),
                                'source_to_sink_rail_counts_by_node': (
                                    source_to_sink_rail_counts),
                                'target_expert_counts': (
                                    _target_expert_counts(
                                        args.case, num_nodes,
                                        num_local_ranks, total, rail_alpha,
                                        expert_alpha, source_alpha, sink,
                                        fan_in, args.incast_total_mode,
                                        args.alltoall_flow_shape)),
                                'demand': demand.to(torch.int64).tolist(),
                                'plan_cpu_median_max_us': pairwise_plan_us,
                                'control_interval_steps': (
                                    args.control_interval),
                                'control_amortized_us_per_step': (
                                    pairwise_plan_us /
                                    args.control_interval),
                                'roundtrip_us': (
                                    roundtrip_stats['median_us']),
                                'roundtrip_p90_us': (
                                    roundtrip_stats['p90_us']),
                                'roundtrip_p99_us': (
                                    roundtrip_stats['p99_us']),
                                'roundtrip_plus_amortized_control_us': (
                                    roundtrip_stats['median_us'] +
                                    pairwise_plan_us /
                                    args.control_interval),
                                'roundtrip_samples_us': (
                                    roundtrip_stats['samples_us']),
                                'ep_total_us': (
                                    roundtrip_stats['median_us']),
                                'ep_total_plus_amortized_control_us': (
                                    roundtrip_stats['median_us'] +
                                    pairwise_plan_us /
                                    args.control_interval),
                                'ep_total_gbps_global': (
                                    2 * payload_bytes /
                                    roundtrip_stats['median_us'] / 1e3),
                                'sink_rail_tokens_after_policy': (
                                    base_sink_metrics[
                                        'sink_rail_tokens_after_policy']),
                                'sink_rail_source_fan_in_after_policy': (
                                    [max_wave_fan_in] * num_local_ranks),
                                'sink_rail_max_over_mean_after_policy': (
                                    phase_max_over_mean),
                                'predicted_moved_tokens': (
                                    base_sink_metrics[
                                        'predicted_moved_tokens']),
                                'predicted_moved_ratio': (
                                    base_sink_metrics[
                                        'predicted_moved_ratio']),
                            }
                            print(
                                'RESULT ' + json.dumps(result), flush=True)

                        del reconstructed_x, reconstructed_weights
                        del timed_wave_dispatch_args
                        del x, topk_idx, topk_weights
                        dist.barrier(group=group)
                        continue

                    dispatch_args = dict(
                        x=x,
                        topk_idx=topk_idx,
                        topk_weights=topk_weights,
                        num_experts=num_experts,
                        num_max_tokens_per_rank=num_max_tokens_per_rank,
                        expert_alignment=1,
                        num_sms=args.num_sms,
                        num_qps=args.num_qps,
                        do_cpu_sync=True,
                    )

                    def dispatch_uncached_once():
                        return buffer.dispatch(**dispatch_args)

                    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = \
                        dispatch_uncached_once()
                    num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
                    src_ids = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)
                    actual_sources = torch.sort(src_ids).values
                    expected_sources = torch.tensor(
                        _expected_source_ids(
                            rank, args.case, num_nodes, num_local_ranks,
                            total, rail_alpha, expert_alpha, sink,
                            source_alpha, fan_in, args.incast_total_mode,
                            num_max_tokens_per_rank, args.rail_phase,
                            args.alltoall_flow_shape),
                        dtype=torch.int64, device='cuda')
                    assert num_recv == expected_sources.numel()
                    if not torch.equal(actual_sources, expected_sources):
                        print(
                            f'DISPATCH_MISMATCH rank={rank} case={args.case} '
                            f'total={total} rail_alpha={rail_alpha} '
                            f'expert_alpha={expert_alpha} '
                            f'source_alpha={source_alpha} fan_in={fan_in} '
                            f'actual_head={actual_sources[:16].cpu().tolist()} '
                            f'expected_head={expected_sources[:16].cpu().tolist()}',
                            flush=True,
                        )
                    assert torch.equal(actual_sources, expected_sources)
                    assert torch.count_nonzero(
                        recv_topk_idx[:num_recv, 0]).item() == 0
                    expected_recv = (src_ids % 127 + 1).to(torch.bfloat16)
                    assert torch.equal(recv_x[:num_recv, 0], expected_recv)
                    assert torch.equal(
                        recv_x[:num_recv],
                        expected_recv[:, None].expand(num_recv, args.hidden))

                    combined_x, combined_weights, _ = buffer.combine(
                        recv_x, handle=handle,
                        topk_weights=recv_topk_weights,
                        num_sms=args.num_sms,
                        num_qps=args.num_qps)
                    assert torch.equal(combined_x, x)
                    assert torch.equal(combined_weights, topk_weights)

                    # Keep the dynamic GPU layout/data path, but do not wait on
                    # the CPU for exact receive sizes.  In production this host
                    # work runs ahead of (and overlaps with) the GPU step.  Rail
                    # balance intentionally does not use DeepEP's cached-layout
                    # mode because its routing decision may change every step.
                    timed_dispatch_args = dict(dispatch_args)
                    timed_dispatch_args['do_cpu_sync'] = False
                    timed_dispatch_args['num_recv_tokens_hint'] = num_recv

                    def dispatch_once():
                        return buffer.dispatch(**timed_dispatch_args)

                    async_recv_x, _, async_recv_weights, async_handle, _ = \
                        dispatch_once()
                    assert async_recv_x.shape[0] == num_recv
                    assert async_handle.recv_src_metadata.shape[0] == num_recv
                    async_combined_x, async_combined_weights, _ = \
                        buffer.combine(
                            async_recv_x, handle=async_handle,
                            topk_weights=async_recv_weights,
                            num_sms=args.num_sms,
                            num_qps=args.num_qps)
                    assert torch.equal(async_combined_x, x)
                    assert torch.equal(async_combined_weights, topk_weights)
                    del async_recv_x, async_recv_weights
                    del async_combined_x, async_combined_weights

                    dispatch_stats = _measure_global_us(
                        dispatch_once, args.warmups, args.iterations, group)
                    recv_x, _, recv_topk_weights, handle, _ = dispatch_once()

                    def combine_once():
                        return buffer.combine(
                            recv_x, handle=handle,
                            topk_weights=recv_topk_weights,
                            num_sms=args.num_sms,
                            num_qps=args.num_qps)

                    combine_stats = _measure_global_us(
                        combine_once, args.warmups, args.iterations, group)

                    def roundtrip_once():
                        rt_recv_x, _, rt_recv_weights, rt_handle, _ = \
                            buffer.dispatch(**timed_dispatch_args)
                        return buffer.combine(
                            rt_recv_x, handle=rt_handle,
                            topk_weights=rt_recv_weights,
                            num_sms=args.num_sms,
                            num_qps=args.num_qps)

                    roundtrip_stats = _measure_global_us(
                        roundtrip_once, args.warmups, args.iterations, group)

                    if (args.profile_dir and
                            (args.profile_total == 0 or
                             args.profile_total == total) and
                            (args.profile_rail_alpha < 0 or
                             math.isclose(
                                 args.profile_rail_alpha, rail_alpha))):
                        profile_name = (
                            f'{_policy_variant(args.rail_balance, args.incast_weighted_quotas, args.pairwise_peer_budget)}'
                            f'_s{total}_ra{rail_alpha}_rank0.json')
                        _profile_global_once(
                            roundtrip_once,
                            os.path.join(args.profile_dir, profile_name),
                            group, rank)

                    if rank == 0:
                        remote_tokens = int(demand.sum().item())
                        source_node_counts = demand.sum(
                            dim=1).to(torch.int64).tolist()
                        source_rail_counts_by_node = [
                            _rail_counts(
                                count, rail_alpha, num_local_ranks,
                                source, (
                                    0 if args.case == 'balanced-alltoall'
                                    else (source + 1) % num_nodes
                                    if args.case == 'rail-ring' else sink),
                                args.rail_phase)
                            if count else [0] * num_local_ranks
                            for source, count in enumerate(source_node_counts)
                        ]
                        source_to_sink_rail_counts = (
                            _source_to_sink_rail_counts(
                                args.case, num_nodes, num_local_ranks, total,
                                rail_alpha, expert_alpha, source_alpha, sink,
                                fan_in, args.incast_total_mode,
                                args.rail_phase,
                                args.alltoall_flow_shape))
                        sink_metrics = _planned_sink_metrics(
                            args.rail_balance, source_to_sink_rail_counts,
                            demand, plan, sink, num_local_ranks,
                            args.incast_weighted_quotas)
                        payload_bytes = remote_tokens * args.hidden * 2
                        ep_total_us = (
                            dispatch_stats['median_us'] +
                            combine_stats['median_us'])
                        result = {
                            'case': args.case,
                            'policy': args.rail_balance,
                            'variant': _policy_variant(
                                args.rail_balance,
                                args.incast_weighted_quotas,
                                args.pairwise_peer_budget),
                            'latency_timing': 'gpu-event',
                            'tokens_parameter': total,
                            'incast_total_mode': (
                                None if args.case == 'rail-ring' else
                                'per-node-fixed'
                                if args.case == 'balanced-alltoall'
                                else args.incast_total_mode),
                            'tokens_per_active_source_node': (
                                total if args.case in (
                                    'rail-ring', 'balanced-alltoall') or (
                                    args.incast_total_mode == 'per-source' and
                                    source_alpha == 0)
                                else None),
                            'remote_tokens_global': remote_tokens,
                            'payload_mib_global': payload_bytes / (1024 ** 2),
                            'source_alpha': source_alpha,
                            'alltoall_flow_shape': (
                                args.alltoall_flow_shape),
                            'rail_alpha': rail_alpha,
                            'rail_phase': args.rail_phase,
                            'expert_alpha': expert_alpha,
                            'fan_in': (
                                1 if args.case == 'rail-ring' else fan_in),
                            'plan_state': args.plan_state,
                            'incast_rail_overlap': args.incast_rail_overlap,
                            'incast_weighted_quotas': (
                                args.incast_weighted_quotas),
                            'num_qps': args.num_qps,
                            'plan_sink': plan_sink,
                            'plan_fan_in': plan_fan_in,
                            'source_node_counts': source_node_counts,
                            'source_rail_counts_by_node': (
                                source_rail_counts_by_node),
                            'source_to_sink_rail_counts_by_node': (
                                source_to_sink_rail_counts),
                            'target_expert_counts': _target_expert_counts(
                                args.case, num_nodes, num_local_ranks, total,
                                rail_alpha, expert_alpha, source_alpha, sink,
                                fan_in, args.incast_total_mode,
                                args.alltoall_flow_shape),
                            'demand': demand.to(torch.int64).tolist(),
                            'plan_demand': plan_demand.to(
                                torch.int64).tolist(),
                            'rail_masks': (
                                _unsigned_plan(plan) if plan is not None else None),
                            'plan_cpu_median_max_us': plan_us,
                            'plan_install_median_max_us': install_us,
                            'control_update_median_max_us': update_us,
                            'control_interval_steps': args.control_interval,
                            'control_amortized_us_per_step': (
                                update_us / args.control_interval),
                            'dispatch_us': dispatch_stats['median_us'],
                            'dispatch_p90_us': dispatch_stats['p90_us'],
                            'dispatch_p99_us': dispatch_stats['p99_us'],
                            'combine_us': combine_stats['median_us'],
                            'combine_p90_us': combine_stats['p90_us'],
                            'combine_p99_us': combine_stats['p99_us'],
                            'ep_total_us': ep_total_us,
                            'ep_total_plus_amortized_control_us': (
                                ep_total_us + update_us / args.control_interval),
                            'ep_total_p90_us': (
                                dispatch_stats['p90_us'] +
                                combine_stats['p90_us']),
                            'roundtrip_us': roundtrip_stats['median_us'],
                            'roundtrip_p90_us': roundtrip_stats['p90_us'],
                            'roundtrip_p99_us': roundtrip_stats['p99_us'],
                            'roundtrip_plus_amortized_control_us': (
                                roundtrip_stats['median_us'] +
                                update_us / args.control_interval),
                            'roundtrip_samples_us': (
                                roundtrip_stats['samples_us']),
                            'ep_total_gbps_global': (
                                2 * payload_bytes / ep_total_us / 1e3
                                if ep_total_us > 0 else 0),
                            **sink_metrics,
                        }
                        print('RESULT ' + json.dumps(result), flush=True)

                    del x, topk_idx, topk_weights
                    del recv_x, recv_topk_idx, recv_topk_weights, handle
                    del combined_x, combined_weights
                    dist.barrier(group=group)

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=16)
    parser.add_argument(
        '--num-qps', type=int, default=0,
        help=(
            'RDMA QPs used by dispatch/combine; 0 keeps the DeepEP '
            'automatic setting'))
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument(
        '--case', choices=(
            'rail-ring', 'incast', 'expert-rail-incast',
            'balanced-alltoall'),
        required=True)
    parser.add_argument(
        '--rail-balance', choices=('off', 'active', 'all', 'incast'),
        required=True)
    parser.add_argument(
        '--totals', type=_parse_int_csv,
        default=_parse_int_csv('2048,8192,16384'))
    parser.add_argument(
        '--rail-alphas', type=_parse_float_csv,
        default=_parse_float_csv('0,0.75,1.5,2.25'))
    parser.add_argument(
        '--rail-phase', choices=('aligned', 'rotated'), default='aligned',
        help='align source Rail hotspots or rotate them across node pairs')
    parser.add_argument(
        '--expert-alphas', type=_parse_float_csv,
        default=_parse_float_csv('0,1.0,2.0'))
    parser.add_argument(
        '--source-alphas', type=_parse_float_csv,
        default=_parse_float_csv('0,1.5'))
    parser.add_argument(
        '--alltoall-flow-shape',
        choices=(
            'directed-zipf', 'symmetric-distance', 'permuted-zipf'),
        default='directed-zipf',
        help=(
            'node-pair demand pattern for balanced-alltoall; symmetric-distance '
            'keeps opposite directions equal; permuted-zipf keeps all totals '
            'balanced while mixing hot/cold edges in fixed cyclic waves'))
    parser.add_argument(
        '--fan-ins', type=_parse_int_csv,
        default=_parse_int_csv('1,2,3'))
    parser.add_argument(
        '--incast-total-mode', choices=('aggregate', 'per-source'),
        default='aggregate',
        help=(
            'aggregate fixes total incast bytes as fan-in changes; '
            'per-source fixes mean bytes per active source'))
    parser.add_argument('--sink', type=int, default=-1)
    parser.add_argument(
        '--plan-state',
        choices=('fresh', 'previous-sink', 'lower-fan-in'),
        default='fresh',
        help=(
            'fresh uses the current demand; previous-sink models a moved '
            'hotspot; lower-fan-in models a sudden extra source'))
    parser.add_argument('--plan-epoch', type=int, default=0)
    parser.add_argument(
        '--incast-rail-overlap', type=float, default=1.0,
        help=(
            'Rail-membership budget: 1 is strict disjoint partition; '
            '1.5 trades bounded overlap for more source Rail parallelism'))
    parser.add_argument(
        '--incast-weighted-quotas', action='store_true',
        help=(
            'use packed per-Rail weights to remove residual overlap-2 '
            'destination imbalance without adding a data-path kernel'))
    parser.add_argument(
        '--pairwise-peer-budget', type=int, default=0,
        help=(
            'split balanced All-to-All into round-robin node-pair waves; '
            '0 disables temporal scheduling, 1 makes node fan-in one'))
    parser.add_argument(
        '--pairwise-planner',
        choices=(
            'round-robin', 'cyclic-local', 'fast-global',
            'chunked-global'),
        default='round-robin',
        help=(
            'round-robin uses topology-only undirected waves; cyclic-local '
            'uses fixed directed matchings plus source-local counts; '
            'fast-global independently reimplements FAST-style weighted '
            'directed permutations from the complete global demand matrix; '
            'chunked-global interleaves equal-sized matching chunks and '
            'drains sub-chunk tails last'))
    parser.add_argument(
        '--pairwise-chunk-tokens', type=int, default=0,
        help='edge chunk size for chunked-global; zero for other planners')
    parser.add_argument(
        '--pairwise-chunk-divisor', type=int, default=0,
        help=(
            'choose chunk size as tokens-per-node divided by this value; '
            'ignored when --pairwise-chunk-tokens is positive'))
    parser.add_argument(
        '--pairwise-execution',
        choices=('prepacked', 'sequential-pack', 'pipelined-pack'),
        default='prepacked',
        help=(
            'prepacked measures communication after wave materialization; '
            'sequential-pack includes index packing on the critical path; '
            'pipelined-pack overlaps next-wave packing with current-wave '
            'dispatch/combine using compute and communication streams'))
    parser.add_argument(
        '--profile-pairwise-waves', action='store_true',
        help='also time every pairwise wave separately for critical-path audit')
    parser.add_argument('--control-interval', type=int, default=32)
    parser.add_argument(
        '--profile-dir', type=str, default='',
        help='optional directory for one rank-0 CUDA kernel trace per point')
    parser.add_argument(
        '--profile-total', type=int, default=0,
        help='when positive, profile only this tokens-per-node point')
    parser.add_argument(
        '--profile-rail-alpha', type=float, default=-1.0,
        help='when non-negative, profile only this Rail-alpha point')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.control_interval <= 0:
        parser.error('--control-interval must be positive')
    if args.pairwise_peer_budget < 0:
        parser.error('--pairwise-peer-budget must be non-negative')
    if args.pairwise_chunk_tokens < 0:
        parser.error('--pairwise-chunk-tokens must be non-negative')
    if args.pairwise_chunk_divisor < 0:
        parser.error('--pairwise-chunk-divisor must be non-negative')
    if args.profile_total < 0:
        parser.error('--profile-total must be non-negative')
    if args.num_qps < 0 or args.num_qps == 1:
        parser.error('--num-qps must be 0 (automatic) or at least 2')
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
