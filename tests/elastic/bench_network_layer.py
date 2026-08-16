"""Generic multi-node EchoP Rail-balance and incast benchmark.

The benchmark deliberately keeps expert placement fixed. It changes only the
source-Rail traffic histogram, expert histogram, and node-pair fan-in so the
network-layer ablations remain comparable across DeepEP/Ultra/Moon-style plans.
"""

import argparse
import itertools
import json
import math
import random
import statistics
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist


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
    assert sum(counts) == total
    return counts


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
        flow_alpha: float) -> list[tuple[int, int]]:
    """Build a regular directed graph with equal send/receive node totals."""
    counts = _zipf_counts(total, flow_alpha, fan_in)
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
        incast_total_mode: str) -> torch.Tensor:
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
                    num_nodes, source, total, fan_in, source_alpha):
                demand[source, destination] = count
    else:
        for source, source_total in enumerate(source_totals):
            demand[source, sink] = source_total
    return demand


def _rank_targets(
        case: str, num_nodes: int, num_rails: int, source_node: int,
        source_rail: int, total: int, rail_alpha: float,
        expert_alpha: float, source_alpha: float, sink: int, fan_in: int,
        incast_total_mode: str, rail_phase: str) -> list[int]:
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
            num_nodes, source_node, total, fan_in, source_alpha)
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
        num_max_tokens_per_rank: int, rail_phase: str) -> list[int]:
    result = []
    for source_node in range(num_nodes):
        for source_rail in range(num_rails):
            targets = _rank_targets(
                case, num_nodes, num_rails, source_node, source_rail,
                total, rail_alpha, expert_alpha, source_alpha, sink, fan_in,
                incast_total_mode, rail_phase)
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
        sink: int, fan_in: int, incast_total_mode: str) -> list[int]:
    if case == 'balanced-alltoall':
        counts = [0] * (num_nodes * num_rails)
        for source in range(num_nodes):
            flows = _balanced_destinations(
                num_nodes, source, total, fan_in, source_alpha)
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
        rail_phase: str) -> list[list[int]]:
    counts = [[0] * num_rails for _ in range(num_nodes)]
    for source in range(num_nodes):
        for rail in range(num_rails):
            targets = _rank_targets(
                case, num_nodes, num_rails, source, rail, total,
                rail_alpha, expert_alpha, source_alpha, sink, fan_in,
                incast_total_mode, rail_phase)
            counts[source][rail] = sum(
                target // num_rails == sink for target in targets)
    return counts


def _max_rank_tokens(num_nodes: int, num_rails: int, args) -> int:
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
        elapsed_us = (time.perf_counter() - start) * 1e6
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


def _planned_sink_metrics(
        policy: str, source_counts: list[list[int]], demand: torch.Tensor,
        plan: torch.Tensor | None, sink: int, num_rails: int) -> dict[str, object]:
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

    num_experts = world_size
    num_max_tokens_per_rank = _max_rank_tokens(
        num_nodes, num_local_ranks, args)
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
            'num_nodes': num_nodes,
            'num_rails_per_node': num_local_ranks,
            'totals_parameter': args.totals,
            'incast_total_mode': (
                None if args.case == 'rail-ring' else
                'per-node-fixed' if args.case == 'balanced-alltoall' else
                args.incast_total_mode),
            'source_alphas': args.source_alphas,
            'rail_alphas': args.rail_alphas,
            'rail_phase': args.rail_phase,
            'expert_alphas': args.expert_alphas,
            'fan_ins': args.fan_ins,
            'sink': sink,
            'plan_state': args.plan_state,
            'hidden': args.hidden,
            'dtype': 'bfloat16',
            'topk': 1,
            'warmups': args.warmups,
            'iterations': args.iterations,
            'control_interval_steps': args.control_interval,
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
                        source_alpha, args.incast_total_mode)
                    plan_demand = _traffic_demand(
                        args.case, num_nodes, total, plan_sink, plan_fan_in,
                        source_alpha, args.incast_total_mode)
                    local_targets = _rank_targets(
                        args.case, num_nodes, num_local_ranks, node, rail,
                        total, rail_alpha, expert_alpha, source_alpha, sink,
                        fan_in, args.incast_total_mode, args.rail_phase)
                    plan = None
                    plan_us = 0.0
                    install_us = 0.0
                    update_us = 0.0
                    if args.rail_balance == 'incast':
                        plan_local_targets = _rank_targets(
                            args.case, num_nodes, num_local_ranks, node, rail,
                            total, rail_alpha, expert_alpha, source_alpha,
                            plan_sink, plan_fan_in, args.incast_total_mode,
                            args.rail_phase)
                        plan_local_demand = torch.bincount(
                            torch.tensor(
                                plan_local_targets, dtype=torch.int64) //
                            num_local_ranks,
                            minlength=num_nodes,
                        ).to(torch.float64)
                        plan, plan_us = _measure_control_us(
                            lambda: deep_ep.utils.plan_incast_rail_masks(
                                plan_demand, num_local_ranks,
                                epoch=args.plan_epoch),
                            group,
                        )
                        _, install_us = _measure_control_us(
                            lambda: buffer.set_incast_rail_masks(plan), group)
                        update_result, update_us = _measure_control_us(
                            lambda: buffer.update_incast_rail_masks(
                                plan_local_demand, epoch=args.plan_epoch),
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

                    dispatch_args = dict(
                        x=x,
                        topk_idx=topk_idx,
                        topk_weights=topk_weights,
                        num_experts=num_experts,
                        num_max_tokens_per_rank=num_max_tokens_per_rank,
                        expert_alignment=1,
                        num_sms=args.num_sms,
                        do_cpu_sync=True,
                    )

                    def dispatch_once():
                        return buffer.dispatch(**dispatch_args)

                    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = \
                        dispatch_once()
                    num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
                    src_ids = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)
                    actual_sources = torch.sort(src_ids).values
                    expected_sources = torch.tensor(
                        _expected_source_ids(
                            rank, args.case, num_nodes, num_local_ranks,
                            total, rail_alpha, expert_alpha, sink,
                            source_alpha, fan_in, args.incast_total_mode,
                            num_max_tokens_per_rank, args.rail_phase),
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
                        num_sms=args.num_sms)
                    assert torch.equal(combined_x, x)
                    assert torch.equal(combined_weights, topk_weights)

                    dispatch_stats = _measure_global_us(
                        dispatch_once, args.warmups, args.iterations, group)
                    recv_x, _, recv_topk_weights, handle, _ = dispatch_once()

                    def combine_once():
                        return buffer.combine(
                            recv_x, handle=handle,
                            topk_weights=recv_topk_weights,
                            num_sms=args.num_sms)

                    combine_stats = _measure_global_us(
                        combine_once, args.warmups, args.iterations, group)

                    def roundtrip_once():
                        rt_recv_x, _, rt_recv_weights, rt_handle, _ = \
                            buffer.dispatch(**dispatch_args)
                        return buffer.combine(
                            rt_recv_x, handle=rt_handle,
                            topk_weights=rt_recv_weights,
                            num_sms=args.num_sms)

                    roundtrip_stats = _measure_global_us(
                        roundtrip_once, args.warmups, args.iterations, group)

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
                                args.rail_phase))
                        sink_metrics = _planned_sink_metrics(
                            args.rail_balance, source_to_sink_rail_counts,
                            demand, plan, sink, num_local_ranks)
                        payload_bytes = remote_tokens * args.hidden * 2
                        ep_total_us = (
                            dispatch_stats['median_us'] +
                            combine_stats['median_us'])
                        result = {
                            'case': args.case,
                            'policy': args.rail_balance,
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
                            'rail_alpha': rail_alpha,
                            'rail_phase': args.rail_phase,
                            'expert_alpha': expert_alpha,
                            'fan_in': (
                                1 if args.case == 'rail-ring' else fan_in),
                            'plan_state': args.plan_state,
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
                                fan_in, args.incast_total_mode),
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
    parser.add_argument('--control-interval', type=int, default=32)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.control_interval <= 0:
        parser.error('--control-interval must be positive')
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
