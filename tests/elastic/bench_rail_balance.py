import argparse
import json
import math
import statistics
import time

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist


def _parse_int_csv(value: str) -> list[int]:
    values = [int(item) for item in value.split(',')]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError('expected comma-separated positive integers')
    return values


def _parse_float_csv(value: str) -> list[float]:
    values = [float(item) for item in value.split(',')]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError('expected comma-separated non-negative floats')
    return values


def _zipf_counts(total: int, alpha: float, num_rails: int) -> list[int]:
    weights = [(rail + 1) ** (-alpha) for rail in range(num_rails)]
    weight_sum = sum(weights)
    exact = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(num_rails),
        key=lambda rail: (-(exact[rail] - counts[rail]), rail),
    )
    for rail in order[:remainder]:
        counts[rail] += 1
    assert sum(counts) == total
    return counts


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


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
        'min_us': min(samples),
    }


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=11)
    assert world_size == 2 * num_local_ranks
    node = rank // num_local_ranks
    rail = rank % num_local_ranks
    remote_node = 1 - node
    num_experts = world_size
    num_max_tokens_per_rank = max(args.totals)

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
    assert buffer.get_logical_domain_size() == (2, num_local_ranks)

    if rank == 0:
        print('CONFIG ' + json.dumps({
            'policy': args.rail_balance,
            'totals_per_node': args.totals,
            'zipf_alphas': args.alphas,
            'hidden': args.hidden,
            'dtype': 'bfloat16',
            'topk': 1,
            'warmups': args.warmups,
            'iterations': args.iterations,
            'num_nodes': 2,
            'num_rails_per_node': num_local_ranks,
            'num_max_tokens_per_rank': num_max_tokens_per_rank,
            'buffer_gib_per_rank': buffer.num_bytes / (1024 ** 3),
        }), flush=True)

    for total in args.totals:
        for alpha in args.alphas:
            counts = _zipf_counts(total, alpha, num_local_ranks)
            num_tokens = counts[rail]
            token = torch.arange(num_tokens, dtype=torch.int64, device='cuda')
            src_global = rank * num_max_tokens_per_rank + token
            x_value = (src_global % 127 + 1).to(torch.bfloat16)
            x = x_value[:, None].expand(num_tokens, args.hidden).contiguous()
            expert = remote_node * num_local_ranks + rail
            topk_idx = torch.full(
                (num_tokens, 1), expert,
                dtype=deep_ep.topk_idx_t, device='cuda')
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

            recv_x, recv_topk_idx, recv_topk_weights, handle, _ = dispatch_once()
            num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
            assert num_recv == num_tokens
            expected_src = (
                (remote_node * num_local_ranks + rail) * num_max_tokens_per_rank +
                torch.arange(num_tokens, dtype=torch.int64, device='cuda'))
            src_ids = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)
            actual_src = torch.sort(src_ids).values
            assert torch.equal(actual_src, expected_src)
            expected_recv_value = (src_ids % 127 + 1).to(torch.bfloat16)
            assert torch.equal(recv_x[:num_recv, 0], expected_recv_value)
            assert torch.equal(
                recv_x[:num_recv],
                expected_recv_value[:, None].expand(num_recv, args.hidden),
            )

            combined_x, combined_weights, _ = buffer.combine(
                recv_x,
                handle=handle,
                topk_weights=recv_topk_weights,
                num_sms=args.num_sms,
            )
            if not torch.equal(combined_x, x):
                mismatch_rows = (combined_x != x).any(dim=1)
                mismatch_indices = mismatch_rows.nonzero().flatten()[:16]
                print(
                    f'COMBINE_MISMATCH rank={rank} total={total} alpha={alpha} '
                    f'num_tokens={num_tokens} mismatch_rows={mismatch_rows.sum().item()} '
                    f'indices={mismatch_indices.cpu().tolist()} '
                    f'actual={combined_x[mismatch_indices, 0].cpu().tolist()} '
                    f'expected={x[mismatch_indices, 0].cpu().tolist()} '
                    f'actual_head={combined_x[:16, 0].cpu().tolist()} '
                    f'expected_head={x[:16, 0].cpu().tolist()}',
                    flush=True,
                )
            assert torch.equal(combined_x, x)
            assert torch.equal(combined_weights, topk_weights)

            dispatch_stats = _measure_global_us(
                dispatch_once, args.warmups, args.iterations, group)

            recv_x, _, recv_topk_weights, handle, _ = dispatch_once()

            def combine_once():
                return buffer.combine(
                    recv_x,
                    handle=handle,
                    topk_weights=recv_topk_weights,
                    num_sms=args.num_sms,
                )

            combine_stats = _measure_global_us(
                combine_once, args.warmups, args.iterations, group)

            if rank == 0:
                payload_bytes = total * args.hidden * 2
                ep_total_us = dispatch_stats['median_us'] + combine_stats['median_us']
                ep_total_p90_us = dispatch_stats['p90_us'] + combine_stats['p90_us']
                result = {
                    'policy': args.rail_balance,
                    'tokens_per_node': total,
                    'payload_mib_per_node': payload_bytes / (1024 ** 2),
                    'zipf_alpha': alpha,
                    'counts': counts,
                    'max_over_mean': max(counts) / (total / num_local_ranks),
                    'dispatch_us': dispatch_stats['median_us'],
                    'dispatch_p90_us': dispatch_stats['p90_us'],
                    'dispatch_gbps_per_node': payload_bytes / dispatch_stats['median_us'] / 1e3,
                    'combine_us': combine_stats['median_us'],
                    'combine_p90_us': combine_stats['p90_us'],
                    'combine_gbps_per_node': payload_bytes / combine_stats['median_us'] / 1e3,
                    'ep_total_us': ep_total_us,
                    'ep_total_p90_us': ep_total_p90_us,
                    'ep_total_gbps_per_node': 2 * payload_bytes / ep_total_us / 1e3,
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
    parser.add_argument('--totals', type=_parse_int_csv, default=_parse_int_csv('2048,8192,16384'))
    parser.add_argument('--alphas', type=_parse_float_csv, default=_parse_float_csv('0,0.75,1.5,2.25'))
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--rail-balance', choices=('off', 'active'), required=True)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
