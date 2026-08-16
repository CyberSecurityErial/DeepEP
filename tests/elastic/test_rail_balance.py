import argparse

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist


def _channel_count(num_tokens: int, channel: int, num_channels: int) -> int:
    return max(0, (num_tokens - channel + num_channels - 1) // num_channels)


def _reference_targets(
        counts, num_channels: int, destination: int, policy: str,
        rail_mask: int | None = None):
    if policy == 'off':
        return list(counts)
    if policy == 'incast' and rail_mask:
        selected = [
            rail for rail in range(len(counts))
            if (rail_mask >> rail) & 1
        ]
    elif policy == 'all':
        selected = list(range(len(counts)))
    else:
        selected = [i for i, count in enumerate(counts) if count > 0]
    targets = [0] * len(counts)
    for channel in range(num_channels):
        channel_counts = [_channel_count(count, channel, num_channels) for count in counts]
        total = sum(channel_counts)
        base, remainder = divmod(total, len(selected))
        rotation = (channel + destination) % len(selected)
        bonus_order = selected[rotation:] + selected[:rotation]
        for rail in selected:
            targets[rail] += base + int(rail in bonus_order[:remainder])
    return targets


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks, seed=7)
    assert world_size == 2 * num_local_ranks
    node = rank // num_local_ranks
    rail = rank % num_local_ranks
    remote_node = 1 - node

    counts = [800, 50, 50] + [0] * (num_local_ranks - 3)
    num_tokens = counts[rail] if rail < 3 else 1
    destination_node = remote_node if rail < 3 else node
    num_max_tokens_per_rank = 800
    hidden = args.hidden
    num_experts = world_size

    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=1,
        allow_hybrid_mode=True,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=True,
        num_allocated_qps=129,
        num_gpu_timeout_secs=60,
        explicitly_destroy=True,
        rail_balance=args.rail_balance,
    )
    assert buffer.get_logical_domain_size() == (2, num_local_ranks)

    incast_rail_mask = None
    if args.rail_balance == 'incast':
        local_demand = torch.zeros((2,), dtype=torch.float64, device='cpu')
        local_demand[destination_node] = num_tokens
        plan, observed_demand = buffer.update_incast_rail_masks(local_demand)
        assert observed_demand[0, 1].item() == sum(counts)
        assert observed_demand[1, 0].item() == sum(counts)
        assert observed_demand[0, 0].item() == num_local_ranks - 3
        assert observed_demand[1, 1].item() == num_local_ranks - 3
        if args.incast_mask >= 0:
            valid_mask = (1 << num_local_ranks) - 1
            if args.incast_mask == 0 or args.incast_mask & ~valid_mask:
                raise ValueError(
                    f'incast mask must select Rails in [0, {num_local_ranks})')
            plan[0, 1] = args.incast_mask
            plan[1, 0] = args.incast_mask
        buffer.set_incast_rail_masks(plan)
        installed = buffer.get_incast_rail_masks().cpu()
        assert torch.equal(installed, plan[node])
        incast_rail_mask = int(plan[node, remote_node].item()) & 0xFFFFFFFF
        if args.incast_mask < 0:
            assert incast_rail_mask == (1 << num_local_ranks) - 1

    token = torch.arange(num_tokens, device='cuda', dtype=torch.int64)
    src_global = rank * num_max_tokens_per_rank + token
    x_value = (src_global % 97 + 1).to(torch.bfloat16)
    x = x_value[:, None].expand(num_tokens, hidden).contiguous()

    expert = destination_node * num_local_ranks + rail
    topk_idx = torch.full(
        (num_tokens, 1), expert, dtype=deep_ep.topk_idx_t, device='cuda')
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float, device='cuda')

    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        expert_alignment=1,
        num_sms=args.num_sms,
        do_cpu_sync=True,
    )
    num_recv = handle.psum_num_recv_tokens_per_scaleup_rank[-1].item()
    src_ids = handle.recv_src_metadata[:num_recv, 0].to(torch.int64)

    source_node = remote_node if rail < 3 else node
    expected_num_recv = counts[rail] if rail < 3 else 1
    expected_src = (
        source_node * num_local_ranks + rail) * num_max_tokens_per_rank + \
        torch.arange(expected_num_recv, device='cuda', dtype=torch.int64)
    assert num_recv == expected_num_recv
    sorted_src_ids = torch.sort(src_ids).values
    if not torch.equal(sorted_src_ids, expected_src):
        source_counts = torch.bincount(
            torch.div(src_ids, num_max_tokens_per_rank, rounding_mode='floor'),
            minlength=world_size,
        )
        print(
            f'DISPATCH_MISMATCH rank={rank} rail={rail} num_recv={num_recv} '
            f'source_counts={source_counts.cpu().tolist()} '
            f'actual_head={sorted_src_ids[:16].cpu().tolist()} '
            f'actual_tail={sorted_src_ids[-16:].cpu().tolist()} '
            f'expected_head={expected_src[:16].cpu().tolist()} '
            f'expected_tail={expected_src[-16:].cpu().tolist()}',
            flush=True,
        )
    assert torch.equal(sorted_src_ids, expected_src)
    # DeepEP exposes expert indices relative to the receiving rank. This test
    # uses exactly one expert per rank, so every received local index is zero.
    expected_topk_idx = torch.zeros_like(recv_topk_idx[:num_recv, 0])
    if not torch.equal(recv_topk_idx[:num_recv, 0], expected_topk_idx):
        actual_values, actual_counts = torch.unique(
            recv_topk_idx[:num_recv, 0], return_counts=True)
        print(
            f'TOPK_MISMATCH rank={rank} rail={rail} num_recv={num_recv} '
            f'values={list(zip(actual_values.cpu().tolist(), actual_counts.cpu().tolist()))} '
            f'actual_head={recv_topk_idx[:min(num_recv, 32), 0].cpu().tolist()} '
            f'src_head={src_ids[:min(num_recv, 32)].cpu().tolist()}',
            flush=True,
        )
    assert torch.equal(recv_topk_idx[:num_recv, 0], expected_topk_idx)
    assert torch.equal(recv_topk_weights[:num_recv, 0], torch.ones_like(recv_topk_weights[:num_recv, 0]))
    expected_recv_x = (src_ids % 97 + 1).to(torch.bfloat16)
    assert torch.equal(recv_x[:num_recv, 0], expected_recv_x)
    assert torch.equal(recv_x[:num_recv], expected_recv_x[:, None].expand(num_recv, hidden))

    result_value = (src_ids % 53 + 3).to(torch.bfloat16)
    combine_input = result_value[:, None].expand(num_recv, hidden).contiguous()
    combined_x, combined_weights, _ = buffer.combine(
        combine_input,
        handle=handle,
        topk_weights=recv_topk_weights,
        num_sms=args.num_sms,
    )
    expected_combined = (src_global % 53 + 3).to(torch.bfloat16)
    assert torch.equal(combined_x[:, 0], expected_combined)
    assert torch.equal(
        combined_x, expected_combined[:, None].expand(num_tokens, hidden))
    assert torch.equal(combined_weights, topk_weights)

    num_channels = handle.token_metadata_at_forward.shape[0]
    targets = _reference_targets(
        counts, num_channels, remote_node, args.rail_balance,
        incast_rail_mask)
    assert sum(targets) == sum(counts)
    if args.rail_balance == 'off':
        assert targets == counts
    else:
        if args.rail_balance == 'incast':
            selected_targets = [
                target for rail_idx, target in enumerate(targets)
                if (incast_rail_mask >> rail_idx) & 1
            ]
            assert all(
                target == 0 for rail_idx, target in enumerate(targets)
                if not ((incast_rail_mask >> rail_idx) & 1)
            )
        else:
            selected_targets = (
                targets if args.rail_balance == 'all' else targets[:3])
        assert max(selected_targets) - min(selected_targets) <= 2
    if args.rail_balance == 'active':
        assert targets[3:] == [0] * (num_local_ranks - 3)

    dist.barrier()
    if rank == 0:
        print(
            f'PASS policy={args.rail_balance} channels={num_channels} '
            f'logical_counts={counts} planned_targets={targets}',
            flush=True,
        )
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=8)
    parser.add_argument('--num-sms', type=int, default=16)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument(
        '--rail-balance', choices=('off', 'active', 'all', 'incast'), required=True)
    parser.add_argument(
        '--incast-mask', type=lambda value: int(value, 0), default=-1,
        help='optional explicit Rail bitmask for the two-node incast test')
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
    )
