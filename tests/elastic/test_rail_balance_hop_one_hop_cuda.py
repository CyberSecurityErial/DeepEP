"""Invariants for the deterministic GPU one-hop assignment materializer."""

from __future__ import annotations

import argparse
import random

import torch

import deep_ep._C as _C


_UNUSED = -(1 << 32)


def _record(destination: int, target_mask: int) -> int:
    assert destination >= 0 and 0 < target_mask < (1 << 32)
    return (destination << 32) | target_mask


def _build(
    records,
    *,
    channels,
    destinations,
    capacity,
    seed=0,
    threshold_percent=0,
    max_two_hop_percent=0,
    hop_penalty_percent=0,
):
    tensor = torch.tensor(records, device="cuda", dtype=torch.int64)
    outputs = _C._build_rail_balance_hop_one_hop_plan(
        tensor,
        channels,
        destinations,
        tensor.size(1),
        capacity,
        seed,
        threshold_percent,
        max_two_hop_percent,
        hop_penalty_percent,
    )
    return tensor, tuple(output.cpu() for output in outputs)


def _decode_record(value: int) -> tuple[int, int]:
    return value >> 32, value & 0xFFFFFFFF


def _validate_greedy_choices(
    records: torch.Tensor, resolutions: torch.Tensor, *, channels: int, seed: int
) -> None:
    records = records.cpu()
    rails, num_tokens, num_topk = records.shape
    destinations = max(2, max(
        (_decode_record(int(value))[0] + 1 for value in records.flatten()
         if _decode_record(int(value))[1]),
        default=0,
    ))
    remaining = [[0] * destinations for _ in range(rails)]
    for owner in range(rails):
        for value in records[owner].flatten():
            destination, mask = _decode_record(int(value))
            if mask:
                remaining[owner][destination] += 1
    pair = [[0] * rails for _ in range(destinations)]
    source = [0] * rails
    groups = [[[0] * destinations for _ in range(channels)]
              for _ in range(rails)]
    for owner in range(rails):
        for token in range(num_tokens):
            for slot in range(num_topk):
                flat = (owner * num_tokens + token) * num_topk + slot
                destination, mask = _decode_record(
                    int(records[owner, token, slot])
                )
                if not mask:
                    continue
                remaining[owner][destination] -= 1
                candidates = [
                    egress for egress in range(rails)
                    if ((mask | (1 << owner)) & (1 << egress))
                    and pair[destination][egress] + 1
                    + remaining[egress][destination] <= num_tokens
                ]

                def score(
                    egress: int,
                    *,
                    destination: int = destination,
                    owner: int = owner,
                    mask: int = mask,
                    flat: int = flat,
                ) -> tuple[int, int, int, int, int]:
                    return (
                        max(load + (rail == egress)
                            for rail, load in enumerate(pair[destination])),
                        max(load + (rail == egress)
                            for rail, load in enumerate(source)),
                        int(egress != owner)
                        + (mask & ~(1 << egress)).bit_count(),
                        (egress - (seed + destination + flat) % rails) % rails,
                        egress,
                    )

                egress = min(candidates, key=score)
                assert resolutions[owner, token, slot, 0].item() == egress
                source_channel = token % channels
                channel_capacity = (num_tokens + channels - 1) // channels
                valid_channels = [
                    channel for channel in range(channels)
                    if groups[egress][channel][destination] < channel_capacity
                ]
                channel = min(valid_channels, key=lambda candidate: (
                    groups[egress][candidate][destination],
                    (candidate - source_channel) % channels,
                    candidate,
                ))
                assert resolutions[owner, token, slot, 1].item() == channel
                pair[destination][egress] += 1
                source[egress] += 1
                groups[egress][channel][destination] += 1


def _validate(
    records: torch.Tensor,
    outputs,
    *,
    channels: int,
    destinations: int,
    seed: int = 0,
    max_two_hop_percent: int = 0,
) -> None:
    (
        resolutions,
        pair_load,
        source_load,
        retained,
        moved,
        group_prefix,
        proxy_required,
        path_units,
        moved_copies,
        status,
    ) = outputs
    rails, num_tokens, num_topk = records.shape
    assert resolutions.shape == (rails, num_tokens, num_topk, 4)
    assert pair_load.shape == (destinations, rails)
    assert retained.shape == moved.shape == group_prefix.shape == (
        rails, channels, destinations
    )
    assert status.tolist() in ([0], [1], [4])

    expected_pair = torch.zeros_like(pair_load)
    expected_source = torch.zeros_like(source_load)
    expected_retained = torch.zeros_like(retained)
    expected_moved = torch.zeros_like(moved)
    expected_paths = torch.zeros_like(path_units)
    remote_slots: dict[tuple[int, int, int, bool], set[int]] = {}
    present = 0
    records_cpu = records.cpu()
    for owner in range(rails):
        for token in range(num_tokens):
            for slot in range(num_topk):
                destination, target_mask = _decode_record(
                    int(records_cpu[owner, token, slot])
                )
                egress, channel, remote_slot, proxy_slot = (
                    int(value) for value in resolutions[owner, token, slot]
                )
                if target_mask == 0:
                    assert destination == -1
                    assert (egress, channel, remote_slot, proxy_slot) == (
                        -1, -1, -1, -1
                    )
                    continue
                present += 1
                assert 0 <= destination < destinations
                assert 0 <= egress < rails
                endpoint = bool(
                    (target_mask | (1 << owner)) & (1 << egress))
                if max_two_hop_percent == 0:
                    assert endpoint
                assert 0 <= channel < channels and remote_slot >= 0
                is_moved = egress != owner
                if is_moved:
                    assert proxy_slot >= 0
                    expected_moved[egress, channel, destination] += 1
                else:
                    assert proxy_slot == -1
                    expected_retained[egress, channel, destination] += 1
                remote_slots.setdefault(
                    (egress, channel, destination, is_moved), set()
                ).add(remote_slot)
                expected_pair[destination, egress] += 1
                expected_source[egress] += 1
                if egress == owner:
                    path = 0 if target_mask == 1 << owner else 1
                elif target_mask & (1 << egress):
                    path = 2
                else:
                    path = 3
                expected_paths[path] += 1

    assert torch.equal(pair_load, expected_pair)
    assert torch.equal(source_load, expected_source)
    assert torch.equal(retained, expected_retained)
    assert torch.equal(moved, expected_moved)
    assert torch.equal(path_units, expected_paths)
    assert path_units[3].item() <= \
        present * max_two_hop_percent // 100
    assert path_units.sum().item() == present
    assert moved_copies.item() == moved.sum().item()
    assert torch.all(pair_load <= num_tokens)
    if status.tolist() != [4] and max_two_hop_percent == 0:
        _validate_greedy_choices(
            records, resolutions, channels=channels, seed=seed
        )

    for key, slots in remote_slots.items():
        egress, channel, destination, is_moved = key
        count = int((moved if is_moved else retained)[
            egress, channel, destination
        ])
        assert slots == set(range(count))

    for egress in range(rails):
        prefix = 0
        seen_proxy = set()
        for channel in range(channels):
            for destination in range(destinations):
                assert group_prefix[egress, channel, destination].item() == prefix
                for owner in range(rails):
                    for token in range(num_tokens):
                        for slot in range(num_topk):
                            resolution = resolutions[owner, token, slot]
                            if (
                                resolution[0].item() == egress
                                and resolution[1].item() == channel
                                and owner != egress
                            ):
                                record_destination, _ = _decode_record(
                                    int(records_cpu[owner, token, slot])
                                )
                                if record_destination == destination:
                                    proxy = resolution[3].item()
                                    assert proxy == prefix + resolution[2].item()
                                    seen_proxy.add(proxy)
                prefix += moved[egress, channel, destination].item()
        assert proxy_required[egress].item() == prefix
        assert seen_proxy == set(range(prefix))


def test_offdiagonal_one_hop_uses_both_endpoint_paths() -> None:
    records = []
    for owner in range(4):
        owner_records = []
        for token in range(12):
            if owner == 0:
                target = 1 + token % 3
                owner_records.append((_record(1, 1 << target),))
            else:
                owner_records.append((_UNUSED,))
        records.append(tuple(owner_records))
    tensor, outputs = _build(
        tuple(records), channels=3, destinations=2, capacity=64
    )
    _validate(tensor, outputs, channels=3, destinations=2)
    assert outputs[7][1].item() > 0
    assert outputs[7][2].item() > 0
    assert max(outputs[1][1]).item() < 12


def test_diagonal_has_no_one_hop_escape() -> None:
    records = tuple(
        tuple(
            (_record(1, 1),) if owner == 0 else (_UNUSED,)
            for _ in range(8)
        )
        for owner in range(4)
    )
    tensor, outputs = _build(
        records, channels=2, destinations=2, capacity=32
    )
    _validate(tensor, outputs, channels=2, destinations=2)
    assert outputs[1][1].tolist() == [8, 0, 0, 0]
    assert outputs[7].tolist() == [8, 0, 0, 0]


def test_adaptive_diagonal_uses_bounded_third_rail_escape() -> None:
    records = tuple(
        tuple(
            (_record(1, 1),) if owner == 0 else (_UNUSED,)
            for _ in range(8)
        )
        for owner in range(4)
    )
    tensor, outputs = _build(
        records, channels=2, destinations=2, capacity=32,
        max_two_hop_percent=50,
    )
    _validate(
        tensor, outputs, channels=2, destinations=2,
        max_two_hop_percent=50,
    )
    assert outputs[7].tolist() == [4, 0, 0, 4]
    assert max(outputs[1][1]).item() == 4


def test_adaptive_threshold_and_cap_stop_extra_hops() -> None:
    records = tuple(
        tuple(
            (_record(1, 1),) if owner == 0 else (_UNUSED,)
            for _ in range(8)
        )
        for owner in range(4)
    )
    _tensor, rejected = _build(
        records, channels=2, destinations=2, capacity=32,
        threshold_percent=100, max_two_hop_percent=50,
    )
    assert rejected[7].tolist() == [8, 0, 0, 0]

    tensor, capped = _build(
        records, channels=2, destinations=2, capacity=32,
        max_two_hop_percent=25,
    )
    _validate(
        tensor, capped, channels=2, destinations=2,
        max_two_hop_percent=25,
    )
    assert capped[7].tolist() == [6, 0, 0, 2]


def test_adaptive_batch_stops_at_the_next_hot_rail() -> None:
    records = []
    for owner, count in enumerate((10, 9, 0, 0)):
        records.append(
            tuple(
                (_record(1, 1 << owner),) if token < count else (_UNUSED,)
                for token in range(10)
            )
        )
    tensor, outputs = _build(
        tuple(records),
        channels=2,
        destinations=2,
        capacity=40,
        max_two_hop_percent=25,
    )
    _validate(
        tensor,
        outputs,
        channels=2,
        destinations=2,
        max_two_hop_percent=25,
    )
    assert outputs[7][3].item() == 4
    assert max(outputs[1][1]).item() <= 8


def test_random_one_hop_invariants_and_determinism() -> None:
    for seed in range(128):
        rng = random.Random(seed)
        rails = rng.choice((2, 4, 8))
        destinations = rng.choice((2, 4))
        num_tokens = rng.randint(1, 16)
        num_topk = rng.randint(1, min(4, destinations))
        records = []
        for _owner in range(rails):
            owner_records = []
            for _token in range(num_tokens):
                selected = sorted(rng.sample(
                    range(destinations), rng.randint(0, num_topk)
                ))
                token_records = [
                    _record(
                        destination,
                        sum(1 << target for target in rng.sample(
                            range(rails), rng.randint(1, min(3, rails))
                        )),
                    )
                    for destination in selected
                ]
                token_records.extend(
                    _UNUSED for _ in range(num_topk - len(token_records))
                )
                owner_records.append(tuple(token_records))
            records.append(tuple(owner_records))
        channels = rng.randint(1, 8)
        kwargs = {
            "channels": channels,
            "destinations": destinations,
            "capacity": rails * num_tokens * num_topk,
            "seed": rng.randrange(64),
        }
        tensor, first = _build(tuple(records), **kwargs)
        _tensor, second = _build(tuple(records), **kwargs)
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second))
        _validate(
            tensor,
            first,
            channels=channels,
            destinations=destinations,
            seed=kwargs["seed"],
        )


def test_proxy_capacity_and_corrupt_records_fail_closed() -> None:
    records = tuple(
        tuple(
            (_record(1, 1 << 1),) if owner == 0 else (_UNUSED,)
            for _ in range(8)
        )
        for owner in range(4)
    )
    tensor, outputs = _build(
        records, channels=2, destinations=2, capacity=0
    )
    _validate(tensor, outputs, channels=2, destinations=2)
    assert outputs[-1].tolist() == [1]

    corrupt = tensor.clone()
    corrupt[0, 0, 0] = _record(1, 1 << 4)
    outputs = tuple(output.cpu() for output in
                    _C._build_rail_balance_hop_one_hop_plan(
                        corrupt, 2, 2, 8, 64, 0
                    ))
    assert outputs[-1].tolist() == [4]


def test_retained_staging_capacity_fails_before_shuffle() -> None:
    records = tuple(
        tuple(
            (_record(1, 1 << owner), _record(2, 1 << owner))
            if owner == 0 else (_UNUSED, _UNUSED)
            for _ in range(4)
        )
        for owner in range(4)
    )
    _tensor, outputs = _build(
        records, channels=2, destinations=3, capacity=4
    )
    assert outputs[-1].tolist() == [1]


def test_random_adaptive_invariants_cap_and_determinism() -> None:
    for seed in range(64):
        rng = random.Random(1000 + seed)
        rails = rng.choice((3, 4, 8))
        num_tokens = rng.randint(1, 12)
        records = []
        for _owner in range(rails):
            owner_records = []
            for _token in range(num_tokens):
                target = rng.randrange(rails)
                owner_records.append((_record(1, 1 << target),))
            records.append(tuple(owner_records))
        ratio = rng.choice((10, 25, 50))
        kwargs = {
            "channels": rng.randint(1, 4),
            "destinations": 2,
            "capacity": rails * num_tokens,
            "seed": rng.randrange(64),
            "max_two_hop_percent": ratio,
            "hop_penalty_percent": rng.choice((0, 25, 50)),
        }
        tensor, first = _build(tuple(records), **kwargs)
        _tensor, second = _build(tuple(records), **kwargs)
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(first, second))
        _validate(
            tensor,
            first,
            channels=kwargs["channels"],
            destinations=2,
            max_two_hop_percent=ratio,
        )


def test_channel_masks_preserve_greedy_order_across_words() -> None:
    records = tuple(
        tuple(
            (_record(1, 1 << 1),) if owner == 0 else (_UNUSED,)
            for _ in range(513)
        )
        for owner in range(4)
    )
    tensor, outputs = _build(
        records, channels=256, destinations=2, capacity=2048
    )
    _validate(tensor, outputs, channels=256, destinations=2)


def test_zero_tokens() -> None:
    records = torch.empty((4, 0, 4), device="cuda", dtype=torch.int64)
    outputs = tuple(output.cpu() for output in
                    _C._build_rail_balance_hop_one_hop_plan(
                        records, 2, 2, 1, 0, 0
                    ))
    assert outputs[0].shape == (4, 0, 4, 4)
    assert outputs[-1].tolist() == [0]


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} hop-aware one-hop CUDA tests")


if __name__ == "__main__":
    _main()
