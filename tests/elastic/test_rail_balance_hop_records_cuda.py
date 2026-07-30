"""GPU/CPU equality for hop-aware endpoint record materialization."""

from __future__ import annotations

import argparse
import random

import torch

import deep_ep._C as _C
from rail_balance_hop_reference import materialize_hop_records


def _dtype() -> torch.dtype:
    return torch.int64 if _C.topk_idx_t == torch.int64 else torch.int32


def _decode(raw: torch.Tensor) -> tuple:
    values = raw.cpu().tolist()
    return tuple(
        tuple(
            tuple((value & 0xFFFFFFFF, value >> 32) for value in token)
            for token in owner
        )
        for owner in values
    )


def _expected(topk_idx, *, num_experts, destinations, rails, local):
    records = materialize_hop_records(
        topk_idx,
        num_experts=num_experts,
        num_destinations=destinations,
        num_rails=rails,
        local_destination=local,
    )
    return tuple(
        tuple(
            tuple((record.target_mask, record.destination) for record in token)
            for token in owner
        )
        for owner in records
    )


def _run(topk_idx, *, num_experts, destinations, channels, local=0):
    tensor = torch.tensor(topk_idx, device="cuda", dtype=_dtype())
    records, status = _C._build_rail_balance_hop_records(
        tensor, channels, num_experts, destinations, local
    )
    assert records.dtype == torch.int64
    assert records.shape == tensor.shape
    assert records.is_contiguous() and records.device == tensor.device
    assert status.cpu().tolist() == [0]
    expected = _expected(
        topk_idx,
        num_experts=num_experts,
        destinations=destinations,
        rails=len(topk_idx),
        local=local,
    )
    assert _decode(records) == expected


def test_fixed_multitarget_and_padding() -> None:
    _run(
        (
            ((8, 10, 11, 0),),
            ((9, 14, 2, 3),),
            ((4, 5, 12, 15),),
            ((1, 6, 13, 7),),
        ),
        num_experts=16,
        destinations=2,
        channels=2,
    )


def test_random_endpoint_records() -> None:
    for seed in range(96):
        rng = random.Random(seed)
        rails = rng.choice((2, 4, 8))
        destinations = rng.choice((2, 4))
        num_topk = rng.choice((1, 2, 4, 8))
        experts_per_rank = max(2, num_topk)
        num_experts = destinations * rails * experts_per_rank
        num_tokens = rng.randint(1, 17)
        topk_idx = tuple(
            tuple(
                tuple(rng.sample(range(num_experts), num_topk))
                for _ in range(num_tokens)
            )
            for _ in range(rails)
        )
        _run(
            topk_idx,
            num_experts=num_experts,
            destinations=destinations,
            channels=rng.randint(1, 8),
            local=rng.randrange(destinations),
        )


def test_nondefault_stream() -> None:
    routes = tuple(
        tuple((owner, 8 + owner, 16 + owner, 24 + owner) for _ in range(5))
        for owner in range(4)
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _run(
            routes,
            num_experts=32,
            destinations=4,
            channels=3,
            local=0,
        )
    stream.synchronize()


def test_zero_tokens() -> None:
    routes = tuple(() for _ in range(4))
    tensor = torch.empty((4, 0, 4), device="cuda", dtype=_dtype())
    records, status = _C._build_rail_balance_hop_records(
        tensor, 2, 32, 2, 0
    )
    assert records.shape == (4, 0, 4)
    assert status.cpu().tolist() == [0]
    assert materialize_hop_records(
        routes,
        num_experts=32,
        num_destinations=2,
        num_rails=4,
        local_destination=0,
    ) == ((), (), (), ())


def test_invalid_routes_raise() -> None:
    duplicate = torch.tensor(
        [[[0, 0]], [[1, 2]]], device="cuda", dtype=_dtype()
    )
    try:
        _C._build_rail_balance_hop_records(duplicate, 1, 8, 2, 0)
    except RuntimeError as error:
        assert "duplicate expert" in str(error)
    else:
        raise AssertionError("duplicate route was accepted")

    invalid = duplicate.clone()
    invalid[0, 0] = torch.tensor((-1, 0), device="cuda", dtype=_dtype())
    try:
        _C._build_rail_balance_hop_records(invalid, 1, 8, 2, 0)
    except RuntimeError as error:
        assert "out-of-range" in str(error)
    else:
        raise AssertionError("masked route was accepted")


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
    print(f"PASS {len(tests)} hop-aware endpoint record CUDA tests")


if __name__ == "__main__":
    _main()
