"""Strict CUDA runner for the private C080-B1 Hybrid plan materializer.

The B1 binding is deliberately a dense-owner, single-process proof.  It does
not expose a production copy manifest and it does not pretend to perform the
later per-rank LSA count exchange.  Its frozen private ABI is::

    _build_rail_balance_hybrid_plan(
        topk_idx,                     # [G, N, K], CUDA, _C.topk_idx_t
        num_channels,
        num_max_tokens_per_rank,
        num_experts,
        num_scaleout_ranks,
        local_scaleout_rank,
        proxy_capacity_per_egress,
        remainder_seed=0,
    )

The return value has fourteen CUDA-contiguous int32 tensors in this order::

    channel_count[G,C,D], count[G,D], quota[G,D], keep_count[G,D],
    segments[D,G-1,5], num_segments[D], owner_channel_prefix[G,C,D],
    retained[G,C,D], moved[G,C,D], moved_channel_prefix[G,D,C+1],
    group_prefix[G,C,D], proxy_required[G], moved_copies[1], status[1]

``segments`` records are ``(owner, egress, owner_begin, count,
egress_begin)`` and unused records are all ``-1``.  ``status == 0`` means the
candidate fits; ``status == 1`` is the only non-throwing failure and means the
per-egress proxy capacity is insufficient.  The other thirteen tensors remain
the complete candidate in both cases.

Run the CPU/golden structure before the extension is rebuilt::

    PYTHONPATH=. python tests/elastic/test_rail_balance_hybrid_plan_cuda.py \
        --oracle-only

Run the CUDA comparison after rebuilding the private binding::

    PYTHONPATH=. python tests/elastic/test_rail_balance_hybrid_plan_cuda.py
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import torch

import deep_ep._C as _C
import test_rail_balance_hybrid_legacy_golden as legacy_golden
from rail_balance_hybrid_reference import (
    HybridRailSchedule,
    build_hybrid_rail_schedule,
)


_ABI_NAME = "_build_rail_balance_hybrid_plan"
_OUTPUT_LABELS = (
    "channel_count",
    "count",
    "quota",
    "keep_count",
    "segments",
    "num_segments",
    "owner_channel_prefix",
    "retained",
    "moved",
    "moved_channel_prefix",
    "group_prefix",
    "proxy_required",
    "moved_copies",
    "status",
)

# C061's three virtual destinations are all remote.  The strict Hybrid entry
# instead reserves real server zero as local, so the fixture is shifted to
# real destination servers 1, 2, and 3.  This preserves all 33 deduplicated
# destination copies and all six required moves.
_C061_VIRTUAL_ROUTES = (
    (
        (0, 1, 2, 0),
        (1, 2, 1, 0),
        (2, 2, 0, 0),
        (2, 0, 0, 2),
        (0, 0, 2, 2),
        (0, 2, 2, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ),
    (
        (1, 2, 1, 0),
        (2, 2, 0, 1),
        (2, 1, 1, 2),
        (1, 1, 2, 2),
        (1, 2, 2, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
        (1, 1, 1, 1),
    ),
)

# C090 keeps the distribution matrix deliberately small and deterministic.
# Each row below is one source rail and each column is one *remote*
# destination (real destination zero remains local).  Route tensors are built
# from these integer counts, so the fixtures exercise destination
# deduplication and the complete B1 materializer instead of injecting counts
# directly into the planner.
_ROUTE_DISTRIBUTION_CASES = (
    (
        "route_level_one_hot",
        (
            (24, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ),
        18,
        (0, 6, 6, 6),
    ),
    (
        "route_level_zipf",
        (
            # floor(16 / rank), rotated once per destination.
            (16, 4, 5),
            (8, 16, 4),
            (5, 8, 16),
            (4, 5, 8),
        ),
        21,
        (7, 4, 3, 7),
    ),
    (
        "route_level_log_normal",
        (
            # round(4 * exp(z)) for z=(1.5, .5, -.5, -1.5), then
            # rotated per destination: the fixed buckets are 18, 7, 2, 1.
            (18, 1, 2),
            (7, 18, 1),
            (2, 7, 18),
            (1, 2, 7),
        ),
        33,
        (11, 6, 5, 11),
    ),
)


@dataclass(frozen=True)
class PlanCase:
    name: str
    topk_idx: tuple[tuple[tuple[int, ...], ...], ...]
    num_channels: int
    num_max_tokens_per_rank: int
    num_experts: int
    num_scaleout_ranks: int
    local_scaleout_rank: int
    proxy_capacity_per_egress: int
    remainder_seed: int = 0
    num_topk_override: int | None = None

    @property
    def num_rails(self) -> int:
        return len(self.topk_idx)

    @property
    def num_tokens(self) -> int:
        return len(self.topk_idx[0])

    @property
    def num_topk(self) -> int:
        if self.num_topk_override is not None:
            return self.num_topk_override
        return len(self.topk_idx[0][0])


def _encode_destination_routes(
    routes: Sequence[Sequence[Sequence[int]]],
    *,
    num_destinations: int,
    experts_per_destination: int,
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], int]:
    """Encode destination lanes as distinct expert ids for the strict entry."""

    num_rails = len(routes)
    assert num_rails >= 2
    num_tokens = len(routes[0])
    assert num_tokens >= 1
    num_topk = len(routes[0][0])
    assert num_topk >= 1
    assert experts_per_destination >= num_topk
    assert experts_per_destination % num_rails == 0

    encoded_owners = []
    for owner_routes in routes:
        assert len(owner_routes) == num_tokens
        encoded_tokens = []
        for destinations in owner_routes:
            assert len(destinations) == num_topk
            occurrence = [0] * num_destinations
            experts = []
            for destination in destinations:
                assert 0 <= destination < num_destinations
                expert_lane = occurrence[destination]
                occurrence[destination] += 1
                assert expert_lane < experts_per_destination
                experts.append(
                    destination * experts_per_destination + expert_lane)
            assert len(experts) == len(set(experts))
            encoded_tokens.append(tuple(experts))
        encoded_owners.append(tuple(encoded_tokens))
    return (
        tuple(encoded_owners),
        num_destinations * experts_per_destination,
    )


def _fixed_route_distribution_cases() -> tuple[PlanCase, ...]:
    """Build the named C090 route fixtures without a stochastic sampler."""

    num_tokens = 32
    num_topk = 4
    cases = []
    for name, remote_counts, _moved_copies, _proxy_required in \
            _ROUTE_DISTRIBUTION_CASES:
        assert len(remote_counts) == 4
        routes = []
        for owner_counts in remote_counts:
            assert len(owner_counts) == 3
            assert all(count >= 0 for count in owner_counts)
            assert sum(owner_counts) <= num_tokens
            owner_routes = tuple(
                (destination,) * num_topk
                for destination, count in enumerate(owner_counts, start=1)
                for _ in range(count)
            )
            owner_routes += ((0,) * num_topk,) * (
                num_tokens - len(owner_routes))
            assert len(owner_routes) == num_tokens
            routes.append(owner_routes)

        topk_idx, num_experts = _encode_destination_routes(
            tuple(routes),
            num_destinations=4,
            experts_per_destination=8,
        )
        cases.append(PlanCase(
            name=name,
            topk_idx=topk_idx,
            num_channels=4,
            num_max_tokens_per_rank=num_tokens,
            num_experts=num_experts,
            num_scaleout_ranks=4,
            local_scaleout_rank=0,
            proxy_capacity_per_egress=num_tokens,
            remainder_seed=90,
        ))
    return tuple(cases)


def _build_schedule(case: PlanCase) -> HybridRailSchedule:
    return build_hybrid_rail_schedule(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=case.num_scaleout_ranks,
        local_scaleout_rank=case.local_scaleout_rank,
        num_channels=case.num_channels,
        num_max_tokens_per_rank=case.num_max_tokens_per_rank,
        proxy_capacity_per_egress=case.proxy_capacity_per_egress,
        remainder_seed=case.remainder_seed,
    )


def _int32_tensor(value: object) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int32).contiguous()


def _expected_outputs(schedule: HybridRailSchedule) -> tuple[torch.Tensor, ...]:
    segments = torch.full(
        (
            schedule.num_destinations,
            schedule.num_rails - 1,
            5,
        ),
        -1,
        dtype=torch.int32,
    )
    for destination, bucket in enumerate(schedule.segments):
        for index, segment in enumerate(bucket):
            segments[destination, index] = _int32_tensor((
                segment.owner,
                segment.egress,
                segment.owner_begin,
                segment.count,
                segment.egress_begin,
            ))

    values = (
        schedule.channel_count,
        schedule.count,
        schedule.quota,
        schedule.keep_count,
        None,
        schedule.num_segments,
        schedule.owner_channel_prefix,
        schedule.retained,
        schedule.moved,
        schedule.moved_channel_prefix,
        schedule.group_prefix,
        schedule.proxy_required,
        (schedule.moved_copies,),
        (0 if schedule.enabled else 1,),
    )
    outputs = tuple(
        segments if value is None else _int32_tensor(value)
        for value in values
    )
    assert len(outputs) == len(_OUTPUT_LABELS) == 14
    assert all(output.dtype == torch.int32 for output in outputs)
    assert all(output.is_contiguous() for output in outputs)
    return outputs


def _c061_case(*, proxy_capacity: int) -> PlanCase:
    remote_routes = tuple(
        tuple(tuple(destination + 1 for destination in token) for token in owner)
        for owner in _C061_VIRTUAL_ROUTES
    )
    topk_idx, num_experts = _encode_destination_routes(
        remote_routes,
        num_destinations=4,
        experts_per_destination=8,
    )
    assert num_experts == 32
    return PlanCase(
        name=f"c061_pcap{proxy_capacity}",
        topk_idx=topk_idx,
        num_channels=2,
        num_max_tokens_per_rank=10,
        num_experts=num_experts,
        num_scaleout_ranks=4,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=proxy_capacity,
        remainder_seed=62,
    )


def _boundary_case() -> PlanCase:
    # Exercise the accepted C=1024/D=32 maxima with real movement.  Owner zero
    # sends both tokens only to server 1 and owner one only to server 2.  The 32
    # lanes use distinct experts within the selected server, so destination
    # dedup reduces each token to one copy without violating the strict K ABI.
    routes = (
        ((1,) * 32, (1,) * 32),
        ((2,) * 32, (2,) * 32),
    )
    topk_idx, num_experts = _encode_destination_routes(
        routes,
        num_destinations=32,
        experts_per_destination=32,
    )
    return PlanCase(
        name="max_c1024_d32",
        topk_idx=topk_idx,
        num_channels=1024,
        num_max_tokens_per_rank=2,
        num_experts=num_experts,
        num_scaleout_ranks=32,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=1,
        remainder_seed=17,
    )


def _wide_destination_case() -> PlanCase:
    destinations = (1, 5, 9, 13, 17, 21, 25, 31)
    routes = tuple(
        ((destination,) * 4,) * 4
        for destination in destinations
    )
    topk_idx, num_experts = _encode_destination_routes(
        routes,
        num_destinations=32,
        experts_per_destination=8,
    )
    return PlanCase(
        name="all_owner_c1024_d32_k4",
        topk_idx=topk_idx,
        num_channels=1024,
        num_max_tokens_per_rank=4,
        num_experts=num_experts,
        num_scaleout_ranks=32,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=4,
        remainder_seed=17,
    )


def _zero_token_case() -> PlanCase:
    return PlanCase(
        name="zero_tokens",
        topk_idx=((), (), (), ()),
        num_channels=3,
        num_max_tokens_per_rank=7,
        num_experts=32,
        num_scaleout_ranks=4,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=1,
        remainder_seed=7,
        num_topk_override=4,
    )


def _random_cases(num_seeds: int) -> Iterable[PlanCase]:
    # Shapes stay fixed so random correctness does not create one JIT cubin per
    # seed.  Inputs still perturb dedup, local exclusion, remainder placement,
    # owner/channel prefixes, segment matching, and grouped proxy slots.
    for seed in range(num_seeds):
        randomizer = random.Random(0xC080B1 + seed)
        routes = tuple(
            tuple(
                tuple(randomizer.randrange(4) for _lane in range(8))
                for _token in range(13)
            )
            for _owner in range(4)
        )
        topk_idx, num_experts = _encode_destination_routes(
            routes,
            num_destinations=4,
            experts_per_destination=8,
        )
        provisional = PlanCase(
            name=f"random_seed_{seed}",
            topk_idx=topk_idx,
            num_channels=5,
            num_max_tokens_per_rank=13,
            num_experts=num_experts,
            num_scaleout_ranks=4,
            local_scaleout_rank=randomizer.randrange(4),
            proxy_capacity_per_egress=13 * 4,
            remainder_seed=randomizer.randrange(1 << 20),
        )
        schedule = _build_schedule(provisional)
        yield replace(
            provisional,
            proxy_capacity_per_egress=max(1, max(schedule.proxy_required)),
        )


def _assert_oracle_contract(num_random_seeds: int) -> tuple[PlanCase, ...]:
    success = _build_schedule(_c061_case(proxy_capacity=3))
    assert success.enabled
    assert success.local_destination == 0
    assert success.count == ((0, 9, 2, 6), (0, 2, 9, 5))
    assert sum(sum(row) for row in success.count) == 33
    assert success.moved_copies == 6
    assert success.proxy_required == (3, 3)

    failure = _build_schedule(_c061_case(proxy_capacity=2))
    assert not failure.enabled
    assert failure.failure_reason == "proxy_capacity_per_egress_exceeded"
    assert failure.proxy_required == success.proxy_required
    assert failure.moved_copies == success.moved_copies
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            _expected_outputs(failure)[:-1],
            _expected_outputs(success)[:-1],
        )
    )

    boundary = _boundary_case()
    boundary_schedule = _build_schedule(boundary)
    assert boundary_schedule.enabled
    assert boundary_schedule.num_channels == 1024
    assert boundary_schedule.num_destinations == 32
    assert boundary_schedule.moved_copies == 2
    assert boundary_schedule.proxy_required == (1, 1)

    wide = _wide_destination_case()
    wide_schedule = _build_schedule(wide)
    assert wide.num_scaleout_ranks == 32 > wide.num_topk == 4
    assert wide_schedule.enabled
    assert wide_schedule.moved_copies == 24
    assert wide_schedule.proxy_required == (4, 1, 4, 4, 3, 2, 3, 3)

    zero_tokens = _zero_token_case()
    zero_schedule = _build_schedule(zero_tokens)
    assert zero_schedule.enabled
    assert zero_schedule.num_tokens_per_owner == (0, 0, 0, 0)
    assert zero_schedule.count == ((0, 0, 0, 0),) * 4
    assert zero_schedule.quota == zero_schedule.keep_count == zero_schedule.count
    assert zero_schedule.moved_copies == 0
    assert zero_schedule.proxy_required == (0, 0, 0, 0)

    large_seed = replace(
        _c061_case(proxy_capacity=3),
        name="c061_int64_seed",
        remainder_seed=1 << 40,
    )
    assert _build_schedule(large_seed).remainder_seed == 0

    distribution_cases = _fixed_route_distribution_cases()
    assert len(distribution_cases) == len(_ROUTE_DISTRIBUTION_CASES)
    for case, expected in zip(
            distribution_cases, _ROUTE_DISTRIBUTION_CASES):
        name, remote_counts, expected_moved, expected_proxy = expected
        schedule = _build_schedule(case)
        expected_count = tuple((0, *row) for row in remote_counts)
        assert case.name == name
        assert schedule.enabled
        assert schedule.count == expected_count
        assert schedule.moved_copies == expected_moved
        assert schedule.proxy_required == expected_proxy

    random_cases = tuple(_random_cases(num_random_seeds))
    assert all(_build_schedule(case).enabled for case in random_cases)
    assert sum(
        _build_schedule(case).moved_copies for case in random_cases
    ) > 0
    return (_c061_case(proxy_capacity=3),
            _c061_case(proxy_capacity=2),
            zero_tokens,
            large_seed,
            wide,
            *distribution_cases,
            *random_cases,
            boundary)


def _assert_tensor_exact(
    label: str,
    actual: object,
    expected_cpu: torch.Tensor,
    device: torch.device,
) -> None:
    assert isinstance(actual, torch.Tensor), (
        f"{label}: expected Tensor, got {type(actual).__name__}")
    assert actual.dtype == torch.int32, (
        f"{label}: expected int32, got {actual.dtype}")
    assert actual.device == device, (
        f"{label}: expected {device}, got {actual.device}")
    assert tuple(actual.shape) == tuple(expected_cpu.shape), (
        f"{label}: expected shape {tuple(expected_cpu.shape)}, "
        f"got {tuple(actual.shape)}")
    assert actual.is_contiguous(), f"{label}: output must be contiguous"
    actual_cpu = actual.detach().cpu()
    if not torch.equal(actual_cpu, expected_cpu):
        mismatch = (actual_cpu != expected_cpu).nonzero(as_tuple=False)
        first = tuple(int(value) for value in mismatch[0].tolist())
        raise AssertionError(
            f"{label}: exact mismatch\n"
            f"first index={first}, expected={expected_cpu[first].item()}, "
            f"actual={actual_cpu[first].item()}, "
            f"total mismatches={mismatch.size(0)}"
        )


def _get_binding():
    if not hasattr(_C, _ABI_NAME):
        raise RuntimeError(
            f"deep_ep._C has no {_ABI_NAME}; rebuild the C080-B1 extension")
    return getattr(_C, _ABI_NAME)


def _call_binding(case: PlanCase, device: torch.device) -> tuple[object, ...]:
    if case.num_tokens == 0:
        topk_idx = torch.empty(
            (case.num_rails, 0, case.num_topk),
            dtype=_C.topk_idx_t,
            device=device,
        )
    else:
        topk_idx = torch.tensor(
            case.topk_idx,
            dtype=_C.topk_idx_t,
            device=device,
        ).contiguous()
    result = _get_binding()(
        topk_idx,
        case.num_channels,
        case.num_max_tokens_per_rank,
        case.num_experts,
        case.num_scaleout_ranks,
        case.local_scaleout_rank,
        case.proxy_capacity_per_egress,
        case.remainder_seed,
    )
    assert isinstance(result, tuple), (
        f"{case.name}: ABI must return tuple, got {type(result).__name__}")
    assert len(result) == len(_OUTPUT_LABELS), (
        f"{case.name}: ABI returned {len(result)} fields, expected 14")
    return result


def _run_cuda_case(
    case: PlanCase,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    schedule = _build_schedule(case)
    expected = _expected_outputs(schedule)
    actual = _call_binding(case, device)
    for label, actual_tensor, expected_tensor in zip(
        _OUTPUT_LABELS, actual, expected
    ):
        _assert_tensor_exact(
            f"{case.name}/{label}", actual_tensor, expected_tensor, device)
    return tuple(tensor.detach().cpu() for tensor in actual)


def _assert_host_rejects(
    label: str,
    message_fragment: str | None,
    function,
) -> None:
    try:
        function()
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        message = str(error)
        if message_fragment is not None:
            assert message_fragment in message, (
                f"{label}: wrong synchronous rejection: {message}")
        assert "CUDA_ERROR" not in message, message
        return
    raise AssertionError(f"{label}: invalid call unexpectedly succeeded")


def _replace_expert(
    case: PlanCase,
    *,
    owner: int,
    token: int,
    lane: int,
    expert: int,
) -> PlanCase:
    topk_idx = [
        [list(token_experts) for token_experts in owner_tokens]
        for owner_tokens in case.topk_idx
    ]
    topk_idx[owner][token][lane] = expert
    return replace(
        case,
        name=f"{case.name}_expert_{owner}_{token}_{lane}_{expert}",
        topk_idx=tuple(
            tuple(tuple(token_experts) for token_experts in owner_tokens)
            for owner_tokens in topk_idx
        ),
    )


def _test_route_rejections(device: torch.device) -> None:
    valid = _c061_case(proxy_capacity=3)
    masked = _replace_expert(
        valid, owner=0, token=0, lane=0, expert=-1)
    out_of_range = _replace_expert(
        valid, owner=0, token=0, lane=0, expert=valid.num_experts)
    duplicate = _replace_expert(
        valid,
        owner=0,
        token=0,
        lane=1,
        expert=valid.topk_idx[0][0][0],
    )
    _assert_host_rejects(
        "masked expert",
        "masked or out-of-range expert id",
        lambda: _call_binding(masked, device),
    )
    _assert_host_rejects(
        "out-of-range expert",
        "masked or out-of-range expert id",
        lambda: _call_binding(out_of_range, device),
    )
    _assert_host_rejects(
        "duplicate expert",
        "duplicate expert ids within one token",
        lambda: _call_binding(duplicate, device),
    )


def _test_seed_rejections(device: torch.device) -> None:
    class IntSubclass(int):
        pass

    valid = _c061_case(proxy_capacity=3)
    for label, seed, message in (
        ("bool remainder_seed", True, "PyLong_CheckExact"),
        ("int-subclass remainder_seed", IntSubclass(0), "PyLong_CheckExact"),
        ("negative remainder_seed", -1, "remainder_seed_i64"),
        ("oversized remainder_seed", 1 << 70, None),
    ):
        _assert_host_rejects(
            label,
            message,
            lambda seed=seed: _call_binding(
                replace(valid, name=label, remainder_seed=seed), device),
        )


def _test_non_default_stream(case: PlanCase, device: torch.device) -> None:
    expected = _expected_outputs(_build_schedule(case))
    stream = torch.cuda.Stream(device=device)
    assert stream != torch.cuda.default_stream(device)
    with torch.cuda.stream(stream):
        actual = _call_binding(
            replace(case, name=f"{case.name}_non_default_stream"), device)
    stream.synchronize()
    for label, actual_tensor, expected_tensor in zip(
        _OUTPUT_LABELS, actual, expected
    ):
        _assert_tensor_exact(
            f"non_default_stream/{label}",
            actual_tensor,
            expected_tensor,
            device,
        )


def _test_host_limits(device: torch.device) -> None:
    valid = _c061_case(proxy_capacity=3)
    _assert_host_rejects(
        "C=1025",
        "num_channels",
        lambda: _call_binding(replace(valid, num_channels=1025), device),
    )

    # D=33 is rejected before device content validation.  K remains at its
    # accepted maximum of 32 and the expert ids are valid and distinct.
    topk_idx = (
        (tuple(range(32)),),
        (tuple(range(32)),),
    )
    invalid_d = PlanCase(
        name="invalid_d33",
        topk_idx=topk_idx,
        num_channels=1,
        num_max_tokens_per_rank=1,
        num_experts=66,
        num_scaleout_ranks=33,
        local_scaleout_rank=0,
        proxy_capacity_per_egress=1,
    )
    _assert_host_rejects(
        "D=33",
        "num_scaleout_ranks",
        lambda: _call_binding(invalid_d, device),
    )


def _select_device(explicit_device: int | None) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("C080-B1 CUDA runner requires a visible CUDA device")
    if explicit_device is not None:
        if not 0 <= explicit_device < torch.cuda.device_count():
            raise ValueError("--device is outside the visible CUDA range")
        return torch.device("cuda", explicit_device)

    candidates = []
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        except RuntimeError:
            continue
        candidates.append((free_bytes, index))
    if not candidates:
        raise RuntimeError("no queryable CUDA device is available")
    _free_bytes, index = max(candidates)
    return torch.device("cuda", index)


def _run_legacy_goldens() -> None:
    legacy_golden.test_immutable_legacy_hybrid_files_match_sha256_golden()
    legacy_golden.test_recursive_include_hash_matches_deepep_algorithm()
    legacy_golden.test_legacy_layout_goldens_are_independently_reproduced()
    legacy_golden.test_transit_scratch_lifetime_and_rank_layout_contract()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-B1 strict CPU-oracle/CUDA Hybrid plan runner")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--random-seeds", type=int, default=64)
    parser.add_argument("--oracle-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.random_seeds < 1:
        parser.error("--random-seeds must be positive")

    cases = _assert_oracle_contract(arguments.random_seeds)
    _run_legacy_goldens()
    print(
        f"PASS C080-B1 CPU expected structure: C061 33 copies/6 moves, "
        f"Pcap 3/2, zero tokens, named route-level one-hot/Zipf/log-normal, "
        f"{arguments.random_seeds} random seeds, C1024/D32 including D32>K4")
    print("PASS 4/4 legacy Hybrid identity goldens")
    if arguments.oracle_only:
        return

    device = _select_device(arguments.device)
    torch.cuda.set_device(device)
    success = _run_cuda_case(cases[0], device)
    failure = _run_cuda_case(cases[1], device)
    assert all(torch.equal(left, right) for left, right in zip(
        success[:-1], failure[:-1]
    ))
    assert success[-1].tolist() == [0]
    assert failure[-1].tolist() == [1]
    _test_route_rejections(device)
    _test_seed_rejections(device)
    _test_non_default_stream(cases[0], device)
    for case in cases[2:]:
        _run_cuda_case(case, device)
    _test_host_limits(device)
    print(
        f"PASS C080-B1 CUDA strict plan: {len(cases)} exact cases on "
        f"{device}, route/seed/C1025/D33 rejection, non-default stream")


if __name__ == "__main__":
    main()
