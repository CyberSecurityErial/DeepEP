"""Direct CUDA correctness runner for the private rail-balance planner.

This file intentionally does not depend on pytest.  Run it directly after the
extension has been rebuilt with the private planner binding::

    PYTHONPATH=$PWD python tests/elastic/test_rail_balance_plan_cuda.py

The output ABI is checked strictly so an accidental binding change fails close
to its source instead of being normalized by a test-side compatibility layer.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch

import deep_ep._C as _C
from rail_balance_reference import CountPlan, build_count_plan


def _matrix_tensor(matrix: Sequence[Sequence[int]], shape: tuple[int, int]) -> torch.Tensor:
    flat = [value for row in matrix for value in row]
    return torch.tensor(flat, dtype=torch.int32).reshape(shape)


def _expected_outputs(plan: CountPlan) -> tuple[torch.Tensor, ...]:
    num_rails = plan.num_rails
    num_destinations = plan.num_destinations
    max_segments = max(num_rails - 1, 0)

    quota = _matrix_tensor(plan.quota, (num_rails, num_destinations))
    keep_count = _matrix_tensor(
        plan.keep_count, (num_rails, num_destinations))
    segments = torch.full(
        (num_destinations, max_segments, 4), -1, dtype=torch.int32)
    num_segments = torch.zeros((num_destinations,), dtype=torch.int32)
    destination_offsets = [0 for _ in range(num_destinations)]
    for segment in plan.segments:
        offset = destination_offsets[segment.destination]
        assert offset < max_segments
        segments[segment.destination, offset] = torch.tensor(
            [segment.owner, segment.egress, segment.owner_begin, segment.count],
            dtype=torch.int32,
        )
        destination_offsets[segment.destination] += 1
        num_segments[segment.destination] += 1
    moved_copies = torch.tensor([plan.moved_copies], dtype=torch.int32)
    return quota, keep_count, segments, num_segments, moved_copies


def _assert_tensor_exact(
    label: str,
    actual: object,
    expected_cpu: torch.Tensor,
    device: torch.device,
) -> None:
    assert isinstance(actual, torch.Tensor), (
        f"{label}: expected Tensor, got {type(actual).__name__}")
    assert actual.dtype == torch.int32, f"{label}: expected int32, got {actual.dtype}"
    assert actual.device == device, f"{label}: expected {device}, got {actual.device}"
    assert tuple(actual.shape) == tuple(expected_cpu.shape), (
        f"{label}: expected shape {tuple(expected_cpu.shape)}, got {tuple(actual.shape)}")
    assert actual.is_contiguous(), f"{label}: output must be contiguous"
    actual_cpu = actual.detach().cpu()
    if not torch.equal(actual_cpu, expected_cpu):
        raise AssertionError(
            f"{label}: exact mismatch\nexpected={expected_cpu.tolist()}\n"
            f"actual={actual_cpu.tolist()}"
        )


def _run_case(
    name: str,
    counts: Sequence[Sequence[int]],
    remainder_seed: int,
    device: torch.device,
) -> None:
    plan = build_count_plan(counts, remainder_seed=remainder_seed)
    counts_cpu = _matrix_tensor(
        counts, (plan.num_rails, plan.num_destinations))
    counts_cuda = counts_cpu.to(device=device)

    try:
        result = _C._build_rail_balance_plan(counts_cuda, remainder_seed)
    except Exception as error:
        raise AssertionError(
            f"{name}: planner call failed on {device} with seed "
            f"{remainder_seed}") from error
    assert isinstance(result, tuple), (
        f"{name}: planner ABI must return tuple, got {type(result).__name__}")
    assert len(result) == 5, f"{name}: planner ABI returned {len(result)} fields"
    expected = _expected_outputs(plan)
    labels = ("quota", "keep_count", "segments", "num_segments", "moved_copies")
    for label, actual, expected_cpu in zip(labels, result, expected):
        _assert_tensor_exact(f"{name}/{label}", actual, expected_cpu, device)


def _fixed_cases() -> Iterable[tuple[str, Sequence[Sequence[int]], int]]:
    yield "g1_d0", [[]], 0
    yield "count_aware_remainder", [[4], [0], [0]], 1
    yield "balanced_identity", [
        [1, 2, 1, 3],
        [2, 1, 1, 2],
        [1, 1, 2, 2],
    ], 0
    yield "rotating_multi_destination", [
        [6, 6, 6, 6],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], 2
    yield "negative_seed", [
        [5, 1, 0, 7],
        [0, 2, 4, 0],
        [1, 3, 0, 0],
    ], -11
    large = [
        [
            (31 if destination % 29 == rail else
             (rail * 11 + destination * 7 + destination // 13) % 17)
            for destination in range(256)
        ]
        for rail in range(8)
    ]
    yield "g8_d256", large, 37


def _random_cases() -> Iterable[tuple[str, Sequence[Sequence[int]], int]]:
    # The complete small shape grid catches degenerate dimensions while three
    # deterministic data seeds exercise different quota remainders and segment
    # matchings without turning this functional runner into a stress test.
    for data_seed in (0x5EED, 0xC0FFEE, 0xBAD5EED):
        for num_rails in range(1, 9):
            for num_destinations in range(18):
                rng = random.Random(
                    data_seed ^ (num_rails << 20) ^ (num_destinations << 8))
                counts = [
                    [rng.randrange(32) for _ in range(num_destinations)]
                    for _ in range(num_rails)
                ]
                remainder_seed = rng.randrange(-4 * num_rails, 4 * num_rails + 1)
                name = (
                    f"random_seed{data_seed}_g{num_rails}_d{num_destinations}"
                )
                yield name, counts, remainder_seed


def _assert_fails(label: str, function) -> None:
    try:
        result = function()
        if isinstance(result, tuple):
            # Force any unexpectedly asynchronous validation path to surface.
            for value in result:
                if isinstance(value, torch.Tensor):
                    value.detach().cpu()
    except Exception:
        return
    raise AssertionError(f"{label}: expected planner call to fail")


def _test_invalid_inputs(device: torch.device) -> None:
    _assert_fails(
        "cpu tensor",
        lambda: _C._build_rail_balance_plan(
            torch.zeros((2, 2), dtype=torch.int32), 0),
    )
    _assert_fails(
        "noncontiguous tensor",
        lambda: _C._build_rail_balance_plan(
            torch.zeros((2, 3), dtype=torch.int32, device=device).transpose(0, 1), 0),
    )
    for dtype in (torch.int64, torch.bool, torch.float32):
        _assert_fails(
            f"wrong dtype {dtype}",
            lambda dtype=dtype: _C._build_rail_balance_plan(
                torch.zeros((2, 2), dtype=dtype, device=device), 0),
        )
    _assert_fails(
        "negative count",
        lambda: _C._build_rail_balance_plan(
            torch.tensor([[1, -1], [0, 2]], dtype=torch.int32, device=device), 0),
    )
    _assert_fails(
        "zero rails",
        lambda: _C._build_rail_balance_plan(
            torch.empty((0, 1), dtype=torch.int32, device=device), 0),
    )
    _assert_fails(
        "more than 32 rails",
        lambda: _C._build_rail_balance_plan(
            torch.zeros((33, 1), dtype=torch.int32, device=device), 0),
    )
    _assert_fails(
        "not 2D",
        lambda: _C._build_rail_balance_plan(
            torch.zeros((4,), dtype=torch.int32, device=device), 0),
    )
    _assert_fails(
        "destination total exceeds int32",
        lambda: _C._build_rail_balance_plan(
            torch.full(
                (2, 1), torch.iinfo(torch.int32).max,
                dtype=torch.int32, device=device),
            0,
        ),
    )
    for invalid_seed in (True, False, 1.0, "1", None, 1 << 70):
        _assert_fails(
            f"invalid remainder_seed {invalid_seed!r}",
            lambda invalid_seed=invalid_seed: _C._build_rail_balance_plan(
                torch.zeros((1, 1), dtype=torch.int32, device=device), invalid_seed),
        )


def _select_device_indices() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA device 0 is required for the planner CUDA runner")

    indices = [0]
    candidates = []
    for index in range(1, torch.cuda.device_count()):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        except RuntimeError:
            continue
        if free_bytes >= 512 * 1024 * 1024:
            candidates.append((free_bytes, index))
    if candidates:
        _free_bytes, index = max(candidates)
        indices.append(index)
    return indices


def _visible_identifier(device_index: int) -> str:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not configured:
        return str(device_index)
    identifiers = [item.strip() for item in configured.split(",")]
    if not 0 <= device_index < len(identifiers):
        raise RuntimeError("selected CUDA device is outside CUDA_VISIBLE_DEVICES")
    return identifiers[device_index]


def _run_visible_child() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cases = tuple(_fixed_cases()) + tuple(_random_cases())
    for name, counts, remainder_seed in cases:
        _run_case(name, counts, remainder_seed, device)
    source_index = os.environ.get("EP_RAIL_BALANCE_TEST_SOURCE_DEVICE", "0")
    print(f"PASS {len(cases)} exact planner cases on source CUDA device {source_index}")

    _test_invalid_inputs(device)
    print(f"PASS invalid-input contract on source CUDA device {source_index}")

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _run_case(
            "non_default_stream",
            [[7, 0, 2], [0, 5, 1], [1, 0, 4]],
            torch.iinfo(torch.int64).min,
            device,
        )
    stream.synchronize()
    print(f"PASS non-default stream on source CUDA device {source_index}")


def _test_same_process_cross_device_rejection(device_indices: Sequence[int]) -> None:
    if len(device_indices) < 2:
        return
    first = torch.device("cuda", device_indices[0])
    second = torch.device("cuda", device_indices[1])
    result = _C._build_rail_balance_plan(
        torch.tensor([[4], [0]], dtype=torch.int32, device=first), 0)
    result[-1].cpu()
    try:
        _C._build_rail_balance_plan(
            torch.tensor([[4], [0]], dtype=torch.int32, device=second), 0)
    except Exception as error:
        message = str(error)
        assert "rail-balance planner supports one CUDA device per process" in message, (
            "same-process cross-device rejection surfaced the wrong failure: "
            f"{message}"
        )
        assert "CUDA_ERROR_INVALID_HANDLE" not in message, message
    else:
        raise AssertionError(
            "same-process cross-device planner call unexpectedly succeeded")
    print("PASS explicit same-process cross-device rejection")


def main() -> None:
    if os.environ.get("EP_RAIL_BALANCE_TEST_CHILD") == "1":
        _run_visible_child()
        return

    device_indices = _select_device_indices()
    cases = tuple(_fixed_cases()) + tuple(_random_cases())
    script = str(Path(__file__).resolve())
    for device_index in device_indices:
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = _visible_identifier(device_index)
        child_env["EP_RAIL_BALANCE_TEST_CHILD"] = "1"
        child_env["EP_RAIL_BALANCE_TEST_SOURCE_DEVICE"] = str(device_index)
        subprocess.run(
            [sys.executable, "-B", script],
            env=child_env,
            check=True,
        )
    _test_same_process_cross_device_rejection(device_indices)
    print(
        f"PASS CUDA rail-balance planner: {len(cases) * len(device_indices)} "
        f"device-cases across {len(device_indices)} isolated process(es)"
    )


if __name__ == "__main__":
    main()
