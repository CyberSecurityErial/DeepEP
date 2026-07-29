"""C080-H3 fixed-tensor WORLD consensus tests.

Run directly; pytest is intentionally not required.  The GPU suite requires
exactly eight visible GPUs and proves only the private gate protocol.  It does
not enable or enter the force Hybrid transaction.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from deep_ep.buffers import elastic as elastic_module
from deep_ep.utils.envs import init_dist


_WORLD_SIZE = 8
_WORDS = 128
_INT64_MAX = (1 << 63) - 1


def _encode(local_error_key: int, fields: tuple[int, ...]) -> torch.Tensor:
    words = torch.empty(_WORDS, dtype=torch.int64, device="cpu")
    elastic_module._encode_rail_balance_world_gate(
        words, local_error_key, fields)
    return words


def _reduce_cpu(payloads: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(payloads).amax(dim=0)


def _assert_cpu_reference() -> None:
    encode = elastic_module._encode_rail_balance_world_gate
    decode = elastic_module._decode_rail_balance_world_gate
    make_error = elastic_module._make_rail_balance_world_gate_error_key
    decode_error = elastic_module._decode_rail_balance_world_gate_error_key

    boundary = _encode(0, (0, _INT64_MAX))
    assert boundary[0].item() == 0 and boundary[1].item() == 0
    assert boundary[2:6].tolist() == [0, 0, _INT64_MAX, -_INT64_MAX]
    assert boundary[6:].count_nonzero().item() == 0
    assert decode(boundary) == (0, -1, 0, 0)

    full = torch.empty(_WORDS, dtype=torch.int64, device="cpu")
    encode(full, 0, tuple(range(63)))
    assert full[-2:].tolist() == [62, -62]
    assert decode(full) == (0, -1, 0, 0)

    mismatch = _reduce_cpu((_encode(0, (5, 7, 11)),
                            _encode(0, (5, 9, 13))))
    assert decode(mismatch) == (0, 1, 7, 9)

    error_rank_5 = make_error(4, 5)
    error_rank_2 = make_error(4, 2)
    error_high = make_error(7, 6)
    assert error_rank_5 < error_rank_2 < error_high <= _INT64_MAX
    assert decode_error(0) == (0, -1)
    assert decode_error(error_rank_2) == (4, 2)
    error_first = _reduce_cpu((_encode(error_rank_5, (1,)),
                               _encode(error_high, (2,))))
    assert decode(error_first) == (error_high, -1, 0, 0)

    invalid = (
        lambda: encode(torch.empty(_WORDS, dtype=torch.int64), 0,
                       tuple(range(64))),
        lambda: encode(torch.empty(_WORDS, dtype=torch.int64), -1, ()),
        lambda: encode(torch.empty(_WORDS, dtype=torch.int64), 0, (-1,)),
        lambda: encode(torch.empty(_WORDS, dtype=torch.int64), 0,
                       (_INT64_MAX + 1,)),
        lambda: make_error(-1, 0),
        lambda: make_error(1, 1 << 32),
    )
    for operation in invalid:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid WORLD-gate input was accepted")


def _worker(local_rank: int, num_processes: int) -> None:
    rank, world_size, group = init_dist(local_rank, num_processes)
    assert rank == local_rank and world_size == _WORLD_SIZE
    device = torch.device("cuda", local_rank)
    device_words = torch.empty(
        _WORDS, dtype=torch.int64, device=device)
    host_words = torch.empty(
        _WORDS, dtype=torch.int64, device="cpu", pin_memory=True)
    elastic_module._validate_rail_balance_world_gate_storage(
        device_words, host_words)
    device_ptr = device_words.data_ptr()
    host_ptr = host_words.data_ptr()

    original_all_reduce = dist.all_reduce
    original_all_gather = dist.all_gather
    original_all_gather_into_tensor = dist.all_gather_into_tensor
    original_all_gather_object = dist.all_gather_object
    original_gather_object = dist.gather_object
    original_device_sync = torch.cuda.synchronize
    calls = 0

    def checked_all_reduce(tensor, op=None, group=None, async_op=False):
        nonlocal calls
        calls += 1
        assert tensor.data_ptr() == device_ptr
        assert tensor.dtype == torch.int64
        assert tuple(tensor.shape) == (_WORDS,)
        assert op == dist.ReduceOp.MAX
        assert group is not None
        assert async_op is False
        return original_all_reduce(
            tensor, op=op, group=group, async_op=async_op)

    def forbidden_collective(*_args, **_kwargs):
        raise AssertionError("fixed WORLD gate entered a gather collective")

    def forbidden_device_sync(*_args, **_kwargs):
        raise AssertionError("fixed WORLD gate synchronized the whole device")

    dist.all_reduce = checked_all_reduce
    dist.all_gather = forbidden_collective
    dist.all_gather_into_tensor = forbidden_collective
    dist.all_gather_object = forbidden_collective
    dist.gather_object = forbidden_collective
    torch.cuda.synchronize = forbidden_device_sync

    def run(local_error_key: int,
            fields: tuple[int, ...],
            stream: torch.cuda.Stream | None = None) -> tuple[int, int, int, int]:
        elastic_module._encode_rail_balance_world_gate(
            host_words, local_error_key, fields)
        assert device_words.data_ptr() == device_ptr
        assert host_words.data_ptr() == host_ptr
        if stream is None:
            result = elastic_module._run_rail_balance_world_gate(
                device_words, host_words, group)
        else:
            with torch.cuda.stream(stream):
                assert torch.cuda.current_stream(device).cuda_stream == \
                    stream.cuda_stream
                result = elastic_module._run_rail_balance_world_gate(
                    device_words, host_words, group)
        assert device_words.data_ptr() == device_ptr
        assert host_words.data_ptr() == host_ptr
        return result

    expected_rounds = 0
    try:
        fields = (0, _INT64_MAX, 8, 1024, 7168, 8, 32)
        assert run(0, fields) == (0, -1, 0, 0)
        expected_rounds += 1
        assert host_words[1].item() == 0
        assert host_words[2 + 2 * len(fields):].count_nonzero().item() == 0

        mismatch = (19, 23, 29 if rank != 5 else 31, 37)
        assert run(0, mismatch) == (0, 2, 29, 31)
        expected_rounds += 1

        # Local token count intentionally differs and is not a common field.
        local_num_tokens = rank * 17
        assert local_num_tokens != (rank + 1) * 17
        assert run(0, (101, 103, 107)) == (0, -1, 0, 0)
        expected_rounds += 1

        make_error = elastic_module._make_rail_balance_world_gate_error_key
        if rank in (2, 6):
            local_error = make_error(9, rank)
        elif rank == 0:
            local_error = make_error(8, rank)
        else:
            local_error = 0
        winning_error = make_error(9, 2)
        # The mismatch must not outrank a synchronized local error.
        assert run(local_error, (41 if rank == 7 else 43,)) == \
            (winning_error, -1, 0, 0)
        expected_rounds += 1
        assert elastic_module._decode_rail_balance_world_gate_error_key(
            winning_error) == (9, 2)

        for repetition in range(3):
            assert run(0, (211, repetition)) == (0, -1, 0, 0)
            expected_rounds += 1

        caller_stream = torch.cuda.Stream(device=device)
        assert run(0, (307, 311), caller_stream) == (0, -1, 0, 0)
        expected_rounds += 1
        assert calls == expected_rounds
    finally:
        dist.all_reduce = original_all_reduce
        dist.all_gather = original_all_gather
        dist.all_gather_into_tensor = original_all_gather_into_tensor
        dist.all_gather_object = original_all_gather_object
        dist.gather_object = original_gather_object
        torch.cuda.synchronize = original_device_sync

    dist.barrier(group=group)
    if rank == 0:
        print(
            "PASS C080-H3 fixed WORLD gate: equal/mismatch/error-first, "
            "variable N, stable storage, non-default stream, one MAX/round",
            flush=True,
        )
    dist.destroy_process_group()


def _run_watchdog(arguments: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        "--num-processes", str(arguments.num_processes),
        "--master-port", str(arguments.master_port),
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=arguments.watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "C080-H3 WORLD-gate watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-H3 fixed CUDA WORLD-gate test")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--master-port", type=int, default=29943)
    parser.add_argument("--watchdog-seconds", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-H3 requires exactly 8 processes")
    if arguments.watchdog_seconds <= 0:
        parser.error("watchdog must be positive")

    _assert_cpu_reference()
    print("PASS C080-H3 fixed WORLD gate CPU reference", flush=True)
    if arguments.cpu_only:
        return
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-H3 requires at least 8 visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    if not arguments.worker_suite:
        _run_watchdog(arguments)
        return

    os.environ["MASTER_PORT"] = str(arguments.master_port)
    torch.multiprocessing.spawn(
        _worker,
        args=(arguments.num_processes,),
        nprocs=arguments.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
