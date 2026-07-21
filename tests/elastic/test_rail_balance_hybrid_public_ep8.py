"""C080-H6 real-EP8 public constructor/dispatch watchdog.

The production capability stays disabled.  Workers temporarily enable the
installed host protocol to test real WORLD gates and truthful D=1 fail-close;
no source payload can be published on this topology.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import deep_ep
import deep_ep._C as _C
from deep_ep.buffers import elastic as elastic_module
from deep_ep.utils.envs import init_dist


_WORLD_SIZE, _GATE_WORDS = 8, 128
_MAX_TOKENS, _HIDDEN, _TOPK = 4, 256, 2
_PROXY_CAPACITY, _NUM_EXPERTS = 4, 8


def _assert_cpu_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    py_source = (root / "deep_ep/buffers/elastic.py").read_text(encoding="utf-8")
    cpp_source = (root / "csrc/elastic/rail_balance.hpp").read_text(encoding="utf-8")
    assert "_RAIL_BALANCE_FORCE_HOST_AVAILABLE = False" in py_source
    assert "return self._dispatch_rail_balance_force(" in py_source
    assert 'm.def("_rail_balance_force_available"' in cpp_source and "return false;" in cpp_source
    assert elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is False
    assert _C._rail_balance_force_available() is False
    required = ("_rail_balance_hybrid_dispatch_prepare",
                "_rail_balance_hybrid_plan_finish",
                "_rail_balance_hybrid_plan_abort")
    assert all(hasattr(_C.ElasticBuffer, name) for name in required)


def _report(label: str, error: str | None, group: dist.ProcessGroup, timeout: int) -> None:
    gathered: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, error, group=group)
    failures = [f"rank {rank}:\n{item}" for rank, item in enumerate(gathered) if item is not None]
    if failures:
        raise AssertionError(f"{label} failed:\n" + "\n".join(failures))
    dist.monitored_barrier(group=group, timeout=timedelta(seconds=timeout), wait_all_ranks=True)


def _constructor_kwargs(mode: str, timeout: int, hidden: int = _HIDDEN) -> dict:
    return dict(num_max_tokens_per_rank=_MAX_TOKENS, hidden=hidden,
                num_topk=_TOPK, use_fp8_dispatch=False, deterministic=False,
                allow_hybrid_mode=True, allow_multiple_reduction=True,
                prefer_overlap_with_compute=False, sl_idx=3,
                num_allocated_qps=1, num_cpu_timeout_secs=timeout,
                num_gpu_timeout_secs=timeout, explicitly_destroy=True,
                rail_balance=mode,
                rail_balance_proxy_slots_per_rank=(
                    _PROXY_CAPACITY if mode == "force" else 0))


def _expect_precomm_rejection(label, ep_group, control_group, timeout,
                              gate_pointers, mode, hidden, fragments) -> None:
    original = elastic_module.get_nccl_comm_handle
    comm_calls = 0
    gates_before = len(gate_pointers)

    def forbidden(*_args, **_kwargs):
        nonlocal comm_calls
        comm_calls += 1
        raise AssertionError("constructor crossed its pre-comm WORLD gate")

    caught = None
    elastic_module.get_nccl_comm_handle = forbidden
    try:
        try:
            deep_ep.ElasticBuffer(ep_group, **_constructor_kwargs(mode, timeout, hidden))
        except BaseException as exception:
            caught = exception
    finally:
        elastic_module.get_nccl_comm_handle = original
    error = None
    try:
        assert isinstance(caught, RuntimeError), repr(caught)
        assert all(fragment in str(caught) for fragment in fragments), str(caught)
        assert comm_calls == 0, comm_calls
        assert len(gate_pointers) == gates_before + 1
    except BaseException:
        error = traceback.format_exc()
    _report(label, error, control_group, timeout)


def _expect_d1_dispatch_rejection(label, buffer, rank, control_group,
                                  timeout, gate_pointers, expected_next) -> None:
    device = torch.device("cuda", rank)
    x = torch.arange(2 * _HIDDEN, dtype=torch.int32, device=device).reshape(
        2, _HIDDEN).to(torch.bfloat16)
    topk_idx = torch.tensor(
        ((rank, (rank + 1) % _NUM_EXPERTS),
         ((rank + 2) % _NUM_EXPERTS, (rank + 3) % _NUM_EXPERTS)),
        dtype=deep_ep.topk_idx_t, device=device)
    topk_weights = torch.ones((2, _TOPK), dtype=torch.float32, device=device)
    before, caught = len(gate_pointers), None
    try:
        buffer.dispatch(x, topk_idx, topk_weights,
                        num_experts=_NUM_EXPERTS,
                        num_max_tokens_per_rank=_MAX_TOKENS,
                        expert_alignment=1, num_sms=2, num_qps=1,
                        do_handle_copy=True, do_cpu_sync=True)
    except BaseException as exception:
        caught = exception
    error = None
    try:
        message = str(caught)
        assert isinstance(caught, RuntimeError), repr(caught)
        assert "CollectivePreflight" in message
        assert "dispatch-prepare rejected rank 0" in message
        assert "priority 31" in message
        assert len(gate_pointers) == before + 1
        assert buffer._rail_balance_next_invocation_id == expected_next
        assert buffer._rail_balance_live_ticket is None
        assert buffer._rail_balance_terminal is False
    except BaseException:
        error = traceback.format_exc()
    _report(label, error, control_group, timeout)


@torch.inference_mode()
def _worker(local_rank: int, num_processes: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(local_rank, num_processes, seed=806)
    control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo",
                                   timeout=timedelta(seconds=args.timeout))
    assert rank == local_rank and world_size == _WORLD_SIZE

    original_reduce = dist.all_reduce
    original_host = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    capability_name = "_rail_balance_force_available"
    original_capability = getattr(_C, capability_name)
    gate_pointers: list[int] = []

    def checked_reduce(tensor, op=None, group=None, async_op=False):
        assert tensor.dtype == torch.int64 and tuple(tensor.shape) == (_GATE_WORDS,)
        assert op == dist.ReduceOp.MAX and group is ep_group and async_op is False
        gate_pointers.append(tensor.data_ptr())
        return original_reduce(tensor, op=op, group=group, async_op=async_op)

    restored = False

    def restore() -> None:
        nonlocal restored
        if not restored:
            dist.all_reduce = original_reduce
            elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = original_host
            setattr(_C, capability_name, original_capability)
            restored = True

    buffer = None
    try:
        assert original_host is False and original_capability() is False
        dist.all_reduce = checked_reduce
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
        setattr(_C, capability_name, lambda: True)

        _expect_precomm_rejection(
            "mixed off/force Gate0", ep_group, control_group, args.timeout,
            gate_pointers, "off" if rank % 2 == 0 else "force", _HIDDEN,
            ("ConfigurationMismatch", "constructor field 5", "min=0", "max=1"))
        _expect_precomm_rejection(
            "single-rank invalid force Gate0", ep_group, control_group, args.timeout,
            gate_pointers, "force", 257 if rank == 3 else _HIDDEN,
            ("CollectivePreflight", "constructor rejected rank 3", "priority 11"))

        buffer = deep_ep.ElasticBuffer(ep_group, **_constructor_kwargs("force", args.timeout))
        device_ptr = buffer._rail_balance_world_gate_device_words.data_ptr()
        host_ptr = buffer._rail_balance_world_gate_host_words.data_ptr()
        error = None
        try:
            assert len(gate_pointers) == 4 and gate_pointers[2:] == [device_ptr] * 2
            assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)
            assert buffer._rail_balance_mode == "force"
            assert buffer._rail_balance_arena_offset > 0
            assert buffer._rail_balance_arena_bytes > 0
            assert buffer.num_bytes == (buffer._rail_balance_arena_offset +
                                        buffer._rail_balance_arena_bytes)
            assert buffer.num_allocated_qps == 1 and buffer.runtime is not None
        except BaseException:
            error = traceback.format_exc()
        _report("unanimous force constructor", error, control_group, args.timeout)

        _expect_d1_dispatch_rejection(
            "public D1 dispatch fail-close", buffer, rank, control_group,
            args.timeout, gate_pointers, 2)
        _expect_d1_dispatch_rejection(
            "public D1 dispatch retry", buffer, rank, control_group,
            args.timeout, gate_pointers, 3)
        error = None
        try:
            assert len(gate_pointers) == 6 and gate_pointers[2:] == [device_ptr] * 4
            assert buffer._rail_balance_world_gate_device_words.data_ptr() == device_ptr
            assert buffer._rail_balance_world_gate_host_words.data_ptr() == host_ptr
        except BaseException:
            error = traceback.format_exc()
        _report("stable six-gate lifecycle", error, control_group, args.timeout)

        buffer.destroy()
        error = None
        try:
            assert buffer.runtime is None and buffer.nccl_comm_handle is None
        except BaseException:
            error = traceback.format_exc()
        _report("collective force destroy", error, control_group, args.timeout)
        buffer = None

        restore()
        error = None
        try:
            assert elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is False
            assert _C._rail_balance_force_available() is False
        except BaseException:
            error = traceback.format_exc()
        _report("restore disabled production capability", error,
                control_group, args.timeout)
        if rank == 0:
            print("PASS C080-H6 public EP8: two constructor Gate0 rejects, "
                  "force construct/destroy, two retryable truthful D1 "
                  "dispatch rejects, six stable MAX gates", flush=True)
        dist.destroy_process_group()
    finally:
        restore()


def _run_watchdog(args: argparse.Namespace) -> None:
    command = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--worker-suite", "--num-processes", str(args.num_processes),
               "--timeout", str(args.timeout), "--master-port", str(args.master_port),
               "--watchdog-seconds", str(args.watchdog_seconds)]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.watchdog_seconds)
    except subprocess.TimeoutExpired as exception:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise RuntimeError(f"C080-H6 watchdog expired after {args.watchdog_seconds}s") from exception
    if return_code:
        # The worker-suite leader may have exited while a failed spawn child is
        # still alive in the dedicated session.  Never leave that group behind.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="C080-H6 public constructor/D1 EP8 watchdog")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29973)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    args = parser.parse_args()
    if not __debug__:
        parser.error("C080-H6 correctness checks require Python assertions")
    if args.num_processes != _WORLD_SIZE:
        parser.error("C080-H6 requires exactly eight processes")
    if args.timeout <= 0 or args.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")
    _assert_cpu_contract()
    print("PASS C080-H6 CPU/source: production host/compiled capability disabled", flush=True)
    if args.cpu_only:
        return
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-H6 requires at least eight visible CUDA devices")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.update(MASTER_PORT=str(args.master_port), WORLD_SIZE="1", RANK="0")
    os.environ["EP_DISABLE_GIN"] = "1"
    os.environ.pop("EP_OVERRIDE_RDMA_SL", None)
    if not args.worker_suite:
        _run_watchdog(args)
        return
    torch.multiprocessing.spawn(_worker, args=(args.num_processes, args),
                               nprocs=args.num_processes, join=True)


if __name__ == "__main__":
    main()
