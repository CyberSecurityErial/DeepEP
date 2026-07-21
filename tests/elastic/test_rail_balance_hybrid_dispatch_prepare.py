"""C080-H4b owning Hybrid-dispatch prepare contract.

H4b deliberately does not enable the public ``rail_balance='force'`` path and
does not submit either source payloads or the force Hybrid dispatch kernel.
This test freezes the complete pre-Gate1 owning bundle and exercises the only
truthful runtime boundary available on this single-node machine: its physical
logical topology is ``D=1, G=8``, so production Hybrid prepare must fail before
publication.  The same runtime must then accept a complete virtual-destination
planner transaction, proving that the rejected prepare leaked no pending
transaction.

Run the source/CPU contract while implementing H4b::

    PYTHONPATH=.:tests/elastic \
      /home/chen/.cache/deepep-sjlgpt/bin/python -B \
      tests/elastic/test_rail_balance_hybrid_dispatch_prepare.py --cpu-only

Run the strict eight-H200 path after rebuilding the extension::

    EP_DISABLE_GIN=1 PYTHONPATH=.:tests/elastic \
      /home/chen/.cache/deepep-sjlgpt/bin/python -B \
      tests/elastic/test_rail_balance_hybrid_dispatch_prepare.py

This is prepare-structure and fail-close evidence.  It is not real Hybrid
Rail/Gin runtime evidence.
"""

from __future__ import annotations

import argparse
import os
import re
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
from deep_ep.utils.math import align
from test_rail_balance_hybrid_plan_lsa import (
    _ARENA_PROXY_CAPACITY,
    _HIDDEN,
    _MAX_TOKENS,
    _MAX_TOPK,
    _WORLD_SIZE,
    _assert_oracle_contract,
    _local_topk,
)
from test_rail_balance_hybrid_plan_world_gate import (
    _GATE_WORDS,
    _abort,
    _gate,
    _make_error,
    _report_local_result,
    _run_success_or_gate2_failure,
)


_ROOT = Path(__file__).resolve().parents[2]
_PENDING_HEADER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp")
_BUFFER_HEADER = _ROOT / "csrc/elastic/buffer.hpp"
_PYTHON_BUFFER = _ROOT / "deep_ep/buffers/elastic.py"

_MANIFEST_MAGIC = int.from_bytes(b"RBH4", byteorder="little")
_MANIFEST_VERSION = 1
_MANIFEST_OPERATION_DISPATCH_PREPARE = 2
_MANIFEST_WIDTH = 9
_PREPARE_EXCEPTION_PRIORITY = 10


def _balanced_brace_section(source: str, anchor: str) -> str:
    """Return one C++ declaration/function containing balanced braces."""

    try:
        start = source.index(anchor)
    except ValueError as error:
        raise AssertionError(
            f"missing expected H4b production section: {anchor}") from error
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated C++ section: {anchor}")


def _assert_in_order(source: str, tokens: tuple[str, ...]) -> None:
    positions = [source.index(token) for token in tokens]
    assert positions == sorted(positions), (tokens, positions)


def _assert_owning_bundle_contract() -> None:
    pending_source = _PENDING_HEADER.read_text(encoding="utf-8")
    buffer_source = _BUFFER_HEADER.read_text(encoding="utf-8")

    bundle = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchBundle")
    pending = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridPlanPending")
    raw = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchRawPointers")
    prepare = _balanced_brace_section(
        buffer_source, "rail_balance_hybrid_dispatch_prepare(")
    signature = prepare[:prepare.index("{")]

    # Tensor handles own every storage whose raw address survives Gate1.  A
    # Python caller dropping or rebinding its local reference therefore cannot
    # release that storage.  This contract intentionally does not promise that
    # an unsupported concurrent in-place resize/write of the same TensorImpl is
    # isolated; making that true would require copying the complete payload.
    owning_fields = (
        "torch::Tensor x;",
        "torch::Tensor topk_idx;",
        "torch::Tensor topk_weights;",
        "torch::Tensor cumulative_local_expert_recv_stats;",
        "torch::Tensor copied_topk_idx;",
        "torch::Tensor psum_num_recv_tokens_per_scaleup_rank;",
        "torch::Tensor psum_num_recv_tokens_per_expert_storage;",
        "torch::Tensor psum_num_recv_tokens_per_expert;",
        "torch::Tensor num_unaligned_recv_tokens_per_expert;",
        "torch::Tensor dst_buffer_slot_idx;",
        "torch::Tensor token_metadata_at_forward;",
        "torch::Tensor channel_linked_list;",
    )
    for field in owning_fields:
        assert field in bundle, field
    for field in (
        "std::shared_ptr<PreparedRailBalanceHybridDispatch> main_dispatch;",
        "std::shared_ptr<PreparedRailBalanceHybridDispatchEpilogue>",
        "std::shared_ptr<PreparedRailBalanceHybridCombine> main_combine;",
        "RailBalanceHybridDispatchRawPointers raw;",
    ):
        assert field in bundle, field
    assert (
        "std::shared_ptr<RailBalanceHybridDispatchBundle> dispatch_bundle;"
        in pending)

    # The committed H4c launch must not rediscover a Tensor address.  Freeze
    # the main-dispatch ABI addresses now while the owning handles above exist.
    raw_fields = (
        "void* x;",
        "topk_idx_t* topk_idx;",
        "float* topk_weights;",
        "topk_idx_t* copied_topk_idx;",
        "int* cumulative_local_expert_recv_stats;",
        "int* psum_num_recv_tokens_per_scaleup_rank;",
        "int* psum_num_recv_tokens_per_expert_storage;",
        "int* psum_num_recv_tokens_per_expert_inclusive;",
        "int* num_unaligned_recv_tokens_per_expert;",
        "int* dst_buffer_slot_idx;",
        "int* token_metadata_at_forward;",
        "int* channel_linked_list;",
        "void* buffer;",
        "void* workspace;",
        "void* mapped_host_workspace;",
        "void* arena;",
        "void* proxy_dispatch_base;",
        "void* proxy_return_base;",
        "void* legacy_reduce_buffer_base;",
        "const int* retained;",
        "const int* moved;",
        "const int* group_prefix;",
        "const int* proxy_required;",
    )
    for field in raw_fields:
        assert field in raw, field

    # D, G, local identities, H, K, and C are production topology/shape facts;
    # allowing a private caller to supply any of them would recreate the old
    # virtual-manifest split brain at the public integration boundary.
    for forbidden_parameter in (
        "num_scaleout_ranks",
        "num_scaleup_ranks",
        "local_scaleout_rank",
        "local_scaleup_rank",
        "num_channels",
        "hidden",
        "num_topk",
    ):
        assert forbidden_parameter not in signature, forbidden_parameter
    for derived in (
        "nccl_context->num_scaleout_ranks",
        "nccl_context->num_scaleup_ranks",
        "nccl_context->scaleout_rank_idx",
        "nccl_context->scaleup_rank_idx",
        "num_channels_per_sm",
        "get_dispatch_token_layout(",
        "get_combine_token_layout(",
        "get_num_notify_smem_bytes(",
        "prefer_overlap_with_compute",
    ):
        assert derived in prepare, derived
    assert re.search(
        r"num_channels_i64\s*=\s*\n?\s*"
        r"static_cast<int64_t>\(num_sms\)\s*\*\s*"
        r"num_channels_per_sm",
        prepare)
    assert (
        "const int num_channels = static_cast<int>(num_channels_i64);"
        in prepare)

    # Force forwarding adds exactly one proxy-slot scalar to legacy 2+2K
    # metadata.  The legacy registered buffer ends at arena_offset; the force
    # tail must never make an undersized legacy dispatch layout appear valid.
    # One occurrence sizes the owning Tensor and one exports the frozen Gate1
    # field; there is no separately recomputed legacy-width variable.
    assert prepare.count("3 + num_topk * 2") == 2
    dispatch_size = prepare.index("get_dispatch_buffer_size(")
    arena_bound = prepare.index("arena_offset", dispatch_size)
    assert dispatch_size < arena_bound
    bound_window = prepare[dispatch_size:arena_bound + len("arena_offset")]
    assert "num_gpu_buffer_bytes" not in bound_window

    # All allocations, JIT builds and address extraction precede ownership.
    # PREPARING is installed before the first arena-writing count launch, so a
    # launch/readback exception remains recoverable by invocation-scoped abort.
    _assert_in_order(prepare, (
        "allocate_rail_balance_hybrid_plan_outputs(",
        "prepare_rail_balance_hybrid_plan(",
        "prepare_rail_balance_hybrid_dispatch(",
        "prepare_rail_balance_hybrid_dispatch_epilogue(",
        "prepare_rail_balance_hybrid_combine(",
        ".data_ptr",
        "std::make_shared<RailBalanceHybridDispatchBundle>",
        "rail_balance_hybrid_plan_pending.emplace(",
        "launch_prepared_rail_balance_hybrid_count(",
        "cudaMemcpyAsync(",
        "cudaStreamSynchronize(",
        "return {",
    ))
    ownership = prepare.index("rail_balance_hybrid_plan_pending.emplace(")
    pre_gate1_suffix = prepare[ownership:]
    for forbidden in (
        "torch::empty",
        "torch::zeros",
        "torch::full",
        ".data_ptr",
        "jit::compiler",
        "prepare_rail_balance_",
        "submit_prepared_",
    ):
        assert forbidden not in pre_gate1_suffix, forbidden

    # Python must encode the fixed WORLD manifest from C++-validated fields,
    # not recompute C, launch geometry, timeout, or metadata width itself.
    result_begin = prepare.rindex("return {")
    result = prepare[result_begin:]
    _assert_in_order(result, (
        "host_status",
        "num_channels",
        "num_channels_per_sm",
        "num_destinations",
        "num_rails",
        "hidden",
        "num_topk",
        "num_max_tokens_per_rank",
        "num_experts",
        "proxy_capacity_per_egress",
        "num_sms",
        "dispatch_spec.num_notify_warps",
        "dispatch_spec.num_scaleout_warps",
        "dispatch_spec.num_forward_warps",
        "num_smem_bytes",
        "num_qps",
        "num_gpu_timeout_cycles",
        "arena_offset",
        "arena_layout.arena_bytes",
        "3 + num_topk * 2",
    ))
    for rank_local in (
        "num_tokens",
        "scaleout_rank_idx",
        "scaleup_rank_idx",
        "rank_idx",
    ):
        assert not re.search(rf"\b{rank_local}\b", result), rank_local

    finish = _balanced_brace_section(
        buffer_source, "rail_balance_hybrid_plan_finish(")
    for forbidden in (
        ".data_ptr",
        "torch::empty",
        "jit::compiler",
        "prepare_rail_balance_",
    ):
        assert forbidden not in finish, forbidden

    # This prepare method still ends at ownership.  H4c publishes only through
    # its separate commit method after the caller completes Gate #2.
    assert "submit_prepared_rail_balance_hybrid_source_shuffle(" not in prepare
    assert "launch_prepared_rail_balance_hybrid_dispatch(" not in prepare
    assert "RailBalanceHybridPlanState::DispatchLive" not in prepare

    abort = _balanced_brace_section(
        buffer_source, "rail_balance_hybrid_plan_abort(")
    _assert_in_order(abort, (
        "cudaStreamSynchronize(comm_stream)",
        "rail_balance_hybrid_plan_pending.reset()",
        "CUDA_RUNTIME_CHECK(stream_status)",
    ))


def _assert_h4b_public_boundary() -> None:
    python_source = _PYTHON_BUFFER.read_text(encoding="utf-8")
    begin = python_source.index("    def dispatch(self,")
    end = python_source.index("    def combine(self,", begin)
    public_dispatch = python_source[begin:end]
    assert "_rail_balance_hybrid" not in public_dispatch
    assert elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is False
    assert _C._rail_balance_force_available() is False


def _assert_cpu_contract() -> None:
    _assert_owning_bundle_contract()
    _assert_h4b_public_boundary()

    # This fixed manifest is used only for the truthful D=1 rejection below.
    # N and local rank identities are deliberately absent from WORLD-common
    # data, matching H3b and the production gate contract.
    manifest = (
        _MANIFEST_MAGIC,
        _MANIFEST_VERSION,
        _MANIFEST_OPERATION_DISPATCH_PREPARE,
        1,
        _MANIFEST_WIDTH,
        8401,
        _WORLD_SIZE,
        1,
        _WORLD_SIZE,
    )
    assert len(manifest) == _MANIFEST_WIDTH
    assert all(type(value) is int and value >= 0 for value in manifest)


def _d1_manifest(invocation_id: int) -> tuple[int, ...]:
    values = (
        _MANIFEST_MAGIC,
        _MANIFEST_VERSION,
        _MANIFEST_OPERATION_DISPATCH_PREPARE,
        1,
        _MANIFEST_WIDTH,
        invocation_id,
        _WORLD_SIZE,
        1,
        _WORLD_SIZE,
    )
    assert len(values) == _MANIFEST_WIDTH
    return values


def _run_truthful_d1_failure(
    *,
    runtime: object,
    rank: int,
    ep_group: dist.ProcessGroup,
    control_group: dist.ProcessGroup,
    timeout: int,
    device_words: torch.Tensor,
    host_words: torch.Tensor,
    gate_calls: list[int],
    arena_offset: int,
    invocation_id: int,
) -> None:
    success, _, _ = _assert_oracle_contract()
    topk_idx = _local_topk(success, rank)
    num_tokens = int(topk_idx.shape[0])
    x = torch.arange(
        max(num_tokens, 1) * _HIDDEN,
        dtype=torch.int32,
        device=torch.device("cuda", rank),
    ).view(max(num_tokens, 1), _HIDDEN).to(torch.bfloat16)[:num_tokens]
    topk_weights = torch.ones(
        (num_tokens, success.num_topk),
        dtype=torch.float32,
        device=torch.device("cuda", rank),
    )
    elastic_module._encode_rail_balance_world_gate(
        host_words, 0, _d1_manifest(invocation_id))
    calls_before = gate_calls[0]

    result = None
    prepare_exception = None
    try:
        result = runtime._rail_balance_hybrid_dispatch_prepare(  # type: ignore[attr-defined]
            x,
            topk_idx,
            topk_weights,
            None,
            success.num_max_tokens_per_rank,
            success.num_experts,
            4,
            4,
            success.proxy_capacity_per_egress,
            arena_offset,
            invocation_id,
            success.remainder_seed,
        )
    except BaseException:
        prepare_exception = traceback.format_exc()

    gate_error = _make_error(
        _PREPARE_EXCEPTION_PRIORITY, rank) \
        if prepare_exception is not None else 0
    elastic_module._patch_rail_balance_world_gate_prevalidated(
        host_words, gate_error)
    gate = _gate(
        device_words=device_words,
        host_words=host_words,
        ep_group=ep_group,
    )
    abort_error = _abort(runtime, invocation_id)

    local_error = abort_error
    if local_error is None:
        try:
            assert result is None, result
            assert prepare_exception is not None
            assert gate == (
                _make_error(_PREPARE_EXCEPTION_PRIORITY, 0), -1, 0, 0)
            assert gate_calls[0] - calls_before == 1
        except BaseException:
            local_error = traceback.format_exc()
    _report_local_result(
        "truthful D1 dispatch-prepare fail-close",
        local_error,
        control_group,
        timeout,
    )


@torch.inference_mode()
def _worker(
    local_rank: int,
    num_processes: int,
    arguments: argparse.Namespace,
) -> None:
    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(
        local_rank, num_processes, seed=804)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=arguments.timeout),
    )
    assert rank == local_rank and world_size == _WORLD_SIZE

    device = torch.device("cuda", rank)
    device_words = torch.empty(
        _GATE_WORDS, dtype=torch.int64, device=device)
    host_words = torch.empty(
        _GATE_WORDS, dtype=torch.int64, device="cpu", pin_memory=True)
    elastic_module._validate_rail_balance_world_gate_storage(
        device_words, host_words)
    device_ptr = device_words.data_ptr()
    host_ptr = host_words.data_ptr()

    original_all_reduce = dist.all_reduce
    gate_calls = [0]

    def checked_all_reduce(
        tensor: torch.Tensor,
        op: dist.ReduceOp | None = None,
        group: dist.ProcessGroup | None = None,
        async_op: bool = False,
    ):
        gate_calls[0] += 1
        assert tensor.data_ptr() == device_ptr
        assert tensor.dtype == torch.int64
        assert tuple(tensor.shape) == (_GATE_WORDS,)
        assert op == dist.ReduceOp.MAX
        assert group is ep_group
        assert async_op is False
        return original_all_reduce(
            tensor, op=op, group=group, async_op=async_op)

    dist.all_reduce = checked_all_reduce
    buffer = None
    clean_shutdown = False
    try:
        success, _, _ = _assert_oracle_contract()
        arena_bytes = int(_C._get_rail_balance_hybrid_layout(
            _HIDDEN, _MAX_TOPK, _ARENA_PROXY_CAPACITY)[-1])
        alignment = int(_C.get_elastic_buffer_alignment())
        base_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
            ep_group,
            num_max_tokens_per_rank=_MAX_TOKENS,
            hidden=_HIDDEN,
            num_topk=_MAX_TOPK,
            use_fp8_dispatch=False,
            allow_hybrid_mode=True,
            allow_multiple_reduction=True,
        )
        arena_offset = align(base_bytes, alignment)
        assert arena_offset == base_bytes
        buffer = deep_ep.ElasticBuffer(
            ep_group,
            num_bytes=arena_offset + arena_bytes,
            num_max_tokens_per_rank=_MAX_TOKENS,
            hidden=_HIDDEN,
            num_topk=_MAX_TOPK,
            allow_hybrid_mode=True,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
            num_gpu_timeout_secs=arguments.timeout,
            num_cpu_timeout_secs=arguments.timeout,
        )
        runtime = buffer.runtime
        assert buffer.get_logical_domain_size() == (1, _WORLD_SIZE)

        _run_truthful_d1_failure(
            runtime=runtime,
            rank=rank,
            ep_group=ep_group,
            control_group=control_group,
            timeout=arguments.timeout,
            device_words=device_words,
            host_words=host_words,
            gate_calls=gate_calls,
            arena_offset=arena_offset,
            invocation_id=8401,
        )

        # The same C++ object must immediately accept the strongest successful
        # single-node transaction available here.  This catches leaked pending
        # ownership from the rejected truthful prepare without inventing a
        # fake Hybrid network topology.
        _run_success_or_gate2_failure(
            runtime=runtime,
            rank=rank,
            ep_group=ep_group,
            control_group=control_group,
            timeout=arguments.timeout,
            device_words=device_words,
            host_words=host_words,
            gate_calls=gate_calls,
            case=success,
            arena_offset=arena_offset,
            arena_bytes=arena_bytes,
            invocation_id=8402,
            label="H3b recovery after truthful D1 rejection",
            stale_abort_invocation_id=8400,
        )

        local_error = None
        try:
            assert gate_calls[0] == 3, gate_calls[0]
            assert device_words.data_ptr() == device_ptr
            assert host_words.data_ptr() == host_ptr
        except BaseException:
            local_error = traceback.format_exc()
        _report_local_result(
            "H4b fixed storage/final gate count",
            local_error,
            control_group,
            arguments.timeout,
        )
        clean_shutdown = True
    finally:
        dist.all_reduce = original_all_reduce
        if clean_shutdown and buffer is not None:
            buffer.destroy()
            buffer = None
            dist.monitored_barrier(
                group=control_group,
                timeout=timedelta(seconds=arguments.timeout),
                wait_all_ranks=True,
            )
            if rank == 0:
                print(
                    "PASS C080-H4b truthful D1 prepare fail-close and "
                    "exact H3b recovery: 3 fixed MAX gates",
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
        "--timeout", str(arguments.timeout),
        "--master-port", str(arguments.master_port),
        "--watchdog-seconds", str(arguments.watchdog_seconds),
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
            "C080-H4b subprocess watchdog expired after "
            f"{arguments.watchdog_seconds}s") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="C080-H4b owning dispatch-prepare contract")
    if not __debug__:
        parser.error("C080-H4b correctness checks require Python assertions")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--num-processes", type=int, default=_WORLD_SIZE)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--master-port", type=int, default=29963)
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    arguments = parser.parse_args()
    if arguments.num_processes != _WORLD_SIZE:
        parser.error("C080-H4b requires exactly 8 processes")
    if arguments.timeout <= 0 or arguments.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")

    _assert_cpu_contract()
    print(
        "PASS C080-H4b CPU/source contract: complete owning bundle, "
        "pre-Gate1 raw pointers/JIT, truthful topology derivation, "
        "3+2K metadata, arena_offset legacy bound, public force disabled",
        flush=True,
    )
    if arguments.cpu_only:
        return

    required = (
        "_rail_balance_hybrid_dispatch_prepare",
        "_rail_balance_hybrid_plan_prepare",
        "_rail_balance_hybrid_plan_finish",
        "_rail_balance_hybrid_plan_abort",
    )
    runtime_type = getattr(_C, "ElasticBuffer", None)
    missing = [
        name for name in required
        if runtime_type is None or not hasattr(runtime_type, name)
    ]
    if missing:
        parser.error("rebuild the C080 private API; missing " + ", ".join(missing))
    if torch.cuda.device_count() < _WORLD_SIZE:
        parser.error("C080-H4b requires at least eight visible CUDA devices")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(arguments.master_port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    if not arguments.worker_suite:
        _run_watchdog(arguments)
        return
    torch.multiprocessing.spawn(
        _worker,
        args=(arguments.num_processes, arguments),
        nprocs=arguments.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
