"""C080-H4c source gate for the production Hybrid dispatch commit.

The local machine has the truthful topology D=1, G=8, so it cannot execute a
successful production Rail/Gin dispatch.  This test therefore freezes the
post-Gate2 host contract without inventing a virtual production topology.  It
does not claim network-runtime evidence.
"""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PENDING_HEADER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp")
_BUFFER_HEADER = _ROOT / "csrc/elastic/buffer.hpp"


def _balanced_brace_section(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated C++ section: {anchor}")


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", source)


def _call_span(source: str, name: str) -> tuple[int, int]:
    start = source.index(name)
    opening = source.index("(", start + len(name))
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                assert source[index + 1] == ";"
                return start, index + 2
    raise AssertionError(f"unterminated call: {name}")


def _assert_prefix_pointer_boundary(
    pending_source: str, prepare: str,
) -> None:
    raw = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchRawPointers")
    assert "int* psum_num_recv_tokens_per_expert_storage;" in raw
    assert "int* psum_num_recv_tokens_per_expert_inclusive;" in raw
    assert "int* psum_num_recv_tokens_per_expert;" not in raw

    # The main persistent dispatch writes an exclusive prefix at storage[0].
    # The later non-expanded epilogue consumes the inclusive view storage[1:].
    # Freeze both addresses before Gate #1; H4c must not recover either from a
    # Tensor after it poisons the transaction state.
    assert (
        "auto psum_num_recv_tokens_per_expert =\n"
        "            psum_num_recv_tokens_per_expert_storage.slice(\n"
        "                0, 1, num_local_experts + 1);"
        in prepare
    )
    assert (
        ".psum_num_recv_tokens_per_expert_storage =\n"
        "                psum_num_recv_tokens_per_expert_storage.data_ptr<int>()"
        in prepare
    )
    inclusive_from_storage = (
        ".psum_num_recv_tokens_per_expert_inclusive =\n"
        "                psum_num_recv_tokens_per_expert_storage.data_ptr<int>() + 1"
        in prepare
    )
    inclusive_from_view = (
        ".psum_num_recv_tokens_per_expert_inclusive =\n"
        "                psum_num_recv_tokens_per_expert.data_ptr<int>()"
        in prepare
    )
    assert inclusive_from_storage or inclusive_from_view


def _assert_commit_contract() -> None:
    pending_source = _PENDING_HEADER.read_text(encoding="utf-8")
    buffer_source = _BUFFER_HEADER.read_text(encoding="utf-8")
    prepare = _balanced_brace_section(
        buffer_source, "rail_balance_hybrid_dispatch_prepare(")
    commit = _balanced_brace_section(
        buffer_source, "rail_balance_hybrid_dispatch_commit(")
    code = _strip_comments(commit)

    _assert_prefix_pointer_boundary(pending_source, prepare)

    # Every validation and raw-ABI capture must finish before Invalid.  From
    # Invalid onward, either enqueue may throw and the transaction must remain
    # permanently poisoned.  Source publication, its cross-rank visibility
    # barrier, and dispatch are adjacent; successful publication consists only
    # of shuffled=true followed by DispatchLive.
    invalid = "pending.state = RailBalanceHybridPlanState::Invalid;"
    shuffled = "pending.shuffled = true;"
    live = "pending.state = RailBalanceHybridPlanState::DispatchLive;"
    invalid_begin = code.index(invalid)
    invalid_end = invalid_begin + len(invalid)
    device_guard = (
        "const c10::cuda::CUDAGuard device_guard(device_index);")
    guard_begin = code.index(device_guard)
    assert code.count(device_guard) == 1
    assert guard_begin < invalid_begin
    source_begin, source_end = _call_span(
        code, "submit_prepared_rail_balance_hybrid_source_shuffle")
    barrier_begin, barrier_end = _call_span(
        code, "submit_prepared_rail_balance_hybrid_local_barrier")
    dispatch_begin, dispatch_end = _call_span(
        code, "launch_prepared_rail_balance_hybrid_dispatch")
    shuffled_begin = code.index(shuffled)
    shuffled_end = shuffled_begin + len(shuffled)
    live_begin = code.index(live)
    live_end = live_begin + len(live)

    assert code[invalid_end:source_begin].strip() == ""
    assert code[source_end:barrier_begin].strip() == ""
    assert code[barrier_end:dispatch_begin].strip() == ""
    assert code[dispatch_end:shuffled_begin].strip() == ""
    assert code[shuffled_end:live_begin].strip() == ""
    assert code[live_end:].strip() == "}"
    critical = code[invalid_begin:live_end]
    assert critical.count(";") == 6
    assert critical.count(
        "submit_prepared_rail_balance_hybrid_source_shuffle(") == 1
    assert critical.count(
        "submit_prepared_rail_balance_hybrid_local_barrier(") == 1
    assert critical.count(
        "launch_prepared_rail_balance_hybrid_dispatch(") == 1

    for forbidden in (
        "EP_HOST_ASSERT", "EP_UNIFIED_ASSERT", ".data_ptr", "torch::",
        "jit::compiler", "prepare_rail_balance_", "get_sym_ptr",
        "cudaMemcpy", "cudaStreamSynchronize", "cudaDeviceSynchronize",
        "stream_wait", "EventHandle", "layout::", "std::make_shared",
        "CUDAGuard",
    ):
        assert forbidden not in critical, forbidden
    assert not re.search(
        r"\b(if|for|while|try|catch|return|throw)\b", critical)

    source_call = code[source_begin:source_end]
    for field in (
        "raw.x", "raw.topk_idx", "raw.topk_weights", "raw.arena",
        "raw.owner_channel_prefix", "raw.keep_count", "raw.segments",
        "raw.num_segments", "raw.retained", "raw.moved_channel_prefix",
        "raw.group_prefix", "raw.proxy_required", "raw.status",
    ):
        assert field in source_call, field
    barrier_call = code[barrier_begin:barrier_end]
    for field in (
        "prepared_local_barrier", "local_barrier_launch_args",
        "nccl_dev_comm", "nccl_window", "raw.workspace", "num_rails",
        "scaleup_rank_idx", "timeout_cycles", "comm_stream",
    ):
        assert field in barrier_call, field
    dispatch_call = code[dispatch_begin:dispatch_end]
    for field in (
        "raw.x", "raw.topk_idx", "raw.topk_weights",
        "raw.copied_topk_idx", "raw.cumulative_local_expert_recv_stats",
        "raw.psum_num_recv_tokens_per_scaleup_rank",
        "raw.psum_num_recv_tokens_per_expert_storage",
        "raw.num_unaligned_recv_tokens_per_expert",
        "raw.dst_buffer_slot_idx", "raw.token_metadata_at_forward",
        "raw.buffer", "raw.workspace", "raw.mapped_host_workspace",
        "raw.arena", "raw.retained", "raw.moved", "raw.group_prefix",
        "raw.proxy_required",
    ):
        assert field in dispatch_call, field
    assert "raw.psum_num_recv_tokens_per_expert_inclusive" not in dispatch_call
    assert critical.count("comm_stream") == 2

    # H4c is private until the full dispatch epilogue/handle transaction is
    # complete.  It must not leak into the public Python dispatch yet.
    assert (
        '"_rail_balance_hybrid_dispatch_commit",\n'
        "            &ElasticBuffer::rail_balance_hybrid_dispatch_commit"
        in buffer_source
    )


def main() -> None:
    _assert_commit_contract()
    print(
        "PASS C080-H4c source contract: Invalid -> source -> dispatch -> "
        "DispatchLive, frozen raw ABI and expert-prefix base boundary",
        flush=True,
    )


if __name__ == "__main__":
    main()
