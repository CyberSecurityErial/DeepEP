"""C080-H5a source contract for private Hybrid dispatch completion.

The production Hybrid success topology cannot be launched truthfully on this
single-node machine.  This gate therefore freezes the owning host-completion
boundary without adding a fake D>1 production path.  It is not runtime Gin
evidence and does not enable public ``rail_balance='force'``.
"""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PENDING_HEADER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp")
_BUFFER_HEADER = _ROOT / "csrc/elastic/buffer.hpp"
_PYTHON_BUFFER = _ROOT / "deep_ep/buffers/elastic.py"


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


def _compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def _split_template_arguments(alias: str) -> tuple[str, ...]:
    tuple_begin = alias.index("std::tuple")
    begin = alias.index("<", tuple_begin)
    depth = 0
    argument_begin = begin + 1
    arguments: list[str] = []
    for index in range(begin + 1, len(alias)):
        character = alias[index]
        if character == "<":
            depth += 1
        elif character == ">":
            if depth == 0:
                arguments.append(alias[argument_begin:index])
                return tuple(
                    re.sub(r"\s+", "", argument) for argument in arguments)
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(alias[argument_begin:index])
            argument_begin = index + 1
    raise AssertionError("unterminated std::tuple alias")


def _call_section(source: str, name: str) -> str:
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
                return source[start:index + 2]
    raise AssertionError(f"unterminated call: {name}")


def _call_arguments(call: str) -> tuple[str, ...]:
    begin = call.index("(")
    depth = 0
    argument_begin = begin + 1
    arguments: list[str] = []
    for index in range(begin + 1, len(call)):
        character = call[index]
        if character in "(<[{":
            depth += 1
        elif character in ")>]}" and depth > 0:
            depth -= 1
        elif character == ")" and depth == 0:
            arguments.append(call[argument_begin:index])
            return tuple(
                re.sub(r"\s+", "", argument) for argument in arguments)
        elif character == "," and depth == 0:
            arguments.append(call[argument_begin:index])
            argument_begin = index + 1
    raise AssertionError("unterminated C++ call arguments")


def _result_initializer_expressions(source: str) -> tuple[str, ...]:
    declaration = "RailBalanceHybridDispatchResult result = {"
    begin = source.index(declaration) + len(declaration)
    end = source.index("};", begin)
    expressions = [
        re.sub(r"\s+", "", expression)
        for expression in source[begin:end].split(",")
    ]
    assert expressions[-1] == ""
    return tuple(expressions[:-1])


def _assert_result_abi(pending_source: str, buffer_source: str) -> None:
    begin = buffer_source.index("using RailBalanceHybridDispatchResult")
    end = buffer_source.index(";", begin) + 1
    alias = buffer_source[begin:end]
    assert _split_template_arguments(alias) == (
        "torch::Tensor",
        "std::optional<torch::Tensor>",
        "std::optional<torch::Tensor>",
        "std::optional<torch::Tensor>",
        "std::optional<torch::Tensor>",
        "int",
        "int",
        "std::vector<int>",
        "torch::Tensor",
        "torch::Tensor",
        "torch::Tensor",
        "torch::Tensor",
        "torch::Tensor",
        "std::optional<torch::Tensor>",
        "std::optional<torch::Tensor>",
        "std::optional<EventHandle>",
    )

    completion = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchCompletion {")
    raw = _balanced_brace_section(
        pending_source,
        "struct RailBalanceHybridDispatchCompletionRawPointers {")
    for field in (
        "torch::Tensor recv_x;",
        "torch::Tensor recv_topk_idx;",
        "torch::Tensor recv_topk_weights;",
        "torch::Tensor recv_src_metadata;",
        "std::vector<int> num_recv_tokens_per_expert_list;",
        "int num_recv_tokens;",
        "int num_expanded_tokens;",
        "RailBalanceHybridDispatchCompletionRawPointers raw;",
    ):
        assert field in completion, field
    for field in (
        "void* recv_x;",
        "topk_idx_t* recv_topk_idx;",
        "float* recv_topk_weights;",
        "int* recv_src_metadata;",
    ):
        assert field in raw, field

    bundle = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchBundle")
    assert re.search(
        r"std::optional<RailBalanceHybridDispatchCompletion>\s+"
        r"dispatch_completion;",
        bundle,
    )


def _assert_finish_contract() -> None:
    pending_source = _PENDING_HEADER.read_text(encoding="utf-8")
    buffer_source = _BUFFER_HEADER.read_text(encoding="utf-8")
    finish = _strip_comments(_balanced_brace_section(
        buffer_source, "rail_balance_hybrid_dispatch_finish("))
    _assert_result_abi(pending_source, buffer_source)

    assert (
        "RailBalanceHybridDispatchResult "
        "rail_balance_hybrid_dispatch_finish(" in buffer_source
    )
    assert finish.count(
        "pending.state == RailBalanceHybridPlanState::DispatchLive") == 1
    assert "not bundle.dispatch_completion.has_value()" in finish
    guard = finish.index(
        "const c10::cuda::CUDAGuard device_guard(device_index);")
    invalid = finish.index(
        "pending.state = RailBalanceHybridPlanState::Invalid;")
    assert invalid < guard

    # Completion reuses the mapped host pointers frozen before Gate #1.  It
    # must not reconstruct WorkspaceLayout or recover either pointer from a
    # Tensor after publication.
    assert "host_scaleup_rank_count" in finish
    assert "host_expert_count" in finish
    for forbidden in (
        "get_scaleup_rank_count_ptr", "get_scaleup_expert_count_ptr",
        "layout::WorkspaceLayout(", "mapped_host_workspace",
    ):
        assert forbidden not in finish, forbidden

    owner = finish.index("dispatch_completion.emplace(")
    epilogue_begin = finish.index(
        "launch_prepared_rail_balance_hybrid_dispatch_epilogue(")
    status_copy = finish.index("cudaMemcpyAsync(", epilogue_begin)
    stream_sync = finish.index(
        "cudaStreamSynchronize(comm_stream)", status_copy)
    result = finish.index("RailBalanceHybridDispatchResult result", stream_sync)
    live = finish.index(
        "pending.state = RailBalanceHybridPlanState::DispatchLive;", result)
    result_return = finish.index("return result;", live)
    assert invalid < owner < epilogue_begin < status_copy < stream_sync
    assert stream_sync < result < live < result_return
    assert finish.count("cudaMemcpyAsync(") == 1
    assert "cudaMemcpyDeviceToHost" in finish[status_copy:stream_sync]

    # Mapped counters are int64 publications. Decode and accumulate without
    # narrowing, prove each addition fits the topology-derived ceiling before
    # performing it, and only then cast the complete totals to the native int
    # dispatch ABI. This prevents a corrupt ready value from wrapping through
    # allocation sizes while preserving DeepEP's native result type.
    raw_pointers = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchRawPointers {")
    assert "int64_t* host_scaleup_rank_count;" in raw_pointers
    assert "int64_t* host_expert_count;" in raw_pointers
    compact_finish = _compact(finish)
    assert (
        "constint64_tmax_num_recv_tokens="
        "static_cast<int64_t>(bundle.num_destinations)*"
        "bundle.num_rails*bundle.num_max_tokens_per_rank;"
    ) in compact_finish
    assert (
        "constint64_tmax_num_expanded_tokens=max_num_recv_tokens*"
        "std::min(num_topk,num_local_experts);"
    ) in compact_finish
    assert "int64_tnum_recv_tokens_i64=0;" in compact_finish
    assert "int64_tnum_expanded_tokens_i64=0;" in compact_finish

    rank_counter = _compact(_balanced_brace_section(
        finish, "while (counter_scaleup_rank_idx"))
    rank_decode = rank_counter.index(
        "constint64_tcount=math::encode_decode_positive(encoded_count);")
    rank_bound = rank_counter.index(
        "count<=max_num_recv_tokens-num_recv_tokens_i64")
    rank_add = rank_counter.index("num_recv_tokens_i64+=count;")
    assert "constint64_tencoded_count=" in rank_counter
    assert "encoded_count!=INT64_MIN" in rank_counter
    assert rank_decode < rank_bound < rank_add

    expert_counter = _compact(_balanced_brace_section(
        finish, "while (counter_local_expert_idx"))
    expert_decode = expert_counter.index(
        "constint64_tcount=math::encode_decode_positive(encoded_count);")
    expert_bound = expert_counter.index(
        "count<=max_num_expanded_tokens-num_expanded_tokens_i64")
    expert_list_cast = expert_counter.index(
        "num_recv_tokens_per_expert_list.push_back(static_cast<int>(count));")
    expert_add = expert_counter.index("num_expanded_tokens_i64+=count;")
    assert "constint64_tencoded_count=" in expert_counter
    assert "encoded_count!=INT64_MIN" in expert_counter
    assert "count>=0andcount<=INT_MAXand" in expert_counter
    assert expert_decode < expert_bound < expert_list_cast < expert_add

    recv_total_bound = finish.index(
        "EP_HOST_ASSERT(num_recv_tokens_i64 <= INT_MAX);")
    expanded_total_bound = finish.index(
        "EP_HOST_ASSERT(num_expanded_tokens_i64 <= INT_MAX);")
    recv_total_cast = finish.index(
        "const int num_recv_tokens =\n"
        "            static_cast<int>(num_recv_tokens_i64);")
    expanded_total_cast = finish.index(
        "const int num_expanded_tokens =\n"
        "            static_cast<int>(num_expanded_tokens_i64);")
    first_allocation = finish.index("auto recv_x = torch::empty(")
    assert (
        recv_total_bound < expanded_total_bound < recv_total_cast <
        expanded_total_cast < first_allocation
    )

    # Types alone cannot distinguish the five adjacent handle Tensors. Freeze
    # every native dispatch-result expression so recv metadata, slot indices,
    # force forward metadata, and linked-list replay cannot be interchanged by
    # a source edit that still compiles.
    assert _result_initializer_expressions(finish) == (
        "completion.recv_x",
        "std::nullopt",
        "completion.recv_topk_idx",
        "completion.recv_topk_weights",
        "bundle.copied_topk_idx",
        "completion.num_recv_tokens",
        "completion.num_expanded_tokens",
        "completion.num_recv_tokens_per_expert_list",
        "bundle.psum_num_recv_tokens_per_scaleup_rank",
        "bundle.psum_num_recv_tokens_per_expert",
        "bundle.num_unaligned_recv_tokens_per_expert",
        "completion.recv_src_metadata",
        "bundle.dst_buffer_slot_idx",
        "bundle.token_metadata_at_forward",
        "bundle.channel_linked_list",
        "std::nullopt",
    )

    # All exact-size outputs are owned before the epilogue can touch them.
    allocation_region = finish[invalid:owner]
    assert "const int hidden = bundle.hidden;" in finish[:invalid]
    assert "const int num_topk = bundle.num_topk;" in finish[:invalid]
    for shape in (
        "{num_recv_tokens, hidden}",
        "{num_recv_tokens, num_topk}",
        "{num_recv_tokens, num_topk + 2}",
    ):
        assert shape in allocation_region, shape
    epilogue = _call_section(
        finish, "launch_prepared_rail_balance_hybrid_dispatch_epilogue")
    assert _call_arguments(epilogue) == (
        "prepared_epilogue",
        "raw.buffer",
        "raw.workspace",
        "raw.psum_num_recv_tokens_per_scaleup_rank",
        "raw.psum_num_recv_tokens_per_expert_inclusive",
        "completion.raw.recv_x",
        "completion.raw.recv_topk_idx",
        "completion.raw.recv_topk_weights",
        "completion.raw.recv_src_metadata",
        "raw.channel_linked_list",
        "num_recv_tokens",
        "scaleout_rank_idx",
        "scaleup_rank_idx",
        "comm_stream",
    )

    # H5a finishes dispatch only. Combine and public force remain closed until
    # the later one-shot consume transaction exists.
    for forbidden in (
        "main_combine", "return_unshuffle", "combine_epilogue",
        "launch_prepared_rail_balance_hybrid_combine(",
        "submit_prepared_rail_balance_hybrid_return_unshuffle(",
        "submit_prepared_rail_balance_hybrid_combine_epilogue(",
    ):
        assert forbidden not in finish, forbidden
    assert (
        '"_rail_balance_hybrid_dispatch_finish",\n'
        "            &ElasticBuffer::rail_balance_hybrid_dispatch_finish"
        in buffer_source
    )

    python_source = _PYTHON_BUFFER.read_text(encoding="utf-8")
    public_begin = python_source.index("    def dispatch(self,")
    public_end = python_source.index("    def combine(self,", public_begin)
    public_dispatch = python_source[public_begin:public_end]
    assert "_rail_balance_hybrid_dispatch_finish" not in public_dispatch
    assert "_rail_balance_hybrid_dispatch_prepare" not in public_dispatch
    assert re.search(
        r"_RAIL_BALANCE_FORCE_HOST_AVAILABLE\s*=\s*False", python_source)


def main() -> None:
    _assert_finish_contract()
    print(
        "PASS C080-H5a source contract: owning exact dispatch completion, "
        "inclusive epilogue, post-epilogue status sync, native 16-item ABI",
        flush=True,
    )


if __name__ == "__main__":
    main()
