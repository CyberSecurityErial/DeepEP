"""C080-H5b source contract for the one-shot Hybrid combine lifecycle.

The local machine cannot truthfully execute a production D>1 Rail/Gin
transaction.  This gate freezes the private owning prepare/abort/commit
boundary without opening public ``rail_balance='force'`` or adding a fake
production transport.  It is source/lifetime evidence, not Gin runtime
evidence.
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
                return tuple(_compact(argument) for argument in arguments)
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(alias[argument_begin:index])
            argument_begin = index + 1
    raise AssertionError("unterminated std::tuple alias")


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
            return tuple(_compact(argument) for argument in arguments)
        elif character == "," and depth == 0:
            arguments.append(call[argument_begin:index])
            argument_begin = index + 1
    raise AssertionError("unterminated C++ call arguments")


def _initializer_expressions(source: str, declaration: str) -> tuple[str, ...]:
    begin = source.index(declaration) + len(declaration)
    opening = source.index("{", begin - 1)
    depth = 0
    argument_begin = opening + 1
    expressions: list[str] = []
    for index in range(opening + 1, len(source)):
        character = source[index]
        if character in "({[<":
            depth += 1
        elif character in ")}]>" and depth > 0:
            depth -= 1
        elif character == "}" and depth == 0:
            tail = source[argument_begin:index]
            if tail.strip():
                expressions.append(tail)
            return tuple(_compact(expression) for expression in expressions)
        elif character == "," and depth == 0:
            expressions.append(source[argument_begin:index])
            argument_begin = index + 1
    raise AssertionError(f"unterminated initializer: {declaration}")


def _assert_owner_abi(pending_source: str, buffer_source: str) -> None:
    begin = buffer_source.index("using RailBalanceHybridCombineResult")
    end = buffer_source.index(";", begin) + 1
    assert _split_template_arguments(buffer_source[begin:end]) == (
        "torch::Tensor",
        "std::optional<torch::Tensor>",
        "std::optional<EventHandle>",
    )

    raw = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridCombineRawPointers {")
    owner = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridCombineCompletion {")
    bundle = _balanced_brace_section(
        pending_source, "struct RailBalanceHybridDispatchBundle {")
    for field in (
        "void* x;",
        "float* topk_weights;",
        "void* combined_x;",
        "float* combined_topk_weights;",
    ):
        assert field in raw, field
    for field in (
        "torch::Tensor x;",
        "std::optional<torch::Tensor> topk_weights;",
        "torch::Tensor combined_x;",
        "std::optional<torch::Tensor> combined_topk_weights;",
        "RailBalanceHybridCombineRawPointers raw;",
    ):
        assert field in owner, field
    assert re.search(
        r"std::optional<RailBalanceHybridCombineCompletion>\s+"
        r"combine_completion;",
        bundle,
    )


def _assert_prepare_contract(buffer_source: str) -> None:
    prepare = _strip_comments(_balanced_brace_section(
        buffer_source, "rail_balance_hybrid_combine_prepare("))
    compact_prepare = _compact(prepare)
    signature = prepare[:prepare.index("{")]
    assert "const torch::Tensor& x" in signature
    assert "const std::optional<torch::Tensor>& topk_weights" in signature
    assert "const int& invocation_id" in signature
    for forbidden_parameter in (
        "src_metadata", "combined_topk_idx",
        "psum_num_recv_tokens_per_scaleup_rank",
        "token_metadata_at_forward", "channel_linked_list",
        "num_reduced_tokens",
    ):
        assert forbidden_parameter not in signature, forbidden_parameter

    # Prepare is the retryable side of the future WORLD combine gate.  It
    # validates the H5a handle, all four prebuilt stages, every raw dependency,
    # force-v1's mandatory FP32 weights, and exact input/output shapes before
    # installing one owner.  It does not publish main combine or poison the
    # live dispatch.
    assert (
        "pending.state==RailBalanceHybridPlanState::DispatchLive"
        in compact_prepare
    )
    assert "bundle.dispatch_completion.has_value()" in compact_prepare
    assert "notbundle.combine_completion.has_value()" in compact_prepare
    for dependency in (
        "bundle.main_combine != nullptr",
        "pending.return_unshuffle != nullptr",
        "pending.local_barrier != nullptr",
        "pending.combine_epilogue != nullptr",
        "bundle.raw.buffer != nullptr",
        "bundle.raw.workspace != nullptr",
        "bundle.raw.arena != nullptr",
        "bundle.raw.proxy_return_base != nullptr",
        "bundle.raw.legacy_reduce_buffer_base != nullptr",
        "bundle.raw.psum_num_recv_tokens_per_scaleup_rank != nullptr",
        "bundle.raw.token_metadata_at_forward != nullptr",
        "bundle.raw.channel_linked_list != nullptr",
        "bundle.raw.moved != nullptr",
        "bundle.raw.group_prefix != nullptr",
        "bundle.raw.proxy_required != nullptr",
        "bundle.raw.status != nullptr",
    ):
        assert _compact(dependency) in compact_prepare, dependency
    assert (
        "dispatch_completion.num_recv_tokens==0or"
        "dispatch_completion.raw.recv_src_metadata!=nullptr"
        in compact_prepare
    )
    assert (
        "bundle.num_tokens==0orbundle.raw.copied_topk_idx!=nullptr"
        in compact_prepare
    )
    assert "x.size(0)==dispatch_completion.num_recv_tokens" in compact_prepare
    assert "x.size(1)==bundle.hidden" in compact_prepare
    assert "x.scalar_type()==torch::kBFloat16" in compact_prepare
    assert "EP_HOST_ASSERT(topk_weights.has_value());" in compact_prepare
    assert "topk_weights->scalar_type()==torch::kFloat" in compact_prepare
    assert (
        "topk_weights->size(0)==dispatch_completion.num_recv_tokens"
        in compact_prepare
    )
    assert "topk_weights->size(1)==bundle.num_topk" in compact_prepare

    assert "{bundle.num_tokens,bundle.hidden}" in compact_prepare
    assert "{bundle.num_tokens,bundle.num_topk}" in compact_prepare
    assert _initializer_expressions(
        prepare, "RailBalanceHybridCombineCompletion completion =") == (
            ".x=x",
            ".topk_weights=topk_weights",
            ".combined_x=std::move(combined_x)",
            ".combined_topk_weights=combined_topk_weights",
            ".raw={}",
        )
    assert _initializer_expressions(prepare, "completion.raw =") == (
        ".x=completion.x.data_ptr()",
        ".topk_weights=completion.topk_weights.has_value()?"
        "completion.topk_weights->data_ptr<float>():nullptr",
        ".combined_x=completion.combined_x.data_ptr()",
        ".combined_topk_weights="
        "completion.combined_topk_weights.has_value()?"
        "completion.combined_topk_weights->data_ptr<float>():nullptr",
    )
    assert "completion.raw.topk_weights!=nullptr" in compact_prepare
    assert "completion.raw.combined_topk_weights!=nullptr" in compact_prepare

    install_begin, install_end = _call_span(
        prepare, "bundle.combine_completion.emplace")
    last_raw = prepare.rindex(".data_ptr")
    last_assert = prepare.rindex("EP_HOST_ASSERT")
    assert last_raw < last_assert < install_begin
    assert prepare[install_end:].strip() == "}"
    assert prepare.count("bundle.combine_completion.emplace(") == 1
    assert not re.search(r"pending\.state\s*=(?!=)", prepare)
    for forbidden in (
        "RailBalanceHybridPlanState::Invalid",
        "launch_prepared_rail_balance_hybrid_combine(",
        "submit_prepared_rail_balance_hybrid_return_unshuffle(",
        "submit_prepared_rail_balance_hybrid_local_barrier(",
        "submit_prepared_rail_balance_hybrid_combine_epilogue(",
        "cudaMemcpy", "cudaStreamSynchronize", "cudaDeviceSynchronize",
        "jit::compiler", "prepare_rail_balance_", "Runtime::",
    ):
        assert forbidden not in prepare, forbidden


def _assert_abort_contract(buffer_source: str) -> None:
    abort = _strip_comments(_balanced_brace_section(
        buffer_source, "rail_balance_hybrid_combine_abort("))
    compact = _compact(abort)
    # Combine-gate rejection is retryable and idempotent.  It releases only
    # the just-prepared combine owner; the synchronized H5a dispatch handle and
    # DispatchLive state remain intact.
    for condition in (
        "destroyedornotrail_balance_hybrid_plan_pending.has_value()",
        "pending.invocation_id!=invocation_id",
        "pending.state!=RailBalanceHybridPlanState::DispatchLive",
        "pending.dispatch_bundle==nullptr",
        "notbundle.combine_completion.has_value()",
    ):
        assert condition in compact, condition
    assert "bundle.combine_completion.reset();" in abort
    assert abort.count(".reset()") == 1
    assert not re.search(r"pending\.state\s*=(?!=)", abort)
    for forbidden in (
        "rail_balance_hybrid_plan_pending.reset",
        "dispatch_completion.reset",
        "RailBalanceHybridPlanState::Invalid", "EP_HOST_ASSERT",
        "CUDAGuard", ".data_ptr", "torch::empty", "jit::compiler",
        "launch_prepared_", "submit_prepared_", "cudaMemcpy",
        "cudaStreamSynchronize", "cudaDeviceSynchronize", "stream_wait",
    ):
        assert forbidden not in abort, forbidden


def _assert_commit_contract(buffer_source: str) -> None:
    commit = _strip_comments(_balanced_brace_section(
        buffer_source, "rail_balance_hybrid_combine_commit("))
    signature = commit[:commit.index("{")]
    assert "const int& invocation_id" in signature
    assert "torch::Tensor" not in signature

    invalid = "pending.state = RailBalanceHybridPlanState::Invalid;"
    invalid_begin = commit.index(invalid)
    invalid_end = invalid_begin + len(invalid)
    main_begin, main_end = _call_span(
        commit, "launch_prepared_rail_balance_hybrid_combine")
    return_begin, return_end = _call_span(
        commit, "submit_prepared_rail_balance_hybrid_return_unshuffle")
    barrier_begin, barrier_end = _call_span(
        commit, "submit_prepared_rail_balance_hybrid_local_barrier")
    epilogue_begin, epilogue_end = _call_span(
        commit, "submit_prepared_rail_balance_hybrid_combine_epilogue")
    assert commit[invalid_end:main_begin].strip() == ""
    assert commit[main_end:return_begin].strip() == ""
    assert commit[return_end:barrier_begin].strip() == ""
    assert commit[barrier_end:epilogue_begin].strip() == ""

    critical = commit[invalid_begin:epilogue_end]
    assert critical.count(";") == 5
    assert critical.count("comm_stream") == 4
    for call in (
        "launch_prepared_rail_balance_hybrid_combine(",
        "submit_prepared_rail_balance_hybrid_return_unshuffle(",
        "submit_prepared_rail_balance_hybrid_local_barrier(",
        "submit_prepared_rail_balance_hybrid_combine_epilogue(",
    ):
        assert critical.count(call) == 1, call
    for forbidden in (
        "EP_HOST_ASSERT", "EP_UNIFIED_ASSERT", ".data_ptr", "torch::",
        "jit::", "prepare_rail_balance_", "get_sym_ptr", "cudaMemcpy",
        "cudaStreamSynchronize", "cudaDeviceSynchronize", "stream_wait",
        "EventHandle", "layout::", "std::make_shared", "CUDAGuard",
        ".emplace", ".reset",
    ):
        assert forbidden not in critical, forbidden
    assert not re.search(
        r"\b(if|for|while|try|catch|return|throw)\b", critical)

    # Freeze every ABI expression, not merely field membership: most of these
    # pointers share a type and a swap can compile while silently corrupting
    # route replay or the legacy reduce layout.
    assert _call_arguments(commit[main_begin:main_end]) == (
        "prepared_combine",
        "combine_completion_raw.x",
        "combine_completion_raw.topk_weights",
        "dispatch_completion_raw.recv_src_metadata",
        "raw.psum_num_recv_tokens_per_scaleup_rank",
        "raw.token_metadata_at_forward",
        "raw.channel_linked_list",
        "nccl_dev_comm",
        "nccl_window",
        "raw.buffer",
        "raw.workspace",
        "raw.proxy_return_base",
        "scaleout_rank_idx",
        "scaleup_rank_idx",
        "num_reduced_tokens",
        "comm_stream",
    )
    assert _call_arguments(commit[return_begin:return_end]) == (
        "prepared_return_unshuffle",
        "nccl_dev_comm",
        "nccl_window",
        "raw.arena",
        "raw.legacy_reduce_buffer_base",
        "raw.moved",
        "raw.group_prefix",
        "raw.proxy_required",
        "raw.status",
        "num_experts",
        "num_destinations",
        "num_rails",
        "scaleup_rank_idx",
        "rank_idx",
        "num_max_tokens_per_rank",
        "proxy_capacity_per_egress",
        "comm_stream",
    )
    assert _call_arguments(commit[barrier_begin:barrier_end]) == (
        "prepared_local_barrier",
        "local_barrier_launch_args",
        "nccl_dev_comm",
        "nccl_window",
        "raw.workspace",
        "num_rails",
        "scaleup_rank_idx",
        "timeout_cycles",
        "comm_stream",
    )
    assert _call_arguments(commit[epilogue_begin:epilogue_end]) == (
        "prepared_combine_epilogue",
        "combine_completion_raw.combined_x",
        "combine_completion_raw.combined_topk_weights",
        "raw.copied_topk_idx",
        "raw.legacy_reduce_buffer_base",
        "num_combined_tokens",
        "scaleout_rank_idx",
        "scaleup_rank_idx",
        "comm_stream",
    )

    # The only synchronization observes the whole committed chain.  Copy the
    # native three-item result before resetting the owning one-shot pending
    # transaction; any earlier failure leaves Invalid plus every owner alive.
    assert commit.count("cudaMemcpyAsync(") == 1
    assert commit.count("cudaStreamSynchronize(comm_stream)") == 1
    assert "cudaMemcpyDeviceToHost" in commit
    status_copy = commit.index("cudaMemcpyAsync(", epilogue_end)
    stream_sync = commit.index("cudaStreamSynchronize(comm_stream)", status_copy)
    status_store = commit.index("pending.plan_status = host_status;", stream_sync)
    status_check = commit.index("EP_HOST_ASSERT(host_status == 0);", status_store)
    result = commit.index("RailBalanceHybridCombineResult result", status_check)
    reset = commit.index("rail_balance_hybrid_plan_pending.reset();", result)
    result_return = commit.index("return result;", reset)
    assert epilogue_end < status_copy < stream_sync < status_store
    assert status_store < status_check < result < reset < result_return
    assert commit.count("rail_balance_hybrid_plan_pending.reset();") == 1
    assert commit[result_return + len("return result;"):].strip() == "}"
    assert _initializer_expressions(
        commit, "RailBalanceHybridCombineResult result =") == (
            "combine_completion.combined_x",
            "combine_completion.combined_topk_weights",
            "std::nullopt",
        )

    pre_poison = commit[:invalid_begin]
    compact_pre_poison = _compact(pre_poison)
    assert "bundle.dispatch_completion.value()" in pre_poison
    assert "bundle.combine_completion.value()" in pre_poison
    assert (
        "constintnum_reduced_tokens="
        "dispatch_completion.num_recv_tokens;"
        in compact_pre_poison
    )
    assert "constintnum_combined_tokens=bundle.num_tokens;" in compact_pre_poison
    assert "pending.state = RailBalanceHybridPlanState::DispatchLive;" not in commit
    for forbidden in (
        ".data_ptr", "torch::empty", "torch::zeros", "torch::full",
        "jit::compiler", "prepare_rail_balance_", "__global__", "_impl<",
        "Runtime::",
    ):
        assert forbidden not in commit, forbidden


def _assert_private_boundary(buffer_source: str) -> None:
    compact = _compact(buffer_source)
    for name in ("prepare", "abort", "commit"):
        assert (
            f'"_rail_balance_hybrid_combine_{name}",'
            f'&ElasticBuffer::rail_balance_hybrid_combine_{name}'
            in compact
        )

    python_source = _PYTHON_BUFFER.read_text(encoding="utf-8")
    dispatch_begin = python_source.index("    def dispatch(self,")
    dispatch_end = python_source.index(
        "    @staticmethod\n    def _unpack_bias", dispatch_begin)
    combine_begin = python_source.index("    def combine(self,", dispatch_end)
    public_source = python_source[dispatch_begin:dispatch_end] + \
        python_source[combine_begin:]
    for name in ("prepare", "abort", "commit"):
        assert f"_rail_balance_hybrid_combine_{name}" not in public_source
    assert re.search(
        r"_RAIL_BALANCE_FORCE_HOST_AVAILABLE\s*=\s*False", python_source)


def main() -> None:
    pending_source = _PENDING_HEADER.read_text(encoding="utf-8")
    buffer_source = _BUFFER_HEADER.read_text(encoding="utf-8")
    _assert_owner_abi(pending_source, buffer_source)
    _assert_prepare_contract(buffer_source)
    _assert_abort_contract(buffer_source)
    _assert_commit_contract(buffer_source)
    _assert_private_boundary(buffer_source)
    print(
        "PASS C080-H5b source contract: retryable owning prepare/abort, "
        "four-stage one-shot combine commit, native result and cleanup",
        flush=True,
    )


if __name__ == "__main__":
    main()
