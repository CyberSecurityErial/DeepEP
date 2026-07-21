"""Source gate for the C080 prepared return-unshuffle submit path."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_unshuffle.hpp"
_PENDING = _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp"
_CALLSITE = _ROOT / "csrc/elastic/buffer.hpp"


def _section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def main() -> None:
    source = _ADAPTER.read_text()
    pending_source = _PENDING.read_text()
    callsite_source = _CALLSITE.read_text()

    prepare = _section(
        source,
        "prepare_rail_balance_hybrid_return_unshuffle(",
        "// legacy_reduce_buffer is the base",
    )
    raw_submit = _section(
        source,
        "static void submit_prepared_rail_balance_hybrid_return_unshuffle(",
        "// Checked wrapper retained",
    )
    checked = _section(
        source,
        "static void launch_prepared_rail_balance_hybrid_return_unshuffle(",
        "}  // namespace deep_ep::elastic",
    )

    # H/K, channel geometry, shared-memory bytes, and launch shape are frozen
    # together with the JIT runtime before the committed transaction.
    for token in (
        "struct PreparedRailBalanceHybridReturnUnshuffle",
        ".hidden = hidden",
        ".num_topk = num_topk",
        ".num_channels = num_channels",
        "layout::TokenLayout(",
        "rail_balance::checked_add_i64(",
        "jit::device_runtime->get_num_smem_bytes()",
        "jit::LaunchArgs(\n        num_channels, 32, num_smem_bytes)",
        'jit::compiler->build(\n            "rail_balance_hybrid_return_unshuffle"',
        ".launch_args = launch_args",
    ):
        assert token in source if token.startswith("struct ") else token in prepare

    # Raw ABI order mirrors RailBalanceHybridReturnUnshuffleRuntime::Args.
    ordered = (
        ".hidden = prepared.spec.hidden",
        ".num_topk = prepared.spec.num_topk",
        ".nccl_dev_comm = nccl_dev_comm",
        ".nccl_window = nccl_window",
        ".arena = arena",
        ".legacy_reduce_buffer = legacy_reduce_buffer",
        ".moved = moved",
        ".group_prefix = group_prefix",
        ".proxy_required = proxy_required",
        ".status = status",
        ".num_experts = num_experts",
        ".num_destinations = num_destinations",
        ".num_rails = num_rails",
        ".egress = egress",
        ".current_rank_idx = current_rank_idx",
        ".num_channels = prepared.spec.num_channels",
        ".num_max_tokens_per_rank = num_max_tokens_per_rank",
        ".proxy_capacity = proxy_capacity",
        ".launch_args = prepared.launch_args",
        "RailBalanceHybridReturnUnshuffleRuntime::launch(",
    )
    cursor = -1
    for token in ordered:
        next_cursor = raw_submit.index(token)
        assert next_cursor > cursor, token
        cursor = next_cursor

    # The committed body is a pure raw bind plus launch. In particular it may
    # not materialize Tensor pointers or reconstruct TokenLayout/smem.
    for forbidden in (
        "EP_HOST_ASSERT",
        "EP_UNIFIED_ASSERT",
        "layout::TokenLayout",
        "torch::",
        ".data_ptr",
        "jit::compiler",
        "::generate(",
        "cudaMalloc",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
    ):
        assert forbidden not in raw_submit, forbidden

    # Keep the old checked test wrapper, but make it delegate to the raw path.
    assert checked.count("EP_HOST_ASSERT(") == 4
    for token in (
        "plan.moved.data_ptr<int>()",
        "plan.group_prefix.data_ptr<int>()",
        "plan.proxy_required.data_ptr<int>()",
        "plan.status.data_ptr<int>()",
        "submit_prepared_rail_balance_hybrid_return_unshuffle(",
    ):
        assert token in checked, token
    assert "layout::TokenLayout" not in checked
    assert "jit::LaunchArgs(" not in checked

    # Pending owns the complete prepared object through a forward-declared
    # shared_ptr because unshuffle.hpp itself consumes PlanOutputs.
    assert "struct PreparedRailBalanceHybridReturnUnshuffle;" in pending_source
    assert (
        "std::shared_ptr<PreparedRailBalanceHybridReturnUnshuffle>\n"
        "        return_unshuffle;"
        in pending_source
    )
    assert (
        "std::make_shared<PreparedRailBalanceHybridReturnUnshuffle>("
        in callsite_source
    )
    assert (
        "prepare_rail_balance_hybrid_return_unshuffle(\n"
        "                    hidden, num_topk, num_channels)"
        in callsite_source
    )

    committed = _section(
        callsite_source,
        "pending.return_unshuffle_tested = true;",
        "// B4:",
    )
    assert "submit_prepared_rail_balance_hybrid_return_unshuffle(" in committed
    assert "*pending.return_unshuffle" in committed
    assert "launch_prepared_rail_balance_hybrid_return_unshuffle(" not in committed
    for forbidden in (
        ".data_ptr",
        "layout::TokenLayout",
        "HybridArenaLayout(",
        "EP_HOST_ASSERT",
        "jit::compiler",
        "cudaStreamSynchronize",
        "cudaMemcpyDeviceToHost",
    ):
        assert forbidden not in committed, forbidden

    print("PASS C080-H1 prepared return-unshuffle raw submit")


if __name__ == "__main__":
    main()
