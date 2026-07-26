"""Source gate for the C080 prepared source-shuffle submit path."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_shuffle.hpp"
_KERNEL = (
    _ROOT / "deep_ep/include/deep_ep/impls/rail_balance_hybrid_shuffle.cuh"
)
_PTX = _ROOT / "deep_ep/include/deep_ep/common/ptx.cuh"
_PENDING = _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp"
_CALLSITE = _ROOT / "csrc/elastic/buffer.hpp"


def _section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def main() -> None:
    source = _ADAPTER.read_text()
    kernel_source = _KERNEL.read_text()
    ptx_source = _PTX.read_text()
    pending_source = _PENDING.read_text()
    callsite_source = _CALLSITE.read_text()

    prepare = _section(
        source,
        "prepare_rail_balance_hybrid_source_shuffle(",
        "// Committed-stage adapter.",
    )
    raw_submit = _section(
        source,
        "static void submit_prepared_rail_balance_hybrid_source_shuffle(",
        "// Compatibility layer for the existing strict LSA tests.",
    )
    checked = _section(
        source,
        "static void launch_prepared_rail_balance_hybrid_source_shuffle(",
        "}  // namespace deep_ep::elastic",
    )

    # Prepare freezes all geometry which controls code generation, shared
    # memory, or launch shape before the collective force epoch begins.
    for token in (
        ".hidden = hidden",
        ".num_topk = num_topk",
        ".num_channels = num_channels",
        ".num_tokens = num_tokens",
        "layout::TokenLayout(",
        "rail_balance::checked_add_i64(",
        "jit::device_runtime->get_num_smem_bytes()",
        "const int num_active_channels",
        'jit::compiler->build(\n            "rail_balance_hybrid_source_shuffle"',
        ".launch_args = jit::LaunchArgs(",
    ):
        assert token in prepare, token

    # The raw committed ABI is a pure pointer/count bind plus launch. In
    # particular N comes only from the same Prepared object which froze grid.
    assert ".num_tokens = prepared.spec.num_tokens" in raw_submit
    assert "const int& num_tokens" not in raw_submit
    assert "RailBalanceHybridSourceShuffleRuntime::launch(" in raw_submit
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

    # Existing strict LSA tests keep their checked owning-plan wrapper.
    assert checked.count("EP_HOST_ASSERT(") == 5
    assert "prepared.spec.num_tokens == num_tokens" in checked
    assert "submit_prepared_rail_balance_hybrid_source_shuffle(" in checked
    assert "RailBalanceHybridSourceShuffleRuntime::launch(" not in checked
    for token in (
        "plan.owner_channel_prefix.data_ptr<int>()",
        "plan.keep_count.data_ptr<int>()",
        "plan.segments.data_ptr<int>()",
        "plan.num_segments.data_ptr<int>()",
        "plan.retained.data_ptr<int>()",
        "plan.moved_channel_prefix.data_ptr<int>()",
        "plan.group_prefix.data_ptr<int>()",
        "plan.proxy_required.data_ptr<int>()",
        "plan.status.data_ptr<int>()",
    ):
        assert token in checked, token

    assert "struct PreparedRailBalanceHybridSourceShuffle;" in pending_source
    assert (
        "std::shared_ptr<PreparedRailBalanceHybridSourceShuffle>\n"
        "        source_shuffle;"
        in pending_source
    )
    assert (
        "std::make_shared<PreparedRailBalanceHybridSourceShuffle>("
        in callsite_source
    )
    assert (
        "prepare_rail_balance_hybrid_source_shuffle(\n"
        "                    hidden, num_topk, num_channels, num_tokens)"
        in callsite_source
    )

    # The descriptor-free handoff relies on one barrier rather than a ready
    # word per proxy copy.  Completed TMA peer writes must cross from the
    # async proxy into the generic-global domain before that barrier signals.
    wait = kernel_source.index("ptx::tma_store_wait();")
    visibility = kernel_source.index(
        "ptx::tma_store_global_visibility_fence();", wait
    )
    assert wait < visibility
    assert 'asm volatile("fence.proxy.async.global;"' in ptx_source

    print("PASS C080-H1b prepared source-shuffle raw submit")


if __name__ == "__main__":
    main()
