"""Source gate for the private C080 prepared epilogue adapters.

The adapters deliberately reuse unchanged production kernels and JIT cache
identities. Their raw submit halves must remain pure argument binds + launch.
"""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_ADAPTER = _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_epilogue.hpp"
_LEGACY = _ROOT / "csrc/kernels/elastic/dispatch.hpp"
_LEGACY_CALLSITE = _ROOT / "csrc/elastic/buffer.hpp"
_LOCAL_BARRIER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp"
)


def _section(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def main() -> None:
    source = _ADAPTER.read_text()
    legacy = _LEGACY.read_text()
    legacy_callsite = _LEGACY_CALLSITE.read_text()
    local_barrier_source = _LOCAL_BARRIER.read_text()
    prepare = _section(
        source,
        "prepare_rail_balance_hybrid_dispatch_epilogue(",
        "// Main dispatch may require its ordinary CPU count/output allocation phase",
    )
    committed = _section(
        source,
        "launch_prepared_rail_balance_hybrid_dispatch_epilogue(",
        "// Private C080 force/test-only wrapper",
    )

    # Same TokenLayout, warp ceiling, launch geometry, generator, and build
    # name imply byte-identical generated source and therefore the same JIT
    # cache signature as launch_dispatch_copy_epilogue.
    identity_tokens = (
        "layout::TokenLayout(\n        num_hidden_bytes, 0, num_topk, true)",
        "num_smem_bytes / token_layout.get_num_bytes<true>(), 32",
        "num_warps * 32, num_smem_bytes, 1, false, true",
        '"dispatch_copy_epilogue"',
        "DispatchCopyEpilogueRuntime::generate(args)",
    )
    for token in identity_tokens:
        assert token in prepare, token
    for token in (
        "layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true)",
        "num_smem_bytes / token_layout.get_num_bytes<true>(), 32",
        "const auto num_threads = num_warps * 32;",
        "num_threads, num_smem_bytes, 1, false, true",
        'jit::compiler->build("dispatch_copy_epilogue", code)',
        "DispatchCopyEpilogueRuntime::generate(args)",
    ):
        assert token in legacy, token

    # Force-v1 supports only BF16, non-cached, non-expanded, no-SF,
    # no-zero-padding, alignment-one.  Freeze this in both generated code and
    # committed launch arguments.
    fixed_fields = (
        ".do_expand = false",
        ".cached_mode = false",
        ".do_zero_padding = false",
        ".num_sf_packs = 0",
        ".expert_alignment = 1",
        ".recv_sf = nullptr",
        ".recv_sf_token_stride = 0",
        ".recv_sf_hidden_stride = 0",
        ".num_unaligned_recv_tokens_per_expert = nullptr",
    )
    for field in fixed_fields:
        assert field in prepare, field
        assert field in committed, field
    assert "hidden * sizeof(nv_bfloat16)" in prepare
    assert "const int& num_sms" in prepare
    assert "num_sms == max_num_sms" in prepare
    assert "const int num_sms = jit::device_runtime->get_num_sms()" not in prepare
    assert ".num_sms = num_sms" in prepare
    # The immutable legacy callsite also launches the copy epilogue on all
    # physical SMs, independently of the main dispatch's selected SM count.
    assert "jit::device_runtime->get_num_sms()," in legacy_callsite[
        legacy_callsite.index("launch_dispatch_copy_epilogue("):
    ]

    # The committed adapter may only bind pointers/counts to the frozen spec
    # and submit the already-built runtime.
    assert (
        "DispatchCopyEpilogueRuntime::launch(prepared.runtime, args, stream);"
        in committed
    )
    for forbidden in (
        "EP_HOST_ASSERT",
        "jit::compiler",
        "::generate(",
        "torch::",
        "cudaMalloc",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "status",
    ):
        assert forbidden not in committed, forbidden

    raw_combine = _section(
        source,
        "submit_prepared_rail_balance_hybrid_combine_epilogue(",
        "// Checked wrapper retained for the existing private standalone test hook.",
    )
    checked_combine = _section(
        source,
        "launch_prepared_rail_balance_hybrid_combine_epilogue(",
        "}  // namespace deep_ep::elastic",
    )
    assert (
        "CombineReduceEpilogueRuntime::launch(\n"
        "        prepared.runtime, args, stream);"
    ) in raw_combine
    for forbidden in (
        "EP_HOST_ASSERT",
        "jit::compiler",
        "::generate(",
        "torch::",
        ".data_ptr",
        "cudaMalloc",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "status",
    ):
        assert forbidden not in raw_combine, forbidden
    assert checked_combine.count("EP_HOST_ASSERT") == 6
    assert (
        "submit_prepared_rail_balance_hybrid_combine_epilogue("
        in checked_combine
    )
    assert "CombineReduceEpilogueRuntime::launch(" not in checked_combine

    # H4b freezes LaunchArgs before Gate1.  The committed raw submit remains
    # assertion-free, while the standalone checked wrapper may only construct
    # the fixed launch geometry and delegate once.
    local_barrier_submit = _section(
        local_barrier_source,
        "submit_prepared_rail_balance_hybrid_local_barrier(",
        "static void launch_prepared_rail_balance_hybrid_local_barrier(",
    )
    local_barrier_checked = _section(
        local_barrier_source,
        "launch_prepared_rail_balance_hybrid_local_barrier(",
        "struct PreparedRailBalanceHybridPlan",
    )
    assert "RailBalanceHybridLocalBarrierRuntime::launch(" in local_barrier_submit
    for forbidden in (
        "EP_HOST_ASSERT",
        "jit::compiler",
        "::generate(",
        "torch::",
        ".data_ptr",
        "cudaMalloc",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "status",
    ):
        assert forbidden not in local_barrier_submit, forbidden
    assert local_barrier_checked.count(
        "submit_prepared_rail_balance_hybrid_local_barrier(") == 1
    assert "RailBalanceHybridLocalBarrierRuntime::launch(" not in \
        local_barrier_checked

    print("PASS C080-H1b prepared epilogue/barrier submit adapters")


if __name__ == "__main__":
    main()
