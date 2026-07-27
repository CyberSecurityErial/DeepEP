"""C080-E force-only Hybrid dispatch cubin/codegen gate.

This is intentionally code-generation evidence only.  A local single-node
machine has no truthful Rail/Gin peer topology on which to launch these
scaleout>1 specializations.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import torch

from deep_ep import _C


_ROOT = Path(__file__).resolve().parents[2]
_FORCE_HEADER = (
    _ROOT / "deep_ep/include/deep_ep/impls/rail_balance_hybrid_dispatch.cuh"
)
_RUNTIME_HEADER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_dispatch.hpp"
)
_LEGACY_HEADER = _ROOT / "deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh"

_CASES = {
    # Labels are local-rails x logical-destinations; the C++ API takes D,G.
    # Production target: eight local rails and the common top-k=8 geometry.
    "8x2_h7168_k8": (2, 8, 7168, 8),
    "4x2_h7168_k4": (2, 4, 7168, 4),
    "2x4_h256_k4": (4, 2, 256, 4),
    # D>K exercises the alternate future combine-row geometry while dispatch
    # still carries only the same four-byte transit key.
    "2x4_h7168_k2_d_gt_k": (4, 2, 7168, 2),
}


def _derive_static_payload_counters(
    retained: tuple[int, ...], moved: tuple[int, ...], token_bytes: int,
) -> tuple[int, int, int, int]:
    """Derive dispatch-payload counters from plan tensors, never atomics."""
    assert token_bytes > 0
    assert len(retained) == len(moved)
    assert all(value >= 0 for value in retained + moved)
    retained_puts = sum(retained)
    moved_puts = sum(moved)
    payload_puts = retained_puts + moved_puts
    payload_gin_bytes = payload_puts * token_bytes
    return retained_puts, moved_puts, payload_puts, payload_gin_bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_case(name: str) -> None:
    scaleout, scaleup, hidden, topk = _CASES[name]
    # JIT KernelRuntime loads the cubin into the process's current CUDA driver
    # context.  Make that context explicit before invoking the C++ compile
    # probe; a cold process otherwise reaches CUDA_ERROR_INVALID_CONTEXT only
    # after ptxas has successfully emitted the cubin.
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    legacy_before = _sha256(_LEGACY_HEADER)
    result = _C._rail_balance_hybrid_dispatch_codegen_test(
        scaleout, scaleup, hidden, topk,
        64, 4, 16,
    )
    legacy_after = _sha256(_LEGACY_HEADER)

    assert legacy_before == legacy_after
    assert result["reuse_slot_indices"] is False
    assert result["num_sf_packs"] == 0
    assert result["proxy_capacity"] == 16
    assert result["num_threads"] == (4 + 4 + 4) * 32
    assert result["num_channels"] == 256
    assert result["num_forward_metadata_dims"] == 3 + 2 * topk

    code = result["code"]
    assert "#include <deep_ep/impls/rail_balance_hybrid_dispatch.cuh>" in code
    assert "#include <deep_ep/impls/hybrid_dispatch.cuh>" not in code
    assert "&rail_balance_hybrid_dispatch_impl<" in code
    assert "true, false," in code
    legacy_code = result["legacy_code"]
    assert "#include <deep_ep/impls/hybrid_dispatch.cuh>" in legacy_code
    assert "rail_balance_hybrid_dispatch.cuh" not in legacy_code
    assert "&hybrid_dispatch_impl<" in legacy_code
    assert "true, false," in legacy_code

    # The generated translation unit includes this audited implementation.
    # Freeze the transit snapshot and 3+2K metadata ABI at the source boundary
    # as well as through the runtime-returned dimension above.
    header = _FORCE_HEADER.read_text()
    runtime_header = _RUNTIME_HEADER.read_text()
    legacy_header = _LEGACY_HEADER.read_text()

    # The post-Gate2 adapter is intentionally a straight ABI bind followed by
    # launch.  Freeze both the kernel-argument order and the absence of a new
    # fallible host phase in this committed section.
    adapter_begin = runtime_header.index(
        "static void launch_prepared_rail_balance_hybrid_dispatch(")
    adapter_end = runtime_header.index(
        "\n}\n\n// Compile the byte-immutable legacy kernel", adapter_begin,
    ) + 2
    adapter = runtime_header[adapter_begin:adapter_end]
    adapter_fields = (
        ".x = x",
        ".sf = sf",
        ".topk_idx = topk_idx",
        ".topk_weights = topk_weights",
        ".copied_topk_idx = copied_topk_idx",
        ".cumulative_local_expert_recv_stats =",
        ".psum_num_recv_tokens_per_scaleup_rank =",
        ".psum_num_recv_tokens_per_expert =",
        ".num_unaligned_recv_tokens_per_expert =",
        ".dst_buffer_slot_idx = dst_buffer_slot_idx",
        ".token_metadata_at_forward = token_metadata_at_forward",
        ".num_tokens = num_tokens",
        ".sf_token_stride = sf_token_stride",
        ".sf_hidden_stride = sf_hidden_stride",
        ".nccl_dev_comm = nccl_dev_comm",
        ".nccl_window = nccl_window",
        ".buffer = buffer",
        ".workspace = workspace",
        ".mapped_host_workspace = mapped_host_workspace",
        ".rail_balance_arena = rail_balance_arena",
        ".rail_balance_retained = rail_balance_retained",
        ".rail_balance_moved = rail_balance_moved",
        ".rail_balance_group_prefix = rail_balance_group_prefix",
        ".rail_balance_proxy_required = rail_balance_proxy_required",
        ".scaleout_rank_idx = scaleout_rank_idx",
        ".scaleup_rank_idx = scaleup_rank_idx",
        ".launch_args = prepared.launch_args",
    )
    adapter_positions = [adapter.index(field) for field in adapter_fields]
    assert adapter_positions == sorted(adapter_positions)
    assert (
        "RailBalanceHybridDispatchRuntime::launch(\n"
        "        prepared.runtime, args, stream);"
    ) in adapter
    for forbidden in (
        "validate_rail_balance_hybrid_dispatch_spec(",
        "make_rail_balance_hybrid_dispatch_args(",
        "jit::compiler",
        "torch::",
        "cudaMemcpy",
        "cudaStreamSynchronize",
        "EP_HOST_ASSERT",
    ):
        assert forbidden not in adapter

    assert (
        "rail_balance::get_num_hybrid_forward_metadata_dims(kNumTopk)"
        in header
    )
    assert "rail_balance::kHybridForwardProxySlotDim" in header
    assert "rail_balance::kHybridForwardRouteBaseDim + lane_idx" in header
    assert (
        "rail_balance::kHybridForwardRouteBaseDim +\n"
        "                                kNumTopk + lane_idx"
        in header
    )
    assert "rail_balance_group_prefix" in header
    for workspace_limit in (
        "layout::WorkspaceLayout::kNumMaxRanks",
        "layout::WorkspaceLayout::kNumMaxExperts",
        "layout::WorkspaceLayout::kNumMaxExpertsPerRank",
    ):
        token = rf"\b{re.escape(workspace_limit)}\b"
        assert re.search(token, header)
        assert re.search(token, runtime_header)
    # proxy_required is intentionally ABI-only in C080-E. Gate2 validates it;
    # the committed kernel must never read it and diverge after Tag0.
    assert header.count("rail_balance_proxy_required") == 1
    retained_threshold = header.index(
        "if (stored_old_slot_idx < retained_count)")
    proxy_begin = header.index("const int proxy_begin =")
    proxy_loop = header.index("for (int proxy_slot = proxy_begin;")
    retained_staging = header.index(
        "arena_layout.get_retained_rail_staging_layout(",
        retained_threshold)
    remote_slot = header.index("const int remote_slot =", proxy_loop)
    grouped_put_helper = header.index("const auto issue_grouped_put =")
    final_action = header.index("ncclGin_VASignalAdd(", grouped_put_helper)
    proxy_put = header.index("issue_grouped_put(", proxy_loop)
    proxy_acquire = header.index(
        "const int ready = ptx::ld_acquire_sys<int>(", proxy_loop
    )
    final_tail = header.index("const auto signaled_tail =", grouped_put_helper)
    final_flush = header.index("gin.flush<ncclCoopWarp>();", proxy_put)
    assert retained_threshold < retained_staging < proxy_begin < proxy_loop
    assert (
        grouped_put_helper < final_action < proxy_loop < remote_slot <
        proxy_acquire < proxy_put < final_flush < final_tail
    )
    assert "RB_PROXY_BAD" not in header
    assert "embedded_proxy" not in header
    proxy_copy = header[proxy_acquire:proxy_put]
    assert "ptx::tma_load_1d(" in proxy_copy
    assert "proxy_token.get_base_ptr(), mbarrier_ptr" in proxy_copy
    assert "ptx::tma_store_1d(" in proxy_copy
    assert "ptx::tma_store_global_visibility_fence();" in proxy_copy
    assert "__threadfence_system();" not in proxy_copy
    assert "ncclGinOptFlagsDefault" in header[
        grouped_put_helper:final_tail
    ]
    assert "retained_count + proxy_slot - proxy_begin" in header[
        remote_slot:proxy_put]

    proxy_snapshot = header.index("int stored_proxy_slot = -1;")
    linked_list_overwrite = header.index(
        "tma_buffer.get_linked_list_idx_ptr()[lane_idx] =",
        proxy_snapshot,
    )
    metadata_snapshot = header.index(
        "rail_balance::kHybridForwardProxySlotDim] =", linked_list_overwrite)
    assert proxy_snapshot < linked_list_overwrite < metadata_snapshot

    # Moved copies change the source-rank buffer that owns a token. The
    # epilogue prefix must therefore be rebuilt from channel counts published
    # after their linked-list tails. The target waits for every channel credit
    # before consuming the reduced count.
    counter_clear = header.index(
        "ptx::st_relaxed_sys(\n"
        "            workspace_layout.get_scaleup_atomic_sender_counter()"
    )
    role_begin = header.index("// Different warp roles")
    assert counter_clear < role_begin
    first_arrival_barrier = header.index(
        "comm::kHybridDispatchTag1", role_begin)
    mailbox_reset = header.index(
        "ptx::st_relaxed_sys(scaleup_count_mailbox + thread_idx, int64_t(0))")
    mailbox_publish = header.index(
        "ptx::red_add_rel_sys(\n"
        "                        peer_mailbox,\n"
        "                        math::pack2<int, int64_t>(",
        role_begin,
    )
    mailbox_snapshot = header.index(
        "published_count = ptx::ld_acquire_sys<int64_t>(",
        first_arrival_barrier,
    )
    tail_publish = header.index(
        "ptx::red_add_rel_sys(\n"
        "                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j)"
    )
    tail_completion = header.index(
        "ptx::fence_acq_rel_sys();", tail_publish)
    assert (
        mailbox_reset < role_begin < tail_publish < mailbox_publish <
        tail_completion < first_arrival_barrier < mailbox_snapshot
    )
    prefix_write = header.index(
        "ptx::st_release_sys(\n"
        "                psum_num_recv_tokens_per_scaleup_rank + lane_idx",
        mailbox_snapshot,
    )
    epilogue_trigger = header.index(
        "cudaTriggerProgrammaticLaunchCompletion()", prefix_write)
    assert (
        first_arrival_barrier < mailbox_snapshot < prefix_write <
        epilogue_trigger
    )
    assert "DeepEP rail count timeout" in header
    assert "static_cast<uint64_t>(kNumChannels)" in header
    assert "kRailBalanceHybridDispatchCountTag" not in header

    # After the opening Tag0 epoch boundary the release specialization trusts
    # immutable Gate2 state. Compare with legacy instead of banning `return;`
    # outright: both headers retain the same safe preload-lambda early exit.
    after_tag0 = header[header.index("comm::kHybridDispatchTag0"):]
    legacy_after_tag0 = legacy_header[
        legacy_header.index("comm::kHybridDispatchTag0"):]
    assert after_tag0.count("return;") == legacy_after_tag0.count("return;")
    for suspicious_token in (
        "EP_DEVICE_ASSERT(",
        "std::clamp(",
        "report_error(",
    ):
        assert after_tag0.count(suspicious_token) <= legacy_after_tag0.count(
            suspicious_token)
    assert "rail_balance_status" not in after_tag0
    assert 'asm("trap' not in after_tag0
    assert "__trap" not in after_tag0

    scaleout_begin = header.index(
        "} else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps)")
    scaleout_end = header.index(
        "\n    } else {\n        const int forward_warp_idx", scaleout_begin)
    scaleout_body = header[scaleout_begin:scaleout_end]
    # The correctness-first grouped put helper doorbells every payload and
    # publishes a per-slot epoch marker as its remote completion action. Local
    # destination remains a TMA bypass.
    assert scaleout_body.count("gin.put<ncclTeamTagRail>(") == 1
    assert "ncclGinOptFlagsAggregateRequests" not in scaleout_body
    assert scaleout_body.count("ncclGin_VASignalAdd(") == 1
    assert "encoded_proxy - forward_ready_epoch_base" in header
    assert "DeepEP rail payload timeout" in header
    assert "if (lane_idx == dst_scaleout_rank_idx)" in scaleout_body
    assert "flush_async<ncclTeamTagRail" not in scaleout_body
    assert "gin.wait(*completion_request);" not in scaleout_body
    assert scaleout_body.count("gin.flush<ncclCoopWarp>();") == 2
    assert "if (++num_remote_groups_since_flush == 2)" in scaleout_body
    assert "stored_old_slot_idx < retained_count" in scaleout_body
    assert "for (int proxy_slot = proxy_begin;" in scaleout_body

    counters = _derive_static_payload_counters((3, 0, 5), (0, 4, 2), 1024)
    assert counters == (8, 6, 14, 14336)

    print(
        "PASS C080-E dispatch codegen "
        f"case={name} topology={scaleout}x{scaleup} hidden={hidden} "
        f"topk={topk} metadata={result['num_forward_metadata_dims']} "
        f"legacy_sha={legacy_after}"
    )


def _expect_codegen_rejection(*args: int) -> None:
    try:
        _C._rail_balance_hybrid_dispatch_codegen_test(*args)
    except RuntimeError:
        return
    raise AssertionError(f"expected codegen preflight rejection: {args}")


def _test_constraint_rejections() -> None:
    # D/G>=2, hidden-elements%256, SM>=2, channel warps>0, and Pcap>0
    # are host gates. kReuse=false and SFPacks=0 are not caller-configurable.
    _expect_codegen_rejection(1, 8, 256, 4, 64, 4, 16)
    _expect_codegen_rejection(8, 1, 256, 4, 64, 4, 16)
    _expect_codegen_rejection(2, 4, 128, 4, 64, 4, 16)
    _expect_codegen_rejection(2, 4, 256, 4, 1, 4, 16)
    _expect_codegen_rejection(2, 4, 256, 4, 64, 0, 16)
    _expect_codegen_rejection(2, 4, 256, 4, 64, 4, 0)
    # E=2112 is total-only: 2112/64=33 remains below the per-rank ceiling.
    _expect_codegen_rejection(2, 32, 256, 4, 64, 4, 16, 2112)
    _expect_codegen_rejection(2, 2, 256, 4, 64, 4, 16, 2048)
    print("PASS C080-E dispatch codegen constraint rejections 8/8")


def _run_watchdog(case: str, watchdog_seconds: int) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--case", case,
        "--watchdog-seconds", str(watchdog_seconds),
    ]
    # NVCC and cuobjdump inherit this process group. A timeout therefore
    # terminates the complete compile tree instead of leaving an orphan that
    # races a later cache rename.
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError(
            "C080-E dispatch codegen watchdog expired after "
            f"{watchdog_seconds}s for {case}") from error
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(_CASES))
    parser.add_argument("--watchdog-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.watchdog_seconds <= 0:
        parser.error("--watchdog-seconds must be positive")
    if args.case is not None:
        _run_case(args.case)
        return
    for name in _CASES:
        _run_watchdog(name, args.watchdog_seconds)
    # The valid child cases have already proved CUDA context setup. Keep the
    # small rejection matrix in the parent; it launches no cubin.
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    _test_constraint_rejections()


if __name__ == "__main__":
    main()
