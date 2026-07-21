"""C080-F isolated force Hybrid combine cubin/codegen gate.

The local machine has no truthful scaleout>1 Rail/Gin topology, so this test
compiles the production specialization and its immutable legacy baseline but
never launches either persistent kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch

from deep_ep import _C


_ROOT = Path(__file__).resolve().parents[2]
_FORCE_HEADER = (
    _ROOT / "deep_ep/include/deep_ep/impls/rail_balance_hybrid_combine.cuh"
)
_RUNTIME_HEADER = (
    _ROOT / "csrc/kernels/elastic/rail_balance_hybrid_combine.hpp"
)
_LEGACY_HEADER = _ROOT / "deep_ep/include/deep_ep/impls/hybrid_combine.cuh"

_CASES = {
    # Labels are local rails G x logical destinations D; the C++ API takes D,G.
    # The first four entries form the minimal complete matrix for
    # (D<=K, G<=K), which selects scaleout/scaleup rank layouts independently.
    "8x2_h7168_k8_rank_tt": (2, 8, 7168, 8),
    "2x4_h7168_k2_rank_ft": (4, 2, 7168, 2),
    "8x2_h7168_k4_rank_tf": (2, 8, 7168, 4),
    "8x4_h256_k2_rank_ff": (4, 8, 256, 2),
    # Dispatch-matching production/codegen geometries.
    "4x2_h7168_k4": (2, 4, 7168, 4),
    "2x4_h256_k4": (4, 2, 256, 4),
}

_EXPECTED_RANK_LAYOUTS = {
    "8x2_h7168_k8_rank_tt": (True, True),
    "2x4_h7168_k2_rank_ft": (False, True),
    "8x2_h7168_k4_rank_tf": (True, False),
    "8x4_h256_k2_rank_ff": (False, False),
    "4x2_h7168_k4": (True, True),
    "2x4_h256_k4": (True, True),
}
assert {
    _EXPECTED_RANK_LAYOUTS[name] for name in tuple(_CASES)[:4]
} == {(True, True), (False, True), (True, False), (False, False)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _derive_static_return_counters(
    retained: tuple[int, ...], moved: tuple[int, ...], token_bytes: int,
) -> tuple[int, int, int, int, int]:
    """Derive release return counters from the immutable plan, never atomics."""
    assert token_bytes > 0
    assert len(retained) == len(moved)
    assert all(value >= 0 for value in retained + moved)
    retained_puts = sum(retained)
    proxy_return_puts = sum(moved)
    payload_puts = retained_puts + proxy_return_puts
    return (
        retained_puts,
        proxy_return_puts,
        payload_puts,
        proxy_return_puts * token_bytes,
        payload_puts * token_bytes,
    )


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    assert count == 1, (count, old)
    return source.replace(old, new, 1)


def _normalize_force_to_legacy(force: str) -> str:
    """Undo exactly the audited force edits; no unrelated drift is accepted."""
    force = _replace_once(
        force, "rail_balance_hybrid_combine_impl(", "hybrid_combine_impl(")
    force = _replace_once(
        force, "                    void* rail_balance_proxy_return_base,\n", "")

    static_gates = '''    // Force-v1 is intentionally a separate non-expanded, multiple-reduction
    // specialization. These compile-time gates keep unsupported replay modes
    // out of the committed Hybrid epoch instead of branching at runtime.
    EP_STATIC_ASSERT(not kUseExpandedLayout,
                     "Rail-balanced Hybrid combine does not support expanded mode");
    EP_STATIC_ASSERT(kAllowMultipleReduction,
                     "Rail-balanced Hybrid combine requires multiple reduction");
    EP_STATIC_ASSERT(kNumSMs >= 2,
                     "Rail-balanced Hybrid combine requires at least two SMs");
    EP_STATIC_ASSERT(kNumScaleoutRanks >= 2 and kNumScaleoutRanks <= 32,
                     "Invalid force Hybrid scale-out size");
    EP_STATIC_ASSERT(kNumScaleupRanks >= 2 and kNumScaleupRanks <= 32,
                     "Invalid force Hybrid scale-up size");
    EP_STATIC_ASSERT(kNumChannels <= deep_ep::kNumMaxChannels,
                     "Force Hybrid channel count exceeds the legacy ceiling");
    EP_STATIC_ASSERT(kNumExperts > 0 and kNumExperts % kNumRanks == 0,
                     "Invalid force Hybrid expert geometry");
    EP_STATIC_ASSERT(kNumRanks <= layout::WorkspaceLayout::kNumMaxRanks,
                     "Force Hybrid rank count exceeds the legacy workspace");
    EP_STATIC_ASSERT(kNumExperts <= layout::WorkspaceLayout::kNumMaxExperts,
                     "Force Hybrid expert count exceeds the legacy workspace");
    EP_STATIC_ASSERT(
        kNumExperts / kNumRanks <=
            layout::WorkspaceLayout::kNumMaxExpertsPerRank,
        "Force Hybrid local expert count exceeds the legacy workspace");
    EP_STATIC_ASSERT(kNumTopk >= 1 and kNumTopk <= 32,
                     "Invalid force Hybrid top-k");
    EP_STATIC_ASSERT(
        kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 and
        kNumHiddenBytes % ptx::kNumTMAAlignBytes == 0,
        "Force Hybrid hidden shape must preserve BF16/TMA alignment");

'''
    force = _replace_once(force, static_gates, "")

    legacy_expanded_assert = '''        // Expanding send mode must not be backward
        if constexpr (kDoExpandedSend)
            EP_DEVICE_ASSERT(topk_weights == nullptr);

'''
    force = _replace_once(
        force, "        // Tail issuer\n",
        legacy_expanded_assert + "        // Tail issuer\n")
    force = _replace_once(
        force,
        "constexpr int kNumForwardMetadataDims = 3 + kNumTopk * 2;",
        "constexpr int kNumForwardMetadataDims = 2 + kNumTopk * 2;")

    force_replay = '''            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            // The ending marker is warp-uniform. Stop before reading the added
            // p field (or the legacy 2K fields) from the otherwise unused
            // sentinel row.
            if (src_token_global_idx < 0)
                break;
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            // Force dispatch snapshots p before overwriting linked-list transit
            // scratch. Gate #2 and the one-live force handle make this metadata
            // immutable for the committed combine epoch: do not add a post-Tag0
            // bounds check, clamp, status branch, trap, or early return here.
            const auto proxy_slot = __ldg(
                token_metadata_at_forward + i * kNumForwardMetadataDims + 2);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 3 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 3 + kNumTopk + lane_idx) : -1;
'''
    legacy_replay = '''            const auto src_token_global_idx = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims);
            const auto is_token_last_in_chunk = __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 1);
            const auto src_rank_idx = src_token_global_idx / kNumMaxTokensPerRank;
            const auto src_scaleout_rank_idx = src_rank_idx / kNumScaleupRanks;
            const auto src_token_idx = src_token_global_idx % kNumMaxTokensPerRank;
            auto stored_src_scaleup_rank_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + lane_idx) : -1;
            auto stored_src_slot_idx = lane_idx < kNumTopk ?
                __ldg(token_metadata_at_forward + i * kNumForwardMetadataDims + 2 + kNumTopk + lane_idx) : -1;
            if (src_token_global_idx < 0)
                break;

'''
    force = _replace_once(force, force_replay, legacy_replay)

    proxy_target = '''
                // A retained record returns to the legacy owner/token slot. A
                // moved record returns along this ingress GPU's Rail context to
                // the symmetric source-egress proxy slot p. Only the remote
                // destination pointer changes; reduction, staging, aggregation,
                // flush, and completion remain byte-for-byte legacy behavior.
                const auto recv_token_buffer_ptr = proxy_slot >= 0 ?
                    math::advance_ptr(
                        rail_balance_proxy_return_base,
                        static_cast<int64_t>(proxy_slot) *
                            token_layout.get_num_bytes<false>()) :
                    recv_token_buffer.get_base_ptr();
'''
    force = _replace_once(force, proxy_target, "")
    force = _replace_once(
        force,
        "last_recv_token_buffer_ptr = recv_token_buffer_ptr;",
        "last_recv_token_buffer_ptr = recv_token_buffer.get_base_ptr();")
    # The immutable legacy root contains two whitespace-only lines. Keep the
    # new file diff-clean, then restore those bytes solely for the exact golden.
    force = _replace_once(
        force,
        "stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);\n\n",
        "stored_num_tokens_recv[j] += static_cast<int>(stored_is_scaleup_rank_needed[j]);\n            \n")
    force = _replace_once(
        force,
        "                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;\n                }\n\n",
        "                        lane_idx * scaleup_buffer.num_max_tokens_per_rank + src_slot_idx;\n                }\n                \n")
    return force


def _run_case(name: str) -> None:
    scaleout, scaleup, hidden, topk = _CASES[name]
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")

    legacy_before = _sha256(_LEGACY_HEADER)
    result = _C._rail_balance_hybrid_combine_codegen_test(
        scaleout, scaleup, hidden, topk, 64, 4, 256)
    legacy_after = _sha256(_LEGACY_HEADER)
    assert legacy_before == legacy_after

    assert result["use_expanded_layout"] is False
    assert result["allow_multiple_reduction"] is True
    assert result["direct_proxy_return_base"] is True
    assert result["num_threads"] == (4 + 4) * 32
    assert result["num_channels"] == 256
    assert result["num_forward_metadata_dims"] == 3 + 2 * topk
    assert result["combine_token_bytes"] > hidden * 2

    code = result["code"]
    assert "#include <deep_ep/impls/rail_balance_hybrid_combine.cuh>" in code
    assert "#include <deep_ep/impls/hybrid_combine.cuh>" not in code
    assert "&rail_balance_hybrid_combine_impl<" in code
    assert "false, true," in code
    legacy_code = result["legacy_code"]
    assert "#include <deep_ep/impls/hybrid_combine.cuh>" in legacy_code
    assert "rail_balance_hybrid_combine.cuh" not in legacy_code
    assert "&hybrid_combine_impl<" in legacy_code
    assert "false, true," in legacy_code

    header = _FORCE_HEADER.read_text()
    runtime_header = _RUNTIME_HEADER.read_text()
    legacy_header = _LEGACY_HEADER.read_text()
    assert _normalize_force_to_legacy(header) == legacy_header
    # The compile-only comparison owns two independent LaunchRuntime-derived
    # caches. It must never initialize CombineRuntime's function-static include
    # hash with a synthetic probe geometry.
    assert "CombineRuntime::generate" not in runtime_header
    assert '#include "combine.hpp"' not in runtime_header
    assert "RailBalanceHybridCombineRuntime::generate" in runtime_header
    assert "RailBalanceHybridLegacyCombineProbeRuntime::generate" in \
        runtime_header
    assert "rail_balance_hybrid_combine_legacy_probe_v1_" in runtime_header
    assert "constexpr int kNumForwardMetadataDims = 3 + kNumTopk * 2;" in header
    assert "i * kNumForwardMetadataDims + 2);" in header
    assert "i * kNumForwardMetadataDims + 3 + lane_idx" in header
    assert "i * kNumForwardMetadataDims + 3 + kNumTopk + lane_idx" in header
    assert header.count("rail_balance_proxy_return_base") == 2

    # The sentinel break must precede every force-only p/2K read.
    replay = header.index("// Replay the dispatch")
    src_load = header.index("const auto src_token_global_idx =", replay)
    sentinel = header.index("if (src_token_global_idx < 0)", src_load)
    proxy_load = header.index("const auto proxy_slot =", sentinel)
    source_slots = header.index("auto stored_src_scaleup_rank_idx", proxy_load)
    assert src_load < sentinel < proxy_load < source_slots

    proxy_choice = header.index(
        "const auto recv_token_buffer_ptr = proxy_slot >= 0 ?")
    proxy_base = header.index("rail_balance_proxy_return_base", proxy_choice)
    legacy_fallback = header.index(
        "recv_token_buffer.get_base_ptr();", proxy_base)
    rdma_record = header.index(
        "last_recv_token_buffer_ptr = recv_token_buffer_ptr;", legacy_fallback)
    final_flush = header.index(
        "// Issue the last RDMA", rdma_record)
    rail_completion = header.index("gin.flush<ncclCoopWarp>();", final_flush)
    assert proxy_choice < proxy_base < legacy_fallback < rdma_record
    assert rdma_record < final_flush < rail_completion

    # Compare the committed region to legacy: force moves the same uniform
    # sentinel break earlier and removes the now-impossible expanded assert;
    # it adds no status, clamp, trap, early return, or payload atomic.
    after_tag0 = header[header.index("comm::kHybridCombineTag0"):]
    legacy_after_tag0 = legacy_header[
        legacy_header.index("comm::kHybridCombineTag0"):]
    assert after_tag0.count("return;") == legacy_after_tag0.count("return;")
    assert after_tag0.count("break;") == legacy_after_tag0.count("break;")
    assert after_tag0.count("EP_DEVICE_ASSERT(") < \
        legacy_after_tag0.count("EP_DEVICE_ASSERT(")
    for suspicious in ("std::clamp(", "report_error(", "atomicAdd("):
        assert after_tag0.count(suspicious) == legacy_after_tag0.count(suspicious)
    assert "rail_balance_status" not in after_tag0
    assert 'asm("trap' not in after_tag0
    assert "__trap" not in after_tag0

    counters = _derive_static_return_counters(
        (3, 0, 5), (0, 4, 2), int(result["combine_token_bytes"]))
    token_bytes = int(result["combine_token_bytes"])
    assert counters == (8, 6, 14, 6 * token_bytes, 14 * token_bytes)

    rank_layouts = (scaleout <= topk, scaleup <= topk)
    assert rank_layouts == _EXPECTED_RANK_LAYOUTS[name]
    print(
        "PASS C080-F combine codegen "
        f"case={name} topology={scaleout}x{scaleup} hidden={hidden} "
        f"topk={topk} layouts={rank_layouts} "
        f"metadata={result['num_forward_metadata_dims']} "
        f"token_bytes={token_bytes} legacy_sha={legacy_after}"
    )


def _expect_codegen_rejection(*args: int) -> None:
    try:
        _C._rail_balance_hybrid_combine_codegen_test(*args)
    except RuntimeError:
        return
    raise AssertionError(f"expected codegen preflight rejection: {args}")


def _test_constraint_rejections() -> None:
    _expect_codegen_rejection(1, 8, 256, 4, 64, 4, 256)
    _expect_codegen_rejection(8, 1, 256, 4, 64, 4, 256)
    _expect_codegen_rejection(2, 4, 128, 4, 64, 4, 256)
    _expect_codegen_rejection(2, 4, 256, 4, 1, 4, 256)
    _expect_codegen_rejection(2, 4, 256, 4, 64, 0, 256)
    _expect_codegen_rejection(2, 4, 256, 4, 64, 17, 256)
    # WorkspaceLayout maxima. E=2112 is total-only because 2112/64=33.
    _expect_codegen_rejection(2, 32, 256, 4, 64, 4, 2112)
    _expect_codegen_rejection(2, 2, 256, 4, 64, 4, 2048)
    print("PASS C080-F combine codegen constraint rejections 8/8")


def _run_watchdog(case: str, watchdog_seconds: int) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--case", case,
        "--watchdog-seconds", str(watchdog_seconds),
    ]
    # NVCC and cuobjdump inherit this process group. Kill the full compile tree
    # on timeout so no orphan can race a later JIT cache rename.
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
            "C080-F combine codegen watchdog expired after "
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
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    _test_constraint_rejections()


if __name__ == "__main__":
    main()
