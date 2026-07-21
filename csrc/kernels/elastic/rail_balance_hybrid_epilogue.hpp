#pragma once

#include <algorithm>
#include <climits>
#include <memory>

#include <deep_ep/common/layout.cuh>

#include "combine.hpp"
#include "dispatch.hpp"

namespace deep_ep::elastic {

// The force-v1 dispatch epilogue is the unchanged production BF16,
// non-cached, non-expanded copy kernel. Freeze every templated field during
// prepare so its later prebuilt launch only binds runtime pointers/counts.
struct RailBalanceHybridDispatchEpilogueSpec {
    int num_sms;
    int num_channels;
    int num_warps;
    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int num_hidden_bytes;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_topk;
};

struct PreparedRailBalanceHybridDispatchEpilogue {
    std::shared_ptr<jit::KernelRuntime> runtime;
    RailBalanceHybridDispatchEpilogueSpec spec;
    jit::LaunchArgs launch_args;
};

static PreparedRailBalanceHybridDispatchEpilogue
prepare_rail_balance_hybrid_dispatch_epilogue(
        const int& hidden,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_topk,
        const int& num_scaleout_ranks,
        const int& num_scaleup_ranks,
        const int& num_sms,
        const int& num_channels) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    EP_HOST_ASSERT(
        static_cast<int64_t>(hidden) * sizeof(nv_bfloat16) <= INT_MAX);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_scaleout_ranks >= 2 and num_scaleout_ranks <= 32);
    EP_HOST_ASSERT(num_scaleup_ranks >= 2 and num_scaleup_ranks <= 32);
    EP_HOST_ASSERT(num_channels >= 1 and num_channels <= kNumMaxChannels);
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    EP_HOST_ASSERT(num_ranks <= layout::WorkspaceLayout::kNumMaxRanks);
    EP_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_experts <= layout::WorkspaceLayout::kNumMaxExperts);
    EP_HOST_ASSERT(
        num_experts / num_ranks <=
        layout::WorkspaceLayout::kNumMaxExpertsPerRank);

    const int max_num_sms = jit::device_runtime->get_num_sms();
    // The legacy dispatch callsite deliberately gives its copy epilogue all
    // physical SMs even when the main communication kernel uses fewer.
    EP_HOST_ASSERT(num_sms == max_num_sms);
    const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
    const int num_hidden_bytes = hidden * sizeof(nv_bfloat16);
    // This is exactly launch_dispatch_copy_epilogue's shared-memory formula.
    const auto token_layout = layout::TokenLayout(
        num_hidden_bytes, 0, num_topk, true);
    const int num_warps = std::min<int>(
        num_smem_bytes / token_layout.get_num_bytes<true>(), 32);
    EP_HOST_ASSERT(num_sms > 0 and num_smem_bytes > 0 and num_warps > 0);

    const auto launch_args = jit::LaunchArgs(
        num_sms, num_warps * 32, num_smem_bytes, 1, false, true);
    const RailBalanceHybridDispatchEpilogueSpec spec = {
        .num_sms = num_sms,
        .num_channels = num_channels,
        .num_warps = num_warps,
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = num_hidden_bytes,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
    };
    const DispatchCopyEpilogueRuntime::Args args = {
        .do_expand = false,
        .cached_mode = false,
        .do_zero_padding = false,
        .num_channels = spec.num_channels,
        .num_warps = spec.num_warps,
        .num_scaleout_ranks = spec.num_scaleout_ranks,
        .num_scaleup_ranks = spec.num_scaleup_ranks,
        .num_hidden_bytes = spec.num_hidden_bytes,
        .num_sf_packs = 0,
        .num_max_tokens_per_rank = spec.num_max_tokens_per_rank,
        .num_experts = spec.num_experts,
        .num_topk = spec.num_topk,
        .expert_alignment = 1,
        .buffer = nullptr,
        .workspace = nullptr,
        .psum_num_recv_tokens_per_scaleup_rank = nullptr,
        .psum_num_recv_tokens_per_expert = nullptr,
        .recv_x = nullptr,
        .recv_sf = nullptr,
        .recv_topk_idx = nullptr,
        .recv_topk_weights = nullptr,
        .recv_src_metadata = nullptr,
        .channel_linked_list = nullptr,
        .num_unaligned_recv_tokens_per_expert = nullptr,
        .num_recv_tokens = 0,
        .recv_sf_token_stride = 0,
        .recv_sf_hidden_stride = 0,
        .scaleout_rank_idx = 0,
        .scaleup_rank_idx = 0,
        .launch_args = launch_args,
    };
    // Identical name + generated source intentionally shares the immutable
    // production dispatch-copy epilogue cache entry.
    return {
        .runtime = jit::compiler->build(
            "dispatch_copy_epilogue",
            DispatchCopyEpilogueRuntime::generate(args)),
        .spec = spec,
        .launch_args = launch_args,
    };
}

// Main dispatch may require its ordinary CPU count/output allocation phase
// before this epilogue. Once the caller enters this prebuilt adapter, however,
// keep it to a pure ABI bind plus launch: no JIT, validation, allocation,
// synchronization, or device-to-host status path belongs here.
static void launch_prepared_rail_balance_hybrid_dispatch_epilogue(
        const PreparedRailBalanceHybridDispatchEpilogue& prepared,
        void* buffer,
        void* workspace,
        int* psum_num_recv_tokens_per_scaleup_rank,
        int* psum_num_recv_tokens_per_expert,
        void* recv_x,
        topk_idx_t* recv_topk_idx,
        float* recv_topk_weights,
        int* recv_src_metadata,
        int* channel_linked_list,
        const int& num_recv_tokens,
        const int& scaleout_rank_idx,
        const int& scaleup_rank_idx,
        const at::cuda::CUDAStream& stream) {
    const auto& spec = prepared.spec;
    const DispatchCopyEpilogueRuntime::Args args = {
        .do_expand = false,
        .cached_mode = false,
        .do_zero_padding = false,
        .num_channels = spec.num_channels,
        .num_warps = spec.num_warps,
        .num_scaleout_ranks = spec.num_scaleout_ranks,
        .num_scaleup_ranks = spec.num_scaleup_ranks,
        .num_hidden_bytes = spec.num_hidden_bytes,
        .num_sf_packs = 0,
        .num_max_tokens_per_rank = spec.num_max_tokens_per_rank,
        .num_experts = spec.num_experts,
        .num_topk = spec.num_topk,
        .expert_alignment = 1,
        .buffer = buffer,
        .workspace = workspace,
        .psum_num_recv_tokens_per_scaleup_rank =
            psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert =
            psum_num_recv_tokens_per_expert,
        .recv_x = recv_x,
        .recv_sf = nullptr,
        .recv_topk_idx = recv_topk_idx,
        .recv_topk_weights = recv_topk_weights,
        .recv_src_metadata = recv_src_metadata,
        .channel_linked_list = channel_linked_list,
        .num_unaligned_recv_tokens_per_expert = nullptr,
        .num_recv_tokens = num_recv_tokens,
        .recv_sf_token_stride = 0,
        .recv_sf_hidden_stride = 0,
        .scaleout_rank_idx = scaleout_rank_idx,
        .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = prepared.launch_args,
    };
    DispatchCopyEpilogueRuntime::launch(prepared.runtime, args, stream);
}

// Private C080 force/test-only wrapper around the unchanged legacy epilogue.
// Keeping both the generated code and the JIT key identical lets this path
// share the production cache entry without changing the public combine path.
struct PreparedRailBalanceHybridCombineEpilogue {
    std::shared_ptr<jit::KernelRuntime> runtime;
    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int hidden;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_topk;
    jit::LaunchArgs launch_args;
};

static PreparedRailBalanceHybridCombineEpilogue
prepare_rail_balance_hybrid_combine_epilogue(
        const int& hidden,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_topk,
        const int& num_scaleout_ranks,
        const int& num_scaleup_ranks) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    EP_HOST_ASSERT(
        static_cast<int64_t>(hidden) * sizeof(nv_bfloat16) <= INT_MAX);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_scaleout_ranks >= 2 and num_scaleout_ranks <= 32);
    EP_HOST_ASSERT(num_scaleup_ranks >= 2 and num_scaleup_ranks <= 32);
    EP_HOST_ASSERT(num_experts > 0 and
                   num_experts %
                       (num_scaleout_ranks * num_scaleup_ranks) == 0);

    const int num_sms = jit::device_runtime->get_num_sms();
    const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
    const auto output_token_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, 0, false);
    const int num_warps = std::min<int>(
        num_smem_bytes / output_token_layout.get_num_bytes<false>(), 32);
    EP_HOST_ASSERT(num_sms > 0 and num_warps > 0);

    // This is deliberately byte-for-byte the launch specialization selected by
    // launch_combine_reduce_epilogue for the C080 non-expanded,
    // multiple-reduction path, including PDL.
    const auto launch_args = jit::LaunchArgs(
        num_sms, num_warps * 32, num_smem_bytes, 1, false, true);
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = false,
        .allow_multiple_reduction = true,
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .combined_x = nullptr,
        .combined_topk_weights = nullptr,
        .combined_topk_idx = nullptr,
        .reduce_buffer = nullptr,
        .bias_0 = nullptr,
        .bias_1 = nullptr,
        .num_combined_tokens = 0,
        .scaleout_rank_idx = 0,
        .scaleup_rank_idx = 0,
        .launch_args = launch_args,
    };
    return {
        .runtime = jit::compiler->build(
            "combine_reduce_epilogue",
            CombineReduceEpilogueRuntime::generate(args)),
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .launch_args = launch_args,
    };
}

static void launch_prepared_rail_balance_hybrid_combine_epilogue(
        const PreparedRailBalanceHybridCombineEpilogue& prepared,
        void* combined_x,
        float* combined_topk_weights,
        topk_idx_t* combined_topk_idx,
        void* reduce_buffer,
        const int& num_combined_tokens,
        const int& scaleout_rank_idx,
        const int& scaleup_rank_idx,
        const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(prepared.runtime != nullptr);
    EP_HOST_ASSERT(num_combined_tokens >= 0 and
                   num_combined_tokens <=
                       prepared.num_max_tokens_per_rank);
    EP_HOST_ASSERT(reduce_buffer != nullptr);
    EP_HOST_ASSERT(num_combined_tokens == 0 or
                   (combined_x != nullptr and
                    combined_topk_weights != nullptr and
                    combined_topk_idx != nullptr));
    EP_HOST_ASSERT(scaleout_rank_idx >= 0 and
                   scaleout_rank_idx < prepared.num_scaleout_ranks);
    EP_HOST_ASSERT(scaleup_rank_idx >= 0 and
                   scaleup_rank_idx < prepared.num_scaleup_ranks);

    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = false,
        .allow_multiple_reduction = true,
        .num_scaleout_ranks = prepared.num_scaleout_ranks,
        .num_scaleup_ranks = prepared.num_scaleup_ranks,
        .hidden = prepared.hidden,
        .num_max_tokens_per_rank = prepared.num_max_tokens_per_rank,
        .num_experts = prepared.num_experts,
        .num_topk = prepared.num_topk,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .reduce_buffer = reduce_buffer,
        .bias_0 = nullptr,
        .bias_1 = nullptr,
        .num_combined_tokens = num_combined_tokens,
        .scaleout_rank_idx = scaleout_rank_idx,
        .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = prepared.launch_args,
    };
    CombineReduceEpilogueRuntime::launch(
        prepared.runtime, args, stream);
}

}  // namespace deep_ep::elastic
