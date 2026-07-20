#pragma once

#include <climits>

#include <deep_ep/common/layout.cuh>

#include "combine.hpp"

namespace deep_ep::elastic {

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
