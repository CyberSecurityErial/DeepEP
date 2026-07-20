#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "rail_balance_hybrid_dispatch.hpp"

namespace deep_ep::elastic {

struct RailBalanceHybridSourceShuffleSpec {
    int hidden;
    int num_topk;
};

class RailBalanceHybridSourceShuffleRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridSourceShuffleRuntime> {
public:
    struct Args {
        int hidden;
        int num_topk;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        const void* x;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        void* arena;
        const int* owner_channel_prefix;
        const int* keep_count;
        const int* segments;
        const int* num_segments;
        const int* retained;
        const int* moved_channel_prefix;
        const int* group_prefix;
        const int* proxy_required;
        int* status;
        int num_tokens;
        int num_experts;
        int num_destinations;
        int local_destination;
        int num_rails;
        int owner;
        int num_channels;
        int num_max_tokens_per_rank;
        int rank_idx;
        int proxy_capacity;
        jit::LaunchArgs launch_args;
    };

    template <typename Spec>
    static std::string generate_impl(const Spec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_shuffle.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_source_shuffle_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.x, args.topk_idx, args.topk_weights,
            args.arena, args.owner_channel_prefix,
            args.keep_count, args.segments, args.num_segments,
            args.retained, args.moved_channel_prefix,
            args.group_prefix, args.proxy_required, args.status,
            args.num_tokens, args.num_experts,
            args.num_destinations, args.local_destination,
            args.num_rails, args.owner, args.num_channels,
            args.num_max_tokens_per_rank, args.rank_idx,
            args.proxy_capacity));
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_source_shuffle(
    const int& hidden,
    const int& num_topk) {
    const RailBalanceHybridSourceShuffleSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    return jit::compiler->build(
        "rail_balance_hybrid_source_shuffle",
        RailBalanceHybridSourceShuffleRuntime::generate(spec));
}

static void launch_prepared_rail_balance_hybrid_source_shuffle(
    const std::shared_ptr<jit::KernelRuntime>& runtime,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    const void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    void* arena,
    const RailBalanceHybridPlanOutputs& plan,
    const int& hidden,
    const int& num_topk,
    const int& num_tokens,
    const int& num_experts,
    const int& num_destinations,
    const int& local_destination,
    const int& num_rails,
    const int& owner,
    const int& num_channels,
    const int& num_max_tokens_per_rank,
    const int& rank_idx,
    const int& proxy_capacity,
    const at::cuda::CUDAStream& stream) {
    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(__nv_bfloat16), 0, num_topk, true);
    const int num_smem_bytes = token_layout.get_num_bytes<false>() +
        ptx::kNumTMAAlignBytes;
    const RailBalanceHybridSourceShuffleRuntime::Args args = {
        .hidden = hidden,
        .num_topk = num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .arena = arena,
        .owner_channel_prefix = plan.owner_channel_prefix.data_ptr<int>(),
        .keep_count = plan.keep_count.data_ptr<int>(),
        .segments = plan.segments.data_ptr<int>(),
        .num_segments = plan.num_segments.data_ptr<int>(),
        .retained = plan.retained.data_ptr<int>(),
        .moved_channel_prefix = plan.moved_channel_prefix.data_ptr<int>(),
        .group_prefix = plan.group_prefix.data_ptr<int>(),
        .proxy_required = plan.proxy_required.data_ptr<int>(),
        .status = plan.status.data_ptr<int>(),
        .num_tokens = num_tokens,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .local_destination = local_destination,
        .num_rails = num_rails,
        .owner = owner,
        .num_channels = num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .rank_idx = rank_idx,
        .proxy_capacity = proxy_capacity,
        .launch_args = jit::LaunchArgs(
            num_channels, 32, num_smem_bytes),
    };
    RailBalanceHybridSourceShuffleRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
