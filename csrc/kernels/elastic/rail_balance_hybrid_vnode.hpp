#pragma once

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/common/rail_balance_vnode_layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "rail_balance_hybrid_dispatch.hpp"

namespace deep_ep::elastic {

struct RailBalanceHybridVNodeSpec {
    int hidden;
    int num_topk;
};

class RailBalanceHybridPackVNodeBaseRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridPackVNodeBaseRuntime> {
public:
    struct Args {
        const void* x;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        const void* proxy_dispatch;
        void* vnode_arena;
        const int* channel_count;
        const int* quota;
        const int* keep_count;
        const int* segments;
        const int* num_segments;
        const int* owner_channel_prefix;
        const int* retained;
        const int* moved;
        const int* moved_channel_prefix;
        const int* group_prefix;
        const int* proxy_required;
        int* status;
        int num_tokens;
        int num_experts;
        int num_destinations;
        int num_rails;
        int egress;
        int num_channels;
        int num_max_tokens_per_rank;
        int proxy_capacity;
        int generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(
            const RailBalanceHybridVNodeSpec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_vnode.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_pack_vnode_base_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x, args.topk_idx, args.topk_weights,
            args.proxy_dispatch, args.vnode_arena,
            args.channel_count, args.quota, args.keep_count,
            args.segments, args.num_segments,
            args.owner_channel_prefix, args.retained, args.moved,
            args.moved_channel_prefix, args.group_prefix,
            args.proxy_required, args.status,
            args.num_tokens, args.num_experts, args.num_destinations,
            args.num_rails, args.egress, args.num_channels,
            args.num_max_tokens_per_rank, args.proxy_capacity,
            args.generation));
    }
};

class RailBalanceHybridReturnDemuxRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridReturnDemuxRuntime> {
public:
    struct Args {
        const void* vnode_arena;
        const void* proxy_dispatch;
        void* reduce_seed;
        void* proxy_return;
        const int* quota;
        const int* keep_count;
        const int* segments;
        const int* num_segments;
        const int* owner_channel_prefix;
        const int* retained;
        const int* moved;
        const int* moved_channel_prefix;
        const int* group_prefix;
        const int* proxy_required;
        int* status;
        int num_experts;
        int num_destinations;
        int num_rails;
        int egress;
        int num_channels;
        int num_max_tokens_per_rank;
        int proxy_capacity;
        int generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(
            const RailBalanceHybridVNodeSpec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_vnode.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_return_demux_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.vnode_arena, args.proxy_dispatch,
            args.reduce_seed, args.proxy_return,
            args.quota, args.keep_count, args.segments,
            args.num_segments, args.owner_channel_prefix,
            args.retained, args.moved, args.moved_channel_prefix,
            args.group_prefix, args.proxy_required, args.status,
            args.num_experts, args.num_destinations,
            args.num_rails, args.egress, args.num_channels,
            args.num_max_tokens_per_rank, args.proxy_capacity,
            args.generation));
    }
};

struct PreparedRailBalanceHybridVNode {
    std::shared_ptr<jit::KernelRuntime> pack;
    std::shared_ptr<jit::KernelRuntime> demux;
};

static void validate_rail_balance_hybrid_vnode_layout(
        const int& hidden,
        const int& num_topk,
        const int& num_destinations,
        const int& num_rails,
        const int& num_channels,
        const int& num_max_tokens_per_rank,
        const int& proxy_capacity) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_destinations >= 2 and
                   num_destinations <=
                       rail_balance::kNumHybridMaxDestinations);
    EP_HOST_ASSERT(num_rails >= 2 and num_rails <= 32);
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and proxy_capacity > 0);

    const auto hybrid_layout = rail_balance::HybridArenaLayout(
        hidden, num_topk, proxy_capacity);
    const auto vnode_layout = rail_balance::VNodeRoundTripLayout(
        hidden * sizeof(nv_bfloat16), num_topk,
        num_destinations - 1, num_max_tokens_per_rank,
        num_rails, num_max_tokens_per_rank);
    const auto dispatch_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, num_topk, true);
    const auto combine_layout = layout::TokenLayout(
        hidden * sizeof(nv_bfloat16), 0, num_topk, false);
    EP_HOST_ASSERT((
        hybrid_layout.dispatch_token_bytes ==
        dispatch_layout.get_num_bytes<false, int64_t>()));
    EP_HOST_ASSERT((
        hybrid_layout.combine_token_bytes ==
        combine_layout.get_num_bytes<false, int64_t>()));
    EP_HOST_ASSERT(
        vnode_layout.rail.records.token_bytes ==
        hybrid_layout.dispatch_token_bytes);
    EP_HOST_ASSERT(
        hybrid_layout.dispatch_token_bytes % sizeof(int4) == 0 and
        hybrid_layout.combine_token_bytes % sizeof(int4) == 0 and
        vnode_layout.rail.records.record_bytes % sizeof(int4) == 0);
}

static PreparedRailBalanceHybridVNode
prepare_rail_balance_hybrid_vnode(
        const int& hidden,
        const int& num_topk) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    const RailBalanceHybridVNodeSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    return {
        .pack = jit::compiler->build(
            "rail_balance_hybrid_pack_vnode_base",
            RailBalanceHybridPackVNodeBaseRuntime::generate(spec)),
        .demux = jit::compiler->build(
            "rail_balance_hybrid_return_demux",
            RailBalanceHybridReturnDemuxRuntime::generate(spec)),
    };
}

static void launch_prepared_rail_balance_hybrid_pack_vnode_base(
        const PreparedRailBalanceHybridVNode& prepared,
        const void* x,
        const topk_idx_t* topk_idx,
        const float* topk_weights,
        const void* proxy_dispatch,
        void* vnode_arena,
        const RailBalanceHybridPlanOutputs& plan,
        int* status,
        const int& num_tokens,
        const int& hidden,
        const int& num_topk,
        const int& num_experts,
        const int& num_destinations,
        const int& num_rails,
        const int& egress,
        const int& num_channels,
        const int& num_max_tokens_per_rank,
        const int& proxy_capacity,
        const int& generation,
        const at::cuda::CUDAStream& stream) {
    validate_rail_balance_hybrid_vnode_layout(
        hidden, num_topk, num_destinations, num_rails,
        num_channels, num_max_tokens_per_rank, proxy_capacity);
    EP_HOST_ASSERT(num_tokens >= 0 and
                   num_tokens <= num_max_tokens_per_rank);
    EP_HOST_ASSERT(egress >= 0 and egress < num_rails);
    EP_HOST_ASSERT(generation > 0);
    const RailBalanceHybridPackVNodeBaseRuntime::Args args = {
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .proxy_dispatch = proxy_dispatch,
        .vnode_arena = vnode_arena,
        .channel_count = plan.channel_count.data_ptr<int>(),
        .quota = plan.quota.data_ptr<int>(),
        .keep_count = plan.keep_count.data_ptr<int>(),
        .segments = plan.segments.data_ptr<int>(),
        .num_segments = plan.num_segments.data_ptr<int>(),
        .owner_channel_prefix = plan.owner_channel_prefix.data_ptr<int>(),
        .retained = plan.retained.data_ptr<int>(),
        .moved = plan.moved.data_ptr<int>(),
        .moved_channel_prefix = plan.moved_channel_prefix.data_ptr<int>(),
        .group_prefix = plan.group_prefix.data_ptr<int>(),
        .proxy_required = plan.proxy_required.data_ptr<int>(),
        .status = status,
        .num_tokens = num_tokens,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .num_rails = num_rails,
        .egress = egress,
        .num_channels = num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .proxy_capacity = proxy_capacity,
        .generation = generation,
        .launch_args = jit::LaunchArgs(num_channels, 32),
    };
    RailBalanceHybridPackVNodeBaseRuntime::launch(
        prepared.pack, args, stream);
}

static void launch_prepared_rail_balance_hybrid_return_demux(
        const PreparedRailBalanceHybridVNode& prepared,
        const void* vnode_arena,
        const void* proxy_dispatch,
        void* reduce_seed,
        void* proxy_return,
        const RailBalanceHybridPlanOutputs& plan,
        int* status,
        const int& hidden,
        const int& num_topk,
        const int& num_experts,
        const int& num_destinations,
        const int& num_rails,
        const int& egress,
        const int& num_channels,
        const int& num_max_tokens_per_rank,
        const int& proxy_capacity,
        const int& generation,
        const at::cuda::CUDAStream& stream) {
    validate_rail_balance_hybrid_vnode_layout(
        hidden, num_topk, num_destinations, num_rails,
        num_channels, num_max_tokens_per_rank, proxy_capacity);
    EP_HOST_ASSERT(egress >= 0 and egress < num_rails);
    EP_HOST_ASSERT(generation > 0);
    const RailBalanceHybridReturnDemuxRuntime::Args args = {
        .vnode_arena = vnode_arena,
        .proxy_dispatch = proxy_dispatch,
        .reduce_seed = reduce_seed,
        .proxy_return = proxy_return,
        .quota = plan.quota.data_ptr<int>(),
        .keep_count = plan.keep_count.data_ptr<int>(),
        .segments = plan.segments.data_ptr<int>(),
        .num_segments = plan.num_segments.data_ptr<int>(),
        .owner_channel_prefix = plan.owner_channel_prefix.data_ptr<int>(),
        .retained = plan.retained.data_ptr<int>(),
        .moved = plan.moved.data_ptr<int>(),
        .moved_channel_prefix = plan.moved_channel_prefix.data_ptr<int>(),
        .group_prefix = plan.group_prefix.data_ptr<int>(),
        .proxy_required = plan.proxy_required.data_ptr<int>(),
        .status = status,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .num_rails = num_rails,
        .egress = egress,
        .num_channels = num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .proxy_capacity = proxy_capacity,
        .generation = generation,
        .launch_args = jit::LaunchArgs(
            (num_destinations - 1) * num_max_tokens_per_rank,
            32),
    };
    RailBalanceHybridReturnDemuxRuntime::launch(
        prepared.demux, args, stream);
}

}  // namespace deep_ep::elastic
