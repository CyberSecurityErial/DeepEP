#pragma once

#include <tuple>
#include <vector>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/common/rail_balance_vnode_layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "rail_balance_hybrid_dispatch.hpp"
#include "rail_balance_vnode.hpp"

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

// Private C080-D result ABI.  The first eleven tensors are owning CUDA
// snapshots from the world buffer; the nested tuple is the independently
// rebuilt production-shaped plan.  No pointer into either symmetric window is
// allowed to escape this entry point.
using RailBalanceHybridVNodeTensors = std::tuple<
    torch::Tensor,  // proxy_return       [Pcap, combine_stride]
    torch::Tensor,  // reduce_seed        [min(D, K) * M, combine_stride]
    torch::Tensor,  // compact_quota      [G, D - 1]
    torch::Tensor,  // rail_records       [P * (K + 1), vnode_record_bytes]
    torch::Tensor,  // rail_ready         [P * (K + 1)]
    torch::Tensor,  // rail_routes        [P * (K + 1), sizeof(VNodeRoute)]
    torch::Tensor,  // expert_records     [G * M * K, vnode_record_bytes]
    torch::Tensor,  // expert_ready       [G * M * K]
    torch::Tensor,  // expert_routes      [G * M * K, sizeof(VNodeRoute)]
    torch::Tensor,  // stage_status       [6, status_stride]
    torch::Tensor,  // arena_guard        [4096]
    RailBalanceHybridPlanTensors>;

using RailBalanceHybridVNodePrepareTensors = std::tuple<
    int,                          // device plan status
    RailBalanceHybridPlanTensors, // owning world plan references
    torch::Tensor>;               // compact quota [G, D - 1]

// Every allocation and every JIT object needed by the fixed B0..B5 sequence
// is owned here before prepare returns.  finish_attempted is sticky: once a
// world barrier may have been entered, replaying the same transaction is
// forbidden even when CUDA reports an exception.
struct RailBalanceHybridVNodePending {
    int invocation_id;
    bool finish_attempted;
    bool finished;
    int plan_status;
    int num_tokens;
    int hidden;
    int num_topk;
    int num_experts;
    int num_destinations;
    int num_rails;
    int num_channels;
    int num_max_tokens_per_rank;
    int proxy_capacity;
    int generation;
    int physical_capacity;
    int rail_capacity;
    int expert_capacity;
    int status_stride;
    int64_t arena_offset;
    int64_t arena_bytes;
    void* arena;

    torch::Tensor x;
    torch::Tensor topk_idx;
    torch::Tensor topk_weights;
    torch::Tensor proxy_dispatch;
    RailBalanceHybridPlanOutputs plan;
    torch::Tensor compact_quota;
    torch::Tensor proxy_return;
    torch::Tensor reduce_seed;
    torch::Tensor rail_records;
    torch::Tensor rail_ready;
    torch::Tensor rail_routes;
    torch::Tensor expert_records;
    torch::Tensor expert_ready;
    torch::Tensor expert_routes;
    torch::Tensor stage_status;
    torch::Tensor arena_guard;
    std::vector<int> host_stage_status;
    bool host_stage_status_ready;

    PreparedRailBalanceHybridPlan prepared_plan;
    PreparedRailBalanceHybridVNode prepared_adapter;
    PreparedRailBalanceVNode prepared_vnode;
    std::shared_ptr<jit::KernelRuntime> prepared_world_barrier;

    RailBalanceHybridVNodeTensors as_tuple() const {
        return {
            proxy_return, reduce_seed, compact_quota,
            rail_records, rail_ready, rail_routes,
            expert_records, expert_ready, expert_routes,
            stage_status, arena_guard, plan.as_tuple(),
        };
    }
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
