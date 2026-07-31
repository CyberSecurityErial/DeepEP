#pragma once

#include <array>
#include <atomic>
#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <nccl.h>
#include <nccl_device.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

struct PreparedRailBalanceHybridCombineEpilogue;
struct PreparedRailBalanceHybridCombine;
struct PreparedRailBalanceHybridDispatchEpilogue;
struct PreparedRailBalanceHybridSourceShuffle;
struct PreparedRailBalanceHybridReturnUnshuffle;

// Private C080-B1 oracle ABI.  All schedule tensors are CUDA int32 tensors;
// the input keeps the compile-time topk_idx_t selected by EP_NUM_TOPK_IDX_BITS.
using RailBalanceHybridPlanTensors = std::tuple<
    torch::Tensor,  // channel_count          [G, C, D]
    torch::Tensor,  // count                  [G, D]
    torch::Tensor,  // quota                  [G, D]
    torch::Tensor,  // keep_count             [G, D]
    torch::Tensor,  // segments               [D, G - 1, 5]
    torch::Tensor,  // num_segments           [D]
    torch::Tensor,  // owner_channel_prefix   [G, C, D]
    torch::Tensor,  // retained               [G, C, D]
    torch::Tensor,  // moved                  [G, C, D]
    torch::Tensor,  // moved_channel_prefix   [G, D, C + 1]
    torch::Tensor,  // group_prefix           [G, C, D]
    torch::Tensor,  // proxy_required         [G]
    torch::Tensor,  // moved_copies           [1]
    torch::Tensor>; // status                 [1]

using RailBalanceHopRecordTensors = std::tuple<
    torch::Tensor,  // records [G, N, K], int64 HopCopyRecord storage
    torch::Tensor>; // status  [1]

using RailBalanceHopPlanTensors = std::tuple<
    torch::Tensor,  // resolutions [G, N, K, 4]
    torch::Tensor,  // pair_load [D, G]
    torch::Tensor,  // source_load [G]
    torch::Tensor,  // retained [G, C, D]
    torch::Tensor,  // moved [G, C, D]
    torch::Tensor,  // group_prefix [G, C, D]
    torch::Tensor,  // proxy_required [G]
    torch::Tensor,  // path_units [4]
    torch::Tensor,  // moved_copies [1]
    torch::Tensor>; // status [1]

// KernelRuntime owns CUmodule handles for the CUDA context in which the JIT
// cubin was loaded. DeepEP binds one process to one GPU, so reject accidental
// same-process cross-device reuse before it can become INVALID_HANDLE.
static std::atomic<int> rail_balance_hybrid_plan_process_device{-1};

class RailBalanceHybridCountRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridCountRuntime> {
public:
    struct Args {
        const topk_idx_t* topk_idx;
        int* channel_count;
        int* status;
        int num_owners;
        int num_tokens;
        int num_topk;
        int num_channels;
        int num_experts;
        int num_destinations;
        int local_destination;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_count_impl<0>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.topk_idx, args.channel_count, args.status,
            args.num_owners, args.num_tokens,
            args.num_topk, args.num_channels,
            args.num_experts, args.num_destinations,
            args.local_destination));
    }
};

class RailBalanceHopRecordRuntime final:
    public jit::LaunchRuntime<RailBalanceHopRecordRuntime> {
public:
    struct Args {
        const topk_idx_t* topk_idx;
        rail_balance::HopCopyRecord* records;
        int* status;
        int num_owners;
        int num_tokens;
        int record_token_capacity;
        int num_topk;
        int num_channels;
        int num_experts;
        int num_destinations;
        int num_rails;
        int local_destination;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance::rail_balance_hop_record_impl<0>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.topk_idx, args.records, args.status,
            args.num_owners, args.num_tokens, args.record_token_capacity,
            args.num_topk,
            args.num_channels, args.num_experts, args.num_destinations,
            args.num_rails, args.local_destination));
    }
};

class RailBalanceHopPlanRuntime final:
    public jit::LaunchRuntime<RailBalanceHopPlanRuntime> {
public:
    struct Args {
        const rail_balance::HopCopyRecord* records;
        rail_balance::HopCopyResolution* resolutions;
        int* pair_load;
        int* source_load;
        int* owner_remaining;
        int* endpoint_count;
        int* endpoint_owner_quota;
        int* owner_group_cursor;
        int* retained;
        int* moved;
        int* retained_prefix;
        int* group_prefix;
        int* proxy_required;
        int* path_units;
        int* moved_copies;
        int* status;
        int num_rails;
        int num_tokens;
        int num_topk;
        int num_channels;
        int num_destinations;
        int num_max_tokens_per_rank;
        int proxy_capacity_per_egress;
        int planner_seed;
        int planner_chunk_size;
        int two_hop_threshold_percent;
        int max_two_hop_percent;
        int hop_penalty_percent;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance::rail_balance_hop_plan_impl<0>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.records, args.resolutions,
            args.pair_load, args.source_load, args.owner_remaining,
            args.endpoint_count, args.endpoint_owner_quota,
            args.owner_group_cursor,
            args.retained, args.moved, args.retained_prefix,
            args.group_prefix,
            args.proxy_required, args.path_units, args.moved_copies,
            args.status, args.num_rails, args.num_tokens, args.num_topk,
            args.num_channels, args.num_destinations,
            args.num_max_tokens_per_rank,
            args.proxy_capacity_per_egress, args.planner_seed,
            args.planner_chunk_size,
            args.two_hop_threshold_percent, args.max_two_hop_percent,
            args.hop_penalty_percent));
    }
};

struct PreparedRailBalanceHopPlan {
    std::shared_ptr<jit::KernelRuntime> record;
    std::shared_ptr<jit::KernelRuntime> plan;
    jit::LaunchArgs record_launch_args;
    jit::LaunchArgs plan_launch_args;
};

static PreparedRailBalanceHopPlan prepare_rail_balance_hop_plan(
    const int& num_channels) {
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    const RailBalanceHopRecordRuntime::Args record_args = {
        .topk_idx = nullptr,
        .records = nullptr,
        .status = nullptr,
        .num_owners = 0,
        .num_tokens = 0,
        .record_token_capacity = 0,
        .num_topk = 0,
        .num_channels = 0,
        .num_experts = 0,
        .num_destinations = 0,
        .num_rails = 0,
        .local_destination = 0,
        .launch_args = jit::LaunchArgs(num_channels, 32),
    };
    const RailBalanceHopPlanRuntime::Args plan_args = {
        .records = nullptr,
        .resolutions = nullptr,
        .pair_load = nullptr,
        .source_load = nullptr,
        .owner_remaining = nullptr,
        .endpoint_count = nullptr,
        .endpoint_owner_quota = nullptr,
        .owner_group_cursor = nullptr,
        .retained = nullptr,
        .moved = nullptr,
        .retained_prefix = nullptr,
        .group_prefix = nullptr,
        .proxy_required = nullptr,
        .path_units = nullptr,
        .moved_copies = nullptr,
        .status = nullptr,
        .num_rails = 0,
        .num_tokens = 0,
        .num_topk = 0,
        .num_channels = 0,
        .num_destinations = 0,
        .num_max_tokens_per_rank = 0,
        .proxy_capacity_per_egress = 0,
        .planner_seed = 0,
        .planner_chunk_size = 1,
        .two_hop_threshold_percent = 0,
        .max_two_hop_percent = 0,
        .hop_penalty_percent = 0,
        .launch_args = jit::LaunchArgs(1, 32),
    };
    return {
        .record = jit::compiler->build(
            "rail_balance_hop_record_v1",
            RailBalanceHopRecordRuntime::generate(record_args)),
        .plan = jit::compiler->build(
            "rail_balance_hop_plan_v2",
            RailBalanceHopPlanRuntime::generate(plan_args)),
        .record_launch_args = record_args.launch_args,
        .plan_launch_args = plan_args.launch_args,
    };
}

static void launch_prepared_rail_balance_hop_record(
    const PreparedRailBalanceHopPlan& prepared,
    const topk_idx_t* topk_idx,
    rail_balance::HopCopyRecord* records,
    int* status,
    const int& num_tokens,
    const int& record_token_capacity,
    const int& num_topk,
    const int& num_channels,
    const int& num_experts,
    const int& num_destinations,
    const int& num_rails,
    const int& local_destination,
    const at::cuda::CUDAStream& stream) {
    RailBalanceHopRecordRuntime::launch(
        prepared.record,
        RailBalanceHopRecordRuntime::Args{
            .topk_idx = topk_idx,
            .records = records,
            .status = status,
            .num_owners = 1,
            .num_tokens = num_tokens,
            .record_token_capacity = record_token_capacity,
            .num_topk = num_topk,
            .num_channels = num_channels,
            .num_experts = num_experts,
            .num_destinations = num_destinations,
            .num_rails = num_rails,
            .local_destination = local_destination,
            .launch_args = prepared.record_launch_args,
        },
        stream);
}

static void launch_prepared_rail_balance_hop_plan(
    const PreparedRailBalanceHopPlan& prepared,
    const rail_balance::HopCopyRecord* records,
    rail_balance::HopCopyResolution* resolutions,
    int* pair_load,
    int* source_load,
    int* owner_remaining,
    int* endpoint_count,
    int* endpoint_owner_quota,
    int* owner_group_cursor,
    int* retained,
    int* moved,
    int* retained_prefix,
    int* group_prefix,
    int* proxy_required,
    int* path_units,
    int* moved_copies,
    int* status,
    const int& num_rails,
    const int& record_token_capacity,
    const int& num_topk,
    const int& num_channels,
    const int& num_destinations,
    const int& num_max_tokens_per_rank,
    const int& proxy_capacity_per_egress,
    const int& planner_seed,
    const int& planner_chunk_size,
    const int& two_hop_threshold_percent,
    const int& max_two_hop_percent,
    const int& hop_penalty_percent,
    const at::cuda::CUDAStream& stream) {
    RailBalanceHopPlanRuntime::launch(
        prepared.plan,
        RailBalanceHopPlanRuntime::Args{
            .records = records,
            .resolutions = resolutions,
            .pair_load = pair_load,
            .source_load = source_load,
            .owner_remaining = owner_remaining,
            .endpoint_count = endpoint_count,
            .endpoint_owner_quota = endpoint_owner_quota,
            .owner_group_cursor = owner_group_cursor,
            .retained = retained,
            .moved = moved,
            .retained_prefix = retained_prefix,
            .group_prefix = group_prefix,
            .proxy_required = proxy_required,
            .path_units = path_units,
            .moved_copies = moved_copies,
            .status = status,
            .num_rails = num_rails,
            .num_tokens = record_token_capacity,
            .num_topk = num_topk,
            .num_channels = num_channels,
            .num_destinations = num_destinations,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .proxy_capacity_per_egress = proxy_capacity_per_egress,
            .planner_seed = planner_seed,
            .planner_chunk_size = planner_chunk_size,
            .two_hop_threshold_percent = two_hop_threshold_percent,
            .max_two_hop_percent = max_two_hop_percent,
            .hop_penalty_percent = hop_penalty_percent,
            .launch_args = prepared.plan_launch_args,
        },
        stream);
}

class RailBalanceHybridPlanRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridPlanRuntime> {
public:
    struct Args {
        const int* channel_count;
        int* count;
        int* quota;
        int* keep_count;
        int* segments;
        int* num_segments;
        int num_rails;
        int num_channels;
        int num_destinations;
        int remainder_seed;
        int policy;
        int threshold_percent;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_plan_v2_impl<0>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.channel_count, args.count, args.quota,
            args.keep_count, args.segments, args.num_segments,
            args.num_rails, args.num_channels,
            args.num_destinations, args.remainder_seed,
            args.policy, args.threshold_percent));
    }
};

class RailBalanceHybridPrefixRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridPrefixRuntime> {
public:
    struct Args {
        const int* channel_count;
        const int* quota;
        const int* keep_count;
        int* owner_channel_prefix;
        int* retained;
        int* moved;
        int* moved_channel_prefix;
        int* group_prefix;
        int* proxy_required;
        int* moved_copies;
        int* status;
        int num_rails;
        int num_channels;
        int num_destinations;
        int num_max_tokens_per_rank;
        int proxy_capacity_per_egress;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_prefix_impl<0>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.channel_count, args.quota, args.keep_count,
            args.owner_channel_prefix, args.retained, args.moved,
            args.moved_channel_prefix, args.group_prefix,
            args.proxy_required, args.moved_copies, args.status,
            args.num_rails, args.num_channels, args.num_destinations,
            args.num_max_tokens_per_rank,
            args.proxy_capacity_per_egress));
    }
};

// Isolated C080-E runtime.  It deliberately has a distinct type, include,
// kernel symbol, and cache key from DispatchRuntime, so merely compiling the
// force specialization cannot change the legacy dispatch JIT identity.
struct RailBalanceHybridDispatchSpec {
    int num_sms;
    int num_notify_warps;
    int num_scaleout_warps;
    int num_forward_warps;
    int num_scaleout_ranks;
    int num_scaleup_ranks;
    int num_hidden_bytes;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_topk;
    int expert_alignment;
    int num_qps;
    int64_t num_timeout_cycles;
    int num_smem_bytes;
    int proxy_capacity;
};

class RailBalanceHybridDispatchRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridDispatchRuntime> {
public:
    struct Args {
        // Compile-time specialization.
        RailBalanceHybridDispatchSpec spec;

        // Legacy Hybrid dispatch ABI.
        void* x;
        sf_pack_t* sf;
        topk_idx_t* topk_idx;
        float* topk_weights;
        topk_idx_t* copied_topk_idx;
        int* cumulative_local_expert_recv_stats;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        int* num_unaligned_recv_tokens_per_expert;
        int* dst_buffer_slot_idx;
        int* token_metadata_at_forward;
        int num_tokens;
        int sf_token_stride;
        int sf_hidden_stride;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        void* mapped_host_workspace;

        // Force-only compact schedule and payload arena.
        void* rail_balance_arena;
        const int* rail_balance_retained;
        const int* rail_balance_moved;
        const int* rail_balance_group_prefix;
        const int* rail_balance_proxy_required;

        int scaleout_rank_idx;
        int scaleup_rank_idx;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        const auto& s = args.spec;
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_dispatch_impl<
            true, false, {}, {}, {}, {}, {}, {}, {}, 0, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
            s.num_sms,
            s.num_notify_warps,
            s.num_scaleout_warps,
            s.num_forward_warps,
            s.num_scaleout_ranks,
            s.num_scaleup_ranks,
            s.num_hidden_bytes,
            s.num_max_tokens_per_rank,
            s.num_experts,
            s.num_topk,
            s.expert_alignment,
            s.num_qps,
            s.num_timeout_cycles,
            s.proxy_capacity);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.x, args.sf, args.topk_idx, args.topk_weights,
            args.copied_topk_idx,
            args.cumulative_local_expert_recv_stats,
            args.psum_num_recv_tokens_per_scaleup_rank,
            args.psum_num_recv_tokens_per_expert,
            args.num_unaligned_recv_tokens_per_expert,
            args.dst_buffer_slot_idx,
            args.token_metadata_at_forward,
            args.num_tokens,
            args.sf_token_stride, args.sf_hidden_stride,
            args.nccl_dev_comm, args.nccl_window,
            args.buffer,
            args.workspace, args.mapped_host_workspace,
            args.rail_balance_arena,
            args.rail_balance_retained,
            args.rail_balance_moved,
            args.rail_balance_group_prefix,
            args.rail_balance_proxy_required,
            args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void validate_rail_balance_hybrid_dispatch_spec(
    const RailBalanceHybridDispatchSpec& spec) {
    EP_HOST_ASSERT(spec.num_sms >= 2);
    EP_HOST_ASSERT(spec.num_notify_warps > 0 and
                   spec.num_notify_warps % 4 == 0);
    EP_HOST_ASSERT(spec.num_scaleout_warps > 0 and
                   spec.num_scaleout_warps == spec.num_forward_warps);
    EP_HOST_ASSERT(spec.num_scaleout_ranks >= 2 and
                   spec.num_scaleout_ranks <= 32);
    EP_HOST_ASSERT(spec.num_scaleup_ranks >= 2 and
                   spec.num_scaleup_ranks <= 32);
    EP_HOST_ASSERT(
        spec.num_hidden_bytes > 0 and
        spec.num_hidden_bytes %
            (256 * static_cast<int>(sizeof(nv_bfloat16))) == 0 and
        spec.num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
    EP_HOST_ASSERT(spec.num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(spec.num_experts > 0 and
                   spec.num_experts %
                       (spec.num_scaleout_ranks * spec.num_scaleup_ranks) == 0);
    const int num_ranks =
        spec.num_scaleout_ranks * spec.num_scaleup_ranks;
    EP_HOST_ASSERT(
        num_ranks <= layout::WorkspaceLayout::kNumMaxRanks);
    EP_HOST_ASSERT(
        spec.num_experts <= layout::WorkspaceLayout::kNumMaxExperts);
    EP_HOST_ASSERT(
        spec.num_experts / num_ranks <=
            layout::WorkspaceLayout::kNumMaxExpertsPerRank);
    EP_HOST_ASSERT(spec.num_topk >= 1 and spec.num_topk <= 32);
    EP_HOST_ASSERT(spec.expert_alignment == 1);
    EP_HOST_ASSERT(spec.num_qps > 0);
    EP_HOST_ASSERT(spec.num_timeout_cycles > 0);
    EP_HOST_ASSERT(spec.num_smem_bytes > 0);
    EP_HOST_ASSERT(spec.proxy_capacity > 0);
    EP_HOST_ASSERT(
        spec.num_sms * spec.num_scaleout_warps <=
        rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(
        (spec.num_notify_warps + spec.num_scaleout_warps +
         spec.num_forward_warps) * 32 <= 1024);
    const auto token_layout = layout::TokenLayout(
        spec.num_hidden_bytes, 0, spec.num_topk, true);
    const int notify_bytes = math::align(
        spec.num_scaleout_ranks * spec.num_scaleup_ranks +
            spec.num_experts,
        spec.num_notify_warps * 32) * sizeof(int);
    EP_HOST_ASSERT(
        notify_bytes +
            (spec.num_scaleout_warps + spec.num_forward_warps) *
                token_layout.get_num_bytes<true>() <=
        spec.num_smem_bytes);
}

static RailBalanceHybridDispatchRuntime::Args
make_rail_balance_hybrid_dispatch_args(
    const RailBalanceHybridDispatchSpec& spec) {
    validate_rail_balance_hybrid_dispatch_spec(spec);
    const int num_threads =
        (spec.num_notify_warps + spec.num_scaleout_warps +
         spec.num_forward_warps) * 32;
    return {
        .spec = spec,
        .x = nullptr,
        .sf = nullptr,
        .topk_idx = nullptr,
        .topk_weights = nullptr,
        .copied_topk_idx = nullptr,
        .cumulative_local_expert_recv_stats = nullptr,
        .psum_num_recv_tokens_per_scaleup_rank = nullptr,
        .psum_num_recv_tokens_per_expert = nullptr,
        .num_unaligned_recv_tokens_per_expert = nullptr,
        .dst_buffer_slot_idx = nullptr,
        .token_metadata_at_forward = nullptr,
        .num_tokens = 0,
        .sf_token_stride = 0,
        .sf_hidden_stride = 0,
        .nccl_dev_comm = {nullptr},
        .nccl_window = nullptr,
        .buffer = nullptr,
        .workspace = nullptr,
        .mapped_host_workspace = nullptr,
        .rail_balance_arena = nullptr,
        .rail_balance_retained = nullptr,
        .rail_balance_moved = nullptr,
        .rail_balance_group_prefix = nullptr,
        .rail_balance_proxy_required = nullptr,
        .scaleout_rank_idx = 0,
        .scaleup_rank_idx = 0,
        .launch_args = jit::LaunchArgs(
            spec.num_sms, num_threads, spec.num_smem_bytes,
            2 - (spec.num_sms % 2), true),
    };
}

struct PreparedRailBalanceHybridDispatch {
    std::shared_ptr<jit::KernelRuntime> runtime;
    RailBalanceHybridDispatchSpec spec;
    jit::LaunchArgs launch_args;
};

static PreparedRailBalanceHybridDispatch
prepare_rail_balance_hybrid_dispatch(
    const RailBalanceHybridDispatchSpec& spec) {
    const auto args = make_rail_balance_hybrid_dispatch_args(spec);
    const auto key = fmt::format(
        "rail_balance_hybrid_dispatch_force_v1_"
        "sm{}_nw{}_sw{}_fw{}_d{}_g{}_h{}_m{}_e{}_k{}_a{}_q{}_t{}_p{}",
        spec.num_sms, spec.num_notify_warps,
        spec.num_scaleout_warps, spec.num_forward_warps,
        spec.num_scaleout_ranks, spec.num_scaleup_ranks,
        spec.num_hidden_bytes, spec.num_max_tokens_per_rank,
        spec.num_experts, spec.num_topk, spec.expert_alignment,
        spec.num_qps, spec.num_timeout_cycles, spec.proxy_capacity);
    return {
        .runtime = jit::compiler->build(
            key, RailBalanceHybridDispatchRuntime::generate(args)),
        .spec = spec,
        .launch_args = args.launch_args,
    };
}

// Gate #2 is the last planned validation/materialization phase. This adapter
// therefore only binds already-owned pointers to the frozen force ABI and
// submits the compiled specialization. Runtime config/launch may still fail;
// the H4c owner poisons the transaction before entering that job-fatal window.
// Keep validation, JIT, allocation, D2H reads, and synchronization out.
static void launch_prepared_rail_balance_hybrid_dispatch(
    const PreparedRailBalanceHybridDispatch& prepared,
    void* x,
    sf_pack_t* sf,
    topk_idx_t* topk_idx,
    float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    int* token_metadata_at_forward,
    const int& num_tokens,
    const int& sf_token_stride,
    const int& sf_hidden_stride,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* buffer,
    void* workspace,
    void* mapped_host_workspace,
    void* rail_balance_arena,
    const int* rail_balance_retained,
    const int* rail_balance_moved,
    const int* rail_balance_group_prefix,
    const int* rail_balance_proxy_required,
    const int& scaleout_rank_idx,
    const int& scaleup_rank_idx,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridDispatchRuntime::Args args = {
        .spec = prepared.spec,
        .x = x,
        .sf = sf,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .copied_topk_idx = copied_topk_idx,
        .cumulative_local_expert_recv_stats =
            cumulative_local_expert_recv_stats,
        .psum_num_recv_tokens_per_scaleup_rank =
            psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert =
            psum_num_recv_tokens_per_expert,
        .num_unaligned_recv_tokens_per_expert =
            num_unaligned_recv_tokens_per_expert,
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        .token_metadata_at_forward = token_metadata_at_forward,
        .num_tokens = num_tokens,
        .sf_token_stride = sf_token_stride,
        .sf_hidden_stride = sf_hidden_stride,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .buffer = buffer,
        .workspace = workspace,
        .mapped_host_workspace = mapped_host_workspace,
        .rail_balance_arena = rail_balance_arena,
        .rail_balance_retained = rail_balance_retained,
        .rail_balance_moved = rail_balance_moved,
        .rail_balance_group_prefix = rail_balance_group_prefix,
        .rail_balance_proxy_required = rail_balance_proxy_required,
        .scaleout_rank_idx = scaleout_rank_idx,
        .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = prepared.launch_args,
    };
    RailBalanceHybridDispatchRuntime::launch(
        prepared.runtime, args, stream);
}

// Compile the byte-immutable legacy kernel through a probe-specific runtime.
// Do not call DispatchRuntime::generate here: its function-static recursive
// include hash must never be initialized by synthetic geometry. This separate
// LaunchRuntime still hashes the complete legacy include closure, so an old
// cubin cannot survive a transitive source change under the probe cache key.
class RailBalanceHybridLegacyDispatchProbeRuntime final:
    public jit::LaunchRuntime<
        RailBalanceHybridLegacyDispatchProbeRuntime> {
public:
    struct Args {
        RailBalanceHybridDispatchSpec spec;
    };

    static std::string generate_impl(const Args& args) {
        const auto& s = args.spec;
        return fmt::format(R"(
#include <deep_ep/impls/hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &hybrid_dispatch_impl<
            true, false, {}, {}, {}, {}, {}, {}, {}, 0, {}, {}, {}, {}, {}, {}>);
}}
)",
        s.num_sms,
        s.num_notify_warps,
        s.num_scaleout_warps,
        s.num_forward_warps,
        s.num_scaleout_ranks,
        s.num_scaleup_ranks,
        s.num_hidden_bytes,
        s.num_max_tokens_per_rank,
        s.num_experts,
        s.num_topk,
        s.expert_alignment,
        s.num_qps,
        s.num_timeout_cycles);
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_legacy_dispatch_probe(
    const RailBalanceHybridDispatchSpec& spec) {
    validate_rail_balance_hybrid_dispatch_spec(spec);
    const RailBalanceHybridLegacyDispatchProbeRuntime::Args args = {
        .spec = spec,
    };
    const auto key = fmt::format(
        "rail_balance_hybrid_dispatch_legacy_probe_v1_"
        "sm{}_nw{}_sw{}_fw{}_d{}_g{}_h{}_m{}_e{}_k{}_a{}_q{}_t{}",
        spec.num_sms, spec.num_notify_warps,
        spec.num_scaleout_warps, spec.num_forward_warps,
        spec.num_scaleout_ranks, spec.num_scaleup_ranks,
        spec.num_hidden_bytes, spec.num_max_tokens_per_rank,
        spec.num_experts, spec.num_topk, spec.expert_alignment,
        spec.num_qps, spec.num_timeout_cycles);
    return jit::compiler->build(
        key, RailBalanceHybridLegacyDispatchProbeRuntime::generate(args));
}

// Test-only compile probe.  It exercises the actual force runtime and returns
// its generated source for ABI assertions; it never launches the kernel or
// changes the disabled force capability.
static pybind11::dict rail_balance_hybrid_dispatch_codegen_test(
    const int& num_scaleout_ranks,
    const int& num_scaleup_ranks,
    const int& hidden,
    const int& num_topk,
    const int& num_sms,
    const int& num_channels_per_sm,
    const int& proxy_capacity,
    const int& num_experts) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0 and
                   hidden <= INT_MAX /
                       static_cast<int>(sizeof(nv_bfloat16)));
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_scaleout_ranks > 0 and num_scaleup_ranks > 0);
    EP_HOST_ASSERT(num_scaleout_ranks <=
                   INT_MAX / num_scaleup_ranks);
    const int num_ranks = num_scaleout_ranks * num_scaleup_ranks;
    EP_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    const RailBalanceHybridDispatchSpec spec = {
        .num_sms = num_sms,
        .num_notify_warps = 4,
        .num_scaleout_warps = num_channels_per_sm,
        .num_forward_warps = num_channels_per_sm,
        .num_scaleout_ranks = num_scaleout_ranks,
        .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = hidden *
            static_cast<int>(sizeof(nv_bfloat16)),
        .num_max_tokens_per_rank = 8192,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .expert_alignment = 1,
        .num_qps = 9,
        .num_timeout_cycles = 1000000000ll,
        .num_smem_bytes = jit::device_runtime->get_num_smem_bytes(),
        .proxy_capacity = proxy_capacity,
    };
    const auto args = make_rail_balance_hybrid_dispatch_args(spec);
    const auto code = RailBalanceHybridDispatchRuntime::generate(args);
    const auto prepared = prepare_rail_balance_hybrid_dispatch(spec);
    EP_HOST_ASSERT(prepared.runtime != nullptr);
    const RailBalanceHybridLegacyDispatchProbeRuntime::Args legacy_args = {
        .spec = spec,
    };
    const auto legacy_code =
        RailBalanceHybridLegacyDispatchProbeRuntime::generate(legacy_args);
    const auto legacy_runtime =
        prepare_rail_balance_hybrid_legacy_dispatch_probe(spec);
    EP_HOST_ASSERT(legacy_runtime != nullptr);

    pybind11::dict result;
    result["code"] = code;
    result["legacy_code"] = legacy_code;
    result["num_threads"] = args.launch_args.num_threads;
    result["num_channels"] = num_sms * num_channels_per_sm;
    result["num_forward_metadata_dims"] =
        rail_balance::get_num_hybrid_forward_metadata_dims(num_topk);
    result["reuse_slot_indices"] = false;
    result["num_sf_packs"] = 0;
    result["proxy_capacity"] = proxy_capacity;
    return result;
}

// Force-v1 count exchange is local to one physical LSA team even in a real
// multi-node Hybrid job. This runtime deliberately specializes the existing
// barrier implementation as synthetic (scaleout=1, scaleup=G); using the
// actual Hybrid topology here would also enter Rail.
class RailBalanceHybridLocalBarrierRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridLocalBarrierRuntime> {
public:
    struct Args {
        int num_rails;
        int64_t num_timeout_cycles;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int nvl_rank_idx;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &barrier_impl<true, 1, 512, 1, {}, {}, true>);
}}
)", args.num_rails, args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window, args.workspace,
            0, args.nvl_rank_idx));
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_local_barrier(
    const int& num_rails,
    const int64_t& num_timeout_cycles) {
    const RailBalanceHybridLocalBarrierRuntime::Args args = {
        .num_rails = num_rails,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = {nullptr},
        .nccl_window = nullptr,
        .workspace = nullptr,
        .nvl_rank_idx = 0,
        .launch_args = jit::LaunchArgs(1, 512, 0, 1, true),
    };
    return jit::compiler->build(
        fmt::format(
            "rail_balance_hybrid_local_barrier_g{}_t{}_v1",
            num_rails, num_timeout_cycles),
        RailBalanceHybridLocalBarrierRuntime::generate(args));
}

static void submit_prepared_rail_balance_hybrid_local_barrier(
    const std::shared_ptr<jit::KernelRuntime>& prepared,
    const jit::LaunchArgs& launch_args,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* workspace,
    const int& num_rails,
    const int& nvl_rank_idx,
    const int64_t& num_timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridLocalBarrierRuntime::Args args = {
        .num_rails = num_rails,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .nvl_rank_idx = nvl_rank_idx,
        .launch_args = launch_args,
    };
    RailBalanceHybridLocalBarrierRuntime::launch(prepared, args, stream);
}

static void launch_prepared_rail_balance_hybrid_local_barrier(
    const std::shared_ptr<jit::KernelRuntime>& prepared,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* workspace,
    const int& num_rails,
    const int& nvl_rank_idx,
    const int64_t& num_timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const auto launch_args = jit::LaunchArgs(1, 512, 0, 1, true);
    submit_prepared_rail_balance_hybrid_local_barrier(
        prepared, launch_args, nccl_dev_comm, nccl_window, workspace,
        num_rails, nvl_rank_idx, num_timeout_cycles, stream);
}

struct PreparedRailBalanceHybridPlan {
    std::shared_ptr<jit::KernelRuntime> count;
    std::shared_ptr<jit::KernelRuntime> plan;
    std::shared_ptr<jit::KernelRuntime> prefix;
    jit::LaunchArgs count_launch_args;
    jit::LaunchArgs plan_launch_args;
    jit::LaunchArgs prefix_launch_args;
};

// Named storage keeps the production-shaped two-phase transaction readable;
// as_tuple() preserves the exact private B1 14-tensor ABI.
struct RailBalanceHybridPlanOutputs {
    torch::Tensor channel_count;
    torch::Tensor count;
    torch::Tensor quota;
    torch::Tensor keep_count;
    torch::Tensor segments;
    torch::Tensor num_segments;
    torch::Tensor owner_channel_prefix;
    torch::Tensor retained;
    torch::Tensor moved;
    torch::Tensor moved_channel_prefix;
    torch::Tensor group_prefix;
    torch::Tensor proxy_required;
    torch::Tensor moved_copies;
    torch::Tensor status;

    RailBalanceHybridPlanTensors as_tuple() const {
        return {
            channel_count, count, quota, keep_count,
            segments, num_segments, owner_channel_prefix,
            retained, moved, moved_channel_prefix, group_prefix,
            proxy_required, moved_copies, status,
        };
    }
};

static RailBalanceHybridPlanOutputs allocate_rail_balance_hybrid_plan_outputs(
    const torch::TensorOptions& int_options,
    const int& num_rails,
    const int& num_channels,
    const int& num_destinations) {
    return {
        .channel_count = torch::empty(
            {num_rails, num_channels, num_destinations}, int_options),
        .count = torch::empty(
            {num_rails, num_destinations}, int_options),
        .quota = torch::empty(
            {num_rails, num_destinations}, int_options),
        .keep_count = torch::empty(
            {num_rails, num_destinations}, int_options),
        .segments = torch::full(
            {num_destinations, num_rails - 1, 5}, -1, int_options),
        .num_segments = torch::empty(
            {num_destinations}, int_options),
        .owner_channel_prefix = torch::empty(
            {num_rails, num_channels, num_destinations}, int_options),
        .retained = torch::empty(
            {num_rails, num_channels, num_destinations}, int_options),
        .moved = torch::empty(
            {num_rails, num_channels, num_destinations}, int_options),
        .moved_channel_prefix = torch::empty(
            {num_rails, num_destinations, num_channels + 1}, int_options),
        .group_prefix = torch::empty(
            {num_rails, num_channels, num_destinations}, int_options),
        .proxy_required = torch::empty({num_rails}, int_options),
        .moved_copies = torch::zeros({1}, int_options),
        .status = torch::zeros({1}, int_options),
    };
}

struct RailBalanceHopPlanState {
    torch::Tensor records;
    torch::Tensor resolutions;
    torch::Tensor pair_load;
    torch::Tensor source_load;
    torch::Tensor owner_remaining;
    torch::Tensor endpoint_count;
    torch::Tensor endpoint_owner_quota;
    torch::Tensor owner_group_cursor;
    torch::Tensor path_units;
    rail_balance::HopCopyRecord* local_records;
    std::array<const rail_balance::HopCopyRecord*, 32> peer_records;
    size_t record_bytes;
    int two_hop_threshold_percent;
    int max_two_hop_percent;
    int hop_penalty_percent;
    int planner_chunk_size;
    PreparedRailBalanceHopPlan prepared;
};

enum class RailBalanceHybridPlanState : uint8_t {
    Preparing,
    PlanReady,
    DispatchLive,
    Invalid,
};

// Gate #1 freezes every pointer which finish-plan and the later committed
// dispatch consume.  Keeping these as raw, non-owning views is safe because
// RailBalanceHybridPlanPending owns all backing tensors and the symmetric
// window for the whole transaction.
struct RailBalanceHybridPlanRawPointers {
    int* channel_count;
    int* count;
    int* quota;
    int* keep_count;
    int* segments;
    int* num_segments;
    int* owner_channel_prefix;
    int* retained;
    int* moved;
    int* moved_channel_prefix;
    int* group_prefix;
    int* proxy_required;
    int* moved_copies;
    int* status;
    int* local_channel_count;
    std::array<const int*, 32> peer_channel_count;
};

struct RailBalanceHybridDispatchRawPointers {
    void* x;
    topk_idx_t* topk_idx;
    float* topk_weights;
    topk_idx_t* copied_topk_idx;
    int* cumulative_local_expert_recv_stats;
    int* psum_num_recv_tokens_per_scaleup_rank;
    int* psum_num_recv_tokens_per_expert_storage;
    int* psum_num_recv_tokens_per_expert_inclusive;
    int* num_unaligned_recv_tokens_per_expert;
    int* dst_buffer_slot_idx;
    int* token_metadata_at_forward;
    int* channel_linked_list;
    void* buffer;
    void* workspace;
    void* mapped_host_workspace;
    void* arena;
    void* proxy_dispatch_base;
    void* proxy_return_base;
    void* legacy_reduce_buffer_base;
    int64_t* host_scaleup_rank_count;
    int64_t* host_expert_count;
    const int* owner_channel_prefix;
    const int* keep_count;
    const int* segments;
    const int* num_segments;
    const int* retained;
    const int* moved;
    const int* moved_channel_prefix;
    const int* group_prefix;
    const int* proxy_required;
    int* status;
};

struct RailBalanceHybridDispatchCompletionRawPointers {
    void* recv_x;
    topk_idx_t* recv_topk_idx;
    float* recv_topk_weights;
    int* recv_src_metadata;
};

// Exact receive storage is the only dispatch ownership which cannot exist
// before Gate #1: its leading dimension is published by the main kernel.
// Install this owner before submitting the copy epilogue so a later launch,
// status-readback, or synchronization failure cannot release in-flight memory.
struct RailBalanceHybridDispatchCompletion {
    torch::Tensor recv_x;
    torch::Tensor recv_topk_idx;
    torch::Tensor recv_topk_weights;
    torch::Tensor recv_src_metadata;
    int num_recv_tokens;
    int num_expanded_tokens;
    std::vector<int> num_recv_tokens_per_expert_list;
    RailBalanceHybridDispatchCompletionRawPointers raw;
};

struct RailBalanceHybridCombineRawPointers {
    void* x;
    float* topk_weights;
    void* combined_x;
    float* combined_topk_weights;
};

// Keep both expert outputs consumed by main combine and the final owner
// outputs alive across the whole committed chain. In particular, a host-side
// launch failure after main combine must not release any in-flight storage.
struct RailBalanceHybridCombineCompletion {
    torch::Tensor x;
    std::optional<torch::Tensor> topk_weights;
    torch::Tensor combined_x;
    std::optional<torch::Tensor> combined_topk_weights;
    RailBalanceHybridCombineRawPointers raw;
};

// The first force dispatch prepares the complete round trip.  Receive payload
// tensors are deliberately absent: their exact leading dimension is only
// known after the main dispatch publishes CPU counts.  Every other Tensor,
// cubin, LaunchArgs, and pointer required by dispatch/combine is owned here
// before WORLD Gate #1.
struct RailBalanceHybridDispatchBundle {
    torch::Tensor x;
    torch::Tensor topk_idx;
    torch::Tensor topk_weights;
    torch::Tensor cumulative_local_expert_recv_stats;
    torch::Tensor copied_topk_idx;
    torch::Tensor psum_num_recv_tokens_per_scaleup_rank;
    torch::Tensor psum_num_recv_tokens_per_expert_storage;
    torch::Tensor psum_num_recv_tokens_per_expert;
    torch::Tensor num_unaligned_recv_tokens_per_expert;
    torch::Tensor dst_buffer_slot_idx;
    torch::Tensor token_metadata_at_forward;
    torch::Tensor channel_linked_list;

    std::shared_ptr<PreparedRailBalanceHybridDispatch> main_dispatch;
    std::shared_ptr<PreparedRailBalanceHybridDispatchEpilogue>
        dispatch_epilogue;
    std::shared_ptr<PreparedRailBalanceHybridCombine> main_combine;

    RailBalanceHybridDispatchRawPointers raw;
    std::optional<RailBalanceHybridDispatchCompletion>
        dispatch_completion;
    std::optional<RailBalanceHybridCombineCompletion>
        combine_completion;
    at::cuda::CUDAStream compute_stream;
    int num_tokens;
    int hidden;
    int num_topk;
    int num_max_tokens_per_rank;
    int num_experts;
    int num_local_experts;
    int num_rails;
    int num_destinations;
    int num_channels;
    int num_channels_per_sm;
    int num_max_tokens_per_channel;
    int num_sms;
    int num_qps;
    int num_smem_bytes;
    int num_physical_sms;
    int scaleout_rank_idx;
    int scaleup_rank_idx;
    int rank_idx;
    int proxy_capacity_per_egress;
    int64_t arena_offset;
    int64_t arena_bytes;
};

// Fixed production prepare result.  It is sufficient to encode Gate #1
// without recomputing topology or launch geometry in Python.  N and local
// rank identities are intentionally absent because they may differ by rank.
using RailBalanceHybridDispatchPrepareResult = std::tuple<
    int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int, int,
    int64_t, int64_t, int64_t, int, int, int>;

// One private prepare may be live per ElasticBuffer. PlanReady remains live
// across the caller's Gate #2 and is released only by the explicit abort for
// now; C080's source-shuffle commit will own the next state transition.
struct RailBalanceHybridPlanPending {
    int invocation_id;
    RailBalanceHybridPlanState state;
    bool shuffled;
    bool return_unshuffle_tested;
    bool combine_epilogue_tested;
    int plan_status;
    int num_rails;
    int num_channels;
    int num_destinations;
    int local_destination;
    int num_tokens;
    int num_topk;
    int hidden;
    int num_experts;
    int num_max_tokens_per_rank;
    int proxy_capacity_per_egress;
    int normalized_remainder_seed;
    int policy;
    int threshold_percent;
    int64_t arena_offset;
    int64_t active_count_values;
    size_t active_count_bytes;
    void* arena;
    int* local_channel_count;
    RailBalanceHybridPlanRawPointers raw;
    torch::Tensor topk_idx;
    PreparedRailBalanceHybridPlan prepared;
    jit::LaunchArgs local_barrier_launch_args;
    std::shared_ptr<jit::KernelRuntime> local_barrier;
    std::shared_ptr<PreparedRailBalanceHybridSourceShuffle>
        source_shuffle;
    std::shared_ptr<PreparedRailBalanceHybridReturnUnshuffle>
        return_unshuffle;
    std::shared_ptr<PreparedRailBalanceHybridCombineEpilogue>
        combine_epilogue;
    std::shared_ptr<RailBalanceHybridDispatchBundle> dispatch_bundle;
    RailBalanceHybridPlanOutputs outputs;
    std::optional<RailBalanceHopPlanState> hop;
};

static PreparedRailBalanceHybridPlan prepare_rail_balance_hybrid_plan(
    const int& num_owners,
    const int& num_rails,
    const int& num_channels,
    const int& num_destinations) {
    EP_HOST_ASSERT(num_owners >= 1 and num_owners <= 32);
    EP_HOST_ASSERT(num_rails >= 1 and num_rails <= 32);
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_destinations >= 1 and
                   num_destinations <=
                       rail_balance::kNumHybridMaxDestinations);
    const RailBalanceHybridCountRuntime::Args count_args = {
        .topk_idx = nullptr,
        .channel_count = nullptr,
        .status = nullptr,
        .num_owners = 0,
        .num_tokens = 0,
        .num_topk = 0,
        .num_channels = 0,
        .num_experts = 0,
        .num_destinations = 0,
        .local_destination = 0,
        .launch_args = jit::LaunchArgs(1, 32),
    };
    const RailBalanceHybridPlanRuntime::Args plan_args = {
        .channel_count = nullptr,
        .count = nullptr,
        .quota = nullptr,
        .keep_count = nullptr,
        .segments = nullptr,
        .num_segments = nullptr,
        .num_rails = 0,
        .num_channels = 0,
        .num_destinations = 0,
        .remainder_seed = 0,
        .policy = 0,
        .threshold_percent = 0,
        .launch_args = jit::LaunchArgs(1, 32),
    };
    const RailBalanceHybridPrefixRuntime::Args prefix_args = {
        .channel_count = nullptr,
        .quota = nullptr,
        .keep_count = nullptr,
        .owner_channel_prefix = nullptr,
        .retained = nullptr,
        .moved = nullptr,
        .moved_channel_prefix = nullptr,
        .group_prefix = nullptr,
        .proxy_required = nullptr,
        .moved_copies = nullptr,
        .status = nullptr,
        .num_rails = 0,
        .num_channels = 0,
        .num_destinations = 0,
        .num_max_tokens_per_rank = 0,
        .proxy_capacity_per_egress = 0,
        .launch_args = jit::LaunchArgs(1, 32),
    };
    return {
        .count = jit::compiler->build(
            "rail_balance_hybrid_count_v1",
            RailBalanceHybridCountRuntime::generate(count_args)),
        .plan = jit::compiler->build(
            "rail_balance_hybrid_plan_v2",
            RailBalanceHybridPlanRuntime::generate(plan_args)),
        .prefix = jit::compiler->build(
            "rail_balance_hybrid_prefix_v1",
            RailBalanceHybridPrefixRuntime::generate(prefix_args)),
        .count_launch_args = jit::LaunchArgs(
            num_owners * num_channels, 32),
        .plan_launch_args = jit::LaunchArgs(num_destinations, 32),
        .prefix_launch_args = jit::LaunchArgs(num_rails, 32),
    };
}

static void launch_prepared_rail_balance_hybrid_count(
    const PreparedRailBalanceHybridPlan& prepared,
    const topk_idx_t* topk_idx,
    int* channel_count,
    int* status,
    const int& num_owners,
    const int& num_tokens,
    const int& num_topk,
    const int& num_channels,
    const int& num_experts,
    const int& num_destinations,
    const int& local_destination,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridCountRuntime::Args args = {
        .topk_idx = topk_idx,
        .channel_count = channel_count,
        .status = status,
        .num_owners = num_owners,
        .num_tokens = num_tokens,
        .num_topk = num_topk,
        .num_channels = num_channels,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .local_destination = local_destination,
        .launch_args = prepared.count_launch_args,
    };
    RailBalanceHybridCountRuntime::launch(prepared.count, args, stream);
}

static void launch_prepared_rail_balance_hybrid_plan(
    const PreparedRailBalanceHybridPlan& prepared,
    const int* channel_count,
    int* count,
    int* quota,
    int* keep_count,
    int* segments,
    int* num_segments,
    const int& num_rails,
    const int& num_channels,
    const int& num_destinations,
    const int& remainder_seed,
    const int& policy,
    const int& threshold_percent,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridPlanRuntime::Args args = {
        .channel_count = channel_count,
        .count = count,
        .quota = quota,
        .keep_count = keep_count,
        .segments = segments,
        .num_segments = num_segments,
        .num_rails = num_rails,
        .num_channels = num_channels,
        .num_destinations = num_destinations,
        .remainder_seed = remainder_seed,
        .policy = policy,
        .threshold_percent = threshold_percent,
        .launch_args = prepared.plan_launch_args,
    };
    RailBalanceHybridPlanRuntime::launch(prepared.plan, args, stream);
}

static void launch_prepared_rail_balance_hybrid_prefix(
    const PreparedRailBalanceHybridPlan& prepared,
    const int* channel_count,
    const int* quota,
    const int* keep_count,
    int* owner_channel_prefix,
    int* retained,
    int* moved,
    int* moved_channel_prefix,
    int* group_prefix,
    int* proxy_required,
    int* moved_copies,
    int* status,
    const int& num_rails,
    const int& num_channels,
    const int& num_destinations,
    const int& num_max_tokens_per_rank,
    const int& proxy_capacity_per_egress,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridPrefixRuntime::Args args = {
        .channel_count = channel_count,
        .quota = quota,
        .keep_count = keep_count,
        .owner_channel_prefix = owner_channel_prefix,
        .retained = retained,
        .moved = moved,
        .moved_channel_prefix = moved_channel_prefix,
        .group_prefix = group_prefix,
        .proxy_required = proxy_required,
        .moved_copies = moved_copies,
        .status = status,
        .num_rails = num_rails,
        .num_channels = num_channels,
        .num_destinations = num_destinations,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .proxy_capacity_per_egress = proxy_capacity_per_egress,
        .launch_args = prepared.prefix_launch_args,
    };
    RailBalanceHybridPrefixRuntime::launch(prepared.prefix, args, stream);
}

static int parse_rail_balance_hybrid_policy(
    const pybind11::object& value) {
    EP_HOST_ASSERT(PyLong_CheckExact(value.ptr()));
    const auto parsed = value.cast<int64_t>();
    EP_HOST_ASSERT(parsed >= 0 and parsed <= INT_MAX);
    const int policy = static_cast<int>(parsed);
    EP_HOST_ASSERT(rail_balance::is_valid_hybrid_policy(policy));
    return policy;
}

static int parse_rail_balance_hybrid_threshold_percent(
    const pybind11::object& value) {
    EP_HOST_ASSERT(PyLong_CheckExact(value.ptr()));
    const auto parsed = value.cast<int64_t>();
    EP_HOST_ASSERT(parsed >= 0 and
                   parsed <=
                       rail_balance::kMaxHybridPolicyThresholdPercent);
    return static_cast<int>(parsed);
}

static RailBalanceHopRecordTensors build_rail_balance_hop_records(
    const torch::Tensor& topk_idx,
    const int& num_channels,
    const int& num_experts,
    const int& num_destinations,
    const int& local_destination) {
    EP_HOST_ASSERT(topk_idx.dim() == 3);
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
    EP_HOST_ASSERT(
        topk_idx.scalar_type() ==
        c10::CppTypeToScalarType<topk_idx_t>::value);

    const auto num_rails_i64 = topk_idx.size(0);
    const auto num_tokens_i64 = topk_idx.size(1);
    const auto num_topk_i64 = topk_idx.size(2);
    EP_HOST_ASSERT(num_rails_i64 >= 1 and num_rails_i64 <= 32);
    EP_HOST_ASSERT(num_tokens_i64 >= 0 and num_tokens_i64 <= INT_MAX);
    EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
    const int num_rails = static_cast<int>(num_rails_i64);
    const int num_tokens = static_cast<int>(num_tokens_i64);
    const int num_topk = static_cast<int>(num_topk_i64);
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_destinations >= 2 and num_destinations <= 32);
    EP_HOST_ASSERT(local_destination >= 0 and
                   local_destination < num_destinations);
    EP_HOST_ASSERT(num_experts % (num_destinations * num_rails) == 0);

    c10::cuda::CUDAGuard device_guard(topk_idx.device());
    const int device_index = topk_idx.get_device();
    int expected_device = -1;
    if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
            expected_device, device_index, std::memory_order_relaxed) and
        expected_device != device_index)
        EP_HOST_UNREACHABLE(
            "Hybrid rail-balance planner supports one CUDA device per process");
    const auto stream = at::cuda::getCurrentCUDAStream(device_index);
    auto records = torch::empty(
        {num_rails, num_tokens, num_topk},
        topk_idx.options().dtype(torch::kLong));
    auto status = torch::zeros(
        {1}, topk_idx.options().dtype(torch::kInt));
    if (num_tokens == 0)
        return {records, status};

    const RailBalanceHopRecordRuntime::Args prototype = {
        .topk_idx = nullptr,
        .records = nullptr,
        .status = nullptr,
        .num_owners = 0,
        .num_tokens = 0,
        .num_topk = 0,
        .num_channels = 0,
        .num_experts = 0,
        .num_destinations = 0,
        .num_rails = 0,
        .local_destination = 0,
        .launch_args = jit::LaunchArgs(
            num_rails * num_channels, 32),
    };
    const auto runtime = jit::compiler->build(
        "rail_balance_hop_record_v1",
        RailBalanceHopRecordRuntime::generate(prototype));
    RailBalanceHopRecordRuntime::launch(
        runtime,
        RailBalanceHopRecordRuntime::Args{
            .topk_idx = topk_idx.data_ptr<topk_idx_t>(),
            .records = reinterpret_cast<rail_balance::HopCopyRecord*>(
                records.data_ptr<int64_t>()),
            .status = status.data_ptr<int>(),
            .num_owners = num_rails,
            .num_tokens = num_tokens,
            .record_token_capacity = num_tokens,
            .num_topk = num_topk,
            .num_channels = num_channels,
            .num_experts = num_experts,
            .num_destinations = num_destinations,
            .num_rails = num_rails,
            .local_destination = local_destination,
            .launch_args = prototype.launch_args,
        },
        stream);

    int host_status = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        &host_status, status.data_ptr<int>(), sizeof(host_status),
        cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    if (host_status == static_cast<int>(
            rail_balance::HybridPlanError::ExpertOutOfRange))
        throw EPExceptionWithLineInfo(
            "Rail balance hop record",
            "topk_idx contains a masked or out-of-range expert id");
    if (host_status == static_cast<int>(
            rail_balance::HybridPlanError::DuplicateExpert))
        throw EPExceptionWithLineInfo(
            "Rail balance hop record",
            "topk_idx contains duplicate expert ids within one token");
    EP_HOST_ASSERT(host_status == 0);
    return {records, status};
}

static RailBalanceHopPlanTensors build_rail_balance_hop_plan(
    const torch::Tensor& records,
    const int& num_channels,
    const int& num_destinations,
    const int& num_max_tokens_per_rank,
    const int& proxy_capacity_per_egress,
    const int& planner_seed,
    const int& two_hop_threshold_percent,
    const int& max_two_hop_percent,
    const int& hop_penalty_percent,
    const int& planner_chunk_size) {
    EP_HOST_ASSERT(records.dim() == 3);
    EP_HOST_ASSERT(records.is_cuda() and records.is_contiguous());
    EP_HOST_ASSERT(records.scalar_type() == torch::kLong);
    const auto num_rails_i64 = records.size(0);
    const auto num_tokens_i64 = records.size(1);
    const auto num_topk_i64 = records.size(2);
    EP_HOST_ASSERT(num_rails_i64 >= 1 and num_rails_i64 <= 32);
    EP_HOST_ASSERT(num_tokens_i64 >= 0 and num_tokens_i64 <= INT_MAX);
    EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
    const int num_rails = static_cast<int>(num_rails_i64);
    const int num_tokens = static_cast<int>(num_tokens_i64);
    const int num_topk = static_cast<int>(num_topk_i64);
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_destinations >= 2 and num_destinations <= 32);
    EP_HOST_ASSERT(num_max_tokens_per_rank >= num_tokens);
    EP_HOST_ASSERT(proxy_capacity_per_egress >= 0);
    EP_HOST_ASSERT(planner_seed >= 0);
    EP_HOST_ASSERT(two_hop_threshold_percent >= 0 and
                   two_hop_threshold_percent <= 10000);
    EP_HOST_ASSERT(max_two_hop_percent >= 0 and
                   max_two_hop_percent <= 100);
    EP_HOST_ASSERT(hop_penalty_percent >= 0 and
                   hop_penalty_percent <= 10000);
    EP_HOST_ASSERT(planner_chunk_size >= 1);

    c10::cuda::CUDAGuard device_guard(records.device());
    const int device_index = records.get_device();
    int expected_device = -1;
    if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
            expected_device, device_index, std::memory_order_relaxed) and
        expected_device != device_index)
        EP_HOST_UNREACHABLE(
            "Hybrid rail-balance planner supports one CUDA device per process");
    const auto stream = at::cuda::getCurrentCUDAStream(device_index);
    const auto int_options = records.options().dtype(torch::kInt);
    auto resolutions = torch::full(
        {num_rails, num_tokens, num_topk, 4}, -1, int_options);
    auto pair_load = torch::zeros(
        {num_destinations, num_rails}, int_options);
    auto source_load = torch::zeros({num_rails}, int_options);
    auto owner_remaining = torch::zeros(
        {num_rails, num_destinations}, int_options);
    auto endpoint_count = torch::zeros(
        {num_rails, num_destinations, num_rails}, int_options);
    auto endpoint_owner_quota = torch::zeros_like(endpoint_count);
    auto owner_group_cursor = planner_chunk_size > 1 ? torch::zeros(
        {32, num_rails, num_channels, num_destinations}, int_options) :
        torch::zeros({1}, int_options);
    auto retained = torch::zeros(
        {num_rails, num_channels, num_destinations}, int_options);
    auto moved = torch::zeros(
        {num_rails, num_channels, num_destinations}, int_options);
    auto retained_prefix = torch::zeros(
        {num_rails, num_channels, num_destinations}, int_options);
    auto group_prefix = torch::zeros(
        {num_rails, num_channels, num_destinations}, int_options);
    auto proxy_required = torch::zeros({num_rails}, int_options);
    auto path_units = torch::zeros({4}, int_options);
    auto moved_copies = torch::zeros({1}, int_options);
    auto status = torch::zeros({1}, int_options);
    if (num_tokens == 0) {
        return {
            resolutions, pair_load, source_load, retained, moved,
            group_prefix, proxy_required, path_units, moved_copies, status,
        };
    }

    const RailBalanceHopPlanRuntime::Args prototype = {
        .records = nullptr,
        .resolutions = nullptr,
        .pair_load = nullptr,
        .source_load = nullptr,
        .owner_remaining = nullptr,
        .endpoint_count = nullptr,
        .endpoint_owner_quota = nullptr,
        .owner_group_cursor = nullptr,
        .retained = nullptr,
        .moved = nullptr,
        .retained_prefix = nullptr,
        .group_prefix = nullptr,
        .proxy_required = nullptr,
        .path_units = nullptr,
        .moved_copies = nullptr,
        .status = nullptr,
        .num_rails = 0,
        .num_tokens = 0,
        .num_topk = 0,
        .num_channels = 0,
        .num_destinations = 0,
        .num_max_tokens_per_rank = 0,
        .proxy_capacity_per_egress = 0,
        .planner_seed = 0,
        .planner_chunk_size = 1,
        .two_hop_threshold_percent = 0,
        .max_two_hop_percent = 0,
        .hop_penalty_percent = 0,
        .launch_args = jit::LaunchArgs(1, 32),
    };
    const auto runtime = jit::compiler->build(
        "rail_balance_hop_plan_v2",
        RailBalanceHopPlanRuntime::generate(prototype));
    RailBalanceHopPlanRuntime::launch(
        runtime,
        RailBalanceHopPlanRuntime::Args{
            .records = reinterpret_cast<const rail_balance::HopCopyRecord*>(
                records.data_ptr<int64_t>()),
            .resolutions =
                reinterpret_cast<rail_balance::HopCopyResolution*>(
                    resolutions.data_ptr<int>()),
            .pair_load = pair_load.data_ptr<int>(),
            .source_load = source_load.data_ptr<int>(),
            .owner_remaining = owner_remaining.data_ptr<int>(),
            .endpoint_count = endpoint_count.data_ptr<int>(),
            .endpoint_owner_quota = endpoint_owner_quota.data_ptr<int>(),
            .owner_group_cursor = owner_group_cursor.data_ptr<int>(),
            .retained = retained.data_ptr<int>(),
            .moved = moved.data_ptr<int>(),
            .retained_prefix = retained_prefix.data_ptr<int>(),
            .group_prefix = group_prefix.data_ptr<int>(),
            .proxy_required = proxy_required.data_ptr<int>(),
            .path_units = path_units.data_ptr<int>(),
            .moved_copies = moved_copies.data_ptr<int>(),
            .status = status.data_ptr<int>(),
            .num_rails = num_rails,
            .num_tokens = num_tokens,
            .num_topk = num_topk,
            .num_channels = num_channels,
            .num_destinations = num_destinations,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .proxy_capacity_per_egress = proxy_capacity_per_egress,
            .planner_seed = planner_seed,
            .planner_chunk_size = planner_chunk_size,
            .two_hop_threshold_percent = two_hop_threshold_percent,
            .max_two_hop_percent = max_two_hop_percent,
            .hop_penalty_percent = hop_penalty_percent,
            .launch_args = prototype.launch_args,
        },
        stream);

    int host_status = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        &host_status, status.data_ptr<int>(), sizeof(host_status),
        cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    EP_HOST_ASSERT(
        host_status == static_cast<int>(rail_balance::HybridPlanError::Success) or
        host_status == static_cast<int>(
            rail_balance::HybridPlanError::CapacityExceeded) or
        host_status == static_cast<int>(
            rail_balance::HybridPlanError::InvalidSchedule));
    return {
        resolutions, pair_load, source_load, retained, moved,
        group_prefix, proxy_required, path_units, moved_copies, status,
    };
}

static RailBalanceHybridPlanTensors build_rail_balance_hybrid_plan(
    const torch::Tensor& topk_idx,
    const int& num_channels,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_scaleout_ranks,
    const int& local_scaleout_rank,
    const int& proxy_capacity_per_egress,
    const pybind11::object& remainder_seed,
    const pybind11::object& policy,
    const pybind11::object& threshold_percent) {
    // This private B1 entry point is deliberately strict before it allocates,
    // compiles, or launches anything.  Its stacked [G,N,K] input is only a
    // single-GPU oracle fixture. The same count runtime accepts G=1 with a
    // production-shaped local owner slice [1,N,K] in C080-B2.
    EP_HOST_ASSERT(topk_idx.dim() == 3);
    EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
    EP_HOST_ASSERT(
        topk_idx.scalar_type() ==
        c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(PyLong_CheckExact(remainder_seed.ptr()));

    const auto num_rails_i64 = topk_idx.size(0);
    const auto num_tokens_i64 = topk_idx.size(1);
    const auto num_topk_i64 = topk_idx.size(2);
    EP_HOST_ASSERT(num_rails_i64 >= 2 and num_rails_i64 <= 32);
    EP_HOST_ASSERT(num_tokens_i64 >= 0 and num_tokens_i64 <= INT_MAX);
    EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
    const int num_rails = static_cast<int>(num_rails_i64);
    const int num_tokens = static_cast<int>(num_tokens_i64);
    const int num_topk = static_cast<int>(num_topk_i64);

    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
    EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_scaleout_ranks >= 2 and
                   num_scaleout_ranks <= 32);
    EP_HOST_ASSERT(local_scaleout_rank >= 0 and
                   local_scaleout_rank < num_scaleout_ranks);
    EP_HOST_ASSERT(proxy_capacity_per_egress > 0);
    EP_HOST_ASSERT(
        num_experts % (num_scaleout_ranks * num_rails) == 0);
    EP_HOST_ASSERT(
        static_cast<int64_t>(num_rails) * num_scaleout_ranks *
            num_max_tokens_per_rank <= INT_MAX);

    // Reject bool/non-PyLong above and reject Python integers outside int64
    // here. The frozen strict oracle accepts only nonnegative seeds, then
    // normalizes them to the local rail ring.
    const int64_t remainder_seed_i64 = remainder_seed.cast<int64_t>();
    EP_HOST_ASSERT(remainder_seed_i64 >= 0);
    const int policy_value = parse_rail_balance_hybrid_policy(policy);
    const int threshold_percent_value =
        parse_rail_balance_hybrid_threshold_percent(threshold_percent);

    c10::cuda::CUDAGuard device_guard(topk_idx.device());
    const int device_index = topk_idx.get_device();
    int expected_device = -1;
    if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
            expected_device, device_index, std::memory_order_relaxed) and
        expected_device != device_index)
        EP_HOST_UNREACHABLE(
            "Hybrid rail-balance planner supports one CUDA device per process");
    const auto stream = at::cuda::getCurrentCUDAStream(
        device_index);
    const auto int_options = topk_idx.options().dtype(torch::kInt);
    const int num_destinations = num_scaleout_ranks;

    // Allocate every local output before the first device status check.  B2
    // can replace the local D2H check below with WORLD GATE #1 without adding
    // a post-gate allocation or compilation failure point.
    auto channel_count = torch::empty(
        {num_rails, num_channels, num_destinations}, int_options);
    auto count = torch::empty(
        {num_rails, num_destinations}, int_options);
    auto quota = torch::empty(
        {num_rails, num_destinations}, int_options);
    auto keep_count = torch::empty(
        {num_rails, num_destinations}, int_options);
    auto segments = torch::full(
        {num_destinations, num_rails - 1, 5}, -1, int_options);
    auto num_segments = torch::empty(
        {num_destinations}, int_options);
    auto owner_channel_prefix = torch::empty(
        {num_rails, num_channels, num_destinations}, int_options);
    auto retained = torch::empty(
        {num_rails, num_channels, num_destinations}, int_options);
    auto moved = torch::empty(
        {num_rails, num_channels, num_destinations}, int_options);
    auto moved_channel_prefix = torch::empty(
        {num_rails, num_destinations, num_channels + 1}, int_options);
    auto group_prefix = torch::empty(
        {num_rails, num_channels, num_destinations}, int_options);
    auto proxy_required = torch::empty({num_rails}, int_options);
    auto moved_copies = torch::zeros({1}, int_options);
    auto status = torch::zeros({1}, int_options);

    // All three cubins are built before validation.  No JIT or allocation is
    // allowed between a successful validation gate and a future collective.
    const auto prepared = prepare_rail_balance_hybrid_plan(
        num_rails, num_rails, num_channels, num_destinations);
    launch_prepared_rail_balance_hybrid_count(
        prepared, topk_idx.data_ptr<topk_idx_t>(),
        channel_count.data_ptr<int>(), status.data_ptr<int>(),
        num_rails, num_tokens, num_topk, num_channels,
        num_experts, num_destinations, local_scaleout_rank, stream);

    int host_status = 0;
    CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
        &host_status, status.data_ptr<int>(), sizeof(host_status),
        cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));
    if (host_status == 2) {
        throw EPExceptionWithLineInfo(
            "Rail balance Hybrid plan",
            "topk_idx contains a masked or out-of-range expert id");
    }
    if (host_status == 3) {
        throw EPExceptionWithLineInfo(
            "Rail balance Hybrid plan",
            "topk_idx contains duplicate expert ids within one token");
    }
    EP_HOST_ASSERT(host_status == 0);

    const int normalized_remainder_seed = static_cast<int>(
        remainder_seed_i64 % num_rails);
    launch_prepared_rail_balance_hybrid_plan(
        prepared, channel_count.data_ptr<int>(), count.data_ptr<int>(),
        quota.data_ptr<int>(), keep_count.data_ptr<int>(),
        segments.data_ptr<int>(), num_segments.data_ptr<int>(),
        num_rails, num_channels, num_destinations,
        normalized_remainder_seed,
        policy_value, threshold_percent_value, stream);
    launch_prepared_rail_balance_hybrid_prefix(
        prepared, channel_count.data_ptr<int>(), quota.data_ptr<int>(),
        keep_count.data_ptr<int>(), owner_channel_prefix.data_ptr<int>(),
        retained.data_ptr<int>(), moved.data_ptr<int>(),
        moved_channel_prefix.data_ptr<int>(), group_prefix.data_ptr<int>(),
        proxy_required.data_ptr<int>(), moved_copies.data_ptr<int>(),
        status.data_ptr<int>(), num_rails, num_channels,
        num_destinations, num_max_tokens_per_rank,
        proxy_capacity_per_egress, stream);

    return {
        channel_count, count, quota, keep_count,
        segments, num_segments, owner_channel_prefix,
        retained, moved, moved_channel_prefix, group_prefix,
        proxy_required, moved_copies, status,
    };
}

static void register_rail_balance_hybrid_plan_apis(pybind11::module_& m) {
    m.def(
        "_build_rail_balance_hop_records",
        &build_rail_balance_hop_records,
        pybind11::arg("topk_idx"),
        pybind11::arg("num_channels"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_destinations"),
        pybind11::arg("local_destination"));
    m.def(
        "_build_rail_balance_hop_one_hop_plan",
        &build_rail_balance_hop_plan,
        pybind11::arg("records"),
        pybind11::arg("num_channels"),
        pybind11::arg("num_destinations"),
        pybind11::arg("num_max_tokens_per_rank"),
        pybind11::arg("proxy_capacity_per_egress"),
        pybind11::arg("planner_seed") = 0,
        pybind11::arg("two_hop_threshold_percent") = 0,
        pybind11::arg("max_two_hop_percent") = 0,
        pybind11::arg("hop_penalty_percent") = 0,
        pybind11::arg("planner_chunk_size") = 1);
    m.def(
        "_build_rail_balance_hybrid_plan",
        &build_rail_balance_hybrid_plan,
        pybind11::arg("topk_idx"),
        pybind11::arg("num_channels"),
        pybind11::arg("num_max_tokens_per_rank"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_scaleout_ranks"),
        pybind11::arg("local_scaleout_rank"),
        pybind11::arg("proxy_capacity_per_egress"),
        pybind11::arg("remainder_seed") = pybind11::int_(0),
        pybind11::arg("policy") =
            static_cast<int>(rail_balance::HybridPolicy::All),
        pybind11::arg("threshold_percent") = 0);
    m.def(
        "_rail_balance_hybrid_dispatch_codegen_test",
        &rail_balance_hybrid_dispatch_codegen_test,
        pybind11::arg("num_scaleout_ranks"),
        pybind11::arg("num_scaleup_ranks"),
        pybind11::arg("hidden"),
        pybind11::arg("num_topk"),
        pybind11::arg("num_sms") = 64,
        pybind11::arg("num_channels_per_sm") = 4,
        pybind11::arg("proxy_capacity") = 16,
        pybind11::arg("num_experts") = 256);
}

}  // namespace deep_ep::elastic
