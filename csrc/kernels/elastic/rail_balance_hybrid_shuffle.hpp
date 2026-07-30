#pragma once

#include <climits>
#include <memory>

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
    int num_channels;
    int num_tokens;
    bool hop_aware;
};

// Everything which depends on the immutable token/channel geometry is frozen
// before the force epoch starts.  The committed submit path below therefore
// does not construct a TokenLayout or derive launch geometry.
struct PreparedRailBalanceHybridSourceShuffle {
    std::shared_ptr<jit::KernelRuntime> runtime;
    RailBalanceHybridSourceShuffleSpec spec;
    jit::LaunchArgs launch_args;
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
        const rail_balance::HopCopyRecord* hop_records;
        const rail_balance::HopCopyResolution* hop_resolutions;
        int* status;
        int num_tokens;
        int num_experts;
        int num_destinations;
        int local_destination;
        int num_rails;
        int owner;
        int num_channels;
        int num_max_tokens_per_rank;
        int hop_record_token_capacity;
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
        &rail_balance_hybrid_source_shuffle_impl<{}, {}, false>);
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
            args.group_prefix, args.proxy_required,
            args.hop_records, args.hop_resolutions, args.status,
            args.num_tokens, args.num_experts,
            args.num_destinations, args.local_destination,
            args.num_rails, args.owner, args.num_channels,
            args.num_max_tokens_per_rank, args.hop_record_token_capacity,
            args.rank_idx,
            args.proxy_capacity));
    }
};

class RailBalanceHopSourceShuffleRuntime final:
    public jit::LaunchRuntime<RailBalanceHopSourceShuffleRuntime> {
public:
    using Args = RailBalanceHybridSourceShuffleRuntime::Args;

    template <typename Spec>
    static std::string generate_impl(const Spec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_shuffle.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_source_shuffle_impl<{}, {}, true>);
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
            args.group_prefix, args.proxy_required,
            args.hop_records, args.hop_resolutions, args.status,
            args.num_tokens, args.num_experts,
            args.num_destinations, args.local_destination,
            args.num_rails, args.owner, args.num_channels,
            args.num_max_tokens_per_rank, args.hop_record_token_capacity,
            args.rank_idx, args.proxy_capacity));
    }
};

static PreparedRailBalanceHybridSourceShuffle
prepare_rail_balance_hybrid_source_shuffle(
    const int& hidden,
    const int& num_topk,
    const int& num_channels,
    const int& num_tokens,
    const bool& hop_aware = false) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0 and
                   hidden <= INT_MAX /
                       static_cast<int>(sizeof(__nv_bfloat16)));
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_channels >= 1 and
                   num_channels <= rail_balance::kNumHybridMaxChannels);
    EP_HOST_ASSERT(num_tokens >= 0);
    const RailBalanceHybridSourceShuffleSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
        .num_channels = num_channels,
        .num_tokens = num_tokens,
        .hop_aware = hop_aware,
    };
    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(__nv_bfloat16), 0, num_topk, true);
    const int64_t num_smem_bytes_i64 = rail_balance::checked_add_i64(
        token_layout.get_num_bytes<false, int64_t>(),
        ptx::kNumTMAAlignBytes);
    EP_HOST_ASSERT(num_smem_bytes_i64 <= INT_MAX);
    const int num_smem_bytes = static_cast<int>(num_smem_bytes_i64);
    EP_HOST_ASSERT(
        num_smem_bytes <= jit::device_runtime->get_num_smem_bytes());
    // Channel c is empty when c >= num_tokens because token traversal starts
    // at c and advances by num_channels. Keep one block for N=0 so the strict
    // checked path retains its launch/error coverage.
    const int num_active_channels = num_tokens <= 0 ? 1 :
        (num_tokens < num_channels ? num_tokens : num_channels);
    return {
        .runtime = jit::compiler->build(
            hop_aware ? "rail_balance_hop_source_shuffle" :
                        "rail_balance_hybrid_source_shuffle",
            hop_aware ? RailBalanceHopSourceShuffleRuntime::generate(spec) :
                        RailBalanceHybridSourceShuffleRuntime::generate(spec)),
        .spec = spec,
        .launch_args = jit::LaunchArgs(
            num_active_channels, 32, num_smem_bytes),
    };
}

// Committed-stage adapter. Gate #2 has already validated every schedule value
// and pointer range. Keep this function to Args construction plus submission:
// no Tensor access, validation, layout work, JIT, allocation, status readback,
// synchronization, or source-local barrier is allowed here.
static void submit_prepared_rail_balance_hybrid_source_shuffle(
    const PreparedRailBalanceHybridSourceShuffle& prepared,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    const void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    void* arena,
    const int* owner_channel_prefix,
    const int* keep_count,
    const int* segments,
    const int* num_segments,
    const int* retained,
    const int* moved_channel_prefix,
    const int* group_prefix,
    const int* proxy_required,
    int* status,
    const int& num_experts,
    const int& num_destinations,
    const int& local_destination,
    const int& num_rails,
    const int& owner,
    const int& num_max_tokens_per_rank,
    const int& rank_idx,
    const int& proxy_capacity,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridSourceShuffleRuntime::Args args = {
        .hidden = prepared.spec.hidden,
        .num_topk = prepared.spec.num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .arena = arena,
        .owner_channel_prefix = owner_channel_prefix,
        .keep_count = keep_count,
        .segments = segments,
        .num_segments = num_segments,
        .retained = retained,
        .moved_channel_prefix = moved_channel_prefix,
        .group_prefix = group_prefix,
        .proxy_required = proxy_required,
        .hop_records = nullptr,
        .hop_resolutions = nullptr,
        .status = status,
        .num_tokens = prepared.spec.num_tokens,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .local_destination = local_destination,
        .num_rails = num_rails,
        .owner = owner,
        .num_channels = prepared.spec.num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hop_record_token_capacity = 0,
        .rank_idx = rank_idx,
        .proxy_capacity = proxy_capacity,
        .launch_args = prepared.launch_args,
    };
    RailBalanceHybridSourceShuffleRuntime::launch(
        prepared.runtime, args, stream);
}

// Compatibility layer for the existing strict LSA tests. It retains the
// owning-plan ABI, verifies that invocation geometry matches prepare, then
// unwraps tensor pointers before crossing into the raw submit layer.
static void launch_prepared_rail_balance_hybrid_source_shuffle(
    const PreparedRailBalanceHybridSourceShuffle& prepared,
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
    EP_HOST_ASSERT(prepared.runtime != nullptr);
    EP_HOST_ASSERT(prepared.spec.hidden == hidden);
    EP_HOST_ASSERT(prepared.spec.num_topk == num_topk);
    EP_HOST_ASSERT(prepared.spec.num_channels == num_channels);
    EP_HOST_ASSERT(prepared.spec.num_tokens == num_tokens);
    EP_HOST_ASSERT(not prepared.spec.hop_aware);
    submit_prepared_rail_balance_hybrid_source_shuffle(
        prepared, nccl_dev_comm, nccl_window,
        x, topk_idx, topk_weights, arena,
        plan.owner_channel_prefix.data_ptr<int>(),
        plan.keep_count.data_ptr<int>(),
        plan.segments.data_ptr<int>(),
        plan.num_segments.data_ptr<int>(),
        plan.retained.data_ptr<int>(),
        plan.moved_channel_prefix.data_ptr<int>(),
        plan.group_prefix.data_ptr<int>(),
        plan.proxy_required.data_ptr<int>(),
        plan.status.data_ptr<int>(),
        num_experts,
        num_destinations, local_destination,
        num_rails, owner, num_max_tokens_per_rank,
        rank_idx, proxy_capacity, stream);
}

static void launch_prepared_rail_balance_hop_source_shuffle(
    const PreparedRailBalanceHybridSourceShuffle& prepared,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    const void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    void* arena,
    const rail_balance::HopCopyRecord* records,
    const rail_balance::HopCopyResolution* resolutions,
    const int* retained,
    const int* group_prefix,
    const int* proxy_required,
    int* status,
    const int& num_experts,
    const int& num_destinations,
    const int& local_destination,
    const int& num_rails,
    const int& owner,
    const int& num_max_tokens_per_rank,
    const int& rank_idx,
    const int& proxy_capacity,
    const at::cuda::CUDAStream& stream) {
    EP_HOST_ASSERT(prepared.spec.hop_aware);
    RailBalanceHopSourceShuffleRuntime::launch(
        prepared.runtime,
        RailBalanceHopSourceShuffleRuntime::Args{
            .hidden = prepared.spec.hidden,
            .num_topk = prepared.spec.num_topk,
            .nccl_dev_comm = nccl_dev_comm,
            .nccl_window = nccl_window,
            .x = x,
            .topk_idx = topk_idx,
            .topk_weights = topk_weights,
            .arena = arena,
            .owner_channel_prefix = nullptr,
            .keep_count = nullptr,
            .segments = nullptr,
            .num_segments = nullptr,
            .retained = retained,
            .moved_channel_prefix = nullptr,
            .group_prefix = group_prefix,
            .proxy_required = proxy_required,
            .hop_records = records,
            .hop_resolutions = resolutions,
            .status = status,
            .num_tokens = prepared.spec.num_tokens,
            .num_experts = num_experts,
            .num_destinations = num_destinations,
            .local_destination = local_destination,
            .num_rails = num_rails,
            .owner = owner,
            .num_channels = prepared.spec.num_channels,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .hop_record_token_capacity = num_max_tokens_per_rank,
            .rank_idx = rank_idx,
            .proxy_capacity = proxy_capacity,
            .launch_args = prepared.launch_args,
        },
        stream);
}

}  // namespace deep_ep::elastic
