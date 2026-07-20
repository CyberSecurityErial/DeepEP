#pragma once

#include <atomic>
#include <climits>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

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
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_plan_impl<0>);
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
            args.num_destinations, args.remainder_seed));
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

struct PreparedRailBalanceHybridPlan {
    std::shared_ptr<jit::KernelRuntime> count;
    std::shared_ptr<jit::KernelRuntime> plan;
    std::shared_ptr<jit::KernelRuntime> prefix;
};

static PreparedRailBalanceHybridPlan prepare_rail_balance_hybrid_plan() {
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
            "rail_balance_hybrid_plan_v1",
            RailBalanceHybridPlanRuntime::generate(plan_args)),
        .prefix = jit::compiler->build(
            "rail_balance_hybrid_prefix_v1",
            RailBalanceHybridPrefixRuntime::generate(prefix_args)),
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
        .launch_args = jit::LaunchArgs(num_owners * num_channels, 32),
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
        .launch_args = jit::LaunchArgs(num_destinations, 32),
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
        .launch_args = jit::LaunchArgs(num_rails, 32),
    };
    RailBalanceHybridPrefixRuntime::launch(prepared.prefix, args, stream);
}

static RailBalanceHybridPlanTensors build_rail_balance_hybrid_plan(
    const torch::Tensor& topk_idx,
    const int& num_channels,
    const int& num_max_tokens_per_rank,
    const int& num_experts,
    const int& num_scaleout_ranks,
    const int& local_scaleout_rank,
    const int& proxy_capacity_per_egress,
    const pybind11::object& remainder_seed) {
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
                   num_scaleout_ranks <= 32 and
                   num_scaleout_ranks <= num_topk);
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
    const auto prepared = prepare_rail_balance_hybrid_plan();
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
        normalized_remainder_seed, stream);
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
        "_build_rail_balance_hybrid_plan",
        &build_rail_balance_hybrid_plan,
        pybind11::arg("topk_idx"),
        pybind11::arg("num_channels"),
        pybind11::arg("num_max_tokens_per_rank"),
        pybind11::arg("num_experts"),
        pybind11::arg("num_scaleout_ranks"),
        pybind11::arg("local_scaleout_rank"),
        pybind11::arg("proxy_capacity_per_egress"),
        pybind11::arg("remainder_seed") = pybind11::int_(0));
}

}  // namespace deep_ep::elastic
