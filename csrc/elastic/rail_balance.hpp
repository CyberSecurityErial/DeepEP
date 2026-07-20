#pragma once

#include <atomic>
#include <climits>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>

#include <deep_ep/common/exception.cuh>

#include "../kernels/elastic/rail_balance_plan.hpp"

namespace deep_ep::elastic {

using RailBalancePlanTensors = std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>;

// DeepEP binds one rank/process to one GPU.  KernelRuntime owns a CUmodule
// handle for the CUDA context in which it was loaded, so reject accidental
// cross-device reuse explicitly instead of surfacing CUDA_ERROR_INVALID_HANDLE.
static std::atomic<int> rail_balance_plan_process_device{-1};

static RailBalancePlanTensors build_rail_balance_plan(
    const torch::Tensor& counts,
    const pybind11::object& remainder_seed
) {
    EP_HOST_ASSERT(counts.is_cuda());
    EP_HOST_ASSERT(counts.dim() == 2);
    EP_HOST_ASSERT(counts.is_contiguous());
    EP_HOST_ASSERT(counts.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(PyLong_CheckExact(remainder_seed.ptr()));

    const auto num_rails_i64 = counts.size(0);
    const auto num_destinations_i64 = counts.size(1);
    EP_HOST_ASSERT(num_rails_i64 >= 1 && num_rails_i64 <= 32);
    EP_HOST_ASSERT(num_destinations_i64 <= INT_MAX);
    const int num_rails = static_cast<int>(num_rails_i64);
    const int num_destinations = static_cast<int>(num_destinations_i64);

    // Convert through int64 so out-of-range Python integers fail at the ABI
    // boundary, then normalize explicitly to Python's nonnegative modulo.
    const int64_t remainder_seed_i64 = remainder_seed.cast<int64_t>();
    const int remainder_seed_mod = static_cast<int>(
        (remainder_seed_i64 % num_rails + num_rails) % num_rails);

    const c10::cuda::CUDAGuard device_guard(counts.device());
    auto quota = torch::empty_like(counts);
    auto keep_count = torch::empty_like(counts);
    auto segments = torch::full(
        {num_destinations_i64, num_rails_i64 - 1, 4}, -1, counts.options());
    auto num_segments = torch::zeros({num_destinations_i64}, counts.options());
    auto moved_copies = torch::zeros({1}, counts.options());

    // Empty destination sets are a host fast path: no JIT compile or launch.
    if (num_destinations == 0)
        return {quota, keep_count, segments, num_segments, moved_copies};

    // Value constraints are part of this private ABI.  Validate on the tensor's
    // current device/stream before launching so malformed input fails at the API
    // boundary instead of surfacing later as an asynchronous device trap.
    EP_HOST_ASSERT(counts.min().item<int>() >= 0);
    const auto total_copies = counts.to(torch::kInt64).sum().item<int64_t>();
    EP_HOST_ASSERT(total_copies <= INT_MAX);

    const int device_index = counts.get_device();
    int expected_device = -1;
    if (not rail_balance_plan_process_device.compare_exchange_strong(
            expected_device, device_index, std::memory_order_relaxed) and
        expected_device != device_index)
        EP_HOST_UNREACHABLE(
            "rail-balance planner supports one CUDA device per process");

    launch_rail_balance_plan(
        counts.data_ptr<int>(),
        quota.data_ptr<int>(),
        keep_count.data_ptr<int>(),
        segments.data_ptr<int>(),
        num_segments.data_ptr<int>(),
        moved_copies.data_ptr<int>(),
        num_rails,
        num_destinations,
        remainder_seed_mod,
        at::cuda::getCurrentCUDAStream());
    return {quota, keep_count, segments, num_segments, moved_copies};
}

static void register_rail_balance_apis(pybind11::module_& m) {
    // The public force path stays unavailable until both Hybrid dispatch and
    // combine are installed.  Python also owns an independent host capability,
    // so a partially upgraded installation fails closed in either direction.
    m.def("_rail_balance_force_available", []() noexcept { return false; });
    m.def(
        "_build_rail_balance_plan",
        &build_rail_balance_plan,
        pybind11::arg("counts"),
        pybind11::arg("remainder_seed") = pybind11::int_(0));
}

}  // namespace deep_ep::elastic
