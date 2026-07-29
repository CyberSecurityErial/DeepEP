#pragma once

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class RailBalancePlanRuntime final : public jit::LaunchRuntime<RailBalancePlanRuntime> {
public:
    struct Args {
        const int* counts;
        int* quota;
        int* keep_count;
        int* segments;
        int* num_segments;
        int* moved_copies;
        int num_rails;
        int num_destinations;
        int remainder_seed_mod;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance_plan.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&rail_balance_plan_impl);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.counts,
            args.quota,
            args.keep_count,
            args.segments,
            args.num_segments,
            args.moved_copies,
            args.num_rails,
            args.num_destinations,
            args.remainder_seed_mod));
    }
};

static void launch_rail_balance_plan(
    const int* counts,
    int* quota,
    int* keep_count,
    int* segments,
    int* num_segments,
    int* moved_copies,
    const int& num_rails,
    const int& num_destinations,
    const int& remainder_seed_mod,
    const at::cuda::CUDAStream& stream
) {
    const RailBalancePlanRuntime::Args args = {
        .counts = counts,
        .quota = quota,
        .keep_count = keep_count,
        .segments = segments,
        .num_segments = num_segments,
        .moved_copies = moved_copies,
        .num_rails = num_rails,
        .num_destinations = num_destinations,
        .remainder_seed_mod = remainder_seed_mod,
        .launch_args = jit::LaunchArgs(num_destinations, 32),
    };
    const auto code = RailBalancePlanRuntime::generate(args);
    const auto runtime = jit::compiler->build("rail_balance_plan", code);
    RailBalancePlanRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
