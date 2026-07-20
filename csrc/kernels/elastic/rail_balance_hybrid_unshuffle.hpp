#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "rail_balance_hybrid_dispatch.hpp"

namespace deep_ep::elastic {

struct RailBalanceHybridReturnUnshuffleSpec {
    int hidden;
    int num_topk;
};

class RailBalanceHybridReturnUnshuffleRuntime final:
    public jit::LaunchRuntime<RailBalanceHybridReturnUnshuffleRuntime> {
public:
    struct Args {
        int hidden;
        int num_topk;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* arena;
        void* legacy_reduce_buffer;
        const int* moved;
        const int* group_prefix;
        const int* proxy_required;
        int* status;
        int num_experts;
        int num_destinations;
        int num_rails;
        int egress;
        int current_rank_idx;
        int num_channels;
        int num_max_tokens_per_rank;
        int proxy_capacity;
        jit::LaunchArgs launch_args;
    };

    template <typename Spec>
    static std::string generate_impl(const Spec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_hybrid_unshuffle.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_hybrid_return_unshuffle_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.arena, args.legacy_reduce_buffer,
            args.moved, args.group_prefix,
            args.proxy_required, args.status,
            args.num_experts, args.num_destinations,
            args.num_rails, args.egress, args.current_rank_idx,
            args.num_channels,
            args.num_max_tokens_per_rank, args.proxy_capacity));
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_hybrid_return_unshuffle(
    const int& hidden,
    const int& num_topk) {
    const RailBalanceHybridReturnUnshuffleSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    return jit::compiler->build(
        "rail_balance_hybrid_return_unshuffle",
        RailBalanceHybridReturnUnshuffleRuntime::generate(spec));
}

// legacy_reduce_buffer is the base of Hybrid combine's scaleout receive
// BufferLayout (not the base of the preceding scaleup buffer). The caller must
// run the existing scaleup-team barrier after this kernel and before launching
// the legacy reduce epilogue.
static void launch_prepared_rail_balance_hybrid_return_unshuffle(
    const std::shared_ptr<jit::KernelRuntime>& runtime,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* arena,
    void* legacy_reduce_buffer,
    const RailBalanceHybridPlanOutputs& plan,
    const int& hidden,
    const int& num_topk,
    const int& num_experts,
    const int& num_destinations,
    const int& num_rails,
    const int& egress,
    const int& current_rank_idx,
    const int& num_channels,
    const int& num_max_tokens_per_rank,
    const int& proxy_capacity,
    const at::cuda::CUDAStream& stream) {
    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(__nv_bfloat16), 0, num_topk, false);
    const int num_smem_bytes = token_layout.get_num_bytes<false>() +
        ptx::kNumTMAAlignBytes;
    const RailBalanceHybridReturnUnshuffleRuntime::Args args = {
        .hidden = hidden,
        .num_topk = num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .arena = arena,
        .legacy_reduce_buffer = legacy_reduce_buffer,
        .moved = plan.moved.data_ptr<int>(),
        .group_prefix = plan.group_prefix.data_ptr<int>(),
        .proxy_required = plan.proxy_required.data_ptr<int>(),
        .status = plan.status.data_ptr<int>(),
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .num_rails = num_rails,
        .egress = egress,
        .current_rank_idx = current_rank_idx,
        .num_channels = num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .proxy_capacity = proxy_capacity,
        .launch_args = jit::LaunchArgs(
            num_channels, 32, num_smem_bytes),
    };
    RailBalanceHybridReturnUnshuffleRuntime::launch(
        runtime, args, stream);
}

}  // namespace deep_ep::elastic
