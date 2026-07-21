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

struct RailBalanceHybridReturnUnshuffleSpec {
    int hidden;
    int num_topk;
    int num_channels;
};

struct PreparedRailBalanceHybridReturnUnshuffle {
    std::shared_ptr<jit::KernelRuntime> runtime;
    RailBalanceHybridReturnUnshuffleSpec spec;
    jit::LaunchArgs launch_args;
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

static PreparedRailBalanceHybridReturnUnshuffle
prepare_rail_balance_hybrid_return_unshuffle(
    const int& hidden,
    const int& num_topk,
    const int& num_channels) {
    EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    EP_HOST_ASSERT(
        static_cast<int64_t>(hidden) * sizeof(__nv_bfloat16) <= INT_MAX);
    EP_HOST_ASSERT(num_topk >= 1 and num_topk <= 32);
    EP_HOST_ASSERT(num_channels >= 1 and num_channels <= kNumMaxChannels);
    const RailBalanceHybridReturnUnshuffleSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
        .num_channels = num_channels,
    };
    const auto token_layout = layout::TokenLayout(
        hidden * sizeof(__nv_bfloat16), 0, num_topk, false);
    const int64_t num_smem_bytes_i64 = rail_balance::checked_add_i64(
        token_layout.get_num_bytes<false, int64_t>(),
        ptx::kNumTMAAlignBytes);
    EP_HOST_ASSERT(num_smem_bytes_i64 <= INT_MAX);
    const int num_smem_bytes = static_cast<int>(num_smem_bytes_i64);
    EP_HOST_ASSERT(
        num_smem_bytes <= jit::device_runtime->get_num_smem_bytes());
    const auto launch_args = jit::LaunchArgs(
        num_channels, 32, num_smem_bytes);
    return {
        .runtime = jit::compiler->build(
            "rail_balance_hybrid_return_unshuffle",
            RailBalanceHybridReturnUnshuffleRuntime::generate(spec)),
        .spec = spec,
        .launch_args = launch_args,
    };
}

// legacy_reduce_buffer is the base of Hybrid combine's scaleout receive
// BufferLayout (not the base of the preceding scaleup buffer). The caller must
// run the existing scaleup-team barrier after this kernel and before launching
// the legacy reduce epilogue.
//
// This raw submit belongs inside the committed combine -> return-unshuffle ->
// local-barrier -> epilogue sequence. Every pointer and dynamic integer must be
// validated/captured before the sequence begins; this body only binds the
// already-prepared ABI and submits it on the caller-owned stream.
static void submit_prepared_rail_balance_hybrid_return_unshuffle(
    const PreparedRailBalanceHybridReturnUnshuffle& prepared,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* arena,
    void* legacy_reduce_buffer,
    const int* moved,
    const int* group_prefix,
    const int* proxy_required,
    int* status,
    const int& num_experts,
    const int& num_destinations,
    const int& num_rails,
    const int& egress,
    const int& current_rank_idx,
    const int& num_max_tokens_per_rank,
    const int& proxy_capacity,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceHybridReturnUnshuffleRuntime::Args args = {
        .hidden = prepared.spec.hidden,
        .num_topk = prepared.spec.num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .arena = arena,
        .legacy_reduce_buffer = legacy_reduce_buffer,
        .moved = moved,
        .group_prefix = group_prefix,
        .proxy_required = proxy_required,
        .status = status,
        .num_experts = num_experts,
        .num_destinations = num_destinations,
        .num_rails = num_rails,
        .egress = egress,
        .current_rank_idx = current_rank_idx,
        .num_channels = prepared.spec.num_channels,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .proxy_capacity = proxy_capacity,
        .launch_args = prepared.launch_args,
    };
    RailBalanceHybridReturnUnshuffleRuntime::launch(
        prepared.runtime, args, stream);
}

// Checked wrapper retained for the existing private standalone test path.
// Production committed code must capture raw tensor pointers before its first
// barrier and call submit_prepared_... directly.
static void launch_prepared_rail_balance_hybrid_return_unshuffle(
    const PreparedRailBalanceHybridReturnUnshuffle& prepared,
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
    EP_HOST_ASSERT(prepared.runtime != nullptr);
    EP_HOST_ASSERT(prepared.spec.hidden == hidden);
    EP_HOST_ASSERT(prepared.spec.num_topk == num_topk);
    EP_HOST_ASSERT(prepared.spec.num_channels == num_channels);
    submit_prepared_rail_balance_hybrid_return_unshuffle(
        prepared, nccl_dev_comm, nccl_window,
        arena, legacy_reduce_buffer,
        plan.moved.data_ptr<int>(),
        plan.group_prefix.data_ptr<int>(),
        plan.proxy_required.data_ptr<int>(),
        plan.status.data_ptr<int>(),
        num_experts, num_destinations, num_rails,
        egress, current_rank_idx,
        num_max_tokens_per_rank, proxy_capacity, stream);
}

}  // namespace deep_ep::elastic
