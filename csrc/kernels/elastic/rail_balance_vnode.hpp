#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

enum class RailBalanceVNodePeerKind {
    Scaleout,
    Return,
    Unshuffle,
};

enum class RailBalanceVNodeExpertKind {
    Forward,
    Expert,
};

struct RailBalanceVNodeSpec {
    int hidden;
    int num_topk;
};

struct RailBalanceVNodePeerSpec: RailBalanceVNodeSpec {
    RailBalanceVNodePeerKind kind;
};

struct RailBalanceVNodeExpertSpec: RailBalanceVNodeSpec {
    RailBalanceVNodeExpertKind kind;
};

class RailBalanceVNodePeerRuntime final:
    public jit::LaunchRuntime<RailBalanceVNodePeerRuntime> {
public:
    struct Args {
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* arena;
        int* status;
        const int* quota;
        int num_destinations;
        int destination_capacity;
        int num_source_ranks;
        int num_max_tokens;
        int generation;
        int rank_idx;
        int expert_begin;
        int experts_per_rank;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(
        const RailBalanceVNodePeerSpec& spec) {
        const char* kernel_name = nullptr;
        switch (spec.kind) {
            case RailBalanceVNodePeerKind::Scaleout:
                kernel_name = "rail_balance_vnode_scaleout_impl";
                break;
            case RailBalanceVNodePeerKind::Return:
                kernel_name = "rail_balance_vnode_return_impl";
                break;
            case RailBalanceVNodePeerKind::Unshuffle:
                kernel_name = "rail_balance_vnode_unshuffle_impl";
                break;
        }
        EP_HOST_ASSERT(kernel_name != nullptr);
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_vnode.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<{}, {}>);
}}
)", kernel_name, spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.arena, args.status,
            args.quota, args.num_destinations, args.destination_capacity,
            args.num_source_ranks, args.num_max_tokens,
            args.generation, args.rank_idx,
            args.expert_begin, args.experts_per_rank));
    }
};

class RailBalanceVNodeExpertRuntime final:
    public jit::LaunchRuntime<RailBalanceVNodeExpertRuntime> {
public:
    struct Args {
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* arena;
        int* status;
        const int* quota;
        int num_destinations;
        int destination_capacity;
        int num_source_ranks;
        int num_max_tokens;
        int generation;
        int rank_idx;
        int expert_begin;
        int experts_per_rank;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(
        const RailBalanceVNodeExpertSpec& spec) {
        const char* kernel_name = nullptr;
        switch (spec.kind) {
            case RailBalanceVNodeExpertKind::Forward:
                kernel_name = "rail_balance_vnode_forward_impl";
                break;
            case RailBalanceVNodeExpertKind::Expert:
                kernel_name = "rail_balance_vnode_expert_impl";
                break;
        }
        EP_HOST_ASSERT(kernel_name != nullptr);
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_vnode.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{}<{}, {}>);
}}
)", kernel_name, spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.arena, args.status,
            args.quota, args.num_destinations, args.destination_capacity,
            args.num_source_ranks, args.num_max_tokens,
            args.generation, args.rank_idx,
            args.expert_begin, args.experts_per_rank));
    }
};

class RailBalanceVNodeReduceRuntime final:
    public jit::LaunchRuntime<RailBalanceVNodeReduceRuntime> {
public:
    struct Args {
        const void* arena;
        const topk_idx_t* topk_idx;
        nv_bfloat16* output;
        int* output_ready;
        int* status;
        int num_destinations;
        int destination_capacity;
        int num_source_ranks;
        int num_max_tokens;
        int generation;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const RailBalanceVNodeSpec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_vnode.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_vnode_owner_reduce_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.arena, args.topk_idx,
            args.output, args.output_ready, args.status,
            args.num_destinations, args.destination_capacity,
            args.num_source_ranks,
            args.num_max_tokens, args.generation));
    }
};

struct PreparedRailBalanceVNode {
    std::shared_ptr<jit::KernelRuntime> scaleout;
    std::shared_ptr<jit::KernelRuntime> forward;
    std::shared_ptr<jit::KernelRuntime> expert;
    std::shared_ptr<jit::KernelRuntime> return_path;
    std::shared_ptr<jit::KernelRuntime> unshuffle;
    std::shared_ptr<jit::KernelRuntime> reduce;
};

static PreparedRailBalanceVNode prepare_rail_balance_vnode(
    const int& hidden,
    const int& num_topk) {
    const RailBalanceVNodeSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    const auto build_peer = [&](const char* name,
                                const RailBalanceVNodePeerKind kind) {
        const RailBalanceVNodePeerSpec peer_spec = {
            {hidden, num_topk},
            kind,
        };
        return jit::compiler->build(
            name, RailBalanceVNodePeerRuntime::generate(peer_spec));
    };
    const auto build_expert = [&](const char* name,
                                  const RailBalanceVNodeExpertKind kind) {
        const RailBalanceVNodeExpertSpec expert_spec = {
            {hidden, num_topk},
            kind,
        };
        return jit::compiler->build(
            name, RailBalanceVNodeExpertRuntime::generate(expert_spec));
    };
    return {
        .scaleout = build_peer(
            "rail_balance_vnode_scaleout",
            RailBalanceVNodePeerKind::Scaleout),
        .forward = build_expert(
            "rail_balance_vnode_forward",
            RailBalanceVNodeExpertKind::Forward),
        .expert = build_expert(
            "rail_balance_vnode_expert",
            RailBalanceVNodeExpertKind::Expert),
        .return_path = build_peer(
            "rail_balance_vnode_return",
            RailBalanceVNodePeerKind::Return),
        .unshuffle = build_peer(
            "rail_balance_vnode_unshuffle",
            RailBalanceVNodePeerKind::Unshuffle),
        .reduce = jit::compiler->build(
            "rail_balance_vnode_owner_reduce",
            RailBalanceVNodeReduceRuntime::generate(spec)),
    };
}

static void launch_prepared_rail_balance_vnode_peer(
    const std::shared_ptr<jit::KernelRuntime>& runtime,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int& num_destinations,
    const int& destination_capacity,
    const int& num_source_ranks,
    const int& num_max_tokens,
    const int& generation,
    const int& rank_idx,
    const int& expert_begin,
    const int& experts_per_rank,
    const int& num_blocks,
    const int& num_smem_bytes,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceVNodePeerRuntime::Args args = {
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .arena = arena,
        .status = status,
        .quota = quota,
        .num_destinations = num_destinations,
        .destination_capacity = destination_capacity,
        .num_source_ranks = num_source_ranks,
        .num_max_tokens = num_max_tokens,
        .generation = generation,
        .rank_idx = rank_idx,
        .expert_begin = expert_begin,
        .experts_per_rank = experts_per_rank,
        .launch_args = jit::LaunchArgs(
            num_blocks, 32, num_smem_bytes),
    };
    RailBalanceVNodePeerRuntime::launch(runtime, args, stream);
}

static void launch_prepared_rail_balance_vnode_expert(
    const std::shared_ptr<jit::KernelRuntime>& runtime,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int& num_destinations,
    const int& destination_capacity,
    const int& num_source_ranks,
    const int& num_max_tokens,
    const int& generation,
    const int& rank_idx,
    const int& expert_begin,
    const int& experts_per_rank,
    const int& num_blocks,
    const int& num_smem_bytes,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceVNodeExpertRuntime::Args args = {
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .arena = arena,
        .status = status,
        .quota = quota,
        .num_destinations = num_destinations,
        .destination_capacity = destination_capacity,
        .num_source_ranks = num_source_ranks,
        .num_max_tokens = num_max_tokens,
        .generation = generation,
        .rank_idx = rank_idx,
        .expert_begin = expert_begin,
        .experts_per_rank = experts_per_rank,
        .launch_args = jit::LaunchArgs(
            num_blocks, 32, num_smem_bytes),
    };
    RailBalanceVNodeExpertRuntime::launch(runtime, args, stream);
}

static void launch_prepared_rail_balance_vnode_reduce(
    const PreparedRailBalanceVNode& prepared,
    const void* arena,
    const topk_idx_t* topk_idx,
    nv_bfloat16* output,
    int* output_ready,
    int* status,
    const int& num_destinations,
    const int& destination_capacity,
    const int& num_source_ranks,
    const int& num_max_tokens,
    const int& generation,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceVNodeReduceRuntime::Args args = {
        .arena = arena,
        .topk_idx = topk_idx,
        .output = output,
        .output_ready = output_ready,
        .status = status,
        .num_destinations = num_destinations,
        .destination_capacity = destination_capacity,
        .num_source_ranks = num_source_ranks,
        .num_max_tokens = num_max_tokens,
        .generation = generation,
        .launch_args = jit::LaunchArgs(num_max_tokens, 32),
    };
    RailBalanceVNodeReduceRuntime::launch(
        prepared.reduce, args, stream);
}

}  // namespace deep_ep::elastic
