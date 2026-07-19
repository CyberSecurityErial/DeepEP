#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

struct RailBalanceSourceShuffleSpec {
    int hidden;
    int num_topk;
};

class RailBalanceSourceShuffleRuntime final:
    public jit::LaunchRuntime<RailBalanceSourceShuffleRuntime> {
public:
    struct Args {
        int hidden;
        int num_topk;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* x;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        const int* send_manifest;
        const int64_t* fingerprints;
        void* arena;
        int physical_capacity;
        int generation;
        int rank_idx;
        int num_max_tokens_per_rank;
        int num_send_records;
        jit::LaunchArgs launch_args;
    };

    template <typename Spec>
    static std::string generate_impl(const Spec& args) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_shuffle.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&rail_balance_source_shuffle_impl<{}, {}>);
}}
)", args.hidden, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.nccl_dev_comm, args.nccl_window,
            args.x, args.topk_idx, args.topk_weights,
            args.send_manifest, args.fingerprints,
            args.arena, args.physical_capacity, args.generation,
            args.rank_idx, args.num_max_tokens_per_rank,
            args.num_send_records));
    }
};

static std::shared_ptr<jit::KernelRuntime>
prepare_rail_balance_source_shuffle(
    const int& hidden,
    const int& num_topk) {
    const RailBalanceSourceShuffleSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    return jit::compiler->build(
        "rail_balance_source_shuffle",
        RailBalanceSourceShuffleRuntime::generate(spec));
}

static void launch_prepared_rail_balance_source_shuffle(
    const std::shared_ptr<jit::KernelRuntime>& runtime,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const int* send_manifest,
    const int64_t* fingerprints,
    void* arena,
    const int& hidden,
    const int& num_topk,
    const int& physical_capacity,
    const int& generation,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_send_records,
    const int& num_smem_bytes,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceSourceShuffleRuntime::Args args = {
        .hidden = hidden,
        .num_topk = num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .send_manifest = send_manifest,
        .fingerprints = fingerprints,
        .arena = arena,
        .physical_capacity = physical_capacity,
        .generation = generation,
        .rank_idx = rank_idx,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_send_records = num_send_records,
        .launch_args = jit::LaunchArgs(
            num_send_records, 32, num_smem_bytes),
    };
    RailBalanceSourceShuffleRuntime::launch(runtime, args, stream);
}

class RailBalanceSourceShuffleReadbackRuntime final:
    public jit::LaunchRuntime<RailBalanceSourceShuffleReadbackRuntime> {
public:
    struct Args {
        int hidden;
        int num_topk;
        const void* arena;
        void* records;
        int* ready_values;
        int physical_capacity;
        int num_recv_records;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_shuffle.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&rail_balance_source_shuffle_readback_impl<{}, {}>);
}}
)", args.hidden, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.arena, args.records, args.ready_values,
            args.physical_capacity, args.num_recv_records));
    }
};

static void launch_rail_balance_source_shuffle(
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const int* send_manifest,
    const int64_t* fingerprints,
    void* arena,
    const int& hidden,
    const int& num_topk,
    const int& physical_capacity,
    const int& generation,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_send_records,
    const int& num_smem_bytes,
    const at::cuda::CUDAStream& stream) {
    const auto runtime =
        prepare_rail_balance_source_shuffle(hidden, num_topk);
    launch_prepared_rail_balance_source_shuffle(
        runtime, nccl_dev_comm, nccl_window,
        x, topk_idx, topk_weights, send_manifest, fingerprints,
        arena, hidden, num_topk, physical_capacity, generation,
        rank_idx, num_max_tokens_per_rank, num_send_records,
        num_smem_bytes, stream);
}

static void launch_rail_balance_source_shuffle_readback(
    const void* arena,
    void* records,
    int* ready_values,
    const int& hidden,
    const int& num_topk,
    const int& physical_capacity,
    const int& num_recv_records,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceSourceShuffleReadbackRuntime::Args args = {
        .hidden = hidden,
        .num_topk = num_topk,
        .arena = arena,
        .records = records,
        .ready_values = ready_values,
        .physical_capacity = physical_capacity,
        .num_recv_records = num_recv_records,
        .launch_args = jit::LaunchArgs(physical_capacity, 32),
    };
    const auto code = RailBalanceSourceShuffleReadbackRuntime::generate(args);
    const auto runtime = jit::compiler->build("rail_balance_source_shuffle_readback", code);
    RailBalanceSourceShuffleReadbackRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
