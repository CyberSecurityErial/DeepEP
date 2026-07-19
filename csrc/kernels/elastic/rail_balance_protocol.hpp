#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

struct RailBalanceProtocolSpec {
    int hidden;
    int num_topk;
};

class RailBalanceProtocolProducerRuntime final:
    public jit::LaunchRuntime<RailBalanceProtocolProducerRuntime> {
public:
    struct Args {
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* x;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        const int* send_manifest;
        const int64_t* fingerprints;
        const int64_t* producer_delay_cycles;
        int* producer_status;
        void* arena;
        int physical_capacity;
        int generation;
        int rank_idx;
        int num_max_tokens_per_rank;
        int num_send_records;
        int producer_phase;
        int force_odd_first;
        int drop_physical_slot;
        int late_publish_start;
        int late_publish_after_consumed;
        int64_t timeout_cycles;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const RailBalanceProtocolSpec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_protocol.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_protocol_producer_impl<{}, {}>);
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
            args.send_manifest, args.fingerprints,
            args.producer_delay_cycles, args.producer_status,
            args.arena, args.physical_capacity, args.generation,
            args.rank_idx, args.num_max_tokens_per_rank,
            args.num_send_records, args.producer_phase,
            args.force_odd_first,
            args.drop_physical_slot, args.late_publish_start,
            args.late_publish_after_consumed, args.timeout_cycles));
    }
};

class RailBalanceProtocolConsumerRuntime final:
    public jit::LaunchRuntime<RailBalanceProtocolConsumerRuntime> {
public:
    struct Args {
        void* arena;
        void* consumed_records;
        int* ready_values;
        uint64_t* publish_sequence_values;
        int* consume_counts;
        int* ready_count_at_consume;
        int64_t* trace;
        void* control_snapshot;
        const int64_t* consumer_delay_cycles;
        int physical_capacity;
        int generation;
        int rank_idx;
        int num_recv_records;
        int64_t timeout_cycles;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const RailBalanceProtocolSpec& spec) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance_protocol.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_protocol_consumer_impl<{}, {}>);
}}
)", spec.hidden, spec.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config,
            args.arena, args.consumed_records,
            args.ready_values, args.publish_sequence_values,
            args.consume_counts, args.ready_count_at_consume,
            args.trace, args.control_snapshot,
            args.consumer_delay_cycles,
            args.physical_capacity, args.generation, args.rank_idx,
            args.num_recv_records, args.timeout_cycles));
    }
};

struct PreparedRailBalanceProtocol {
    std::shared_ptr<jit::KernelRuntime> producer;
    std::shared_ptr<jit::KernelRuntime> consumer;
};

static PreparedRailBalanceProtocol prepare_rail_balance_protocol(
    const int& hidden,
    const int& num_topk) {
    const RailBalanceProtocolSpec spec = {
        .hidden = hidden,
        .num_topk = num_topk,
    };
    const auto producer_code =
        RailBalanceProtocolProducerRuntime::generate(spec);
    const auto producer = jit::compiler->build(
        "rail_balance_protocol_producer", producer_code);
    const auto consumer_code =
        RailBalanceProtocolConsumerRuntime::generate(spec);
    const auto consumer = jit::compiler->build(
        "rail_balance_protocol_consumer", consumer_code);
    return {
        .producer = producer,
        .consumer = consumer,
    };
}

static void launch_prepared_rail_balance_protocol_producer(
    const PreparedRailBalanceProtocol& prepared,
    const jit::NoRefPtr& nccl_dev_comm,
    const ncclWindow_t& nccl_window,
    void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const int* send_manifest,
    const int64_t* fingerprints,
    const int64_t* producer_delay_cycles,
    int* producer_status,
    void* arena,
    const int& physical_capacity,
    const int& generation,
    const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_send_records,
    const int& producer_phase,
    const int& force_odd_first,
    const int& drop_physical_slot,
    const int& late_publish_start,
    const int& late_publish_after_consumed,
    const int64_t& timeout_cycles,
    const int& num_smem_bytes,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceProtocolProducerRuntime::Args args = {
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .send_manifest = send_manifest,
        .fingerprints = fingerprints,
        .producer_delay_cycles = producer_delay_cycles,
        .producer_status = producer_status,
        .arena = arena,
        .physical_capacity = physical_capacity,
        .generation = generation,
        .rank_idx = rank_idx,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_send_records = num_send_records,
        .producer_phase = producer_phase,
        .force_odd_first = force_odd_first,
        .drop_physical_slot = drop_physical_slot,
        .late_publish_start = late_publish_start,
        .late_publish_after_consumed =
            late_publish_after_consumed,
        .timeout_cycles = timeout_cycles,
        .launch_args = jit::LaunchArgs(
            num_send_records, 32, num_smem_bytes),
    };
    RailBalanceProtocolProducerRuntime::launch(
        prepared.producer, args, stream);
}

static void launch_prepared_rail_balance_protocol_consumer(
    const PreparedRailBalanceProtocol& prepared,
    void* arena,
    void* consumed_records,
    int* ready_values,
    uint64_t* publish_sequence_values,
    int* consume_counts,
    int* ready_count_at_consume,
    int64_t* trace,
    void* control_snapshot,
    const int64_t* consumer_delay_cycles,
    const int& physical_capacity,
    const int& generation,
    const int& rank_idx,
    const int& num_recv_records,
    const int64_t& timeout_cycles,
    const at::cuda::CUDAStream& stream) {
    const RailBalanceProtocolConsumerRuntime::Args args = {
        .arena = arena,
        .consumed_records = consumed_records,
        .ready_values = ready_values,
        .publish_sequence_values = publish_sequence_values,
        .consume_counts = consume_counts,
        .ready_count_at_consume = ready_count_at_consume,
        .trace = trace,
        .control_snapshot = control_snapshot,
        .consumer_delay_cycles = consumer_delay_cycles,
        .physical_capacity = physical_capacity,
        .generation = generation,
        .rank_idx = rank_idx,
        .num_recv_records = num_recv_records,
        .timeout_cycles = timeout_cycles,
        .launch_args = jit::LaunchArgs(1, 64),
    };
    RailBalanceProtocolConsumerRuntime::launch(
        prepared.consumer, args, stream);
}

}  // namespace deep_ep::elastic
