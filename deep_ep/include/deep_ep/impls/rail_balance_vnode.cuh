#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_vnode_layout.cuh>

namespace deep_ep::elastic {

namespace rail_balance::vnode_detail {

enum class StageError : int {
    None = 0,
    ReadyMismatch = 1,
    DescriptorMismatch = 2,
    CanaryMismatch = 3,
    RouteMismatch = 4,
    ExpertMismatch = 5,
    DuplicateOwnerPartial = 6,
    InvalidTopk = 7,
    QuotaMismatch = 8,
    UnexpectedRecord = 9,
    DestinationMismatch = 10,
};

__forceinline__ __device__ int validate_record(
    const SourceShuffleLayout& layout,
    const int& storage_slot,
    const int& generation,
    const int& expected_physical_slot,
    const int& expected_egress) {
    if (ptx::ld_acquire_sys(layout.get_ready_ptr(storage_slot)) != generation)
        return static_cast<int>(StageError::ReadyMismatch);
    const auto descriptor = layout.get_descriptor_ptr(storage_slot);
    if (descriptor->generation != generation or
        descriptor->physical_slot != expected_physical_slot or
        descriptor->egress != expected_egress or
        descriptor->owner_rank < 0 or
        descriptor->owner_token < 0 or
        descriptor->src_token_global_idx < 0)
        return static_cast<int>(StageError::DescriptorMismatch);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        if (descriptor->reserved[i] != 0)
            return static_cast<int>(StageError::DescriptorMismatch);
    #pragma unroll
    for (int i = 0; i < kNumCanaryBytes /
                            static_cast<int>(sizeof(uint32_t)); ++i)
        if (layout.get_head_canary_ptr(storage_slot)[i] != kHeadCanary or
            layout.get_tail_canary_ptr(storage_slot)[i] != kTailCanary)
            return static_cast<int>(StageError::CanaryMismatch);
    return static_cast<int>(StageError::None);
}

__forceinline__ __device__ int validate_route(
    const VNodeRoute& route,
    const ProxyDescriptor* descriptor,
    const int& generation,
    const int& ingress_rank,
    const int& ingress_slot,
    const int& topk_lane,
    const int& expert_rank,
    const int& expert_slot) {
    if (route.fingerprint != descriptor->fingerprint or
        route.generation != generation or
        route.ingress_rank != ingress_rank or
        route.ingress_slot != ingress_slot or
        route.topk_lane != topk_lane or
        route.expert_rank != expert_rank or
        route.expert_slot != expert_slot)
        return static_cast<int>(StageError::RouteMismatch);
    return static_cast<int>(StageError::None);
}

__forceinline__ __device__ int validate_empty_stage(
    const VNodeStageLayout& layout, const int& storage_slot) {
    if (ptx::ld_acquire_sys(
            layout.records.get_ready_ptr(storage_slot)) != 0)
        return static_cast<int>(StageError::UnexpectedRecord);
    const auto route = layout.get_route_ptr(storage_slot);
    #pragma unroll
    for (int i = 0;
         i < static_cast<int>(sizeof(VNodeRoute) / sizeof(uint32_t)); ++i)
        if (reinterpret_cast<const uint32_t*>(route)[i] != 0)
            return static_cast<int>(StageError::UnexpectedRecord);
    return static_cast<int>(StageError::None);
}

__forceinline__ __device__ int validate_base_record(
    const SourceShuffleLayout& layout,
    const int& storage_slot,
    const int& generation,
    const int& expected_physical_slot,
    const int& expected_egress,
    const int& expected_destination) {
    const int error = validate_record(
        layout, storage_slot, generation,
        expected_physical_slot, expected_egress);
    if (error != static_cast<int>(StageError::None))
        return error;
    return layout.get_descriptor_ptr(storage_slot)->destination ==
                   expected_destination
        ? static_cast<int>(StageError::None)
        : static_cast<int>(StageError::DestinationMismatch);
}

__forceinline__ __device__ int get_prefix_expectation(
    const int* quota,
    const int& num_source_ranks,
    const int& num_destinations,
    const int& destination_capacity,
    const int& source_egress,
    const int& destination,
    const int& destination_slot,
    bool& expected) {
    if (quota == nullptr or
        source_egress < 0 or source_egress >= num_source_ranks or
        destination < 0 or destination >= num_destinations or
        destination_slot < 0 or destination_slot >= destination_capacity)
        return static_cast<int>(StageError::QuotaMismatch);
    const int count = quota[
        static_cast<int64_t>(source_egress) * num_destinations +
        destination];
    if (count < 0 or count > destination_capacity)
        return static_cast<int>(StageError::QuotaMismatch);
    expected = destination_slot < count;
    return static_cast<int>(StageError::None);
}

__forceinline__ __device__ int decode_expert(
    const int& expert_idx,
    const int& expert_begin,
    const int& experts_per_rank,
    const int& num_source_ranks,
    const int& num_destinations,
    int& destination,
    int& destination_rank) {
    const int64_t relative =
        static_cast<int64_t>(expert_idx) - expert_begin;
    const int64_t experts_per_destination =
        static_cast<int64_t>(num_source_ranks) * experts_per_rank;
    const int64_t num_remote_experts =
        experts_per_destination * num_destinations;
    if (experts_per_rank <= 0 or relative < 0 or
        relative >= num_remote_experts)
        return static_cast<int>(StageError::ExpertMismatch);
    destination = static_cast<int>(relative / experts_per_destination);
    destination_rank = static_cast<int>(
        (relative % experts_per_destination) / experts_per_rank);
    return static_cast<int>(StageError::None);
}

__forceinline__ __device__ void set_status(
    int* status, const int& index, const int& error) {
    if (ptx::elect_one_sync())
        status[index] = error;
}

}  // namespace rail_balance::vnode_detail

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_scaleout_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation,
    const int rank_idx,
    const int expert_begin,
    const int experts_per_rank) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int slot = static_cast<int>(blockIdx.x);
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, arena);
    (void)expert_begin;
    (void)experts_per_rank;
    const int destination = layout.get_destination(slot);
    const int destination_slot = layout.get_destination_slot(slot);

    int error = 0;
    bool expected = false;
    if (ptx::elect_one_sync()) {
        error = rail_balance::vnode_detail::get_prefix_expectation(
            quota, num_source_ranks, num_destinations,
            destination_capacity, rank_idx, destination,
            destination_slot, expected);
        if (error == 0)
            error = expected
                ? rail_balance::vnode_detail::validate_base_record(
                      layout.rail.records, slot, generation,
                      slot, rank_idx, destination)
                : rail_balance::vnode_detail::validate_empty_stage(
                      layout.rail, slot);
    }
    error = __shfl_sync(0xffffffff, error, 0);
    expected = __shfl_sync(
        0xffffffff, static_cast<int>(expected), 0) != 0;
    if (error != 0) {
        rail_balance::vnode_detail::set_status(status, slot, error);
        return;
    }
    if (not expected)
        return;

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        ptx::tma_load_1d(
            staged.get_record_ptr(0), layout.rail.records.get_record_ptr(slot),
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_arrive_and_set_tx(
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();

    const int peer_rank = layout.get_ingress_rank(destination, rank_idx);
    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto peer_arena =
        gin.get_sym_ptr<ncclTeamTagLsa>(arena, peer_rank);
    const auto peer_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, peer_arena);
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.rail.records.get_record_ptr(slot),
            staged.get_record_ptr(0), static_cast<int>(staged.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync())
        ptx::st_release_sys(
            peer_layout.rail.records.get_ready_ptr(slot), generation);
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_forward_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation,
    const int rank_idx,
    const int expert_begin,
    const int experts_per_rank) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int work_idx = static_cast<int>(blockIdx.x);
    const int ingress_slot = work_idx / kNumTopk;
    const int topk_lane = work_idx % kNumTopk;
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, arena);
    const int destination = layout.get_destination(ingress_slot);
    const int destination_slot =
        layout.get_destination_slot(ingress_slot);
    const int destination_rank_idx = rank_idx - num_source_ranks;
    const int rank_destination =
        destination_rank_idx / num_source_ranks;
    const int source_egress =
        destination_rank_idx % num_source_ranks;

    int error = 0;
    int expert_rank = -1;
    int expert_slot = -1;
    int expert_idx = -1;
    bool process = false;
    if (ptx::elect_one_sync()) {
        bool in_prefix = false;
        error = rail_balance::vnode_detail::get_prefix_expectation(
            quota, num_source_ranks, num_destinations,
            destination_capacity, source_egress, destination,
            destination_slot, in_prefix);
        const bool expected =
            rank_destination == destination and in_prefix;
        if (error == 0)
            error = expected
                ? rail_balance::vnode_detail::validate_base_record(
                      layout.rail.records, ingress_slot, generation,
                      ingress_slot, source_egress, destination)
                : rail_balance::vnode_detail::validate_empty_stage(
                      layout.rail, ingress_slot);
        if (error == 0 and expected) {
            expert_idx = layout.rail.records
                .get_token_layout(ingress_slot)
                .get_topk_idx_ptr()[topk_lane];
            int expert_destination = -1;
            int destination_expert_rank = -1;
            error = rail_balance::vnode_detail::decode_expert(
                expert_idx, expert_begin, experts_per_rank,
                num_source_ranks, num_destinations,
                expert_destination, destination_expert_rank);
            if (error == 0 and expert_destination == destination) {
                process = true;
                expert_rank = layout.get_ingress_rank(
                    destination, destination_expert_rank);
                expert_slot = layout.get_expert_slot(
                    source_egress, destination_slot, topk_lane);
            }
        }
    }
    error = __shfl_sync(0xffffffff, error, 0);
    expert_rank = __shfl_sync(0xffffffff, expert_rank, 0);
    expert_slot = __shfl_sync(0xffffffff, expert_slot, 0);
    expert_idx = __shfl_sync(0xffffffff, expert_idx, 0);
    process = __shfl_sync(
        0xffffffff, static_cast<int>(process), 0) != 0;
    if (error != 0) {
        rail_balance::vnode_detail::set_status(status, work_idx, error);
        return;
    }
    if (not process)
        return;

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        ptx::tma_load_1d(
            staged.get_record_ptr(0),
            layout.rail.records.get_record_ptr(ingress_slot),
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_arrive_and_set_tx(
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto peer_arena =
        gin.get_sym_ptr<ncclTeamTagLsa>(arena, expert_rank);
    const auto peer_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, peer_arena);
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.expert.records.get_record_ptr(expert_slot),
            staged.get_record_ptr(0), static_cast<int>(staged.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync()) {
        const auto descriptor = staged.get_descriptor_ptr(0);
        *peer_layout.expert.get_route_ptr(expert_slot) = {
            descriptor->fingerprint,
            generation,
            rank_idx,
            ingress_slot,
            topk_lane,
            expert_rank,
            expert_slot,
        };
        __threadfence_system();
        ptx::st_release_sys(
            peer_layout.expert.records.get_ready_ptr(expert_slot), generation);
    }
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_expert_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation,
    const int rank_idx,
    const int expert_begin,
    const int experts_per_rank) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int expert_slot = static_cast<int>(blockIdx.x);
    const int lane_idx = ptx::get_lane_idx();
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, arena);
    const int destination_rank_idx = rank_idx - num_source_ranks;
    const int rank_destination =
        destination_rank_idx / num_source_ranks;
    const int rank_destination_expert =
        destination_rank_idx % num_source_ranks;

    __shared__ rail_balance::VNodeRoute shared_route;
    __shared__ int shared_error;
    __shared__ int shared_expert_idx;
    if (ptx::elect_one_sync()) {
        shared_error = 0;
        const int ready = ptx::ld_acquire_sys(
            layout.expert.records.get_ready_ptr(expert_slot));
        if (ready == 0) {
            const int empty_error =
                rail_balance::vnode_detail::validate_empty_stage(
                    layout.expert, expert_slot);
            shared_error = empty_error == 0 ? -1 : empty_error;
        } else if (ready != generation) {
            shared_error = static_cast<int>(
                rail_balance::vnode_detail::StageError::ReadyMismatch);
        } else {
            shared_route = *layout.expert.get_route_ptr(expert_slot);
            const auto descriptor =
                layout.expert.records.get_descriptor_ptr(expert_slot);
            const int expected_topk_lane = expert_slot % kNumTopk;
            const int expected_destination_slot =
                (expert_slot / kNumTopk) % destination_capacity;
            const int expected_source_egress =
                expert_slot / (destination_capacity * kNumTopk);
            const int expected_ingress_slot = layout.get_ingress_slot(
                rank_destination, expected_destination_slot);
            const int expected_ingress_rank = layout.get_ingress_rank(
                rank_destination, expected_source_egress);
            bool in_prefix = false;
            shared_error =
                rail_balance::vnode_detail::get_prefix_expectation(
                    quota, num_source_ranks, num_destinations,
                    destination_capacity, expected_source_egress,
                    rank_destination, expected_destination_slot,
                    in_prefix);
            if (shared_error == 0 and not in_prefix)
                shared_error = static_cast<int>(
                    rail_balance::vnode_detail::StageError::QuotaMismatch);
            if (shared_error == 0)
                shared_error =
                    rail_balance::vnode_detail::validate_base_record(
                layout.expert.records, expert_slot, generation,
                expected_ingress_slot, expected_source_egress,
                rank_destination);
            if (shared_error == 0)
                shared_error = rail_balance::vnode_detail::validate_route(
                    shared_route, descriptor, generation,
                    expected_ingress_rank, expected_ingress_slot,
                    expected_topk_lane, rank_idx, expert_slot);
            if (shared_error == 0) {
                shared_expert_idx = layout.expert.records
                    .get_token_layout(expert_slot)
                    .get_topk_idx_ptr()[expected_topk_lane];
                int expert_destination = -1;
                int destination_expert_rank = -1;
                shared_error = rail_balance::vnode_detail::decode_expert(
                    shared_expert_idx, expert_begin, experts_per_rank,
                    num_source_ranks, num_destinations,
                    expert_destination, destination_expert_rank);
                if (shared_error == 0 and
                    (expert_destination != rank_destination or
                     destination_expert_rank != rank_destination_expert))
                    shared_error = static_cast<int>(
                        rail_balance::vnode_detail::StageError::ExpertMismatch);
            }
        }
    }
    __syncwarp();
    if (shared_error == -1)
        return;
    if (shared_error != 0) {
        rail_balance::vnode_detail::set_status(
            status, expert_slot, shared_error);
        return;
    }

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        ptx::tma_load_1d(
            staged.get_record_ptr(0),
            layout.expert.records.get_record_ptr(expert_slot),
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_arrive_and_set_tx(
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();

    const float weight =
        staged.get_token_layout(0).get_topk_weights_ptr()[
            shared_route.topk_lane];
    const float bias = static_cast<float>(
        8 * (shared_expert_idx - expert_begin + 1));
    auto hidden = static_cast<nv_bfloat16*>(
        staged.get_token_layout(0).get_hidden_ptr());
    for (int index = lane_idx; index < kHidden; index += 32) {
        const float value = __bfloat162float(hidden[index]);
        hidden[index] = __float2bfloat16_rn((value + bias) * weight);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();

    const int contribution_slot = layout.get_contribution_slot(
        shared_route.ingress_slot, shared_route.topk_lane);
    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto peer_arena = gin.get_sym_ptr<ncclTeamTagLsa>(
        arena, shared_route.ingress_rank);
    const auto peer_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, peer_arena);
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.rail.records.get_record_ptr(contribution_slot),
            staged.get_record_ptr(0), static_cast<int>(staged.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync()) {
        *peer_layout.rail.get_route_ptr(contribution_slot) = shared_route;
        __threadfence_system();
        ptx::st_release_sys(
            peer_layout.rail.records.get_ready_ptr(contribution_slot),
            generation);
    }
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_return_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation,
    const int rank_idx,
    const int expert_begin,
    const int experts_per_rank) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int work_idx = static_cast<int>(blockIdx.x);
    const int ingress_slot = work_idx / kNumTopk;
    const int topk_lane = work_idx % kNumTopk;
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, arena);
    const int contribution_slot =
        layout.get_contribution_slot(ingress_slot, topk_lane);
    const int destination = layout.get_destination(ingress_slot);
    const int destination_slot =
        layout.get_destination_slot(ingress_slot);
    const int destination_rank_idx = rank_idx - num_source_ranks;
    const int rank_destination =
        destination_rank_idx / num_source_ranks;
    const int source_egress =
        destination_rank_idx % num_source_ranks;

    __shared__ rail_balance::VNodeRoute shared_route;
    int error = 0;
    bool process = false;
    if (ptx::elect_one_sync()) {
        bool in_prefix = false;
        error = rail_balance::vnode_detail::get_prefix_expectation(
            quota, num_source_ranks, num_destinations,
            destination_capacity, source_egress, destination,
            destination_slot, in_prefix);
        const bool expected_record =
            destination == rank_destination and in_prefix;
        if (error == 0)
            error = expected_record
                ? rail_balance::vnode_detail::validate_base_record(
                      layout.rail.records, ingress_slot, generation,
                      ingress_slot, source_egress, destination)
                : rail_balance::vnode_detail::validate_empty_stage(
                      layout.rail, ingress_slot);
        int expected_expert_rank = -1;
        int expected_expert_slot = -1;
        if (error == 0 and expected_record) {
            const int expert_idx = layout.rail.records
                .get_token_layout(ingress_slot)
                .get_topk_idx_ptr()[topk_lane];
            int expert_destination = -1;
            int destination_expert_rank = -1;
            error = rail_balance::vnode_detail::decode_expert(
                expert_idx, expert_begin, experts_per_rank,
                num_source_ranks, num_destinations,
                expert_destination, destination_expert_rank);
            process = error == 0 and expert_destination == destination;
            if (process) {
                expected_expert_rank = layout.get_ingress_rank(
                    destination, destination_expert_rank);
                expected_expert_slot = layout.get_expert_slot(
                    source_egress, destination_slot, topk_lane);
            }
        }
        if (error == 0 and not process) {
            error = rail_balance::vnode_detail::validate_empty_stage(
                layout.rail, contribution_slot);
        } else if (error == 0) {
            error = rail_balance::vnode_detail::validate_base_record(
                layout.rail.records, contribution_slot, generation,
                ingress_slot, source_egress, destination);
        }
        if (error == 0 and process) {
            shared_route = *layout.rail.get_route_ptr(contribution_slot);
            const auto descriptor =
                layout.rail.records.get_descriptor_ptr(contribution_slot);
            error = rail_balance::vnode_detail::validate_route(
                shared_route, descriptor, generation, rank_idx,
                ingress_slot, topk_lane,
                expected_expert_rank, expected_expert_slot);
        }
    }
    error = __shfl_sync(0xffffffff, error, 0);
    process = __shfl_sync(
        0xffffffff, static_cast<int>(process), 0) != 0;
    __syncwarp();
    if (error != 0) {
        rail_balance::vnode_detail::set_status(status, work_idx, error);
        return;
    }
    if (not process)
        return;

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        ptx::tma_load_1d(
            staged.get_record_ptr(0),
            layout.rail.records.get_record_ptr(contribution_slot),
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_arrive_and_set_tx(
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto peer_arena =
        gin.get_sym_ptr<ncclTeamTagLsa>(arena, source_egress);
    const auto peer_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, peer_arena);
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.rail.records.get_record_ptr(contribution_slot),
            staged.get_record_ptr(0), static_cast<int>(staged.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync()) {
        *peer_layout.rail.get_route_ptr(contribution_slot) = shared_route;
        __threadfence_system();
        ptx::st_release_sys(
            peer_layout.rail.records.get_ready_ptr(contribution_slot),
            generation);
    }
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_unshuffle_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* arena,
    int* status,
    const int* quota,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation,
    const int rank_idx,
    const int expert_begin,
    const int experts_per_rank) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int work_idx = static_cast<int>(blockIdx.x);
    const int ingress_slot = work_idx / kNumTopk;
    const int topk_lane = work_idx % kNumTopk;
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, arena);
    const int contribution_slot =
        layout.get_contribution_slot(ingress_slot, topk_lane);
    const int destination = layout.get_destination(ingress_slot);
    const int destination_slot =
        layout.get_destination_slot(ingress_slot);

    __shared__ rail_balance::VNodeRoute shared_route;
    __shared__ int shared_owner;
    __shared__ int shared_owner_token;
    int error = 0;
    bool process = false;
    if (ptx::elect_one_sync()) {
        bool in_prefix = false;
        error = rail_balance::vnode_detail::get_prefix_expectation(
            quota, num_source_ranks, num_destinations,
            destination_capacity, rank_idx, destination,
            destination_slot, in_prefix);
        if (error == 0)
            error = in_prefix
                ? rail_balance::vnode_detail::validate_base_record(
                      layout.rail.records, ingress_slot, generation,
                      ingress_slot, rank_idx, destination)
                : rail_balance::vnode_detail::validate_empty_stage(
                      layout.rail, ingress_slot);
        int expected_expert_rank = -1;
        int expected_expert_slot = -1;
        if (error == 0 and in_prefix) {
            const int expert_idx = layout.rail.records
                .get_token_layout(ingress_slot)
                .get_topk_idx_ptr()[topk_lane];
            int expert_destination = -1;
            int destination_expert_rank = -1;
            error = rail_balance::vnode_detail::decode_expert(
                expert_idx, expert_begin, experts_per_rank,
                num_source_ranks, num_destinations,
                expert_destination, destination_expert_rank);
            process = error == 0 and expert_destination == destination;
            if (process) {
                expected_expert_rank = layout.get_ingress_rank(
                    destination, destination_expert_rank);
                expected_expert_slot = layout.get_expert_slot(
                    rank_idx, destination_slot, topk_lane);
            }
        }
        if (error == 0 and not process) {
            error = rail_balance::vnode_detail::validate_empty_stage(
                layout.rail, contribution_slot);
        } else if (error == 0) {
            error = rail_balance::vnode_detail::validate_base_record(
                layout.rail.records, contribution_slot, generation,
                ingress_slot, rank_idx, destination);
        }
        if (error == 0 and process) {
            shared_route = *layout.rail.get_route_ptr(contribution_slot);
            const auto descriptor =
                layout.rail.records.get_descriptor_ptr(contribution_slot);
            error = rail_balance::vnode_detail::validate_route(
                shared_route, descriptor, generation,
                layout.get_ingress_rank(destination, rank_idx),
                ingress_slot, topk_lane,
                expected_expert_rank, expected_expert_slot);
            shared_owner = descriptor->owner_rank;
            shared_owner_token = descriptor->owner_token;
            if (error == 0 and
                (shared_owner < 0 or shared_owner >= num_source_ranks or
                 shared_owner_token < 0 or
                 shared_owner_token >= num_max_tokens or
                 descriptor->src_token_global_idx !=
                     shared_owner * num_max_tokens + shared_owner_token))
                error = static_cast<int>(
                    rail_balance::vnode_detail::StageError::DescriptorMismatch);
        }
    }
    error = __shfl_sync(0xffffffff, error, 0);
    process = __shfl_sync(
        0xffffffff, static_cast<int>(process), 0) != 0;
    __syncwarp();
    if (error != 0) {
        rail_balance::vnode_detail::set_status(status, work_idx, error);
        return;
    }
    if (not process)
        return;

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    void* peer_arena = arena;
    if (shared_owner != rank_idx)
        peer_arena = gin.get_sym_ptr<ncclTeamTagLsa>(arena, shared_owner);
    const auto peer_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, peer_arena);
    int claimed = 0;
    if (ptx::elect_one_sync())
        claimed = atomicCAS_system(
            peer_layout.owner.get_ready_ptr(
                shared_owner_token, topk_lane),
            0, -generation);
    claimed = __shfl_sync(0xffffffff, claimed, 0);
    if (claimed != 0) {
        rail_balance::vnode_detail::set_status(
            status, work_idx,
            static_cast<int>(
                rail_balance::vnode_detail::StageError::
                    DuplicateOwnerPartial));
        return;
    }

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync()) {
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
        ptx::tma_load_1d(
            staged.get_record_ptr(0),
            layout.rail.records.get_record_ptr(contribution_slot),
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_arrive_and_set_tx(
            mbarrier_ptr, static_cast<int>(staged.record_bytes));
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.owner.get_value_ptr(
                shared_owner_token, topk_lane),
            staged.get_token_layout(0).get_hidden_ptr(), kNumHiddenBytes);
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync())
        ptx::st_release_sys(
            peer_layout.owner.get_ready_ptr(
                shared_owner_token, topk_lane),
            generation);
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_vnode_owner_reduce_impl(
    const void* arena,
    const topk_idx_t* topk_idx,
    nv_bfloat16* output,
    int* output_ready,
    int* status,
    const int num_destinations,
    const int destination_capacity,
    const int num_source_ranks,
    const int num_max_tokens,
    const int generation) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int token_idx = static_cast<int>(blockIdx.x);
    const int lane_idx = ptx::get_lane_idx();
    const auto layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_destinations, destination_capacity,
        num_source_ranks, num_max_tokens, const_cast<void*>(arena));

    int action = 0;  // 0: masked, 1: reduce, 2: error
    if (ptx::elect_one_sync()) {
        int valid_lanes = 0;
        for (int topk_lane = 0; topk_lane < kNumTopk; ++topk_lane) {
            const int expert = topk_idx[
                static_cast<int64_t>(token_idx) * kNumTopk + topk_lane];
            valid_lanes += expert >= 0;
        }
        if (valid_lanes == 0) {
            action = 0;
        } else if (valid_lanes != kNumTopk) {
            action = 2;
        } else {
            action = 1;
            for (int topk_lane = 0; topk_lane < kNumTopk; ++topk_lane)
                if (ptx::ld_acquire_sys(layout.owner.get_ready_ptr(
                        token_idx, topk_lane)) != generation) {
                    action = 2;
                    break;
                }
        }
    }
    action = __shfl_sync(0xffffffff, action, 0);
    if (action == 0)
        return;
    if (action == 2) {
        rail_balance::vnode_detail::set_status(
            status, token_idx,
            static_cast<int>(
                rail_balance::vnode_detail::StageError::InvalidTopk));
        return;
    }
    __syncwarp();

    for (int hidden_idx = lane_idx;
         hidden_idx < kHidden; hidden_idx += 32) {
        float reduced = 0.0f;
        #pragma unroll
        for (int topk_lane = 0; topk_lane < kNumTopk; ++topk_lane) {
            const auto partial = static_cast<const nv_bfloat16*>(
                layout.owner.get_value_ptr(token_idx, topk_lane));
            reduced += __bfloat162float(partial[hidden_idx]);
        }
        output[static_cast<int64_t>(token_idx) * kHidden + hidden_idx] =
            __float2bfloat16_rn(reduced);
    }
    __syncwarp();
    if (ptx::elect_one_sync())
        ptx::st_release_sys(output_ready + token_idx, generation);
}

}  // namespace deep_ep::elastic
