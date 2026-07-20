#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

namespace deep_ep::elastic {

// One warp owns one compact (egress, channel) group sequence. Moved combine
// records return to proxy_return[p]; the preserved proxy_dispatch[p] record is
// the route source, so no return descriptor, ready flag, ring, or sidecar is
// needed. All successful-path offsets are static plan prefixes.
// The compact plan is a frozen Gate-2 product. A later corruption makes the
// transaction invalid, but this fused validator/mover does not roll back peer
// stores completed before another block reports the sticky error.
template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_return_unshuffle_impl(
        const ncclDevComm_t nccl_dev_comm,
        const ncclWindow_t nccl_window,
        void* arena,
        void* legacy_reduce_buffer,
        const int* moved,
        const int* group_prefix,
        const int* proxy_required,
        int* status,
        const int num_experts,
        const int num_destinations,
        const int num_rails,
        const int egress,
        const int current_rank_idx,
        const int num_channels,
        const int num_max_tokens_per_rank,
        const int proxy_capacity) {
    static_assert(kHidden > 0 and kHidden % 256 == 0);
    static_assert(kNumTopk >= 1 and kNumTopk <= 32);
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);

    const int channel = static_cast<int>(blockIdx.x);
    const int lane = ptx::get_lane_idx();
    const bool invalid_arguments =
        channel >= num_channels or threadIdx.x >= 32 or
        arena == nullptr or legacy_reduce_buffer == nullptr or
        moved == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or status == nullptr or
        num_experts < 1 or
        num_destinations < 2 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        num_rails < 2 or num_rails > 32 or
        egress < 0 or egress >= num_rails or
        current_rank_idx < 0 or current_rank_idx % num_rails != egress or
        current_rank_idx / num_rails >= num_destinations or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_max_tokens_per_rank < 1 or proxy_capacity < 1 or
        num_experts % num_destinations != 0;
    if (invalid_arguments) {
        if (lane == 0)
            rail_balance::hybrid_plan_detail::report_error(
                status, rail_balance::HybridPlanError::InvalidSchedule);
        return;
    }
    if (__ldg(status) != static_cast<int>(
            rail_balance::HybridPlanError::Success))
        return;

    const int required = __ldg(proxy_required + egress);
    if (required < 0 or required > proxy_capacity) {
        if (lane == 0)
            rail_balance::hybrid_plan_detail::report_error(
                status,
                required > proxy_capacity ?
                    rail_balance::HybridPlanError::CapacityExceeded :
                    rail_balance::HybridPlanError::InvalidSchedule);
        return;
    }

    const int experts_per_destination = num_experts / num_destinations;
    const int current_source_scaleout_rank = current_rank_idx / num_rails;
    const int num_reduce_rows =
        num_destinations <= kNumTopk ? num_destinations : kNumTopk;
    const auto arena_layout = rail_balance::HybridArenaLayout(
        kHidden, kNumTopk, proxy_capacity, arena);
    const auto combine_token_layout = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false);
    const auto legacy_reduce_layout = layout::BufferLayout<false>(
        combine_token_layout, num_reduce_rows,
        num_max_tokens_per_rank, legacy_reduce_buffer);

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged_token = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false, smem);
    const int token_bytes = staged_token.get_num_bytes<false>();
    auto* mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + token_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // group_prefix is channel-major/destination-minor. Checking the immediate
    // predecessor makes every block verify its slice is part of one dense
    // [0, proxy_required) partition without a global scan or hot-path atomic.
    int expected_proxy_slot = 0;
    if (channel > 0) {
        const auto previous_offset =
            rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel - 1, num_destinations - 1,
                num_channels, num_destinations);
        const int previous_begin = __ldg(group_prefix + previous_offset);
        const int previous_count = __ldg(moved + previous_offset);
        if (previous_begin < 0 or previous_begin > required or
            previous_count < 0 or previous_count > required - previous_begin) {
            if (lane == 0)
                rail_balance::hybrid_plan_detail::report_error(
                    status, rail_balance::HybridPlanError::InvalidSchedule);
            return;
        }
        expected_proxy_slot = previous_begin + previous_count;
    }

    for (int destination = 0;
         destination < num_destinations;
         ++destination) {
        if (__ldg(status) != static_cast<int>(
                rail_balance::HybridPlanError::Success))
            break;

        const auto group_offset =
            rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel, destination,
                num_channels, num_destinations);
        const int begin = __ldg(group_prefix + group_offset);
        const int count = __ldg(moved + group_offset);
        const bool valid_group =
            begin == expected_proxy_slot and count >= 0 and
            begin >= 0 and begin <= required and count <= required - begin;
        const bool valid_destination =
            destination != current_source_scaleout_rank or count == 0;
        if (not valid_group or not valid_destination) {
            if (lane == 0)
                rail_balance::hybrid_plan_detail::report_error(
                    status, rail_balance::HybridPlanError::InvalidSchedule);
            break;
        }
        expected_proxy_slot = begin + count;

        for (int proxy_slot = begin;
             proxy_slot < expected_proxy_slot;
             ++proxy_slot) {
            if (__ldg(status) != static_cast<int>(
                    rail_balance::HybridPlanError::Success))
                break;

            const auto dispatch_token =
                arena_layout.get_proxy_dispatch_layout(proxy_slot);
            int src_token_global_idx = lane == 0 ?
                __ldg(dispatch_token.get_src_token_global_idx_ptr()) : 0;
            int transit_proxy_slot = lane == 0 ?
                __ldg(dispatch_token.get_linked_list_idx_ptr()) : 0;
            src_token_global_idx =
                ptx::exchange(src_token_global_idx, 0);
            transit_proxy_slot = ptx::exchange(transit_proxy_slot, 0);
            if (src_token_global_idx < 0 or
                transit_proxy_slot != proxy_slot) {
                if (lane == 0)
                    rail_balance::hybrid_plan_detail::report_error(
                        status,
                        rail_balance::HybridPlanError::InvalidSchedule);
                break;
            }

            const int src_rank_idx =
                src_token_global_idx / num_max_tokens_per_rank;
            const int source_scaleout_rank = src_rank_idx / num_rails;
            const int owner = src_rank_idx % num_rails;
            // src_token_global_idx >= 0 and M > 0 make this modulo naturally
            // land in [0, M); no separate owner-token range branch is needed.
            const int owner_token =
                src_token_global_idx % num_max_tokens_per_rank;
            if (source_scaleout_rank != current_source_scaleout_rank or
                owner == egress) {
                if (lane == 0)
                    rail_balance::hybrid_plan_detail::report_error(
                        status,
                        rail_balance::HybridPlanError::InvalidSchedule);
                break;
            }

            int reduce_row = destination;
            if constexpr (kNumTopk <
                          rail_balance::kNumHybridMaxDestinations) {
                if (num_destinations > kNumTopk) {
                    const bool active = lane < kNumTopk;
                    const int expert = active ?
                        __ldg(dispatch_token.get_topk_idx_ptr() + lane) : -1;
                    const unsigned invalid_expert_mask = ptx::gather(
                        active and (expert < 0 or expert >= num_experts));
                    if (invalid_expert_mask != 0) {
                        if (lane == 0)
                            rail_balance::hybrid_plan_detail::report_error(
                                status,
                                rail_balance::HybridPlanError::
                                    ExpertOutOfRange);
                        break;
                    }

                    const unsigned matching_lane_mask = ptx::gather(
                        active and
                        expert / experts_per_destination == destination);
                    if (matching_lane_mask == 0) {
                        if (lane == 0)
                            rail_balance::hybrid_plan_detail::report_error(
                                status,
                                rail_balance::HybridPlanError::
                                    InvalidSchedule);
                        break;
                    }
                    // Legacy Hybrid combine uses bfind/get_master_lane_idx,
                    // i.e. the highest (last) matching top-k lane.
                    reduce_row =
                        ptx::get_master_lane_idx(matching_lane_mask);
                }
            }

            const auto reduce_token = legacy_reduce_layout
                .get_rank_buffer(reduce_row)
                .get_token_buffer(owner_token);
            void* peer_reduce_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                reduce_token.get_base_ptr(), owner);

            const auto return_token =
                arena_layout.get_proxy_return_layout(proxy_slot);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(
                    staged_token.get_base_ptr(),
                    return_token.get_base_ptr(),
                    mbarrier_ptr, token_bytes);
                ptx::mbarrier_arrive_and_set_tx(
                    mbarrier_ptr, token_bytes);
                ptx::mbarrier_wait_and_flip_phase(
                    mbarrier_ptr, phase);
                ptx::tma_store_1d(
                    peer_reduce_ptr,
                    staged_token.get_base_ptr(), token_bytes);
                ptx::tma_store_commit();
                ptx::tma_store_wait();
            }
            __syncwarp();
        }
    }

    if (channel == num_channels - 1 and
        expected_proxy_slot != required and
        __ldg(status) == static_cast<int>(
            rail_balance::HybridPlanError::Success)) {
        if (lane == 0)
            rail_balance::hybrid_plan_detail::report_error(
                status, rail_balance::HybridPlanError::InvalidSchedule);
    }
}

}  // namespace deep_ep::elastic
