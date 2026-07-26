#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>

namespace deep_ep::elastic {

// One warp owns one source channel. Destinations are lane-owned, so the
// channel-local ordinal consumed by the compact resolver needs no atomic or
// per-copy manifest. Only moved copies are staged into peer proxy arenas.
template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_source_shuffle_impl(
        const ncclDevComm_t nccl_dev_comm,
        const ncclWindow_t nccl_window,
        const __nv_bfloat16* x,
        const topk_idx_t* topk_idx,
        const float* topk_weights,
        void* arena,
        const int* owner_channel_prefix,
        const int* keep_count,
        const int* segments,
        const int* num_segments,
        const int* retained,
        const int* moved_channel_prefix,
        const int* group_prefix,
        const int* proxy_required,
        int* status,
        const int num_tokens,
        const int num_experts,
        const int num_destinations,
        const int local_destination,
        const int num_rails,
        const int owner,
        const int num_channels,
        const int num_max_tokens_per_rank,
        const int rank_idx,
        const int proxy_capacity) {
    static_assert(kHidden > 0 and kHidden % 256 == 0);
    static_assert(kNumTopk >= 1 and kNumTopk <= 32);
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    const int source_channel = static_cast<int>(blockIdx.x);
    if (source_channel >= num_channels or threadIdx.x >= 32 or
        (num_tokens > 0 and
         (x == nullptr or topk_idx == nullptr or topk_weights == nullptr)) or
        arena == nullptr or owner_channel_prefix == nullptr or
        keep_count == nullptr or segments == nullptr or
        num_segments == nullptr or retained == nullptr or
        moved_channel_prefix == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or status == nullptr or
        num_tokens < 0 or num_experts < 1 or
        num_destinations < 2 or num_destinations > 32 or
        local_destination < 0 or local_destination >= num_destinations or
        num_rails < 2 or num_rails > 32 or
        owner < 0 or owner >= num_rails or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_max_tokens_per_rank < 1 or num_tokens > num_max_tokens_per_rank or
        rank_idx < 0 or proxy_capacity < 1 or
        num_experts % num_destinations != 0) {
        return;
    }
    if (__ldg(status) != static_cast<int>(
            rail_balance::HybridPlanError::Success))
        return;

    const int lane = ptx::get_lane_idx();
    const int experts_per_destination = num_experts / num_destinations;
    const int channel_capacity =
        num_max_tokens_per_rank / num_channels +
        (num_max_tokens_per_rank % num_channels != 0);
    int destination_ordinal = 0;

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto local_arena_layout = rail_balance::HybridArenaLayout(
        kHidden, kNumTopk, proxy_capacity, arena);
    const int invocation_key = ptx::ld_acquire_sys<int>(
        &local_arena_layout.get_control_ptr()->invocation_id);
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged_token = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, true, smem);
    const int staged_token_bytes = staged_token.get_num_bytes<false>();
    auto* mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged_token_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    const int metadata_bytes = staged_token_bytes - kNumHiddenBytes;
    EP_DEVICE_ASSERT(metadata_bytes >= 0 and metadata_bytes % sizeof(int4) == 0);

    for (int token = source_channel;
         token < num_tokens;
         token += num_channels) {
        if (__ldg(status) != static_cast<int>(
                rail_balance::HybridPlanError::Success))
            break;
        // Metadata and its aligned padding are deterministic. Hidden is loaded
        // exactly once and is never copied through a second local staging area.
        auto* metadata = reinterpret_cast<int4*>(
            staged_token.get_metadata_ptr());
        for (int i = lane; i < metadata_bytes / sizeof(int4); i += 32)
            metadata[i] = make_int4(0, 0, 0, 0);
        __syncwarp();

        const int64_t token_offset = static_cast<int64_t>(token) * kNumTopk;
        const bool active = lane < kNumTopk;
        const topk_idx_t expert = active ?
            __ldg(topk_idx + token_offset + lane) : topk_idx_t{-1};
        const bool in_range = not active or
            (expert >= topk_idx_t{0} and expert < num_experts);
        const unsigned invalid_route_mask = ptx::gather(not in_range);

        bool duplicate = false;
        for (int source_lane = 0; source_lane < kNumTopk; ++source_lane) {
            const auto other = ptx::exchange(expert, source_lane);
            duplicate |= active and source_lane < lane and expert == other;
        }
        const unsigned duplicate_mask = ptx::gather(duplicate);
        if (invalid_route_mask != 0 or duplicate_mask != 0) {
            if (lane == 0) {
                rail_balance::hybrid_plan_detail::report_error(
                    status,
                    invalid_route_mask != 0 ?
                        rail_balance::HybridPlanError::ExpertOutOfRange :
                        rail_balance::HybridPlanError::DuplicateExpert);
            }
            // Preserve ordinal alignment with the plan only for valid input.
            // A mutated route poisons the transaction and publishes no copy
            // for this token.
            break;
        }

        if (active) {
            staged_token.get_topk_idx_ptr()[lane] = static_cast<int>(expert);
            staged_token.get_topk_weights_ptr()[lane] =
                __ldg(topk_weights + token_offset + lane);
            staged_token.get_linked_list_idx_ptr()[lane] = -1;
        }
        if (lane == 0) {
            *staged_token.get_src_token_global_idx_ptr() =
                rank_idx * num_max_tokens_per_rank + token;
        }

        const int destination = active ?
            static_cast<int>(expert) / experts_per_destination : -1;
        const unsigned destination_bit =
            destination >= 0 and destination != local_destination ?
                (1u << destination) : 0u;
        const unsigned destination_mask = ptx::reduce_or(destination_bit);
        const bool present = lane < num_destinations and
            ((destination_mask >> lane) & 1u) != 0;
        const int local_ordinal = destination_ordinal;
        destination_ordinal += present;

        rail_balance::HybridCopyResolution resolution = {};
        bool resolved = true;
        bool schedule_valid = true;
        if (present) {
            const int destination_num_segments = __ldg(num_segments + lane);
            schedule_valid = destination_num_segments >= 0 and
                destination_num_segments <= num_rails - 1;
            if (schedule_valid) {
                resolved = rail_balance::resolve_hybrid_copy(
                    owner_channel_prefix, keep_count, segments,
                    num_segments, retained, moved_channel_prefix,
                    group_prefix, num_rails, num_channels,
                    num_destinations, owner, source_channel, lane,
                    local_ordinal, &resolution);
                schedule_valid = resolved and
                    (resolution.moved == 0 or resolution.moved == 1) and
                    resolution.channel >= 0 and
                    resolution.channel < num_channels and
                    resolution.remote_slot >= 0 and
                    resolution.remote_slot < channel_capacity;
                if (schedule_valid and resolution.moved == 0) {
                    schedule_valid = resolution.egress == owner and
                        resolution.channel == source_channel and
                        resolution.proxy_slot == -1;
                }
                if (schedule_valid and resolution.moved == 1) {
                    const int egress = resolution.egress;
                    schedule_valid = egress >= 0 and egress < num_rails and
                        egress != owner and resolution.proxy_slot >= 0 and
                        resolution.proxy_slot < proxy_capacity;
                    if (schedule_valid) {
                        const int required = __ldg(proxy_required + egress);
                        schedule_valid = required >= 0 and
                            required <= proxy_capacity and
                            resolution.proxy_slot < required;
                    }
                }
            }
        }
        const unsigned bad_schedule_mask =
            ptx::gather(present and not schedule_valid);
        if (bad_schedule_mask != 0) {
            if (lane == 0)
                rail_balance::hybrid_plan_detail::report_error(
                    status, rail_balance::HybridPlanError::InvalidSchedule);
            break;
        }

        const unsigned retained_mask = ptx::gather(
            present and resolution.moved == 0);
        unsigned moved_mask = ptx::gather(
            present and resolution.moved == 1);
        if (retained_mask == 0 and moved_mask == 0)
            continue;

        if (ptx::elect_one_sync()) {
            ptx::tma_load_1d(
                staged_token.get_hidden_ptr(),
                x + static_cast<int64_t>(token) * kHidden,
                mbarrier_ptr, kNumHiddenBytes);
        }
        __syncwarp();
        if (ptx::elect_one_sync()) {
            ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
            ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
        }
        __syncwarp();

        // Pack retained payloads in channel-major/destination-minor order.
        // The egress dispatch warp reconstructs the same prefix and posts both
        // retained and moved payloads through one elected-lane completion path.
        unsigned pending_retained_mask = retained_mask;
        while (pending_retained_mask != 0) {
            const int source_lane = __ffs(pending_retained_mask) - 1;
            const int destination = source_lane;
            const int remote_slot =
                ptx::exchange(resolution.remote_slot, source_lane);
            int retained_prefix = 0;
            for (int channel = 0; channel <= source_channel; ++channel) {
                const int destination_end = channel < source_channel ?
                    num_destinations : destination;
                for (int previous_destination = 0;
                     previous_destination < destination_end;
                     ++previous_destination) {
                    retained_prefix += __ldg(retained +
                        rail_balance::hybrid_plan_detail::gcd_offset(
                            owner, channel, previous_destination,
                            num_channels, num_destinations));
                }
            }
            const int retained_slot = retained_prefix + remote_slot;
            const auto retained_token =
                local_arena_layout.get_retained_rail_staging_layout(
                    retained_slot);
            ptx::tma_store_fence();
            __syncwarp();
            if (ptx::elect_one_sync())
                ptx::tma_store_1d(
                    retained_token.get_base_ptr(), staged_token.get_base_ptr(),
                    staged_token_bytes);
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            ptx::tma_store_global_visibility_fence();
            __syncwarp();
            pending_retained_mask &= pending_retained_mask - 1;
        }

        while (moved_mask != 0) {
            const int source_lane = __ffs(moved_mask) - 1;
            const int egress = ptx::exchange(resolution.egress, source_lane);
            const int proxy_slot =
                ptx::exchange(resolution.proxy_slot, source_lane);
            ptx::tma_store_fence();
            __syncwarp();

            void* peer_arena = gin.get_sym_ptr<ncclTeamTagLsa>(arena, egress);
            const auto peer_layout = rail_balance::HybridArenaLayout(
                kHidden, kNumTopk, proxy_capacity, peer_arena);
            if (ptx::elect_one_sync()) {
                ptx::tma_store_1d(
                    peer_layout.get_proxy_dispatch_layout(proxy_slot)
                        .get_base_ptr(),
                    staged_token.get_base_ptr(), staged_token_bytes);
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            // Bridge the completed async-proxy payload into the generic
            // global domain before publishing its release-ready word.
            ptx::tma_store_global_visibility_fence();
            // Publish the completed async-proxy copy with an epoch-specific
            // ready key.  The plan's channel-count arena is dead after Gate #2
            // and is large enough to provide one word per proxy slot.
            if (ptx::elect_one_sync())
                ptx::st_release_sys(
                    peer_layout.get_proxy_ready_ptr(proxy_slot),
                    invocation_key);
            __syncwarp();
            moved_mask &= moved_mask - 1;
        }
    }

}

}  // namespace deep_ep::elastic
