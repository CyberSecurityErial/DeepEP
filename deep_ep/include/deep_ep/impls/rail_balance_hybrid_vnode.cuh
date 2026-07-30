#pragma once

#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/common/rail_balance_vnode_layout.cuh>
#include <deep_ep/impls/combine_utils.cuh>
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>
#include <deep_ep/impls/rail_balance_vnode.cuh>

namespace deep_ep::elastic {

namespace rail_balance::hybrid_vnode_detail {

enum class AdapterError : int {
    InvalidArguments = 32,
    InvalidPlan = 33,
    InvalidSource = 34,
    InvalidProxy = 35,
    InvalidBase = 36,
    InvalidContribution = 37,
};

__forceinline__ __device__ void report(
        int* status, const int work_idx, const int error) {
    if (status != nullptr and ptx::elect_one_sync())
        status[work_idx] = error;
}

__forceinline__ __device__ uint64_t fingerprint(
        const int generation,
        const int owner,
        const int owner_token,
        const int num_max_tokens_per_rank,
        const int num_destinations,
        const int destination) {
    const uint64_t identity =
        (static_cast<uint64_t>(owner) * num_max_tokens_per_rank +
         owner_token) * num_destinations + destination;
    return (static_cast<uint64_t>(static_cast<uint32_t>(generation)) << 32) |
           static_cast<uint32_t>(identity);
}

__forceinline__ __device__ bool descriptor_equal(
        const ProxyDescriptor* lhs,
        const ProxyDescriptor* rhs) {
    const auto* lhs_words = reinterpret_cast<const uint32_t*>(lhs);
    const auto* rhs_words = reinterpret_cast<const uint32_t*>(rhs);
    #pragma unroll
    for (int word = 0;
         word < static_cast<int>(sizeof(ProxyDescriptor) / sizeof(uint32_t));
         ++word)
        if (__ldg(lhs_words + word) != __ldg(rhs_words + word))
            return false;
    return true;
}

__forceinline__ __device__ bool metadata_equal(
        const layout::TokenLayout& lhs,
        const layout::TokenLayout& rhs) {
    const int metadata_bytes = lhs.get_num_bytes<false>() -
                               lhs.num_hidden_bytes;
    if (metadata_bytes != rhs.get_num_bytes<false>() - rhs.num_hidden_bytes or
        metadata_bytes % static_cast<int>(sizeof(int4)) != 0)
        return false;
    const auto* lhs_words = reinterpret_cast<const int4*>(
        lhs.get_metadata_ptr());
    const auto* rhs_words = reinterpret_cast<const int4*>(
        rhs.get_metadata_ptr());
    for (int word = 0; word < metadata_bytes / sizeof(int4); ++word) {
        const int4 a = __ldg(lhs_words + word);
        const int4 b = __ldg(rhs_words + word);
        if (a.x != b.x or a.y != b.y or a.z != b.z or a.w != b.w)
            return false;
    }
    return true;
}

__forceinline__ __device__ bool invert_moved_ordinal(
        const int* segments,
        const int* num_segments,
        const int num_rails,
        const int destination,
        const int egress,
        const int incoming_ordinal,
        int& owner,
        int& owner_ordinal) {
    const int count = __ldg(num_segments + destination);
    if (count < 0 or count > num_rails - 1)
        return false;

    int matches = 0;
    owner = -1;
    owner_ordinal = -1;
    for (int segment = 0; segment < count; ++segment) {
        const auto base = hybrid_plan_detail::segment_offset(
            destination, segment, 0, num_rails);
        const int segment_owner =
            __ldg(segments + base + kHybridSegmentOwner);
        const int segment_egress =
            __ldg(segments + base + kHybridSegmentEgress);
        const int owner_begin =
            __ldg(segments + base + kHybridSegmentOwnerBegin);
        const int segment_count =
            __ldg(segments + base + kHybridSegmentCount);
        const int egress_begin =
            __ldg(segments + base + kHybridSegmentEgressBegin);
        if (segment_owner < 0 or segment_owner >= num_rails or
            segment_egress < 0 or segment_egress >= num_rails or
            segment_owner == segment_egress or owner_begin < 0 or
            segment_count <= 0 or egress_begin < 0)
            return false;
        if (segment_egress == egress and
            egress_begin <= incoming_ordinal and
            incoming_ordinal < egress_begin + segment_count) {
            owner = segment_owner;
            owner_ordinal = owner_begin + incoming_ordinal - egress_begin;
            ++matches;
        }
    }
    return matches == 1;
}

__forceinline__ __device__ bool validate_dispatch_token(
        const layout::TokenLayout& token,
        const int expected_proxy_slot,
        const int expected_owner,
        const int expected_destination,
        const int num_experts,
        const int num_destinations,
        const int num_rails,
        const int num_max_tokens_per_rank,
        int& owner_token) {
    const int lane = ptx::get_lane_idx();
    const int experts_per_destination = num_experts / num_destinations;
    const bool active = lane < token.num_topk;
    const int expert = active ?
        __ldg(token.get_topk_idx_ptr() + lane) : -1;
    const bool expert_valid = not active or
        (expert >= 0 and expert < num_experts);
    unsigned invalid_mask = ptx::gather(not expert_valid);

    bool duplicate = false;
    for (int source_lane = 0; source_lane < token.num_topk; ++source_lane) {
        const int other = ptx::exchange(expert, source_lane);
        duplicate |= active and source_lane < lane and expert == other;
    }
    invalid_mask |= ptx::gather(duplicate);

    const int destination = active ?
        expert / experts_per_destination : -1;
    // The old vnode forwarder decodes every lane as a remote expert. Keep the
    // adapter's supported fixture domain explicit instead of weakening it.
    invalid_mask |= ptx::gather(active and destination == 0);
    const unsigned destination_mask = ptx::gather(
        active and destination == expected_destination);

    int src_token_global_idx = 0;
    int linked = 0;
    if (lane == 0) {
        src_token_global_idx =
            __ldg(token.get_src_token_global_idx_ptr());
        linked = __ldg(token.get_linked_list_idx_ptr());
    }
    src_token_global_idx = ptx::exchange(src_token_global_idx, 0);
    linked = ptx::exchange(linked, 0);
    const bool linked_valid = active ?
        __ldg(token.get_linked_list_idx_ptr() + lane) ==
            (lane == 0 ? expected_proxy_slot : -1) : true;
    invalid_mask |= ptx::gather(not linked_valid);

    owner_token = src_token_global_idx % num_max_tokens_per_rank;
    const int owner = src_token_global_idx / num_max_tokens_per_rank;
    return invalid_mask == 0 and destination_mask != 0 and
           linked == expected_proxy_slot and src_token_global_idx >= 0 and
           owner == expected_owner and owner >= 0 and owner < num_rails and
           owner_token >= 0 and owner_token < num_max_tokens_per_rank;
}

__forceinline__ __device__ int validate_vnode_descriptor(
        const ProxyDescriptor* descriptor,
        const int generation,
        const int expected_owner,
        const int expected_owner_token,
        const int expected_destination,
        const int expected_owner_ordinal,
        const int expected_egress,
        const int expected_channel,
        const int expected_logical_slot,
        const int expected_physical_slot,
        const int num_max_tokens_per_rank,
        const int num_destinations) {
    if (descriptor->fingerprint != fingerprint(
            generation, expected_owner, expected_owner_token,
            num_max_tokens_per_rank, num_destinations,
            expected_destination + 1) or
        descriptor->owner_rank != expected_owner or
        descriptor->owner_token != expected_owner_token or
        descriptor->destination != expected_destination or
        descriptor->owner_ordinal != expected_owner_ordinal or
        descriptor->egress != expected_egress or
        descriptor->channel != expected_channel or
        descriptor->logical_slot != expected_logical_slot or
        descriptor->physical_slot != expected_physical_slot or
        descriptor->generation != generation or
        descriptor->src_token_global_idx !=
            expected_owner * num_max_tokens_per_rank + expected_owner_token)
        return static_cast<int>(AdapterError::InvalidBase);
    #pragma unroll
    for (int index = 0; index < 4; ++index)
        if (descriptor->reserved[index] != 0)
            return static_cast<int>(AdapterError::InvalidBase);
    return 0;
}

}  // namespace rail_balance::hybrid_vnode_detail

// Test-only adapter from the production compact Hybrid schedule to the old
// vnode transport namespace. One warp owns one source/target egress channel.
// It emits both retained local records and moved proxy records into the same
// dense [destination][slot] prefix consumed by the old vnode scaleout stage.
template <int kHidden, int kNumTopk, bool kHopAware = false>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_pack_vnode_base_impl(
        const __nv_bfloat16* x,
        const topk_idx_t* topk_idx,
        const float* topk_weights,
        const void* proxy_dispatch,
        void* vnode_arena,
        const int* channel_count,
        const int* quota,
        const int* keep_count,
        const int* segments,
        const int* num_segments,
        const int* owner_channel_prefix,
        const int* retained,
        const int* moved,
        const int* moved_channel_prefix,
        const int* group_prefix,
        const int* proxy_required,
        const rail_balance::HopCopyRecord* hop_records,
        const rail_balance::HopCopyResolution* hop_resolutions,
        const int* hop_pair_load,
        int* status,
        const int num_tokens,
        const int num_experts,
        const int num_destinations,
        const int num_rails,
        const int egress,
        const int num_channels,
        const int num_max_tokens_per_rank,
        const int proxy_capacity,
        const int generation) {
    static_assert(kHidden > 0 and kHidden % 256 == 0);
    static_assert(kNumTopk >= 1 and kNumTopk <= 32);
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    constexpr int kVectorBytes = sizeof(int4);

    const int channel = static_cast<int>(blockIdx.x);
    if (channel >= num_channels or threadIdx.x >= 32)
        return;
    const int lane = ptx::get_lane_idx();
    const int work_idx = channel;
    const bool invalid_arguments =
        (num_tokens > 0 and
         (x == nullptr or topk_idx == nullptr or topk_weights == nullptr)) or
        proxy_dispatch == nullptr or vnode_arena == nullptr or
        (kHopAware and (hop_records == nullptr or
                        hop_resolutions == nullptr or
                        hop_pair_load == nullptr)) or
        (not kHopAware and
         (channel_count == nullptr or quota == nullptr or
          keep_count == nullptr or segments == nullptr or
          num_segments == nullptr or owner_channel_prefix == nullptr)) or
        retained == nullptr or moved == nullptr or
        moved_channel_prefix == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or status == nullptr or
        num_tokens < 0 or num_experts < 1 or
        num_destinations < 2 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        num_rails < 2 or num_rails > 32 or
        egress < 0 or egress >= num_rails or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_max_tokens_per_rank < 1 or
        num_tokens > num_max_tokens_per_rank or proxy_capacity < 1 or
        generation <= 0 or num_experts % num_destinations != 0 or
        num_experts % (num_destinations * num_rails) != 0;
    if (invalid_arguments) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::
                    InvalidArguments));
        return;
    }
    if (__ldg(status + work_idx) != 0)
        return;

    const int num_remote_destinations = num_destinations - 1;
    const int destination_capacity = num_max_tokens_per_rank;
    const int experts_per_destination = num_experts / num_destinations;
    const int channel_capacity =
        num_max_tokens_per_rank / num_channels +
        (num_max_tokens_per_rank % num_channels != 0);
    const auto vnode_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_remote_destinations,
        destination_capacity, num_rails, num_max_tokens_per_rank,
        vnode_arena);
    const auto dispatch_layout = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, true);
    const int dispatch_token_bytes = dispatch_layout.get_num_bytes<false>();
    EP_DEVICE_ASSERT(dispatch_token_bytes % kVectorBytes == 0);
    EP_DEVICE_ASSERT(vnode_layout.rail.records.record_bytes % kVectorBytes == 0);

    const auto publish_record = [&] __device__ (
            const int physical_slot,
            const int destination,
            const int owner,
            const int owner_token,
            const int owner_ordinal,
            const int logical_slot,
            const int proxy_slot) -> bool {
        int empty_error = 0;
        if (lane == 0)
            empty_error = rail_balance::vnode_detail::validate_empty_stage(
                vnode_layout.rail, physical_slot);
        empty_error = ptx::exchange(empty_error, 0);
        if (empty_error != 0)
            return false;

        auto* dst_record = reinterpret_cast<int4*>(
            vnode_layout.rail.records.get_record_ptr(physical_slot));
        for (int64_t word = lane;
             word < vnode_layout.rail.records.record_bytes / kVectorBytes;
             word += 32)
            dst_record[word] = make_int4(0, 0, 0, 0);
        __syncwarp();

        const auto dst_token =
            vnode_layout.rail.records.get_token_layout(physical_slot);
        if (proxy_slot < 0) {
            const auto* src_hidden = reinterpret_cast<const int4*>(
                x + static_cast<int64_t>(owner_token) * kHidden);
            auto* dst_hidden = reinterpret_cast<int4*>(
                dst_token.get_hidden_ptr());
            for (int word = lane;
                 word < kNumHiddenBytes / kVectorBytes;
                 word += 32)
                dst_hidden[word] = __ldg(src_hidden + word);
            if (lane < kNumTopk) {
                const auto source_offset =
                    static_cast<int64_t>(owner_token) * kNumTopk + lane;
                dst_token.get_topk_idx_ptr()[lane] =
                    static_cast<int>(__ldg(topk_idx + source_offset));
                dst_token.get_topk_weights_ptr()[lane] =
                    __ldg(topk_weights + source_offset);
                dst_token.get_linked_list_idx_ptr()[lane] = -1;
            }
            if (lane == 0)
                *dst_token.get_src_token_global_idx_ptr() =
                    owner * num_max_tokens_per_rank + owner_token;
        } else {
            const auto* src_record = reinterpret_cast<const int4*>(
                math::advance_ptr(
                    const_cast<void*>(proxy_dispatch),
                    static_cast<int64_t>(proxy_slot) *
                        dispatch_token_bytes));
            auto* dst_token_words =
                reinterpret_cast<int4*>(dst_token.get_base_ptr());
            for (int word = lane;
                 word < dispatch_token_bytes / kVectorBytes;
                 word += 32)
                dst_token_words[word] = __ldg(src_record + word);
        }

        if (lane < rail_balance::kNumCanaryBytes /
                       static_cast<int>(sizeof(uint32_t))) {
            vnode_layout.rail.records
                .get_head_canary_ptr(physical_slot)[lane] =
                    rail_balance::kHeadCanary;
            vnode_layout.rail.records
                .get_tail_canary_ptr(physical_slot)[lane] =
                    rail_balance::kTailCanary;
        }
        if (lane == 0) {
            auto* descriptor = vnode_layout.rail.records
                .get_descriptor_ptr(physical_slot);
            *descriptor = rail_balance::ProxyDescriptor{
                rail_balance::hybrid_vnode_detail::fingerprint(
                    generation, owner, owner_token,
                    num_max_tokens_per_rank, num_destinations,
                    destination),
                owner,
                owner_token,
                destination - 1,
                owner_ordinal,
                egress,
                channel,
                logical_slot,
                physical_slot,
                generation,
                owner * num_max_tokens_per_rank + owner_token,
                {0, 0, 0, 0},
            };
        }
        __syncwarp();
        __threadfence_system();
        __syncwarp();
        if (lane == 0)
            ptx::st_release_sys(
                vnode_layout.rail.records.get_ready_ptr(physical_slot),
                generation);
        return true;
    };

    if constexpr (kHopAware) {
        const int64_t owner_stride =
            static_cast<int64_t>(num_max_tokens_per_rank) * kNumTopk;
        for (int destination = 1;
             destination < num_destinations; ++destination) {
            const auto group_offset =
                rail_balance::hybrid_plan_detail::gcd_offset(
                    egress, channel, destination,
                    num_channels, num_destinations);
            const int retained_count = __ldg(retained + group_offset);
            const int moved_count = __ldg(moved + group_offset);
            const int target = __ldg(
                hop_pair_load + destination * num_rails + egress);
            int channel_base = 0;
            for (int previous = 0; previous < channel; ++previous) {
                const auto previous_offset =
                    rail_balance::hybrid_plan_detail::gcd_offset(
                        egress, previous, destination,
                        num_channels, num_destinations);
                channel_base += __ldg(retained + previous_offset) +
                                __ldg(moved + previous_offset);
            }
            const int required = __ldg(proxy_required + egress);
            const bool group_valid =
                retained_count >= 0 and moved_count >= 0 and
                channel_base >= 0 and
                channel_base + retained_count + moved_count <= target and
                (channel != num_channels - 1 or
                 channel_base + retained_count + moved_count == target) and
                target >= 0 and target <= destination_capacity and
                required >= 0 and required <= proxy_capacity;
            if (not group_valid) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidPlan));
                return;
            }

            int emitted = 0;
            for (int owner = 0; owner < num_rails; ++owner) {
                for (int token = 0;
                     token < num_max_tokens_per_rank; ++token) {
                    for (int record_slot = 0;
                         record_slot < kNumTopk; ++record_slot) {
                        const int64_t index =
                            static_cast<int64_t>(owner) * owner_stride +
                            static_cast<int64_t>(token) * kNumTopk +
                            record_slot;
                        const auto record = hop_records[index];
                        const auto resolution = hop_resolutions[index];
                        if (record.target_mask == 0 or
                            record.destination != destination or
                            resolution.egress != egress or
                            resolution.channel != channel)
                            continue;
                        const bool is_moved = owner != egress;
                        const int count = is_moved ? moved_count :
                                                   retained_count;
                        const int proxy_slot = is_moved ?
                            resolution.proxy_slot : -1;
                        const int expected_proxy_slot = is_moved ?
                            __ldg(group_prefix + group_offset) +
                                resolution.remote_slot : -1;
                        const int logical_slot =
                            (is_moved ? retained_count : 0) +
                            resolution.remote_slot;
                        const int dense_slot = channel_base + logical_slot;
                        const int physical_slot =
                            (destination - 1) * destination_capacity +
                            dense_slot;
                        const bool resolution_valid =
                            resolution.remote_slot >= 0 and
                            resolution.remote_slot < count and
                            ((not is_moved and proxy_slot == -1 and
                              token < num_tokens) or
                             (is_moved and proxy_slot >= 0 and
                              proxy_slot < required and
                              proxy_slot == expected_proxy_slot));
                        if (not resolution_valid or
                            not publish_record(
                                physical_slot, destination, owner, token,
                                token * kNumTopk + record_slot,
                                logical_slot, proxy_slot)) {
                            rail_balance::hybrid_vnode_detail::report(
                                status, work_idx,
                                static_cast<int>(
                                    rail_balance::hybrid_vnode_detail::
                                        AdapterError::InvalidProxy));
                            return;
                        }
                        ++emitted;
                    }
                }
            }
            if (emitted != retained_count + moved_count) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidPlan));
                return;
            }
        }
        return;
    }

    for (int destination = 1;
         destination < num_destinations;
         ++destination) {
        const auto tensor_offset =
            rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel, destination,
                num_channels, num_destinations);
        const auto matrix_offset =
            rail_balance::hybrid_plan_detail::gd_offset(
                egress, destination, num_destinations);
        const auto moved_prefix_offset =
            rail_balance::hybrid_plan_detail::moved_prefix_offset(
                egress, destination, channel,
                num_channels, num_destinations);
        const int available = __ldg(channel_count + tensor_offset);
        const int target = __ldg(quota + matrix_offset);
        const int keep = __ldg(keep_count + matrix_offset);
        const int owner_prefix =
            __ldg(owner_channel_prefix + tensor_offset);
        const int retained_count = __ldg(retained + tensor_offset);
        const int moved_count = __ldg(moved + tensor_offset);
        const int incoming_begin =
            __ldg(moved_channel_prefix + moved_prefix_offset);
        const int incoming_end =
            __ldg(moved_channel_prefix + moved_prefix_offset + 1);
        const int proxy_begin = __ldg(group_prefix + tensor_offset);
        const int required = __ldg(proxy_required + egress);
        const int channel_base =
            (owner_prefix < keep ? owner_prefix : keep) + incoming_begin;
        const int channel_end =
            channel_base + retained_count + moved_count;

        const int remaining_keep = keep - owner_prefix;
        const int expected_retained = remaining_keep <= 0 ? 0 :
            (available < remaining_keep ? available : remaining_keep);
        const bool plan_valid =
            available >= 0 and target >= 0 and
            target <= destination_capacity and
            keep >= 0 and keep <= target and owner_prefix >= 0 and
            retained_count == expected_retained and moved_count >= 0 and
            incoming_begin >= 0 and incoming_end >= incoming_begin and
            incoming_end - incoming_begin == moved_count and
            proxy_begin >= 0 and proxy_begin + moved_count <= required and
            required >= 0 and required <= proxy_capacity and
            retained_count + moved_count <= channel_capacity and
            channel_base >= 0 and channel_end <= target and
            (channel != num_channels - 1 or channel_end == target);
        if (not plan_valid) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidPlan));
            return;
        }

        int destination_ordinal = 0;
        int emitted_retained = 0;
        for (int token = channel;
             token < num_tokens;
             token += num_channels) {
            const auto topk_offset =
                static_cast<int64_t>(token) * kNumTopk;
            const bool active = lane < kNumTopk;
            const int expert = active ?
                static_cast<int>(__ldg(topk_idx + topk_offset + lane)) : -1;
            const bool expert_valid = not active or
                (expert >= 0 and expert < num_experts);
            unsigned invalid_mask = ptx::gather(not expert_valid);
            bool duplicate = false;
            for (int source_lane = 0;
                 source_lane < kNumTopk;
                 ++source_lane) {
                const int other = ptx::exchange(expert, source_lane);
                duplicate |= active and source_lane < lane and
                             expert == other;
            }
            invalid_mask |= ptx::gather(duplicate);
            const int token_destination = active ?
                expert / experts_per_destination : -1;
            invalid_mask |= ptx::gather(
                active and token_destination == 0);
            if (invalid_mask != 0) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidSource));
                return;
            }
            const unsigned destination_mask = ptx::gather(
                active and token_destination == destination);
            if (destination_mask == 0)
                continue;

            const int local_ordinal = destination_ordinal++;
            if (local_ordinal >= retained_count)
                continue;
            const int dense_slot = channel_base + local_ordinal;
            const int physical_slot =
                (destination - 1) * destination_capacity + dense_slot;
            if (dense_slot < 0 or dense_slot >= target or
                not publish_record(
                    physical_slot, destination, egress, token,
                    owner_prefix + local_ordinal, local_ordinal, -1)) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidBase));
                return;
            }
            ++emitted_retained;
        }
        if (destination_ordinal != available or
            emitted_retained != retained_count) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidPlan));
            return;
        }

        for (int local = 0; local < moved_count; ++local) {
            const int incoming_ordinal = incoming_begin + local;
            int owner = -1;
            int owner_ordinal = -1;
            if (not rail_balance::hybrid_vnode_detail::invert_moved_ordinal(
                    segments, num_segments, num_rails, destination,
                    egress, incoming_ordinal, owner, owner_ordinal)) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidPlan));
                return;
            }
            const int proxy_slot = proxy_begin + local;
            const auto proxy_token = layout::TokenLayout(
                kNumHiddenBytes, 0, kNumTopk, true,
                math::advance_ptr(
                    const_cast<void*>(proxy_dispatch),
                    static_cast<int64_t>(proxy_slot) *
                        dispatch_token_bytes));
            int owner_token = -1;
            const bool proxy_valid =
                proxy_slot >= 0 and proxy_slot < required and
                owner != egress and
                rail_balance::hybrid_vnode_detail::validate_dispatch_token(
                    proxy_token, proxy_slot, owner, destination,
                    num_experts, num_destinations, num_rails,
                    num_max_tokens_per_rank, owner_token);
            const int logical_slot = retained_count + local;
            const int dense_slot = channel_base + logical_slot;
            const int physical_slot =
                (destination - 1) * destination_capacity + dense_slot;
            if (not proxy_valid or logical_slot < 0 or
                logical_slot >= channel_capacity or dense_slot < 0 or
                dense_slot >= target or
                not publish_record(
                    physical_slot, destination, owner, owner_token,
                    owner_ordinal, logical_slot, proxy_slot)) {
                rail_balance::hybrid_vnode_detail::report(
                    status, work_idx,
                    static_cast<int>(
                        rail_balance::hybrid_vnode_detail::AdapterError::
                            InvalidProxy));
                return;
            }
        }
    }
}

// Test-only adapter from old vnode lane contributions back into Hybrid's
// destination-level combine records. One warp validates and reduces one old
// vnode base slot. Retained records seed the legacy reduce buffer locally;
// moved records return to proxy_return[p] for the production LSA unshuffle.
template <int kHidden, int kNumTopk, bool kHopAware = false>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_return_demux_impl(
        const void* vnode_arena,
        const void* proxy_dispatch,
        void* reduce_seed,
        void* proxy_return,
        const int* quota,
        const int* keep_count,
        const int* segments,
        const int* num_segments,
        const int* owner_channel_prefix,
        const int* retained,
        const int* moved,
        const int* moved_channel_prefix,
        const int* group_prefix,
        const int* proxy_required,
        const rail_balance::HopCopyRecord* hop_records,
        const rail_balance::HopCopyResolution* hop_resolutions,
        const int* hop_pair_load,
        int* status,
        const int num_experts,
        const int num_destinations,
        const int num_rails,
        const int egress,
        const int num_channels,
        const int num_max_tokens_per_rank,
        const int proxy_capacity,
        const int generation) {
    static_assert(kHidden > 0 and kHidden % 256 == 0);
    static_assert(kNumTopk >= 1 and kNumTopk <= 32);
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    constexpr int kVectorBytes = sizeof(int4);

    const int physical_slot = static_cast<int>(blockIdx.x);
    const int num_remote_destinations = num_destinations - 1;
    const int physical_capacity =
        num_remote_destinations * num_max_tokens_per_rank;
    if (physical_slot >= physical_capacity or threadIdx.x >= 32)
        return;
    const int lane = ptx::get_lane_idx();
    const int work_idx = physical_slot;
    const bool invalid_arguments =
        vnode_arena == nullptr or proxy_dispatch == nullptr or
        reduce_seed == nullptr or proxy_return == nullptr or
        (kHopAware and (hop_records == nullptr or
                        hop_resolutions == nullptr or
                        hop_pair_load == nullptr)) or
        (not kHopAware and
         (quota == nullptr or keep_count == nullptr or segments == nullptr or
          num_segments == nullptr or owner_channel_prefix == nullptr)) or
        retained == nullptr or moved == nullptr or
        moved_channel_prefix == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or status == nullptr or
        num_experts < 1 or num_destinations < 2 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        num_rails < 2 or num_rails > 32 or
        egress < 0 or egress >= num_rails or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_max_tokens_per_rank < 1 or proxy_capacity < 1 or
        generation <= 0 or num_experts % num_destinations != 0 or
        num_experts % (num_destinations * num_rails) != 0;
    if (invalid_arguments) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::
                    InvalidArguments));
        return;
    }
    if (__ldg(status + work_idx) != 0)
        return;

    const auto vnode_layout = rail_balance::VNodeRoundTripLayout(
        kNumHiddenBytes, kNumTopk, num_remote_destinations,
        num_max_tokens_per_rank, num_rails,
        num_max_tokens_per_rank, const_cast<void*>(vnode_arena));
    const int old_destination =
        physical_slot / num_max_tokens_per_rank;
    const int destination = old_destination + 1;
    const int dense_slot =
        physical_slot % num_max_tokens_per_rank;
    const auto matrix_offset =
        rail_balance::hybrid_plan_detail::gd_offset(
            egress, destination, num_destinations);
    const int target = kHopAware ?
        __ldg(hop_pair_load + destination * num_rails + egress) :
        __ldg(quota + matrix_offset);
    if (target < 0 or target > num_max_tokens_per_rank) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidPlan));
        return;
    }

    if (dense_slot >= target) {
        int error = 0;
        if (lane == 0)
            error = rail_balance::vnode_detail::validate_empty_stage(
                vnode_layout.rail, physical_slot);
        error = ptx::exchange(error, 0);
        for (int topk_lane = 0;
             topk_lane < kNumTopk and error == 0;
             ++topk_lane) {
            if (lane == 0)
                error = rail_balance::vnode_detail::validate_empty_stage(
                    vnode_layout.rail,
                    vnode_layout.get_contribution_slot(
                        physical_slot, topk_lane));
            error = ptx::exchange(error, 0);
        }
        if (error != 0)
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx, error);
        return;
    }

    int error = 0;
    if (lane == 0)
        error = rail_balance::vnode_detail::validate_base_record(
            vnode_layout.rail.records, physical_slot, generation,
            physical_slot, egress, old_destination);
    error = ptx::exchange(error, 0);
    if (error != 0) {
        rail_balance::hybrid_vnode_detail::report(status, work_idx, error);
        return;
    }

    int source_channel = -1;
    int channel_local = -1;
    int retained_count = -1;
    int moved_count = -1;
    int owner_prefix = -1;
    int incoming_begin = -1;
    int proxy_begin = -1;
    int matches = 0;
    int cursor = 0;
    const int keep = kHopAware ? 0 : __ldg(keep_count + matrix_offset);
    for (int channel = 0; channel < num_channels; ++channel) {
        const auto tensor_offset =
            rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel, destination,
                num_channels, num_destinations);
        const auto prefix_offset =
            rail_balance::hybrid_plan_detail::moved_prefix_offset(
                egress, destination, channel,
                num_channels, num_destinations);
        const int channel_owner_prefix = kHopAware ? 0 :
            __ldg(owner_channel_prefix + tensor_offset);
        const int channel_retained = __ldg(retained + tensor_offset);
        const int channel_moved = __ldg(moved + tensor_offset);
        const int channel_incoming_begin = kHopAware ? 0 :
            __ldg(moved_channel_prefix + prefix_offset);
        const int channel_incoming_end = kHopAware ? channel_moved :
            __ldg(moved_channel_prefix + prefix_offset + 1);
        const int channel_proxy_begin = __ldg(group_prefix + tensor_offset);
        const int computed_base = kHopAware ? cursor :
            (channel_owner_prefix < keep ? channel_owner_prefix : keep) +
                channel_incoming_begin;
        if (channel_owner_prefix < 0 or channel_retained < 0 or
            channel_moved < 0 or channel_incoming_begin < 0 or
            channel_incoming_end - channel_incoming_begin != channel_moved or
            computed_base != cursor) {
            error = static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidPlan);
            break;
        }
        const int end = computed_base + channel_retained + channel_moved;
        if (computed_base <= dense_slot and dense_slot < end) {
            source_channel = channel;
            channel_local = dense_slot - computed_base;
            retained_count = channel_retained;
            moved_count = channel_moved;
            owner_prefix = channel_owner_prefix;
            incoming_begin = channel_incoming_begin;
            proxy_begin = channel_proxy_begin;
            ++matches;
        }
        cursor = end;
    }
    if (error != 0 or matches != 1 or cursor != target) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            error != 0 ? error : static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidPlan));
        return;
    }

    const bool is_moved = channel_local >= retained_count;
    const int group_local = channel_local - retained_count;
    const int logical_slot = channel_local;
    int proxy_slot = -1;
    int expected_owner = egress;
    int expected_owner_ordinal = owner_prefix + channel_local;
    if (is_moved) {
        if (group_local < 0 or group_local >= moved_count) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidPlan));
            return;
        }
        proxy_slot = proxy_begin + group_local;
        const int required = __ldg(proxy_required + egress);
        bool route_valid = proxy_slot >= 0 and proxy_slot < required and
            required >= 0 and required <= proxy_capacity;
        if constexpr (not kHopAware) {
            route_valid = route_valid and
                rail_balance::hybrid_vnode_detail::invert_moved_ordinal(
                    segments, num_segments, num_rails, destination,
                    egress, incoming_begin + group_local,
                    expected_owner, expected_owner_ordinal) and
                expected_owner != egress;
        }
        if (not route_valid) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidPlan));
            return;
        }
    }

    const auto base_token =
        vnode_layout.rail.records.get_token_layout(physical_slot);
    const auto* base_descriptor =
        vnode_layout.rail.records.get_descriptor_ptr(physical_slot);
    int owner_token = lane == 0 ? base_descriptor->owner_token : 0;
    int base_src_token_global_idx = lane == 0 ?
        __ldg(base_token.get_src_token_global_idx_ptr()) : 0;
    owner_token = ptx::exchange(owner_token, 0);
    base_src_token_global_idx = ptx::exchange(
        base_src_token_global_idx, 0);
    const int descriptor_owner = base_src_token_global_idx >= 0 ?
        base_src_token_global_idx / num_max_tokens_per_rank : -1;
    if constexpr (kHopAware) {
        int matched_slot = -1;
        int record_matches = 0;
        if (descriptor_owner >= 0 and descriptor_owner < num_rails and
            owner_token >= 0 and
            owner_token < num_max_tokens_per_rank) {
            const int64_t base_index =
                (static_cast<int64_t>(descriptor_owner) *
                     num_max_tokens_per_rank + owner_token) * kNumTopk;
            for (int record_slot = 0;
                 record_slot < kNumTopk; ++record_slot) {
                const auto record = hop_records[base_index + record_slot];
                const auto resolution =
                    hop_resolutions[base_index + record_slot];
                const int expected_remote = is_moved ? group_local :
                                                       channel_local;
                if (record.target_mask != 0 and
                    record.destination == destination and
                    resolution.egress == egress and
                    resolution.channel == source_channel and
                    resolution.remote_slot == expected_remote and
                    resolution.proxy_slot == proxy_slot) {
                    matched_slot = record_slot;
                    ++record_matches;
                }
            }
        }
        if (record_matches == 1 and
            (descriptor_owner != egress) == is_moved) {
            expected_owner = descriptor_owner;
            expected_owner_ordinal =
                owner_token * kNumTopk + matched_slot;
        } else {
            error = static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidPlan);
        }
    }
    if (lane == 0)
        error = error != 0 ? error :
            rail_balance::hybrid_vnode_detail::validate_vnode_descriptor(
                base_descriptor, generation, expected_owner, owner_token,
                old_destination, expected_owner_ordinal, egress,
                source_channel, logical_slot, physical_slot,
                num_max_tokens_per_rank, num_destinations);
    error = ptx::exchange(error, 0);
    if (owner_token < 0 or owner_token >= num_max_tokens_per_rank or
        static_cast<int64_t>(base_src_token_global_idx) !=
            static_cast<int64_t>(expected_owner) *
                num_max_tokens_per_rank + owner_token or
        error != 0) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            error != 0 ? error : static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidBase));
        return;
    }

    if (is_moved) {
        const auto proxy_token = layout::TokenLayout(
            kNumHiddenBytes, 0, kNumTopk, true,
            math::advance_ptr(
                const_cast<void*>(proxy_dispatch),
                static_cast<int64_t>(proxy_slot) *
                    base_token.get_num_bytes<false>()));
        int proxy_owner_token = -1;
        const bool proxy_valid =
            rail_balance::hybrid_vnode_detail::validate_dispatch_token(
                proxy_token, proxy_slot, expected_owner, destination,
                num_experts, num_destinations, num_rails,
                num_max_tokens_per_rank, proxy_owner_token);
        unsigned mismatch = 0;
        const auto* lhs = reinterpret_cast<const int4*>(
            base_token.get_base_ptr());
        const auto* rhs = reinterpret_cast<const int4*>(
            proxy_token.get_base_ptr());
        for (int word = lane;
             word < base_token.get_num_bytes<false>() / kVectorBytes;
             word += 32) {
            const int4 a = __ldg(lhs + word);
            const int4 b = __ldg(rhs + word);
            mismatch |= a.x != b.x or a.y != b.y or
                        a.z != b.z or a.w != b.w;
        }
        if (not proxy_valid or proxy_owner_token != owner_token or
            ptx::reduce_or(mismatch) != 0) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidProxy));
            return;
        }
    } else {
        const int linked = lane < kNumTopk ?
            __ldg(base_token.get_linked_list_idx_ptr() + lane) : -1;
        if (ptx::gather(lane < kNumTopk and linked != -1) != 0 or
            (not kHopAware and
             owner_token % num_channels != source_channel)) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx,
                static_cast<int>(
                    rail_balance::hybrid_vnode_detail::AdapterError::
                        InvalidBase));
            return;
        }
    }

    const int experts_per_destination = num_experts / num_destinations;
    const int experts_per_physical_rank =
        num_experts / (num_destinations * num_rails);
    const bool active = lane < kNumTopk;
    const int expert = active ?
        __ldg(base_token.get_topk_idx_ptr() + lane) : -1;
    const bool expert_valid = not active or
        (expert >= experts_per_destination and expert < num_experts);
    unsigned invalid_mask = ptx::gather(not expert_valid);
    bool duplicate = false;
    for (int source_lane = 0; source_lane < kNumTopk; ++source_lane) {
        const int other = ptx::exchange(expert, source_lane);
        duplicate |= active and source_lane < lane and expert == other;
    }
    invalid_mask |= ptx::gather(duplicate);
    const int lane_destination = active ?
        expert / experts_per_destination : -1;
    const unsigned matching_mask = ptx::gather(
        active and lane_destination == destination);
    if (invalid_mask != 0 or matching_mask == 0) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::
                    InvalidBase));
        return;
    }

    for (int topk_lane = 0; topk_lane < kNumTopk; ++topk_lane) {
        const int contribution_slot = vnode_layout.get_contribution_slot(
            physical_slot, topk_lane);
        const bool expected = ((matching_mask >> topk_lane) & 1u) != 0;
        if (lane == 0) {
            if (expected) {
                error = rail_balance::vnode_detail::validate_base_record(
                    vnode_layout.rail.records, contribution_slot,
                    generation, physical_slot, egress, old_destination);
                if (error == 0) {
                    const auto* contribution_descriptor =
                        vnode_layout.rail.records.get_descriptor_ptr(
                            contribution_slot);
                    if (not rail_balance::hybrid_vnode_detail::descriptor_equal(
                            contribution_descriptor, base_descriptor) or
                        not rail_balance::hybrid_vnode_detail::metadata_equal(
                            vnode_layout.rail.records.get_token_layout(
                                contribution_slot),
                            base_token))
                        error = static_cast<int>(
                            rail_balance::hybrid_vnode_detail::AdapterError::
                                InvalidContribution);
                    if (error == 0) {
                        const auto route = *vnode_layout.rail.get_route_ptr(
                            contribution_slot);
                        const int contribution_expert = __ldg(
                            base_token.get_topk_idx_ptr() + topk_lane);
                        const int expected_expert_rank =
                            contribution_expert /
                            experts_per_physical_rank;
                        const int destination_slot = dense_slot;
                        const int expected_expert_slot =
                            vnode_layout.get_expert_slot(
                                egress, destination_slot, topk_lane);
                        error = rail_balance::vnode_detail::validate_route(
                            route, contribution_descriptor, generation,
                            vnode_layout.get_ingress_rank(
                                old_destination, egress),
                            physical_slot, topk_lane,
                            expected_expert_rank, expected_expert_slot);
                    }
                }
            } else {
                error = rail_balance::vnode_detail::validate_empty_stage(
                    vnode_layout.rail, contribution_slot);
            }
        }
        error = ptx::exchange(error, 0);
        if (error != 0) {
            rail_balance::hybrid_vnode_detail::report(
                status, work_idx, error);
            return;
        }
    }

    int reduce_row = destination;
    if constexpr (kNumTopk < rail_balance::kNumHybridMaxDestinations) {
        if (num_destinations > kNumTopk)
            reduce_row = ptx::get_master_lane_idx(matching_mask);
    }
    const int num_reduce_rows =
        num_destinations <= kNumTopk ? num_destinations : kNumTopk;
    if (reduce_row < 0 or reduce_row >= num_reduce_rows) {
        rail_balance::hybrid_vnode_detail::report(
            status, work_idx,
            static_cast<int>(
                rail_balance::hybrid_vnode_detail::AdapterError::InvalidPlan));
        return;
    }

    const auto combine_template = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false);
    const int combine_token_bytes = combine_template.get_num_bytes<false>();
    EP_DEVICE_ASSERT(combine_token_bytes % kVectorBytes == 0);
    void* output_base = nullptr;
    if (is_moved) {
        output_base = math::advance_ptr(
            proxy_return,
            static_cast<int64_t>(proxy_slot) * combine_token_bytes);
    } else {
        const auto reduce_layout = layout::BufferLayout<false>(
            combine_template, num_reduce_rows,
            num_max_tokens_per_rank, reduce_seed);
        output_base = reduce_layout.get_rank_buffer(reduce_row)
            .get_token_buffer(owner_token).get_base_ptr();
    }
    const auto output_token = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false, output_base);

    const int metadata_bytes = combine_token_bytes - kNumHiddenBytes;
    auto* output_metadata = reinterpret_cast<int4*>(
        output_token.get_metadata_ptr());
    for (int word = lane; word < metadata_bytes / kVectorBytes; word += 32)
        output_metadata[word] = make_int4(0, 0, 0, 0);
    __syncwarp();
    if (lane < kNumTopk) {
        output_token.get_topk_idx_ptr()[lane] =
            __ldg(base_token.get_topk_idx_ptr() + lane);
        output_token.get_topk_weights_ptr()[lane] =
            __ldg(base_token.get_topk_weights_ptr() + lane);
    }

    int contribution_slots[kNumTopk];
    compute_topk_slots(
        contribution_slots, matching_mask,
        [=] __device__ (const int& topk_lane) {
            return topk_lane < 0 ? -1 :
                vnode_layout.get_contribution_slot(
                    physical_slot, topk_lane);
        });
    using combine_vec_t =
        typename CombineVecTraits<kNumHiddenBytes>::vec_t;
    constexpr int kHiddenVec = kNumHiddenBytes / sizeof(combine_vec_t);
    constexpr int kUnrollFactor =
        get_max_unroll_factor<kHiddenVec, 4>();
    combine_reduce<kHiddenVec, kUnrollFactor, 1>(
        lane, contribution_slots,
        static_cast<combine_vec_t*>(output_token.get_hidden_ptr()),
        [=] __device__ (const int& slot) {
            // ldg_with_gez_pred suppresses the load for slot < 0, but its
            // caller still performs pointer arithmetic. Keep that pointer in
            // a valid allocation instead of manufacturing nullptr arithmetic.
            return slot < 0 ? static_cast<combine_vec_t*>(
                vnode_layout.rail.records.get_token_layout(physical_slot)
                    .get_hidden_ptr()) :
                static_cast<combine_vec_t*>(
                    vnode_layout.rail.records.get_token_layout(slot)
                        .get_hidden_ptr());
        },
        [] __device__ () {});
}

}  // namespace deep_ep::elastic
