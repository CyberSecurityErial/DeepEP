#pragma once

#include <climits>
#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>

namespace deep_ep::elastic::rail_balance {

static constexpr int kNumHybridSegmentFields = 5;

enum HybridSegmentField : int {
    kHybridSegmentOwner = 0,
    kHybridSegmentEgress = 1,
    kHybridSegmentOwnerBegin = 2,
    kHybridSegmentCount = 3,
    kHybridSegmentEgressBegin = 4,
};

struct HybridCopyResolution {
    int moved;
    int egress;
    int channel;
    int remote_slot;
    int proxy_slot;
    int incoming_ordinal;
};

namespace hybrid_plan_detail {

__forceinline__ __device__ __host__ int64_t gcd_offset(
        const int owner,
        const int channel,
        const int destination,
        const int num_channels,
        const int num_destinations) {
    return (static_cast<int64_t>(owner) * num_channels + channel) *
        num_destinations + destination;
}

__forceinline__ __device__ __host__ int64_t gd_offset(
        const int owner,
        const int destination,
        const int num_destinations) {
    return static_cast<int64_t>(owner) * num_destinations + destination;
}

__forceinline__ __device__ __host__ int64_t segment_offset(
        const int destination,
        const int segment,
        const int field,
        const int num_rails) {
    return (static_cast<int64_t>(destination) * (num_rails - 1) + segment) *
        kNumHybridSegmentFields + field;
}

__forceinline__ __device__ __host__ int64_t moved_prefix_offset(
        const int egress,
        const int destination,
        const int channel,
        const int num_channels,
        const int num_destinations) {
    return (static_cast<int64_t>(egress) * num_destinations + destination) *
        (num_channels + 1) + channel;
}

__forceinline__ __device__ void report_error(
        int* status, const HybridPlanError error) {
    if (status != nullptr)
        atomicCAS(status, static_cast<int>(HybridPlanError::Success),
                  static_cast<int>(error));
}

}  // namespace hybrid_plan_detail

// Materialize the endpoint information omitted by the legacy [G,C,D] count.
// Output shape is [owner, token, num_topk]. Remote destinations are sorted by
// destination id; unused slots have target_mask=0 and destination=-1.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hop_record_impl(
        const topk_idx_t* topk_idx,
        HopCopyRecord* records,
        int* status,
        const int num_owners,
        const int num_tokens,
        const int record_token_capacity,
        const int num_topk,
        const int num_channels,
        const int num_experts,
        const int num_destinations,
        const int num_rails,
        const int local_destination) {
    if ((num_tokens > 0 and topk_idx == nullptr) or
        records == nullptr or status == nullptr or
        num_owners < 1 or num_owners > 32 or num_tokens < 0 or
        record_token_capacity < num_tokens or
        num_topk < 1 or num_topk > 32 or num_channels < 1 or
        num_channels > kNumHybridMaxChannels or num_experts < 1 or
        num_destinations < 2 or num_destinations > 32 or
        num_rails < 1 or num_rails > 32 or
        num_experts % (num_destinations * num_rails) != 0 or
        local_destination < 0 or local_destination >= num_destinations) {
        if (threadIdx.x == 0)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::InvalidSchedule);
        return;
    }

    const int owner_channel = static_cast<int>(blockIdx.x);
    const int owner = owner_channel / num_channels;
    const int channel = owner_channel % num_channels;
    if (owner >= num_owners)
        return;

    const int lane = ptx::get_lane_idx();
    const int experts_per_destination = num_experts / num_destinations;
    const int experts_per_rank = experts_per_destination / num_rails;
    for (int token = channel;
         token < record_token_capacity;
         token += num_channels) {
        const auto record_offset =
            (static_cast<int64_t>(owner) * record_token_capacity + token) *
            num_topk;
        if (lane < num_topk)
            records[record_offset + lane] = {0u, -1};
        if (token >= num_tokens)
            continue;

        const auto token_offset =
            (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;

        const bool active = lane < num_topk;
        const topk_idx_t expert = active ?
            __ldg(topk_idx + token_offset + lane) : topk_idx_t{-1};
        const bool in_range = not active or
            (expert >= topk_idx_t{0} and expert < num_experts);
        if (ptx::gather(not in_range) != 0) {
            if (lane == 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::ExpertOutOfRange);
            continue;
        }

        bool duplicate = false;
        for (int source_lane = 0; source_lane < num_topk; ++source_lane) {
            const auto other = ptx::exchange(expert, source_lane);
            duplicate |= active and source_lane < lane and expert == other;
        }
        if (ptx::gather(duplicate) != 0) {
            if (lane == 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::DuplicateExpert);
            continue;
        }

        const int destination = active ?
            static_cast<int>(expert) / experts_per_destination : -1;
        const int target = active ?
            (static_cast<int>(expert) % experts_per_destination) /
                experts_per_rank : -1;
        uint32_t target_mask = 0;
        for (int source_lane = 0; source_lane < num_topk; ++source_lane) {
            // Every lane must execute full-mask shuffles. Only the endpoint
            // lanes consume the exchanged values.
            const int other_destination =
                ptx::exchange(destination, source_lane);
            const int other_target = ptx::exchange(target, source_lane);
            if (lane < num_destinations and lane != local_destination and
                    other_destination == lane)
                target_mask |= uint32_t{1} << other_target;
        }
        const bool present = target_mask != 0;
        const unsigned present_mask = ptx::gather(present);
        if (present) {
            const int slot = __popc(present_mask & ((1u << lane) - 1u));
            records[record_offset + slot] = {target_mask, lane};
        }
        __syncwarp();
    }
}

// Correctness-first deterministic one-hop planner. One lane performs the
// fixed-order assignment; later profiling decides whether this phase warrants
// a parallel implementation. There are no data-plane atomics or queues.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hop_one_hop_plan_impl(
        const HopCopyRecord* records,
        HopCopyResolution* resolutions,
        int* pair_load,
        int* source_load,
        int* owner_remaining,
        int* retained,
        int* moved,
        int* group_prefix,
        int* proxy_required,
        int* path_units,
        int* moved_copies,
        int* status,
        const int num_rails,
        const int num_tokens,
        const int num_topk,
        const int num_channels,
        const int num_destinations,
        const int num_max_tokens_per_rank,
        const int proxy_capacity_per_egress,
        const int planner_seed) {
    if (threadIdx.x != 0)
        return;
    if (records == nullptr or resolutions == nullptr or pair_load == nullptr or
        source_load == nullptr or owner_remaining == nullptr or
        retained == nullptr or moved == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or path_units == nullptr or
        moved_copies == nullptr or status == nullptr or num_rails < 1 or
        num_rails > 32 or num_tokens < 0 or num_topk < 1 or num_topk > 32 or
        num_channels < 1 or num_channels > kNumHybridMaxChannels or
        num_destinations < 2 or num_destinations > 32 or
        num_max_tokens_per_rank < num_tokens or
        proxy_capacity_per_egress < 0 or planner_seed < 0) {
        hybrid_plan_detail::report_error(
            status, HybridPlanError::InvalidSchedule);
        return;
    }

    const int64_t num_records = static_cast<int64_t>(num_rails) *
        num_tokens * num_topk;
    const int num_groups = num_rails * num_channels * num_destinations;
    for (int rail = 0; rail < num_rails; ++rail) {
        source_load[rail] = 0;
        proxy_required[rail] = 0;
    }
    for (int i = 0; i < num_rails * num_destinations; ++i) {
        pair_load[i] = 0;
        owner_remaining[i] = 0;
    }
    for (int i = 0; i < num_groups; ++i) {
        retained[i] = 0;
        moved[i] = 0;
        group_prefix[i] = 0;
    }
    for (int path = 0; path < 4; ++path)
        path_units[path] = 0;
    *moved_copies = 0;

    // Validate the fixed table and reserve every owner's still-unscheduled
    // traffic. This prevents inbound moves from consuming an owner's only
    // feasible direct capacity.
    for (int64_t index = 0; index < num_records; ++index) {
        resolutions[index] = {-1, -1, -1, -1};
        const auto record = records[index];
        if (record.target_mask == 0) {
            if (record.destination != -1) {
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
                return;
            }
            continue;
        }
        if (record.destination < 0 or record.destination >= num_destinations or
                (num_rails < 32 and
                 (record.target_mask >> num_rails) != 0)) {
            hybrid_plan_detail::report_error(
                status, HybridPlanError::InvalidSchedule);
            return;
        }
        const int owner = static_cast<int>(
            index / (static_cast<int64_t>(num_tokens) * num_topk));
        ++owner_remaining[owner * num_destinations + record.destination];
    }

    for (int64_t index = 0; index < num_records; ++index) {
        const auto record = records[index];
        if (record.target_mask == 0)
            continue;
        const int owner = static_cast<int>(
            index / (static_cast<int64_t>(num_tokens) * num_topk));
        const int token = static_cast<int>(
            (index / num_topk) % num_tokens);
        const int destination = record.destination;
        --owner_remaining[owner * num_destinations + destination];

        const uint32_t candidates =
            record.target_mask | (uint32_t{1} << owner);
        int best_egress = -1;
        int best_pair_peak = INT_MAX;
        int best_source_peak = INT_MAX;
        int best_local_forwards = INT_MAX;
        int best_tie = INT_MAX;
        for (int egress = 0; egress < num_rails; ++egress) {
            if ((candidates & (uint32_t{1} << egress)) == 0)
                continue;
            const int pair_offset = destination * num_rails + egress;
            if (pair_load[pair_offset] + 1 +
                    owner_remaining[egress * num_destinations + destination] >
                    num_max_tokens_per_rank)
                continue;

            int pair_peak = 0;
            int source_peak = 0;
            for (int rail = 0; rail < num_rails; ++rail) {
                pair_peak = max(
                    pair_peak,
                    pair_load[destination * num_rails + rail] +
                        (rail == egress));
                source_peak = max(
                    source_peak, source_load[rail] + (rail == egress));
            }
            const int local_forwards = (egress != owner) +
                __popc(record.target_mask & ~(uint32_t{1} << egress));
            const int origin = static_cast<int>(
                (static_cast<int64_t>(planner_seed) + destination + index) %
                num_rails);
            const int tie = (egress - origin + num_rails) % num_rails;
            const bool better =
                pair_peak < best_pair_peak or
                (pair_peak == best_pair_peak and
                 (source_peak < best_source_peak or
                  (source_peak == best_source_peak and
                   (local_forwards < best_local_forwards or
                    (local_forwards == best_local_forwards and
                     (tie < best_tie or
                      (tie == best_tie and egress < best_egress)))))));
            if (better) {
                best_egress = egress;
                best_pair_peak = pair_peak;
                best_source_peak = source_peak;
                best_local_forwards = local_forwards;
                best_tie = tie;
            }
        }
        if (best_egress < 0) {
            hybrid_plan_detail::report_error(
                status, HybridPlanError::CapacityExceeded);
            return;
        }

        int best_channel = -1;
        int best_group_load = INT_MAX;
        int best_channel_tie = INT_MAX;
        const int source_channel = token % num_channels;
        const int channel_capacity =
            (num_max_tokens_per_rank + num_channels - 1) / num_channels;
        for (int channel = 0; channel < num_channels; ++channel) {
            const int group =
                (best_egress * num_channels + channel) * num_destinations +
                destination;
            const int group_load = retained[group] + moved[group];
            const int channel_tie =
                (channel - source_channel + num_channels) % num_channels;
            if (group_load < channel_capacity and
                    (group_load < best_group_load or
                     (group_load == best_group_load and
                      (channel_tie < best_channel_tie or
                       (channel_tie == best_channel_tie and
                        channel < best_channel))))) {
                best_channel = channel;
                best_group_load = group_load;
                best_channel_tie = channel_tie;
            }
        }
        if (best_channel < 0) {
            hybrid_plan_detail::report_error(
                status, HybridPlanError::CapacityExceeded);
            return;
        }

        const int group =
            (best_egress * num_channels + best_channel) * num_destinations +
            destination;
        const bool is_moved = best_egress != owner;
        int& group_count = is_moved ? moved[group] : retained[group];
        const int remote_slot = group_count++;
        resolutions[index] = {
            best_egress, best_channel, remote_slot, -1};
        ++pair_load[destination * num_rails + best_egress];
        ++source_load[best_egress];

        int path = static_cast<int>(HopPathKind::TwoHop);
        if (best_egress == owner) {
            path = record.target_mask == (uint32_t{1} << owner) ?
                static_cast<int>(HopPathKind::Direct) :
                static_cast<int>(HopPathKind::DestinationForward);
        } else if ((record.target_mask &
                    (uint32_t{1} << best_egress)) != 0) {
            path = static_cast<int>(HopPathKind::SourceForward);
        }
        ++path_units[path];
        *moved_copies += is_moved;
    }

    for (int egress = 0; egress < num_rails; ++egress) {
        int prefix = 0;
        for (int channel = 0; channel < num_channels; ++channel) {
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const int group =
                    (egress * num_channels + channel) * num_destinations +
                    destination;
                group_prefix[group] = prefix;
                prefix += moved[group];
            }
        }
        proxy_required[egress] = prefix;
        if (prefix > proxy_capacity_per_egress)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::CapacityExceeded);
    }

    for (int64_t index = 0; index < num_records; ++index) {
        auto& resolution = resolutions[index];
        if (resolution.egress < 0)
            continue;
        const int owner = static_cast<int>(
            index / (static_cast<int64_t>(num_tokens) * num_topk));
        if (resolution.egress == owner)
            continue;
        const int destination = records[index].destination;
        const int group =
            (resolution.egress * num_channels + resolution.channel) *
                num_destinations + destination;
        resolution.proxy_slot =
            group_prefix[group] + resolution.remote_slot;
    }
}

// Resolve one deduplicated (owner, source-channel, destination) copy from the
// compact plan. This is shared by source shuffle and strict device tests. It
// never consults a per-copy manifest or performs a global atomic.
__forceinline__ __device__ __host__ bool resolve_hybrid_copy(
        const int* owner_channel_prefix,
        const int* keep_count,
        const int* segments,
        const int* num_segments,
        const int* retained,
        const int* moved_channel_prefix,
        const int* group_prefix,
        const int num_rails,
        const int num_channels,
        const int num_destinations,
        const int owner,
        const int source_channel,
        const int destination,
        const int channel_local_ordinal,
        HybridCopyResolution* resolution) {
    if (resolution == nullptr)
        return false;

    *resolution = {-1, -1, -1, -1, -1, -1};
    if (owner_channel_prefix == nullptr or keep_count == nullptr or
        num_segments == nullptr or retained == nullptr or
        moved_channel_prefix == nullptr or group_prefix == nullptr or
        num_rails < 1 or num_rails > 32 or
        num_channels < 1 or num_channels > kNumHybridMaxChannels or
        num_destinations < 1 or
        num_destinations > kNumHybridMaxDestinations or
        owner < 0 or owner >= num_rails or
        source_channel < 0 or source_channel >= num_channels or
        destination < 0 or destination >= num_destinations or
        channel_local_ordinal < 0) {
        return false;
    }

    const auto source_offset = hybrid_plan_detail::gcd_offset(
        owner, source_channel, destination,
        num_channels, num_destinations);
    const auto owner_destination_offset = hybrid_plan_detail::gd_offset(
        owner, destination, num_destinations);
    const int owner_ordinal =
        owner_channel_prefix[source_offset] + channel_local_ordinal;

    if (owner_ordinal < keep_count[owner_destination_offset]) {
        *resolution = {
            0,
            owner,
            source_channel,
            channel_local_ordinal,
            -1,
            -1,
        };
        return true;
    }

    if (segments == nullptr)
        return false;

    const int destination_num_segments = num_segments[destination];
    int egress = -1;
    int incoming_ordinal = -1;
    for (int segment = 0; segment < destination_num_segments; ++segment) {
        const auto base = hybrid_plan_detail::segment_offset(
            destination, segment, 0, num_rails);
        const int segment_owner = segments[base + kHybridSegmentOwner];
        const int owner_begin = segments[base + kHybridSegmentOwnerBegin];
        const int segment_count = segments[base + kHybridSegmentCount];
        if (segment_owner == owner and owner_begin <= owner_ordinal and
            owner_ordinal < owner_begin + segment_count) {
            egress = segments[base + kHybridSegmentEgress];
            incoming_ordinal =
                segments[base + kHybridSegmentEgressBegin] +
                owner_ordinal - owner_begin;
            break;
        }
    }
    if (egress < 0 or egress >= num_rails or incoming_ordinal < 0)
        return false;

    const auto prefix_offset = hybrid_plan_detail::moved_prefix_offset(
        egress, destination, 0, num_channels, num_destinations);
    if (moved_channel_prefix[prefix_offset] != 0 or
        incoming_ordinal >= moved_channel_prefix[
            prefix_offset + num_channels])
        return false;

    // Find the first channel whose exclusive end exceeds the incoming
    // ordinal. The plan producer emits a nondecreasing prefix, including
    // repeated values for empty channels.
    int lower_channel = 0;
    int upper_channel = num_channels - 1;
    while (lower_channel < upper_channel) {
        const int middle_channel =
            lower_channel + (upper_channel - lower_channel) / 2;
        if (moved_channel_prefix[prefix_offset + middle_channel + 1] <=
            incoming_ordinal) {
            lower_channel = middle_channel + 1;
        } else {
            upper_channel = middle_channel;
        }
    }
    const int target_channel = lower_channel;
    const int begin = moved_channel_prefix[prefix_offset + target_channel];
    const int end = moved_channel_prefix[prefix_offset + target_channel + 1];
    if (begin < 0 or begin > incoming_ordinal or incoming_ordinal >= end)
        return false;
    const int group_local_ordinal = incoming_ordinal - begin;

    const auto target_offset = hybrid_plan_detail::gcd_offset(
        egress, target_channel, destination,
        num_channels, num_destinations);
    *resolution = {
        1,
        egress,
        target_channel,
        retained[target_offset] + group_local_ordinal,
        group_prefix[target_offset] + group_local_ordinal,
        incoming_ordinal,
    };
    return true;
}

}  // namespace deep_ep::elastic::rail_balance

namespace deep_ep::elastic {

// Input is stacked contiguous [num_owners, num_tokens, num_topk]. One warp
// owns one (owner, channel) pair and writes compact [G, C, D] counts. Invalid
// routes publish a status and are skipped before destination division or bit
// indexing. Duplicate destinations remain valid and are deduplicated by the
// destination mask; duplicate expert ids are rejected by force-v1.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_count_impl(
        const topk_idx_t* topk_idx,
        int* channel_count,
        int* status,
        const int num_owners,
        const int num_tokens,
        const int num_topk,
        const int num_channels,
        const int num_experts,
        const int num_destinations,
        const int local_destination) {
    if (threadIdx.x >= 32 or
        (num_tokens > 0 and topk_idx == nullptr) or
        channel_count == nullptr or
        num_owners < 1 or num_tokens < 0 or num_topk < 1 or num_topk > 32 or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_experts < 1 or num_destinations < 1 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        num_experts % num_destinations != 0 or
        local_destination < 0 or local_destination >= num_destinations) {
        return;
    }

    const int owner_channel = static_cast<int>(blockIdx.x);
    const int owner = owner_channel / num_channels;
    const int channel = owner_channel % num_channels;
    if (owner >= num_owners)
        return;

    const int lane = ptx::get_lane_idx();
    const int experts_per_destination = num_experts / num_destinations;
    int destination_count = 0;

    for (int token = channel; token < num_tokens; token += num_channels) {
        const auto input_offset =
            (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;
        const bool active = lane < num_topk;
        const topk_idx_t expert = active ?
            __ldg(topk_idx + input_offset + lane) : topk_idx_t{-1};
        const bool in_range = not active or
            (expert >= topk_idx_t{0} and expert < num_experts);
        const unsigned invalid_mask = ptx::gather(not in_range);
        if (invalid_mask != 0) {
            if (lane == 0)
                rail_balance::hybrid_plan_detail::report_error(
                    status, rail_balance::HybridPlanError::ExpertOutOfRange);
            continue;
        }

        bool duplicate = false;
        for (int source_lane = 0; source_lane < num_topk; ++source_lane) {
            const auto other = ptx::exchange(expert, source_lane);
            duplicate |= active and source_lane < lane and expert == other;
        }
        const unsigned duplicate_mask = ptx::gather(duplicate);
        if (duplicate_mask != 0) {
            if (lane == 0)
                rail_balance::hybrid_plan_detail::report_error(
                    status, rail_balance::HybridPlanError::DuplicateExpert);
            continue;
        }

        const int destination = active ?
            static_cast<int>(expert) / experts_per_destination : -1;
        const unsigned destination_bit =
            destination >= 0 and destination != local_destination ?
            (1u << destination) : 0u;
        const unsigned destination_mask = ptx::reduce_or(destination_bit);
        if (lane < num_destinations)
            destination_count += (destination_mask >> lane) & 1u;
    }

    if (lane < num_destinations) {
        const auto output_offset = rail_balance::hybrid_plan_detail::gcd_offset(
            owner, channel, lane, num_channels, num_destinations);
        // B1 writes ordinary CUDA memory, while B2 points this output at an
        // LSA-visible symmetric arena. Publish each compact count with system
        // release semantics so the following local-team barrier can establish
        // visibility before peers gather their snapshots.
        ptx::st_release_sys(channel_count + output_offset, destination_count);
    }
}

// Build the canonical minimum-move quota and destination-bucketed five-int
// segments from a node-consistent compact [G, C, D] count snapshot. Snapshot
// exchange and its LSA barrier intentionally belong to C080-C/B2; this kernel
// contains no guessed NCCL interface.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_plan_v2_impl(
        const int* channel_count,
        int* count,
        int* quota,
        int* keep_count,
        int* segments,
        int* num_segments,
        const int num_rails,
        const int num_channels,
        const int num_destinations,
        const int remainder_seed,
        const int policy,
        const int threshold_percent) {
    const int destination = static_cast<int>(blockIdx.x);
    if (destination >= num_destinations or threadIdx.x != 0 or
        channel_count == nullptr or count == nullptr or quota == nullptr or
        keep_count == nullptr or num_segments == nullptr or
        num_rails < 1 or num_rails > 32 or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_destinations < 1 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        not rail_balance::is_valid_hybrid_policy(policy) or
        threshold_percent < 0 or
        threshold_percent >
            rail_balance::kMaxHybridPolicyThresholdPercent) {
        return;
    }

    int64_t total = 0;
    int max_count = 0;
    uint32_t active_mask = 0;
    for (int owner = 0; owner < num_rails; ++owner) {
        int owner_count = 0;
        for (int channel = 0; channel < num_channels; ++channel) {
            owner_count += channel_count[
                rail_balance::hybrid_plan_detail::gcd_offset(
                    owner, channel, destination,
                    num_channels, num_destinations)];
        }
        count[rail_balance::hybrid_plan_detail::gd_offset(
            owner, destination, num_destinations)] = owner_count;
        total += owner_count;
        max_count = owner_count > max_count ? owner_count : max_count;
        if (owner_count > 0)
            active_mask |= uint32_t{1} << owner;
    }

    int normalized_seed = remainder_seed % num_rails;
    if (normalized_seed < 0)
        normalized_seed += num_rails;
    const int start = (normalized_seed + destination % num_rails) % num_rails;
    const uint32_t all_mask = num_rails == 32 ?
        uint32_t{0xffffffff} : (uint32_t{1} << num_rails) - 1;
    uint32_t selected_mask = policy ==
            static_cast<int>(rail_balance::HybridPolicy::All) ?
        all_mask : active_mask;
    int selected_count = __popc(selected_mask);

    if (policy == static_cast<int>(rail_balance::HybridPolicy::Adaptive) and
        total > 0) {
        while (selected_count < num_rails) {
            const int64_t current_tail =
                (total + selected_count - 1) / selected_count;
            const int64_t next_tail =
                (total + selected_count) / (selected_count + 1);
            if (current_tail * 100 <=
                next_tail * (100 + threshold_percent))
                break;
            for (int offset = 0; offset < num_rails; ++offset) {
                const int rail = (start + offset) % num_rails;
                const uint32_t bit = uint32_t{1} << rail;
                if ((selected_mask & bit) == 0) {
                    selected_mask |= bit;
                    ++selected_count;
                    break;
                }
            }
        }
    }

    const int64_t target_tail = selected_count > 0 ?
        (total + selected_count - 1) / selected_count : 0;
    const bool should_balance = total > 0 and
        (threshold_percent == 0 or
         static_cast<int64_t>(max_count) * 100 >
             target_tail * (100 + threshold_percent));

    if (not should_balance) {
        for (int owner = 0; owner < num_rails; ++owner) {
            const auto matrix_offset =
                rail_balance::hybrid_plan_detail::gd_offset(
                    owner, destination, num_destinations);
            quota[matrix_offset] = count[matrix_offset];
        }
    } else {
        const int base = static_cast<int>(total / selected_count);
        const int remainder = static_cast<int>(total % selected_count);

        for (int owner = 0; owner < num_rails; ++owner) {
            const bool selected =
                (selected_mask & (uint32_t{1} << owner)) != 0;
            quota[rail_balance::hybrid_plan_detail::gd_offset(
                owner, destination, num_destinations)] =
                    selected ? base : 0;
        }

        int assigned = 0;
        for (int offset = 0;
             offset < num_rails and assigned < remainder; ++offset) {
            const int owner = (start + offset) % num_rails;
            const uint32_t bit = uint32_t{1} << owner;
            const auto matrix_offset =
                rail_balance::hybrid_plan_detail::gd_offset(
                    owner, destination, num_destinations);
            if ((selected_mask & bit) != 0 and
                count[matrix_offset] > base) {
                quota[matrix_offset] = base + 1;
                ++assigned;
            }
        }
        for (int offset = 0;
             offset < num_rails and assigned < remainder; ++offset) {
            const int owner = (start + offset) % num_rails;
            const uint32_t bit = uint32_t{1} << owner;
            const auto matrix_offset =
                rail_balance::hybrid_plan_detail::gd_offset(
                    owner, destination, num_destinations);
            if ((selected_mask & bit) != 0 and
                count[matrix_offset] <= base) {
                quota[matrix_offset] = base + 1;
                ++assigned;
            }
        }
    }

    for (int owner = 0; owner < num_rails; ++owner) {
        const auto matrix_offset = rail_balance::hybrid_plan_detail::gd_offset(
            owner, destination, num_destinations);
        const int value = count[matrix_offset];
        const int target = quota[matrix_offset];
        keep_count[matrix_offset] = value < target ? value : target;
    }

    if (num_rails > 1 and segments != nullptr) {
        for (int segment = 0; segment < num_rails - 1; ++segment) {
            const auto base_offset = rail_balance::hybrid_plan_detail::segment_offset(
                destination, segment, 0, num_rails);
            for (int field = 0; field < rail_balance::kNumHybridSegmentFields; ++field)
                segments[base_offset + field] = -1;
        }
    }

    int owner = 0;
    int egress = 0;
    int owner_consumed = 0;
    int egress_consumed = 0;
    int segment_idx = 0;
    while (owner < num_rails and egress < num_rails) {
        while (owner < num_rails) {
            const auto owner_offset = rail_balance::hybrid_plan_detail::gd_offset(
                owner, destination, num_destinations);
            if (count[owner_offset] - keep_count[owner_offset] -
                    owner_consumed > 0)
                break;
            ++owner;
            owner_consumed = 0;
        }
        while (egress < num_rails) {
            const auto egress_offset = rail_balance::hybrid_plan_detail::gd_offset(
                egress, destination, num_destinations);
            if (quota[egress_offset] - keep_count[egress_offset] -
                    egress_consumed > 0)
                break;
            ++egress;
            egress_consumed = 0;
        }
        if (owner == num_rails or egress == num_rails)
            break;

        const auto owner_offset = rail_balance::hybrid_plan_detail::gd_offset(
            owner, destination, num_destinations);
        const auto egress_offset = rail_balance::hybrid_plan_detail::gd_offset(
            egress, destination, num_destinations);
        const int owner_remaining =
            count[owner_offset] - keep_count[owner_offset] - owner_consumed;
        const int egress_remaining =
            quota[egress_offset] - keep_count[egress_offset] - egress_consumed;
        const int amount = owner_remaining < egress_remaining ?
            owner_remaining : egress_remaining;
        EP_DEVICE_ASSERT(amount > 0);

        if (segments != nullptr and segment_idx < num_rails - 1) {
            const auto base_offset = rail_balance::hybrid_plan_detail::segment_offset(
                destination, segment_idx, 0, num_rails);
            segments[base_offset + rail_balance::kHybridSegmentOwner] = owner;
            segments[base_offset + rail_balance::kHybridSegmentEgress] = egress;
            segments[base_offset + rail_balance::kHybridSegmentOwnerBegin] =
                keep_count[owner_offset] + owner_consumed;
            segments[base_offset + rail_balance::kHybridSegmentCount] = amount;
            segments[base_offset + rail_balance::kHybridSegmentEgressBegin] =
                egress_consumed;
        }
        owner_consumed += amount;
        egress_consumed += amount;
        ++segment_idx;
    }
    num_segments[destination] = segment_idx;
}

// Materialize all inverse schedule prefixes. One warp owns one egress; lanes
// own destinations. group_prefix is an exclusive scan in exactly
// channel-major/destination-minor order. Only one plan-stage atomic per egress
// contributes to moved_copies; the dispatch/resolve hot path has no atomic.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hybrid_prefix_impl(
        const int* channel_count,
        const int* quota,
        const int* keep_count,
        int* owner_channel_prefix,
        int* retained,
        int* moved,
        int* moved_channel_prefix,
        int* group_prefix,
        int* proxy_required,
        int* moved_copies,
        int* status,
        const int num_rails,
        const int num_channels,
        const int num_destinations,
        const int num_max_tokens_per_rank,
        const int proxy_capacity_per_egress) {
    const int egress = static_cast<int>(blockIdx.x);
    if (egress >= num_rails or threadIdx.x >= 32 or
        channel_count == nullptr or quota == nullptr or keep_count == nullptr or
        owner_channel_prefix == nullptr or retained == nullptr or
        moved == nullptr or moved_channel_prefix == nullptr or
        group_prefix == nullptr or proxy_required == nullptr or
        moved_copies == nullptr or
        num_rails < 1 or num_rails > 32 or
        num_channels < 1 or
        num_channels > rail_balance::kNumHybridMaxChannels or
        num_destinations < 1 or
        num_destinations > rail_balance::kNumHybridMaxDestinations or
        num_max_tokens_per_rank < 1) {
        return;
    }

    const int lane = ptx::get_lane_idx();
    const bool active = lane < num_destinations;
    const int channel_capacity =
        num_max_tokens_per_rank / num_channels +
        (num_max_tokens_per_rank % num_channels != 0);

    int keep = 0;
    int incoming_remaining = 0;
    if (active) {
        const auto matrix_offset = rail_balance::hybrid_plan_detail::gd_offset(
            egress, lane, num_destinations);
        keep = keep_count[matrix_offset];
        incoming_remaining = quota[matrix_offset] - keep;
        moved_channel_prefix[
            rail_balance::hybrid_plan_detail::moved_prefix_offset(
                egress, lane, 0, num_channels, num_destinations)] = 0;
    }

    // Rotate only pure-deficit groups. Their retained counts are all zero, so
    // full/tail is an exact closed-form window. A partial-deficit group keeps
    // the legacy channel-zero fill and contributes no cursor increment.
    const bool rotate_window = active and incoming_remaining > 0 and keep == 0;
    const int full_channels = rotate_window ?
        incoming_remaining / channel_capacity : 0;
    const int tail = rotate_window ?
        incoming_remaining % channel_capacity : 0;
    const int window_span = full_channels + (tail != 0);
    const int exclusive_span = ptx::warp_exclusive_sum(window_span, lane);
    const int window_start = rotate_window ?
        exclusive_span % num_channels : 0;

    int owner_prefix = 0;
    int moved_prefix = 0;
    for (int channel = 0; channel < num_channels; ++channel) {
        if (active) {
            const auto tensor_offset = rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel, lane, num_channels, num_destinations);
            const int available = channel_count[tensor_offset];
            owner_channel_prefix[tensor_offset] = owner_prefix;

            const int remaining_keep = keep - owner_prefix;
            const int retained_count = remaining_keep <= 0 ? 0 :
                (available < remaining_keep ? available : remaining_keep);
            int moved_count;
            if (rotate_window) {
                int relative_channel = channel - window_start;
                if (relative_channel < 0)
                    relative_channel += num_channels;
                moved_count = relative_channel < full_channels ?
                    channel_capacity :
                    (relative_channel == full_channels ? tail : 0);
            } else {
                const int spare = channel_capacity - retained_count;
                moved_count = spare < incoming_remaining ?
                    spare : incoming_remaining;
            }

            retained[tensor_offset] = retained_count;
            moved[tensor_offset] = moved_count;
            owner_prefix += available;
            incoming_remaining -= moved_count;
            moved_prefix += moved_count;
            moved_channel_prefix[
                rail_balance::hybrid_plan_detail::moved_prefix_offset(
                    egress, lane, channel + 1,
                    num_channels, num_destinations)] = moved_prefix;
        }
    }
    __syncwarp();

    int group_cursor = 0;
    for (int channel = 0; channel < num_channels; ++channel) {
        const auto tensor_offset = active ?
            rail_balance::hybrid_plan_detail::gcd_offset(
                egress, channel, lane, num_channels, num_destinations) : 0;
        const int value = active ? moved[tensor_offset] : 0;
        const int inclusive = ptx::warp_inclusive_sum(value, lane);
        const int exclusive = inclusive - value;
        if (active)
            group_prefix[tensor_offset] = group_cursor + exclusive;
        group_cursor += ptx::exchange(inclusive, 31);
    }

    if (lane == 0) {
        proxy_required[egress] = group_cursor;
        atomicAdd(moved_copies, group_cursor);
        if (group_cursor > proxy_capacity_per_egress) {
            rail_balance::hybrid_plan_detail::report_error(
                status, rail_balance::HybridPlanError::CapacityExceeded);
        }
    }
}

}  // namespace deep_ep::elastic
