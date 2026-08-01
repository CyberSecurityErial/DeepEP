#pragma once

#include <climits>
#include <cstdint>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>

namespace deep_ep::elastic::rail_balance {

static constexpr int kNumHybridSegmentFields = 5;
static constexpr int kNumMaxEndpointChunks = 32;
static constexpr int kNumMaxAdaptiveRoundsPerRail = 8;

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

__forceinline__ __device__ __host__ int64_t odt_offset(
        const int owner,
        const int destination,
        const int target,
        const int num_destinations,
        const int num_rails) {
    return (static_cast<int64_t>(owner) * num_destinations + destination) *
        num_rails + target;
}

__forceinline__ __device__ __host__ int64_t odte_offset(
        const int owner,
        const int destination,
        const int target,
        const int egress,
        const int num_destinations,
        const int num_rails) {
    return ((static_cast<int64_t>(owner) * num_destinations + destination) *
        num_rails + target) * num_rails + egress;
}

__forceinline__ __device__ __host__ int64_t odme_offset(
        const int owner,
        const int destination,
        const int target_mask,
        const int egress,
        const int num_destinations,
        const int num_target_masks,
        const int num_rails) {
    return ((static_cast<int64_t>(owner) * num_destinations + destination) *
        num_target_masks + target_mask) * num_rails + egress;
}

__forceinline__ __device__ __host__ int64_t odm_offset(
        const int owner,
        const int destination,
        const int target_mask,
        const int num_destinations,
        const int num_target_masks) {
    return (static_cast<int64_t>(owner) * num_destinations + destination) *
        num_target_masks + target_mask;
}

__forceinline__ __device__ __host__ int64_t lgcd_offset(
        const int worker_lane,
        const int egress,
        const int channel,
        const int destination,
        const int num_rails,
        const int num_channels,
        const int num_destinations) {
    return ((static_cast<int64_t>(worker_lane) * num_rails + egress) *
        num_channels + channel) * num_destinations + destination;
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

struct LoadPeaks {
    int first;
    int second;
    int first_count;
};

struct AdaptiveCandidate {
    int64_t index;
    int64_t net_gain;
    int egress;
    int batch;
    int meets_batch_floor;
    int pair_after;
    int source_after;
    int added_hops;
    int tie;
};

__forceinline__ __device__ bool better_candidate(
        const AdaptiveCandidate& candidate,
        const AdaptiveCandidate& current) {
    if (candidate.index < 0)
        return false;
    if (current.index >= 0 and
        candidate.meets_batch_floor != current.meets_batch_floor)
        return candidate.meets_batch_floor > current.meets_batch_floor;
    if (current.index >= 0 and not candidate.meets_batch_floor) {
        const int64_t candidate_gain = candidate.net_gain * candidate.batch;
        const int64_t current_gain = current.net_gain * current.batch;
        if (candidate_gain != current_gain)
            return candidate_gain > current_gain;
    }
    if (current.index < 0 or candidate.net_gain != current.net_gain)
        return current.index < 0 or candidate.net_gain > current.net_gain;
    if (candidate.pair_after != current.pair_after)
        return candidate.pair_after < current.pair_after;
    if (candidate.source_after != current.source_after)
        return candidate.source_after < current.source_after;
    if (candidate.added_hops != current.added_hops)
        return candidate.added_hops < current.added_hops;
    if (candidate.tie != current.tie)
        return candidate.tie < current.tie;
    if (candidate.egress != current.egress)
        return candidate.egress < current.egress;
    return candidate.index < current.index;
}

__forceinline__ __device__ AdaptiveCandidate warp_best_candidate(
        AdaptiveCandidate best) {
    const int lane = ptx::get_lane_idx();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const auto candidate = ptx::exchange(best, lane + offset);
        if (lane < offset and better_candidate(candidate, best))
            best = candidate;
    }
    return ptx::exchange(best, 0);
}

__forceinline__ __device__ LoadPeaks find_load_peaks(
        const int* loads, const int count) {
    LoadPeaks peaks = {-1, 0, 0};
    for (int index = 0; index < count; ++index) {
        const int value = loads[index];
        if (value > peaks.first) {
            peaks.second = max(peaks.first, 0);
            peaks.first = value;
            peaks.first_count = 1;
        } else if (value == peaks.first) {
            ++peaks.first_count;
        } else {
            peaks.second = max(peaks.second, value);
        }
    }
    return peaks;
}

__forceinline__ __device__ int peak_after_move(
        const LoadPeaks& peaks,
        const int old_load,
        const int new_load) {
    const int removed_first =
        (old_load == peaks.first) + (new_load == peaks.first);
    const int unchanged_peak = peaks.first_count > removed_first ?
        peaks.first : peaks.second;
    return max(unchanged_peak, max(old_load - 1, new_load + 1));
}

__forceinline__ __device__ int adaptive_batch_limit(
        const int quota,
        const int old_pair,
        const int new_pair,
        const int old_source,
        const int new_source,
        const LoadPeaks& pair_peaks,
        const LoadPeaks& source_peaks,
        const int remaining_cap,
        const int pair_capacity,
        const int proxy_capacity) {
    int batch = min(quota, remaining_cap);
    batch = min(batch, pair_capacity);
    batch = min(batch, pair_peaks.first - new_pair);
    batch = min(batch, source_peaks.first - new_source);
    batch = min(batch, proxy_capacity);

    const int pair_gap = old_pair - new_pair;
    const int source_gap = old_source - new_source;
    if (pair_gap <= 0 or source_gap <= 0) {
        batch = min(batch, 1);
    } else {
        batch = min(batch, pair_gap / 2 + pair_gap % 2);
        batch = min(batch, source_gap / 2 + source_gap % 2);
    }

    const int pair_other =
        (pair_peaks.first_count >
         static_cast<int>(old_pair == pair_peaks.first)) ?
            pair_peaks.first : pair_peaks.second;
    const int source_other =
        (source_peaks.first_count >
         static_cast<int>(old_source == source_peaks.first)) ?
            source_peaks.first : source_peaks.second;
    if (old_pair == pair_peaks.first and pair_peaks.first_count > 1)
        batch = min(
            batch,
            remaining_cap / pair_peaks.first_count +
                static_cast<int>(
                    remaining_cap % pair_peaks.first_count != 0));
    if (old_source == source_peaks.first and source_peaks.first_count > 1)
        batch = min(
            batch,
            remaining_cap / source_peaks.first_count +
                static_cast<int>(
                    remaining_cap % source_peaks.first_count != 0));
    int critical_batch = 0;
    if (old_pair > pair_other) {
        const int difference = old_pair - pair_other;
        critical_batch = difference >= batch ? batch : difference + 1;
    }
    if (old_source > source_other) {
        const int difference = old_source - source_other;
        critical_batch = max(
            critical_batch, difference >= batch ? batch : difference + 1);
    }
    if (critical_batch > 0)
        batch = min(batch, critical_batch);
    return max(batch, 0);
}

__forceinline__ __device__ int choose_endpoint(
        const uint32_t candidates,
        const uint32_t target_mask,
        const int owner,
        const int destination,
        const int amount,
        const int64_t tie_index,
        const int* pair_load,
        const int* source_load,
        const int* owner_remaining,
        const int num_rails,
        const int num_destinations,
        const int num_max_tokens_per_rank,
        const int planner_seed) {
    int best_egress = -1;
    int best_pair_peak = INT_MAX;
    int best_source_peak = INT_MAX;
    int best_local_forwards = INT_MAX;
    int best_tie = INT_MAX;
    if ((candidates & (candidates - 1)) == 0) {
        best_egress = __ffs(static_cast<int>(candidates)) - 1;
        const int pair_offset = destination * num_rails + best_egress;
        return pair_load[pair_offset] + amount + owner_remaining[
            best_egress * num_destinations + destination] <=
                num_max_tokens_per_rank ? best_egress : -1;
    }

    const auto pair_peaks = find_load_peaks(
        pair_load + destination * num_rails, num_rails);
    const auto source_peaks = find_load_peaks(source_load, num_rails);
    for (int egress = 0; egress < num_rails; ++egress) {
        if ((candidates & (uint32_t{1} << egress)) == 0)
            continue;
        const int pair_offset = destination * num_rails + egress;
        if (pair_load[pair_offset] + amount + owner_remaining[
                egress * num_destinations + destination] >
                num_max_tokens_per_rank)
            continue;

        const int pair_peak = max(
            pair_peaks.first, pair_load[pair_offset] + amount);
        const int source_peak = max(
            source_peaks.first, source_load[egress] + amount);
        const int local_forwards = (egress != owner) +
            __popc(target_mask & ~(uint32_t{1} << egress));
        const int origin = static_cast<int>(
            (static_cast<int64_t>(planner_seed) + destination + tie_index) %
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
    return best_egress;
}

__forceinline__ __device__ bool assign_endpoint_group(
        int* egress_quota,
        int* pair_load,
        int* source_load,
        int* owner_remaining,
        int* proxy_required,
        const int owner,
        const int destination,
        const uint32_t target_mask,
        const int count,
        const int64_t tie_group,
        const int num_rails,
        const int num_destinations,
        const int num_tokens,
        const int num_max_tokens_per_rank,
        const int planner_seed,
        const int planner_chunk_size) {
    owner_remaining[owner * num_destinations + destination] -= count;
    const int chunk_size = max(
        planner_chunk_size,
        count / kNumMaxEndpointChunks +
            static_cast<int>(count % kNumMaxEndpointChunks != 0));
    for (int begin = 0; begin < count;) {
        const int amount = min(chunk_size, count - begin);
        const int egress = choose_endpoint(
            target_mask | (uint32_t{1} << owner), target_mask,
            owner, destination, amount,
            tie_group * static_cast<int64_t>(num_tokens) + begin,
            pair_load, source_load, owner_remaining,
            num_rails, num_destinations, num_max_tokens_per_rank,
            planner_seed);
        if (egress < 0)
            return false;
        pair_load[destination * num_rails + egress] += amount;
        source_load[egress] += amount;
        egress_quota[egress] += amount;
        if (egress != owner)
            proxy_required[egress] += amount;
        begin += amount;
    }
    return true;
}

// Move group quota, not individual records. For G <= 8, active_groups
// compacts singleton [o,d,t] and dense multi-target [o,d,mask] rows; later
// G*C and G blocks materialize their egress quotas.
__forceinline__ __device__ bool rebalance_endpoint_quotas(
        int* endpoint_egress_quota,
        int* multi_target_egress_quota,
        const int* active_groups,
        const int num_active_groups,
        int* pair_load,
        int* source_load,
        int* peak_scratch,
        int* proxy_required,
        const int total_units,
        const int num_rails,
        const int num_destinations,
        const int num_target_masks,
        const int num_max_tokens_per_rank,
        const int proxy_capacity_per_egress,
        const int planner_seed,
        const int two_hop_threshold_percent,
        const int max_two_hop_percent,
        const int hop_penalty_percent) {
    const int lane = ptx::get_lane_idx();
    const int two_hop_cap = num_rails < 2 ? 0 : static_cast<int>(
        static_cast<int64_t>(total_units) * max_two_hop_percent / 100);
    const int64_t num_endpoint_quotas = static_cast<int64_t>(num_rails) *
        num_destinations * num_rails * num_rails;
    const int num_endpoint_groups =
        num_rails * num_destinations * num_rails;
    const int64_t num_multi_target_quotas =
        num_target_masks > 1 ? static_cast<int64_t>(num_rails) *
            num_destinations * num_target_masks * num_rails : 0;
    const int64_t num_quotas =
        num_endpoint_quotas + num_multi_target_quotas;
    const int64_t num_scan_quotas = active_groups == nullptr ?
        num_quotas :
        static_cast<int64_t>(num_active_groups) * num_rails;
    const int max_rounds = num_rails * kNumMaxAdaptiveRoundsPerRail;
    int selected = 0;
    int round = 0;
    int tiny_tail_rounds = 0;

    while (selected < two_hop_cap and round < max_rounds) {
        const int remaining_cap = two_hop_cap - selected;
        const int remaining_rounds = max_rounds - round;
        const int required_batch = remaining_cap / remaining_rounds +
            static_cast<int>(remaining_cap % remaining_rounds != 0);
        if (lane == 0)
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const auto peaks = find_load_peaks(
                    pair_load + destination * num_rails, num_rails);
                peak_scratch[destination] = peaks.first;
                peak_scratch[num_destinations + destination] = peaks.second;
                peak_scratch[2 * num_destinations + destination] =
                    peaks.first_count;
            }
        __syncwarp();
        const auto source_peaks = find_load_peaks(source_load, num_rails);

        AdaptiveCandidate best = {
            -1, INT64_MIN, -1, 0, 0,
            INT_MAX, INT_MAX, INT_MAX, INT_MAX};
        for (int64_t scan_index = lane;
             scan_index < num_scan_quotas; scan_index += 32) {
            int64_t index = scan_index;
            if (active_groups != nullptr) {
                const int group =
                    active_groups[scan_index / num_rails];
                const int old_egress =
                    static_cast<int>(scan_index % num_rails);
                index = group < num_endpoint_groups ?
                    static_cast<int64_t>(group) * num_rails + old_egress :
                    num_endpoint_quotas +
                        static_cast<int64_t>(
                            group - num_endpoint_groups) * num_rails +
                        old_egress;
            }
            const bool multi_target = index >= num_endpoint_quotas;
            int64_t local_index = multi_target ?
                index - num_endpoint_quotas : index;
            const int quota = multi_target ?
                multi_target_egress_quota[local_index] :
                endpoint_egress_quota[local_index];
            if (quota <= 0)
                continue;
            int64_t decoded = local_index;
            const int old_egress = static_cast<int>(decoded % num_rails);
            decoded /= num_rails;
            const uint32_t target_mask = multi_target ?
                static_cast<uint32_t>(decoded % num_target_masks) :
                uint32_t{1} << static_cast<int>(decoded % num_rails);
            decoded /= multi_target ? num_target_masks : num_rails;
            const int destination =
                static_cast<int>(decoded % num_destinations);
            const int owner =
                static_cast<int>(decoded / num_destinations);
            if (old_egress != owner and
                (target_mask & (uint32_t{1} << old_egress)) == 0)
                continue;

            const LoadPeaks pair_peaks = {
                peak_scratch[destination],
                peak_scratch[num_destinations + destination],
                peak_scratch[2 * num_destinations + destination],
            };
            const int pair_before = pair_peaks.first;
            const int source_before = source_peaks.first;
            const int old_pair =
                pair_load[destination * num_rails + old_egress];
            const int old_source = source_load[old_egress];
            if (old_pair < pair_before and old_source < source_before)
                continue;
            const int old_hops = (old_egress != owner) +
                __popc(target_mask & ~(uint32_t{1} << old_egress));
            const int added_hops = 1 + __popc(target_mask) - old_hops;

            for (int egress = 0; egress < num_rails; ++egress) {
                if (egress == owner or
                    (target_mask & (uint32_t{1} << egress)) != 0 or
                    pair_load[destination * num_rails + egress] >=
                        num_max_tokens_per_rank or
                    proxy_required[egress] >= proxy_capacity_per_egress)
                    continue;
                const int new_pair =
                    pair_load[destination * num_rails + egress];
                const int new_source = source_load[egress];
                const int pair_after =
                    peak_after_move(pair_peaks, old_pair, new_pair);
                const int source_after =
                    peak_after_move(source_peaks, old_source, new_source);
                if (pair_after > pair_before or
                    source_after > source_before)
                    continue;
                const int pair_relief =
                    min(1, max(0, old_pair - new_pair));
                const int source_relief =
                    min(1, max(0, old_source - new_source));
                const int64_t net_gain = static_cast<int64_t>(
                    pair_relief + source_relief) * 100 -
                    static_cast<int64_t>(hop_penalty_percent) * added_hops;
                const int64_t threshold =
                    static_cast<int64_t>(two_hop_threshold_percent) *
                    (static_cast<int64_t>(pair_before) + source_before);
                if (net_gain <= 0 or net_gain <= threshold)
                    continue;
                const int batch = adaptive_batch_limit(
                    quota, old_pair, new_pair, old_source, new_source,
                    pair_peaks, source_peaks, remaining_cap,
                    num_max_tokens_per_rank - new_pair,
                    proxy_capacity_per_egress - proxy_required[egress]);
                if (batch <= 0)
                    continue;
                const int origin = static_cast<int>(
                    (static_cast<int64_t>(planner_seed) + destination +
                     index) % num_rails);
                const AdaptiveCandidate candidate = {
                    index,
                    net_gain,
                    egress,
                    batch,
                    static_cast<int>(batch >= required_batch),
                    pair_after,
                    source_after,
                    added_hops,
                    (egress - origin + num_rails) % num_rails,
                };
                if (better_candidate(candidate, best))
                    best = candidate;
            }
        }
        best = warp_best_candidate(best);
        if (best.index < 0 or
            (not best.meets_batch_floor and
             tiny_tail_rounds >= num_rails))
            break;

        const bool multi_target = best.index >= num_endpoint_quotas;
        const int64_t local_index = multi_target ?
            best.index - num_endpoint_quotas : best.index;
        int64_t decoded = local_index;
        const int old_egress = static_cast<int>(decoded % num_rails);
        decoded /= num_rails;
        const uint32_t target_mask = multi_target ?
            static_cast<uint32_t>(decoded % num_target_masks) :
            uint32_t{1} << static_cast<int>(decoded % num_rails);
        decoded /= multi_target ? num_target_masks : num_rails;
        const int destination =
            static_cast<int>(decoded % num_destinations);
        const int owner = static_cast<int>(decoded / num_destinations);
        const int new_egress = best.egress;
        const int old_pair =
            pair_load[destination * num_rails + old_egress];
        const int new_pair =
            pair_load[destination * num_rails + new_egress];
        const int old_source = source_load[old_egress];
        const int new_source = source_load[new_egress];
        const LoadPeaks pair_peaks = {
            peak_scratch[destination],
            peak_scratch[num_destinations + destination],
            peak_scratch[2 * num_destinations + destination],
        };
        const int batch = adaptive_batch_limit(
            multi_target ?
                multi_target_egress_quota[local_index] :
                endpoint_egress_quota[local_index],
            old_pair, new_pair, old_source, new_source,
            pair_peaks, source_peaks, remaining_cap,
            num_max_tokens_per_rank - new_pair,
            proxy_capacity_per_egress - proxy_required[new_egress]);
        if (batch <= 0 or
            (best.meets_batch_floor and batch < required_batch))
            return false;

        if (lane == 0) {
            int* quotas = multi_target ?
                multi_target_egress_quota : endpoint_egress_quota;
            const auto new_quota = multi_target ?
                odme_offset(
                    owner, destination, target_mask, new_egress,
                    num_destinations, num_target_masks, num_rails) :
                odte_offset(
                    owner, destination,
                    __ffs(static_cast<int>(target_mask)) - 1, new_egress,
                    num_destinations, num_rails);
            quotas[local_index] -= batch;
            quotas[new_quota] += batch;
            pair_load[destination * num_rails + old_egress] -= batch;
            pair_load[destination * num_rails + new_egress] += batch;
            source_load[old_egress] -= batch;
            source_load[new_egress] += batch;
            if (old_egress != owner)
                proxy_required[old_egress] -= batch;
            proxy_required[new_egress] += batch;
        }
        selected += batch;
        ++round;
        tiny_tail_rounds += not best.meets_batch_floor;
        __syncwarp();
    }
    return true;
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
    for (int64_t token = channel;
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

// Count and validate independent (owner, channel) token stripes before the
// ordered endpoint decision. One warp owns one stripe; lanes own top-k slots.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hop_precount_impl(
        const HopCopyRecord* records,
        HopCopyResolution* resolutions,
        int* owner_remaining,
        int* endpoint_count,
        int* endpoint_egress_quota,
        int* multi_target_egress_quota,
        int* owner_group_cursor,
        int* total_units,
        int* status,
        const int num_rails,
        const int num_tokens,
        const int num_topk,
        const int num_channels,
        const int num_destinations) {
    const int lane = ptx::get_lane_idx();
    if (records == nullptr or resolutions == nullptr or
        owner_remaining == nullptr or endpoint_count == nullptr or
        endpoint_egress_quota == nullptr or
        multi_target_egress_quota == nullptr or
        owner_group_cursor == nullptr or
        total_units == nullptr or status == nullptr or
        num_rails < 1 or num_rails > 32 or
        num_tokens < 0 or num_topk < 1 or num_topk > 32 or
        num_channels < 1 or num_channels > kNumHybridMaxChannels or
        num_destinations < 2 or num_destinations > 32) {
        if (lane == 0)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::InvalidSchedule);
        return;
    }

    const int owner_channel = static_cast<int>(blockIdx.x);
    const int owner = owner_channel / num_channels;
    const int channel = owner_channel % num_channels;
    if (owner >= num_rails)
        return;

    for (int64_t token = channel; token < num_tokens; token += num_channels) {
        const int64_t begin =
            (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;
        const bool in_table = lane < num_topk;
        const auto record = in_table ? records[begin + lane] :
            HopCopyRecord{0u, -1};
        if (in_table)
            resolutions[begin + lane] = {-1, -1, -1, -1};

        const bool active = in_table and record.target_mask != 0;
        const uint32_t active_mask = ptx::gather(active);
        const int active_count = __popc(active_mask);
        const uint32_t packed_mask = active_count == 32 ? UINT32_MAX :
            (uint32_t{1} << active_count) - 1;
        const bool valid_padding = not in_table or active or
            record.destination == -1;
        const bool valid_active = not active or
            (record.destination >= 0 and
             record.destination < num_destinations and
             (num_rails == 32 or (record.target_mask >> num_rails) == 0));
        const int previous_destination = __shfl_up_sync(
            0xffffffff, record.destination, 1);
        const bool valid_order = not active or lane == 0 or
            record.destination > previous_destination;
        if (active_mask != packed_mask or
            ptx::gather(
                not valid_padding or not valid_active or not valid_order) != 0) {
            if (lane == 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
            continue;
        }
        if (not active)
            continue;

        atomicAdd(owner_remaining + owner * num_destinations +
                  record.destination, 1);
        if (__popc(record.target_mask) == 1) {
            const int target =
                __ffs(static_cast<int>(record.target_mask)) - 1;
            atomicAdd(endpoint_count + hybrid_plan_detail::odt_offset(
                owner, record.destination, target,
                num_destinations, num_rails), 1);
            atomicAdd(owner_group_cursor + hybrid_plan_detail::lgcd_offset(
                owner, target, channel, record.destination,
                num_rails, num_channels, num_destinations), 1);
        } else {
            atomicExch(endpoint_egress_quota, 1);
            const int num_target_masks = get_num_dense_target_masks(num_rails);
            if (num_target_masks > 1)
                atomicAdd(
                    multi_target_egress_quota +
                        hybrid_plan_detail::odme_offset(
                            owner, record.destination,
                            static_cast<int>(record.target_mask), 0,
                            num_destinations, num_target_masks, num_rails),
                    1);
        }
        if (lane == 0)
            atomicAdd(total_units, active_count);
    }
}

// Multi-block one-hop materializer. Launch boundaries are the only grid-wide
// synchronization; every mutable cursor has exactly one owning block.
template <int kStage, int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_hop_materialize_impl(
        const HopCopyRecord* records,
        HopCopyResolution* resolutions,
        const int* endpoint_egress_quota,
        const int* multi_target_egress_quota,
        int* multi_target_cursor,
        int* owner_group_cursor,
        int* retained,
        int* moved,
        int* retained_prefix,
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
        const int proxy_capacity_per_egress) {
    const int lane = ptx::get_lane_idx();
    if (records == nullptr or resolutions == nullptr or
        endpoint_egress_quota == nullptr or
        multi_target_egress_quota == nullptr or
        multi_target_cursor == nullptr or owner_group_cursor == nullptr or
        retained == nullptr or moved == nullptr or
        retained_prefix == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or path_units == nullptr or
        moved_copies == nullptr or status == nullptr or
        num_rails < 1 or num_rails > 32 or num_tokens < 0 or
        num_topk < 1 or num_topk > 32 or num_channels < 1 or
        num_channels > kNumHybridMaxChannels or num_destinations < 2 or
        num_destinations > 32 or proxy_capacity_per_egress < 0) {
        if (lane == 0)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::InvalidSchedule);
        return;
    }
    if (*status != static_cast<int>(HybridPlanError::Success))
        return;

    if constexpr (kStage == kHopEndpointPrefix) {
        const int flat = static_cast<int>(blockIdx.x);
        const int target = flat % num_rails;
        const int destination = (flat / num_rails) % num_destinations;
        const int owner = flat / (num_rails * num_destinations);
        if (owner >= num_rails)
            return;

        int base = 0;
        for (int tile = 0; tile < num_channels; tile += 32) {
            const int channel = tile + lane;
            const auto offset = hybrid_plan_detail::lgcd_offset(
                owner, target, channel, destination,
                num_rails, num_channels, num_destinations);
            const int count = channel < num_channels ?
                owner_group_cursor[offset] : 0;
            int inclusive = count;
            for (int delta = 1; delta < 32; delta *= 2) {
                const int upper = __shfl_up_sync(
                    0xffffffff, inclusive, delta);
                if (lane >= delta)
                    inclusive += upper;
            }
            if (channel < num_channels)
                owner_group_cursor[offset] = base + inclusive - count;
            base += __shfl_sync(0xffffffff, inclusive, 31);
        }
        return;
    }

    if constexpr (kStage == kHopEndpointAssign) {
        const int owner_channel = static_cast<int>(blockIdx.x);
        const int owner = owner_channel / num_channels;
        const int channel = owner_channel % num_channels;
        if (owner >= num_rails)
            return;
        for (int64_t token = channel;
             token < num_tokens; token += num_channels) {
            const int64_t begin =
                (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;
            if (lane >= num_topk)
                continue;
            const auto record = records[begin + lane];
            if (record.target_mask == 0)
                continue;
            auto& resolution = resolutions[begin + lane];
            if (__popc(record.target_mask) != 1) {
                if (resolution.egress < 0)
                    hybrid_plan_detail::report_error(
                        status, HybridPlanError::InvalidSchedule);
                continue;
            }
            const int target =
                __ffs(static_cast<int>(record.target_mask)) - 1;
            const auto cursor = hybrid_plan_detail::lgcd_offset(
                owner, target, channel, record.destination,
                num_rails, num_channels, num_destinations);
            const int ordinal = owner_group_cursor[cursor]++;
            int cumulative = 0;
            int egress = -1;
            for (int candidate = 0;
                 candidate < num_rails; ++candidate) {
                cumulative += endpoint_egress_quota[
                    hybrid_plan_detail::odte_offset(
                        owner, record.destination, target, candidate,
                        num_destinations, num_rails)];
                if (ordinal < cumulative) {
                    egress = candidate;
                    break;
                }
            }
            if (egress < 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
            else
                resolution.egress = egress;
        }
        return;
    }

    if constexpr (kStage == kHopMultiTargetAssign) {
        const int owner = static_cast<int>(blockIdx.x);
        const int num_target_masks = get_num_dense_target_masks(num_rails);
        if (owner >= num_rails or num_target_masks == 1)
            return;
        const int64_t owner_begin =
            static_cast<int64_t>(owner) * num_tokens * num_topk;
        const int64_t owner_records =
            static_cast<int64_t>(num_tokens) * num_topk;
        for (int64_t base = 0; base < owner_records; base += 32) {
            const int64_t record_offset = base + lane;
            const bool in_range = record_offset < owner_records;
            const auto record = in_range ?
                records[owner_begin + record_offset] :
                HopCopyRecord{0u, -1};
            const bool active = in_range and
                __popc(record.target_mask) > 1;
            const unsigned peers = ptx::match(active ?
                record.destination * num_target_masks +
                    static_cast<int>(record.target_mask) : -1) &
                ptx::gather(active);
            if (active) {
                const auto cursor = hybrid_plan_detail::odm_offset(
                    owner, record.destination,
                    static_cast<int>(record.target_mask),
                    num_destinations, num_target_masks);
                const int group_base = multi_target_cursor[cursor];
                const int ordinal = group_base +
                    __popc(peers & ((uint32_t{1} << lane) - 1));
                int cumulative = 0;
                int egress = -1;
                for (int candidate = 0;
                     candidate < num_rails; ++candidate) {
                    cumulative += multi_target_egress_quota[
                        hybrid_plan_detail::odme_offset(
                            owner, record.destination,
                            static_cast<int>(record.target_mask), candidate,
                            num_destinations, num_target_masks, num_rails)];
                    if (ordinal < cumulative) {
                        egress = candidate;
                        break;
                    }
                }
                if (egress < 0)
                    hybrid_plan_detail::report_error(
                        status, HybridPlanError::InvalidSchedule);
                else
                    resolutions[owner_begin + record_offset].egress = egress;
                if (lane == __ffs(static_cast<int>(peers)) - 1)
                    multi_target_cursor[cursor] =
                        group_base + __popc(peers);
            }
            __syncwarp();
        }

        const int64_t num_groups =
            static_cast<int64_t>(num_destinations) * num_target_masks;
        for (int64_t flat = lane; flat < num_groups; flat += 32) {
            const int target_mask =
                static_cast<int>(flat % num_target_masks);
            if (__popc(static_cast<unsigned>(target_mask)) <= 1)
                continue;
            const int destination =
                static_cast<int>(flat / num_target_masks);
            int expected = 0;
            for (int egress = 0; egress < num_rails; ++egress)
                expected += multi_target_egress_quota[
                    hybrid_plan_detail::odme_offset(
                        owner, destination, target_mask, egress,
                        num_destinations, num_target_masks, num_rails)];
            if (multi_target_cursor[
                    hybrid_plan_detail::odm_offset(
                        owner, destination, target_mask,
                        num_destinations, num_target_masks)] != expected)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
        }
        return;
    }

    if constexpr (kStage == kHopGroupCount) {
        const int owner_source_channel = static_cast<int>(blockIdx.x);
        const int owner = owner_source_channel / num_channels;
        const int source_channel = owner_source_channel % num_channels;
        if (owner >= num_rails)
            return;
        for (int flat = lane; flat < num_rails * num_destinations; flat += 32) {
            const int egress = flat / num_destinations;
            const int destination = flat % num_destinations;
            owner_group_cursor[hybrid_plan_detail::lgcd_offset(
                owner, egress, source_channel, destination,
                num_rails, num_channels, num_destinations)] = 0;
        }
        __syncwarp();

        int local_paths[4] = {};
        int local_moved = 0;
        for (int64_t token = source_channel;
             token < num_tokens; token += num_channels) {
            const int64_t begin =
                (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;
            if (lane >= num_topk)
                continue;
            const auto record = records[begin + lane];
            if (record.target_mask == 0)
                continue;
            const int egress = resolutions[begin + lane].egress;
            if (egress < 0 or egress >= num_rails) {
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
                continue;
            }
            ++owner_group_cursor[hybrid_plan_detail::lgcd_offset(
                owner, egress, source_channel, record.destination,
                num_rails, num_channels, num_destinations)];

            int path = static_cast<int>(HopPathKind::TwoHop);
            if (egress == owner) {
                path = record.target_mask == (uint32_t{1} << owner) ?
                    static_cast<int>(HopPathKind::Direct) :
                    static_cast<int>(HopPathKind::DestinationForward);
            } else if ((record.target_mask &
                        (uint32_t{1} << egress)) != 0) {
                path = static_cast<int>(HopPathKind::SourceForward);
            }
            ++local_paths[path];
            local_moved += egress != owner;
        }
        for (int path = 0; path < 4; ++path) {
            const int units = ptx::reduce_add(local_paths[path]);
            if (lane == 0 and units != 0)
                atomicAdd(path_units + path, units);
        }
        const int moved_units = ptx::reduce_add(local_moved);
        if (lane == 0 and moved_units != 0)
            atomicAdd(moved_copies, moved_units);
        return;
    }

    if constexpr (kStage == kHopGroupPrefix) {
        const int egress = static_cast<int>(blockIdx.x);
        if (egress >= num_rails)
            return;

        // Prefix retained copies first, then moved copies, in one combined
        // sequence. Modulo striping therefore bounds their sum in every
        // target channel without a cross-block atomic or serial record scan.
        for (int destination = 0;
             destination < num_destinations; ++destination) {
            int retained_total = 0;
            for (int tile = 0; tile < num_channels; tile += 32) {
                const int source_channel = tile + lane;
                const auto offset = hybrid_plan_detail::lgcd_offset(
                    egress, egress, source_channel, destination,
                    num_rails, num_channels, num_destinations);
                const int count = source_channel < num_channels ?
                    owner_group_cursor[offset] : 0;
                int inclusive = count;
                for (int delta = 1; delta < 32; delta *= 2) {
                    const int upper = __shfl_up_sync(
                        0xffffffff, inclusive, delta);
                    if (lane >= delta)
                        inclusive += upper;
                }
                if (source_channel < num_channels)
                    owner_group_cursor[offset] =
                        retained_total + inclusive - count;
                retained_total += __shfl_sync(
                    0xffffffff, inclusive, 31);
            }

            int combined_total = retained_total;
            for (int owner = 0; owner < num_rails; ++owner) {
                if (owner == egress)
                    continue;
                for (int tile = 0; tile < num_channels; tile += 32) {
                    const int source_channel = tile + lane;
                    const auto offset = hybrid_plan_detail::lgcd_offset(
                        owner, egress, source_channel, destination,
                        num_rails, num_channels, num_destinations);
                    const int count = source_channel < num_channels ?
                        owner_group_cursor[offset] : 0;
                    int inclusive = count;
                    for (int delta = 1; delta < 32; delta *= 2) {
                        const int upper = __shfl_up_sync(
                            0xffffffff, inclusive, delta);
                        if (lane >= delta)
                            inclusive += upper;
                    }
                    if (source_channel < num_channels)
                        owner_group_cursor[offset] =
                            combined_total + inclusive - count;
                    combined_total += __shfl_sync(
                        0xffffffff, inclusive, 31);
                }
            }

            for (int channel = lane;
                 channel < num_channels; channel += 32) {
                const int group =
                    (egress * num_channels + channel) * num_destinations +
                    destination;
                retained[group] = retained_total > channel ?
                    1 + (retained_total - 1 - channel) / num_channels : 0;
                const int combined_count = combined_total > channel ?
                    1 + (combined_total - 1 - channel) / num_channels : 0;
                moved[group] = combined_count - retained[group];
            }
        }
        __syncwarp();

        int moved_base = 0;
        int retained_base = 0;
        const int num_groups = num_channels * num_destinations;
        for (int tile = 0; tile < num_groups; tile += 32) {
            const int group_index = tile + lane;
            int moved_count = 0;
            int retained_count = 0;
            int channel = 0;
            int destination = 0;
            if (group_index < num_groups) {
                channel = group_index / num_destinations;
                destination = group_index % num_destinations;
                const int group =
                    (egress * num_channels + channel) * num_destinations +
                    destination;
                retained_count = retained[group];
                moved_count = moved[group];
            }

            int moved_inclusive = moved_count;
            int retained_inclusive = retained_count;
            for (int delta = 1; delta < 32; delta *= 2) {
                const int moved_upper = __shfl_up_sync(
                    0xffffffff, moved_inclusive, delta);
                const int retained_upper = __shfl_up_sync(
                    0xffffffff, retained_inclusive, delta);
                if (lane >= delta) {
                    moved_inclusive += moved_upper;
                    retained_inclusive += retained_upper;
                }
            }
            if (group_index < num_groups) {
                const int group =
                    (egress * num_channels + channel) * num_destinations +
                    destination;
                retained[group] = retained_count;
                moved[group] = moved_count;
                retained_prefix[group] =
                    retained_base + retained_inclusive - retained_count;
                group_prefix[group] =
                    moved_base + moved_inclusive - moved_count;
            }
            moved_base += __shfl_sync(0xffffffff, moved_inclusive, 31);
            retained_base +=
                __shfl_sync(0xffffffff, retained_inclusive, 31);
        }
        if (lane == 0) {
            proxy_required[egress] = moved_base;
            if (moved_base > proxy_capacity_per_egress or
                retained_base > proxy_capacity_per_egress)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::CapacityExceeded);
        }
        return;
    }

    if constexpr (kStage == kHopSlotFinalize) {
        const int owner_source_channel = static_cast<int>(blockIdx.x);
        const int owner = owner_source_channel / num_channels;
        const int source_channel = owner_source_channel % num_channels;
        if (owner >= num_rails)
            return;
        for (int64_t token = source_channel;
             token < num_tokens; token += num_channels) {
            const int64_t begin =
                (static_cast<int64_t>(owner) * num_tokens + token) * num_topk;
            if (lane >= num_topk)
                continue;
            const auto record = records[begin + lane];
            if (record.target_mask == 0)
                continue;
            auto& resolution = resolutions[begin + lane];
            const int egress = resolution.egress;
            const int ordinal = owner_group_cursor[
                hybrid_plan_detail::lgcd_offset(
                    owner, egress, source_channel, record.destination,
                    num_rails, num_channels, num_destinations)]++;
            const int channel = ordinal % num_channels;
            const int retained_count = retained[
                (egress * num_channels + channel) * num_destinations +
                record.destination];
            const int remote_slot = ordinal / num_channels -
                (egress == owner ? 0 : retained_count);
            if (remote_slot < 0) {
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
                continue;
            }
            resolution.channel = channel;
            resolution.remote_slot = remote_slot;
            resolution.proxy_slot = egress == owner ? -1 :
                group_prefix[
                    (egress * num_channels + channel) * num_destinations +
                    record.destination] + remote_slot;
        }
    }
}

// Correctness-first deterministic hop planner. It always builds the endpoint
// plan first, then optionally moves only profitable residual copies to a
// third Rail. One lane keeps selection order reproducible; later profiling
// decides whether this phase warrants parallelization.
template <int kInstantiation = 0, bool kPrecounted = false,
          bool kAggregateAdaptive = false>
__global__ __launch_bounds__(32, 1)
void rail_balance_hop_plan_impl(
        const HopCopyRecord* records,
        HopCopyResolution* resolutions,
        int* pair_load,
        int* source_load,
        int* owner_remaining,
        int* endpoint_count,
        int* endpoint_egress_quota,
        int* multi_target_egress_quota,
        int* multi_target_cursor,
        int* owner_group_cursor,
        int* retained,
        int* moved,
        int* retained_prefix,
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
        const int planner_seed,
        const int planner_chunk_size,
        const int two_hop_threshold_percent,
        const int max_two_hop_percent,
        const int hop_penalty_percent) {
    static_assert(not kAggregateAdaptive or kPrecounted);
    const int lane = ptx::get_lane_idx();
    if (records == nullptr or resolutions == nullptr or pair_load == nullptr or
        source_load == nullptr or owner_remaining == nullptr or
        endpoint_count == nullptr or endpoint_egress_quota == nullptr or
        multi_target_egress_quota == nullptr or
        multi_target_cursor == nullptr or owner_group_cursor == nullptr or
        retained == nullptr or moved == nullptr or
        retained_prefix == nullptr or group_prefix == nullptr or
        proxy_required == nullptr or path_units == nullptr or
        moved_copies == nullptr or status == nullptr or num_rails < 1 or
        num_rails > 32 or num_tokens < 0 or num_topk < 1 or num_topk > 32 or
        num_channels < 1 or num_channels > kNumHybridMaxChannels or
        num_destinations < 2 or num_destinations > 32 or
        num_max_tokens_per_rank < num_tokens or
        proxy_capacity_per_egress < 0 or planner_seed < 0 or
        planner_chunk_size < 1 or
        kPrecounted != (planner_chunk_size > 1) or
        two_hop_threshold_percent < 0 or
        two_hop_threshold_percent > 10000 or
        max_two_hop_percent < 0 or max_two_hop_percent > 100 or
        hop_penalty_percent < 0 or hop_penalty_percent > 10000) {
        if (lane == 0)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::InvalidSchedule);
        return;
    }

    const int64_t num_records = static_cast<int64_t>(num_rails) *
        num_tokens * num_topk;
    const int num_groups = num_rails * num_channels * num_destinations;
    const int num_endpoint_groups =
        num_rails * num_destinations * num_rails;
    const int64_t num_endpoint_egress_groups =
        static_cast<int64_t>(num_endpoint_groups) * num_rails;
    const int num_target_masks = get_num_dense_target_masks(num_rails);
    const int64_t num_multi_target_groups =
        static_cast<int64_t>(num_rails) * num_destinations *
        num_target_masks;
    // Precount stores a multi-target marker in endpoint quota[0], then turns
    // each multi-target count row into per-egress quotas. The decision pass
    // uses multi_target_cursor as its compact active-group list, clears that
    // prefix, and the materializer reuses it as the consumed-record cursor.
    const bool has_multi_target =
        kPrecounted and endpoint_egress_quota[0] != 0;
    const bool has_unaggregated_multi_target =
        has_multi_target and num_target_masks == 1;
    if constexpr (kAggregateAdaptive)
        if (has_unaggregated_multi_target) {
            if (lane == 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
            return;
        }
    __syncwarp();
    for (int rail = lane; rail < num_rails; rail += 32) {
        source_load[rail] = 0;
        proxy_required[rail] = 0;
    }
    for (int i = lane; i < num_rails * num_destinations; i += 32) {
        pair_load[i] = 0;
        if constexpr (not kPrecounted)
            owner_remaining[i] = 0;
    }
    if constexpr (not kPrecounted)
        for (int i = lane; i < num_endpoint_groups; i += 32) {
            endpoint_count[i] = 0;
        }
    for (int64_t i = lane; i < num_endpoint_egress_groups; i += 32)
        endpoint_egress_quota[i] = 0;
    if constexpr (kPrecounted)
        if (num_target_masks > 1)
            for (int64_t i = lane;
                 i < num_multi_target_groups; i += 32)
                multi_target_cursor[i] = 0;
    for (int i = lane; i < num_groups; i += 32) {
        retained[i] = 0;
        moved[i] = 0;
        group_prefix[i] = 0;
    }
    for (int path = lane; path < 4; path += 32)
        path_units[path] = 0;
    const int precounted_units = kPrecounted ? *moved_copies : 0;
    if (lane == 0)
        *moved_copies = 0;
    __syncwarp();

    // Validate the fixed table and reserve every owner's still-unscheduled
    // traffic. This prevents inbound moves from consuming an owner's only
    // feasible direct capacity.
    int lane_units = 0;
    const int64_t owner_records =
        static_cast<int64_t>(num_tokens) * num_topk;
    if constexpr (kPrecounted) {
        lane_units = lane == 0 ? precounted_units : 0;
    } else {
        bool reached_padding = false;
        int previous_destination = -1;
        for (int64_t index = static_cast<int64_t>(lane) * owner_records;
             lane < num_rails and index < (lane + 1) * owner_records; ++index) {
            if (index % num_topk == 0) {
                reached_padding = false;
                previous_destination = -1;
            }
            resolutions[index] = {-1, -1, -1, -1};
            const auto record = records[index];
            if (record.target_mask == 0) {
                if (record.destination != -1)
                    hybrid_plan_detail::report_error(
                        status, HybridPlanError::InvalidSchedule);
                reached_padding = true;
                continue;
            }
            if (reached_padding or record.destination < 0 or
                    record.destination >= num_destinations or
                    record.destination <= previous_destination or
                    (num_rails < 32 and
                     (record.target_mask >> num_rails) != 0)) {
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
                continue;
            }
            previous_destination = record.destination;
            ++owner_remaining[lane * num_destinations + record.destination];
            if (__popc(record.target_mask) == 1) {
                const int target =
                    __ffs(static_cast<int>(record.target_mask)) - 1;
                ++endpoint_count[hybrid_plan_detail::odt_offset(
                    lane, record.destination, target,
                    num_destinations, num_rails)];
            }
            ++lane_units;
        }
    }
    const int total_units = ptx::reduce_add(lane_units);
    if (*status != static_cast<int>(HybridPlanError::Success))
        return;

    int num_active_groups = 0;
    if (lane == 0 and
        (not kPrecounted or has_unaggregated_multi_target)) {
        for (int64_t index = 0; index < num_records; ++index) {
            const auto record = records[index];
            if (record.target_mask == 0) {
                index += num_topk - 1 - index % num_topk;
                continue;
            }
            if constexpr (kPrecounted)
                if (__popc(record.target_mask) == 1)
                    continue;
            const int owner = static_cast<int>(
                index / (static_cast<int64_t>(num_tokens) * num_topk));
            const int destination = record.destination;
            --owner_remaining[owner * num_destinations + destination];
            const int best_egress = hybrid_plan_detail::choose_endpoint(
                record.target_mask | (uint32_t{1} << owner),
                record.target_mask, owner, destination, 1, index,
                pair_load, source_load, owner_remaining,
                num_rails, num_destinations, num_max_tokens_per_rank,
                planner_seed);
            if (best_egress < 0) {
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::CapacityExceeded);
                break;
            }
            resolutions[index] = {best_egress, -1, -1, -1};
            ++pair_load[destination * num_rails + best_egress];
            ++source_load[best_egress];
            if (best_egress != owner)
                ++proxy_required[best_egress];
        }
    }

    if constexpr (kPrecounted) {
        // Load dense counts in parallel, but apply non-empty groups in stable
        // dense order because each choice updates the shared load state.
        if (has_multi_target and num_target_masks > 1) {
            for (int64_t base = 0;
                 base < num_multi_target_groups; base += 32) {
                const int64_t group = base + lane;
                const int target_mask = group < num_multi_target_groups ?
                    static_cast<int>(group % num_target_masks) : 0;
                const auto count_offset = group < num_multi_target_groups ?
                    group * num_rails : 0;
                const int count =
                    group < num_multi_target_groups and
                    __popc(static_cast<unsigned>(target_mask)) > 1 ?
                        multi_target_egress_quota[count_offset] : 0;
                unsigned active = ptx::gather(count > 0);
                if constexpr (kAggregateAdaptive) {
                    if (count > 0)
                        multi_target_cursor[
                            num_active_groups +
                            __popc(active &
                                   ((uint32_t{1} << lane) - 1))] =
                            num_endpoint_groups +
                            static_cast<int>(group);
                    num_active_groups += __popc(active);
                }

                while (active != 0) {
                    const int source_lane =
                        __ffs(static_cast<int>(active)) - 1;
                    const int64_t selected_group = base + source_lane;
                    const int selected_count =
                        ptx::exchange(count, source_lane);
                    if (lane == 0) {
                        int64_t decoded = selected_group;
                        const int selected_mask = static_cast<int>(
                            decoded % num_target_masks);
                        decoded /= num_target_masks;
                        const int destination = static_cast<int>(
                            decoded % num_destinations);
                        const int owner = static_cast<int>(
                            decoded / num_destinations);
                        int* quota = multi_target_egress_quota +
                            selected_group * num_rails;
                        for (int egress = 0;
                             egress < num_rails; ++egress)
                            quota[egress] = 0;
                        if (not hybrid_plan_detail::assign_endpoint_group(
                                quota, pair_load, source_load,
                                owner_remaining, proxy_required,
                                owner, destination,
                                static_cast<uint32_t>(selected_mask),
                                selected_count, selected_group,
                                num_rails, num_destinations, num_tokens,
                                num_max_tokens_per_rank, planner_seed,
                                planner_chunk_size))
                            hybrid_plan_detail::report_error(
                                status, HybridPlanError::CapacityExceeded);
                    }
                    active &= active - 1;
                }
            }
        }

        if (lane == 0) {
            for (int owner = 0; owner < num_rails; ++owner) {
                for (int destination = 0;
                     destination < num_destinations; ++destination) {
                    for (int target = 0; target < num_rails; ++target) {
                        const auto group = hybrid_plan_detail::odt_offset(
                            owner, destination, target,
                            num_destinations, num_rails);
                        const int count = endpoint_count[group];
                        if (count == 0)
                            continue;
                        if constexpr (kAggregateAdaptive)
                            if (num_target_masks > 1)
                                multi_target_cursor[num_active_groups++] =
                                    static_cast<int>(group);
                        int* quota = endpoint_egress_quota +
                            static_cast<int64_t>(group) * num_rails;
                        if (not hybrid_plan_detail::assign_endpoint_group(
                                quota, pair_load, source_load,
                                owner_remaining, proxy_required,
                                owner, destination, uint32_t{1} << target,
                                count, group, num_rails, num_destinations,
                                num_tokens, num_max_tokens_per_rank,
                                planner_seed, planner_chunk_size))
                            hybrid_plan_detail::report_error(
                                status, HybridPlanError::CapacityExceeded);
                    }
                }
            }
        }
        if constexpr (kAggregateAdaptive)
            if (num_target_masks > 1)
                num_active_groups =
                    ptx::exchange(num_active_groups, 0);
    }
    __syncwarp();
    if (*status != static_cast<int>(HybridPlanError::Success))
        return;
    if constexpr (kPrecounted) {
        if constexpr (kAggregateAdaptive) {
            const bool valid = hybrid_plan_detail::rebalance_endpoint_quotas(
                    endpoint_egress_quota, multi_target_egress_quota,
                    num_target_masks > 1 ? multi_target_cursor : nullptr,
                    num_active_groups,
                    pair_load, source_load, owner_remaining, proxy_required,
                    total_units, num_rails, num_destinations,
                    has_multi_target ? num_target_masks : 1,
                    num_max_tokens_per_rank,
                    proxy_capacity_per_egress, planner_seed,
                    two_hop_threshold_percent, max_two_hop_percent,
                    hop_penalty_percent);
            if (num_target_masks > 1) {
                for (int index = lane;
                     index < num_active_groups; index += 32)
                    multi_target_cursor[index] = 0;
                __syncwarp();
            }
            if (not valid) {
                if (lane == 0)
                    hybrid_plan_detail::report_error(
                        status, HybridPlanError::InvalidSchedule);
            }
        }
        return;
    }

    const int two_hop_cap = num_rails < 2 ? 0 : static_cast<int>(
        static_cast<int64_t>(total_units) * max_two_hop_percent / 100);
    int selected_two_hop = 0;
    while (selected_two_hop < two_hop_cap) {
        if (lane == 0)
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const auto peaks = hybrid_plan_detail::find_load_peaks(
                    pair_load + destination * num_rails, num_rails);
                owner_remaining[destination] = peaks.first;
                owner_remaining[num_destinations + destination] = peaks.second;
                owner_remaining[2 * num_destinations + destination] =
                    peaks.first_count;
            }
        __syncwarp();
        const auto source_peaks =
            hybrid_plan_detail::find_load_peaks(source_load, num_rails);

        hybrid_plan_detail::AdaptiveCandidate best = {
            -1, INT64_MIN, -1, 1, 1,
            INT_MAX, INT_MAX, INT_MAX, INT_MAX};
        const int64_t num_owner_tokens =
            static_cast<int64_t>(num_rails) * num_tokens;
        for (int64_t owner_token = lane;
             owner_token < num_owner_tokens; owner_token += 32) {
            const int owner = static_cast<int>(owner_token / num_tokens);
            const int64_t first_index = owner_token * num_topk;
            for (int slot = 0; slot < num_topk; ++slot) {
                const int64_t index = first_index + slot;
                const auto record = records[index];
                if (record.target_mask == 0)
                    break;
                const int old_egress = resolutions[index].egress;
                const uint32_t endpoints =
                    record.target_mask | (uint32_t{1} << owner);
                if ((endpoints & (uint32_t{1} << old_egress)) == 0)
                    continue;

                const hybrid_plan_detail::LoadPeaks pair_peaks = {
                    owner_remaining[record.destination],
                    owner_remaining[num_destinations + record.destination],
                    owner_remaining[2 * num_destinations + record.destination],
                };
                const int pair_before = pair_peaks.first;
                const int source_before = source_peaks.first;
                const int old_pair = pair_load[
                    record.destination * num_rails + old_egress];
                const int old_source = source_load[old_egress];
                if (old_pair < pair_before and old_source < source_before)
                    continue;

                const int old_forwards = (old_egress != owner) +
                    __popc(record.target_mask &
                           ~(uint32_t{1} << old_egress));
                for (int egress = 0; egress < num_rails; ++egress) {
                    if ((endpoints & (uint32_t{1} << egress)) != 0 or
                        pair_load[record.destination * num_rails + egress] >=
                            num_max_tokens_per_rank or
                        proxy_required[egress] >= proxy_capacity_per_egress)
                        continue;

                    const int new_pair = pair_load[
                        record.destination * num_rails + egress];
                    const int new_source = source_load[egress];
                    const int pair_after =
                        hybrid_plan_detail::peak_after_move(
                            pair_peaks, old_pair, new_pair);
                    const int source_after =
                        hybrid_plan_detail::peak_after_move(
                            source_peaks, old_source, new_source);
                    const int pair_relief =
                        min(1, max(0, old_pair - new_pair));
                    const int source_relief =
                        min(1, max(0, old_source - new_source));
                    const int added_hops = 1 + __popc(record.target_mask) -
                        old_forwards;
                    const int64_t net_gain = static_cast<int64_t>(
                        pair_relief + source_relief) * 100 -
                        static_cast<int64_t>(hop_penalty_percent) * added_hops;
                    const int64_t threshold =
                        static_cast<int64_t>(two_hop_threshold_percent) *
                        (static_cast<int64_t>(pair_before) + source_before);
                    if (net_gain <= 0 or net_gain <= threshold)
                        continue;

                    const int origin = static_cast<int>(
                        (static_cast<int64_t>(planner_seed) +
                         record.destination + index) % num_rails);
                    const hybrid_plan_detail::AdaptiveCandidate candidate = {
                        index,
                        net_gain,
                        egress,
                        1,
                        1,
                        pair_after,
                        source_after,
                        added_hops,
                        (egress - origin + num_rails) % num_rails,
                    };
                    if (hybrid_plan_detail::better_candidate(candidate, best))
                        best = candidate;
                }
            }
        }
        for (int source_lane = 0; source_lane < 32; ++source_lane) {
            const auto candidate = ptx::exchange(best, source_lane);
            if (hybrid_plan_detail::better_candidate(candidate, best))
                best = candidate;
        }
        const int64_t best_index = best.index;
        const int best_egress = best.egress;
        const int best_added_hops = best.added_hops;
        if (best_index < 0)
            break;

        const auto best_record = records[best_index];
        const int old_egress = resolutions[best_index].egress;
        const int pair_gap = pair_load[
            best_record.destination * num_rails + old_egress] -
            pair_load[best_record.destination * num_rails + best_egress];
        const int source_gap =
            source_load[old_egress] - source_load[best_egress];
        int batch = min(
            two_hop_cap - selected_two_hop,
            proxy_capacity_per_egress - proxy_required[best_egress]);
        batch = min(
            batch, num_max_tokens_per_rank -
                pair_load[best_record.destination * num_rails + best_egress]);
        // Batch only while both score dimensions improve. If one is already
        // tied or inverted, the next copy may change the preferred candidate.
        if (pair_gap <= 0 or source_gap <= 0) {
            batch = 1;
        } else {
            batch = min(batch, pair_gap / 2 + pair_gap % 2);
            batch = min(batch, source_gap / 2 + source_gap % 2);
        }
        const hybrid_plan_detail::LoadPeaks pair_peaks = {
            owner_remaining[best_record.destination],
            owner_remaining[num_destinations + best_record.destination],
            owner_remaining[2 * num_destinations + best_record.destination],
        };
        const int old_pair = pair_load[
            best_record.destination * num_rails + old_egress];
        const int pair_other =
            (pair_peaks.first_count >
             static_cast<int>(old_pair == pair_peaks.first)) ?
                pair_peaks.first : pair_peaks.second;
        const int old_source = source_load[old_egress];
        const int source_other =
            (source_peaks.first_count >
             static_cast<int>(old_source == source_peaks.first)) ?
                source_peaks.first : source_peaks.second;
        int critical_batch = 0;
        if (old_pair >= pair_other) {
            const int difference = old_pair - pair_other;
            critical_batch = difference >= batch ? batch : difference + 1;
        }
        if (old_source >= source_other) {
            const int difference = old_source - source_other;
            critical_batch = max(
                critical_batch, difference >= batch ? batch : difference + 1);
        }
        batch = min(batch, critical_batch);

        // Relief and hop cost stay admissible until a gap closes or the old
        // Rail stops being critical. Move that deterministic prefix at once.
        int migrated = 0;
        if (lane == 0)
            for (int64_t index = 0;
                 index < num_records and migrated < batch; ++index) {
                const auto record = records[index];
                if (record.target_mask == 0) {
                    index += num_topk - 1 - index % num_topk;
                    continue;
                }
                if (record.destination != best_record.destination or
                    resolutions[index].egress != old_egress)
                    continue;
                const int owner = static_cast<int>(
                    index / (static_cast<int64_t>(num_tokens) * num_topk));
                const uint32_t endpoints =
                    record.target_mask | (uint32_t{1} << owner);
                if ((endpoints & (uint32_t{1} << old_egress)) == 0 or
                    (endpoints & (uint32_t{1} << best_egress)) != 0)
                    continue;
                const int old_forwards = (old_egress != owner) +
                    __popc(record.target_mask &
                           ~(uint32_t{1} << old_egress));
                const int added_hops = 1 + __popc(record.target_mask) -
                    old_forwards;
                if (added_hops != best_added_hops)
                    continue;

                --pair_load[record.destination * num_rails + old_egress];
                ++pair_load[record.destination * num_rails + best_egress];
                --source_load[old_egress];
                ++source_load[best_egress];
                if (old_egress != owner)
                    --proxy_required[old_egress];
                ++proxy_required[best_egress];
                resolutions[index].egress = best_egress;
                ++migrated;
            }
        migrated = __shfl_sync(0xffffffff, migrated, 0);
        if (migrated == 0) {
            if (lane == 0)
                hybrid_plan_detail::report_error(
                    status, HybridPlanError::InvalidSchedule);
            break;
        }
        selected_two_hop += migrated;
        __syncwarp();
    }
    if (*status != static_cast<int>(HybridPlanError::Success))
        return;

    if (lane != 0)
        return;

    // Materialize dense channel-local slots only after two-hop selection has
    // finalized every egress. A bit set denotes a channel at the current
    // minimum load. Selecting the first set bit from source_channel preserves
    // the original (load, circular distance, channel) ordering while reducing
    // each lookup from C counters to ceil(C / 32) words. owner_remaining is
    // dead after endpoint assignment and holds the number of minimum-load
    // channels until final slot materialization completes.
    const int channel_mask_words = (num_channels + 31) / 32;
    for (int egress = 0; egress < num_rails; ++egress) {
        for (int destination = 0;
             destination < num_destinations; ++destination) {
            owner_remaining[egress * num_destinations + destination] =
                num_channels;
            for (int word = 0; word < channel_mask_words; ++word) {
                const int valid_bits = min(32, num_channels - word * 32);
                const uint32_t mask = valid_bits == 32 ? UINT32_MAX :
                    (uint32_t{1} << valid_bits) - 1;
                group_prefix[hybrid_plan_detail::gcd_offset(
                    egress, word, destination,
                    num_channels, num_destinations)] =
                        static_cast<int32_t>(mask);
            }
        }
    }

    for (int64_t index = 0; index < num_records; ++index) {
        const auto record = records[index];
        if (record.target_mask == 0) {
            index += num_topk - 1 - index % num_topk;
            continue;
        }
        const int owner = static_cast<int>(
            index / (static_cast<int64_t>(num_tokens) * num_topk));
        const int token = static_cast<int>((index / num_topk) % num_tokens);
        const int egress = resolutions[index].egress;
        const int source_channel = token % num_channels;
        const bool is_moved = egress != owner;
        int best_channel = -1;
        const int source_word = source_channel / 32;
        const int source_bit = source_channel % 32;
        for (int word = source_word;
             word < channel_mask_words and best_channel < 0; ++word) {
            uint32_t mask = static_cast<uint32_t>(group_prefix[
                hybrid_plan_detail::gcd_offset(
                    egress, word, record.destination,
                    num_channels, num_destinations)]);
            if (word == source_word)
                mask &= UINT32_MAX << source_bit;
            if (mask != 0)
                best_channel = word * 32 + __ffs(mask) - 1;
        }
        for (int word = 0;
             word <= source_word and best_channel < 0; ++word) {
            uint32_t mask = static_cast<uint32_t>(group_prefix[
                hybrid_plan_detail::gcd_offset(
                    egress, word, record.destination,
                    num_channels, num_destinations)]);
            if (word == source_word) {
                mask &= source_bit == 0 ? 0 :
                    (uint32_t{1} << source_bit) - 1;
            }
            if (mask != 0)
                best_channel = word * 32 + __ffs(mask) - 1;
        }
        if (best_channel < 0) {
            hybrid_plan_detail::report_error(
                status, HybridPlanError::CapacityExceeded);
            return;
        }

        const int group =
            (egress * num_channels + best_channel) * num_destinations +
            record.destination;
        int& group_count = is_moved ? moved[group] : retained[group];
        resolutions[index] = {egress, best_channel, group_count, -1};
        ++group_count;

        const int mask_word = best_channel / 32;
        const int mask_bit = best_channel % 32;
        const auto mask_offset = hybrid_plan_detail::gcd_offset(
            egress, mask_word, record.destination,
            num_channels, num_destinations);
        group_prefix[mask_offset] &=
            static_cast<int32_t>(~(uint32_t{1} << mask_bit));
        int& channels_at_minimum = owner_remaining[
            egress * num_destinations + record.destination];
        if (--channels_at_minimum == 0) {
            channels_at_minimum = num_channels;
            for (int word = 0; word < channel_mask_words; ++word) {
                const int valid_bits = min(
                    32, num_channels - word * 32);
                const uint32_t mask = valid_bits == 32 ? UINT32_MAX :
                    (uint32_t{1} << valid_bits) - 1;
                group_prefix[hybrid_plan_detail::gcd_offset(
                    egress, word, record.destination,
                    num_channels, num_destinations)] =
                        static_cast<int32_t>(mask);
            }
        }

        int path = static_cast<int>(HopPathKind::TwoHop);
        if (egress == owner) {
            path = record.target_mask == (uint32_t{1} << owner) ?
                static_cast<int>(HopPathKind::Direct) :
                static_cast<int>(HopPathKind::DestinationForward);
        } else if ((record.target_mask &
                    (uint32_t{1} << egress)) != 0) {
            path = static_cast<int>(HopPathKind::SourceForward);
        }
        ++path_units[path];
        *moved_copies += is_moved;
    }

    for (int egress = 0; egress < num_rails; ++egress) {
        int prefix = 0;
        int retained_cursor = 0;
        for (int channel = 0; channel < num_channels; ++channel) {
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const int group =
                    (egress * num_channels + channel) * num_destinations +
                    destination;
                retained_prefix[group] = retained_cursor;
                retained_cursor += retained[group];
                group_prefix[group] = prefix;
                prefix += moved[group];
            }
        }
        proxy_required[egress] = prefix;
        if (prefix > proxy_capacity_per_egress or
                retained_cursor > proxy_capacity_per_egress)
            hybrid_plan_detail::report_error(
                status, HybridPlanError::CapacityExceeded);
    }

    for (int64_t index = 0; index < num_records; ++index) {
        auto& resolution = resolutions[index];
        if (resolution.egress < 0) {
            index += num_topk - 1 - index % num_topk;
            continue;
        }
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

    for (int64_t token = channel; token < num_tokens; token += num_channels) {
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
