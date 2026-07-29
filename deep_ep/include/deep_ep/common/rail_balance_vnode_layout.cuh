#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>

#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/rail_balance_layout.cuh>

namespace deep_ep::elastic::rail_balance {

// C060 keeps the frozen C040/C050 record bytes immutable. Per-contribution
// forwarding state lives in this sidecar instead of ProxyDescriptor::reserved
// or TokenLayout metadata, both of which have existing replay semantics.
struct alignas(16) VNodeRoute {
    uint64_t fingerprint;
    int generation;
    int ingress_rank;
    int ingress_slot;
    int topk_lane;
    int expert_rank;
    int expert_slot;
};
static_assert(sizeof(VNodeRoute) == 32);
static_assert(alignof(VNodeRoute) == 16);
static_assert(offsetof(VNodeRoute, fingerprint) == 0);
static_assert(offsetof(VNodeRoute, generation) == 8);
static_assert(offsetof(VNodeRoute, expert_slot) == 28);

struct VNodeStageLayout {
    SourceShuffleLayout records;
    int64_t route_offset;
    int64_t arena_bytes;
    void* base;

    __forceinline__ __device__ __host__
    VNodeStageLayout(const int& num_hidden_bytes,
                     const int& num_topk,
                     const int& capacity,
                     void* base = nullptr):
        records(num_hidden_bytes, num_topk, capacity, base),
        route_offset(math::align<int64_t>(
            records.ready_offset + math::align<int64_t>(
                static_cast<int64_t>(capacity) * sizeof(int),
                ptx::kNumTMAAlignBytes),
            ptx::kNumTMAAlignBytes)),
        arena_bytes(math::align<int64_t>(
            route_offset + static_cast<int64_t>(capacity) *
                               sizeof(VNodeRoute),
            kArenaAlignmentBytes)),
        base(base) {
        EP_UNIFIED_ASSERT(capacity > 0);
        EP_UNIFIED_ASSERT(route_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes >= records.arena_bytes);
    }

    __forceinline__ __device__ __host__
    VNodeRoute* get_route_ptr(const int& slot = 0) const {
        EP_UNIFIED_ASSERT(slot >= 0 and slot < records.physical_capacity);
        return math::advance_ptr<VNodeRoute>(base, route_offset) + slot;
    }
};

// Every destination expert contribution returns independently. It first lands
// in the original owner's symmetric partial buffer; only after the final
// all-rank barrier does a local kernel reduce lanes into a PyTorch output.
// Ordinary torch allocations are never passed to get_sym_ptr<LSA>.
struct VNodeOwnerLayout {
    int num_hidden_bytes;
    int num_max_tokens;
    int num_topk;
    int num_partial_slots;
    int64_t values_bytes;
    int64_t ready_offset;
    int64_t arena_bytes;
    void* base;

    __forceinline__ __device__ __host__
    VNodeOwnerLayout(const int& num_hidden_bytes,
                     const int& num_max_tokens,
                     const int& num_topk,
                     void* base = nullptr):
        num_hidden_bytes(num_hidden_bytes),
        num_max_tokens(num_max_tokens),
        num_topk(num_topk),
        num_partial_slots(static_cast<int>(
            static_cast<int64_t>(num_max_tokens) * num_topk)),
        values_bytes(static_cast<int64_t>(num_hidden_bytes) *
                     num_partial_slots),
        ready_offset(math::align<int64_t>(
            values_bytes, ptx::kNumTMAAlignBytes)),
        arena_bytes(math::align<int64_t>(
            ready_offset + math::align<int64_t>(
                static_cast<int64_t>(num_partial_slots) * sizeof(int),
                ptx::kNumTMAAlignBytes),
            kArenaAlignmentBytes)),
        base(base) {
        EP_UNIFIED_ASSERT(num_hidden_bytes > 0);
        EP_UNIFIED_ASSERT(
            num_hidden_bytes % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(num_max_tokens > 0);
        EP_UNIFIED_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_UNIFIED_ASSERT(
            static_cast<int64_t>(num_max_tokens) * num_topk <= INT_MAX);
    }

    __forceinline__ __device__ __host__
    int get_partial_slot(const int& token_idx,
                         const int& topk_lane) const {
        EP_UNIFIED_ASSERT(token_idx >= 0 and token_idx < num_max_tokens);
        EP_UNIFIED_ASSERT(topk_lane >= 0 and topk_lane < num_topk);
        return token_idx * num_topk + topk_lane;
    }

    __forceinline__ __device__ __host__
    void* get_value_ptr(const int& token_idx,
                        const int& topk_lane) const {
        const int partial_slot = get_partial_slot(token_idx, topk_lane);
        return math::advance_ptr(
            base, static_cast<int64_t>(partial_slot) * num_hidden_bytes);
    }

    __forceinline__ __device__ __host__
    int* get_ready_ptr(const int& token_idx,
                       const int& topk_lane) const {
        return math::advance_ptr<int>(base, ready_offset) +
               get_partial_slot(token_idx, topk_lane);
    }
};

struct VNodeRoundTripLayout {
    int num_hidden_bytes;
    int num_topk;
    // Base records are globally addressed as [destination][local slot].
    // `physical_capacity` remains a compatibility alias for the flattened
    // base capacity used by the C060 single-destination prototype.
    int physical_capacity;
    int num_destinations;
    int destination_capacity;
    int num_source_ranks;
    int num_max_tokens;
    int rail_capacity;
    int expert_capacity;
    VNodeStageLayout rail;
    int64_t expert_offset;
    VNodeStageLayout expert;
    VNodeOwnerLayout owner;
    int64_t role_arena_bytes;
    int64_t arena_bytes;
    void* base;

    __forceinline__ __device__ __host__
    VNodeRoundTripLayout(const int& num_hidden_bytes,
                         const int& num_topk,
                         const int& num_destinations,
                         const int& destination_capacity,
                         const int& num_source_ranks,
                         const int& num_max_tokens,
                         void* base = nullptr):
        num_hidden_bytes(num_hidden_bytes),
        num_topk(num_topk),
        physical_capacity(static_cast<int>(
            static_cast<int64_t>(num_destinations) *
            destination_capacity)),
        num_destinations(num_destinations),
        destination_capacity(destination_capacity),
        num_source_ranks(num_source_ranks),
        num_max_tokens(num_max_tokens),
        rail_capacity(static_cast<int>(
            static_cast<int64_t>(num_destinations) * destination_capacity *
            (num_topk + 1))),
        expert_capacity(static_cast<int>(
            static_cast<int64_t>(num_source_ranks) * destination_capacity *
            num_topk)),
        rail(num_hidden_bytes, num_topk, rail_capacity, base),
        expert_offset(rail.arena_bytes),
        expert(num_hidden_bytes, num_topk, expert_capacity,
               base ? math::advance_ptr(base, expert_offset) : nullptr),
        owner(num_hidden_bytes, num_max_tokens, num_topk,
              base ? math::advance_ptr(base, expert_offset) : nullptr),
        role_arena_bytes(expert.arena_bytes > owner.arena_bytes
                             ? expert.arena_bytes
                             : owner.arena_bytes),
        arena_bytes(expert_offset + role_arena_bytes),
        base(base) {
        const int64_t physical_capacity_i64 =
            static_cast<int64_t>(num_destinations) * destination_capacity;
        const int64_t rail_capacity_i64 =
            physical_capacity_i64 * (num_topk + 1);
        const int64_t expert_capacity_i64 =
            static_cast<int64_t>(num_source_ranks) * destination_capacity *
            num_topk;
        EP_UNIFIED_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_UNIFIED_ASSERT(num_destinations > 0);
        EP_UNIFIED_ASSERT(destination_capacity > 0);
        EP_UNIFIED_ASSERT(
            physical_capacity_i64 > 0 and physical_capacity_i64 <= INT_MAX);
        EP_UNIFIED_ASSERT(num_source_ranks > 0);
        EP_UNIFIED_ASSERT(num_max_tokens > 0);
        EP_UNIFIED_ASSERT(
            rail_capacity_i64 > 0 and rail_capacity_i64 <= INT_MAX);
        EP_UNIFIED_ASSERT(
            expert_capacity_i64 > 0 and expert_capacity_i64 <= INT_MAX);
        EP_UNIFIED_ASSERT(expert_offset % kArenaAlignmentBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes % kArenaAlignmentBytes == 0);
    }

    // C060 compatibility: one destination with P base slots.
    __forceinline__ __device__ __host__
    VNodeRoundTripLayout(const int& num_hidden_bytes,
                         const int& num_topk,
                         const int& physical_capacity,
                         const int& num_source_ranks,
                         const int& num_max_tokens,
                         void* base = nullptr):
        VNodeRoundTripLayout(
            num_hidden_bytes, num_topk, 1, physical_capacity,
            num_source_ranks, num_max_tokens, base) {}

    __forceinline__ __device__ __host__
    int get_ingress_slot(const int& destination,
                         const int& destination_slot) const {
        EP_UNIFIED_ASSERT(
            destination >= 0 and destination < num_destinations);
        EP_UNIFIED_ASSERT(
            destination_slot >= 0 and
            destination_slot < destination_capacity);
        return destination * destination_capacity + destination_slot;
    }

    __forceinline__ __device__ __host__
    int get_destination(const int& ingress_slot) const {
        EP_UNIFIED_ASSERT(
            ingress_slot >= 0 and ingress_slot < physical_capacity);
        return ingress_slot / destination_capacity;
    }

    __forceinline__ __device__ __host__
    int get_destination_slot(const int& ingress_slot) const {
        EP_UNIFIED_ASSERT(
            ingress_slot >= 0 and ingress_slot < physical_capacity);
        return ingress_slot % destination_capacity;
    }

    __forceinline__ __device__ __host__
    int get_ingress_rank(const int& destination,
                         const int& source_egress_rank) const {
        EP_UNIFIED_ASSERT(
            destination >= 0 and destination < num_destinations);
        EP_UNIFIED_ASSERT(
            source_egress_rank >= 0 and
            source_egress_rank < num_source_ranks);
        return num_source_ranks + destination * num_source_ranks +
               source_egress_rank;
    }

    __forceinline__ __device__ __host__
    int get_contribution_slot(const int& ingress_slot,
                              const int& topk_lane) const {
        EP_UNIFIED_ASSERT(
            ingress_slot >= 0 and ingress_slot < physical_capacity);
        EP_UNIFIED_ASSERT(topk_lane >= 0 and topk_lane < num_topk);
        return physical_capacity + ingress_slot * num_topk + topk_lane;
    }

    __forceinline__ __device__ __host__
    int get_expert_slot(const int& source_egress_rank,
                        const int& destination_slot,
                        const int& topk_lane) const {
        EP_UNIFIED_ASSERT(
            source_egress_rank >= 0 and
            source_egress_rank < num_source_ranks);
        EP_UNIFIED_ASSERT(
            destination_slot >= 0 and
            destination_slot < destination_capacity);
        EP_UNIFIED_ASSERT(topk_lane >= 0 and topk_lane < num_topk);
        return ((source_egress_rank * destination_capacity +
                 destination_slot) *
                num_topk + topk_lane);
    }
};

}  // namespace deep_ep::elastic::rail_balance
