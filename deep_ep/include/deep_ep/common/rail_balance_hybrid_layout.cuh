#pragma once

#include <cstdint>
#include <limits>

#include <deep_ep/common/layout.cuh>

namespace deep_ep::elastic::rail_balance {

// These are the force-v1 ABI maxima. Each rank stores only its own channel
// counts; peer ranks read the fixed-capacity array through LSA. Force-v1 must
// inherit the immutable legacy Hybrid workspace ceiling.
static constexpr int kNumHybridMaxChannels = deep_ep::kNumMaxChannels;
static constexpr int kNumHybridMaxDestinations = 32;
static constexpr int64_t kNumHybridBufferAlignmentBytes = 2 * 1024 * 1024;
static constexpr int kMaxHybridPolicyThresholdPercent = 3100;

// Planner-only policy ABI. Dispatch, shuffle, and combine consume only the
// resulting quota/segments and therefore stay policy-free.
enum class HybridPolicy : int {
    All = 0,
    Active = 1,
    Adaptive = 2,
};

__forceinline__ __device__ __host__ constexpr bool
is_valid_hybrid_policy(const int policy) {
    return policy >= static_cast<int>(HybridPolicy::All) and
           policy <= static_cast<int>(HybridPolicy::Adaptive);
}

// Force-forward metadata adds one proxy slot to the legacy two-field header.
// Keep its ABI here so host allocation and both persistent kernels cannot
// drift independently.
static constexpr int kHybridForwardSrcTokenDim = 0;
static constexpr int kHybridForwardLastTokenDim = 1;
static constexpr int kHybridForwardProxySlotDim = 2;
static constexpr int kHybridForwardRouteBaseDim = 3;
__forceinline__ __device__ __host__ constexpr int
get_num_hybrid_forward_metadata_dims(const int num_topk) {
    return kHybridForwardRouteBaseDim + 2 * num_topk;
}

// Stable force-v1 device status shared by host transaction code and the JIT
// planner/shuffle kernels.
enum class HybridPlanError : int {
    Success = 0,
    CapacityExceeded = 1,
    ExpertOutOfRange = 2,
    DuplicateExpert = 3,
    InvalidSchedule = 4,
};

struct alignas(ptx::kNumTMAAlignBytes) HybridControl {
    int32_t invocation_id;
    int32_t state;
    int32_t error_code;
    int32_t error_detail;
    int32_t num_channels;
    int32_t num_destinations;
    int32_t proxy_required;
    int32_t reserved;
};

EP_STATIC_ASSERT(sizeof(HybridControl) == ptx::kNumTMAAlignBytes,
                 "Hybrid control ABI must occupy one TMA-aligned unit");

// Capacity-independent force-arena prefix.
static constexpr int64_t kHybridChannelCountOffsetBytes =
    math::constexpr_align<int64_t>(
        sizeof(HybridControl), ptx::kNumTMAAlignBytes);
static constexpr int64_t kHybridChannelCountBytes =
    static_cast<int64_t>(kNumHybridMaxChannels) *
        kNumHybridMaxDestinations * sizeof(int32_t);
static constexpr int64_t kHybridProxyReadyOffsetBytes =
    math::constexpr_align<int64_t>(
        kHybridChannelCountOffsetBytes + kHybridChannelCountBytes,
        ptx::kNumTMAAlignBytes);

__forceinline__ __device__ __host__ int64_t checked_add_i64(
        const int64_t lhs, const int64_t rhs) {
    EP_UNIFIED_ASSERT(lhs >= 0 and rhs >= 0);
    EP_UNIFIED_ASSERT(lhs <= std::numeric_limits<int64_t>::max() - rhs);
    return lhs + rhs;
}

__forceinline__ __device__ __host__ int64_t checked_mul_i64(
        const int64_t lhs, const int64_t rhs) {
    EP_UNIFIED_ASSERT(lhs >= 0 and rhs >= 0);
    EP_UNIFIED_ASSERT(lhs == 0 or
                      rhs <= std::numeric_limits<int64_t>::max() / lhs);
    return lhs * rhs;
}

__forceinline__ __device__ __host__ int64_t checked_align_i64(
        const int64_t value, const int64_t alignment) {
    EP_UNIFIED_ASSERT(value >= 0 and alignment > 0);
    const auto remainder = value % alignment;
    return remainder == 0 ? value : checked_add_i64(value, alignment - remainder);
}

struct HybridArenaLayout {
    int hidden;
    int num_topk;
    int proxy_capacity;
    void* base;

    int64_t control_offset;
    int64_t channel_count_offset;
    int64_t channel_count_bytes;
    int64_t proxy_ready_offset;
    int64_t proxy_ready_bytes;
    int64_t proxy_dispatch_offset;
    int64_t dispatch_token_bytes;
    int64_t proxy_rail_staging_offset;
    int64_t proxy_return_offset;
    int64_t combine_token_bytes;
    int64_t raw_bytes;
    int64_t arena_bytes;

    __forceinline__ __device__ __host__
    HybridArenaLayout(const int hidden,
                      const int num_topk,
                      const int proxy_capacity,
                      void* base = nullptr):
        hidden(hidden),
        num_topk(num_topk),
        proxy_capacity(proxy_capacity),
        base(base) {
        EP_UNIFIED_ASSERT(hidden > 0 and hidden % 256 == 0);
        EP_UNIFIED_ASSERT(hidden <= std::numeric_limits<int>::max() /
                                   static_cast<int>(sizeof(nv_bfloat16)));
        EP_UNIFIED_ASSERT(num_topk >= 1 and num_topk <= 32);
        EP_UNIFIED_ASSERT(proxy_capacity > 0);

        const auto dispatch_layout = layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, true);
        const auto combine_layout = layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, false);
        dispatch_token_bytes = dispatch_layout.get_num_bytes<false, int64_t>();
        combine_token_bytes = combine_layout.get_num_bytes<false, int64_t>();

        control_offset = 0;
        channel_count_offset = kHybridChannelCountOffsetBytes;
        channel_count_bytes = kHybridChannelCountBytes;
        proxy_ready_offset = kHybridProxyReadyOffsetBytes;
        proxy_ready_bytes = checked_mul_i64(
            proxy_capacity, static_cast<int64_t>(sizeof(int32_t)));
        proxy_dispatch_offset = checked_align_i64(
            checked_add_i64(proxy_ready_offset, proxy_ready_bytes),
            ptx::kNumTMAAlignBytes);
        proxy_rail_staging_offset = checked_align_i64(
            checked_add_i64(
                proxy_dispatch_offset,
                checked_mul_i64(proxy_capacity, dispatch_token_bytes)),
            ptx::kNumTMAAlignBytes);
        proxy_return_offset = checked_align_i64(
            checked_add_i64(
                proxy_rail_staging_offset,
                checked_mul_i64(proxy_capacity, dispatch_token_bytes)),
            ptx::kNumTMAAlignBytes);
        raw_bytes = checked_add_i64(
            proxy_return_offset,
            checked_mul_i64(proxy_capacity, combine_token_bytes));
        arena_bytes = checked_align_i64(
            raw_bytes, kNumHybridBufferAlignmentBytes);

        EP_UNIFIED_ASSERT(channel_count_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_ready_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_dispatch_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_rail_staging_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_return_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes % kNumHybridBufferAlignmentBytes == 0);
    }

    __forceinline__ __device__ __host__ HybridControl* get_control_ptr() const {
        return math::advance_ptr<HybridControl>(base, control_offset);
    }

    __forceinline__ __device__ __host__ int32_t* get_channel_count_ptr() const {
        return math::advance_ptr<int32_t>(base, channel_count_offset);
    }

    // Planning owns this region until Gate #2.  Committed dispatch then
    // reuses its inactive contents as per-proxy generation-ready words.
    __forceinline__ __device__ __host__ int32_t*
    get_proxy_ready_ptr(const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return math::advance_ptr<int32_t>(base, proxy_ready_offset) +
            proxy_slot;
    }

    __forceinline__ __device__ __host__ layout::TokenLayout
    get_proxy_dispatch_layout(const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, true,
            math::advance_ptr(base, checked_add_i64(
                proxy_dispatch_offset,
                checked_mul_i64(proxy_slot, dispatch_token_bytes))));
    }

    __forceinline__ __device__ __host__ layout::TokenLayout
    get_proxy_rail_staging_layout(const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, true,
            math::advance_ptr(base, checked_add_i64(
                proxy_rail_staging_offset,
                checked_mul_i64(proxy_slot, dispatch_token_bytes))));
    }

    __forceinline__ __device__ __host__ layout::TokenLayout
    get_proxy_return_layout(const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, false,
            math::advance_ptr(base, checked_add_i64(
                proxy_return_offset,
                checked_mul_i64(proxy_slot, combine_token_bytes))));
    }
};

}  // namespace deep_ep::elastic::rail_balance
