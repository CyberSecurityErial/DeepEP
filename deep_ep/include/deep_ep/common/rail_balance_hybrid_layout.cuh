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
    int64_t proxy_dispatch_offset;
    int64_t dispatch_token_bytes;
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
        channel_count_offset = checked_align_i64(
            sizeof(HybridControl), ptx::kNumTMAAlignBytes);
        channel_count_bytes = checked_mul_i64(
            checked_mul_i64(kNumHybridMaxChannels,
                            kNumHybridMaxDestinations),
            sizeof(int32_t));
        proxy_dispatch_offset = checked_align_i64(
            checked_add_i64(channel_count_offset, channel_count_bytes),
            ptx::kNumTMAAlignBytes);
        proxy_return_offset = checked_align_i64(
            checked_add_i64(
                proxy_dispatch_offset,
                checked_mul_i64(proxy_capacity, dispatch_token_bytes)),
            ptx::kNumTMAAlignBytes);
        raw_bytes = checked_add_i64(
            proxy_return_offset,
            checked_mul_i64(proxy_capacity, combine_token_bytes));
        arena_bytes = checked_align_i64(
            raw_bytes, kNumHybridBufferAlignmentBytes);

        EP_UNIFIED_ASSERT(channel_count_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_dispatch_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(proxy_return_offset % ptx::kNumTMAAlignBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes % kNumHybridBufferAlignmentBytes == 0);
    }

    __forceinline__ __device__ __host__ HybridControl* get_control_ptr() const {
        return math::advance_ptr<HybridControl>(base, control_offset);
    }

    __forceinline__ __device__ __host__ int32_t* get_channel_count_ptr() const {
        return math::advance_ptr<int32_t>(base, channel_count_offset);
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
