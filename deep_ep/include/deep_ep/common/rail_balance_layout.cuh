#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::rail_balance {

constexpr int kNumManifestFields = 7;
constexpr int kNumCanaryBytes = 32;
constexpr int kNumDescriptorBytes = 64;
constexpr int64_t kArenaAlignmentBytes = 2097152;
constexpr uint32_t kHeadCanary = 0xDEADBEEF;
constexpr uint32_t kTailCanary = 0xC001D00D;

struct alignas(16) ProxyDescriptor {
    uint64_t fingerprint;
    int owner_rank;
    int owner_token;
    int destination;
    int owner_ordinal;
    int egress;
    int channel;
    int logical_slot;
    int physical_slot;
    int generation;
    int src_token_global_idx;
    int reserved[4];
};
static_assert(sizeof(ProxyDescriptor) == kNumDescriptorBytes,
              "Rail-balance proxy descriptor must remain a stable 64-byte ABI");
static_assert(offsetof(ProxyDescriptor, fingerprint) == 0);
static_assert(offsetof(ProxyDescriptor, owner_rank) == 8);
static_assert(offsetof(ProxyDescriptor, src_token_global_idx) == 44);
static_assert(offsetof(ProxyDescriptor, reserved) == 48);

struct SourceShuffleLayout {
    int num_hidden_bytes;
    int num_topk;
    int physical_capacity;
    int64_t token_bytes;
    int64_t record_bytes;
    int64_t ready_offset;
    int64_t arena_bytes;
    void* base;

    __forceinline__ __device__ __host__
    SourceShuffleLayout(const int& num_hidden_bytes,
                        const int& num_topk,
                        const int& physical_capacity,
                        void* base = nullptr):
        num_hidden_bytes(num_hidden_bytes),
        num_topk(num_topk),
        physical_capacity(physical_capacity),
        token_bytes(layout::TokenLayout(
            num_hidden_bytes, 0, num_topk, true).get_num_bytes<false, int64_t>()),
        record_bytes(math::align<int64_t>(
            kNumCanaryBytes + token_bytes + kNumDescriptorBytes + kNumCanaryBytes,
            ptx::kNumTMAAlignBytes)),
        ready_offset(record_bytes * physical_capacity),
        arena_bytes(math::align<int64_t>(
            ready_offset + math::align<int64_t>(
                static_cast<int64_t>(physical_capacity) * sizeof(int),
                ptx::kNumTMAAlignBytes),
            kArenaAlignmentBytes)),
        base(base) {
        EP_UNIFIED_ASSERT(num_hidden_bytes > 0);
        EP_UNIFIED_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_UNIFIED_ASSERT(physical_capacity > 0);
        EP_UNIFIED_ASSERT(record_bytes % ptx::kNumTMAAlignBytes == 0);
    }

    __forceinline__ __device__ __host__ void* get_record_ptr(const int& physical_slot) const {
        EP_UNIFIED_ASSERT(physical_slot >= 0 and physical_slot < physical_capacity);
        return math::advance_ptr(base, record_bytes * physical_slot);
    }

    __forceinline__ __device__ __host__ uint32_t* get_head_canary_ptr(const int& physical_slot) const {
        return static_cast<uint32_t*>(get_record_ptr(physical_slot));
    }

    __forceinline__ __device__ __host__ layout::TokenLayout get_token_layout(const int& physical_slot) const {
        return layout::TokenLayout(
            num_hidden_bytes, 0, num_topk, true,
            math::advance_ptr(get_record_ptr(physical_slot), kNumCanaryBytes));
    }

    __forceinline__ __device__ __host__ ProxyDescriptor* get_descriptor_ptr(const int& physical_slot) const {
        return math::advance_ptr<ProxyDescriptor>(
            get_record_ptr(physical_slot), kNumCanaryBytes + token_bytes);
    }

    __forceinline__ __device__ __host__ uint32_t* get_tail_canary_ptr(const int& physical_slot) const {
        return math::advance_ptr<uint32_t>(
            get_descriptor_ptr(physical_slot), kNumDescriptorBytes);
    }

    __forceinline__ __device__ __host__ int* get_ready_ptr(const int& physical_slot = 0) const {
        EP_UNIFIED_ASSERT(physical_slot >= 0 and physical_slot < physical_capacity);
        return math::advance_ptr<int>(base, ready_offset) + physical_slot;
    }

    __forceinline__ __device__ __host__ int64_t get_smem_bytes() const {
        return record_bytes + math::align<int64_t>(
            sizeof(ptx::mbarrier), ptx::kNumTMAAlignBytes);
    }
};

}  // namespace deep_ep::elastic::rail_balance
