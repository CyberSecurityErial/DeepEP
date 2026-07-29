#pragma once

#include <cstddef>
#include <cstdint>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/rail_balance_layout.cuh>

namespace deep_ep::elastic::rail_balance {

constexpr int64_t kPublishSequenceAlignmentBytes = 32;
constexpr int64_t kProtocolControlAlignmentBytes = 64;
constexpr uint32_t kMaxPositiveInt32 = 0x7fffffffU;

enum class OneShotProtocolStatus : int {
    Idle = 0,
    Running = 1,
    Complete = 2,
    Failed = 3,
    Bypassed = 4,
};

enum class OneShotProtocolError : int {
    None = 0,
    ProducerTimeout = 1,
    ConsumerTimeout = 2,
    ReadyGenerationMismatch = 3,
    PublishSequenceMismatch = 4,
    DescriptorMismatch = 5,
    CanaryMismatch = 6,
    DuplicateConsumption = 7,
    TailRegression = 8,
};

static_assert(sizeof(int) == sizeof(uint32_t), "Protocol ABI requires 32-bit int");
static_assert(sizeof(OneShotProtocolStatus) == sizeof(int));
static_assert(sizeof(OneShotProtocolError) == sizeof(int));

__forceinline__ __device__ __host__ bool is_valid_protocol_generation(const int& generation) {
    return generation > 0;
}

__forceinline__ __device__ __host__ uint64_t make_publish_key(const int& generation,
                                                               const int& physical_slot) {
    EP_UNIFIED_ASSERT(is_valid_protocol_generation(generation));
    EP_UNIFIED_ASSERT(physical_slot >= 0);
    EP_UNIFIED_ASSERT(static_cast<uint32_t>(physical_slot) < kMaxPositiveInt32);
    return (static_cast<uint64_t>(static_cast<uint32_t>(generation)) << 32) |
           (static_cast<uint32_t>(physical_slot) + 1U);
}

__forceinline__ __device__ __host__ int get_publish_key_generation(const uint64_t& publish_key) {
    const auto generation = static_cast<uint32_t>(publish_key >> 32);
    EP_UNIFIED_ASSERT(generation > 0 and generation <= kMaxPositiveInt32);
    return static_cast<int>(generation);
}

__forceinline__ __device__ __host__ int get_publish_key_physical_slot(const uint64_t& publish_key) {
    const auto encoded_slot = static_cast<uint32_t>(publish_key);
    EP_UNIFIED_ASSERT(encoded_slot > 0 and encoded_slot <= kMaxPositiveInt32);
    return static_cast<int>(encoded_slot - 1U);
}

__forceinline__ __device__ __host__ uint64_t pack_generation_tail(const int& generation,
                                                                   const int& tail) {
    EP_UNIFIED_ASSERT(is_valid_protocol_generation(generation));
    EP_UNIFIED_ASSERT(tail >= 0);
    return (static_cast<uint64_t>(static_cast<uint32_t>(generation)) << 32) |
           static_cast<uint32_t>(tail);
}

__forceinline__ __device__ __host__ int get_packed_generation(const uint64_t& packed_generation_tail) {
    const auto generation = static_cast<uint32_t>(packed_generation_tail >> 32);
    EP_UNIFIED_ASSERT(generation > 0 and generation <= kMaxPositiveInt32);
    return static_cast<int>(generation);
}

__forceinline__ __device__ __host__ int get_packed_tail(const uint64_t& packed_generation_tail) {
    const auto tail = static_cast<uint32_t>(packed_generation_tail);
    EP_UNIFIED_ASSERT(tail <= kMaxPositiveInt32);
    return static_cast<int>(tail);
}

struct alignas(kProtocolControlAlignmentBytes) OneShotProxyControl {
    uint64_t published_tail;
    uint64_t consumed_tail;
    int generation;
    int expected_records;
    int status;
    int error_code;
    int error_slot;
    int first_hole_tail;
    int first_hole_ready_slot;
    int max_tail_gap;
    int hole_seen_generation;
    int trace_count;
    int consumer_start_count;
    int stale_observations;
};

static_assert(alignof(OneShotProxyControl) == kProtocolControlAlignmentBytes);
static_assert(sizeof(OneShotProxyControl) == kProtocolControlAlignmentBytes,
              "One-shot proxy control must remain a stable 64-byte ABI");
static_assert(offsetof(OneShotProxyControl, published_tail) == 0);
static_assert(offsetof(OneShotProxyControl, consumed_tail) == 8);
static_assert(offsetof(OneShotProxyControl, generation) == 16);
static_assert(offsetof(OneShotProxyControl, expected_records) == 20);
static_assert(offsetof(OneShotProxyControl, status) == 24);
static_assert(offsetof(OneShotProxyControl, error_code) == 28);
static_assert(offsetof(OneShotProxyControl, error_slot) == 32);
static_assert(offsetof(OneShotProxyControl, first_hole_tail) == 36);
static_assert(offsetof(OneShotProxyControl, first_hole_ready_slot) == 40);
static_assert(offsetof(OneShotProxyControl, max_tail_gap) == 44);
static_assert(offsetof(OneShotProxyControl, hole_seen_generation) == 48);
static_assert(offsetof(OneShotProxyControl, trace_count) == 52);
static_assert(offsetof(OneShotProxyControl, consumer_start_count) == 56);
static_assert(offsetof(OneShotProxyControl, stale_observations) == 60);

struct OneShotProtocolLayout {
    SourceShuffleLayout source_shuffle;
    int64_t publish_sequence_offset;
    int64_t control_offset;
    int64_t arena_bytes;
    void* base;

    __forceinline__ __device__ __host__
    OneShotProtocolLayout(const int& num_hidden_bytes,
                          const int& num_topk,
                          const int& physical_capacity,
                          void* base = nullptr):
        source_shuffle(num_hidden_bytes, num_topk, physical_capacity, base),
        publish_sequence_offset(math::align<int64_t>(
            source_shuffle.ready_offset +
                static_cast<int64_t>(physical_capacity) * sizeof(int),
            kPublishSequenceAlignmentBytes)),
        control_offset(math::align<int64_t>(
            publish_sequence_offset +
                static_cast<int64_t>(physical_capacity) * sizeof(uint64_t),
            kProtocolControlAlignmentBytes)),
        arena_bytes(math::align<int64_t>(
            control_offset + static_cast<int64_t>(sizeof(OneShotProxyControl)),
            kArenaAlignmentBytes)),
        base(base) {
        EP_UNIFIED_ASSERT(
            publish_sequence_offset >= source_shuffle.ready_offset +
                static_cast<int64_t>(physical_capacity) * sizeof(int));
        EP_UNIFIED_ASSERT(publish_sequence_offset % kPublishSequenceAlignmentBytes == 0);
        EP_UNIFIED_ASSERT(control_offset % kProtocolControlAlignmentBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes % kArenaAlignmentBytes == 0);
        EP_UNIFIED_ASSERT(arena_bytes >= source_shuffle.arena_bytes);
    }

    __forceinline__ __device__ __host__ const SourceShuffleLayout& get_source_shuffle_layout() const {
        return source_shuffle;
    }

    __forceinline__ __device__ __host__ uint64_t* get_publish_sequence_ptr(
        const int& physical_slot = 0) const {
        EP_UNIFIED_ASSERT(physical_slot >= 0 and
                          physical_slot < source_shuffle.physical_capacity);
        return math::advance_ptr<uint64_t>(base, publish_sequence_offset) + physical_slot;
    }

    __forceinline__ __device__ __host__ OneShotProxyControl* get_control_ptr() const {
        return math::advance_ptr<OneShotProxyControl>(base, control_offset);
    }
};

}  // namespace deep_ep::elastic::rail_balance
