#include "balance/rail_balance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

namespace balance {
namespace {

constexpr std::int32_t kVectorTileBytes = 16 * 1024;
constexpr std::int32_t kTmaTileBytes = 16 * 1024;
constexpr std::size_t kWorkspaceAlignment = 256;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct DeviceCopyParams {
  const std::uint8_t* source[kNumRails];
  std::uint8_t* destination[kNumRails];
  const Span* spans;
  const MetadataHeader* header;
  std::int32_t num_spans;
  std::int32_t record_bytes;
  std::int32_t workers_per_span;
};

struct CapacityArray {
  std::int64_t value[kNumRails];
};

__device__ __forceinline__ void set_error(MetadataHeader* header,
                                           Status status) {
  atomicCAS(&header->status,
            static_cast<std::int32_t>(Status::kSuccess),
            static_cast<std::int32_t>(status));
}

__global__ __launch_bounds__(32, 1) void make_plan_kernel(
    const std::int32_t* __restrict__ all_counts,
    CapacityArray input_capacities,
    std::int32_t rail,
    std::int32_t num_partitions,
    std::int32_t record_bytes,
    std::int32_t remainder_seed,
    bool zero_padding,
    std::int64_t output_capacity,
    MetadataHeader* __restrict__ header,
    PartitionInfo* __restrict__ partitions,
    Span* __restrict__ spans) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  header->magic = kMetadataMagic;
  header->version = kMetadataVersion;
  header->status = static_cast<std::int32_t>(Status::kSuccess);
  header->rail = rail;
  header->num_partitions = num_partitions;
  header->record_bytes = record_bytes;
  header->remainder_seed = remainder_seed;
  header->reserved0 = 0;
  header->total_output_records = 0;
  header->total_valid_records = 0;
  header->local_input_records = 0;
  header->num_spans = 0;

  std::int64_t source_prefix[kNumRails] = {};
  std::int64_t output_prefix = 0;
  std::int64_t local_valid_total = 0;
  std::int64_t emitted_spans = 0;

  for (std::int32_t partition = 0; partition < num_partitions; ++partition) {
    std::int32_t count[kNumRails];
    std::int32_t quota[kNumRails];
    std::int32_t surplus[kNumRails];
    std::int32_t deficit[kNumRails];
    std::int32_t owner_consumed[kNumRails] = {};
    std::int64_t total = 0;

#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      const std::int32_t value = all_counts[
          static_cast<std::int64_t>(r) * num_partitions + partition];
      if (value < 0) {
        set_error(header, Status::kInvalidCount);
        count[r] = 0;
      } else {
        count[r] = value;
        total += value;
      }
    }

    const std::int64_t base64 = total / kNumRails;
    const std::int32_t remainder = static_cast<std::int32_t>(total % kNumRails);
    if (base64 > INT_MAX) {
      set_error(header, Status::kInvalidCount);
    }
    const std::int32_t base = static_cast<std::int32_t>(base64);
    const std::int32_t start =
        ((remainder_seed % kNumRails) + kNumRails + partition % kNumRails) %
        kNumRails;

#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      quota[r] = base;
    }

    // Prefer assigning remainder records to rails which already own one.  It
    // is the exact minimum-move tie-break used by the previous implementation.
    std::int32_t assigned = 0;
#pragma unroll
    for (std::int32_t offset = 0; offset < kNumRails; ++offset) {
      const std::int32_t r = (start + offset) % kNumRails;
      if (assigned < remainder && count[r] > base) {
        ++quota[r];
        ++assigned;
      }
    }
#pragma unroll
    for (std::int32_t offset = 0; offset < kNumRails; ++offset) {
      const std::int32_t r = (start + offset) % kNumRails;
      if (assigned < remainder && count[r] <= base) {
        ++quota[r];
        ++assigned;
      }
    }

#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      surplus[r] = count[r] > quota[r] ? count[r] - quota[r] : 0;
      deficit[r] = quota[r] > count[r] ? quota[r] - count[r] : 0;
    }

    const std::int32_t physical_count = base + (remainder != 0);
    const std::int32_t span_begin = partition * kMaxSpansPerPartition;
    std::int32_t span_count = 0;
    std::int32_t destination_written = 0;

    // Retained records stay first in the local partition.
    const std::int32_t retained = min(count[rail], quota[rail]);
    if (retained > 0) {
      spans[span_begin + span_count++] = Span{
          source_prefix[rail], output_prefix, retained,
          static_cast<std::int16_t>(rail),
          static_cast<std::int16_t>(partition), kSpanData};
      destination_written += retained;
    }

    // Canonically pair global surplus and deficit streams.  Every rail runs
    // the same tiny plan and only emits segments whose egress is itself.
    std::int32_t owner = 0;
    std::int32_t egress = 0;
    while (owner < kNumRails && egress < kNumRails) {
      while (owner < kNumRails && surplus[owner] == 0) {
        ++owner;
      }
      while (egress < kNumRails && deficit[egress] == 0) {
        ++egress;
      }
      if (owner == kNumRails || egress == kNumRails) {
        break;
      }
      const std::int32_t amount = min(surplus[owner], deficit[egress]);
      if (egress == rail && amount > 0) {
        if (span_count >= kNumRails) {
          set_error(header, Status::kSpanCapacityExceeded);
        } else {
          spans[span_begin + span_count++] = Span{
              source_prefix[owner] + quota[owner] + owner_consumed[owner],
              output_prefix + destination_written,
              amount,
              static_cast<std::int16_t>(owner),
              static_cast<std::int16_t>(partition),
              kSpanData};
          destination_written += amount;
        }
      }
      surplus[owner] -= amount;
      deficit[egress] -= amount;
      owner_consumed[owner] += amount;
    }

    if (destination_written != quota[rail]) {
      set_error(header, Status::kInternalError);
    }

    const std::int32_t padding = physical_count - quota[rail];
    if (zero_padding && padding > 0) {
      if (span_count >= kMaxSpansPerPartition) {
        set_error(header, Status::kSpanCapacityExceeded);
      } else {
        spans[span_begin + span_count++] = Span{
            0, output_prefix + quota[rail], padding,
            static_cast<std::int16_t>(-1),
            static_cast<std::int16_t>(partition), kSpanPadding};
      }
    }

    for (std::int32_t slot = span_count;
         slot < kMaxSpansPerPartition;
         ++slot) {
      spans[span_begin + slot] = Span{
          0, 0, 0, static_cast<std::int16_t>(-1),
          static_cast<std::int16_t>(partition), kSpanData};
    }

    partitions[partition] = PartitionInfo{
        output_prefix, total, physical_count, quota[rail], span_begin,
        span_count};
    output_prefix += physical_count;
    local_valid_total += quota[rail];
    emitted_spans += span_count;

#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      source_prefix[r] += max(count[r], 0);
    }
  }

#pragma unroll
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    if (source_prefix[r] > input_capacities.value[r]) {
      set_error(header, Status::kInputCapacityExceeded);
    }
  }
  if (output_prefix > output_capacity) {
    set_error(header, Status::kOutputCapacityExceeded);
  }

  header->total_output_records = output_prefix;
  header->total_valid_records = local_valid_total;
  header->local_input_records = source_prefix[rail];
  header->num_spans = emitted_spans;
}

// The production topology has eight partitions.  Planning them serially with
// one GPU thread costs ~16 us beyond launch on H200.  This fast path maps one
// partition to one lane, shares the tiny [partition, rail] count tile in SMEM,
// and keeps the exact canonical quota/span ordering of the serial fallback.
// One warp covers up to 32 partitions without another kernel or global scan.
__global__ __launch_bounds__(32, 1) void make_plan_warp_kernel(
    const std::int32_t* __restrict__ all_counts,
    CapacityArray input_capacities,
    std::int32_t rail,
    std::int32_t num_partitions,
    std::int32_t record_bytes,
    std::int32_t remainder_seed,
    bool zero_padding,
    std::int64_t output_capacity,
    MetadataHeader* __restrict__ header,
    PartitionInfo* __restrict__ partitions,
    Span* __restrict__ spans) {
  __shared__ std::int32_t shared_counts[32][kNumRails];
  __shared__ std::int32_t shared_physical[32];
  const std::int32_t partition = static_cast<std::int32_t>(threadIdx.x);
  const bool active = partition < num_partitions;

  if (partition == 0) {
    header->magic = kMetadataMagic;
    header->version = kMetadataVersion;
    header->status = static_cast<std::int32_t>(Status::kSuccess);
    header->rail = rail;
    header->num_partitions = num_partitions;
    header->record_bytes = record_bytes;
    header->remainder_seed = remainder_seed;
    header->reserved0 = 0;
    header->total_output_records = 0;
    header->total_valid_records = 0;
    header->local_input_records = 0;
    header->num_spans = 0;
  }
  __syncwarp();

  std::int32_t count[kNumRails] = {};
  std::int64_t total = 0;
#pragma unroll
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    std::int32_t value = 0;
    if (active) {
      value = all_counts[
          static_cast<std::int64_t>(r) * num_partitions + partition];
      if (value < 0) {
        set_error(header, Status::kInvalidCount);
        value = 0;
      }
    }
    count[r] = value;
    shared_counts[partition][r] = value;
    total += value;
  }

  const std::int64_t base64 = total / kNumRails;
  if (base64 > INT_MAX) {
    set_error(header, Status::kInvalidCount);
  }
  const std::int32_t base = static_cast<std::int32_t>(base64);
  const std::int32_t remainder = static_cast<std::int32_t>(total % kNumRails);
  shared_physical[partition] = active ? base + (remainder != 0) : 0;
  __syncwarp();

  if (active) {
    std::int64_t source_prefix[kNumRails] = {};
#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      std::int64_t prefix = 0;
      for (std::int32_t previous = 0; previous < partition; ++previous) {
        prefix += shared_counts[previous][r];
      }
      source_prefix[r] = prefix;
    }
    std::int64_t output_prefix = 0;
    for (std::int32_t previous = 0; previous < partition; ++previous) {
      output_prefix += shared_physical[previous];
    }

    std::int32_t quota[kNumRails];
    std::int32_t surplus[kNumRails];
    std::int32_t deficit[kNumRails];
    std::int32_t owner_consumed[kNumRails] = {};
#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      quota[r] = base;
    }
    const std::int32_t start =
        ((remainder_seed % kNumRails) + kNumRails +
         partition % kNumRails) % kNumRails;
    std::int32_t assigned = 0;
#pragma unroll
    for (std::int32_t offset = 0; offset < kNumRails; ++offset) {
      const std::int32_t r = (start + offset) % kNumRails;
      if (assigned < remainder && count[r] > base) {
        ++quota[r];
        ++assigned;
      }
    }
#pragma unroll
    for (std::int32_t offset = 0; offset < kNumRails; ++offset) {
      const std::int32_t r = (start + offset) % kNumRails;
      if (assigned < remainder && count[r] <= base) {
        ++quota[r];
        ++assigned;
      }
    }
#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      surplus[r] = count[r] > quota[r] ? count[r] - quota[r] : 0;
      deficit[r] = quota[r] > count[r] ? quota[r] - count[r] : 0;
    }

    const std::int32_t physical_count = base + (remainder != 0);
    const std::int32_t span_begin = partition * kMaxSpansPerPartition;
    std::int32_t span_count = 0;
    std::int32_t destination_written = 0;
    const std::int32_t retained = min(count[rail], quota[rail]);
    if (retained > 0) {
      spans[span_begin + span_count++] = Span{
          source_prefix[rail], output_prefix, retained,
          static_cast<std::int16_t>(rail),
          static_cast<std::int16_t>(partition), kSpanData};
      destination_written += retained;
    }

    std::int32_t owner = 0;
    std::int32_t egress = 0;
    while (owner < kNumRails && egress < kNumRails) {
      while (owner < kNumRails && surplus[owner] == 0) ++owner;
      while (egress < kNumRails && deficit[egress] == 0) ++egress;
      if (owner == kNumRails || egress == kNumRails) break;
      const std::int32_t amount = min(surplus[owner], deficit[egress]);
      if (egress == rail && amount > 0) {
        if (span_count >= kNumRails) {
          set_error(header, Status::kSpanCapacityExceeded);
        } else {
          spans[span_begin + span_count++] = Span{
              source_prefix[owner] + quota[owner] + owner_consumed[owner],
              output_prefix + destination_written, amount,
              static_cast<std::int16_t>(owner),
              static_cast<std::int16_t>(partition), kSpanData};
          destination_written += amount;
        }
      }
      surplus[owner] -= amount;
      deficit[egress] -= amount;
      owner_consumed[owner] += amount;
    }
    if (destination_written != quota[rail]) {
      set_error(header, Status::kInternalError);
    }

    const std::int32_t padding = physical_count - quota[rail];
    if (zero_padding && padding > 0) {
      if (span_count >= kMaxSpansPerPartition) {
        set_error(header, Status::kSpanCapacityExceeded);
      } else {
        spans[span_begin + span_count++] = Span{
            0, output_prefix + quota[rail], padding,
            static_cast<std::int16_t>(-1),
            static_cast<std::int16_t>(partition), kSpanPadding};
      }
    }
    for (std::int32_t slot = span_count;
         slot < kMaxSpansPerPartition;
         ++slot) {
      spans[span_begin + slot] = Span{
          0, 0, 0, static_cast<std::int16_t>(-1),
          static_cast<std::int16_t>(partition), kSpanData};
    }
    partitions[partition] = PartitionInfo{
        output_prefix, total, physical_count, quota[rail], span_begin,
        span_count};
  }
  __syncwarp();

  if (partition == 0) {
    std::int64_t total_output = 0;
    std::int64_t total_valid = 0;
    std::int64_t emitted_spans = 0;
    std::int64_t input_totals[kNumRails] = {};
    for (std::int32_t p = 0; p < num_partitions; ++p) {
      total_output += partitions[p].physical_count;
      total_valid += partitions[p].local_valid_count;
      emitted_spans += partitions[p].span_count;
#pragma unroll
      for (std::int32_t r = 0; r < kNumRails; ++r) {
        input_totals[r] += shared_counts[p][r];
      }
    }
#pragma unroll
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      if (input_totals[r] > input_capacities.value[r]) {
        set_error(header, Status::kInputCapacityExceeded);
      }
    }
    if (total_output > output_capacity) {
      set_error(header, Status::kOutputCapacityExceeded);
    }
    header->total_output_records = total_output;
    header->total_valid_records = total_valid;
    header->local_input_records = input_totals[rail];
    header->num_spans = emitted_spans;
  }
}

template <bool kReverse>
__global__ __launch_bounds__(256, 2) void vector_copy_kernel(
    DeviceCopyParams params) {
  if (params.header->status != static_cast<std::int32_t>(Status::kSuccess)) {
    return;
  }
  const std::int32_t span_index = static_cast<std::int32_t>(blockIdx.x);
  const std::int32_t worker = static_cast<std::int32_t>(blockIdx.y);
  const Span span = params.spans[span_index];
  if (span.record_count <= 0 || (kReverse && (span.flags & kSpanPadding))) {
    return;
  }

  const std::int64_t bytes =
      static_cast<std::int64_t>(span.record_count) * params.record_bytes;
  const bool padding = (span.flags & kSpanPadding) != 0;
  const std::uint8_t* source = nullptr;
  std::uint8_t* destination = nullptr;
  if constexpr (kReverse) {
    source = params.source[0] + span.dst_record * params.record_bytes;
    destination = params.destination[span.src_rail] +
                  span.src_record * params.record_bytes;
  } else {
    if (!padding) {
      source = params.source[span.src_rail] +
               span.src_record * params.record_bytes;
    }
    destination = params.destination[0] +
                  span.dst_record * params.record_bytes;
    if (padding) {
      source = destination;
    }
  }

  for (std::int64_t tile =
           static_cast<std::int64_t>(worker) * kVectorTileBytes;
       tile < bytes;
       tile += static_cast<std::int64_t>(params.workers_per_span) *
               kVectorTileBytes) {
    const std::int64_t remaining = bytes - tile;
    const std::int32_t tile_bytes = static_cast<std::int32_t>(
        remaining < kVectorTileBytes ? remaining : kVectorTileBytes);
    const std::uintptr_t src_address =
        reinterpret_cast<std::uintptr_t>(source == nullptr ? destination + tile
                                                            : source + tile);
    const std::uintptr_t dst_address =
        reinterpret_cast<std::uintptr_t>(destination + tile);
    const bool vector_aligned =
        ((src_address | dst_address | static_cast<std::uintptr_t>(tile_bytes)) &
         15u) == 0;
    if (vector_aligned) {
      const auto* src4 = reinterpret_cast<const uint4*>(source + tile);
      auto* dst4 = reinterpret_cast<uint4*>(destination + tile);
      for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x);
           index < tile_bytes / 16;
           index += static_cast<std::int32_t>(blockDim.x)) {
        dst4[index] = padding ? make_uint4(0, 0, 0, 0) : src4[index];
      }
    } else {
      for (std::int32_t index = static_cast<std::int32_t>(threadIdx.x);
           index < tile_bytes;
           index += static_cast<std::int32_t>(blockDim.x)) {
        destination[tile + index] = padding ? 0 : source[tile + index];
      }
    }
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ std::uint32_t shared_address(const void* pointer) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(std::uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
               "r"(shared_address(barrier)));
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_expect(std::uint64_t* barrier,
                                                std::int32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::
      "r"(bytes), "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void mbarrier_wait(std::uint64_t* barrier,
                                              std::uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred done;\n"
      "WAIT%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, 10000000;\n"
      "@!done bra WAIT%=;\n"
      "}" ::
      "r"(shared_address(barrier)), "r"(phase) : "memory");
}

__device__ __forceinline__ void tma_load(void* shared,
                                         const void* global,
                                         std::int32_t bytes,
                                         std::uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::
      "r"(shared_address(shared)), "l"(global), "r"(bytes),
      "r"(shared_address(barrier)) : "memory");
}

__device__ __forceinline__ void tma_store_release_stage(
    void* global, const void* shared, std::int32_t bytes) {
  asm volatile(
      "fence.proxy.async.shared::cta;\n"
      "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
      "cp.async.bulk.commit_group;\n"
      // The next load only needs the TMA engine to have consumed this shared
      // stage.  Destination-global completion is drained once at CTA exit,
      // allowing a store to overlap the following peer load.
      "cp.async.bulk.wait_group.read 0;" ::
      "l"(global), "r"(shared_address(shared)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_store_wait_all() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_invalidate(std::uint64_t* barrier) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::
               "r"(shared_address(barrier)) : "memory");
}
#endif

template <bool kReverse>
__global__ __launch_bounds__(32, 2) void tma_copy_kernel(
    DeviceCopyParams params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(16) std::uint8_t shared_storage[];
  auto* stage = shared_storage;
  auto* barrier = reinterpret_cast<std::uint64_t*>(
      shared_storage + kTmaTileBytes);
  const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
  if (lane == 0) {
    mbarrier_init(barrier);
  }
  __syncwarp();

  if (params.header->status == static_cast<std::int32_t>(Status::kSuccess)) {
    const std::int32_t span_index = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t worker = static_cast<std::int32_t>(blockIdx.y);
    const Span span = params.spans[span_index];
    if (span.record_count > 0 && !(kReverse && (span.flags & kSpanPadding))) {
      const std::int64_t bytes =
          static_cast<std::int64_t>(span.record_count) * params.record_bytes;
      const bool padding = (span.flags & kSpanPadding) != 0;
      const std::uint8_t* source = nullptr;
      std::uint8_t* destination = nullptr;
      if constexpr (kReverse) {
        source = params.source[0] + span.dst_record * params.record_bytes;
        destination = params.destination[span.src_rail] +
                      span.src_record * params.record_bytes;
      } else {
        if (!padding) {
          source = params.source[span.src_rail] +
                   span.src_record * params.record_bytes;
        }
        destination = params.destination[0] +
                      span.dst_record * params.record_bytes;
      }

      std::uint32_t phase = 0;
      for (std::int64_t tile =
               static_cast<std::int64_t>(worker) * kTmaTileBytes;
           tile < bytes;
           tile += static_cast<std::int64_t>(params.workers_per_span) *
                   kTmaTileBytes) {
        const std::int64_t remaining = bytes - tile;
        const std::int32_t tile_bytes = static_cast<std::int32_t>(
            remaining < kTmaTileBytes ? remaining : kTmaTileBytes);
        if (padding) {
          auto* stage4 = reinterpret_cast<uint4*>(stage);
          for (std::int32_t index = lane; index < tile_bytes / 16;
               index += 32) {
            stage4[index] = make_uint4(0, 0, 0, 0);
          }
          __syncwarp();
          if (lane == 0) {
            tma_store_release_stage(destination + tile, stage, tile_bytes);
          }
          __syncwarp();
        } else if (lane == 0) {
          mbarrier_expect(barrier, tile_bytes);
          tma_load(stage, source + tile, tile_bytes, barrier);
          mbarrier_wait(barrier, phase);
          phase ^= 1;
          tma_store_release_stage(destination + tile, stage, tile_bytes);
        }
      }
    }
  }
  __syncwarp();
  if (lane == 0) {
    tma_store_wait_all();
    mbarrier_invalidate(barrier);
  }
#else
  (void)params;
#endif
}

cudaError_t validate_common(std::int32_t rail,
                            std::int32_t num_partitions,
                            std::int32_t record_bytes,
                            const void* workspace,
                            std::size_t supplied_workspace) {
  if (rail < 0 || rail >= kNumRails || num_partitions <= 0 ||
      num_partitions > std::numeric_limits<std::int16_t>::max() ||
      record_bytes <= 0 || workspace == nullptr ||
      supplied_workspace < workspace_size(num_partitions) ||
      (reinterpret_cast<std::uintptr_t>(workspace) & 15u) != 0) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

std::int32_t workers_per_span(std::int64_t max_records,
                              std::int32_t record_bytes,
                              std::int32_t tile_bytes,
                              std::int32_t max_workers) {
  if (max_records <= 0 || max_workers <= 0) {
    return 0;
  }
  const unsigned long long records =
      static_cast<unsigned long long>(max_records);
  const unsigned long long bytes = records *
      static_cast<unsigned long long>(record_bytes);
  if (records != 0 && bytes / records !=
                          static_cast<unsigned long long>(record_bytes)) {
    return 0;
  }
  const unsigned long long tiles =
      (bytes + static_cast<unsigned long long>(tile_bytes) - 1) /
      static_cast<unsigned long long>(tile_bytes);
  return static_cast<std::int32_t>(
      std::max<unsigned long long>(
          1, std::min<unsigned long long>(tiles, max_workers)));
}

template <bool kReverse>
cudaError_t launch_copy(const DeviceCopyParams& params,
                        CopyEngine engine,
                        cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned>(params.num_spans),
                  static_cast<unsigned>(params.workers_per_span), 1);
  if (engine == CopyEngine::kTma) {
    constexpr std::size_t dynamic_shared = kTmaTileBytes + 16;
    tma_copy_kernel<kReverse><<<grid, 32, dynamic_shared, stream>>>(params);
  } else {
    vector_copy_kernel<kReverse><<<grid, 256, 0, stream>>>(params);
  }
  return cudaGetLastError();
}

}  // namespace

std::size_t workspace_size(std::int32_t num_partitions) {
  if (num_partitions <= 0) {
    return 0;
  }
  const std::size_t header_bytes = align_up(sizeof(MetadataHeader), 16);
  const std::size_t partition_bytes = align_up(
      static_cast<std::size_t>(num_partitions) * sizeof(PartitionInfo), 16);
  const std::size_t span_bytes =
      static_cast<std::size_t>(num_partitions) * kMaxSpansPerPartition *
      sizeof(Span);
  return align_up(header_bytes + partition_bytes + span_bytes,
                  kWorkspaceAlignment);
}

MetadataView metadata_view(void* workspace, std::int32_t num_partitions) {
  auto* base = static_cast<std::uint8_t*>(workspace);
  const std::size_t header_bytes = align_up(sizeof(MetadataHeader), 16);
  const std::size_t partition_bytes = align_up(
      static_cast<std::size_t>(num_partitions) * sizeof(PartitionInfo), 16);
  return MetadataView{
      reinterpret_cast<MetadataHeader*>(base),
      reinterpret_cast<PartitionInfo*>(base + header_bytes),
      reinterpret_cast<Span*>(base + header_bytes + partition_bytes)};
}

ConstMetadataView metadata_view(const void* workspace,
                                std::int32_t num_partitions) {
  const auto mutable_view =
      metadata_view(const_cast<void*>(workspace), num_partitions);
  return ConstMetadataView{
      mutable_view.header, mutable_view.partitions, mutable_view.spans};
}

std::int64_t max_output_records(std::int64_t total_input_capacity,
                                std::int32_t num_partitions) {
  if (total_input_capacity < 0 || num_partitions <= 0) {
    return 0;
  }
  // sum(ceil(T[p]/8)) <= floor((sum(T[p]) + 7*P) / 8).
  if (total_input_capacity >
      std::numeric_limits<std::int64_t>::max() -
          static_cast<std::int64_t>(7) * num_partitions) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return (total_input_capacity + static_cast<std::int64_t>(7) *
                                     num_partitions) /
         kNumRails;
}

cudaError_t select_copy_engine(CopyEngine requested,
                               std::int32_t record_bytes,
                               std::int32_t tma_threshold_bytes,
                               CopyEngine* selected) {
  if (selected == nullptr || record_bytes <= 0 || tma_threshold_bytes <= 0) {
    return cudaErrorInvalidValue;
  }
  if (requested == CopyEngine::kVector ||
      (requested == CopyEngine::kAuto &&
       (record_bytes < tma_threshold_bytes || record_bytes % 16 != 0))) {
    *selected = CopyEngine::kVector;
    return cudaSuccess;
  }
  if (requested != CopyEngine::kAuto && requested != CopyEngine::kTma) {
    return cudaErrorInvalidValue;
  }
  int device = 0;
  int major = 0;
  cudaError_t result = cudaGetDevice(&device);
  if (result != cudaSuccess) {
    return result;
  }
  result = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                  device);
  if (result != cudaSuccess) {
    return result;
  }
  const bool tma_eligible =
      major >= 9 && record_bytes % 16 == 0;
  if (requested == CopyEngine::kTma) {
    if (!tma_eligible) {
      return cudaErrorNotSupported;
    }
    *selected = CopyEngine::kTma;
  } else if (requested == CopyEngine::kAuto) {
    *selected = tma_eligible && record_bytes >= tma_threshold_bytes
                    ? CopyEngine::kTma
                    : CopyEngine::kVector;
  }
  return cudaSuccess;
}

cudaError_t launch_balance(const BalanceParams& params,
                           const Options& options,
                           cudaStream_t stream) {
  cudaError_t result = validate_common(
      params.rail, params.num_partitions, params.record_bytes,
      params.workspace, params.workspace_bytes);
  if (result != cudaSuccess || params.all_counts == nullptr ||
      params.output == nullptr || params.output_capacity_records < 0 ||
      params.max_input_records <= 0 || options.max_blocks_per_span <= 0 ||
      options.max_blocks_per_span > 65535) {
    return result == cudaSuccess ? cudaErrorInvalidValue : result;
  }
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    if (params.peer_input[r] == nullptr ||
        params.input_capacity_records[r] < 0 ||
        params.max_input_records < params.input_capacity_records[r]) {
      return cudaErrorInvalidValue;
    }
  }

  CopyEngine engine;
  result = select_copy_engine(options.engine, params.record_bytes,
                              options.tma_threshold_bytes, &engine);
  if (result != cudaSuccess) {
    return result;
  }
  if (engine == CopyEngine::kTma) {
    std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(params.output);
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      alignment |= reinterpret_cast<std::uintptr_t>(params.peer_input[r]);
    }
    if ((alignment & 15u) != 0) {
      if (options.engine == CopyEngine::kTma) {
        return cudaErrorInvalidValue;
      }
      engine = CopyEngine::kVector;
    }
  }

  const std::int32_t workers = workers_per_span(
      params.max_input_records, params.record_bytes,
      engine == CopyEngine::kTma ? kTmaTileBytes : kVectorTileBytes,
      options.max_blocks_per_span);
  if (workers <= 0) {
    return cudaErrorInvalidValue;
  }

  auto metadata = metadata_view(params.workspace, params.num_partitions);
  CapacityArray capacities{};
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    capacities.value[r] = params.input_capacity_records[r];
  }

  if (params.num_partitions <= 32) {
    make_plan_warp_kernel<<<1, 32, 0, stream>>>(
        params.all_counts, capacities, params.rail, params.num_partitions,
        params.record_bytes, options.remainder_seed, options.zero_padding,
        params.output_capacity_records, metadata.header, metadata.partitions,
        metadata.spans);
  } else {
    make_plan_kernel<<<1, 32, 0, stream>>>(
        params.all_counts, capacities, params.rail, params.num_partitions,
        params.record_bytes, options.remainder_seed, options.zero_padding,
        params.output_capacity_records, metadata.header, metadata.partitions,
        metadata.spans);
  }
  result = cudaGetLastError();
  if (result != cudaSuccess) {
    return result;
  }

  DeviceCopyParams copy{};
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    copy.source[r] = static_cast<const std::uint8_t*>(params.peer_input[r]);
  }
  copy.destination[0] = static_cast<std::uint8_t*>(params.output);
  copy.spans = metadata.spans;
  copy.header = metadata.header;
  copy.num_spans = params.num_partitions * kMaxSpansPerPartition;
  copy.record_bytes = params.record_bytes;
  copy.workers_per_span = workers;
  return engine == CopyEngine::kTma
             ? launch_copy<false>(copy, CopyEngine::kTma, stream)
             : launch_copy<false>(copy, CopyEngine::kVector, stream);
}

cudaError_t launch_unbalance(const UnbalanceParams& params,
                             const Options& options,
                             cudaStream_t stream) {
  cudaError_t result = validate_common(
      params.rail, params.num_partitions, params.record_bytes,
      params.workspace, params.workspace_bytes);
  if (result != cudaSuccess || params.input == nullptr ||
      params.max_input_records <= 0 || options.max_blocks_per_span <= 0 ||
      options.max_blocks_per_span > 65535) {
    return result == cudaSuccess ? cudaErrorInvalidValue : result;
  }
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    if (params.peer_output[r] == nullptr) {
      return cudaErrorInvalidValue;
    }
  }

  CopyEngine engine;
  result = select_copy_engine(options.engine, params.record_bytes,
                              options.tma_threshold_bytes, &engine);
  if (result != cudaSuccess) {
    return result;
  }
  if (engine == CopyEngine::kTma) {
    std::uintptr_t alignment = reinterpret_cast<std::uintptr_t>(params.input);
    for (std::int32_t r = 0; r < kNumRails; ++r) {
      alignment |= reinterpret_cast<std::uintptr_t>(params.peer_output[r]);
    }
    if ((alignment & 15u) != 0) {
      if (options.engine == CopyEngine::kTma) {
        return cudaErrorInvalidValue;
      }
      engine = CopyEngine::kVector;
    }
  }

  const std::int32_t workers = workers_per_span(
      params.max_input_records, params.record_bytes,
      engine == CopyEngine::kTma ? kTmaTileBytes : kVectorTileBytes,
      options.max_blocks_per_span);
  if (workers <= 0) {
    return cudaErrorInvalidValue;
  }

  const auto metadata = metadata_view(params.workspace,
                                      params.num_partitions);
  DeviceCopyParams copy{};
  copy.source[0] = static_cast<const std::uint8_t*>(params.input);
  for (std::int32_t r = 0; r < kNumRails; ++r) {
    copy.destination[r] = static_cast<std::uint8_t*>(params.peer_output[r]);
  }
  copy.spans = metadata.spans;
  copy.header = metadata.header;
  copy.num_spans = params.num_partitions * kMaxSpansPerPartition;
  copy.record_bytes = params.record_bytes;
  copy.workers_per_span = workers;
  return engine == CopyEngine::kTma
             ? launch_copy<true>(copy, CopyEngine::kTma, stream)
             : launch_copy<true>(copy, CopyEngine::kVector, stream);
}

const char* status_string(Status status) {
  switch (status) {
    case Status::kSuccess: return "success";
    case Status::kInvalidCount: return "invalid count";
    case Status::kInputCapacityExceeded: return "input capacity exceeded";
    case Status::kOutputCapacityExceeded: return "output capacity exceeded";
    case Status::kSpanCapacityExceeded: return "span capacity exceeded";
    case Status::kInternalError: return "internal planner error";
  }
  return "unknown status";
}

}  // namespace balance
