// Copyright (c) 2026
// Standalone 8-rail CUDA buffer balancer.  This API intentionally has no
// dependency on DeepEP, NCCL, NVSHMEM, or CUTLASS.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace balance {

constexpr int kNumRails = 8;
constexpr int kMaxSpansPerPartition = kNumRails + 1;  // 8 data + 1 padding.
constexpr std::uint32_t kMetadataMagic = 0x52424c38u; // "RBL8"
constexpr std::uint32_t kMetadataVersion = 1;

enum class Status : std::int32_t {
  kSuccess = 0,
  kInvalidCount = 1,
  kInputCapacityExceeded = 2,
  kOutputCapacityExceeded = 3,
  kSpanCapacityExceeded = 4,
  kInternalError = 5,
};

enum class CopyEngine : std::int32_t {
  kAuto = 0,
  kVector = 1,
  kTma = 2,
};

enum SpanFlags : std::uint32_t {
  kSpanData = 0,
  kSpanPadding = 1u << 0,
};

// A data span means:
//   output[dst_record : dst_record + record_count] =
//       peer_input[src_rail][src_record : src_record + record_count].
// Padding spans have src_rail == -1 and are only emitted when zero_padding is
// requested.  All record indices are in records, not bytes.
struct alignas(16) Span {
  std::int64_t src_record;
  std::int64_t dst_record;
  std::int32_t record_count;
  std::int16_t src_rail;
  std::int16_t partition;
  std::uint32_t flags;
};
static_assert(sizeof(Span) == 32, "Span is a stable 32-byte ABI");

struct alignas(16) PartitionInfo {
  std::int64_t output_offset;
  std::int64_t global_valid_count;
  std::int32_t physical_count;
  std::int32_t local_valid_count;
  std::int32_t span_begin;
  std::int32_t span_count;
};
static_assert(sizeof(PartitionInfo) == 32,
              "PartitionInfo is a stable 32-byte ABI");

struct alignas(16) MetadataHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::int32_t status;
  std::int32_t rail;
  std::int32_t num_partitions;
  std::int32_t record_bytes;
  std::int32_t remainder_seed;
  std::int32_t reserved0;
  std::int64_t total_output_records;
  std::int64_t total_valid_records;
  std::int64_t local_input_records;
  std::int64_t num_spans;
};
static_assert(sizeof(MetadataHeader) == 64,
              "MetadataHeader is a stable 64-byte ABI");

struct MetadataView {
  MetadataHeader* header;
  PartitionInfo* partitions;
  Span* spans;
};

struct ConstMetadataView {
  const MetadataHeader* header;
  const PartitionInfo* partitions;
  const Span* spans;
};

struct Options {
  CopyEngine engine = CopyEngine::kAuto;

  // Auto uses TMA only on SM90+ and only when one record is at least this
  // large.  The threshold is deliberately based on record size: small token
  // metadata should stay on the vector path even if the batch is large.
  std::int32_t tma_threshold_bytes = 4096;

  // The exact-size output contains at most one padding record per partition.
  // Clearing it is optional because an RDMA consumer can skip it using the
  // metadata.  Disabling the clear removes all padding traffic.
  bool zero_padding = false;

  // Rotates ownership of the remainder and therefore avoids permanently
  // favoring rail zero.  The planner also prefers rails which can retain the
  // extra record, minimizing P2P traffic.
  std::int32_t remainder_seed = 0;

  // Bounds parallelism assigned to a single span.  This is not a persistent
  // kernel: every CTA handles finite, statically indexed tiles and exits.
  std::int32_t max_blocks_per_span = 16;
};

struct BalanceParams {
  std::int32_t rail;
  std::int32_t num_partitions;
  std::int32_t record_bytes;

  // Device array [8, num_partitions], rank-major.  Every rail must see the
  // same snapshot.  Counts describe packed partition-major peer_input data.
  const std::int32_t* all_counts;

  // Peer-accessible UVA pointers.  In multi-process deployments these may be
  // CUDA-IPC/VMM/NCCL-window mappings; this library does not own or exchange
  // them.  The local pointer is peer_input[rail].
  const void* peer_input[kNumRails];

  // Per-peer allocation bounds and a conservative maximum used to size the
  // launch grid without synchronously reading counts back to the CPU.
  std::int64_t input_capacity_records[kNumRails];
  std::int64_t max_input_records;

  void* output;
  std::int64_t output_capacity_records;

  // Reusable device workspace.  No allocation or host synchronization occurs
  // in launch_balance().
  void* workspace;
  std::size_t workspace_bytes;
};

struct UnbalanceParams {
  std::int32_t rail;
  std::int32_t num_partitions;
  std::int32_t record_bytes;
  const void* input;
  void* peer_output[kNumRails];
  std::int64_t max_input_records;
  const void* workspace;
  std::size_t workspace_bytes;
};

// Workspace is deterministic and contains only the public metadata structs.
std::size_t workspace_size(std::int32_t num_partitions);
MetadataView metadata_view(void* workspace, std::int32_t num_partitions);
ConstMetadataView metadata_view(const void* workspace,
                                std::int32_t num_partitions);

// A safe allocation bound when only aggregate per-rank capacities are known.
// It includes worst-case per-partition rounding to ceil(total / 8).
std::int64_t max_output_records(std::int64_t total_input_capacity,
                                std::int32_t num_partitions);

// Returns the engine Auto would choose on the current device.  Forced TMA
// returns cudaErrorNotSupported on pre-SM90 devices or unaligned records.
cudaError_t select_copy_engine(CopyEngine requested,
                               std::int32_t record_bytes,
                               std::int32_t tma_threshold_bytes,
                               CopyEngine* selected);

// Enqueues planner + copy kernels on stream.  Device-side count/capacity
// errors are reported asynchronously in MetadataHeader::status and suppress
// all copies.  Host argument errors are returned directly.
cudaError_t launch_balance(const BalanceParams& params,
                           const Options& options,
                           cudaStream_t stream = nullptr);

// Reverses the data spans produced by launch_balance().  All eight rails may
// enqueue this concurrently; the spans target disjoint original records.
// Padding spans are ignored.
cudaError_t launch_unbalance(const UnbalanceParams& params,
                             const Options& options,
                             cudaStream_t stream = nullptr);

const char* status_string(Status status);

}  // namespace balance
