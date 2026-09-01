#include "balance/rail_balance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expression)                                                \
  do {                                                                        \
    const cudaError_t error__ = (expression);                                 \
    if (error__ != cudaSuccess) {                                             \
      throw std::runtime_error(std::string(#expression) + ": " +             \
                               cudaGetErrorString(error__));                  \
    }                                                                         \
  } while (false)

constexpr int kPartitions = 7;

struct DeviceState {
  std::uint8_t* input = nullptr;
  std::uint8_t* balanced = nullptr;
  std::uint8_t* restored = nullptr;
  std::int32_t* counts = nullptr;
  void* workspace = nullptr;
  cudaStream_t stream = nullptr;
  std::int64_t input_records = 0;
};

std::uint64_t record_id(int rail, int partition, int ordinal) {
  return 1ull + (static_cast<std::uint64_t>(partition) << 48) +
         (static_cast<std::uint64_t>(rail) << 40) +
         static_cast<std::uint64_t>(ordinal);
}

std::uint8_t payload_byte(std::uint64_t id, std::int64_t byte) {
  return static_cast<std::uint8_t>((id * 1315423911ull + byte * 17 +
                                    (byte >> 4) * 29) & 0xffu);
}

void fill_record(std::uint8_t* record, int bytes, std::uint64_t id) {
  std::memcpy(record, &id, sizeof(id));
  for (int byte = sizeof(id); byte < bytes; ++byte) {
    record[byte] = payload_byte(id, byte);
  }
}

void check_record(const std::uint8_t* record, int bytes, std::uint64_t id) {
  std::uint64_t observed = 0;
  std::memcpy(&observed, record, sizeof(observed));
  if (observed != id) {
    throw std::runtime_error("record id mismatch");
  }
  for (int byte = sizeof(id); byte < bytes; ++byte) {
    if (record[byte] != payload_byte(id, byte)) {
      throw std::runtime_error("record payload mismatch");
    }
  }
}

void enable_peer_access() {
  for (int source = 0; source < balance::kNumRails; ++source) {
    CUDA_CHECK(cudaSetDevice(source));
    for (int peer = 0; peer < balance::kNumRails; ++peer) {
      if (source == peer) {
        continue;
      }
      int accessible = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, source, peer));
      if (!accessible) {
        throw std::runtime_error("all eight GPUs must have P2P access");
      }
      const cudaError_t result = cudaDeviceEnablePeerAccess(peer, 0);
      if (result != cudaSuccess && result != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(result);
      }
      if (result == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
      }
    }
  }
}

void run_case(int record_bytes) {
  if (record_bytes < static_cast<int>(sizeof(std::uint64_t))) {
    throw std::runtime_error("test record is too small");
  }

  std::array<std::array<std::int32_t, kPartitions>, balance::kNumRails>
      count{};
  std::array<std::int64_t, balance::kNumRails> input_records{};
  std::array<std::int64_t, kPartitions> partition_totals{};
  std::vector<std::int32_t> flat_counts(balance::kNumRails * kPartitions);
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    for (int partition = 0; partition < kPartitions; ++partition) {
      int value = (rail * 11 + partition * 7 + 3) % 10;
      if (rail == (partition * 3) % balance::kNumRails) {
        value += 17 + partition;
      }
      if ((rail + 2 * partition) % 9 == 0) {
        value = 0;
      }
      count[rail][partition] = value;
      flat_counts[rail * kPartitions + partition] = value;
      input_records[rail] += value;
      partition_totals[partition] += value;
    }
  }

  std::int64_t output_records = 0;
  for (const auto total : partition_totals) {
    output_records += (total + balance::kNumRails - 1) /
                      balance::kNumRails;
  }
  const std::int64_t max_input =
      *std::max_element(input_records.begin(), input_records.end());
  const std::size_t workspace_bytes = balance::workspace_size(kPartitions);
  std::array<DeviceState, balance::kNumRails> device{};
  std::array<std::vector<std::uint8_t>, balance::kNumRails> original;

  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    CUDA_CHECK(cudaStreamCreateWithFlags(&device[rail].stream,
                                         cudaStreamNonBlocking));
    device[rail].input_records = input_records[rail];
    const std::size_t input_bytes = std::max<std::size_t>(
        1, static_cast<std::size_t>(input_records[rail]) * record_bytes);
    const std::size_t output_bytes = std::max<std::size_t>(
        1, static_cast<std::size_t>(output_records) * record_bytes);
    original[rail].resize(input_bytes);
    std::int64_t record = 0;
    for (int partition = 0; partition < kPartitions; ++partition) {
      for (int ordinal = 0; ordinal < count[rail][partition]; ++ordinal) {
        fill_record(original[rail].data() + record * record_bytes,
                    record_bytes, record_id(rail, partition, ordinal));
        ++record;
      }
    }
    CUDA_CHECK(cudaMalloc(&device[rail].input, input_bytes));
    CUDA_CHECK(cudaMalloc(&device[rail].balanced, output_bytes));
    CUDA_CHECK(cudaMalloc(&device[rail].restored, input_bytes));
    CUDA_CHECK(cudaMalloc(&device[rail].counts,
                          flat_counts.size() * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&device[rail].workspace, workspace_bytes));
    CUDA_CHECK(cudaMemcpy(device[rail].input, original[rail].data(),
                          input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device[rail].counts, flat_counts.data(),
                          flat_counts.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device[rail].balanced, 0xa5, output_bytes));
    CUDA_CHECK(cudaMemset(device[rail].restored, 0xcd, input_bytes));
  }

  balance::Options options{};
  options.engine = balance::CopyEngine::kAuto;
  options.zero_padding = true;
  options.remainder_seed = 5;
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    balance::BalanceParams params{};
    params.rail = rail;
    params.num_partitions = kPartitions;
    params.record_bytes = record_bytes;
    params.all_counts = device[rail].counts;
    for (int peer = 0; peer < balance::kNumRails; ++peer) {
      params.peer_input[peer] = device[peer].input;
      params.input_capacity_records[peer] = input_records[peer];
    }
    params.max_input_records = max_input;
    params.output = device[rail].balanced;
    params.output_capacity_records = output_records;
    params.workspace = device[rail].workspace;
    params.workspace_bytes = workspace_bytes;
    CUDA_CHECK(balance::launch_balance(params, options, device[rail].stream));
  }
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    CUDA_CHECK(cudaStreamSynchronize(device[rail].stream));
  }

  std::array<std::map<std::uint64_t, int>, kPartitions> expected_ids;
  std::array<std::map<std::uint64_t, int>, kPartitions> actual_ids;
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    for (int partition = 0; partition < kPartitions; ++partition) {
      for (int ordinal = 0; ordinal < count[rail][partition]; ++ordinal) {
        ++expected_ids[partition][record_id(rail, partition, ordinal)];
      }
    }
  }

  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    balance::MetadataHeader header{};
    std::vector<balance::PartitionInfo> partitions(kPartitions);
    std::vector<balance::Span> spans(
        kPartitions * balance::kMaxSpansPerPartition);
    const auto metadata =
        balance::metadata_view(device[rail].workspace, kPartitions);
    CUDA_CHECK(cudaMemcpy(&header, metadata.header, sizeof(header),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(partitions.data(), metadata.partitions,
                          partitions.size() * sizeof(partitions[0]),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(spans.data(), metadata.spans,
                          spans.size() * sizeof(spans[0]),
                          cudaMemcpyDeviceToHost));
    if (header.magic != balance::kMetadataMagic ||
        header.version != balance::kMetadataVersion ||
        header.status != static_cast<int>(balance::Status::kSuccess) ||
        header.total_output_records != output_records ||
        header.local_input_records != input_records[rail]) {
      throw std::runtime_error("invalid metadata header");
    }

    std::vector<std::uint8_t> output(
        static_cast<std::size_t>(output_records) * record_bytes);
    CUDA_CHECK(cudaMemcpy(output.data(), device[rail].balanced, output.size(),
                          cudaMemcpyDeviceToHost));
    std::int64_t expected_offset = 0;
    for (int partition = 0; partition < kPartitions; ++partition) {
      const auto& info = partitions[partition];
      const int physical = static_cast<int>(
          (partition_totals[partition] + balance::kNumRails - 1) /
          balance::kNumRails);
      if (info.output_offset != expected_offset ||
          info.global_valid_count != partition_totals[partition] ||
          info.physical_count != physical ||
          info.local_valid_count < physical - 1 ||
          info.local_valid_count > physical ||
          info.span_begin != partition * balance::kMaxSpansPerPartition ||
          info.span_count < 0 ||
          info.span_count > balance::kMaxSpansPerPartition) {
        throw std::runtime_error("invalid partition metadata");
      }
      for (int local = 0; local < info.local_valid_count; ++local) {
        const auto* record = output.data() +
            (info.output_offset + local) * record_bytes;
        std::uint64_t id = 0;
        std::memcpy(&id, record, sizeof(id));
        check_record(record, record_bytes, id);
        ++actual_ids[partition][id];
      }
      for (int local = info.local_valid_count; local < physical; ++local) {
        const auto* record = output.data() +
            (info.output_offset + local) * record_bytes;
        for (int byte = 0; byte < record_bytes; ++byte) {
          if (record[byte] != 0) {
            throw std::runtime_error("padding was not cleared");
          }
        }
      }
      expected_offset += physical;
    }
  }
  if (actual_ids != expected_ids) {
    throw std::runtime_error("balanced records are not a global permutation");
  }

  // Exercise the saved split metadata in the reverse direction.
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    balance::UnbalanceParams params{};
    params.rail = rail;
    params.num_partitions = kPartitions;
    params.record_bytes = record_bytes;
    params.input = device[rail].balanced;
    params.max_input_records = max_input;
    params.workspace = device[rail].workspace;
    params.workspace_bytes = workspace_bytes;
    for (int peer = 0; peer < balance::kNumRails; ++peer) {
      params.peer_output[peer] = device[peer].restored;
    }
    CUDA_CHECK(balance::launch_unbalance(params, options,
                                         device[rail].stream));
  }
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    CUDA_CHECK(cudaStreamSynchronize(device[rail].stream));
  }
  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    std::vector<std::uint8_t> restored(original[rail].size());
    CUDA_CHECK(cudaMemcpy(restored.data(), device[rail].restored,
                          restored.size(), cudaMemcpyDeviceToHost));
    if (restored != original[rail]) {
      const auto mismatch = std::mismatch(restored.begin(), restored.end(),
                                          original[rail].begin());
      const auto byte = std::distance(restored.begin(), mismatch.first);
      throw std::runtime_error(
          "unbalance did not restore rail " + std::to_string(rail) +
          " at byte " + std::to_string(byte) + " (got " +
          std::to_string(static_cast<int>(*mismatch.first)) + ", expected " +
          std::to_string(static_cast<int>(*mismatch.second)) + ")");
    }
  }

  for (int rail = 0; rail < balance::kNumRails; ++rail) {
    CUDA_CHECK(cudaSetDevice(rail));
    CUDA_CHECK(cudaFree(device[rail].workspace));
    CUDA_CHECK(cudaFree(device[rail].counts));
    CUDA_CHECK(cudaFree(device[rail].restored));
    CUDA_CHECK(cudaFree(device[rail].balanced));
    CUDA_CHECK(cudaFree(device[rail].input));
    CUDA_CHECK(cudaStreamDestroy(device[rail].stream));
  }
  std::cout << "PASS record_bytes=" << record_bytes << '\n';
}

}  // namespace

int main() {
  try {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < balance::kNumRails) {
      std::cout << "SKIP: requires 8 GPUs, found " << devices << '\n';
      return 0;
    }
    enable_peer_access();
    run_case(30);      // Unaligned byte-tail fallback.
    run_case(64);      // Small-record vector path.
    run_case(14368);   // hidden=7168 BF16; TMA on SM90, vector otherwise.
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
