#include "balance/rail_balance.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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

constexpr int kPartitions = 8;

struct DeviceState {
  void* input = nullptr;
  void* output = nullptr;
  void* restored = nullptr;
  void* workspace = nullptr;
  std::int32_t* counts = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
};

class SpinBarrier {
 public:
  explicit SpinBarrier(int parties) : parties_(parties) {}

  void wait() {
    const int generation = generation_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == parties_) {
      arrived_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
      return;
    }
    while (generation_.load(std::memory_order_acquire) == generation) {
      std::this_thread::yield();
    }
  }

 private:
  int parties_;
  std::atomic<int> arrived_{0};
  std::atomic<int> generation_{0};
};

void enable_peer_access() {
  for (int source = 0; source < balance::kNumRails; ++source) {
    CUDA_CHECK(cudaSetDevice(source));
    for (int peer = 0; peer < balance::kNumRails; ++peer) {
      if (source == peer) continue;
      int accessible = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, source, peer));
      if (!accessible) throw std::runtime_error("P2P access is required");
      const cudaError_t result = cudaDeviceEnablePeerAccess(peer, 0);
      if (result != cudaSuccess && result != cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_CHECK(result);
      }
      if (result == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
    }
  }
}

std::int64_t moved_records(const std::vector<std::int32_t>& counts,
                           int partitions,
                           int remainder_seed) {
  std::int64_t moved = 0;
  for (int partition = 0; partition < partitions; ++partition) {
    std::array<int, balance::kNumRails> quota{};
    std::int64_t total = 0;
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      total += counts[rail * partitions + partition];
    }
    const int base = static_cast<int>(total / balance::kNumRails);
    const int remainder = static_cast<int>(total % balance::kNumRails);
    quota.fill(base);
    const int start =
        ((remainder_seed % balance::kNumRails) + balance::kNumRails +
         partition % balance::kNumRails) % balance::kNumRails;
    int assigned = 0;
    for (int offset = 0;
         offset < balance::kNumRails && assigned < remainder;
         ++offset) {
      const int rail = (start + offset) % balance::kNumRails;
      if (counts[rail * partitions + partition] > base) {
        ++quota[rail];
        ++assigned;
      }
    }
    for (int offset = 0;
         offset < balance::kNumRails && assigned < remainder;
         ++offset) {
      const int rail = (start + offset) % balance::kNumRails;
      if (counts[rail * partitions + partition] <= base) {
        ++quota[rail];
        ++assigned;
      }
    }
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      moved += std::max(
          counts[rail * partitions + partition] - quota[rail], 0);
    }
  }
  return moved;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int record_bytes = argc > 1 ? std::atoi(argv[1]) : 14368;
    const int records_per_rail = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 200;
    const std::string requested_engine = argc > 4 ? argv[4] : "auto";
    const int max_blocks_per_span = argc > 5 ? std::atoi(argv[5]) : 16;
    const std::string launch_mode = argc > 6 ? argv[6] : "eager";
    const std::string measurement_mode = argc > 7 ? argv[7] : "throughput";
    const std::string traffic_pattern = argc > 8 ? argv[8] : "skew";
    const std::string operation = argc > 9 ? argv[9] : "balance";
    if (record_bytes <= 0 || records_per_rail <= 0 || iterations <= 0) {
      throw std::runtime_error(
          "usage: rail_balance_bench [record_bytes] [records_per_rail] "
          "[iterations] [auto|vector|tma] [max_blocks_per_span] "
          "[eager|graph] [throughput|latency] [skew|uniform] "
          "[balance|roundtrip]");
    }
    if (traffic_pattern != "skew" && traffic_pattern != "uniform") {
      throw std::runtime_error("traffic pattern must be skew or uniform");
    }
    const bool measure_roundtrip = operation == "roundtrip";
    if (!measure_roundtrip && operation != "balance") {
      throw std::runtime_error("operation must be balance or roundtrip");
    }
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < balance::kNumRails) {
      std::cout << "SKIP: requires 8 GPUs, found " << devices << '\n';
      return 0;
    }
    enable_peer_access();

    // Every rail owns the same total number of records, but 60% are placed in
    // a rotating hot partition.  This creates substantial inter-rail movement
    // while keeping all partition totals comparable.
    std::vector<std::int32_t> host_counts(
        balance::kNumRails * kPartitions, 0);
    std::array<std::int64_t, kPartitions> partition_total{};
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      if (traffic_pattern == "uniform") {
        // Every rail owns the same number of records for the same remote
        // partition. The planner should retain all records locally, so this
        // isolates planner + launch + local materialization overhead.
        host_counts[rail * kPartitions] = records_per_rail;
      } else {
        const int common_total = records_per_rail * 2 / 5;
        const int common = common_total / kPartitions;
        int assigned = 0;
        for (int partition = 0; partition < kPartitions; ++partition) {
          host_counts[rail * kPartitions + partition] = common;
          assigned += common;
        }
        const int hot = rail * 3 % kPartitions;
        host_counts[rail * kPartitions + hot] +=
            records_per_rail - assigned;
      }
      for (int partition = 0; partition < kPartitions; ++partition) {
        partition_total[partition] +=
            host_counts[rail * kPartitions + partition];
      }
    }
    std::int64_t output_records = 0;
    for (const auto total : partition_total) {
      output_records += (total + balance::kNumRails - 1) /
                        balance::kNumRails;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(records_per_rail) * record_bytes;
    const std::size_t output_bytes =
        static_cast<std::size_t>(output_records) * record_bytes;
    const std::size_t workspace_bytes =
        balance::workspace_size(kPartitions);
    std::array<DeviceState, balance::kNumRails> state{};
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      CUDA_CHECK(cudaSetDevice(rail));
      CUDA_CHECK(cudaStreamCreateWithFlags(&state[rail].stream,
                                           cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreate(&state[rail].start));
      CUDA_CHECK(cudaEventCreate(&state[rail].stop));
      CUDA_CHECK(cudaMalloc(&state[rail].input, input_bytes));
      CUDA_CHECK(cudaMalloc(&state[rail].output, output_bytes));
      CUDA_CHECK(cudaMalloc(&state[rail].restored, input_bytes));
      CUDA_CHECK(cudaMalloc(&state[rail].workspace, workspace_bytes));
      CUDA_CHECK(cudaMalloc(&state[rail].counts,
                            host_counts.size() * sizeof(std::int32_t)));
      CUDA_CHECK(cudaMemset(state[rail].input, rail + 1, input_bytes));
      CUDA_CHECK(cudaMemcpy(state[rail].counts, host_counts.data(),
                            host_counts.size() * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice));
    }

    balance::Options options{};
    if (requested_engine == "auto") {
      options.engine = balance::CopyEngine::kAuto;
    } else if (requested_engine == "vector") {
      options.engine = balance::CopyEngine::kVector;
    } else if (requested_engine == "tma") {
      options.engine = balance::CopyEngine::kTma;
    } else {
      throw std::runtime_error("engine must be auto, vector, or tma");
    }
    const bool use_graph = launch_mode == "graph";
    if (!use_graph && launch_mode != "eager") {
      throw std::runtime_error("launch mode must be eager or graph");
    }
    const bool measure_latency = measurement_mode == "latency";
    if (!measure_latency && measurement_mode != "throughput") {
      throw std::runtime_error(
          "measurement mode must be throughput or latency");
    }
    options.zero_padding = false;
    options.max_blocks_per_span = max_blocks_per_span;
    std::array<balance::BalanceParams, balance::kNumRails> params{};
    std::array<balance::UnbalanceParams, balance::kNumRails>
        unbalance_params{};
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      auto& p = params[rail];
      p.rail = rail;
      p.num_partitions = kPartitions;
      p.record_bytes = record_bytes;
      p.all_counts = state[rail].counts;
      for (int peer = 0; peer < balance::kNumRails; ++peer) {
        p.peer_input[peer] = state[peer].input;
        p.input_capacity_records[peer] = records_per_rail;
      }
      p.max_input_records = records_per_rail;
      p.output = state[rail].output;
      p.output_capacity_records = output_records;
      p.workspace = state[rail].workspace;
      p.workspace_bytes = workspace_bytes;

      auto& u = unbalance_params[rail];
      u.rail = rail;
      u.num_partitions = kPartitions;
      u.record_bytes = record_bytes;
      u.input = state[rail].output;
      for (int peer = 0; peer < balance::kNumRails; ++peer) {
        u.peer_output[peer] = state[peer].restored;
      }
      u.max_input_records = records_per_rail;
      u.workspace = state[rail].workspace;
      u.workspace_bytes = workspace_bytes;
    }

    if (use_graph) {
      for (int rail = 0; rail < balance::kNumRails; ++rail) {
        CUDA_CHECK(cudaSetDevice(rail));
        CUDA_CHECK(cudaStreamSynchronize(state[rail].stream));
        CUDA_CHECK(cudaStreamBeginCapture(state[rail].stream,
                                          cudaStreamCaptureModeThreadLocal));
        CUDA_CHECK(balance::launch_balance(params[rail], options,
                                           state[rail].stream));
        if (measure_roundtrip) {
          CUDA_CHECK(balance::launch_unbalance(
              unbalance_params[rail], options, state[rail].stream));
        }
        CUDA_CHECK(cudaStreamEndCapture(state[rail].stream,
                                        &state[rail].graph));
        CUDA_CHECK(cudaGraphInstantiate(&state[rail].graph_exec,
                                        state[rail].graph, 0));
      }
    }

    auto enqueue = [&](int rail) {
      if (use_graph) {
        return cudaGraphLaunch(state[rail].graph_exec, state[rail].stream);
      }
      cudaError_t result = balance::launch_balance(
          params[rail], options, state[rail].stream);
      if (result == cudaSuccess && measure_roundtrip) {
        result = balance::launch_unbalance(
            unbalance_params[rail], options, state[rail].stream);
      }
      return result;
    };

    // Context creation leaves several devices at idle clocks.  A longer
    // warmup makes payload sweeps stable without requiring privileged clock
    // locking.
    constexpr int warmups = 100;
    for (int iteration = 0; iteration < warmups; ++iteration) {
      for (int rail = 0; rail < balance::kNumRails; ++rail) {
        CUDA_CHECK(cudaSetDevice(rail));
        CUDA_CHECK(enqueue(rail));
      }
    }
    // Match the real one-process-per-rank launch pattern: eight host workers
    // remain bound to their GPUs and enqueue concurrently.  A single host
    // thread switching devices adds ~30 us of artificial latency for tiny
    // payloads and materially distorts the crossover point.
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> worker_error{static_cast<int>(cudaSuccess)};
    std::array<float, balance::kNumRails> elapsed_ms{};
    std::vector<std::array<float, balance::kNumRails>> latency_ms(
        measure_latency ? iterations : 0);
    SpinBarrier iteration_barrier(balance::kNumRails);
    std::vector<std::thread> workers;
    workers.reserve(balance::kNumRails);
    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      workers.emplace_back([&, rail] {
        auto record_error = [&](cudaError_t error) {
          int expected = static_cast<int>(cudaSuccess);
          worker_error.compare_exchange_strong(expected,
                                               static_cast<int>(error));
        };
        cudaError_t error = cudaSetDevice(rail);
        if (error == cudaSuccess) {
          error = cudaStreamSynchronize(state[rail].stream);
        }
        if (error != cudaSuccess) {
          record_error(error);
        }
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        if (error != cudaSuccess) return;
        if (measure_latency) {
          for (int iteration = 0; iteration < iterations; ++iteration) {
            iteration_barrier.wait();
            error = cudaEventRecord(state[rail].start, state[rail].stream);
            if (error == cudaSuccess) error = enqueue(rail);
            if (error == cudaSuccess) {
              error = cudaEventRecord(state[rail].stop, state[rail].stream);
            }
            if (error == cudaSuccess) {
              error = cudaEventSynchronize(state[rail].stop);
            }
            if (error == cudaSuccess) {
              error = cudaEventElapsedTime(&latency_ms[iteration][rail],
                                           state[rail].start,
                                           state[rail].stop);
            }
            if (error != cudaSuccess) record_error(error);
          }
        } else {
          error = cudaEventRecord(state[rail].start, state[rail].stream);
          for (int iteration = 0;
               iteration < iterations && error == cudaSuccess;
               ++iteration) {
            error = enqueue(rail);
          }
          if (error == cudaSuccess) {
            error = cudaEventRecord(state[rail].stop, state[rail].stream);
          }
          if (error == cudaSuccess) {
            error = cudaEventSynchronize(state[rail].stop);
          }
          if (error == cudaSuccess) {
            error = cudaEventElapsedTime(&elapsed_ms[rail], state[rail].start,
                                         state[rail].stop);
          }
        }
        if (error != cudaSuccess) record_error(error);
      });
    }
    while (ready.load(std::memory_order_acquire) < balance::kNumRails) {
      std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    if (worker_error.load() != static_cast<int>(cudaSuccess)) {
      CUDA_CHECK(static_cast<cudaError_t>(worker_error.load()));
    }
    double us = 0.0;
    double p95_us = 0.0;
    if (measure_latency) {
      std::vector<float> max_latency_us(iterations);
      for (int iteration = 0; iteration < iterations; ++iteration) {
        max_latency_us[iteration] = 1000.0f * *std::max_element(
            latency_ms[iteration].begin(), latency_ms[iteration].end());
      }
      std::sort(max_latency_us.begin(), max_latency_us.end());
      us = max_latency_us[iterations / 2];
      p95_us = max_latency_us[std::min(
          iterations - 1, static_cast<int>(iterations * 0.95))];
    } else {
      const float max_ms =
          *std::max_element(elapsed_ms.begin(), elapsed_ms.end());
      us = max_ms * 1000.0 / iterations;
    }
    balance::CopyEngine engine = balance::CopyEngine::kVector;
    CUDA_CHECK(cudaSetDevice(0));
    int compute_major = 0;
    int compute_minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &compute_major, cudaDevAttrComputeCapabilityMajor, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &compute_minor, cudaDevAttrComputeCapabilityMinor, 0));
    CUDA_CHECK(balance::select_copy_engine(
        options.engine, record_bytes, options.tma_threshold_bytes, &engine));
    const std::int64_t cross_rail_records =
        moved_records(host_counts, kPartitions, options.remainder_seed);
    const int copy_passes = measure_roundtrip ? 2 : 1;
    const double valid_copy_gb = copy_passes *
        static_cast<double>(records_per_rail) * balance::kNumRails *
        record_bytes / 1.0e9;
    const double cross_rail_gb = copy_passes *
        static_cast<double>(cross_rail_records) * record_bytes / 1.0e9;
    const double valid_copy_gbps = valid_copy_gb / (us * 1.0e-6);
    const double nvlink_gbps = cross_rail_gb / (us * 1.0e-6);
    std::cout << std::fixed << std::setprecision(2)
              << "engine="
              << (engine == balance::CopyEngine::kTma ? "tma" : "vector")
              << " launch=" << launch_mode
              << " measurement=" << measurement_mode
              << " traffic=" << traffic_pattern
              << " operation=" << operation
              << " sm=" << compute_major << compute_minor
              << " record_bytes=" << record_bytes
              << " records_per_rail=" << records_per_rail
              << " max_8rail_us=" << us
              << " p95_us=" << p95_us
              << " valid_copy_GB/s=" << valid_copy_gbps
              << " cross_rail_records=" << cross_rail_records
              << " logical_nvlink_GB/s=" << nvlink_gbps << '\n';

    for (int rail = 0; rail < balance::kNumRails; ++rail) {
      CUDA_CHECK(cudaSetDevice(rail));
      if (state[rail].graph_exec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(state[rail].graph_exec));
      }
      if (state[rail].graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(state[rail].graph));
      }
      CUDA_CHECK(cudaFree(state[rail].counts));
      CUDA_CHECK(cudaFree(state[rail].workspace));
      CUDA_CHECK(cudaFree(state[rail].restored));
      CUDA_CHECK(cudaFree(state[rail].output));
      CUDA_CHECK(cudaFree(state[rail].input));
      CUDA_CHECK(cudaEventDestroy(state[rail].stop));
      CUDA_CHECK(cudaEventDestroy(state[rail].start));
      CUDA_CHECK(cudaStreamDestroy(state[rail].stream));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
