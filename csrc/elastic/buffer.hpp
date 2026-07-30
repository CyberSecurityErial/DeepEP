#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#include <pybind11/functional.h>
#include <c10/cuda/CUDAGuard.h>

#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_layout.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/common/rail_balance_protocol_layout.cuh>
#include <deep_ep/common/rail_balance_vnode_layout.cuh>

#include "../kernels/backend/api.cuh"
#include "../kernels/elastic/api.hpp"
#include "../utils/event.hpp"
#include "utils.hpp"

namespace deep_ep::elastic {

using RailBalanceProtocolTensors = std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>;

// Keep the private force completion byte-for-byte compatible with the native
// dispatch return contract so Python can reuse its existing unpack/EPHandle
// construction without adding a second public result shape.
using RailBalanceHybridDispatchResult = std::tuple<
    torch::Tensor,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    int,
    int,
    std::vector<int>,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    std::optional<torch::Tensor>,
    std::optional<torch::Tensor>,
    std::optional<EventHandle>>;

using RailBalanceHybridCombineResult = std::tuple<
    torch::Tensor,
    std::optional<torch::Tensor>,
    std::optional<EventHandle>>;

using RailBalanceVNodeTensors = std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>;

class ElasticBuffer {
    // Buffer bytes = GPU buffer + CPU buffer (excludes workspace)
    // Memory layout: [[[Workspace] GPU buffer] CPU buffer]
    int64_t num_buffer_bytes;
    int64_t num_gpu_buffer_bytes;
    int64_t num_cpu_buffer_bytes;
    void* buffer;
    int device_index = -1;

    // Destructor settings
    bool explicitly_destroy;
    bool destroyed = false;

    // Workspace
    // NOTES: for all workspace, we must keep them as zeros
    void *workspace;
    void *host_workspace, *mapped_host_workspace;
    std::shared_ptr<layout::WorkspaceLayout> workspace_layout_wo_expert;

    // CUDA streams
    at::cuda::CUDAStream comm_stream;

    // Whether to use hybrid mode (scale-out with scale-up)
    bool allow_hybrid_mode;

    // Whether to allow multiple reductions
    bool allow_multiple_reduction;

    // Whether to prefer overlapping communication with compute (use more SMs and channels if false)
    bool prefer_overlap_with_compute;

    // Timeout settings
    int num_cpu_timeout_secs;
    int64_t num_gpu_timeout_cycles;

    // NCCL context
    std::shared_ptr<nccl::NCCLSymmetricMemoryContext> nccl_context;

    // Some EP hybrid mode settings
    static constexpr int kNumMaxChannelsPerSM = 8;
    static constexpr int kNumMaxSMs = 160;
    static constexpr int kNumMaxChannels = kNumMaxChannelsPerSM * kNumMaxSMs;

    // Private C080-B2 prepare -> local snapshot -> plan transaction. Only one
    // may own the force arena and the legacy workspace barrier epoch at once.
    std::optional<RailBalanceHybridPlanPending>
        rail_balance_hybrid_plan_pending;

    // Private C080-D world-emulator transaction.  It is intentionally
    // independent from the source-subgroup transaction above because the two
    // ElasticBuffer instances own different NCCL symmetric windows.
    std::optional<RailBalanceHybridVNodePending>
        rail_balance_hybrid_vnode_pending;

    // Some Engram storage settings
    int num_engram_entries = 0, engram_hidden = 0;
    std::optional<torch::Tensor> engram_sf;

    // PP settings
    int prev_rank_idx = 0, next_rank_idx = 0;
    int64_t num_max_pp_tensor_bytes = 0;
    int num_max_pp_inflight_tensors = 0;

    // AGRS session settings
    int64_t num_max_agrs_session_bytes = 0;
    int num_max_agrs_per_session = 0;
    int agrs_session_idx = 0;
    bool agrs_in_session = false;

    // AGRS in-session settings
    int64_t agrs_buffer_offset = 0;
    int agrs_buffer_slot_idx = 0;

public:
    ElasticBuffer(const int& rank_idx, const int& num_ranks,
                  const int64_t& nccl_comm, const symmetric::cpu_comm_t& cpu_comm,
                  const int64_t& num_buffer_bytes, const int64_t& num_cpu_buffer_bytes,
                  const bool& allow_hybrid_mode,
                  const bool& allow_multiple_reduction,
                  const bool& prefer_overlap_with_compute,
                  const int& sl_idx, const int& num_allocated_qps,
                  const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                  const bool& explicitly_destroy):
        num_buffer_bytes(num_buffer_bytes),
        num_cpu_buffer_bytes(num_cpu_buffer_bytes),
        explicitly_destroy(explicitly_destroy),
        comm_stream(get_global_comm_stream()),
        allow_hybrid_mode(allow_hybrid_mode),
        allow_multiple_reduction(allow_multiple_reduction),
        prefer_overlap_with_compute(prefer_overlap_with_compute) {
        // Check buffer bytes alignment (2 MB)
        EP_HOST_ASSERT(num_buffer_bytes > 0 and num_buffer_bytes % symmetric::kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_buffer_bytes >= 0 and num_cpu_buffer_bytes % symmetric::kNumAlignmentBytes == 0);
        EP_HOST_ASSERT(num_cpu_buffer_bytes <= num_buffer_bytes);
        num_gpu_buffer_bytes = num_buffer_bytes - num_cpu_buffer_bytes;
        CUDA_RUNTIME_CHECK(cudaGetDevice(&device_index));

        // Workspace is aligned to 2 MB so that it sits cleanly at the front of the GPU segment
        const auto num_workspace_bytes = math::align<int64_t>(
            layout::WorkspaceLayout::get_num_bytes(), symmetric::kNumAlignmentBytes);

        // Create NCCL symmetric memory context
        // Symmetric memory layout: [[[Workspace] GPU buffer] CPU buffer]
        // sym.num_bytes = workspace + buffer, sym.num_cpu_bytes = CPU buffer
        const auto num_sym_bytes = num_workspace_bytes + num_buffer_bytes;
        this->nccl_context = std::make_shared<nccl::NCCLSymmetricMemoryContext>(
            nccl_comm, cpu_comm, num_ranks, rank_idx,
            num_sym_bytes, num_cpu_buffer_bytes,
            allow_hybrid_mode, sl_idx, num_allocated_qps);

        // Verify the symmetric memory layout matches our expectations
        EP_HOST_ASSERT(num_workspace_bytes + num_gpu_buffer_bytes == nccl_context->num_gpu_bytes);
        EP_HOST_ASSERT(num_cpu_buffer_bytes == nccl_context->num_cpu_bytes);

        // Timeout
        this->num_cpu_timeout_secs = num_cpu_timeout_secs;
        this->num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs);
        this->num_gpu_timeout_cycles *= jit::device_runtime->get_clock_rate();

        // Assign workspaces and buffers
        workspace = this->nccl_context->mapped_window_ptr;
        workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
            workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, 0);
        buffer = static_cast<uint8_t*>(workspace) + num_workspace_bytes;
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));

        // Allocate host workspaces
        CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes(), cudaHostAllocMapped));
        CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
        std::memset(host_workspace, 0, layout::WorkspaceLayout::get_num_bytes());

        // We should call a barrier at the end
        // The barrier should be called by Python `dist.barrier`
        // NOTES: do not call our barrier, as the workspace is not ready yet
    }

    ~ElasticBuffer() noexcept(false) {
        if (not explicitly_destroy)
            destroy();

        if (not destroyed) {
            printf("`destroy()` is not called before DeepEP elastic buffer destruction, which can leak resources.\n");
            fflush(stdout);
        }
    }

    void destroy() {
        EP_HOST_ASSERT(not destroyed);

        // Finish all works on all GPUs
        barrier(true, true);

        // Deallocate host workspaces
        CUDA_RUNTIME_CHECK(cudaFreeHost(host_workspace));

        // Destroy NCCL context
        nccl_context->finalize();

        // Cannot use anymore
        destroyed = true;
    }

    torch::Stream get_comm_stream() const {
        return comm_stream;
    }

    std::tuple<int, int> get_physical_domain_size() const {
        return {nccl_context->num_rdma_ranks, nccl_context->num_nvl_ranks};
    }

    std::tuple<int, int> get_logical_domain_size() const {
        return {nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks};
    }

    static std::tuple<
        int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t>
    get_rail_balance_hybrid_layout(
        const int& hidden,
        const int& num_topk,
        const int& proxy_capacity) {
        EP_STATIC_ASSERT(
            rail_balance::kNumHybridMaxChannels == deep_ep::kNumMaxChannels,
            "Hybrid rail-balance channel capacity must match legacy workspace");
        const auto layout = rail_balance::HybridArenaLayout(
            hidden, num_topk, proxy_capacity);
        return {
            layout.control_offset,
            sizeof(rail_balance::HybridControl),
            layout.channel_count_offset,
            layout.channel_count_bytes,
            layout.proxy_dispatch_offset,
            layout.dispatch_token_bytes,
            layout.proxy_return_offset,
            layout.combine_token_bytes,
            layout.raw_bytes,
            layout.arena_bytes,
        };
    }

    static int64_t calculate_rail_balance_hybrid_buffer_size(
        const int64_t& nccl_comm,
        const int& num_max_tokens_per_rank,
        const int& hidden,
        const int& num_topk,
        const int& proxy_capacity) {
        const auto legacy_bytes = calculate_buffer_size(
            nccl_comm,
            num_max_tokens_per_rank, hidden, num_topk,
            false, true, true);
        const auto layout = rail_balance::HybridArenaLayout(
            hidden, num_topk, proxy_capacity);
        EP_HOST_ASSERT(
            legacy_bytes % rail_balance::kNumHybridBufferAlignmentBytes == 0);
        return rail_balance::checked_add_i64(legacy_bytes, layout.arena_bytes);
    }

    int rail_balance_hybrid_plan_prepare(
        const torch::Tensor& topk_idx,
        const int& hidden,
        const int& num_channels,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_scaleout_ranks,
        const int& local_scaleout_rank,
        const int& proxy_capacity_per_egress,
        const int64_t& arena_offset,
        const int& invocation_id,
        const pybind11::object& remainder_seed,
        const pybind11::object& policy,
        const pybind11::object& threshold_percent,
        const bool& hop_aware) {
        // Everything below is noncollective. Fail before touching the arena if
        // another private force transaction still owns it.
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not rail_balance_hybrid_plan_pending.has_value());
        EP_HOST_ASSERT(topk_idx.dim() == 2);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());
        EP_HOST_ASSERT(topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(PyLong_CheckExact(remainder_seed.ptr()));
        EP_HOST_ASSERT(invocation_id >= 0);
        int expected_device = -1;
        if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
                expected_device, device_index, std::memory_order_relaxed) and
            expected_device != device_index)
            EP_HOST_UNREACHABLE(
                "Hybrid rail-balance planner supports one CUDA device per process");

        const auto num_tokens_i64 = topk_idx.size(0);
        const auto num_topk_i64 = topk_idx.size(1);
        EP_HOST_ASSERT(num_tokens_i64 >= 0 and num_tokens_i64 <= INT_MAX);
        EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
        const int num_tokens = static_cast<int>(num_tokens_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        const int num_rails = nccl_context->num_nvl_ranks;
        const int nvl_rank_idx = nccl_context->nvl_rank_idx;
        EP_HOST_ASSERT(num_rails >= 2 and num_rails <= 32);
        EP_HOST_ASSERT(nvl_rank_idx >= 0 and nvl_rank_idx < num_rails);
        EP_HOST_ASSERT(nccl_context->is_scaleup_nvlink);
        EP_HOST_ASSERT(nccl_context->num_scaleup_ranks == num_rails);
        EP_HOST_ASSERT(
            static_cast<int>(nccl_context->nvl_window_ptrs.size()) ==
            num_rails);
        EP_HOST_ASSERT(num_channels >= 1 and
                       num_channels <= rail_balance::kNumHybridMaxChannels);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(num_scaleout_ranks >= 2 and
                       num_scaleout_ranks <= 32);
        EP_HOST_ASSERT(local_scaleout_rank >= 0 and
                       local_scaleout_rank < num_scaleout_ranks);
        EP_HOST_ASSERT(proxy_capacity_per_egress >= num_tokens);
        EP_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
        EP_HOST_ASSERT(
            num_experts % (num_scaleout_ranks * num_rails) == 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(num_rails) * num_scaleout_ranks *
                num_max_tokens_per_rank <= INT_MAX);

        const int64_t remainder_seed_i64 = remainder_seed.cast<int64_t>();
        EP_HOST_ASSERT(remainder_seed_i64 >= 0);
        const int policy_value = parse_rail_balance_hybrid_policy(policy);
        const int threshold_percent_value =
            parse_rail_balance_hybrid_threshold_percent(threshold_percent);
        const int normalized_remainder_seed = static_cast<int>(
            remainder_seed_i64 % num_rails);

        // The force arena is an aligned tail of the symmetric GPU buffer. Its
        // offset is relative to ElasticBuffer::buffer, never to workspace.
        EP_HOST_ASSERT(arena_offset >= 0 and
                       arena_offset %
                           rail_balance::kNumHybridBufferAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        auto* arena = math::advance_ptr(buffer, arena_offset);
        const auto arena_layout = rail_balance::HybridArenaLayout(
            hidden, num_topk, proxy_capacity_per_egress, arena);
        EP_HOST_ASSERT(arena_layout.arena_bytes <=
                       num_gpu_buffer_bytes - arena_offset);
        const auto dispatch_layout = layout::TokenLayout(
            hidden * sizeof(__nv_bfloat16), 0, num_topk, true);
        const int shuffle_smem_bytes =
            dispatch_layout.get_num_bytes<false>() +
            ptx::kNumTMAAlignBytes;
        cudaDeviceProp device_prop{};
        CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(
            &device_prop, device_index));
        EP_HOST_ASSERT(
            shuffle_smem_bytes <= device_prop.sharedMemPerBlockOptin);
        EP_HOST_ASSERT(
            static_cast<int64_t>(nccl_context->rank_idx) *
                    num_max_tokens_per_rank +
                (num_tokens > 0 ? num_tokens - 1 : 0) <= INT_MAX);

        const c10::cuda::CUDAGuard device_guard(topk_idx.device());
        const auto compute_stream = at::cuda::getCurrentCUDAStream(device_index);
        const int invocation_key = ~invocation_id;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &arena_layout.get_control_ptr()->invocation_id,
            &invocation_key, sizeof(invocation_key), cudaMemcpyHostToDevice,
            comm_stream));
        const auto int_options = topk_idx.options().dtype(torch::kInt);

        // Allocate every Gate #2 output and build every private cubin before the
        // first arena store. Cold JIT/allocation failures therefore remain on
        // the noncollective side of WORLD Gate #1.
        auto outputs = allocate_rail_balance_hybrid_plan_outputs(
            int_options, num_rails, num_channels, num_scaleout_ranks);
        auto prepared = prepare_rail_balance_hybrid_plan(
            1, num_rails, num_channels, num_scaleout_ranks);
        auto local_barrier = prepare_rail_balance_hybrid_local_barrier(
            num_rails, num_gpu_timeout_cycles);
        auto source_shuffle =
            std::make_shared<PreparedRailBalanceHybridSourceShuffle>(
                prepare_rail_balance_hybrid_source_shuffle(
                    hidden, num_topk, num_channels, num_tokens, hop_aware));
        auto return_unshuffle =
            std::make_shared<PreparedRailBalanceHybridReturnUnshuffle>(
                prepare_rail_balance_hybrid_return_unshuffle(
                    hidden, num_topk, num_channels));
        auto combine_epilogue =
            std::make_shared<PreparedRailBalanceHybridCombineEpilogue>(
                prepare_rail_balance_hybrid_combine_epilogue(
                    hidden, num_max_tokens_per_rank, num_experts, num_topk,
                    num_scaleout_ranks, num_rails));

        // topk_idx and the zeros/full tensor initializers above belong to the
        // caller's current stream. Count and all later B2 work use comm_stream.
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        auto* local_channel_count = arena_layout.get_channel_count_ptr();
        RailBalanceHybridPlanRawPointers raw = {
            .channel_count = outputs.channel_count.data_ptr<int>(),
            .count = outputs.count.data_ptr<int>(),
            .quota = outputs.quota.data_ptr<int>(),
            .keep_count = outputs.keep_count.data_ptr<int>(),
            .segments = outputs.segments.data_ptr<int>(),
            .num_segments = outputs.num_segments.data_ptr<int>(),
            .owner_channel_prefix =
                outputs.owner_channel_prefix.data_ptr<int>(),
            .retained = outputs.retained.data_ptr<int>(),
            .moved = outputs.moved.data_ptr<int>(),
            .moved_channel_prefix =
                outputs.moved_channel_prefix.data_ptr<int>(),
            .group_prefix = outputs.group_prefix.data_ptr<int>(),
            .proxy_required = outputs.proxy_required.data_ptr<int>(),
            .moved_copies = outputs.moved_copies.data_ptr<int>(),
            .status = outputs.status.data_ptr<int>(),
            .local_channel_count = local_channel_count,
            .peer_channel_count = {},
        };
        for (int peer = 0; peer < num_rails; ++peer) {
            raw.peer_channel_count[peer] = static_cast<const int*>(
                nccl_context->get_sym_ptr(local_channel_count, peer));
        }
        std::optional<RailBalanceHopPlanState> hop;
        if (hop_aware) {
            const int64_t record_count =
                static_cast<int64_t>(num_max_tokens_per_rank) * num_topk;
            const int64_t record_bytes_i64 = rail_balance::checked_mul_i64(
                record_count,
                static_cast<int64_t>(sizeof(rail_balance::HopCopyRecord)));
            EP_HOST_ASSERT(record_bytes_i64 <=
                           arena_layout.get_hop_plan_record_capacity_bytes());
            const auto long_options = topk_idx.options().dtype(torch::kLong);
            auto records = torch::empty(
                {num_rails, num_max_tokens_per_rank, num_topk}, long_options);
            auto resolutions = torch::full(
                {num_rails, num_max_tokens_per_rank, num_topk, 4},
                -1, int_options);
            auto pair_load = torch::zeros(
                {num_scaleout_ranks, num_rails}, int_options);
            auto source_load = torch::zeros({num_rails}, int_options);
            auto owner_remaining = torch::zeros(
                {num_rails, num_scaleout_ranks}, int_options);
            auto path_units = torch::zeros({4}, int_options);
            auto* local_records = arena_layout.get_hop_plan_record_ptr();
            std::array<const rail_balance::HopCopyRecord*, 32>
                peer_records{};
            for (int peer = 0; peer < num_rails; ++peer) {
                peer_records[peer] = static_cast<
                    const rail_balance::HopCopyRecord*>(
                        nccl_context->get_sym_ptr(local_records, peer));
            }
            hop.emplace(RailBalanceHopPlanState{
                .records = std::move(records),
                .resolutions = std::move(resolutions),
                .pair_load = std::move(pair_load),
                .source_load = std::move(source_load),
                .owner_remaining = std::move(owner_remaining),
                .path_units = std::move(path_units),
                .local_records = local_records,
                .peer_records = peer_records,
                .record_bytes = static_cast<size_t>(record_bytes_i64),
                .prepared = prepare_rail_balance_hop_plan(num_channels),
            });
        }
        const int64_t active_count_values =
            static_cast<int64_t>(num_channels) * num_scaleout_ranks;
        EP_HOST_ASSERT(active_count_values > 0 and
                       active_count_values <= INT_MAX);
        const auto active_count_bytes = static_cast<size_t>(
            active_count_values * sizeof(int));
        const auto local_barrier_launch_args = jit::LaunchArgs(
            1, 512, 0, 1, true);
        if (hop_aware) {
            launch_prepared_rail_balance_hop_record(
                hop->prepared, topk_idx.data_ptr<topk_idx_t>(),
                hop->local_records, raw.status,
                num_tokens, num_max_tokens_per_rank, num_topk, num_channels,
                num_experts, num_scaleout_ranks, num_rails,
                local_scaleout_rank, comm_stream);
        } else {
            launch_prepared_rail_balance_hybrid_count(
                prepared, topk_idx.data_ptr<topk_idx_t>(),
                local_channel_count, raw.status,
                1, num_tokens, num_topk, num_channels,
                num_experts, num_scaleout_ranks,
                local_scaleout_rank, comm_stream);
        }

        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, outputs.status.data_ptr<int>(), sizeof(host_status),
            cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        EP_HOST_ASSERT(host_status == 0 or host_status == 2 or
                       host_status == 3);
        if (host_status != 0)
            return host_status;

        rail_balance_hybrid_plan_pending.emplace(
            RailBalanceHybridPlanPending{
                .invocation_id = invocation_id,
                .state = RailBalanceHybridPlanState::Preparing,
                .shuffled = false,
                .return_unshuffle_tested = false,
                .combine_epilogue_tested = false,
                .plan_status = -1,
                .num_rails = num_rails,
                .num_channels = num_channels,
                .num_destinations = num_scaleout_ranks,
                .local_destination = local_scaleout_rank,
                .num_tokens = num_tokens,
                .num_topk = num_topk,
                .hidden = hidden,
                .num_experts = num_experts,
                .num_max_tokens_per_rank = num_max_tokens_per_rank,
                .proxy_capacity_per_egress = proxy_capacity_per_egress,
                .normalized_remainder_seed = normalized_remainder_seed,
                .policy = policy_value,
                .threshold_percent = threshold_percent_value,
                .arena_offset = arena_offset,
                .active_count_values = active_count_values,
                .active_count_bytes = active_count_bytes,
                .arena = arena,
                .local_channel_count = local_channel_count,
                .raw = raw,
                .topk_idx = topk_idx,
                .prepared = std::move(prepared),
                .local_barrier_launch_args = local_barrier_launch_args,
                .local_barrier = std::move(local_barrier),
                .source_shuffle = std::move(source_shuffle),
                .return_unshuffle = std::move(return_unshuffle),
                .combine_epilogue = std::move(combine_epilogue),
                .dispatch_bundle = nullptr,
                .outputs = std::move(outputs),
                .hop = std::move(hop),
            });
        return 0;
    }

    RailBalanceHybridDispatchPrepareResult
    rail_balance_hybrid_dispatch_prepare(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const std::optional<torch::Tensor>&
            cumulative_local_expert_recv_stats,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_sms,
        const int& num_qps,
        const int& proxy_capacity_per_egress,
        const int64_t& arena_offset,
        const int& invocation_id,
        const pybind11::object& remainder_seed,
        const pybind11::object& policy,
        const pybind11::object& threshold_percent) {
        // This is the production-shaped, noncollective half of force dispatch.
        // It deliberately accepts no topology, local-rank, hidden, top-k, or
        // channel argument: all of those values are derived and frozen here.
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not rail_balance_hybrid_plan_pending.has_value());
        EP_HOST_ASSERT(allow_hybrid_mode);
        EP_HOST_ASSERT(allow_multiple_reduction);
        EP_HOST_ASSERT(num_cpu_buffer_bytes == 0);
        EP_HOST_ASSERT(invocation_id >= 0);
        EP_HOST_ASSERT(PyLong_CheckExact(remainder_seed.ptr()));

        EP_HOST_ASSERT(x.dim() == 2 and x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.get_device() == device_index);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_cuda() and
                       topk_idx.is_contiguous());
        EP_HOST_ASSERT(topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_weights.dim() == 2 and
                       topk_weights.is_cuda() and
                       topk_weights.is_contiguous());
        EP_HOST_ASSERT(topk_weights.get_device() == device_index);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);

        const auto num_tokens_i64 = x.size(0);
        const auto hidden_i64 = x.size(1);
        const auto num_topk_i64 = topk_idx.size(1);
        EP_HOST_ASSERT(num_tokens_i64 >= 0 and num_tokens_i64 <= INT_MAX);
        EP_HOST_ASSERT(hidden_i64 > 0 and hidden_i64 <= INT_MAX);
        EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens_i64);
        EP_HOST_ASSERT(topk_weights.size(0) == num_tokens_i64 and
                       topk_weights.size(1) == num_topk_i64);
        const int num_tokens = static_cast<int>(num_tokens_i64);
        const int hidden = static_cast<int>(hidden_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        EP_HOST_ASSERT(hidden % 256 == 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(hidden) * sizeof(nv_bfloat16) <= INT_MAX);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and
                       num_tokens <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(num_sms >= 2 and num_sms <= kNumMaxSMs);
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
        EP_HOST_ASSERT(num_qps > 0 and
                       num_qps <= nccl_context->num_allocated_qps);
        EP_HOST_ASSERT(proxy_capacity_per_egress >= num_tokens);

        const int num_destinations = nccl_context->num_scaleout_ranks;
        const int num_rails = nccl_context->num_scaleup_ranks;
        const int scaleout_rank_idx = nccl_context->scaleout_rank_idx;
        const int scaleup_rank_idx = nccl_context->scaleup_rank_idx;
        const int rank_idx = nccl_context->rank_idx;
        EP_HOST_ASSERT(num_destinations >= 2 and num_destinations <= 32);
        EP_HOST_ASSERT(num_rails >= 2 and num_rails <= 32);
        EP_HOST_ASSERT(nccl_context->is_scaleup_nvlink);
        EP_HOST_ASSERT(nccl_context->num_rdma_ranks == num_destinations);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == num_rails);
        EP_HOST_ASSERT(nccl_context->rdma_rank_idx == scaleout_rank_idx);
        EP_HOST_ASSERT(nccl_context->nvl_rank_idx == scaleup_rank_idx);
        EP_HOST_ASSERT(
            static_cast<int>(nccl_context->nvl_window_ptrs.size()) ==
            num_rails);
        EP_HOST_ASSERT(
            nccl_context->num_ranks == num_destinations * num_rails);
        EP_HOST_ASSERT(
            rank_idx == scaleout_rank_idx * num_rails + scaleup_rank_idx);
        EP_HOST_ASSERT(
            num_experts % (num_destinations * num_rails) == 0);
        const int num_local_experts =
            num_experts / (num_destinations * num_rails);
        EP_HOST_ASSERT(
            num_destinations * num_rails <=
            layout::WorkspaceLayout::kNumMaxRanks);
        EP_HOST_ASSERT(
            num_experts <= layout::WorkspaceLayout::kNumMaxExperts);
        EP_HOST_ASSERT(
            num_local_experts <=
            layout::WorkspaceLayout::kNumMaxExpertsPerRank);
        EP_HOST_ASSERT(
            static_cast<int64_t>(num_destinations) * num_rails *
                num_max_tokens_per_rank <= INT_MAX);
        EP_HOST_ASSERT(
            static_cast<int64_t>(num_destinations) * num_rails *
                num_max_tokens_per_rank *
                std::min(num_topk, num_local_experts) <= INT_MAX);
        EP_HOST_ASSERT(
            static_cast<int64_t>(rank_idx) * num_max_tokens_per_rank +
                (num_tokens > 0 ? num_tokens - 1 : 0) <= INT_MAX);

        int expected_device = -1;
        if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
                expected_device, device_index, std::memory_order_relaxed) and
            expected_device != device_index)
            EP_HOST_UNREACHABLE(
                "Hybrid rail-balance planner supports one CUDA device per process");

        const int num_smem_bytes =
            jit::device_runtime->get_num_smem_bytes();
        const int num_hidden_bytes =
            hidden * static_cast<int>(sizeof(nv_bfloat16));
        const auto dispatch_token_layout = get_dispatch_token_layout(
            hidden, sizeof(nv_bfloat16), 0, num_topk);
        const auto combine_token_layout = get_combine_token_layout(
            hidden, sizeof(nv_bfloat16), num_topk);
        const int notify_smem_bytes = get_num_notify_smem_bytes(
            num_destinations * num_rails, num_experts);
        EP_HOST_ASSERT(num_smem_bytes > notify_smem_bytes);
        int num_channels_per_sm = std::min<int>(
            (num_smem_bytes - notify_smem_bytes) /
                dispatch_token_layout.get_num_bytes<true>(),
            32 - kNumNotifyWarps);
        num_channels_per_sm = std::min<int>(
            num_smem_bytes /
                combine_token_layout.get_num_bytes<true>(),
            num_channels_per_sm);
        num_channels_per_sm = std::min<int>(
            num_channels_per_sm / 2, kNumMaxChannelsPerSM);
        if (not prefer_overlap_with_compute)
            num_channels_per_sm = std::min<int>(
                num_channels_per_sm, 4);
        EP_HOST_ASSERT(num_channels_per_sm > 0);
        const int64_t num_channels_i64 =
            static_cast<int64_t>(num_sms) * num_channels_per_sm;
        EP_HOST_ASSERT(
            num_channels_i64 > 0 and
            num_channels_i64 <= rail_balance::kNumHybridMaxChannels);
        const int num_channels = static_cast<int>(num_channels_i64);
        const int num_max_tokens_per_channel =
            math::ceil_div(num_max_tokens_per_rank, num_channels);

        const int64_t remainder_seed_i64 = remainder_seed.cast<int64_t>();
        EP_HOST_ASSERT(remainder_seed_i64 >= 0);
        const int policy_value = parse_rail_balance_hybrid_policy(policy);
        const int threshold_percent_value =
            parse_rail_balance_hybrid_threshold_percent(threshold_percent);
        const int normalized_remainder_seed = static_cast<int>(
            remainder_seed_i64 % num_rails);

        // Legacy dispatch/combine may use only the prefix before the appended
        // arena.  Checking against num_buffer_bytes would incorrectly allow a
        // legacy layout to overlap proxy payloads (or a future CPU segment).
        const int64_t legacy_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, 0, num_topk,
            sizeof(nv_bfloat16), num_destinations, num_rails, true);
        const int64_t legacy_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_destinations, num_rails, true, true);
        const int64_t expected_legacy_bytes = math::align<int64_t>(
            std::max(legacy_dispatch_bytes, legacy_combine_bytes),
            symmetric::kNumAlignmentBytes);
        EP_HOST_ASSERT(legacy_dispatch_bytes <= arena_offset);
        EP_HOST_ASSERT(legacy_combine_bytes <= arena_offset);
        EP_HOST_ASSERT(arena_offset == expected_legacy_bytes);
        EP_HOST_ASSERT(
            arena_offset >= 0 and
            arena_offset % rail_balance::kNumHybridBufferAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        auto* arena = math::advance_ptr(buffer, arena_offset);
        const auto arena_layout = rail_balance::HybridArenaLayout(
            hidden, num_topk, proxy_capacity_per_egress, arena);
        EP_HOST_ASSERT(
            arena_layout.arena_bytes <= num_gpu_buffer_bytes - arena_offset);
        EP_HOST_ASSERT(
            arena_offset + arena_layout.arena_bytes == num_gpu_buffer_bytes);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(device_index);
        const int invocation_key = ~invocation_id;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &arena_layout.get_control_ptr()->invocation_id,
            &invocation_key, sizeof(invocation_key), cudaMemcpyHostToDevice,
            comm_stream));
        const auto int_options = topk_idx.options().dtype(torch::kInt);

        // Allocate every plan/handle tensor before the first symmetric arena
        // store.  Exact receive tensors remain a documented post-main local
        // allocation because their leading dimension is not known yet.
        auto outputs = allocate_rail_balance_hybrid_plan_outputs(
            int_options, num_rails, num_channels, num_destinations);
        auto copied_topk_idx = torch::empty_like(topk_idx);
        auto psum_num_recv_tokens_per_scaleup_rank = torch::empty(
            {num_rails}, int_options);
        auto psum_num_recv_tokens_per_expert_storage = torch::empty(
            {num_local_experts + 1}, int_options);
        auto psum_num_recv_tokens_per_expert =
            psum_num_recv_tokens_per_expert_storage.slice(
                0, 1, num_local_experts + 1);
        auto num_unaligned_recv_tokens_per_expert = torch::empty(
            {num_local_experts}, int_options);
        auto dst_buffer_slot_idx = torch::empty(
            {num_channels, num_destinations,
             num_max_tokens_per_channel, num_topk}, int_options);
        auto token_metadata_at_forward = torch::empty(
            {num_channels,
             num_destinations * num_max_tokens_per_channel + 1,
             rail_balance::get_num_hybrid_forward_metadata_dims(num_topk)},
            int_options);
        auto channel_linked_list = torch::empty(
            {num_channels,
             num_destinations * num_max_tokens_per_channel + 1,
             num_rails}, int_options);
        auto cumulative_stats =
            cumulative_local_expert_recv_stats.value_or(torch::Tensor());
        if (cumulative_stats.defined()) {
            EP_HOST_ASSERT(cumulative_stats.dim() == 1 and
                           cumulative_stats.is_cuda() and
                           cumulative_stats.is_contiguous());
            EP_HOST_ASSERT(cumulative_stats.get_device() == device_index);
            EP_HOST_ASSERT(cumulative_stats.scalar_type() == torch::kInt32);
            EP_HOST_ASSERT(cumulative_stats.size(0) == num_local_experts);
        }

        // Cold JIT, layout validation, and every LaunchArgs derivation for the
        // whole round trip remain on the pre-Gate1 side.
        auto prepared_plan = prepare_rail_balance_hybrid_plan(
            1, num_rails, num_channels, num_destinations);
        auto local_barrier = prepare_rail_balance_hybrid_local_barrier(
            num_rails, num_gpu_timeout_cycles);
        auto source_shuffle =
            std::make_shared<PreparedRailBalanceHybridSourceShuffle>(
                prepare_rail_balance_hybrid_source_shuffle(
                    hidden, num_topk, num_channels, num_tokens));
        const RailBalanceHybridDispatchSpec dispatch_spec = {
            .num_sms = num_sms,
            .num_notify_warps = kNumNotifyWarps,
            .num_scaleout_warps = num_channels_per_sm,
            .num_forward_warps = num_channels_per_sm,
            .num_scaleout_ranks = num_destinations,
            .num_scaleup_ranks = num_rails,
            .num_hidden_bytes = num_hidden_bytes,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .num_experts = num_experts,
            .num_topk = num_topk,
            .expert_alignment = 1,
            .num_qps = num_qps,
            .num_timeout_cycles = num_gpu_timeout_cycles,
            .num_smem_bytes = num_smem_bytes,
            .proxy_capacity = proxy_capacity_per_egress,
        };
        auto main_dispatch =
            std::make_shared<PreparedRailBalanceHybridDispatch>(
                prepare_rail_balance_hybrid_dispatch(dispatch_spec));
        const int num_physical_sms = jit::device_runtime->get_num_sms();
        auto dispatch_epilogue =
            std::make_shared<PreparedRailBalanceHybridDispatchEpilogue>(
                prepare_rail_balance_hybrid_dispatch_epilogue(
                    hidden, num_max_tokens_per_rank, num_experts, num_topk,
                    num_destinations, num_rails, num_physical_sms,
                    num_channels));
        const RailBalanceHybridCombineSpec combine_spec = {
            .num_sms = num_sms,
            .num_scaleup_warps = num_channels_per_sm,
            .num_forward_warps = num_channels_per_sm,
            .num_scaleout_ranks = num_destinations,
            .num_scaleup_ranks = num_rails,
            .hidden = hidden,
            .num_max_tokens_per_rank = num_max_tokens_per_rank,
            .num_experts = num_experts,
            .num_topk = num_topk,
            .num_qps = num_qps,
            .num_timeout_cycles = num_gpu_timeout_cycles,
            .num_smem_bytes = num_smem_bytes,
        };
        auto main_combine =
            std::make_shared<PreparedRailBalanceHybridCombine>(
                prepare_rail_balance_hybrid_combine(combine_spec));
        auto return_unshuffle =
            std::make_shared<PreparedRailBalanceHybridReturnUnshuffle>(
                prepare_rail_balance_hybrid_return_unshuffle(
                    hidden, num_topk, num_channels));
        auto combine_epilogue =
            std::make_shared<PreparedRailBalanceHybridCombineEpilogue>(
                prepare_rail_balance_hybrid_combine_epilogue(
                    hidden, num_max_tokens_per_rank, num_experts, num_topk,
                    num_destinations, num_rails));

        auto* local_channel_count = arena_layout.get_channel_count_ptr();
        RailBalanceHybridPlanRawPointers plan_raw = {
            .channel_count = outputs.channel_count.data_ptr<int>(),
            .count = outputs.count.data_ptr<int>(),
            .quota = outputs.quota.data_ptr<int>(),
            .keep_count = outputs.keep_count.data_ptr<int>(),
            .segments = outputs.segments.data_ptr<int>(),
            .num_segments = outputs.num_segments.data_ptr<int>(),
            .owner_channel_prefix =
                outputs.owner_channel_prefix.data_ptr<int>(),
            .retained = outputs.retained.data_ptr<int>(),
            .moved = outputs.moved.data_ptr<int>(),
            .moved_channel_prefix =
                outputs.moved_channel_prefix.data_ptr<int>(),
            .group_prefix = outputs.group_prefix.data_ptr<int>(),
            .proxy_required = outputs.proxy_required.data_ptr<int>(),
            .moved_copies = outputs.moved_copies.data_ptr<int>(),
            .status = outputs.status.data_ptr<int>(),
            .local_channel_count = local_channel_count,
            .peer_channel_count = {},
        };
        for (int peer = 0; peer < num_rails; ++peer) {
            plan_raw.peer_channel_count[peer] = static_cast<const int*>(
                nccl_context->get_sym_ptr(local_channel_count, peer));
        }

        auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace, num_destinations, num_rails, num_experts);
        auto* host_scaleup_rank_count =
            host_workspace_layout.get_scaleup_rank_count_ptr<false>();
        auto* host_expert_count =
            host_workspace_layout.get_scaleup_expert_count_ptr<false>();
        std::fill_n(host_scaleup_rank_count, num_rails, 0);
        std::fill_n(host_expert_count, num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        RailBalanceHybridDispatchRawPointers dispatch_raw = {
            .x = x.data_ptr(),
            .topk_idx = topk_idx.data_ptr<topk_idx_t>(),
            .topk_weights = topk_weights.data_ptr<float>(),
            .copied_topk_idx = copied_topk_idx.data_ptr<topk_idx_t>(),
            .cumulative_local_expert_recv_stats =
                cumulative_stats.defined() ?
                    cumulative_stats.data_ptr<int>() : nullptr,
            .psum_num_recv_tokens_per_scaleup_rank =
                psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            .psum_num_recv_tokens_per_expert_storage =
                psum_num_recv_tokens_per_expert_storage.data_ptr<int>(),
            .psum_num_recv_tokens_per_expert_inclusive =
                psum_num_recv_tokens_per_expert_storage.data_ptr<int>() + 1,
            .num_unaligned_recv_tokens_per_expert =
                num_unaligned_recv_tokens_per_expert.data_ptr<int>(),
            .dst_buffer_slot_idx = dst_buffer_slot_idx.data_ptr<int>(),
            .token_metadata_at_forward =
                token_metadata_at_forward.data_ptr<int>(),
            .channel_linked_list = channel_linked_list.data_ptr<int>(),
            .buffer = buffer,
            .workspace = workspace,
            .mapped_host_workspace = mapped_host_workspace,
            .arena = arena,
            .proxy_dispatch_base = arena_layout
                .get_proxy_dispatch_layout(0).get_base_ptr(),
            .proxy_return_base = arena_layout
                .get_proxy_return_layout(0).get_base_ptr(),
            .legacy_reduce_buffer_base =
                get_rail_balance_hybrid_legacy_reduce_buffer_base(
                    *main_combine, buffer),
            .host_scaleup_rank_count = host_scaleup_rank_count,
            .host_expert_count = host_expert_count,
            .owner_channel_prefix = plan_raw.owner_channel_prefix,
            .keep_count = plan_raw.keep_count,
            .segments = plan_raw.segments,
            .num_segments = plan_raw.num_segments,
            .retained = plan_raw.retained,
            .moved = plan_raw.moved,
            .moved_channel_prefix = plan_raw.moved_channel_prefix,
            .group_prefix = plan_raw.group_prefix,
            .proxy_required = plan_raw.proxy_required,
            .status = plan_raw.status,
        };

        auto dispatch_bundle =
            std::make_shared<RailBalanceHybridDispatchBundle>(
                RailBalanceHybridDispatchBundle{
                    .x = x,
                    .topk_idx = topk_idx,
                    .topk_weights = topk_weights,
                    .cumulative_local_expert_recv_stats = cumulative_stats,
                    .copied_topk_idx = std::move(copied_topk_idx),
                    .psum_num_recv_tokens_per_scaleup_rank =
                        std::move(psum_num_recv_tokens_per_scaleup_rank),
                    .psum_num_recv_tokens_per_expert_storage =
                        std::move(psum_num_recv_tokens_per_expert_storage),
                    .psum_num_recv_tokens_per_expert =
                        std::move(psum_num_recv_tokens_per_expert),
                    .num_unaligned_recv_tokens_per_expert =
                        std::move(num_unaligned_recv_tokens_per_expert),
                    .dst_buffer_slot_idx =
                        std::move(dst_buffer_slot_idx),
                    .token_metadata_at_forward =
                        std::move(token_metadata_at_forward),
                    .channel_linked_list =
                        std::move(channel_linked_list),
                    .main_dispatch = std::move(main_dispatch),
                    .dispatch_epilogue = std::move(dispatch_epilogue),
                    .main_combine = std::move(main_combine),
                    .raw = dispatch_raw,
                    .dispatch_completion = std::nullopt,
                    .combine_completion = std::nullopt,
                    .compute_stream = compute_stream,
                    .num_tokens = num_tokens,
                    .hidden = hidden,
                    .num_topk = num_topk,
                    .num_max_tokens_per_rank = num_max_tokens_per_rank,
                    .num_experts = num_experts,
                    .num_local_experts = num_local_experts,
                    .num_rails = num_rails,
                    .num_destinations = num_destinations,
                    .num_channels = num_channels,
                    .num_channels_per_sm = num_channels_per_sm,
                    .num_max_tokens_per_channel =
                        num_max_tokens_per_channel,
                    .num_sms = num_sms,
                    .num_qps = num_qps,
                    .num_smem_bytes = num_smem_bytes,
                    .num_physical_sms = num_physical_sms,
                    .scaleout_rank_idx = scaleout_rank_idx,
                    .scaleup_rank_idx = scaleup_rank_idx,
                    .rank_idx = rank_idx,
                    .proxy_capacity_per_egress =
                        proxy_capacity_per_egress,
                    .arena_offset = arena_offset,
                    .arena_bytes = arena_layout.arena_bytes,
                });

        // All caller-stream initializers and allocations above are now ordered
        // before comm-stream count/finish/commit.  Event construction is
        // intentionally pre-Gate1 because it may throw.
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        const int64_t active_count_values =
            static_cast<int64_t>(num_channels) * num_destinations;
        EP_HOST_ASSERT(active_count_values > 0 and
                       active_count_values <= INT_MAX);
        const auto active_count_bytes = static_cast<size_t>(
            active_count_values * sizeof(int));
        const auto local_barrier_launch_args = jit::LaunchArgs(
            1, 512, 0, 1, true);

        // PREPARING owns every resource before the first symmetric arena
        // store.  A launch/synchronization exception is therefore recoverable
        // by the invocation-scoped precommit abort after WORLD Gate #1.
        rail_balance_hybrid_plan_pending.emplace(
            RailBalanceHybridPlanPending{
                .invocation_id = invocation_id,
                .state = RailBalanceHybridPlanState::Preparing,
                .shuffled = false,
                .return_unshuffle_tested = false,
                .combine_epilogue_tested = false,
                .plan_status = -1,
                .num_rails = num_rails,
                .num_channels = num_channels,
                .num_destinations = num_destinations,
                .local_destination = scaleout_rank_idx,
                .num_tokens = num_tokens,
                .num_topk = num_topk,
                .hidden = hidden,
                .num_experts = num_experts,
                .num_max_tokens_per_rank = num_max_tokens_per_rank,
                .proxy_capacity_per_egress = proxy_capacity_per_egress,
                .normalized_remainder_seed = normalized_remainder_seed,
                .policy = policy_value,
                .threshold_percent = threshold_percent_value,
                .arena_offset = arena_offset,
                .active_count_values = active_count_values,
                .active_count_bytes = active_count_bytes,
                .arena = arena,
                .local_channel_count = local_channel_count,
                .raw = plan_raw,
                .topk_idx = topk_idx,
                .prepared = std::move(prepared_plan),
                .local_barrier_launch_args = local_barrier_launch_args,
                .local_barrier = std::move(local_barrier),
                .source_shuffle = std::move(source_shuffle),
                .return_unshuffle = std::move(return_unshuffle),
                .combine_epilogue = std::move(combine_epilogue),
                .dispatch_bundle = std::move(dispatch_bundle),
                .outputs = std::move(outputs),
                .hop = std::nullopt,
            });
        auto& pending = rail_balance_hybrid_plan_pending.value();
        launch_prepared_rail_balance_hybrid_count(
            pending.prepared, pending.dispatch_bundle->raw.topk_idx,
            pending.raw.local_channel_count, pending.raw.status,
            1, num_tokens, num_topk, num_channels,
            num_experts, num_destinations, scaleout_rank_idx, comm_stream);

        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, pending.raw.status, sizeof(host_status),
            cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        EP_HOST_ASSERT(host_status == 0 or host_status == 2 or
                       host_status == 3);

        return {
            host_status,
            num_channels,
            num_channels_per_sm,
            num_destinations,
            num_rails,
            hidden,
            num_topk,
            num_max_tokens_per_rank,
            num_experts,
            proxy_capacity_per_egress,
            num_sms,
            dispatch_spec.num_notify_warps,
            dispatch_spec.num_scaleout_warps,
            dispatch_spec.num_forward_warps,
            num_smem_bytes,
            num_qps,
            num_gpu_timeout_cycles,
            arena_offset,
            arena_layout.arena_bytes,
            rail_balance::get_num_hybrid_forward_metadata_dims(num_topk),
            policy_value,
            threshold_percent_value,
        };
    }

    RailBalanceHybridPlanTensors rail_balance_hybrid_plan_finish(
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::Preparing or
            pending.state == RailBalanceHybridPlanState::PlanReady);
        if (pending.state == RailBalanceHybridPlanState::PlanReady)
            return pending.outputs.as_tuple();

        const c10::cuda::CUDAGuard device_guard(device_index);
        // WORLD Gate #1 is owned by Python. Once it succeeds, every local rank
        // consumes exactly one monotonic legacy-workspace barrier epoch.
        submit_prepared_rail_balance_hybrid_local_barrier(
            pending.local_barrier,
            pending.local_barrier_launch_args,
            nccl_context->dev_comm, nccl_context->window,
            workspace,
            pending.num_rails, nccl_context->nvl_rank_idx,
            num_gpu_timeout_cycles, comm_stream);

        if (pending.hop.has_value()) {
            auto& hop = *pending.hop;
            auto* records = reinterpret_cast<rail_balance::HopCopyRecord*>(
                hop.records.data_ptr<int64_t>());
            const int64_t record_stride =
                static_cast<int64_t>(pending.num_max_tokens_per_rank) *
                pending.num_topk;
            for (int owner = 0; owner < pending.num_rails; ++owner) {
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    records + owner * record_stride,
                    hop.peer_records[owner], hop.record_bytes,
                    cudaMemcpyDeviceToDevice, comm_stream));
            }
            launch_prepared_rail_balance_hop_one_hop_plan(
                hop.prepared, records,
                reinterpret_cast<rail_balance::HopCopyResolution*>(
                    hop.resolutions.data_ptr<int>()),
                hop.pair_load.data_ptr<int>(),
                hop.source_load.data_ptr<int>(),
                hop.owner_remaining.data_ptr<int>(),
                pending.raw.retained, pending.raw.moved,
                pending.raw.group_prefix, pending.raw.proxy_required,
                hop.path_units.data_ptr<int>(), pending.raw.moved_copies,
                pending.raw.status, pending.num_rails,
                pending.num_max_tokens_per_rank, pending.num_topk,
                pending.num_channels, pending.num_destinations,
                pending.num_max_tokens_per_rank,
                pending.proxy_capacity_per_egress,
                pending.normalized_remainder_seed, comm_stream);
        } else {
            auto* snapshot = pending.raw.channel_count;
            for (int owner = 0; owner < pending.num_rails; ++owner) {
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    snapshot + owner * pending.active_count_values,
                    pending.raw.peer_channel_count[owner],
                    pending.active_count_bytes,
                    cudaMemcpyDeviceToDevice, comm_stream));
            }

            launch_prepared_rail_balance_hybrid_plan(
                pending.prepared,
                pending.raw.channel_count,
                pending.raw.count,
                pending.raw.quota,
                pending.raw.keep_count,
                pending.raw.segments,
                pending.raw.num_segments,
                pending.num_rails, pending.num_channels,
                pending.num_destinations,
                pending.normalized_remainder_seed,
                pending.policy, pending.threshold_percent, comm_stream);
            launch_prepared_rail_balance_hybrid_prefix(
                pending.prepared,
                pending.raw.channel_count,
                pending.raw.quota,
                pending.raw.keep_count,
                pending.raw.owner_channel_prefix,
                pending.raw.retained,
                pending.raw.moved,
                pending.raw.moved_channel_prefix,
                pending.raw.group_prefix,
                pending.raw.proxy_required,
                pending.raw.moved_copies,
                pending.raw.status,
                pending.num_rails, pending.num_channels,
                pending.num_destinations,
                pending.num_max_tokens_per_rank,
                pending.proxy_capacity_per_egress, comm_stream);
        }

        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, pending.raw.status,
            sizeof(host_status), cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        EP_HOST_ASSERT(host_status == 0 or host_status == 1 or
                       (pending.hop.has_value() and host_status == 4));
        pending.plan_status = host_status;
        pending.state = RailBalanceHybridPlanState::PlanReady;
        return pending.outputs.as_tuple();
    }

    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor>
    rail_balance_hop_plan_snapshot(const int& invocation_id) {
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        const auto& pending = *rail_balance_hybrid_plan_pending;
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.hop.has_value());
        const auto& hop = *pending.hop;
        return {
            hop.records, hop.resolutions, hop.pair_load, hop.source_load,
            pending.outputs.retained, pending.outputs.moved,
            pending.outputs.group_prefix, pending.outputs.proxy_required,
            hop.path_units, pending.outputs.moved_copies,
            pending.outputs.status,
        };
    }

    void rail_balance_hybrid_dispatch_commit(const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(not pending.shuffled);
        EP_HOST_ASSERT(pending.source_shuffle != nullptr);
        EP_HOST_ASSERT(pending.dispatch_bundle != nullptr);
        auto& bundle = *pending.dispatch_bundle;
        EP_HOST_ASSERT(bundle.main_dispatch != nullptr);
        const c10::cuda::CUDAGuard device_guard(device_index);

        // Freeze the complete raw submission ABI before poisoning ownership.
        // From Invalid through the three adjacent submissions there may be no
        // Tensor access, validation, allocation, JIT, status readback, or
        // host synchronization. The local barrier is required because source
        // shuffle publishes into peer arenas: stream ordering protects the
        // local producer only, while an egress rank may otherwise consume a
        // peer proxy slot before its owner has finished writing it. All three
        // launch adapters are still allowed to throw;
        // in that case the transaction remains permanently Invalid.
        const auto& prepared_source = *pending.source_shuffle;
        const auto& prepared_local_barrier = pending.local_barrier;
        const auto local_barrier_launch_args =
            pending.local_barrier_launch_args;
        const auto& prepared_dispatch = *bundle.main_dispatch;
        const auto raw = bundle.raw;
        const auto nccl_dev_comm = nccl_context->dev_comm;
        const auto nccl_window = nccl_context->window;
        const int num_experts = bundle.num_experts;
        const int num_destinations = bundle.num_destinations;
        const int scaleout_rank_idx = bundle.scaleout_rank_idx;
        const int num_rails = bundle.num_rails;
        const int scaleup_rank_idx = bundle.scaleup_rank_idx;
        const int num_max_tokens_per_rank =
            bundle.num_max_tokens_per_rank;
        const int rank_idx = bundle.rank_idx;
        const int proxy_capacity_per_egress =
            bundle.proxy_capacity_per_egress;
        const int num_tokens = bundle.num_tokens;
        const int64_t timeout_cycles = num_gpu_timeout_cycles;

        pending.state = RailBalanceHybridPlanState::Invalid;
        submit_prepared_rail_balance_hybrid_source_shuffle(
            prepared_source,
            nccl_dev_comm, nccl_window,
            raw.x, raw.topk_idx, raw.topk_weights,
            raw.arena,
            raw.owner_channel_prefix, raw.keep_count,
            raw.segments, raw.num_segments,
            raw.retained, raw.moved_channel_prefix,
            raw.group_prefix, raw.proxy_required, raw.status,
            num_experts, num_destinations, scaleout_rank_idx,
            num_rails, scaleup_rank_idx,
            num_max_tokens_per_rank, rank_idx,
            proxy_capacity_per_egress, comm_stream);
        submit_prepared_rail_balance_hybrid_local_barrier(
            prepared_local_barrier, local_barrier_launch_args,
            nccl_dev_comm, nccl_window, raw.workspace,
            num_rails, scaleup_rank_idx, timeout_cycles, comm_stream);
        launch_prepared_rail_balance_hybrid_dispatch(
            prepared_dispatch,
            raw.x, nullptr,
            raw.topk_idx, raw.topk_weights, raw.copied_topk_idx,
            raw.cumulative_local_expert_recv_stats,
            raw.psum_num_recv_tokens_per_scaleup_rank,
            raw.psum_num_recv_tokens_per_expert_storage,
            raw.num_unaligned_recv_tokens_per_expert,
            raw.dst_buffer_slot_idx, raw.token_metadata_at_forward,
            num_tokens, 0, 0,
            nccl_dev_comm, nccl_window,
            raw.buffer, raw.workspace, raw.mapped_host_workspace,
            raw.arena,
            raw.retained, raw.moved,
            raw.group_prefix, raw.proxy_required,
            scaleout_rank_idx, scaleup_rank_idx, comm_stream);
        pending.shuffled = true;
        pending.state = RailBalanceHybridPlanState::DispatchLive;
    }

    RailBalanceHybridDispatchResult rail_balance_hybrid_dispatch_finish(
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::DispatchLive);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(pending.shuffled);
        EP_HOST_ASSERT(pending.dispatch_bundle != nullptr);
        auto& bundle = *pending.dispatch_bundle;
        EP_HOST_ASSERT(bundle.dispatch_epilogue != nullptr);
        EP_HOST_ASSERT(not bundle.dispatch_completion.has_value());
        EP_HOST_ASSERT(bundle.raw.host_scaleup_rank_count != nullptr);
        EP_HOST_ASSERT(bundle.raw.host_expert_count != nullptr);
        EP_HOST_ASSERT(bundle.raw.status != nullptr);
        EP_HOST_ASSERT(
            at::cuda::getCurrentCUDAStream(device_index).id() ==
            bundle.compute_stream.id());

        const auto& prepared_epilogue = *bundle.dispatch_epilogue;
        const auto raw = bundle.raw;
        const int num_local_experts = bundle.num_local_experts;
        const int hidden = bundle.hidden;
        const int num_topk = bundle.num_topk;
        const int scaleout_rank_idx = bundle.scaleout_rank_idx;
        const int scaleup_rank_idx = bundle.scaleup_rank_idx;
        const int64_t max_num_recv_tokens =
            static_cast<int64_t>(bundle.num_destinations) *
            bundle.num_rails * bundle.num_max_tokens_per_rank;
        const int64_t max_num_expanded_tokens =
            max_num_recv_tokens * std::min(num_topk, num_local_experts);

        // Main dispatch has already crossed the publication boundary. Every
        // fallible completion step below is therefore fail-closed; only a
        // fully owned, synchronized result may restore DispatchLive.
        pending.state = RailBalanceHybridPlanState::Invalid;
        const c10::cuda::CUDAGuard device_guard(device_index);

        int64_t num_recv_tokens_i64 = 0;
        int64_t num_expanded_tokens_i64 = 0;
        int counter_scaleup_rank_idx = 0;
        int counter_local_expert_idx = 0;
        std::vector<int> num_recv_tokens_per_expert_list;
        num_recv_tokens_per_expert_list.reserve(num_local_experts);
        const auto start_cpu_time =
            std::chrono::high_resolution_clock::now();
        while (true) {
            bool ready = true;
            while (counter_scaleup_rank_idx < bundle.num_rails and ready) {
                const int64_t encoded_count =
                    raw.host_scaleup_rank_count[counter_scaleup_rank_idx];
                EP_HOST_ASSERT(encoded_count != INT64_MIN);
                const int64_t count =
                    math::encode_decode_positive(encoded_count);
                if ((ready = math::is_decoded_positive_ready(count))) {
                    EP_HOST_ASSERT(
                        count >= 0 and
                        count <= max_num_recv_tokens - num_recv_tokens_i64);
                    num_recv_tokens_i64 += count;
                    ++counter_scaleup_rank_idx;
                }
            }
            while (counter_local_expert_idx < num_local_experts and ready) {
                const int64_t encoded_count =
                    raw.host_expert_count[counter_local_expert_idx];
                EP_HOST_ASSERT(encoded_count != INT64_MIN);
                const int64_t count =
                    math::encode_decode_positive(encoded_count);
                if ((ready = math::is_decoded_positive_ready(count))) {
                    EP_HOST_ASSERT(
                        count >= 0 and count <= INT_MAX and
                        count <= max_num_expanded_tokens -
                            num_expanded_tokens_i64);
                    num_recv_tokens_per_expert_list.push_back(
                        static_cast<int>(count));
                    num_expanded_tokens_i64 += count;
                    ++counter_local_expert_idx;
                }
            }
            if (ready)
                break;
            const auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    now - start_cpu_time).count() > num_cpu_timeout_secs)
                throw EPExceptionWithLineInfo(
                    "Rail-balance dispatch CPU wait",
                    "mapped receive counters did not become ready");
        }
        EP_HOST_ASSERT(num_recv_tokens_i64 <= INT_MAX);
        EP_HOST_ASSERT(num_expanded_tokens_i64 <= INT_MAX);
        const int num_recv_tokens =
            static_cast<int>(num_recv_tokens_i64);
        const int num_expanded_tokens =
            static_cast<int>(num_expanded_tokens_i64);

        auto recv_x = torch::empty(
            {num_recv_tokens, hidden}, bundle.x.options());
        auto recv_topk_idx = torch::empty(
            {num_recv_tokens, num_topk}, bundle.topk_idx.options());
        auto recv_topk_weights = torch::empty(
            {num_recv_tokens, num_topk}, bundle.topk_weights.options());
        auto recv_src_metadata = torch::empty(
            {num_recv_tokens, num_topk + 2},
            bundle.dst_buffer_slot_idx.options());

        bundle.dispatch_completion.emplace(
            RailBalanceHybridDispatchCompletion{
                .recv_x = std::move(recv_x),
                .recv_topk_idx = std::move(recv_topk_idx),
                .recv_topk_weights = std::move(recv_topk_weights),
                .recv_src_metadata = std::move(recv_src_metadata),
                .num_recv_tokens = num_recv_tokens,
                .num_expanded_tokens = num_expanded_tokens,
                .num_recv_tokens_per_expert_list =
                    std::move(num_recv_tokens_per_expert_list),
                .raw = {},
            });
        auto& completion = bundle.dispatch_completion.value();
        completion.raw = {
            .recv_x = completion.recv_x.data_ptr(),
            .recv_topk_idx =
                completion.recv_topk_idx.data_ptr<topk_idx_t>(),
            .recv_topk_weights =
                completion.recv_topk_weights.data_ptr<float>(),
            .recv_src_metadata =
                completion.recv_src_metadata.data_ptr<int>(),
        };

        launch_prepared_rail_balance_hybrid_dispatch_epilogue(
            prepared_epilogue,
            raw.buffer, raw.workspace,
            raw.psum_num_recv_tokens_per_scaleup_rank,
            raw.psum_num_recv_tokens_per_expert_inclusive,
            completion.raw.recv_x,
            completion.raw.recv_topk_idx,
            completion.raw.recv_topk_weights,
            completion.raw.recv_src_metadata,
            raw.channel_linked_list,
            num_recv_tokens,
            scaleout_rank_idx, scaleup_rank_idx, comm_stream);

        // This is the first safe synchronization after H4c's adjacent
        // source/main commit. It observes source status and turns every
        // asynchronous source, main, or epilogue failure into sticky Invalid.
        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, raw.status, sizeof(host_status),
            cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        pending.plan_status = host_status;
        EP_HOST_ASSERT(host_status == 0);

        RailBalanceHybridDispatchResult result = {
            completion.recv_x,
            std::nullopt,
            completion.recv_topk_idx,
            completion.recv_topk_weights,
            bundle.copied_topk_idx,
            completion.num_recv_tokens,
            completion.num_expanded_tokens,
            completion.num_recv_tokens_per_expert_list,
            bundle.psum_num_recv_tokens_per_scaleup_rank,
            bundle.psum_num_recv_tokens_per_expert,
            bundle.num_unaligned_recv_tokens_per_expert,
            completion.recv_src_metadata,
            bundle.dst_buffer_slot_idx,
            bundle.token_metadata_at_forward,
            bundle.channel_linked_list,
            std::nullopt,
        };
        pending.state = RailBalanceHybridPlanState::DispatchLive;
        return result;
    }

    void rail_balance_hybrid_combine_prepare(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::DispatchLive);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(pending.shuffled);
        EP_HOST_ASSERT(pending.dispatch_bundle != nullptr);
        auto& bundle = *pending.dispatch_bundle;
        EP_HOST_ASSERT(bundle.dispatch_completion.has_value());
        EP_HOST_ASSERT(not bundle.combine_completion.has_value());
        EP_HOST_ASSERT(bundle.main_combine != nullptr);
        EP_HOST_ASSERT(pending.return_unshuffle != nullptr);
        EP_HOST_ASSERT(pending.local_barrier != nullptr);
        EP_HOST_ASSERT(pending.combine_epilogue != nullptr);
        EP_HOST_ASSERT(bundle.raw.buffer != nullptr);
        EP_HOST_ASSERT(bundle.raw.workspace != nullptr);
        EP_HOST_ASSERT(bundle.raw.arena != nullptr);
        EP_HOST_ASSERT(bundle.raw.proxy_return_base != nullptr);
        EP_HOST_ASSERT(bundle.raw.legacy_reduce_buffer_base != nullptr);
        EP_HOST_ASSERT(
            bundle.raw.psum_num_recv_tokens_per_scaleup_rank != nullptr);
        EP_HOST_ASSERT(bundle.raw.token_metadata_at_forward != nullptr);
        EP_HOST_ASSERT(bundle.raw.channel_linked_list != nullptr);
        EP_HOST_ASSERT(bundle.raw.moved != nullptr);
        EP_HOST_ASSERT(bundle.raw.group_prefix != nullptr);
        EP_HOST_ASSERT(bundle.raw.proxy_required != nullptr);
        EP_HOST_ASSERT(bundle.raw.status != nullptr);
        const auto& dispatch_completion =
            bundle.dispatch_completion.value();
        EP_HOST_ASSERT(dispatch_completion.num_recv_tokens >= 0);
        EP_HOST_ASSERT(
            dispatch_completion.num_recv_tokens == 0 or
            dispatch_completion.raw.recv_src_metadata != nullptr);
        EP_HOST_ASSERT(
            bundle.num_tokens == 0 or
            bundle.raw.copied_topk_idx != nullptr);

        EP_HOST_ASSERT(x.dim() == 2 and x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.get_device() == device_index);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(
            x.size(0) == dispatch_completion.num_recv_tokens and
            x.size(1) == bundle.hidden);
        EP_HOST_ASSERT(topk_weights.has_value());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and
                       topk_weights->is_cuda() and
                       topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->get_device() == device_index);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        EP_HOST_ASSERT(
            topk_weights->size(0) ==
                dispatch_completion.num_recv_tokens and
            topk_weights->size(1) == bundle.num_topk);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(device_index);
        auto combined_x = torch::empty(
            {bundle.num_tokens, bundle.hidden}, x.options());
        auto combined_topk_weights = torch::empty(
            {bundle.num_tokens, bundle.num_topk},
            topk_weights->options());

        RailBalanceHybridCombineCompletion completion = {
            .x = x,
            .topk_weights = topk_weights,
            .combined_x = std::move(combined_x),
            .combined_topk_weights = combined_topk_weights,
            .raw = {},
        };
        completion.raw = {
            .x = completion.x.data_ptr(),
            .topk_weights = completion.topk_weights.has_value() ?
                completion.topk_weights->data_ptr<float>() : nullptr,
            .combined_x = completion.combined_x.data_ptr(),
            .combined_topk_weights =
                completion.combined_topk_weights.has_value() ?
                    completion.combined_topk_weights->data_ptr<float>() :
                    nullptr,
        };
        EP_HOST_ASSERT(
            dispatch_completion.num_recv_tokens == 0 or
            (completion.raw.x != nullptr and
             completion.raw.topk_weights != nullptr));
        EP_HOST_ASSERT(
            bundle.num_tokens == 0 or
            (completion.raw.combined_x != nullptr and
             completion.raw.combined_topk_weights != nullptr));
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        // Installation is the final prepare operation. A future WORLD
        // combine gate may now either commit this exact owner or abort it.
        bundle.combine_completion.emplace(std::move(completion));
    }

    void rail_balance_hybrid_combine_abort(const int& invocation_id) {
        if (destroyed or not rail_balance_hybrid_plan_pending.has_value())
            return;
        auto& pending = rail_balance_hybrid_plan_pending.value();
        if (pending.invocation_id != invocation_id or
            pending.state != RailBalanceHybridPlanState::DispatchLive or
            pending.dispatch_bundle == nullptr)
            return;
        auto& bundle = *pending.dispatch_bundle;
        if (not bundle.combine_completion.has_value())
            return;
        bundle.combine_completion.reset();
    }

    RailBalanceHybridCombineResult rail_balance_hybrid_combine_commit(
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::DispatchLive);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(pending.shuffled);
        EP_HOST_ASSERT(pending.dispatch_bundle != nullptr);
        auto& bundle = *pending.dispatch_bundle;
        EP_HOST_ASSERT(bundle.dispatch_completion.has_value());
        EP_HOST_ASSERT(bundle.combine_completion.has_value());
        EP_HOST_ASSERT(bundle.main_combine != nullptr);
        EP_HOST_ASSERT(pending.return_unshuffle != nullptr);
        EP_HOST_ASSERT(pending.local_barrier != nullptr);
        EP_HOST_ASSERT(pending.combine_epilogue != nullptr);
        EP_HOST_ASSERT(bundle.raw.legacy_reduce_buffer_base != nullptr);
        EP_HOST_ASSERT(bundle.raw.status != nullptr);

        const auto& dispatch_completion =
            bundle.dispatch_completion.value();
        const auto& combine_completion =
            bundle.combine_completion.value();
        const auto& prepared_combine = *bundle.main_combine;
        const auto& prepared_return_unshuffle =
            *pending.return_unshuffle;
        const auto prepared_local_barrier = pending.local_barrier;
        const auto local_barrier_launch_args =
            pending.local_barrier_launch_args;
        const auto& prepared_combine_epilogue =
            *pending.combine_epilogue;
        const auto raw = bundle.raw;
        const auto dispatch_completion_raw = dispatch_completion.raw;
        const auto combine_completion_raw = combine_completion.raw;
        const auto nccl_dev_comm = nccl_context->dev_comm;
        const auto nccl_window = nccl_context->window;
        const int num_reduced_tokens =
            dispatch_completion.num_recv_tokens;
        const int num_combined_tokens = bundle.num_tokens;
        const int num_experts = bundle.num_experts;
        const int num_destinations = bundle.num_destinations;
        const int num_rails = bundle.num_rails;
        const int scaleout_rank_idx = bundle.scaleout_rank_idx;
        const int scaleup_rank_idx = bundle.scaleup_rank_idx;
        const int rank_idx = bundle.rank_idx;
        const int num_max_tokens_per_rank =
            bundle.num_max_tokens_per_rank;
        const int proxy_capacity_per_egress =
            bundle.proxy_capacity_per_egress;
        const int64_t timeout_cycles = num_gpu_timeout_cycles;
        const c10::cuda::CUDAGuard device_guard(device_index);

        // No host work may split this committed sequence. Any submission or
        // synchronization failure leaves the complete owner installed and the
        // transaction permanently Invalid.
        pending.state = RailBalanceHybridPlanState::Invalid;
        launch_prepared_rail_balance_hybrid_combine(
            prepared_combine,
            combine_completion_raw.x,
            combine_completion_raw.topk_weights,
            dispatch_completion_raw.recv_src_metadata,
            raw.psum_num_recv_tokens_per_scaleup_rank,
            raw.token_metadata_at_forward,
            raw.channel_linked_list,
            nccl_dev_comm, nccl_window,
            raw.buffer, raw.workspace,
            raw.proxy_return_base,
            scaleout_rank_idx, scaleup_rank_idx,
            num_reduced_tokens, comm_stream);
        submit_prepared_rail_balance_hybrid_return_unshuffle(
            prepared_return_unshuffle,
            nccl_dev_comm, nccl_window,
            raw.arena, raw.legacy_reduce_buffer_base,
            raw.moved, raw.group_prefix, raw.proxy_required, raw.status,
            num_experts, num_destinations, num_rails,
            scaleup_rank_idx, rank_idx,
            num_max_tokens_per_rank,
            proxy_capacity_per_egress, comm_stream);
        submit_prepared_rail_balance_hybrid_local_barrier(
            prepared_local_barrier, local_barrier_launch_args,
            nccl_dev_comm, nccl_window, raw.workspace,
            num_rails, scaleup_rank_idx, timeout_cycles, comm_stream);
        submit_prepared_rail_balance_hybrid_combine_epilogue(
            prepared_combine_epilogue,
            combine_completion_raw.combined_x,
            combine_completion_raw.combined_topk_weights,
            raw.copied_topk_idx, raw.legacy_reduce_buffer_base,
            num_combined_tokens,
            scaleout_rank_idx, scaleup_rank_idx, comm_stream);

        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, raw.status, sizeof(host_status),
            cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        pending.plan_status = host_status;
        EP_HOST_ASSERT(host_status == 0);

        RailBalanceHybridCombineResult result = {
            combine_completion.combined_x,
            combine_completion.combined_topk_weights,
            std::nullopt,
        };
        rail_balance_hybrid_plan_pending.reset();
        return result;
    }

    void rail_balance_hybrid_source_shuffle(
        const torch::Tensor& x,
        const torch::Tensor& topk_weights,
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(not pending.shuffled);

        EP_HOST_ASSERT(x.dim() == 2 and x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.get_device() == device_index);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(x.size(0) == pending.num_tokens and
                       x.size(1) == pending.hidden);
        EP_HOST_ASSERT(topk_weights.dim() == 2 and
                       topk_weights.is_cuda() and
                       topk_weights.is_contiguous());
        EP_HOST_ASSERT(topk_weights.get_device() == device_index);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat);
        EP_HOST_ASSERT(topk_weights.size(0) == pending.num_tokens and
                       topk_weights.size(1) == pending.num_topk);
        EP_HOST_ASSERT(pending.topk_idx.defined() and
                       pending.topk_idx.is_cuda() and
                       pending.topk_idx.is_contiguous());
        EP_HOST_ASSERT(pending.topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(pending.topk_idx.size(0) == pending.num_tokens and
                       pending.topk_idx.size(1) == pending.num_topk);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(device_index);
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        int host_status = 0;
        try {
            if (pending.hop.has_value()) {
                const auto& hop = *pending.hop;
                launch_prepared_rail_balance_hop_source_shuffle(
                    *pending.source_shuffle,
                    nccl_context->dev_comm, nccl_context->window,
                    x.data_ptr(), pending.topk_idx.data_ptr<topk_idx_t>(),
                    topk_weights.data_ptr<float>(), pending.arena,
                    reinterpret_cast<const rail_balance::HopCopyRecord*>(
                        hop.records.data_ptr<int64_t>()),
                    reinterpret_cast<
                        const rail_balance::HopCopyResolution*>(
                            hop.resolutions.data_ptr<int>()),
                    pending.raw.retained, pending.raw.group_prefix,
                    pending.raw.proxy_required, pending.raw.status,
                    pending.num_experts, pending.num_destinations,
                    pending.local_destination, pending.num_rails,
                    nccl_context->nvl_rank_idx,
                    pending.num_max_tokens_per_rank,
                    nccl_context->rank_idx,
                    pending.proxy_capacity_per_egress, comm_stream);
            } else {
                launch_prepared_rail_balance_hybrid_source_shuffle(
                    *pending.source_shuffle,
                    nccl_context->dev_comm, nccl_context->window,
                    x.data_ptr(), pending.topk_idx.data_ptr<topk_idx_t>(),
                    topk_weights.data_ptr<float>(), pending.arena,
                    pending.outputs, pending.hidden, pending.num_topk,
                    pending.num_tokens, pending.num_experts,
                    pending.num_destinations, pending.local_destination,
                    pending.num_rails, nccl_context->nvl_rank_idx,
                    pending.num_channels, pending.num_max_tokens_per_rank,
                    nccl_context->rank_idx,
                    pending.proxy_capacity_per_egress, comm_stream);
            }
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                &host_status, pending.outputs.status.data_ptr<int>(),
                sizeof(host_status), cudaMemcpyDeviceToHost, comm_stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        } catch (...) {
            // A launch/synchronization failure may follow partial peer
            // publication. Keep the transaction for explicit abort, but make
            // a second shuffle impossible.
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw;
        }
        if (host_status != 0 and host_status != 2 and host_status != 3 and
            host_status != 4) {
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw EPExceptionWithLineInfo(
                "Rail balance Hybrid source shuffle",
                "device returned an unknown source-shuffle status");
        }
        if (host_status != 0) {
            pending.plan_status = host_status;
            throw EPExceptionWithLineInfo(
                "Rail balance Hybrid source shuffle",
                host_status == 2 ?
                    "topk_idx contains an out-of-range expert id" :
                host_status == 3 ?
                    "topk_idx contains duplicate expert ids within one token" :
                    "compact plan contains an invalid source-shuffle route");
        }
        pending.shuffled = true;
    }

    torch::Tensor rail_balance_hybrid_return_unshuffle_test(
        const torch::Tensor& proxy_return_bytes,
        const torch::Tensor& reduce_seed_bytes,
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(pending.shuffled);
        EP_HOST_ASSERT(not pending.return_unshuffle_tested);

        const auto arena_layout = rail_balance::HybridArenaLayout(
            pending.hidden, pending.num_topk,
            pending.proxy_capacity_per_egress, pending.arena);
        const int num_reduce_rows =
            pending.num_destinations <= pending.num_topk ?
                pending.num_destinations : pending.num_topk;
        const int64_t num_reduce_records = rail_balance::checked_mul_i64(
            num_reduce_rows, pending.num_max_tokens_per_rank);
        const int64_t proxy_return_num_bytes =
            rail_balance::checked_mul_i64(
                pending.proxy_capacity_per_egress,
                arena_layout.combine_token_bytes);
        const int64_t reduce_num_bytes = rail_balance::checked_mul_i64(
            num_reduce_records, arena_layout.combine_token_bytes);

        EP_HOST_ASSERT(proxy_return_bytes.dim() == 2 and
                       proxy_return_bytes.is_cuda() and
                       proxy_return_bytes.is_contiguous());
        EP_HOST_ASSERT(proxy_return_bytes.get_device() == device_index);
        EP_HOST_ASSERT(proxy_return_bytes.scalar_type() == torch::kUInt8);
        EP_HOST_ASSERT(
            proxy_return_bytes.size(0) ==
                pending.proxy_capacity_per_egress and
            proxy_return_bytes.size(1) ==
                arena_layout.combine_token_bytes);
        EP_HOST_ASSERT(reduce_seed_bytes.dim() == 2 and
                       reduce_seed_bytes.is_cuda() and
                       reduce_seed_bytes.is_contiguous());
        EP_HOST_ASSERT(reduce_seed_bytes.get_device() == device_index);
        EP_HOST_ASSERT(reduce_seed_bytes.scalar_type() == torch::kUInt8);
        EP_HOST_ASSERT(reduce_seed_bytes.size(0) == num_reduce_records and
                       reduce_seed_bytes.size(1) ==
                           arena_layout.combine_token_bytes);
        EP_HOST_ASSERT(proxy_return_bytes.nbytes() == proxy_return_num_bytes);
        EP_HOST_ASSERT(reduce_seed_bytes.nbytes() == reduce_num_bytes);
        // This direct single-node harness intentionally treats buffer base as
        // the legacy scaleout-reduce base. Keep both the seed and its peer
        // writes strictly before the force arena appended at arena_offset.
        EP_HOST_ASSERT(reduce_num_bytes <= pending.arena_offset);
        EP_HOST_ASSERT(reduce_num_bytes <= num_gpu_buffer_bytes);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(device_index);

        // Every allocation and JIT build precedes local barrier B1. The
        // prepared runtime lives in pending; this is the only snapshot needed.
        auto reduce_snapshot = torch::empty(
            reduce_seed_bytes.sizes(), reduce_seed_bytes.options());
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        // Capture every raw pointer before B1. The fixed committed sequence
        // below must not enter Tensor accessors or construct another layout.
        auto* proxy_return_base = arena_layout
            .get_proxy_return_layout(0).get_base_ptr();
        const auto* proxy_return_input = proxy_return_bytes.data_ptr();
        const auto* reduce_seed = reduce_seed_bytes.data_ptr();
        auto* reduce_snapshot_ptr = reduce_snapshot.data_ptr();
        const auto* moved = pending.outputs.moved.data_ptr<int>();
        const auto* group_prefix =
            pending.outputs.group_prefix.data_ptr<int>();
        const auto* proxy_required =
            pending.outputs.proxy_required.data_ptr<int>();
        auto* status = pending.outputs.status.data_ptr<int>();

        // Mark the adapter one-shot before queueing either copy. A partial
        // enqueue must never be replayed. Device status is deliberately not
        // observed until both local barriers have completed on every rank.
        pending.return_unshuffle_tested = true;
        int host_status = 0;
        try {
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                proxy_return_base, proxy_return_input,
                static_cast<size_t>(proxy_return_num_bytes),
                cudaMemcpyDeviceToDevice, comm_stream));
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                buffer, reduce_seed,
                static_cast<size_t>(reduce_num_bytes),
                cudaMemcpyDeviceToDevice, comm_stream));

            // B1: all local proxy-return and reduce seeds are visible before
            // any egress starts peer writes.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.local_barrier,
                nccl_context->dev_comm, nccl_context->window,
                workspace,
                pending.num_rails, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            submit_prepared_rail_balance_hybrid_return_unshuffle(
                *pending.return_unshuffle,
                nccl_context->dev_comm, nccl_context->window,
                pending.arena, buffer,
                moved, group_prefix, proxy_required, status,
                pending.num_experts,
                pending.num_destinations, pending.num_rails,
                nccl_context->nvl_rank_idx, nccl_context->rank_idx,
                pending.num_max_tokens_per_rank,
                pending.proxy_capacity_per_egress, comm_stream);

            // B4: even a rank whose kernel published a sticky error
            // participates; only after this cross-GPU visibility point may
            // any host inspect it.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.local_barrier,
                nccl_context->dev_comm, nccl_context->window,
                workspace,
                pending.num_rails, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                reduce_snapshot_ptr, buffer,
                static_cast<size_t>(reduce_num_bytes),
                cudaMemcpyDeviceToDevice, comm_stream));
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                &host_status, status,
                sizeof(host_status), cudaMemcpyDeviceToHost, comm_stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        } catch (...) {
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw;
        }
        if (host_status != 0 and host_status != 1 and host_status != 2 and
            host_status != 3 and host_status != 4) {
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw EPExceptionWithLineInfo(
                "Rail balance Hybrid return unshuffle",
                "device returned an unknown return-unshuffle status");
        }
        if (host_status != 0) {
            pending.plan_status = host_status;
            throw EPExceptionWithLineInfo(
                "Rail balance Hybrid return unshuffle",
                host_status == 1 ?
                    "proxy-return capacity is invalid" :
                host_status == 2 ?
                    "preserved dispatch contains an out-of-range expert id" :
                    "preserved dispatch or compact schedule is invalid");
        }
        return reduce_snapshot;
    }

    std::tuple<torch::Tensor, torch::Tensor>
    rail_balance_hybrid_combine_epilogue_test(
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(pending.shuffled);
        EP_HOST_ASSERT(pending.return_unshuffle_tested);
        EP_HOST_ASSERT(not pending.combine_epilogue_tested);
        EP_HOST_ASSERT(pending.combine_epilogue != nullptr);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(device_index);
        auto combined_x = torch::empty(
            {pending.num_tokens, pending.hidden},
            pending.topk_idx.options().dtype(torch::kBFloat16));
        auto combined_topk_weights = torch::empty(
            pending.topk_idx.sizes(),
            pending.topk_idx.options().dtype(torch::kFloat32));
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);

        // This hook is intentionally one-shot. A launch failure may leave a
        // partial output, so retrying the same transaction is not supported.
        pending.combine_epilogue_tested = true;
        try {
            launch_prepared_rail_balance_hybrid_combine_epilogue(
                *pending.combine_epilogue,
                combined_x.data_ptr<c10::BFloat16>(),
                combined_topk_weights.data_ptr<float>(),
                pending.topk_idx.data_ptr<topk_idx_t>(),
                buffer, pending.num_tokens,
                0, nccl_context->nvl_rank_idx, comm_stream);
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        } catch (...) {
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw;
        }
        return {combined_x, combined_topk_weights};
    }

    torch::Tensor rail_balance_hybrid_proxy_dispatch_snapshot(
        const int& invocation_id) {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_plan_pending.has_value());
        auto& pending = rail_balance_hybrid_plan_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(
            pending.state == RailBalanceHybridPlanState::PlanReady);
        EP_HOST_ASSERT(pending.plan_status == 0);

        const c10::cuda::CUDAGuard device_guard(device_index);
        const auto stream = at::cuda::getCurrentCUDAStream(device_index);
        const auto arena_layout = rail_balance::HybridArenaLayout(
            pending.hidden, pending.num_topk,
            pending.proxy_capacity_per_egress, pending.arena);
        auto snapshot = torch::empty(
            {pending.proxy_capacity_per_egress,
             arena_layout.dispatch_token_bytes},
            pending.topk_idx.options().dtype(torch::kUInt8));
        const auto num_bytes = rail_balance::checked_mul_i64(
            pending.proxy_capacity_per_egress,
            arena_layout.dispatch_token_bytes);
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            snapshot.data_ptr(),
            arena_layout.get_proxy_dispatch_layout(0).get_base_ptr(),
            static_cast<size_t>(num_bytes),
            cudaMemcpyDeviceToDevice, stream));
        return snapshot;
    }

    void rail_balance_hybrid_plan_abort(const int& invocation_id) {
        // Idempotent and transaction-specific: a stale abort may not clear a
        // newer owner. In particular, never reset the shared legacy barrier
        // workspace; its phase is monotonic across all EP operations.
        if (rail_balance_hybrid_plan_pending.has_value() and
            rail_balance_hybrid_plan_pending->invocation_id == invocation_id) {
            const auto state = rail_balance_hybrid_plan_pending->state;
            if (state == RailBalanceHybridPlanState::Preparing or
                state == RailBalanceHybridPlanState::PlanReady) {
                // Preparing owns pointers already visible to comm_stream.  A
                // failed enqueue/readback may leave count work in flight, so
                // quiesce it before releasing the backing tensors.  Preserve
                // fail-close cleanup even when CUDA reports an async error.
                const auto stream_status =
                    cudaStreamSynchronize(comm_stream);
                rail_balance_hybrid_plan_pending.reset();
                CUDA_RUNTIME_CHECK(stream_status);
            } else if (state == RailBalanceHybridPlanState::DispatchLive)
                rail_balance_hybrid_plan_pending->state =
                    RailBalanceHybridPlanState::Invalid;
        }
    }

    RailBalanceHybridVNodePrepareTensors rail_balance_hybrid_vnode_prepare(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& proxy_dispatch,
        const torch::Tensor& channel_count,
        const int64_t& arena_offset,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_destinations,
        const int& num_source_ranks,
        const int& proxy_capacity,
        const int& generation,
        const int& invocation_id,
        const pybind11::object& remainder_seed,
        const pybind11::object& policy,
        const pybind11::object& threshold_percent) {
        constexpr int kWorldRanks = 8;
        constexpr int64_t kArenaGuardBytes = 4096;

        // This bridge is a private, pure-single-node proof.  The real Hybrid
        // buffer and its public ABI remain untouched.
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(not allow_hybrid_mode);
        EP_HOST_ASSERT(not rail_balance_hybrid_vnode_pending.has_value());
        EP_HOST_ASSERT(nccl_context->num_ranks == kWorldRanks);
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1);
        EP_HOST_ASSERT(nccl_context->num_scaleup_ranks == kWorldRanks);
        EP_HOST_ASSERT(nccl_context->num_rdma_ranks == 1);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == kWorldRanks);
        EP_HOST_ASSERT(nccl_context->is_scaleup_nvlink);
        EP_HOST_ASSERT(nccl_context->scaleup_rank_idx ==
                       nccl_context->rank_idx);
        EP_HOST_ASSERT(nccl_context->nvl_rank_idx == nccl_context->rank_idx);

        EP_HOST_ASSERT(num_source_ranks >= 2 and num_source_ranks <= 32);
        EP_HOST_ASSERT(num_destinations >= 2 and
                       num_destinations <=
                           rail_balance::kNumHybridMaxDestinations);
        EP_HOST_ASSERT(
            static_cast<int64_t>(num_source_ranks) * num_destinations ==
            kWorldRanks);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(proxy_capacity > 0);
        EP_HOST_ASSERT(generation > 0);
        EP_HOST_ASSERT(invocation_id >= 0);
        EP_HOST_ASSERT(num_experts > 0 and
                       num_experts %
                           (num_source_ranks * num_destinations) == 0);
        EP_HOST_ASSERT(PyLong_CheckExact(remainder_seed.ptr()));
        const int64_t remainder_seed_i64 = remainder_seed.cast<int64_t>();
        EP_HOST_ASSERT(remainder_seed_i64 >= 0);
        const int normalized_remainder_seed = static_cast<int>(
            remainder_seed_i64 % num_source_ranks);
        const int policy_value = parse_rail_balance_hybrid_policy(policy);
        const int threshold_percent_value =
            parse_rail_balance_hybrid_threshold_percent(threshold_percent);

        const int rank_idx = nccl_context->rank_idx;
        const bool is_source = rank_idx < num_source_ranks;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.dim() == 2);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous() and
                       topk_idx.dim() == 2);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_weights.is_cuda() and
                       topk_weights.is_contiguous() and
                       topk_weights.dim() == 2);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(proxy_dispatch.is_cuda() and
                       proxy_dispatch.is_contiguous() and
                       proxy_dispatch.dim() == 2);
        EP_HOST_ASSERT(proxy_dispatch.scalar_type() == torch::kUInt8);
        EP_HOST_ASSERT(channel_count.is_cuda() and
                       channel_count.is_contiguous() and
                       channel_count.dim() == 3);
        EP_HOST_ASSERT(channel_count.scalar_type() == torch::kInt32);

        const int64_t num_tokens_i64 = x.size(0);
        const int64_t hidden_i64 = x.size(1);
        const int64_t num_topk_i64 = topk_idx.size(1);
        EP_HOST_ASSERT(num_tokens_i64 >= 0 and
                       num_tokens_i64 <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(hidden_i64 > 0 and hidden_i64 <= INT_MAX and
                       hidden_i64 % 256 == 0);
        EP_HOST_ASSERT(hidden_i64 <=
                       INT_MAX / static_cast<int>(sizeof(c10::BFloat16)));
        EP_HOST_ASSERT(num_topk_i64 >= 1 and num_topk_i64 <= 32);
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens_i64);
        EP_HOST_ASSERT(topk_weights.sizes() == topk_idx.sizes());
        EP_HOST_ASSERT(channel_count.size(0) == num_source_ranks);
        EP_HOST_ASSERT(channel_count.size(2) == num_destinations);
        EP_HOST_ASSERT(channel_count.size(1) >= 1 and
                       channel_count.size(1) <=
                           rail_balance::kNumHybridMaxChannels);
        EP_HOST_ASSERT(is_source or num_tokens_i64 == 0);

        const int hidden = static_cast<int>(hidden_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        const int num_tokens = static_cast<int>(num_tokens_i64);
        const int num_channels = static_cast<int>(channel_count.size(1));
        const auto hybrid_layout = rail_balance::HybridArenaLayout(
            hidden, num_topk, proxy_capacity);
        const int64_t dispatch_token_bytes =
            hybrid_layout.dispatch_token_bytes;
        const int64_t combine_token_bytes = hybrid_layout.combine_token_bytes;
        EP_HOST_ASSERT(proxy_dispatch.size(1) == dispatch_token_bytes);
        EP_HOST_ASSERT(proxy_dispatch.size(0) ==
                       (is_source ? proxy_capacity : 0));

        const int tensor_device = x.get_device();
        EP_HOST_ASSERT(tensor_device == device_index);
        EP_HOST_ASSERT(topk_idx.get_device() == tensor_device);
        EP_HOST_ASSERT(topk_weights.get_device() == tensor_device);
        EP_HOST_ASSERT(proxy_dispatch.get_device() == tensor_device);
        EP_HOST_ASSERT(channel_count.get_device() == tensor_device);
        const c10::cuda::CUDAGuard device_guard(tensor_device);
        int expected_process_device = -1;
        if (not rail_balance_hybrid_plan_process_device.compare_exchange_strong(
                expected_process_device, tensor_device,
                std::memory_order_relaxed) and
            expected_process_device != tensor_device)
            EP_HOST_UNREACHABLE(
                "Hybrid vnode bridge supports one CUDA device per process");

        // Integer-domain checks precede VNode layout construction, whose
        // public fields are intentionally compact int values.
        const int64_t physical_capacity_i64 =
            rail_balance::checked_mul_i64(
                num_destinations - 1, num_max_tokens_per_rank);
        const int64_t rail_capacity_i64 = rail_balance::checked_mul_i64(
            physical_capacity_i64, num_topk + 1);
        const int64_t expert_capacity_i64 = rail_balance::checked_mul_i64(
            rail_balance::checked_mul_i64(
                num_source_ranks, num_max_tokens_per_rank),
            num_topk);
        const int64_t reduce_rows_i64 = rail_balance::checked_mul_i64(
            std::min(num_destinations, num_topk),
            num_max_tokens_per_rank);
        const int64_t source_identity_i64 = rail_balance::checked_mul_i64(
            num_source_ranks, num_max_tokens_per_rank);
        EP_HOST_ASSERT(physical_capacity_i64 > 0 and
                       physical_capacity_i64 <= INT_MAX);
        EP_HOST_ASSERT(rail_capacity_i64 > 0 and
                       rail_capacity_i64 <= INT_MAX);
        EP_HOST_ASSERT(expert_capacity_i64 > 0 and
                       expert_capacity_i64 <= INT_MAX);
        EP_HOST_ASSERT(reduce_rows_i64 > 0 and
                       reduce_rows_i64 <= INT_MAX);
        EP_HOST_ASSERT(source_identity_i64 > 0 and
                       source_identity_i64 <= INT_MAX);
        const int physical_capacity =
            static_cast<int>(physical_capacity_i64);
        const int rail_capacity = static_cast<int>(rail_capacity_i64);
        const int expert_capacity = static_cast<int>(expert_capacity_i64);

        const auto vnode_layout = rail_balance::VNodeRoundTripLayout(
            hidden * static_cast<int>(sizeof(c10::BFloat16)), num_topk,
            num_destinations - 1, num_max_tokens_per_rank,
            num_source_ranks, num_max_tokens_per_rank);
        validate_rail_balance_hybrid_vnode_layout(
            hidden, num_topk, num_destinations, num_source_ranks,
            num_channels, num_max_tokens_per_rank, proxy_capacity);
        EP_HOST_ASSERT(vnode_layout.rail.records.get_smem_bytes() <=
                       jit::device_runtime->get_num_smem_bytes());
        EP_HOST_ASSERT(arena_offset >= 0 and
                       arena_offset %
                           rail_balance::kArenaAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        const int64_t guarded_arena_bytes =
            rail_balance::checked_add_i64(
                vnode_layout.arena_bytes, kArenaGuardBytes);
        EP_HOST_ASSERT(guarded_arena_bytes <=
                       num_gpu_buffer_bytes - arena_offset);
        void* arena = math::advance_ptr(buffer, arena_offset);
        const auto mapped_vnode_layout = rail_balance::VNodeRoundTripLayout(
            hidden * static_cast<int>(sizeof(c10::BFloat16)), num_topk,
            num_destinations - 1, num_max_tokens_per_rank,
            num_source_ranks, num_max_tokens_per_rank, arena);

        // Reject accidental wrapping of any part of this world's symmetric
        // GPU buffer as an ordinary tensor.  The source-subgroup snapshot may
        // come from a different allocator, but no raw symmetric pointer is
        // accepted by this API.
        const uintptr_t symmetric_begin =
            reinterpret_cast<uintptr_t>(workspace);
        const uintptr_t symmetric_end =
            reinterpret_cast<uintptr_t>(buffer) + num_gpu_buffer_bytes;
        const auto assert_owning_input = [&](const torch::Tensor& tensor) {
            if (tensor.nbytes() == 0)
                return;
            const uintptr_t begin = reinterpret_cast<uintptr_t>(
                tensor.data_ptr());
            const uintptr_t end = begin + tensor.nbytes();
            EP_HOST_ASSERT(end <= symmetric_begin or begin >= symmetric_end);
        };
        assert_owning_input(x);
        assert_owning_input(topk_idx);
        assert_owning_input(topk_weights);
        assert_owning_input(proxy_dispatch);
        assert_owning_input(channel_count);

        // The common count tensor is the one cross-object bridge input.  A
        // host snapshot is used only for strict pre-commit validation; the GPU
        // planner below consumes an owning D2D copy.
        const auto count_cpu = channel_count.cpu().contiguous();
        const int* count_ptr = count_cpu.data_ptr<int>();
        const int channel_capacity =
            num_max_tokens_per_rank / num_channels +
            (num_max_tokens_per_rank % num_channels != 0);
        for (int owner = 0; owner < num_source_ranks; ++owner) {
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                int64_t owner_total = 0;
                for (int channel = 0; channel < num_channels; ++channel) {
                    const auto offset =
                        (static_cast<int64_t>(owner) * num_channels +
                         channel) * num_destinations + destination;
                    const int value = count_ptr[offset];
                    EP_HOST_ASSERT(value >= 0 and
                                   value <= channel_capacity);
                    if (destination == 0)
                        EP_HOST_ASSERT(value == 0);
                    owner_total += value;
                }
                EP_HOST_ASSERT(owner_total <= num_max_tokens_per_rank);
            }
        }

        // Source ranks additionally prove that their ordinary top-k input is
        // remote-only, unmasked, duplicate-free, and exactly represented by
        // their owner row of channel_count.  Thus pack cannot discover an
        // input/count mismatch after it starts publishing into the arena.
        if (is_source) {
            const auto topk_cpu = topk_idx.cpu().contiguous();
            const auto* topk_ptr = topk_cpu.data_ptr<topk_idx_t>();
            std::vector<int> expected_count(
                static_cast<size_t>(num_channels) * num_destinations, 0);
            const int experts_per_destination =
                num_experts / num_destinations;
            for (int token = 0; token < num_tokens; ++token) {
                bool seen_destination[
                    rail_balance::kNumHybridMaxDestinations] = {};
                for (int lane = 0; lane < num_topk; ++lane) {
                    const auto expert_value = topk_ptr[
                        static_cast<int64_t>(token) * num_topk + lane];
                    const int64_t expert =
                        static_cast<int64_t>(expert_value);
                    EP_HOST_ASSERT(expert >= experts_per_destination and
                                   expert < num_experts);
                    for (int prior = 0; prior < lane; ++prior)
                        EP_HOST_ASSERT(expert_value != topk_ptr[
                            static_cast<int64_t>(token) * num_topk + prior]);
                    seen_destination[expert / experts_per_destination] = true;
                }
                const int channel = token % num_channels;
                for (int destination = 1;
                     destination < num_destinations; ++destination)
                    if (seen_destination[destination])
                        ++expected_count[
                            channel * num_destinations + destination];
            }
            for (int channel = 0; channel < num_channels; ++channel)
                for (int destination = 0;
                     destination < num_destinations; ++destination) {
                    const auto global_offset =
                        (static_cast<int64_t>(rank_idx) * num_channels +
                         channel) * num_destinations + destination;
                    EP_HOST_ASSERT(count_ptr[global_offset] ==
                        expected_count[
                            channel * num_destinations + destination]);
                }
        }

        const auto int_options = channel_count.options();
        auto plan = allocate_rail_balance_hybrid_plan_outputs(
            int_options, num_source_ranks, num_channels, num_destinations);
        auto compact_quota = torch::empty(
            {num_source_ranks, num_destinations - 1}, int_options);
        auto proxy_return = torch::zeros(
            {proxy_capacity, combine_token_bytes},
            proxy_dispatch.options());
        auto reduce_seed = torch::zeros(
            {reduce_rows_i64, combine_token_bytes},
            proxy_dispatch.options());
        auto rail_records = torch::empty(
            {rail_capacity, mapped_vnode_layout.rail.records.record_bytes},
            proxy_dispatch.options());
        auto rail_ready = torch::empty({rail_capacity}, int_options);
        auto rail_routes = torch::empty(
            {rail_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            proxy_dispatch.options());
        auto expert_records = torch::empty(
            {expert_capacity,
             mapped_vnode_layout.expert.records.record_bytes},
            proxy_dispatch.options());
        auto expert_ready = torch::empty({expert_capacity}, int_options);
        auto expert_routes = torch::empty(
            {expert_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            proxy_dispatch.options());
        const int status_stride = std::max({
            num_channels,
            physical_capacity * num_topk,
            expert_capacity,
        });
        EP_HOST_ASSERT(status_stride > 0);
        auto stage_status = torch::zeros(
            {6, status_stride}, int_options);
        auto arena_guard = torch::empty(
            {kArenaGuardBytes}, proxy_dispatch.options());
        std::vector<int> host_stage_status(
            static_cast<size_t>(6) * status_stride, 0);

        // Every cold-build failure remains pre-commit.  finish performs no
        // build, allocation, tensor validation, or quota compaction.
        auto prepared_plan = prepare_rail_balance_hybrid_plan(
            1, num_source_ranks, num_channels, num_destinations);
        auto prepared_adapter = prepare_rail_balance_hybrid_vnode(
            hidden, num_topk);
        auto prepared_vnode = prepare_rail_balance_vnode(hidden, num_topk);
        auto prepared_world_barrier =
            prepare_rail_balance_hybrid_local_barrier(
                kWorldRanks, num_gpu_timeout_cycles);

        const auto compute_stream =
            at::cuda::getCurrentCUDAStream(tensor_device);
        if (comm_stream.id() != compute_stream.id())
            stream_wait(comm_stream, compute_stream);
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            plan.channel_count.data_ptr<int>(), channel_count.data_ptr<int>(),
            channel_count.nbytes(), cudaMemcpyDeviceToDevice, comm_stream));
        launch_prepared_rail_balance_hybrid_plan(
            prepared_plan,
            plan.channel_count.data_ptr<int>(), plan.count.data_ptr<int>(),
            plan.quota.data_ptr<int>(), plan.keep_count.data_ptr<int>(),
            plan.segments.data_ptr<int>(),
            plan.num_segments.data_ptr<int>(),
            num_source_ranks, num_channels, num_destinations,
            normalized_remainder_seed,
            policy_value, threshold_percent_value, comm_stream);
        launch_prepared_rail_balance_hybrid_prefix(
            prepared_plan,
            plan.channel_count.data_ptr<int>(), plan.quota.data_ptr<int>(),
            plan.keep_count.data_ptr<int>(),
            plan.owner_channel_prefix.data_ptr<int>(),
            plan.retained.data_ptr<int>(), plan.moved.data_ptr<int>(),
            plan.moved_channel_prefix.data_ptr<int>(),
            plan.group_prefix.data_ptr<int>(),
            plan.proxy_required.data_ptr<int>(),
            plan.moved_copies.data_ptr<int>(), plan.status.data_ptr<int>(),
            num_source_ranks, num_channels, num_destinations,
            num_max_tokens_per_rank, proxy_capacity, comm_stream);

        // Dropping destination zero with a pointer offset would retain the D
        // source-row pitch.  Copy every row into its true contiguous [G,D-1]
        // representation before the old vnode kernels can observe it.
        CUDA_RUNTIME_CHECK(cudaMemcpy2DAsync(
            compact_quota.data_ptr<int>(),
            static_cast<size_t>(num_destinations - 1) * sizeof(int),
            plan.quota.data_ptr<int>() + 1,
            static_cast<size_t>(num_destinations) * sizeof(int),
            static_cast<size_t>(num_destinations - 1) * sizeof(int),
            num_source_ranks, cudaMemcpyDeviceToDevice, comm_stream));

        int host_status = 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            &host_status, plan.status.data_ptr<int>(), sizeof(host_status),
            cudaMemcpyDeviceToHost, comm_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
        EP_HOST_ASSERT(host_status == 0 or host_status == 1);
        if (host_status == 0) {
            const auto compact_cpu = compact_quota.cpu().contiguous();
            const int* compact_ptr = compact_cpu.data_ptr<int>();
            for (int owner = 0; owner < num_source_ranks; ++owner)
                for (int destination = 0;
                     destination < num_destinations - 1; ++destination) {
                    const int value = compact_ptr[
                        owner * (num_destinations - 1) + destination];
                    EP_HOST_ASSERT(value >= 0 and
                                   value <= num_max_tokens_per_rank);
                }
        }

        rail_balance_hybrid_vnode_pending.emplace(
            RailBalanceHybridVNodePending{
                .invocation_id = invocation_id,
                .finish_attempted = false,
                .finished = false,
                .plan_status = host_status,
                .num_tokens = num_tokens,
                .hidden = hidden,
                .num_topk = num_topk,
                .num_experts = num_experts,
                .num_destinations = num_destinations,
                .num_rails = num_source_ranks,
                .num_channels = num_channels,
                .num_max_tokens_per_rank = num_max_tokens_per_rank,
                .proxy_capacity = proxy_capacity,
                .generation = generation,
                .physical_capacity = physical_capacity,
                .rail_capacity = rail_capacity,
                .expert_capacity = expert_capacity,
                .status_stride = status_stride,
                .arena_offset = arena_offset,
                .arena_bytes = vnode_layout.arena_bytes,
                .arena = arena,
                .x = x,
                .topk_idx = topk_idx,
                .topk_weights = topk_weights,
                .proxy_dispatch = proxy_dispatch,
                .plan = std::move(plan),
                .compact_quota = std::move(compact_quota),
                .proxy_return = std::move(proxy_return),
                .reduce_seed = std::move(reduce_seed),
                .rail_records = std::move(rail_records),
                .rail_ready = std::move(rail_ready),
                .rail_routes = std::move(rail_routes),
                .expert_records = std::move(expert_records),
                .expert_ready = std::move(expert_ready),
                .expert_routes = std::move(expert_routes),
                .stage_status = std::move(stage_status),
                .arena_guard = std::move(arena_guard),
                .host_stage_status = std::move(host_stage_status),
                .host_stage_status_ready = false,
                .prepared_plan = std::move(prepared_plan),
                .prepared_adapter = std::move(prepared_adapter),
                .prepared_vnode = std::move(prepared_vnode),
                .prepared_world_barrier =
                    std::move(prepared_world_barrier),
            });
        const auto& pending = rail_balance_hybrid_vnode_pending.value();
        return {
            host_status, pending.plan.as_tuple(), pending.compact_quota,
        };
    }

    RailBalanceHybridVNodeTensors rail_balance_hybrid_vnode_finish(
        const int& invocation_id) {
        constexpr int64_t kArenaGuardBytes = 4096;
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_vnode_pending.has_value());
        auto& pending = rail_balance_hybrid_vnode_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(pending.plan_status == 0);
        EP_HOST_ASSERT(not pending.finish_attempted);
        EP_HOST_ASSERT(not pending.finished);

        const bool is_source =
            nccl_context->rank_idx < pending.num_rails;
        const int old_num_destinations = pending.num_destinations - 1;
        const int expert_begin =
            pending.num_experts / pending.num_destinations;
        const int experts_per_rank = pending.num_experts /
            (pending.num_destinations * pending.num_rails);
        const auto mapped_layout = rail_balance::VNodeRoundTripLayout(
            pending.hidden * static_cast<int>(sizeof(c10::BFloat16)),
            pending.num_topk, old_num_destinations,
            pending.num_max_tokens_per_rank, pending.num_rails,
            pending.num_max_tokens_per_rank, pending.arena);
        const int smem_bytes = static_cast<int>(
            mapped_layout.rail.records.get_smem_bytes());
        int* status = pending.stage_status.data_ptr<int>();

        // From this point onward a world barrier may be entered.  Any failure
        // is sticky and the same invocation can only be released by abort.
        pending.finish_attempted = true;
        try {
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(
                pending.arena, 0, pending.arena_bytes, comm_stream));
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(
                math::advance_ptr(pending.arena, pending.arena_bytes),
                0xA5, kArenaGuardBytes, comm_stream));

            if (is_source)
                launch_prepared_rail_balance_hybrid_pack_vnode_base(
                    pending.prepared_adapter,
                    pending.x.data_ptr(),
                    pending.topk_idx.data_ptr<topk_idx_t>(),
                    pending.topk_weights.data_ptr<float>(),
                    pending.proxy_dispatch.data_ptr(), pending.arena,
                    pending.plan, status,
                    pending.num_tokens, pending.hidden,
                    pending.num_topk, pending.num_experts,
                    pending.num_destinations, pending.num_rails,
                    nccl_context->rank_idx, pending.num_channels,
                    pending.num_max_tokens_per_rank,
                    pending.proxy_capacity, pending.generation,
                    comm_stream);
            // B0: arena zeroing and every source base publication complete.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            if (is_source)
                launch_prepared_rail_balance_vnode_peer(
                    pending.prepared_vnode.scaleout,
                    nccl_context->dev_comm, nccl_context->window,
                    pending.arena, status + pending.status_stride,
                    pending.compact_quota.data_ptr<int>(),
                    old_num_destinations,
                    pending.num_max_tokens_per_rank, pending.num_rails,
                    pending.num_max_tokens_per_rank, pending.generation,
                    nccl_context->rank_idx, expert_begin, experts_per_rank,
                    pending.physical_capacity, smem_bytes, comm_stream);
            // B1: all virtual scaleout writes are visible.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            if (not is_source)
                launch_prepared_rail_balance_vnode_expert(
                    pending.prepared_vnode.forward,
                    nccl_context->dev_comm, nccl_context->window,
                    pending.arena, status + 2 * pending.status_stride,
                    pending.compact_quota.data_ptr<int>(),
                    old_num_destinations,
                    pending.num_max_tokens_per_rank, pending.num_rails,
                    pending.num_max_tokens_per_rank, pending.generation,
                    nccl_context->rank_idx, expert_begin, experts_per_rank,
                    pending.physical_capacity * pending.num_topk,
                    smem_bytes, comm_stream);
            // B2: destination forwarding complete.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            if (not is_source)
                launch_prepared_rail_balance_vnode_expert(
                    pending.prepared_vnode.expert,
                    nccl_context->dev_comm, nccl_context->window,
                    pending.arena, status + 3 * pending.status_stride,
                    pending.compact_quota.data_ptr<int>(),
                    old_num_destinations,
                    pending.num_max_tokens_per_rank, pending.num_rails,
                    pending.num_max_tokens_per_rank, pending.generation,
                    nccl_context->rank_idx, expert_begin, experts_per_rank,
                    pending.expert_capacity, smem_bytes, comm_stream);
            // B3: synthetic expert contributions complete.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            if (not is_source)
                launch_prepared_rail_balance_vnode_peer(
                    pending.prepared_vnode.return_path,
                    nccl_context->dev_comm, nccl_context->window,
                    pending.arena, status + 4 * pending.status_stride,
                    pending.compact_quota.data_ptr<int>(),
                    old_num_destinations,
                    pending.num_max_tokens_per_rank, pending.num_rails,
                    pending.num_max_tokens_per_rank, pending.generation,
                    nccl_context->rank_idx, expert_begin, experts_per_rank,
                    pending.physical_capacity * pending.num_topk,
                    smem_bytes, comm_stream);
            // B4: every returned contribution is visible at source egress.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            if (is_source)
                launch_prepared_rail_balance_hybrid_return_demux(
                    pending.prepared_adapter,
                    pending.arena, pending.proxy_dispatch.data_ptr(),
                    pending.reduce_seed.data_ptr(),
                    pending.proxy_return.data_ptr(), pending.plan,
                    status + 5 * pending.status_stride,
                    pending.hidden, pending.num_topk, pending.num_experts,
                    pending.num_destinations, pending.num_rails,
                    nccl_context->rank_idx, pending.num_channels,
                    pending.num_max_tokens_per_rank,
                    pending.proxy_capacity, pending.generation,
                    comm_stream);
            // B5: demux has completed on every source before host inspection.
            launch_prepared_rail_balance_hybrid_local_barrier(
                pending.prepared_world_barrier,
                nccl_context->dev_comm, nccl_context->window, workspace,
                8, nccl_context->nvl_rank_idx,
                num_gpu_timeout_cycles, comm_stream);

            const auto copy_snapshot = [&](void* dst, const void* src,
                                           const int64_t num_bytes) {
                CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                    dst, src, static_cast<size_t>(num_bytes),
                    cudaMemcpyDeviceToDevice, comm_stream));
            };
            copy_snapshot(
                pending.rail_records.data_ptr(),
                mapped_layout.rail.records.get_record_ptr(0),
                static_cast<int64_t>(pending.rail_capacity) *
                    mapped_layout.rail.records.record_bytes);
            copy_snapshot(
                pending.rail_ready.data_ptr<int>(),
                mapped_layout.rail.records.get_ready_ptr(),
                static_cast<int64_t>(pending.rail_capacity) * sizeof(int));
            copy_snapshot(
                pending.rail_routes.data_ptr(),
                mapped_layout.rail.get_route_ptr(),
                static_cast<int64_t>(pending.rail_capacity) *
                    sizeof(rail_balance::VNodeRoute));
            copy_snapshot(
                pending.expert_records.data_ptr(),
                mapped_layout.expert.records.get_record_ptr(0),
                static_cast<int64_t>(pending.expert_capacity) *
                    mapped_layout.expert.records.record_bytes);
            copy_snapshot(
                pending.expert_ready.data_ptr<int>(),
                mapped_layout.expert.records.get_ready_ptr(),
                static_cast<int64_t>(pending.expert_capacity) * sizeof(int));
            copy_snapshot(
                pending.expert_routes.data_ptr(),
                mapped_layout.expert.get_route_ptr(),
                static_cast<int64_t>(pending.expert_capacity) *
                    sizeof(rail_balance::VNodeRoute));
            copy_snapshot(
                pending.arena_guard.data_ptr(),
                math::advance_ptr(pending.arena, pending.arena_bytes),
                kArenaGuardBytes);
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                pending.host_stage_status.data(), status,
                static_cast<size_t>(6) * pending.status_stride * sizeof(int),
                cudaMemcpyDeviceToHost, comm_stream));
            CUDA_RUNTIME_CHECK(cudaStreamSynchronize(comm_stream));
            pending.host_stage_status_ready = true;
        } catch (...) {
            pending.plan_status = static_cast<int>(
                rail_balance::HybridPlanError::InvalidSchedule);
            throw;
        }

        for (const int value : pending.host_stage_status) {
            if (value != 0) {
                pending.plan_status = value;
                throw EPExceptionWithLineInfo(
                    "Rail balance Hybrid vnode finish",
                    "a fixed vnode stage reported a sticky device error");
            }
        }
        pending.finished = true;
        return pending.as_tuple();
    }

    std::tuple<int, int, std::vector<int>>
    rail_balance_hybrid_vnode_status_snapshot(
            const int& invocation_id) const {
        EP_HOST_ASSERT(not destroyed);
        EP_HOST_ASSERT(rail_balance_hybrid_vnode_pending.has_value());
        const auto& pending = rail_balance_hybrid_vnode_pending.value();
        EP_HOST_ASSERT(pending.invocation_id == invocation_id);
        EP_HOST_ASSERT(pending.finish_attempted);
        EP_HOST_ASSERT(pending.host_stage_status_ready);
        EP_HOST_ASSERT(
            pending.host_stage_status.size() ==
            static_cast<size_t>(6) * pending.status_stride);
        return {
            pending.plan_status,
            pending.status_stride,
            pending.host_stage_status,
        };
    }

    void rail_balance_hybrid_vnode_abort(const int& invocation_id) {
        // Idempotent and transaction-specific.  After finish starts, scratch
        // rollback is deliberately not promised, but owning tensor/JIT state
        // can still be released safely once the Python control plane has
        // converged all ranks.
        if (rail_balance_hybrid_vnode_pending.has_value() and
            rail_balance_hybrid_vnode_pending->invocation_id == invocation_id)
            rail_balance_hybrid_vnode_pending.reset();
    }

    static std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
    get_rail_balance_source_shuffle_layout(
        const int& hidden,
        const int& num_topk,
        const int& physical_capacity) {
        EP_HOST_ASSERT(hidden > 0);
        EP_HOST_ASSERT(hidden % 16 == 0);
        EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_HOST_ASSERT(physical_capacity > 0);
        EP_HOST_ASSERT(static_cast<int64_t>(hidden) * sizeof(c10::BFloat16) <= INT_MAX);
        const auto layout = rail_balance::SourceShuffleLayout(
            hidden * sizeof(c10::BFloat16), num_topk, physical_capacity);
        const auto descriptor_offset =
            rail_balance::kNumCanaryBytes + layout.token_bytes;
        const auto tail_canary_offset =
            descriptor_offset + rail_balance::kNumDescriptorBytes;
        return {
            rail_balance::kNumCanaryBytes,
            layout.token_bytes,
            descriptor_offset,
            tail_canary_offset,
            layout.record_bytes,
            layout.ready_offset,
            layout.arena_bytes,
        };
    }

    static std::tuple<
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>
    get_rail_balance_protocol_layout(
        const int& hidden,
        const int& num_topk,
        const int& physical_capacity) {
        EP_HOST_ASSERT(hidden > 0 and hidden % 16 == 0);
        EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_HOST_ASSERT(physical_capacity > 0 and physical_capacity <= 64);
        EP_HOST_ASSERT(
            static_cast<int64_t>(hidden) * sizeof(c10::BFloat16) <= INT_MAX);
        const auto layout = rail_balance::OneShotProtocolLayout(
            hidden * sizeof(c10::BFloat16), num_topk, physical_capacity);
        return {
            layout.source_shuffle.record_bytes,
            layout.source_shuffle.ready_offset,
            layout.publish_sequence_offset,
            layout.control_offset,
            sizeof(rail_balance::OneShotProxyControl),
            layout.arena_bytes,
        };
    }

    static std::tuple<
        int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t>
    get_rail_balance_vnode_layout(
        const int& hidden,
        const int& num_topk,
        const int& physical_capacity,
        const int& num_source_ranks,
        const int& num_max_tokens) {
        EP_HOST_ASSERT(hidden > 0 and hidden % 16 == 0);
        EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_HOST_ASSERT(physical_capacity > 0);
        EP_HOST_ASSERT(num_source_ranks > 0);
        EP_HOST_ASSERT(num_max_tokens > 0);
        const int64_t hidden_bytes =
            static_cast<int64_t>(hidden) * sizeof(c10::BFloat16);
        const int64_t rail_capacity =
            static_cast<int64_t>(physical_capacity) * (num_topk + 1);
        const int64_t expert_capacity =
            static_cast<int64_t>(num_source_ranks) * physical_capacity *
            num_topk;
        const int64_t partial_capacity =
            static_cast<int64_t>(num_max_tokens) * num_topk;
        EP_HOST_ASSERT(hidden_bytes <= INT_MAX);
        EP_HOST_ASSERT(
            rail_capacity > 0 and rail_capacity <= INT_MAX);
        EP_HOST_ASSERT(
            expert_capacity > 0 and expert_capacity <= INT_MAX);
        EP_HOST_ASSERT(
            partial_capacity > 0 and partial_capacity <= INT_MAX);
        const auto layout = rail_balance::VNodeRoundTripLayout(
            static_cast<int>(hidden_bytes), num_topk,
            physical_capacity, num_source_ranks, num_max_tokens);
        return {
            layout.rail.records.record_bytes,
            sizeof(rail_balance::VNodeRoute),
            layout.rail_capacity,
            layout.rail.records.ready_offset,
            layout.rail.route_offset,
            layout.expert_offset,
            layout.expert_capacity,
            layout.expert_offset + layout.expert.records.ready_offset,
            layout.expert_offset + layout.expert.route_offset,
            layout.expert_offset,
            layout.expert_offset + layout.owner.ready_offset,
            layout.role_arena_bytes,
            layout.arena_bytes,
        };
    }

    static std::tuple<
        int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t>
    get_rail_balance_vnode_multidst_layout(
        const int& hidden,
        const int& num_topk,
        const int& physical_capacity,
        const int& destination_capacity,
        const int& num_destinations,
        const int& num_source_ranks,
        const int& num_max_tokens) {
        EP_HOST_ASSERT(hidden > 0 and hidden % 16 == 0);
        EP_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
        EP_HOST_ASSERT(destination_capacity > 0);
        EP_HOST_ASSERT(num_destinations > 0);
        EP_HOST_ASSERT(num_source_ranks > 0);
        EP_HOST_ASSERT(num_max_tokens > 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(destination_capacity) *
                num_destinations == physical_capacity);
        const int64_t hidden_bytes =
            static_cast<int64_t>(hidden) * sizeof(c10::BFloat16);
        const int64_t rail_capacity =
            static_cast<int64_t>(physical_capacity) * (num_topk + 1);
        const int64_t expert_capacity =
            static_cast<int64_t>(num_source_ranks) *
            destination_capacity * num_topk;
        const int64_t partial_capacity =
            static_cast<int64_t>(num_max_tokens) * num_topk;
        EP_HOST_ASSERT(hidden_bytes <= INT_MAX);
        EP_HOST_ASSERT(
            rail_capacity > 0 and rail_capacity <= INT_MAX);
        EP_HOST_ASSERT(
            expert_capacity > 0 and expert_capacity <= INT_MAX);
        EP_HOST_ASSERT(
            partial_capacity > 0 and partial_capacity <= INT_MAX);
        const auto layout = rail_balance::VNodeRoundTripLayout(
            static_cast<int>(hidden_bytes), num_topk,
            num_destinations, destination_capacity,
            num_source_ranks, num_max_tokens);
        return {
            layout.rail.records.record_bytes,
            sizeof(rail_balance::VNodeRoute),
            layout.rail_capacity,
            layout.rail.records.ready_offset,
            layout.rail.route_offset,
            layout.expert_offset,
            layout.expert_capacity,
            layout.expert_offset + layout.expert.records.ready_offset,
            layout.expert_offset + layout.expert.route_offset,
            layout.expert_offset,
            layout.expert_offset + layout.owner.ready_offset,
            layout.role_arena_bytes,
            layout.arena_bytes,
        };
    }

    std::tuple<torch::Tensor, torch::Tensor> rail_balance_source_shuffle(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& send_manifest,
        const torch::Tensor& fingerprints,
        const int64_t& arena_offset,
        const int& physical_capacity,
        const int& generation,
        const int& num_max_tokens_per_rank,
        const int& num_recv_records) const {
        // Collective private prototype: the caller must preflight identical
        // configuration and globally unique (egress, physical_slot) keys on
        // all ranks before entering this method.
        EP_HOST_ASSERT(not allow_hybrid_mode and
                       "C040 source shuffle is restricted to pure single-node mode");
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1);
        EP_HOST_ASSERT(nccl_context->num_scaleup_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(nccl_context->scaleup_rank_idx == nccl_context->rank_idx);

        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.dim() == 2);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous() and topk_idx.dim() == 2);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_weights.is_cuda() and topk_weights.is_contiguous() and topk_weights.dim() == 2);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(send_manifest.is_cuda() and send_manifest.is_contiguous() and send_manifest.dim() == 2);
        EP_HOST_ASSERT(send_manifest.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(send_manifest.size(1) == rail_balance::kNumManifestFields);
        EP_HOST_ASSERT(fingerprints.is_cuda() and fingerprints.is_contiguous() and fingerprints.dim() == 1);
        EP_HOST_ASSERT(fingerprints.scalar_type() == torch::kInt64);

        const auto num_tokens_i64 = x.size(0);
        const auto hidden_i64 = x.size(1);
        const auto num_topk_i64 = topk_idx.size(1);
        const auto num_send_records_i64 = send_manifest.size(0);
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens_i64);
        EP_HOST_ASSERT(topk_weights.sizes() == topk_idx.sizes());
        EP_HOST_ASSERT(fingerprints.size(0) == num_send_records_i64);
        EP_HOST_ASSERT(hidden_i64 > 0 and hidden_i64 <= INT_MAX);
        EP_HOST_ASSERT(hidden_i64 * static_cast<int64_t>(sizeof(c10::BFloat16)) <= INT_MAX);
        EP_HOST_ASSERT(num_topk_i64 > 0 and num_topk_i64 <= 32);
        EP_HOST_ASSERT(num_send_records_i64 <= INT_MAX);
        EP_HOST_ASSERT(num_tokens_i64 <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(physical_capacity > 0);
        EP_HOST_ASSERT(generation > 0);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(num_recv_records >= 0 and num_recv_records <= physical_capacity);
        EP_HOST_ASSERT(
            static_cast<int64_t>(nccl_context->num_ranks - 1) * num_max_tokens_per_rank +
                std::max<int64_t>(num_tokens_i64 - 1, 0) <= INT_MAX);

        const int device_index = x.get_device();
        EP_HOST_ASSERT(device_index == this->device_index);
        EP_HOST_ASSERT(topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(topk_weights.get_device() == device_index);
        EP_HOST_ASSERT(send_manifest.get_device() == device_index);
        EP_HOST_ASSERT(fingerprints.get_device() == device_index);
        const c10::cuda::CUDAGuard device_guard(x.device());
        if (num_tokens_i64 > 0) {
            EP_HOST_ASSERT(topk_idx.min().item<topk_idx_t>() >= -1);
            EP_HOST_ASSERT(topk_idx.max().item<int64_t>() <= INT_MAX);
        }

        const int hidden = static_cast<int>(hidden_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        const int num_send_records = static_cast<int>(num_send_records_i64);
        const auto arena_layout = rail_balance::SourceShuffleLayout(
            hidden * sizeof(c10::BFloat16), num_topk, physical_capacity);
        EP_HOST_ASSERT(hidden % 16 == 0);
        EP_HOST_ASSERT(arena_offset >= 0 and
                       arena_offset % rail_balance::kArenaAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        EP_HOST_ASSERT(arena_layout.arena_bytes <= num_gpu_buffer_bytes - arena_offset);
        EP_HOST_ASSERT(arena_layout.get_smem_bytes() <=
                       jit::device_runtime->get_num_smem_bytes());

        if (num_send_records > 0) {
            EP_HOST_ASSERT(send_manifest.select(1, 0).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 0).max().item<int64_t>() < num_tokens_i64);
            EP_HOST_ASSERT(send_manifest.select(1, 1).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 2).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 3).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 3).max().item<int>() < nccl_context->num_ranks);
            EP_HOST_ASSERT((send_manifest.select(1, 3) != nccl_context->rank_idx).all().item<bool>());
            EP_HOST_ASSERT(send_manifest.select(1, 4).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 5).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 6).min().item<int>() >= 0);
            EP_HOST_ASSERT(send_manifest.select(1, 6).max().item<int>() < physical_capacity);
        }

        const auto stream = at::cuda::getCurrentCUDAStream();
        auto arena = math::advance_ptr(buffer, arena_offset);
        const auto mapped_arena_layout = rail_balance::SourceShuffleLayout(
            hidden * sizeof(c10::BFloat16), num_topk, physical_capacity, arena);
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            arena, 0xA5, mapped_arena_layout.ready_offset, stream));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            mapped_arena_layout.get_ready_ptr(), 0,
            static_cast<int64_t>(physical_capacity) * sizeof(int), stream));

        // All ranks clear their local arena before any peer can publish into it.
        barrier(false, true);
        if (num_send_records > 0) {
            launch_rail_balance_source_shuffle(
                nccl_context->dev_comm, nccl_context->window,
                x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(),
                topk_weights.data_ptr<float>(),
                send_manifest.data_ptr<int>(), fingerprints.data_ptr<int64_t>(),
                arena, hidden, num_topk, physical_capacity, generation,
                nccl_context->rank_idx, num_max_tokens_per_rank,
                num_send_records, static_cast<int>(arena_layout.get_smem_bytes()), stream);
        }

        // C040 is one-shot: a collective barrier replaces consumer spinning.
        barrier(false, true);
        auto records = torch::empty(
            {num_recv_records, arena_layout.record_bytes},
            x.options().dtype(torch::kUInt8));
        auto ready_values = torch::empty(
            {physical_capacity}, send_manifest.options());
        launch_rail_balance_source_shuffle_readback(
            arena, records.data_ptr(), ready_values.data_ptr<int>(),
            hidden, num_topk, physical_capacity, num_recv_records, stream);
        return {records, ready_values};
    }

    RailBalanceProtocolTensors rail_balance_source_shuffle_protocol(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& send_manifest,
        const torch::Tensor& fingerprints,
        const torch::Tensor& producer_delay_cycles,
        const torch::Tensor& consumer_delay_cycles,
        const int64_t& arena_offset,
        const int& physical_capacity,
        const int& generation,
        const int& stale_generation,
        const int& num_max_tokens_per_rank,
        const int& num_recv_records,
        const bool& preserve_record_payload,
        const bool& force_odd_first,
        const int& drop_physical_slot,
        const int& late_publish_start,
        const int& late_publish_after_consumed,
        const int64_t& timeout_cycles) const {
        // Collective private prototype. The caller must preflight an identical
        // configuration, compact per-egress physical prefixes, and unique
        // (egress, slot, generation) keys before entering.
        EP_HOST_ASSERT(not allow_hybrid_mode);
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1);
        EP_HOST_ASSERT(
            nccl_context->num_scaleup_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(
            nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(
            nccl_context->scaleup_rank_idx == nccl_context->rank_idx);

        EP_HOST_ASSERT(
            x.is_cuda() and x.is_contiguous() and x.dim() == 2);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(
            topk_idx.is_cuda() and topk_idx.is_contiguous() and
            topk_idx.dim() == 2);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(
            topk_weights.is_cuda() and topk_weights.is_contiguous() and
            topk_weights.dim() == 2 and
            topk_weights.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(
            send_manifest.is_cuda() and send_manifest.is_contiguous() and
            send_manifest.dim() == 2 and
            send_manifest.scalar_type() == torch::kInt32 and
            send_manifest.size(1) == rail_balance::kNumManifestFields);
        EP_HOST_ASSERT(
            fingerprints.is_cuda() and fingerprints.is_contiguous() and
            fingerprints.dim() == 1 and
            fingerprints.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(
            producer_delay_cycles.is_cuda() and
            producer_delay_cycles.is_contiguous() and
            producer_delay_cycles.dim() == 1 and
            producer_delay_cycles.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(
            consumer_delay_cycles.is_cuda() and
            consumer_delay_cycles.is_contiguous() and
            consumer_delay_cycles.dim() == 1 and
            consumer_delay_cycles.scalar_type() == torch::kInt64);

        const auto num_tokens_i64 = x.size(0);
        const auto hidden_i64 = x.size(1);
        const auto num_topk_i64 = topk_idx.size(1);
        const auto num_send_records_i64 = send_manifest.size(0);
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens_i64);
        EP_HOST_ASSERT(topk_weights.sizes() == topk_idx.sizes());
        EP_HOST_ASSERT(fingerprints.size(0) == num_send_records_i64);
        EP_HOST_ASSERT(
            producer_delay_cycles.size(0) == num_send_records_i64);
        EP_HOST_ASSERT(consumer_delay_cycles.size(0) == num_recv_records);
        EP_HOST_ASSERT(hidden_i64 > 0 and hidden_i64 <= INT_MAX);
        EP_HOST_ASSERT(
            hidden_i64 * static_cast<int64_t>(sizeof(c10::BFloat16)) <=
            INT_MAX);
        EP_HOST_ASSERT(num_topk_i64 > 0 and num_topk_i64 <= 32);
        EP_HOST_ASSERT(num_send_records_i64 <= INT_MAX);
        EP_HOST_ASSERT(num_tokens_i64 <= num_max_tokens_per_rank);
        EP_HOST_ASSERT(
            physical_capacity > 0 and physical_capacity <= 64);
        EP_HOST_ASSERT(generation > 0);
        EP_HOST_ASSERT(
            stale_generation >= 0 and stale_generation < generation);
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0);
        EP_HOST_ASSERT(
            num_recv_records >= 0 and
            num_recv_records <= physical_capacity);
        EP_HOST_ASSERT(
            not force_odd_first or
            num_recv_records == 0 or num_recv_records >= 2);
        EP_HOST_ASSERT(
            drop_physical_slot >= -1 and
            drop_physical_slot < physical_capacity);
        EP_HOST_ASSERT(
            drop_physical_slot < 0 or
            drop_physical_slot < num_recv_records);
        EP_HOST_ASSERT(
            late_publish_start >= -1 and
            late_publish_start < physical_capacity);
        EP_HOST_ASSERT(
            late_publish_start < 0 or
            late_publish_start < num_recv_records);
        EP_HOST_ASSERT(
            late_publish_after_consumed >= 0 and
            late_publish_after_consumed <= num_recv_records);
        EP_HOST_ASSERT(
            late_publish_start < 0 or
            late_publish_after_consumed < late_publish_start);
        EP_HOST_ASSERT(timeout_cycles > 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(nccl_context->num_ranks - 1) *
                    num_max_tokens_per_rank +
                std::max<int64_t>(num_tokens_i64 - 1, 0) <=
            INT_MAX);

        const int device_index = x.get_device();
        EP_HOST_ASSERT(device_index == this->device_index);
        EP_HOST_ASSERT(topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(topk_weights.get_device() == device_index);
        EP_HOST_ASSERT(send_manifest.get_device() == device_index);
        EP_HOST_ASSERT(fingerprints.get_device() == device_index);
        EP_HOST_ASSERT(
            producer_delay_cycles.get_device() == device_index);
        EP_HOST_ASSERT(
            consumer_delay_cycles.get_device() == device_index);
        const c10::cuda::CUDAGuard device_guard(x.device());
        if (num_tokens_i64 > 0) {
            EP_HOST_ASSERT(
                topk_idx.min().item<topk_idx_t>() >= -1);
            EP_HOST_ASSERT(topk_idx.max().item<int64_t>() <= INT_MAX);
        }
        if (num_send_records_i64 > 0) {
            EP_HOST_ASSERT(
                producer_delay_cycles.min().item<int64_t>() >= 0);
            EP_HOST_ASSERT(
                producer_delay_cycles.max().item<int64_t>() <
                timeout_cycles);
        }
        if (num_recv_records > 0) {
            EP_HOST_ASSERT(
                consumer_delay_cycles.min().item<int64_t>() >= 0);
            EP_HOST_ASSERT(
                consumer_delay_cycles.max().item<int64_t>() <
                timeout_cycles);
        }

        const int hidden = static_cast<int>(hidden_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        const int num_send_records =
            static_cast<int>(num_send_records_i64);
        EP_HOST_ASSERT(hidden % 16 == 0);
        const auto protocol_layout = rail_balance::OneShotProtocolLayout(
            hidden * sizeof(c10::BFloat16),
            num_topk,
            physical_capacity);
        EP_HOST_ASSERT(
            arena_offset >= 0 and
            arena_offset % rail_balance::kArenaAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        EP_HOST_ASSERT(
            protocol_layout.arena_bytes <=
            num_gpu_buffer_bytes - arena_offset);
        EP_HOST_ASSERT(
            protocol_layout.source_shuffle.get_smem_bytes() <=
            jit::device_runtime->get_num_smem_bytes());

        if (num_send_records > 0) {
            EP_HOST_ASSERT(
                send_manifest.select(1, 0).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 0).max().item<int64_t>() <
                num_tokens_i64);
            EP_HOST_ASSERT(
                send_manifest.select(1, 1).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 2).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 3).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 3).max().item<int>() <
                nccl_context->num_ranks);
            EP_HOST_ASSERT(
                (send_manifest.select(1, 3) !=
                 nccl_context->rank_idx).all().item<bool>());
            EP_HOST_ASSERT(
                send_manifest.select(1, 4).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 5).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 6).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 6).max().item<int>() <
                physical_capacity);
        }

        const auto stream = at::cuda::getCurrentCUDAStream();
        EP_HOST_ASSERT(stream.id() != comm_stream.id());
        auto arena = math::advance_ptr(buffer, arena_offset);
        const auto mapped_layout = rail_balance::OneShotProtocolLayout(
            hidden * sizeof(c10::BFloat16),
            num_topk,
            physical_capacity,
            arena);

        auto consumed_records = torch::full(
            {num_recv_records,
             mapped_layout.source_shuffle.record_bytes},
            0xCC,
            x.options().dtype(torch::kUInt8));
        auto ready_values = torch::full(
            {physical_capacity}, -1, send_manifest.options());
        auto publish_sequence_values = torch::full(
            {physical_capacity}, -1, fingerprints.options());
        auto consume_counts = torch::zeros(
            {physical_capacity}, send_manifest.options());
        auto ready_count_at_consume = torch::full(
            {physical_capacity}, -1, send_manifest.options());
        auto trace = torch::full(
            {physical_capacity + 1, 4}, -1, fingerprints.options());
        auto control_snapshot = torch::full(
            {static_cast<int64_t>(
                sizeof(rail_balance::OneShotProxyControl))},
            0xCC,
            x.options().dtype(torch::kUInt8));
        auto producer_status = torch::full(
            {num_send_records}, -1, send_manifest.options());

        // All generated cubins are built and loaded before the consumer starts
        // its bounded polling loop. Cold JIT work after this point would create
        // a false timeout.
        const auto prepared =
            prepare_rail_balance_protocol(hidden, num_topk);

        std::vector<int> initial_ready(
            physical_capacity, stale_generation);
        std::vector<uint64_t> initial_sequence(physical_capacity, 0);
        if (stale_generation > 0) {
            for (int slot = 0; slot < physical_capacity; ++slot)
                initial_sequence[slot] =
                    rail_balance::make_publish_key(
                        stale_generation, slot);
        }
        rail_balance::OneShotProxyControl initial_control = {};
        initial_control.published_tail =
            rail_balance::pack_generation_tail(generation, 0);
        initial_control.consumed_tail =
            rail_balance::pack_generation_tail(generation, 0);
        initial_control.generation = generation;
        initial_control.expected_records = num_recv_records;
        initial_control.status = static_cast<int>(
            rail_balance::OneShotProtocolStatus::Idle);
        initial_control.error_code = static_cast<int>(
            rail_balance::OneShotProtocolError::None);
        initial_control.error_slot = -1;
        initial_control.first_hole_tail = -1;
        initial_control.first_hole_ready_slot = -1;

        if (not preserve_record_payload) {
            CUDA_RUNTIME_CHECK(cudaMemsetAsync(
                arena,
                0xA5,
                mapped_layout.source_shuffle.ready_offset,
                stream));
        }
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            mapped_layout.source_shuffle.get_ready_ptr(),
            initial_ready.data(),
            static_cast<int64_t>(physical_capacity) * sizeof(int),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            mapped_layout.get_publish_sequence_ptr(),
            initial_sequence.data(),
            static_cast<int64_t>(physical_capacity) *
                sizeof(uint64_t),
            cudaMemcpyHostToDevice,
            stream));
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            mapped_layout.get_control_ptr(),
            &initial_control,
            sizeof(initial_control),
            cudaMemcpyHostToDevice,
            stream));

        // The only collective barrier is before either protocol kernel. It
        // makes every stale seed visible; no producer/consumer overlap is
        // replaced by a post-producer barrier.
        barrier(false, true);
        launch_prepared_rail_balance_protocol_consumer(
            prepared,
            arena,
            consumed_records.data_ptr(),
            ready_values.data_ptr<int>(),
            reinterpret_cast<uint64_t*>(
                publish_sequence_values.data_ptr<int64_t>()),
            consume_counts.data_ptr<int>(),
            ready_count_at_consume.data_ptr<int>(),
            trace.data_ptr<int64_t>(),
            control_snapshot.data_ptr(),
            consumer_delay_cycles.data_ptr<int64_t>(),
            physical_capacity,
            generation,
            nccl_context->rank_idx,
            num_recv_records,
            timeout_cycles,
            comm_stream);
        if (num_send_records > 0) {
            for (int producer_phase = 0;
                 producer_phase < 3; ++producer_phase) {
                launch_prepared_rail_balance_protocol_producer(
                    prepared,
                    nccl_context->dev_comm,
                    nccl_context->window,
                    x.data_ptr(),
                    topk_idx.data_ptr<topk_idx_t>(),
                    topk_weights.data_ptr<float>(),
                    send_manifest.data_ptr<int>(),
                    fingerprints.data_ptr<int64_t>(),
                    producer_delay_cycles.data_ptr<int64_t>(),
                    producer_status.data_ptr<int>(),
                    arena,
                    physical_capacity,
                    generation,
                    nccl_context->rank_idx,
                    num_max_tokens_per_rank,
                    num_send_records,
                    producer_phase,
                    static_cast<int>(force_odd_first),
                    drop_physical_slot,
                    late_publish_start,
                    late_publish_after_consumed,
                    timeout_cycles,
                    static_cast<int>(
                        protocol_layout.source_shuffle.get_smem_bytes()),
                    stream);
            }
        }

        // This wait is enqueued after the producer on the compute stream, so
        // it cannot form a consumer↔producer stream-dependency cycle.
        stream_wait(stream, comm_stream);
        return {
            consumed_records,
            ready_values,
            publish_sequence_values,
            consume_counts,
            ready_count_at_consume,
            trace,
            control_snapshot,
            producer_status,
        };
    }

    RailBalanceVNodeTensors rail_balance_vnode_roundtrip(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& send_manifest,
        const torch::Tensor& fingerprints,
        const torch::Tensor& quota,
        const int64_t& arena_offset,
        const int& physical_capacity,
        const int& destination_capacity,
        const int& num_destinations,
        const int& generation,
        const int& num_max_tokens,
        const int& num_source_ranks,
        const int& expert_begin,
        const int& experts_per_rank) const {
        constexpr int64_t kArenaGuardBytes = 4096;
        // C060/C061 are private, finite, pure-single-node transport emulators.
        // They never change the public dispatch/combine ABI and must not be
        // confused with a real Hybrid/Gin topology.
        EP_HOST_ASSERT(not allow_hybrid_mode);
        EP_HOST_ASSERT(num_source_ranks > 0);
        EP_HOST_ASSERT(destination_capacity > 0);
        EP_HOST_ASSERT(num_destinations > 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(destination_capacity) *
                num_destinations == physical_capacity);
        EP_HOST_ASSERT(
            nccl_context->num_ranks ==
            num_source_ranks * (num_destinations + 1));
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1);
        EP_HOST_ASSERT(
            nccl_context->num_scaleup_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(nccl_context->num_rdma_ranks == 1);
        EP_HOST_ASSERT(
            nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(
            nccl_context->scaleup_rank_idx == nccl_context->rank_idx);
        EP_HOST_ASSERT(nccl_context->nvl_rank_idx == nccl_context->rank_idx);

        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.dim() == 2);
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(
            topk_idx.is_cuda() and topk_idx.is_contiguous() and
            topk_idx.dim() == 2);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(
            topk_weights.is_cuda() and topk_weights.is_contiguous() and
            topk_weights.dim() == 2);
        EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(
            send_manifest.is_cuda() and send_manifest.is_contiguous() and
            send_manifest.dim() == 2);
        EP_HOST_ASSERT(send_manifest.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(
            send_manifest.size(1) == rail_balance::kNumManifestFields);
        EP_HOST_ASSERT(
            fingerprints.is_cuda() and fingerprints.is_contiguous() and
            fingerprints.dim() == 1);
        EP_HOST_ASSERT(fingerprints.scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(
            quota.is_cuda() and quota.is_contiguous() and quota.dim() == 2);
        EP_HOST_ASSERT(quota.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(quota.size(0) == num_source_ranks);
        EP_HOST_ASSERT(quota.size(1) == num_destinations);

        const auto num_tokens_i64 = x.size(0);
        const auto hidden_i64 = x.size(1);
        const auto num_topk_i64 = topk_idx.size(1);
        const auto num_send_records_i64 = send_manifest.size(0);
        EP_HOST_ASSERT(num_tokens_i64 == num_max_tokens);
        EP_HOST_ASSERT(topk_idx.size(0) == num_tokens_i64);
        EP_HOST_ASSERT(topk_weights.sizes() == topk_idx.sizes());
        EP_HOST_ASSERT(fingerprints.size(0) == num_send_records_i64);
        EP_HOST_ASSERT(hidden_i64 > 0 and hidden_i64 <= INT_MAX);
        EP_HOST_ASSERT(hidden_i64 % 16 == 0);
        EP_HOST_ASSERT(
            hidden_i64 * static_cast<int64_t>(sizeof(c10::BFloat16)) <=
            INT_MAX);
        EP_HOST_ASSERT(num_topk_i64 > 0 and num_topk_i64 <= 32);
        EP_HOST_ASSERT(num_send_records_i64 <= INT_MAX);
        EP_HOST_ASSERT(physical_capacity > 0);
        EP_HOST_ASSERT(generation > 0);
        EP_HOST_ASSERT(num_max_tokens > 0);
        EP_HOST_ASSERT(expert_begin >= 0);
        EP_HOST_ASSERT(experts_per_rank > 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(expert_begin) +
                static_cast<int64_t>(num_source_ranks) * num_destinations *
                    experts_per_rank <=
            INT_MAX);
        EP_HOST_ASSERT(
            static_cast<int64_t>(num_source_ranks - 1) * num_max_tokens +
                num_max_tokens - 1 <=
            INT_MAX);

        const int device_index = x.get_device();
        EP_HOST_ASSERT(device_index == this->device_index);
        EP_HOST_ASSERT(topk_idx.get_device() == device_index);
        EP_HOST_ASSERT(topk_weights.get_device() == device_index);
        EP_HOST_ASSERT(send_manifest.get_device() == device_index);
        EP_HOST_ASSERT(fingerprints.get_device() == device_index);
        EP_HOST_ASSERT(quota.get_device() == device_index);
        const c10::cuda::CUDAGuard device_guard(x.device());

        const int hidden = static_cast<int>(hidden_i64);
        const int num_topk = static_cast<int>(num_topk_i64);
        const int num_send_records =
            static_cast<int>(num_send_records_i64);
        const int rank_idx = nccl_context->rank_idx;
        const bool is_source = rank_idx < num_source_ranks;
        const auto quota_cpu = quota.cpu().contiguous();
        const auto quota_ptr = quota_cpu.data_ptr<int>();
        for (int egress = 0; egress < num_source_ranks; ++egress)
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const int value = quota_ptr[
                    egress * num_destinations + destination];
                EP_HOST_ASSERT(
                    value >= 0 and value <= destination_capacity);
            }
        if (is_source) {
            EP_HOST_ASSERT(topk_idx.min().item<topk_idx_t>() >= -1);
            EP_HOST_ASSERT(
                not topk_idx.ge(0)
                        .logical_and(topk_idx.lt(expert_begin))
                        .any()
                        .item<bool>());
            EP_HOST_ASSERT(
                topk_idx.max().item<int64_t>() <
                static_cast<int64_t>(expert_begin) +
                    num_source_ranks * num_destinations * experts_per_rank);
            const auto valid_lanes = topk_idx.ge(0).sum(1);
            EP_HOST_ASSERT(
                valid_lanes.eq(0)
                    .logical_or(valid_lanes.eq(num_topk))
                    .all()
                    .item<bool>());
        } else {
            EP_HOST_ASSERT(num_send_records == 0);
        }

        const int64_t rail_capacity_i64 =
            static_cast<int64_t>(physical_capacity) * (num_topk + 1);
        const int64_t expert_capacity_i64 =
            static_cast<int64_t>(num_source_ranks) *
            destination_capacity * num_topk;
        const int64_t partial_capacity_i64 =
            static_cast<int64_t>(num_max_tokens) * num_topk;
        EP_HOST_ASSERT(
            rail_capacity_i64 > 0 and rail_capacity_i64 <= INT_MAX);
        EP_HOST_ASSERT(
            expert_capacity_i64 > 0 and expert_capacity_i64 <= INT_MAX);
        EP_HOST_ASSERT(
            partial_capacity_i64 > 0 and partial_capacity_i64 <= INT_MAX);
        const int rail_capacity = static_cast<int>(rail_capacity_i64);
        const int expert_capacity = static_cast<int>(expert_capacity_i64);

        if (num_send_records > 0) {
            EP_HOST_ASSERT(
                send_manifest.select(1, 0).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 0).max().item<int64_t>() <
                num_tokens_i64);
            EP_HOST_ASSERT(
                send_manifest.select(1, 1).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 1).max().item<int>() <
                num_destinations);
            EP_HOST_ASSERT(
                send_manifest.select(1, 2).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 3).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 3).max().item<int>() <
                num_source_ranks);
            EP_HOST_ASSERT(
                send_manifest.select(1, 4).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 5).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 6).min().item<int>() >= 0);
            EP_HOST_ASSERT(
                send_manifest.select(1, 6).max().item<int>() <
                physical_capacity);

            const auto manifest_cpu = send_manifest.cpu().contiguous();
            const auto manifest_ptr = manifest_cpu.data_ptr<int>();
            for (int row = 0; row < num_send_records; ++row) {
                const auto fields = manifest_ptr +
                    static_cast<int64_t>(row) *
                        rail_balance::kNumManifestFields;
                const int destination = fields[1];
                const int egress = fields[3];
                const int logical_slot = fields[5];
                const int physical_slot = fields[6];
                EP_HOST_ASSERT(
                    destination >= 0 and
                    destination < num_destinations);
                EP_HOST_ASSERT(
                    egress >= 0 and egress < num_source_ranks);
                EP_HOST_ASSERT(
                    physical_slot / destination_capacity == destination);
                EP_HOST_ASSERT(
                    physical_slot % destination_capacity <
                    quota_ptr[egress * num_destinations + destination]);
                EP_HOST_ASSERT(
                    logical_slot >= 0 and
                    logical_slot <
                    quota_ptr[egress * num_destinations + destination]);
            }
        }

        const auto layout = rail_balance::VNodeRoundTripLayout(
            hidden * sizeof(c10::BFloat16), num_topk,
            num_destinations, destination_capacity,
            num_source_ranks, num_max_tokens);
        EP_HOST_ASSERT(
            arena_offset >= 0 and
            arena_offset % rail_balance::kArenaAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        EP_HOST_ASSERT(
            layout.arena_bytes + kArenaGuardBytes <=
            num_gpu_buffer_bytes - arena_offset);
        EP_HOST_ASSERT(
            layout.rail.records.get_smem_bytes() <=
            jit::device_runtime->get_num_smem_bytes());

        const auto stream = at::cuda::getCurrentCUDAStream();
        auto arena = math::advance_ptr(buffer, arena_offset);
        const auto mapped_layout = rail_balance::VNodeRoundTripLayout(
            hidden * sizeof(c10::BFloat16), num_topk,
            num_destinations, destination_capacity,
            num_source_ranks, num_max_tokens, arena);
        const int status_stride = std::max({
            num_max_tokens,
            physical_capacity * num_topk,
            expert_capacity,
        });
        auto combined_output = torch::zeros_like(x);
        auto output_ready = torch::zeros(
            {num_max_tokens}, send_manifest.options());
        auto owner_partials = torch::zeros(
            {num_max_tokens, num_topk, hidden}, x.options());
        auto owner_partial_ready = torch::zeros(
            {num_max_tokens, num_topk}, send_manifest.options());
        auto rail_records = torch::zeros(
            {rail_capacity, mapped_layout.rail.records.record_bytes},
            x.options().dtype(torch::kUInt8));
        auto rail_ready = torch::zeros(
            {rail_capacity}, send_manifest.options());
        auto rail_routes = torch::zeros(
            {rail_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            x.options().dtype(torch::kUInt8));
        auto expert_records = torch::zeros(
            {expert_capacity, mapped_layout.expert.records.record_bytes},
            x.options().dtype(torch::kUInt8));
        auto expert_ready = torch::zeros(
            {expert_capacity}, send_manifest.options());
        auto expert_routes = torch::zeros(
            {expert_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            x.options().dtype(torch::kUInt8));
        auto stage_status = torch::zeros(
            {6, status_stride}, send_manifest.options());
        auto arena_guard = torch::zeros(
            {kArenaGuardBytes}, x.options().dtype(torch::kUInt8));

        // All ranks build all seven cubins before the first collective. A
        // cold NVRTC compile inside one virtual role would make its peers wait
        // at different barriers and can create a false timeout.
        const auto prepared_source =
            prepare_rail_balance_source_shuffle(hidden, num_topk);
        const auto prepared_vnode =
            prepare_rail_balance_vnode(hidden, num_topk);

        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            arena, 0, mapped_layout.arena_bytes, stream));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            math::advance_ptr(arena, mapped_layout.arena_bytes),
            0xA5, kArenaGuardBytes, stream));
        barrier(false, true);

        if (is_source and num_send_records > 0)
            launch_prepared_rail_balance_source_shuffle(
                prepared_source,
                nccl_context->dev_comm, nccl_context->window,
                x.data_ptr(), topk_idx.data_ptr<topk_idx_t>(),
                topk_weights.data_ptr<float>(),
                send_manifest.data_ptr<int>(),
                fingerprints.data_ptr<int64_t>(),
                arena, hidden, num_topk, rail_capacity, generation,
                rank_idx, num_max_tokens, num_send_records,
                static_cast<int>(mapped_layout.rail.records.get_smem_bytes()),
                stream);
        barrier(false, true);

        int* status_ptr = stage_status.data_ptr<int>();
        const int smem_bytes = static_cast<int>(
            mapped_layout.rail.records.get_smem_bytes());
        if (is_source)
            launch_prepared_rail_balance_vnode_peer(
                prepared_vnode.scaleout,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr, quota.data_ptr<int>(),
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity, smem_bytes, stream);
        barrier(false, true);

        if (not is_source)
            launch_prepared_rail_balance_vnode_expert(
                prepared_vnode.forward,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + status_stride, quota.data_ptr<int>(),
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity * num_topk, smem_bytes, stream);
        barrier(false, true);

        if (not is_source)
            launch_prepared_rail_balance_vnode_expert(
                prepared_vnode.expert,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + 2 * status_stride,
                quota.data_ptr<int>(), num_destinations,
                destination_capacity, num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                expert_capacity, smem_bytes, stream);
        barrier(false, true);

        if (not is_source)
            launch_prepared_rail_balance_vnode_peer(
                prepared_vnode.return_path,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + 3 * status_stride,
                quota.data_ptr<int>(), num_destinations,
                destination_capacity, num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity * num_topk, smem_bytes, stream);
        barrier(false, true);

        if (is_source)
            launch_prepared_rail_balance_vnode_peer(
                prepared_vnode.unshuffle,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + 4 * status_stride,
                quota.data_ptr<int>(), num_destinations,
                destination_capacity, num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity * num_topk, smem_bytes, stream);
        barrier(false, true);

        if (is_source)
            launch_prepared_rail_balance_vnode_reduce(
                prepared_vnode, arena,
                topk_idx.data_ptr<topk_idx_t>(),
                reinterpret_cast<nv_bfloat16*>(
                    combined_output.data_ptr<c10::BFloat16>()),
                output_ready.data_ptr<int>(),
                status_ptr + 5 * status_stride,
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, stream);
        barrier(false, true);

        const auto copy_snapshot = [&](void* dst, const void* src,
                                       const int64_t num_bytes) {
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream));
        };
        copy_snapshot(
            rail_records.data_ptr(),
            mapped_layout.rail.records.get_record_ptr(0),
            static_cast<int64_t>(rail_capacity) *
                mapped_layout.rail.records.record_bytes);
        copy_snapshot(
            rail_ready.data_ptr<int>(),
            mapped_layout.rail.records.get_ready_ptr(),
            static_cast<int64_t>(rail_capacity) * sizeof(int));
        copy_snapshot(
            rail_routes.data_ptr(), mapped_layout.rail.get_route_ptr(),
            static_cast<int64_t>(rail_capacity) *
                sizeof(rail_balance::VNodeRoute));
        if (is_source) {
            copy_snapshot(
                owner_partials.data_ptr(),
                mapped_layout.owner.get_value_ptr(0, 0),
                partial_capacity_i64 * hidden *
                    static_cast<int64_t>(sizeof(c10::BFloat16)));
            copy_snapshot(
                owner_partial_ready.data_ptr<int>(),
                mapped_layout.owner.get_ready_ptr(0, 0),
                partial_capacity_i64 * sizeof(int));
        } else {
            copy_snapshot(
                expert_records.data_ptr(),
                mapped_layout.expert.records.get_record_ptr(0),
                static_cast<int64_t>(expert_capacity) *
                    mapped_layout.expert.records.record_bytes);
            copy_snapshot(
                expert_ready.data_ptr<int>(),
                mapped_layout.expert.records.get_ready_ptr(),
                static_cast<int64_t>(expert_capacity) * sizeof(int));
            copy_snapshot(
                expert_routes.data_ptr(),
                mapped_layout.expert.get_route_ptr(),
                static_cast<int64_t>(expert_capacity) *
                    sizeof(rail_balance::VNodeRoute));
        }
        copy_snapshot(
            arena_guard.data_ptr(),
            math::advance_ptr(arena, mapped_layout.arena_bytes),
            kArenaGuardBytes);
        return {
            combined_output,
            output_ready,
            owner_partials,
            owner_partial_ready,
            rail_records,
            rail_ready,
            rail_routes,
            expert_records,
            expert_ready,
            expert_routes,
            stage_status,
            arena_guard,
        };
    }

    RailBalanceVNodeTensors rail_balance_vnode_replay(
        const torch::Tensor& topk_idx,
        const torch::Tensor& quota,
        const torch::Tensor& rail_base_records,
        const torch::Tensor& rail_base_ready,
        const torch::Tensor& expert_records_snapshot,
        const torch::Tensor& expert_ready_snapshot,
        const torch::Tensor& expert_routes_snapshot,
        const int& hidden,
        const int64_t& arena_offset,
        const int& physical_capacity,
        const int& destination_capacity,
        const int& num_destinations,
        const int& generation,
        const int& num_max_tokens,
        const int& num_source_ranks,
        const int& expert_begin,
        const int& experts_per_rank) const {
        constexpr int64_t kArenaGuardBytes = 4096;
        // C070 is a deliberately narrow replay proof for the frozen C061
        // 2-source x 3-destination virtual topology.  Its input snapshots are
        // ordinary owning CUDA tensors; only this symmetric arena is ever
        // passed to LSA address translation.
        EP_HOST_ASSERT(not allow_hybrid_mode);
        EP_HOST_ASSERT(num_source_ranks == 2);
        EP_HOST_ASSERT(num_destinations == 3);
        EP_HOST_ASSERT(destination_capacity == 6);
        EP_HOST_ASSERT(physical_capacity == 18);
        EP_HOST_ASSERT(
            destination_capacity * num_destinations == physical_capacity);
        EP_HOST_ASSERT(
            nccl_context->num_ranks ==
            num_source_ranks * (num_destinations + 1));
        EP_HOST_ASSERT(nccl_context->num_scaleout_ranks == 1);
        EP_HOST_ASSERT(
            nccl_context->num_scaleup_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(nccl_context->num_rdma_ranks == 1);
        EP_HOST_ASSERT(
            nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        EP_HOST_ASSERT(
            nccl_context->scaleup_rank_idx == nccl_context->rank_idx);
        EP_HOST_ASSERT(nccl_context->nvl_rank_idx == nccl_context->rank_idx);

        EP_HOST_ASSERT(hidden > 0 and hidden % 16 == 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(hidden) * sizeof(c10::BFloat16) <= INT_MAX);
        EP_HOST_ASSERT(generation > 0);
        EP_HOST_ASSERT(num_max_tokens > 0);
        EP_HOST_ASSERT(expert_begin >= 0);
        EP_HOST_ASSERT(experts_per_rank > 0);
        EP_HOST_ASSERT(
            static_cast<int64_t>(expert_begin) +
                static_cast<int64_t>(num_source_ranks) * num_destinations *
                    experts_per_rank <=
            INT_MAX);

        EP_HOST_ASSERT(
            topk_idx.is_cuda() and topk_idx.is_contiguous() and
            topk_idx.dim() == 2);
        EP_HOST_ASSERT(
            topk_idx.scalar_type() ==
            c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.size(0) == num_max_tokens);
        const auto num_topk_i64 = topk_idx.size(1);
        EP_HOST_ASSERT(num_topk_i64 > 0 and num_topk_i64 <= 32);
        const int num_topk = static_cast<int>(num_topk_i64);

        EP_HOST_ASSERT(
            quota.is_cuda() and quota.is_contiguous() and quota.dim() == 2);
        EP_HOST_ASSERT(quota.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(quota.size(0) == num_source_ranks);
        EP_HOST_ASSERT(quota.size(1) == num_destinations);

        const auto layout = rail_balance::VNodeRoundTripLayout(
            hidden * sizeof(c10::BFloat16), num_topk,
            num_destinations, destination_capacity,
            num_source_ranks, num_max_tokens);
        const int expert_capacity = layout.expert_capacity;
        const int rail_capacity = layout.rail_capacity;
        const int64_t record_bytes = layout.rail.records.record_bytes;
        EP_HOST_ASSERT(
            rail_base_records.is_cuda() and
            rail_base_records.is_contiguous() and
            rail_base_records.scalar_type() == torch::kUInt8 and
            rail_base_records.dim() == 2 and
            rail_base_records.size(0) == physical_capacity and
            rail_base_records.size(1) == record_bytes);
        EP_HOST_ASSERT(
            rail_base_ready.is_cuda() and rail_base_ready.is_contiguous() and
            rail_base_ready.scalar_type() == torch::kInt32 and
            rail_base_ready.dim() == 1 and
            rail_base_ready.size(0) == physical_capacity);
        EP_HOST_ASSERT(
            expert_records_snapshot.is_cuda() and
            expert_records_snapshot.is_contiguous() and
            expert_records_snapshot.scalar_type() == torch::kUInt8 and
            expert_records_snapshot.dim() == 2 and
            expert_records_snapshot.size(0) == expert_capacity and
            expert_records_snapshot.size(1) == record_bytes);
        EP_HOST_ASSERT(
            expert_ready_snapshot.is_cuda() and
            expert_ready_snapshot.is_contiguous() and
            expert_ready_snapshot.scalar_type() == torch::kInt32 and
            expert_ready_snapshot.dim() == 1 and
            expert_ready_snapshot.size(0) == expert_capacity);
        EP_HOST_ASSERT(
            expert_routes_snapshot.is_cuda() and
            expert_routes_snapshot.is_contiguous() and
            expert_routes_snapshot.scalar_type() == torch::kUInt8 and
            expert_routes_snapshot.dim() == 2 and
            expert_routes_snapshot.size(0) == expert_capacity and
            expert_routes_snapshot.size(1) ==
                static_cast<int64_t>(sizeof(rail_balance::VNodeRoute)));

        const int device_index = topk_idx.get_device();
        EP_HOST_ASSERT(device_index == this->device_index);
        EP_HOST_ASSERT(quota.get_device() == device_index);
        EP_HOST_ASSERT(rail_base_records.get_device() == device_index);
        EP_HOST_ASSERT(rail_base_ready.get_device() == device_index);
        EP_HOST_ASSERT(expert_records_snapshot.get_device() == device_index);
        EP_HOST_ASSERT(expert_ready_snapshot.get_device() == device_index);
        EP_HOST_ASSERT(expert_routes_snapshot.get_device() == device_index);
        const c10::cuda::CUDAGuard device_guard(topk_idx.device());

        const int rank_idx = nccl_context->rank_idx;
        const bool is_source = rank_idx < num_source_ranks;
        const int local_egress = is_source
            ? rank_idx
            : (rank_idx - num_source_ranks) % num_source_ranks;
        const int local_destination = is_source
            ? -1
            : (rank_idx - num_source_ranks) / num_source_ranks;

        const auto quota_cpu = quota.cpu().contiguous();
        const auto quota_ptr = quota_cpu.data_ptr<int>();
        for (int egress = 0; egress < num_source_ranks; ++egress)
            for (int destination = 0;
                 destination < num_destinations; ++destination) {
                const int value = quota_ptr[
                    egress * num_destinations + destination];
                EP_HOST_ASSERT(
                    value >= 0 and value <= destination_capacity);
            }

        if (is_source) {
            EP_HOST_ASSERT(topk_idx.min().item<topk_idx_t>() >= -1);
            EP_HOST_ASSERT(
                not topk_idx.ge(0)
                        .logical_and(topk_idx.lt(expert_begin))
                        .any()
                        .item<bool>());
            EP_HOST_ASSERT(
                topk_idx.max().item<int64_t>() <
                static_cast<int64_t>(expert_begin) +
                    num_source_ranks * num_destinations * experts_per_rank);
            const auto valid_lanes = topk_idx.ge(0).sum(1);
            EP_HOST_ASSERT(
                valid_lanes.eq(0)
                    .logical_or(valid_lanes.eq(num_topk))
                    .all()
                    .item<bool>());
        }

        // Reject stale/mismatched snapshots before publishing them into the
        // symmetric arena.  Base readiness is exact for the owning role;
        // every published record and expert route must carry this generation.
        const auto base_records_cpu = rail_base_records.cpu().contiguous();
        const auto base_ready_cpu = rail_base_ready.cpu().contiguous();
        const auto base_records_ptr =
            base_records_cpu.data_ptr<uint8_t>();
        const auto base_ready_ptr = base_ready_cpu.data_ptr<int>();
        for (int slot = 0; slot < physical_capacity; ++slot) {
            const int destination = slot / destination_capacity;
            const int destination_slot = slot % destination_capacity;
            const bool expected =
                (is_source or destination == local_destination) and
                destination_slot < quota_ptr[
                    local_egress * num_destinations + destination];
            EP_HOST_ASSERT(base_ready_ptr[slot] ==
                           (expected ? generation : 0));
            if (expected) {
                const auto record = rail_balance::SourceShuffleLayout(
                    hidden * sizeof(c10::BFloat16), num_topk, 1,
                    const_cast<uint8_t*>(
                        base_records_ptr + slot * record_bytes));
                const auto descriptor = record.get_descriptor_ptr(0);
                EP_HOST_ASSERT(descriptor->generation == generation);
                EP_HOST_ASSERT(descriptor->egress == local_egress);
                EP_HOST_ASSERT(descriptor->destination == destination);
                EP_HOST_ASSERT(descriptor->physical_slot == slot);
            }
        }

        const auto expert_records_cpu =
            expert_records_snapshot.cpu().contiguous();
        const auto expert_ready_cpu =
            expert_ready_snapshot.cpu().contiguous();
        const auto expert_routes_cpu =
            expert_routes_snapshot.cpu().contiguous();
        const auto expert_records_ptr =
            expert_records_cpu.data_ptr<uint8_t>();
        const auto expert_ready_ptr = expert_ready_cpu.data_ptr<int>();
        const auto expert_routes_ptr =
            expert_routes_cpu.data_ptr<uint8_t>();
        for (int slot = 0; slot < expert_capacity; ++slot) {
            const int ready = expert_ready_ptr[slot];
            EP_HOST_ASSERT(ready == 0 or ready == generation);
            if (is_source)
                EP_HOST_ASSERT(ready == 0);
            if (ready == generation) {
                const auto record = rail_balance::SourceShuffleLayout(
                    hidden * sizeof(c10::BFloat16), num_topk, 1,
                    const_cast<uint8_t*>(
                        expert_records_ptr + slot * record_bytes));
                const auto descriptor = record.get_descriptor_ptr(0);
                const auto route = reinterpret_cast<
                    const rail_balance::VNodeRoute*>(
                        expert_routes_ptr +
                        static_cast<int64_t>(slot) *
                            sizeof(rail_balance::VNodeRoute));
                EP_HOST_ASSERT(descriptor->generation == generation);
                EP_HOST_ASSERT(route->generation == generation);
                EP_HOST_ASSERT(route->expert_rank == rank_idx);
                EP_HOST_ASSERT(route->expert_slot == slot);
            }
        }

        EP_HOST_ASSERT(
            arena_offset >= 0 and
            arena_offset % rail_balance::kArenaAlignmentBytes == 0);
        EP_HOST_ASSERT(arena_offset <= num_gpu_buffer_bytes);
        EP_HOST_ASSERT(
            layout.arena_bytes + kArenaGuardBytes <=
            num_gpu_buffer_bytes - arena_offset);
        EP_HOST_ASSERT(
            layout.rail.records.get_smem_bytes() <=
            jit::device_runtime->get_num_smem_bytes());

        const auto stream = at::cuda::getCurrentCUDAStream();
        auto arena = math::advance_ptr(buffer, arena_offset);
        const auto mapped_layout = rail_balance::VNodeRoundTripLayout(
            hidden * sizeof(c10::BFloat16), num_topk,
            num_destinations, destination_capacity,
            num_source_ranks, num_max_tokens, arena);
        const auto arena_begin = reinterpret_cast<uintptr_t>(arena);
        const auto arena_end = arena_begin + mapped_layout.arena_bytes +
                               kArenaGuardBytes;
        const auto assert_snapshot_does_not_alias_arena =
            [&](const torch::Tensor& snapshot) {
                const auto begin = reinterpret_cast<uintptr_t>(
                    snapshot.data_ptr());
                const auto end = begin + snapshot.nbytes();
                EP_HOST_ASSERT(end <= arena_begin or begin >= arena_end);
            };
        assert_snapshot_does_not_alias_arena(rail_base_records);
        assert_snapshot_does_not_alias_arena(rail_base_ready);
        assert_snapshot_does_not_alias_arena(expert_records_snapshot);
        assert_snapshot_does_not_alias_arena(expert_ready_snapshot);
        assert_snapshot_does_not_alias_arena(expert_routes_snapshot);

        const int status_stride = std::max({
            num_max_tokens,
            physical_capacity * num_topk,
            expert_capacity,
        });
        auto combined_output = torch::zeros(
            {num_max_tokens, hidden},
            rail_base_records.options().dtype(torch::kBFloat16));
        auto output_ready = torch::zeros(
            {num_max_tokens}, quota.options());
        auto owner_partials = torch::zeros(
            {num_max_tokens, num_topk, hidden},
            rail_base_records.options().dtype(torch::kBFloat16));
        auto owner_partial_ready = torch::zeros(
            {num_max_tokens, num_topk}, quota.options());
        auto rail_records = torch::zeros(
            {rail_capacity, record_bytes}, rail_base_records.options());
        auto rail_ready = torch::zeros(
            {rail_capacity}, quota.options());
        auto rail_routes = torch::zeros(
            {rail_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            rail_base_records.options());
        auto expert_records = torch::zeros(
            {expert_capacity, record_bytes}, rail_base_records.options());
        auto expert_ready = torch::zeros(
            {expert_capacity}, quota.options());
        auto expert_routes = torch::zeros(
            {expert_capacity,
             static_cast<int64_t>(sizeof(rail_balance::VNodeRoute))},
            rail_base_records.options());
        auto stage_status = torch::zeros(
            {4, status_stride}, quota.options());
        auto arena_guard = torch::zeros(
            {kArenaGuardBytes}, rail_base_records.options());

        // Build on every rank before entering a collective.  Only the four
        // post-forward stages below are launched by this replay entry point.
        const auto prepared_vnode =
            prepare_rail_balance_vnode(hidden, num_topk);

        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            arena, 0, mapped_layout.arena_bytes, stream));
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(
            math::advance_ptr(arena, mapped_layout.arena_bytes),
            0xA5, kArenaGuardBytes, stream));
        const auto restore = [&](void* dst, const void* src,
                                 const int64_t num_bytes) {
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream));
        };
        // Publication order is record -> route -> ready.  Only base rail
        // slots are restored; old contribution slots stay zero.  The source
        // role deliberately leaves the expert/owner overlay zeroed.
        restore(
            mapped_layout.rail.records.get_record_ptr(0),
            rail_base_records.data_ptr(),
            static_cast<int64_t>(physical_capacity) * record_bytes);
        if (not is_source) {
            restore(
                mapped_layout.expert.records.get_record_ptr(0),
                expert_records_snapshot.data_ptr(),
                static_cast<int64_t>(expert_capacity) * record_bytes);
            restore(
                mapped_layout.expert.get_route_ptr(),
                expert_routes_snapshot.data_ptr(),
                static_cast<int64_t>(expert_capacity) *
                    sizeof(rail_balance::VNodeRoute));
        }
        restore(
            mapped_layout.rail.records.get_ready_ptr(),
            rail_base_ready.data_ptr<int>(),
            static_cast<int64_t>(physical_capacity) * sizeof(int));
        if (not is_source)
            restore(
                mapped_layout.expert.records.get_ready_ptr(),
                expert_ready_snapshot.data_ptr<int>(),
                static_cast<int64_t>(expert_capacity) * sizeof(int));
        barrier(false, true);

        int* status_ptr = stage_status.data_ptr<int>();
        const int smem_bytes = static_cast<int>(
            mapped_layout.rail.records.get_smem_bytes());
        if (not is_source)
            launch_prepared_rail_balance_vnode_expert(
                prepared_vnode.expert,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr, quota.data_ptr<int>(),
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                expert_capacity, smem_bytes, stream);
        barrier(false, true);

        if (not is_source)
            launch_prepared_rail_balance_vnode_peer(
                prepared_vnode.return_path,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + status_stride, quota.data_ptr<int>(),
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity * num_topk, smem_bytes, stream);
        barrier(false, true);

        if (is_source)
            launch_prepared_rail_balance_vnode_peer(
                prepared_vnode.unshuffle,
                nccl_context->dev_comm, nccl_context->window,
                arena, status_ptr + 2 * status_stride,
                quota.data_ptr<int>(), num_destinations,
                destination_capacity, num_source_ranks, num_max_tokens,
                generation, rank_idx, expert_begin, experts_per_rank,
                physical_capacity * num_topk, smem_bytes, stream);
        barrier(false, true);

        if (is_source)
            launch_prepared_rail_balance_vnode_reduce(
                prepared_vnode, arena,
                topk_idx.data_ptr<topk_idx_t>(),
                reinterpret_cast<nv_bfloat16*>(
                    combined_output.data_ptr<c10::BFloat16>()),
                output_ready.data_ptr<int>(),
                status_ptr + 3 * status_stride,
                num_destinations, destination_capacity,
                num_source_ranks, num_max_tokens,
                generation, stream);
        barrier(false, true);

        const auto copy_snapshot = [&](void* dst, const void* src,
                                       const int64_t num_bytes) {
            CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
                dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream));
        };
        copy_snapshot(
            rail_records.data_ptr(),
            mapped_layout.rail.records.get_record_ptr(0),
            static_cast<int64_t>(rail_capacity) * record_bytes);
        copy_snapshot(
            rail_ready.data_ptr<int>(),
            mapped_layout.rail.records.get_ready_ptr(),
            static_cast<int64_t>(rail_capacity) * sizeof(int));
        copy_snapshot(
            rail_routes.data_ptr(), mapped_layout.rail.get_route_ptr(),
            static_cast<int64_t>(rail_capacity) *
                sizeof(rail_balance::VNodeRoute));
        if (is_source) {
            copy_snapshot(
                owner_partials.data_ptr(),
                mapped_layout.owner.get_value_ptr(0, 0),
                static_cast<int64_t>(num_max_tokens) * num_topk * hidden *
                    sizeof(c10::BFloat16));
            copy_snapshot(
                owner_partial_ready.data_ptr<int>(),
                mapped_layout.owner.get_ready_ptr(0, 0),
                static_cast<int64_t>(num_max_tokens) * num_topk *
                    sizeof(int));
        } else {
            copy_snapshot(
                expert_records.data_ptr(),
                mapped_layout.expert.records.get_record_ptr(0),
                static_cast<int64_t>(expert_capacity) * record_bytes);
            copy_snapshot(
                expert_ready.data_ptr<int>(),
                mapped_layout.expert.records.get_ready_ptr(),
                static_cast<int64_t>(expert_capacity) * sizeof(int));
            copy_snapshot(
                expert_routes.data_ptr(),
                mapped_layout.expert.get_route_ptr(),
                static_cast<int64_t>(expert_capacity) *
                    sizeof(rail_balance::VNodeRoute));
        }
        copy_snapshot(
            arena_guard.data_ptr(),
            math::advance_ptr(arena, mapped_layout.arena_bytes),
            kArenaGuardBytes);
        return {
            combined_output,
            output_ready,
            owner_partials,
            owner_partial_ready,
            rail_records,
            rail_ready,
            rail_routes,
            expert_records,
            expert_ready,
            expert_routes,
            stage_status,
            arena_guard,
        };
    }

    // ReSharper disable once CppMemberFunctionMayBeStatic
    void barrier(const bool& use_comm_stream, const bool& with_cpu_sync, const bool& sequential = true) const {
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        const auto stream = use_comm_stream ? comm_stream : compute_stream;
        if (use_comm_stream)
            stream_wait(comm_stream, compute_stream);

        // Wait all streams to finish on this GPU
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Launch GPU barrier
        launch_barrier(nccl_context->dev_comm, nccl_context->window,
                       workspace,
                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       num_gpu_timeout_cycles,
                       nccl_context->is_scaleup_nvlink,
                       sequential,
                       stream);

        // Let CPU wait
        if (with_cpu_sync)
            CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

        // Compute stream should also wait for the barrier
        if (use_comm_stream)
            stream_wait(compute_stream, comm_stream);
    }

    void engram_write(const torch::Tensor& storage, const std::optional<torch::Tensor>& sf) {
        // Ensure previous fetch are finished
        barrier(false, true);

        const auto compute_stream = at::cuda::getCurrentCUDAStream();

        // Check storage
        const auto [num_entries, hidden] = get_shape<2>(storage);
        EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16 or
                       storage.scalar_type() == torch::kFloat8_e4m3fn);
        EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
        num_engram_entries = num_entries, engram_hidden = hidden;

        // Store globally-replicated FP8 scaling factors for local gather during fetch.
        if (sf.has_value())
            EP_HOST_ASSERT(sf->dim() == 2 and sf->is_cuda() and sf->is_contiguous() and
                           sf->element_size() == sizeof(sf_pack_t));
        engram_sf = sf;

        // Write storage to CPU segment at back of buffer
        EP_HOST_ASSERT(storage.nbytes() <= num_cpu_buffer_bytes and "Engram storage exceeds CPU buffer size");
        const auto cpu_write_offset = allow_hybrid_mode
            ? static_cast<int64_t>(nccl_context->scaleup_rank_idx) * num_cpu_buffer_bytes : 0;
        CUDA_RUNTIME_CHECK(cudaMemcpyAsync(
            math::advance_ptr(buffer, num_gpu_buffer_bytes + cpu_write_offset),
            storage.data_ptr(), storage.nbytes(),
            cudaMemcpyDeviceToDevice, compute_stream));

        // Ensure data is visible for all ranks
        barrier(false, true);
    }

    std::function<std::tuple<torch::Tensor, std::optional<torch::Tensor>>()>
    engram_fetch(const torch::Tensor& indices, int num_qps, const bool& use_tma_aligned_col_major_sf) const {
        const auto use_fp8 = engram_sf.has_value();
        const auto fetched_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
        const int elem_size = use_fp8 ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto [num_tokens, num_entries_per_token] = get_shape<2>(indices);
        EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(indices.is_cuda() and indices.is_contiguous());
        EP_HOST_ASSERT(num_tokens * num_entries_per_token * engram_hidden * elem_size <= num_gpu_buffer_bytes);

        // Calculate a QP count
        if (num_qps == 0)
            num_qps = nccl_context->num_allocated_qps;

        // Return tensor from the raw buffer: each token's entries are concatenated along hidden
        EP_HOST_ASSERT(num_engram_entries > 0);
        const auto fetched = torch::from_blob(
            buffer,
            {num_tokens, engram_hidden * num_entries_per_token},
            torch::TensorOptions().dtype(fetched_dtype).device(torch::kCUDA)
        );

        auto fetched_sf = std::optional<torch::Tensor>();
        void* sf_table_ptr = nullptr;
        void* fetched_sf_ptr = nullptr;
        int num_sf_packs = 0, sf_token_stride = 0, sf_hidden_stride = 0;
        if (use_fp8) {
            num_sf_packs = static_cast<int>(engram_sf->size(1));
            if (use_tma_aligned_col_major_sf) {
                // TMA-aligned column-major layout for the next GEMM input
                sf_token_stride = 1, sf_hidden_stride = math::align(num_tokens, kNumAlignedSFPacks);
            } else {
                sf_token_stride = num_entries_per_token * num_sf_packs, sf_hidden_stride = 1;
            }
            fetched_sf = torch::empty_strided({num_tokens, num_entries_per_token * num_sf_packs},
                                              {sf_token_stride, sf_hidden_stride}, engram_sf->options());
            sf_table_ptr = engram_sf->data_ptr();
            fetched_sf_ptr = fetched_sf->data_ptr();
        }

        // Last issued Gin requests
        const auto num_gin_ranks = allow_hybrid_mode
            ? nccl_context->num_scaleout_ranks
            : nccl_context->num_ranks;
        const auto last_gin_requests = torch::empty(
            {num_gin_ranks * num_qps, sizeof(ncclGinRequest_t)},
            torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)
        );

        // Launch the fetch kernel
        launch_engram_fetch(
            nccl_context->dev_comm, nccl_context->window,
            math::advance_ptr(buffer, num_gpu_buffer_bytes),
            fetched.data_ptr(),
            indices.data_ptr<int>(),
            static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
            sf_table_ptr, fetched_sf_ptr, sf_token_stride, sf_hidden_stride,
            num_engram_entries,
            engram_hidden, elem_size, num_sf_packs,
            num_entries_per_token,
            num_tokens,
            nccl_context->num_scaleout_ranks,
            nccl_context->num_scaleup_ranks,
            num_cpu_buffer_bytes,
            num_qps,
            allow_hybrid_mode,
            at::cuda::getCurrentCUDAStream()
        );

        return [=, this]() -> std::tuple<torch::Tensor, std::optional<torch::Tensor>> {
            // Wait for all RDMA gets to complete
            launch_engram_fetch_wait(
                static_cast<ncclGinRequest_t*>(last_gin_requests.data_ptr()),
                nccl_context->dev_comm,
                nccl_context->window,
                nccl_context->num_scaleout_ranks,
                nccl_context->num_scaleup_ranks,
                num_qps,
                allow_hybrid_mode,
                at::cuda::getCurrentCUDAStream()
            );
            return {fetched, fetched_sf};
        };
    }

    void pp_set_config(const int64_t& num_max_tensor_bytes, const int& num_max_inflight_tensors) {
        // Flush previous operations
        barrier(false, true);

        EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
        EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);
        this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;
        this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
        this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);
        this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
    }

    void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);

        launch_pp_send(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void pp_recv(const torch::Tensor& x, const int& src_rank_idx, const int& num_sms) const {
        EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
        EP_HOST_ASSERT(src_rank_idx == prev_rank_idx or src_rank_idx == next_rank_idx);

        launch_pp_recv(
            nccl_context->dev_comm, nccl_context->window,
            x.data_ptr(), x.nbytes(),
            buffer, workspace,
            nccl_context->rank_idx, src_rank_idx, nccl_context->num_ranks,
            num_max_pp_tensor_bytes,
            num_max_pp_inflight_tensors,
            num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
            num_gpu_timeout_cycles,
            jit::device_runtime->get_num_smem_bytes(),
            at::cuda::getCurrentCUDAStream()
        );
    }

    void agrs_set_config(const int64_t& num_max_session_bytes,
                         const int& new_num_max_agrs_per_session) {
        // Flush previous operations
        barrier(true, true);

        EP_HOST_ASSERT(nccl_context->num_ranks > 1);
        EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);
        EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);
        EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS);
        EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
        this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32);
        this->num_max_agrs_per_session = new_num_max_agrs_per_session;
    }

    void create_agrs_session() {
        EP_HOST_ASSERT(not agrs_in_session);
        agrs_in_session = true;
        agrs_buffer_offset = 0;
        agrs_buffer_slot_idx = 0;
        agrs_session_idx += 1;
    }

    void destroy_agrs_session() {
        // Must be in a session
        EP_HOST_ASSERT(agrs_in_session);
        agrs_in_session = false;

        // Wait compute stream
        stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());

        // Notify that the buffer is now available & Wait for the buffer to be ready
        // NOTES: self-wait is guaranteed by in-stream order
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = static_cast<int*>(
                nccl_context->get_sym_ptr(workspace_layout_wo_expert->get_agrs_session_signal_ptr(nccl_context->rank_idx), dst_rank_idx));
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_session_signal_ptr(dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);
    }

    std::vector<torch::Tensor> agrs_get_inplace_tensor(const std::vector<int64_t>& num_bytes_list) const {
        EP_HOST_ASSERT(num_bytes_list.size() >= 1);
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);

        std::vector<torch::Tensor> out;
        out.reserve(num_bytes_list.size());
        int64_t offset = agrs_buffer_offset;
        for (const auto& num_bytes: num_bytes_list) {
            EP_HOST_ASSERT(offset + num_bytes * nccl_context->num_ranks <= num_max_agrs_session_bytes and
                           agrs_buffer_slot_idx < num_max_agrs_per_session and
                           "Not enough session buffer size. Did you forget to flush session?");
            out.push_back(torch::from_blob(math::advance_ptr(buffer, offset + num_bytes * nccl_context->rank_idx),
                          {num_bytes}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA)));
            offset += math::align<int64_t>(num_bytes * nccl_context->num_ranks, 32);
        }
        return out;
    }

    std::pair<std::vector<torch::Tensor>, std::function<void()>>
    all_gather(const std::vector<torch::Tensor>& tensors) {
        const int num_tensors = tensors.size();
        EP_HOST_ASSERT(num_max_agrs_session_bytes > 0 and num_max_agrs_per_session > 0 and agrs_in_session);
        EP_HOST_ASSERT(num_tensors >= 1);

        int num_copies = 0;
        std::vector<int64_t> offset(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            const auto& x = tensors[i];
            EP_HOST_ASSERT(x.is_contiguous());

            const auto x_offset = math::ptr_diff(x.data_ptr(), buffer);
            const bool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;
            offset[i] = agrs_buffer_offset;
            num_copies += nccl_context->num_ranks - is_inplace;
            agrs_buffer_offset += math::align<int64_t>(x.nbytes() * nccl_context->num_ranks, 32);
            EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));
        }
        EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and
                       agrs_buffer_slot_idx < num_max_agrs_per_session and
                       "Not enough session buffer size. Did you forget to flush session?");

        // Wait compute stream
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        stream_wait(comm_stream, compute_stream);

        // Send data to all ranks
        std::vector<size_t> sizes(num_copies);
        std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);
        int count = 0;
        for (int i = 0; i < nccl_context->num_ranks; ++ i) {
            for (int j = 0; j < num_tensors; ++ j) {
                const auto& x = tensors[j];
                const auto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;
                void* src_ptr = x.data_ptr();
                void* dst_ptr =
                    nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);
                if (src_ptr != dst_ptr) {
                    src_ptrs[count] = src_ptr;
                    dst_ptrs[count] = dst_ptr;
                    sizes[count] = x.nbytes();
                    count += 1;
                }
            }
        }
        cudaMemcpyAttributes attrs = {
            .srcAccessOrder = cudaMemcpySrcAccessOrderStream,
            .flags = cudaMemcpyFlagPreferOverlapWithCompute
        };
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, comm_stream));
#else
        CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), num_copies, attrs, nullptr, comm_stream));
#endif

        // Wait for data from other ranks
        const int current_session = agrs_session_idx;
        const int slot_idx = agrs_buffer_slot_idx;
        agrs_buffer_slot_idx += 1;
        std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
        std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
        for (int i = 0; i < nccl_context->num_ranks - 1; ++ i) {
            const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
            write_ptrs[i] = nccl_context->get_sym_ptr(
                workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, nccl_context->rank_idx), dst_rank_idx);
            wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);
        }
        cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);

        // Build output tensors eagerly
        std::vector<torch::Tensor> out(num_tensors);
        for (int i = 0; i < num_tensors; ++ i) {
            auto shape = tensors[i].sizes().vec();
            shape.insert(shape.begin(), nccl_context->num_ranks);
            out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());
        }

        // Return tensors and a handle to wait for data arrival
        const auto event = EventHandle(comm_stream);
        auto handle = [=, this]() {
            EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());
            EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);
            stream_wait(compute_stream, event);
        };
        return {std::move(out), std::move(handle)};
    }

    torch::cuda::CUDAStream stream_control_prologue(const std::optional<EventHandle>& previous_event,
                                                    const bool& allocate_on_comm_stream,
                                                    const bool& async_with_compute_stream) const {
        // Allocate all tensors on communication stream if set
        // NOTES: do not allocate tensors upfront!
        const auto compute_stream = at::cuda::getCurrentCUDAStream();
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(comm_stream);

        // Assertion for safety
        // `previous_event` implicitly means the overlapping computation kernels are launched first,
        // in order not to use the memory on the compute stream, we must allocate on the communication stream.
        // If you launch the communication kernels firstly, then `previous_event` must be unnecessary.
        if (previous_event.has_value())
            EP_HOST_ASSERT(allocate_on_comm_stream);

        // Wait previous tasks to finish
        if (previous_event.has_value()) {
            stream_wait(comm_stream, previous_event.value());
        } else {
            stream_wait(comm_stream, compute_stream);
        }
        return compute_stream;
    }

    void stream_control_before_epilogue(const std::optional<EventHandle>& previous_event_before_epilogue) const {
        if (previous_event_before_epilogue.has_value())
            stream_wait(comm_stream, previous_event_before_epilogue.value());
    }

    std::optional<EventHandle> stream_control_epilogue(const std::vector<std::optional<torch::Tensor>>& tensors,
                                                       const at::cuda::CUDAStream& compute_stream,
                                                       const bool& allocate_on_comm_stream,
                                                       const bool& async_with_compute_stream) const {
        // Ensure memory access safety between two streams
        std::optional<EventHandle> event;
        if (async_with_compute_stream) {
            event = EventHandle(comm_stream);

            // NOTES: this environment only applies to V2 APIs
            if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
                event->tensors_to_record = tensors;
            } else {
                for (auto& t: tensors) if (t.has_value()) {
                    t->record_stream(compute_stream);
                    t->record_stream(comm_stream);
                }
            }
        } else {
            stream_wait(compute_stream, comm_stream);
        }

        // Switch back compute stream
        if (allocate_on_comm_stream)
            at::cuda::setCurrentCUDAStream(compute_stream);

        // The CUDA event marking the finishing
        return event;
    }

    static int64_t get_dispatch_buffer_size(const int& num_max_tokens_per_rank,
                                            const int& hidden, const int& num_sf_packs, const int& num_topk,
                                            const int& elem_size,
                                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                            const bool& is_scaleup_nvlink) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct dispatch
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : 1, num_max_tokens_per_rank);
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_ranks, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid dispatch
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleup_ranks, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, 1, num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_scaleout_ranks,
                /* kNumChannels * kNumMaxTokensPerChannel */ num_max_tokens_per_rank + kNumMaxChannels);
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t get_combine_buffer_size(const int& num_max_tokens_per_rank, const int& hidden, const int& num_topk,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const bool& is_scaleup_nvlink,
                                           const bool& allow_multiple_reduction) {
        const auto num_ranks = num_scaleup_ranks * num_scaleout_ranks;
        const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);

        if (num_scaleout_ranks == 1) {
            // Direct combine
            const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_ranks, num_topk) : num_topk;
            const auto send_buffer_layout = layout::BufferLayout<false>(
                token_layout, is_scaleup_nvlink ? 0 : num_ranks,
                // For single reduction cases, the maximum number of received tokens is
                // `num_ranks * num_topk * num_max_tokens_per_rank` (we assume the bad case of `do_expand=True`)
                num_max_tokens_per_rank * (allow_multiple_reduction ? 1 : num_topk));
            const auto recv_buffer_layout = layout::BufferLayout<false>(
                token_layout, num_tokens_in_layout, num_max_tokens_per_rank);
            return send_buffer_layout.get_num_bytes() + recv_buffer_layout.get_num_bytes();
        } else {
            // Hybrid combine
            const int num_tokens_in_scaleup_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, num_topk) : num_topk;
            const int num_tokens_in_scaleout_layout = allow_multiple_reduction ? std::min(num_scaleout_ranks, num_topk) : num_topk;
            const auto scaleup_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleup_layout, num_scaleout_ranks * num_max_tokens_per_rank);
            const auto scaleout_recv_buffer = layout::BufferLayout<false>(
                token_layout, num_tokens_in_scaleout_layout, num_max_tokens_per_rank);
            const auto scaleout_send_buffer = layout::BufferLayout<false>(
                token_layout, allow_multiple_reduction ? 1 : num_topk,
                /* kNumChannels * num_scaleout_ranks * kNumMaxTokensPerChannel */
                num_scaleout_ranks * (num_max_tokens_per_rank + kNumMaxChannels));
            return scaleup_recv_buffer.get_num_bytes() +
                   scaleout_send_buffer.get_num_bytes() +
                   scaleout_recv_buffer.get_num_bytes();
        }
    }

    static int64_t calculate_buffer_size(const int64_t& nccl_comm,
                                         const int& num_max_tokens_per_rank, const int& hidden,
                                         int num_topk, const bool& use_fp8_dispatch,
                                         const bool& allow_hybrid_mode,
                                         const bool& allow_multiple_reduction) {
        EP_HOST_ASSERT(num_max_tokens_per_rank > 0 and hidden > 0);

        // The worst case SF bytes must be less than the main part
        EP_HOST_ASSERT(math::ceil_div(hidden, 32) * sizeof(float) <= hidden);

        // NOTES: there are lots of `kNumTopk <= 32` restrictions, so we use 32 to calculate token size
        num_topk = num_topk == 0 ? 32 : num_topk;

        // Topology
        const auto [num_rdma_ranks, num_nvl_ranks] = nccl::get_physical_domain_size(nccl_comm);
        const auto [num_scaleout_ranks, num_scaleup_ranks] = nccl::get_logical_domain_size(nccl_comm, allow_hybrid_mode);
        const auto is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

        // Dispatch size
        const auto elem_size = use_fp8_dispatch ? sizeof(__nv_fp8_e4m3) : sizeof(nv_bfloat16);
        const auto num_sf_packs = use_fp8_dispatch ? math::ceil_div(hidden, 32) : 0; // An approximation for number of SF packs
        const auto num_dispatch_bytes = get_dispatch_buffer_size(
            num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, elem_size,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink);

        // Combine layout
        const auto num_combine_bytes = get_combine_buffer_size(
            num_max_tokens_per_rank, hidden, num_topk,
            num_scaleout_ranks, num_scaleup_ranks,
            is_scaleup_nvlink, allow_multiple_reduction);

        // Return the maximum of those layouts, aligned to 2 MB
        return math::align(std::max(num_dispatch_bytes, num_combine_bytes), symmetric::kNumAlignmentBytes);
    }

    static symmetric::cpu_handle_t create_cpu_handle(const int64_t& num_cpu_bytes) {
        EP_HOST_ASSERT(num_cpu_bytes > 0 and num_cpu_bytes % symmetric::kNumAlignmentBytes == 0);
        return symmetric::HybridElasticSymmetricMemory::create_cpu_handle(num_cpu_bytes);
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               int, int,
               std::vector<int>,
               torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               std::optional<torch::Tensor>, std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    dispatch(const torch::Tensor& x,
             const std::optional<torch::Tensor>& sf,
             const torch::Tensor& topk_idx,
             const std::optional<torch::Tensor>& topk_weights,
             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
             const std::optional<int>& cached_num_recv_tokens,
             const std::optional<int>& cached_num_expanded_tokens,
             const std::optional<std::vector<int>>& cached_num_recv_tokens_per_expert_list,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_scaleup_rank,
             const std::optional<torch::Tensor>& cached_psum_num_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_num_unaligned_recv_tokens_per_expert,
             const std::optional<torch::Tensor>& cached_dst_buffer_slot_idx,
             const std::optional<torch::Tensor>& cached_token_metadata_at_forward,
             const std::optional<torch::Tensor>& cached_recv_src_metadata,
             const std::optional<torch::Tensor>& cached_channel_linked_list,
             const int& num_max_tokens_per_rank,
             const int& num_experts, const int& expert_alignment,
             const int& num_sms, const int& num_qps,
             const std::optional<EventHandle>& previous_event,
             const std::optional<EventHandle>& previous_event_before_epilogue,
             const bool& async_with_compute_stream,
             const bool& allocate_on_comm_stream,
             const bool& do_handle_copy, const bool& do_cpu_sync,
             const bool& do_expand, const bool& do_zero_padding,
             const bool& use_tma_aligned_col_major_sf) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        // Zero padding only makes sense with expand mode
        EP_HOST_ASSERT(not do_zero_padding or do_expand);

        // Cached mode must have responding handles
        const bool cached_mode = cached_num_recv_tokens.has_value();
        if (cached_mode) {
            EP_HOST_ASSERT(cached_num_recv_tokens.has_value());
            EP_HOST_ASSERT(cached_num_recv_tokens_per_expert_list.has_value());
            EP_HOST_ASSERT(cached_num_expanded_tokens.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_scaleup_rank.has_value());
            EP_HOST_ASSERT(cached_psum_num_recv_tokens_per_expert.has_value());
            EP_HOST_ASSERT(cached_dst_buffer_slot_idx.has_value());

            // Hybrid kernels require more
            if (nccl_context->num_scaleout_ranks > 1) {
                EP_HOST_ASSERT(cached_token_metadata_at_forward.has_value());
                EP_HOST_ASSERT(cached_channel_linked_list.has_value());
            }
        }

        // Check data tensor
        const auto [num_tokens, hidden] = get_shape<2>(x);
        const auto num_hidden_bytes = hidden * static_cast<int>(x.element_size());
        const auto num_local_experts = num_experts / nccl_context->num_ranks;
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
        EP_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);

        // Check SF stuffs
        int num_sf_packs = 0;
        void* sf_ptr = nullptr;
        int sf_token_stride = 0, sf_hidden_stride = 0;
        if (sf.has_value()) {
            // SF must be FP32 or packed UE8M0x4
            const auto [num_tokens_, num_sf_packs_] = get_shape<2>(sf.value());
            EP_HOST_ASSERT(num_tokens == num_tokens_);
            EP_HOST_ASSERT(sf->is_cuda());
            EP_HOST_ASSERT(sf->element_size() == sizeof(sf_pack_t));
            num_sf_packs = num_sf_packs_;
            sf_ptr = sf->data_ptr();
            sf_token_stride = sf->stride(0);
            sf_hidden_stride = sf->stride(1);
        }

        // Check top-k stuffs
        const auto [num_tokens_, num_topk] = get_shape<2>(topk_idx);
        EP_HOST_ASSERT(num_tokens == num_tokens_);
        EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(topk_idx.is_cuda() and topk_idx.is_contiguous());

        // Weights are optional for training backward
        float* topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            const auto [num_tokens__, num_topk_] = get_shape<2>(topk_weights.value());
            EP_HOST_ASSERT(num_tokens == num_tokens__);
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            topk_weights_ptr = topk_weights->data_ptr<float>();
        }

        // Expert receiving counter
        int* cumulative_local_expert_recv_stats_ptr = nullptr;
        if (cumulative_local_expert_recv_stats.has_value()) {
            const auto [num_local_experts_] = get_shape<1>(cumulative_local_expert_recv_stats.value());
            EP_HOST_ASSERT(cumulative_local_expert_recv_stats->is_cuda() and
                           cumulative_local_expert_recv_stats->is_contiguous());
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // The number of received tokens per expert
        // This is useful for expanding mode
        EP_HOST_ASSERT(num_experts % nccl_context->num_ranks == 0);
        auto psum_num_recv_tokens_per_expert = cached_psum_num_recv_tokens_per_expert.value_or(torch::Tensor());
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(psum_num_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.is_cuda() and psum_num_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            // NOTES: for expand mode, the input is exclusive prefix sum, while for non-expand, it is inclusive
            psum_num_recv_tokens_per_expert = torch::empty(
                {num_local_experts + 1}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // The unaligned (actual) number of received tokens per expert
        // Written by the dispatch kernel's notify warps, used by the epilogue for zero padding
        auto num_unaligned_recv_tokens_per_expert = cached_num_unaligned_recv_tokens_per_expert.value_or(torch::Tensor());
        int* num_unaligned_recv_tokens_per_expert_ptr = nullptr;
        if (cached_mode) {
            const auto& [num_local_experts_] = get_shape<1>(num_unaligned_recv_tokens_per_expert);
            EP_HOST_ASSERT(num_local_experts == num_local_experts_);
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.is_cuda() and num_unaligned_recv_tokens_per_expert.is_contiguous());
            EP_HOST_ASSERT(num_unaligned_recv_tokens_per_expert.scalar_type() == torch::kInt);
        } else {
            num_unaligned_recv_tokens_per_expert = torch::empty(
                {num_local_experts}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }
        num_unaligned_recv_tokens_per_expert_ptr = num_unaligned_recv_tokens_per_expert.data_ptr<int>();

        // The prefix sum tensor of number of received tokens from each rank
        // Will also be used in combine as the dispatch handle
        auto psum_num_recv_tokens_per_scaleup_rank = cached_psum_num_recv_tokens_per_scaleup_rank.value_or(torch::Tensor());
        if (cached_mode) {
            const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
            EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
            EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        } else {
            psum_num_recv_tokens_per_scaleup_rank = torch::empty(
                {nccl_context->num_scaleup_ranks}, at::TensorOptions(torch::kCUDA).dtype(torch::kInt));
        }

        // Decide number of channels by shared memory consumption
        // Only for hybrid version
        int num_channels_per_sm = 1, num_channels = 1;
        const int num_smem_bytes = jit::device_runtime->get_num_smem_bytes();
        if (nccl_context->num_scaleout_ranks > 1) {
            const auto dispatch_token_layout = get_dispatch_token_layout(hidden, x.element_size(), num_sf_packs, num_topk);
            const auto combine_token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
            EP_HOST_ASSERT(num_sms <= kNumMaxSMs);
            num_channels_per_sm = std::min<int>(
                (num_smem_bytes - get_num_notify_smem_bytes(nccl_context->num_ranks, num_experts)) / dispatch_token_layout.get_num_bytes<true>(),
                32 - kNumNotifyWarps);
            num_channels_per_sm = std::min<int>(
                num_smem_bytes / combine_token_layout.get_num_bytes<true>(),
                num_channels_per_sm);
            num_channels_per_sm = std::min<int>(
                /* 2 kinds of warps */ num_channels_per_sm / 2, kNumMaxChannelsPerSM);
            if (not prefer_overlap_with_compute)
                num_channels_per_sm = std::min<int>(num_channels_per_sm, 4);
            num_channels = num_sms * num_channels_per_sm;
            if (get_env<int>("EP_BUFFER_DEBUG"))
                printf("Elastic buffer uses %d channels per SM\n", num_channels_per_sm);
        }

        // Non-hybrid mode handles
        auto dst_buffer_slot_idx = cached_dst_buffer_slot_idx.value_or(torch::Tensor());
        if (nccl_context->num_scaleout_ranks == 1) {
            if (cached_mode) {
                const auto [num_tokens__, num_topk_] = get_shape<2>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                // Allocate a new tensor
                dst_buffer_slot_idx = torch::empty(
                    {num_tokens, num_topk}, torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));
            }
        }

        // Hybrid mode handles
        std::optional<torch::Tensor> token_metadata_at_forward, channel_linked_list;
        int *token_metadata_at_forward_ptr = nullptr, *channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            // The token destination slot idx during forward
            // `[i, j, k, l]` means: from channel i from scale-out peer k, the j-th token's index in the l-th rank buffer
            // NOTES: Used primarily for cached mode
            // TODO: May make it a linked list to remove the redundant info in `token_metadata_at_forward`
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels);
            if (cached_mode) {
                const auto [num_channels_, num_scaleout_ranks_, num_max_tokens_per_channel_, num_topk_] =
                    get_shape<4>(dst_buffer_slot_idx);
                EP_HOST_ASSERT(num_channels == num_channels_ and nccl_context->num_scaleout_ranks == num_scaleout_ranks_ and
                               num_max_tokens_per_channel == num_max_tokens_per_channel_ and num_topk == num_topk_);
                EP_HOST_ASSERT(dst_buffer_slot_idx.is_cuda() and dst_buffer_slot_idx.is_contiguous());
                EP_HOST_ASSERT(dst_buffer_slot_idx.scalar_type() == torch::kInt);
            } else {
                dst_buffer_slot_idx = torch::empty(
                    {num_channels, nccl_context->num_scaleout_ranks, num_max_tokens_per_channel, num_topk},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }

            // The token metadata during forward
            // `[i, j]` means: in channel i, the j-th forwarded token's metadata
            // Info contains:
            //   - Scaleout rank index and source token index in the original rank (0)
            //   - Whether the token is the last one in the chunk (1)
            //   - cached top-k scaleup peer indices (top-k)
            //   - each selections' destination slot indices (top-k)
            const auto num_max_forwarded_tokens = nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1;
            const auto num_forward_metadata_dims = 2 + num_topk * 2;
            if (cached_mode) {
                token_metadata_at_forward = cached_token_metadata_at_forward;
                const auto [num_channels_, num_max_forwarded_tokens_, num_forward_metadata_dims_] = get_shape<3>(token_metadata_at_forward.value());
                EP_HOST_ASSERT(num_channels == num_channels_ and num_max_forwarded_tokens == num_max_forwarded_tokens_
                               and num_forward_metadata_dims == num_forward_metadata_dims_);
                EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
                EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);
            } else {
                token_metadata_at_forward = torch::empty(
                    {num_channels, num_max_forwarded_tokens, num_forward_metadata_dims},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();

            // Per-scaleup-peer-per-channel linked list
            // `[i, j, k]` means: from channel i from scaleup peer k, the j-th token's index in the combine's input
            if (cached_mode) {
                channel_linked_list = cached_channel_linked_list;
                const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
                channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
                EP_HOST_ASSERT(num_channels == num_channels__);
                EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
                EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
                EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
                EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
            } else {
                channel_linked_list = torch::empty(
                    // Index 0 of the list means the starting item
                    {num_channels,
                    nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1,
                    nccl_context->num_scaleup_ranks},
                    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt)
                );
            }
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
        }

        // Clone `topk_idx` for saving in the handle (to prevent users' modification)
        auto copied_topk_idx = std::optional<torch::Tensor>();
        topk_idx_t* copied_topk_idx_ptr = nullptr;
        if (do_handle_copy and not cached_mode) {
            copied_topk_idx = torch::empty_like(topk_idx);
            copied_topk_idx_ptr = copied_topk_idx->data_ptr<topk_idx_t>();
        }

        // Check buffer size
        EP_HOST_ASSERT(get_dispatch_buffer_size(
                       num_max_tokens_per_rank, hidden, num_sf_packs, num_topk, x.element_size(),
                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                       nccl_context->is_scaleup_nvlink) <= num_buffer_bytes);

        // Ready and clean host workspace for this round
        const auto host_workspace_layout = layout::WorkspaceLayout(
            host_workspace,
            nccl_context->num_scaleout_ranks,
            nccl_context->num_scaleup_ranks,
            num_experts);
        std::fill_n(host_workspace_layout.get_scaleup_rank_count_ptr<false>(), nccl_context->num_scaleup_ranks, 0);
        std::fill_n(host_workspace_layout.get_scaleup_expert_count_ptr<false>(), num_local_experts, 0);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Do dispatch into the buffers (with SM limitation)
        EP_HOST_ASSERT(num_sms <= jit::device_runtime->get_num_sms());
        launch_dispatch(x.data_ptr(), sf_ptr,
                        topk_idx.data_ptr<topk_idx_t>(), topk_weights_ptr,
                        copied_topk_idx_ptr,
                        cumulative_local_expert_recv_stats_ptr,
                        psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                        psum_num_recv_tokens_per_expert.data_ptr<int>(),
                        num_unaligned_recv_tokens_per_expert_ptr,
                        dst_buffer_slot_idx.data_ptr<int>(),
                        token_metadata_at_forward_ptr,
                        num_tokens, num_max_tokens_per_rank,
                        hidden, x.element_size(),
                        num_sf_packs, sf_token_stride, sf_hidden_stride,
                        num_experts, num_topk, expert_alignment,
                        nccl_context->dev_comm, nccl_context->window,
                        buffer,
                        workspace, mapped_host_workspace,
                        nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                        nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                        nccl_context->is_scaleup_nvlink,
                        num_sms, num_channels_per_sm,
                        num_smem_bytes,
                        num_qps, num_gpu_timeout_cycles,
                        cached_mode, do_cpu_sync,
                        comm_stream);

        // Received token counters
        int num_recv_tokens = 0, num_expanded_tokens = 0;
        int counter_scaleup_rank_idx = 0, counter_local_expert_idx = 0;
        std::vector<int> num_recv_tokens_per_expert_list;

        // Assign these values according to modes
        if (cached_mode) {
            // Cached mode
            EP_HOST_ASSERT(not do_cpu_sync and "Cannot do CPU sync with cached mode");
            num_recv_tokens = cached_num_recv_tokens.value();
            num_recv_tokens_per_expert_list = cached_num_recv_tokens_per_expert_list.value();
            num_expanded_tokens = cached_num_expanded_tokens.value();
        } else if (do_cpu_sync) {
            // Non-cached mode with sync
            const auto start_cpu_time = std::chrono::high_resolution_clock::now();
            while (true) {
                bool ready = true;

                // Read number of received tokens from each scaleup rank
                while (counter_scaleup_rank_idx < nccl_context->num_scaleup_ranks and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_rank_count_ptr<false>()[counter_scaleup_rank_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens += count;
                        ++ counter_scaleup_rank_idx;
                    }
                }

                // Read expert counts
                while (counter_local_expert_idx < num_local_experts and ready) {
                    const auto count = math::encode_decode_positive(
                        host_workspace_layout.get_scaleup_expert_count_ptr<false>()[counter_local_expert_idx]);
                    if ((ready = math::is_decoded_positive_ready(count))) {
                        num_recv_tokens_per_expert_list.push_back(count);
                        num_expanded_tokens += count;
                        ++ counter_local_expert_idx;
                    }
                }

                // Ready and do next steps
                const auto get_buffer_info = [&]() {
                    std::stringstream ss;
                    ss << "CPU side received count (scaleup: " << nccl_context->scaleup_rank_idx << "): ";
                    for (int i = 0; i < nccl_context->num_scaleup_ranks + num_local_experts; ++ i) {
                        ss << host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[i];
                        ss << (i == nccl_context->num_scaleup_ranks - 1 ? " # ": " ");
                    }
                    return ss.str();
                };
                if (ready) {
                    if (get_env<int>("EP_BUFFER_DEBUG"))
                        printf("%s\n", get_buffer_info().c_str());
                    break;
                }

                // Timeout checks
                const auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > num_cpu_timeout_secs)
                    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
            }
        } else {
            // Non-cached mode without CPU sync, allocate with the worst case
            num_recv_tokens = num_max_tokens_per_rank * nccl_context->num_ranks;
            num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts);
            num_expanded_tokens += (expert_alignment - 1) * num_local_experts;
            num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment);
        }

        // Allocate received tensors
        // `recv_src_metadata` includes source token indices and buffer slot indices
        const auto num_allocated_tokens = do_expand ? num_expanded_tokens : num_recv_tokens;
        auto recv_x = torch::empty({num_allocated_tokens, hidden}, x.options());
        auto recv_sf = std::optional<torch::Tensor>();
        auto recv_topk_idx = std::optional<torch::Tensor>();
        auto recv_topk_weights = std::optional<torch::Tensor>();
        auto recv_src_metadata = cached_mode ?
            cached_recv_src_metadata.value() :
            torch::empty({num_recv_tokens, num_topk + 2},
                         torch::TensorOptions(torch::kCUDA).dtype(torch::kInt));

        // Optional tensors
        void* recv_sf_ptr = nullptr;
        topk_idx_t* recv_topk_idx_ptr = nullptr;
        float* recv_topk_weights_ptr = nullptr;
        int recv_sf_token_stride = 0, recv_sf_hidden_stride = 0;
        if (sf.has_value()) {
            if (not use_tma_aligned_col_major_sf) {
                recv_sf_token_stride = num_sf_packs, recv_sf_hidden_stride = 1;
            } else {
                // TMA-aligned layout for the next GEMM input
                recv_sf_token_stride = 1, recv_sf_hidden_stride = math::align(num_allocated_tokens, kNumAlignedSFPacks);
            }
            recv_sf = torch::empty_strided({num_allocated_tokens, num_sf_packs},
                                           {recv_sf_token_stride, recv_sf_hidden_stride},
                                           sf->options());
            recv_sf_ptr = recv_sf->data_ptr();
        }
        if (not do_expand) {
            recv_topk_idx = torch::empty({num_allocated_tokens, num_topk}, topk_idx.options());
            recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        }
        if (topk_weights.has_value()) {
            recv_topk_weights = do_expand ?
                torch::empty({num_allocated_tokens}, topk_weights->options()) :
                torch::empty({num_allocated_tokens, num_topk}, topk_weights->options());
            recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
        }

        // Process prefix sum, in expanding mode, it is also atomic counters
        if (not cached_mode) {
            if (do_expand) {
                // Slice the exclusive part and do atomic additions into inclusive
                psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 0, num_local_experts);
            } else {
                // Slice the inclusive part (and will not be used in the epilogue)
                psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert.slice(0, 1, num_local_experts + 1);
            }
        }
        EP_HOST_ASSERT(psum_num_recv_tokens_per_expert.size(0) == num_local_experts);

        // Launch copy kernels with full SMs
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_dispatch_copy_epilogue(buffer, workspace,
                                      psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
                                      psum_num_recv_tokens_per_expert.data_ptr<int>(),
                                      recv_x.data_ptr(), recv_sf_ptr,
                                      recv_topk_idx_ptr, recv_topk_weights_ptr,
                                      recv_src_metadata.data_ptr<int>(),
                                      channel_linked_list_ptr,
                                      num_unaligned_recv_tokens_per_expert_ptr,
                                      num_recv_tokens, num_max_tokens_per_rank,
                                      num_hidden_bytes,
                                      num_sf_packs, recv_sf_token_stride, recv_sf_hidden_stride,
                                      num_experts, num_topk, expert_alignment,
                                      nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                      nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                      jit::device_runtime->get_num_sms(),
                                      jit::device_runtime->get_num_smem_bytes(),
                                      num_channels,
                                      do_expand, cached_mode,
                                      do_zero_padding,
                                      comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, sf, topk_idx, topk_weights,
             recv_x, recv_sf, recv_topk_idx, recv_topk_weights,
             cumulative_local_expert_recv_stats,
             copied_topk_idx,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);

        return {recv_x, recv_sf,
                recv_topk_idx, recv_topk_weights,
                copied_topk_idx,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list,
                event};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
    combine(const torch::Tensor& x,
            const std::optional<torch::Tensor>& topk_weights,
            const std::optional<torch::Tensor>& bias_0,
            const std::optional<torch::Tensor>& bias_1,
            const torch::Tensor& src_metadata,
            const torch::Tensor& combined_topk_idx,
            const torch::Tensor& psum_num_recv_tokens_per_scaleup_rank,
            const std::optional<torch::Tensor>& token_metadata_at_forward,
            const std::optional<torch::Tensor>& channel_linked_list,
            const int& num_experts,
            const int& num_max_tokens_per_rank,
            const int& num_sms, const int& num_qps,
            const std::optional<EventHandle>& previous_event,
            const std::optional<EventHandle>& previous_event_before_epilogue,
            const bool& async_with_compute_stream,
            const bool& allocate_on_comm_stream,
            const bool& use_expanded_layout) const {
        // Check SM count
        EP_HOST_ASSERT(num_sms > 0);

        // Check data
        const auto [num_tokens, hidden] = get_shape<2>(x);
        EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous());
        EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);

        // Check tensors at dispatch
        const auto [num_combined_tokens, num_topk] = get_shape<2>(combined_topk_idx);
        const auto [num_scaleup_ranks] = get_shape<1>(psum_num_recv_tokens_per_scaleup_rank);
        EP_HOST_ASSERT(combined_topk_idx.is_cuda() and combined_topk_idx.is_contiguous());
        EP_HOST_ASSERT(combined_topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
        EP_HOST_ASSERT(num_scaleup_ranks == nccl_context->num_scaleup_ranks);
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.is_cuda() and psum_num_recv_tokens_per_scaleup_rank.is_contiguous());
        EP_HOST_ASSERT(psum_num_recv_tokens_per_scaleup_rank.scalar_type() == torch::kInt);
        EP_HOST_ASSERT(num_combined_tokens <= num_max_tokens_per_rank);

        // Check metadata
        // For reduction mode, `num_tokens_` means the number of unexpanded tokens
        const auto [num_reduced_tokens, num_topk_p2] = get_shape<2>(src_metadata);
        EP_HOST_ASSERT(num_reduced_tokens == (use_expanded_layout ? num_reduced_tokens : num_tokens));
        EP_HOST_ASSERT(num_topk_p2 == num_topk + 2);
        EP_HOST_ASSERT(src_metadata.is_cuda() and src_metadata.is_contiguous());
        EP_HOST_ASSERT(src_metadata.scalar_type() == torch::kInt);

        // Check optional tensors
        if (topk_weights.has_value()) {
            if (use_expanded_layout) {
                const auto [num_tokens__] = get_shape<1>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__);
            } else {
                const auto [num_tokens__, num_topk__] = get_shape<2>(topk_weights.value());
                EP_HOST_ASSERT(num_tokens == num_tokens__ and num_topk == num_topk__);
            }
            EP_HOST_ASSERT(topk_weights->is_cuda() and topk_weights->is_contiguous());
            EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat);
        }

        const auto bias_opts = std::vector({bias_0, bias_1});
        void* bias_ptrs[2] = {nullptr, nullptr};
        for (int i = 0; i < 2; ++ i) {
            if (bias_opts[i].has_value()) {
                auto bias = bias_opts[i].value();
                EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
                EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
                EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
                bias_ptrs[i] = bias.data_ptr();
            }
        }

        // Stream control
        // All new tensor allocations should happen after this
        const auto compute_stream = stream_control_prologue(previous_event, allocate_on_comm_stream, async_with_compute_stream);

        // Check buffer size
        EP_HOST_ASSERT(get_combine_buffer_size(num_max_tokens_per_rank, hidden, num_topk,
                                               nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                               nccl_context->is_scaleup_nvlink, allow_multiple_reduction) <= num_buffer_bytes);

        // Optional configs and metadata for hybrid combine
        int num_channels = 1;
        int* token_metadata_at_forward_ptr = nullptr;
        int* channel_linked_list_ptr = nullptr;
        if (nccl_context->num_scaleout_ranks > 1) {
            // The token metadata during forward
            const auto [num_channels_, d1, d2] = get_shape<3>(token_metadata_at_forward.value());
            const auto num_max_tokens_per_channel = math::ceil_div(num_max_tokens_per_rank, num_channels_);
            num_channels = num_channels_;
            token_metadata_at_forward_ptr = token_metadata_at_forward->data_ptr<int>();
            EP_HOST_ASSERT(d1 == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2 == 2 + num_topk * 2);
            EP_HOST_ASSERT(token_metadata_at_forward->is_cuda() and token_metadata_at_forward->is_contiguous());
            EP_HOST_ASSERT(token_metadata_at_forward->scalar_type() == torch::kInt);

            // Per-scaleup-peer-per-channel linked list
            const auto [num_channels__, d1_, d2_] = get_shape<3>(channel_linked_list.value());
            channel_linked_list_ptr = channel_linked_list->data_ptr<int>();
            EP_HOST_ASSERT(num_channels == num_channels__);
            EP_HOST_ASSERT(d1_ == nccl_context->num_scaleout_ranks * num_max_tokens_per_channel + 1);
            EP_HOST_ASSERT(d2_ == nccl_context->num_scaleup_ranks);
            EP_HOST_ASSERT(channel_linked_list->is_cuda() and channel_linked_list->is_contiguous());
            EP_HOST_ASSERT(channel_linked_list->scalar_type() == torch::kInt);
        }

        // Push data into remote buffers
        // NOTES: we don't use `num_hidden_bytes` due to enable later quantization possibility
        const auto reduce_buffer = launch_combine(
            x.data_ptr(),
            topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
            src_metadata.data_ptr<int>(),
            psum_num_recv_tokens_per_scaleup_rank.data_ptr<int>(),
            token_metadata_at_forward_ptr,
            channel_linked_list_ptr,
            nccl_context->dev_comm, nccl_context->window,
            buffer, workspace,
            num_reduced_tokens, num_max_tokens_per_rank,
            hidden, num_experts, num_topk,
            num_qps, num_gpu_timeout_cycles,
            nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
            nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
            nccl_context->is_scaleup_nvlink,
            num_sms, jit::device_runtime->get_num_smem_bytes(),
            num_channels,
            use_expanded_layout, allow_multiple_reduction,
            comm_stream);

        // Allocate output tensors
        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        auto combined_topk_weights = std::optional<torch::Tensor>();
        float* combined_topk_weights_ptr = nullptr;
        if (topk_weights.has_value()) {
            combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
            combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
        }

        // Combine pushed data
        stream_control_before_epilogue(previous_event_before_epilogue);
        launch_combine_reduce_epilogue(combined_x.data_ptr(),
                                       combined_topk_weights_ptr,
                                       combined_topk_idx.data_ptr<topk_idx_t>(),
                                       num_combined_tokens, num_max_tokens_per_rank,
                                       hidden,
                                       num_experts, num_topk,
                                       reduce_buffer,
                                       bias_ptrs[0], bias_ptrs[1],
                                       nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                                       nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                                       jit::device_runtime->get_num_sms(),
                                       jit::device_runtime->get_num_smem_bytes(),
                                       use_expanded_layout, allow_multiple_reduction,
                                       comm_stream);

        // Stream control
        const auto event = stream_control_epilogue(
            {x, topk_weights, bias_0, bias_1,
             src_metadata,
             combined_topk_idx,
             combined_x, combined_topk_weights,
             psum_num_recv_tokens_per_scaleup_rank,
             token_metadata_at_forward,
             channel_linked_list},
            compute_stream,
            allocate_on_comm_stream, async_with_compute_stream);
        return {combined_x, combined_topk_weights, event};
    }
};

static void register_apis(pybind11::module_& m) {
    pybind11::class_<ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<int, int, int64_t, symmetric::cpu_comm_t, int64_t, int64_t, bool, bool, bool, int, int, int, int, bool>())
        .def("destroy", &ElasticBuffer::destroy)
        .def("get_comm_stream", &ElasticBuffer::get_comm_stream)
        .def("get_physical_domain_size", &ElasticBuffer::get_physical_domain_size)
        .def("get_logical_domain_size", &ElasticBuffer::get_logical_domain_size)
        .def(
            "_rail_balance_hybrid_dispatch_prepare",
            &ElasticBuffer::rail_balance_hybrid_dispatch_prepare,
            pybind11::arg("x"),
            pybind11::arg("topk_idx"),
            pybind11::arg("topk_weights"),
            pybind11::arg("cumulative_local_expert_recv_stats"),
            pybind11::arg("num_max_tokens_per_rank"),
            pybind11::arg("num_experts"),
            pybind11::arg("num_sms"),
            pybind11::arg("num_qps"),
            pybind11::arg("proxy_capacity_per_egress"),
            pybind11::arg("arena_offset"),
            pybind11::arg("invocation_id"),
            pybind11::arg("remainder_seed") = pybind11::int_(0),
            pybind11::arg("policy") =
                static_cast<int>(rail_balance::HybridPolicy::All),
            pybind11::arg("threshold_percent") = 0)
        .def(
            "_rail_balance_hybrid_plan_prepare",
            &ElasticBuffer::rail_balance_hybrid_plan_prepare,
            pybind11::arg("topk_idx"),
            pybind11::arg("hidden"),
            pybind11::arg("num_channels"),
            pybind11::arg("num_max_tokens_per_rank"),
            pybind11::arg("num_experts"),
            pybind11::arg("num_scaleout_ranks"),
            pybind11::arg("local_scaleout_rank"),
            pybind11::arg("proxy_capacity_per_egress"),
            pybind11::arg("arena_offset"),
            pybind11::arg("invocation_id"),
            pybind11::arg("remainder_seed") = pybind11::int_(0),
            pybind11::arg("policy") =
                static_cast<int>(rail_balance::HybridPolicy::All),
            pybind11::arg("threshold_percent") = 0,
            pybind11::arg("hop_aware") = false)
        .def(
            "_rail_balance_hybrid_plan_finish",
            &ElasticBuffer::rail_balance_hybrid_plan_finish,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hop_plan_snapshot",
            &ElasticBuffer::rail_balance_hop_plan_snapshot,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_dispatch_commit",
            &ElasticBuffer::rail_balance_hybrid_dispatch_commit,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_dispatch_finish",
            &ElasticBuffer::rail_balance_hybrid_dispatch_finish,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_combine_prepare",
            &ElasticBuffer::rail_balance_hybrid_combine_prepare,
            pybind11::arg("x"),
            pybind11::arg("topk_weights"),
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_combine_abort",
            &ElasticBuffer::rail_balance_hybrid_combine_abort,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_combine_commit",
            &ElasticBuffer::rail_balance_hybrid_combine_commit,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_source_shuffle",
            &ElasticBuffer::rail_balance_hybrid_source_shuffle,
            pybind11::arg("x"),
            pybind11::arg("topk_weights"),
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_return_unshuffle_test",
            &ElasticBuffer::rail_balance_hybrid_return_unshuffle_test,
            pybind11::arg("proxy_return_bytes"),
            pybind11::arg("reduce_seed_bytes"),
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_combine_epilogue_test",
            &ElasticBuffer::rail_balance_hybrid_combine_epilogue_test,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_proxy_dispatch_snapshot",
            &ElasticBuffer::rail_balance_hybrid_proxy_dispatch_snapshot,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_plan_abort",
            &ElasticBuffer::rail_balance_hybrid_plan_abort,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_vnode_prepare",
            &ElasticBuffer::rail_balance_hybrid_vnode_prepare,
            pybind11::arg("x"),
            pybind11::arg("topk_idx"),
            pybind11::arg("topk_weights"),
            pybind11::arg("proxy_dispatch"),
            pybind11::arg("channel_count"),
            pybind11::arg("arena_offset"),
            pybind11::arg("num_max_tokens_per_rank"),
            pybind11::arg("num_experts"),
            pybind11::arg("num_destinations"),
            pybind11::arg("num_source_ranks"),
            pybind11::arg("proxy_capacity"),
            pybind11::arg("generation"),
            pybind11::arg("invocation_id"),
            pybind11::arg("remainder_seed") = pybind11::int_(0),
            pybind11::arg("policy") =
                static_cast<int>(rail_balance::HybridPolicy::All),
            pybind11::arg("threshold_percent") = 0)
        .def(
            "_rail_balance_hybrid_vnode_finish",
            &ElasticBuffer::rail_balance_hybrid_vnode_finish,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_vnode_status_snapshot",
            &ElasticBuffer::rail_balance_hybrid_vnode_status_snapshot,
            pybind11::arg("invocation_id"))
        .def(
            "_rail_balance_hybrid_vnode_abort",
            &ElasticBuffer::rail_balance_hybrid_vnode_abort,
            pybind11::arg("invocation_id"))
        .def("_rail_balance_source_shuffle", &ElasticBuffer::rail_balance_source_shuffle)
        .def("_rail_balance_source_shuffle_protocol",
             &ElasticBuffer::rail_balance_source_shuffle_protocol)
        .def("_rail_balance_vnode_roundtrip",
             &ElasticBuffer::rail_balance_vnode_roundtrip)
        .def("_rail_balance_vnode_replay",
             &ElasticBuffer::rail_balance_vnode_replay)
        .def("barrier", &ElasticBuffer::barrier)
        .def("engram_write", &ElasticBuffer::engram_write)
        .def("engram_fetch", &ElasticBuffer::engram_fetch)
        .def("pp_set_config", &ElasticBuffer::pp_set_config)
        .def("pp_send", &ElasticBuffer::pp_send)
        .def("pp_recv", &ElasticBuffer::pp_recv)
        .def("create_agrs_session", &ElasticBuffer::create_agrs_session)
        .def("destroy_agrs_session", &ElasticBuffer::destroy_agrs_session)
        .def("agrs_set_config", &ElasticBuffer::agrs_set_config)
        .def("agrs_get_inplace_tensor", &ElasticBuffer::agrs_get_inplace_tensor)
        .def("all_gather", &ElasticBuffer::all_gather)
        .def("dispatch", &ElasticBuffer::dispatch)
        .def("combine", &ElasticBuffer::combine);
    m.def("create_cpu_handle", &ElasticBuffer::create_cpu_handle);
    m.def("calculate_elastic_buffer_size", &ElasticBuffer::calculate_buffer_size);
    m.def("_calculate_rail_balance_hybrid_buffer_size",
          &ElasticBuffer::calculate_rail_balance_hybrid_buffer_size);
    m.def("_get_rail_balance_hybrid_layout",
          &ElasticBuffer::get_rail_balance_hybrid_layout);
    m.def("_get_rail_balance_source_shuffle_layout",
          &ElasticBuffer::get_rail_balance_source_shuffle_layout);
    m.def("_get_rail_balance_protocol_layout",
          &ElasticBuffer::get_rail_balance_protocol_layout);
    m.def("_get_rail_balance_vnode_layout",
          &ElasticBuffer::get_rail_balance_vnode_layout);
    m.def("_get_rail_balance_vnode_multidst_layout",
          &ElasticBuffer::get_rail_balance_vnode_multidst_layout);
    m.def("get_elastic_buffer_alignment", [=]() {
        return symmetric::kNumAlignmentBytes;
    });

    // NCCL communicator handle
    m.def("get_local_nccl_unique_id", &nccl::get_local_unique_id);
    m.def("create_nccl_comm", &nccl::create_nccl_comm);
    m.def("destroy_nccl_comm", &nccl::destroy_nccl_comm);

    // Communication domain utilities
    m.def("get_physical_domain_size", &nccl::get_physical_domain_size);
    m.def("get_logical_domain_size", &nccl::get_logical_domain_size);
}

}  // namespace deep_ep
