#pragma once

#include <nccl_device.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/rail_balance_layout.cuh>

namespace deep_ep::elastic {

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_source_shuffle_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const int* send_manifest,
    const int64_t* fingerprints,
    void* arena,
    const int physical_capacity,
    const int generation,
    const int rank_idx,
    const int num_max_tokens_per_rank,
    const int num_send_records) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    const int record_idx = static_cast<int>(blockIdx.x);
    if (record_idx >= num_send_records)
        return;

    const int lane_idx = ptx::get_lane_idx();
    const int* manifest = send_manifest +
        static_cast<int64_t>(record_idx) * rail_balance::kNumManifestFields;
    const int token_idx = __ldg(manifest + 0);
    const int destination = __ldg(manifest + 1);
    const int owner_ordinal = __ldg(manifest + 2);
    const int egress = __ldg(manifest + 3);
    const int channel = __ldg(manifest + 4);
    const int logical_slot = __ldg(manifest + 5);
    const int physical_slot = __ldg(manifest + 6);

    EP_DEVICE_ASSERT(token_idx >= 0);
    EP_DEVICE_ASSERT(physical_slot >= 0 and physical_slot < physical_capacity);

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto local_layout = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1);

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    const auto staged_layout = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, 1, smem);
    const auto staged_token = staged_layout.get_token_layout(0);
    const auto mbarrier_ptr = reinterpret_cast<ptx::mbarrier*>(
        smem + staged_layout.record_bytes);
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // The whole record is transferred, including TokenLayout padding.  Clear
    // it deterministically before filling fields so no stale shared bytes are
    // published into the symmetric arena.
    auto staged_record_int4 = static_cast<int4*>(staged_layout.get_record_ptr(0));
    for (int64_t i = lane_idx; i < staged_layout.record_bytes / sizeof(int4); i += 32)
        staged_record_int4[i] = make_int4(0, 0, 0, 0);
    __syncwarp();

    if (ptx::elect_one_sync()) {
        ptx::tma_load_1d(
            staged_token.get_hidden_ptr(),
            math::advance_ptr(x, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
            mbarrier_ptr,
            kNumHiddenBytes);
    }
    __syncwarp();

    if (lane_idx < rail_balance::kNumCanaryBytes / sizeof(uint32_t)) {
        staged_layout.get_head_canary_ptr(0)[lane_idx] = rail_balance::kHeadCanary;
        staged_layout.get_tail_canary_ptr(0)[lane_idx] = rail_balance::kTailCanary;
    }
    if (lane_idx < kNumTopk) {
        const auto source_offset = static_cast<int64_t>(token_idx) * kNumTopk + lane_idx;
        staged_token.get_topk_idx_ptr()[lane_idx] = static_cast<int>(__ldg(topk_idx + source_offset));
        staged_token.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + source_offset);
        staged_token.get_linked_list_idx_ptr()[lane_idx] = -1;
    }
    if (ptx::elect_one_sync()) {
        const int src_token_global_idx = rank_idx * num_max_tokens_per_rank + token_idx;
        *staged_token.get_src_token_global_idx_ptr() = src_token_global_idx;
        auto descriptor = staged_layout.get_descriptor_ptr(0);
        descriptor->fingerprint = static_cast<uint64_t>(__ldg(fingerprints + record_idx));
        descriptor->owner_rank = rank_idx;
        descriptor->owner_token = token_idx;
        descriptor->destination = destination;
        descriptor->owner_ordinal = owner_ordinal;
        descriptor->egress = egress;
        descriptor->channel = channel;
        descriptor->logical_slot = logical_slot;
        descriptor->physical_slot = physical_slot;
        descriptor->generation = generation;
        descriptor->src_token_global_idx = src_token_global_idx;
        #pragma unroll
        for (int i = 0; i < 4; ++ i)
            descriptor->reserved[i] = 0;
    }
    __syncwarp();

    if (ptx::elect_one_sync()) {
        ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
    }
    __syncwarp();
    ptx::tma_store_fence();
    __syncwarp();

    // The C040 moved-only API never emits self-egress records. C060 packs the
    // complete retained+moved manifest, so retained records deliberately use
    // the local symmetric arena instead of asking NCCL for a peer pointer to
    // ourselves.
    const auto peer_arena = egress == rank_idx
        ? arena
        : gin.get_sym_ptr<ncclTeamTagLsa>(arena, egress);
    const auto peer_layout = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, physical_capacity, peer_arena);
    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_layout.get_record_ptr(physical_slot),
            staged_layout.get_record_ptr(0),
            static_cast<int>(local_layout.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync())
        ptx::st_release_sys(peer_layout.get_ready_ptr(physical_slot), generation);
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_source_shuffle_readback_impl(
    const void* arena,
    void* records,
    int* ready_values,
    const int physical_capacity,
    const int num_recv_records) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    const int physical_slot = static_cast<int>(blockIdx.x);
    const int lane_idx = ptx::get_lane_idx();
    const auto arena_layout = rail_balance::SourceShuffleLayout(
        kNumHiddenBytes, kNumTopk, physical_capacity, const_cast<void*>(arena));

    if (ptx::elect_one_sync())
        ready_values[physical_slot] =
            ptx::ld_acquire_sys(arena_layout.get_ready_ptr(physical_slot));
    __syncwarp();

    if (physical_slot < num_recv_records) {
        const auto src = static_cast<const int4*>(arena_layout.get_record_ptr(physical_slot));
        auto dst = math::advance_ptr<int4>(
            records, static_cast<int64_t>(physical_slot) * arena_layout.record_bytes);
        for (int64_t i = lane_idx; i < arena_layout.record_bytes / sizeof(int4); i += 32)
            dst[i] = __ldg(src + i);
    }
}

}  // namespace deep_ep::elastic
