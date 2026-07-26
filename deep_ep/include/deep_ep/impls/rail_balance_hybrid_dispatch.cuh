#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/rail_balance_hybrid_layout.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/rail_balance_hybrid_plan.cuh>


namespace deep_ep::elastic {

template <bool kDoCPUSync,
          bool kReuseSlotIndices,
          int kNumSMs,
          int kNumNotifyWarps, int kNumScaleoutWarps, int kNumForwardWarps,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kNumHiddenBytes, int kNumSFPacks,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk, int kExpertAlignment,
          int kNumQPs, int64_t kNumTimeoutCycles, int kProxyCapacity,
          int kNumScaleupRanksPerLane = math::constexpr_ceil_div(kNumScaleupRanks, 32),
          int kNumChannelsPerSM = kNumScaleoutWarps,
          int kNumChannels = kNumScaleoutWarps * kNumSMs,
          int kNumMaxTokensPerChannel = math::constexpr_ceil_div(kNumMaxTokensPerRank, kNumChannels),
          int kScaleoutUpdateInterval = 6,
          int kNumSlotsPerForwardChunk = kScaleoutUpdateInterval,
          int kNumRanks = kNumScaleoutRanks * kNumScaleupRanks,
          int kNumNotifyThreads = kNumNotifyWarps * 32,
          int kNumScaleoutSendThreads = kNumScaleoutWarps * 32,
          int kNumForwardThreads = kNumForwardWarps * 32,
          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads + kNumForwardThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
rail_balance_hybrid_dispatch_impl(
    void* x, sf_pack_t* sf, topk_idx_t* topk_idx, float* topk_weights,
    topk_idx_t* copied_topk_idx,
    int* cumulative_local_expert_recv_stats,
    int* psum_num_recv_tokens_per_scaleup_rank,
    int* psum_num_recv_tokens_per_expert,
    int* num_unaligned_recv_tokens_per_expert,
    int* dst_buffer_slot_idx,
    int* token_metadata_at_forward,
    const int num_tokens,
    const int sf_token_stride, const int sf_hidden_stride,
    // TODO(NCCL): so many params, plans to optimize?
    const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
    void* buffer,
    void* workspace, void* mapped_host_workspace,
    void* rail_balance_arena,
    const int* rail_balance_retained,
    const int* rail_balance_moved,
    const int* rail_balance_group_prefix,
    // ABI-visible for the eventual force transaction/counters. Gate #2 has
    // already validated this immutable tensor. Do not add a post-Tag0 assert,
    // trap, early return, or clamp based on it: that can strand peer ranks.
    const int* rail_balance_proxy_required,
    const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerRank = kNumExperts / kNumRanks;
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    EP_STATIC_ASSERT(kNumExperts % kNumScaleupRanks == 0, "Invalid number of experts or ranks");
    EP_STATIC_ASSERT(kNumNotifyWarps % 4 == 0, "Invalid warpgroup size");
    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size");
    // C080 force-v1 is deliberately a separate, non-cached BF16
    // specialization.  Keeping these restrictions compile-time prevents a
    // partially wired cached/FP8 path from silently using the transit key.
    EP_STATIC_ASSERT(not kReuseSlotIndices,
                     "Rail-balanced Hybrid dispatch does not support cached mode");
    EP_STATIC_ASSERT(kNumSFPacks == 0,
                     "Rail-balanced Hybrid dispatch does not support FP8");
    EP_STATIC_ASSERT(kNumScaleoutRanks >= 2 and kNumScaleoutRanks <= 32,
                     "Invalid force Hybrid scale-out size");
    EP_STATIC_ASSERT(kNumScaleupRanks >= 2 and kNumScaleupRanks <= 32,
                     "Invalid force Hybrid scale-up size");
    EP_STATIC_ASSERT(kNumRanks <= layout::WorkspaceLayout::kNumMaxRanks,
                     "Force Hybrid rank geometry exceeds workspace capacity");
    EP_STATIC_ASSERT(kNumExperts <= layout::WorkspaceLayout::kNumMaxExperts,
                     "Force Hybrid expert geometry exceeds workspace capacity");
    EP_STATIC_ASSERT(
        kNumExpertsPerRank <=
            layout::WorkspaceLayout::kNumMaxExpertsPerRank,
        "Force Hybrid per-rank experts exceed workspace capacity");
    EP_STATIC_ASSERT(kNumChannels <= rail_balance::kNumHybridMaxChannels,
                     "Force Hybrid channel count exceeds the legacy ceiling");
    EP_STATIC_ASSERT(
        kNumHiddenBytes % (256 * sizeof(nv_bfloat16)) == 0 and
        kNumHiddenBytes % ptx::kNumTMAAlignBytes == 0,
        "Force Hybrid hidden shape must preserve BF16/TMA alignment");

    // Utils
    // NOTES: a warp is a channel (different channels may share QPs)
    const auto sm_idx = static_cast<int>(blockIdx.x), thread_idx = static_cast<int>(threadIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    const auto rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx;

    // Workspaces
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    const auto host_workspace_layout = layout::WorkspaceLayout(mapped_host_workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
    auto scaleup_count_mailbox = static_cast<int64_t*>(
        workspace_layout.get_scaleout_channel_gin_request_ptr(0, 0));

    // The kernel uses a fixed space of dynamic shared memory (no static shared memory)
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    constexpr int kNumSmemBytesForNotify = kNumNotifyThreads > 0 ?
        math::constexpr_align(kNumRanks + kNumExperts, kNumNotifyThreads) * sizeof(int) : 0;
    EP_STATIC_ASSERT(kNumSmemBytesForNotify % ptx::kNumTMAAlignBytes == 0, "Invalid TMA alignment");

    // Named barrier indices
    constexpr int kNotifyBarrierIndex = 1;

    // NCCL Gin handle
    // Each warp is a channel
    const auto [qp_idx, sharing_mode] = comm::get_qp_mode<kNumSMs, kNumQPs, kNumChannelsPerSM, (kNumNotifyWarps > 0)>(
        sm_idx, (warp_idx - kNumNotifyWarps) % kNumChannelsPerSM, warp_idx < kNumNotifyWarps);
    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx, sharing_mode);

    // Reset the target-local mailbox before Tag0. No peer can publish the next
    // epoch until every rank has entered and left that barrier.
    if (sm_idx == 0 and thread_idx < kNumScaleupRanks)
        ptx::st_relaxed_sys(scaleup_count_mailbox + thread_idx, int64_t(0));
    cooperative_groups::this_grid().sync();

    // Global parallel barriers for scale-out subteam and scale-up subteam
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag0, false, false, true>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);

    // Sender counters are an epoch snapshot: clear them before any forward
    // warp can increment, then leave the completed values stable until the
    // next Tag0.  Resetting at the previous epoch's tail races peer readers.
    if (not kReuseSlotIndices and sm_idx == 0 and
        thread_idx < kNumScaleupRanks)
        ptx::st_relaxed_sys(
            workspace_layout.get_scaleup_atomic_sender_counter() +
                thread_idx,
            0);
    cooperative_groups::this_grid().sync();

    // The golden layout during the whole process for both scale-out and forward warps
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, kNumSFPacks * sizeof(sf_pack_t), kNumTopk, true);
    const auto tma_buffer = layout::BufferLayout<true>(token_layout, kNumScaleoutWarps + kNumForwardWarps, 1,
            math::advance_ptr<int>(smem, kNumSmemBytesForNotify)).get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);

    // All the buffers
    auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleupRanks, kNumScaleoutRanks * kNumMaxTokensPerRank, buffer);
    auto scaleout_send_buffer = layout::BufferLayout<false>(
        token_layout, 1, kNumMaxTokensPerRank, scaleup_buffer.get_buffer_end_ptr());
    auto scaleout_recv_buffer = layout::BufferLayout<false>(
        token_layout, kNumScaleoutRanks, kNumChannels * kNumMaxTokensPerChannel, scaleout_send_buffer.get_buffer_end_ptr());

    // Init TMA for scale-out and forward warps
    ptx::arrival_phase phase = 0;
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    if (warp_idx >= kNumNotifyWarps and ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    // Different warp roles
    if (warp_idx < kNumNotifyWarps) {
        // Assign shared memory
        constexpr int kNumAlignedElems = kNumSmemBytesForNotify / sizeof(int);
        const auto rank_expert_count = math::advance_ptr<int>(smem, 0);

        // Clean initial counts
        // NOTES: if you want to change the order of different warp roles, please take care of the `thread_idx`
        int *rank_count = rank_expert_count, *expert_count = rank_expert_count + kNumRanks;
        #pragma unroll
        for (int i = 0; i < kNumAlignedElems / kNumNotifyThreads; ++ i)
            rank_expert_count[i * kNumNotifyThreads + thread_idx] = 0;
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Atomic add on shared memory
        EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes");
        const auto global_warp_idx = sm_idx * kNumNotifyWarps + warp_idx;
        for (int i = global_warp_idx; i < num_tokens; i += kNumNotifyWarps * kNumSMs) {
            // Expert choice can not be redundant
            // NOTES: no assertions here as they are expensive
            const auto dst_expert_idx = lane_idx < kNumTopk ?
                static_cast<int>(__ldg(topk_idx + i * kNumTopk + lane_idx)) : -1;
            if (dst_expert_idx >= 0)
                atomicAdd_block(expert_count + dst_expert_idx, 1);

            // Rank choice should do deduplication here
            const auto dst_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerRank : -1;
            if (ptx::deduplicate(dst_rank_idx, lane_idx) and dst_rank_idx >= 0)
                atomicAdd_block(rank_count + dst_rank_idx, 1);
        }
        ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

        // Do full-grid reduction
        #pragma unroll
        for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
            const int64_t counter = (1ll << 32ll) | rank_expert_count[i];
            ptx::red_add(workspace_layout.get_notify_reduction_workspace_ptr() + i, counter);
        }

        // Do the remaining work by SM 0
        if (sm_idx == 0) {
            // Reduce all SM's count
            // Wait all SMs' arrival
            #pragma unroll
            for (int i = thread_idx; i < kNumRanks + kNumExperts; i += kNumNotifyThreads) {
                comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
                    const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_notify_reduction_workspace_ptr() + i);
                    if ((status >> 32) == kNumSMs) {
                        // Encode and write into the send buffer
                        workspace_layout.get_scaleout_rank_expert_count_ptr<true>()[i] =
                            math::encode_decode_positive<int>(status & 0xffffffffll);

                        // Clean for the next usage
                        workspace_layout.get_notify_reduction_workspace_ptr()[i] = 0;
                        return true;
                    }

                    if (is_last_check) {
                        printf("DeepEP hybrid notify (GPU reduction) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                               "thread: %d, status: %d | %d, expected: %d\n",
                               scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                               static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumSMs);
                    }
                    return false;
                });
            }
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Issue scaleout writes to peers
            EP_STATIC_ASSERT(kReuseSlotIndices or kNumScaleoutRanks <= kNumNotifyThreads,
                             "kNumScaleoutRanks must be less than kNumNotifyThreads");
            if (thread_idx < kNumScaleoutRanks) {
                const auto dst_scaleout_rank_idx = thread_idx;
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_rank_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumScaleupRanks * sizeof(int), dst_scaleout_rank_idx,
                    ncclGinOptFlagsAggregateRequests);
                gin.put<ncclTeamTagRail>(
                    workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_rank_idx),
                    workspace_layout.get_scaleout_expert_count_ptr<true>(dst_scaleout_rank_idx),
                    kNumExpertsPerScaleout * sizeof(int), dst_scaleout_rank_idx);
            }
            __syncwarp();

            // Util functions to get metadata from scale-out peers
            // NOTES: this is correct as RDMA operations has a minimum write granularity of 1024 bytes (a whole integer write is atomic)
            const auto recv_and_reduce = [=](const auto& get_ptr_func, const bool& is_expert_reduction = false) -> int {
                int count = 0;
                #pragma unroll
                for (int j = 0; j < kNumScaleoutRanks; ++ j) {
                    const auto ptr = get_ptr_func(j);
                    int decoded;
                    comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check){
                        decoded = math::encode_decode_positive(ptx::ld_acquire_sys<int>(ptr));
                        if (math::is_decoded_positive_ready(decoded))
                            return true;

                        if (is_last_check) {
                            printf("DeepEP hybrid notify (scale-out %s reduction) timeout, "
                                   "scale-out: %d, scale-up: %d, "
                                   "thread: %d, wait scale-out: %d, decoded: %d\n",
                                   is_expert_reduction ? "expert" : "rank",
                                   scaleout_rank_idx, scaleup_rank_idx, thread_idx, j,
                                   decoded);
                        }
                        return false;
                    });

                    // Add and clean for next usages
                    count += decoded, *ptr = 0;
                }
                return count;
            };

            // Write into all scale-up peers' rank-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumScaleupRanks; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_rank_count_ptr<false>(scaleout_peer_idx, i);
                });

                // Write into the remote scale-up peer
                const int64_t counter = (static_cast<int64_t>(kNumScaleupRanks) << 32ll) | count;
                gin.put_value<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_rank_count_ptr<false>() + scaleup_rank_idx,
                    counter, i);
            }
            __syncwarp();

            // Atomic add into all scale-up peers' expert-level counters
            #pragma unroll
            for (int i = thread_idx; i < kNumExpertsPerScaleout; i += kNumNotifyThreads) {
                // Wait scale-out arrival and reduce
                const auto count = recv_and_reduce([=](const int& scaleout_peer_idx) {
                    return workspace_layout.get_scaleout_expert_count_ptr<false>(scaleout_peer_idx, i);
                }, true);

                // Write into the remote scale-up peer
                const int64_t counter = (1ll << 32ll) | count;
                const auto dst_scaleup_rank_idx = i / kNumExpertsPerRank;
                const auto expert_idx_in_dst_rank = i % kNumExpertsPerRank;
                gin.red_add_rel<ncclTeamTagLsa>(
                    workspace_layout.get_scaleup_expert_count_ptr<false>() + expert_idx_in_dst_rank,
                    counter, dst_scaleup_rank_idx);
            }
            // There are shared memory reads above, a barrier is necessary
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // NOTES: from now on, the `rank` and `expert`s size change into the local size
            expert_count = rank_expert_count + kNumScaleupRanks;

            // Wait local counters to be ready
            // NOTES: here we only care the prefix sum by scale-up peers (used for later epilogue), not all ranks
            EP_STATIC_ASSERT(kNumNotifyWarps == 0 or kNumScaleupRanks + kNumExpertsPerRank <= kNumNotifyWarps * 32,
                             "Insufficient notify threads");
            comm::timeout_while<kNumTimeoutCycles>(thread_idx < kNumScaleupRanks + kNumExpertsPerRank,
                [&](const bool& is_last_check) {
                const auto status = ptx::ld_volatile<int64_t>(workspace_layout.get_scaleup_rank_expert_count_ptr<false>() + thread_idx);
                if ((status >> 32ull) == kNumScaleupRanks) {
                    // Clean GPU workspace and write into host workspace
                    const auto count = static_cast<int>(status & 0xffffffffll);
                    const auto aligned_count = math::align<int>(
                        count, thread_idx < kNumScaleupRanks ? 1 : kExpertAlignment);

                    workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] = 0;
                    if constexpr (kDoCPUSync) {
                        host_workspace_layout.get_scaleup_rank_expert_count_ptr<false>()[thread_idx] =
                            math::encode_decode_positive(aligned_count);
                    }

                    // Update statistics counters
                    if (cumulative_local_expert_recv_stats != nullptr and thread_idx >= kNumScaleupRanks)
                        atomicAdd(cumulative_local_expert_recv_stats + (thread_idx - kNumScaleupRanks), count);

                    // Write unaligned count before aligning
                    if (num_unaligned_recv_tokens_per_expert != nullptr and thread_idx >= kNumScaleupRanks)
                        num_unaligned_recv_tokens_per_expert[thread_idx - kNumScaleupRanks] = count;

                    // Save for later prefix sum calculation
                    rank_expert_count[thread_idx] = aligned_count;
                    return true;
                }

                if (is_last_check) {
                    printf("DeepEP hybrid notify (scale-up reduction) timeout,"
                           "scale-out: %d/%d, scale-up: %d/%d, "
                           "thread: %d, status: %d | %d, expected: %d\n",
                           scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks, thread_idx,
                           static_cast<int>(status >> 32), static_cast<int>(status & 0xffffffff), kNumScaleupRanks);
                }
                return false;
            });
            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);

            // Do prefix sum by the warps of the first SM
            // NOTES: we may have fast implementation with `cub::BlockScan`, but it is too heavy to use
            const auto do_psum = [=](const int* count, int* out, const int n, const int is_exclusive) {
                int psum = 0;
                #pragma unroll
                for (int i = 0; i < math::ceil_div(n + is_exclusive, 32); ++ i) {
                    const auto idx = i * 32 + lane_idx;
                    const auto mem_idx = idx - is_exclusive;
                    const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
                    const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);

                    // Store into global memory
                    if (idx < n + is_exclusive)
                        out[idx] = sum;

                    // Update `psum` by using the last lane's value
                    psum = ptx::exchange(sum, 31);
                }
            };
            if (warp_idx == 0) {
                // Inclusive prefix sum
                do_psum(rank_count, psum_num_recv_tokens_per_scaleup_rank, kNumScaleupRanks, 0);
            } else if (warp_idx == 1) {
                // Exclusive prefix sum for later expanding
                do_psum(expert_count, psum_num_recv_tokens_per_expert, kNumExpertsPerRank, 1);
            }
        }
    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps) {
        const int scaleout_warp_idx = warp_idx - kNumNotifyWarps;
        const int channel_idx = sm_idx * kNumChannelsPerSM + scaleout_warp_idx;
        // Gate #2 has already validated every p before Tag0, so the committed
        // kernel uses the shared ABI offset without a fallible device check.
        scaleout_recv_buffer = scaleout_recv_buffer.get_rank_buffer(scaleout_rank_idx);
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);

        // The lane-owned counter is the ordinal in the original owner's
        // (channel,destination) stream.  It advances for every deduplicated
        // copy, including copies that source shuffle moved to another rail.
        // A separate queue/tail is unnecessary because the compact plan gives
        // the exact retained and moved dense prefixes.
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32,
                         "Invalid number of scale-out ranks");
        int stored_owner_tail = 0;
        const auto arena_layout = rail_balance::HybridArenaLayout(
            kNumHiddenBytes / static_cast<int>(sizeof(nv_bfloat16)),
            kNumTopk, kProxyCapacity, rail_balance_arena);

        // Preload next token
        const auto preload_next_token = [&](const int& token_idx) {
            if (token_idx >= num_tokens)
                return;

            // Issue TMA load
            const auto token_i64_idx = static_cast<int64_t>(token_idx);
            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(tma_buffer.get_hidden_ptr(), math::advance_ptr(x, token_i64_idx * kNumHiddenBytes),
                                 mbarrier_ptr, kNumHiddenBytes);
            }
            __syncwarp();

            // Issue SF `cp.async`
            if constexpr (kNumSFPacks > 0) {
                EP_STATIC_ASSERT(sizeof(sf_pack_t) % 4 == 0, "Unaligned SF element type");
                const auto gmem_src_ptr = math::advance_ptr<sf_pack_t>(sf, token_i64_idx * sf_token_stride * sizeof(sf_pack_t));
                const auto smem_dst_ptr = tma_buffer.get_sf_ptr();

                constexpr auto kNumFullIters = kNumSFPacks / 32;
                #pragma unroll
                for (int k = 0; k < kNumFullIters; ++ k) {
                    ptx::cp_async_ca(gmem_src_ptr + (k * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + k * 32 + lane_idx);
                }
                if (kNumFullIters * 32 + lane_idx < kNumSFPacks) {
                    ptx::cp_async_ca(gmem_src_ptr + (kNumFullIters * 32 + lane_idx) * sf_hidden_stride,
                                     smem_dst_ptr + kNumFullIters * 32 + lane_idx);
                }
                ptx::cp_async_mbarrier_arrive(mbarrier_ptr);
                __syncwarp();
            }
        };

        // Iterate all tokens
        preload_next_token(channel_idx);
        for (int token_idx = channel_idx; token_idx < num_tokens; token_idx += kNumChannels) {
            // Load top-k indices and weights
            EP_STATIC_ASSERT(kNumTopk <= 32, "Insufficient lanes for loading top-k indices");
            int stored_dst_scaleout_rank_idx = -1;
            if (lane_idx < kNumTopk) {
                const auto uncasted_dst_expert_idx = __ldg(topk_idx + token_idx * kNumTopk + lane_idx);
                const auto dst_expert_idx = static_cast<int>(uncasted_dst_expert_idx);
                stored_dst_scaleout_rank_idx = dst_expert_idx >= 0 ? dst_expert_idx / kNumExpertsPerScaleout : -1;
                tma_buffer.get_topk_idx_ptr()[lane_idx] = dst_expert_idx;
                if (topk_weights != nullptr)
                    tma_buffer.get_topk_weights_ptr()[lane_idx] = __ldg(topk_weights + token_idx * kNumTopk + lane_idx);
                if (copied_topk_idx != nullptr)
                    copied_topk_idx[token_idx * kNumTopk + lane_idx] = uncasted_dst_expert_idx;
            }
            __syncwarp();

            // Add source metadata (rank index and token index)
            if (ptx::elect_one_sync())
                *tma_buffer.get_src_token_global_idx_ptr() = rank_idx * kNumMaxTokensPerRank + token_idx;
            ptx::tma_store_fence();
            __syncwarp();

            // Deduplicate destinations and retain only the prefix assigned to
            // this physical rail.  Local-destination traffic is deliberately
            // outside the rail plan and preserves the legacy bypass.
            int stored_dst_slot_idx = -1;
            const auto stored_old_slot_idx = ptx::exchange(
                stored_owner_tail,
                stored_dst_scaleout_rank_idx >= 0 ?
                    stored_dst_scaleout_rank_idx : 0);
            const bool owns_destination =
                ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx) and
                stored_dst_scaleout_rank_idx >= 0;
            if (owns_destination) {
                if (stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                    stored_dst_slot_idx = stored_old_slot_idx;
                } else {
                    const auto plan_offset =
                        rail_balance::hybrid_plan_detail::gcd_offset(
                            scaleup_rank_idx, channel_idx,
                            stored_dst_scaleout_rank_idx,
                            kNumChannels, kNumScaleoutRanks);
                    const int retained_count =
                        __ldg(rail_balance_retained + plan_offset);
                    if (stored_old_slot_idx < retained_count) {
                        stored_dst_slot_idx = stored_old_slot_idx;
                    }
                }
            }

            // Keep the original owner ordinal aligned with the source-shuffle
            // resolver even when this copy is not retained here.
            const auto scaleout_rank_mask = ptx::reduce_or(stored_dst_scaleout_rank_idx >= 0 ? (1u << stored_dst_scaleout_rank_idx) : 0u);
            stored_owner_tail += (scaleout_rank_mask >> lane_idx) & 1;

            // Wait for the owner payload used by the local bypass.
            if (ptx::elect_one_sync()) {
                ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes);
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
            }
            __syncwarp();

            // Local rank can be bypassed
            if (stored_dst_slot_idx >= 0 and stored_dst_scaleout_rank_idx == scaleout_rank_idx) {
                ptx::tma_store_1d(scaleout_recv_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                                  tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
            }
            ptx::tma_store_commit();
            ptx::tma_store_wait();
            __syncwarp();

            // Preload the next token (overlapping with the IBGDA issues)
            preload_next_token(token_idx + kNumChannels);

        }

        // Consume the descriptor-free moved groups owned by this egress. NIC
        // reads use a dedicated egress-local staging region rather than the
        // peer-written arena directly; the cooperative copy also establishes
        // system visibility before posting the Rail put.
        const int invocation_key = ptx::ld_acquire_sys<int>(
            &arena_layout.get_control_ptr()->invocation_id);
        const int token_bytes = token_layout.get_num_bytes<false>();
        const uint32_t ready_epoch_base =
            static_cast<uint32_t>(invocation_key) *
            static_cast<uint32_t>(kProxyCapacity + 1);
        int stored_grouped_tail = -1;
        const auto issue_grouped_put = [&] (
                const auto& recv_token, void* send_ptr,
                const int& num_tokens_in_put,
                const auto& completion_token,
                const int& dst_scaleout_rank_idx) {
            if (lane_idx == dst_scaleout_rank_idx) {
                gin.put<ncclTeamTagRail>(
                    recv_token.get_base_ptr(), send_ptr,
                    num_tokens_in_put * token_bytes,
                    dst_scaleout_rank_idx, ncclGinOptFlagsDefault,
                    ncclGin_VASignalAdd(
                        nccl_window,
                        gin.get_sym_offset(
                            completion_token
                                .get_src_token_global_idx_ptr()),
                        static_cast<uint64_t>(ready_epoch_base) << 32));
            }
        };
        for (int dst_scaleout_rank_idx = 0;
             dst_scaleout_rank_idx < kNumScaleoutRanks;
             ++dst_scaleout_rank_idx) {
            if (dst_scaleout_rank_idx == scaleout_rank_idx)
                continue;
            const auto plan_offset =
                rail_balance::hybrid_plan_detail::gcd_offset(
                    scaleup_rank_idx, channel_idx, dst_scaleout_rank_idx,
                    kNumChannels, kNumScaleoutRanks);
            const int retained_count =
                __ldg(rail_balance_retained + plan_offset);
            const int moved_count = __ldg(rail_balance_moved + plan_offset);
            const int final_tail = retained_count + moved_count;
            if (lane_idx == dst_scaleout_rank_idx)
                stored_grouped_tail = final_tail;
            int retained_begin = 0;
            for (int previous_channel = 0;
                 previous_channel <= channel_idx; ++previous_channel) {
                const int destination_end = previous_channel < channel_idx ?
                    kNumScaleoutRanks : dst_scaleout_rank_idx;
                for (int previous_destination = 0;
                     previous_destination < destination_end;
                     ++previous_destination) {
                    retained_begin += __ldg(rail_balance_retained +
                        rail_balance::hybrid_plan_detail::gcd_offset(
                            scaleup_rank_idx, previous_channel,
                            previous_destination, kNumChannels,
                            kNumScaleoutRanks));
                }
            }
            if (retained_count > 0 and moved_count == 0) {
                const auto first_retained_token =
                    arena_layout.get_retained_rail_staging_layout(
                        retained_begin);
                issue_grouped_put(
                    scaleout_recv_buffer.get_token_buffer(0),
                    first_retained_token.get_base_ptr(), retained_count,
                    scaleout_recv_buffer
                        .get_token_buffer(retained_count - 1),
                    dst_scaleout_rank_idx);
            }
            const int proxy_begin =
                __ldg(rail_balance_group_prefix + plan_offset);
            const int proxy_end = proxy_begin + moved_count;
            #pragma unroll 1
            for (int proxy_slot = proxy_begin;
                 proxy_slot < proxy_end; ++proxy_slot) {
                const auto proxy_token =
                    arena_layout.get_proxy_dispatch_layout(proxy_slot);
                comm::timeout_while<kNumTimeoutCycles>([&](
                        const bool& is_last_check) {
                    const int ready = ptx::ld_acquire_sys<int>(
                        arena_layout.get_proxy_ready_ptr(proxy_slot));
                    if (ready == invocation_key)
                        return true;
                    if (is_last_check)
                        printf("DeepEP rail-balance proxy timeout, scale-out: %d, "
                               "scale-up: %d, channel: %d, proxy: %d, "
                               "ready: %d, invocation: %d\n",
                               scaleout_rank_idx, scaleup_rank_idx, channel_idx,
                               proxy_slot, ready, invocation_key);
                    return false;
                });
                const auto staged_token =
                    arena_layout.get_proxy_rail_staging_layout(proxy_slot);
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(
                        tma_buffer.get_base_ptr(),
                        proxy_token.get_base_ptr(), mbarrier_ptr,
                        token_bytes);
                    ptx::mbarrier_arrive_and_set_tx(
                        mbarrier_ptr, token_bytes);
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
                __syncwarp();
                if (ptx::elect_one_sync())
                    ptx::tma_store_1d(
                        staged_token.get_base_ptr(),
                        tma_buffer.get_base_ptr(), token_bytes);
                ptx::tma_store_commit();
                ptx::tma_store_wait();
                ptx::tma_store_global_visibility_fence();
                __syncwarp();
            }
            if (moved_count > 0) {
                // The retained and proxy staging arrays are individually
                // dense.  At most two bulk puts therefore cover the complete
                // destination stream.  The completion action on the final
                // put orders both payload ranges; the receiver waits on that
                // final marker before consuming any slot from this source.
                if (retained_count > 0) {
                    const auto first_retained_token =
                        arena_layout.get_retained_rail_staging_layout(
                            retained_begin);
                    if (lane_idx == dst_scaleout_rank_idx) {
                        gin.put<ncclTeamTagRail>(
                            scaleout_recv_buffer.get_token_buffer(0)
                                .get_base_ptr(),
                            first_retained_token.get_base_ptr(),
                            retained_count * token_bytes,
                            dst_scaleout_rank_idx,
                            ncclGinOptFlagsDefault);
                    }
                }
                const auto first_staged_token =
                    arena_layout.get_proxy_rail_staging_layout(proxy_begin);
                issue_grouped_put(
                    scaleout_recv_buffer
                        .get_token_buffer(retained_count),
                    first_staged_token.get_base_ptr(), moved_count,
                    scaleout_recv_buffer.get_token_buffer(final_tail - 1),
                    dst_scaleout_rank_idx);
            }
            __syncwarp();
        }

        // The grouped force path has no legacy interval update to terminate
        // the grouped queue. Complete every payload issue before publishing
        // the one dense tail below, matching Hybrid combine's final protocol.
        gin.flush<ncclCoopWarp>();
        __syncwarp();

        // Tag0 begins a fresh dispatch epoch and the legacy forwarder clears
        // every signaled tail before leaving the preceding epoch. Therefore
        // the value below is the complete packed value (not a delta from an
        // interval publication). Correctness-first force-v1 publishes exactly
        // one final dense tail per (channel,destination); it intentionally
        // gives up the legacy interval overlap until C100 can measure a safe
        // grouped alternative.
        // Remote payloads publish a per-slot epoch marker as their completion
        // action. The dense tail may arrive first; forwarders validate that
        // marker before consuming each advertised slot.
        if (lane_idx < kNumScaleoutRanks) {
            const int final_tail = lane_idx == scaleout_rank_idx ?
                stored_owner_tail : stored_grouped_tail;
            const auto signaled_tail =
                math::pack2<int, int64_t>(1, final_tail);
            const auto tail_ptr =
                workspace_layout.get_scaleout_channel_signaled_tail_ptr(
                    channel_idx, scaleout_rank_idx);
            gin.red_add_rel<ncclTeamTagRail>(
                tail_ptr, signaled_tail, lane_idx);
        }
        __syncwarp();
    } else {
        const int forward_warp_idx = warp_idx - (kNumNotifyWarps + kNumScaleoutWarps);
        const int channel_idx = sm_idx * kNumChannelsPerSM + forward_warp_idx;
        scaleout_recv_buffer = scaleout_recv_buffer.get_channel_buffer<kNumMaxTokensPerChannel>(channel_idx);
        scaleup_buffer = scaleup_buffer.get_rank_buffer(scaleup_rank_idx);

        // Shape of `token_metadata_at_forward`: `[kNumChannels, kNumScaleoutRanks * kNumMaxTokensPerChannel + 1, kNumForwardMetadataDims]`
        constexpr int kNumForwardMetadataDims =
            rail_balance::get_num_hybrid_forward_metadata_dims(kNumTopk);
        token_metadata_at_forward += channel_idx * ((kNumScaleoutRanks * kNumMaxTokensPerChannel + 1) * kNumForwardMetadataDims);

        // Shape of `dst_buffer_slot_idx`: `[kNumChannels, kNumScaleoutRanks, kNumMaxTokensPerChannel, kNumTopk]`
        dst_buffer_slot_idx += channel_idx * (kNumScaleoutRanks * kNumMaxTokensPerChannel * kNumTopk);

        // Transform linked list index
        const auto transform_linked_list_idx = [=](const int& idx) {
            constexpr int kNumTokensInLinkedList = kNumMaxTokensPerChannel * kNumScaleoutRanks + 1;
            return channel_idx * (kNumTokensInLinkedList * kNumScaleupRanks) +
                idx * kNumScaleupRanks + scaleup_rank_idx;
        };

        // Forward tokens from scale-out ranks
        EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Too many scale-out ranks");
        int num_tokens_processed = 0;
        int stored_scaleout_old_tail_idx = 0;
        int stored_scaleup_send_counters[kNumScaleupRanksPerLane] = {};
        int stored_finish_flag = lane_idx >= kNumScaleoutRanks;
        int stored_scaleout_tail_idx = 0;
        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;
        const auto forward_arena_layout = rail_balance::HybridArenaLayout(
            kNumHiddenBytes / static_cast<int>(sizeof(nv_bfloat16)),
            kNumTopk, kProxyCapacity, rail_balance_arena);
        const int forward_invocation_key = ptx::ld_acquire_sys<int>(
            &forward_arena_layout.get_control_ptr()->invocation_id);
        const uint32_t forward_ready_epoch_base =
            static_cast<uint32_t>(forward_invocation_key) *
            static_cast<uint32_t>(kProxyCapacity + 1);
        uint32_t wip_mask;
        while ((wip_mask = ptx::gather(stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag == 0))) {
            // Pick next rank in round-robin
            const auto offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks;
            const auto hi_mask = (wip_mask >> offset) << offset;
            recv_scaleout_rank_idx = hi_mask ? ptx::ffs(hi_mask) : ptx::ffs(wip_mask);

            // Wait for this rank to have data (or finish)
            comm::timeout_while<kNumTimeoutCycles>([&](const bool& is_last_check) {
                const uint32_t arrived_or_finished =
                    stored_scaleout_tail_idx > stored_scaleout_old_tail_idx or stored_finish_flag > 0;
                if (ptx::exchange(arrived_or_finished, recv_scaleout_rank_idx))
                    return true;

                // Timeout
                if (is_last_check) {
                    if (lane_idx < kNumScaleoutRanks) {
                        printf("DeepEP hybrid dispatch (forwarding) timeout, scale-out: %d, scale-up: %d, "
                               "channel: %d, lane: %d, old scale-out tail: %d, scale-out tail: (%d, %d)\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, lane_idx, stored_scaleout_old_tail_idx,
                               stored_finish_flag, stored_scaleout_tail_idx);
                    }
                    return false;
                }

                // Read new signaled tails
                if (lane_idx < kNumScaleoutRanks) {
                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(
                        workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx));
                    math::unpack2<int, int64_t>(signaled_tail, stored_finish_flag, stored_scaleout_tail_idx);
                }
                __syncwarp();
                return false;
            });

            // Process one chunk from the current rank
            const auto start_slot_idx = ptx::exchange(stored_scaleout_old_tail_idx, recv_scaleout_rank_idx);
            const auto end_slot_idx = std::min(
                ptx::exchange(stored_scaleout_tail_idx, recv_scaleout_rank_idx),
                start_slot_idx + kNumSlotsPerForwardChunk
            );
            if (lane_idx == recv_scaleout_rank_idx)
                stored_scaleout_old_tail_idx = end_slot_idx;

            const auto recv_buffer = scaleout_recv_buffer.get_rank_buffer(recv_scaleout_rank_idx);
            // Force-v1 publishes one final dense tail per remote source.  Its
            // last slot carries the completion action for the at-most-two
            // bulk puts that populate the whole stream, so gate the first
            // chunk on that marker instead of attaching a completion action
            // to every token-sized put.
            if (recv_scaleout_rank_idx != scaleout_rank_idx and
                start_slot_idx == 0) {
                const int final_slot_idx = ptx::exchange(
                    stored_scaleout_tail_idx,
                    recv_scaleout_rank_idx) - 1;
                const auto completion_token =
                    recv_buffer.get_token_buffer(final_slot_idx);
                comm::timeout_while<kNumTimeoutCycles>([&](
                        const bool& is_last_check) {
                    const uint32_t encoded_proxy =
                        ptx::ld_acquire_sys<uint32_t>(
                            reinterpret_cast<const uint32_t*>(
                                completion_token
                                    .get_linked_list_idx_ptr()));
                    const uint32_t decoded_proxy =
                        encoded_proxy - forward_ready_epoch_base;
                    const bool ready =
                        decoded_proxy == ~uint32_t(0) or
                        decoded_proxy <
                            static_cast<uint32_t>(kProxyCapacity);
                    if (ready)
                        return true;
                    if (is_last_check and ptx::elect_one_sync())
                        printf("DeepEP rail payload timeout, scale-out: %d, "
                               "scale-up: %d, channel: %d, slot: %d, "
                               "encoded: %u, epoch: %u\n",
                               scaleout_rank_idx, scaleup_rank_idx,
                               channel_idx, final_slot_idx, encoded_proxy,
                               forward_ready_epoch_base);
                    return false;
                });
            }
            for (int slot_idx = start_slot_idx; slot_idx < end_slot_idx; ++ slot_idx) {
                const auto token_buffer = recv_buffer.get_token_buffer(slot_idx);

                // Wait TMA arrival
                ptx::tma_store_wait();
                __syncwarp();

                // TMA load into shared memory
                if (ptx::elect_one_sync()) {
                    ptx::tma_load_1d(tma_buffer.get_base_ptr(), token_buffer.get_base_ptr(),
                                     mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_arrive_and_set_tx(mbarrier_ptr, token_layout.get_num_bytes<false>());
                    ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                }
                __syncwarp();

                // linked_list_idx[0] is free transit scratch until the
                // overwrite below.  A retained token enters on its original
                // owner rail; a moved token enters on a different local rail
                // and carries its compact proxy slot p in those four bytes.
                int stored_proxy_slot = -1;
                if (ptx::elect_one_sync()) {
                    const int src_token_global_idx =
                        *tma_buffer.get_src_token_global_idx_ptr();
                    const int src_rank_idx =
                        src_token_global_idx / kNumMaxTokensPerRank;
                    const int owner_scaleup_rank_idx =
                        src_rank_idx % kNumScaleupRanks;
                    if (owner_scaleup_rank_idx != scaleup_rank_idx) {
                        const uint32_t encoded_proxy = static_cast<uint32_t>(
                            tma_buffer.get_linked_list_idx_ptr()[0]);
                        stored_proxy_slot = static_cast<int>(
                            encoded_proxy - forward_ready_epoch_base);
                    }
                }

                // Read top-k indices
                EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
                int stored_dst_scaleup_rank_idx = -1;
                auto dst_expert_idx = lane_idx < kNumTopk ? tma_buffer.get_topk_idx_ptr()[lane_idx] : -1;
                dst_expert_idx -= scaleout_rank_idx * kNumExpertsPerScaleout;
                stored_dst_scaleup_rank_idx = 0 <= dst_expert_idx and dst_expert_idx < kNumExpertsPerScaleout ?
                    dst_expert_idx / kNumExpertsPerRank : -1;

                // Write the per-scaleup channel index for this token
                int linked_list_idx = -1;
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j) {
                    const auto src_lane_idx = stored_dst_scaleup_rank_idx - j * 32;
                    const bool valid = 0 <= src_lane_idx and src_lane_idx < 32;
                    const auto exchanged = ptx::exchange(
                        stored_scaleup_send_counters[j], valid ? src_lane_idx : 0);
                    linked_list_idx = valid ? exchanged : linked_list_idx;
                }
                if (not kReuseSlotIndices and lane_idx < kNumTopk) {
                    tma_buffer.get_linked_list_idx_ptr()[lane_idx] = transform_linked_list_idx(linked_list_idx);
                    ptx::tma_store_fence();
                }
                __syncwarp();

                // Deduplicate for scale-up ranks
                int stored_dst_slot_idx = -1;
                const auto dst_slot_idx_ptr = dst_buffer_slot_idx +
                    recv_scaleout_rank_idx * (kNumMaxTokensPerChannel * kNumTopk) + slot_idx * kNumTopk;
                if constexpr (kReuseSlotIndices) {
                    if (lane_idx < kNumTopk)
                        stored_dst_slot_idx = __ldg(dst_slot_idx_ptr + lane_idx);
                } else {
                    // Deduplicate for NVLink ranks
                    if (ptx::deduplicate(stored_dst_scaleup_rank_idx, lane_idx) and stored_dst_scaleup_rank_idx >= 0)
                        stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter() + stored_dst_scaleup_rank_idx, 1);
                }
                __syncwarp();

                // Issue TMAs
                if (stored_dst_slot_idx >= 0) {
                    const auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
                        scaleup_buffer.get_token_buffer(stored_dst_slot_idx).get_base_ptr(),
                        stored_dst_scaleup_rank_idx);
                    ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), tma_buffer.get_num_bytes<false>());
                    ptx::tma_store_commit();
                }
                __syncwarp();

                // Add per-scale-up counter
                EP_STATIC_ASSERT(kNumScaleupRanks <= 64, "Invalid number of scale-up peers");
                using mask_t = std::conditional_t<kNumScaleupRanks <= 32, unsigned, unsigned long long>;
                const auto scaleup_send_mask = ptx::reduce_or(
                    stored_dst_scaleup_rank_idx >= 0 ?
                    (mask_t(1) << stored_dst_scaleup_rank_idx) : mask_t(0));
                #pragma unroll
                for (int j = 0; j < kNumScaleupRanksPerLane; ++ j)
                    stored_scaleup_send_counters[j] += (scaleup_send_mask >> (j * 32 + lane_idx)) & 1;

                // Record metadata at forward
                if constexpr (not kReuseSlotIndices) {
                    EP_STATIC_ASSERT(kNumTopk <= 32, "Invalid number of selections");
                    const auto metadata_ptr = token_metadata_at_forward +
                        num_tokens_processed * kNumForwardMetadataDims;

                    // Source token index and last token index flag
                    if (ptx::elect_one_sync()) {
                        metadata_ptr[rail_balance::kHybridForwardSrcTokenDim] =
                            tma_buffer.get_src_token_global_idx_ptr()[0];
                        metadata_ptr[rail_balance::kHybridForwardLastTokenDim] =
                            slot_idx == (end_slot_idx - 1);
                        metadata_ptr[rail_balance::kHybridForwardProxySlotDim] =
                            stored_proxy_slot;
                    }

                    // Second, original top-k indices and destination slots
                    if (lane_idx < kNumTopk) {
                        metadata_ptr[
                            rail_balance::kHybridForwardRouteBaseDim + lane_idx] =
                            stored_dst_scaleup_rank_idx;
                        metadata_ptr[
                            rail_balance::kHybridForwardRouteBaseDim +
                                kNumTopk + lane_idx] = stored_dst_slot_idx;
                        dst_slot_idx_ptr[lane_idx] = stored_dst_slot_idx;
                    }
                }
                num_tokens_processed += 1;
                __syncwarp();
            }
        }

        // Assign the source token index part of the metadata into `-1` as an ending mark
        if (not kReuseSlotIndices and ptx::elect_one_sync())
            token_metadata_at_forward[
                num_tokens_processed * kNumForwardMetadataDims +
                    rail_balance::kHybridForwardSrcTokenDim] = -1;
        __syncwarp();

        // Update linked list's ending position
        if constexpr (not kReuseSlotIndices) {
            const auto tail_ptr = workspace_layout.get_channel_scaleup_tail_ptr(channel_idx, scaleup_rank_idx);
            #pragma unroll
            for (int i = 0; i < kNumScaleupRanksPerLane; ++ i) {
                if (const auto j = i * 32 + lane_idx; i < (kNumScaleupRanksPerLane - 1) or j < kNumScaleupRanks) {
                    ptx::red_add_rel_sys(
                        gin.get_sym_ptr<ncclTeamTagLsa>(tail_ptr, j),
                        transform_linked_list_idx(stored_scaleup_send_counters[i]));
                    auto peer_mailbox = gin.get_sym_ptr<ncclTeamTagLsa>(
                        scaleup_count_mailbox + scaleup_rank_idx, j);
                    // The release reduction is ordered after this channel's
                    // tail store. The target waits for all channel credits,
                    // so Tag1 cannot expose an incomplete combine list.
                    ptx::red_add_rel_sys(
                        peer_mailbox,
                        math::pack2<int, int64_t>(
                            stored_scaleup_send_counters[i], 1));
                }
            }
            // The NVLink barrier signal is issued by SM 0 after the grid
            // rendezvous, not by the lanes that publish these peer tails.
            // Complete each lane's store before handing off to that signaler.
            ptx::fence_acq_rel_sys();
        }
        __syncwarp();

        // Clean tails for next usages
        if (lane_idx < kNumScaleoutRanks)
            *workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx) = 0;
        __syncwarp();
    }

    // Wait until every warp has issued its channel completion credit.
    cooperative_groups::this_grid().sync();

    // Scale-up barrier to ensure data arrival
    // As scale-out tokens have already been consumed by forwarders, no need to do scale-out barrier again
    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,
                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, comm::kHybridDispatchTag1, true, true, false>(
        gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx, /* do not scale-out */ false, true);

    // Notify counted tokens by their original owner rail, but source shuffle
    // changes the rank-buffer owner for moved copies. Rebuild the rank prefix
    // from sender-owned dense counts reduced into this target before Tag1.
    if (sm_idx == 0 and warp_idx == 0) {
        int actual_count = 0;
        int64_t published_count = 0;
        comm::timeout_while<kNumTimeoutCycles>(
            lane_idx < kNumScaleupRanks,
            [&](const bool& is_last_check) {
                published_count = ptx::ld_acquire_sys<int64_t>(
                    scaleup_count_mailbox + lane_idx);
                if ((static_cast<uint64_t>(published_count) >> 32ull) ==
                    static_cast<uint64_t>(kNumChannels))
                    return true;
                if (is_last_check)
                    printf("DeepEP rail count timeout, scale-out: %d, "
                           "scale-up: %d, source: %d, status: %lld\n",
                           scaleout_rank_idx, scaleup_rank_idx, lane_idx,
                           static_cast<long long>(published_count));
                return false;
            });
        if (lane_idx < kNumScaleupRanks)
            actual_count = static_cast<int>(
                static_cast<uint32_t>(published_count));
        const int actual_prefix =
            ptx::warp_inclusive_sum(actual_count, lane_idx);
        if (lane_idx < kNumScaleupRanks)
            ptx::st_release_sys(
                psum_num_recv_tokens_per_scaleup_rank + lane_idx,
                actual_prefix);
    }

    // Order the local prefix write before any SM triggers the epilogue.
    cooperative_groups::this_grid().sync();
    // Trigger the copy epilogue kernel
    cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace deep_ep::elastic
