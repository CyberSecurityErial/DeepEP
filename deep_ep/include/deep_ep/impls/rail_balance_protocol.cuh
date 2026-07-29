#pragma once

#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/rail_balance_protocol_layout.cuh>

namespace deep_ep::elastic {

namespace rail_balance::protocol_detail {

__forceinline__ __device__ void delay_cycles(const int64_t& cycles) {
    if (cycles <= 0)
        return;
    const auto start = clock64();
    while (clock64() - start < cycles)
        __nanosleep(64);
}

__forceinline__ __device__ int load_error(
    const rail_balance::OneShotProxyControl* control) {
    return ptx::ld_acquire_sys(&control->error_code);
}

__forceinline__ __device__ void set_error(
    rail_balance::OneShotProxyControl* control,
    const rail_balance::OneShotProtocolError error,
    const int& physical_slot) {
    // `control` may be an LSA peer pointer. Error arbitration therefore needs
    // system scope just like the surrounding release/acquire protocol.
    const int claimed = atomicCAS_system(&control->error_code, 0, -1);
    if (claimed == 0) {
        control->error_slot = physical_slot;
        ptx::st_release_sys(
            &control->error_code, static_cast<int>(error));
    }
}

__forceinline__ __device__ uint64_t current_ready_mask(
    const rail_balance::OneShotProtocolLayout& layout,
    const int& generation,
    const int& expected_records) {
    uint64_t mask = 0;
    for (int slot = 0; slot < expected_records; ++slot) {
        const auto key = ptx::ld_acquire_sys(
            layout.get_publish_sequence_ptr(slot));
        if (key == rail_balance::make_publish_key(generation, slot))
            mask |= uint64_t{1} << slot;
    }
    return mask;
}

__forceinline__ __device__ bool validate_record(
    const rail_balance::OneShotProtocolLayout& layout,
    const int& generation,
    const int& rank_idx,
    const int& physical_slot,
    rail_balance::OneShotProtocolError& error) {
    const auto& source_layout = layout.get_source_shuffle_layout();
    if (ptx::ld_acquire_sys(
            source_layout.get_ready_ptr(physical_slot)) != generation) {
        error = rail_balance::OneShotProtocolError::ReadyGenerationMismatch;
        return false;
    }

    const auto descriptor = source_layout.get_descriptor_ptr(physical_slot);
    if (descriptor->generation != generation or
        descriptor->physical_slot != physical_slot or
        descriptor->egress != rank_idx) {
        error = rail_balance::OneShotProtocolError::DescriptorMismatch;
        return false;
    }
    #pragma unroll
    for (int i = 0; i < rail_balance::kNumCanaryBytes /
                            static_cast<int>(sizeof(uint32_t)); ++i) {
        if (source_layout.get_head_canary_ptr(physical_slot)[i] !=
                rail_balance::kHeadCanary or
            source_layout.get_tail_canary_ptr(physical_slot)[i] !=
                rail_balance::kTailCanary) {
            error = rail_balance::OneShotProtocolError::CanaryMismatch;
            return false;
        }
    }
    return true;
}

}  // namespace rail_balance::protocol_detail

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_protocol_producer_impl(
    const ncclDevComm_t nccl_dev_comm,
    const ncclWindow_t nccl_window,
    void* x,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const int* send_manifest,
    const int64_t* fingerprints,
    const int64_t* producer_delay_cycles,
    int* producer_status,
    void* arena,
    const int physical_capacity,
    const int generation,
    const int rank_idx,
    const int num_max_tokens_per_rank,
    const int num_send_records,
    const int producer_phase,
    const int force_odd_first,
    const int drop_physical_slot,
    const int late_publish_start,
    const int late_publish_after_consumed,
    const int64_t timeout_cycles) {
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
    EP_DEVICE_ASSERT(
        producer_phase >= 0 and producer_phase <= 2);

    const bool waits_for_hole =
        force_odd_first and physical_slot % 2 == 0;
    const bool waits_for_consumption =
        late_publish_start >= 0 and
        physical_slot >= late_publish_start and
        late_publish_after_consumed > 0;
    const int required_phase = waits_for_consumption
        ? 2 : (waits_for_hole ? 1 : 0);
    if (producer_phase != required_phase)
        return;

    const auto gin = handle::NCCLGin(
        nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto peer_arena = gin.get_sym_ptr<ncclTeamTagLsa>(arena, egress);
    const auto peer_protocol_layout = rail_balance::OneShotProtocolLayout(
        kNumHiddenBytes, kNumTopk, physical_capacity, peer_arena);
    auto peer_control = peer_protocol_layout.get_control_ptr();

    int can_publish = 1;
    if (waits_for_hole and
        ptx::elect_one_sync()) {
        const auto start = clock64();
        while (ptx::ld_acquire_sys(
                   &peer_control->hole_seen_generation) != generation) {
            if (clock64() - start >= timeout_cycles) {
                can_publish = 0;
                break;
            }
            __nanosleep(64);
        }
    }
    can_publish = __shfl_sync(0xffffffff, can_publish, 0);
    if (not can_publish) {
        if (ptx::elect_one_sync())
            producer_status[record_idx] = static_cast<int>(
                rail_balance::OneShotProtocolError::ProducerTimeout);
        return;
    }

    if (late_publish_start >= 0 and
        physical_slot >= late_publish_start) {
        can_publish = 1;
        if (ptx::elect_one_sync()) {
            auto progress_clock = clock64();
            int last_consumed_count = -1;
            while (true) {
                const auto consumed_tail =
                    ptx::ld_acquire_sys(&peer_control->consumed_tail);
                const auto consumed_generation =
                    static_cast<uint32_t>(consumed_tail >> 32);
                const auto consumed_count_u =
                    static_cast<uint32_t>(consumed_tail);
                const auto expected_generation =
                    static_cast<uint32_t>(generation);
                if (consumed_generation == expected_generation) {
                    if (consumed_count_u >
                            static_cast<uint32_t>(physical_capacity) or
                        (last_consumed_count >= 0 and
                         consumed_count_u < static_cast<uint32_t>(
                             last_consumed_count))) {
                        rail_balance::protocol_detail::set_error(
                            peer_control,
                            rail_balance::OneShotProtocolError::TailRegression,
                            physical_slot);
                        can_publish = 0;
                        break;
                    }
                    const int consumed_count =
                        static_cast<int>(consumed_count_u);
                    if (consumed_count >= late_publish_after_consumed)
                        break;
                    if (consumed_count > last_consumed_count) {
                        last_consumed_count = consumed_count;
                        progress_clock = clock64();
                    }
                } else if (consumed_generation > expected_generation) {
                    rail_balance::protocol_detail::set_error(
                        peer_control,
                        rail_balance::OneShotProtocolError::TailRegression,
                        physical_slot);
                    can_publish = 0;
                    break;
                }
                if (clock64() - progress_clock >= timeout_cycles) {
                    can_publish = 0;
                    break;
                }
                __nanosleep(64);
            }
        }
        can_publish = __shfl_sync(0xffffffff, can_publish, 0);
        if (not can_publish) {
            if (ptx::elect_one_sync())
                producer_status[record_idx] = static_cast<int>(
                    rail_balance::OneShotProtocolError::ProducerTimeout);
            return;
        }
    }

    rail_balance::protocol_detail::delay_cycles(
        __ldg(producer_delay_cycles + record_idx));
    if (physical_slot == drop_physical_slot) {
        if (ptx::elect_one_sync())
            producer_status[record_idx] = -2;
        return;
    }

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

    auto staged_record_int4 =
        static_cast<int4*>(staged_layout.get_record_ptr(0));
    for (int64_t i = lane_idx;
         i < staged_layout.record_bytes / sizeof(int4); i += 32)
        staged_record_int4[i] = make_int4(0, 0, 0, 0);
    __syncwarp();

    if (ptx::elect_one_sync()) {
        ptx::tma_load_1d(
            staged_token.get_hidden_ptr(),
            math::advance_ptr(
                x, static_cast<int64_t>(token_idx) * kNumHiddenBytes),
            mbarrier_ptr,
            kNumHiddenBytes);
    }
    __syncwarp();

    if (lane_idx < rail_balance::kNumCanaryBytes /
                       static_cast<int>(sizeof(uint32_t))) {
        staged_layout.get_head_canary_ptr(0)[lane_idx] =
            rail_balance::kHeadCanary;
        staged_layout.get_tail_canary_ptr(0)[lane_idx] =
            rail_balance::kTailCanary;
    }
    if (lane_idx < kNumTopk) {
        const auto source_offset =
            static_cast<int64_t>(token_idx) * kNumTopk + lane_idx;
        staged_token.get_topk_idx_ptr()[lane_idx] =
            static_cast<int>(__ldg(topk_idx + source_offset));
        staged_token.get_topk_weights_ptr()[lane_idx] =
            __ldg(topk_weights + source_offset);
        staged_token.get_linked_list_idx_ptr()[lane_idx] = -1;
    }
    if (ptx::elect_one_sync()) {
        const int src_token_global_idx =
            rank_idx * num_max_tokens_per_rank + token_idx;
        *staged_token.get_src_token_global_idx_ptr() = src_token_global_idx;
        auto descriptor = staged_layout.get_descriptor_ptr(0);
        descriptor->fingerprint =
            static_cast<uint64_t>(__ldg(fingerprints + record_idx));
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
        for (int i = 0; i < 4; ++i)
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

    if (ptx::elect_one_sync())
        ptx::tma_store_1d(
            peer_protocol_layout.get_source_shuffle_layout().get_record_ptr(
                physical_slot),
            staged_layout.get_record_ptr(0),
            static_cast<int>(local_layout.record_bytes));
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if (ptx::elect_one_sync()) {
        ptx::st_release_sys(
            peer_protocol_layout.get_source_shuffle_layout().get_ready_ptr(
                physical_slot),
            generation);
        ptx::st_release_sys(
            peer_protocol_layout.get_publish_sequence_ptr(physical_slot),
            rail_balance::make_publish_key(generation, physical_slot));
        producer_status[record_idx] = 0;
    }
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(64, 1)
void rail_balance_protocol_consumer_impl(
    void* arena,
    void* consumed_records,
    int* ready_values,
    uint64_t* publish_sequence_values,
    int* consume_counts,
    int* ready_count_at_consume,
    int64_t* trace,
    void* control_snapshot,
    const int64_t* consumer_delay_cycles,
    const int physical_capacity,
    const int generation,
    const int rank_idx,
    const int num_recv_records,
    const int64_t timeout_cycles) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(__nv_bfloat16);
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int warp_idx = thread_idx / 32;
    const int lane_idx = ptx::get_lane_idx();
    const auto protocol_layout = rail_balance::OneShotProtocolLayout(
        kNumHiddenBytes, kNumTopk, physical_capacity, arena);
    const auto& source_layout =
        protocol_layout.get_source_shuffle_layout();
    auto control = protocol_layout.get_control_ptr();

    __shared__ int scanner_done;
    __shared__ int downstream_done;
    if (thread_idx == 0) {
        scanner_done = 0;
        downstream_done = 0;
        if (control->generation != generation or
            control->expected_records != num_recv_records) {
            rail_balance::protocol_detail::set_error(
                control,
                rail_balance::OneShotProtocolError::DescriptorMismatch,
                -1);
        }
        control->consumer_start_count = 1;
        ptx::st_release_sys(
            &control->status,
            static_cast<int>(
                rail_balance::OneShotProtocolStatus::Running));
    }
    __syncthreads();

    if (warp_idx == 0 and lane_idx == 0) {
        int tail = 0;
        int trace_count = 0;
        uint64_t last_stale_front = ~uint64_t{0};
        int last_hole_tail = -1;
        int last_hole_ready_slot = -1;
        int last_consumed_count = -1;
        auto progress_clock = clock64();
        while (tail < num_recv_records) {
            if (rail_balance::protocol_detail::load_error(control) != 0)
                break;

            const int old_tail = tail;
            while (tail < num_recv_records) {
                const auto expected_key =
                    rail_balance::make_publish_key(generation, tail);
                const auto observed_key = ptx::ld_acquire_sys(
                    protocol_layout.get_publish_sequence_ptr(tail));
                if (observed_key != expected_key)
                    break;
                rail_balance::OneShotProtocolError validation_error =
                    rail_balance::OneShotProtocolError::None;
                if (not rail_balance::protocol_detail::validate_record(
                        protocol_layout, generation, rank_idx, tail,
                        validation_error)) {
                    rail_balance::protocol_detail::set_error(
                        control, validation_error, tail);
                    break;
                }
                ++tail;
            }
            if (rail_balance::protocol_detail::load_error(control) != 0)
                break;

            if (tail > old_tail) {
                ptx::st_release_sys(
                    &control->published_tail,
                    rail_balance::pack_generation_tail(generation, tail));
                const auto consumed_packed =
                    ptx::ld_acquire_sys(&control->consumed_tail);
                int consumed = 0;
                if (static_cast<uint32_t>(consumed_packed >> 32) ==
                    static_cast<uint32_t>(generation))
                    consumed = static_cast<int>(
                        static_cast<uint32_t>(consumed_packed));
                atomicMax(&control->max_tail_gap, tail - consumed);
                if (trace_count < physical_capacity + 1) {
                    auto row = trace + static_cast<int64_t>(trace_count) * 4;
                    row[0] = 1;
                    row[1] = old_tail;
                    row[2] = tail;
                    row[3] = static_cast<int64_t>(
                        rail_balance::protocol_detail::current_ready_mask(
                            protocol_layout, generation, num_recv_records));
                    ++trace_count;
                    control->trace_count = trace_count;
                }
                progress_clock = clock64();
                continue;
            }

            const auto front_key = ptx::ld_acquire_sys(
                protocol_layout.get_publish_sequence_ptr(tail));
            if (front_key != 0 and front_key != last_stale_front) {
                const auto observed_generation =
                    static_cast<uint32_t>(front_key >> 32);
                const auto observed_slot = static_cast<uint32_t>(front_key);
                if (observed_generation >
                        static_cast<uint32_t>(generation) or
                    (observed_generation ==
                         static_cast<uint32_t>(generation) and
                     observed_slot != static_cast<uint32_t>(tail + 1))) {
                    rail_balance::protocol_detail::set_error(
                        control,
                        rail_balance::OneShotProtocolError::
                            PublishSequenceMismatch,
                        tail);
                    break;
                }
                if (observed_generation !=
                    static_cast<uint32_t>(generation))
                    ++control->stale_observations;
                last_stale_front = front_key;
            }

            int later_ready_slot = -1;
            for (int slot = tail + 1; slot < num_recv_records; ++slot) {
                if (ptx::ld_acquire_sys(
                        protocol_layout.get_publish_sequence_ptr(slot)) ==
                    rail_balance::make_publish_key(generation, slot)) {
                    later_ready_slot = slot;
                    break;
                }
            }
            if (later_ready_slot >= 0) {
                if (tail != last_hole_tail or
                    later_ready_slot != last_hole_ready_slot) {
                    // A newly observed later-ready slot is real progress and
                    // may be the event that releases gated producers. Reset
                    // once per distinct hole state, never once per poll.
                    progress_clock = clock64();
                    last_hole_tail = tail;
                    last_hole_ready_slot = later_ready_slot;
                }
                if (control->first_hole_tail < 0) {
                    control->first_hole_tail = tail;
                    control->first_hole_ready_slot = later_ready_slot;
                    if (trace_count < physical_capacity + 1) {
                        auto row =
                            trace + static_cast<int64_t>(trace_count) * 4;
                        row[0] = 0;
                        row[1] = tail;
                        row[2] = tail;
                        row[3] = later_ready_slot;
                        ++trace_count;
                        control->trace_count = trace_count;
                    }
                }
                ptx::st_release_sys(
                    &control->hole_seen_generation, generation);
            }

            // Publication can intentionally be gated on downstream progress.
            // Treat a strictly advancing current-generation consumed tail as
            // progress, but never let a repeated observation renew the lease.
            const auto consumed_packed =
                ptx::ld_acquire_sys(&control->consumed_tail);
            const auto consumed_generation =
                static_cast<uint32_t>(consumed_packed >> 32);
            const auto consumed_count_u =
                static_cast<uint32_t>(consumed_packed);
            const auto expected_generation =
                static_cast<uint32_t>(generation);
            if (consumed_generation == expected_generation) {
                if (consumed_count_u >
                        static_cast<uint32_t>(num_recv_records) or
                    (last_consumed_count >= 0 and
                     consumed_count_u < static_cast<uint32_t>(
                         last_consumed_count))) {
                    rail_balance::protocol_detail::set_error(
                        control,
                        rail_balance::OneShotProtocolError::TailRegression,
                        tail);
                    break;
                }
                const int consumed_count =
                    static_cast<int>(consumed_count_u);
                if (consumed_count > last_consumed_count) {
                    last_consumed_count = consumed_count;
                    progress_clock = clock64();
                }
            } else if (consumed_generation > expected_generation) {
                rail_balance::protocol_detail::set_error(
                    control,
                    rail_balance::OneShotProtocolError::TailRegression,
                    tail);
                break;
            }

            if (clock64() - progress_clock >= timeout_cycles) {
                rail_balance::protocol_detail::set_error(
                    control,
                    rail_balance::OneShotProtocolError::ConsumerTimeout,
                    tail);
                break;
            }
            __nanosleep(64);
        }
        scanner_done = 1;
    }

    if (warp_idx == 1) {
        constexpr int kWait = 0;
        constexpr int kBreak = 1;
        constexpr int kCopy = 2;
        int consumed = 0;
        auto progress_clock = clock64();
        while (consumed < num_recv_records) {
            int action = kWait;
            if (lane_idx == 0) {
                const int error =
                    rail_balance::protocol_detail::load_error(control);
                if (error != 0) {
                    action = kBreak;
                } else {
                    const auto packed_tail =
                        ptx::ld_acquire_sys(&control->published_tail);
                    const auto published_generation =
                        static_cast<uint32_t>(packed_tail >> 32);
                    const int published_tail =
                        static_cast<int>(
                            static_cast<uint32_t>(packed_tail));
                    if (published_generation !=
                            static_cast<uint32_t>(generation) or
                        consumed == published_tail) {
                        if (clock64() - progress_clock >=
                            timeout_cycles) {
                            rail_balance::protocol_detail::set_error(
                                control,
                                rail_balance::OneShotProtocolError::
                                    ConsumerTimeout,
                                consumed);
                            action = kBreak;
                        }
                    } else if (published_tail < consumed or
                               published_tail > num_recv_records) {
                        rail_balance::protocol_detail::set_error(
                            control,
                            rail_balance::OneShotProtocolError::
                                TailRegression,
                            consumed);
                        action = kBreak;
                    } else {
                        action = kCopy;
                    }
                }
            }
            action = __shfl_sync(0xffffffff, action, 0);
            if (action == kBreak)
                break;
            if (action == kWait) {
                __nanosleep(64);
                continue;
            }

            // lane 0's acquire-load of published_tail is the transitive
            // producer→scanner→consumer visibility edge. This warp barrier is
            // required before the other lanes read payload bytes.
            __syncwarp(0xffffffff);
            rail_balance::protocol_detail::delay_cycles(
                __ldg(consumer_delay_cycles + consumed));
            const auto src = static_cast<const int4*>(
                source_layout.get_record_ptr(consumed));
            auto dst = math::advance_ptr<int4>(
                consumed_records,
                static_cast<int64_t>(consumed) *
                    source_layout.record_bytes);
            for (int64_t i = lane_idx;
                 i < source_layout.record_bytes / sizeof(int4); i += 32)
                dst[i] = __ldg(src + i);
            __syncwarp();

            if (lane_idx == 0) {
                int ready_count = 0;
                for (int slot = 0; slot < num_recv_records; ++slot)
                    ready_count += ptx::ld_acquire_sys(
                        protocol_layout.get_publish_sequence_ptr(slot)) ==
                        rail_balance::make_publish_key(generation, slot);
                ready_count_at_consume[consumed] = ready_count;
                if (atomicAdd(consume_counts + consumed, 1) != 0)
                    rail_balance::protocol_detail::set_error(
                        control,
                        rail_balance::OneShotProtocolError::
                            DuplicateConsumption,
                        consumed);
                ptx::st_release_sys(
                    &control->consumed_tail,
                    rail_balance::pack_generation_tail(
                        generation, consumed + 1));
            }
            __syncwarp();
            ++consumed;
            if (lane_idx == 0)
                progress_clock = clock64();
        }
        if (lane_idx == 0)
            downstream_done = 1;
    }

    __syncthreads();
    if (thread_idx == 0) {
        const auto published =
            ptx::ld_acquire_sys(&control->published_tail);
        const auto consumed =
            ptx::ld_acquire_sys(&control->consumed_tail);
        int error = rail_balance::protocol_detail::load_error(control);
        const auto expected_tail =
            rail_balance::pack_generation_tail(
                generation, num_recv_records);
        if (error == 0 and
            (not scanner_done or not downstream_done or
             published != expected_tail or consumed != expected_tail)) {
            rail_balance::protocol_detail::set_error(
                control,
                rail_balance::OneShotProtocolError::TailRegression,
                num_recv_records);
            error = rail_balance::protocol_detail::load_error(control);
        }
        ptx::st_release_sys(
            &control->status,
            static_cast<int>(
                error == 0
                    ? rail_balance::OneShotProtocolStatus::Complete
                    : rail_balance::OneShotProtocolStatus::Failed));
    }
    __syncthreads();

    for (int slot = thread_idx; slot < physical_capacity; slot += 64) {
        ready_values[slot] =
            ptx::ld_acquire_sys(source_layout.get_ready_ptr(slot));
        publish_sequence_values[slot] = ptx::ld_acquire_sys(
            protocol_layout.get_publish_sequence_ptr(slot));
    }
    const auto control_bytes =
        reinterpret_cast<const uint8_t*>(control);
    auto snapshot_bytes =
        static_cast<uint8_t*>(control_snapshot);
    if (thread_idx < static_cast<int>(
            sizeof(rail_balance::OneShotProxyControl)))
        snapshot_bytes[thread_idx] = control_bytes[thread_idx];
}

}  // namespace deep_ep::elastic
