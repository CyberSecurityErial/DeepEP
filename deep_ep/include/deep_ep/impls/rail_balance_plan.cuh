#pragma once

#include <cstdint>

namespace deep_ep::elastic {

// Build one exact, minimum-move plan per destination.  The host launches one
// warp per destination; lane 0 deliberately performs the small (G <= 32)
// canonical algorithm serially so its tie-breaking and segment order match the
// CPU reference exactly.
__global__ void rail_balance_plan_impl(
    const int* counts,
    int* quota,
    int* keep_count,
    int* segments,
    int* num_segments,
    int* moved_copies,
    const int num_rails,
    const int num_destinations,
    const int remainder_seed_mod
) {
    const int destination = static_cast<int>(blockIdx.x);
    if (destination >= num_destinations || threadIdx.x != 0)
        return;

    int column[32];
    int destination_quota[32];
    int surplus[32];
    int deficit[32];
    int owner_consumed[32];

    int64_t total = 0;
    for (int rail = 0; rail < num_rails; ++ rail) {
        const int64_t matrix_offset =
            static_cast<int64_t>(rail) * num_destinations + destination;
        const int value = counts[matrix_offset];
        column[rail] = value;
        total += static_cast<int64_t>(value);
    }

    const int base = static_cast<int>(total / num_rails);
    const int remainder = static_cast<int>(total % num_rails);
    const int start = (remainder_seed_mod + destination % num_rails) % num_rails;

    for (int rail = 0; rail < num_rails; ++ rail)
        destination_quota[rail] = base;

    // Raising a quota from base to base + 1 retains one additional copy iff
    // count > base.  Prefer those rails; canonical ring order breaks ties.
    int assigned = 0;
    for (int offset = 0; offset < num_rails && assigned < remainder; ++ offset) {
        const int rail = (start + offset) % num_rails;
        if (column[rail] > base) {
            ++ destination_quota[rail];
            ++ assigned;
        }
    }
    for (int offset = 0; offset < num_rails && assigned < remainder; ++ offset) {
        const int rail = (start + offset) % num_rails;
        if (column[rail] <= base) {
            ++ destination_quota[rail];
            ++ assigned;
        }
    }

    for (int rail = 0; rail < num_rails; ++ rail) {
        const int value = column[rail];
        const int target = destination_quota[rail];
        const int keep = value < target ? value : target;
        const int64_t matrix_offset =
            static_cast<int64_t>(rail) * num_destinations + destination;
        quota[matrix_offset] = target;
        keep_count[matrix_offset] = keep;
        surplus[rail] = value > target ? value - target : 0;
        deficit[rail] = target > value ? target - value : 0;
        owner_consumed[rail] = 0;
    }

    int owner = 0;
    int egress = 0;
    int segment_idx = 0;
    int destination_moved = 0;
    while (owner < num_rails && egress < num_rails) {
        while (owner < num_rails && surplus[owner] == 0)
            ++ owner;
        while (egress < num_rails && deficit[egress] == 0)
            ++ egress;
        if (owner == num_rails || egress == num_rails)
            break;

        const int count = surplus[owner] < deficit[egress] ? surplus[owner] : deficit[egress];
        const int64_t owner_matrix_offset =
            static_cast<int64_t>(owner) * num_destinations + destination;
        const int owner_begin = keep_count[owner_matrix_offset] + owner_consumed[owner];
        const int64_t segment_offset =
            (static_cast<int64_t>(destination) * (num_rails - 1) + segment_idx) * 4;
        segments[segment_offset + 0] = owner;
        segments[segment_offset + 1] = egress;
        segments[segment_offset + 2] = owner_begin;
        segments[segment_offset + 3] = count;

        surplus[owner] -= count;
        deficit[egress] -= count;
        owner_consumed[owner] += count;
        destination_moved += count;
        ++ segment_idx;
    }

    num_segments[destination] = segment_idx;
    if (destination_moved != 0)
        atomicAdd(moved_copies, destination_moved);
}

}  // namespace deep_ep::elastic
