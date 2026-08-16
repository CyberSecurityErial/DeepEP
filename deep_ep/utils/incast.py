"""Control-plane planning for EchoP cross-node incast avoidance."""

from __future__ import annotations

import torch


def _as_signed_int32(value: int) -> int:
    """Encode a 32-bit Rail bitset in a signed torch.int32 element."""
    value &= 0xFFFFFFFF
    return value if value < (1 << 31) else value - (1 << 32)


def _rotated_order(length: int, offset: int) -> list[int]:
    if length == 0:
        return []
    offset %= length
    return [(offset + index) % length for index in range(length)]


def _allocate_exclusive_counts(
        active_sources: list[int], demand: list[list[float]], destination: int,
        num_rails: int, source_order: list[int]) -> dict[int, int]:
    """Allocate every destination Rail once while minimizing peak flow/Rail."""
    counts = {source: 1 for source in active_sources}
    priority = {source: index for index, source in enumerate(source_order)}
    for _ in range(num_rails - len(active_sources)):
        # Splitting the currently hottest per-Rail flow is the greedy optimum for
        # the next Rail. Stable rotated ties avoid pinning the same node forever.
        source = max(
            active_sources,
            key=lambda item: (
                demand[item][destination] / counts[item],
                demand[item][destination],
                -priority[item],
            ),
        )
        counts[source] += 1
    return counts


def plan_incast_rail_masks(
        demand: torch.Tensor, num_rails: int, *, epoch: int = 0) -> torch.Tensor:
    """Plan per-node-pair Rail subsets from a predicted traffic matrix.

    Args:
        demand: Square ``[num_nodes, num_nodes]`` tensor. ``demand[s, d]`` is
            the predicted cross-node byte or token volume from source node
            ``s`` to destination node ``d`` in the next control window.
        num_rails: Number of corresponding egress/ingress Rails per node.
        epoch: Optional deterministic rotation. Advancing it changes equally
            good Rail identities without changing the volume allocation.

    Returns:
        A CPU int32 tensor of shape ``[num_nodes, num_nodes]``. Bit ``r`` in
        element ``[s, d]`` allows flow ``s -> d`` to use Rail ``r``. Diagonal
        and zero-demand entries are zero.

    When no more sources target a destination than there are Rails, the flows
    receive disjoint subsets and collectively use every Rail. Subset sizes are
    chosen greedily to minimize the largest predicted flow volume per Rail.
    When sources outnumber Rails, every flow still receives one Rail and a
    largest-first list scheduler minimizes unavoidable destination contention.
    """
    if not isinstance(demand, torch.Tensor):
        raise TypeError('demand must be a torch.Tensor')
    if demand.ndim != 2 or demand.shape[0] != demand.shape[1]:
        raise ValueError('demand must have square shape [num_nodes, num_nodes]')
    if not isinstance(num_rails, int) or isinstance(num_rails, bool):
        raise TypeError('num_rails must be an integer')
    if not 1 <= num_rails <= 32:
        raise ValueError('num_rails must be in [1, 32]')
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        raise TypeError('epoch must be an integer')

    demand_cpu = demand.detach().to(device='cpu', dtype=torch.float64)
    if not bool(torch.isfinite(demand_cpu).all().item()):
        raise ValueError('demand must contain only finite values')
    if bool((demand_cpu < 0).any().item()):
        raise ValueError('demand must be non-negative')

    num_nodes = demand_cpu.shape[0]
    demand_values = demand_cpu.tolist()
    masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Predicted source-NIC load is carried across destinations so masks also
    # avoid creating a new egress Rail hotspot while fixing destination incast.
    source_rail_load = [
        [0.0 for _ in range(num_rails)] for _ in range(num_nodes)
    ]

    destination_order = _rotated_order(num_nodes, epoch)
    for destination in destination_order:
        active_sources = [
            source for source in range(num_nodes)
            if source != destination and demand_values[source][destination] > 0
        ]
        if not active_sources:
            continue

        source_rotation = (destination + epoch) % num_nodes
        rotated_sources = _rotated_order(num_nodes, source_rotation)
        source_priority = {source: index for index, source in enumerate(rotated_sources)}
        source_order = sorted(
            active_sources,
            key=lambda source: (
                -demand_values[source][destination],
                source_priority[source],
            ),
        )
        rail_rotation = (destination * 3 + epoch) % num_rails
        rail_priority_order = _rotated_order(num_rails, rail_rotation)
        rail_priority = {rail: index for index, rail in enumerate(rail_priority_order)}

        if len(active_sources) <= num_rails:
            counts = _allocate_exclusive_counts(
                active_sources, demand_values, destination, num_rails,
                rotated_sources)
            available_rails = set(range(num_rails))

            # Place the largest per-Rail flows first. For each source, select
            # its least-loaded egress Rails from those still free at this sink.
            placement_order = sorted(
                active_sources,
                key=lambda source: (
                    -(demand_values[source][destination] / counts[source]),
                    source_priority[source],
                ),
            )
            for source in placement_order:
                count = counts[source]
                selected = sorted(
                    available_rails,
                    key=lambda rail: (
                        source_rail_load[source][rail],
                        rail_priority[rail],
                    ),
                )[:count]
                per_rail = demand_values[source][destination] / count
                mask = 0
                for rail in selected:
                    mask |= 1 << rail
                    source_rail_load[source][rail] += per_rail
                    available_rails.remove(rail)
                masks[source][destination] = mask
            assert not available_rails
        else:
            # Disjointness is impossible. LPT placement minimizes the
            # destination's maximum Rail load; source load breaks equal ties.
            destination_rail_load = [0.0 for _ in range(num_rails)]
            for source in source_order:
                volume = demand_values[source][destination]
                rail = min(
                    range(num_rails),
                    key=lambda item: (
                        destination_rail_load[item],
                        source_rail_load[source][item],
                        rail_priority[item],
                    ),
                )
                masks[source][destination] = 1 << rail
                destination_rail_load[rail] += volume
                source_rail_load[source][rail] += volume

    encoded = [
        [_as_signed_int32(mask) for mask in row]
        for row in masks
    ]
    return torch.tensor(encoded, dtype=torch.int32, device='cpu')
