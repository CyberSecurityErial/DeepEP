"""Pure-CPU C080 Hybrid virtual-node round-trip reference.

The model keeps only state required by the staged force-v1 design: compact
Hybrid schedule state, retained/moved destination copies, and moved proxy slot
``p``.  It intentionally has no descriptor, ready word, generation, ring,
queue, route sidecar, or return header.

Topology notation is ``G x D``: ``G`` rails per node and ``D`` nodes.  The
single-machine physical rank used by a vnode test is always::

    physical(node, rail) = node * G + rail

The source node is node zero.  A remote retained copy enters destination node
``d`` through the owner's rail; a moved copy enters through its planned egress
and returns to the same source egress's ``proxy_return[p]`` before unshuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from rail_balance_hybrid_reference import (
    HybridRailSchedule,
    ResolvedHybridDestinationCopy,
    build_hybrid_rail_schedule,
    enumerate_resolved_copies,
    map_topk_experts_to_destinations,
)


Scalar = Fraction
Vector = tuple[Scalar, ...]


def _exact_int(name: str, value: int, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class VnodeTopology:
    rails_per_node: int
    num_nodes: int
    source_node: int = 0

    def __post_init__(self) -> None:
        _exact_int("rails_per_node", self.rails_per_node, 1)
        _exact_int("num_nodes", self.num_nodes, 2)
        _exact_int("source_node", self.source_node, 0)
        if self.source_node >= self.num_nodes:
            raise ValueError("source_node is outside the node range")

    @property
    def world_size(self) -> int:
        return self.rails_per_node * self.num_nodes

    def physical(self, node: int, rail: int) -> int:
        _exact_int("node", node, 0)
        _exact_int("rail", rail, 0)
        if node >= self.num_nodes or rail >= self.rails_per_node:
            raise ValueError("virtual coordinate is outside the topology")
        return node * self.rails_per_node + rail

    def coordinates(self, physical_rank: int) -> tuple[int, int]:
        _exact_int("physical_rank", physical_rank, 0)
        if physical_rank >= self.world_size:
            raise ValueError("physical rank is outside the topology")
        return divmod(physical_rank, self.rails_per_node)


@dataclass(frozen=True)
class VnodeRoundTripCase:
    name: str
    topology: VnodeTopology
    topk_idx: tuple[tuple[tuple[int, ...], ...], ...]
    topk_weights: tuple[tuple[tuple[Scalar, ...], ...], ...]
    source_values: tuple[tuple[Vector, ...], ...]
    num_topk: int
    num_channels: int
    num_max_tokens_per_rank: int
    num_experts: int
    experts_per_physical_rank: int
    proxy_capacity_per_egress: int
    remainder_seed: int

    @property
    def num_tokens_per_owner(self) -> tuple[int, ...]:
        return tuple(len(tokens) for tokens in self.topk_idx)


@dataclass(frozen=True, order=True)
class VnodeCopyRoute:
    owner: int
    token: int
    destination: int
    source_channel: int
    owner_ordinal: int
    moved: bool
    egress: int
    proxy_slot: int
    remote_slot: int
    reduce_row: int
    source_physical: int
    egress_physical: int
    ingress_physical: int
    return_physical: int
    final_owner_physical: int
    expert_physicals: tuple[int, ...]
    topk_lanes: tuple[int, ...]

    @property
    def is_remote(self) -> bool:
        return self.destination != 0


@dataclass(frozen=True)
class VnodeContribution:
    route: VnodeCopyRoute
    value: Vector


@dataclass(frozen=True)
class VnodeRoundTripResult:
    case: VnodeRoundTripCase
    schedule: HybridRailSchedule
    routes: tuple[VnodeCopyRoute, ...]
    contributions: tuple[VnodeContribution, ...]
    reduce_rows: tuple[tuple[tuple[Vector, ...], ...], ...]
    combined: tuple[tuple[Vector, ...], ...]
    direct: tuple[tuple[Vector, ...], ...]


def expert_destination(expert: int, num_experts: int,
                       num_nodes: int) -> int:
    _exact_int("expert", expert, 0)
    _exact_int("num_experts", num_experts, 1)
    _exact_int("num_nodes", num_nodes, 2)
    if expert >= num_experts or num_experts % num_nodes:
        raise ValueError("expert or node partition is invalid")
    return expert // (num_experts // num_nodes)


def derive_reduce_row(
    *,
    destination: int,
    topk_idx: Sequence[int],
    num_destinations: int,
    num_experts: int,
) -> int:
    """Derive the legacy multiple-reduction row from preserved top-k ids.

    Rank layout is selected when ``D <= K`` and the row is the destination.
    Otherwise top-k layout is selected.  The legacy forwarding semantics leave
    the *highest* matching lane as the destination's row; this matters when a
    token has multiple experts on the same destination.  No match is invalid.
    """

    _exact_int("destination", destination, 0)
    _exact_int("num_destinations", num_destinations, 2)
    _exact_int("num_experts", num_experts, 1)
    if destination >= num_destinations:
        raise ValueError("destination is outside the node range")
    if not topk_idx:
        raise ValueError("topk_idx must not be empty")
    if num_experts % num_destinations:
        raise ValueError("num_experts must be divisible by num_destinations")
    for expert in topk_idx:
        expert_destination(expert, num_experts, num_destinations)

    if num_destinations <= len(topk_idx):
        return destination
    matches = [
        lane for lane, expert in enumerate(topk_idx)
        if expert_destination(expert, num_experts, num_destinations)
        == destination
    ]
    if not matches:
        raise ValueError("destination has no matching top-k lane")
    return matches[-1]


def _zero(width: int) -> Vector:
    return (Fraction(0),) * width


def _add(left: Vector, right: Vector) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector widths do not match")
    return tuple(a + b for a, b in zip(left, right))


def _weighted_expert(
    source: Vector,
    expert: int,
    lane: int,
    weight: Scalar,
) -> Vector:
    # Integer affine expert transforms and rational weights make the oracle
    # exact while ensuring expert/lane mix-ups change the final answer.
    scale = 1 + expert % 7
    return tuple(
        weight * (scale * value + expert * 3 + lane * 5 + column)
        for column, value in enumerate(source)
    )


def _validate_case(case: VnodeRoundTripCase) -> int:
    if not isinstance(case, VnodeRoundTripCase):
        raise ValueError("case must be a VnodeRoundTripCase")
    topology = case.topology
    if topology.source_node != 0:
        raise ValueError("the C080 vnode source node must be zero")
    g = topology.rails_per_node
    d = topology.num_nodes
    if len(case.topk_idx) != g or len(case.topk_weights) != g or \
            len(case.source_values) != g:
        raise ValueError("source owner count must equal rails_per_node")
    if case.num_topk < 1 or case.num_channels < 1 or \
            case.num_max_tokens_per_rank < 1:
        raise ValueError("top-k, channels, and token capacity must be positive")
    if case.num_experts != (
            topology.world_size * case.experts_per_physical_rank):
        raise ValueError("num_experts does not match topology partitioning")
    if case.proxy_capacity_per_egress < 1:
        raise ValueError("proxy capacity must be positive")

    width = -1
    for owner in range(g):
        if not (
            len(case.topk_idx[owner])
            == len(case.topk_weights[owner])
            == len(case.source_values[owner])
        ):
            raise ValueError("per-owner token arrays have different lengths")
        if len(case.topk_idx[owner]) > case.num_max_tokens_per_rank:
            raise ValueError("owner token count exceeds capacity")
        for token, experts in enumerate(case.topk_idx[owner]):
            weights = case.topk_weights[owner][token]
            source = case.source_values[owner][token]
            if len(experts) != case.num_topk or len(weights) != case.num_topk:
                raise ValueError("top-k row width is invalid")
            if len(set(experts)) != case.num_topk:
                raise ValueError("top-k experts must be distinct")
            for expert in experts:
                expert_destination(expert, case.num_experts, d)
            if width < 0:
                width = len(source)
            if not source or len(source) != width:
                raise ValueError("source hidden vectors must have one width")
            if not all(isinstance(weight, Fraction) for weight in weights):
                raise ValueError("weights must be exact Fractions")
    if width < 1:
        raise ValueError("case must contain at least one source token")
    return width


def run_vnode_roundtrip(case: VnodeRoundTripCase) -> VnodeRoundTripResult:
    """Execute dispatch, synthetic experts, combine, and unshuffle on CPU."""

    width = _validate_case(case)
    topology = case.topology
    g, d = topology.rails_per_node, topology.num_nodes
    schedule = build_hybrid_rail_schedule(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=d,
        local_scaleout_rank=topology.source_node,
        num_channels=case.num_channels,
        num_max_tokens_per_rank=case.num_max_tokens_per_rank,
        proxy_capacity_per_egress=case.proxy_capacity_per_egress,
        remainder_seed=case.remainder_seed,
    )
    if not schedule.enabled:
        raise RuntimeError(schedule.failure_reason)

    destinations = map_topk_experts_to_destinations(
        case.topk_idx,
        num_topk=case.num_topk,
        num_experts=case.num_experts,
        num_scaleout_ranks=d,
        local_scaleout_rank=topology.source_node,
    )
    resolved = enumerate_resolved_copies(destinations, schedule)
    remote_by_key: dict[
        tuple[int, int, int], ResolvedHybridDestinationCopy
    ] = {record.copy.key: record for record in resolved}
    if len(remote_by_key) != len(resolved):
        raise AssertionError("remote destination-copy identity is not unique")

    reduce_rows: list[list[list[Vector]]] = []
    combined: list[list[Vector]] = []
    direct: list[list[Vector]] = []
    routes: list[VnodeCopyRoute] = []
    contributions: list[VnodeContribution] = []

    experts_per_destination = case.num_experts // d
    num_reduce_rows = d if d <= case.num_topk else case.num_topk
    for owner, tokens in enumerate(case.topk_idx):
        owner_rows: list[list[Vector]] = []
        owner_combined: list[Vector] = []
        owner_direct: list[Vector] = []
        for token, experts in enumerate(tokens):
            source = case.source_values[owner][token]
            weights = case.topk_weights[owner][token]
            token_rows = [_zero(width) for _ in range(num_reduce_rows)]
            row_destinations: dict[int, int] = {}
            direct_value = _zero(width)
            lanes_by_destination: dict[int, list[int]] = {}
            for lane, expert in enumerate(experts):
                destination = expert // experts_per_destination
                lanes_by_destination.setdefault(destination, []).append(lane)
                direct_value = _add(
                    direct_value,
                    _weighted_expert(source, expert, lane, weights[lane]),
                )

            for destination in sorted(lanes_by_destination):
                lanes = tuple(lanes_by_destination[destination])
                row = derive_reduce_row(
                    destination=destination,
                    topk_idx=experts,
                    num_destinations=d,
                    num_experts=case.num_experts,
                )
                if row in row_destinations:
                    raise AssertionError("two destinations selected one reduce row")
                row_destinations[row] = destination

                destination_value = _zero(width)
                expert_physicals = []
                for lane in lanes:
                    expert = experts[lane]
                    expert_rank = expert // case.experts_per_physical_rank
                    expert_node, _expert_rail = topology.coordinates(expert_rank)
                    if expert_node != destination:
                        raise AssertionError("expert destination mapping changed")
                    expert_physicals.append(expert_rank)
                    destination_value = _add(
                        destination_value,
                        _weighted_expert(
                            source, expert, lane, weights[lane]),
                    )

                source_physical = topology.physical(
                    topology.source_node, owner)
                if destination == topology.source_node:
                    moved = False
                    egress = owner
                    proxy_slot = -1
                    remote_slot = -1
                    source_channel = token % case.num_channels
                    owner_ordinal = -1
                else:
                    record = remote_by_key.pop((owner, token, destination))
                    copy, resolution = record.copy, record.resolution
                    moved = resolution.moved
                    egress = resolution.egress
                    proxy_slot = resolution.proxy_slot
                    remote_slot = resolution.remote_slot
                    source_channel = copy.source_channel
                    owner_ordinal = copy.owner_ordinal
                    if moved != (egress != owner):
                        raise AssertionError("moved and egress identities disagree")
                    if moved != (proxy_slot >= 0):
                        raise AssertionError("proxy p exists on the wrong path")

                egress_physical = topology.physical(
                    topology.source_node, egress)
                ingress_physical = topology.physical(destination, egress)
                route = VnodeCopyRoute(
                    owner=owner,
                    token=token,
                    destination=destination,
                    source_channel=source_channel,
                    owner_ordinal=owner_ordinal,
                    moved=moved,
                    egress=egress,
                    proxy_slot=proxy_slot,
                    remote_slot=remote_slot,
                    reduce_row=row,
                    source_physical=source_physical,
                    egress_physical=egress_physical,
                    ingress_physical=ingress_physical,
                    return_physical=egress_physical,
                    final_owner_physical=source_physical,
                    expert_physicals=tuple(expert_physicals),
                    topk_lanes=lanes,
                )
                routes.append(route)
                contributions.append(VnodeContribution(route, destination_value))
                token_rows[row] = _add(token_rows[row], destination_value)

            combined_value = _zero(width)
            for value in token_rows:
                combined_value = _add(combined_value, value)
            if combined_value != direct_value:
                raise AssertionError("vnode round trip differs from direct expert sum")
            owner_rows.append(token_rows)
            owner_combined.append(combined_value)
            owner_direct.append(direct_value)
        reduce_rows.append(owner_rows)
        combined.append(owner_combined)
        direct.append(owner_direct)

    if remote_by_key:
        raise AssertionError("one or more scheduled copies were not consumed")
    moved_routes = [route for route in routes if route.moved]
    if len(moved_routes) != schedule.moved_copies:
        raise AssertionError("moved route total differs from the compact plan")
    for egress in range(g):
        slots = sorted(
            route.proxy_slot for route in moved_routes
            if route.egress == egress)
        if slots != list(range(schedule.proxy_required[egress])):
            raise AssertionError("proxy slots are not one dense egress prefix")

    return VnodeRoundTripResult(
        case=case,
        schedule=schedule,
        routes=tuple(routes),
        contributions=tuple(contributions),
        reduce_rows=tuple(
            tuple(tuple(rows) for rows in owner) for owner in reduce_rows),
        combined=tuple(tuple(owner) for owner in combined),
        direct=tuple(tuple(owner) for owner in direct),
    )


def _encode_patterns(
    topology: VnodeTopology,
    patterns: Sequence[Sequence[Sequence[int]]],
    *,
    experts_per_physical_rank: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if len(patterns) != topology.rails_per_node:
        raise ValueError("pattern owner count differs from topology")
    encoded = []
    rank_choices = topology.rails_per_node * experts_per_physical_rank
    for owner, tokens in enumerate(patterns):
        owner_rows = []
        for token, destinations in enumerate(tokens):
            experts = []
            for lane, destination in enumerate(destinations):
                if destination < 0 or destination >= topology.num_nodes:
                    raise ValueError("fixture destination is outside topology")
                choice = (owner * 3 + token * 5 + lane) % rank_choices
                expert_rail = choice % topology.rails_per_node
                local_expert = choice // topology.rails_per_node
                physical_rank = topology.physical(destination, expert_rail)
                experts.append(
                    physical_rank * experts_per_physical_rank + local_expert)
            if len(experts) != len(set(experts)):
                raise AssertionError("fixture generated duplicate experts")
            owner_rows.append(tuple(experts))
        encoded.append(tuple(owner_rows))
    return tuple(encoded)


def _case_from_patterns(
    *,
    name: str,
    topology: VnodeTopology,
    patterns: Sequence[Sequence[Sequence[int]]],
    num_topk: int,
    num_channels: int,
    num_max_tokens_per_rank: int,
    proxy_capacity_per_egress: int,
    remainder_seed: int,
) -> VnodeRoundTripCase:
    experts_per_physical_rank = 2
    topk_idx = _encode_patterns(
        topology, patterns,
        experts_per_physical_rank=experts_per_physical_rank)
    weights_by_k = {
        2: (Fraction(3, 4), Fraction(1, 4)),
        4: (Fraction(1, 2), Fraction(1, 4),
            Fraction(1, 8), Fraction(1, 8)),
    }
    weights = weights_by_k[num_topk]
    topk_weights = tuple(
        tuple(weights for _token in owner) for owner in topk_idx)
    source_values = tuple(
        tuple(
            tuple(Fraction(100 * owner + 10 * token + column + 1)
                  for column in range(3))
            for token in range(len(owner_tokens))
        )
        for owner, owner_tokens in enumerate(topk_idx)
    )
    return VnodeRoundTripCase(
        name=name,
        topology=topology,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        source_values=source_values,
        num_topk=num_topk,
        num_channels=num_channels,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_experts=topology.world_size * experts_per_physical_rank,
        experts_per_physical_rank=experts_per_physical_rank,
        proxy_capacity_per_egress=proxy_capacity_per_egress,
        remainder_seed=remainder_seed,
    )


def build_4x2_case() -> VnodeRoundTripCase:
    topology = VnodeTopology(rails_per_node=4, num_nodes=2)
    patterns = (
        ((1, 1, 0, 0),) * 4,
        ((1, 1, 0, 0),) * 2,
        ((0, 0, 0, 0),),
        ((0, 0, 0, 0),),
    )
    return _case_from_patterns(
        name="hybrid_vnode_4x2_d_le_k",
        topology=topology,
        patterns=patterns,
        num_topk=4,
        num_channels=2,
        num_max_tokens_per_rank=4,
        proxy_capacity_per_egress=2,
        remainder_seed=0,
    )


def build_2x4_case() -> VnodeRoundTripCase:
    topology = VnodeTopology(rails_per_node=2, num_nodes=4)
    patterns = (
        ((1, 1), (1, 2), (1, 3), (1, 0), (1, 0), (2, 0)),
        ((3, 3), (2, 3), (2, 3), (3, 0), (3, 0), (1, 0)),
    )
    return _case_from_patterns(
        name="hybrid_vnode_2x4_d_gt_k",
        topology=topology,
        patterns=patterns,
        num_topk=2,
        num_channels=2,
        num_max_tokens_per_rank=6,
        proxy_capacity_per_egress=2,
        remainder_seed=0,
    )
