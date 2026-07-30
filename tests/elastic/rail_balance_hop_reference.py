"""Deterministic endpoint-aware RailBalance reference planner.

The scale-out unit is one payload per token and destination node.  ``targets``
retains every destination-local rank served by that shared payload; the usual
``(o, d, t, e)`` model is the special case ``targets == (t,)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, Sequence


Mode = Literal["off", "legacy_exact", "one_hop", "adaptive"]
PathKind = Literal["direct", "dst_forward", "src_forward", "two_hop"]


class PlanCapacityError(ValueError):
    """The complete plan exceeds a configured egress proxy capacity."""


def _require_int(name: str, value: object, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _fraction(name: str, value: object, *, maximum: int | None = None) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number") from error
    if result < 0 or (maximum is not None and result > maximum):
        upper = f" <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must satisfy 0 <= value{upper}")
    return result


@dataclass(frozen=True, order=True)
class HopFlow:
    """One deduplicated payload stream from owner ``o`` to node ``d``."""

    owner: int
    destination: int
    targets: tuple[int, ...]
    count: int

    def __post_init__(self) -> None:
        _require_int("owner", self.owner, 0)
        _require_int("destination", self.destination, 0)
        _require_int("count", self.count, 1)
        if not isinstance(self.targets, tuple) or not self.targets:
            raise ValueError("targets must be a non-empty tuple")
        if any(type(target) is not int or target < 0 for target in self.targets):
            raise ValueError("targets must contain nonnegative integers")
        if self.targets != tuple(sorted(set(self.targets))):
            raise ValueError("targets must be sorted and unique")


@dataclass(frozen=True, order=True)
class HopCopyRecord:
    target_mask: int
    destination: int


@dataclass(frozen=True)
class HopPlannerConfig:
    num_rails: int
    num_destinations: int
    mode: Mode
    chunk_size: int = 1
    two_hop_threshold: float = 0.0
    max_two_hop_ratio: float = 0.0
    hop_penalty: float = 0.0
    planner_seed: int = 0
    proxy_capacity_per_egress: int | None = None


@dataclass(frozen=True, order=True)
class HopAssignment:
    flow_index: int
    begin: int
    count: int
    owner: int
    destination: int
    targets: tuple[int, ...]
    egress: int
    path_kind: PathKind


@dataclass(frozen=True)
class HopPlan:
    config: HopPlannerConfig
    assignments: tuple[HopAssignment, ...]
    pair_load: tuple[tuple[int, ...], ...]
    source_rail_load: tuple[int, ...]
    proxy_load: tuple[int, ...]
    path_units: tuple[tuple[PathKind, int], ...]
    total_units: int
    local_forward_units: int
    extra_local_forward_units: int
    two_hop_units: int

    @property
    def two_hop_ratio(self) -> float:
        return self.two_hop_units / self.total_units if self.total_units else 0.0

    def units_for(self, path_kind: PathKind) -> int:
        return dict(self.path_units)[path_kind]


@dataclass(frozen=True)
class _Chunk:
    flow_index: int
    begin: int
    count: int
    flow: HopFlow


def materialize_hop_records(
    topk_idx: Sequence[Sequence[Sequence[int]]],
    *,
    num_experts: int,
    num_destinations: int,
    num_rails: int,
    local_destination: int,
) -> tuple[tuple[tuple[HopCopyRecord, ...], ...], ...]:
    """Build the fixed token-major endpoint table consumed by the GPU plan."""

    _require_int("num_experts", num_experts, 1)
    _require_int("num_destinations", num_destinations, 2)
    _require_int("num_rails", num_rails, 1)
    _require_int("local_destination", local_destination, 0)
    if num_rails > 32 or local_destination >= num_destinations:
        raise ValueError("record topology is outside its supported range")
    if num_experts % (num_destinations * num_rails):
        raise ValueError("num_experts must be divisible by destinations * rails")
    if isinstance(topk_idx, (str, bytes)) or not isinstance(topk_idx, Sequence):
        raise ValueError("topk_idx must be a sequence")
    owners = tuple(topk_idx)
    if len(owners) != num_rails:
        raise ValueError("topk_idx owner dimension must equal num_rails")
    num_topk = None
    experts_per_destination = num_experts // num_destinations
    experts_per_rank = experts_per_destination // num_rails
    result = []
    for tokens in owners:
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise ValueError("every owner route must be a sequence")
        owner_records = []
        for lanes in tokens:
            if isinstance(lanes, (str, bytes)) or not isinstance(lanes, Sequence):
                raise ValueError("every token route must be a sequence")
            lanes = tuple(lanes)
            if num_topk is None:
                num_topk = len(lanes)
                if not 1 <= num_topk <= 32:
                    raise ValueError("num_topk must be in [1, 32]")
            elif len(lanes) != num_topk:
                raise ValueError("all token routes must have the same num_topk")
            if any(type(expert) is not int or not 0 <= expert < num_experts
                   for expert in lanes):
                raise ValueError("expert index is outside [0, num_experts)")
            if len(set(lanes)) != len(lanes):
                raise ValueError("a token route contains duplicate expert ids")

            masks: dict[int, int] = {}
            for expert in lanes:
                destination = expert // experts_per_destination
                if destination == local_destination:
                    continue
                target = (expert % experts_per_destination) // experts_per_rank
                masks[destination] = masks.get(destination, 0) | (1 << target)
            records = [
                HopCopyRecord(mask, destination)
                for destination, mask in sorted(masks.items())
            ]
            records.extend(
                HopCopyRecord(0, -1) for _ in range(len(lanes) - len(records))
            )
            owner_records.append(tuple(records))
        result.append(tuple(owner_records))
    return tuple(result)


def _validate_config(config: HopPlannerConfig) -> tuple[Fraction, Fraction, Fraction]:
    g = _require_int("num_rails", config.num_rails, 1)
    _require_int("num_destinations", config.num_destinations, 1)
    _require_int("chunk_size", config.chunk_size, 1)
    if config.mode not in ("off", "legacy_exact", "one_hop", "adaptive"):
        raise ValueError("mode must be off, legacy_exact, one_hop, or adaptive")
    _require_int("planner_seed", config.planner_seed, 0)
    if g > 32:
        raise ValueError("num_rails must not exceed the uint32 target-mask limit")
    if config.proxy_capacity_per_egress is not None:
        _require_int("proxy_capacity_per_egress", config.proxy_capacity_per_egress, 0)
    return (
        _fraction("two_hop_threshold", config.two_hop_threshold, maximum=1),
        _fraction("max_two_hop_ratio", config.max_two_hop_ratio, maximum=1),
        _fraction("hop_penalty", config.hop_penalty),
    )


def _validate_flows(flows: Sequence[HopFlow], config: HopPlannerConfig) -> tuple[HopFlow, ...]:
    if isinstance(flows, (str, bytes)) or not isinstance(flows, Sequence):
        raise ValueError("flows must be a sequence")
    normalized = tuple(flows)
    for flow in normalized:
        if not isinstance(flow, HopFlow):
            raise ValueError("every flow must be a HopFlow")
        if flow.owner >= config.num_rails:
            raise ValueError("flow owner is outside num_rails")
        if flow.destination >= config.num_destinations:
            raise ValueError("flow destination is outside num_destinations")
        if flow.targets[-1] >= config.num_rails:
            raise ValueError("flow target is outside num_rails")
    return normalized


def endpoint_rails(flow: HopFlow) -> tuple[int, ...]:
    """Return the owner and target endpoint Rails without duplicates."""

    return tuple(sorted({flow.owner, *flow.targets}))


def classify_path(owner: int, targets: tuple[int, ...], egress: int) -> PathKind:
    if egress == owner:
        return "direct" if targets == (owner,) else "dst_forward"
    if egress in targets:
        return "src_forward"
    return "two_hop"


def local_forward_count(owner: int, targets: tuple[int, ...], egress: int) -> int:
    """Count actual cross-GPU local forwards for one shared payload."""

    return int(egress != owner) + sum(target != egress for target in targets)


def _chunks(flows: tuple[HopFlow, ...], chunk_size: int) -> list[_Chunk]:
    chunks = []
    for flow_index, flow in enumerate(flows):
        for begin in range(0, flow.count, chunk_size):
            chunks.append(_Chunk(
                flow_index=flow_index,
                begin=begin,
                count=min(chunk_size, flow.count - begin),
                flow=flow,
            ))
    return sorted(
        chunks,
        key=lambda chunk: (
            -chunk.count,
            chunk.flow.destination,
            chunk.flow.owner,
            chunk.flow.targets,
            chunk.flow_index,
            chunk.begin,
        ),
    )


def _tie_key(chunk: _Chunk, egress: int, config: HopPlannerConfig) -> tuple[int, int]:
    origin = (
        config.planner_seed
        + chunk.flow.destination
        + chunk.flow_index
        + chunk.begin // config.chunk_size
    ) % config.num_rails
    return ((egress - origin) % config.num_rails, egress)


def _initial_assignments(
    flows: tuple[HopFlow, ...], config: HopPlannerConfig
) -> list[HopAssignment]:
    pair_load = [
        [0 for _ in range(config.num_rails)]
        for _ in range(config.num_destinations)
    ]
    source_load = [0 for _ in range(config.num_rails)]
    assignments = []

    for chunk in _chunks(flows, config.chunk_size):
        flow = chunk.flow
        if config.mode == "off":
            candidates = (flow.owner,)
        elif config.mode == "legacy_exact":
            candidates = tuple(range(config.num_rails))
        else:
            candidates = endpoint_rails(flow)

        def score(egress: int) -> tuple[int, int, int, int, int]:
            pair_after = max(
                load + (chunk.count if rail == egress else 0)
                for rail, load in enumerate(pair_load[flow.destination])
            )
            source_after = max(
                load + (chunk.count if rail == egress else 0)
                for rail, load in enumerate(source_load)
            )
            return (
                pair_after,
                source_after,
                local_forward_count(flow.owner, flow.targets, egress),
                *_tie_key(chunk, egress, config),
            )

        egress = min(candidates, key=score)
        pair_load[flow.destination][egress] += chunk.count
        source_load[egress] += chunk.count
        assignments.append(HopAssignment(
            flow_index=chunk.flow_index,
            begin=chunk.begin,
            count=chunk.count,
            owner=flow.owner,
            destination=flow.destination,
            targets=flow.targets,
            egress=egress,
            path_kind=classify_path(flow.owner, flow.targets, egress),
        ))
    return assignments


def _loads(
    assignments: Sequence[HopAssignment], config: HopPlannerConfig
) -> tuple[list[list[int]], list[int], list[int]]:
    pair_load = [
        [0 for _ in range(config.num_rails)]
        for _ in range(config.num_destinations)
    ]
    source_load = [0 for _ in range(config.num_rails)]
    proxy_load = [0 for _ in range(config.num_rails)]
    for assignment in assignments:
        pair_load[assignment.destination][assignment.egress] += assignment.count
        source_load[assignment.egress] += assignment.count
        if assignment.egress != assignment.owner:
            proxy_load[assignment.egress] += assignment.count
    return pair_load, source_load, proxy_load


def _selective_two_hop(
    assignments: list[HopAssignment],
    config: HopPlannerConfig,
    threshold: Fraction,
    max_ratio: Fraction,
    hop_penalty: Fraction,
) -> list[HopAssignment]:
    total = sum(assignment.count for assignment in assignments)
    cap = total * max_ratio.numerator // max_ratio.denominator
    moved = 0

    while moved < cap:
        pair_load, source_load, _ = _loads(assignments, config)
        remaining = cap - moved
        best: tuple[tuple[Fraction, int, int, int, int, int], int, int, int] | None = None

        for index, assignment in enumerate(assignments):
            if assignment.path_kind == "two_hop":
                continue
            amount = min(assignment.count, remaining)
            if amount <= 0:
                continue
            endpoints = {assignment.owner, *assignment.targets}
            for egress in range(config.num_rails):
                if egress in endpoints:
                    continue

                before_pair = max(pair_load[assignment.destination])
                old_pair_load = pair_load[assignment.destination][assignment.egress]
                new_pair_load = pair_load[assignment.destination][egress]
                after_pair_row = list(pair_load[assignment.destination])
                after_pair_row[assignment.egress] -= amount
                after_pair_row[egress] += amount
                after_pair = max(after_pair_row)

                before_source = max(source_load)
                old_source_load = source_load[assignment.egress]
                new_source_load = source_load[egress]
                after_source_load = list(source_load)
                after_source_load[assignment.egress] -= amount
                after_source_load[egress] += amount
                after_source = max(after_source_load)

                if old_pair_load < before_pair and old_source_load < before_source:
                    continue

                added_hops = (
                    local_forward_count(assignment.owner, assignment.targets, egress)
                    - local_forward_count(
                        assignment.owner, assignment.targets, assignment.egress
                    )
                ) * amount
                # A tied peak needs more than one move before its numeric max
                # falls.  Local relief keeps that first move admissible while
                # still refusing transfers from a noncritical Rail.
                pair_relief = min(amount, max(0, old_pair_load - new_pair_load))
                source_relief = min(
                    amount, max(0, old_source_load - new_source_load)
                )
                raw_gain = pair_relief + source_relief
                net_gain = Fraction(raw_gain) - hop_penalty * added_hops
                critical_before = before_pair + before_source
                if net_gain <= 0 or net_gain <= threshold * critical_before:
                    continue

                origin = (
                    config.planner_seed
                    + assignment.destination
                    + assignment.flow_index
                    + assignment.begin // config.chunk_size
                ) % config.num_rails
                rank = (
                    -net_gain,
                    after_pair,
                    after_source,
                    added_hops,
                    (egress - origin) % config.num_rails,
                    egress,
                )
                candidate = (rank, index, egress, amount)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            break

        _rank, index, egress, amount = best
        assignment = assignments[index]
        moved_assignment = replace(
            assignment,
            begin=assignment.begin + assignment.count - amount,
            count=amount,
            egress=egress,
            path_kind="two_hop",
        )
        if amount == assignment.count:
            assignments[index] = moved_assignment
        else:
            assignments[index] = replace(
                assignment, count=assignment.count - amount
            )
            assignments.append(moved_assignment)
        moved += amount

    return assignments


def _summarize(
    assignments: list[HopAssignment],
    config: HopPlannerConfig,
    base_local_forward_units: int,
) -> HopPlan:
    assignments.sort(key=lambda item: (item.flow_index, item.begin, item.egress))
    pair_load, source_load, proxy_load = _loads(assignments, config)
    capacity = config.proxy_capacity_per_egress
    if capacity is not None and any(load > capacity for load in proxy_load):
        raise PlanCapacityError(
            f"complete plan proxy load {proxy_load} exceeds per-egress capacity {capacity}"
        )

    path_units: dict[PathKind, int] = {
        "direct": 0,
        "dst_forward": 0,
        "src_forward": 0,
        "two_hop": 0,
    }
    local_units = 0
    for assignment in assignments:
        path_units[assignment.path_kind] += assignment.count
        local_units += assignment.count * local_forward_count(
            assignment.owner, assignment.targets, assignment.egress
        )
    total = sum(path_units.values())
    return HopPlan(
        config=config,
        assignments=tuple(assignments),
        pair_load=tuple(tuple(row) for row in pair_load),
        source_rail_load=tuple(source_load),
        proxy_load=tuple(proxy_load),
        path_units=tuple(path_units.items()),
        total_units=total,
        local_forward_units=local_units,
        extra_local_forward_units=local_units - base_local_forward_units,
        two_hop_units=path_units["two_hop"],
    )


def validate_hop_plan(flows: Sequence[HopFlow], plan: HopPlan) -> None:
    """Validate conservation, unique ranges, endpoints, paths, and cap."""

    normalized = _validate_flows(flows, plan.config)
    by_flow: list[list[HopAssignment]] = [[] for _ in normalized]
    for assignment in plan.assignments:
        if not 0 <= assignment.flow_index < len(normalized):
            raise ValueError("assignment flow_index is outside the input")
        flow = normalized[assignment.flow_index]
        if (
            assignment.owner,
            assignment.destination,
            assignment.targets,
        ) != (flow.owner, flow.destination, flow.targets):
            raise ValueError("assignment changed a transport endpoint")
        if assignment.count <= 0 or assignment.begin < 0:
            raise ValueError("assignment range is invalid")
        if not 0 <= assignment.egress < plan.config.num_rails:
            raise ValueError("assignment egress is outside num_rails")
        if assignment.path_kind != classify_path(
            assignment.owner, assignment.targets, assignment.egress
        ):
            raise ValueError("assignment path_kind disagrees with its endpoints")
        if plan.config.mode == "one_hop" and assignment.egress not in endpoint_rails(flow):
            raise ValueError("one_hop selected a third Rail")
        by_flow[assignment.flow_index].append(assignment)

    for flow, assignments in zip(normalized, by_flow):
        cursor = 0
        for assignment in sorted(assignments, key=lambda item: item.begin):
            if assignment.begin != cursor:
                raise ValueError("assignment ranges overlap or contain a hole")
            cursor += assignment.count
        if cursor != flow.count:
            raise ValueError("assignment ranges do not conserve the flow")

    rebuilt = _summarize(
        list(plan.assignments),
        plan.config,
        plan.local_forward_units - plan.extra_local_forward_units,
    )
    for field in (
        "pair_load",
        "source_rail_load",
        "proxy_load",
        "path_units",
        "total_units",
        "local_forward_units",
        "two_hop_units",
    ):
        if getattr(rebuilt, field) != getattr(plan, field):
            raise ValueError(f"plan summary field {field} is inconsistent")
    if plan.config.mode == "adaptive":
        ratio = _fraction(
            "max_two_hop_ratio", plan.config.max_two_hop_ratio, maximum=1
        )
        cap = plan.total_units * ratio.numerator // ratio.denominator
        if plan.two_hop_units > cap:
            raise ValueError("adaptive plan exceeds max_two_hop_ratio")


def build_hop_plan(
    flows: Sequence[HopFlow], config: HopPlannerConfig
) -> HopPlan:
    """Build and fully validate one deterministic hop-aware plan."""

    threshold, max_ratio, hop_penalty = _validate_config(config)
    normalized = _validate_flows(flows, config)
    assignments = _initial_assignments(normalized, config)
    base_local = sum(
        assignment.count
        * local_forward_count(
            assignment.owner, assignment.targets, assignment.egress
        )
        for assignment in assignments
    )
    if config.mode == "adaptive" and max_ratio > 0:
        assignments = _selective_two_hop(
            assignments, config, threshold, max_ratio, hop_penalty
        )
    plan = _summarize(assignments, config, base_local)
    validate_hop_plan(normalized, plan)
    return plan
