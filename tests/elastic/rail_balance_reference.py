"""Deterministic CPU reference for source-side rail balancing.

This module intentionally has no dependency on ``deep_ep`` or CUDA.  It defines
the correctness contract that the standalone GPU planner must match before the
data path is integrated into the Hybrid kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Matrix = Tuple[Tuple[int, ...], ...]


def _require_int(name: str, value: int, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, order=True)
class DestinationCopy:
    """One hidden-payload copy after per-token destination deduplication."""

    owner: int
    token: int
    destination: int
    source_channel: int
    owner_ordinal: int

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.owner, self.token, self.destination


@dataclass(frozen=True, order=True)
class TransferSegment:
    """A stable suffix range moved from one owner rail to one egress rail."""

    destination: int
    owner: int
    egress: int
    owner_begin: int
    count: int


@dataclass(frozen=True)
class CountPlan:
    counts: Matrix
    quota: Matrix
    keep_count: Matrix
    segments: Tuple[TransferSegment, ...]
    moved_copies: int
    remainder_seed: int

    @property
    def num_rails(self) -> int:
        return len(self.counts)

    @property
    def num_destinations(self) -> int:
        return len(self.counts[0]) if self.counts else 0


@dataclass(frozen=True, order=True)
class SlotAssignment:
    """Committed queue position in the unified retained/proxy namespace."""

    egress: int
    destination: int
    channel: int
    slot: int
    generation: int
    copy: DestinationCopy
    moved: bool

    @property
    def slot_key(self) -> Tuple[int, int, int, int, int]:
        return self.egress, self.destination, self.channel, self.slot, self.generation


@dataclass(frozen=True)
class StaticSlotPlan:
    """Candidate exact plan plus an atomically committed or bypassed result."""

    candidate: CountPlan
    enabled: bool
    committed_quota: Matrix
    committed_keep_count: Matrix
    committed_segments: Tuple[TransferSegment, ...]
    assignments: Tuple[SlotAssignment, ...]
    moved_copies: int
    bypass_reason: Optional[str]
    generation: int
    num_channels: int
    channel_capacity: int


@dataclass(frozen=True, order=True)
class PhysicalProxyAssignment:
    """One moved copy in an egress-local compact payload slot.

    ``assignment.slot`` remains the logical slot in the unified retained/proxy
    namespace.  ``physical_slot`` is a separate, compact index into the moved-
    only payload arena on ``egress``.
    """

    egress: int
    physical_slot: int
    assignment: SlotAssignment

    @property
    def physical_slot_key(self) -> Tuple[int, int, int]:
        return self.egress, self.physical_slot, self.assignment.generation


@dataclass(frozen=True)
class PhysicalProxyPlan:
    """Atomic commitment result for moved-only physical proxy payloads."""

    logical_candidate: StaticSlotPlan
    committed_logical: StaticSlotPlan
    enabled: bool
    assignments: Tuple[PhysicalProxyAssignment, ...]
    moved_copies: int
    incoming_copies_per_egress: Tuple[int, ...]
    global_policy_budget: int
    physical_capacity_per_egress: int
    bypass_reason: Optional[str]


@dataclass(frozen=True, order=True)
class FullEgressAssignment:
    """One retained or moved copy in a compact source-egress record slot."""

    egress: int
    physical_slot: int
    assignment: SlotAssignment

    @property
    def physical_slot_key(self) -> Tuple[int, int, int]:
        return self.egress, self.physical_slot, self.assignment.generation


@dataclass(frozen=True)
class FullEgressPlan:
    """Atomic full-dispatch packing used by the C060 virtual-node loop."""

    logical_candidate: StaticSlotPlan
    enabled: bool
    assignments: Tuple[FullEgressAssignment, ...]
    records_per_egress: Tuple[int, ...]
    physical_capacity_per_egress: int
    bypass_reason: Optional[str]


@dataclass(frozen=True)
class DestinationSegmentedEgressPlan:
    """Full-dispatch packing with one fixed physical segment per destination.

    C061 keeps destination prefixes independent so unequal/non-divisible
    quotas create explicit holes instead of being hidden by aggregate compact
    packing.  A record for destination ``d`` uses
    ``d * capacity_per_destination + local_slot`` on its selected egress.
    """

    logical_candidate: StaticSlotPlan
    enabled: bool
    assignments: Tuple[FullEgressAssignment, ...]
    records_per_egress_destination: Matrix
    capacity_per_destination: int
    physical_capacity_per_egress: int
    bypass_reason: Optional[str]


@dataclass(frozen=True, order=True)
class ReadyHoleObservation:
    """A current-generation record observed beyond the first unready slot."""

    publication_step: int
    contiguous_tail: int
    max_ready_plus_one: int


@dataclass(frozen=True)
class ContiguousReadyTrace:
    """Scalar oracle trace for one generation of a one-shot proxy queue."""

    generation: int
    publication_order: Tuple[int, ...]
    tail_history: Tuple[int, ...]
    holes: Tuple[ReadyHoleObservation, ...]
    stale_front_observations: int
    final_ready: Tuple[int, ...]


def advance_contiguous_ready_tail(
    ready_generations: Sequence[int],
    *,
    generation: int,
    start_tail: int = 0,
) -> int:
    """Advance only across a current-generation ready prefix.

    A nonzero value from an older generation is a hole, not readiness.  The
    helper is intentionally scalar and side-effect free so GPU protocol traces
    can be compared against it exactly.
    """

    generation = _require_int("generation", generation, minimum=1)
    if generation > 0x7FFFFFFF:
        raise ValueError("generation exceeds the signed int32 CUDA ABI")
    start_tail = _require_int("start_tail", start_tail, minimum=0)
    normalized = tuple(
        _require_int("ready generation", value, minimum=0)
        for value in ready_generations
    )
    if any(value > 0x7FFFFFFF for value in normalized):
        raise ValueError("ready generation exceeds the signed int32 CUDA ABI")
    if start_tail > len(normalized):
        raise ValueError("start_tail exceeds ready capacity")
    if any(value != generation for value in normalized[:start_tail]):
        raise ValueError(
            "start_tail is not backed by a current-generation ready prefix"
        )
    tail = start_tail
    while tail < len(normalized) and normalized[tail] == generation:
        tail += 1
    return tail


def simulate_contiguous_ready_publication(
    publication_order: Sequence[int],
    *,
    generation: int,
    initial_ready: Optional[Sequence[int]] = None,
) -> ContiguousReadyTrace:
    """Publish a complete one-shot generation and trace its safe tail.

    ``publication_order`` must be a permutation of ``[0, N)``.  This models a
    statically allocated one-shot call, not ring reuse.  ``initial_ready`` may
    contain zero or stale generations but must not already contain the current
    generation; accepting that would hide an ABA/configuration error.
    """

    generation = _require_int("generation", generation, minimum=1)
    if generation > 0x7FFFFFFF:
        raise ValueError("generation exceeds the signed int32 CUDA ABI")
    order = tuple(
        _require_int("publication slot", value, minimum=0)
        for value in publication_order
    )
    num_slots = len(order)
    if sorted(order) != list(range(num_slots)):
        raise ValueError("publication_order must be a complete slot permutation")

    if initial_ready is None:
        ready = [0] * num_slots
    else:
        ready = [
            _require_int("initial ready generation", value, minimum=0)
            for value in initial_ready
        ]
        if any(value > 0x7FFFFFFF for value in ready):
            raise ValueError("ready generation exceeds the signed int32 CUDA ABI")
        if len(ready) != num_slots:
            raise ValueError("initial_ready length must match publication_order")
        if generation in ready:
            raise ValueError("initial_ready already contains the current generation")

    tail = advance_contiguous_ready_tail(
        ready, generation=generation, start_tail=0)
    tail_history = [tail]
    holes = []
    stale_front_observations = 0
    for step, slot in enumerate(order, start=1):
        ready[slot] = generation
        if tail < num_slots and ready[tail] != generation:
            if ready[tail] != 0:
                stale_front_observations += 1
            later = [
                index for index in range(tail + 1, num_slots)
                if ready[index] == generation
            ]
            if later:
                holes.append(ReadyHoleObservation(
                    publication_step=step,
                    contiguous_tail=tail,
                    max_ready_plus_one=max(later) + 1,
                ))

        new_tail = advance_contiguous_ready_tail(
            ready, generation=generation, start_tail=tail)
        if not tail <= new_tail <= num_slots:
            raise AssertionError("contiguous ready tail is not monotonic")
        if any(ready[index] != generation for index in range(new_tail)):
            raise AssertionError("contiguous ready tail crossed a hole")
        tail = new_tail
        tail_history.append(tail)

    if tail != num_slots:
        raise AssertionError("complete publication did not reach the final tail")
    return ContiguousReadyTrace(
        generation=generation,
        publication_order=order,
        tail_history=tuple(tail_history),
        holes=tuple(holes),
        stale_front_observations=stale_front_observations,
        final_ready=tuple(ready),
    )


def _as_nonnegative_matrix(counts: Sequence[Sequence[int]]) -> Matrix:
    if not counts:
        raise ValueError("counts must contain at least one rail")
    width = len(counts[0])
    matrix = []
    for row in counts:
        if len(row) != width:
            raise ValueError("counts must be rectangular")
        normalized_row = []
        for value in row:
            try:
                normalized_row.append(_require_int("count", value, minimum=0))
            except ValueError as error:
                raise ValueError("counts must contain nonnegative integers") from error
        matrix.append(tuple(normalized_row))
    return tuple(matrix)


def _ring_order(size: int, start: int) -> Tuple[int, ...]:
    return tuple((start + offset) % size for offset in range(size))


def _minimum_move_quota(matrix: Matrix, remainder_seed: int) -> Matrix:
    num_rails = len(matrix)
    num_destinations = len(matrix[0])
    quota = [[0 for _ in range(num_destinations)] for _ in range(num_rails)]
    for destination in range(num_destinations):
        column = [matrix[rail][destination] for rail in range(num_rails)]
        total = sum(column)
        base, remainder = divmod(total, num_rails)
        start = (remainder_seed + destination) % num_rails
        order = _ring_order(num_rails, start)

        # Raising a quota from base to base + 1 retains one extra copy exactly
        # when the original count is greater than base.  Prefer those rails;
        # ring order only breaks equal-gain ties.
        positive_gain = [rail for rail in order if column[rail] > base]
        zero_gain = [rail for rail in order if column[rail] <= base]
        winners = set((positive_gain + zero_gain)[:remainder])
        for rail in range(num_rails):
            quota[rail][destination] = base + int(rail in winners)
    return tuple(tuple(row) for row in quota)


def build_destination_copies(
    topk_destinations: Sequence[Sequence[Sequence[int]]],
    *,
    num_destinations: int,
    num_channels: int,
    local_destination: Optional[int],
) -> Tuple[DestinationCopy, ...]:
    """Normalize routes into unique ``(owner, token, destination)`` copies.

    Negative destinations are masks.  Destination order within a token is not
    semantically relevant, so the helper sorts the deduplicated set.  This makes
    the planner invariant to top-k lane permutations.
    """

    num_destinations = _require_int(
        "num_destinations", num_destinations, minimum=0)
    num_channels = _require_int("num_channels", num_channels, minimum=1)
    if local_destination is not None:
        local_destination = _require_int(
            "local_destination", local_destination, minimum=0)
        if local_destination >= num_destinations:
            raise ValueError("local_destination is out of range")

    ordinals = [[0 for _ in range(num_destinations)] for _ in topk_destinations]
    copies = []
    for owner, owner_tokens in enumerate(topk_destinations):
        for token, destinations in enumerate(owner_tokens):
            canonical = set()
            for destination in destinations:
                if isinstance(destination, bool) or not isinstance(destination, int):
                    raise ValueError("destinations must be integers")
                if destination < 0:
                    continue
                if destination >= num_destinations:
                    raise ValueError("destination is out of range")
                if destination != local_destination:
                    canonical.add(destination)
            for destination in sorted(canonical):
                ordinal = ordinals[owner][destination]
                copies.append(DestinationCopy(
                    owner=owner,
                    token=token,
                    destination=destination,
                    source_channel=token % num_channels,
                    owner_ordinal=ordinal,
                ))
                ordinals[owner][destination] += 1
    return tuple(copies)


def counts_from_copies(
    copies: Sequence[DestinationCopy],
    *,
    num_rails: int,
    num_destinations: int,
    num_channels: Optional[int] = None,
) -> Matrix:
    num_rails = _require_int("num_rails", num_rails, minimum=1)
    num_destinations = _require_int(
        "num_destinations", num_destinations, minimum=0)
    if num_channels is not None:
        num_channels = _require_int("num_channels", num_channels, minimum=1)

    counts = [[0 for _ in range(num_destinations)] for _ in range(num_rails)]
    seen = set()
    grouped_ordinals = {}
    for copy in copies:
        _require_int("copy owner", copy.owner, minimum=0)
        _require_int("copy token", copy.token, minimum=0)
        _require_int("copy destination", copy.destination, minimum=0)
        _require_int("copy source channel", copy.source_channel, minimum=0)
        _require_int("copy owner ordinal", copy.owner_ordinal, minimum=0)
        if copy.key in seen:
            raise ValueError(f"duplicate destination copy: {copy.key}")
        seen.add(copy.key)
        if not 0 <= copy.owner < num_rails:
            raise ValueError("copy owner is out of range")
        if not 0 <= copy.destination < num_destinations:
            raise ValueError("copy destination is out of range")
        if num_channels is not None and not 0 <= copy.source_channel < num_channels:
            raise ValueError("copy source channel is out of range")

        group = copy.owner, copy.destination
        expected_ordinal = grouped_ordinals.get(group, 0)
        if copy.owner_ordinal != expected_ordinal:
            raise ValueError(
                f"non-canonical owner ordinal for {group}: "
                f"expected {expected_ordinal}, got {copy.owner_ordinal}"
            )
        grouped_ordinals[group] = expected_ordinal + 1
        counts[copy.owner][copy.destination] += 1
    return tuple(tuple(row) for row in counts)


def build_count_plan(
    counts: Sequence[Sequence[int]],
    *,
    remainder_seed: int = 0,
) -> CountPlan:
    """Build the minimum-move exact quota and stable rail-level segments."""

    remainder_seed = _require_int("remainder_seed", remainder_seed)
    matrix = _as_nonnegative_matrix(counts)
    num_rails = len(matrix)
    num_destinations = len(matrix[0])
    quota = _minimum_move_quota(matrix, remainder_seed)

    keep = [
        [min(matrix[rail][destination], quota[rail][destination])
         for destination in range(num_destinations)]
        for rail in range(num_rails)
    ]
    segments = []
    for destination in range(num_destinations):
        surplus = [max(matrix[rail][destination] - quota[rail][destination], 0)
                   for rail in range(num_rails)]
        deficit = [max(quota[rail][destination] - matrix[rail][destination], 0)
                   for rail in range(num_rails)]
        owner_consumed = [0 for _ in range(num_rails)]
        owner = egress = 0
        while owner < num_rails and egress < num_rails:
            while owner < num_rails and surplus[owner] == 0:
                owner += 1
            while egress < num_rails and deficit[egress] == 0:
                egress += 1
            if owner == num_rails or egress == num_rails:
                break
            count = min(surplus[owner], deficit[egress])
            segments.append(TransferSegment(
                destination=destination,
                owner=owner,
                egress=egress,
                owner_begin=keep[owner][destination] + owner_consumed[owner],
                count=count,
            ))
            surplus[owner] -= count
            deficit[egress] -= count
            owner_consumed[owner] += count

    plan = CountPlan(
        counts=matrix,
        quota=quota,
        keep_count=tuple(tuple(row) for row in keep),
        segments=tuple(segments),
        moved_copies=sum(segment.count for segment in segments),
        remainder_seed=remainder_seed,
    )
    validate_count_plan(plan)
    return plan


def validate_count_plan(plan: CountPlan) -> None:
    remainder_seed = _require_int("remainder_seed", plan.remainder_seed)
    _require_int("moved_copies", plan.moved_copies, minimum=0)
    counts = _as_nonnegative_matrix(plan.counts)
    quota = _as_nonnegative_matrix(plan.quota)
    keep = _as_nonnegative_matrix(plan.keep_count)
    if not (len(counts) == len(quota) == len(keep)):
        raise AssertionError("count-plan rail dimensions differ")
    if counts and not (len(counts[0]) == len(quota[0]) == len(keep[0])):
        raise AssertionError("count-plan destination dimensions differ")

    num_rails = len(counts)
    num_destinations = len(counts[0])
    if quota != _minimum_move_quota(counts, remainder_seed):
        raise AssertionError("quota is not the canonical minimum-move quota")
    moved_by_owner = [[0 for _ in range(num_destinations)] for _ in range(num_rails)]
    incoming_by_egress = [[0 for _ in range(num_destinations)] for _ in range(num_rails)]
    expected_begin = [[keep[rail][destination] for destination in range(num_destinations)]
                      for rail in range(num_rails)]
    segments_per_destination = [0 for _ in range(num_destinations)]

    for segment in plan.segments:
        _require_int("segment destination", segment.destination, minimum=0)
        _require_int("segment owner", segment.owner, minimum=0)
        _require_int("segment egress", segment.egress, minimum=0)
        _require_int("segment owner_begin", segment.owner_begin, minimum=0)
        _require_int("segment count", segment.count, minimum=1)
        if segment.count <= 0 or segment.owner == segment.egress:
            raise AssertionError("invalid transfer segment")
        if not 0 <= segment.destination < num_destinations:
            raise AssertionError("segment destination is out of range")
        if not 0 <= segment.owner < num_rails or not 0 <= segment.egress < num_rails:
            raise AssertionError("segment rail is out of range")
        if segment.owner_begin != expected_begin[segment.owner][segment.destination]:
            raise AssertionError("segment owner ranges are not contiguous")
        expected_begin[segment.owner][segment.destination] += segment.count
        moved_by_owner[segment.owner][segment.destination] += segment.count
        incoming_by_egress[segment.egress][segment.destination] += segment.count
        segments_per_destination[segment.destination] += 1

    expected_moved = 0
    for destination in range(num_destinations):
        count_column = [counts[rail][destination] for rail in range(num_rails)]
        quota_column = [quota[rail][destination] for rail in range(num_rails)]
        if sum(count_column) != sum(quota_column):
            raise AssertionError("quota does not conserve copies")
        if max(quota_column) - min(quota_column) > 1:
            raise AssertionError("quota is not exact-balanced")
        if max(count_column) - min(count_column) <= 1 and count_column != quota_column:
            raise AssertionError("balanced input is not an identity plan")
        if segments_per_destination[destination] > max(num_rails - 1, 0):
            raise AssertionError("too many rail-level segments")

        for rail in range(num_rails):
            expected_keep = min(counts[rail][destination], quota[rail][destination])
            if keep[rail][destination] != expected_keep:
                raise AssertionError("keep count is inconsistent")
            expected_surplus = max(counts[rail][destination] - quota[rail][destination], 0)
            expected_deficit = max(quota[rail][destination] - counts[rail][destination], 0)
            if moved_by_owner[rail][destination] != expected_surplus:
                raise AssertionError("surplus was not consumed exactly")
            if incoming_by_egress[rail][destination] != expected_deficit:
                raise AssertionError("deficit was not filled exactly")
            expected_moved += expected_surplus

    if plan.moved_copies != expected_moved:
        raise AssertionError("moved copy total is inconsistent")


def validate_static_slot_plan(plan: StaticSlotPlan) -> None:
    """Fail-closed validation for a committed logical slot plan."""

    if not isinstance(plan, StaticSlotPlan):
        raise ValueError("plan must be a StaticSlotPlan")
    if not isinstance(plan.enabled, bool):
        raise ValueError("enabled must be a boolean")
    validate_count_plan(plan.candidate)
    generation = _require_int("generation", plan.generation, minimum=0)
    num_channels = _require_int(
        "num_channels", plan.num_channels, minimum=1)
    channel_capacity = _require_int(
        "channel_capacity", plan.channel_capacity, minimum=1)
    _require_int("moved_copies", plan.moved_copies, minimum=0)

    quota = _as_nonnegative_matrix(plan.committed_quota)
    keep = _as_nonnegative_matrix(plan.committed_keep_count)
    if len(quota) != plan.candidate.num_rails or any(
            len(row) != plan.candidate.num_destinations for row in quota):
        raise AssertionError("committed quota dimensions are inconsistent")
    if len(keep) != plan.candidate.num_rails or any(
            len(row) != plan.candidate.num_destinations for row in keep):
        raise AssertionError("committed keep dimensions are inconsistent")

    if not plan.enabled:
        if not isinstance(plan.bypass_reason, str) or not plan.bypass_reason:
            raise AssertionError("disabled plan must have a bypass reason")
        if quota != plan.candidate.counts:
            raise AssertionError("disabled plan must commit identity quota")
        if keep != plan.candidate.counts:
            raise AssertionError("disabled plan must keep every source copy")
        if plan.committed_segments or plan.assignments or plan.moved_copies != 0:
            raise AssertionError(
                "disabled plan must be an atomic identity commit")
        return

    if plan.bypass_reason is not None:
        raise AssertionError("enabled plan cannot have a bypass reason")
    if quota != plan.candidate.quota:
        raise AssertionError("enabled plan must commit candidate quota")
    if keep != plan.candidate.keep_count:
        raise AssertionError("enabled plan must commit candidate keep counts")
    if plan.committed_segments != plan.candidate.segments:
        raise AssertionError("enabled plan must commit candidate segments")
    if (plan.moved_copies != plan.candidate.moved_copies or
            plan.moved_copies <= 0):
        raise AssertionError(
            "enabled moved-copy count differs from candidate")

    num_rails = plan.candidate.num_rails
    num_destinations = plan.candidate.num_destinations
    if len(plan.assignments) != sum(map(sum, plan.candidate.counts)):
        raise AssertionError(
            "slot assignments do not cover every source copy")

    moved_egress = {}
    for segment in plan.committed_segments:
        for ordinal in range(
                segment.owner_begin, segment.owner_begin + segment.count):
            key = segment.owner, segment.destination, ordinal
            if key in moved_egress:
                raise AssertionError("committed segments overlap source copies")
            moved_egress[key] = segment.egress

    copy_keys = set()
    ordinal_keys = set()
    slot_keys = set()
    assigned_counts = [
        [0] * num_destinations for _ in range(num_rails)]
    namespaces = {}
    copies_by_owner_destination = {}
    observed_moved = 0
    for assignment in plan.assignments:
        if not isinstance(assignment, SlotAssignment):
            raise AssertionError(
                "assignments must contain SlotAssignment values")
        if not isinstance(assignment.moved, bool):
            raise ValueError("assignment moved must be a boolean")
        copy = assignment.copy
        if not isinstance(copy, DestinationCopy):
            raise AssertionError(
                "assignment copy must be a DestinationCopy")
        for name, value in (
                ("assignment egress", assignment.egress),
                ("assignment destination", assignment.destination),
                ("assignment channel", assignment.channel),
                ("assignment slot", assignment.slot),
                ("assignment generation", assignment.generation),
                ("copy owner", copy.owner),
                ("copy token", copy.token),
                ("copy destination", copy.destination),
                ("copy source channel", copy.source_channel),
                ("copy owner ordinal", copy.owner_ordinal)):
            _require_int(name, value, minimum=0)
        if not 0 <= assignment.egress < num_rails:
            raise AssertionError("assignment egress is out of range")
        if not 0 <= assignment.destination < num_destinations:
            raise AssertionError("assignment destination is out of range")
        if (not 0 <= copy.owner < num_rails or
                not 0 <= copy.destination < num_destinations):
            raise AssertionError("assignment copy is out of range")
        if assignment.destination != copy.destination:
            raise AssertionError(
                "assignment destination differs from its copy")
        if assignment.generation != generation:
            raise AssertionError(
                "assignment generation differs from its plan")
        if not 0 <= assignment.channel < num_channels:
            raise AssertionError("assignment channel is out of range")
        if not 0 <= copy.source_channel < num_channels:
            raise AssertionError("copy source channel is out of range")
        if copy.source_channel != copy.token % num_channels:
            raise AssertionError("copy source channel is not canonical")
        if not 0 <= assignment.slot < channel_capacity:
            raise AssertionError("assignment slot is out of range")
        if copy.key in copy_keys:
            raise AssertionError("destination copy is assigned more than once")
        copy_keys.add(copy.key)
        ordinal_key = copy.owner, copy.destination, copy.owner_ordinal
        if ordinal_key in ordinal_keys:
            raise AssertionError(
                "source copy ordinal is assigned more than once")
        ordinal_keys.add(ordinal_key)

        segment_egress = moved_egress.get(ordinal_key)
        if assignment.moved:
            observed_moved += 1
            if copy.owner == assignment.egress:
                raise AssertionError("moved copy must use a different egress")
            if segment_egress is None:
                raise AssertionError(
                    "moved assignment is absent from segments")
            if assignment.egress != segment_egress:
                raise AssertionError(
                    "moved assignment egress differs from segment")
            if copy.owner_ordinal < keep[copy.owner][copy.destination]:
                raise AssertionError(
                    "moved assignment overlaps retained prefix")
        else:
            if assignment.egress != copy.owner:
                raise AssertionError(
                    "retained copy must stay on its owner egress")
            if assignment.channel != copy.source_channel:
                raise AssertionError(
                    "retained copy must stay on its source channel")
            if segment_egress is not None:
                raise AssertionError("segment copy was marked retained")
            if copy.owner_ordinal >= keep[copy.owner][copy.destination]:
                raise AssertionError(
                    "retained assignment exceeds committed prefix")

        if assignment.slot_key in slot_keys:
            raise AssertionError("unified retained/proxy slot collision")
        slot_keys.add(assignment.slot_key)
        assigned_counts[assignment.egress][assignment.destination] += 1
        namespace = (
            assignment.egress, assignment.destination, assignment.channel)
        namespaces.setdefault(namespace, []).append(assignment)
        copies_by_owner_destination.setdefault(
            (copy.owner, copy.destination), []).append(copy)

    for owner in range(num_rails):
        for destination in range(num_destinations):
            copies = sorted(
                copies_by_owner_destination.get((owner, destination), ()),
                key=lambda copy: copy.token,
            )
            expected_count = plan.candidate.counts[owner][destination]
            if [copy.owner_ordinal for copy in copies] != list(
                    range(expected_count)):
                raise AssertionError(
                    "slot assignments have incomplete or noncanonical copy coverage")
    if tuple(map(tuple, assigned_counts)) != quota:
        raise AssertionError(
            "assignment egress counts differ from committed quota")
    if (observed_moved != plan.moved_copies or
            len(moved_egress) != plan.moved_copies):
        raise AssertionError(
            "moved assignments and committed segments disagree")

    for values in namespaces.values():
        ordered = sorted(values, key=lambda item: item.slot)
        if [item.slot for item in ordered] != list(range(len(ordered))):
            raise AssertionError("slot namespace is not dense")
        seen_proxy = False
        for item in ordered:
            if item.moved:
                seen_proxy = True
            elif seen_proxy:
                raise AssertionError(
                    "retained slots must precede proxy slots")


def _identity_matrix(matrix: Matrix) -> Matrix:
    return tuple(tuple(row) for row in matrix)


def _bypassed_plan(
    candidate: CountPlan,
    *,
    reason: str,
    generation: int,
    num_channels: int,
    channel_capacity: int,
) -> StaticSlotPlan:
    plan = StaticSlotPlan(
        candidate=candidate,
        enabled=False,
        committed_quota=_identity_matrix(candidate.counts),
        committed_keep_count=_identity_matrix(candidate.counts),
        committed_segments=(),
        assignments=(),
        moved_copies=0,
        bypass_reason=reason,
        generation=generation,
        num_channels=num_channels,
        channel_capacity=channel_capacity,
    )
    validate_static_slot_plan(plan)
    return plan


def materialize_static_slots(
    copies: Sequence[DestinationCopy],
    candidate: CountPlan,
    *,
    num_channels: int,
    channel_capacity: int,
    generation: int = 0,
    channel_seed: int = 0,
    proxy_payload_capacity: Optional[int] = None,
    metadata_capacity: Optional[int] = None,
) -> StaticSlotPlan:
    """Assign one-shot slots, committing atomically or bypassing the whole plan."""

    num_channels = _require_int("num_channels", num_channels, minimum=1)
    channel_capacity = _require_int(
        "channel_capacity", channel_capacity, minimum=1)
    generation = _require_int("generation", generation, minimum=0)
    channel_seed = _require_int("channel_seed", channel_seed)
    if proxy_payload_capacity is not None:
        proxy_payload_capacity = _require_int(
            "proxy_payload_capacity", proxy_payload_capacity, minimum=0)
    if metadata_capacity is not None:
        metadata_capacity = _require_int(
            "metadata_capacity", metadata_capacity, minimum=0)
    validate_count_plan(candidate)

    canonical_copies = tuple(sorted(copies, key=lambda copy: (
        copy.owner, copy.destination, copy.owner_ordinal, copy.token)))
    observed_counts = counts_from_copies(
        canonical_copies,
        num_rails=candidate.num_rails,
        num_destinations=candidate.num_destinations,
        num_channels=num_channels,
    )
    if observed_counts != candidate.counts:
        raise ValueError("copies do not match candidate counts")

    def bypass(reason: str) -> StaticSlotPlan:
        return _bypassed_plan(
            candidate,
            reason=reason,
            generation=generation,
            num_channels=num_channels,
            channel_capacity=channel_capacity,
        )

    if candidate.moved_copies == 0:
        return bypass("already_balanced")
    if (proxy_payload_capacity is not None and
            candidate.moved_copies > proxy_payload_capacity):
        return bypass("proxy_payload_capacity")
    if metadata_capacity is not None and len(candidate.segments) > metadata_capacity:
        return bypass("metadata_capacity")

    by_owner_destination = {}
    for copy in canonical_copies:
        by_owner_destination.setdefault((copy.owner, copy.destination), []).append(copy)

    retained = []
    moved_lookup = {}
    for (owner, destination), group in by_owner_destination.items():
        keep_count = candidate.keep_count[owner][destination]
        retained.extend(group[:keep_count])
        for copy in group[keep_count:]:
            moved_lookup[(owner, destination, copy.owner_ordinal)] = copy

    # Candidate state remains private until every capacity check succeeds.
    occupancy = {}
    temporary = []
    for copy in sorted(retained, key=lambda item: (
            item.owner, item.destination, item.source_channel, item.owner_ordinal)):
        key = copy.owner, copy.destination, copy.source_channel
        slot = occupancy.get(key, 0)
        if slot >= channel_capacity:
            return bypass("retained_channel_capacity")
        occupancy[key] = slot + 1
        temporary.append(SlotAssignment(
            egress=copy.owner,
            destination=copy.destination,
            channel=copy.source_channel,
            slot=slot,
            generation=generation,
            copy=copy,
            moved=False,
        ))

    for segment in candidate.segments:
        for offset in range(segment.count):
            ordinal = segment.owner_begin + offset
            copy = moved_lookup.get((segment.owner, segment.destination, ordinal))
            if copy is None:
                raise AssertionError("segment does not resolve to a unique source copy")
            start = (channel_seed + segment.destination + segment.owner) % num_channels
            channel_order = _ring_order(num_channels, start)
            selected = min(
                channel_order,
                key=lambda channel: occupancy.get(
                    (segment.egress, segment.destination, channel), 0),
            )
            key = segment.egress, segment.destination, selected
            if occupancy.get(key, 0) >= channel_capacity:
                return bypass("channel_capacity")
            slot = occupancy.get(key, 0)
            occupancy[key] = slot + 1
            temporary.append(SlotAssignment(
                egress=segment.egress,
                destination=segment.destination,
                channel=selected,
                slot=slot,
                generation=generation,
                copy=copy,
                moved=True,
            ))

    assignments = tuple(sorted(temporary))
    slot_keys = [assignment.slot_key for assignment in assignments]
    if len(slot_keys) != len(set(slot_keys)):
        raise AssertionError("unified retained/proxy slot collision")
    moved_keys = [assignment.copy.key for assignment in assignments if assignment.moved]
    if len(moved_keys) != candidate.moved_copies or len(moved_keys) != len(set(moved_keys)):
        raise AssertionError("moved copies were not assigned exactly once")

    result = StaticSlotPlan(
        candidate=candidate,
        enabled=True,
        committed_quota=candidate.quota,
        committed_keep_count=candidate.keep_count,
        committed_segments=candidate.segments,
        assignments=assignments,
        moved_copies=candidate.moved_copies,
        bypass_reason=None,
        generation=generation,
        num_channels=num_channels,
        channel_capacity=channel_capacity,
    )
    validate_static_slot_plan(result)
    return result


def _physical_assignment_key(assignment: SlotAssignment) -> Tuple[int, ...]:
    """Canonical physical packing order, independent of input iteration order."""

    copy = assignment.copy
    return (
        assignment.egress,
        assignment.destination,
        assignment.channel,
        assignment.slot,
        assignment.generation,
        copy.owner,
        copy.destination,
        copy.owner_ordinal,
        copy.token,
        copy.source_channel,
    )


def materialize_full_egress_slots(
    logical_candidate: StaticSlotPlan,
    *,
    physical_capacity_per_egress: int,
) -> FullEgressPlan:
    """Pack retained and moved copies into compact per-egress prefixes.

    C040's physical arena intentionally contains moved copies only. A complete
    virtual dispatch must additionally transmit retained copies, so C060 uses a
    separate full-egress packing contract. Capacity failure is atomic and
    returns no partial manifest.
    """

    if not isinstance(logical_candidate, StaticSlotPlan):
        raise ValueError("logical_candidate must be a StaticSlotPlan")
    physical_capacity_per_egress = _require_int(
        "physical_capacity_per_egress",
        physical_capacity_per_egress,
        minimum=0,
    )
    validate_static_slot_plan(logical_candidate)
    num_egresses = logical_candidate.candidate.num_rails

    if not logical_candidate.enabled:
        return FullEgressPlan(
            logical_candidate=logical_candidate,
            enabled=False,
            assignments=(),
            records_per_egress=(0,) * num_egresses,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason=logical_candidate.bypass_reason,
        )

    grouped = [[] for _ in range(num_egresses)]
    for assignment in sorted(
            logical_candidate.assignments, key=_physical_assignment_key):
        grouped[assignment.egress].append(assignment)
    records_per_egress = tuple(len(values) for values in grouped)
    if any(count > physical_capacity_per_egress
           for count in records_per_egress):
        return FullEgressPlan(
            logical_candidate=logical_candidate,
            enabled=False,
            assignments=(),
            records_per_egress=(0,) * num_egresses,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason="full_egress_physical_capacity",
        )

    packed = tuple(
        FullEgressAssignment(
            egress=egress,
            physical_slot=physical_slot,
            assignment=assignment,
        )
        for egress, values in enumerate(grouped)
        for physical_slot, assignment in enumerate(values)
    )
    if len(packed) != len(logical_candidate.assignments):
        raise AssertionError("full-egress packing lost logical assignments")
    if len({item.assignment.copy.key for item in packed}) != len(packed):
        raise AssertionError("full-egress packing duplicated a destination copy")
    physical_keys = [item.physical_slot_key for item in packed]
    if len(physical_keys) != len(set(physical_keys)):
        raise AssertionError("full-egress physical slot collision")
    for egress, count in enumerate(records_per_egress):
        observed = sorted(
            item.physical_slot for item in packed if item.egress == egress)
        if observed != list(range(count)):
            raise AssertionError("full-egress slots are not a compact prefix")

    return FullEgressPlan(
        logical_candidate=logical_candidate,
        enabled=True,
        assignments=packed,
        records_per_egress=records_per_egress,
        physical_capacity_per_egress=physical_capacity_per_egress,
        bypass_reason=None,
    )


def materialize_destination_segmented_egress_slots(
    logical_candidate: StaticSlotPlan,
    *,
    capacity_per_destination: int,
) -> DestinationSegmentedEgressPlan:
    """Pack full records into dense per-(egress,destination) prefixes.

    Capacity commitment is atomic across every destination segment.  A single
    overflowing quota returns no assignments and no partially committed count
    matrix.
    """

    if not isinstance(logical_candidate, StaticSlotPlan):
        raise ValueError("logical_candidate must be a StaticSlotPlan")
    capacity_per_destination = _require_int(
        "capacity_per_destination", capacity_per_destination, minimum=0)
    validate_static_slot_plan(logical_candidate)
    num_egresses = logical_candidate.candidate.num_rails
    num_destinations = logical_candidate.candidate.num_destinations
    physical_capacity_per_egress = (
        capacity_per_destination * num_destinations)
    empty_counts = tuple(
        (0,) * num_destinations for _ in range(num_egresses))

    if not logical_candidate.enabled:
        return DestinationSegmentedEgressPlan(
            logical_candidate=logical_candidate,
            enabled=False,
            assignments=(),
            records_per_egress_destination=empty_counts,
            capacity_per_destination=capacity_per_destination,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason=logical_candidate.bypass_reason,
        )

    grouped = [
        [[] for _ in range(num_destinations)]
        for _ in range(num_egresses)
    ]
    for assignment in sorted(
            logical_candidate.assignments, key=_physical_assignment_key):
        grouped[assignment.egress][assignment.destination].append(assignment)
    records = tuple(tuple(len(values) for values in row) for row in grouped)
    if records != logical_candidate.committed_quota:
        raise AssertionError(
            "destination-segmented counts differ from committed quota")
    if any(count > capacity_per_destination
           for row in records for count in row):
        return DestinationSegmentedEgressPlan(
            logical_candidate=logical_candidate,
            enabled=False,
            assignments=(),
            records_per_egress_destination=empty_counts,
            capacity_per_destination=capacity_per_destination,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason="destination_segment_physical_capacity",
        )

    packed = tuple(
        FullEgressAssignment(
            egress=egress,
            physical_slot=(
                destination * capacity_per_destination + local_slot),
            assignment=assignment,
        )
        for egress, row in enumerate(grouped)
        for destination, values in enumerate(row)
        for local_slot, assignment in enumerate(values)
    )
    if len(packed) != len(logical_candidate.assignments):
        raise AssertionError(
            "destination-segmented packing lost logical assignments")
    if len({item.assignment.copy.key for item in packed}) != len(packed):
        raise AssertionError(
            "destination-segmented packing duplicated a destination copy")
    physical_keys = [item.physical_slot_key for item in packed]
    if len(physical_keys) != len(set(physical_keys)):
        raise AssertionError(
            "destination-segmented physical slot collision")
    for item in packed:
        if (item.physical_slot // capacity_per_destination !=
                item.assignment.destination):
            raise AssertionError(
                "destination-segmented physical slot escaped its namespace")

    return DestinationSegmentedEgressPlan(
        logical_candidate=logical_candidate,
        enabled=True,
        assignments=packed,
        records_per_egress_destination=records,
        capacity_per_destination=capacity_per_destination,
        physical_capacity_per_egress=physical_capacity_per_egress,
        bypass_reason=None,
    )


def materialize_physical_proxy_slots(
    logical_candidate: StaticSlotPlan,
    *,
    global_policy_budget: int,
    physical_capacity_per_egress: int,
) -> PhysicalProxyPlan:
    """Pack moved copies into compact, egress-local physical payload slots.

    The two capacities deliberately have different meanings:

    * ``global_policy_budget`` bounds total moved copies across all egresses;
    * ``physical_capacity_per_egress`` bounds payload records resident on each
      individual egress GPU.  Symmetric memory reserves this many records on
      every egress, even when the actual incoming counts differ.

    Capacity failure is atomic.  Candidate counts remain available for
    diagnostics, but the committed logical plan is replaced by a complete
    identity bypass and no physical assignment is returned.
    """

    if not isinstance(logical_candidate, StaticSlotPlan):
        raise ValueError("logical_candidate must be a StaticSlotPlan")
    global_policy_budget = _require_int(
        "global_policy_budget", global_policy_budget, minimum=0)
    physical_capacity_per_egress = _require_int(
        "physical_capacity_per_egress",
        physical_capacity_per_egress,
        minimum=0,
    )
    validate_static_slot_plan(logical_candidate)

    num_egresses = logical_candidate.candidate.num_rails
    incoming = [0 for _ in range(num_egresses)]
    moved = []
    seen_logical_slots = set()
    seen_copy_keys = set()
    for assignment in logical_candidate.assignments:
        if not isinstance(assignment, SlotAssignment):
            raise ValueError("logical assignments must be SlotAssignment values")
        if not 0 <= assignment.egress < num_egresses:
            raise ValueError("logical assignment egress is out of range")
        if assignment.generation != logical_candidate.generation:
            raise ValueError("logical assignment generation does not match its plan")
        if assignment.slot_key in seen_logical_slots:
            raise ValueError("duplicate logical slot assignment")
        seen_logical_slots.add(assignment.slot_key)
        if assignment.copy.key in seen_copy_keys:
            raise ValueError("duplicate logical destination copy")
        seen_copy_keys.add(assignment.copy.key)
        if assignment.moved:
            moved.append(assignment)
            incoming[assignment.egress] += 1

    if logical_candidate.enabled:
        if logical_candidate.bypass_reason is not None:
            raise ValueError("enabled logical plan cannot have a bypass reason")
        if len(moved) != logical_candidate.moved_copies:
            raise ValueError("logical moved-copy count does not match assignments")
    else:
        if logical_candidate.assignments or logical_candidate.moved_copies != 0:
            raise ValueError("disabled logical plan must be an atomic identity bypass")
        return PhysicalProxyPlan(
            logical_candidate=logical_candidate,
            committed_logical=logical_candidate,
            enabled=False,
            assignments=(),
            moved_copies=0,
            incoming_copies_per_egress=tuple(incoming),
            global_policy_budget=global_policy_budget,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason=logical_candidate.bypass_reason,
        )

    def bypass(reason: str) -> PhysicalProxyPlan:
        return PhysicalProxyPlan(
            logical_candidate=logical_candidate,
            committed_logical=_bypassed_plan(
                logical_candidate.candidate,
                reason=reason,
                generation=logical_candidate.generation,
                num_channels=logical_candidate.num_channels,
                channel_capacity=logical_candidate.channel_capacity,
            ),
            enabled=False,
            assignments=(),
            moved_copies=0,
            incoming_copies_per_egress=tuple(incoming),
            global_policy_budget=global_policy_budget,
            physical_capacity_per_egress=physical_capacity_per_egress,
            bypass_reason=reason,
        )

    if len(moved) > global_policy_budget:
        return bypass("global_policy_budget")
    if any(count > physical_capacity_per_egress for count in incoming):
        return bypass("physical_egress_capacity")

    physical_tail = [0 for _ in range(num_egresses)]
    physical_assignments = []
    for assignment in sorted(moved, key=_physical_assignment_key):
        physical_slot = physical_tail[assignment.egress]
        physical_tail[assignment.egress] += 1
        physical_assignments.append(PhysicalProxyAssignment(
            egress=assignment.egress,
            physical_slot=physical_slot,
            assignment=assignment,
        ))

    physical_slot_keys = [
        assignment.physical_slot_key for assignment in physical_assignments
    ]
    if len(physical_slot_keys) != len(set(physical_slot_keys)):
        raise AssertionError("duplicate physical proxy payload slot")
    if any(not assignment.assignment.moved for assignment in physical_assignments):
        raise AssertionError("retained copy received a physical proxy payload slot")
    if tuple(physical_tail) != tuple(incoming):
        raise AssertionError("physical proxy payload packing is incomplete")

    return PhysicalProxyPlan(
        logical_candidate=logical_candidate,
        committed_logical=logical_candidate,
        enabled=True,
        assignments=tuple(physical_assignments),
        moved_copies=len(physical_assignments),
        incoming_copies_per_egress=tuple(incoming),
        global_policy_budget=global_policy_budget,
        physical_capacity_per_egress=physical_capacity_per_egress,
        bypass_reason=None,
    )
