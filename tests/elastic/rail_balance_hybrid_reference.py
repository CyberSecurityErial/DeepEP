"""CPU oracle for the C080 Hybrid rail-balance scheduling contract.

The production contract is deliberately compact: :class:`HybridRailSchedule`
contains counts, prefixes, and transfer segments, but never a full per-copy
manifest.  Tests may enumerate copies with :func:`enumerate_resolved_copies`
to prove that the compact resolver is complete and collision-free.

Input ``topk_destinations`` is indexed as ``[owner][token][topk_lane]`` and
contains destination-server indices (not expert indices).  Negative entries
are masked.  Each token contributes at most one payload copy to a destination;
duplicate lanes and the optional local destination are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


Matrix = Tuple[Tuple[int, ...], ...]
Tensor3 = Tuple[Tuple[Tuple[int, ...], ...], ...]

_INT32_MAX = 0x7FFFFFFF
_INT64_MAX = 0x7FFFFFFFFFFFFFFF
_MAX_RAILS = 32
_MAX_CHANNELS = 1024
_MAX_DESTINATIONS = 32
_MAX_TOPK = 32
_CAPACITY_FAILURE = "proxy_capacity_per_egress_exceeded"


def _require_int(name: str, value: int, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if value > _INT32_MAX:
        raise ValueError(f"{name} exceeds the signed int32 CUDA ABI")
    return value


def _require_nonnegative_int64(name: str, value: int) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be at least 0")
    if value > _INT64_MAX:
        raise ValueError(f"{name} exceeds the signed int64 host ABI")
    return value


def _require_sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


@dataclass(frozen=True, order=True)
class HybridTransferSegment:
    """A channel-major suffix moved from one owner to one egress rail.

    ``owner_begin`` is an ordinal in the owner's ``(owner, destination)``
    stream.  ``egress_begin`` is an ordinal in the incoming
    ``(egress, destination)`` stream and is therefore independent for every
    egress/destination pair.
    """

    owner: int
    egress: int
    owner_begin: int
    count: int
    egress_begin: int


@dataclass(frozen=True)
class HybridRailSchedule:
    """Compact, deterministic C080 Hybrid schedule (no per-copy manifest)."""

    num_rails: int
    num_channels: int
    num_destinations: int
    num_max_tokens_per_rank: int
    channel_capacity: int
    num_tokens_per_owner: Tuple[int, ...]
    local_destination: Optional[int]
    remainder_seed: int

    # [owner][channel][destination]
    channel_count: Tensor3
    owner_channel_prefix: Tensor3
    # [owner][destination]
    count: Matrix
    quota: Matrix
    keep_count: Matrix

    # [destination][segment], with at most G - 1 entries in each bucket.
    segments: Tuple[Tuple[HybridTransferSegment, ...], ...]
    num_segments: Tuple[int, ...]

    # [egress][channel][destination]
    retained: Tensor3
    moved: Tensor3
    # [egress][destination][channel prefix, length C + 1]
    moved_channel_prefix: Tensor3
    # [egress][channel][destination], channel-major/destination-minor
    group_prefix: Tensor3

    proxy_required: Tuple[int, ...]
    moved_copies: int
    proxy_capacity_per_egress: Optional[int]
    enabled: bool
    failure_reason: Optional[str]


@dataclass(frozen=True, order=True)
class HybridDestinationCopy:
    """One deduplicated copy, exposed only for oracle/test enumeration."""

    owner: int
    token: int
    destination: int
    source_channel: int
    channel_local_ordinal: int
    owner_ordinal: int

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.owner, self.token, self.destination


@dataclass(frozen=True, order=True)
class HybridCopyResolution:
    """Result of resolving one copy through the compact schedule."""

    owner: int
    source_channel: int
    destination: int
    channel_local_ordinal: int
    owner_ordinal: int
    moved: bool
    egress: int
    channel: int
    remote_slot: int
    proxy_slot: int
    incoming_ordinal: int


@dataclass(frozen=True, order=True)
class ResolvedHybridDestinationCopy:
    """Test-only pairing of a logical copy and its compact resolution."""

    copy: HybridDestinationCopy
    resolution: HybridCopyResolution


def _normalize_topk_destinations(
    topk_destinations: Sequence[Sequence[Sequence[int]]],
    *,
    num_destinations: int,
    num_max_tokens_per_rank: int,
    local_destination: Optional[int],
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    owners = _require_sequence("topk_destinations", topk_destinations)
    if not owners:
        raise ValueError("topk_destinations must contain at least one owner")
    if len(owners) > _MAX_RAILS:
        raise ValueError(
            f"topk_destinations exceeds the force-v1 {_MAX_RAILS}-rail limit")

    normalized_owners = []
    for owner, raw_tokens in enumerate(owners):
        tokens = _require_sequence(f"topk_destinations[{owner}]", raw_tokens)
        if len(tokens) > num_max_tokens_per_rank:
            raise ValueError(
                f"owner {owner} has {len(tokens)} tokens, exceeding "
                f"num_max_tokens_per_rank={num_max_tokens_per_rank}"
            )
        normalized_tokens = []
        for token, raw_lanes in enumerate(tokens):
            lanes = _require_sequence(
                f"topk_destinations[{owner}][{token}]", raw_lanes)
            destinations = set()
            for lane, raw_destination in enumerate(lanes):
                destination = _require_int(
                    f"topk_destinations[{owner}][{token}][{lane}]",
                    raw_destination,
                )
                if destination < 0:
                    raise ValueError("masked destinations are not supported")
                if destination >= num_destinations:
                    raise ValueError(
                        f"destination {destination} is outside "
                        f"[0, {num_destinations})"
                    )
                if destination != local_destination:
                    destinations.add(destination)
            normalized_tokens.append(tuple(sorted(destinations)))
        normalized_owners.append(tuple(normalized_tokens))
    return tuple(normalized_owners)


def map_topk_experts_to_destinations(
    topk_idx: Sequence[Sequence[Sequence[int]]],
    *,
    num_topk: int,
    num_experts: int,
    num_scaleout_ranks: int,
    local_scaleout_rank: int,
) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    """Validate force-v1 top-k input and map experts to remote servers."""

    num_topk = _require_int("num_topk", num_topk, minimum=1)
    num_experts = _require_int("num_experts", num_experts, minimum=1)
    num_scaleout_ranks = _require_int(
        "num_scaleout_ranks", num_scaleout_ranks, minimum=2)
    local_scaleout_rank = _require_int(
        "local_scaleout_rank", local_scaleout_rank, minimum=0)
    if num_topk > _MAX_TOPK:
        raise ValueError(f"num_topk exceeds the force-v1 {_MAX_TOPK} limit")
    if num_scaleout_ranks > _MAX_DESTINATIONS:
        raise ValueError(
            f"num_scaleout_ranks exceeds the force-v1 "
            f"{_MAX_DESTINATIONS} limit")
    if local_scaleout_rank >= num_scaleout_ranks:
        raise ValueError("local_scaleout_rank is outside the scaleout range")
    if num_experts % num_scaleout_ranks:
        raise ValueError("num_experts must be divisible by num_scaleout_ranks")

    owners = _require_sequence("topk_idx", topk_idx)
    if not owners:
        raise ValueError("topk_idx must contain at least one owner")
    if len(owners) > _MAX_RAILS:
        raise ValueError(f"topk_idx exceeds the force-v1 {_MAX_RAILS}-rail limit")
    if num_experts % (num_scaleout_ranks * len(owners)):
        raise ValueError(
            "num_experts must be divisible by scaleout_ranks * local_rails")

    experts_per_scaleout = num_experts // num_scaleout_ranks
    mapped_owners = []
    for owner, raw_tokens in enumerate(owners):
        tokens = _require_sequence(f"topk_idx[{owner}]", raw_tokens)
        mapped_tokens = []
        for token, raw_lanes in enumerate(tokens):
            lanes = _require_sequence(f"topk_idx[{owner}][{token}]", raw_lanes)
            if len(lanes) != num_topk:
                raise ValueError(
                    f"topk_idx[{owner}][{token}] must have exactly "
                    f"num_topk={num_topk} lanes")
            experts = []
            for lane, raw_expert in enumerate(lanes):
                expert = _require_int(
                    f"topk_idx[{owner}][{token}][{lane}]",
                    raw_expert,
                    minimum=0,
                )
                if expert >= num_experts:
                    raise ValueError(
                        f"expert {expert} is outside [0, {num_experts})")
                experts.append(expert)
            if len(set(experts)) != num_topk:
                raise ValueError("top-k expert ids must be distinct per token")
            destinations = {
                expert // experts_per_scaleout for expert in experts
            }
            destinations.discard(local_scaleout_rank)
            mapped_tokens.append(tuple(sorted(destinations)))
        mapped_owners.append(tuple(mapped_tokens))
    return tuple(mapped_owners)


def _build_quota_and_segments(
    count: Matrix,
    *,
    remainder_seed: int,
) -> Tuple[
    Matrix,
    Matrix,
    Tuple[Tuple[HybridTransferSegment, ...], ...],
    Tuple[int, ...],
]:
    num_rails = len(count)
    num_destinations = len(count[0])
    quota = [[0] * num_destinations for _ in range(num_rails)]

    for destination in range(num_destinations):
        total = sum(count[owner][destination] for owner in range(num_rails))
        base, remainder = divmod(total, num_rails)
        ring = tuple(
            (remainder_seed + destination + offset) % num_rails
            for offset in range(num_rails)
        )
        # Giving remainder slots to already-heavy owners first minimizes the
        # number of copies that must cross an intra-node link.
        preferred = tuple(
            owner for owner in ring if count[owner][destination] > base
        )
        fallback = tuple(
            owner for owner in ring if count[owner][destination] <= base
        )
        winners = set((preferred + fallback)[:remainder])
        for owner in range(num_rails):
            quota[owner][destination] = base + int(owner in winners)

    keep = [
        [min(count[owner][d], quota[owner][d]) for d in range(num_destinations)]
        for owner in range(num_rails)
    ]
    segments = []
    for destination in range(num_destinations):
        destination_segments = []
        surplus = [
            count[owner][destination] - keep[owner][destination]
            for owner in range(num_rails)
        ]
        deficit = [
            quota[egress][destination] - keep[egress][destination]
            for egress in range(num_rails)
        ]
        owner_consumed = [0] * num_rails
        incoming_cursor = [0] * num_rails
        owner = 0
        egress = 0
        while True:
            while owner < num_rails and surplus[owner] == 0:
                owner += 1
            while egress < num_rails and deficit[egress] == 0:
                egress += 1
            if owner == num_rails or egress == num_rails:
                break
            moved = min(surplus[owner], deficit[egress])
            if owner == egress:
                raise AssertionError("an owner cannot be its own deficit egress")
            destination_segments.append(HybridTransferSegment(
                owner=owner,
                egress=egress,
                owner_begin=keep[owner][destination] + owner_consumed[owner],
                count=moved,
                egress_begin=incoming_cursor[egress],
            ))
            owner_consumed[owner] += moved
            incoming_cursor[egress] += moved
            surplus[owner] -= moved
            deficit[egress] -= moved
        if any(surplus) or any(deficit):
            raise AssertionError("surplus/deficit matching did not terminate")
        segments.append(tuple(destination_segments))

    return (
        tuple(tuple(row) for row in quota),
        tuple(tuple(row) for row in keep),
        tuple(segments),
        tuple(len(bucket) for bucket in segments),
    )


def build_hybrid_rail_schedule_from_destinations(
    topk_destinations: Sequence[Sequence[Sequence[int]]],
    *,
    num_destinations: int,
    num_channels: int,
    num_max_tokens_per_rank: int,
    local_destination: Optional[int] = None,
    remainder_seed: int = 0,
    proxy_capacity_per_egress: Optional[int] = None,
) -> HybridRailSchedule:
    """Build a compact schedule from pre-mapped destination sets.

    This lower-level candidate oracle keeps ``None`` and zero capacity useful
    for mathematical tests. Public force-v1 enters through
    :func:`build_hybrid_rail_schedule`, which requires positive capacity and
    validates real top-k expert indices first.
    """

    num_destinations = _require_int(
        "num_destinations", num_destinations, minimum=1)
    num_channels = _require_int("num_channels", num_channels, minimum=1)
    if num_destinations > _MAX_DESTINATIONS:
        raise ValueError(
            f"num_destinations exceeds the force-v1 {_MAX_DESTINATIONS} limit")
    if num_channels > _MAX_CHANNELS:
        raise ValueError(
            f"num_channels exceeds the legacy Hybrid {_MAX_CHANNELS} limit")
    num_max_tokens_per_rank = _require_int(
        "num_max_tokens_per_rank", num_max_tokens_per_rank, minimum=1)
    remainder_seed = _require_nonnegative_int64(
        "remainder_seed", remainder_seed)
    if local_destination is not None:
        local_destination = _require_int(
            "local_destination", local_destination, minimum=0)
        if local_destination >= num_destinations:
            raise ValueError("local_destination is outside the destination range")
    if proxy_capacity_per_egress is not None:
        proxy_capacity_per_egress = _require_int(
            "proxy_capacity_per_egress",
            proxy_capacity_per_egress,
            minimum=0,
        )

    normalized = _normalize_topk_destinations(
        topk_destinations,
        num_destinations=num_destinations,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        local_destination=local_destination,
    )
    num_rails = len(normalized)
    if num_rails * num_destinations * num_max_tokens_per_rank > _INT32_MAX:
        raise ValueError(
            "num_rails * num_destinations * num_max_tokens_per_rank "
            "exceeds the signed int32 prefix ABI")
    remainder_seed %= num_rails
    channel_capacity = (
        num_max_tokens_per_rank + num_channels - 1) // num_channels

    channel_count = [
        [[0] * num_destinations for _ in range(num_channels)]
        for _ in range(num_rails)
    ]
    for owner, tokens in enumerate(normalized):
        for token, destinations in enumerate(tokens):
            channel = token % num_channels
            for destination in destinations:
                channel_count[owner][channel][destination] += 1

    owner_channel_prefix = [
        [[0] * num_destinations for _ in range(num_channels)]
        for _ in range(num_rails)
    ]
    count = [[0] * num_destinations for _ in range(num_rails)]
    for owner in range(num_rails):
        for destination in range(num_destinations):
            prefix = 0
            for channel in range(num_channels):
                owner_channel_prefix[owner][channel][destination] = prefix
                prefix += channel_count[owner][channel][destination]
            count[owner][destination] = prefix

    count_tuple = tuple(tuple(row) for row in count)
    quota, keep, segments, num_segments = _build_quota_and_segments(
        count_tuple, remainder_seed=remainder_seed)

    retained = [
        [[0] * num_destinations for _ in range(num_channels)]
        for _ in range(num_rails)
    ]
    moved = [
        [[0] * num_destinations for _ in range(num_channels)]
        for _ in range(num_rails)
    ]
    for egress in range(num_rails):
        for destination in range(num_destinations):
            for channel in range(num_channels):
                available = channel_count[egress][channel][destination]
                prefix = owner_channel_prefix[egress][channel][destination]
                retained[egress][channel][destination] = min(
                    available,
                    max(keep[egress][destination] - prefix, 0),
                )

            incoming_remaining = (
                quota[egress][destination] - keep[egress][destination])
            for channel in range(num_channels):
                spare = (
                    channel_capacity
                    - retained[egress][channel][destination]
                )
                take = min(spare, incoming_remaining)
                moved[egress][channel][destination] = take
                incoming_remaining -= take
            if incoming_remaining:
                raise AssertionError("channel capacity proof was violated")

    moved_channel_prefix = [
        [[0] * (num_channels + 1) for _ in range(num_destinations)]
        for _ in range(num_rails)
    ]
    group_prefix = [
        [[0] * num_destinations for _ in range(num_channels)]
        for _ in range(num_rails)
    ]
    proxy_required = [0] * num_rails
    for egress in range(num_rails):
        for destination in range(num_destinations):
            for channel in range(num_channels):
                moved_channel_prefix[egress][destination][channel + 1] = (
                    moved_channel_prefix[egress][destination][channel]
                    + moved[egress][channel][destination]
                )
        cursor = 0
        for channel in range(num_channels):
            for destination in range(num_destinations):
                group_prefix[egress][channel][destination] = cursor
                cursor += moved[egress][channel][destination]
        proxy_required[egress] = cursor

    for egress, required in enumerate(proxy_required):
        _require_int(f"proxy_required[{egress}]", required, minimum=0)
    moved_copies = _require_int(
        "moved_copies", sum(proxy_required), minimum=0)

    capacity_exceeded = (
        proxy_capacity_per_egress is not None
        and any(
            required > proxy_capacity_per_egress
            for required in proxy_required
        )
    )
    schedule = HybridRailSchedule(
        num_rails=num_rails,
        num_channels=num_channels,
        num_destinations=num_destinations,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        channel_capacity=channel_capacity,
        num_tokens_per_owner=tuple(len(tokens) for tokens in normalized),
        local_destination=local_destination,
        remainder_seed=remainder_seed,
        channel_count=tuple(
            tuple(tuple(row) for row in owner) for owner in channel_count),
        owner_channel_prefix=tuple(
            tuple(tuple(row) for row in owner)
            for owner in owner_channel_prefix),
        count=count_tuple,
        quota=quota,
        keep_count=keep,
        segments=segments,
        num_segments=num_segments,
        retained=tuple(
            tuple(tuple(row) for row in egress) for egress in retained),
        moved=tuple(
            tuple(tuple(row) for row in egress) for egress in moved),
        moved_channel_prefix=tuple(
            tuple(tuple(row) for row in egress)
            for egress in moved_channel_prefix),
        group_prefix=tuple(
            tuple(tuple(row) for row in egress) for egress in group_prefix),
        proxy_required=tuple(proxy_required),
        moved_copies=moved_copies,
        proxy_capacity_per_egress=proxy_capacity_per_egress,
        enabled=not capacity_exceeded,
        failure_reason=_CAPACITY_FAILURE if capacity_exceeded else None,
    )
    validate_hybrid_rail_schedule(schedule)
    return schedule


def build_hybrid_rail_schedule(
    topk_idx: Sequence[Sequence[Sequence[int]]],
    *,
    num_topk: int,
    num_experts: int,
    num_scaleout_ranks: int,
    local_scaleout_rank: int,
    num_channels: int,
    num_max_tokens_per_rank: int,
    remainder_seed: int = 0,
    proxy_capacity_per_egress: int,
) -> HybridRailSchedule:
    """Build the strict force-v1 schedule from real top-k expert indices."""

    proxy_capacity_per_egress = _require_int(
        "proxy_capacity_per_egress",
        proxy_capacity_per_egress,
        minimum=1,
    )
    destinations = map_topk_experts_to_destinations(
        topk_idx,
        num_topk=num_topk,
        num_experts=num_experts,
        num_scaleout_ranks=num_scaleout_ranks,
        local_scaleout_rank=local_scaleout_rank,
    )
    return build_hybrid_rail_schedule_from_destinations(
        destinations,
        num_destinations=num_scaleout_ranks,
        num_channels=num_channels,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        local_destination=local_scaleout_rank,
        remainder_seed=remainder_seed,
        proxy_capacity_per_egress=proxy_capacity_per_egress,
    )


def _validate_matrix(
    name: str,
    matrix: Matrix,
    rows: int,
    columns: int,
) -> None:
    if len(matrix) != rows:
        raise ValueError(f"{name} must have {rows} rows")
    for row, values in enumerate(matrix):
        if len(values) != columns:
            raise ValueError(f"{name}[{row}] must have {columns} columns")
        for column, value in enumerate(values):
            _require_int(f"{name}[{row}][{column}]", value, minimum=0)


def _validate_tensor3(
    name: str,
    tensor: Tensor3,
    outer: int,
    middle: int,
    inner: int,
) -> None:
    if len(tensor) != outer:
        raise ValueError(f"{name} must have outer dimension {outer}")
    for first, matrix in enumerate(tensor):
        if len(matrix) != middle:
            raise ValueError(
                f"{name}[{first}] must have middle dimension {middle}")
        for second, row in enumerate(matrix):
            if len(row) != inner:
                raise ValueError(
                    f"{name}[{first}][{second}] must have "
                    f"inner dimension {inner}"
                )
            for third, value in enumerate(row):
                _require_int(
                    f"{name}[{first}][{second}][{third}]",
                    value,
                    minimum=0,
                )


def validate_hybrid_rail_schedule(schedule: HybridRailSchedule) -> None:
    """Validate every algebraic invariant of a compact C080 schedule."""

    if not isinstance(schedule, HybridRailSchedule):
        raise ValueError("schedule must be a HybridRailSchedule")
    g = _require_int("num_rails", schedule.num_rails, minimum=1)
    c = _require_int("num_channels", schedule.num_channels, minimum=1)
    d = _require_int("num_destinations", schedule.num_destinations, minimum=1)
    if g > _MAX_RAILS:
        raise ValueError(f"num_rails exceeds the force-v1 {_MAX_RAILS} limit")
    if c > _MAX_CHANNELS:
        raise ValueError(
            f"num_channels exceeds the legacy Hybrid {_MAX_CHANNELS} limit")
    if d > _MAX_DESTINATIONS:
        raise ValueError(
            f"num_destinations exceeds the force-v1 {_MAX_DESTINATIONS} limit")
    t = _require_int(
        "num_max_tokens_per_rank",
        schedule.num_max_tokens_per_rank,
        minimum=1,
    )
    if g * d * t > _INT32_MAX:
        raise ValueError(
            "num_rails * num_destinations * num_max_tokens_per_rank "
            "exceeds the signed int32 prefix ABI")
    remainder_seed = _require_int(
        "remainder_seed", schedule.remainder_seed, minimum=0)
    if remainder_seed >= g:
        raise ValueError("remainder_seed must be normalized to the rail range")
    if schedule.local_destination is not None:
        local_destination = _require_int(
            "local_destination", schedule.local_destination, minimum=0)
        if local_destination >= d:
            raise ValueError("local_destination is outside the destination range")
    expected_capacity = (t + c - 1) // c
    if schedule.channel_capacity != expected_capacity:
        raise ValueError("channel_capacity is not ceil(T / C)")
    if len(schedule.num_tokens_per_owner) != g:
        raise ValueError("num_tokens_per_owner has the wrong length")
    for owner, tokens in enumerate(schedule.num_tokens_per_owner):
        _require_int(f"num_tokens_per_owner[{owner}]", tokens, minimum=0)
        if tokens > t:
            raise ValueError("num_tokens_per_owner exceeds T")

    _validate_tensor3("channel_count", schedule.channel_count, g, c, d)
    _validate_tensor3(
        "owner_channel_prefix", schedule.owner_channel_prefix, g, c, d)
    _validate_matrix("count", schedule.count, g, d)
    _validate_matrix("quota", schedule.quota, g, d)
    _validate_matrix("keep_count", schedule.keep_count, g, d)
    _validate_tensor3("retained", schedule.retained, g, c, d)
    _validate_tensor3("moved", schedule.moved, g, c, d)
    _validate_tensor3(
        "moved_channel_prefix", schedule.moved_channel_prefix, g, d, c + 1)
    _validate_tensor3("group_prefix", schedule.group_prefix, g, c, d)

    for owner in range(g):
        for destination in range(d):
            prefix = 0
            for channel in range(c):
                if schedule.channel_count[owner][channel][destination] > expected_capacity:
                    raise ValueError("channel_count exceeds channel capacity")
                if schedule.owner_channel_prefix[owner][channel][destination] != prefix:
                    raise ValueError("owner_channel_prefix is not exclusive")
                prefix += schedule.channel_count[owner][channel][destination]
            if schedule.count[owner][destination] != prefix:
                raise ValueError("count does not equal the channel sum")
            if prefix > t:
                raise ValueError("deduplicated owner/destination count exceeds T")
            if (
                schedule.local_destination is not None
                and destination == schedule.local_destination
                and prefix != 0
            ):
                raise ValueError("local destination copies must be excluded")

    expected_quota, expected_keep, expected_segments, expected_num_segments = (
        _build_quota_and_segments(
            schedule.count,
            remainder_seed=schedule.remainder_seed,
        )
    )
    if schedule.quota != expected_quota:
        raise ValueError("quota is not the deterministic minimum-move quota")
    if schedule.keep_count != expected_keep:
        raise ValueError("keep_count is not min(count, quota)")
    if schedule.segments != expected_segments:
        raise ValueError("segments do not match canonical surplus/deficit matching")
    if schedule.num_segments != expected_num_segments:
        raise ValueError("num_segments does not match segments")
    if any(value > g - 1 for value in schedule.num_segments):
        raise ValueError("a destination has more than G - 1 segments")

    expected_proxy = [0] * g
    for egress in range(g):
        for destination in range(d):
            incoming = (
                schedule.quota[egress][destination]
                - schedule.keep_count[egress][destination]
            )
            remaining = incoming
            moved_prefix = 0
            if schedule.moved_channel_prefix[egress][destination][0] != 0:
                raise ValueError("moved_channel_prefix must begin at zero")
            for channel in range(c):
                available = schedule.channel_count[egress][channel][destination]
                owner_prefix = (
                    schedule.owner_channel_prefix[egress][channel][destination])
                expected_retained = min(
                    available,
                    max(
                        schedule.keep_count[egress][destination] - owner_prefix,
                        0,
                    ),
                )
                if schedule.retained[egress][channel][destination] != expected_retained:
                    raise ValueError("retained does not match the owner prefix")
                spare = expected_capacity - expected_retained
                expected_moved = min(spare, remaining)
                if schedule.moved[egress][channel][destination] != expected_moved:
                    raise ValueError("moved is not the canonical channel fill")
                remaining -= expected_moved
                moved_prefix += expected_moved
                if (
                    schedule.moved_channel_prefix[egress][destination][channel + 1]
                    != moved_prefix
                ):
                    raise ValueError("moved_channel_prefix is not exclusive")
            if remaining:
                raise ValueError("incoming copies were not fully assigned")

        cursor = 0
        for channel in range(c):
            for destination in range(d):
                if schedule.group_prefix[egress][channel][destination] != cursor:
                    raise ValueError(
                        "group_prefix is not channel-major/destination-minor")
                cursor += schedule.moved[egress][channel][destination]
        expected_proxy[egress] = cursor

    if len(schedule.proxy_required) != g:
        raise ValueError("proxy_required has the wrong length")
    for egress, required in enumerate(schedule.proxy_required):
        _require_int(f"proxy_required[{egress}]", required, minimum=0)
    _require_int("moved_copies", schedule.moved_copies, minimum=0)
    if schedule.proxy_required != tuple(expected_proxy):
        raise ValueError("proxy_required does not equal the moved group sum")
    if schedule.moved_copies != sum(expected_proxy):
        raise ValueError("moved_copies does not equal proxy_required sum")
    if schedule.moved_copies != sum(
        segment.count
        for destination_segments in schedule.segments
        for segment in destination_segments
    ):
        raise ValueError("segment and channel moved totals disagree")

    capacity = schedule.proxy_capacity_per_egress
    if capacity is not None:
        _require_int("proxy_capacity_per_egress", capacity, minimum=0)
    exceeded = (
        capacity is not None
        and any(required > capacity for required in expected_proxy)
    )
    if exceeded:
        if schedule.enabled or schedule.failure_reason != _CAPACITY_FAILURE:
            raise ValueError("capacity overflow must fail closed")
    elif not schedule.enabled or schedule.failure_reason is not None:
        raise ValueError("a feasible schedule must be enabled without a reason")


def resolve_hybrid_copy(
    schedule: HybridRailSchedule,
    *,
    owner: int,
    source_channel: int,
    destination: int,
    channel_local_ordinal: int,
) -> HybridCopyResolution:
    """Resolve one copy without consulting a materialized copy assignment."""

    if not schedule.enabled:
        raise RuntimeError(
            f"cannot resolve a disabled schedule: {schedule.failure_reason}")
    owner = _require_int("owner", owner, minimum=0)
    source_channel = _require_int("source_channel", source_channel, minimum=0)
    destination = _require_int("destination", destination, minimum=0)
    channel_local_ordinal = _require_int(
        "channel_local_ordinal", channel_local_ordinal, minimum=0)
    if owner >= schedule.num_rails:
        raise ValueError("owner is outside the rail range")
    if source_channel >= schedule.num_channels:
        raise ValueError("source_channel is outside the channel range")
    if destination >= schedule.num_destinations:
        raise ValueError("destination is outside the destination range")
    source_count = schedule.channel_count[owner][source_channel][destination]
    if channel_local_ordinal >= source_count:
        raise ValueError("channel_local_ordinal is outside the source group")

    owner_ordinal = (
        schedule.owner_channel_prefix[owner][source_channel][destination]
        + channel_local_ordinal
    )
    if owner_ordinal < schedule.keep_count[owner][destination]:
        return HybridCopyResolution(
            owner=owner,
            source_channel=source_channel,
            destination=destination,
            channel_local_ordinal=channel_local_ordinal,
            owner_ordinal=owner_ordinal,
            moved=False,
            egress=owner,
            channel=source_channel,
            remote_slot=channel_local_ordinal,
            proxy_slot=-1,
            incoming_ordinal=-1,
        )

    matching = [
        segment
        for segment in schedule.segments[destination]
        if segment.owner == owner
        and segment.owner_begin <= owner_ordinal
        < segment.owner_begin + segment.count
    ]
    if len(matching) != 1:
        raise ValueError("moved copy does not resolve to exactly one segment")
    segment = matching[0]
    egress = segment.egress
    incoming_ordinal = (
        segment.egress_begin + owner_ordinal - segment.owner_begin)
    prefix = schedule.moved_channel_prefix[egress][destination]
    target_channels = [
        channel
        for channel in range(schedule.num_channels)
        if prefix[channel] <= incoming_ordinal < prefix[channel + 1]
    ]
    if len(target_channels) != 1:
        raise ValueError("incoming copy does not resolve to exactly one channel")
    channel = target_channels[0]
    channel_ordinal = incoming_ordinal - prefix[channel]
    return HybridCopyResolution(
        owner=owner,
        source_channel=source_channel,
        destination=destination,
        channel_local_ordinal=channel_local_ordinal,
        owner_ordinal=owner_ordinal,
        moved=True,
        egress=egress,
        channel=channel,
        remote_slot=(
            schedule.retained[egress][channel][destination]
            + channel_ordinal
        ),
        proxy_slot=(
            schedule.group_prefix[egress][channel][destination]
            + channel_ordinal
        ),
        incoming_ordinal=incoming_ordinal,
    )


def enumerate_destination_copies(
    topk_destinations: Sequence[Sequence[Sequence[int]]],
    schedule: HybridRailSchedule,
) -> Tuple[HybridDestinationCopy, ...]:
    """Enumerate channel-major logical copies for tests and diagnostics only."""

    normalized = _normalize_topk_destinations(
        topk_destinations,
        num_destinations=schedule.num_destinations,
        num_max_tokens_per_rank=schedule.num_max_tokens_per_rank,
        local_destination=schedule.local_destination,
    )
    if len(normalized) != schedule.num_rails:
        raise ValueError("input owner count does not match the schedule")
    if tuple(len(tokens) for tokens in normalized) != schedule.num_tokens_per_owner:
        raise ValueError("input token counts do not match the schedule")

    observed_count = [
        [[0] * schedule.num_destinations for _ in range(schedule.num_channels)]
        for _ in range(schedule.num_rails)
    ]
    copies = []
    for owner, tokens in enumerate(normalized):
        for channel in range(schedule.num_channels):
            for token in range(channel, len(tokens), schedule.num_channels):
                for destination in tokens[token]:
                    local_ordinal = observed_count[owner][channel][destination]
                    copies.append(HybridDestinationCopy(
                        owner=owner,
                        token=token,
                        destination=destination,
                        source_channel=channel,
                        channel_local_ordinal=local_ordinal,
                        owner_ordinal=(
                            schedule.owner_channel_prefix[owner][channel][destination]
                            + local_ordinal
                        ),
                    ))
                    observed_count[owner][channel][destination] += 1
    observed_tuple = tuple(
        tuple(tuple(row) for row in owner) for owner in observed_count)
    if observed_tuple != schedule.channel_count:
        raise ValueError("input destinations do not match schedule.channel_count")
    return tuple(copies)


def enumerate_resolved_copies(
    topk_destinations: Sequence[Sequence[Sequence[int]]],
    schedule: HybridRailSchedule,
) -> Tuple[ResolvedHybridDestinationCopy, ...]:
    """Materialize resolutions for exhaustive oracle assertions only."""

    return tuple(
        ResolvedHybridDestinationCopy(
            copy=copy,
            resolution=resolve_hybrid_copy(
                schedule,
                owner=copy.owner,
                source_channel=copy.source_channel,
                destination=copy.destination,
                channel_local_ordinal=copy.channel_local_ordinal,
            ),
        )
        for copy in enumerate_destination_copies(topk_destinations, schedule)
    )
