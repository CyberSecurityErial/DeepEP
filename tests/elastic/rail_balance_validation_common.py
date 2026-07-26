"""CPU-only contracts shared by the C105 rail-balance validation tests."""

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rail_balance_hybrid_reference import build_hybrid_rail_schedule


RESULT_SCHEMA_VERSION = 1
CANONICAL_CASES = ("balanced", "two_hot", "one_hot", "capacity")
VALIDATION_MODES = ("off", "force")
EVIDENCE_LABEL = "REAL_HYBRID_RUNTIME_UNTESTED"
_MAX_FORCE_V1_DIM = 32
_MAX_FORCE_V1_EXPERTS = 2048
_MAX_FORCE_V1_EXPERTS_PER_RANK = 256
_MAX_FORCE_V1_CHANNELS = 1024
_MAX_INT32 = (1 << 31) - 1
_PAYLOAD_DTYPE_BYTES = {"bf16": 2}


def _require_exact_int(name: str, value: Any, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("%s must be an exact int in [%d, %d]" % (name, minimum, maximum))


def _validate_topology(
    *,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
) -> None:
    _require_exact_int("num_scaleout_ranks", num_scaleout_ranks, 2, _MAX_FORCE_V1_DIM)
    _require_exact_int("num_scaleup_ranks", num_scaleup_ranks, 2, _MAX_FORCE_V1_DIM)
    _require_exact_int("num_tokens_per_rank", num_tokens_per_rank, 0, _MAX_INT32)
    _require_exact_int("num_topk", num_topk, 1, _MAX_FORCE_V1_DIM)
    _require_exact_int(
        "num_experts", num_experts, 1, _MAX_FORCE_V1_EXPERTS)

    world_size = num_scaleout_ranks * num_scaleup_ranks
    if world_size * num_tokens_per_rank > _MAX_INT32:
        raise ValueError("world_size * num_tokens_per_rank exceeds int32")
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    experts_per_rank = num_experts // world_size
    if experts_per_rank > _MAX_FORCE_V1_EXPERTS_PER_RANK:
        raise ValueError("num_experts / world_size exceeds 256")
    if (world_size * num_tokens_per_rank
            * min(num_topk, experts_per_rank) > _MAX_INT32):
        raise ValueError("flattened token contribution count exceeds int32")
    if num_topk > num_experts // num_scaleout_ranks:
        raise ValueError("num_topk exceeds the experts available on one server")


def build_deterministic_topk(
    case: str,
    *,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
) -> List[List[List[int]]]:
    """Build deterministic ``[world_size][tokens][topk]`` expert indices.

    Global ranks and experts are grouped as ``server * G + local_rank``.  A
    remote token's experts all belong to the same destination server, so the
    generated routes also exercise server-level payload deduplication.
    """
    if type(case) is not str or case not in CANONICAL_CASES:
        raise ValueError("unknown C105 route case: %s" % case)
    _validate_topology(
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_tokens_per_rank=num_tokens_per_rank,
        num_topk=num_topk,
        num_experts=num_experts,
    )
    if case == "two_hot" and num_scaleup_ranks < 3:
        raise ValueError("two_hot requires at least three scaleup ranks")

    world_size = num_scaleout_ranks * num_scaleup_ranks
    experts_per_server = num_experts // num_scaleout_ranks
    active_rails = {
        "balanced": num_scaleup_ranks,
        "two_hot": 2,
        "one_hot": 1,
        "capacity": 1,
    }[case]

    routes: List[List[List[int]]] = []
    for rank in range(world_size):
        source_server = rank // num_scaleup_ranks
        local_rank = rank % num_scaleup_ranks
        target_server = (
            (source_server + 1) % num_scaleout_ranks
            if local_rank < active_rails
            else source_server
        )
        expert_base = target_server * experts_per_server

        rank_routes: List[List[int]] = []
        for token_idx in range(num_tokens_per_rank):
            first_expert = (rank * num_tokens_per_rank + token_idx * num_topk) % experts_per_server
            rank_routes.append(
                [
                    expert_base + (first_expert + topk_idx) % experts_per_server
                    for topk_idx in range(num_topk)
                ]
            )
        routes.append(rank_routes)
    return routes


def count_distinct_remote_destinations(
    topk_idx: List[List[List[int]]],
    *,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_experts: int,
) -> List[List[List[int]]]:
    """Count unique remote destination servers for every source rail."""
    _require_exact_int("num_scaleout_ranks", num_scaleout_ranks, 2, _MAX_FORCE_V1_DIM)
    _require_exact_int("num_scaleup_ranks", num_scaleup_ranks, 2, _MAX_FORCE_V1_DIM)
    _require_exact_int(
        "num_experts", num_experts, 1, _MAX_FORCE_V1_EXPERTS)
    world_size = num_scaleout_ranks * num_scaleup_ranks
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    experts_per_rank = num_experts // world_size
    if experts_per_rank > _MAX_FORCE_V1_EXPERTS_PER_RANK:
        raise ValueError("num_experts / world_size exceeds 256")
    if type(topk_idx) is not list:
        raise ValueError("topk_idx must be a list")
    if len(topk_idx) != world_size:
        raise ValueError("topk_idx rank dimension does not match world_size")

    experts_per_server = num_experts // num_scaleout_ranks
    counts = [
        [[0 for _ in range(num_scaleout_ranks)] for _ in range(num_scaleup_ranks)]
        for _ in range(num_scaleout_ranks)
    ]
    expected_num_tokens = None
    expected_num_topk = None
    for rank, rank_routes in enumerate(topk_idx):
        if type(rank_routes) is not list:
            raise ValueError("every rank route must be a list")
        if expected_num_tokens is None:
            expected_num_tokens = len(rank_routes)
            if world_size * expected_num_tokens > _MAX_INT32:
                raise ValueError("world_size * num_tokens_per_rank exceeds int32")
        elif len(rank_routes) != expected_num_tokens:
            raise ValueError("all ranks must have the same token count")

        source_server = rank // num_scaleup_ranks
        local_rank = rank % num_scaleup_ranks
        for token_route in rank_routes:
            if type(token_route) is not list:
                raise ValueError("every token route must be a list")
            if expected_num_topk is None:
                _require_exact_int("inferred num_topk", len(token_route), 1, _MAX_FORCE_V1_DIM)
                expected_num_topk = len(token_route)
                if (world_size * expected_num_tokens
                        * min(expected_num_topk, experts_per_rank)
                        > _MAX_INT32):
                    raise ValueError(
                        "flattened token contribution count exceeds int32")
            elif len(token_route) != expected_num_topk:
                raise ValueError("all token routes must have the same top-k")

            remote_servers = set()
            seen_experts = set()
            for expert_idx in token_route:
                if type(expert_idx) is not int or not 0 <= expert_idx < num_experts:
                    raise ValueError("expert index is outside [0, num_experts)")
                if expert_idx in seen_experts:
                    raise ValueError("a token route must not contain duplicate experts")
                seen_experts.add(expert_idx)
                destination_server = expert_idx // experts_per_server
                if destination_server != source_server:
                    remote_servers.add(destination_server)
            for destination_server in remote_servers:
                counts[source_server][local_rank][destination_server] += 1
    return counts


def _reject_non_string_json_keys(name: str, value: Any, active_ids: set) -> None:
    if not isinstance(value, (dict, list, tuple)):
        return
    value_id = id(value)
    if value_id in active_ids:
        raise ValueError("%s must not contain recursive values" % name)
    active_ids.add(value_id)
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError("%s JSON object keys must be str" % name)
                _reject_non_string_json_keys(name, child, active_ids)
        else:
            for child in value:
                _reject_non_string_json_keys(name, child, active_ids)
    finally:
        active_ids.remove(value_id)


def _json_roundtrip_dict(name: str, value: Any) -> Dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("%s must be a dict" % name)
    _reject_non_string_json_keys(name, value, set())
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError("%s must contain only finite JSON values" % name) from error


def new_result_record(
    *,
    run_id: str,
    case: str,
    mode: str,
    topology: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the stable, JSON-serializable C105 result skeleton."""
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a non-empty str")
    if type(case) is not str or case not in CANONICAL_CASES:
        raise ValueError("case must be a canonical C105 case")
    if type(mode) is not str or mode not in ("off", "force"):
        raise ValueError("mode must be 'off' or 'force'")

    topology_copy = _json_roundtrip_dict("topology", topology)
    config_copy = _json_roundtrip_dict("config", config)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "evidence_label": EVIDENCE_LABEL,
        "case": case,
        "mode": mode,
        "topology": topology_copy,
        "config": config_copy,
        "correctness": {
            "dispatch_matches_reference": None,
            "combine_matches_reference": None,
            "off_force_equal": None,
            "output_digest": None,
        },
        "plan": {
            "source": "not_available",
            "count": None,
            "quota": None,
            "original_rail_bytes": None,
            "balanced_rail_bytes": None,
            "moved_copies": None,
            "moved_bytes": None,
            "proxy_required": None,
            "group_count": None,
            "num_segments": None,
        },
        "traffic": {
            "source": "not_available",
            "scope": "payload_only",
            "expected_gin_puts": None,
            "expected_gin_bytes": None,
        },
        "runtime": {
            "completed": False,
            "availability": "not_instrumented",
            "wait_cycles": None,
            "qp_utilization": None,
            "nic_bytes": None,
        },
        "error": None,
        "claim_scope": "validation_only",
    }


def _require_modes(modes: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(modes, (str, bytes)) or not isinstance(modes, Sequence):
        raise ValueError("modes must be a sequence")
    normalized = tuple(modes)
    if not normalized:
        raise ValueError("modes must not be empty")
    for mode in normalized:
        if type(mode) is not str or mode not in VALIDATION_MODES:
            raise ValueError("every mode must be 'off' or 'force'")
    if len(set(normalized)) != len(normalized):
        raise ValueError("modes must not contain duplicates")
    return normalized


def _payload_bytes_per_token(hidden: int, payload_dtype: str) -> int:
    _require_exact_int("hidden", hidden, 1, _MAX_INT32)
    if hidden % 256 != 0:
        raise ValueError("hidden must be a positive multiple of 256")
    if type(payload_dtype) is not str or payload_dtype not in _PAYLOAD_DTYPE_BYTES:
        raise ValueError("payload_dtype must be exactly 'bf16'")
    bytes_per_element = _PAYLOAD_DTYPE_BYTES[payload_dtype]
    if hidden > _MAX_INT32 // bytes_per_element:
        raise ValueError("hidden payload bytes exceed int32")
    return hidden * bytes_per_element


def _total_remote_copies(counts: List[List[List[int]]]) -> int:
    return sum(
        count
        for source_server in counts
        for source_rail in source_server
        for count in source_rail
    )


def _scale_matrix_bytes(matrix: Sequence[Sequence[int]],
                        bytes_per_token: int) -> List[List[int]]:
    return [
        [int(value) * bytes_per_token for value in row]
        for row in matrix
    ]


def _source_schedule(
    topk_idx: List[List[List[int]]],
    *,
    source_server: int,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
    num_channels: int,
    proxy_slots_per_rank: int,
):
    start = source_server * num_scaleup_ranks
    stop = start + num_scaleup_ranks
    return build_hybrid_rail_schedule(
        topk_idx[start:stop],
        num_topk=num_topk,
        num_experts=num_experts,
        num_scaleout_ranks=num_scaleout_ranks,
        local_scaleout_rank=source_server,
        num_channels=num_channels,
        num_max_tokens_per_rank=num_tokens_per_rank,
        proxy_capacity_per_egress=proxy_slots_per_rank,
    )


def _build_schedules(
    topk_idx: List[List[List[int]]],
    *,
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
    num_channels: int,
    proxy_slots_per_rank: int,
):
    return [
        _source_schedule(
            topk_idx,
            source_server=source_server,
            num_scaleout_ranks=num_scaleout_ranks,
            num_scaleup_ranks=num_scaleup_ranks,
            num_tokens_per_rank=num_tokens_per_rank,
            num_topk=num_topk,
            num_experts=num_experts,
            num_channels=num_channels,
            proxy_slots_per_rank=proxy_slots_per_rank,
        )
        for source_server in range(num_scaleout_ranks)
    ]


def build_validation_bundle(
    *,
    run_id: str,
    case: str,
    modes: Sequence[str],
    num_scaleout_ranks: int,
    num_scaleup_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts: int,
    hidden: int,
    num_channels: int,
    payload_dtype: str = "bf16",
    proxy_slots_per_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic C105 JSON bundle without claiming runtime success."""
    modes_tuple = _require_modes(modes)
    _validate_topology(
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_tokens_per_rank=num_tokens_per_rank,
        num_topk=num_topk,
        num_experts=num_experts,
    )
    _require_exact_int("num_channels", num_channels, 1, _MAX_FORCE_V1_CHANNELS)
    if num_tokens_per_rank == 0:
        raise ValueError("C105 validation bundles require nonzero token capacity")
    bytes_per_token = _payload_bytes_per_token(hidden, payload_dtype)

    topk_idx = build_deterministic_topk(
        case,
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_tokens_per_rank=num_tokens_per_rank,
        num_topk=num_topk,
        num_experts=num_experts,
    )
    counts = count_distinct_remote_destinations(
        topk_idx,
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_experts=num_experts,
    )

    safe_capacity = max(1, num_scaleout_ranks * num_tokens_per_rank)
    required_schedules = _build_schedules(
        topk_idx,
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_tokens_per_rank=num_tokens_per_rank,
        num_topk=num_topk,
        num_experts=num_experts,
        num_channels=num_channels,
        proxy_slots_per_rank=safe_capacity,
    )
    max_required = max(
        max(schedule.proxy_required)
        for schedule in required_schedules
    )
    if case == "capacity":
        if proxy_slots_per_rank is None:
            proxy_slots_per_rank = max_required - 1
        if max_required <= 1 or proxy_slots_per_rank >= max_required:
            raise ValueError("capacity case must exceed proxy_slots_per_rank")
    elif proxy_slots_per_rank is None:
        proxy_slots_per_rank = max(1, max_required)
    _require_exact_int(
        "proxy_slots_per_rank", proxy_slots_per_rank, 1, _MAX_INT32)

    schedules = _build_schedules(
        topk_idx,
        num_scaleout_ranks=num_scaleout_ranks,
        num_scaleup_ranks=num_scaleup_ranks,
        num_tokens_per_rank=num_tokens_per_rank,
        num_topk=num_topk,
        num_experts=num_experts,
        num_channels=num_channels,
        proxy_slots_per_rank=proxy_slots_per_rank,
    )
    capacity_failed = any(not schedule.enabled for schedule in schedules)
    if case == "capacity" and not capacity_failed:
        raise ValueError("capacity case did not fail closed")
    if case != "capacity" and capacity_failed:
        raise ValueError("non-capacity case exceeded proxy capacity")

    topology = {
        "world_size": num_scaleout_ranks * num_scaleup_ranks,
        "num_scaleout_ranks": num_scaleout_ranks,
        "num_scaleup_ranks": num_scaleup_ranks,
    }
    config = {
        "num_tokens_per_rank": num_tokens_per_rank,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "hidden": hidden,
        "payload_dtype": payload_dtype,
        "num_channels": num_channels,
        "proxy_slots_per_rank": proxy_slots_per_rank,
    }

    total_remote_copies = _total_remote_copies(counts)
    moved_copies = sum(schedule.moved_copies for schedule in schedules)
    plan = {
        "source": "cpu_oracle",
        "count": [schedule.count for schedule in schedules],
        "quota": [schedule.quota for schedule in schedules],
        "original_rail_bytes": [
            _scale_matrix_bytes(schedule.count, bytes_per_token)
            for schedule in schedules
        ],
        "balanced_rail_bytes": [
            _scale_matrix_bytes(schedule.quota, bytes_per_token)
            for schedule in schedules
        ],
        "moved_copies": moved_copies,
        "moved_bytes": moved_copies * bytes_per_token,
        "proxy_required": [schedule.proxy_required for schedule in schedules],
        "group_count": [schedule.moved for schedule in schedules],
        "num_segments": [schedule.num_segments for schedule in schedules],
    }
    traffic = {
        "source": "cpu_oracle_expected",
        "scope": "payload_only",
        "expected_gin_puts": total_remote_copies,
        "expected_gin_bytes": total_remote_copies * bytes_per_token,
    }
    records = {}
    for mode in modes_tuple:
        record = new_result_record(
            run_id=run_id,
            case=case,
            mode=mode,
            topology=topology,
            config=config,
        )
        record["plan"] = plan
        record["traffic"] = traffic
        records[mode] = record

    bundle = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "evidence_label": EVIDENCE_LABEL,
        "case": case,
        "modes": modes_tuple,
        "topology": topology,
        "config": config,
        "remote_counts": counts,
        "total_remote_copies": total_remote_copies,
        "max_proxy_required": max_required,
        "capacity_failed": capacity_failed,
        "records": records,
    }
    return _json_roundtrip_dict("validation_bundle", bundle)
