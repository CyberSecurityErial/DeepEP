"""CPU-only contracts shared by the C105 rail-balance validation tests."""

import json
from typing import Any, Dict, List


RESULT_SCHEMA_VERSION = 1
CANONICAL_CASES = ("balanced", "two_hot", "one_hot")
EVIDENCE_LABEL = "REAL_HYBRID_RUNTIME_UNTESTED"
_MAX_FORCE_V1_DIM = 32
_MAX_FORCE_V1_EXPERTS = 2048
_MAX_FORCE_V1_EXPERTS_PER_RANK = 256
_MAX_INT32 = (1 << 31) - 1


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
