#!/usr/bin/env python3
"""Run one truthful D>1 Hybrid Rail off/force correctness case.

Launch this script once per node. ``WORLD_SIZE`` is the node count and
``RANK`` is the node index, matching ``deep_ep.utils.envs.init_dist``.
The production force capability is enabled only inside spawned workers and is
restored before exit. Physical wait/QP/NIC counters remain unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import traceback
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from rail_balance_validation_common import (
    CANONICAL_CASES,
    build_deterministic_topk,
    build_validation_bundle,
)


_RUNTIME_LABEL = "REAL_HYBRID_D_GT_1_CORRECTNESS_VALIDATED_COUNTERS_UNAVAILABLE"
_CAPACITY_LABEL = "REAL_HYBRID_D_GT_1_FAIL_CLOSED_VALIDATED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_launch_environment() -> None:
    required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"missing required launch environment: {', '.join(missing)}")
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        node_rank = int(os.environ["RANK"])
        master_port = int(os.environ["MASTER_PORT"])
    except ValueError as error:
        raise ValueError(
            "WORLD_SIZE, RANK, and MASTER_PORT must be integers"
        ) from error
    if world_size < 2:
        raise ValueError(
            "WORLD_SIZE must be the truthful node count and greater than one"
        )
    if not 0 <= node_rank < world_size:
        raise ValueError("RANK must be in [0, WORLD_SIZE)")
    if not 1 <= master_port <= 65535:
        raise ValueError("MASTER_PORT must be in [1, 65535]")
    master_addr = os.environ["MASTER_ADDR"].strip().lower()
    if master_addr in {
        "localhost",
        "::1",
        "[::1]",
        "0.0.0.0",
    } or master_addr.startswith("127."):
        raise ValueError("MASTER_ADDR must be reachable from every node, not loopback")
    if os.environ.get("EP_DISABLE_GIN") not in (None, "0"):
        raise ValueError("EP_DISABLE_GIN must not disable the real Gin path")


def _phase(
    label: str,
    group: dist.ProcessGroup,
    timeout: int,
    operation: Callable[[], Any],
) -> Any:
    """Run one rank-local phase and report every failure before proceeding."""
    result, error = None, None
    try:
        result = operation()
    except BaseException:
        error = traceback.format_exc()
    gathered: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, error, group=group)
    failures = [
        f"rank {rank}:\n{item}"
        for rank, item in enumerate(gathered)
        if item is not None
    ]
    if failures:
        raise AssertionError(f"{label} failed:\n" + "\n".join(failures))
    dist.monitored_barrier(
        group=group,
        timeout=timedelta(seconds=timeout),
        wait_all_ranks=True,
    )
    return result


def _digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        cpu = tensor.detach().contiguous().cpu()
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _global_digest(
    local_digest: str,
    group: dist.ProcessGroup,
) -> tuple[str, list[str]]:
    rank_digests: list[str | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(rank_digests, local_digest, group=group)
    values = [value for value in rank_digests if value is not None]
    _require(len(values) == len(rank_digests), "missing rank digest")
    joined = "\n".join(values).encode()
    return hashlib.sha256(joined).hexdigest(), values


def _consensus_int(value: int, group: dist.ProcessGroup) -> int:
    values: list[int | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    _require(all(item == value for item in values), f"rank values differ: {values}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_identity(args: argparse.Namespace, extension: Any) -> dict:
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    _require(not dirty, "real-runtime evidence requires a clean worktree")
    extension_path = Path(extension.__file__).resolve()
    environment_keys = (
        "EP_NIC_NAME",
        "EP_OVERRIDE_RDMA_SL",
        "NCCL_GIN_CROSS_NIC",
        "NCCL_IB_HCA",
        "NCCL_SOCKET_IFNAME",
    )
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_dirty": False,
        "extension_path": str(extension_path),
        "extension_sha256": _sha256_file(extension_path),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": str(torch.cuda.nccl.version()),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "node_count": int(os.environ["WORLD_SIZE"]),
        "local_processes": args.num_processes,
        "config": {
            "case": args.case,
            "num_tokens": args.num_tokens,
            "hidden": args.hidden,
            "num_topk": args.num_topk,
            "num_experts": args.num_experts,
            "num_sms": args.num_sms,
            "num_allocated_qps": args.num_allocated_qps,
            "proxy_slots_per_rank": args.proxy_slots_per_rank,
            "rail_balance_policy": args.rail_policy,
            "rail_balance_threshold_percent": args.rail_threshold_percent,
            "sl_idx": args.sl_idx,
            "seed": args.seed,
        },
        "environment": {key: os.environ.get(key) for key in environment_keys},
    }


def _consensus_identity(identity: dict, group: dist.ProcessGroup) -> dict:
    identities: list[dict | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(identities, identity, group=group)
    _require(
        all(item == identity for item in identities),
        "cluster source, binary, hardware, or config identity differs",
    )
    return identity


def _make_input(
    rank: int,
    routes: list[list[list[int]]],
    num_tokens: int,
    hidden: int,
    device: torch.device,
    topk_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token = torch.arange(num_tokens, dtype=torch.int32, device=device)[:, None]
    column = torch.arange(hidden, dtype=torch.int32, device=device)[None, :]
    x = ((rank * 257 + token * 17 + column) % 1024).to(torch.bfloat16)
    topk_idx = torch.tensor(routes[rank], dtype=topk_dtype, device=device)
    num_topk = topk_idx.shape[1]
    first_weight = rank * num_tokens * num_topk + 1
    topk_weights = (
        torch.arange(
            first_weight,
            first_weight + num_tokens * num_topk,
            dtype=torch.int32,
            device=device,
        )
        .reshape(topk_idx.shape)
        .to(torch.float32)
    )
    return x, topk_idx, topk_weights


def _dispatch_reference(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    from deep_ep.utils.refs import dispatch as ref_dispatch

    return ref_dispatch(
        x,
        topk_idx,
        topk_weights,
        num_tokens,
        num_experts,
    )


def _source_combine_input(
    topk_idx: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    hidden: int,
) -> torch.Tensor:
    from deep_ep.utils.refs import generate_pre_combine_data

    rank = dist.get_rank()
    source_ids = rank * num_tokens + torch.arange(num_tokens, device=topk_idx.device)
    source_y = generate_pre_combine_data(
        source_ids,
        num_tokens,
        num_topk,
        hidden,
    )
    source_y[topk_idx < 0] = 0
    return source_y


def _check_dispatch(actual, reference) -> int:
    recv_x, recv_topk_idx, recv_topk_weights, handle, _event = actual
    (ref_x, ref_topk_idx, ref_topk_weights, ref_src_ids, ref_counts) = reference
    num_recv = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    _require(num_recv == int(ref_counts.sum().item()), "receive count mismatch")
    _require(recv_x.shape[0] == num_recv, "force-v1 must return exact storage")

    src_ids = handle.recv_src_metadata[:num_recv, 0]
    order = torch.argsort(src_ids, stable=True)
    _require(torch.equal(src_ids[order], ref_src_ids), "source token ids mismatch")
    _require(torch.equal(recv_x[:num_recv][order], ref_x), "dispatch payload mismatch")
    _require(
        torch.equal(recv_topk_idx[:num_recv][order], ref_topk_idx),
        "dispatch top-k indices mismatch",
    )
    ref_weights = ref_topk_weights.masked_fill(ref_topk_idx < 0, 0)
    actual_weights = recv_topk_weights[:num_recv][order].masked_fill(
        ref_topk_idx < 0, 0
    )
    _require(torch.equal(actual_weights, ref_weights), "dispatch weights mismatch")
    return num_recv


def _run_round_trip(
    buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    reference,
    combined_reference: torch.Tensor,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    num_experts: int,
    num_sms: int,
    control_group: dist.ProcessGroup,
    timeout: int,
):
    from deep_ep.utils.refs import generate_pre_combine_data, ordered_accumulate

    dispatch_result = _phase(
        "dispatch",
        control_group,
        timeout,
        lambda: buffer.dispatch(
            x,
            topk_idx,
            topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_tokens,
            expert_alignment=1,
            num_sms=num_sms,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        ),
    )
    num_recv = _phase(
        "dispatch reference comparison",
        control_group,
        timeout,
        lambda: _check_dispatch(dispatch_result, reference),
    )
    recv_x, recv_topk_idx, recv_topk_weights, handle, _event = dispatch_result

    def prepare_combine_input() -> torch.Tensor:
        src_ids = handle.recv_src_metadata[:num_recv, 0]
        local_y = generate_pre_combine_data(src_ids, num_tokens, num_topk, hidden)
        local_y[recv_topk_idx[:num_recv] < 0] = 0
        return ordered_accumulate(local_y)

    combine_x = _phase(
        "combine input",
        control_group,
        timeout,
        prepare_combine_input,
    )
    combined_x, combined_weights, _event = _phase(
        "combine",
        control_group,
        timeout,
        lambda: buffer.combine(
            combine_x,
            handle,
            topk_weights=recv_topk_weights[:num_recv],
            num_sms=0,
            num_qps=0,
        ),
    )

    def check_combine() -> None:
        _require(
            torch.equal(combined_x, combined_reference), "combine payload mismatch"
        )
        _require(
            torch.equal(combined_weights, topk_weights), "combine weights mismatch"
        )

    _phase("combine reference comparison", control_group, timeout, check_combine)
    channels = int(handle.channel_linked_list.shape[0])
    local_digest = _phase(
        "local output digest",
        control_group,
        timeout,
        lambda: _digest(combined_x, combined_weights),
    )
    global_digest, rank_digests = _phase(
        "global output digest",
        control_group,
        timeout,
        lambda: _global_digest(local_digest, control_group),
    )
    return {
        "combined_x": combined_x.detach().cpu(),
        "combined_weights": combined_weights.detach().cpu(),
        "num_channels": channels,
        "global_digest": global_digest,
        "rank_digests": rank_digests,
    }


def _constructor_kwargs(args: argparse.Namespace, mode: str, capacity: int) -> dict:
    return {
        "num_max_tokens_per_rank": args.num_tokens,
        "hidden": args.hidden,
        "num_topk": args.num_topk,
        "use_fp8_dispatch": False,
        "deterministic": False,
        "allow_hybrid_mode": True,
        "allow_multiple_reduction": True,
        "prefer_overlap_with_compute": False,
        "sl_idx": args.sl_idx,
        "num_allocated_qps": args.num_allocated_qps,
        "num_cpu_timeout_secs": args.timeout,
        "num_gpu_timeout_secs": args.timeout,
        "explicitly_destroy": True,
        "rail_balance": mode,
        "rail_balance_proxy_slots_per_rank": capacity if mode == "force" else 0,
        "rail_balance_policy": args.rail_policy if mode == "force" else "all",
        "rail_balance_threshold_percent": (
            args.rail_threshold_percent if mode == "force" else 0
        ),
    }


def _destroy(buffer, control_group: dist.ProcessGroup, timeout: int) -> None:
    _phase("buffer destroy", control_group, timeout, buffer.destroy)


def _expect_capacity_failure(
    buffer,
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    args: argparse.Namespace,
) -> str:
    before = buffer._rail_balance_next_invocation_id
    caught = None
    try:
        buffer.dispatch(
            x,
            topk_idx,
            topk_weights,
            num_experts=args.num_experts,
            num_max_tokens_per_rank=args.num_tokens,
            expert_alignment=1,
            num_sms=args.num_sms,
            num_qps=0,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
    except BaseException as exception:
        caught = exception
    _require(isinstance(caught, RuntimeError), repr(caught))
    message = str(caught)
    _require("dispatch-plan rejected rank 0" in message, message)
    _require("priority 36" in message, message)
    _require(
        buffer._rail_balance_next_invocation_id == before + 1,
        "capacity failure did not consume exactly one invocation id",
    )
    _require(buffer._rail_balance_live_ticket is None, "capacity failure leaked ticket")
    _require(buffer._rail_balance_terminal is False, "capacity failure poisoned buffer")
    return message


def _mark_completed(record: dict, result: dict) -> None:
    record["evidence_label"] = _RUNTIME_LABEL
    record["claim_scope"] = "real_d_gt_1_runtime_correctness"
    record["correctness"].update(
        dispatch_matches_reference=True,
        combine_matches_reference=True,
        output_digest=result["global_digest"],
    )
    record["runtime"]["completed"] = True
    record["runtime"]["availability"] = "real_d_gt_1_roundtrip_counters_unavailable"
    record["runtime"]["rank_output_digests"] = result["rank_digests"]


@torch.inference_mode()
def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    import deep_ep
    from deep_ep.buffers import elastic as elastic_module
    from deep_ep.utils.envs import init_dist

    _C = importlib.import_module("deep_ep._C")

    torch.cuda.set_device(local_rank)
    rank, world_size, ep_group = init_dist(local_rank, num_local_ranks, seed=args.seed)
    control_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(seconds=args.timeout),
    )
    off_buffer = force_buffer = None
    original_host = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    original_capability = _C._rail_balance_force_available
    try:
        _phase(
            "production capability gate",
            control_group,
            args.timeout,
            lambda: (
                _require(
                    original_host is False,
                    "production host capability must be disabled",
                ),
                _require(
                    original_capability() is False,
                    "production compiled capability must be disabled",
                ),
            ),
        )
        identity = _phase(
            "local evidence identity",
            control_group,
            args.timeout,
            lambda: _local_identity(args, _C),
        )
        identity = _phase(
            "cluster evidence identity",
            control_group,
            args.timeout,
            lambda: _consensus_identity(identity, control_group),
        )

        off_buffer = _phase(
            "off constructor",
            control_group,
            args.timeout,
            lambda: deep_ep.ElasticBuffer(
                ep_group, **_constructor_kwargs(args, "off", 0)
            ),
        )
        num_scaleout_ranks, num_scaleup_ranks = off_buffer.get_logical_domain_size()
        physical_domain = tuple(off_buffer.get_physical_domain_size())

        def check_topology() -> None:
            _require(num_scaleout_ranks > 1, "C105 requires truthful D>1 Hybrid")
            _require(
                num_scaleup_ranks == num_local_ranks,
                "logical scale-up size differs from local process count",
            )
            _require(
                world_size == num_scaleout_ranks * num_scaleup_ranks,
                "logical topology does not cover WORLD",
            )
            node_rank = int(os.environ["RANK"])
            _require(
                rank == node_rank * num_local_ranks + local_rank,
                "global rank ordering is not server-major",
            )
            _require(
                world_size * args.num_tokens * args.num_topk <= (1 << 24),
                "unique FP32 validation weights exceed exact integer range",
            )

        _phase("logical topology", control_group, args.timeout, check_topology)

        def prepare_input():
            routes = build_deterministic_topk(
                args.case,
                num_scaleout_ranks=num_scaleout_ranks,
                num_scaleup_ranks=num_scaleup_ranks,
                num_tokens_per_rank=args.num_tokens,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
            )
            return _make_input(
                rank,
                routes,
                args.num_tokens,
                args.hidden,
                torch.device("cuda", local_rank),
                deep_ep.topk_idx_t,
            )

        x, topk_idx, topk_weights = _phase(
            "input preparation", control_group, args.timeout, prepare_input
        )
        reference = _phase(
            "official dispatch reference",
            control_group,
            args.timeout,
            lambda: _dispatch_reference(
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                args.num_experts,
            ),
        )
        source_y = _phase(
            "reference combine input",
            control_group,
            args.timeout,
            lambda: _source_combine_input(
                topk_idx, args.num_tokens, args.num_topk, args.hidden
            ),
        )

        def combine_reference():
            from deep_ep.utils.refs import combine as ref_combine

            return ref_combine(
                source_y,
                topk_idx,
                num_scaleout_ranks,
                num_scaleup_ranks,
                args.num_experts,
                None,
                True,
                True,
            )

        combined_reference = _phase(
            "official combine reference",
            control_group,
            args.timeout,
            combine_reference,
        )
        off_result = _run_round_trip(
            off_buffer,
            x,
            topk_idx,
            topk_weights,
            reference,
            combined_reference,
            args.num_tokens,
            args.num_topk,
            args.hidden,
            args.num_experts,
            args.num_sms,
            control_group,
            args.timeout,
        )
        _destroy(off_buffer, control_group, args.timeout)
        off_buffer = None

        # Proxy demand is the total incoming deficit per egress and does not
        # depend on channel striping. C=1 is therefore enough to choose the
        # constructor capacity before a force handle exists.
        capacity_bundle = _phase(
            "capacity oracle",
            control_group,
            args.timeout,
            lambda: build_validation_bundle(
                run_id=f"{args.run_id}/{args.case}",
                case=args.case,
                modes=("off", "force"),
                num_scaleout_ranks=num_scaleout_ranks,
                num_scaleup_ranks=num_scaleup_ranks,
                num_tokens_per_rank=args.num_tokens,
                num_topk=args.num_topk,
                num_experts=args.num_experts,
                hidden=args.hidden,
                num_channels=1,
                proxy_slots_per_rank=args.proxy_slots_per_rank,
                policy=args.rail_policy,
                threshold_percent=args.rail_threshold_percent,
            ),
        )
        capacity = capacity_bundle["config"]["proxy_slots_per_rank"]

        def enable_validation_capability() -> None:
            elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
            _C._rail_balance_force_available = lambda: True
            _require(
                elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is True,
                "host validation capability was not enabled",
            )
            _require(
                _C._rail_balance_force_available() is True,
                "compiled validation capability was not enabled",
            )

        _phase(
            "enable validation capability",
            control_group,
            args.timeout,
            enable_validation_capability,
        )
        force_buffer = _phase(
            "force constructor",
            control_group,
            args.timeout,
            lambda: deep_ep.ElasticBuffer(
                ep_group, **_constructor_kwargs(args, "force", capacity)
            ),
        )
        if args.case == "capacity":
            bundle = capacity_bundle
            _mark_completed(bundle["records"]["off"], off_result)
            bundle["config"]["num_channels_source"] = "capacity_oracle"
            first_error = _phase(
                "capacity fail-close #1",
                control_group,
                args.timeout,
                lambda: _expect_capacity_failure(
                    force_buffer, x, topk_idx, topk_weights, args
                ),
            )
            second_error = _phase(
                "capacity fail-close #2",
                control_group,
                args.timeout,
                lambda: _expect_capacity_failure(
                    force_buffer, x, topk_idx, topk_weights, args
                ),
            )
            force_record = bundle["records"]["force"]
            force_record["evidence_label"] = _CAPACITY_LABEL
            force_record["claim_scope"] = "real_d_gt_1_precommit_fail_closed"
            force_record["runtime"]["availability"] = "expected_plan_rejection"
            force_record["error"] = {
                "first": first_error,
                "retry": second_error,
            }
            bundle["evidence_label"] = _CAPACITY_LABEL
        else:
            force_result = _run_round_trip(
                force_buffer,
                x,
                topk_idx,
                topk_weights,
                reference,
                combined_reference,
                args.num_tokens,
                args.num_topk,
                args.hidden,
                args.num_experts,
                args.num_sms,
                control_group,
                args.timeout,
            )

            def check_ab() -> None:
                _require(
                    torch.equal(force_result["combined_x"], off_result["combined_x"]),
                    "off/force combined payload differs",
                )
                _require(
                    torch.equal(
                        force_result["combined_weights"], off_result["combined_weights"]
                    ),
                    "off/force combined weights differ",
                )

            _phase("off/force A/B", control_group, args.timeout, check_ab)
            force_channels = _phase(
                "force channel geometry",
                control_group,
                args.timeout,
                lambda: _consensus_int(force_result["num_channels"], control_group),
            )
            bundle = _phase(
                "final force oracle",
                control_group,
                args.timeout,
                lambda: build_validation_bundle(
                    run_id=f"{args.run_id}/{args.case}",
                    case=args.case,
                    modes=("off", "force"),
                    num_scaleout_ranks=num_scaleout_ranks,
                    num_scaleup_ranks=num_scaleup_ranks,
                    num_tokens_per_rank=args.num_tokens,
                    num_topk=args.num_topk,
                    num_experts=args.num_experts,
                    hidden=args.hidden,
                    num_channels=force_channels,
                    proxy_slots_per_rank=capacity,
                    policy=args.rail_policy,
                    threshold_percent=args.rail_threshold_percent,
                ),
            )
            _mark_completed(bundle["records"]["off"], off_result)
            _mark_completed(bundle["records"]["force"], force_result)
            for record in bundle["records"].values():
                record["correctness"]["off_force_equal"] = True
            bundle["evidence_label"] = _RUNTIME_LABEL

        _destroy(force_buffer, control_group, args.timeout)
        force_buffer = None

        def restore_production_capability() -> None:
            elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = original_host
            _C._rail_balance_force_available = original_capability
            _require(
                elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is original_host,
                "host capability object was not restored",
            )
            _require(
                _C._rail_balance_force_available is original_capability,
                "compiled capability function was not restored",
            )
            _require(
                elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE is False,
                "production host capability is not disabled after restoration",
            )
            _require(
                _C._rail_balance_force_available() is False,
                "production compiled capability is not disabled after restoration",
            )

        _phase(
            "restore production capability",
            control_group,
            args.timeout,
            restore_production_capability,
        )
        bundle["runtime_environment"] = {
            **identity,
            "physical_domain": list(physical_domain),
            "off_num_channels": off_result["num_channels"],
            "force_num_channels": (
                None if args.case == "capacity" else force_result["num_channels"]
            ),
        }

        def write_result() -> None:
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(bundle, allow_nan=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"PASS C105 {args.case}: "
                    f"{bundle['evidence_label']} -> {args.output}",
                    flush=True,
                )

        _phase("write evidence bundle", control_group, args.timeout, write_result)
        dist.destroy_process_group()
    finally:
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = original_host
        _C._rail_balance_force_available = original_capability
        for buffer in (force_buffer, off_buffer):
            if buffer is not None:
                with suppress(BaseException):
                    buffer.destroy()


def _terminate_process_group(
    process: subprocess.Popen,
    grace_seconds: int = 10,
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)
    # The spawn leader can exit before its CUDA children. Always kill the
    # original dedicated process group after the grace period.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def _run_watchdog(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-suite",
        *sys.argv[1:],
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.watchdog_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RuntimeError(
            f"C105 watchdog expired after {args.watchdog_seconds}s"
        ) from error
    if return_code:
        # The spawn leader may exit before every CUDA child in its process
        # group. Kill the dedicated group so a failed validation leaves no
        # rank behind on the GPUs.
        _terminate_process_group(process, grace_seconds=0)
        raise SystemExit(return_code)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", choices=CANONICAL_CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-sms", type=int, default=2)
    parser.add_argument("--num-allocated-qps", type=int, default=1)
    parser.add_argument("--proxy-slots-per-rank", type=int)
    parser.add_argument(
        "--rail-policy", choices=("all", "active", "adaptive"), default="all"
    )
    parser.add_argument(
        "--rail-threshold-percent", type=int, default=0,
        help=(
            "tolerated peak overload above the balanced target; "
            "0 disables the threshold gate"
        ),
    )
    parser.add_argument("--sl-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--watchdog-seconds", type=int, default=1800)
    parser.add_argument("--worker-suite", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not __debug__:
        parser.error("C105 correctness checks require Python assertions")
    if args.num_processes < 2 or torch.cuda.device_count() < args.num_processes:
        parser.error("C105 requires at least two visible local CUDA devices")
    try:
        _validate_launch_environment()
    except ValueError as error:
        parser.error(str(error))
    if args.timeout <= 0 or args.watchdog_seconds <= 0:
        parser.error("timeouts must be positive")
    if args.num_tokens <= 0 or args.hidden <= 0 or args.hidden % 256 != 0:
        parser.error("tokens must be positive and hidden a positive multiple of 256")
    if args.num_topk <= 0 or args.num_sms < 2 or args.num_allocated_qps <= 0:
        parser.error("top-k, SM count, and allocated QP count are invalid")
    if not 0 <= args.rail_threshold_percent <= 3100:
        parser.error("rail threshold percent must be in [0, 3100]")
    if args.case == "capacity" and (
        args.rail_policy != "all" or args.rail_threshold_percent != 0
    ):
        parser.error("capacity case requires --rail-policy all --rail-threshold-percent 0")
    return args


def main() -> None:
    args = _parse_args()
    if not args.worker_suite:
        _run_watchdog(args)
        return
    torch.multiprocessing.spawn(
        _worker,
        args=(args.num_processes, args),
        nprocs=args.num_processes,
        join=True,
    )


if __name__ == "__main__":
    main()
