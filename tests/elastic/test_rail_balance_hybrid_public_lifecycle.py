"""CPU-only contract for the public C080 force handle lifecycle.

This test deliberately does not claim Gin, RDMA, or CUDA data-plane evidence.
It drives the real :class:`ElasticBuffer` public methods with a tiny fake
runtime and a fixed-gate monkeypatch.  The goal is to freeze the host ordering,
failure ownership, and one-shot ``EPHandle`` ticket before public force is
advertised.

Run directly (pytest is not required)::

    PYTHONPATH=.:tests/elastic python -B \
      tests/elastic/test_rail_balance_hybrid_public_lifecycle.py
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Callable

import torch

import deep_ep.buffers.elastic as elastic_module
from deep_ep.buffers.elastic import EPHandle, ElasticBuffer
from deep_ep.utils.event import EventOverlap


_ROOT = Path(__file__).resolve().parents[2]
_ELASTIC_SOURCE = _ROOT / "deep_ep/buffers/elastic.py"
_PASS = (0, -1, 0, 0)
_REMOTE_REJECT = (
    elastic_module._make_rail_balance_world_gate_error_key(7, 0),
    -1, 0, 0)


def _node_source(source: str, node: ast.AST) -> str:
    text = ast.get_source_segment(source, node)
    assert text is not None
    return text


def _assert_in_order(source: str, tokens: tuple[str, ...]) -> None:
    cursor = 0
    positions = []
    for token in tokens:
        position = source.index(token, cursor)
        positions.append(position)
        cursor = position + len(token)
    assert positions == sorted(positions), (tokens, positions)


def _assert_source_contract() -> None:
    """Freeze the minimal public shape without prescribing a framework."""
    source = _ELASTIC_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

    # The ticket is metadata-only.  Its three semantic fields are deliberately
    # separate from the ordinary EPHandle constructor and routing tensors.
    ticket_classes = []
    for node in classes:
        for child in node.body:
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target = child.targets[0]
            if not isinstance(target, ast.Name) or target.id != "__slots__":
                continue
            if ast.literal_eval(child.value) == (
                    "owner_token", "invocation_id", "state"):
                ticket_classes.append(_node_source(source, node))
    assert len(ticket_classes) == 1, "expected one private rail-balance ticket"

    ep_handle = next(node for node in classes if node.name == "EPHandle")
    ep_init = next(
        child for child in ep_handle.body
        if isinstance(child, ast.FunctionDef) and child.name == "__init__")
    ep_init_source = _node_source(source, ep_init)
    assert "rail_balance" not in ep_init_source
    assert "_rail_balance_ticket" in source

    public_dispatch = next(
        node for node in functions
        if node.name == "dispatch" and
        isinstance(getattr(node, "args", None), ast.arguments) and
        any(arg.arg == "handle" for arg in node.args.args))
    public_combine = next(
        node for node in functions
        if node.name == "combine" and
        isinstance(getattr(node, "args", None), ast.arguments) and
        any(arg.arg == "handle" for arg in node.args.args))
    dispatch_source = _node_source(source, public_dispatch)
    combine_source = _node_source(source, public_combine)

    # Off still owns the old call sites.  Force branches before either legacy
    # call and does not repurpose cached dispatch.
    assert "self.runtime.dispatch(" in dispatch_source
    assert "self.runtime.combine(" in combine_source
    force_dispatch_call = dispatch_source.index("rail_balance")
    legacy_dispatch_call = dispatch_source.index("self.runtime.dispatch(")
    force_combine_call = combine_source.index("rail_balance")
    legacy_combine_call = combine_source.index("self.runtime.combine(")
    assert force_dispatch_call < legacy_dispatch_call
    assert force_combine_call < legacy_combine_call

    # Find the actual force helpers even if their private names change.  The
    # runtime ABI and publication sequence are not negotiable.
    force_dispatch_helpers = [
        _node_source(source, node) for node in functions
        if "_rail_balance_hybrid_dispatch_prepare" in _node_source(source, node)
    ]
    force_combine_helpers = [
        _node_source(source, node) for node in functions
        if "_rail_balance_hybrid_combine_prepare" in _node_source(source, node)
    ]
    assert len(force_dispatch_helpers) == 1
    assert len(force_combine_helpers) == 1
    force_dispatch = force_dispatch_helpers[0]
    force_combine = force_combine_helpers[0]
    _assert_in_order(force_dispatch, (
        "_rail_balance_hybrid_dispatch_prepare",
        "_run_rail_balance_world_gate",
        "_rail_balance_hybrid_plan_finish",
        "_run_rail_balance_world_gate",
        "_rail_balance_hybrid_dispatch_commit",
        "_rail_balance_hybrid_dispatch_finish",
    ))
    _assert_in_order(force_combine, (
        "_rail_balance_hybrid_combine_prepare",
        "_run_rail_balance_world_gate",
        "CONSUMED",
        "_rail_balance_hybrid_combine_commit",
    ))
    assert "_rail_balance_hybrid_plan_abort" in force_dispatch
    assert "_rail_balance_hybrid_combine_abort" in force_combine
    assert source.count(
        "self._rail_balance_world_gate_stream = torch.cuda.Stream(") == 1
    assert force_dispatch.count("with gate_context:") == 2
    assert force_combine.count("with gate_context:") == 1
    assert "'_rail_balance_world_gate_stream'" in force_dispatch
    assert "'_rail_balance_world_gate_stream'" in force_combine


def _assert_multinode_hybrid_auto_resources_are_bounded() -> None:
    assert elastic_module._hybrid_auto_sm_floor(True, 2) == 32
    assert elastic_module._hybrid_auto_sm_floor(True, 8) == 32
    assert elastic_module._hybrid_auto_sm_floor(False, 2) == 64
    assert elastic_module._hybrid_auto_sm_floor(True, 1) == 64
    assert elastic_module._hybrid_auto_qp_limit(True, 2) == 2
    assert elastic_module._hybrid_auto_qp_limit(True, 16) == 8
    assert elastic_module._hybrid_auto_qp_limit(False, 2) is None
    assert elastic_module._hybrid_auto_qp_limit(True, 1) is None
    assert elastic_module._rail_only_auto_allocated_qps(True, '0') == 2
    assert elastic_module._rail_only_auto_allocated_qps(True, '1') is None
    assert elastic_module._rail_only_auto_allocated_qps(True, None) is None
    assert elastic_module._rail_only_auto_allocated_qps(False, '0') is None


class _FakeGroup:
    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 4


class _GateController:
    _by_host_id: dict[int, "_GateController"] = {}

    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.host_words = torch.zeros(
            elastic_module._RAIL_BALANCE_WORLD_GATE_WORDS,
            dtype=torch.int64,
        )
        # The operation is monkeypatched, so CPU storage is intentional.  It
        # lets this lifecycle test run without allocating a CUDA context.
        self.device_words = torch.zeros_like(self.host_words)
        self.overrides: list[tuple[int, int, int, int]] = []
        self.accepted_manifests: list[tuple[int, ...]] = []
        self._by_host_id[id(self.host_words)] = self

    def reject_next(self) -> None:
        self.overrides.append(_REMOTE_REJECT)

    def run(self) -> tuple[int, int, int, int]:
        self.trace.append("gate")
        if self.overrides:
            return self.overrides.pop(0)
        error = int(self.host_words[0].item())
        priority = (
            elastic_module._decode_rail_balance_world_gate_error_key(error)[0]
            if error else 0
        )
        if error and priority != \
                elastic_module._RAIL_BALANCE_DISPATCH_HAS_MOVES:
            return error, -1, 0, 0

        width = int(self.host_words[2 + 2 * 4].item())
        assert 0 < width <= elastic_module._RAIL_BALANCE_WORLD_GATE_MAX_FIELDS
        manifest = tuple(
            int(self.host_words[2 + 2 * index].item())
            for index in range(width)
        )
        assert all(
            int(self.host_words[3 + 2 * index].item()) == -value
            for index, value in enumerate(manifest)
        )
        self.accepted_manifests.append(manifest)
        if error:
            return error, -1, 0, 0
        return _PASS

    @classmethod
    def patched_gate(
        cls, device_words: torch.Tensor, host_words: torch.Tensor, group: object,
    ) -> tuple[int, int, int, int]:
        del device_words, group
        return cls._by_host_id[id(host_words)].run()


def _ticket_state(ticket: object) -> str:
    state = getattr(ticket, "state")
    value = getattr(state, "value", state)
    value = getattr(value, "value", value)
    return str(value).upper()


_DISPATCH_COMMON_FIELDS = (
    2, 1, 2, 2, 256, 2, 8, 8, 8,
    4, 1, 1, 1, 1024, 4, 100, 2 << 20, 2 << 20, 7, 0, 0,
)


def _dispatch_result(
    x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
) -> tuple[object, ...]:
    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    num_experts = 8
    num_local_experts = num_experts // 4
    return (
        x.clone(),                         # recv_x
        None,                              # recv_sf
        topk_idx.clone(),                  # recv_topk_idx
        topk_weights.clone(),              # recv_topk_weights
        topk_idx.clone(),                  # copied topk
        num_tokens,
        0,
        [num_tokens] + [0] * (num_local_experts - 1),
        torch.tensor([num_tokens, num_tokens], dtype=torch.int32),
        torch.tensor([num_tokens, num_tokens], dtype=torch.int32),
        torch.zeros(num_local_experts, dtype=torch.int32),
        torch.zeros((num_tokens, 2 + num_topk), dtype=torch.int32),
        torch.arange(num_tokens, dtype=torch.int32),
        torch.zeros((1, 1, 3 + 2 * num_topk), dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int32),
        None,
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.trace: list[str] = []
        self.buffer: ElasticBuffer | None = None
        self.plan_status = 0
        self.bypass_plan = False
        self.fail_dispatch_prepare = False
        self.fail_plan_finish = False
        self.fail_dispatch_commit = False
        self.fail_dispatch_finish = False
        self.fail_combine_prepare = False
        self.fail_combine_commit = False
        self._x: torch.Tensor | None = None
        self._topk_idx: torch.Tensor | None = None
        self._topk_weights: torch.Tensor | None = None
        self._combine_x: torch.Tensor | None = None
        self._combine_weights: torch.Tensor | None = None
        self._prepared_ticket: object | None = None
        self.hop_aware: bool | None = None
        self.activation_threshold_percent: int | None = None
        self.two_hop_config: tuple[int, int, int] | None = None

    def _rail_balance_hybrid_dispatch_prepare(self, *args: object) -> tuple[int, ...]:
        self.trace.append("dispatch_prepare")
        if self.fail_dispatch_prepare:
            raise RuntimeError("injected dispatch prepare failure")
        self._x = args[0]  # type: ignore[assignment]
        self._topk_idx = args[1]  # type: ignore[assignment]
        self._topk_weights = args[2]  # type: ignore[assignment]
        assert args[-6] == 0
        assert type(args[-5]) is int
        self.activation_threshold_percent = args[-5]
        assert type(args[-4]) is bool
        self.hop_aware = args[-4]
        self.two_hop_config = args[-3:]  # type: ignore[assignment]
        # Exact H4b manifest width: status then immutable public geometry/ABI.
        return 0, *_DISPATCH_COMMON_FIELDS

    def _rail_balance_hybrid_plan_finish(self, invocation_id: int) -> tuple[torch.Tensor, ...]:
        del invocation_id
        self.trace.append("plan_finish")
        if self.fail_plan_finish:
            raise RuntimeError("injected plan finish failure")
        outputs = [torch.zeros(1, dtype=torch.int32) for _ in range(13)]
        outputs[12] = torch.tensor(
            0 if self.bypass_plan else 1, dtype=torch.int32)
        outputs.append(torch.tensor(self.plan_status, dtype=torch.int32))
        return tuple(outputs)

    def _rail_balance_hybrid_dispatch_commit(self, invocation_id: int) -> None:
        del invocation_id
        self.trace.append("dispatch_commit")
        if self.fail_dispatch_commit:
            raise RuntimeError("injected post-Gate dispatch commit failure")

    def _rail_balance_hybrid_dispatch_finish(self, invocation_id: int) -> tuple[object, ...]:
        del invocation_id
        self.trace.append("dispatch_finish")
        if self.fail_dispatch_finish:
            raise RuntimeError("injected post-Gate dispatch finish failure")
        assert self._x is not None
        assert self._topk_idx is not None
        assert self._topk_weights is not None
        return _dispatch_result(self._x, self._topk_idx, self._topk_weights)

    def _rail_balance_hybrid_plan_abort(self, invocation_id: int) -> None:
        del invocation_id
        self.trace.append("dispatch_abort")

    def _rail_balance_hybrid_combine_prepare(self, *args: object) -> None:
        self.trace.append("combine_prepare")
        assert self.buffer is not None
        self._prepared_ticket = self.buffer._rail_balance_live_ticket  # type: ignore[attr-defined]
        assert self._prepared_ticket is not None
        assert _ticket_state(self._prepared_ticket) == "PREPARING"
        assert args[2] == getattr(self._prepared_ticket, "invocation_id")
        self._combine_x = args[0]  # type: ignore[assignment]
        self._combine_weights = args[1]  # type: ignore[assignment]
        if self.fail_combine_prepare:
            raise RuntimeError("injected combine prepare failure")

    def _rail_balance_hybrid_combine_abort(self, invocation_id: int) -> None:
        del invocation_id
        self.trace.append("combine_abort")
        self._prepared_ticket = None

    def _rail_balance_hybrid_combine_commit(self, invocation_id: int) -> tuple[object, ...]:
        del invocation_id
        assert self.buffer is not None
        assert self._prepared_ticket is not None
        state = _ticket_state(self._prepared_ticket)
        self.trace.append(f"combine_commit:{state}")
        if self.fail_combine_commit:
            raise RuntimeError("injected post-Gate combine commit failure")
        assert state == "CONSUMED"
        assert self._combine_x is not None and self._combine_weights is not None
        return self._combine_x.clone(), self._combine_weights.clone(), None

    # Legacy methods are used only by the explicit off-path identity test.
    def dispatch(self, *args: object) -> tuple[object, ...]:
        self.trace.append("legacy_dispatch")
        return _dispatch_result(args[0], args[2], args[3])  # type: ignore[arg-type]

    def combine(self, x: torch.Tensor, topk_weights: torch.Tensor, *args: object) \
            -> tuple[object, ...]:
        del args
        self.trace.append("legacy_combine")
        return x.clone(), topk_weights.clone(), None


def _make_buffer(
    *, force: bool, mode: str = "force"
) -> tuple[ElasticBuffer, _FakeRuntime, _GateController]:
    buffer = object.__new__(ElasticBuffer)
    runtime = _FakeRuntime()
    runtime.buffer = buffer
    buffer.runtime = runtime
    buffer.group = _FakeGroup()
    buffer.rank_idx = 0
    buffer.num_ranks = 4
    buffer.num_scaleout_ranks = 2
    buffer.num_scaleup_ranks = 2
    buffer.scaleout_rank_idx = 0
    buffer.scaleup_rank_idx = 0
    buffer.num_rdma_ranks = 2
    buffer.num_nvlink_ranks = 2
    buffer.num_allocated_qps = 16
    buffer.num_max_tokens_per_rank = 8
    buffer.allow_hybrid_mode = True
    buffer.allow_multiple_reduction = True
    buffer.prefer_overlap_with_compute = True
    buffer.deterministic = False

    gate = _GateController(runtime.trace)
    if force:
        buffer._rail_balance_mode = mode
        buffer._rail_balance_proxy_slots_per_rank = 8
        buffer._rail_balance_policy = 0
        buffer._rail_balance_threshold_percent = 0
        buffer._rail_balance_two_hop_threshold_percent = 0
        buffer._rail_balance_max_two_hop_percent = 0
        buffer._rail_balance_hop_penalty_percent = 0
        buffer._rail_balance_arena_offset = 2 << 20
        buffer._rail_balance_arena_bytes = 2 << 20
        buffer._rail_balance_owner_token = object()
        buffer._rail_balance_live_ticket = None
        buffer._rail_balance_terminal = False
        buffer._rail_balance_zero_move_bypass_budget = 0
        buffer._rail_balance_zero_move_common_fields = None
        # Epoch zero stays reserved for uninitialized/control words.
        buffer._rail_balance_next_invocation_id = 1
        # Keep aliases until the public implementation chooses one spelling;
        # only one pair is consumed by production code.
        buffer._rail_balance_world_gate_device_words = gate.device_words
        buffer._rail_balance_world_gate_host_words = gate.host_words
    return buffer, runtime, gate


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.arange(2 * 256, dtype=torch.float32).view(2, 256).to(torch.bfloat16)
    topk_idx = torch.tensor([[2, 6], [3, 7]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3]], dtype=torch.float32)
    return x, topk_idx, topk_weights


def _force_dispatch(
    buffer: ElasticBuffer,
    *,
    handle: EPHandle | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, EPHandle, EventOverlap]:
    x, topk_idx, topk_weights = _inputs()
    return buffer.dispatch(
        x,
        None if handle is not None else topk_idx,
        None if handle is not None else topk_weights,
        num_experts=8,
        num_max_tokens_per_rank=8,
        expert_alignment=1,
        num_sms=4,
        num_qps=4,
        handle=handle,
        do_cpu_sync=True if handle is None else None,
    )


def _expect_raises(call: Callable[[], object]) -> BaseException:
    try:
        call()
    except BaseException as exception:
        return exception
    raise AssertionError("expected the lifecycle operation to fail")


def _assert_successful_round_trip() -> None:
    buffer, runtime, gate = _make_buffer(force=True)
    recv_x, recv_idx, recv_weights, handle, event = _force_dispatch(buffer)
    assert isinstance(handle, EPHandle)
    assert isinstance(event, EventOverlap)
    assert runtime.hop_aware is False
    ticket = handle._rail_balance_ticket  # type: ignore[attr-defined]
    assert ticket.owner_token is buffer._rail_balance_owner_token  # type: ignore[attr-defined]
    assert ticket.invocation_id == 1
    assert _ticket_state(ticket) == "LIVE"
    assert buffer._rail_balance_live_ticket is ticket  # type: ignore[attr-defined]
    assert runtime.trace == [
        "dispatch_prepare", "gate", "plan_finish", "gate",
        "dispatch_commit", "dispatch_finish",
    ]

    runtime.trace.clear()
    combine_x = recv_x + 2
    combine_input_weights = recv_weights + 0.125
    combined_x, combined_weights, event = buffer.combine(
        combine_x, handle, combine_input_weights, num_sms=4, num_qps=0)
    assert torch.equal(combined_x, combine_x)
    assert torch.equal(combined_weights, combine_input_weights)
    assert isinstance(event, EventOverlap)
    assert _ticket_state(ticket) == "CONSUMED"
    assert buffer._rail_balance_live_ticket is None  # type: ignore[attr-defined]
    assert runtime.trace == [
        "combine_prepare", "gate", "combine_commit:CONSUMED",
    ]
    assert recv_idx.shape == (2, 2)

    prefix = (
        elastic_module._RAIL_BALANCE_PROTOCOL_MAGIC,
        elastic_module._RAIL_BALANCE_PROTOCOL_VERSION,
    )
    assert gate.accepted_manifests == [
        (
            *prefix,
            elastic_module._RAIL_BALANCE_OPERATION_DISPATCH,
            elastic_module._RAIL_BALANCE_PHASE_PREPARE,
            elastic_module._RAIL_BALANCE_DISPATCH_MANIFEST_WIDTH,
            4, 1,
            *_DISPATCH_COMMON_FIELDS,
        ),
        (
            *prefix,
            elastic_module._RAIL_BALANCE_OPERATION_DISPATCH,
            elastic_module._RAIL_BALANCE_PHASE_PLAN,
            elastic_module._RAIL_BALANCE_DISPATCH_MANIFEST_WIDTH,
            4, 1,
            *_DISPATCH_COMMON_FIELDS,
        ),
        (
            *prefix,
            elastic_module._RAIL_BALANCE_OPERATION_COMBINE,
            elastic_module._RAIL_BALANCE_PHASE_PREPARE,
            elastic_module._RAIL_BALANCE_COMBINE_MANIFEST_WIDTH,
            4, 1,
        ),
    ]


def _assert_one_hop_selects_endpoint_planner() -> None:
    buffer, runtime, _ = _make_buffer(force=True, mode="one_hop")
    buffer._rail_balance_threshold_percent = 20
    _force_dispatch(buffer)
    assert runtime.hop_aware is True
    assert runtime.activation_threshold_percent == 20


def _assert_adaptive_does_not_fall_back() -> None:
    buffer, runtime, _ = _make_buffer(force=True, mode="adaptive")
    buffer._rail_balance_two_hop_threshold_percent = 7
    buffer._rail_balance_max_two_hop_percent = 40
    buffer._rail_balance_hop_penalty_percent = 90
    _force_dispatch(buffer)
    assert runtime.hop_aware is True
    assert runtime.two_hop_config == (7, 40, 90)


def _assert_zero_move_plan_bypasses_force() -> None:
    buffer, runtime, _ = _make_buffer(force=True)
    runtime.bypass_plan = True
    recv_x, _, recv_weights, handle, _ = _force_dispatch(buffer)
    assert getattr(handle, "_rail_balance_ticket", None) is None
    buffer.combine(recv_x, handle, recv_weights, num_sms=4, num_qps=0)

    assert runtime.trace == [
        "dispatch_prepare", "gate", "plan_finish", "gate",
        "dispatch_abort", "legacy_dispatch", "legacy_combine",
    ]

    runtime.trace.clear()
    recv_x, _, recv_weights, handle, _ = _force_dispatch(buffer)
    assert getattr(handle, "_rail_balance_ticket", None) is None
    buffer.combine(recv_x, handle, recv_weights, num_sms=4, num_qps=0)
    assert runtime.trace == ["gate", "legacy_dispatch", "legacy_combine"]
    assert buffer._rail_balance_zero_move_bypass_budget == \
        elastic_module._RAIL_BALANCE_ZERO_MOVE_RECHECK_INTERVAL - 2


def _assert_gate_rejection_is_retryable() -> None:
    # A Gate2 capacity/status rejection aborts the private prepare and does not
    # make the buffer terminal.  The same Python buffer can dispatch again.
    buffer, runtime, _ = _make_buffer(force=True)
    runtime.plan_status = 1
    _expect_raises(lambda: _force_dispatch(buffer))
    assert runtime.trace == [
        "dispatch_prepare", "gate", "plan_finish", "gate", "dispatch_abort",
    ]
    assert buffer._rail_balance_live_ticket is None  # type: ignore[attr-defined]
    assert buffer._rail_balance_terminal is False  # type: ignore[attr-defined]

    runtime.trace.clear()
    runtime.plan_status = 0
    recv_x, _, recv_weights, handle, _ = _force_dispatch(buffer)
    ticket = handle._rail_balance_ticket  # type: ignore[attr-defined]
    assert ticket.invocation_id == 2

    # Combine rejection drops only the prepared output owner.  The exact same
    # ticket remains LIVE and can be retried.
    runtime.trace.clear()
    gate = _GateController._by_host_id[
        id(buffer._rail_balance_world_gate_host_words)]  # type: ignore[attr-defined]
    gate.reject_next()
    _expect_raises(lambda: buffer.combine(
        recv_x, handle, recv_weights, num_sms=4, num_qps=0))
    assert runtime.trace == ["combine_prepare", "gate", "combine_abort"]
    assert _ticket_state(ticket) == "LIVE"
    assert buffer._rail_balance_live_ticket is ticket  # type: ignore[attr-defined]
    assert buffer._rail_balance_terminal is False  # type: ignore[attr-defined]

    runtime.trace.clear()
    buffer.combine(recv_x, handle, recv_weights, num_sms=4, num_qps=0)
    assert runtime.trace == [
        "combine_prepare", "gate", "combine_commit:CONSUMED",
    ]


def _assert_dispatch_entry_failures_are_retryable() -> None:
    cases = (
        (
            lambda runtime, gate: gate.reject_next(),
            ["dispatch_prepare", "gate", "dispatch_abort"],
        ),
        (
            lambda runtime, gate: setattr(
                runtime, "fail_dispatch_prepare", True),
            ["dispatch_prepare", "gate", "dispatch_abort"],
        ),
        (
            lambda runtime, gate: setattr(
                runtime, "fail_plan_finish", True),
            [
                "dispatch_prepare", "gate", "plan_finish", "gate",
                "dispatch_abort",
            ],
        ),
    )
    for inject, expected_trace in cases:
        buffer, runtime, gate = _make_buffer(force=True)
        inject(runtime, gate)
        _expect_raises(lambda: _force_dispatch(buffer))
        assert runtime.trace == expected_trace
        assert buffer._rail_balance_live_ticket is None  # type: ignore[attr-defined]
        assert buffer._rail_balance_terminal is False  # type: ignore[attr-defined]

        runtime.trace.clear()
        runtime.fail_dispatch_prepare = False
        runtime.fail_plan_finish = False
        _, _, _, handle, _ = _force_dispatch(buffer)
        assert handle._rail_balance_ticket.invocation_id == 2  # type: ignore[attr-defined]


def _assert_combine_prepare_failure_is_retryable() -> None:
    buffer, runtime, _ = _make_buffer(force=True)
    recv_x, _, recv_weights, handle, _ = _force_dispatch(buffer)
    ticket = handle._rail_balance_ticket  # type: ignore[attr-defined]

    runtime.trace.clear()
    runtime.fail_combine_prepare = True
    _expect_raises(lambda: buffer.combine(
        recv_x, handle, recv_weights, num_sms=4, num_qps=0))
    assert runtime.trace == ["combine_prepare", "gate", "combine_abort"]
    assert _ticket_state(ticket) == "LIVE"
    assert buffer._rail_balance_live_ticket is ticket  # type: ignore[attr-defined]
    assert buffer._rail_balance_terminal is False  # type: ignore[attr-defined]

    runtime.trace.clear()
    runtime.fail_combine_prepare = False
    combined_x, combined_weights, _ = buffer.combine(
        recv_x + 3, handle, recv_weights + 0.25, num_sms=4, num_qps=0)
    assert torch.equal(combined_x, recv_x + 3)
    assert torch.equal(combined_weights, recv_weights + 0.25)
    assert runtime.trace == [
        "combine_prepare", "gate", "combine_commit:CONSUMED",
    ]


def _assert_cached_foreign_and_duplicate_reach_the_gate() -> None:
    owner, owner_runtime, _ = _make_buffer(force=True)
    recv_x, _, recv_weights, handle, _ = _force_dispatch(owner)
    ticket = handle._rail_balance_ticket  # type: ignore[attr-defined]

    # A second ordinary dispatch is invalid while the first ticket is live,
    # but the entry gate must not consume or replace that existing ticket.
    owner_runtime.trace.clear()
    _expect_raises(lambda: _force_dispatch(owner))
    assert owner_runtime.trace == ["gate"]
    assert _ticket_state(ticket) == "LIVE"
    assert owner._rail_balance_live_ticket is ticket  # type: ignore[attr-defined]

    # Cached dispatch is unsupported, but it must report through Gate1 so one
    # rank cannot return while peers enter the source-node barrier.
    owner_runtime.trace.clear()
    _expect_raises(lambda: _force_dispatch(owner, handle=handle))
    assert "gate" in owner_runtime.trace
    assert "dispatch_commit" not in owner_runtime.trace
    assert _ticket_state(ticket) == "LIVE"
    assert owner._rail_balance_live_ticket is ticket  # type: ignore[attr-defined]

    # A foreign buffer also participates in the combine entry gate.  Its stale-
    # safe abort cannot consume the real owner's live handle.
    foreign, foreign_runtime, _ = _make_buffer(force=True)
    _expect_raises(lambda: foreign.combine(
        recv_x, handle, recv_weights, num_sms=4, num_qps=0))
    assert "gate" in foreign_runtime.trace
    assert "combine_commit" not in " ".join(foreign_runtime.trace)
    assert _ticket_state(ticket) == "LIVE"

    # Shallow copies share the mutable one-shot ticket state.  Once one copy is
    # accepted, the other copy cannot submit combine and is rejected via gate.
    duplicate = copy.copy(handle)
    assert duplicate._rail_balance_ticket is ticket  # type: ignore[attr-defined]
    assert duplicate._rail_balance_ticket.state is ticket.state  # type: ignore[attr-defined]
    owner_runtime.trace.clear()
    owner.combine(recv_x, handle, recv_weights, num_sms=4, num_qps=0)
    assert _ticket_state(ticket) == "CONSUMED"

    owner_runtime.trace.clear()
    _expect_raises(lambda: owner.combine(
        recv_x, duplicate, recv_weights, num_sms=4, num_qps=0))
    assert "gate" in owner_runtime.trace
    assert not any(item.startswith("combine_commit") for item in owner_runtime.trace)


def _assert_post_gate_failure_is_terminal() -> None:
    # Dispatch has published proxy payload after Gate2 once commit starts.
    # Neither the Python object nor C++ pending owner may be silently reused.
    buffer, runtime, _ = _make_buffer(force=True)
    runtime.fail_dispatch_commit = True
    _expect_raises(lambda: _force_dispatch(buffer))
    assert runtime.trace[:5] == [
        "dispatch_prepare", "gate", "plan_finish", "gate", "dispatch_commit",
    ]
    assert runtime.trace[-1] == "dispatch_abort"
    assert buffer._rail_balance_terminal is True  # type: ignore[attr-defined]

    # Finish owns the host-visible outputs after commit.  Failure there has the
    # same terminal/abort semantics as a commit failure.
    buffer, runtime, _ = _make_buffer(force=True)
    runtime.fail_dispatch_finish = True
    _expect_raises(lambda: _force_dispatch(buffer))
    assert runtime.trace == [
        "dispatch_prepare", "gate", "plan_finish", "gate",
        "dispatch_commit", "dispatch_finish", "dispatch_abort",
    ]
    assert buffer._rail_balance_live_ticket is None  # type: ignore[attr-defined]
    assert buffer._rail_balance_terminal is True  # type: ignore[attr-defined]

    # Combine gate acceptance consumes the shared ticket before a fallible
    # commit.  A commit exception is terminal and cannot resurrect it.
    buffer, runtime, _ = _make_buffer(force=True)
    recv_x, _, recv_weights, handle, _ = _force_dispatch(buffer)
    ticket = handle._rail_balance_ticket  # type: ignore[attr-defined]
    duplicate = copy.copy(handle)
    runtime.trace.clear()
    runtime.fail_combine_commit = True
    _expect_raises(lambda: buffer.combine(
        recv_x, handle, recv_weights, num_sms=4, num_qps=0))
    assert runtime.trace == [
        "combine_prepare", "gate", "combine_commit:CONSUMED",
    ]
    assert _ticket_state(ticket) == "CONSUMED"
    assert buffer._rail_balance_terminal is True  # type: ignore[attr-defined]

    runtime.trace.clear()
    _expect_raises(lambda: buffer.combine(
        recv_x, duplicate, recv_weights, num_sms=4, num_qps=0))
    assert "gate" in runtime.trace
    assert not any(item.startswith("combine_commit") for item in runtime.trace)


def _assert_off_path_never_touches_force_helpers() -> None:
    # This object deliberately has no force fields.  Any public off-path read,
    # gate, ticket allocation, or private runtime call therefore fails loudly.
    buffer, runtime, _ = _make_buffer(force=False)
    x, topk_idx, topk_weights = _inputs()
    recv_x, _, recv_weights, handle, _ = buffer.dispatch(
        x, topk_idx, topk_weights,
        num_experts=8,
        num_max_tokens_per_rank=8,
        expert_alignment=1,
        num_sms=4,
        num_qps=4,
    )
    assert runtime.trace == ["legacy_dispatch"]
    assert not hasattr(handle, "_rail_balance_ticket")
    runtime.trace.clear()
    buffer.combine(recv_x, handle, recv_weights, num_sms=4, num_qps=4)
    assert runtime.trace == ["legacy_combine"]

    # A force ticket from another buffer must be rejected before either
    # legacy method dereferences its routing tensors.
    force_buffer, _, _ = _make_buffer(force=True)
    force_recv_x, _, force_recv_weights, force_handle, _ = \
        _force_dispatch(force_buffer)
    runtime.trace.clear()
    _expect_raises(lambda: _force_dispatch(buffer, handle=force_handle))
    assert runtime.trace == []
    _expect_raises(lambda: buffer.combine(
        force_recv_x, force_handle, force_recv_weights,
        num_sms=4, num_qps=4))
    assert runtime.trace == []


def main() -> None:
    original_gate = elastic_module._run_rail_balance_world_gate
    elastic_module._run_rail_balance_world_gate = _GateController.patched_gate
    try:
        _assert_source_contract()
        _assert_multinode_hybrid_auto_resources_are_bounded()
        _assert_successful_round_trip()
        _assert_one_hop_selects_endpoint_planner()
        _assert_adaptive_does_not_fall_back()
        _assert_zero_move_plan_bypasses_force()
        _assert_gate_rejection_is_retryable()
        _assert_dispatch_entry_failures_are_retryable()
        _assert_combine_prepare_failure_is_retryable()
        _assert_cached_foreign_and_duplicate_reach_the_gate()
        _assert_post_gate_failure_is_terminal()
        _assert_off_path_never_touches_force_helpers()
    finally:
        elastic_module._run_rail_balance_world_gate = original_gate
    print("PASS C080-H6 public rail-balance lifecycle (CPU fake runtime)")


if __name__ == "__main__":
    main()
