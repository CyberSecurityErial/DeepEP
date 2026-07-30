import functools
import os
import math
import torch
import torch.distributed as dist
from typing import Callable, Optional, Tuple, Union, List, Sequence
from contextlib import contextmanager

# noinspection PyUnresolvedReferences
import deep_ep._C as _C
# noinspection PyUnresolvedReferences
from deep_ep._C import EventHandle

from ..utils.event import EventOverlap
from ..utils.math import align
from ..utils.semantic import value_or, weak_lru
from ..utils.envs import (
    check_fast_rdma_atomic_support,
    check_nvlink_connections, check_torch_deterministic,
    get_nvlink_gbs, get_rdma_gbs
)
from ..utils.comm import get_nccl_comm_handle


_RAIL_BALANCE_MODES = (
    'off', 'force', 'legacy_exact', 'one_hop', 'adaptive')
_RAIL_BALANCE_MODE_IDS = {
    'off': 0,
    'force': 1,
    'legacy_exact': 1,
    'one_hop': 2,
    'adaptive': 3,
}
_RAIL_BALANCE_POLICIES = ('all', 'active', 'adaptive')
_RAIL_BALANCE_POLICY_IDS = {
    policy: index for index, policy in enumerate(_RAIL_BALANCE_POLICIES)
}
_RAIL_BALANCE_MAX_THRESHOLD_PERCENT = 3100
_RAIL_BALANCE_MAX_PROXY_SLOTS = (1 << 31) - 1
_RAIL_BALANCE_FORCE_HOST_AVAILABLE = False
_RAIL_BALANCE_WORLD_GATE_WORDS = 128
_RAIL_BALANCE_WORLD_GATE_MAX_FIELDS = 63
_RAIL_BALANCE_WORLD_GATE_INT64_MAX = (1 << 63) - 1
_RAIL_BALANCE_WORLD_GATE_INT32_MAX = (1 << 31) - 1
_RAIL_BALANCE_BUFFER_ALIGNMENT = 2 * 1024 * 1024
_RAIL_BALANCE_WORLD_GATE_RANK_MASK = (1 << 32) - 1
_RAIL_BALANCE_WORLD_GATE_MAX_ERROR_PRIORITY = (1 << 31) - 1
_RAIL_BALANCE_PROTOCOL_MAGIC = int.from_bytes(b'RBH6', byteorder='little')
_RAIL_BALANCE_PROTOCOL_VERSION = 2
_RAIL_BALANCE_OPERATION_CONSTRUCTOR = 1
_RAIL_BALANCE_OPERATION_DISPATCH = 2
_RAIL_BALANCE_OPERATION_COMBINE = 3
_RAIL_BALANCE_PHASE_PREPARE = 1
_RAIL_BALANCE_PHASE_SIZING = 2
_RAIL_BALANCE_PHASE_PLAN = 2
_RAIL_BALANCE_CONSTRUCTOR_LAYOUT_FIELDS = 10
_RAIL_BALANCE_CONSTRUCTOR_MANIFEST_WIDTH = 18 + _RAIL_BALANCE_CONSTRUCTOR_LAYOUT_FIELDS
_RAIL_BALANCE_CONSTRUCTOR_SIZING_MANIFEST_WIDTH = 20
_RAIL_BALANCE_CONSTRUCTOR_PARSE_ERROR = 10
_RAIL_BALANCE_CONSTRUCTOR_VALIDATION_ERROR = 11
_RAIL_BALANCE_CONSTRUCTOR_CAPABILITY_ERROR = 12
_RAIL_BALANCE_CONSTRUCTOR_LAYOUT_ERROR = 13
_RAIL_BALANCE_CONSTRUCTOR_MANIFEST_ERROR = 14
_RAIL_BALANCE_CONSTRUCTOR_ENCODE_ERROR = 15
_RAIL_BALANCE_CONSTRUCTOR_SIZING_ERROR = 17
_RAIL_BALANCE_CONSTRUCTOR_RUNTIME_CONFIG_ERROR = 18
_RAIL_BALANCE_CONSTRUCTOR_SIZING_MANIFEST_ERROR = 19
_RAIL_BALANCE_CONSTRUCTOR_SIZING_ENCODE_ERROR = 20
_RAIL_BALANCE_DISPATCH_COMMON_FIELDS = 21
_RAIL_BALANCE_DISPATCH_MANIFEST_WIDTH = 7 + \
    _RAIL_BALANCE_DISPATCH_COMMON_FIELDS
_RAIL_BALANCE_COMBINE_MANIFEST_WIDTH = 7
_RAIL_BALANCE_DISPATCH_VALIDATION_ERROR = 30
_RAIL_BALANCE_DISPATCH_PREPARE_ERROR = 31
_RAIL_BALANCE_DISPATCH_PREPARE_STATUS_ERROR = 32
_RAIL_BALANCE_DISPATCH_MANIFEST_ERROR = 33
_RAIL_BALANCE_DISPATCH_ENCODE_ERROR = 34
_RAIL_BALANCE_DISPATCH_PLAN_ERROR = 35
_RAIL_BALANCE_DISPATCH_PLAN_STATUS_ERROR = 36
_RAIL_BALANCE_DISPATCH_PLAN_ENCODE_ERROR = 37
# Priority one is below every real error and doubles as a world-wide OR bit.
_RAIL_BALANCE_DISPATCH_HAS_MOVES = 1
_RAIL_BALANCE_ZERO_MOVE_RECHECK_INTERVAL = 32
_RAIL_BALANCE_COMBINE_VALIDATION_ERROR = 40
_RAIL_BALANCE_COMBINE_PREPARE_ERROR = 41
_RAIL_BALANCE_COMBINE_MANIFEST_ERROR = 42
_RAIL_BALANCE_COMBINE_ENCODE_ERROR = 43


def _rail_balance_error(code: str, detail: str) -> str:
    return f'[DeepEP rail_balance:{code}] {detail}'


def _parse_rail_balance_config(
        mode: str,
        proxy_slots_per_rank: int,
        policy: str = 'all',
        threshold_percent: int = 0) -> Tuple[str, int, int, int]:
    """Validate the constructor-fixed rail-balance mode without touching CUDA or collectives."""
    if type(mode) is not str or mode not in _RAIL_BALANCE_MODES:
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            f"rail_balance must be exactly one of {_RAIL_BALANCE_MODES}, got {mode!r}"))
    if type(proxy_slots_per_rank) is not int or not (
            0 <= proxy_slots_per_rank <= _RAIL_BALANCE_MAX_PROXY_SLOTS):
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            'rail_balance_proxy_slots_per_rank must be an integer in '
            f'[0, {_RAIL_BALANCE_MAX_PROXY_SLOTS}]'))
    if type(policy) is not str or policy not in _RAIL_BALANCE_POLICIES:
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            'rail_balance_policy must be exactly one of '
            f'{_RAIL_BALANCE_POLICIES}, got {policy!r}'))
    if type(threshold_percent) is not int or not (
            0 <= threshold_percent <= _RAIL_BALANCE_MAX_THRESHOLD_PERCENT):
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            'rail_balance_threshold_percent must be an integer in '
            f'[0, {_RAIL_BALANCE_MAX_THRESHOLD_PERCENT}]'))
    if mode == 'off' and proxy_slots_per_rank != 0:
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            "rail_balance_proxy_slots_per_rank must be 0 when rail_balance='off'"))
    if mode != 'off' and proxy_slots_per_rank == 0:
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            'rail_balance_proxy_slots_per_rank must be positive when rail balancing is enabled'))
    if mode == 'off' and policy != 'all':
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            "rail_balance_policy must be 'all' when rail_balance='off'"))
    if mode == 'off' and threshold_percent != 0:
        raise ValueError(_rail_balance_error(
            'InvalidConfiguration',
            'rail_balance_threshold_percent must be 0 when '
            "rail_balance='off'"))
    return (mode, proxy_slots_per_rank,
            _RAIL_BALANCE_POLICY_IDS[policy], threshold_percent)


def _validate_rail_balance_force_constructor(
        num_bytes: Optional[int], num_cpu_bytes: int,
        num_max_tokens_per_rank: int, hidden: int, num_topk: int,
        use_fp8_dispatch: bool, deterministic: bool,
        allow_hybrid_mode: bool, allow_multiple_reduction: bool) -> None:
    """Validate the deliberately narrow force-v1 constructor contract."""
    invalid_reason = None
    if num_bytes is not None:
        invalid_reason = 'manual num_bytes is not supported'
    elif type(num_cpu_bytes) is not int or num_cpu_bytes != 0:
        invalid_reason = 'num_cpu_bytes must be 0'
    elif type(num_max_tokens_per_rank) is not int or num_max_tokens_per_rank <= 0:
        invalid_reason = 'num_max_tokens_per_rank must be a positive integer'
    elif type(hidden) is not int or hidden <= 0 or hidden % 256 != 0:
        invalid_reason = 'hidden must be a positive multiple of 256'
    elif type(num_topk) is not int or not (1 <= num_topk <= 32):
        invalid_reason = 'num_topk must be an integer in [1, 32]'
    elif type(use_fp8_dispatch) is not bool:
        invalid_reason = 'use_fp8_dispatch must be a bool'
    elif use_fp8_dispatch:
        invalid_reason = 'FP8 dispatch is not supported'
    elif type(deterministic) is not bool:
        invalid_reason = 'deterministic must be a bool'
    elif deterministic:
        invalid_reason = 'deterministic mode is not supported'
    elif type(allow_hybrid_mode) is not bool:
        invalid_reason = 'allow_hybrid_mode must be a bool'
    elif not allow_hybrid_mode:
        invalid_reason = 'allow_hybrid_mode must be true'
    elif type(allow_multiple_reduction) is not bool:
        invalid_reason = 'allow_multiple_reduction must be a bool'
    elif not allow_multiple_reduction:
        invalid_reason = 'allow_multiple_reduction must be true'

    if invalid_reason is not None:
        raise ValueError(_rail_balance_error(
            'UnsupportedConfiguration', f'force-v1: {invalid_reason}'))


def _validate_rail_balance_force_runtime_config(
        sl_idx: int, num_allocated_qps: int,
        num_cpu_timeout_secs: int, num_gpu_timeout_secs: int,
        prefer_overlap_with_compute: bool,
        explicitly_destroy: bool) -> None:
    """Reject values that could fail Python-to-C++ conversion after consensus."""
    invalid_reason = None
    if type(sl_idx) is not int or not (
            0 <= sl_idx <= _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
        invalid_reason = 'sl_idx must be a nonnegative int32'
    elif type(num_allocated_qps) is not int or not (
            1 <= num_allocated_qps <= _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
        invalid_reason = 'num_allocated_qps must be a positive int32'
    elif type(num_cpu_timeout_secs) is not int or not (
            0 <= num_cpu_timeout_secs <= _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
        invalid_reason = 'num_cpu_timeout_secs must be a nonnegative int32'
    elif type(num_gpu_timeout_secs) is not int or not (
            0 <= num_gpu_timeout_secs <= _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
        invalid_reason = 'num_gpu_timeout_secs must be a nonnegative int32'
    elif type(prefer_overlap_with_compute) is not bool:
        invalid_reason = 'prefer_overlap_with_compute must be a bool'
    elif type(explicitly_destroy) is not bool:
        invalid_reason = 'explicitly_destroy must be a bool'

    if invalid_reason is not None:
        raise ValueError(_rail_balance_error(
            'UnsupportedConfiguration', f'force-v1: {invalid_reason}'))


def _rail_balance_force_available() -> bool:
    """Fail closed across old extensions and partial Python/C++ installations."""
    if not _RAIL_BALANCE_FORCE_HOST_AVAILABLE:
        return False
    compiled_capability = getattr(_C, '_rail_balance_force_available', None)
    if not callable(compiled_capability):
        return False
    try:
        return compiled_capability() is True
    except Exception:
        return False


def _make_rail_balance_world_gate_error_key(priority: int, rank: int) -> int:
    """Encode an error so MAX selects higher priority, then the lowest rank."""
    if type(priority) is not int or not (
            0 <= priority <= _RAIL_BALANCE_WORLD_GATE_MAX_ERROR_PRIORITY):
        raise ValueError('rail-balance WORLD gate error priority is out of range')
    if type(rank) is not int or not (
            0 <= rank <= _RAIL_BALANCE_WORLD_GATE_RANK_MASK):
        raise ValueError('rail-balance WORLD gate error rank is out of range')
    if priority == 0:
        return 0
    return (priority << 32) | (_RAIL_BALANCE_WORLD_GATE_RANK_MASK - rank)


def _decode_rail_balance_world_gate_error_key(error_key: int) -> Tuple[int, int]:
    """Return ``(priority, rank)`` for a nonnegative WORLD-gate error key."""
    if type(error_key) is not int or not (
            0 <= error_key <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX):
        raise ValueError('rail-balance WORLD gate error key is out of range')
    if error_key == 0:
        return 0, -1
    return (error_key >> 32,
            _RAIL_BALANCE_WORLD_GATE_RANK_MASK -
            (error_key & _RAIL_BALANCE_WORLD_GATE_RANK_MASK))


def _validate_rail_balance_world_gate_storage(
        device_words: torch.Tensor, host_words: torch.Tensor) -> None:
    """Validate the fixed storage once, before it enters a force transaction."""
    if device_words.dtype != torch.int64 or device_words.device.type != 'cuda' or \
            tuple(device_words.shape) != (_RAIL_BALANCE_WORLD_GATE_WORDS,) or \
            not device_words.is_contiguous():
        raise ValueError(
            'rail-balance WORLD gate device storage must be contiguous '
            'CUDA int64[128]')
    if host_words.dtype != torch.int64 or host_words.device.type != 'cpu' or \
            tuple(host_words.shape) != (_RAIL_BALANCE_WORLD_GATE_WORDS,) or \
            not host_words.is_contiguous() or not host_words.is_pinned():
        raise ValueError(
            'rail-balance WORLD gate host storage must be contiguous pinned '
            'CPU int64[128]')


def _encode_rail_balance_world_gate(
        host_words: torch.Tensor,
        local_error_key: int,
        common_fields: Sequence[int]) -> None:
    """Encode one checked fixed gate payload into caller-owned host storage."""
    if type(local_error_key) is not int or not (
            0 <= local_error_key <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX):
        raise ValueError('rail-balance WORLD gate error key is out of range')
    if len(common_fields) > _RAIL_BALANCE_WORLD_GATE_MAX_FIELDS:
        raise ValueError('rail-balance WORLD gate has more than 63 fields')

    checked_fields = []
    for value in common_fields:
        if type(value) is not int or not (
                0 <= value <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX):
            raise ValueError(
                'rail-balance WORLD gate fields must be nonnegative int64')
        checked_fields.append(value)

    host_words.zero_()
    host_words[0] = local_error_key
    for index, value in enumerate(checked_fields):
        host_words[2 + 2 * index] = value
        host_words[3 + 2 * index] = -value


def _patch_rail_balance_world_gate_prevalidated(
        host_words: torch.Tensor,
        local_error_key: int,
        field_index: int = -1,
        field_value: int = 0) -> None:
    """Patch prepared gate words without field iteration or validation.

    The caller must validate the fixed storage, error key, field index, and
    value before entering the collective transaction.  ``field_index == -1``
    patches only the error word.  This deliberately tiny helper is for the
    Gate1/Gate2 interval where rank-local dynamic validation is unsafe.
    """
    host_words[0] = local_error_key
    if field_index >= 0:
        offset = 2 + 2 * field_index
        host_words[offset] = field_value
        host_words[offset + 1] = -field_value


def _decode_rail_balance_world_gate(
        host_words: torch.Tensor) -> Tuple[int, int, int, int]:
    """Return ``(error_key, field, minimum, maximum)`` without rank-local throws."""
    error_key = int(host_words[0].item())
    if error_key != 0:
        return error_key, -1, 0, 0
    for index in range(_RAIL_BALANCE_WORLD_GATE_MAX_FIELDS):
        maximum = int(host_words[2 + 2 * index].item())
        minimum = -int(host_words[3 + 2 * index].item())
        if minimum != maximum:
            return 0, index, minimum, maximum
    return 0, -1, 0, 0


def _run_rail_balance_world_gate(
        device_words: torch.Tensor,
        host_words: torch.Tensor,
        group: dist.ProcessGroup) -> Tuple[int, int, int, int]:
    """Run one prevalidated fixed-tensor WORLD consensus on the caller stream."""
    device_words.copy_(host_words, non_blocking=True)
    dist.all_reduce(device_words, op=dist.ReduceOp.MAX, group=group)
    host_words.copy_(device_words, non_blocking=True)
    torch.cuda.current_stream(device_words.device).synchronize()
    return _decode_rail_balance_world_gate(host_words)


def _make_rail_balance_constructor_manifest(
        mode: str,
        world_size: int,
        num_max_tokens_per_rank: int = 0,
        hidden: int = 0,
        num_topk: int = 0,
        proxy_slots_per_rank: int = 0,
        use_fp8_dispatch: bool = False,
        deterministic: bool = False,
        allow_hybrid_mode: bool = False,
        allow_multiple_reduction: bool = False,
        arena_layout: Sequence[int] = (),
        policy: int = 0,
        threshold_percent: int = 0) -> Tuple[int, ...]:
    """Encode only fields that must agree before the symmetric window exists."""
    if type(mode) is not str or mode not in _RAIL_BALANCE_MODES:
        raise ValueError('rail-balance constructor manifest mode is invalid')
    if type(policy) is not int or policy not in \
            _RAIL_BALANCE_POLICY_IDS.values():
        raise ValueError('rail-balance constructor policy ABI is invalid')
    if type(threshold_percent) is not int or not (
            0 <= threshold_percent <= _RAIL_BALANCE_MAX_THRESHOLD_PERCENT):
        raise ValueError('rail-balance constructor threshold is invalid')
    mode_id = _RAIL_BALANCE_MODE_IDS[mode]
    if mode_id:
        if len(arena_layout) != _RAIL_BALANCE_CONSTRUCTOR_LAYOUT_FIELDS or not all(
                type(value) is int and value >= 0 for value in arena_layout):
            raise ValueError('rail-balance constructor arena layout ABI is invalid')
        force_fields = (
            num_max_tokens_per_rank,
            hidden,
            num_topk,
            proxy_slots_per_rank,
            int(use_fp8_dispatch),
            int(deterministic),
            int(allow_hybrid_mode),
            int(allow_multiple_reduction),
            policy,
            threshold_percent,
            len(arena_layout),
        )
        layout_fields = tuple(arena_layout)
    else:
        force_fields = (0,) * 11
        layout_fields = (0,) * _RAIL_BALANCE_CONSTRUCTOR_LAYOUT_FIELDS

    fields = (
        _RAIL_BALANCE_PROTOCOL_MAGIC,
        _RAIL_BALANCE_PROTOCOL_VERSION,
        _RAIL_BALANCE_OPERATION_CONSTRUCTOR,
        _RAIL_BALANCE_PHASE_PREPARE,
        _RAIL_BALANCE_CONSTRUCTOR_MANIFEST_WIDTH,
        mode_id,
        world_size,
        *force_fields,
        *layout_fields,
    )
    if len(fields) != _RAIL_BALANCE_CONSTRUCTOR_MANIFEST_WIDTH:
        raise RuntimeError('rail-balance constructor manifest width is invalid')
    if not all(type(value) is int and
               0 <= value <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX
               for value in fields):
        raise ValueError(
            'rail-balance constructor manifest fields must be nonnegative int64')
    return fields


def _make_rail_balance_constructor_sizing_manifest(
        world_size: int,
        num_max_tokens_per_rank: int,
        hidden: int,
        num_topk: int,
        proxy_slots_per_rank: int,
        legacy_bytes: int,
        arena_bytes: int,
        total_bytes: int,
        sl_idx: int,
        num_allocated_qps: int,
        num_cpu_timeout_secs: int,
        num_gpu_timeout_secs: int,
        prefer_overlap_with_compute: bool,
        explicitly_destroy: bool) -> Tuple[int, ...]:
    """Encode the force facts resolved only after a common NCCL comm exists."""
    fields = (
        _RAIL_BALANCE_PROTOCOL_MAGIC,
        _RAIL_BALANCE_PROTOCOL_VERSION,
        _RAIL_BALANCE_OPERATION_CONSTRUCTOR,
        _RAIL_BALANCE_PHASE_SIZING,
        _RAIL_BALANCE_CONSTRUCTOR_SIZING_MANIFEST_WIDTH,
        1,
        world_size,
        num_max_tokens_per_rank,
        hidden,
        num_topk,
        proxy_slots_per_rank,
        legacy_bytes,
        arena_bytes,
        total_bytes,
        sl_idx,
        num_allocated_qps,
        num_cpu_timeout_secs,
        num_gpu_timeout_secs,
        int(prefer_overlap_with_compute),
        int(explicitly_destroy),
    )
    if len(fields) != _RAIL_BALANCE_CONSTRUCTOR_SIZING_MANIFEST_WIDTH:
        raise RuntimeError('rail-balance constructor sizing manifest width is invalid')
    if not all(type(value) is int and
               0 <= value <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX
               for value in fields):
        raise ValueError(
            'rail-balance constructor sizing fields must be nonnegative int64')
    return fields


def _make_rail_balance_operation_manifest(
        operation: int, phase: int, world_size: int, invocation_id: int,
        common_fields: Sequence[int] = ()) -> Tuple[int, ...]:
    """Encode one fixed dispatch/combine transaction identity and geometry."""
    if operation == _RAIL_BALANCE_OPERATION_DISPATCH:
        expected_common_fields = _RAIL_BALANCE_DISPATCH_COMMON_FIELDS
        expected_width = _RAIL_BALANCE_DISPATCH_MANIFEST_WIDTH
        if phase not in (_RAIL_BALANCE_PHASE_PREPARE,
                         _RAIL_BALANCE_PHASE_PLAN):
            raise ValueError('rail-balance dispatch phase is invalid')
    elif operation == _RAIL_BALANCE_OPERATION_COMBINE:
        expected_common_fields = 0
        expected_width = _RAIL_BALANCE_COMBINE_MANIFEST_WIDTH
        if phase != _RAIL_BALANCE_PHASE_PREPARE:
            raise ValueError('rail-balance combine phase is invalid')
    else:
        raise ValueError('rail-balance operation is invalid')
    if len(common_fields) != expected_common_fields:
        raise ValueError('rail-balance operation manifest width is invalid')

    fields = (
        _RAIL_BALANCE_PROTOCOL_MAGIC,
        _RAIL_BALANCE_PROTOCOL_VERSION,
        operation,
        phase,
        expected_width,
        world_size,
        invocation_id,
        *common_fields,
    )
    if len(fields) != expected_width or not all(
            type(value) is int and
            0 <= value <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX
            for value in fields):
        raise ValueError(
            'rail-balance operation manifest fields must be nonnegative int64')
    return fields


def _raise_rail_balance_world_gate_failure(
        stage: str, result: Tuple[int, int, int, int]) -> None:
    """Raise the same compact post-consensus error on every participating rank."""
    error_key, field, minimum, maximum = result
    if error_key != 0:
        priority, rank = _decode_rail_balance_world_gate_error_key(error_key)
        raise RuntimeError(_rail_balance_error(
            'CollectivePreflight',
            f'{stage} rejected rank {rank} with error priority {priority}'))
    if field >= 0:
        raise RuntimeError(_rail_balance_error(
            'ConfigurationMismatch',
            f'{stage} field {field} differs across ranks: min={minimum}, max={maximum}'))


class _RailBalanceForceTicket:
    """Shared mutable one-shot state attached dynamically to an EPHandle."""
    __slots__ = ('owner_token', 'invocation_id', 'state')

    LIVE = 'LIVE'
    PREPARING = 'PREPARING'
    CONSUMED = 'CONSUMED'

    def __init__(self, owner_token: object, invocation_id: int):
        self.owner_token = owner_token
        self.invocation_id = invocation_id
        self.state = self.LIVE


class EPHandle:
    """
    Communication handle returned by `ElasticBuffer.dispatch`.
    Can be reused as a cached handle in subsequent `ElasticBuffer.dispatch` calls to skip layout recomputation,
    and is consumed by `ElasticBuffer.combine` to reverse the token routing.

    Attributes:
        do_expand: whether the expanding (one-token-per-expert-slot) layout is used.
        num_experts: the number of all experts.
        expert_alignment: align the number of tokens received by each local expert to this variable.
        num_max_tokens_per_rank: the maximum number of tokens per rank, all the ranks must hold the same value.
        num_sms: the SM count used during dispatch (reused in combine).
        topk_idx: cloned top-k expert indices from dispatch, `[num_tokens, num_topk]`.
        psum_num_recv_tokens_per_scaleup_rank: inclusive prefix sum of deduplicated received token counts
            per scaleup rank, shape `[num_scaleup_ranks]`. A token is counted once per rank even if
            multiple of its top-k experts land on the same rank. The last element equals the total number
            of received tokens.
        psum_num_recv_tokens_per_expert: prefix sum of alignment-padded received token counts per local
            expert, shape `[num_local_experts]`. Each expert's count is padded to `expert_alignment`.
            In non-expand mode, this is the inclusive prefix sum. In expand mode, `psum[i]` equals
            the aligned cumulative count of experts before `i` plus the actual (unaligned) token count
            of expert `i` — so `psum[i] - align(psum[i-1], expert_alignment)` recovers the real
            count for expert `i`, and `align(psum[i], expert_alignment)` gives expert `i+1`'s
            starting offset.
        num_recv_tokens_per_expert_list: Python list of per-expert received token counts (CPU-side).
        num_unaligned_recv_tokens_per_expert: the actual (unaligned) number of tokens received per local
            expert, shape `[num_local_experts]` with `torch.int`. Only populated in expand mode.
        recv_src_metadata: source token indices and buffer slot indices.
        dst_buffer_slot_idx: destination buffer slot indices from dispatch.
        token_metadata_at_forward: per-channel forwarded token metadata (hybrid mode only).
        channel_linked_list: per-channel per-scaleup-peer linked list (hybrid mode only).
        num_recv_tokens: the total number of received tokens.
    """

    def __init__(self,
                 do_expand: bool,
                 num_experts: int, expert_alignment: int,
                 num_max_tokens_per_rank: int,
                 num_sms: int,
                 topk_idx: torch.Tensor,
                 num_recv_tokens: int,
                 num_expanded_tokens: int,
                 num_recv_tokens_per_expert_list: list,
                 psum_num_recv_tokens_per_scaleup_rank: torch.Tensor,
                 psum_num_recv_tokens_per_expert: torch.Tensor,
                 num_unaligned_recv_tokens_per_expert: torch.Tensor,
                 recv_src_metadata: torch.Tensor,
                 dst_buffer_slot_idx: torch.Tensor,
                 token_metadata_at_forward: Optional[torch.Tensor],
                 channel_linked_list: Optional[torch.Tensor]):
        # NOTES: remember to copy the original users' input to prevent uncasual modifications on them
        assert topk_idx is not None

        self.do_expand = do_expand
        self.num_experts = num_experts
        self.expert_alignment = expert_alignment
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.topk_idx = topk_idx
        self.psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank
        self.psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert
        self.num_unaligned_recv_tokens_per_expert = num_unaligned_recv_tokens_per_expert
        self.num_recv_tokens_per_expert_list = num_recv_tokens_per_expert_list
        self.recv_src_metadata = recv_src_metadata
        self.dst_buffer_slot_idx = dst_buffer_slot_idx
        self.token_metadata_at_forward = token_metadata_at_forward
        self.channel_linked_list = channel_linked_list

        # May not be accurate without CPU sync
        self.num_recv_tokens = num_recv_tokens
        self.num_expanded_tokens = num_expanded_tokens

        # For deterministic features
        self.cached_recv_src_metadata_before_sort = None

    def deterministic_sort(self,
                           do_cpu_sync: bool,
                           is_cached_dispatch: bool,
                           recv_x: torch.Tensor,
                           recv_sf: Optional[torch.Tensor],
                           recv_topk_idx: torch.Tensor,
                           recv_topk_weights: torch.Tensor,
                           channel_linked_list: Optional[torch.Tensor]):
        """
        Sort received tokens to guarantee deterministic dispatch output.
        The principle:
          - Non-expand mode: sort everything that depends on the receive order, including
            `recv_x`, `recv_sf`, `recv_topk_weights`, `recv_topk_idx`, and `self.recv_src_metadata`
            (`recv_src_metadata` is sorted only for non-cached dispatch, since it is not regenerated in cached mode).
          - Expand mode: only sort the expanded arrays — `recv_x`, `recv_sf`, and `recv_topk_weights`.
            The slot pointers in `self.recv_src_metadata[:, 2:]` are updated to reflect the new positions, but `self.recv_src_metadata` itself is not permuted.
        """

        # NOTE: `self.recv_src_metadata` is generated once during non-cached dispatch and is not
        # regenerated during cached dispatch (applies to both expand and non-expand mode). So we:
        #  1. Cache it for later sorting
        #  2. Only permute `self.recv_src_metadata` in non-cached mode
        if not is_cached_dispatch:
            self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()
        assert self.cached_recv_src_metadata_before_sort is not None
        sort_keys = self.cached_recv_src_metadata_before_sort[:, 0]

        # Ignore trailing tokens by setting their `sort_keys` to max
        num_recv_tokens = self.psum_num_recv_tokens_per_scaleup_rank[-1] if not do_cpu_sync else self.recv_src_metadata.shape[0]
        if not do_cpu_sync:
            oob_tokens_mask = torch.arange(0, self.recv_src_metadata.shape[0], device=self.recv_src_metadata.device) >= num_recv_tokens
            sort_keys = sort_keys.clone()
            sort_keys[oob_tokens_mask] = torch.iinfo(sort_keys.dtype).max
        orig_indices = torch.sort(sort_keys).indices

        def get_reverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            assert perm.dim() == 1
            result = torch.empty_like(perm)
            result[perm] = torch.arange(0, perm.shape[0], dtype=perm.dtype, device=perm.device)
            return result

        def permute(tensor: Optional[torch.Tensor], orig_indices: torch.Tensor):
            if tensor is not None:
                tmp = tensor[orig_indices]
                tensor.copy_(tmp)

        if not self.do_expand:
            # Non-expand mode
            # If cached dispatch is enabled, the `dispatch` kernel stores values according to `dst_buffer_slot_idx`, and the `dispatch_copy_epilogue_impl` kernel writes the info of token i into the i-th slot
            permute(recv_x, orig_indices)
            permute(recv_sf, orig_indices)
            permute(recv_topk_weights, orig_indices)
            permute(recv_topk_idx, orig_indices)
            if not is_cached_dispatch:
                permute(self.recv_src_metadata, orig_indices)

            if not is_cached_dispatch and channel_linked_list is not None:
                valid_mask = (channel_linked_list >= 0) & (channel_linked_list < num_recv_tokens)
                to_indices = get_reverse_permutation(orig_indices)
                channel_linked_list[valid_mask] = to_indices[channel_linked_list[valid_mask]].to(channel_linked_list.dtype)

        elif not is_cached_dispatch:
            # Expand mode. In cached mode the copy epilogue places tokens according to
            # `self.recv_src_metadata[:, 2:]`, so we only need to permute when `is_cached_dispatch` is `False`.
            # In expand mode, `recv_x`, `recv_sf`, and `recv_topk_weights` are grouped by expert ID, possibly with padding (expert alignment). We permute tokens within each expert and update `self.recv_src_metadata[:, 2:]` accordingly.

            # Now we're going to construct the sorting key, which is:
            #  - `expert_idx*src_token_global_index_max_x2 + (-src_token_global_index_max) + src_token_global_idx`, for valid tokens
            #  - `expert_idx * src_token_global_index_max_x2`, for padding slots
            # This guarantees a two-key sort: first by expert, then by order within each expert.
            # Valid tokens precede padding tokens, and valid tokens are sorted by `src_token_global_idx`.
            src_token_global_index_max_x2 = 10000000000    # 1e10
            tensor_dim0_after_expand = recv_x.shape[0]

            expert_token_idx_start = self.psum_num_recv_tokens_per_expert - self.num_unaligned_recv_tokens_per_expert
            token_idx2expert_idx = torch.bucketize(torch.arange(tensor_dim0_after_expand, device='cuda'),
                                                   expert_token_idx_start[1:], right=True, out_int32=False)
            sort_keys_for_expanded_tensors = token_idx2expert_idx * src_token_global_index_max_x2

            slots = self.cached_recv_src_metadata_before_sort[:, 2:]    # [num_recv_tokens, topk]
            src_global_idx = self.cached_recv_src_metadata_before_sort[:, 0]
            valid_mask = slots >= 0
            if not do_cpu_sync:
                valid_mask[oob_tokens_mask] = False
            sort_keys_for_expanded_tensors.scatter_add_(0, slots[valid_mask], -src_token_global_index_max_x2//2 + src_global_idx.unsqueeze(1).expand_as(slots)[valid_mask].to(torch.int64))

            orig_indices_for_expanded_tensors = torch.sort(sort_keys_for_expanded_tensors, stable=True).indices.to(torch.int32)
            permute(recv_x, orig_indices_for_expanded_tensors)
            permute(recv_sf, orig_indices_for_expanded_tensors)
            permute(recv_topk_weights, orig_indices_for_expanded_tensors)

            to_indices_for_expanded_tensors = get_reverse_permutation(orig_indices_for_expanded_tensors)
            self.recv_src_metadata[:, 2:][valid_mask] = to_indices_for_expanded_tensors[self.recv_src_metadata[:, 2:][valid_mask]]


class ElasticBuffer:
    """
    The elastic communication buffer, which supports:
        - high-throughput expert-parallel all-to-all (dispatch and combine, using NVLink and/or RDMA)
        - Engram (remote KV cache fetch, using RDMA)
        - pipeline-parallel send/recv (PP, using NVLink)
        - all-gather reduce-scatter (AGRS, using NVLink)
    "Elastic" refers to the flexibility of underlying memory: currently GPU-only, with CPU and mixed
        (GPU+CPU) backends on the roadmap

    Attributes:
        group: the communication group.
        rank_idx: the rank index.
        num_ranks: the number of ranks in the group.
        allow_hybrid_mode: whether to enable hybrid mode for multi-node communication. Hybrid mode uses
            hierarchical RDMA + NVLink communication to achieve higher bandwidth, and is more friendly
            to multi-plane/multi-rail networks.
        allow_multiple_reduction: whether to allow multiple reductions in combine. If disabled,
            only one reduction will be done in the combine epilogue for best precision,
            but it may increase data transfer size.
        prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            If enabled, we tend to use fewer SMs.
        num_bytes: the total buffer size in bytes.
        num_max_tokens_per_rank: the default maximum tokens per rank.
        num_scaleout_ranks: the number of scaleout ranks.
        num_scaleup_ranks: the number of scaleup ranks.
        scaleout_rank_idx: the scaleout rank index of this rank.
        scaleup_rank_idx: the scaleup rank index of this rank.
        num_rdma_ranks: the number of physical RDMA ranks.
        num_nvlink_ranks: the number of physical NVLink ranks.
        runtime: the C++ runtime.
    """

    def __init__(self,
                 group: dist.ProcessGroup,
                 # Provide `num_bytes` (GPU + CPU buffer, excludes workspace)
                 num_bytes: Optional[int] = None,
                 num_cpu_bytes: int = 0,
                 # Or provide MoE settings (BF16 by default)
                 num_max_tokens_per_rank: int = 0,
                 hidden: int = 0,
                 num_topk: int = 0,
                 use_fp8_dispatch: bool = False,
                 # Configs
                 deterministic: bool = False,
                 allow_hybrid_mode: bool = True,
                 allow_multiple_reduction: bool = True,
                 prefer_overlap_with_compute: bool = True,
                 sl_idx: int = 3,
                 num_allocated_qps: int = 0,
                 num_cpu_timeout_secs: int = 300, num_gpu_timeout_secs: int = 100,
                 explicitly_destroy: bool = False,
                 *,
                 rail_balance: str = 'off',
                 rail_balance_proxy_slots_per_rank: int = 0,
                 rail_balance_policy: str = 'all',
                 rail_balance_threshold_percent: int = 0):
        """
        Initialize the elastic communication buffer.

        Arguments:
            group: the communication group.
            num_bytes: the total buffer size in bytes (GPU + CPU, excludes workspace), if set, overrides MoE-based calculation.
                Must be aligned to 2 MB (``get_elastic_buffer_alignment()``).
            num_cpu_bytes: the number of CPU buffer bytes (e.g. for Engram storage). Must be aligned to 2 MB.
            num_max_tokens_per_rank: the maximum number of tokens per rank, used for buffer size calculation.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to enable FP8 casting, with this, the received data will be a tuple of FP8 tensor and scaling factors.
            deterministic: whether to use deterministic routing algorithms.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.
            prefer_overlap_with_compute: whether to prefer overlapping communication with compute.
            sl_idx: the RDMA service level index, can be overridden by `EP_OVERRIDE_RDMA_SL` env var.
            num_allocated_qps: the number of QPs to allocate for RDMA (0 for automatic).
            num_cpu_timeout_secs: CPU-side timeout in seconds for CPU sync.
            num_gpu_timeout_secs: GPU-side timeout in seconds for GPU operations.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
            rail_balance: experimental source-side rail-balancing mode. ``'off'``
                preserves DeepEP; ``'force'``/``'legacy_exact'`` use the original
                all-Rail prototype; ``'one_hop'`` restricts egress to endpoint
                Rails; ``'adaptive'`` is reserved for bounded selective 2-hop.
                The default ``'off'`` path preserves the legacy buffer sizing, runtime arguments,
                JIT specialization, handles, and results; only local configuration parsing and
                mode guards are added.
            rail_balance_proxy_slots_per_rank: moved-copy capacity reserved on each egress rank.
                Must be zero for ``'off'`` and positive otherwise.
            rail_balance_policy: planner policy: ``'all'`` balances across every
                local rail, ``'active'`` keeps the original nonempty rail set,
                and ``'adaptive'`` expands that set only when each added rail
                clears ``rail_balance_threshold_percent``.
            rail_balance_threshold_percent: tolerated integer percentage by
                which the current peak may exceed the selected-set balanced
                target. Zero disables the gate and preserves the original
                exact ``'all'`` plan.
        """
        rail_balance_arena_layout = None
        rail_balance_policy_id = 0
        constructor_gate_device_words = None
        constructor_gate_host_words = None
        if not _RAIL_BALANCE_FORCE_HOST_AVAILABLE:
            # Development/partial-install fail-close path. Once the public host
            # protocol is enabled, every mode instead joins the universal gate
            # below so a valid off rank cannot diverge from a valid force rank.
            (rail_balance, rail_balance_proxy_slots_per_rank,
             rail_balance_policy_id,
             rail_balance_threshold_percent) = \
                _parse_rail_balance_config(
                    rail_balance, rail_balance_proxy_slots_per_rank,
                    rail_balance_policy,
                    rail_balance_threshold_percent)
            if rail_balance != 'off':
                _validate_rail_balance_force_constructor(
                    num_bytes, num_cpu_bytes,
                    num_max_tokens_per_rank, hidden, num_topk,
                    use_fp8_dispatch, deterministic,
                    allow_hybrid_mode, allow_multiple_reduction)
                if not _rail_balance_force_available():
                    raise RuntimeError(_rail_balance_error(
                        'FeatureUnavailable',
                        'force-v1 remains disabled until truthful D>1 Rail/Gin correctness passes'))
        else:
            # This is intentionally before get_nccl_comm_handle, buffer sizing,
            # and window registration. Off participates once, then keeps no
            # rail-balance field or storage after construction.
            constructor_rank = group.rank()
            constructor_world_size = group.size()
            constructor_error_priority = 0
            constructor_manifest = None
            try:
                (rail_balance, rail_balance_proxy_slots_per_rank,
                 rail_balance_policy_id,
                 rail_balance_threshold_percent) = \
                    _parse_rail_balance_config(
                        rail_balance, rail_balance_proxy_slots_per_rank,
                        rail_balance_policy,
                        rail_balance_threshold_percent)
            except BaseException:
                rail_balance = 'off'
                rail_balance_proxy_slots_per_rank = 0
                rail_balance_policy_id = 0
                rail_balance_threshold_percent = 0
                constructor_error_priority = \
                    _RAIL_BALANCE_CONSTRUCTOR_PARSE_ERROR

            if constructor_error_priority == 0 and rail_balance != 'off':
                try:
                    _validate_rail_balance_force_constructor(
                        num_bytes, num_cpu_bytes,
                        num_max_tokens_per_rank, hidden, num_topk,
                        use_fp8_dispatch, deterministic,
                        allow_hybrid_mode, allow_multiple_reduction)
                except BaseException:
                    constructor_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_VALIDATION_ERROR

            if constructor_error_priority == 0 and rail_balance != 'off' and \
                    not _rail_balance_force_available():
                constructor_error_priority = \
                    _RAIL_BALANCE_CONSTRUCTOR_CAPABILITY_ERROR

            if constructor_error_priority == 0 and rail_balance != 'off':
                try:
                    rail_balance_arena_layout = tuple(
                        _C._get_rail_balance_hybrid_layout(
                            hidden, num_topk,
                            rail_balance_proxy_slots_per_rank))
                except BaseException:
                    rail_balance_arena_layout = None
                    constructor_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_LAYOUT_ERROR

            if constructor_error_priority == 0:
                try:
                    constructor_manifest = \
                        _make_rail_balance_constructor_manifest(
                            rail_balance, constructor_world_size,
                            num_max_tokens_per_rank, hidden, num_topk,
                            rail_balance_proxy_slots_per_rank,
                            use_fp8_dispatch, deterministic,
                            allow_hybrid_mode, allow_multiple_reduction,
                            rail_balance_arena_layout or (),
                            policy=rail_balance_policy_id,
                            threshold_percent=
                                rail_balance_threshold_percent)
                except BaseException:
                    constructor_manifest = None
                    constructor_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_MANIFEST_ERROR

            constructor_gate_device_words = torch.empty(
                _RAIL_BALANCE_WORLD_GATE_WORDS,
                dtype=torch.int64, device='cuda')
            constructor_gate_host_words = torch.empty(
                _RAIL_BALANCE_WORLD_GATE_WORDS,
                dtype=torch.int64, device='cpu', pin_memory=True)
            _validate_rail_balance_world_gate_storage(
                constructor_gate_device_words, constructor_gate_host_words)
            if constructor_manifest is None:
                # The error key wins before any field comparison. Keep this
                # fallback independent of every caller-controlled value.
                constructor_manifest = \
                    _make_rail_balance_constructor_manifest('off', 0)
            constructor_error_key = _make_rail_balance_world_gate_error_key(
                constructor_error_priority, constructor_rank)
            try:
                _encode_rail_balance_world_gate(
                    constructor_gate_host_words,
                    constructor_error_key,
                    constructor_manifest)
            except BaseException:
                constructor_error_key = \
                    _make_rail_balance_world_gate_error_key(
                        _RAIL_BALANCE_CONSTRUCTOR_ENCODE_ERROR,
                        constructor_rank)
                _encode_rail_balance_world_gate(
                    constructor_gate_host_words,
                    constructor_error_key,
                    _make_rail_balance_constructor_manifest('off', 0))
            constructor_gate_result = _run_rail_balance_world_gate(
                constructor_gate_device_words,
                constructor_gate_host_words,
                group)
            _raise_rail_balance_world_gate_failure(
                'constructor', constructor_gate_result)

        # Some useful utilities
        self.group = group
        self.rank_idx = group.rank()
        self.num_ranks = group.size()
        self.allow_hybrid_mode = allow_hybrid_mode
        self.allow_multiple_reduction = allow_multiple_reduction
        self.prefer_overlap_with_compute = prefer_overlap_with_compute
        self.deterministic = deterministic

        if os.environ.get('NCCL_GIN_CROSS_NIC') == '0':
            # TODO: move this variable into NCCL runtime
            # Multi-plane: all ranks share CPU segments, skip proxy re-export for sysmem handles
            os.environ.setdefault('NCCL_SYM_REUSE_SYSMEM_HANDLES', '1')

        # For extreme large buffer size, we have to enlarge the NCCL VA space
        if num_cpu_bytes > 0:
            assert num_bytes is not None
            num_gpu_bytes = num_bytes - num_cpu_bytes
            num_max_local_ranks = int(os.getenv('EP_NUM_MAX_LOCAL_RANKS', 16)) if allow_hybrid_mode else 1

            # Add 4 GiB of slack for the workspace
            num_registered_bytes = num_gpu_bytes + num_cpu_bytes * num_max_local_ranks + (1 << 32)
            num_total_gpu_bytes = torch.cuda.get_device_properties('cuda').total_memory
            if num_registered_bytes > num_total_gpu_bytes:
                # NCCL aligns the stride up to 4 GiB internally.
                win_stride = align(num_registered_bytes, 1 << 32)
                # TODO: setting the window stride via an env var is fragile. Replace this once
                # NCCL exposes a better way to configure the symmetric window stride.
                os.environ['NCCL_WIN_STRIDE'] = str(win_stride)

        # Create NCCL comm handle
        self.nccl_comm_handle = get_nccl_comm_handle(group, force_new_comm=num_cpu_bytes > 0)

        legacy_num_bytes = 0
        rail_balance_arena_bytes = 0
        force_runtime = None
        if rail_balance != 'off' and _RAIL_BALANCE_FORCE_HOST_AVAILABLE:
            # Gate 0 made mode and geometry unanimous.  Resolve the remaining
            # rank-local sizing/runtime facts before any symmetric window is
            # registered, then reuse the same fixed storage for one force-only
            # Gate 1.  Failures returned by local helpers still participate in
            # the gate so a healthy peer cannot enter window creation alone.
            sizing_error_priority = 0
            sizing_manifest = None
            force_num_bytes = 0
            force_sl_idx = 0
            force_num_allocated_qps = 0
            force_runtime_args = None
            raw_nccl_comm = 0

            if sizing_error_priority == 0:
                try:
                    raw_nccl_comm = self.nccl_comm_handle.get()
                    legacy_num_bytes = _C.calculate_elastic_buffer_size(
                        raw_nccl_comm,
                        num_max_tokens_per_rank, hidden, num_topk,
                        use_fp8_dispatch,
                        allow_hybrid_mode, allow_multiple_reduction)
                    rail_balance_arena_bytes = rail_balance_arena_layout[-1]
                    force_num_bytes = \
                        _C._calculate_rail_balance_hybrid_buffer_size(
                            raw_nccl_comm,
                            num_max_tokens_per_rank, hidden, num_topk,
                            rail_balance_proxy_slots_per_rank)
                    if type(raw_nccl_comm) is not int or not (
                            0 < raw_nccl_comm <=
                            _RAIL_BALANCE_WORLD_GATE_INT64_MAX) or \
                            not all(type(value) is int and
                               0 < value <= _RAIL_BALANCE_WORLD_GATE_INT64_MAX
                               for value in (
                                   legacy_num_bytes,
                                   rail_balance_arena_bytes,
                                   force_num_bytes)) or \
                            any(value % _RAIL_BALANCE_BUFFER_ALIGNMENT != 0
                                for value in (
                                    legacy_num_bytes,
                                    rail_balance_arena_bytes,
                                    force_num_bytes)) or \
                            force_num_bytes != (
                                legacy_num_bytes + rail_balance_arena_bytes):
                        raise RuntimeError(_rail_balance_error(
                            'InternalInvariant',
                            'force-v1 buffer size must equal legacy bytes '
                            'plus arena bytes'))
                except BaseException:
                    sizing_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_SIZING_ERROR

            if sizing_error_priority == 0:
                try:
                    force_sl_idx = sl_idx
                    if 'EP_OVERRIDE_RDMA_SL' in os.environ:
                        force_sl_idx = int(
                            os.environ['EP_OVERRIDE_RDMA_SL'])

                    force_num_allocated_qps = num_allocated_qps
                    if type(force_num_allocated_qps) is not int:
                        raise ValueError('num_allocated_qps is not an int')
                    if force_num_allocated_qps == 0:
                        force_num_allocated_qps = \
                            65 if check_fast_rdma_atomic_support() else 129
                    _validate_rail_balance_force_runtime_config(
                        force_sl_idx, force_num_allocated_qps,
                        num_cpu_timeout_secs, num_gpu_timeout_secs,
                        prefer_overlap_with_compute,
                        explicitly_destroy)
                    force_runtime_args = (
                        self.rank_idx, self.num_ranks,
                        raw_nccl_comm, [],
                        force_num_bytes, 0,
                        allow_hybrid_mode,
                        allow_multiple_reduction,
                        prefer_overlap_with_compute,
                        force_sl_idx, force_num_allocated_qps,
                        num_cpu_timeout_secs, num_gpu_timeout_secs,
                        explicitly_destroy)
                except BaseException:
                    sizing_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_RUNTIME_CONFIG_ERROR

            if sizing_error_priority == 0:
                try:
                    sizing_manifest = \
                        _make_rail_balance_constructor_sizing_manifest(
                            self.num_ranks,
                            num_max_tokens_per_rank, hidden, num_topk,
                            rail_balance_proxy_slots_per_rank,
                            legacy_num_bytes, rail_balance_arena_bytes,
                            force_num_bytes,
                            force_sl_idx, force_num_allocated_qps,
                            num_cpu_timeout_secs, num_gpu_timeout_secs,
                            prefer_overlap_with_compute,
                            explicitly_destroy)
                except BaseException:
                    sizing_error_priority = \
                        _RAIL_BALANCE_CONSTRUCTOR_SIZING_MANIFEST_ERROR

            if sizing_manifest is None:
                sizing_manifest = \
                    _make_rail_balance_constructor_sizing_manifest(
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        False, False)
            sizing_error_key = _make_rail_balance_world_gate_error_key(
                sizing_error_priority, self.rank_idx)
            try:
                _encode_rail_balance_world_gate(
                    constructor_gate_host_words,
                    sizing_error_key,
                    sizing_manifest)
            except BaseException:
                sizing_error_key = \
                    _make_rail_balance_world_gate_error_key(
                        _RAIL_BALANCE_CONSTRUCTOR_SIZING_ENCODE_ERROR,
                        self.rank_idx)
                _encode_rail_balance_world_gate(
                    constructor_gate_host_words,
                    sizing_error_key,
                    _make_rail_balance_constructor_sizing_manifest(
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        False, False))
            sizing_gate_result = _run_rail_balance_world_gate(
                constructor_gate_device_words,
                constructor_gate_host_words,
                group)
            _raise_rail_balance_world_gate_failure(
                'constructor-sizing', sizing_gate_result)
            force_runtime = _C.ElasticBuffer(*force_runtime_args)

            num_bytes = force_num_bytes
            sl_idx = force_sl_idx
            num_allocated_qps = force_num_allocated_qps
        elif num_bytes is None:
            # NOTES: we allow `num_topk == 0`, as the buffer size can also be calculated by number of ranks (maybe bigger though)
            num_bytes = _C.calculate_elastic_buffer_size(
                self.nccl_comm_handle.get(),
                num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
                allow_hybrid_mode, allow_multiple_reduction)

        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'Initializing EP elastic buffer with {num_bytes} bytes '
                  f'(cpu: {num_cpu_bytes}) at rank EP {group.rank()}/{group.size()}')
        self.num_bytes = num_bytes

        if rail_balance != 'off':
            self._rail_balance_mode = rail_balance
            self._rail_balance_proxy_slots_per_rank = rail_balance_proxy_slots_per_rank
            self._rail_balance_policy = rail_balance_policy_id
            self._rail_balance_threshold_percent = \
                rail_balance_threshold_percent
            self._rail_balance_arena_offset = legacy_num_bytes
            self._rail_balance_arena_bytes = rail_balance_arena_bytes
            self._rail_balance_world_gate_device_words = \
                constructor_gate_device_words
            self._rail_balance_world_gate_host_words = \
                constructor_gate_host_words
            self._rail_balance_owner_token = object()
            self._rail_balance_next_invocation_id = 1
            self._rail_balance_live_ticket = None
            self._rail_balance_terminal = False
            self._rail_balance_zero_move_bypass_budget = 0
            self._rail_balance_zero_move_common_fields = None

        # Store default values
        self.num_max_tokens_per_rank = num_max_tokens_per_rank

        if rail_balance == 'off':
            # Preserve the exact legacy path when rail balancing is disabled.
            check_nvlink_connections(group)

            # RDMA SL
            if 'EP_OVERRIDE_RDMA_SL' in os.environ:
                sl_idx = int(os.environ['EP_OVERRIDE_RDMA_SL'])

            # Automatic maximum QP count allowed
            # TODO(tianr22): revise the QP count in consideration of Engram
            if num_allocated_qps == 0:
                # Hybrid mode will consume more QPs
                # The extra QP is for notify warps
                if self.allow_hybrid_mode:
                    num_allocated_qps = 65 if check_fast_rdma_atomic_support() else 129
                else:
                    num_allocated_qps = 17
        self.num_allocated_qps = num_allocated_qps

        self.explicitly_destroy = explicitly_destroy
        if rail_balance != 'off':
            # The force runtime/window was created immediately after Gate 1.
            self.runtime = force_runtime
        else:
            # Create CPU communicator (exchange POSIX FD handles for CPU segments)
            cpu_comm = []
            if allow_hybrid_mode and num_cpu_bytes > 0:
                pid, fd = _C.create_cpu_handle(num_cpu_bytes)
                cpu_comm = [None] * self.num_ranks
                dist.all_gather_object(cpu_comm, (pid, fd), self.group)

            # Create CPP handle
            self.runtime = _C.ElasticBuffer(
                group.rank(), group.size(),
                self.nccl_comm_handle.get(), cpu_comm,
                num_bytes, num_cpu_bytes,
                allow_hybrid_mode,
                allow_multiple_reduction,
                prefer_overlap_with_compute,
                sl_idx, num_allocated_qps,
                num_cpu_timeout_secs, num_gpu_timeout_secs,
                self.explicitly_destroy)

        # Logical rank indices
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.get_logical_domain_size()
        self.scaleout_rank_idx = self.rank_idx // self.num_scaleup_ranks
        self.scaleup_rank_idx = self.rank_idx % self.num_scaleup_ranks

        # Physical rank indices
        self.num_rdma_ranks, self.num_nvlink_ranks = self.get_physical_domain_size()

        # Call a barrier to ensure initialization visibility for all peers
        torch.cuda.synchronize()
        group.barrier()
        torch.cuda.synchronize()

    def destroy(self) -> None:
        """
        Destroy the C++ runtime and release resources. Requires `explicitly_destroy=True` at construction.
        """
        assert self.explicitly_destroy

        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None  # Cannot use anymore
            self.nccl_comm_handle = None

    @staticmethod
    def get_buffer_size_hint(group: dist.ProcessGroup,
                             num_max_tokens_per_rank: int, hidden: int,
                             num_topk: int = 0, use_fp8_dispatch: bool = False,
                             allow_hybrid_mode: bool = True,
                             allow_multiple_reduction: bool = True) -> int:
        """
        Get a recommended buffer size (in bytes) for the given MoE settings, without constructing the buffer.
        The returned value is aligned to 2 MB.

        Arguments:
            group: the communication group.
            num_max_tokens_per_rank: the maximum number of tokens per rank.
            hidden: the hidden dimension of each token.
            num_topk: the number of top-k experts per token.
            use_fp8_dispatch: whether to use FP8 for dispatch.
            allow_hybrid_mode: whether to enable hybrid mode.
            allow_multiple_reduction: whether to allow multiple reductions in combine.

        Returns:
            size: the recommended buffer size in bytes (2 MB-aligned).
        """
        # NOTES: calculate_elastic_buffer_size already returns 2 MB-aligned values
        return _C.calculate_elastic_buffer_size(
            get_nccl_comm_handle(group).get(),
            num_max_tokens_per_rank, hidden, num_topk, use_fp8_dispatch,
            allow_hybrid_mode, allow_multiple_reduction)

    @staticmethod
    def get_engram_storage_size_hint(num_entries: int, hidden: int,
                                     num_max_tokens_per_rank: int,
                                     dtype: torch.dtype = torch.bfloat16) -> Tuple[int, int]:
        """
        (Experimental) Get a minimum buffer size requirement for Engram storage.
        Both returned values are aligned to 2 MB.

        Arguments:
            num_entries: the number of entries in the Engram storage.
            hidden: the hidden dimension of each entry.
            num_max_tokens_per_rank: the maximum number of tokens per rank (reserved for receive space).
            dtype: the data type, defaults to `torch.bfloat16`.

        Returns:
            num_gpu_bytes: the recommended GPU buffer size in bytes for fetch recv area (2 MB-aligned).
            num_cpu_bytes: the recommended CPU buffer size in bytes for engram local storage (2 MB-aligned).
        """
        # TODO: refactor all APIs to allow more parallelism
        # TODO: consider FP4
        # NOTES: only the data (BF16 or FP8) is transported via RDMA; FP8 scaling factors are
        # locally redundant.
        buffer_alignment = _C.get_elastic_buffer_alignment()
        # NOTES: we align per-entry size with 32 bytes (LDG.256)
        num_bytes_per_entry = align(hidden * dtype.itemsize, 32)
        num_gpu_bytes = align(num_bytes_per_entry * num_max_tokens_per_rank, buffer_alignment)
        num_cpu_bytes = align(num_bytes_per_entry * num_entries, buffer_alignment)
        return num_gpu_bytes, num_cpu_bytes

    @staticmethod
    def get_pp_buffer_size_hint(num_max_tensor_bytes: int,
                                num_max_inflight_tensors: int) -> int:
        """
        (Experimental) Get a minimum buffer size requirement for pipeline-parallel (PP) send/recv.
        The returned value is aligned to 2 MB.

        Arguments:
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.

        Returns:
            size: the recommended PP buffer size in bytes (2 MB-aligned).
        """
        # Align with `LDG.256`
        num_max_tensor_bytes = align(num_max_tensor_bytes, 32)

        # Each buffer (send and recv, * 2) contains prev and next rank (* 2) in the ring
        buffer_alignment = _C.get_elastic_buffer_alignment()
        return align(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2, buffer_alignment)

    @staticmethod
    def get_agrs_num_max_session_bytes(group: dist.ProcessGroup,
                                       shapes: Union[Tuple[int, ...], torch.Size, Sequence[Union[Tuple[int, ...], torch.Size]]],
                                       dtype: torch.dtype) -> int:
        """
        (Experimental) Calculate the total buffer bytes required for all-gather reduce-scatter (AGRS)
        in a single session.

        Arguments:
            group: the communication group.
            shapes: the local shape(s) of the tensor(s) before gathering. Pass a single shape
                tuple, or a sequence of shape tuples for batched mode.
            dtype: the data type for the tensor(s).

        Returns:
            size: the total number of bytes that will be used in this session.
        """
        if not isinstance(shapes[0], tuple):
            shapes = (shapes,)
        return sum(align(group.size() * math.prod(x) * dtype.itemsize, 32) for x in shapes)

    @staticmethod
    def get_agrs_buffer_size_hint(group: dist.ProcessGroup,
                                  num_max_session_bytes: int) -> int:
        """
        (Experimental) Get a minimum buffer size requirement for all-gather reduce-scatter (AGRS) sessions.
        The returned value is aligned to 2 MB.

        Arguments:
            group: the communication group.
            num_max_session_bytes: the maximum total bytes of all gathered tensors in a single session
                (calculated by rounding each tensor up to 32 bytes).

        Returns:
            size: the recommended AGRS buffer size in bytes (2 MB-aligned).
        """
        buffer_alignment = _C.get_elastic_buffer_alignment()
        return align(num_max_session_bytes, buffer_alignment)

    def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False, sequential: bool = True) -> None:
        """
        Perform a GPU-level barrier across all ranks, optionally with CPU synchronization.

        Arguments:
            use_comm_stream: whether to use the communication stream (otherwise uses the current compute stream).
            with_cpu_sync: whether to also call `cudaDeviceSynchronize` before and after the barrier.
            sequential: whether to run the scaleout and scaleup barriers sequentially (on a single SM) instead of
                in parallel across SMs. Sequential mode provides better synchronization guarantees,
                mainly used for test synchronization.
        """
        self.runtime.barrier(use_comm_stream, with_cpu_sync, sequential)

    @staticmethod
    def _unpack_handle(handle: Optional[EPHandle] = None) \
        -> Tuple[Optional[int], Optional[int], Optional[list],
                 Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                 Optional[torch.Tensor], Optional[torch.Tensor]]:
        if handle is None:
            return None, None, None, None, None, None, None, None, None, None
        return (handle.num_recv_tokens,
                handle.num_expanded_tokens,
                handle.num_recv_tokens_per_expert_list,
                handle.psum_num_recv_tokens_per_scaleup_rank,
                handle.psum_num_recv_tokens_per_expert,
                handle.num_unaligned_recv_tokens_per_expert,
                handle.dst_buffer_slot_idx,
                handle.token_metadata_at_forward,
                handle.recv_src_metadata,
                handle.channel_linked_list)

    @staticmethod
    def capture() -> EventHandle:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event_handle: the captured event handle.
        """
        return EventHandle()

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    def get_physical_domain_size(self) -> Tuple[int, int]:
        """
        Get the physical domain sizes (RDMA ranks and NVLink ranks).

        Returns:
            num_rdma_ranks: the number of physical RDMA ranks.
            num_nvlink_ranks: the number of physical NVLink ranks.
        """
        return self.runtime.get_physical_domain_size()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        """
        Get the logical domain sizes (scaleout ranks and scaleup ranks).

        Returns:
            num_scaleout_ranks: the number of logical scaleout ranks.
            num_scaleup_ranks: the number of logical scaleup ranks.
        """
        return self.runtime.get_logical_domain_size()

    def engram_write(self, storage: torch.Tensor,
                     sf: Optional[torch.Tensor] = None) -> None:
        """
        (Experimental) Write Engram storage data into the buffer.
        This call includes a barrier before and after the write to ensure visibility.

        Arguments:
            storage: `[num_entries, hidden]`, the Engram storage tensor. Either `torch.bfloat16`,
                or `torch.float8_e4m3fn` for FP8 mode.
            sf: `[num_total_entries, num_sf_packs]`, the globally replicated per-entry FP8 scaling
                factors (row-major). Each pack is an opaque 4-byte element, either `torch.float32` or
                packed UE8M0x4 (`torch.int32`). Must be provided iff the storage is FP8.
        """
        self.runtime.engram_write(storage, sf)

    def engram_fetch(self, indices: torch.Tensor, num_qps: int = 0,
                     use_tma_aligned_col_major_sf: bool = False) -> Callable:
        """
        (Experimental) Fetch Engram entries from remote ranks via RDMA.
        Returns a callable that, when invoked, waits for the RDMA gets to complete and returns the fetched tensor.

        Arguments:
            indices: `[num_tokens, num_entries_per_token]` with `torch.int`, the entry indices to fetch.
                Each token concatenates its `num_entries_per_token` entries along the hidden dimension.
            num_qps: the number of QPs to use (0 for all allocated QPs).
            use_tma_aligned_col_major_sf: whether to gather the fetched factors into the TMA-aligned
                column-major layout (otherwise a plain row-major layout).

        Returns:
            hook: a callable that blocks until data arrives and returns `(data, sf)`, where `data` has
                shape `[num_tokens * num_entries_per_token, hidden]` (`torch.bfloat16`, or
                `torch.float8_e4m3fn` in FP8 mode) and `sf` is the gathered scaling factors with shape
                `[num_tokens, num_entries_per_token * num_sf_packs]` in FP8 mode, otherwise `None`.
                In FP8 mode the factors come from the `sf` tensor supplied at `engram_write`.
        """
        return self.runtime.engram_fetch(indices, num_qps, use_tma_aligned_col_major_sf)

    def pp_set_config(self, num_max_tensor_bytes: int, num_max_inflight_tensors: int):
        """
        (Experimental) Configure pipeline-parallel (PP) send/recv parameters. Includes a barrier to flush previous operations.

        Arguments:
            num_max_tensor_bytes: the maximum tensor size in bytes per send/recv operation.
            num_max_inflight_tensors: the maximum number of in-flight tensors at once.
        """
        self.runtime.pp_set_config(num_max_tensor_bytes, num_max_inflight_tensors)

    def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
        """
        (Experimental) Send a tensor to an adjacent rank in the PP ring (prev or next rank only).

        Arguments:
            t: the tensor to send, must be contiguous and fit within `num_max_tensor_bytes`.
            dst_rank_idx: the destination rank index (must be prev or next rank in the ring).
            num_sms: the number of SMs to use (0 for all SMs).
        """
        self.runtime.pp_send(t, dst_rank_idx, num_sms)

    def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
        """
        (Experimental) Receive a tensor from an adjacent rank in the PP ring (prev or next rank only).

        Arguments:
            t: the output tensor to receive into, must be contiguous and fit within `num_max_tensor_bytes`.
            src_rank_idx: the source rank index (must be prev or next rank in the ring).
            num_sms: the number of SMs to use (0 for all SMs).
        """
        self.runtime.pp_recv(t, src_rank_idx, num_sms)

    def create_agrs_session(self) -> None:
        """
        (Experimental) Begin a new all-gather reduce-scatter (AGRS) session. Must be paired with `destroy_agrs_session`.

        """
        self.runtime.create_agrs_session()

    def destroy_agrs_session(self) -> None:
        """
        (Experimental) End the current AGRS session. Waits for the compute stream, signals session completion to all peers.

        """
        self.runtime.destroy_agrs_session()

    @contextmanager
    def agrs_new_session(self, enabled: bool = True):
        """
        (Experimental) Context manager that wraps `create_agrs_session` and `destroy_agrs_session`.

        Arguments:
            enabled: if `False`, the context manager is a no-op.
        """
        if not enabled:
            yield
            return

        self.runtime.create_agrs_session()
        try:
            yield
        finally:
            self.runtime.destroy_agrs_session()

    def agrs_set_config(self, num_max_session_bytes: int,
                        num_max_all_gathers_per_session: int) -> None:
        """
        (Experimental) Configure AGRS session parameters. Includes a barrier to flush previous operations.

        Arguments:
            num_max_session_bytes: the maximum total bytes of gathered tensors per session.
            num_max_all_gathers_per_session: the maximum number of all-gather operations per session.
        """
        self.runtime.agrs_set_config(num_max_session_bytes, num_max_all_gathers_per_session)

    # noinspection PyTypeChecker
    def agrs_get_inplace_tensor(self,
                                shapes: Union[Tuple[int, ...], torch.Size, Sequence[Union[Tuple[int, ...], torch.Size]]],
                                dtype: torch.dtype) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        (Experimental) Get in-place tensor(s) from the AGRS buffer for this rank's slot, without copying.
        Must be called within an active AGRS session.

        Arguments:
            shapes: the shape(s) of tensor(s) to allocate. Pass a single shape tuple, or a sequence of shape tuples
                for batched mode.
            dtype: the data type for the tensor(s).

        Returns:
            tensor: a single tensor if a single shape is given, or a tuple of tensors for batched mode.
        """
        is_batched_mode = isinstance(shapes[0], tuple)
        if not is_batched_mode:
            shapes = (shapes, )
        tensors = self.runtime.agrs_get_inplace_tensor(
            (math.prod(shape) * dtype.itemsize for shape in shapes)
        )
        out = tuple(tensor.view(dtype).view(shape) for tensor, shape in zip(tensors, shapes, strict=True))
        return out if is_batched_mode else out[0]

    def all_gather(self, t: Union[torch.Tensor, Sequence[torch.Tensor]]):
        """
        (Experimental) Perform an all-gather operation within an active AGRS session.
        Each rank's data is gathered to all ranks via NVLink symmetric memory.

        Arguments:
            t: a single tensor or a sequence of tensors to all-gather. Each tensor must be contiguous and
                CUDA-allocated.

        Returns:
            For a single tensor: `(gathered, handle)` where `gathered` has an extra leading dimension of
                `num_ranks`, and `handle` is a callable to wait for data arrival.
            For a sequence: `(*gathered_tensors, handle)` with one gathered tensor per input.
        """
        if isinstance(t, torch.Tensor):
            tensors, handle = self.runtime.all_gather((t,))
            return tensors[0], handle

        # Batched
        tensors, handle = self.runtime.all_gather(t)
        return *tensors, handle

    @weak_lru(maxsize=None)
    def get_theoretical_num_sms(self, num_experts: int, num_topk: int,
                                num_scaleout_topk: int = 0,
                                rdma_gbs: float = 0, nvlink_gbs: float = 0,
                                # TODO: use different values for other architectures
                                sm_read_gbs: float = 200, sm_write_gbs: float = 50) -> int:
        """
        Estimate the optimal number of SMs for dispatch/combine kernels based on bandwidth modeling.
        The result is cached. This assumes a balanced gate distribution.

        Arguments:
            num_experts: the number of all experts.
            num_topk: the number of top-k experts per token.
            num_scaleout_topk: reserved for balanced gate (must be 0 currently).
            rdma_gbs: the RDMA bandwidth in GB/s (0 for auto-detect).
            nvlink_gbs: the NVLink bandwidth in GB/s (0 for auto-detect).
            sm_read_gbs: the per-SM HBM read bandwidth in GB/s.
            sm_write_gbs: the per-SM HBM write bandwidth in GB/s.

        Returns:
            num_sms: the recommended SM count (even, at least 4).
        """
        # TODO: support `do_expand` and `allow_multiple_reduction`

        # The `1` in this function means scale-up traffic
        # i.e. the HBM read volume of the dispatch copy epilogue, equals to "the number of tokens" * "num_expected_topk" * "data size per token"
        # NOTES: this is for balanced gate
        # For V3.0's group-limited gate, please do not use this function
        # TODO: support this
        assert num_scaleout_topk == 0

        # Get bandwidth
        if rdma_gbs == 0 and self.num_rdma_ranks > 1:
            rdma_gbs = get_rdma_gbs()
        if nvlink_gbs == 0:
            nvlink_gbs = get_nvlink_gbs()

        # Initial count
        # NOTES: we don't count HBM traffic
        sm_read, sm_write = 0, 0
        rdma_traffic, nvlink_traffic = 0, 0

        def get_expected_topk(num_groups: int) -> float:
            assert num_experts % num_groups == 0
            return num_groups * (1 - math.comb(num_experts - num_experts // num_groups, num_topk) / math.comb(num_experts, num_topk))

        # Expected top-k scale-out ranks
        num_expected_scaleout_topk = get_expected_topk(self.num_scaleout_ranks) if self.num_scaleout_ranks > 1 else 0

        # Expected top-k scale-up ranks
        num_expected_topk = get_expected_topk(self.num_ranks)

        # Read tokens
        sm_read += 1 / num_expected_topk

        # NOTES: we don't consider the skip-send-buffer cases (all selections fall in the local)
        if self.num_scaleout_ranks > 1:
            # Scaleup warps: write send buffer
            sm_write += 1 / num_expected_topk

            # Scaleout traffic
            sm_write += (1 / num_expected_topk) * (num_expected_scaleout_topk / self.num_scaleout_ranks)  # Local bypass
            rdma_traffic += (1 / num_expected_topk) * (num_expected_scaleout_topk * (1 - 1 / self.num_scaleout_ranks))

            # Forward warps
            sm_read += num_expected_scaleout_topk / num_expected_topk
            sm_write += 1  # Issue scaleup
            nvlink_traffic += 1 - (1 / self.num_scaleup_ranks)
        else:
            # Write send buffer
            if self.num_rdma_ranks > 1:
                sm_write += 1 / num_expected_topk

            # Issue NVLink
            sm_write += self.num_nvlink_ranks / self.num_ranks

            # NVLink and RDMA traffic
            nvlink_traffic += self.num_nvlink_ranks / self.num_ranks * (1 - 1 / self.num_nvlink_ranks)  # Except local bypass
            rdma_traffic += (self.num_ranks - self.num_nvlink_ranks) / self.num_ranks

        # Found the bounded one
        if self.num_scaleout_ranks > 1 and (rdma_traffic / rdma_gbs) > (nvlink_traffic / nvlink_gbs):
            bounded_traffic, bounded_gbs = rdma_traffic, rdma_gbs
        else:
            bounded_traffic, bounded_gbs = nvlink_traffic, nvlink_gbs

        # Calculate SM count
        # NOTES: will try to use more SMs if not overlap with compute
        num_device_sms = torch.cuda.get_device_properties('cuda').multi_processor_count
        num_sms = num_device_sms  # No traffic, e.g., EP=1
        if bounded_traffic > 0:
            num_sms = max(
                bounded_gbs / bounded_traffic * sm_read / sm_read_gbs,
                bounded_gbs / bounded_traffic * sm_write / sm_write_gbs,
            )
        num_sms = align(max(4, math.ceil(num_sms * 1.25)), 2)
        num_sms = num_sms if self.prefer_overlap_with_compute else max(num_sms, 64)
        num_sms = min(num_sms, num_device_sms)

        # Summary
        if os.environ.get('EP_BUFFER_DEBUG', 0):
            print(f'EP SM approximation: '
                  f'{sm_read=}, {sm_write=}, {rdma_traffic=}, {nvlink_traffic=}, '
                  f'{rdma_gbs=}, {nvlink_gbs=}, '
                  f'{num_expected_scaleout_topk=}, {num_expected_topk=}, '
                  f'{bounded_traffic=}, {bounded_gbs=}, {num_sms=}')
        return num_sms

    def get_theoretical_num_qps(self, num_sms: int) -> int:
        """
        Estimate the optimal number of RDMA QPs based on SM count and mode.

        Arguments:
            num_sms: the number of SMs used for the dispatch/combine kernel.

        Returns:
            num_qps: the recommended QP count, capped by `num_allocated_qps`.
        """
        # For direct mode, we encourage less QPs to reduce DB ringing overhead
        num_qps = min(num_sms, 8 + 1)

        # For hybrid mode, we encourage every channel (and notify) to have an independent QP
        if self.allow_hybrid_mode:
            num_qps = num_sms * 16 + 1

        return min(num_qps, self.num_allocated_qps)

    def _dispatch_rail_balance_force(
            self,
            x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            topk_idx: Optional[torch.Tensor],
            topk_weights: Optional[torch.Tensor],
            cumulative_local_expert_recv_stats: Optional[torch.Tensor],
            num_experts: Optional[int],
            num_max_tokens_per_rank: Optional[int],
            expert_alignment: Optional[int],
            num_sms: int, num_qps: int,
            previous_event: Optional[EventHandle],
            previous_event_before_epilogue: Optional[EventHandle],
            async_with_compute_stream: bool,
            allocate_on_comm_stream: bool,
            handle: Optional[EPHandle],
            do_handle_copy: bool,
            do_cpu_sync: Optional[bool],
            do_expand: bool,
            do_zero_padding: bool,
            use_tma_aligned_col_major_sf: bool):
        """Run the narrow force-v1 publication transaction."""
        invocation_id = 0
        local_error_priority = 0
        prepare_attempted = False
        prepare_common_fields = None
        dispatch_manifest = None
        ticket = None
        cached_bypass = False

        # Reserve the epoch before validating caller inputs. Every attempt,
        # including a locally invalid one, must rendezvous with its peers.
        try:
            invocation_id = self._rail_balance_next_invocation_id
            if type(invocation_id) is not int or not (
                    1 <= invocation_id <=
                    _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
                raise ValueError('rail-balance invocation is out of range')
            self._rail_balance_next_invocation_id = invocation_id + 1
        except BaseException:
            invocation_id = 0
            local_error_priority = _RAIL_BALANCE_DISPATCH_VALIDATION_ERROR

        resolved_num_max_tokens = 0
        resolved_num_experts = 0
        resolved_num_sms = 0
        resolved_num_qps = 0
        if local_error_priority == 0:
            try:
                if self._rail_balance_terminal:
                    raise RuntimeError('rail-balance buffer is terminal')
                if self._rail_balance_live_ticket is not None:
                    raise RuntimeError('rail-balance dispatch is already live')
                if self._rail_balance_mode == 'adaptive':
                    raise RuntimeError(
                        'adaptive RailBalance requires the selective 2-hop GPU planner')
                if handle is not None:
                    raise ValueError('cached force dispatch is unsupported')
                if isinstance(x, tuple):
                    raise ValueError('FP8 force dispatch is unsupported')
                if topk_idx is None or topk_weights is None:
                    raise ValueError(
                        'force dispatch requires top-k indices and weights')
                if type(num_experts) is not int or num_experts <= 0:
                    raise ValueError('force dispatch requires num_experts')
                resolved_num_experts = num_experts
                resolved_num_max_tokens = value_or(
                    num_max_tokens_per_rank,
                    self.num_max_tokens_per_rank)
                if type(resolved_num_max_tokens) is not int or \
                        resolved_num_max_tokens != \
                        self.num_max_tokens_per_rank:
                    raise ValueError(
                        'force dispatch must use the constructor token capacity')
                resolved_expert_alignment = value_or(expert_alignment, 1)
                if type(resolved_expert_alignment) is not int or \
                        resolved_expert_alignment != 1:
                    raise ValueError(
                        'force dispatch supports expert_alignment=1 only')
                resolved_do_cpu_sync = value_or(do_cpu_sync, True)
                if type(resolved_do_cpu_sync) is not bool or \
                        not resolved_do_cpu_sync:
                    raise ValueError(
                        'force dispatch requires do_cpu_sync=True')
                if type(do_handle_copy) is not bool or not do_handle_copy:
                    raise ValueError(
                        'force dispatch requires do_handle_copy=True')
                if type(do_expand) is not bool or do_expand or \
                        type(do_zero_padding) is not bool or do_zero_padding or \
                        type(use_tma_aligned_col_major_sf) is not bool or \
                        use_tma_aligned_col_major_sf:
                    raise ValueError(
                        'force dispatch does not support expanded/FP8 layouts')
                if previous_event is not None or \
                        previous_event_before_epilogue is not None or \
                        type(async_with_compute_stream) is not bool or \
                        async_with_compute_stream or \
                        type(allocate_on_comm_stream) is not bool or \
                        allocate_on_comm_stream:
                    raise ValueError(
                        'force dispatch does not support public event/async options')
                if self.deterministic:
                    raise ValueError(
                        'force dispatch does not support deterministic mode')
                check_torch_deterministic()

                if type(num_sms) is not int or num_sms < 0:
                    raise ValueError('force dispatch num_sms is invalid')
                resolved_num_sms = num_sms
                if resolved_num_sms == 0:
                    resolved_num_sms = self.get_theoretical_num_sms(
                        resolved_num_experts, topk_idx.shape[1])
                if type(resolved_num_sms) is not int or not (
                        2 <= resolved_num_sms <=
                        _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
                    raise ValueError('force dispatch num_sms is invalid')

                if type(num_qps) is not int or num_qps < 0:
                    raise ValueError('force dispatch num_qps is invalid')
                resolved_num_qps = num_qps
                if resolved_num_qps == 0:
                    resolved_num_qps = self.get_theoretical_num_qps(
                        resolved_num_sms)
                if type(resolved_num_qps) is not int or not (
                        1 <= resolved_num_qps <= self.num_allocated_qps):
                    raise ValueError('force dispatch num_qps is invalid')
                ticket = _RailBalanceForceTicket(
                    self._rail_balance_owner_token, invocation_id)
            except BaseException:
                local_error_priority = \
                    _RAIL_BALANCE_DISPATCH_VALIDATION_ERROR

        if local_error_priority == 0:
            cached_bypass = self._rail_balance_zero_move_bypass_budget > 0
            if cached_bypass:
                try:
                    cached_fields = \
                        self._rail_balance_zero_move_common_fields
                    if type(cached_fields) is not tuple or \
                            len(cached_fields) != \
                            _RAIL_BALANCE_DISPATCH_COMMON_FIELDS:
                        raise ValueError(
                            'rail-balance zero-move cache ABI is invalid')
                    if x.dim() != 2 or topk_idx.dim() != 2 or \
                            topk_weights.dim() != 2 or \
                            not x.is_contiguous() or \
                            not topk_idx.is_contiguous() or \
                            not topk_weights.is_contiguous() or \
                            x.dtype != torch.bfloat16 or \
                            topk_weights.dtype != torch.float32 or \
                            x.device != topk_idx.device or \
                            x.device != topk_weights.device or \
                            x.shape[0] != topk_idx.shape[0] or \
                            x.shape[0] != topk_weights.shape[0] or \
                            topk_idx.shape != topk_weights.shape or \
                            x.shape[1] != cached_fields[4] or \
                            topk_idx.shape[1] != cached_fields[5] or \
                            resolved_num_max_tokens != cached_fields[6] or \
                            resolved_num_experts != cached_fields[7] or \
                            resolved_num_sms != cached_fields[9] or \
                            resolved_num_qps != cached_fields[14]:
                        raise ValueError(
                            'rail-balance zero-move cache input mismatch')
                    prepare_common_fields = cached_fields
                except BaseException:
                    local_error_priority = \
                        _RAIL_BALANCE_DISPATCH_VALIDATION_ERROR
            else:
                prepare_attempted = True
                try:
                    prepare_result = \
                        self.runtime._rail_balance_hybrid_dispatch_prepare(
                            x, topk_idx, topk_weights,
                            cumulative_local_expert_recv_stats,
                            resolved_num_max_tokens, resolved_num_experts,
                            resolved_num_sms, resolved_num_qps,
                            self._rail_balance_proxy_slots_per_rank,
                            self._rail_balance_arena_offset,
                            invocation_id, 0,
                            self._rail_balance_policy,
                            self._rail_balance_threshold_percent,
                            self._rail_balance_mode == 'one_hop')
                    if type(prepare_result) is not tuple or \
                            len(prepare_result) != 1 + \
                            _RAIL_BALANCE_DISPATCH_COMMON_FIELDS or \
                            type(prepare_result[0]) is not int or \
                            prepare_result[0] not in (0, 2, 3):
                        raise ValueError(
                            'rail-balance dispatch prepare ABI is invalid')
                    if prepare_result[0] != 0:
                        local_error_priority = \
                            _RAIL_BALANCE_DISPATCH_PREPARE_STATUS_ERROR
                    else:
                        prepare_common_fields = tuple(prepare_result[1:])
                except BaseException:
                    local_error_priority = \
                        _RAIL_BALANCE_DISPATCH_PREPARE_ERROR

        if local_error_priority == 0:
            try:
                dispatch_manifest = _make_rail_balance_operation_manifest(
                    _RAIL_BALANCE_OPERATION_DISPATCH,
                    _RAIL_BALANCE_PHASE_PREPARE,
                    self.num_ranks, invocation_id,
                    prepare_common_fields)
            except BaseException:
                local_error_priority = \
                    _RAIL_BALANCE_DISPATCH_MANIFEST_ERROR
        if dispatch_manifest is None:
            dispatch_manifest = _make_rail_balance_operation_manifest(
                _RAIL_BALANCE_OPERATION_DISPATCH,
                _RAIL_BALANCE_PHASE_PREPARE,
                0, 0,
                (0,) * _RAIL_BALANCE_DISPATCH_COMMON_FIELDS)

        local_error_key = _make_rail_balance_world_gate_error_key(
            local_error_priority, self.rank_idx)
        try:
            _encode_rail_balance_world_gate(
                self._rail_balance_world_gate_host_words,
                local_error_key, dispatch_manifest)
        except BaseException:
            local_error_key = _make_rail_balance_world_gate_error_key(
                _RAIL_BALANCE_DISPATCH_ENCODE_ERROR, self.rank_idx)
            _encode_rail_balance_world_gate(
                self._rail_balance_world_gate_host_words,
                local_error_key,
                _make_rail_balance_operation_manifest(
                    _RAIL_BALANCE_OPERATION_DISPATCH,
                    _RAIL_BALANCE_PHASE_PREPARE,
                    0, 0,
                    (0,) * _RAIL_BALANCE_DISPATCH_COMMON_FIELDS))

        try:
            gate_result = _run_rail_balance_world_gate(
                self._rail_balance_world_gate_device_words,
                self._rail_balance_world_gate_host_words,
                self.group)
        except BaseException:
            self._rail_balance_terminal = True
            raise
        try:
            _raise_rail_balance_world_gate_failure(
                'dispatch-prepare', gate_result)
        except BaseException:
            if prepare_attempted:
                try:
                    self.runtime._rail_balance_hybrid_plan_abort(
                        invocation_id)
                except BaseException:
                    self._rail_balance_terminal = True
                    raise
            raise

        if cached_bypass:
            self._rail_balance_zero_move_bypass_budget -= 1
            original_mode = self._rail_balance_mode
            self._rail_balance_mode = 'off'
            try:
                return self.dispatch(
                    x, topk_idx, topk_weights,
                    cumulative_local_expert_recv_stats,
                    num_experts, num_max_tokens_per_rank,
                    expert_alignment, num_sms, num_qps,
                    previous_event,
                    previous_event_before_epilogue,
                    async_with_compute_stream,
                    allocate_on_comm_stream, handle,
                    do_handle_copy, do_cpu_sync, do_expand,
                    do_zero_padding,
                    use_tma_aligned_col_major_sf)
            finally:
                self._rail_balance_mode = original_mode

        plan_error_priority = 0
        local_moved_copies = 0
        try:
            plan_outputs = \
                self.runtime._rail_balance_hybrid_plan_finish(invocation_id)
            if type(plan_outputs) is not tuple or len(plan_outputs) != 14:
                raise ValueError('rail-balance plan result ABI is invalid')
            plan_status = plan_outputs[-1].item()
            if type(plan_status) is not int or plan_status not in (0, 1):
                raise ValueError('rail-balance plan status is invalid')
            local_moved_copies = plan_outputs[12].item()
            if type(local_moved_copies) is not int or \
                    local_moved_copies < 0:
                raise ValueError(
                    'rail-balance moved-copy count is invalid')
            if plan_status != 0:
                plan_error_priority = \
                    _RAIL_BALANCE_DISPATCH_PLAN_STATUS_ERROR
            elif local_moved_copies > 0:
                plan_error_priority = _RAIL_BALANCE_DISPATCH_HAS_MOVES
        except BaseException:
            plan_error_priority = _RAIL_BALANCE_DISPATCH_PLAN_ERROR

        plan_error_key = _make_rail_balance_world_gate_error_key(
            plan_error_priority, self.rank_idx)
        try:
            _patch_rail_balance_world_gate_prevalidated(
                self._rail_balance_world_gate_host_words,
                plan_error_key,
                3, _RAIL_BALANCE_PHASE_PLAN)
        except BaseException:
            plan_error_key = _make_rail_balance_world_gate_error_key(
                _RAIL_BALANCE_DISPATCH_PLAN_ENCODE_ERROR,
                self.rank_idx)
            _encode_rail_balance_world_gate(
                self._rail_balance_world_gate_host_words,
                plan_error_key,
                _make_rail_balance_operation_manifest(
                    _RAIL_BALANCE_OPERATION_DISPATCH,
                    _RAIL_BALANCE_PHASE_PLAN,
                    0, 0,
                    (0,) * _RAIL_BALANCE_DISPATCH_COMMON_FIELDS))

        try:
            gate_result = _run_rail_balance_world_gate(
                self._rail_balance_world_gate_device_words,
                self._rail_balance_world_gate_host_words,
                self.group)
        except BaseException:
            self._rail_balance_terminal = True
            raise
        world_has_moves = False
        try:
            if gate_result[0] != 0:
                priority, _ = _decode_rail_balance_world_gate_error_key(
                    gate_result[0])
                if priority == _RAIL_BALANCE_DISPATCH_HAS_MOVES:
                    world_has_moves = True
                else:
                    _raise_rail_balance_world_gate_failure(
                        'dispatch-plan', gate_result)
            else:
                _raise_rail_balance_world_gate_failure(
                    'dispatch-plan', gate_result)
        except BaseException:
            try:
                self.runtime._rail_balance_hybrid_plan_abort(invocation_id)
            except BaseException:
                self._rail_balance_terminal = True
                raise
            raise

        # Gate2's low-priority control bit computes a world-wide OR without an
        # extra collective. If no source rank moves a copy, the native Hybrid
        # path already has the desired rail placement.
        if not world_has_moves:
            self.runtime._rail_balance_hybrid_plan_abort(invocation_id)
            self._rail_balance_zero_move_bypass_budget = \
                _RAIL_BALANCE_ZERO_MOVE_RECHECK_INTERVAL - 1
            self._rail_balance_zero_move_common_fields = \
                prepare_common_fields
            original_mode = self._rail_balance_mode
            self._rail_balance_mode = 'off'
            try:
                return self.dispatch(
                    x, topk_idx, topk_weights,
                    cumulative_local_expert_recv_stats,
                    num_experts, num_max_tokens_per_rank,
                    expert_alignment, num_sms, num_qps,
                    previous_event,
                    previous_event_before_epilogue,
                    async_with_compute_stream,
                    allocate_on_comm_stream, handle,
                    do_handle_copy, do_cpu_sync, do_expand,
                    do_zero_padding,
                    use_tma_aligned_col_major_sf)
            finally:
                self._rail_balance_mode = original_mode

        try:
            self.runtime._rail_balance_hybrid_dispatch_commit(invocation_id)
            (recv_x, recv_sf,
             recv_topk_idx, recv_topk_weights,
             cloned_topk_idx,
             num_recv_tokens, num_expanded_tokens,
             num_recv_tokens_per_expert_list,
             psum_num_recv_tokens_per_scaleup_rank,
             psum_num_recv_tokens_per_expert,
             num_unaligned_recv_tokens_per_expert,
             recv_src_metadata,
             dst_buffer_slot_idx,
             token_metadata_at_forward,
             channel_linked_list,
             event) = self.runtime._rail_balance_hybrid_dispatch_finish(
                invocation_id)
            if recv_sf is not None:
                raise RuntimeError(
                    'force-v1 dispatch unexpectedly returned FP8 scales')
            force_handle = EPHandle(
                False,
                resolved_num_experts, 1,
                resolved_num_max_tokens,
                resolved_num_sms,
                cloned_topk_idx,
                num_recv_tokens, num_expanded_tokens,
                num_recv_tokens_per_expert_list,
                psum_num_recv_tokens_per_scaleup_rank,
                psum_num_recv_tokens_per_expert,
                num_unaligned_recv_tokens_per_expert,
                recv_src_metadata,
                dst_buffer_slot_idx,
                token_metadata_at_forward,
                channel_linked_list)
            force_handle._rail_balance_ticket = ticket
            event_overlap = EventOverlap(event)
            self._rail_balance_live_ticket = ticket
            return (recv_x, recv_topk_idx, recv_topk_weights,
                    force_handle, event_overlap)
        except BaseException:
            self._rail_balance_terminal = True
            self.runtime._rail_balance_hybrid_plan_abort(invocation_id)
            raise

    def dispatch(self,
                 x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 topk_idx: Optional[torch.Tensor] = None,
                 topk_weights: Optional[torch.Tensor] = None,
                 cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                 num_experts: Optional[int] = None,
                 num_max_tokens_per_rank: Optional[int] = None,
                 expert_alignment: Optional[int] = None,
                 num_sms: int = 0, num_qps: int = 0,
                 previous_event: Optional[EventHandle] = None,
                 previous_event_before_epilogue: Optional[EventHandle] = None,
                 async_with_compute_stream: bool = False,
                 allocate_on_comm_stream: bool = False,
                 handle: Optional[EPHandle] = None,
                 do_handle_copy: bool = True,
                 do_cpu_sync: Optional[bool] = None,
                 do_expand: bool = False,
                 do_zero_padding: bool = False,
                 use_tma_aligned_col_major_sf: bool = False) \
            -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     Optional[torch.Tensor], Optional[torch.Tensor],
                     EPHandle, EventOverlap]:
        """
        Dispatch tokens to different ranks. Supports both single-node and multi-node settings.
            SM and QP counts are automatically determined if not specified.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be
                `[num_tokens, hidden]`, and type must be `torch.bfloat16`; for the second type (FP8 mode),
                the first element of the tuple must be `[num_tokens, hidden]` with type `torch.float8_e4m3fn`,
                the second is the scale factors.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
                Must be `None` if `handle` is provided.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
                Must be `None` if `handle` is provided.
            cumulative_local_expert_recv_stats: `[num_local_experts]` with `torch.int`, a cumulative expert count
                tensor for statistics, useful for online EP load balance monitoring.
            num_experts: the number of all experts. Inferred from `handle` if provided.
            num_max_tokens_per_rank: the maximum number of tokens per rank. Inferred from constructor default
                or `handle` if provided.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_sms: the number of SMs to use (0 for automatic via `get_theoretical_num_sms`).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the copy epilogue.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.
            handle: an optional cached `EPHandle` from a previous dispatch, if set, the CPU will reuse the layout
                information to save some time. `topk_idx` must be `None` (reused from handle).
                `topk_weights` can be optionally provided (e.g. for backward pass with cached expand).
            do_handle_copy: whether to clone `topk_idx` in the returned handle (to prevent user modification).
            do_cpu_sync: whether to synchronize with CPU to get exact received token counts.
                `None` defaults to `True` unless `handle` is provided.
            do_expand: whether to use the expanding layout (one slot per expert per token).
            do_zero_padding: whether to zero out the alignment padding slots in the expanded output.
                Only valid when `do_expand` is True. Ensures alignment gaps between experts are zeroed.
            use_tma_aligned_col_major_sf: whether to use TMA-aligned column-major layout for scale factors.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`
            recv_topk_idx: received expert indices
            recv_topk_weights: received expert weights (`None` if `topk_weights` was not provided).
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        if getattr(self, '_rail_balance_mode', 'off') != 'off':
            return self._dispatch_rail_balance_force(
                x, topk_idx, topk_weights,
                cumulative_local_expert_recv_stats,
                num_experts, num_max_tokens_per_rank, expert_alignment,
                num_sms, num_qps,
                previous_event, previous_event_before_epilogue,
                async_with_compute_stream, allocate_on_comm_stream,
                handle, do_handle_copy, do_cpu_sync,
                do_expand, do_zero_padding,
                use_tma_aligned_col_major_sf)
        if type(getattr(handle, '_rail_balance_ticket', None)) is \
                _RailBalanceForceTicket:
            raise ValueError(_rail_balance_error(
                'InvalidHandle',
                'a force dispatch handle cannot be reused by an off buffer'))

        check_torch_deterministic()

        # Automatic decide SM and QP count
        num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]
        num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else num_sms
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        # Unpack SF
        x, sf = x if isinstance(x, tuple) else (x, None)

        # Unpack handles
        # Reuse some values if possible
        if handle is not None:
            assert topk_idx is None
            assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'
            topk_idx = handle.topk_idx
            num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
            num_experts = value_or(num_experts, handle.num_experts)
            expert_alignment = value_or(expert_alignment, handle.expert_alignment)
            do_cpu_sync = False

            # Should be aligned with the handle context
            assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
                   (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)
        (cached_num_recv_tokens, cached_num_expanded_tokens,
         cached_num_recv_tokens_per_expert_list,
         cached_psum_num_recv_tokens_per_scaleup_rank, cached_psum_num_recv_tokens_per_expert,
         cached_num_unaligned_recv_tokens_per_expert,
         cached_dst_buffer_slot_idx,
         cached_token_metadata_at_forward,
         cached_recv_src_metadata,
         cached_channel_linked_list) = self._unpack_handle(handle)

        # Some default values
        num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)
        expert_alignment = value_or(expert_alignment, 1)
        do_cpu_sync = value_or(do_cpu_sync, True)

        # Do dispatch
        (recv_x, recv_sf,
         recv_topk_idx, recv_topk_weights,
         cloned_topk_idx,
         num_recv_tokens, num_expanded_tokens,
         num_recv_tokens_per_expert_list,
         psum_num_recv_tokens_per_scaleup_rank,
         psum_num_recv_tokens_per_expert,
         num_unaligned_recv_tokens_per_expert,
         recv_src_metadata,
         dst_buffer_slot_idx,
         token_metadata_at_forward,
         channel_linked_list,
         event) = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
                                        cumulative_local_expert_recv_stats,
                                        cached_num_recv_tokens,
                                        cached_num_expanded_tokens,
                                        cached_num_recv_tokens_per_expert_list,
                                        cached_psum_num_recv_tokens_per_scaleup_rank,
                                        cached_psum_num_recv_tokens_per_expert,
                                        cached_num_unaligned_recv_tokens_per_expert,
                                        cached_dst_buffer_slot_idx,
                                        cached_token_metadata_at_forward,
                                        cached_recv_src_metadata,
                                        cached_channel_linked_list,
                                        num_max_tokens_per_rank,
                                        num_experts, expert_alignment,
                                        num_sms, num_qps,
                                        previous_event,
                                        previous_event_before_epilogue,
                                        async_with_compute_stream, allocate_on_comm_stream,
                                        do_handle_copy, do_cpu_sync, do_expand,
                                        do_zero_padding,
                                        use_tma_aligned_col_major_sf)

        # Create handle
        is_cached_dispatch = handle is not None
        if not is_cached_dispatch:
            handle = EPHandle(do_expand,
                              num_experts, expert_alignment,
                              num_max_tokens_per_rank,
                              num_sms,
                              cloned_topk_idx if do_handle_copy else topk_idx,
                              num_recv_tokens, num_expanded_tokens,
                              num_recv_tokens_per_expert_list,
                              psum_num_recv_tokens_per_scaleup_rank,
                              psum_num_recv_tokens_per_expert,
                              num_unaligned_recv_tokens_per_expert,
                              recv_src_metadata,
                              dst_buffer_slot_idx,
                              token_metadata_at_forward,
                              channel_linked_list)

        # Create event
        event_overlap = EventOverlap(event)

        # Deterministic epilogue
        # NOTES: when we change the metadata layout, the epilogue should also be changed
        if self.deterministic:
            epilogue = functools.partial(
                handle.deterministic_sort,
                do_cpu_sync, is_cached_dispatch,
                recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list
            )
            event_overlap.register_hook_after_wait(epilogue) if async_with_compute_stream else epilogue()

        # Repack SF
        recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x

        # Return
        return recv_x, recv_topk_idx, recv_topk_weights, handle, event_overlap

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    def _combine_rail_balance_force(
            self,
            x: torch.Tensor,
            handle: EPHandle,
            topk_weights: Optional[torch.Tensor],
            bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            num_sms: int, num_qps: int,
            previous_event: EventHandle,
            previous_event_before_epilogue: Optional[EventHandle],
            async_with_compute_stream: bool,
            allocate_on_comm_stream: bool):
        """Consume one buffer-owned force ticket through the combine gate."""
        local_error_priority = 0
        invocation_id = 0
        ticket = None
        prepare_attempted = False
        combine_manifest = None

        try:
            ticket = getattr(handle, '_rail_balance_ticket', None)
            if type(ticket) is not _RailBalanceForceTicket:
                raise ValueError('combine handle has no force ticket')
            # Owner identity is checked before operation/state so a foreign
            # handle can never mutate or abort this buffer's live completion.
            if ticket.owner_token is not self._rail_balance_owner_token:
                raise ValueError('combine handle belongs to another buffer')
            invocation_id = ticket.invocation_id
            if type(invocation_id) is not int or not (
                    1 <= invocation_id <=
                    _RAIL_BALANCE_WORLD_GATE_INT32_MAX):
                raise ValueError('combine invocation is invalid')
            if self._rail_balance_terminal:
                raise RuntimeError('rail-balance buffer is terminal')
            if self._rail_balance_live_ticket is not ticket:
                raise ValueError('combine handle is not the live operation')
            if ticket.state != _RailBalanceForceTicket.LIVE:
                raise ValueError('combine handle has already been consumed')
            if topk_weights is None:
                raise ValueError('force combine requires top-k weights')
            if bias is not None:
                raise ValueError('force combine does not support bias')
            if type(num_qps) is not int or num_qps != 0:
                raise ValueError('force combine requires num_qps=0')
            if type(num_sms) is not int or num_sms not in (
                    0, handle.num_sms):
                raise ValueError(
                    'force combine must reuse the dispatch SM count')
            if previous_event is not None or \
                    previous_event_before_epilogue is not None or \
                    type(async_with_compute_stream) is not bool or \
                    async_with_compute_stream or \
                    type(allocate_on_comm_stream) is not bool or \
                    allocate_on_comm_stream:
                raise ValueError(
                    'force combine does not support public event/async options')
            if self.deterministic:
                raise ValueError(
                    'force combine does not support deterministic mode')
            check_torch_deterministic()
            ticket.state = _RailBalanceForceTicket.PREPARING
        except BaseException:
            local_error_priority = \
                _RAIL_BALANCE_COMBINE_VALIDATION_ERROR

        if local_error_priority == 0:
            # Once the call starts, Python owns cleanup even if an exception
            # arrives after C++ has installed the pending completion but
            # before control returns to the next Python statement.
            prepare_attempted = True
            try:
                self.runtime._rail_balance_hybrid_combine_prepare(
                    x, topk_weights, invocation_id)
            except BaseException:
                local_error_priority = \
                    _RAIL_BALANCE_COMBINE_PREPARE_ERROR

        if local_error_priority == 0:
            try:
                combine_manifest = _make_rail_balance_operation_manifest(
                    _RAIL_BALANCE_OPERATION_COMBINE,
                    _RAIL_BALANCE_PHASE_PREPARE,
                    self.num_ranks, invocation_id)
            except BaseException:
                local_error_priority = \
                    _RAIL_BALANCE_COMBINE_MANIFEST_ERROR
        if combine_manifest is None:
            combine_manifest = _make_rail_balance_operation_manifest(
                _RAIL_BALANCE_OPERATION_COMBINE,
                _RAIL_BALANCE_PHASE_PREPARE,
                0, 0)

        local_error_key = _make_rail_balance_world_gate_error_key(
            local_error_priority, self.rank_idx)
        try:
            _encode_rail_balance_world_gate(
                self._rail_balance_world_gate_host_words,
                local_error_key, combine_manifest)
        except BaseException:
            local_error_key = _make_rail_balance_world_gate_error_key(
                _RAIL_BALANCE_COMBINE_ENCODE_ERROR, self.rank_idx)
            _encode_rail_balance_world_gate(
                self._rail_balance_world_gate_host_words,
                local_error_key,
                _make_rail_balance_operation_manifest(
                    _RAIL_BALANCE_OPERATION_COMBINE,
                    _RAIL_BALANCE_PHASE_PREPARE,
                    0, 0))

        try:
            gate_result = _run_rail_balance_world_gate(
                self._rail_balance_world_gate_device_words,
                self._rail_balance_world_gate_host_words,
                self.group)
        except BaseException:
            self._rail_balance_terminal = True
            raise
        try:
            _raise_rail_balance_world_gate_failure(
                'combine-prepare', gate_result)
        except BaseException:
            if prepare_attempted:
                try:
                    self.runtime._rail_balance_hybrid_combine_abort(
                        invocation_id)
                except BaseException:
                    self._rail_balance_terminal = True
                    raise
            if type(ticket) is _RailBalanceForceTicket and \
                    ticket.owner_token is self._rail_balance_owner_token and \
                    ticket.state == _RailBalanceForceTicket.PREPARING:
                ticket.state = _RailBalanceForceTicket.LIVE
            raise

        ticket.state = _RailBalanceForceTicket.CONSUMED
        self._rail_balance_live_ticket = None
        try:
            combined_x, combined_topk_weights, event = \
                self.runtime._rail_balance_hybrid_combine_commit(
                    invocation_id)
            return (combined_x, combined_topk_weights,
                    EventOverlap(event))
        except BaseException:
            self._rail_balance_terminal = True
            raise

    def combine(self,
                x: torch.Tensor,
                handle: EPHandle,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                num_sms: int = 0, num_qps: int = 0,
                previous_event: EventHandle = None,
                previous_event_before_epilogue: Optional[EventHandle] = None,
                async_with_compute_stream: bool = False,
                allocate_on_comm_stream: bool = False) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens from different ranks back to their original ranks.
        Supports both single-node and multi-node settings.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the `dispatch` function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float` for non-expand mode, or
                `[num_tokens]` 1D for expand mode. The tokens' top-k weights for reducing to
                its original ranks.
            bias: 0, 1 or 2 `[num_combined_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            num_sms: the number of SMs to use (0 to reuse the SM count from the dispatch handle).
            num_qps: the number of RDMA QPs to use (0 for automatic via `get_theoretical_num_qps`).
            previous_event: the event to wait before actually executing the kernel.
                If set, `allocate_on_comm_stream` must also be `True`.
            previous_event_before_epilogue: the event to wait before actually executing the reduce epilogue.
            async_with_compute_stream: the current stream will not wait for the communication kernels to be
                finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the
                communication stream.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            combined_topk_weights: the reduced top-k weights, with shape `[num_combined_tokens, num_topk]` and type `torch.float`.
            event: the event after executing the kernel (valid only if `async_with_compute_stream` is set).
        """
        if getattr(self, '_rail_balance_mode', 'off') != 'off' and \
                type(getattr(handle, '_rail_balance_ticket', None)) is \
                _RailBalanceForceTicket:
            return self._combine_rail_balance_force(
                x, handle, topk_weights, bias,
                num_sms, num_qps,
                previous_event, previous_event_before_epilogue,
                async_with_compute_stream, allocate_on_comm_stream)
        if type(getattr(handle, '_rail_balance_ticket', None)) is \
                _RailBalanceForceTicket:
            raise ValueError(_rail_balance_error(
                'InvalidHandle',
                'a force handle cannot be combined by an off buffer'))

        check_torch_deterministic()

        # Automatic decide SM and QP count
        num_sms = handle.num_sms if num_sms == 0 else num_sms
        num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
        assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'

        bias_0, bias_1 = ElasticBuffer._unpack_bias(bias)
        combined_x, combined_topk_weights, event = \
            self.runtime.combine(x, topk_weights,
                                 bias_0, bias_1,
                                 handle.recv_src_metadata,
                                 handle.topk_idx,
                                 handle.psum_num_recv_tokens_per_scaleup_rank,
                                 handle.token_metadata_at_forward,
                                 handle.channel_linked_list,
                                 handle.num_experts,
                                 handle.num_max_tokens_per_rank,
                                 num_sms, num_qps,
                                 previous_event,
                                 previous_event_before_epilogue,
                                 async_with_compute_stream,
                                 allocate_on_comm_stream,
                                 handle.do_expand)
        return combined_x, combined_topk_weights, EventOverlap(event)
