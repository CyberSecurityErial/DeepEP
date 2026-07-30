"""H6 constructor pre-window consensus and force-owner contract.

This is a direct source/fake test; pytest and a live ProcessGroup are not
required.  The fake collective executes the real fixed WORLD-gate runner and
models the elementwise MAX result that every rank would observe.
"""

from __future__ import annotations

import inspect
import os

import torch

from deep_ep.buffers import elastic as elastic_module


_WORDS = 128
_MANIFEST_WIDTH = 31
_MODE_FIELD = 5
_WORLD_FIELD = 6
_M_FIELD = 7
_H_FIELD = 8
_K_FIELD = 9
_PCAP_FIELD = 10
_POLICY_FIELD = 15
_THRESHOLD_FIELD = 16
_TWO_HOP_THRESHOLD_FIELD = 17
_MAX_TWO_HOP_FIELD = 18
_HOP_PENALTY_FIELD = 19
_LAYOUT_LEN_FIELD = 20
_LAYOUT_BEGIN = 21
_POLICY_IDS = {'all': 0, 'active': 1, 'adaptive': 2}

_SIZING_WIDTH = 20
_SIZING_MODE_FIELD = 5
_SIZING_WORLD_FIELD = 6
_SIZING_M_FIELD = 7
_SIZING_H_FIELD = 8
_SIZING_K_FIELD = 9
_SIZING_PCAP_FIELD = 10
_SIZING_LEGACY_FIELD = 11
_SIZING_ARENA_FIELD = 12
_SIZING_TOTAL_FIELD = 13
_SIZING_SL_FIELD = 14
_SIZING_QPS_FIELD = 15
_SIZING_CPU_TIMEOUT_FIELD = 16
_SIZING_GPU_TIMEOUT_FIELD = 17
_SIZING_OVERLAP_FIELD = 18
_SIZING_DESTROY_FIELD = 19

_LEGACY_BYTES = 4 * 1024 * 1024
_ARENA_BYTES = 2 * 1024 * 1024


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _token_bytes(hidden: int, num_topk: int, with_metadata: bool) -> int:
    metadata_bytes = num_topk * 8
    if with_metadata:
        metadata_bytes += (1 + num_topk) * 4
    return _align(hidden * 2, 32) + _align(metadata_bytes, 32)


def _reference_layout(hidden: int, num_topk: int, pcap: int):
    """Independent mirror of the ten-field HybridArenaLayout ABI."""
    channel_count_offset = _align(32, 32)
    channel_count_bytes = 1024 * 32 * 4
    proxy_ready_offset = _align(
        channel_count_offset + channel_count_bytes, 32)
    proxy_dispatch_offset = _align(
        proxy_ready_offset + pcap * 4, 32)
    dispatch_token_bytes = _token_bytes(hidden, num_topk, True)
    proxy_rail_staging_offset = _align(
        proxy_dispatch_offset + pcap * dispatch_token_bytes, 32)
    retained_rail_staging_offset = _align(
        proxy_rail_staging_offset + pcap * dispatch_token_bytes, 32)
    proxy_return_offset = _align(
        retained_rail_staging_offset + pcap * dispatch_token_bytes, 32)
    combine_token_bytes = _token_bytes(hidden, num_topk, False)
    raw_bytes = proxy_return_offset + pcap * combine_token_bytes
    arena_bytes = _align(raw_bytes, _ARENA_BYTES)
    return (
        0, 32,
        channel_count_offset, channel_count_bytes,
        proxy_dispatch_offset, dispatch_token_bytes,
        proxy_return_offset, combine_token_bytes,
        raw_bytes, arena_bytes,
    )


_LAYOUT = _reference_layout(1280, 7, 37)
assert _LAYOUT == (
    0, 32, 32, 131072, 131264, 2656,
    426080, 2624, 523168, 2097152)


class _FakeScalar:
    def __init__(self, value: int):
        self._value = value

    def item(self) -> int:
        return self._value


class _FakeTensor:
    _next_pointer = 0x100000

    def __init__(self, device_type: str, pinned: bool):
        self.dtype = torch.int64
        self.device = torch.device(device_type)
        self.shape = (_WORDS,)
        self._pinned = pinned
        self._values = [0] * _WORDS
        self._pointer = _FakeTensor._next_pointer
        _FakeTensor._next_pointer += 0x1000

    def is_contiguous(self) -> bool:
        return True

    def is_pinned(self) -> bool:
        return self._pinned

    def data_ptr(self) -> int:
        return self._pointer

    def zero_(self):
        self._values[:] = [0] * _WORDS
        return self

    def copy_(self, source, non_blocking=False):
        assert isinstance(source, _FakeTensor)
        assert non_blocking is True
        self._values[:] = source._values
        return self

    def __getitem__(self, index: int) -> _FakeScalar:
        return _FakeScalar(self._values[index])

    def __setitem__(self, index: int, value: int) -> None:
        self._values[index] = int(value)


class _FakeStream:
    def __init__(self, calls):
        self._calls = calls

    def synchronize(self):
        self._calls['order'].append('gate_stream_sync')
        self._calls['gate_stream_sync'] = \
            self._calls.get('gate_stream_sync', 0) + 1


def _decode_local_manifest(words: _FakeTensor, width=_MANIFEST_WIDTH):
    result = []
    for index in range(width):
        maximum = words._values[2 + 2 * index]
        minimum = -words._values[3 + 2 * index]
        assert minimum == maximum, (index, minimum, maximum)
        result.append(maximum)
    return result


def _configuration(mode: str, *, m=131, h=1280, k=7, pcap=37,
                   sl=3, qps=0, cpu_timeout=300, gpu_timeout=100,
                   prefer_overlap=True, explicitly_destroy=False,
                   policy='all', threshold=0):
    if mode == 'off':
        pcap = 0
        policy = 'all'
        threshold = 0
    return dict(
        mode=mode, m=m, h=h, k=k, pcap=pcap,
        sl=sl, qps=qps,
        cpu_timeout=cpu_timeout, gpu_timeout=gpu_timeout,
        prefer_overlap=prefer_overlap,
        explicitly_destroy=explicitly_destroy,
        policy=policy, threshold=threshold)


def _assert_manifest_semantics(fields, config, world_size):
    # The first five words are the constructor protocol identity.  Their
    # concrete magic is private, but width and all configuration positions are
    # fixed so MAX mismatch reports are deterministic on every rank.
    assert len(fields) == _MANIFEST_WIDTH
    assert all(type(value) is int and value >= 0 for value in fields)
    assert fields[4] == _MANIFEST_WIDTH
    assert fields[_MODE_FIELD] == (0 if config['mode'] == 'off' else 1)
    assert fields[_WORLD_FIELD] == world_size
    if config['mode'] == 'force':
        expected_layout = _reference_layout(
            config['h'], config['k'], config['pcap'])
        assert fields[_M_FIELD] == config['m']
        assert fields[_H_FIELD] == config['h']
        assert fields[_K_FIELD] == config['k']
        assert fields[_PCAP_FIELD] == config['pcap']
        assert fields[11:15] == [0, 0, 1, 1]
        assert fields[_POLICY_FIELD] == _POLICY_IDS[config['policy']]
        assert fields[_THRESHOLD_FIELD] == config['threshold']
        assert fields[_LAYOUT_LEN_FIELD] == len(expected_layout)
        assert tuple(fields[_LAYOUT_BEGIN:_MANIFEST_WIDTH]) == expected_layout
    else:
        # Off contributes only mode/world protocol identity.  It must not call
        # a force sizing/layout helper merely to populate the temporary gate.
        assert fields[_M_FIELD:_LAYOUT_LEN_FIELD + 1] == [0] * 14
        assert fields[_LAYOUT_LEN_FIELD] == 0
        assert fields[_LAYOUT_BEGIN:_MANIFEST_WIDTH] == [0] * len(_LAYOUT)


def _peer_manifest(local_fields, peer_config):
    fields = list(local_fields)
    fields[_MODE_FIELD] = 0 if peer_config['mode'] == 'off' else 1
    if peer_config['mode'] == 'force':
        peer_layout = _reference_layout(
            peer_config['h'], peer_config['k'], peer_config['pcap'])
        fields[_M_FIELD] = peer_config['m']
        fields[_H_FIELD] = peer_config['h']
        fields[_K_FIELD] = peer_config['k']
        fields[_PCAP_FIELD] = peer_config['pcap']
        fields[11:15] = [0, 0, 1, 1]
        fields[_POLICY_FIELD] = _POLICY_IDS[peer_config['policy']]
        fields[_THRESHOLD_FIELD] = peer_config['threshold']
        fields[_LAYOUT_LEN_FIELD] = len(peer_layout)
        fields[_LAYOUT_BEGIN:_MANIFEST_WIDTH] = peer_layout
    else:
        fields[_M_FIELD:_LAYOUT_LEN_FIELD + 1] = [0] * 14
        fields[_LAYOUT_LEN_FIELD] = 0
        fields[_LAYOUT_BEGIN:_MANIFEST_WIDTH] = [0] * len(_LAYOUT)
    # Protocol identity and WORLD size must be identical for this fake pair.
    assert fields[:5] == local_fields[:5]
    assert fields[_WORLD_FIELD] == local_fields[_WORLD_FIELD]
    _assert_manifest_semantics(fields, peer_config, fields[_WORLD_FIELD])
    return fields


def _resolved_qps(config, fast_atomic=False):
    if config['qps'] != 0:
        return config['qps']
    return 65 if fast_atomic else 129


def _assert_sizing_manifest_semantics(
        fields, config, world_size, legacy_bytes, arena_bytes, total_bytes,
        *, resolved_sl=None, fast_atomic=False):
    assert len(fields) == _SIZING_WIDTH
    assert all(type(value) is int and value >= 0 for value in fields)
    assert fields[4] == _SIZING_WIDTH
    assert fields[_SIZING_MODE_FIELD] == 1
    assert fields[_SIZING_WORLD_FIELD] == world_size
    assert fields[_SIZING_M_FIELD] == config['m']
    assert fields[_SIZING_H_FIELD] == config['h']
    assert fields[_SIZING_K_FIELD] == config['k']
    assert fields[_SIZING_PCAP_FIELD] == config['pcap']
    assert fields[_SIZING_LEGACY_FIELD] == legacy_bytes
    assert fields[_SIZING_ARENA_FIELD] == arena_bytes
    assert fields[_SIZING_TOTAL_FIELD] == total_bytes
    assert fields[_SIZING_SL_FIELD] == (
        config['sl'] if resolved_sl is None else resolved_sl)
    assert fields[_SIZING_QPS_FIELD] == _resolved_qps(
        config, fast_atomic)
    assert fields[_SIZING_CPU_TIMEOUT_FIELD] == config['cpu_timeout']
    assert fields[_SIZING_GPU_TIMEOUT_FIELD] == config['gpu_timeout']
    assert fields[_SIZING_OVERLAP_FIELD] == int(config['prefer_overlap'])
    assert fields[_SIZING_DESTROY_FIELD] == int(
        config['explicitly_destroy'])


def _peer_sizing_manifest(local_fields, overrides):
    fields = list(local_fields)
    for index, value in overrides.items():
        fields[index] = value
    return fields


def _run_constructor(config, *, rank=0, peer_config=None,
                     layout_override=None,
                     expected_local_error_priority=0,
                     peer_error_priority=0, peer_error_rank=0,
                     legacy_override=None, force_size_override=None,
                     fast_atomic=False, fast_atomic_error=None,
                     env_sl_override=None,
                     expected_sizing_error_priority=0,
                     peer_sizing_error_priority=0,
                     peer_sizing_error_rank=0,
                     peer_sizing_overrides=None):
    calls = {'order': [], 'allocations': [], 'all_reduce': []}

    class FakeGroup:
        def rank(self):
            return rank

        def size(self):
            return 2

        def barrier(self):
            calls['order'].append('barrier')
            calls['barrier'] = calls.get('barrier', 0) + 1

    class FakeCommHandle:
        def get(self):
            return 1234

    class FakeRuntime:
        def __init__(self, *args):
            calls['order'].append('runtime')
            calls['runtime_args'] = args

        def get_logical_domain_size(self):
            return 1, 2

        def get_physical_domain_size(self):
            return 1, 2

        def __getattr__(self, name):
            raise AssertionError(
                f'constructor touched runtime/JIT helper {name!r}')

    group = FakeGroup()

    def fake_empty(size, *, dtype=None, device=None, pin_memory=False,
                   **kwargs):
        assert not kwargs, kwargs
        normalized_size = (size,) if type(size) is int else tuple(size)
        assert normalized_size == (_WORDS,)
        assert dtype == torch.int64
        if isinstance(device, torch.device):
            device_type = device.type
        elif type(device) is int:
            device_type = 'cuda'
        elif device is None:
            device_type = 'cpu'
        else:
            device_type = str(device).split(':', 1)[0]
        if device_type == 'cuda':
            assert pin_memory is False
        else:
            assert device_type == 'cpu'
            assert pin_memory is True
        tensor = _FakeTensor(device_type, pin_memory)
        calls['allocations'].append(tensor)
        calls['order'].append(f'allocate_{device_type}')
        return tensor

    def fake_all_reduce(tensor, op=None, group=None, async_op=False):
        assert tensor is calls['allocations'][0]
        assert tensor.device.type == 'cuda'
        assert tensor.dtype == torch.int64
        assert tuple(tensor.shape) == (_WORDS,)
        assert op == elastic_module.dist.ReduceOp.MAX
        assert group is FakeGroup_instance
        assert async_op is False
        gate_index = len(calls['all_reduce'])
        calls['order'].append(f'max{gate_index + 1}')
        calls['all_reduce'].append((tensor, op, group))

        local_error_key = tensor._values[0]
        local_priority, local_error_rank = \
            elastic_module._decode_rail_balance_world_gate_error_key(
                local_error_key)
        expected_priority = (
            expected_local_error_priority if gate_index == 0
            else expected_sizing_error_priority)
        expected_peer_priority = (
            peer_error_priority if gate_index == 0
            else peer_sizing_error_priority)
        expected_peer_rank = (
            peer_error_rank if gate_index == 0
            else peer_sizing_error_rank)
        assert local_priority == expected_priority
        assert local_error_rank == (
            rank if expected_priority else -1)

        if gate_index == 0:
            local_fields = _decode_local_manifest(tensor)
            if expected_priority:
                # Any rank-local failure must encode a caller-independent
                # fallback manifest; word zero wins before field inspection.
                _assert_manifest_semantics(
                    local_fields, _configuration('off'), 0)
            else:
                _assert_manifest_semantics(
                    local_fields, config, FakeGroup_instance.size())

            if not expected_priority and not expected_peer_priority:
                remote_fields = _peer_manifest(
                    local_fields, peer_config or config)
                for index, (local_value, remote_value) in enumerate(
                        zip(local_fields, remote_fields)):
                    tensor._values[2 + 2 * index] = max(
                        local_value, remote_value)
                    tensor._values[3 + 2 * index] = -min(
                        local_value, remote_value)
        else:
            assert gate_index == 1
            local_fields = _decode_local_manifest(tensor, _SIZING_WIDTH)
            assert local_fields[3] == 2
            assert local_fields[4] == _SIZING_WIDTH
            if not expected_priority:
                resolved_sl = int(env_sl_override) \
                    if env_sl_override is not None else config['sl']
                _assert_sizing_manifest_semantics(
                    local_fields, config, FakeGroup_instance.size(),
                    calls['legacy_value'], calls['arena_value'],
                    calls['force_size_value'],
                    resolved_sl=resolved_sl, fast_atomic=fast_atomic)
            if not expected_priority and not expected_peer_priority:
                remote_fields = _peer_sizing_manifest(
                    local_fields, peer_sizing_overrides or {})
                for index, (local_value, remote_value) in enumerate(
                        zip(local_fields, remote_fields)):
                    tensor._values[2 + 2 * index] = max(
                        local_value, remote_value)
                    tensor._values[3 + 2 * index] = -min(
                        local_value, remote_value)

        peer_error_key = \
            elastic_module._make_rail_balance_world_gate_error_key(
                expected_peer_priority, expected_peer_rank)
        tensor._values[0] = max(local_error_key, peer_error_key)
        return None

    def fake_get_comm(actual_group, force_new_comm=False):
        assert actual_group is FakeGroup_instance
        calls['order'].append('comm')
        calls['comm'] = (actual_group, force_new_comm)
        return FakeCommHandle()

    def fake_legacy(*args):
        calls['order'].append('legacy')
        calls['legacy_args'] = args
        if isinstance(legacy_override, BaseException):
            raise legacy_override
        value = _LEGACY_BYTES if legacy_override is None else legacy_override
        calls['legacy_value'] = value
        return value

    def fake_layout(*args):
        if config['mode'] == 'off':
            raise AssertionError('off constructor called force layout helper')
        calls['order'].append('layout')
        calls['layout_args'] = args
        if isinstance(layout_override, BaseException):
            raise layout_override
        if layout_override is not None:
            value = layout_override
        else:
            value = _reference_layout(*args)
        calls['arena_value'] = value[-1]
        return value

    def fake_force_size(*args):
        if config['mode'] == 'off':
            raise AssertionError('off constructor called force size helper')
        calls['order'].append('force_size')
        calls['force_size_args'] = args
        if isinstance(force_size_override, BaseException):
            raise force_size_override
        if force_size_override is None:
            value = calls.get('legacy_value', _LEGACY_BYTES) + \
                calls.get('arena_value', _ARENA_BYTES)
        else:
            value = force_size_override
        calls['force_size_value'] = value
        return value

    def fake_fast_atomic_support():
        calls['order'].append('fast_atomic')
        if fast_atomic_error is not None:
            raise fast_atomic_error
        return fast_atomic

    def fake_check_nvlink(_group):
        if config['mode'] == 'force':
            raise AssertionError(
                'force constructor entered legacy NVLink/object-collective helper')
        calls['order'].append('nvlink_check')

    FakeGroup_instance = group
    names = (
        'get_nccl_comm_handle', 'check_nvlink_connections',
        'check_fast_rdma_atomic_support')
    originals = {name: getattr(elastic_module, name) for name in names}
    original_empty = elastic_module.torch.empty
    original_current_stream = elastic_module.torch.cuda.current_stream
    original_synchronize = elastic_module.torch.cuda.synchronize
    original_all_reduce = elastic_module.dist.all_reduce
    original_all_gather_object = elastic_module.dist.all_gather_object
    original_legacy = elastic_module._C.calculate_elastic_buffer_size
    original_layout = elastic_module._C._get_rail_balance_hybrid_layout
    original_force_size = \
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size
    original_runtime = elastic_module._C.ElasticBuffer
    capability_name = '_rail_balance_force_available'
    had_capability = hasattr(elastic_module._C, capability_name)
    original_capability = getattr(elastic_module._C, capability_name, None)
    original_host_capability = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    original_override = os.environ.pop('EP_OVERRIDE_RDMA_SL', None)

    def fake_capability():
        calls['order'].append('capability')
        calls['capability'] = calls.get('capability', 0) + 1
        return True

    buffer = error = None
    try:
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
        setattr(elastic_module._C, capability_name, fake_capability)
        elastic_module.torch.empty = fake_empty
        elastic_module.torch.cuda.current_stream = \
            lambda _device=None: _FakeStream(calls)
        elastic_module.torch.cuda.synchronize = \
            lambda: calls['order'].append('cuda_sync')
        elastic_module.dist.all_reduce = fake_all_reduce
        elastic_module.dist.all_gather_object = \
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError('constructor used an object collective'))
        elastic_module.get_nccl_comm_handle = fake_get_comm
        elastic_module.check_nvlink_connections = fake_check_nvlink
        elastic_module.check_fast_rdma_atomic_support = \
            fake_fast_atomic_support
        elastic_module._C.calculate_elastic_buffer_size = fake_legacy
        elastic_module._C._get_rail_balance_hybrid_layout = fake_layout
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = \
            fake_force_size
        elastic_module._C.ElasticBuffer = FakeRuntime
        if env_sl_override is not None:
            os.environ['EP_OVERRIDE_RDMA_SL'] = str(env_sl_override)

        try:
            buffer = elastic_module.ElasticBuffer(
                group,
                num_max_tokens_per_rank=config['m'],
                hidden=config['h'],
                num_topk=config['k'],
                prefer_overlap_with_compute=config['prefer_overlap'],
                sl_idx=config['sl'],
                num_allocated_qps=config['qps'],
                num_cpu_timeout_secs=config['cpu_timeout'],
                num_gpu_timeout_secs=config['gpu_timeout'],
                explicitly_destroy=config['explicitly_destroy'],
                rail_balance=config['mode'],
                rail_balance_proxy_slots_per_rank=config['pcap'],
                rail_balance_policy=config['policy'],
                rail_balance_threshold_percent=config['threshold'])
        except Exception as caught:
            error = caught
    finally:
        for name, value in originals.items():
            setattr(elastic_module, name, value)
        elastic_module.torch.empty = original_empty
        elastic_module.torch.cuda.current_stream = original_current_stream
        elastic_module.torch.cuda.synchronize = original_synchronize
        elastic_module.dist.all_reduce = original_all_reduce
        elastic_module.dist.all_gather_object = original_all_gather_object
        elastic_module._C.calculate_elastic_buffer_size = original_legacy
        elastic_module._C._get_rail_balance_hybrid_layout = original_layout
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = \
            original_force_size
        elastic_module._C.ElasticBuffer = original_runtime
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = \
            original_host_capability
        if had_capability:
            setattr(elastic_module._C, capability_name, original_capability)
        else:
            delattr(elastic_module._C, capability_name)
        os.environ.pop('EP_OVERRIDE_RDMA_SL', None)
        if original_override is not None:
            os.environ['EP_OVERRIDE_RDMA_SL'] = original_override
    return calls, buffer, error


def _assert_fixed_max_gates(calls, *, force, count):
    # Off joins based on the enabled host protocol and does not need to probe
    # a force-only compiled capability. Force probes it exactly once.
    assert calls.get('capability', 0) == int(force)
    assert len(calls['allocations']) == 2
    device_words, host_words = calls['allocations']
    assert device_words.device.type == 'cuda'
    assert device_words.dtype == torch.int64
    assert tuple(device_words.shape) == (_WORDS,)
    assert host_words.device.type == 'cpu'
    assert host_words.dtype == torch.int64
    assert tuple(host_words.shape) == (_WORDS,)
    assert host_words.is_pinned()
    assert len(calls['all_reduce']) == count
    assert all(entry[0] is device_words for entry in calls['all_reduce'])
    assert calls['gate_stream_sync'] == count
    if 'comm' in calls['order']:
        assert calls['order'].index('max1') < calls['order'].index('comm')
    if 'runtime' in calls['order']:
        assert calls['order'].index(f'max{count}') < \
            calls['order'].index('runtime')


def test_unanimous_off_pays_one_gate_then_is_exact_legacy_identity():
    config = _configuration('off')
    calls, buffer, error = _run_constructor(config)
    assert error is None, error
    assert buffer is not None
    _assert_fixed_max_gates(calls, force=False, count=1)
    assert calls['legacy_args'] == (
        1234, config['m'], config['h'], config['k'], False, True, True)
    assert calls['runtime_args'] == (
        0, 2, 1234, [], _LEGACY_BYTES, 0,
        True, True, True, 3, 129, 300, 100, False)
    assert calls['barrier'] == 1
    assert buffer.num_bytes == _LEGACY_BYTES
    assert calls['order'].count('nvlink_check') == 1
    assert not any(name.startswith('_rail_balance_')
                   for name in buffer.__dict__)
    assert 'layout' not in calls['order']
    assert 'force_size' not in calls['order']


def test_unanimous_force_retains_the_exact_gate_storage_and_idle_owner():
    config = _configuration('force')
    calls, buffer, error = _run_constructor(config)
    assert error is None, error
    assert buffer is not None
    _assert_fixed_max_gates(calls, force=True, count=2)
    assert calls['layout_args'] == (config['h'], config['k'], config['pcap'])
    assert calls['force_size_args'] == (
        1234, config['m'], config['h'], config['k'], config['pcap'])
    assert calls['runtime_args'] == (
        0, 2, 1234, [], _LEGACY_BYTES + _ARENA_BYTES, 0,
        True, True, True, 3, 129, 300, 100, False)
    assert buffer.num_bytes == _LEGACY_BYTES + _ARENA_BYTES
    assert 'nvlink_check' not in calls['order']
    # Gate 1's caller-stream completion is the final observable operation
    # before the already-prepared C++ runtime/window constructor.
    gate_sync = len(calls['order']) - 1 - \
        calls['order'][::-1].index('gate_stream_sync')
    assert calls['order'][gate_sync + 1] == 'runtime'

    device_words, host_words = calls['allocations']
    expected_fields = {
        '_rail_balance_mode',
        '_rail_balance_proxy_slots_per_rank',
        '_rail_balance_policy',
        '_rail_balance_threshold_percent',
        '_rail_balance_two_hop_threshold_percent',
        '_rail_balance_max_two_hop_percent',
        '_rail_balance_hop_penalty_percent',
        '_rail_balance_arena_offset',
        '_rail_balance_arena_bytes',
        '_rail_balance_world_gate_device_words',
        '_rail_balance_world_gate_host_words',
        '_rail_balance_owner_token',
        '_rail_balance_next_invocation_id',
        '_rail_balance_live_ticket',
        '_rail_balance_terminal',
        '_rail_balance_zero_move_bypass_budget',
        '_rail_balance_zero_move_common_fields',
    }
    assert {
        name for name in buffer.__dict__ if name.startswith('_rail_balance_')
    } == expected_fields
    assert buffer._rail_balance_world_gate_device_words is device_words
    assert buffer._rail_balance_world_gate_host_words is host_words
    assert buffer._rail_balance_owner_token is not None
    assert buffer._rail_balance_owner_token is not buffer
    assert buffer._rail_balance_next_invocation_id == 1
    assert buffer._rail_balance_live_ticket is None
    assert buffer._rail_balance_terminal is False
    assert buffer._rail_balance_zero_move_bypass_budget == 0
    assert buffer._rail_balance_zero_move_common_fields is None


def test_force_runtime_arguments_are_frozen_before_gate_two_commit():
    source = inspect.getsource(elastic_module.ElasticBuffer.__init__)
    prepare = source.index('force_runtime_args = (')
    gate = source.index('sizing_gate_result = _run_rail_balance_world_gate(')
    accepted = source.index(
        "_raise_rail_balance_world_gate_failure(\n"
        "                'constructor-sizing', sizing_gate_result)", gate)
    runtime = source.index(
        'force_runtime = _C.ElasticBuffer(*force_runtime_args)', accepted)
    assert prepare < gate < accepted < runtime
    between = source[accepted:runtime]
    # After the accepted fixed gate there may be only the failure helper call
    # itself and whitespace/comments before window/runtime construction.
    assert 'self.' not in between
    assert 'get_nccl_comm_handle' not in between
    assert 'calculate_' not in between
    assert 'check_' not in between
    assert 'os.environ' not in between


def _assert_consistent_pre_window_rejection(first, second, field_index):
    calls_a, buffer_a, error_a = first
    calls_b, buffer_b, error_b = second
    assert buffer_a is None and buffer_b is None
    assert error_a is not None and error_b is not None
    assert type(error_a) is type(error_b)
    assert str(error_a) == str(error_b)
    message = str(error_a)
    assert message.startswith('[DeepEP rail_balance:ConfigurationMismatch] ')
    assert str(field_index) in message
    for calls in (calls_a, calls_b):
        _assert_fixed_max_gates(
            calls, force='capability' in calls, count=1)
        assert 'comm' not in calls['order']
        assert 'runtime' not in calls['order']
        assert 'legacy' not in calls['order']
        assert 'force_size' not in calls['order']


def test_mixed_off_force_is_rejected_consistently_before_window_creation():
    off = _configuration('off')
    force = _configuration('force')
    _assert_consistent_pre_window_rejection(
        _run_constructor(off, rank=0, peer_config=force),
        _run_constructor(force, rank=1, peer_config=off),
        _MODE_FIELD)


def test_force_geometry_mismatches_are_consistent_and_pre_window():
    base = _configuration('force')
    cases = (
        ('m', _M_FIELD, base['m'] + 2),
        ('h', _H_FIELD, base['h'] + 256),
        ('k', _K_FIELD, base['k'] + 1),
        ('pcap', _PCAP_FIELD, base['pcap'] + 2),
        ('policy', _POLICY_FIELD, 'active'),
        ('threshold', _THRESHOLD_FIELD, 20),
    )
    for name, field_index, peer_value in cases:
        peer = dict(base)
        peer[name] = peer_value
        _assert_consistent_pre_window_rejection(
            _run_constructor(base, rank=0, peer_config=peer),
            _run_constructor(peer, rank=1, peer_config=base),
            field_index)


def _assert_consistent_sizing_rejection(
        first, second, *, field_index=None, priority=None):
    calls_a, buffer_a, error_a = first
    calls_b, buffer_b, error_b = second
    assert buffer_a is None and buffer_b is None
    assert type(error_a) is RuntimeError
    assert type(error_b) is RuntimeError
    assert str(error_a) == str(error_b)
    if field_index is not None:
        message = str(error_a)
        assert message.startswith(
            '[DeepEP rail_balance:ConfigurationMismatch] '
            'constructor-sizing ')
        assert f'field {field_index} ' in message
    else:
        assert priority is not None
        assert str(error_a) == (
            '[DeepEP rail_balance:CollectivePreflight] '
            'constructor-sizing rejected rank 0 with error priority '
            f'{priority}')
    for calls in (calls_a, calls_b):
        _assert_fixed_max_gates(calls, force=True, count=2)
        assert 'comm' in calls['order']
        assert 'runtime' not in calls['order']
        assert 'barrier' not in calls['order']


def test_force_sizing_and_runtime_config_mismatches_use_gate_two():
    base = _configuration('force')
    base_total = _LEGACY_BYTES + _ARENA_BYTES

    # Real rank-local legacy sizing differences also change total. The first
    # reported mismatch must still be the stable legacy-bytes field.
    peer_legacy = _LEGACY_BYTES + _ARENA_BYTES
    peer_total = peer_legacy + _ARENA_BYTES
    _assert_consistent_sizing_rejection(
        _run_constructor(
            base, rank=0,
            peer_sizing_overrides={
                _SIZING_LEGACY_FIELD: peer_legacy,
                _SIZING_TOTAL_FIELD: peer_total}),
        _run_constructor(
            base, rank=1, legacy_override=peer_legacy,
            peer_sizing_overrides={
                _SIZING_LEGACY_FIELD: _LEGACY_BYTES,
                _SIZING_TOTAL_FIELD: base_total}),
        field_index=_SIZING_LEGACY_FIELD)

    # Arena and total are repeated in Gate 1 to detect helper/ABI drift after
    # the geometry gate, even though the normal implementation is deterministic.
    for field_index, peer_value in (
            (_SIZING_ARENA_FIELD, _ARENA_BYTES + 32),
            (_SIZING_TOTAL_FIELD, base_total + 32)):
        _assert_consistent_sizing_rejection(
            _run_constructor(
                base, rank=0,
                peer_sizing_overrides={field_index: peer_value}),
            _run_constructor(
                base, rank=1,
                peer_sizing_overrides={field_index: peer_value}),
            field_index=field_index)

    runtime_cases = (
        ('sl', _SIZING_SL_FIELD, 4, 3),
        ('qps', _SIZING_QPS_FIELD, 64, 129),
        ('cpu_timeout', _SIZING_CPU_TIMEOUT_FIELD, 301, 300),
        ('gpu_timeout', _SIZING_GPU_TIMEOUT_FIELD, 101, 100),
        ('prefer_overlap', _SIZING_OVERLAP_FIELD, False, True),
        ('explicitly_destroy', _SIZING_DESTROY_FIELD, True, False),
    )
    for name, field_index, peer_value, base_value in runtime_cases:
        peer = dict(base)
        peer[name] = peer_value
        _assert_consistent_sizing_rejection(
            _run_constructor(
                base, rank=0, peer_config=peer,
                peer_sizing_overrides={field_index: int(peer_value)}),
            _run_constructor(
                peer, rank=1, peer_config=base,
                peer_sizing_overrides={field_index: int(base_value)}),
            field_index=field_index)


def _assert_local_sizing_error(kwargs, priority):
    healthy = _configuration('force')
    bad_kwargs = dict(kwargs)
    bad_kwargs.update(
        rank=0, expected_sizing_error_priority=priority)
    _assert_consistent_sizing_rejection(
        _run_constructor(healthy, **bad_kwargs),
        _run_constructor(
            healthy, rank=1,
            peer_sizing_error_priority=priority,
            peer_sizing_error_rank=0),
        priority=priority)


def test_force_local_sizing_and_runtime_failures_still_join_gate_two():
    sizing_failures = (
        dict(legacy_override=RuntimeError('legacy sizing failed')),
        dict(force_size_override=RuntimeError('force sizing failed')),
        dict(force_size_override=_LEGACY_BYTES + _ARENA_BYTES + 1),
        dict(legacy_override=1 << 63),
        # Both independent formulas remain mutually consistent, so only the
        # 2 MiB registered-window alignment check can reject this value.
        dict(legacy_override=_LEGACY_BYTES + 1),
    )
    for kwargs in sizing_failures:
        _assert_local_sizing_error(kwargs, 17)

    runtime_failures = (
        dict(env_sl_override='not-an-integer'),
        dict(fast_atomic_error=RuntimeError('QP probe failed')),
    )
    for kwargs in runtime_failures:
        _assert_local_sizing_error(kwargs, 18)

    invalid_qps = _configuration('force', qps=-1)
    _assert_consistent_sizing_rejection(
        _run_constructor(
            invalid_qps, rank=0,
            expected_sizing_error_priority=18),
        _run_constructor(
            _configuration('force'), rank=1,
            peer_sizing_error_priority=18,
            peer_sizing_error_rank=0),
        priority=18)


def _assert_consistent_fixed_error(first, second, priority):
    calls_a, buffer_a, error_a = first
    calls_b, buffer_b, error_b = second
    assert buffer_a is None and buffer_b is None
    assert type(error_a) is RuntimeError
    assert type(error_b) is RuntimeError
    expected = (
        '[DeepEP rail_balance:CollectivePreflight] constructor rejected '
        f'rank 0 with error priority {priority}')
    assert str(error_a) == expected
    assert str(error_b) == expected
    for calls in (calls_a, calls_b):
        _assert_fixed_max_gates(calls, force=True, count=1)
        assert 'comm' not in calls['order']
        assert 'runtime' not in calls['order']
        assert 'legacy' not in calls['order']
        assert 'force_size' not in calls['order']


def test_local_int64_envelope_errors_reach_the_same_fixed_gate():
    healthy = _configuration('force')

    huge_m = dict(healthy)
    huge_m['m'] = 1 << 63
    _assert_consistent_fixed_error(
        _run_constructor(
            huge_m, rank=0, expected_local_error_priority=14),
        _run_constructor(
            healthy, rank=1,
            peer_error_priority=14, peer_error_rank=0),
        14)

    huge_layout = list(_LAYOUT)
    huge_layout[8] = 1 << 63
    _assert_consistent_fixed_error(
        _run_constructor(
            healthy, rank=0, layout_override=tuple(huge_layout),
            expected_local_error_priority=14),
        _run_constructor(
            healthy, rank=1,
            peer_error_priority=14, peer_error_rank=0),
        14)


def test_layout_helper_failure_reaches_priority_13_gate():
    healthy = _configuration('force')
    _assert_consistent_fixed_error(
        _run_constructor(
            healthy, rank=0,
            layout_override=RuntimeError('rank-local layout failure'),
            expected_local_error_priority=13),
        _run_constructor(
            healthy, rank=1,
            peer_error_priority=13, peer_error_rank=0),
        13)


def run_all():
    tests = (
        test_unanimous_off_pays_one_gate_then_is_exact_legacy_identity,
        test_unanimous_force_retains_the_exact_gate_storage_and_idle_owner,
        test_force_runtime_arguments_are_frozen_before_gate_two_commit,
        test_mixed_off_force_is_rejected_consistently_before_window_creation,
        test_force_geometry_mismatches_are_consistent_and_pre_window,
        test_force_sizing_and_runtime_config_mismatches_use_gate_two,
        test_force_local_sizing_and_runtime_failures_still_join_gate_two,
        test_local_int64_envelope_errors_reach_the_same_fixed_gate,
        test_layout_helper_failure_reaches_priority_13_gate,
    )
    for test in tests:
        test()
    print(f'PASS {len(tests)}/{len(tests)} H6 constructor preflight tests')


if __name__ == '__main__':
    run_all()
