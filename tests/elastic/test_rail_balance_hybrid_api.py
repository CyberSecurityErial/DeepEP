"""C080-A constructor contract and default-off identity tests.

Run directly; pytest is intentionally not required.
"""

from __future__ import annotations

import inspect
import os

from deep_ep.buffers import elastic as elastic_module


_INVALID_PREFIX = '[DeepEP rail_balance:InvalidConfiguration] '
_UNSUPPORTED_PREFIX = '[DeepEP rail_balance:UnsupportedConfiguration] '

_LEGACY_BYTES = 4 * 1024 * 1024
_ARENA_BYTES = 2 * 1024 * 1024


def _expect_error(error_type, expected_message, function):
    try:
        function()
    except error_type as error:
        assert str(error) == expected_message, str(error)
    else:
        raise AssertionError(f'expected {error_type.__name__}: {expected_message}')


def _run_force_constructor(legacy_helper=None, layout_helper=None,
                           force_size_helper=None, **rail_balance_kwargs):
    calls = {'order': []}

    class FakeGroup:
        def rank(self):
            return 3

        def size(self):
            return 8

        def barrier(self):
            calls['barrier'] = calls.get('barrier', 0) + 1

    class FakeCommHandle:
        def get(self):
            return 1234

    class FakeRuntime:
        def __init__(self, *args):
            calls['order'].append('runtime')
            calls['runtime_args'] = args

        def get_logical_domain_size(self):
            return 1, 8

        def get_physical_domain_size(self):
            return 1, 8

    if legacy_helper is None:
        legacy_helper = lambda *_args: _LEGACY_BYTES
    if layout_helper is None:
        layout_helper = lambda *_args: (0,) * 9 + (_ARENA_BYTES,)
    if force_size_helper is None:
        force_size_helper = lambda *_args: _LEGACY_BYTES + _ARENA_BYTES

    def fake_get_comm(group, force_new_comm=False):
        calls['order'].append('comm')
        calls['comm'] = (group, force_new_comm)
        return FakeCommHandle()

    def fake_legacy(*args):
        calls['order'].append('legacy')
        calls['legacy_args'] = args
        return legacy_helper(*args)

    def fake_layout(*args):
        calls['order'].append('layout')
        calls['layout_args'] = args
        return layout_helper(*args)

    def fake_force_size(*args):
        calls['order'].append('force_size')
        calls['force_size_args'] = args
        return force_size_helper(*args)

    capability_name = '_rail_balance_force_available'
    names = (
        'get_nccl_comm_handle', 'check_nvlink_connections',
        'check_fast_rdma_atomic_support')
    originals = {name: getattr(elastic_module, name) for name in names}
    original_legacy = elastic_module._C.calculate_elastic_buffer_size
    original_layout = elastic_module._C._get_rail_balance_hybrid_layout
    original_force_size = elastic_module._C._calculate_rail_balance_hybrid_buffer_size
    original_runtime = elastic_module._C.ElasticBuffer
    original_synchronize = elastic_module.torch.cuda.synchronize
    original_world_gate = elastic_module._run_rail_balance_world_gate
    old_host_capability = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    had_capability = hasattr(elastic_module._C, capability_name)
    old_capability = getattr(elastic_module._C, capability_name, None)
    original_override = os.environ.pop('EP_OVERRIDE_RDMA_SL', None)
    buffer = error = None
    try:
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
        setattr(elastic_module._C, capability_name,
                lambda: calls['order'].append('capability') or True)
        elastic_module.get_nccl_comm_handle = fake_get_comm
        elastic_module.check_nvlink_connections = lambda _group: None
        elastic_module.check_fast_rdma_atomic_support = lambda: False
        elastic_module._C.calculate_elastic_buffer_size = fake_legacy
        elastic_module._C._get_rail_balance_hybrid_layout = fake_layout
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = fake_force_size
        elastic_module._C.ElasticBuffer = FakeRuntime
        elastic_module.torch.cuda.synchronize = lambda: None

        def fake_world_gate(device_words, host_words, group):
            calls['order'].append('constructor_gate')
            calls['constructor_gate'] = (device_words, host_words, group)
            return elastic_module._decode_rail_balance_world_gate(host_words)

        elastic_module._run_rail_balance_world_gate = fake_world_gate

        try:
            buffer = elastic_module.ElasticBuffer(
                FakeGroup(),
                num_max_tokens_per_rank=128,
                hidden=1024,
                num_topk=4,
                rail_balance='force',
                rail_balance_proxy_slots_per_rank=32,
                **rail_balance_kwargs)
        except Exception as caught:
            error = caught
    finally:
        for name, value in originals.items():
            setattr(elastic_module, name, value)
        elastic_module._C.calculate_elastic_buffer_size = original_legacy
        elastic_module._C._get_rail_balance_hybrid_layout = original_layout
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = original_force_size
        elastic_module._C.ElasticBuffer = original_runtime
        elastic_module.torch.cuda.synchronize = original_synchronize
        elastic_module._run_rail_balance_world_gate = original_world_gate
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = old_host_capability
        if had_capability:
            setattr(elastic_module._C, capability_name, old_capability)
        else:
            delattr(elastic_module._C, capability_name)
        if original_override is not None:
            os.environ['EP_OVERRIDE_RDMA_SL'] = original_override
    return calls, buffer, error


def test_config_parser_is_strict_and_deterministic():
    parse = elastic_module._parse_rail_balance_config
    assert parse('off', 0) == ('off', 0, 0, 0)
    assert parse('force', 1) == ('force', 1, 0, 0)
    assert parse('force', 1, 'active', 20) == ('force', 1, 1, 20)
    assert parse('force', 1, 'adaptive', 3100) == \
        ('force', 1, 2, 3100)
    assert parse('legacy_exact', 1) == ('legacy_exact', 1, 0, 0)
    assert parse('one_hop', 1) == ('one_hop', 1, 0, 0)
    assert parse('adaptive', 1) == ('adaptive', 1, 0, 0)
    assert parse('force', (1 << 31) - 1) == \
        ('force', (1 << 31) - 1, 0, 0)
    make_manifest = elastic_module._make_rail_balance_constructor_manifest
    layout = (0,) * elastic_module._RAIL_BALANCE_CONSTRUCTOR_LAYOUT_FIELDS
    assert make_manifest('force', 8, arena_layout=layout)[5] == 1
    assert make_manifest('legacy_exact', 8, arena_layout=layout)[5] == 1
    assert make_manifest('one_hop', 8, arena_layout=layout)[5] == 2
    assert make_manifest('adaptive', 8, arena_layout=layout)[5] == 3

    invalid = (
        (None, 0,
         "rail_balance must be exactly one of ('off', 'force', "
         "'legacy_exact', 'one_hop', 'adaptive'), got None"),
        ('auto', 0,
         "rail_balance must be exactly one of ('off', 'force', "
         "'legacy_exact', 'one_hop', 'adaptive'), got 'auto'"),
        ('OFF', 0,
         "rail_balance must be exactly one of ('off', 'force', "
         "'legacy_exact', 'one_hop', 'adaptive'), got 'OFF'"),
        ('off', True,
         'rail_balance_proxy_slots_per_rank must be an integer in '
         '[0, 2147483647]'),
        ('off', -1,
         'rail_balance_proxy_slots_per_rank must be an integer in '
         '[0, 2147483647]'),
        ('force', 1 << 31,
         'rail_balance_proxy_slots_per_rank must be an integer in '
         '[0, 2147483647]'),
        ('off', 1,
         "rail_balance_proxy_slots_per_rank must be 0 when rail_balance='off'"),
        ('force', 0,
         'rail_balance_proxy_slots_per_rank must be positive when rail balancing is enabled'),
    )
    for mode, capacity, detail in invalid:
        _expect_error(
            ValueError, _INVALID_PREFIX + detail,
            lambda mode=mode, capacity=capacity: parse(mode, capacity))

    policy_invalid = (
        ('force', 1, None, 0,
         "rail_balance_policy must be exactly one of "
         "('all', 'active', 'adaptive'), got None"),
        ('force', 1, 'ACTIVE', 0,
         "rail_balance_policy must be exactly one of "
         "('all', 'active', 'adaptive'), got 'ACTIVE'"),
        ('force', 1, 'all', True,
         'rail_balance_threshold_percent must be an integer in [0, 3100]'),
        ('force', 1, 'all', 3101,
         'rail_balance_threshold_percent must be an integer in [0, 3100]'),
        ('off', 0, 'active', 0,
         "rail_balance_policy must be 'all' when rail_balance='off'"),
        ('off', 0, 'all', 1,
         "rail_balance_threshold_percent must be 0 when rail_balance='off'"),
    )
    for mode, capacity, policy, threshold, detail in policy_invalid:
        _expect_error(
            ValueError, _INVALID_PREFIX + detail,
            lambda mode=mode, capacity=capacity, policy=policy,
            threshold=threshold: parse(
                mode, capacity, policy, threshold))


def test_force_constructor_matrix_fails_with_stable_reasons():
    validate = elastic_module._validate_rail_balance_force_constructor
    valid = dict(
        num_bytes=None,
        num_cpu_bytes=0,
        num_max_tokens_per_rank=128,
        hidden=1024,
        num_topk=4,
        use_fp8_dispatch=False,
        deterministic=False,
        allow_hybrid_mode=True,
        allow_multiple_reduction=True,
    )
    validate(**valid)

    cases = (
        ('num_bytes', 2 * 1024 * 1024,
         'manual num_bytes is not supported'),
        ('num_cpu_bytes', 2 * 1024 * 1024,
         'num_cpu_bytes must be 0'),
        ('num_max_tokens_per_rank', 0,
         'num_max_tokens_per_rank must be a positive integer'),
        ('num_max_tokens_per_rank', True,
         'num_max_tokens_per_rank must be a positive integer'),
        ('hidden', 255,
         'hidden must be a positive multiple of 256'),
        ('num_topk', 0,
         'num_topk must be an integer in [1, 32]'),
        ('use_fp8_dispatch', 0,
         'use_fp8_dispatch must be a bool'),
        ('use_fp8_dispatch', True,
         'FP8 dispatch is not supported'),
        ('deterministic', 0,
         'deterministic must be a bool'),
        ('deterministic', True,
         'deterministic mode is not supported'),
        ('allow_hybrid_mode', 1,
         'allow_hybrid_mode must be a bool'),
        ('allow_hybrid_mode', False,
         'allow_hybrid_mode must be true'),
        ('allow_multiple_reduction', 1,
         'allow_multiple_reduction must be a bool'),
        ('allow_multiple_reduction', False,
         'allow_multiple_reduction must be true'),
    )
    for name, value, reason in cases:
        kwargs = dict(valid)
        kwargs[name] = value
        _expect_error(
            ValueError,
            _UNSUPPORTED_PREFIX + f'force-v1: {reason}',
            lambda kwargs=kwargs: validate(**kwargs))


def test_new_constructor_arguments_are_keyword_only_suffixes():
    parameters = tuple(inspect.signature(
        elastic_module.ElasticBuffer.__init__).parameters.values())
    old_names = (
        'self', 'group', 'num_bytes', 'num_cpu_bytes',
        'num_max_tokens_per_rank', 'hidden', 'num_topk',
        'use_fp8_dispatch', 'deterministic', 'allow_hybrid_mode',
        'allow_multiple_reduction', 'prefer_overlap_with_compute', 'sl_idx',
        'num_allocated_qps', 'num_cpu_timeout_secs', 'num_gpu_timeout_secs',
        'explicitly_destroy',
    )
    assert tuple(parameter.name for parameter in parameters[:-4]) == old_names
    assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for parameter in parameters[:-4])
    assert tuple(parameter.name for parameter in parameters[-4:]) == (
        'rail_balance', 'rail_balance_proxy_slots_per_rank',
        'rail_balance_policy', 'rail_balance_threshold_percent')
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in parameters[-4:])
    assert tuple(parameter.default for parameter in parameters[-4:]) == (
        'off', 0, 'all', 0)


def test_default_ep_handle_fields_are_unchanged():
    sentinel = object()
    handle = elastic_module.EPHandle(
        False,
        64, 1,
        128,
        12,
        sentinel,
        7, 9,
        [1, 2],
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        None,
        None)
    assert tuple(handle.__dict__) == (
        'do_expand',
        'num_experts',
        'expert_alignment',
        'num_max_tokens_per_rank',
        'num_sms',
        'topk_idx',
        'psum_num_recv_tokens_per_scaleup_rank',
        'psum_num_recv_tokens_per_expert',
        'num_unaligned_recv_tokens_per_expert',
        'num_recv_tokens_per_expert_list',
        'recv_src_metadata',
        'dst_buffer_slot_idx',
        'token_metadata_at_forward',
        'channel_linked_list',
        'num_recv_tokens',
        'num_expanded_tokens',
        'cached_recv_src_metadata_before_sort',
    )


def test_force_capability_rejects_before_group_cuda_or_collectives():
    class TripwireGroup:
        def __getattr__(self, name):
            raise AssertionError(f'force capability gate touched group.{name}')

    expected = (
        '[DeepEP rail_balance:FeatureUnavailable] force-v1 remains disabled '
        'until truthful D>1 Rail/Gin correctness passes')
    _expect_error(
        RuntimeError, expected,
        lambda: elastic_module.ElasticBuffer(
            TripwireGroup(),
            num_max_tokens_per_rank=128,
            hidden=1024,
            num_topk=4,
            rail_balance='force',
            rail_balance_proxy_slots_per_rank=32))

    # A partially upgraded extension must remain unavailable while the host
    # path is disabled, without even calling its capability symbol.
    capability_name = '_rail_balance_force_available'
    had_capability = hasattr(elastic_module._C, capability_name)
    old_capability = getattr(elastic_module._C, capability_name, None)
    old_host_capability = elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE
    called = []
    try:
        setattr(elastic_module._C, capability_name,
                lambda: called.append(True) or True)
        assert not elastic_module._rail_balance_force_available()
        assert called == []

        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = True
        delattr(elastic_module._C, capability_name)
        assert not elastic_module._rail_balance_force_available()

        setattr(elastic_module._C, capability_name, lambda: False)
        assert not elastic_module._rail_balance_force_available()
        setattr(elastic_module._C, capability_name, lambda: True)
        assert elastic_module._rail_balance_force_available()

        def broken_capability():
            raise RuntimeError('partial extension failure')

        setattr(elastic_module._C, capability_name, broken_capability)
        assert not elastic_module._rail_balance_force_available()
    finally:
        elastic_module._RAIL_BALANCE_FORCE_HOST_AVAILABLE = old_host_capability
        if had_capability:
            setattr(elastic_module._C, capability_name, old_capability)
        else:
            delattr(elastic_module._C, capability_name)


def test_off_path_uses_exact_legacy_size_and_runtime_arguments():
    calls = {}

    class FakeGroup:
        def rank(self):
            return 3

        def size(self):
            return 8

        def barrier(self):
            calls['barrier'] = calls.get('barrier', 0) + 1

    class FakeCommHandle:
        def get(self):
            return 1234

    class FakeRuntime:
        def __init__(self, *args):
            calls['runtime_args'] = args

        def get_logical_domain_size(self):
            return 1, 8

        def get_physical_domain_size(self):
            return 1, 8

    def fake_get_comm(group, force_new_comm=False):
        calls['comm'] = (group, force_new_comm)
        return FakeCommHandle()

    def fake_calculate(*args):
        calls['calculate_args'] = args
        return 2 * 1024 * 1024

    def force_helper_tripwire(*_args):
        raise AssertionError('off path called a force-only sizing helper')

    names = (
        'get_nccl_comm_handle', 'check_nvlink_connections',
        'check_fast_rdma_atomic_support')
    originals = {name: getattr(elastic_module, name) for name in names}
    original_calculate = elastic_module._C.calculate_elastic_buffer_size
    original_layout = elastic_module._C._get_rail_balance_hybrid_layout
    original_force_size = elastic_module._C._calculate_rail_balance_hybrid_buffer_size
    original_runtime = elastic_module._C.ElasticBuffer
    original_synchronize = elastic_module.torch.cuda.synchronize
    original_override = os.environ.pop('EP_OVERRIDE_RDMA_SL', None)
    try:
        elastic_module.get_nccl_comm_handle = fake_get_comm
        elastic_module.check_nvlink_connections = lambda _group: None
        elastic_module.check_fast_rdma_atomic_support = lambda: False
        elastic_module._C.calculate_elastic_buffer_size = fake_calculate
        elastic_module._C._get_rail_balance_hybrid_layout = force_helper_tripwire
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = force_helper_tripwire
        elastic_module._C.ElasticBuffer = FakeRuntime
        elastic_module.torch.cuda.synchronize = lambda: None

        group = FakeGroup()
        buffer = elastic_module.ElasticBuffer(
            group,
            num_max_tokens_per_rank=128,
            hidden=1024,
            num_topk=4)
    finally:
        for name, value in originals.items():
            setattr(elastic_module, name, value)
        elastic_module._C.calculate_elastic_buffer_size = original_calculate
        elastic_module._C._get_rail_balance_hybrid_layout = original_layout
        elastic_module._C._calculate_rail_balance_hybrid_buffer_size = original_force_size
        elastic_module._C.ElasticBuffer = original_runtime
        elastic_module.torch.cuda.synchronize = original_synchronize
        if original_override is not None:
            os.environ['EP_OVERRIDE_RDMA_SL'] = original_override

    assert calls['comm'] == (group, False)
    assert calls['calculate_args'] == (
        1234, 128, 1024, 4, False, True, True)
    assert calls['runtime_args'] == (
        3, 8, 1234, [], 2 * 1024 * 1024, 0,
        True, True, True, 3, 129, 300, 100, False)
    assert calls['barrier'] == 1
    assert buffer.num_bytes == 2 * 1024 * 1024
    assert 'rail_balance' not in buffer.__dict__
    assert 'rail_balance_proxy_slots_per_rank' not in buffer.__dict__
    assert not any(name.startswith('_rail_balance_') for name in buffer.__dict__)
    assert buffer.num_scaleout_ranks == 1
    assert buffer.num_scaleup_ranks == 8


def test_force_path_owns_checked_tail_arena_and_runtime_total():
    calls, buffer, error = _run_force_constructor()
    assert error is None, error
    assert buffer is not None
    assert calls['order'] == [
        'capability', 'layout', 'constructor_gate',
        'comm', 'legacy', 'force_size', 'constructor_gate', 'runtime']
    assert calls['comm'][1] is False
    assert calls['legacy_args'] == (
        1234, 128, 1024, 4, False, True, True)
    assert calls['layout_args'] == (1024, 4, 32)
    assert calls['force_size_args'] == (1234, 128, 1024, 4, 32)
    assert calls['runtime_args'] == (
        3, 8, 1234, [], _LEGACY_BYTES + _ARENA_BYTES, 0,
        True, True, True, 3, 129, 300, 100, False)
    assert calls['barrier'] == 1
    assert buffer.num_bytes == _LEGACY_BYTES + _ARENA_BYTES
    assert {
        name for name in buffer.__dict__ if name.startswith('_rail_balance_')
    } == {
        '_rail_balance_mode',
        '_rail_balance_proxy_slots_per_rank',
        '_rail_balance_policy',
        '_rail_balance_threshold_percent',
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
    assert buffer._rail_balance_mode == 'force'
    assert buffer._rail_balance_proxy_slots_per_rank == 32
    assert buffer._rail_balance_policy == 0
    assert buffer._rail_balance_threshold_percent == 0
    assert buffer._rail_balance_arena_offset == _LEGACY_BYTES
    assert buffer._rail_balance_arena_bytes == _ARENA_BYTES
    assert buffer._rail_balance_next_invocation_id == 1
    assert buffer._rail_balance_live_ticket is None
    assert buffer._rail_balance_terminal is False
    assert buffer._rail_balance_world_gate_device_words is \
        calls['constructor_gate'][0]
    assert buffer._rail_balance_world_gate_host_words is \
        calls['constructor_gate'][1]

    calls, buffer, error = _run_force_constructor(
        rail_balance_policy='adaptive',
        rail_balance_threshold_percent=20)
    assert error is None, error
    assert buffer._rail_balance_policy == 2
    assert buffer._rail_balance_threshold_percent == 20


def test_force_size_mismatch_fails_before_runtime_construction():
    calls, buffer, error = _run_force_constructor(
        force_size_helper=lambda *_args: _LEGACY_BYTES + _ARENA_BYTES + 1)
    assert buffer is None
    assert type(error) is RuntimeError
    assert str(error) == (
        '[DeepEP rail_balance:CollectivePreflight] constructor-sizing '
        'rejected rank 3 with error priority 17')
    assert calls['order'] == [
        'capability', 'layout', 'constructor_gate',
        'comm', 'legacy', 'force_size', 'constructor_gate']
    assert 'runtime_args' not in calls


def test_force_sizing_helper_exceptions_fail_before_runtime_construction():
    for failing_helper in ('legacy', 'layout', 'force_size'):
        marker = RuntimeError(f'{failing_helper} helper failed')

        def fail(*_args, marker=marker):
            raise marker

        overrides = {
            'legacy_helper': None,
            'layout_helper': None,
            'force_size_helper': None,
        }
        overrides[f'{failing_helper}_helper'] = fail
        calls, buffer, error = _run_force_constructor(**overrides)
        assert buffer is None
        if failing_helper == 'layout':
            assert type(error) is RuntimeError
            assert str(error) == (
                '[DeepEP rail_balance:CollectivePreflight] constructor '
                'rejected rank 3 with error priority 13')
        else:
            assert type(error) is RuntimeError
            assert str(error) == (
                '[DeepEP rail_balance:CollectivePreflight] '
                'constructor-sizing rejected rank 3 with error priority 17')
        assert calls['order'][-1] == 'constructor_gate'
        assert 'runtime' not in calls['order']
        assert 'runtime_args' not in calls


def run_all():
    tests = (
        test_config_parser_is_strict_and_deterministic,
        test_force_constructor_matrix_fails_with_stable_reasons,
        test_new_constructor_arguments_are_keyword_only_suffixes,
        test_default_ep_handle_fields_are_unchanged,
        test_force_capability_rejects_before_group_cuda_or_collectives,
        test_off_path_uses_exact_legacy_size_and_runtime_arguments,
        test_force_path_owns_checked_tail_arena_and_runtime_total,
        test_force_size_mismatch_fails_before_runtime_construction,
        test_force_sizing_helper_exceptions_fail_before_runtime_construction,
    )
    for test in tests:
        test()
    print(f'PASS {len(tests)}/{len(tests)} C080-A Hybrid API tests')


if __name__ == '__main__':
    run_all()
