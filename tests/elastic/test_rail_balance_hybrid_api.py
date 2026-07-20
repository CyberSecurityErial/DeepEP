"""C080-A constructor contract and default-off identity tests.

Run directly; pytest is intentionally not required.
"""

from __future__ import annotations

import inspect
import os

from deep_ep.buffers import elastic as elastic_module


_INVALID_PREFIX = '[DeepEP rail_balance:InvalidConfiguration] '
_UNSUPPORTED_PREFIX = '[DeepEP rail_balance:UnsupportedConfiguration] '


def _expect_error(error_type, expected_message, function):
    try:
        function()
    except error_type as error:
        assert str(error) == expected_message, str(error)
    else:
        raise AssertionError(f'expected {error_type.__name__}: {expected_message}')


def test_config_parser_is_strict_and_deterministic():
    parse = elastic_module._parse_rail_balance_config
    assert parse('off', 0) == ('off', 0)
    assert parse('force', 1) == ('force', 1)
    assert parse('force', (1 << 31) - 1) == ('force', (1 << 31) - 1)

    invalid = (
        (None, 0,
         "rail_balance must be exactly one of ('off', 'force'), got None"),
        ('auto', 0,
         "rail_balance must be exactly one of ('off', 'force'), got 'auto'"),
        ('OFF', 0,
         "rail_balance must be exactly one of ('off', 'force'), got 'OFF'"),
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
         "rail_balance_proxy_slots_per_rank must be positive when rail_balance='force'"),
    )
    for mode, capacity, detail in invalid:
        _expect_error(
            ValueError, _INVALID_PREFIX + detail,
            lambda mode=mode, capacity=capacity: parse(mode, capacity))


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
        ('use_fp8_dispatch', True,
         'FP8 dispatch is not supported'),
        ('deterministic', True,
         'deterministic mode is not supported'),
        ('allow_hybrid_mode', False,
         'allow_hybrid_mode must be true'),
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
    assert tuple(parameter.name for parameter in parameters[:-2]) == old_names
    assert all(parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for parameter in parameters[:-2])
    assert tuple(parameter.name for parameter in parameters[-2:]) == (
        'rail_balance', 'rail_balance_proxy_slots_per_rank')
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in parameters[-2:])
    assert parameters[-2].default == 'off'
    assert parameters[-1].default == 0


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
        '[DeepEP rail_balance:FeatureUnavailable] force-v1 is disabled until '
        'both Hybrid dispatch and combine are installed')
    _expect_error(
        RuntimeError, expected,
        lambda: elastic_module.ElasticBuffer(
            TripwireGroup(),
            num_max_tokens_per_rank=128,
            hidden=1024,
            num_topk=4,
            rail_balance='force',
            rail_balance_proxy_slots_per_rank=32))

    # An old extension has no capability symbol; a partially upgraded
    # extension may claim device support before the Python round trip exists.
    # Both states must remain unavailable.
    assert not elastic_module._rail_balance_force_available()
    capability_name = '_rail_balance_force_available'
    had_capability = hasattr(elastic_module._C, capability_name)
    old_capability = getattr(elastic_module._C, capability_name, None)
    try:
        setattr(elastic_module._C, capability_name, lambda: True)
        assert not elastic_module._rail_balance_force_available()
    finally:
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

    names = (
        'get_nccl_comm_handle', 'check_nvlink_connections',
        'check_fast_rdma_atomic_support')
    originals = {name: getattr(elastic_module, name) for name in names}
    original_calculate = elastic_module._C.calculate_elastic_buffer_size
    original_runtime = elastic_module._C.ElasticBuffer
    original_synchronize = elastic_module.torch.cuda.synchronize
    original_override = os.environ.pop('EP_OVERRIDE_RDMA_SL', None)
    try:
        elastic_module.get_nccl_comm_handle = fake_get_comm
        elastic_module.check_nvlink_connections = lambda _group: None
        elastic_module.check_fast_rdma_atomic_support = lambda: False
        elastic_module._C.calculate_elastic_buffer_size = fake_calculate
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
    assert buffer.num_scaleout_ranks == 1
    assert buffer.num_scaleup_ranks == 8


def run_all():
    tests = (
        test_config_parser_is_strict_and_deterministic,
        test_force_constructor_matrix_fails_with_stable_reasons,
        test_new_constructor_arguments_are_keyword_only_suffixes,
        test_default_ep_handle_fields_are_unchanged,
        test_force_capability_rejects_before_group_cuda_or_collectives,
        test_off_path_uses_exact_legacy_size_and_runtime_arguments,
    )
    for test in tests:
        test()
    print(f'PASS {len(tests)}/{len(tests)} C080-A Hybrid API tests')


if __name__ == '__main__':
    run_all()
