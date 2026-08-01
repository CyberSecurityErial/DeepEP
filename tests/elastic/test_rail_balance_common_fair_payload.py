from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import rail_balance_common_fair_payload as payload


MODULE_PATH = Path(payload.__file__)


def test_frozen_generator_identity() -> None:
    assert payload.GENERATOR_ID == "rail-balance-deterministic-bf16-v1"
    assert payload.GENERATOR_VERSION == 1
    assert payload.SEED == 20260801
    assert payload.CONFORMANCE_VECTOR_VERSION == 1
    assert payload.ROW_MAJOR_INDEX_FORMULA == (
        "((rank * tokens_per_rank) + token) * hidden_size + hidden"
    )

    identity = payload.specification_identity()
    assert identity["generator_id"] == payload.GENERATOR_ID
    assert identity["seed"] == payload.SEED
    assert identity["byte_order"] == "little"
    assert identity["value_class"] == "FINITE_IEEE754_NORMAL_BF16"
    assert identity["stream_conformance"]["raw_bf16_le_sha256"] == (
        payload.STREAM_CONFORMANCE_SHA256
    )
    assert identity["stream_conformance"]["element_count"] == 4454
    assert json.loads(json.dumps(identity, allow_nan=False, sort_keys=True)) == identity


@pytest.mark.parametrize(
    ("counter", "expected"),
    (
        (0, 0xE220A8397B1DCDAF),
        (1, 0x910A2DEC89025CC1),
        (2, 0x975835DE1C9756CE),
        (20260801, 0xE1FC8B667B4CD66D),
        (0xFFFFFFFFFFFFFFFF, 0xE4D971771B652C20),
    ),
)
def test_splitmix64_standard_vectors(counter: int, expected: int) -> None:
    assert payload.splitmix64(counter) == expected


def test_versioned_coordinate_conformance_vectors() -> None:
    expected = (
        (0, 0x3F80, 1.0),
        (1, 0xBF80, -1.0),
        (2, 0x3C00, 0.0078125),
        (3, 0x3FFF, 1.9921875),
        (4, 0x3F29, 0.66015625),
        (127, 0xBCF3, -0.0296630859375),
        (4095, 0xBEEB, -0.458984375),
        (4096, 0x3F80, 1.0),
        (917504, 0x3F80, 1.0),
        (917509, 0x3DE8, 0.11328125),
        (939524095, 0x3E04, 0.12890625),
    )
    assert len(payload.CONFORMANCE_VECTORS) == len(expected)
    for vector, (expected_index, expected_bits, expected_float) in zip(
        payload.CONFORMANCE_VECTORS,
        expected,
        strict=True,
    ):
        observed_index = payload.global_index(
            vector.rank,
            vector.token,
            vector.hidden,
            world_size=vector.world_size,
            tokens_per_rank=vector.tokens_per_rank,
            hidden_size=vector.hidden_size,
        )
        assert observed_index == vector.global_index == expected_index
        assert (
            payload.payload_bf16_bits(
                vector.rank,
                vector.token,
                vector.hidden,
                world_size=vector.world_size,
                tokens_per_rank=vector.tokens_per_rank,
                hidden_size=vector.hidden_size,
            )
            == vector.bf16_bits
            == expected_bits
        )
        assert payload.bf16_bits_to_float(expected_bits) == expected_float


def test_row_major_coordinate_mapping_is_injective() -> None:
    world_size = 4
    tokens_per_rank = 17
    hidden_size = 31
    observed: set[int] = set()
    expected = 0
    for rank in range(world_size):
        for token in range(tokens_per_rank):
            for hidden in range(hidden_size):
                index = payload.global_index(
                    rank,
                    token,
                    hidden,
                    world_size=world_size,
                    tokens_per_rank=tokens_per_rank,
                    hidden_size=hidden_size,
                )
                assert index == expected
                assert index not in observed
                observed.add(index)
                expected += 1
    assert observed == set(range(world_size * tokens_per_rank * hidden_size))


def test_random_access_is_deterministic_and_order_independent() -> None:
    indices = [4, 5, 17, 127, 4095, 4099, 65537, 917509, 939524095]
    forward = {index: payload.bf16_bits_from_global_index(index) for index in indices}
    reverse = {
        index: payload.bf16_bits_from_global_index(index) for index in reversed(indices)
    }
    repeated = {index: payload.bf16_bits_from_global_index(index) for index in indices}
    assert forward == reverse == repeated


def test_every_sampled_value_is_bounded_finite_and_normal() -> None:
    for index in range(20_000):
        bits = payload.bf16_bits_from_global_index(index)
        value = payload.bf16_bits_to_float(bits)
        assert payload.is_finite_normal_bf16(bits)
        assert math.isfinite(value)
        assert payload.MIN_ABS_VALUE <= abs(value) <= payload.MAX_ABS_VALUE
        assert ((bits >> 7) & 0xFF) in range(120, 128)


def test_special_value_strategy_is_periodic_and_stays_normal() -> None:
    special_bits = (0x3F80, 0xBF80, 0x3C00, 0x3FFF)
    for period in (0, 4096, 8192):
        observed = tuple(
            payload.bf16_bits_from_global_index(period + offset) for offset in range(4)
        )
        assert observed == special_bits
        assert all(payload.is_finite_normal_bf16(bits) for bits in observed)


def test_bf16_to_float_is_exact_bit_expansion() -> None:
    assert payload.bf16_bits_to_float(0x0000) == 0.0
    assert payload.bf16_bits_to_float(0x8000) == -0.0
    assert payload.bf16_bits_to_float(0x3F80) == 1.0
    assert payload.bf16_bits_to_float(0xBF80) == -1.0
    assert math.isinf(payload.bf16_bits_to_float(0x7F80))
    assert math.isnan(payload.bf16_bits_to_float(0x7FC1))


def test_small_range_streaming_sha_is_frozen() -> None:
    observed = payload.streaming_payload_sha256(
        world_size=2,
        tokens_per_rank=17,
        hidden_size=131,
    )
    assert observed == (
        "4ad5880f10a886e72470cc50428b8ff960d12dfaa4292e6f8b159d5057416733"
    )
    assert observed == payload.STREAM_CONFORMANCE_SHA256

    # Independently exercise the declared raw little-endian stream contract.
    reference = hashlib.sha256()
    for index in range(2 * 17 * 131):
        reference.update(
            payload.bf16_bits_from_global_index(index).to_bytes(2, "little")
        )
    assert reference.hexdigest() == observed


@pytest.mark.parametrize(
    ("function", "args", "kwargs"),
    (
        (payload.splitmix64, (True,), {}),
        (payload.splitmix64, (-1,), {}),
        (payload.splitmix64, (1 << 64,), {}),
        (payload.bf16_bits_from_global_index, (False,), {}),
        (payload.bf16_bits_from_global_index, (-1,), {}),
        (
            payload.bf16_bits_from_global_index,
            (payload.MAX_GLOBAL_ELEMENTS,),
            {},
        ),
        (payload.bf16_bits_to_float, (True,), {}),
        (payload.bf16_bits_to_float, (-1,), {}),
        (payload.bf16_bits_to_float, (0x10000,), {}),
        (payload.is_finite_normal_bf16, (False,), {}),
        (payload.is_finite_normal_bf16, (0x10000,), {}),
    ),
)
def test_scalar_apis_reject_bool_negative_and_overflow(
    function: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(payload.PayloadSpecError):
        function(*args, **kwargs)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("coordinate", "extents"),
    (
        ((True, 0, 0), (2, 3, 4)),
        ((-1, 0, 0), (2, 3, 4)),
        ((2, 0, 0), (2, 3, 4)),
        ((0, True, 0), (2, 3, 4)),
        ((0, -1, 0), (2, 3, 4)),
        ((0, 3, 0), (2, 3, 4)),
        ((0, 0, True), (2, 3, 4)),
        ((0, 0, -1), (2, 3, 4)),
        ((0, 0, 4), (2, 3, 4)),
        ((0, 0, 0), (True, 3, 4)),
        ((0, 0, 0), (0, 3, 4)),
        ((0, 0, 0), (33, 3, 4)),
        ((0, 0, 0), (2, False, 4)),
        ((0, 0, 0), (2, 0, 4)),
        ((0, 0, 0), (2, 4097, 4)),
        ((0, 0, 0), (2, 3, True)),
        ((0, 0, 0), (2, 3, 0)),
        ((0, 0, 0), (2, 3, 7169)),
    ),
)
def test_coordinate_api_strictly_validates_every_axis(
    coordinate: tuple[object, object, object],
    extents: tuple[object, object, object],
) -> None:
    with pytest.raises(payload.PayloadSpecError):
        payload.global_index(
            *coordinate,
            world_size=extents[0],
            tokens_per_rank=extents[1],
            hidden_size=extents[2],
        )


def test_streaming_hash_has_a_hard_small_range_limit() -> None:
    with pytest.raises(payload.PayloadSpecError, match="conformance range"):
        payload.streaming_payload_sha256(
            world_size=32,
            tokens_per_rank=4096,
            hidden_size=7168,
        )
    with pytest.raises(payload.PayloadSpecError):
        payload.streaming_payload_sha256(
            world_size=True,
            tokens_per_rank=1,
            hidden_size=1,
        )


def test_self_check_and_cli_outputs_are_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = payload.self_check()
    assert result == {
        "generator_id": "rail-balance-deterministic-bf16-v1",
        "seed": 20260801,
        "status": "PASS",
        "stream_sha256": payload.STREAM_CONFORMANCE_SHA256,
    }

    assert payload.main(["--self-check"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == result

    assert payload.main(["--print-identity"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == payload.specification_identity()


def test_direct_cli_defaults_to_fail_closed_self_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASS"
    assert completed.stderr == ""


def test_payload_module_has_only_standard_library_imports() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".", 1)[0])
    assert imports <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "struct",
        "typing",
    }
