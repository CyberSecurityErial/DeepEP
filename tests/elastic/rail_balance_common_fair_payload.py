"""Deterministic, random-access BF16 payloads for COMMON_FAIR experiments.

The generator is deliberately independent of CUDA, PyTorch, NumPy, and the
host C library's random-number implementation.  A payload element is a pure
function of its row-major global index.  Callers can therefore regenerate any
element without allocating or walking the full experiment tensor.

Version 1 emits only finite IEEE-754 normal BF16 values.  Four exact values are
reserved at the start of every 4096-element period; all other values use a
SplitMix64 word to select sign, exponent, and fraction.  The exponent is kept
in [120, 127], so every generated magnitude is in [2**-7, 1.9921875].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from typing import Sequence


GENERATOR_ID = "rail-balance-deterministic-bf16-v1"
GENERATOR_VERSION = 1
SEED = 20260801

MAX_WORLD_SIZE = 32
MAX_TOKENS_PER_RANK = 4096
MAX_HIDDEN_SIZE = 7168
MAX_GLOBAL_ELEMENTS = MAX_WORLD_SIZE * MAX_TOKENS_PER_RANK * MAX_HIDDEN_SIZE
MAX_STREAM_ELEMENTS = 1_000_000

ROW_MAJOR_INDEX_FORMULA = "((rank * tokens_per_rank) + token) * hidden_size + hidden"
BYTE_ORDER = "little"
VALUE_CLASS = "FINITE_IEEE754_NORMAL_BF16"
MIN_ABS_VALUE = 2.0**-7
MAX_ABS_VALUE = 1.9921875

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB

_EXPONENT_MIN = 120
_EXPONENT_COUNT = 8
_SPECIAL_PERIOD = 4096
_SPECIAL_BITS = (0x3F80, 0xBF80, 0x3C00, 0x3FFF)

CONFORMANCE_VECTOR_VERSION = 1
STREAM_CONFORMANCE_WORLD_SIZE = 2
STREAM_CONFORMANCE_TOKENS_PER_RANK = 17
STREAM_CONFORMANCE_HIDDEN_SIZE = 131
STREAM_CONFORMANCE_ELEMENTS = (
    STREAM_CONFORMANCE_WORLD_SIZE
    * STREAM_CONFORMANCE_TOKENS_PER_RANK
    * STREAM_CONFORMANCE_HIDDEN_SIZE
)


class PayloadSpecError(ValueError):
    """Raised when a payload coordinate or extent violates the frozen spec."""


@dataclass(frozen=True)
class ConformanceVector:
    """One versioned coordinate-to-BF16 compatibility vector."""

    world_size: int
    tokens_per_rank: int
    hidden_size: int
    rank: int
    token: int
    hidden: int
    global_index: int
    bf16_bits: int


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum:
        raise PayloadSpecError(f"{name} must be an integer >= {minimum}")
    assert isinstance(value, int)
    if maximum is not None and value > maximum:
        raise PayloadSpecError(f"{name} must be <= {maximum}")
    return value


def _extents(
    world_size: object,
    tokens_per_rank: object,
    hidden_size: object,
) -> tuple[int, int, int]:
    return (
        _integer(
            world_size,
            "world_size",
            minimum=1,
            maximum=MAX_WORLD_SIZE,
        ),
        _integer(
            tokens_per_rank,
            "tokens_per_rank",
            minimum=1,
            maximum=MAX_TOKENS_PER_RANK,
        ),
        _integer(
            hidden_size,
            "hidden_size",
            minimum=1,
            maximum=MAX_HIDDEN_SIZE,
        ),
    )


def global_index(
    rank: object,
    token: object,
    hidden: object,
    *,
    world_size: object,
    tokens_per_rank: object,
    hidden_size: object,
) -> int:
    """Return the exact row-major counter for one payload coordinate."""

    world, tokens, width = _extents(world_size, tokens_per_rank, hidden_size)
    rank_value = _integer(rank, "rank", maximum=world - 1)
    token_value = _integer(token, "token", maximum=tokens - 1)
    hidden_value = _integer(hidden, "hidden", maximum=width - 1)
    return ((rank_value * tokens) + token_value) * width + hidden_value


def splitmix64(counter: object) -> int:
    """Return the standard SplitMix64 output for one unsigned 64-bit counter."""

    value = _integer(counter, "counter", maximum=_MASK64)
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def bf16_bits_from_global_index(index: object) -> int:
    """Generate one finite normal BF16 bit pattern from a global index."""

    counter = _integer(index, "global_index", maximum=MAX_GLOBAL_ELEMENTS - 1)
    special_slot = counter % _SPECIAL_PERIOD
    if special_slot < len(_SPECIAL_BITS):
        return _SPECIAL_BITS[special_slot]

    random_word = splitmix64((SEED + counter) & _MASK64)
    sign = (random_word >> 63) & 1
    exponent = _EXPONENT_MIN + ((random_word >> 7) % _EXPONENT_COUNT)
    fraction = random_word & 0x7F
    return (sign << 15) | (exponent << 7) | fraction


def payload_bf16_bits(
    rank: object,
    token: object,
    hidden: object,
    *,
    world_size: object,
    tokens_per_rank: object,
    hidden_size: object,
) -> int:
    """Generate one payload element without materializing neighboring values."""

    index = global_index(
        rank,
        token,
        hidden,
        world_size=world_size,
        tokens_per_rank=tokens_per_rank,
        hidden_size=hidden_size,
    )
    return bf16_bits_from_global_index(index)


def bf16_bits_to_float(bits: object) -> float:
    """Expand one BF16 bit pattern to its exact IEEE binary32 value."""

    value = _integer(bits, "bf16_bits", maximum=0xFFFF)
    return struct.unpack(">f", struct.pack(">I", value << 16))[0]


def is_finite_normal_bf16(bits: object) -> bool:
    """Return whether ``bits`` encodes a finite, nonzero, non-subnormal BF16."""

    value = _integer(bits, "bf16_bits", maximum=0xFFFF)
    exponent = (value >> 7) & 0xFF
    return 0 < exponent < 0xFF


def streaming_payload_sha256(
    *,
    world_size: object,
    tokens_per_rank: object,
    hidden_size: object,
) -> str:
    """Hash a bounded row-major BF16 stream without retaining the payload.

    The digest covers exactly two little-endian bytes per element and no
    header.  The element count is capped because this API is a conformance
    check, not an invitation to synthesize a full benchmark tensor on CPU.
    """

    world, tokens, width = _extents(world_size, tokens_per_rank, hidden_size)
    element_count = world * tokens * width
    if element_count > MAX_STREAM_ELEMENTS:
        raise PayloadSpecError(
            "streaming conformance range exceeds " f"{MAX_STREAM_ELEMENTS} elements"
        )

    digest = hashlib.sha256()
    chunk = bytearray()
    for index in range(element_count):
        chunk.extend(bf16_bits_from_global_index(index).to_bytes(2, BYTE_ORDER))
        if len(chunk) >= 4096:
            digest.update(chunk)
            chunk.clear()
    digest.update(chunk)
    return digest.hexdigest()


# These vectors are compatibility data, not examples computed at import time.
# Changing any field requires a new generator id and conformance-vector version.
CONFORMANCE_VECTORS = (
    ConformanceVector(2, 128, 7168, 0, 0, 0, 0, 0x3F80),
    ConformanceVector(2, 128, 7168, 0, 0, 1, 1, 0xBF80),
    ConformanceVector(2, 128, 7168, 0, 0, 2, 2, 0x3C00),
    ConformanceVector(2, 128, 7168, 0, 0, 3, 3, 0x3FFF),
    ConformanceVector(2, 128, 7168, 0, 0, 4, 4, 0x3F29),
    ConformanceVector(2, 128, 7168, 0, 0, 127, 127, 0xBCF3),
    ConformanceVector(2, 128, 7168, 0, 0, 4095, 4095, 0xBEEB),
    ConformanceVector(2, 128, 7168, 0, 0, 4096, 4096, 0x3F80),
    ConformanceVector(4, 128, 7168, 1, 0, 0, 917504, 0x3F80),
    ConformanceVector(4, 128, 7168, 1, 0, 5, 917509, 0x3DE8),
    ConformanceVector(32, 4096, 7168, 31, 4095, 7167, 939524095, 0x3E04),
)

# SHA256 of the raw little-endian BF16 stream for the frozen small range above.
STREAM_CONFORMANCE_SHA256 = (
    "4ad5880f10a886e72470cc50428b8ff960d12dfaa4292e6f8b159d5057416733"
)


def specification_identity() -> dict[str, object]:
    """Return the complete machine-readable identity needed by a manifest."""

    return {
        "algorithm": {
            "counter_input": "(seed + global_index) modulo 2**64",
            "counter_hash": "SplitMix64",
            "fraction_bits": "splitmix64_word bits [6:0]",
            "index_formula": ROW_MAJOR_INDEX_FORMULA,
            "normal_exponent_max": _EXPONENT_MIN + _EXPONENT_COUNT - 1,
            "normal_exponent_min": _EXPONENT_MIN,
            "normal_exponent_selector": ("120 + ((splitmix64_word >> 7) modulo 8)"),
            "sign_bit": "splitmix64_word bit 63",
            "special_bits": list(_SPECIAL_BITS),
            "special_selector": (
                "if (global_index modulo 4096) < 4, return "
                "special_bits[global_index modulo 4096]"
            ),
            "special_period": _SPECIAL_PERIOD,
            "splitmix64_constants_hex": [
                "9e3779b97f4a7c15",
                "bf58476d1ce4e5b9",
                "94d049bb133111eb",
            ],
        },
        "byte_order": BYTE_ORDER,
        "conformance_vector_version": CONFORMANCE_VECTOR_VERSION,
        "conformance_vectors": [asdict(vector) for vector in CONFORMANCE_VECTORS],
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "limits": {
            "hidden_size": MAX_HIDDEN_SIZE,
            "tokens_per_rank": MAX_TOKENS_PER_RANK,
            "world_size": MAX_WORLD_SIZE,
        },
        "seed": SEED,
        "stream_conformance": {
            "encoding": "raw row-major uint16 little-endian; no header",
            "element_count": STREAM_CONFORMANCE_ELEMENTS,
            "hidden_size": STREAM_CONFORMANCE_HIDDEN_SIZE,
            "raw_bf16_le_sha256": STREAM_CONFORMANCE_SHA256,
            "tokens_per_rank": STREAM_CONFORMANCE_TOKENS_PER_RANK,
            "world_size": STREAM_CONFORMANCE_WORLD_SIZE,
        },
        "value_class": VALUE_CLASS,
        "value_range": {
            "max_abs_inclusive": MAX_ABS_VALUE,
            "min_abs_inclusive": MIN_ABS_VALUE,
        },
    }


def self_check() -> dict[str, object]:
    """Verify all frozen vectors and the bounded streaming digest."""

    for vector in CONFORMANCE_VECTORS:
        observed_index = global_index(
            vector.rank,
            vector.token,
            vector.hidden,
            world_size=vector.world_size,
            tokens_per_rank=vector.tokens_per_rank,
            hidden_size=vector.hidden_size,
        )
        if observed_index != vector.global_index:
            raise RuntimeError(f"global index mismatch at conformance vector {vector}")
        observed_bits = bf16_bits_from_global_index(observed_index)
        if observed_bits != vector.bf16_bits:
            raise RuntimeError(f"BF16 mismatch at conformance vector {vector}")
        if not is_finite_normal_bf16(observed_bits):
            raise RuntimeError(f"non-normal BF16 at conformance vector {vector}")

    observed_sha256 = streaming_payload_sha256(
        world_size=STREAM_CONFORMANCE_WORLD_SIZE,
        tokens_per_rank=STREAM_CONFORMANCE_TOKENS_PER_RANK,
        hidden_size=STREAM_CONFORMANCE_HIDDEN_SIZE,
    )
    if observed_sha256 != STREAM_CONFORMANCE_SHA256:
        raise RuntimeError(
            "stream conformance mismatch: "
            f"expected={STREAM_CONFORMANCE_SHA256}, observed={observed_sha256}"
        )
    return {
        "generator_id": GENERATOR_ID,
        "seed": SEED,
        "status": "PASS",
        "stream_sha256": observed_sha256,
    }


def _json_line(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or self-check the COMMON_FAIR BF16 payload spec."
    )
    parser.add_argument(
        "--print-identity",
        action="store_true",
        help="print the canonical machine-readable specification identity",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify conformance vectors and the frozen streaming SHA256",
    )
    arguments = parser.parse_args(argv)

    # No arguments is intentionally a self-check, making the direct invocation
    # a useful fail-closed health probe.
    run_self_check = arguments.self_check or not arguments.print_identity
    if arguments.print_identity:
        print(_json_line(specification_identity()))
    if run_self_check:
        print(_json_line(self_check()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
