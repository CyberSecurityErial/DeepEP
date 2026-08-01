from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import struct
import types
from pathlib import Path

import pytest

import rail_balance_common_fair_route as route


def _rehash(artifact: dict[str, object]) -> None:
    artifact["canonical_sha256"] = route.canonical_sha256(artifact)


@pytest.mark.parametrize("mode", ["HT", "LL"])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("pattern", route.ROUTE_PATTERNS)
def test_all_frozen_route_cases_have_exact_shape_and_dtypes(
    mode: str,
    world_size: int,
    pattern: str,
) -> None:
    artifact = route.generate_route_artifact(
        mode=mode,
        world_size=world_size,
        pattern=pattern,
    )

    tokens = route.MODE_TOKENS[mode]
    count = world_size * tokens * route.TOP_K
    assert artifact["shape"] == [world_size, tokens, route.TOP_K]
    assert len(artifact["expert_ids"]) == count
    assert len(artifact["topk_weights_fp32_hex"]) == count
    assert artifact["case"]["num_experts"] == 256
    assert artifact["case"]["top_k"] == 8
    assert artifact["case"]["expert_owner_policy"] == "CONTIGUOUS_EQUAL"
    assert artifact["case"]["manifest_case_id"] == (
        f"derived-{mode.lower()}-n1-w{world_size}"
    )
    assert artifact["case"]["claim_scope"] == "DERIVED_SCALE_DOWN_CORRECTNESS"
    assert artifact["case"]["performance_claim_allowed"] is False
    assert artifact["case"]["semantic_scope"] == route.SEMANTIC_SCOPES[pattern]
    assert artifact["dtypes"] == {
        "expert_ids": "INT64",
        "topk_weights": "IEEE754_BINARY32_HEX_BITS",
    }
    assert route.validate_route_artifact(artifact) is artifact


def test_balanced_pattern_is_owner_balanced_per_token() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=4, pattern="balanced"
    )
    experts = artifact["expert_ids"]
    experts_per_owner = 256 // 4
    for rank in range(4):
        for token in range(128):
            offset = (rank * 128 + token) * 8
            owners = [
                expert // experts_per_owner for expert in experts[offset : offset + 8]
            ]
            assert [owners.count(owner) for owner in range(4)] == [2, 2, 2, 2]


def test_rail_hot_pattern_concentrates_each_source_on_one_owner() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=4, pattern="rail_hot"
    )
    experts = artifact["expert_ids"]
    assert artifact["case"]["semantic_scope"] == (
        "LOGICAL_OWNER_HOTSPOT_NOT_PHYSICAL_NIC_EVIDENCE"
    )
    assert artifact["case"]["performance_claim_allowed"] is False
    experts_per_owner = 256 // 4
    for rank in range(4):
        rank_values = experts[rank * 128 * 8 : (rank + 1) * 128 * 8]
        assert {expert // experts_per_owner for expert in rank_values} == {
            (rank + 1) % 4
        }
        assert len(set(rank_values)) == experts_per_owner


def test_expert_skew_is_owner_balanced_and_uses_two_hot_experts_per_owner() -> None:
    artifact = route.generate_route_artifact(
        mode="HT", world_size=4, pattern="expert_skew"
    )
    experts = artifact["expert_ids"]
    assert set(experts) == {0, 1, 64, 65, 128, 129, 192, 193}
    for rank in range(4):
        for token in range(4096):
            offset = (rank * 4096 + token) * 8
            owners = [expert // 64 for expert in experts[offset : offset + 8]]
            assert [owners.count(owner) for owner in range(4)] == [2, 2, 2, 2]


def test_weights_are_exact_finite_fp32_bits_and_sum_to_one() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    first_route = artifact["topk_weights_fp32_hex"][:8]
    decoded = [
        struct.unpack(">f", bytes.fromhex(value[2:]))[0] for value in first_route
    ]
    assert tuple(first_route) == route.WEIGHT_FP32_HEX
    assert sum(decoded) == 1.0


def test_generation_is_bit_deterministic() -> None:
    first = route.generate_route_artifact(mode="HT", world_size=4, pattern="rail_hot")
    second = route.generate_route_artifact(mode="HT", world_size=4, pattern="rail_hot")
    assert first == second
    assert route.canonical_bytes(first) == route.canonical_bytes(second)
    assert first["canonical_sha256"] == second["canonical_sha256"]


@pytest.mark.parametrize(
    ("mode", "world_size", "pattern"),
    [
        ("ll", 2, "balanced"),
        ("LL", 3, "balanced"),
        ("LL", True, "balanced"),
        ("LL", 2, "unknown"),
    ],
)
def test_generation_rejects_non_frozen_arguments(
    mode: object,
    world_size: object,
    pattern: object,
) -> None:
    with pytest.raises(route.CommonFairRouteError):
        route.generate_route_artifact(
            mode=mode,  # type: ignore[arg-type]
            world_size=world_size,  # type: ignore[arg-type]
            pattern=pattern,  # type: ignore[arg-type]
        )


def test_validator_rejects_shape_drift_with_a_matching_hash() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["shape"] = [2, 64, 8]
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match="shape"):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_case_id", "confirm-ll-n2-r8-w16"),
        ("claim_scope", "CONFIRMATORY_COMMON_FAIR"),
        ("performance_claim_allowed", True),
        ("semantic_scope", "PHYSICAL_NIC_EVIDENCE"),
    ],
)
def test_validator_rejects_promotion_of_derived_routes(
    field: str, value: object
) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="rail_hot"
    )
    artifact["case"][field] = value
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match=field):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), 1.0),
        (("shape", 0), 2.0),
        (("case", "tokens_per_rank"), 128.0),
        (("case", "num_experts"), 256.0),
        (("case", "top_k"), 8.0),
    ],
)
def test_validator_rejects_numeric_metadata_with_wrong_json_type(
    path: tuple[object, ...], value: object
) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    target: object = artifact
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError):
        route.validate_route_artifact(artifact)


def test_validator_rejects_out_of_range_expert_with_a_matching_hash() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["expert_ids"][0] = 256
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match=r"outside \[0, 256\)"):
        route.validate_route_artifact(artifact)


def test_validator_rejects_bool_expert_id() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["expert_ids"][0] = True
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match=r"expert_ids\[0\]"):
        route.validate_route_artifact(artifact)


def test_validator_rejects_duplicate_expert_with_a_matching_hash() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["expert_ids"][1] = artifact["expert_ids"][0]
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match="duplicate experts"):
        route.validate_route_artifact(artifact)


def test_validator_rejects_non_deterministic_valid_expert() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["expert_ids"][0] = 255
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match="deterministic route"):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize("weight", ["0X3f000000", "0x7f800000", "0x3f00000"])
def test_validator_rejects_invalid_fp32_weight_bits(weight: str) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["topk_weights_fp32_hex"][0] = weight
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match="FP32|non-finite"):
        route.validate_route_artifact(artifact)


def test_validator_rejects_hash_mismatch() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["canonical_sha256"] = "0" * 64
    with pytest.raises(route.CommonFairRouteError, match="canonical_sha256 mismatch"):
        route.validate_route_artifact(artifact)


def test_validator_rejects_unknown_fields_even_with_a_matching_hash() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    artifact["unexpected"] = None
    _rehash(artifact)
    with pytest.raises(route.CommonFairRouteError, match="fields differ"):
        route.validate_route_artifact(artifact)


def test_safe_write_and_load_round_trip(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=4, pattern="expert_skew"
    )
    path = tmp_path / "route.json"

    raw_sha, canonical_sha, size = route.write_route(path, artifact)
    loaded, loaded_raw_sha, loaded_canonical_sha = route.load_route(path)

    assert loaded == artifact
    assert loaded_raw_sha == raw_sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded_canonical_sha == canonical_sha == artifact["canonical_sha256"]
    assert size == path.stat().st_size
    assert path.read_bytes() == route.canonical_bytes(artifact) + b"\n"
    assert path.stat().st_mode & 0o777 == 0o644


def test_safe_writer_never_replaces_an_existing_file(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    path = tmp_path / "route.json"
    path.write_bytes(b"sentinel")
    with pytest.raises(route.CommonFairRouteError, match="cannot safely create"):
        route.write_route(path, artifact)
    assert path.read_bytes() == b"sentinel"


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"format":"first","format":"second"}\n', encoding="utf-8")
    with pytest.raises(route.CommonFairRouteError, match="duplicate JSON key"):
        route.load_route(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_nonfinite_json_constants(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(route.CommonFairRouteError, match="non-finite JSON constant"):
        route.load_route(path)


def test_loader_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(route.CommonFairRouteError, match="canonical byte encoding"):
        route.load_route(path)


def test_loader_rejects_utf8_bom_and_utf16(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    canonical = route.canonical_bytes(artifact) + b"\n"
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + canonical)
    utf16 = tmp_path / "utf16.json"
    utf16.write_bytes(canonical.decode("utf-8").encode("utf-16"))

    with pytest.raises(route.CommonFairRouteError, match="BOM"):
        route.load_route(bom)
    with pytest.raises(route.CommonFairRouteError, match="invalid route JSON"):
        route.load_route(utf16)


def test_loader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "route.fifo"
    os.mkfifo(path, 0o600)
    with pytest.raises(route.CommonFairRouteError, match="regular file"):
        route.load_route(path)


def test_loader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    target = tmp_path / "target.json"
    route.write_route(target, artifact)
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)

    with pytest.raises(route.CommonFairRouteError, match="cannot safely open"):
        route.load_route(symlink)
    with pytest.raises(route.CommonFairRouteError, match="exactly one hard link"):
        route.load_route(hardlink)


def test_loader_rejects_symlinked_parent(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    route.write_route(real_parent / "route.json", artifact)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(route.CommonFairRouteError, match="safely traverse"):
        route.load_route(linked_parent / "route.json")


def test_loader_rejects_writable_and_oversized_files(tmp_path: Path) -> None:
    writable = tmp_path / "writable.json"
    writable.write_bytes(b"{}\n")
    writable.chmod(0o666)
    with pytest.raises(route.CommonFairRouteError, match="group/world writable"):
        route.load_route(writable)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(route.MAX_ROUTE_BYTES + 1)
    with pytest.raises(route.CommonFairRouteError, match=r"1\.\."):
        route.load_route(oversized)


def test_loader_rejects_group_writable_parent(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    path = tmp_path / "route.json"
    route.write_route(path, artifact)
    tmp_path.chmod(0o770)
    try:
        with pytest.raises(route.CommonFairRouteError, match="route parent"):
            route.load_route(path)
    finally:
        tmp_path.chmod(0o700)


def test_loader_rejects_group_writable_intermediate_parent(tmp_path: Path) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    safe = unsafe / "safe"
    safe.mkdir(mode=0o700)
    path = safe / "route.json"
    path.write_bytes(route.canonical_bytes(artifact) + b"\n")
    unsafe.chmod(0o770)

    with pytest.raises(route.CommonFairRouteError, match="unsafe writable ancestor"):
        route.load_route(path)


@pytest.mark.parametrize(
    "changed_field", ["st_mode", "st_uid", "st_nlink", "st_ctime_ns"]
)
def test_loader_rejects_concurrent_file_metadata_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    path = tmp_path / "route.json"
    route.write_route(path, artifact)
    real_fstat = os.fstat
    regular_calls = 0

    def racing_fstat(descriptor: int) -> object:
        nonlocal regular_calls
        metadata = real_fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return metadata
        regular_calls += 1
        if regular_calls != 2:
            return metadata
        values = {
            name: getattr(metadata, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        if changed_field == "st_mode":
            values[changed_field] ^= stat.S_IWGRP
        else:
            values[changed_field] += 1
        return types.SimpleNamespace(**values)

    monkeypatch.setattr(route.os, "fstat", racing_fstat)
    with pytest.raises(route.CommonFairRouteError, match="changed while"):
        route.load_route(path)


def test_cli_generate_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli-route.json"
    assert (
        route.main(
            [
                "generate",
                "--mode",
                "LL",
                "--world-size",
                "2",
                "--pattern",
                "balanced",
                "--output",
                os.fspath(path),
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["status"] == "GENERATED"
    assert generated["manifest_case_id"] == "derived-ll-n1-w2"
    assert generated["claim_scope"] == "DERIVED_SCALE_DOWN_CORRECTNESS"
    assert generated["performance_claim_allowed"] is False

    assert route.main(["verify", os.fspath(path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "VERIFIED"
    assert verified["canonical_sha256"] == generated["canonical_sha256"]
    assert verified["semantic_scope"] == "LOGICAL_BALANCE_CORRECTNESS_ONLY"


def test_cli_reports_validation_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}\n", encoding="utf-8")
    assert route.main(["verify", os.fspath(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "COMMON_FAIR route error" in captured.err


def test_canonical_hash_excludes_only_the_self_hash_field() -> None:
    artifact = route.generate_route_artifact(
        mode="LL", world_size=2, pattern="balanced"
    )
    changed_self_hash = copy.deepcopy(artifact)
    changed_self_hash["canonical_sha256"] = "f" * 64
    assert route.canonical_sha256(changed_self_hash) == artifact["canonical_sha256"]

    changed_payload = copy.deepcopy(artifact)
    changed_payload["case"]["id"] = "different"
    assert route.canonical_sha256(changed_payload) != artifact["canonical_sha256"]
