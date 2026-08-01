from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import stat
import struct
import types
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

import rail_balance_common_fair_route_compact as route


@lru_cache(maxsize=1)
def _cached_base() -> dict[str, object]:
    return route.generate_route_artifact(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="balanced",
    )


def _base() -> dict[str, object]:
    return copy.deepcopy(_cached_base())


def _rehash(artifact: dict[str, object]) -> None:
    artifact["canonical_sha256"] = route.canonical_sha256(artifact)


def _contains_float(value: object) -> bool:
    if type(value) is float:
        return True
    if type(value) is list:
        return any(_contains_float(item) for item in value)
    if type(value) is dict:
        return any(_contains_float(item) for item in value.values())
    return False


@pytest.mark.parametrize("manifest_case_id", sorted(route.CASE_SPECS))
@pytest.mark.parametrize("pattern", route.ROUTE_PATTERNS)
def test_all_confirmatory_cases_and_patterns_are_compact_and_exact(
    manifest_case_id: str,
    pattern: str,
) -> None:
    artifact = route.generate_route_artifact(
        manifest_case_id=manifest_case_id,
        pattern=pattern,
    )
    mode, nodes, world_size, tokens = route.CASE_SPECS[manifest_case_id]
    count = world_size * tokens * route.TOP_K

    assert artifact["format"] == "rail-balance-logical-route-compact-v2"
    assert artifact["version"] == 2
    assert artifact["case"]["manifest_case_id"] == manifest_case_id
    assert artifact["case"]["mode"] == mode
    assert artifact["case"]["nodes"] == nodes
    assert artifact["case"]["ranks_per_node"] == 8
    assert artifact["case"]["world_size"] == world_size
    assert artifact["case"]["tokens_per_rank"] == tokens
    assert artifact["case"]["num_experts"] == 256
    assert artifact["case"]["hidden"] == 7168
    assert artifact["case"]["top_k"] == 8
    assert artifact["case"]["expert_owner_policy"] == "CONTIGUOUS_EQUAL"
    assert artifact["shape"] == [world_size, tokens, 8]
    assert artifact["logical_tensors"]["expert_ids"]["count"] == count
    assert artifact["logical_tensors"]["topk_weights"]["count"] == count
    assert "expert_ids" not in artifact
    assert "topk_weights" not in artifact
    assert len(artifact["weight_slot_fp32_bits"]) == 8
    assert len(route.canonical_bytes(artifact)) < 2048
    assert not _contains_float(artifact)


def test_the_four_manifest_case_ids_are_frozen_without_cross_product_drift() -> None:
    assert set(route.CASE_SPECS) == {
        "confirmatory-ht-n2-w16",
        "confirmatory-ht-n4-w32",
        "confirmatory-ll-n2-w16",
        "confirmatory-ll-n4-w32",
    }


def test_maximum_route_generation_is_bit_deterministic() -> None:
    first = route.generate_route_artifact(
        manifest_case_id="confirmatory-ht-n4-w32",
        pattern="rail_hot",
        seed=123456789,
    )
    second = route.generate_route_artifact(
        manifest_case_id="confirmatory-ht-n4-w32",
        pattern="rail_hot",
        seed=123456789,
    )
    assert first == second
    assert route.canonical_bytes(first) == route.canonical_bytes(second)
    assert first["canonical_sha256"] == second["canonical_sha256"]


def test_balanced_pattern_has_exact_global_owner_and_expert_balance() -> None:
    values = route.iter_expert_ids(
        manifest_case_id="confirmatory-ll-n4-w32",
        pattern="balanced",
        seed=7,
    )
    expert_counts = Counter(values)
    assert set(expert_counts) == set(range(256))
    assert len(set(expert_counts.values())) == 1
    owner_counts = Counter(expert // 8 for expert in expert_counts.elements())
    assert set(owner_counts) == set(range(32))
    assert len(set(owner_counts.values())) == 1


def test_expert_skew_is_owner_balanced_with_two_hot_experts_per_owner() -> None:
    values = route.iter_expert_ids(
        manifest_case_id="confirmatory-ll-n4-w32",
        pattern="expert_skew",
        seed=17,
    )
    expert_counts = Counter(values)
    owner_counts: Counter[int] = Counter()
    locals_by_owner: dict[int, set[int]] = {owner: set() for owner in range(32)}
    for expert, count in expert_counts.items():
        owner = expert // 8
        owner_counts[owner] += count
        locals_by_owner[owner].add(expert % 8)
    assert len(set(owner_counts.values())) == 1
    assert all(locals_seen == {0, 1} for locals_seen in locals_by_owner.values())
    assert len(expert_counts) == 64


def test_rail_hot_uses_only_remote_nodes_and_one_fixed_logical_rail() -> None:
    manifest_case_id = "confirmatory-ll-n4-w32"
    seed = 21
    values = iter(
        route.iter_expert_ids(
            manifest_case_id=manifest_case_id,
            pattern="rail_hot",
            seed=seed,
        )
    )
    remote_nodes_by_source = {node: set() for node in range(4)}
    local_ranks: set[int] = set()
    for source_rank in range(32):
        source_node = source_rank // 8
        for _token in range(128):
            token_values = [next(values) for _slot in range(8)]
            assert len(set(token_values)) == 8
            for expert in token_values:
                owner = expert // 8
                remote_node = owner // 8
                assert remote_node != source_node
                remote_nodes_by_source[source_node].add(remote_node)
                local_ranks.add(owner % 8)
    with pytest.raises(StopIteration):
        next(values)
    assert local_ranks == {seed % 8}
    for source_node, remote_nodes in remote_nodes_by_source.items():
        assert remote_nodes == set(range(4)) - {source_node}


def test_rail_hot_never_claims_physical_evidence() -> None:
    artifact = route.generate_route_artifact(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="rail_hot",
    )
    case = artifact["case"]
    assert case["physical_evidence_status"] == (
        "REQUIRES_PHYSICAL_TOPOLOGY_AND_COUNTER_VERIFICATION"
    )
    assert case["physical_topology_or_counter_evidence_present"] is False
    assert case["route_artifact_claims_performance"] is False
    assert "PHYSICAL_TOPOLOGY" not in artifact["algorithm"]["pattern_invariant"]


def test_streaming_hashes_use_frozen_little_endian_encodings() -> None:
    artifact = route.generate_route_artifact(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="expert_skew",
        seed=9,
    )
    expert_hash = hashlib.sha256()
    for expert in route.iter_expert_ids(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="expert_skew",
        seed=9,
    ):
        expert_hash.update(struct.pack("<q", expert))
    weight_hash = hashlib.sha256()
    for bits in route.iter_weight_fp32_bits(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="expert_skew",
        seed=9,
    ):
        weight_hash.update(struct.pack("<I", bits))
    assert (
        expert_hash.hexdigest()
        == artifact["logical_tensors"]["expert_ids"]["streaming_sha256"]
    )
    assert (
        weight_hash.hexdigest()
        == artifact["logical_tensors"]["topk_weights"]["streaming_sha256"]
    )


@pytest.mark.parametrize(
    ("manifest_case_id", "pattern", "seed"),
    [
        ("derived-ll-n1-w4", "balanced", 0),
        ("confirmatory-ll-n2-w32", "balanced", 0),
        ("confirmatory-ll-n2-w16", "unknown", 0),
        ("confirmatory-ll-n2-w16", "balanced", True),
        ("confirmatory-ll-n2-w16", "balanced", 1.0),
        ("confirmatory-ll-n2-w16", "balanced", -1),
        ("confirmatory-ll-n2-w16", "balanced", route.MAX_SEED + 1),
    ],
)
def test_generation_rejects_non_frozen_arguments(
    manifest_case_id: object,
    pattern: object,
    seed: object,
) -> None:
    with pytest.raises(route.CommonFairCompactRouteError):
        route.generate_route_artifact(
            manifest_case_id=manifest_case_id,  # type: ignore[arg-type]
            pattern=pattern,  # type: ignore[arg-type]
            seed=seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), True),
        (("version",), 2.0),
        (("shape", 0), True),
        (("shape", 1), 128.0),
        (("case", "nodes"), 2.0),
        (("case", "world_size"), True),
        (("case", "hidden"), 7168.0),
        (("algorithm", "seed"), False),
        (("algorithm", "seed"), 0.0),
        (("logical_tensors", "expert_ids", "count"), True),
        (("logical_tensors", "topk_weights", "count"), 16384.0),
    ],
)
def test_validator_rejects_bool_and_float_numeric_metadata(
    path: tuple[object, ...],
    value: object,
) -> None:
    artifact = _base()
    target: object = artifact
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    _rehash(artifact)
    with pytest.raises(route.CommonFairCompactRouteError):
        route.validate_route_artifact(artifact)


def test_validator_rejects_seed_tamper_even_with_new_self_hash() -> None:
    artifact = _base()
    artifact["algorithm"]["seed"] += 1
    _rehash(artifact)
    with pytest.raises(route.CommonFairCompactRouteError, match="streaming_sha256"):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize(
    ("tensor", "field", "value", "match"),
    [
        ("expert_ids", "count", 1, "count changed"),
        ("expert_ids", "element_encoding", "NATIVE_INT64", "encoding changed"),
        ("expert_ids", "streaming_sha256", "0" * 64, "mismatch"),
        ("topk_weights", "count", 1, "count changed"),
        ("topk_weights", "element_encoding", "FLOAT32", "encoding changed"),
        ("topk_weights", "streaming_sha256", "F" * 64, "invalid"),
    ],
)
def test_validator_rejects_tensor_commitment_tamper(
    tensor: str,
    field: str,
    value: object,
    match: str,
) -> None:
    artifact = _base()
    artifact["logical_tensors"][tensor][field] = value
    _rehash(artifact)
    with pytest.raises(route.CommonFairCompactRouteError, match=match):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_evidence_status", "PHYSICAL_COUNTERS_VERIFIED"),
        ("physical_topology_or_counter_evidence_present", True),
        ("route_artifact_claims_performance", True),
        ("claim_scope", "CONFIRMATORY_PERFORMANCE_RESULT"),
    ],
)
def test_validator_rejects_claim_promotion_with_matching_hash(
    field: str,
    value: object,
) -> None:
    artifact = route.generate_route_artifact(
        manifest_case_id="confirmatory-ll-n2-w16",
        pattern="rail_hot",
    )
    artifact["case"][field] = value
    _rehash(artifact)
    with pytest.raises(route.CommonFairCompactRouteError):
        route.validate_route_artifact(artifact)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("format",), "rail-balance-logical-route-v1"),
        (("layout",), "TOKEN_RANK_TOPK"),
        (("dtypes", "expert_ids"), "INT32"),
        (("dtypes", "topk_weights"), "FLOAT32_DECIMAL"),
        (("algorithm", "expert_id_formula"), "UNVERSIONED"),
        (("weight_slot_fp32_bits", 0), "0x7f800000"),
    ],
)
def test_validator_rejects_format_algorithm_dtype_and_weight_drift(
    path: tuple[object, ...],
    value: object,
) -> None:
    artifact = _base()
    target: object = artifact
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    _rehash(artifact)
    with pytest.raises(route.CommonFairCompactRouteError):
        route.validate_route_artifact(artifact)


def test_validator_rejects_unknown_field_and_hash_mismatch() -> None:
    unknown = _base()
    unknown["expanded_expert_ids"] = []
    _rehash(unknown)
    with pytest.raises(route.CommonFairCompactRouteError, match="fields differ"):
        route.validate_route_artifact(unknown)

    mismatch = _base()
    mismatch["canonical_sha256"] = "0" * 64
    with pytest.raises(route.CommonFairCompactRouteError, match="mismatch"):
        route.validate_route_artifact(mismatch)


def test_canonical_hash_excludes_only_self_hash() -> None:
    artifact = _base()
    changed_self = copy.deepcopy(artifact)
    changed_self["canonical_sha256"] = "f" * 64
    assert route.canonical_sha256(changed_self) == artifact["canonical_sha256"]
    changed_payload = copy.deepcopy(artifact)
    changed_payload["algorithm"]["seed"] += 1
    assert route.canonical_sha256(changed_payload) != artifact["canonical_sha256"]


def test_safe_atomic_write_and_load_round_trip(tmp_path: Path) -> None:
    artifact = _base()
    output = tmp_path / "compact.json"
    raw_sha, canonical_sha, size = route.write_route(output, artifact)
    loaded, loaded_raw_sha, loaded_canonical_sha = route.load_route(output)

    assert loaded == artifact
    assert loaded_raw_sha == raw_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert loaded_canonical_sha == canonical_sha == artifact["canonical_sha256"]
    assert size == output.stat().st_size < 2048
    assert output.read_bytes() == route.canonical_bytes(artifact) + b"\n"
    assert stat.S_IMODE(output.stat().st_mode) == 0o644
    assert output.stat().st_nlink == 1


def test_writer_publishes_complete_inode_and_never_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _base()
    payload = route.canonical_bytes(artifact) + b"\n"
    output = tmp_path / "atomic.json"
    real_link = os.link
    observed = False

    def observing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal observed
        assert destination == output.name
        assert not output.exists()
        descriptor = os.open(source, os.O_RDONLY, dir_fd=src_dir_fd)
        try:
            assert os.read(descriptor, route.MAX_ARTIFACT_BYTES) == payload
        finally:
            os.close(descriptor)
        observed = True
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(route.os, "link", observing_link)
    route.write_route(output, artifact)
    assert observed
    assert output.read_bytes() == payload

    monkeypatch.setattr(route.os, "link", real_link)
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_bytes(b"sentinel")
    with pytest.raises(route.CommonFairCompactRouteError, match="without replacement"):
        route.write_route(sentinel, artifact)
    assert sentinel.read_bytes() == b"sentinel"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_writer_cleans_temporary_file_when_atomic_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(route.os, "link", failing_link)
    output = tmp_path / "failed.json"
    with pytest.raises(route.CommonFairCompactRouteError, match="atomically publish"):
        route.write_route(output, _base())
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_loader_rejects_duplicate_keys_nonfinite_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"format":"a","format":"b"}\n', encoding="utf-8")
    duplicate.chmod(0o644)
    with pytest.raises(route.CommonFairCompactRouteError, match="duplicate JSON key"):
        route.load_route(duplicate)

    for index, constant in enumerate(("NaN", "Infinity", "-Infinity")):
        path = tmp_path / f"nonfinite-{index}.json"
        path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
        path.chmod(0o644)
        with pytest.raises(
            route.CommonFairCompactRouteError, match="non-finite JSON constant"
        ):
            route.load_route(path)

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(_base(), indent=2) + "\n", encoding="utf-8")
    pretty.chmod(0o644)
    with pytest.raises(route.CommonFairCompactRouteError, match="canonical UTF-8"):
        route.load_route(pretty)


def test_loader_rejects_bom_utf16_and_invalid_utf8(tmp_path: Path) -> None:
    canonical = route.canonical_bytes(_base()) + b"\n"
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + canonical)
    bom.chmod(0o644)
    utf16 = tmp_path / "utf16.json"
    utf16.write_bytes(canonical.decode("utf-8").encode("utf-16"))
    utf16.chmod(0o644)
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff\n")
    invalid.chmod(0o644)

    with pytest.raises(route.CommonFairCompactRouteError, match="BOM"):
        route.load_route(bom)
    with pytest.raises(route.CommonFairCompactRouteError, match="invalid route JSON"):
        route.load_route(utf16)
    with pytest.raises(route.CommonFairCompactRouteError, match="invalid route JSON"):
        route.load_route(invalid)


def test_loader_rejects_fifo_symlink_hardlink_and_symlink_parent(
    tmp_path: Path,
) -> None:
    fifo = tmp_path / "route.fifo"
    os.mkfifo(fifo, 0o644)
    with pytest.raises(route.CommonFairCompactRouteError, match="regular file"):
        route.load_route(fifo)

    target = tmp_path / "target.json"
    route.write_route(target, _base())
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(route.CommonFairCompactRouteError, match="safely open"):
        route.load_route(symlink)
    with pytest.raises(
        route.CommonFairCompactRouteError, match="exactly one hard link"
    ):
        route.load_route(hardlink)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    direct = real_parent / "route.json"
    route.write_route(direct, _base())
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(route.CommonFairCompactRouteError, match="safely traverse"):
        route.load_route(linked_parent / "route.json")


def test_loader_rejects_wrong_mode_writable_parent_and_oversized_file(
    tmp_path: Path,
) -> None:
    wrong_mode = tmp_path / "wrong-mode.json"
    wrong_mode.write_bytes(route.canonical_bytes(_base()) + b"\n")
    wrong_mode.chmod(0o600)
    with pytest.raises(route.CommonFairCompactRouteError, match="mode must be 0644"):
        route.load_route(wrong_mode)

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(route.MAX_ARTIFACT_BYTES + 1)
    oversized.chmod(0o644)
    with pytest.raises(route.CommonFairCompactRouteError, match=r"1\.\."):
        route.load_route(oversized)

    safe_file = tmp_path / "safe.json"
    route.write_route(safe_file, _base())
    tmp_path.chmod(0o770)
    try:
        with pytest.raises(route.CommonFairCompactRouteError, match="route parent"):
            route.load_route(safe_file)
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.parametrize(
    "changed_field",
    ["st_mode", "st_uid", "st_nlink", "st_ctime_ns"],
)
def test_loader_rejects_concurrent_security_metadata_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    output = tmp_path / "race.json"
    route.write_route(output, _base())
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
    with pytest.raises(route.CommonFairCompactRouteError, match="changed while"):
        route.load_route(output)


def test_cli_generate_verify_and_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cli.json"
    assert (
        route.main(
            [
                "generate",
                "--case",
                "confirmatory-ll-n2-w16",
                "--pattern",
                "rail_hot",
                "--seed",
                "5",
                "--output",
                os.fspath(output),
            ]
        )
        == 0
    )
    generated = json.loads(capsys.readouterr().out)
    assert generated["status"] == "GENERATED"
    assert generated["format"] == "rail-balance-logical-route-compact-v2"
    assert generated["manifest_case_id"] == "confirmatory-ll-n2-w16"
    assert generated["physical_evidence_status"] == (
        "REQUIRES_PHYSICAL_TOPOLOGY_AND_COUNTER_VERIFICATION"
    )

    assert route.main(["verify", os.fspath(output)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "VERIFIED"
    assert verified["canonical_sha256"] == generated["canonical_sha256"]

    bad = tmp_path / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    assert route.main(["verify", os.fspath(bad)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "COMMON_FAIR compact route error" in captured.err


def test_implementation_imports_only_python_standard_library() -> None:
    source_path = Path(route.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "contextlib",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "secrets",
        "stat",
        "struct",
        "sys",
        "typing",
    }
