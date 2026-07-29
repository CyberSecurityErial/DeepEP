import hashlib
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = Path(__file__).with_name("goldens") / "legacy_hybrid.json"
INCLUDE_ROOT = REPO_ROOT / "deep_ep" / "include"
MASK64 = (1 << 64) - 1
TMA_ALIGNMENT = 32
BUFFER_ALIGNMENT = 2 * 1024 * 1024


def _digest(data):
    if isinstance(data, str):
        data = data.encode()

    def fnv1a(seed):
        value = seed
        for byte in data:
            value ^= byte
            value = (value * 0x100000001B3) & MASK64
        return value

    def split_mix(value):
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    state_0 = split_mix(fnv1a(0xC6A4A7935BD1E995))
    state_1 = split_mix(fnv1a(0x9E3779B97F4A7C15))
    return f"{state_0:016x}{state_1:016x}"


def _deep_ep_includes(code):
    matches = re.findall(r'#\s*include\s*[<\"]([^>\"]+)[>\"]', code)
    return [item for item in matches if item.startswith("deep_ep")]


def _recursive_include_hash(root_include, cache=None):
    cache = {} if cache is None else cache

    def visit(relative_path):
        if relative_path in cache:
            return cache[relative_path]
        source = (INCLUDE_ROOT / relative_path).read_bytes()
        code = source.decode()
        payload = "".join(
            f"{visit(child)}$" for child in _deep_ep_includes(code)
        )
        payload += f"#{_digest(source)}"
        cache[relative_path] = _digest(payload)
        return cache[relative_path]

    # IncludeParser::get_hash_value(generated_code) hashes each root include
    # followed by '$'; generated Hybrid source contains exactly one such root.
    return _digest(f"{visit(root_include)}$")


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _token_bytes(hidden, element_size, num_topk, with_metadata,
                 num_sf_bytes=0):
    metadata_bytes = num_topk * (4 + 4)
    if with_metadata:
        metadata_bytes += (1 + num_topk) * 4
    return (
        _align(hidden * element_size, TMA_ALIGNMENT)
        + _align(num_sf_bytes, TMA_ALIGNMENT)
        + _align(metadata_bytes, TMA_ALIGNMENT)
    )


def _legacy_sizes(num_scaleout_ranks, num_scaleup_ranks, hidden, num_topk,
                  num_max_tokens_per_rank, num_max_channels,
                  is_scaleup_nvlink):
    dispatch_token_bytes = _token_bytes(
        hidden, element_size=2, num_topk=num_topk, with_metadata=True)
    combine_token_bytes = _token_bytes(
        hidden, element_size=2, num_topk=num_topk, with_metadata=False)
    num_ranks = num_scaleout_ranks * num_scaleup_ranks

    if num_scaleout_ranks == 1:
        dispatch_send_ranks = 0 if is_scaleup_nvlink else 1
        dispatch_bytes = dispatch_token_bytes * num_max_tokens_per_rank * (
            dispatch_send_ranks + num_ranks)

        combine_send_ranks = 0 if is_scaleup_nvlink else num_ranks
        combine_recv_ranks = min(num_ranks, num_topk)
        combine_bytes = combine_token_bytes * num_max_tokens_per_rank * (
            combine_send_ranks + combine_recv_ranks)
    else:
        dispatch_tokens = (
            num_scaleup_ranks * num_scaleout_ranks
            * num_max_tokens_per_rank
            + num_max_tokens_per_rank
            + num_scaleout_ranks
            * (num_max_tokens_per_rank + num_max_channels)
        )
        dispatch_bytes = dispatch_token_bytes * dispatch_tokens

        num_scaleup_layouts = min(num_scaleup_ranks, num_topk)
        num_scaleout_layouts = min(num_scaleout_ranks, num_topk)
        combine_tokens = (
            num_scaleup_layouts * num_scaleout_ranks
            * num_max_tokens_per_rank
            + num_scaleout_layouts * num_max_tokens_per_rank
            + num_scaleout_ranks
            * (num_max_tokens_per_rank + num_max_channels)
        )
        combine_bytes = combine_token_bytes * combine_tokens

    return {
        "dispatch_bytes": dispatch_bytes,
        "combine_bytes": combine_bytes,
        "aligned_max_bytes": _align(
            max(dispatch_bytes, combine_bytes), BUFFER_ALIGNMENT),
    }


def _workspace_bytes(num_max_channels):
    num_max_ranks = 1024
    num_max_experts = 2048
    num_max_inflight_agrs = 32
    value = 16
    value += (num_max_ranks + num_max_experts) * 8
    value += num_max_ranks * 8 * 2
    value += num_max_experts * 8 * 2
    value += num_max_ranks * 4
    value += num_max_ranks * 4 * 2
    value += num_max_experts * 4 * 2
    value += num_max_ranks * num_max_channels * 8
    value += num_max_ranks * num_max_channels * 4
    value += 2 * 2 * 8
    value += (num_max_inflight_agrs + 1) * num_max_ranks * 4
    return value


def _load_golden():
    return json.loads(GOLDEN_PATH.read_text())


def test_immutable_legacy_hybrid_files_match_sha256_golden():
    golden = _load_golden()
    for relative_path, expected in golden["immutable_sha256"].items():
        observed = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert observed == expected, (relative_path, observed, expected)


def test_recursive_include_hash_matches_deepep_algorithm():
    golden = _load_golden()
    for relative_path, expected in golden["recursive_include_hash"].items():
        observed = _recursive_include_hash(relative_path)
        assert observed == expected, (relative_path, observed, expected)


def test_legacy_layout_goldens_are_independently_reproduced():
    layout = _load_golden()["canonical_layout"]
    hidden = layout["hidden"]
    num_topk = layout["num_topk"]
    max_tokens = layout["num_max_tokens_per_rank"]
    num_max_channels = layout["buffer_num_max_channels"]

    assert _token_bytes(hidden, 2, num_topk, True) == \
        layout["bf16_dispatch_token_bytes"]
    assert _token_bytes(hidden, 2, num_topk, False) == \
        layout["combine_token_bytes"]

    cases = (
        ("num_scaleout_1_num_scaleup_8_nvlink", 1, 8, True),
        ("num_scaleout_2_num_scaleup_4", 2, 4, False),
        ("num_scaleout_4_num_scaleup_2", 4, 2, False),
    )
    for name, num_scaleout, num_scaleup, is_nvlink in cases:
        observed = _legacy_sizes(
            num_scaleout, num_scaleup, hidden, num_topk, max_tokens,
            num_max_channels, is_nvlink)
        assert observed == layout["cases"][name], (name, observed)

    workspace_raw = _workspace_bytes(layout["workspace_num_max_channels"])
    assert workspace_raw == layout["workspace_raw_bytes"]
    assert _align(workspace_raw, BUFFER_ALIGNMENT) == \
        layout["workspace_aligned_bytes"]


def test_transit_scratch_lifetime_and_rank_layout_contract():
    dispatch = (
        REPO_ROOT / "deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh"
    ).read_text()
    source_begin = dispatch.index("// Iterate all tokens")
    tag0_barrier = dispatch.index("comm::kHybridDispatchTag0")
    assert tag0_barrier < source_begin
    assert (
        "comm::kHybridDispatchTag0, false, false, true>" in dispatch
    )
    source_end = dispatch.index("\n    } else {\n", source_begin)
    source_region = dispatch[source_begin:source_end]
    assert "get_linked_list_idx_ptr" not in source_region

    forward_load = dispatch.index(
        "// TMA load into shared memory", source_end)
    linked_list_overwrite = dispatch.index(
        "tma_buffer.get_linked_list_idx_ptr()[lane_idx] =",
        forward_load)
    forward_metadata = dispatch.index(
        "// Record metadata at forward", linked_list_overwrite)
    assert forward_load < linked_list_overwrite < forward_metadata

    combine_utils = (
        REPO_ROOT / "deep_ep/include/deep_ep/impls/combine_utils.cuh"
    ).read_text()
    assert "return kNumRanks <= kNumTopk;" in combine_utils

    combine = (
        REPO_ROOT / "deep_ep/include/deep_ep/impls/hybrid_combine.cuh"
    ).read_text()
    assert (
        "kHidden % (32 * sizeof(int4) / sizeof(nv_bfloat16)) == 0"
        in combine
    )


def run_all():
    tests = [
        test_immutable_legacy_hybrid_files_match_sha256_golden,
        test_recursive_include_hash_matches_deepep_algorithm,
        test_legacy_layout_goldens_are_independently_reproduced,
        test_transit_scratch_lifetime_and_rank_layout_contract,
    ]
    for test in tests:
        test()
    print(f"PASS {len(tests)}/{len(tests)} legacy Hybrid identity goldens")


if __name__ == "__main__":
    run_all()
