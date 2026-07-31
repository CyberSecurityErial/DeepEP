"""CPU-only contracts for the formal source-round live executor."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import rail_balance_campaign_schema as campaign_schema
import rail_balance_source_round_executor as executor
import run_rail_balance_hop_campaign as runner


_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _ROOT / executor.CAMPAIGN_TEMPLATE_PATH


def _expect_executor_error(function: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    try:
        function(*args, **kwargs)
    except executor.ExecutorError as error:
        return str(error)
    raise AssertionError("unsafe executor input was accepted")


def _contract() -> dict[str, Any]:
    template, _, _ = campaign_schema.load_manifest(_TEMPLATE)
    return {
        "algorithm": {
            "hop_mode": "adaptive",
            "two_hop_threshold_percent": 0,
            "max_two_hop_percent": 25,
            "hop_penalty_percent": 50,
        },
        "shape": {
            "stage": "source",
            "case_name": "c100_volume_h256",
            "world_size": 8,
            "hidden": 256,
            "input_iteration": 100,
            "init_dist_seed": 100,
            "remainder_seed": 100,
            "tokens_per_rank": [1024] * 8,
            "num_topk": 4,
            "num_channels": 256,
            "num_experts": 72,
            "num_destinations": 9,
            "local_destination": 0,
            "proxy_capacity_per_egress": 896,
            "dtype": "bfloat16",
            "interference_mode": "none",
            "interference_compute_shape": [2048, 2048, 2048],
        },
        "timing": {
            "warmup_iterations": 10,
            "steady_iterations": 100,
            "clock": "time.perf_counter_ns common monotonic host clock",
            "stage_truth": "max(stage_end)-min(stage_start) across 8 ranks",
            "percentile_method": "linear interpolation over sorted samples",
            "std_method": "population",
            "logical_token_bytes": 576,
            "logical_bytes_per_iteration": 1,
            "logical_bandwidth_denominator": "global target-stage span",
        },
        "toolchain": {"cuda": "12.8", "arch": "sm90"},
        "gpu_mapping": copy.deepcopy(
            template["resources"]["expected_gpu_index_uuid_mapping"]
        ),
    }


def _source_manifest() -> dict[str, Any]:
    contract = _contract()
    contract_sha = executor.canonical_sha256(contract)
    parent = {
        "id": "parent",
        "worktree": str(_ROOT),
        "source_commit": "1" * 40,
        "frozen_execution_contract_sha256": contract_sha,
        "build_bundle_ref": "bundles/parent/build",
        "gate_bundle_ref": "bundles/parent/gates",
    }
    candidates = []
    for index in range(2):
        candidate_id = f"candidate-{index + 1}"
        candidates.append(
            {
                "id": candidate_id,
                "worktree": str(_ROOT),
                "source_commit": str(index + 2) * 40,
                "frozen_execution_contract_sha256": contract_sha,
                "build_bundle_ref": f"bundles/{candidate_id}/build",
                "gate_bundle_ref": f"bundles/{candidate_id}/gates",
                "declared_patch": {
                    "base_commit": "1" * 40,
                    "diff_sha256": hashlib.sha256(candidate_id.encode()).hexdigest(),
                    "changed_files": ["csrc/kernels/internode_ll.cu"],
                },
                "primary_change": f"one CUDA change {index}",
                "hypothesis": "reduce target kernel duration",
                "expected_profile_metrics": ["lower kernel duration"],
                "risks": ["register pressure"],
            }
        )
    return {
        "round_id": "round-a",
        "frozen_execution_contract": contract,
        "parent": parent,
        "candidates": candidates,
    }


def test_import_is_stdlib_only() -> None:
    code = (
        "import sys; import rail_balance_source_round_executor; "
        "assert 'torch' not in sys.modules; assert 'deep_ep' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parent)
    subprocess.run(
        (sys.executable, "-B", "-c", code),
        cwd=_ROOT,
        env=environment,
        check=True,
    )


def test_default_mode_is_check_only_and_never_calls_live() -> None:
    original_prepare = executor._prepare_inputs
    original_live = executor._run_live
    called = False

    def fake_prepare(*_args: Any) -> dict[str, Any]:
        return {
            "source_manifest": {"round_id": "round-a"},
            "plan": {"blocks": [1, 2, 3]},
        }

    def forbidden_live(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        raise AssertionError("default mode launched live execution")

    executor._prepare_inputs = fake_prepare
    executor._run_live = forbidden_live
    try:
        output = io.StringIO()
        with redirect_stdout(output):
            result = executor.main(
                [
                    "--source-manifest",
                    "/no/read/in/mock",
                    "--plan",
                    "/no/read/in/mock",
                    "--campaign-template",
                    "/no/read/in/mock",
                ]
            )
        assert result == 0
        assert output.getvalue().startswith("NOT_RUN")
        assert not called
    finally:
        executor._prepare_inputs = original_prepare
        executor._run_live = original_live


def test_live_contract_rejects_missing_and_wrong_noncritical_fields() -> None:
    manifest = _source_manifest()
    executor._validate_live_contract(manifest)

    missing = copy.deepcopy(manifest)
    del missing["frozen_execution_contract"]["algorithm"][
        "two_hop_threshold_percent"
    ]
    assert "algorithm contract" in _expect_executor_error(
        executor._validate_live_contract, missing
    )

    bad_hidden = copy.deepcopy(manifest)
    bad_hidden["frozen_execution_contract"]["shape"]["hidden"] = "256"
    assert "hidden" in _expect_executor_error(
        executor._validate_live_contract, bad_hidden
    )

    bad_dtype = copy.deepcopy(manifest)
    bad_dtype["frozen_execution_contract"]["shape"]["dtype"] = "float16"
    assert "dtype" in _expect_executor_error(executor._validate_live_contract, bad_dtype)

    bad_timer = copy.deepcopy(manifest)
    bad_timer["frozen_execution_contract"]["timing"]["logical_token_bytes"] = 0
    assert "byte counts" in _expect_executor_error(
        executor._validate_live_contract, bad_timer
    )

    dynamic_toolchain = copy.deepcopy(manifest)
    dynamic_toolchain["frozen_execution_contract"]["toolchain"][
        "created_utc"
    ] = "2026-08-01T00:00:00+00:00"
    assert "normalized" in _expect_executor_error(
        executor._validate_live_contract, dynamic_toolchain
    )


def test_block_manifest_selects_only_fixed_stage_and_vnode_gates() -> None:
    template, _, _ = campaign_schema.load_manifest(_TEMPLATE)
    manifest = _source_manifest()
    block = {
        "ordinal": 0,
        "role": "parent",
        "variant_id": "parent",
        "repetition": 0,
        "run_id": "round-a-00-parent-0",
    }
    generated = executor._block_manifest(template, manifest, block)
    benchmarks = [row for row in generated["stages"] if row["kind"] == "benchmark"]
    enabled = [row for row in benchmarks if row.get("enabled", True)]
    assert [row["id"] for row in enabled] == [executor.FIXED_BENCHMARK_STAGE]
    assert enabled[0]["depends_on"] == list(executor.FIXED_VNODE_GATES)
    assert enabled[0]["repeat"] == 1
    assert enabled[0]["artifacts"] == ["report.json"]
    assert enabled[0]["env"]["EP_JIT_CACHE_DIR"] == "{stage_dir}/jit"
    interference_index = enabled[0]["argv"].index("--interference-mode")
    assert enabled[0]["argv"][interference_index + 1] == "none"
    assert all(
        not row.get("enabled", True)
        for row in generated["stages"]
        if str(row["kind"]).startswith("profile_")
    )


def test_round_disk_formula_reserves_every_remaining_block() -> None:
    original = executor._home_usage_bytes
    executor._home_usage_bytes = lambda: 270_000_000_000
    try:
        result = executor._round_disk_preflight(12, 300_000_000_000)
    finally:
        executor._home_usage_bytes = original
    assert result["all_blocks_artifact_reserve_bytes"] == 9_000_000_000
    assert result["projected_home_bytes"] == 284_020_000_000
    assert result["projected_home_bytes"] <= result["home_hard_limit_bytes"]


def test_toolchain_identity_ignores_only_capture_timestamp() -> None:
    first = {
        "created_utc": "2026-08-01T00:00:00+00:00",
        "target_platform": {"cuda": "12.8"},
        "tools": [{"name": "nvcc", "sha256": "a" * 64}],
        "valid": True,
    }
    second = copy.deepcopy(first)
    second["created_utc"] = "2026-08-01T00:01:00+00:00"
    changed = copy.deepcopy(second)
    changed["tools"][0]["sha256"] = "b" * 64
    assert executor.canonical_sha256(
        executor._normalized_toolchain(first)
    ) == executor.canonical_sha256(executor._normalized_toolchain(second))
    assert executor.canonical_sha256(
        executor._normalized_toolchain(first)
    ) != executor.canonical_sha256(executor._normalized_toolchain(changed))


def test_atomic_nofollow_no_clobber_and_hardlink_tamper() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        output = root / "record.json"
        executor._atomic_json_no_clobber(output, {"value": 1})
        assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
        _expect_executor_error(
            executor._atomic_json_no_clobber, output, {"value": 2}
        )
        assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

        linked = root / "linked.json"
        os.link(output, linked)
        _expect_executor_error(executor._strict_json, linked)

        real = root / "real"
        real.mkdir(mode=0o700)
        symlink = root / "alias"
        symlink.symlink_to(real, target_is_directory=True)
        _expect_executor_error(
            executor._atomic_json_no_clobber, symlink / "escape.json", {"x": 1}
        )
        assert not (real / "escape.json").exists()


def _lease_fixture(root: Path) -> tuple[int, Path, dict[str, Any], dict[str, Any]]:
    artifact_root = root / "artifacts"
    artifact_root.mkdir(mode=0o700)
    (artifact_root / "runs").mkdir(mode=0o700)
    manifest_path = root / "campaign.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    lock_path = root / "round.lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    os.fchmod(descriptor, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_stat = os.fstat(descriptor)
    root_stat = artifact_root.stat()
    nonce = "a" * 64
    executor_sha = executor._sha256_file(Path(executor.__file__).resolve())
    record = {
        "schema_version": 1,
        "protocol": executor.LOCK_PROTOCOL,
        "round_id": "round-a",
        "plan_raw_sha256": "b" * 64,
        "executor_pid": os.getpid(),
        "executor_code_sha256": executor_sha,
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "boot_id": executor._boot_id(),
        "lock_path": str(lock_path),
        "lock_device": lock_stat.st_dev,
        "lock_inode": lock_stat.st_ino,
        "shared_offset": 1_234_567,
        "artifact_root": str(artifact_root),
        "artifact_root_device": root_stat.st_dev,
        "artifact_root_inode": root_stat.st_ino,
        "output_dir": str(artifact_root / "runs" / "run-a"),
        "run_id": "run-a",
        "campaign_manifest_path": str(manifest_path),
        "campaign_manifest_raw_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "round_lock_identity_sha256": "",
    }
    record["round_lock_identity_sha256"] = executor._round_lock_identity(record)
    payload = executor._canonical_bytes(record) + b"\n"
    os.ftruncate(descriptor, 0)
    assert os.pwrite(descriptor, payload, 0) == len(payload)
    os.fsync(descriptor)
    os.lseek(descriptor, record["shared_offset"], os.SEEK_SET)
    lease = {
        "fd": descriptor,
        "round_id": "round-a",
        "plan_raw_sha256": "b" * 64,
        "nonce": nonce,
        "parent_pid": os.getpid(),
        "identity_sha256": executor.canonical_sha256(record),
    }
    return descriptor, lock_path, record, lease


def _fork_lease_validation(
    lock_path: Path, lease: dict[str, Any], *, become_session_leader: bool
) -> str:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            if become_session_leader:
                os.setsid()
            with runner._campaign_lock(lock_path, lease):
                assert os.get_inheritable(lease["fd"]) is False
            message = b"accepted"
        except BaseException:
            message = b"rejected"
        with os.fdopen(write_fd, "wb", closefd=True) as stream:
            stream.write(message)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb", closefd=True) as stream:
        result = stream.read().decode("ascii")
    waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0
    return result


def test_inherited_lease_is_auditable_noninheritable_and_parent_retained() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        descriptor, lock_path, _record, lease = _lease_fixture(root)
        try:
            assert _fork_lease_validation(
                lock_path, lease, become_session_leader=True
            ) == "accepted"
            probe = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError("child close released the parent's flock")
            finally:
                os.close(probe)
            assert _fork_lease_validation(
                lock_path, lease, become_session_leader=False
            ) == "rejected"
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def test_lease_rejects_wrong_output_and_replaced_root() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        os.chmod(root, 0o700)
        descriptor, lock_path, record, lease = _lease_fixture(root)
        try:
            attacked = copy.deepcopy(record)
            attacked["output_dir"] = "/home/chen"
            attacked["round_lock_identity_sha256"] = executor._round_lock_identity(
                attacked
            )
            payload = executor._canonical_bytes(attacked) + b"\n"
            os.ftruncate(descriptor, 0)
            os.pwrite(descriptor, payload, 0)
            os.fsync(descriptor)
            os.lseek(descriptor, attacked["shared_offset"], os.SEEK_SET)
            bad_lease = dict(lease)
            bad_lease["identity_sha256"] = executor.canonical_sha256(attacked)
            assert _fork_lease_validation(
                lock_path, bad_lease, become_session_leader=True
            ) == "rejected"

            artifact_root = Path(record["artifact_root"])
            moved = root / "artifacts-old"
            artifact_root.rename(moved)
            artifact_root.mkdir(mode=0o700)
            (artifact_root / "runs").mkdir(mode=0o700)
            # Restore the original record: its bound root dev/inode is now stale.
            payload = executor._canonical_bytes(record) + b"\n"
            os.ftruncate(descriptor, 0)
            os.pwrite(descriptor, payload, 0)
            os.fsync(descriptor)
            os.lseek(descriptor, record["shared_offset"], os.SEEK_SET)
            lease["identity_sha256"] = executor.canonical_sha256(record)
            assert _fork_lease_validation(
                lock_path, lease, become_session_leader=True
            ) == "rejected"
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def test_lock_probe_has_no_fd_leak() -> None:
    before = len(os.listdir("/proc/self/fd"))
    with executor._whole_round_lock(Path(executor.LOCK_PATH)) as (descriptor, metadata):
        for _ in range(32):
            executor._assert_lock_retained(descriptor, metadata)
        during = len(os.listdir("/proc/self/fd"))
        assert during <= before + 1
    assert len(os.listdir("/proc/self/fd")) <= before


def test_executor_reaps_runner_group_after_leader_exits() -> None:
    code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "sys.exit(0)"
    )
    process = subprocess.Popen(
        (sys.executable, "-B", "-c", code), start_new_session=True
    )
    process.wait(timeout=5)
    assert not executor._wait_process_group_empty(process.pid, timeout_seconds=0.1)
    executor._terminate_owned_runner(process)
    assert executor._wait_process_group_empty(process.pid, timeout_seconds=1)


def test_launch_runner_cleans_leak_and_still_rejects_success_code() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        worktree = root / "worktree"
        runner_path = worktree / executor.RUNNER_PATH
        runner_path.parent.mkdir(parents=True)
        pid_path = worktree / "runner-pid"
        runner_path.write_text(
            "import os,subprocess,sys\n"
            f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n",
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        lock = root / "lock"
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            message = _expect_executor_error(
                executor._launch_runner,
                descriptor=descriptor,
                worktree=worktree,
                manifest_path=manifest,
                plan_block={"run_id": "run-a", "ordinal": 0},
                campaign_root=root / "campaign",
                round_id="round-a",
                plan_sha="a" * 64,
                nonce="b" * 64,
                lease_identity="c" * 64,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                timeout_seconds=5,
                expected_runner_sha256=executor._sha256_file(runner_path),
            )
            assert "leaked descendants" in message
            process_group = int(pid_path.read_text(encoding="utf-8"))
            assert executor._wait_process_group_empty(process_group, timeout_seconds=1)
        finally:
            os.close(descriptor)


def test_runner_proc_fd_resolves_frozen_sibling_schema() -> None:
    runner_path = Path(runner.__file__).resolve()
    descriptor = os.open(
        runner_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        code = (
            "import hashlib,runpy; "
            f"d=runpy.run_path('/proc/self/fd/{descriptor}', run_name='frozen_runner'); "
            "assert d['_RUNNER_PATH'].is_file(); assert d['_SCHEMA_PATH'].is_file(); "
            "assert d['_sha256'](d['_RUNNER_PATH']) == "
            f"{executor._sha256_file(runner_path)!r}"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(runner_path.parent)
        subprocess.run(
            (sys.executable, "-B", "-c", code),
            cwd=_ROOT,
            env=environment,
            pass_fds=(descriptor,),
            check=True,
        )
    finally:
        os.close(descriptor)


def test_inherited_attempt_group_and_repeat_stop_are_immediate() -> None:
    helper = r'''
import json, os, signal, sys, tempfile, time
from pathlib import Path
import run_rail_balance_hop_campaign as runner
runner._home_usage = lambda _path: {"bytes": 0}
stage = {
    "id": "cpu-nested", "kind": "correctness", "resource": "cpu",
    "argv": [sys.executable, "-c", "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"],
    "env": {}, "timeout_seconds": 5, "depends_on": []
}
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    inherited = runner._run_attempt(
        stage=stage, attempt=1, output_dir=root,
        artifact_budget_bytes=100_000_000, home_path=Path(directory),
        home_hard_limit_bytes=300_000_000_000,
        monitor_exclusive_gpus=False, required_gpu_count=8,
        expected_gpu_mapping=[], inherited_round_lease=True,
    )
    members = [pid for pid in runner._process_group_members(os.getpgrp()) if pid != os.getpid()]
    for pid in members:
        try: os.kill(pid, signal.SIGKILL)
        except ProcessLookupError: pass
    deadline = time.monotonic() + 3
    while any(pid != os.getpid() for pid in runner._process_group_members(os.getpgrp())) and time.monotonic() < deadline:
        time.sleep(0.05)
    stage["id"] = "cpu-normal"
    normal = runner._run_attempt(
        stage=stage, attempt=1, output_dir=root,
        artifact_budget_bytes=100_000_000, home_path=Path(directory),
        home_hard_limit_bytes=300_000_000_000,
        monitor_exclusive_gpus=False, required_gpu_count=8,
        expected_gpu_mapping=[], inherited_round_lease=False,
    )
    print(json.dumps({"inherited": inherited, "normal": normal}))
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(runner.__file__).parent)
    result = subprocess.run(
        (sys.executable, "-B", "-c", helper),
        cwd=_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        timeout=20,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["inherited"]["owned_process_group_leaked"] is True
    assert observed["inherited"]["requires_executor_cleanup"] is True
    assert observed["normal"]["owned_process_group_leaked"] is True
    assert observed["normal"]["requires_executor_cleanup"] is False

    original_attempt = runner._run_attempt
    calls = 0

    def fake_attempt(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"passed": False, "requires_executor_cleanup": True}

    runner._run_attempt = fake_attempt
    try:
        manifest = {
            "resources": {
                "artifact_budget_bytes": 1,
                "home_path": "/tmp",
                "home_hard_limit_bytes": 300_000_000_000,
                "required_gpu_count": 8,
                "expected_gpu_index_uuid_mapping": [],
            },
            "stages": [
                {
                    "id": "cpu-repeat",
                    "kind": "correctness",
                    "resource": "cpu",
                    "depends_on": [],
                    "repeat": 2,
                }
            ],
        }
        try:
            runner._run_stages(
                manifest=manifest,
                output_dir=Path("/tmp/not-used-by-fake-attempt"),
                gpu_ready=True,
                no_execute=False,
                expected_source_identity_sha256="a" * 64,
                inherited_round_lease=True,
            )
        except RuntimeError as error:
            assert "executor cleanup" in str(error)
        else:
            raise AssertionError("repeat continued after inherited cleanup request")
        assert calls == 1
    finally:
        runner._run_attempt = original_attempt


def test_runner_rejects_hidden_lease_material_without_capability() -> None:
    name = "DEEP_EP_SOURCE_ROUND_LEASE_FD"
    original = os.environ.get(name)
    os.environ[name] = "9"
    try:
        try:
            runner._parse_args(["--manifest", "/tmp/x", "--run-id", "run-a"])
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("hidden lease environment was accepted")
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def test_live_loop_keeps_global_order_after_terminal_block_failures() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_root = root / "artifacts"
        artifact_root.mkdir(mode=0o700)
        worktrees = []
        for name in ("parent", "candidate-1", "candidate-2"):
            path = root / name
            path.mkdir(mode=0o700)
            worktrees.append(path)
        variants = ["parent", "candidate-1", "candidate-2"]
        schedule = [
            ("parent", "parent", 0),
            ("candidate", "candidate-1", 0),
            ("candidate", "candidate-2", 0),
            ("candidate", "candidate-1", 1),
            ("candidate", "candidate-2", 1),
            ("parent", "parent", 1),
            ("candidate", "candidate-1", 2),
            ("candidate", "candidate-2", 2),
            ("parent", "parent", 2),
            ("parent", "parent", 3),
            ("candidate", "candidate-1", 3),
            ("candidate", "candidate-2", 3),
        ]
        blocks = []
        worktree_by_variant = dict(zip(variants, worktrees))
        for ordinal, (role, variant, repetition) in enumerate(schedule):
            run_id = f"round-a-{ordinal:02d}-{variant}-{repetition}"
            blocks.append(
                {
                    "ordinal": ordinal,
                    "role": role,
                    "variant_id": variant,
                    "repetition": repetition,
                    "run_id": run_id,
                    "worktree": str(worktree_by_variant[variant]),
                    "source_commit": str(variants.index(variant) + 1) * 40,
                    "artifact_dir": f"runs/{run_id}",
                    "raw_benchmark_artifact": (
                        f"runs/{run_id}/stages/bench-abba-a1/attempt-01/report.json"
                    ),
                    "jit_dir": f"runs/{run_id}/stages/bench-abba-a1/attempt-01/jit",
                }
            )
        contract = _contract()
        harness = {
            executor.RUNNER_PATH: "1" * 64,
            executor.SCHEMA_PATH: "2" * 64,
            executor.EVALUATOR_PATH: "3" * 64,
            executor.BENCHMARK_PATH: "4" * 64,
        }
        source = {
            "round_id": "round-a",
            "coordinator": {"control_worktree": str(_ROOT), "control_commit": "1" * 40},
            "parent": {"id": "parent", "source_commit": "1" * 40},
            "candidates": [
                {
                    "id": name,
                    "primary_change": "one CUDA change",
                    "hypothesis": "faster",
                    "expected_profile_metrics": ["duration"],
                    "risks": ["registers"],
                }
                for name in variants[1:]
            ],
            "frozen_execution_contract": contract,
            "source_policy": {"harness_sha256": harness},
        }
        prepared = {
            "source_manifest": source,
            "plan": {"blocks": blocks, "preflight": {"status": "passed"}},
            "artifact_root": artifact_root,
            "artifact_root_stat": artifact_root.stat(),
            "source_raw_sha256": "5" * 64,
            "source_canonical_sha256": "6" * 64,
            "plan_raw_sha256": "7" * 64,
            "plan_canonical_sha256": "8" * 64,
            "executor_sha256": executor._sha256_file(Path(executor.__file__).resolve()),
            "coordinator_sha256": "9" * 64,
            "runner_sha256": "1" * 64,
            "schema_sha256": "2" * 64,
            "template": {},
            "source_manifest_path": root / "source.json",
            "plan_path": root / "plan.json",
        }
        observed: list[int] = []
        originals = {
            "preflight": executor.source_coordinator.preflight,
            "home": executor._home_usage_bytes,
            "block_manifest": executor._block_manifest,
            "evidence": executor._validate_campaign_evidence,
            "coordinator": executor._write_coordinator_record,
            "round_manifest": executor._build_round_manifest,
            "evaluate": executor.round_evaluator.evaluate_round,
            "validate_evaluation": executor._validate_embedded_evaluation,
        }

        def fake_launcher(**kwargs: Any) -> int:
            observed.append(kwargs["plan_block"]["ordinal"])
            return 3

        def fake_evidence(**kwargs: Any) -> dict[str, Any]:
            block = kwargs["plan_block"]
            return {
                "complete": False,
                "source_identity": {
                    "working_tree_identity_sha256": hashlib.sha256(
                        block["variant_id"].encode()
                    ).hexdigest()
                },
            }

        def fake_coordinator(**kwargs: Any) -> dict[str, Any]:
            block = kwargs["plan_block"]
            return {
                "block": {
                    "global_ordinal": block["ordinal"],
                    "role": block["role"],
                }
            }

        try:
            executor.source_coordinator.preflight = lambda *_args, **_kwargs: {
                "status": "passed"
            }
            executor._home_usage_bytes = lambda: 100
            executor._block_manifest = lambda *_args, **_kwargs: {"mock": True}
            executor._validate_campaign_evidence = fake_evidence
            executor._write_coordinator_record = fake_coordinator
            executor._build_round_manifest = lambda **_kwargs: {"mock": True}
            executor.round_evaluator.evaluate_round = lambda _path: {"status": "mock"}
            executor._validate_embedded_evaluation = lambda *_args, **_kwargs: None
            result = executor._run_live(
                prepared,
                timeout_seconds=1,
                gpu_preflight=lambda _mapping: {"idle": True, "reasons": []},
                launcher=fake_launcher,
            )
        finally:
            executor.source_coordinator.preflight = originals["preflight"]
            executor._home_usage_bytes = originals["home"]
            executor._block_manifest = originals["block_manifest"]
            executor._validate_campaign_evidence = originals["evidence"]
            executor._write_coordinator_record = originals["coordinator"]
            executor._build_round_manifest = originals["round_manifest"]
            executor.round_evaluator.evaluate_round = originals["evaluate"]
            executor._validate_embedded_evaluation = originals[
                "validate_evaluation"
            ]
        assert result == 0
        assert observed == list(range(12))
        terminal = json.loads(
            (artifact_root / executor.ROUND_TERMINAL_NAME).read_text(encoding="utf-8")
        )
        assert terminal["completed_block_count"] == 12


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} source-round executor CPU contracts")
