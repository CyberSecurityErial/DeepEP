"""CPU-only diagnostic for post-Gloo rank release skew.

This is not a CUDA or operator benchmark.  It uses the same eight spawned
rank shape as the C100 checked-adapter benchmark, records the common-host
timestamps immediately before and after each Gloo monitored barrier, and
retains every per-rank interval in an atomic JSON report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist


_WORLD_SIZE = 8
_CHILD_ENV = "EP_RAIL_BALANCE_GLOO_GATE_CHILD"
_LAUNCH_ENV = "EP_RAIL_BALANCE_GLOO_GATE_LAUNCH"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C100 CPU-only eight-rank Gloo gate diagnostic")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--steady-iters", type=int, default=100)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--master-port", type=int, default=30121)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--watchdog-seconds", type=int, default=180)
    parser.add_argument("--worker-suite", action="store_true",
                        help=argparse.SUPPRESS)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("diagnostic contract requires Python assertions")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters must be non-negative")
    if args.steady_iters <= 0:
        parser.error("--steady-iters must be positive")
    if not 1 <= args.master_port <= 65535:
        parser.error("--master-port must be in [1, 65535]")
    if args.timeout <= 0 or args.watchdog_seconds <= args.timeout:
        parser.error("watchdog must be longer than the Gloo timeout")
    return args


def _run_text(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_identity() -> dict[str, str]:
    source = Path(__file__).resolve()
    return {
        "git_commit": _run_text(("git", "rev-parse", "HEAD")),
        "git_status": _run_text(("git", "status", "--porcelain")),
        "source": str(source.relative_to(_REPOSITORY_ROOT)),
        "source_sha256": _sha256(source),
    }


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _cgroup_snapshot() -> dict[str, object]:
    v1_cpu = Path("/sys/fs/cgroup/cpu,cpuacct")
    v1_cpuset = Path("/sys/fs/cgroup/cpuset")
    v2 = Path("/sys/fs/cgroup")
    result: dict[str, object] = {
        "proc_self_cgroup": _read_optional(Path("/proc/self/cgroup")),
    }
    files = {
        "cpu_stat": v1_cpu / "cpu.stat",
        "cpu_quota_us": v1_cpu / "cpu.cfs_quota_us",
        "cpu_period_us": v1_cpu / "cpu.cfs_period_us",
        "cpuset_cpus": v1_cpuset / "cpuset.cpus",
        "cpuset_mems": v1_cpuset / "cpuset.mems",
    }
    if not (v1_cpu / "cpu.stat").exists():
        files = {
            "cpu_stat": v2 / "cpu.stat",
            "cpu_max": v2 / "cpu.max",
            "cpuset_cpus": v2 / "cpuset.cpus.effective",
            "cpuset_mems": v2 / "cpuset.mems.effective",
        }
    for key, path in files.items():
        value = _read_optional(path)
        if value is not None:
            result[key] = value
    return result


def _selected_proc_fields(path: Path, names: set[str]) -> dict[str, str]:
    text = _read_optional(path)
    if text is None:
        return {}
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in names:
            result[key.strip()] = value.strip()
    return result


def _thread_snapshot(rank: int) -> dict[str, object]:
    native_id = threading.get_native_id()
    return {
        "rank": rank,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "native_thread_id": native_id,
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "proc_task_status": _selected_proc_fields(
            Path(f"/proc/self/task/{native_id}/status"),
            {
                "Cpus_allowed_list",
                "Mems_allowed_list",
                "Threads",
                "voluntary_ctxt_switches",
                "nonvoluntary_ctxt_switches",
            },
        ),
        "proc_task_sched": _selected_proc_fields(
            Path(f"/proc/self/task/{native_id}/sched"),
            {
                "nr_involuntary_switches",
                "nr_migrations",
                "nr_switches",
                "nr_voluntary_switches",
                "se.nr_migrations",
                "se.statistics.wait_sum",
            },
        ),
        "proc_task_schedstat": _read_optional(
            Path(f"/proc/self/task/{native_id}/schedstat")),
    }


def _gather_objects(
    value: object,
    group: dist.ProcessGroup,
) -> list[object]:
    output: list[object] = [None] * _WORLD_SIZE
    dist.all_gather_object(output, value, group=group)
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower)


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    assert values
    floating = [float(value) for value in values]
    mean = statistics.fmean(floating)
    std = statistics.pstdev(floating)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(floating),
        "p95": _percentile(floating, 0.95),
        "p99": _percentile(floating, 0.99),
        "max": max(values),
        "mean": mean,
        "std_population": std,
        "cv_population": 0.0 if mean == 0 else std / mean,
    }


def _atomic_json(path: Path, report: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _aggregate_iteration(
    category: str,
    category_index: int,
    rows: Sequence[object],
) -> dict[str, object]:
    typed = []
    for expected_rank, raw in enumerate(rows):
        assert isinstance(raw, dict)
        assert raw["rank"] == expected_rank
        typed.append(raw)
    entries = [int(row["entry_ns"]) for row in typed]
    exits = [int(row["exit_ns"]) for row in typed]
    durations = [int(row["duration_ns"]) for row in typed]
    assert all(
        duration == exit_ns - entry_ns >= 0
        for duration, entry_ns, exit_ns in zip(durations, entries, exits)
    )
    last_arrival_to_first_exit = min(exits) - max(entries)
    assert last_arrival_to_first_exit >= 0
    return {
        "category": category,
        "category_index": category_index,
        "rank_raw": typed,
        "entry_span_ns": max(entries) - min(entries),
        "exit_span_ns": max(exits) - min(exits),
        "barrier_envelope_ns": max(exits) - min(entries),
        "last_arrival_to_first_exit_ns": last_arrival_to_first_exit,
        "rank_duration_min_ns": min(durations),
        "rank_duration_max_ns": max(durations),
        "entry_order": sorted(range(_WORLD_SIZE), key=entries.__getitem__),
        "exit_order": sorted(range(_WORLD_SIZE), key=exits.__getitem__),
    }


def _worker(rank: int, args: argparse.Namespace) -> None:
    assert not torch.cuda.is_initialized()
    timeout = timedelta(seconds=args.timeout)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{args.master_port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timeout,
    )
    control_group = dist.new_group(
        ranks=list(range(_WORLD_SIZE)),
        backend="gloo",
        timeout=timeout,
    )
    records = []
    pre_identity = _code_identity() if rank == 0 else None
    pre_threads = _gather_objects(_thread_snapshot(rank), control_group)
    pre_cgroup = _cgroup_snapshot() if rank == 0 else None
    total = 1 + args.warmup_iters + args.steady_iters
    try:
        for ordinal in range(total):
            if ordinal == 0:
                category, category_index = "cold", 0
            elif ordinal <= args.warmup_iters:
                category, category_index = "warm", ordinal - 1
            else:
                category, category_index = (
                    "steady", ordinal - 1 - args.warmup_iters)

            entry_ns = time.perf_counter_ns()
            dist.monitored_barrier(
                group=control_group,
                timeout=timeout,
                wait_all_ranks=True,
            )
            exit_ns = time.perf_counter_ns()
            rows = _gather_objects({
                "rank": rank,
                "entry_ns": entry_ns,
                "exit_ns": exit_ns,
                "duration_ns": exit_ns - entry_ns,
            }, control_group)
            if rank == 0:
                records.append(_aggregate_iteration(
                    category, category_index, rows))

        post_threads = _gather_objects(_thread_snapshot(rank), control_group)
        post_cgroup = _cgroup_snapshot() if rank == 0 else None
        post_identity = _code_identity() if rank == 0 else None
        assert all(
            isinstance(row, dict) and not row["cuda_initialized"]
            for row in (*pre_threads, *post_threads)
        )
        report = None
        build_error = None
        if rank == 0:
            try:
                assert pre_identity is not None
                assert post_identity == pre_identity
                steady = [row for row in records
                          if row["category"] == "steady"]
                launch = json.loads(os.environ[_LAUNCH_ENV])
                semantic_config = {
                    "world_size": _WORLD_SIZE,
                    "warmup_iterations": args.warmup_iters,
                    "steady_iterations": args.steady_iters,
                    "timeout_seconds": args.timeout,
                    "master_port": args.master_port,
                    "CUDA_VISIBLE_DEVICES": os.environ.get(
                        "CUDA_VISIBLE_DEVICES"),
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                }
                config_bytes = json.dumps(
                    semantic_config,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                created_utc = datetime.now(timezone.utc).isoformat()
                assert len(records) == total
                assert records[0]["category"] == "cold"
                assert len([row for row in records
                            if row["category"] == "warm"]) == \
                    args.warmup_iters
                assert len(steady) == args.steady_iters
                report = {
                    "schema_version": 1,
                    "run_id": f"gloo-gate-{created_utc}-p{os.getpid()}",
                    "claim_scope": (
                        "CPU-only bare Gloo monitored-barrier return plus "
                        "host wakeup diagnostic; not a CUDA, NCCL, "
                        "ElasticBuffer, operator, transport, or end-to-end "
                        "benchmark"),
                    "created_utc": created_utc,
                    "command": launch,
                    "identity": {
                        "pre": pre_identity,
                        "post": post_identity,
                        "stable_during_run": True,
                        "semantic_config": semantic_config,
                        "semantic_config_sha256": hashlib.sha256(
                            config_bytes).hexdigest(),
                    },
                    "environment": {
                        "hostname": platform.node(),
                        "platform": platform.platform(),
                        "python": sys.version,
                        "torch": torch.__version__,
                        "CUDA_VISIBLE_DEVICES": os.environ.get(
                            "CUDA_VISIBLE_DEVICES"),
                        "OMP_NUM_THREADS": os.environ.get(
                            "OMP_NUM_THREADS"),
                        "MKL_NUM_THREADS": os.environ.get(
                            "MKL_NUM_THREADS"),
                        "pre_threads": pre_threads,
                        "post_threads": post_threads,
                        "pre_cgroup": pre_cgroup,
                        "post_cgroup": post_cgroup,
                    },
                    "measurement": {
                        "world_size": _WORLD_SIZE,
                        "clock": (
                            "time.perf_counter_ns common monotonic host "
                            "clock; all ranks are processes on one host"),
                        "cold_iterations": 1,
                        "warmup_iterations": args.warmup_iters,
                        "steady_iterations": args.steady_iters,
                        "truth": (
                            "max barrier-return timestamp minus min "
                            "barrier-return timestamp across eight ranks"),
                        "percentile_method": "linear interpolation",
                        "std_method": "population",
                        "cuda_initialized": False,
                        "scope_limit": (
                            "bare Gloo lacks the NCCL/CUDA/ElasticBuffer "
                            "helper threads present in the C100 adapter"),
                    },
                    "cold": records[0],
                    "warm": [row for row in records
                             if row["category"] == "warm"],
                    "steady": steady,
                    "steady_summary": {
                        key: _summary([int(row[key]) for row in steady])
                        for key in (
                            "entry_span_ns",
                            "exit_span_ns",
                            "barrier_envelope_ns",
                            "last_arrival_to_first_exit_ns",
                            "rank_duration_max_ns",
                        )
                    },
                }
            except BaseException:
                build_error = traceback.format_exc()
        build_errors = _gather_objects(build_error, control_group)
        messages = [
            f"rank {index}:\n{error}"
            for index, error in enumerate(build_errors)
            if error is not None
        ]
        if messages:
            raise AssertionError(
                "Gloo diagnostic report build failed:\n" +
                "\n".join(messages))

        dist.monitored_barrier(
            group=control_group,
            timeout=timeout,
            wait_all_ranks=True,
        )
        write_error = None
        if rank == 0:
            try:
                assert report is not None
                _atomic_json(args.json_out, report)
            except BaseException:
                write_error = traceback.format_exc()
        write_errors = _gather_objects(write_error, control_group)
        messages = [
            f"rank {index}:\n{error}"
            for index, error in enumerate(write_errors)
            if error is not None
        ]
        if messages:
            raise AssertionError(
                "Gloo diagnostic report write failed:\n" +
                "\n".join(messages))
        dist.monitored_barrier(
            group=control_group,
            timeout=timeout,
            wait_all_ranks=True,
        )
        if rank == 0:
            assert report is not None
            summary = report["steady_summary"]["exit_span_ns"]
            print(
                "GLOO_GATE_PASS "
                f"median_us={summary['median'] / 1000:.3f} "
                f"p95_us={summary['p95'] / 1000:.3f} "
                f"max_us={summary['max'] / 1000:.3f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


def _run_watchdog(
    args: argparse.Namespace,
    requested: Sequence[str],
) -> None:
    command = [*requested, "--worker-suite"]
    environment = os.environ.copy()
    environment[_CHILD_ENV] = "1"
    environment[_LAUNCH_ENV] = json.dumps(list(requested))
    process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    cleanup_signals = (
        signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)

    def process_group_exists() -> bool:
        if process_group_id is None:
            return False
        try:
            os.killpg(process_group_id, 0)
            return True
        except ProcessLookupError:
            return False

    def stop_process_group() -> None:
        if process is None:
            return
        cleanup_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, cleanup_signals)
        cleanup_failure = None
        try:
            if process_group_exists():
                try:
                    assert process_group_id is not None
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 10
            while process_group_exists() and time.monotonic() < deadline:
                process.poll()
                time.sleep(0.1)
            if process_group_exists():
                try:
                    assert process_group_id is not None
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                deadline = time.monotonic() + 10
                while (process_group_exists() and
                       time.monotonic() < deadline):
                    process.poll()
                    time.sleep(0.1)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                cleanup_failure = "watchdog group leader was not reaped"
            if process_group_exists():
                cleanup_failure = (
                    "watchdog process group survived SIGTERM and SIGKILL")
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, cleanup_mask)
        if cleanup_failure is not None:
            raise RuntimeError(cleanup_failure)

    def interrupt_parent(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received parent signal {signum}")

    spawn_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, cleanup_signals)
    signals_unblocked = False
    previous_handlers = {}
    try:
        for signum in (signal.SIGHUP, signal.SIGTERM, signal.SIGQUIT):
            previous_handlers[signum] = signal.signal(
                signum, interrupt_parent)
        try:
            process = subprocess.Popen(
                command,
                start_new_session=True,
                env=environment,
            )
            process_group_id = process.pid
            signals_unblocked = True
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)
            return_code = process.wait(timeout=args.watchdog_seconds)
        except subprocess.TimeoutExpired as error:
            stop_process_group()
            raise RuntimeError(
                "Gloo diagnostic watchdog expired after "
                f"{args.watchdog_seconds}s") from error
        except BaseException:
            stop_process_group()
            raise
        if return_code:
            stop_process_group()
            raise SystemExit(return_code)
        stop_process_group()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if not signals_unblocked:
            signal.pthread_sigmask(signal.SIG_SETMASK, spawn_mask)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    parser = _parser()
    if args.worker_suite:
        if os.environ.get(_CHILD_ENV) != "1":
            parser.error("--worker-suite is an internal watchdog mode")
        torch.multiprocessing.spawn(
            _worker,
            args=(args,),
            nprocs=_WORLD_SIZE,
            join=True,
        )
        return
    requested = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
    ]
    _run_watchdog(args, requested)


if __name__ == "__main__":
    main()
