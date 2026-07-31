"""CPU-only contracts for campaign-owned nested process groups."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock


_ROOT = Path(__file__).resolve().parents[2]
_SUPERVISED_ENV = "DEEP_EP_CAMPAIGN_SUPERVISED"
_TARGETS = {
    "tests/elastic/run_rail_balance_build_warmup.py": "environment",
    "tests/elastic/test_rail_balance_hybrid_dispatch_codegen.py": None,
    "tests/elastic/test_rail_balance_hybrid_combine_codegen.py": None,
    "tests/elastic/test_rail_balance_hop_vnode_cuda.py": None,
    "tests/elastic/bench_rail_balance_hybrid_lsa.py": "child_environment",
}


def _tree(relative: str) -> ast.Module:
    path = _ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _switch(relative: str):
    function = next(
        node
        for node in _tree(relative).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_nested_start_new_session"
    )
    namespace = {
        "os": os,
        "_CAMPAIGN_SUPERVISED_ENV": _SUPERVISED_ENV,
    }
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, relative, "exec"), namespace)
    return namespace["_nested_start_new_session"]


def test_every_nested_popen_uses_the_supervision_switch() -> None:
    for relative, expected_argument in _TARGETS.items():
        calls = [
            node
            for node in ast.walk(_tree(relative))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "Popen"
        ]
        assert len(calls) == 1, (relative, len(calls))
        keywords = {
            keyword.arg: keyword.value
            for keyword in calls[0].keywords
            if keyword.arg is not None
        }
        value = keywords.get("start_new_session")
        assert isinstance(value, ast.Name), (relative, ast.dump(value))
        assert value.id == "start_new_session", (relative, ast.dump(value))

        assignments = [
            node
            for node in ast.walk(_tree(relative))
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "start_new_session"
                for target in node.targets
            )
        ]
        assert len(assignments) == 1, (relative, len(assignments))
        call = assignments[0].value
        assert isinstance(call, ast.Call), (relative, ast.dump(call))
        assert isinstance(call.func, ast.Name), (relative, ast.dump(call))
        assert call.func.id == "_nested_start_new_session", relative
        if expected_argument is None:
            assert not call.args, (relative, ast.dump(call))
        else:
            assert len(call.args) == 1, (relative, ast.dump(call))
            assert isinstance(call.args[0], ast.Name)
            assert call.args[0].id == expected_argument


def test_switch_defaults_to_setsid_and_only_one_disables_it() -> None:
    for relative in _TARGETS:
        switch = _switch(relative)
        with mock.patch.dict(os.environ, {}, clear=True):
            assert switch() is True, relative
        with mock.patch.dict(
            os.environ, {_SUPERVISED_ENV: "1"}, clear=True
        ):
            assert switch() is False, relative
        with mock.patch.dict(
            os.environ, {_SUPERVISED_ENV: "0"}, clear=True
        ):
            assert switch() is True, relative
        assert switch({}) is True, relative
        assert switch({_SUPERVISED_ENV: "1"}) is False, relative


def _session_probe(start_new_session: bool) -> tuple[int, int, int]:
    code = (
        "import os; "
        "print(os.getpid(), os.getsid(0), os.getpgrp(), flush=True)"
    )
    output = subprocess.check_output(
        [sys.executable, "-B", "-c", code],
        env={"CUDA_VISIBLE_DEVICES": ""},
        start_new_session=start_new_session,
        text=True,
    )
    child_pid, session_id, process_group = map(int, output.split())
    return child_pid, session_id, process_group


def test_switch_controls_real_cpu_child_session_membership() -> None:
    switch = _switch("tests/elastic/run_rail_balance_build_warmup.py")
    default_pid, default_session, default_group = _session_probe(switch({}))
    assert default_pid == default_session == default_group

    child_pid, inherited_session, inherited_group = _session_probe(
        switch({_SUPERVISED_ENV: "1"})
    )
    assert child_pid != inherited_group
    assert inherited_session == os.getsid(0)
    assert inherited_group == os.getpgrp()


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} campaign supervision CPU contracts")
