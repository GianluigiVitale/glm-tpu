"""Remote helpers are self-contained stdlib programs; the command builder never templates them."""
from __future__ import annotations

import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import shlex
import socket
import subprocess

import pytest

from glm_tpu.engine import resident_protocol as protocol
import glm_tpu.envs as envs
from glm_tpu.executor import fleet as remote
from glm_tpu.executor.remote import HELPERS
from tests.fixtures.site import example_site

REMOTE_DIR = Path(remote.__file__).resolve().parent / "remote"
# Standard-library modules that exist on Python 3.10 (the hosts' system python3).
STDLIB_310 = {"base64", "hashlib", "io", "json", "os", "pathlib", "re", "signal", "socket", "subprocess", "sys",
              "tarfile"}


def _tree(name: str) -> ast.Module:
    return ast.parse((REMOTE_DIR / f"{name}.py").read_text(), feature_version=(3, 10))


def test_the_package_lists_exactly_its_helper_files():
    files = {p.stem for p in REMOTE_DIR.glob("*.py")} - {"__init__"}
    assert files == set(HELPERS)


@pytest.mark.parametrize("name", HELPERS)
def test_a_helper_is_a_stdlib_only_python310_program_with_one_entry_point(name):
    tree = _tree(name)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a helper is a standalone program: no relative imports"
            imported.add((node.module or "").split(".")[0])
    assert imported <= STDLIB_310 | {"__future__"}, imported - STDLIB_310
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert [a.arg for a in functions["main"].args.args] == ["argv"]
    guard = tree.body[-1]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "__name__ == '__main__'"
    assert ast.unparse(guard.body[0]) == "raise SystemExit(main(sys.argv[1:]))"


def test_helpers_compile_under_the_helper_interpreter():
    python = envs.GLM_TPU_TEST_HELPER_PYTHON
    for name in HELPERS:
        path = REMOTE_DIR / f"{name}.py"
        result = subprocess.run([python, "-c", "import sys; compile(open(sys.argv[1]).read(), sys.argv[1], 'exec')",
                                 str(path)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (name, result.stderr[-2000:])


def test_libtpu_holders_is_the_collector_guard_verbatim():
    from scripts.greenfield.watch_ws32_run import REMOTE

    start = REMOTE.index("def libtpu_holders():")
    original = REMOTE[start:REMOTE.index("\ntag, pin =")].strip()
    text = (REMOTE_DIR / "idle_probe.py").read_text()
    copy = text[text.index("def libtpu_holders():"):text.index("\ndef arguments(")].strip()
    assert copy == original


# ----------------------------------------------------------------------------- command builder
@pytest.mark.parametrize("root", ["/runs/run_20260923T000000000000Z", "/runs/with space/it's \"quoted\"",
                                  "/runs/ünïcödé/日本"])
def test_remote_command_carries_the_file_text_and_one_json_argument(root):
    args = dict(root=root, hosts=["example-w-0", "example-w-1"])
    command = remote.remote_command("idle_probe", args, interpreter="python3")
    words = shlex.split(command)
    assert words[:2] == ["python3", "-c"] and len(words) == 4
    assert words[2] == (REMOTE_DIR / "idle_probe.py").read_text()
    assert json.loads(words[3]) == args and words[3].isascii()


def test_the_command_builder_refuses_unknown_helpers_and_oversized_commands():
    with pytest.raises(ValueError, match="unknown remote helper"):
        remote.remote_command("shell", {}, interpreter="python3")
    with pytest.raises(ValueError, match="exceeds"):
        remote.remote_command("fetch", dict(dir="/x", hosts=["h"], names=["n" * (100 << 10)]), interpreter="python3")


def test_interpreters_are_the_181c013e_ones(tmp_path):
    fleet = example_site(tmp_path).fleet
    for name in HELPERS:
        words = shlex.split(remote.command(fleet, name, {}))
        expected = fleet.worker_python if name == "stage_bundle" else fleet.helper_python
        assert words[0] == expected, name


def test_helpers_record_names_each_helper_digest_and_role():
    record = remote.helpers_record()
    assert record["schema"] == "glm_tpu_remote_helpers_v1" and set(record["helpers"]) == set(HELPERS)
    for name, entry in record["helpers"].items():
        assert entry["sha256"] == sha256((REMOTE_DIR / f"{name}.py").read_bytes()).hexdigest()
    assert record["helpers"]["stage_bundle"]["interpreter"] == "fleet.worker_python"
    assert record["helpers"]["cleanup"]["interpreter"] == "fleet.helper_python"


def test_preflight_command_is_the_181c013e_form(tmp_path):
    fleet = example_site(tmp_path).fleet
    root = Path("/runs/run x")
    command = remote.preflight_command(fleet, root, [fleet.worker_python, "-m", protocol.WORKER_MODULE, "--output",
                                                     str(root)])
    assert command == ("cd '/runs/run x/source' && env JAX_PLATFORMS=cpu GLM_OPTIMIZED_REQUEST=1 "
                       f"'PYTHONPATH=/runs/run x/source:{fleet.worker_pythonpath[0]}' {fleet.worker_python} "
                       f"-m {protocol.WORKER_MODULE} --output '/runs/run x' --preflight-only")


# ----------------------------------------------------------------------------- helper mains in-process
def _run(name: str, args: object, monkeypatch, capsys, *, hostname: str = "example-w-2") -> tuple[int, str]:
    namespace: dict[str, object] = {"__name__": "helper"}
    exec(compile((REMOTE_DIR / f"{name}.py").read_text(), name, "exec"), namespace)
    monkeypatch.setattr(socket, "gethostname", lambda: hostname)
    code = namespace["main"]([json.dumps(args)])
    return code, capsys.readouterr().out


def test_fetch_derives_names_from_the_rank_in_the_authenticated_list(tmp_path, monkeypatch, capsys):
    (tmp_path / "runner.rank2.json").write_bytes(b'{"rank": 2}\n')
    hosts = [f"example-w-{r}" for r in range(8)]
    code, out = _run("fetch", dict(dir=str(tmp_path), hosts=hosts,
                                   names=["runner.rank{rank}.json", "worker_started.rank{rank}.json"]),
                     monkeypatch, capsys)
    assert code == 0
    assert {k: base64.b64decode(v) for k, v in json.loads(out).items()} == {"runner.rank2.json": b'{"rank": 2}\n'}


@pytest.mark.parametrize("names", [["../runner.rank{rank}.json"], ["a/b"], [".."], [], ["{other}"]])
def test_fetch_refuses_path_like_names(tmp_path, monkeypatch, capsys, names):
    with pytest.raises(SystemExit):
        _run("fetch", dict(dir=str(tmp_path), hosts=["example-w-2"], names=names), monkeypatch, capsys)


@pytest.mark.parametrize("name", ["stage_bundle", "start_worker", "cleanup", "fetch"])
def test_post_idle_helpers_refuse_a_host_outside_the_authenticated_list(tmp_path, monkeypatch, capsys, name):
    keys = dict(stage_bundle=dict(root=str(tmp_path), digest="0" * 64),
                start_worker=dict(root=str(tmp_path), pin="a" * 40, worker_python="/usr/bin/python3",
                                  pythonpath=[], module="stub", env={}, argv=[]),
                cleanup=dict(root=str(tmp_path), pin="a" * 40, module="stub"),
                fetch=dict(dir=str(tmp_path), names=["x"]))[name]
    with pytest.raises(RuntimeError, match="not in the authenticated fleet"):
        _run(name, dict(keys, hosts=["example-w-0", "example-w-1"]), monkeypatch, capsys)


@pytest.mark.parametrize("name", HELPERS)
def test_helpers_refuse_unknown_or_missing_argument_keys(tmp_path, monkeypatch, capsys, name):
    with pytest.raises(SystemExit):
        _run(name, dict(root=str(tmp_path), unexpected=True), monkeypatch, capsys)
