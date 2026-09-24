"""G8 contracts: every resident-protocol constant names what the processes really use."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from glm_tpu.engine import resident_protocol as protocol

REPO = Path(__file__).resolve().parents[2]
MODULES = (protocol.CONTROLLER_MODULE, protocol.WORKER_MODULE, protocol.PACK_WORKER_MODULE)


@pytest.mark.parametrize("module", MODULES)
def test_every_process_module_constant_is_a_real_module_path(module):
    origin = importlib.util.find_spec(module).origin
    assert origin.endswith(protocol.module_path(module))
    assert protocol.source_root(origin, module) == REPO


def test_the_remote_helper_package_constant_is_the_helper_package():
    from glm_tpu.executor import remote

    assert remote.__name__ == protocol.REMOTE_HELPER_PACKAGE


@pytest.mark.parametrize(
    ("module", "flag"), [("WORKER_MODULE", "WORKER_ENV_FLAG"), ("PACK_WORKER_MODULE", "PACK_WORKER_ENV_FLAG")]
)
def test_a_worker_derives_its_manifest_path_from_its_file_and_checks_its_handshake_flag(module, flag):
    name = getattr(protocol, module)
    worker = importlib.import_module(name)
    # The source root and the manifest self-path come from the module's own file, never a literal.
    assert worker.REPO == protocol.source_root(worker.__file__, name) == REPO
    assert worker.SELF == protocol.module_path(name)  # noqa: SIM300 (actual == expected)
    text = Path(worker.__file__).read_text()
    assert "SELF not in manifest" in text
    assert all(f"{quote}{protocol.module_path(name)}{quote}" not in text for quote in "'\"")
    assert re.search(rf"os\.environ\.get\(protocol\.{flag}\)\s*!=\s*[\"']1[\"']", text)
    assert getattr(protocol, flag).startswith("GLM_TPU_")


def test_the_controller_starts_and_authenticates_the_worker_module():
    from glm_tpu.executor import multihost_executor as launch

    assert launch.MODULE == protocol.WORKER_MODULE
    assert launch.REPO == protocol.source_root(launch.__file__, protocol.CONTROLLER_MODULE) == REPO
    import inspect

    assert inspect.signature(launch.cleanup_owned).parameters["module"].default == protocol.WORKER_MODULE


@pytest.mark.parametrize(
    ("argv_module", "accepted"),
    [
        (protocol.CONTROLLER_MODULE, True),
        ("scripts.release.launch_ws32_optimized_request", False),  # D11: no legacy controller acceptance
    ],
)
def test_the_resident_client_accepts_only_the_controller_module(tmp_path, argv_module, accepted):
    from glm_tpu.engine.resident_client import Resident

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", argv_module, "--keep-loaded"])
    try:
        ticks = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        client = Resident.__new__(Resident)
        client.run, client.identity = tmp_path, dict(pid=process.pid, start_ticks=ticks)
        if accepted:
            client.check()
        else:
            with pytest.raises(ValueError, match="resident model is unavailable"):
                client.check()
    finally:
        process.kill()
        process.wait()


def test_source_root_refuses_a_file_that_is_not_the_module(tmp_path):
    other = tmp_path / "a" / "b.py"
    other.parent.mkdir()
    other.write_text("")
    with pytest.raises(ValueError, match="is not the file of module"):
        protocol.source_root(other, "x.b")


def test_run_directory_names_and_command_bytes():
    root = Path("/runs/run")
    assert protocol.result_dir(root, 0) == root and protocol.result_dir(root, 12) == root / "resident-0012"
    assert protocol.inbox_name(1) == "0001.json" and protocol.runner_file(3) == "runner.rank3.json"
    assert protocol.worker_started_file(7) == "worker_started.rank7.json"
    assert json.loads(protocol.STOP_COMMAND) == {"stop": True} and protocol.STOP_COMMAND.endswith(b"\n")
    line = protocol.encode_command(2, {"request_id": "prüfung-日本"})
    assert line == '{"request":{"request_id":"prüfung-日本"},"sequence":2}\n'.encode()


@pytest.mark.parametrize("module", MODULES)
def test_python_m_help_resolves_for_every_process_module(module):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    env.pop(protocol.WORKER_ENV_FLAG, None)
    env.pop(protocol.PACK_WORKER_ENV_FLAG, None)
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
