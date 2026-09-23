"""G8 contracts: every resident-protocol constant names what the processes really use."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
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


def test_the_worker_checks_its_own_manifest_path_and_handshake_flag():
    from scripts.release import ws32_optimized_worker as worker

    text = Path(worker.__file__).read_text()
    assert f"'{protocol.module_path(protocol.WORKER_MODULE)}' not in manifest" in text
    assert f"os.environ.get('{protocol.WORKER_ENV_FLAG}')!='1'" in text
    assert Path(worker.__file__).resolve().relative_to(REPO).as_posix() == protocol.module_path(protocol.WORKER_MODULE)


def test_the_pack_worker_checks_its_own_manifest_path_and_handshake_flag():
    text = (REPO / protocol.module_path(protocol.PACK_WORKER_MODULE)).read_text()
    assert f"'{protocol.module_path(protocol.PACK_WORKER_MODULE)}' not in manifest" in text
    assert f"os.environ.get('{protocol.PACK_WORKER_ENV_FLAG}') != '1'" in text


def test_the_controller_starts_and_authenticates_the_worker_module():
    from scripts.release import launch_ws32_optimized_request as launch

    assert launch.MODULE == protocol.WORKER_MODULE
    import inspect

    assert inspect.signature(launch.cleanup_owned).parameters["module"].default == protocol.WORKER_MODULE


def test_the_resident_client_checks_the_controller_module():
    text = (REPO / "glm_tpu" / "ui.py").read_text()
    assert f"b'{protocol.CONTROLLER_MODULE}' in argv" in text


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
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO, env=env, capture_output=True,
                            text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
