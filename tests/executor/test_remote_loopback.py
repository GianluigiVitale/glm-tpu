"""G8 loopback: execute the exact remote command strings the controller builds, on this host.

Each string comes from ``glm_tpu.executor.fleet`` (the builders the controller uses) and runs under
``bash -c`` with a host-like environment (``PATH`` and ``HOME`` only), the helper tier on
``GLM_TPU_TEST_HELPER_PYTHON`` (CI: ``python3.10``) and ``sys.executable`` as ``worker_python``.
The source is a tiny git repository whose stand-in for the worker module (at the path of
``resident_protocol.WORKER_MODULE``, started and authenticated under that name) sleeps; the bundle
is built by the launcher's real ``stage_bundle``; the run directory's name has spaces, quotes and
non-ASCII characters. ``hosts`` is ``[socket.gethostname()]`` (rank 0). ``sudo`` is a stand-in on
``PATH`` that answers the idle probe's ``fuser`` question ("no holder"), so no privileged command
and no TPU is ever touched. This is the test that would have caught F-C1 (post-staging helpers that
cannot import).
"""
from __future__ import annotations

import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from glm_tpu.config.site import SiteConfig
from glm_tpu.engine import resident_protocol as protocol
import glm_tpu.envs as envs
from glm_tpu.executor import fleet as remote
from tests.fixtures.site import example_mapping

STUB = '''"""Loopback stand-in for the worker module (standard library only)."""
import json, os, sys, time
state = dict(cwd=os.getcwd(), jax=os.environ.get("JAX_PLATFORMS"), flag=os.environ.get("{flag}"),
             pythonpath=os.environ.get("PYTHONPATH"), argv=sys.argv[1:])
if "--preflight-only" in sys.argv:
    print(json.dumps(state))
    raise SystemExit(0)
print("STUB " + json.dumps(state), flush=True)
time.sleep(600)
'''.replace("{flag}", protocol.WORKER_ENV_FLAG)
MODULE = protocol.WORKER_MODULE             # the stand-in runs as the real worker module name
STUB_PATH = protocol.module_path(MODULE)    # glm_tpu/worker/tpu_worker.py in the stub repository
NO_HOLDER_SUDO = '''#!/bin/sh
# Loopback stand-in for `sudo -n env LC_ALL=C fuser /tmp/libtpu_lockfile`: answers "no holder".
[ "$*" = "-n env LC_ALL=C fuser /tmp/libtpu_lockfile" ] || exit 99
if [ -e /tmp/libtpu_lockfile ]; then exit 1; fi
echo "Specified filename /tmp/libtpu_lockfile does not exist." >&2
exit 1
'''
HOLDER_SUDO = '''#!/bin/sh
# Stand-in reporting one libtpu holder.
[ "$*" = "-n env LC_ALL=C fuser /tmp/libtpu_lockfile" ] || exit 99
echo " 4242"
echo "/tmp/libtpu_lockfile:" >&2
exit 0
'''
GIT_ENV = dict(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="fixture",
               GIT_AUTHOR_EMAIL="fixture@example.invalid", GIT_COMMITTER_NAME="fixture",
               GIT_COMMITTER_EMAIL="fixture@example.invalid")
SCALE = envs.GLM_TPU_TEST_TIMEOUT_SCALE


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o700)
    return path


class Host:
    """This machine as one fleet host: ``bash -c <exact command>`` with ``PATH`` and ``HOME`` only."""

    def __init__(self, base: Path, sudo: str = NO_HOLDER_SUDO) -> None:
        self.home = base / "home"
        self.home.mkdir()
        tools = base / "host-bin"
        tools.mkdir()
        _executable(tools / "sudo", sudo)
        self.env = dict(PATH=os.pathsep.join([str(tools), os.environ.get("PATH", "")]), HOME=str(self.home))

    def run(self, command: str, *, payload: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["bash", "-c", command], cwd=self.home, env=self.env, input=payload,
                              capture_output=True, timeout=120 * SCALE)

    def start(self, command: str) -> subprocess.Popen[bytes]:
        return subprocess.Popen(["bash", "-c", command], cwd=self.home, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture
def fleet_run(tmp_path, monkeypatch):
    """A pinned tiny source repository, a synthetic site and a run directory with a hostile name."""
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / STUB_PATH).parent.mkdir(parents=True)
    for package in Path(STUB_PATH).parents:  # regular packages, so the staged source wins on sys.path
        if package != Path("."):
            (repo / package / "__init__.py").write_text("")
    (repo / STUB_PATH).write_text(STUB)
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "stub"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    pin = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                         text=True).stdout.strip()
    binding = tmp_path / "binding"
    (binding / "captures").mkdir(parents=True)
    digests = {}
    for rank in range(8):
        raw = json.dumps(dict(rank=rank)).encode()
        (binding / "captures" / f"topology.rank{rank}.json").write_bytes(raw)
        digests[f"topology.rank{rank}.json"] = sha256(raw).hexdigest()
    rebinding = json.dumps(dict(capture_sha256=digests), sort_keys=True).encode()
    (binding / "topology_rebinding.json").write_bytes(rebinding)
    extra = tmp_path / "site-packages"
    extra.mkdir()
    site = SiteConfig.from_mapping(example_mapping(
        tmp_path, fleet=dict(helper_python=envs.GLM_TPU_TEST_HELPER_PYTHON, worker_python=sys.executable,
                             worker_pythonpath=[str(extra)]),
        topology=dict(binding_dir=str(binding), binding_sha256=sha256(rebinding).hexdigest())))
    root = tmp_path / "runs with space 'single' \"double\" ünïcödé 日本" / "run_20260923T000000000000Z"
    root.mkdir(parents=True, mode=0o700)
    return dict(repo=repo, pin=pin, site=site, root=root, hosts=[socket.gethostname()], extra=extra)


def _idle(fleet, root: Path, hosts: list[str] | None) -> str:
    return remote.command(fleet, "idle_probe", dict(root=str(root), hosts=hosts))


def _wait_for(predicate, what: str, seconds: float = 60) -> None:
    deadline = time.monotonic() + seconds * SCALE
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def test_the_exact_command_strings_run_a_whole_host_lifecycle(tmp_path, fleet_run):
    from glm_tpu.executor import staging

    site, root, pin, hosts = fleet_run["site"], fleet_run["root"], fleet_run["pin"], fleet_run["hosts"]
    fleet, host, hostname = site.fleet, Host(tmp_path), hosts[0]

    # idle_before: no hosts yet (fresh root), no rank needed
    result = host.run(_idle(fleet, root, None))
    assert result.returncode == 0, result.stderr.decode()[-2000:]
    assert result.stdout.decode() == f"IDLE {hostname}\n"

    # stage_bundle on worker_python: a wrong digest refuses before extraction, the right one extracts
    bundle, manifest_sha = staging.stage_bundle(fleet_run["repo"], pin, root, b'{"request": "loopback"}\n', site)
    wrong = host.run(remote.command(fleet, "stage_bundle", dict(root=str(root), digest="0" * 64, hosts=hosts)),
                     payload=bundle)
    assert wrong.returncode != 0 and b"staging transport differs" in wrong.stderr
    assert not (root / "source").exists()
    staged = host.run(remote.command(fleet, "stage_bundle",
                                     dict(root=str(root), digest=sha256(bundle).hexdigest(), hosts=hosts)),
                      payload=bundle)
    assert staged.returncode == 0, staged.stderr.decode()[-2000:]
    assert (root / "source" / STUB_PATH).read_text() == STUB
    assert sha256((root / "source_manifest.json").read_bytes()).hexdigest() == manifest_sha
    assert (root / "site.json").read_bytes() == site.resolved_json()
    assert oct((root / "site.json").stat().st_mode & 0o777) == "0o600"

    # CPU preflight: the worker module on worker_python from <root>/source
    worker_argv = ["--output", str(root), "--code-hash", pin]
    preflight = host.run(remote.preflight_command(fleet, root, [fleet.worker_python, "-m", MODULE, *worker_argv]))
    assert preflight.returncode == 0, preflight.stderr.decode()[-2000:]
    state = json.loads(preflight.stdout)
    source = str(root / "source")
    assert state == dict(cwd=source, jax="cpu", flag="1", pythonpath=f"{source}:{fleet_run['extra']}",
                         argv=[*worker_argv, "--preflight-only"])

    # start_worker: the marker first, then execv into the stub (this process becomes the worker)
    start = remote.command(fleet, "start_worker", dict(
        root=str(root), hosts=hosts, pin=pin, worker_python=fleet.worker_python,
        pythonpath=list(fleet.worker_pythonpath), module=MODULE,
        env={"JAX_PLATFORMS": "cpu", protocol.WORKER_ENV_FLAG: "1"}, argv=worker_argv))
    process = host.start(start)
    try:
        marker = root / protocol.worker_started_file(0)
        _wait_for(marker.exists, "the start marker")
        line = process.stdout.readline().decode()
        assert line.startswith("STUB "), process.stderr.read().decode()[-2000:] if process.poll() else line
        assert json.loads(line[5:]) == dict(cwd=source, jax="cpu", flag="1",
                                            pythonpath=f"{source}:{fleet_run['extra']}", argv=worker_argv)
        owner = json.loads(marker.read_text())
        assert sorted(owner) == ["boot_id", "code_hash", "hostname", "pid", "start_ticks"]
        assert owner["code_hash"] == pin and owner["hostname"] == hostname
        argv = (Path("/proc") / str(owner["pid"]) / "cmdline").read_bytes().split(b"\0")
        assert MODULE.encode() in argv and str(root).encode() in argv and pin.encode() in argv

        # while it runs the idle probe refuses
        live = host.run(_idle(fleet, root, hosts))
        assert live.returncode != 0 and b"request worker is live" in live.stderr

        # cleanup refuses an identity or argv that differs, and leaves the process alone
        other_pin = host.run(remote.command(fleet, "cleanup", dict(root=str(root), hosts=hosts, pin="b" * 40,
                                                                   module=MODULE)))
        assert other_pin.returncode != 0 and b"cleanup identity differs" in other_pin.stderr
        other_module = host.run(remote.command(fleet, "cleanup", dict(root=str(root), hosts=hosts, pin=pin,
                                                                      module="other_worker")))
        assert other_module.returncode != 0 and b"cleanup argv differs" in other_module.stderr
        assert process.poll() is None

        # authenticated cleanup kills exactly this worker
        cleanup = host.run(remote.command(fleet, "cleanup", dict(root=str(root), hosts=hosts, pin=pin,
                                                                 module=MODULE)))
        assert cleanup.returncode == 0, cleanup.stderr.decode()[-2000:]
        process.wait(timeout=30 * SCALE)
        assert process.returncode in (-9, 137)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    # fetch: base64 records round-trip, names derived from the rank
    record = b'{"complete": true, "rank": 0}\n'
    (root / protocol.runner_file(0)).write_bytes(record)
    fetched = host.run(remote.command(fleet, "fetch", dict(dir=str(root), hosts=hosts,
                                                           names=["runner.rank{rank}.json",
                                                                  "worker_started.rank{rank}.json",
                                                                  "missing.rank{rank}.json"])))
    assert fetched.returncode == 0, fetched.stderr.decode()[-2000:]
    files = {name: base64.b64decode(data) for name, data in json.loads(fetched.stdout).items()}
    assert files == {"runner.rank0.json": record, "worker_started.rank0.json": marker.read_bytes()}

    # idle_after: idle again, with the authenticated host list
    after = host.run(_idle(fleet, root, hosts))
    assert after.returncode == 0, after.stderr.decode()[-2000:]
    assert after.stdout.decode() == f"IDLE {hostname}\n"
    # a second cleanup finds nothing to do (process gone)
    assert host.run(remote.command(fleet, "cleanup", dict(root=str(root), hosts=hosts, pin=pin,
                                                          module=MODULE))).returncode == 0


def test_the_idle_probe_refuses_a_libtpu_holder_and_a_host_outside_the_fleet(tmp_path, fleet_run):
    fleet, root = fleet_run["site"].fleet, fleet_run["root"]
    (tmp_path / "held").mkdir()
    result = Host(tmp_path / "held", sudo=HOLDER_SUDO).run(_idle(fleet, root, None))
    assert result.returncode != 0 and not result.stdout  # "libtpu is owned" (or its state is unknowable)
    (tmp_path / "stranger").mkdir()
    outside = Host(tmp_path / "stranger").run(_idle(fleet, root, ["example-w-0"]))
    assert outside.returncode != 0 and b"not in the authenticated fleet" in outside.stderr


def test_the_helper_interpreter_is_at_least_python_310():
    result = subprocess.run([envs.GLM_TPU_TEST_HELPER_PYTHON, "-c", "import sys; print(sys.version_info[:2])"],
                            capture_output=True, text=True, timeout=60 * SCALE)
    assert result.returncode == 0 and tuple(json.loads(result.stdout.replace("(", "[").replace(")", "]"))) >= (3, 10)
