"""A fake eight-host fleet for the fleet-job tests (``glm_tpu.executor.jobs`` and the jobs built on it).

Every ``subprocess.run``/``Popen`` of an SSH command is answered by :class:`FakeJobFleet`: a remote helper is
identified by its exact file text and its JSON argument decoded, each host keeps its own files (rank 0's are the real
run directory, as on the controller host), the staged bundle is extracted per host, and the job module's CPU steps
(``cd <root>/source && env ... -m <module> ...``) and started processes are answered by the test's callbacks. Faked
besides: ``gcloud`` discovery, the hostname, the launch policy's pin and the helper snapshot (this package's files).
Neutral example values only."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
from typing import Any

from glm_tpu.executor import fleet as remote
from glm_tpu.executor import jobs
from glm_tpu.executor import multihost_executor as launch
from glm_tpu.executor.remote import HELPERS

FAKE_SSH = "glm-test-ssh"
HOSTS = [f"example-w-{rank}" for rank in range(8)]
GIT_ENV = dict(
    GIT_CONFIG_GLOBAL="/dev/null",
    GIT_CONFIG_NOSYSTEM="1",
    GIT_AUTHOR_NAME="fixture",
    GIT_AUTHOR_EMAIL="fixture@example.invalid",
    GIT_COMMITTER_NAME="fixture",
    GIT_COMMITTER_EMAIL="fixture@example.invalid",
)


class FakeProcess:
    def __init__(self, code: int | None) -> None:
        self.code = code

    def poll(self):
        return self.code

    def wait(self):
        assert self.code is not None, "a fake host process never ended"
        return self.code

    def kill(self) -> None:
        self.code = -9


@dataclass
class FakeJobFleet:
    """Eight hosts. ``cpu(fleet, rank, words) -> (code, output)`` answers a CPU step; ``start(fleet, rank, args) ->
    code`` runs a started process to its end (``None``: still running until cleanup)."""

    runs: Path
    cpu: Callable[[FakeJobFleet, int, list[str]], tuple[int, bytes]]
    start: Callable[[FakeJobFleet, int, dict], int | None]
    idle_failure: bool = False
    hosts: Path | None = None  # ranks 1..7 keep their files under hosts/rank<r>/ (default: beside runs)
    calls: list[tuple[str, int, Any]] = field(default_factory=list)
    processes: dict[int, FakeProcess] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.helpers = {remote.helper_text(name): name for name in HELPERS}
        self._earlier = set(self.runs.iterdir())

    @property
    def root(self) -> Path:
        """The run directory the job created (the one new directory under the run root)."""
        (root,) = [p for p in self.runs.iterdir() if p.is_dir() and p not in self._earlier]
        return root

    def run_dir(self, rank: int) -> Path:
        """Where host ``rank`` keeps this run's directory: rank 0's is the real one."""
        if rank == 0:
            return self.root
        base = self.hosts if self.hosts is not None else self.runs.parent / "hosts"
        return base / f"rank{rank}" / self.root.name

    def put(self, rank: int, relative: str, data: bytes) -> None:
        path = self.run_dir(rank) / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)

    def get(self, rank: int, relative: str) -> bytes | None:
        path = self.run_dir(rank) / relative
        return path.read_bytes() if path.is_file() else None

    def run(self, argv: list[str], stream, payload: bytes | None) -> int:
        rank = HOSTS.index(argv[1])
        words = shlex.split(argv[-1])
        if words[0] == "cd":
            self.calls.append(("cpu", rank, words))
            code, output = self.cpu(self, rank, words)
            stream.write(output)
            return code
        assert words[1] == "-c" and len(words) == 4, words[:2]
        helper, args = self.helpers[words[2]], json.loads(words[3])
        self.calls.append((helper, rank, args))
        if helper == "idle_probe":
            if self.idle_failure and args["hosts"] is not None:
                return 1
            stream.write(f"IDLE {HOSTS[rank]}\n".encode())
        elif helper == "stage_bundle":
            assert sha256(payload).hexdigest() == args["digest"] and args["hosts"] == HOSTS
            relative = Path(args["root"]).relative_to(self.root)
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
                for member in tar:
                    name = str(relative / member.name) if str(relative) != "." else member.name
                    self.put(rank, name, tar.extractfile(member).read())
        elif helper == "cleanup":
            process = self.processes.get(rank)
            if process is not None and process.code is None:
                process.code = -9
        elif helper == "fetch":
            directory = Path(args["dir"]).relative_to(self.root)
            found = {}
            for template in args["names"]:
                name = template.replace("{rank}", str(rank))
                data = self.get(rank, str(directory / name) if str(directory) != "." else name)
                if data is not None:
                    found[name] = base64.b64encode(data).decode()
            stream.write(json.dumps(found, sort_keys=True).encode())
        return 0

    def popen(self, argv: list[str]) -> FakeProcess:
        rank = HOSTS.index(argv[1])
        words = shlex.split(argv[-1])
        assert self.helpers[words[2]] == "start_worker"
        args = json.loads(words[3])
        self.calls.append(("start_worker", rank, args))
        marker = dict(pid=5000 + rank, hostname=HOSTS[rank], code_hash=args["pin"])
        self.put(rank, f"worker_started.rank{rank}.json", json.dumps(marker).encode())
        process = FakeProcess(self.start(self, rank, args))
        self.processes[rank] = process
        return process

    def helper_calls(self, name: str) -> list[tuple[int, Any]]:
        return [(rank, args) for helper, rank, args in self.calls if helper == name]


def synthetic_repo(root: Path) -> str:
    """A committed checkout (one file) whose ``git archive`` the job stages; returns its commit."""
    root.mkdir()
    (root / "README.md").write_text("synthetic checkout\n")
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(command, cwd=root, check=True, capture_output=True, env=dict(os.environ, **GIT_ENV))
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def install(monkeypatch, fleet: FakeJobFleet, repo: Path, pin: str) -> list[tuple[int, int]]:
    """Route SSH to ``fleet`` and fake discovery, hostname, pin and helper snapshot; returns the recorded lease
    operations (index in workload + sync order, flags)."""
    real_run, real_popen, real_flock = subprocess.run, subprocess.Popen, fcntl.flock
    leases: list[tuple[int, int]] = []

    def fake_run(argv, *args, **kwargs):
        if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
            return subprocess.CompletedProcess(argv, fleet.run(argv, kwargs["stdout"], kwargs.get("input")))
        return real_run(argv, *args, **kwargs)

    def fake_popen(argv, *args, **kwargs):
        if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
            return fleet.popen(argv)
        return real_popen(argv, *args, **kwargs)

    def flock(stream, flags):
        leases.append((int(Path(stream.name).name.removeprefix("lock")), flags))
        return real_flock(stream, flags)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(jobs.fcntl, "flock", flock)
    monkeypatch.setattr(launch, "REPO", repo)
    monkeypatch.setattr(launch, "source_identity", lambda path, policy: pin)
    monkeypatch.setattr(launch, "pinned_helpers", lambda path, commit: remote.HelperTexts.from_package())
    monkeypatch.setattr(jobs, "ssh_commands", lambda fleet_: [[FAKE_SSH, host, "--", "true"] for host in HOSTS])
    monkeypatch.setattr(jobs.socket, "gethostname", lambda: HOSTS[0])
    monkeypatch.setattr(jobs, "POLL_SECONDS", 0)
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)
    return leases


ENVIRONMENT = dict(python="3.12.0", jax="0.10.1", jaxlib="0.10.1", libtpu="0.0.41")  # every host's, synthetic


def preflight_facts(pin: str, **changes: Any) -> Callable[[FakeJobFleet, int, list[str]], tuple[int, bytes]]:
    """A CPU-step answer: each host's preflight facts (rank, hostname, pin, no TPU, the environment); ``changes``
    maps a rank to the facts it reports instead (e.g. another pin)."""

    def answer(fleet: FakeJobFleet, rank: int, words: list[str]) -> tuple[int, bytes]:
        facts = dict(rank=rank, hostname=HOSTS[rank], code_hash=pin, tpu_initialized=False, **ENVIRONMENT)
        facts.update(changes.get(f"rank{rank}", {}))
        return 0, (json.dumps(facts) + "\n").encode()

    return answer
