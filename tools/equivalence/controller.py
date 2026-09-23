"""The controller's launch path, driven for real (G9 wire record; G6 ``controller`` stage).

``launcher_record()`` runs the real launcher ``main`` twice -- a successful sequential request and
one whose rank-3 worker exits 1 -- against a synthetic host: a tiny committed git repository as
the source tree (``stage_bundle`` archives it with ``git archive``), synthetic topology captures, a
temporary run root and lock files, all named by a synthetic owner-only site file the launcher loads
and validates (``--site``; ``site_fixture``). Faked are only ``ssh_commands`` (``gcloud`` discovery),
``source_identity`` (it compares the private origin and runs ``git ls-remote`` over the network),
``socket.gethostname`` and the eight SSH hosts: every ``subprocess.run``/``Popen`` of an SSH
command is answered by an in-process emulation of the remote host (``IDLE <host>`` for the idle
probes, a preflight environment record, the worker's records for ``collect``). The real
``remote_all``, ``idle``, ``stage_bundle``, ``cleanup_owned`` and ``summarize`` run; the worker
wrapper the launcher sends is executed in-process with ``os.execv`` and ``os.chdir`` captured, so
the environment the worker process would start with is recorded (``LIBTPU_INIT_ARGS``,
``XLA_FLAGS`` or any other variable the launcher adds shows up).

Recorded (normalized: ``<run>`` for the run directory, ``<python>``/``<site>`` for the synthetic
site's interpreter and site-packages, ``<pin>`` for the source commit, ``<coordinator>`` for its
coordinator address, ``<site_sha256>`` for the digest of the staged ``site.json``, whose content
names the temporary paths, and ``<tmp>`` for those paths): the lock calls
(workload locks non-blocking, sync locks blocking and released before dispatch), every remote
command (inline Python programs by the digest of their normalized text), the staged bundle's
members and manifest keys, the preflight and worker command lines, the worker environment and
``execv`` arguments, the controller's files and stdout markers, and the failure path (authenticated
cleanup, idle-after, refusal).

Standard library only at import time, and the exercise imports nothing beyond the launcher's own
modules: the G6 controller stage runs it and must stay JAX-free.
"""

from __future__ import annotations

import base64
from contextlib import ExitStack
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
from typing import Any
from unittest import mock

FAKE_SSH = "glm-equivalence-ssh"
PYTHON = "/opt/example/bin/python3.12"          # the synthetic site's fleet.worker_python
SITE = "/opt/example/site-packages"             # the synthetic site's fleet.worker_pythonpath
HOSTS = [f"example-w-{rank}" for rank in range(8)]
SOURCE_FILES = {"README.md": "synthetic source tree\n",
                "scripts/release/ws32_optimized_worker.py": "# synthetic worker placeholder\n"}
_GIT_ENV = dict(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="glm-equivalence",
                GIT_AUTHOR_EMAIL="glm-equivalence@example.invalid", GIT_COMMITTER_NAME="glm-equivalence",
                GIT_COMMITTER_EMAIL="glm-equivalence@example.invalid", GIT_AUTHOR_DATE="2000-01-01T00:00:00Z",
                GIT_COMMITTER_DATE="2000-01-01T00:00:00Z")


class _Exec(Exception):
    """Raised by the captured ``os.execv``: the wrapper handed control to the worker."""


def _synthetic_repo(root: Path) -> str:
    """A committed, clean git repository with a fixed commit (dates and identity pinned)."""
    root.mkdir()
    for name, text in SOURCE_FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    env = dict(os.environ, **_GIT_ENV)
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
                    ["git", "commit", "-q", "-m", "synthetic"]):
        subprocess.run(command, cwd=root, env=env, check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


def _binding(root: Path) -> str:
    """Topology rebinding and eight captures as the launcher stages them (content is opaque to the
    launcher: it checks digests only)."""
    captures = root / "captures"
    captures.mkdir(parents=True)
    digests = {}
    for rank in range(8):
        raw = json.dumps(dict(launch_process_id=rank, hostname=HOSTS[rank]), sort_keys=True).encode()
        (captures / f"topology.rank{rank}.json").write_bytes(raw)
        digests[f"topology.rank{rank}.json"] = sha256(raw).hexdigest()
    raw = json.dumps(dict(schema="glm_perf_topology_rebinding_v1", capture_sha256=digests), sort_keys=True).encode()
    (root / "topology_rebinding.json").write_bytes(raw)
    return sha256(raw).hexdigest()


class _Process:
    """A worker process as the controller's supervision loop sees it (``poll`` results in order)."""

    def __init__(self, polls: list[int | None], code: int) -> None:
        self._polls, self.code, self.stdin = list(polls), code, None

    def poll(self) -> int | None:
        return self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]

    def wait(self) -> int:
        return self.code


class _Host:
    """The eight SSH hosts: answers the launcher's remote commands and runs the worker wrapper."""

    def __init__(self, pin: str, request_sha256: str, *, failing_rank: int | None) -> None:
        self.pin, self.request_sha256, self.failing_rank = pin, request_sha256, failing_rank
        self.commands: dict[str, list[str]] = {}
        self.staged: list[bytes] = []
        self.workers: list[list[str]] = []
        self.remote_files: dict[int, dict[str, bytes]] = {}
        self.worker: dict[str, Any] = {}

    def _row(self, rank: int) -> dict[str, Any]:
        return dict(schema="glm_optimized_worker_v1", rank=rank, hostname=HOSTS[rank], code_hash=self.pin,
                    request_sha256=self.request_sha256, complete=rank != self.failing_rank,
                    request=dict(token_sha256="c" * 64, emitted=3),
                    programs=dict(decode=dict(stablehlo_sha256="d" * 64, optimized_hlo_sha256="e" * 64)))

    def run(self, argv: list[str], stream: Any, payload: bytes | None) -> int:
        rank = HOSTS.index(argv[1])
        label = Path(stream.name).name.rsplit(".rank", 1)[0]
        self.commands.setdefault(label, [None] * 8)[rank] = argv[-1]
        if label.startswith("idle"):
            stream.write(f"IDLE {HOSTS[rank]}\n".encode())
        elif label == "stage":
            self.staged.append(payload or b"")
        elif label == "preflight":
            stream.write(json.dumps(dict(hostname=HOSTS[rank], jax="0.10.1", jaxlib="0.10.1", numpy="synthetic",
                                         python="synthetic", sha256=dict(python="0" * 64))).encode())
        elif label == "collect":
            files = self.remote_files.get(rank, {}) if rank else {}
            stream.write(json.dumps({name: base64.b64encode(data).decode() for name, data in files.items()}).encode())
        return 0

    def popen(self, argv: list[str], root: Path) -> Any:
        rank = HOSTS.index(argv[1])
        self.workers.append(argv)
        wrapper = shlex.split(argv[-1])
        if rank == 0:
            self.worker = self.execute_wrapper(wrapper[2], root)
            (root / "runner.rank0.json").write_text(json.dumps(self._row(0), sort_keys=True))
        else:
            self.remote_files[rank] = {
                f"runner.rank{rank}.json": json.dumps(self._row(rank), sort_keys=True).encode(),
                f"worker_started.rank{rank}.json": json.dumps(dict(pid=1000 + rank, hostname=HOSTS[rank]),
                                                               sort_keys=True).encode()}
        code = 1 if rank == self.failing_rank else 0
        # With a failing rank, the others are still running at the controller's first look.
        return _Process([code] if self.failing_rank in (None, rank) else [None, 0], code)

    @staticmethod
    def execute_wrapper(code: str, root: Path) -> dict[str, Any]:
        """Run the wrapper text in-process: it writes its start marker, changes directory, updates
        the environment and ``execv``-s the worker (captured)."""
        captured: dict[str, Any] = {}
        before = dict(os.environ)

        def execv(path: str, argv: Any) -> None:
            captured["execv"] = [path, list(argv)]
            captured["environment"] = {k: os.environ.get(k, "<unset>") for k in sorted(set(before) | set(os.environ))
                                       if before.get(k) != os.environ.get(k)}
            raise _Exec

        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ))
            stack.enter_context(mock.patch.object(os, "execv", execv))
            stack.enter_context(mock.patch.object(os, "chdir", lambda path: captured.setdefault("chdir", str(path))))
            try:
                exec(compile(code, "<launcher worker wrapper>", "exec"), {"__name__": "__main__"})
            except _Exec:
                pass
        captured["start_marker_keys"] = sorted(json.loads((root / "worker_started.rank0.json").read_text()))
        return captured


def _normalizer(replacements: dict[str, str]) -> Any:
    pairs = sorted(((value, label) for value, label in replacements.items() if value), key=lambda kv: -len(kv[0]))

    def normalize(text: str) -> str:
        for value, label in pairs:
            text = text.replace(value, label)
        return text

    return normalize


def _command(text: str | None, normalize: Any) -> Any:
    """A remote command: shell words, with an inline ``-c`` program replaced by its digest."""
    if text is None:
        return None
    words = shlex.split(text)
    if "-c" in words:
        index = words.index("-c")
        program = normalize(words[index + 1])
        return dict(argv=[normalize(w) for w in words[:index + 1]] + ["<program>"],
                    program_sha256=sha256(program.encode()).hexdigest(), program_lines=len(program.splitlines()))
    return dict(argv=[normalize(w) for w in words])


def _tree(root: Path, normalize: Any) -> list[str]:
    return sorted(normalize(str(p.relative_to(root))) + ("/" if p.is_dir() else "") for p in root.rglob("*"))


def _bundle(raw: bytes, base: Path) -> dict[str, Any]:
    """Members (name, mode, size and digest of the content with the temporary base path normalized
    to ``<tmp>``: the staged ``site.json`` names the synthetic site's paths) and manifest keys."""
    members = []
    manifest: dict[str, Any] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar:
            data = tar.extractfile(member).read() if member.isfile() else b""
            data = data.replace(str(base).encode(), b"<tmp>")
            members.append([member.name, oct(member.mode), len(data), sha256(data).hexdigest()])
            if member.name == "source_manifest.json":
                manifest = json.loads(data)
    return dict(members=sorted(members), manifest_keys=sorted(manifest))


def _staged_site_sha(raw: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        return sha256(tar.extractfile("site.json").read()).hexdigest()


def _scenario(launch: Any, worker: Any, base: Path, value: dict[str, Any], *,
              failing_rank: int | None) -> dict[str, Any]:
    import fcntl
    import socket

    from glm_tpu import user_request as legacy

    from .site_fixture import EXAMPLE_COORDINATOR, site_mapping, write_site

    repo, runs, locks, binding = base / "repo", base / "runs", base / "locks", base / "binding"
    pin = _synthetic_repo(repo)
    runs.mkdir(mode=0o700)
    locks.mkdir()
    binding_sha = _binding(binding)
    request_path = base / "request.json"
    raw = legacy.canonical(value) + b"\n"
    fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
    host = _Host(pin, value["request_sha256"], failing_rank=failing_rank)
    lock_paths = [str(locks / f"lock{index}") for index in range(4)]
    site_path = write_site(base / "site.toml", site_mapping(
        base, fleet=dict(worker_python=PYTHON, worker_pythonpath=[SITE], coordinator_address=EXAMPLE_COORDINATOR),
        paths=dict(run_root=str(runs)), topology=dict(binding_dir=str(binding), binding_sha256=binding_sha),
        locks=dict(workload=lock_paths[:2], sync=lock_paths[2:])))
    flocks: list[list[Any]] = []
    real_flock, real_run, real_popen = fcntl.flock, subprocess.run, subprocess.Popen
    identity_calls: list[str] = []

    def flock(stream: Any, flags: int) -> None:
        names = [name for name in ("LOCK_EX", "LOCK_NB", "LOCK_UN", "LOCK_SH") if flags & getattr(fcntl, name)]
        flocks.append([lock_paths.index(stream.name), names])
        real_flock(stream, flags)

    def run(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
            return subprocess.CompletedProcess(argv, host.run(argv, kwargs["stdout"], kwargs.get("input")))
        return real_run(argv, *args, **kwargs)

    def popen(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
            root = next(p for p in runs.iterdir() if p.is_dir())
            return host.popen(argv, root)
        return real_popen(argv, *args, **kwargs)

    def source_identity(path: Any) -> str:
        identity_calls.append("<source root>" if Path(path) == repo else "<other>")
        return pin

    printed = io.StringIO()
    outcome = "returned 0"
    umask = os.umask(0o077)
    os.umask(umask)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, _GIT_ENV))
        for name in [k for k in os.environ if k.startswith("GLM_TPU_")]:  # the site file alone decides
            del os.environ[name]  # restored by the patch.dict above
        for name, replacement in dict(REPO=repo, source_identity=source_identity,
                                      ssh_commands=lambda fleet: [[FAKE_SSH, h, "--", "true"] for h in HOSTS]).items():
            stack.enter_context(mock.patch.object(launch, name, replacement))
        stack.enter_context(mock.patch.object(socket, "gethostname", lambda: HOSTS[0]))
        stack.enter_context(mock.patch.object(fcntl, "flock", flock))
        stack.enter_context(mock.patch.object(subprocess, "run", run))
        stack.enter_context(mock.patch.object(subprocess, "Popen", popen))
        stack.enter_context(mock.patch("sys.stdout", printed))
        try:
            code = launch.main(["--request", str(request_path), "--wall-seconds", "60", "--site", str(site_path)])
            outcome = f"returned {code}"
        except Exception as exc:  # the failure scenario's refusal
            outcome = f"{type(exc).__name__}: {exc}"
        finally:
            os.umask(umask)
    root = next(p for p in runs.iterdir() if p.is_dir())
    bundle_sha = sha256(host.staged[0]).hexdigest() if host.staged else ""  # gzip header carries a timestamp
    site_sha = _staged_site_sha(host.staged[0]) if host.staged else ""
    normalize = _normalizer({str(root): "<run>", PYTHON: "<python>", SITE: "<site>", pin: "<pin>",
                             EXAMPLE_COORDINATOR: "<coordinator>", str(base): "<tmp>", bundle_sha: "<bundle_sha256>",
                             site_sha: "<site_sha256>"})
    commands = {label: dict(_command(texts[0], normalize) or {}, same_on_all_hosts=len(set(texts)) == 1)
                for label, texts in sorted(host.commands.items())}
    worker_argv = host.workers[0] if host.workers else []
    record: dict[str, Any] = dict(
        outcome=outcome, source_identity_calls=identity_calls, locks=flocks, remote_commands=commands,
        staged_bundle=_bundle(host.staged[0], base) if host.staged else None,
        stage_payload_same_on_all_hosts=len(set(host.staged)) == 1,
        worker_command=_command(worker_argv[-1], normalize) if worker_argv else None,
        worker_ssh_prefix=[normalize(w) for w in worker_argv[:-1]],
        worker_process=dict(
            execv=[normalize(host.worker["execv"][0]), [normalize(a) for a in host.worker["execv"][1]]],
            chdir=normalize(host.worker.get("chdir", "")),
            environment={k: normalize(v) for k, v in host.worker["environment"].items()},
            start_marker_keys=host.worker["start_marker_keys"]) if host.worker else None,
        layout=_tree(root, normalize),
        stdout=[normalize(line) if not line.startswith("{") else "<summary json: "
                + ",".join(sorted(json.loads(line))) + ">" for line in printed.getvalue().splitlines()],
    )
    for name in ("controller_identity.json", "controller_terminal.json", "summary.json"):
        path = root / name
        if path.is_file():
            value_ = json.loads(path.read_text())
            record[name] = value_ if name == "controller_terminal.json" else sorted(value_)
    return record


def launcher_record() -> dict[str, Any]:
    """The real launcher ``main``: a successful sequential request, and a run whose rank-3 worker
    exits 1 (authenticated cleanup, idle-after, refusal)."""
    from glm_tpu.optimized import request
    from scripts.release import launch_ws32_optimized_request as launch
    from scripts.release import ws32_optimized_worker as worker

    value = request.from_token_ids([30, 31, 32], request_id="golden-launch", max_new_tokens=3)
    out: dict[str, Any] = {}
    for label, failing in (("success", None), ("worker_failure", 3)):
        with tempfile.TemporaryDirectory(prefix="glm-equivalence-launch-") as scratch:
            out[label] = _scenario(launch, worker, Path(scratch), value, failing_rank=failing)
    return out
