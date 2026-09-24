"""The resident controller's stop and failure paths against a fake eight-host fleet (DESIGN 6.8, H1).

The real launcher ``main`` runs with ``--keep-loaded``. Faked are only the SSH hosts (every
``subprocess.run``/``Popen`` of an SSH command is answered by :class:`FakeFleet`: the helper is
identified by its exact file text, its JSON argument is decoded, and each host keeps its own
files), ``gcloud`` discovery, the hostname, the launch policy (no network) and the clock's sleep,
which drives the private inbox. The helper snapshot is the real one: the synthetic checkout
commits the helper files, and the controller reads its package from a copy that a test may edit
while the run is live. Rank 0's files live in the real run directory, as on the
controller host. Regressions for the 181c013e defect: after ``stop.json`` the final collection
re-fetched ``runner.rank{1..7}.json`` that sequence 0 had already stored and raised
``FileExistsError`` before ``summary.json``. The final collection is per host: a host whose fetch
fails is listed (``uncollected_ranks``) without losing the other hosts' records or the resident
failure, and a local SSH client that outlives the verified-idle hosts is ended after a bounded wait.
"""
from __future__ import annotations

import base64
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import pytest

from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.executor import fleet as remote
from glm_tpu.executor.remote import HELPERS
from glm_tpu.engine import request
from glm_tpu.engine._s3_user_request import canonical
from glm_tpu.executor import multihost_executor as launch
from glm_tpu.worker import tpu_worker as worker
from tests.fixtures.site import example_mapping, write_example_site

FAKE_SSH = "glm-test-ssh"
HOSTS = [f"example-w-{rank}" for rank in range(8)]
GIT_ENV = dict(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="fixture",
               GIT_AUTHOR_EMAIL="fixture@example.invalid", GIT_COMMITTER_NAME="fixture",
               GIT_COMMITTER_EMAIL="fixture@example.invalid")


def persisted(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class FakeProcess:
    def __init__(self, fleet: FakeFleet, rank: int) -> None:
        self.fleet, self.rank, self.code = fleet, rank, None
        self.stdin = self
        self.killed = False

    def write(self, line: bytes) -> None:
        self.fleet.deliver(self.rank, line)

    def flush(self) -> None:
        pass

    def poll(self):
        return self.code

    def wait(self):
        assert self.code is not None, f"rank {self.rank} never ended"
        return self.code

    def kill(self) -> None:  # the local SSH client (the host is already verified idle)
        self.killed = True
        self.code = -9


class LauncherClock:
    """The launcher module's ``time`` with only its own sleeps faked (``launch.time`` is replaced,
    not ``time.sleep``). ``subprocess`` polls with ``time.sleep`` while a real child it waits on
    with a timeout exits (``git cat-file`` of the pinned helpers, ``git archive``); a global patch
    handed those polls to the fake fleet, which under load then looked for the run directory
    before ``main`` had created it."""

    def __init__(self, sleep) -> None:
        self.sleep = sleep

    def __getattr__(self, name: str):
        return getattr(time, name)


class FakeFleet:
    """Eight hosts: helper commands by their exact text, per-host files, resident workers."""

    def __init__(self, runs: Path, pin: str, *, fail_rank: int | None = None, rewrite_equal_rank: int | None = None,
                 collect_failures: dict[int, tuple[int, bytes]] | None = None, stalled_rank: int | None = None,
                 edit_package: Path | None = None):
        self.runs, self.pin = runs, pin
        self.fail_rank, self.rewrite_equal_rank = fail_rank, rewrite_equal_rank
        # rank -> (exit code, output) of that host's *final* fetch; a stalled rank's SSH client
        # outlives its (cleaned-up) worker until the controller ends it
        self.collect_failures, self.stalled_rank = collect_failures or {}, stalled_rank
        self.files: dict[int, dict[str, bytes]] = {rank: {} for rank in range(8)}  # ranks 1..7
        self.processes: list[FakeProcess] = []
        self.helpers = {remote.helper_text(name): name for name in HELPERS}
        # a package directory edited at the first supervision sleep (after main has started)
        self.edit_package, self.edited_at, self.edited_record = edit_package, None, None
        self.unknown_texts: list[tuple[str, int]] = []  # helper texts that are not the launch-time ones
        self.calls: list[tuple[str, int, dict]] = []
        self.value: dict = {}
        self.pending: list[bytes] = []
        self.inbox_step = 0
        self.ticks = 0

    # ---------------------------------------------------------------- host files
    @property
    def root(self) -> Path:
        return next(p for p in self.runs.iterdir() if p.is_dir())

    def put(self, rank: int, relative: str, data: bytes) -> None:
        if rank == 0:
            path = self.root / relative
            path.parent.mkdir(mode=0o700, exist_ok=True)
            path.write_bytes(data)
        else:
            self.files[rank][relative] = data

    def get(self, rank: int, relative: str) -> bytes | None:
        if rank == 0:
            path = self.root / relative
            return path.read_bytes() if path.is_file() else None
        return self.files[rank].get(relative)

    def record(self, rank: int, value: dict, **changes) -> dict:
        return dict(schema="glm_optimized_worker_v1", rank=rank, hostname=HOSTS[rank], code_hash=self.pin,
                    request_sha256=value["request_sha256"], complete=True,
                    request=dict(token_sha256=sha256(canonical(value)).hexdigest(), emitted=2),
                    programs=dict(decode=dict(stablehlo_sha256="d" * 64, optimized_hlo_sha256="e" * 64)), **changes)

    # ---------------------------------------------------------------- SSH
    def run(self, argv: list[str], stream, payload: bytes | None) -> int:
        rank = HOSTS.index(argv[1])
        words = shlex.split(argv[-1])
        if words[0] == "cd":
            stream.write(json.dumps(dict(hostname=HOSTS[rank], jax="0.10.1", jaxlib="0.10.1")).encode())
            self.calls.append(("preflight", rank, {}))
            return 0
        assert words[1] == "-c" and len(words) == 4, words[:2]
        if words[2] not in self.helpers:  # a host would refuse a helper whose arguments differ
            self.unknown_texts.append((Path(stream.name).name, rank))
            return 97
        helper, args = self.helpers[words[2]], json.loads(words[3])
        self.calls.append((helper, rank, args))
        if helper == "idle_probe":
            stream.write(f"IDLE {HOSTS[rank]}\n".encode())
        elif helper == "cleanup":
            assert args["pin"] == self.pin and args["module"] == protocol.WORKER_MODULE
            process = self.processes[rank]
            if process.code is None and rank != self.stalled_rank:
                process.code = -9
        elif helper == "fetch":
            if Path(stream.name).name.startswith("collect.") and rank in self.collect_failures:
                code, output = self.collect_failures[rank]
                stream.write(output)
                return code
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
        assert args["hosts"] == HOSTS and args["module"] == protocol.WORKER_MODULE and "--keep-loaded" in args["argv"]
        process = FakeProcess(self, rank)
        self.processes.append(process)
        self.put(rank, protocol.worker_started_file(rank), json.dumps(dict(pid=4000 + rank, hostname=HOSTS[rank],
                                                                           code_hash=self.pin)).encode())
        self.put(rank, protocol.runner_file(rank), persisted(self.record(rank, self.value)))
        if rank == 7:
            worker.persist(self.root / protocol.READY_FILE, dict(sequence=0))
        return process

    # ---------------------------------------------------------------- resident workers
    def deliver(self, rank: int, line: bytes) -> None:
        if rank != 7:
            return
        command = json.loads(line)
        if command == {"stop": True}:
            for process in self.processes:
                if process.code is None:
                    process.code = 0
            if self.rewrite_equal_rank is not None:  # same JSON value, other bytes
                rank_ = self.rewrite_equal_rank
                value = json.loads(self.get(rank_, protocol.runner_file(rank_)))
                self.put(rank_, protocol.runner_file(rank_), json.dumps(value, separators=(",", ":")).encode())
            return
        sequence, value = command["sequence"], command["request"]
        job = f"resident-{sequence:04d}"
        (self.root / job).mkdir(mode=0o700)
        for rank_ in range(8):
            if rank_ == self.fail_rank:
                # the worker's except branch: its root record gains failure_type, exit 1
                failed = self.record(rank_, self.value, failure_type="RuntimeError")
                self.put(rank_, protocol.runner_file(rank_), persisted(failed))
                self.processes[rank_].code = 1
                continue
            self.put(rank_, f"{job}/{protocol.runner_file(rank_)}",
                     persisted(self.record(rank_, value, resident_sequence=sequence)))
        if self.fail_rank is None:
            worker.persist(self.root / protocol.READY_FILE, dict(sequence=sequence))

    def sleep(self, seconds: float) -> None:
        """The controller's supervision sleep: publishes inbox 0001, then stop, like an operator."""
        self.ticks += 1
        if self.ticks > 500:
            raise RuntimeError("fake fleet: the controller did not finish")
        if self.edit_package is not None and self.edited_at is None:
            for name in HELPERS:  # e.g. a later refactor stage or a branch switch in the checkout
                path = self.edit_package / f"{name}.py"
                path.write_text(path.read_text() + "# edited while the controller runs\n")
            self.edited_at, self.edited_record = len(self.calls), remote.helpers_record()
        root = self.root
        inbox = root / protocol.INBOX_DIR
        if self.inbox_step == 0 and (root / protocol.MEASUREMENT_FILE).exists():
            path = inbox / protocol.inbox_name(1)
            path.write_bytes(canonical(request.from_token_ids([8, 9], request_id="fixture-r1", max_new_tokens=2)))
            path.chmod(0o600)
            self.inbox_step = 1
        elif self.inbox_step == 1 and (root / "resident-0001" / protocol.MEASUREMENT_FILE).exists():
            path = inbox / protocol.STOP_FILE
            path.write_text('{"stop":true}')
            path.chmod(0o600)
            self.inbox_step = 2


@pytest.fixture
def resident(tmp_path, monkeypatch):
    """A synthetic site, source checkout and request; returns ``run(fleet_kwargs) -> (fleet, outcome)``."""
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)
    repo = tmp_path / "repo"
    (repo / "glm_tpu" / "worker").mkdir(parents=True)
    (repo / "glm_tpu" / "worker" / "tpu_worker.py").write_text("# synthetic worker\n")
    # The pinned commit holds the helper files; the controller's package is a copy of them.
    package = tmp_path / "package"
    package.mkdir()
    for name in HELPERS:
        (repo / remote.helper_path(name)).parent.mkdir(parents=True, exist_ok=True)
        (repo / remote.helper_path(name)).write_text(remote.helper_text(name))
        (package / f"{name}.py").write_text(remote.helper_text(name))
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "x"]):
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
    rebinding = json.dumps(dict(capture_sha256=digests)).encode()
    (binding / "topology_rebinding.json").write_bytes(rebinding)
    runs, locks = tmp_path / "runs", tmp_path / "locks"
    runs.mkdir(mode=0o700)
    locks.mkdir()
    site = write_example_site(tmp_path / "site.toml", example_mapping(
        tmp_path, paths=dict(run_root=str(runs)),
        topology=dict(binding_dir=str(binding), binding_sha256=sha256(rebinding).hexdigest())))
    value = request.from_token_ids([7], request_id="fixture-r0", max_new_tokens=2)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical(value) + b"\n")
    request_path.chmod(0o600)

    real_run, real_popen = subprocess.run, subprocess.Popen

    def run(client_exit_seconds=None, edit_helpers=False, **fleet_kwargs):
        monkeypatch.setattr(remote, "_package_file", lambda name: package / f"{name}.py")
        fleet = FakeFleet(runs, pin, edit_package=package if edit_helpers else None, **fleet_kwargs)
        fleet.value = value

        def fake_run(argv, *args, **kwargs):
            if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
                return subprocess.CompletedProcess(argv, fleet.run(argv, kwargs["stdout"], kwargs.get("input")))
            return real_run(argv, *args, **kwargs)

        def fake_popen(argv, *args, **kwargs):
            if isinstance(argv, list) and argv and argv[0] == FAKE_SSH:
                return fleet.popen(argv)
            return real_popen(argv, *args, **kwargs)

        monkeypatch.setattr(launch, "REPO", repo)
        monkeypatch.setattr(launch, "source_identity", lambda path, policy: pin)
        monkeypatch.setattr(launch, "ssh_commands", lambda fleet_: [[FAKE_SSH, host, "--", "true"] for host in HOSTS])
        monkeypatch.setattr(launch.socket, "gethostname", lambda: HOSTS[0])
        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        monkeypatch.setattr(launch.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(launch, "time", LauncherClock(fleet.sleep))
        if client_exit_seconds is not None:
            monkeypatch.setattr(launch, "SSH_CLIENT_EXIT_SECONDS", client_exit_seconds)
        printed = io.StringIO()
        monkeypatch.setattr("sys.stdout", printed)
        umask = os.umask(0o077)
        try:
            outcome = launch.main(["--request", str(request_path), "--site", str(site), "--wall-seconds", "60",
                                   "--keep-loaded"])
        except Exception as exc:  # noqa: BLE001 -- the outcome is what the test asserts
            outcome = exc
        finally:
            os.umask(umask)
            monkeypatch.undo()
        return fleet, outcome, printed.getvalue()

    return run


def _terminal(root: Path) -> dict:
    return json.loads((root / protocol.CONTROLLER_TERMINAL_FILE).read_text())


def test_a_resident_stop_collects_once_and_writes_the_summary(resident):
    fleet, outcome, printed = resident()
    root = fleet.root
    assert outcome == 0, repr(outcome)
    terminal = _terminal(root)
    assert terminal == dict(codes=[0] * 8, all_hosts_idle=True, collected=True, divergent_records=[],
                            uncollected_ranks=[], stalled_ssh_clients=[], failure=None, collect_error=None)
    assert not (root / "final").exists()
    summary = json.loads((root / protocol.SUMMARY_FILE).read_text())
    assert summary["passed"] is True and summary["all_hosts_idle_after"] is True
    assert summary["request"]["emitted"] == 2
    for rank in range(1, 8):  # the sequence-0 receipt and the final collection agree; nothing replaced
        assert (root / protocol.runner_file(rank)).read_bytes() == fleet.files[rank][protocol.runner_file(rank)]
        assert (root / protocol.worker_started_file(rank)).read_bytes() == \
            fleet.files[rank][protocol.worker_started_file(rank)]
    lines = printed.splitlines()
    assert lines[0].startswith(protocol.STDOUT_RUN)
    assert sum(line.startswith(protocol.STDOUT_RESIDENT_RESULT) for line in lines) == 2
    assert json.loads(lines[-1]) == summary
    assert [c[0] for c in fleet.calls if c[0] == "cleanup"] == []  # a clean stop needs no kill
    stops = [call for call in fleet.calls if call[0] == "idle_probe" and call[2]["hosts"] == HOSTS]
    assert len(stops) == 8  # idle_after on every host, with the authenticated list


def test_a_json_equal_but_byte_different_record_is_skipped(resident):
    fleet, outcome, _ = resident(rewrite_equal_rank=5)
    root = fleet.root
    assert outcome == 0, repr(outcome)
    assert _terminal(root)["divergent_records"] == [] and not (root / "final").exists()
    stored = (root / protocol.runner_file(5)).read_bytes()
    assert stored != fleet.files[5][protocol.runner_file(5)]           # other bytes on the host
    assert json.loads(stored) == json.loads(fleet.files[5][protocol.runner_file(5)])  # same record
    assert (root / protocol.SUMMARY_FILE).is_file()


def test_a_worker_failure_after_sequence_0_is_preserved_in_final_and_reported(resident):
    fleet, outcome, printed = resident(fail_rank=3)
    root = fleet.root
    assert isinstance(outcome, ValueError) and not isinstance(outcome, FileExistsError), repr(outcome)
    assert str(outcome) == "resident worker exited unexpectedly"
    terminal = _terminal(root)
    assert terminal["divergent_records"] == ["runner.rank3.json"] and terminal["collected"] is True
    assert terminal["failure"] == dict(type="ValueError", message="resident worker exited unexpectedly")
    assert terminal["uncollected_ranks"] == [] and terminal["collect_error"] is None
    assert terminal["stalled_ssh_clients"] == []
    assert terminal["codes"][3] == 1 and all(code == -9 for rank, code in enumerate(terminal["codes"]) if rank != 3)
    final = json.loads((root / "final" / "runner.rank3.json").read_text())
    assert final["failure_type"] == "RuntimeError"
    assert "failure_type" not in json.loads((root / protocol.runner_file(3)).read_text())  # receipt untouched
    assert not (root / protocol.SUMMARY_FILE).exists()
    cleaned = sorted(rank for helper, rank, _ in fleet.calls if helper == "cleanup")
    assert cleaned == list(range(8))  # authenticated cleanup on every host before collection
    assert "Cleanup unresolved" not in printed


def test_a_failed_final_fetch_on_one_host_keeps_every_other_record(resident):
    # A clean stop whose final fetch fails on rank 4 (ssh exit 255): at 181c013e and until H1
    # was per host, the whole collection raised before any record was written.
    fleet, outcome, _ = resident(collect_failures={4: (255, b"ssh: connection closed\n")})
    root = fleet.root
    assert isinstance(outcome, ValueError) and str(outcome) == \
        "collect failed on one or more hosts; see private originals", repr(outcome)
    terminal = _terminal(root)
    assert terminal["codes"] == [0] * 8 and terminal["collected"] is False
    assert terminal["uncollected_ranks"] == [4] and terminal["collect_error"] is None and terminal["failure"] is None
    for rank in (1, 2, 3, 5, 6, 7):  # every other host's start marker arrived and was written
        assert (root / protocol.worker_started_file(rank)).read_bytes() == \
            fleet.files[rank][protocol.worker_started_file(rank)]
    assert not (root / protocol.worker_started_file(4)).exists()
    assert (root / protocol.runner_file(4)).is_file()  # the sequence-0 receipt is untouched
    assert not (root / protocol.SUMMARY_FILE).exists()


@pytest.mark.parametrize("output", [b"not json", b'["list"]', b'{"runner.rank2.json": 5}',
                                    b'{"runner.rank2.json": "***"}', b'{"../escape.json": "e30="}',
                                    b'{"runner.rank3.json": "e30="}'])
def test_unreadable_or_unexpected_fetch_output_is_that_host_uncollected(resident, output):
    fleet, outcome, _ = resident(collect_failures={2: (0, output)})
    root = fleet.root
    assert isinstance(outcome, ValueError) and "collect failed" in str(outcome), repr(outcome)
    terminal = _terminal(root)
    assert terminal["uncollected_ranks"] == [2] and terminal["collected"] is False
    assert terminal["divergent_records"] == [] and not (root / "final").exists()
    assert not (root.parent / "escape.json").exists() and not (root / "escape.json").exists()
    assert (root / protocol.worker_started_file(1)).is_file()


def test_a_resident_failure_is_not_replaced_by_a_failed_collection(resident):
    fleet, outcome, _ = resident(fail_rank=3, collect_failures={6: (255, b"")})
    root = fleet.root
    assert isinstance(outcome, ValueError) and str(outcome) == "resident worker exited unexpectedly", repr(outcome)
    terminal = _terminal(root)
    assert terminal["failure"] == dict(type="ValueError", message="resident worker exited unexpectedly")
    assert terminal["uncollected_ranks"] == [6] and terminal["collected"] is False
    assert terminal["divergent_records"] == ["runner.rank3.json"]  # rank 3's failure record still arrived
    assert json.loads((root / "final" / "runner.rank3.json").read_text())["failure_type"] == "RuntimeError"


def test_a_stalled_local_ssh_client_is_ended_after_the_bounded_wait(resident):
    # rank 5's worker is cleaned up on the host, but its local SSH client never exits: the
    # controller ends it once idle_after has verified every host, instead of waiting forever.
    fleet, outcome, printed = resident(fail_rank=3, stalled_rank=5, client_exit_seconds=0)
    root = fleet.root
    assert isinstance(outcome, ValueError) and str(outcome) == "resident worker exited unexpectedly", repr(outcome)
    terminal = _terminal(root)
    assert terminal["stalled_ssh_clients"] == [5] and terminal["codes"][5] == -9
    assert [p.rank for p in fleet.processes if p.killed] == [5]
    assert terminal["collected"] is True and terminal["divergent_records"] == ["runner.rank3.json"]
    assert "Cleanup unresolved" not in printed


@pytest.mark.parametrize("fail_rank", [None, 3])
def test_an_edit_of_the_helper_files_during_the_run_changes_nothing_that_is_sent(resident, fail_rank):
    # The package files change at the first supervision sleep, after idle_before, staging, the
    # dispatch and the sequence-0 collection. Every later command -- the resident and final
    # fetches, cleanup after a failure, idle_after -- still carries the launch-time (pinned)
    # texts that helpers.json records; a changed text would make the host refuse it.
    fleet, outcome, printed = resident(edit_helpers=True, fail_rank=fail_rank)
    root = fleet.root
    assert fleet.edited_at is not None and fleet.unknown_texts == []
    after = [call[0] for call in fleet.calls[fleet.edited_at:]]
    expected = {"fetch", "idle_probe"} | ({"cleanup"} if fail_rank is not None else set())
    assert expected <= set(after), after
    assert after.count("idle_probe") == 8  # idle_after on every host
    launch_time = remote.HelperTexts({name: text for text, name in fleet.helpers.items()}).record()
    assert json.loads((root / protocol.HELPERS_FILE).read_text()) == launch_time
    edited = fleet.edited_record["helpers"]
    assert all(edited[name]["sha256"] != launch_time["helpers"][name]["sha256"] for name in HELPERS)
    if fail_rank is None:
        assert outcome == 0, repr(outcome)
        assert json.loads((root / protocol.SUMMARY_FILE).read_text())["passed"] is True
    else:
        assert str(outcome) == "resident worker exited unexpectedly", repr(outcome)
        assert _terminal(root)["collected"] is True
    assert "Cleanup unresolved" not in printed
