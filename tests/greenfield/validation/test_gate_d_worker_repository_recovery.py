from __future__ import annotations

import fcntl
import importlib.util
import io
import os
import shlex
import signal
import stat
import subprocess
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
CONTROLLER_PATH = ROOT / "scripts/greenfield/recover_gate_d_worker_repositories.py"
WORKER_PATH = ROOT / "scripts/greenfield/gate_d_worker_repository_transaction.py"
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh"
ORIGIN = "git@example.invalid:owner/repo.git"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CONTROLLER = _load(CONTROLLER_PATH, "gate_d_recovery_controller_test")
WORKER = _load(WORKER_PATH, "gate_d_recovery_worker_test")


def _worker_verifier() -> str:
    source = WRAPPER.read_text()
    marker = (
        "read -r -d '' WORKER_REPO_VERIFY_SCRIPT <<'WORKER_REPO_VERIFY_EOF' || true\n"
    )
    return source.split(marker, 1)[1].split("\nWORKER_REPO_VERIFY_EOF", 1)[0]


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "Gate D Test",
        "GIT_AUTHOR_EMAIL": "gate-d@example.invalid",
        "GIT_COMMITTER_NAME": "Gate D Test",
        "GIT_COMMITTER_EMAIL": "gate-d@example.invalid",
    }


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        env=_git_environment(),
        text=True,
    )


def _make_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "switch", "-q", "-c", "rewrite/topology-first-decode")
    (source / "state.txt").write_text("old\n")
    _git(source, "add", "state.txt")
    _git(source, "commit", "-q", "-m", "old")
    old_pin = _git(source, "rev-parse", "HEAD").stdout.strip()
    (source / "state.txt").write_text("new\n")
    _git(source, "commit", "-q", "-am", "new")
    pin = _git(source, "rev-parse", "HEAD").stdout.strip()

    canonical = tmp_path / "canonical"
    subprocess.run(
        ["/usr/bin/git", "clone", "-q", "--no-checkout", str(source), str(canonical)],
        check=True,
        env=_git_environment(),
    )
    _git(canonical, "checkout", "-q", "--detach", old_pin)
    _git(canonical, "remote", "set-url", "origin", ORIGIN)
    bundle_path = tmp_path / "source.bundle"
    _git(
        source,
        "bundle",
        "create",
        str(bundle_path),
        "refs/heads/rewrite/topology-first-decode",
    )
    bundle = bundle_path.read_bytes()
    bundle_path.unlink()

    monkeypatch.setattr(WORKER, "_PARENT", tmp_path)
    monkeypatch.setenv("GLM_GATE_D_REPO_RECOVERY_CARRIER", "test-carrier")
    WORKER._LOCK_FD = -1
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    common = {
        "pin": pin,
        "worktree": "canonical",
        "new": "prepared.new",
        "old": "preserved.old",
        "bundle": ".transport.bundle",
        "bundle_sha": sha256(bundle).hexdigest(),
        "bundle_bytes": len(bundle),
        "origin": ORIGIN,
        "verifier": _worker_verifier(),
        "old_pin": "",
        "old_dev": -1,
        "old_ino": -1,
        "new_dev": -1,
        "new_ino": -1,
    }
    preflight = WORKER.preflight(SimpleNamespace(**common), parent_fd)
    common.update(preflight)
    prepared = WORKER.receive_prepare(
        SimpleNamespace(**common), parent_fd, io.BytesIO(bundle)
    )
    common.update(prepared)
    return {
        "arguments": common,
        "bundle": bundle,
        "canonical": canonical,
        "old_pin": old_pin,
        "parent_fd": parent_fd,
        "pin": pin,
    }


def test_source_contract_is_descriptor_bound_timeout_and_terminal_last() -> None:
    controller = CONTROLLER_PATH.read_text()
    worker = WORKER_PATH.read_text()
    assert "controller must run from the reviewed /opt path" in controller
    assert "os.O_NOFOLLOW" in controller
    assert "os.O_EXCL" in controller
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in controller
    assert "remote recovery quiescence" in controller
    assert "no automatic fail-open deadline" in controller
    assert '"--kill-after=15"' in controller
    assert '"/opt/glm-tpu/locks"' in controller
    assert 'REMOTE_LOCK = "/opt/glm-tpu/locks/gate_d_repo_recovery.lock"' in controller
    assert (
        '"/usr/bin/python3",\n            "-I",\n            "-S",\n            "-B"'
        in controller
    )
    assert 'with open("/proc/locks"' in controller
    assert "psutil" not in controller[controller.index("def _quiescence_audit_code") :]
    assert "fuser" not in controller[controller.index("def _quiescence_audit_code") :]
    assert "gcloud" in controller
    assert "tpu-vm" in controller
    assert " scp" not in controller
    assert controller.index('label="preterminal"') < controller.index(
        'evidence.write("RECOVERY_COMPLETE"'
    )
    assert '"preterminal_objects": preterminal_objects' in controller
    assert controller.index("evidence.close_writes()") < controller.index(
        '_upload_fd(\n                terminal_fd, f"{remote_prefix}/RECOVERY_COMPLETE"'
    )
    assert (
        "if evidence is not None and not terminal_started and not evidence.closed"
        in controller
    )
    assert "_RENAME_EXCHANGE" in worker
    assert "_RENAME_NOREPLACE" in worker
    assert "_rollback_after_exchange" in worker
    assert "bundle_identity_drift" in worker
    assert '_LOCK_PARENT = Path("/opt/glm-tpu/locks")' in worker
    assert '_LOCK_NAME = "gate_d_repo_recovery.lock"' in worker

    wrapper = WRAPPER.read_text()
    assert 'root = "/opt/glm-tpu/locks"' in wrapper
    assert 'names = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")' in wrapper
    assert "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW" in wrapper
    assert "held.st_uid != 0" in wrapper and "held.st_gid != 0" in wrapper


def test_prepare_exchange_and_final_preserve_exact_old_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _make_transaction(tmp_path, monkeypatch)
    try:
        arguments = SimpleNamespace(**transaction["arguments"])
        swapped = WORKER.swap(arguments, transaction["parent_fd"])
        transaction["arguments"].update(swapped)
        verified = WORKER.final(
            SimpleNamespace(**transaction["arguments"]), transaction["parent_fd"]
        )
        assert verified["marker"] == "REPO_FINAL_OK"
        assert (
            _git(transaction["canonical"], "rev-parse", "HEAD").stdout.strip()
            == transaction["pin"]
        )
        preserved = tmp_path / "preserved.old"
        assert (
            _git(preserved, "rev-parse", "HEAD").stdout.strip()
            == transaction["old_pin"]
        )
        assert (transaction["canonical"] / "state.txt").read_text() == "new\n"
        assert (preserved / "state.txt").read_text() == "old\n"
        assert not (tmp_path / ".transport.bundle").exists()
    finally:
        os.close(transaction["parent_fd"])


def test_post_exchange_verifier_failure_rolls_back_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _make_transaction(tmp_path, monkeypatch)
    original_verify = WORKER._verify

    def injected_verify(verifier: str, pin: str, path: str, origin: str) -> None:
        if pin == transaction["pin"] and path == str(transaction["canonical"]):
            raise WORKER.TransactionError("injected_post_exchange_failure")
        original_verify(verifier, pin, path, origin)

    monkeypatch.setattr(WORKER, "_verify", injected_verify)
    try:
        with pytest.raises(
            WORKER.TransactionError, match="injected_post_exchange_failure"
        ):
            WORKER.swap(
                SimpleNamespace(**transaction["arguments"]), transaction["parent_fd"]
            )
        assert (
            _git(transaction["canonical"], "rev-parse", "HEAD").stdout.strip()
            == transaction["old_pin"]
        )
        assert (
            _git(tmp_path / "preserved.old", "rev-parse", "HEAD").stdout.strip()
            == transaction["pin"]
        )
    finally:
        os.close(transaction["parent_fd"])


def test_canonical_substitution_refuses_before_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _make_transaction(tmp_path, monkeypatch)
    displaced = tmp_path / "displaced"
    transaction["canonical"].rename(displaced)
    transaction["canonical"].mkdir()
    (transaction["canonical"] / "intruder.txt").write_text("unchanged\n")
    try:
        with pytest.raises(WORKER.TransactionError, match="canonical_prestate"):
            WORKER.swap(
                SimpleNamespace(**transaction["arguments"]), transaction["parent_fd"]
            )
        assert (transaction["canonical"] / "intruder.txt").read_text() == "unchanged\n"
        assert (
            _git(displaced, "rev-parse", "HEAD").stdout.strip()
            == transaction["old_pin"]
        )
        assert (
            _git(tmp_path / "preserved.old", "rev-parse", "HEAD").stdout.strip()
            == transaction["pin"]
        )
    finally:
        os.close(transaction["parent_fd"])


def test_bundle_receiver_refuses_raced_symlink_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _make_transaction(tmp_path, monkeypatch)
    target = tmp_path / "target"
    target.write_bytes(b"do-not-overwrite")
    (tmp_path / ".second.bundle").symlink_to(target)
    arguments = dict(transaction["arguments"])
    arguments.update(new="second.new", old="second.old", bundle=".second.bundle")
    try:
        with pytest.raises(WORKER.TransactionError, match="occupied_.second.bundle"):
            WORKER.receive_prepare(
                SimpleNamespace(**arguments),
                transaction["parent_fd"],
                io.BytesIO(transaction["bundle"]),
            )
        assert target.read_bytes() == b"do-not-overwrite"
    finally:
        os.close(transaction["parent_fd"])


def test_fresh_tag_can_resume_after_partial_fleet_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = _make_transaction(tmp_path, monkeypatch)
    try:
        swapped = WORKER.swap(
            SimpleNamespace(**transaction["arguments"]), transaction["parent_fd"]
        )
        second = dict(transaction["arguments"])
        second.update(
            new="next-tag.new",
            old="next-tag.old",
            bundle=".next-tag.bundle",
            old_pin="",
            old_dev=-1,
            old_ino=-1,
            new_dev=-1,
            new_ino=-1,
        )
        preflight = WORKER.preflight(
            SimpleNamespace(**second), transaction["parent_fd"]
        )
        assert preflight["old_pin"] == transaction["pin"]
        assert preflight["old_dev"] == swapped["new_dev"]
        assert preflight["old_ino"] == swapped["new_ino"]
    finally:
        os.close(transaction["parent_fd"])


def test_installed_source_authenticates_bytes_owner_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = tmp_path / "controller.py"
    raw = CONTROLLER_PATH.read_bytes()
    installed.write_bytes(raw)
    installed.chmod(0o555)
    monkeypatch.setattr(CONTROLLER, "INSTALLED_CONTROLLER", installed)

    assert (
        CONTROLLER._validate_installed_source(
            sha256(raw).hexdigest(),
            running_path=installed,
            required_uid=os.getuid(),
            required_gid=os.getgid(),
        )
        == raw
    )
    installed.chmod(0o755)
    with pytest.raises(CONTROLLER.RecoveryError, match="mode 0555"):
        CONTROLLER._validate_installed_source(
            sha256(raw).hexdigest(),
            running_path=installed,
            required_uid=os.getuid(),
            required_gid=os.getgid(),
        )
    installed.chmod(0o555)
    with pytest.raises(CONTROLLER.RecoveryError, match="hash mismatch"):
        CONTROLLER._validate_installed_source(
            "0" * 64,
            running_path=installed,
            required_uid=os.getuid(),
            required_gid=os.getgid(),
        )


def test_held_lock_refuses_pathname_replacement(tmp_path: Path) -> None:
    lock = tmp_path / "lease.lock"
    lock.touch(mode=0o600)
    fd = CONTROLLER._open_lock(
        tmp_path,
        lock.name,
        required_uid=os.getuid(),
        required_gid=os.getgid(),
        required_mode=0o600,
        required_parent_uid=os.getuid(),
        required_parent_gid=os.getgid(),
        forbid_parent_group_other_write=False,
    )
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        lock.unlink()
        lock.touch(mode=0o600)
        with pytest.raises(CONTROLLER.RecoveryError, match="pathname identity drift"):
            CONTROLLER._corroborate_lock(parent_fd, lock.name, fd)
    finally:
        os.close(parent_fd)
        os.close(fd)


def test_evidence_refuses_same_name_regular_file_substitution(tmp_path: Path) -> None:
    evidence = CONTROLLER.EvidenceDirectory("test-substitution", home=tmp_path)
    evidence.write("member.txt", b"reviewed bytes\n")
    member = tmp_path / "gate-d-repo-recovery" / "test-substitution" / "member.txt"
    member.unlink()
    member.write_bytes(b"substituted bytes\n")
    try:
        with pytest.raises(CONTROLLER.RecoveryError, match="pathname identity drift"):
            evidence.open_read("member.txt")
    finally:
        evidence.close()


def test_exact_object_set_rejects_unexpected_preterminal_object() -> None:
    expected = {"gs://bucket/prefix/a", "gs://bucket/prefix/b"}
    CONTROLLER._assert_exact_object_set(expected, expected, label="preterminal")
    with pytest.raises(CONTROLLER.RecoveryError, match="unexpected=.*intruder"):
        CONTROLLER._assert_exact_object_set(
            expected | {"gs://bucket/prefix/intruder"}, expected, label="preterminal"
        )


def test_privileged_quiescence_audit_observes_kernel_lock_state(tmp_path: Path) -> None:
    sudo = subprocess.run(
        ["/usr/bin/sudo", "-n", "true"], check=False, capture_output=True, text=True
    )
    if sudo.returncode:
        pytest.skip(
            "passwordless sudo is required for the production-equivalent /proc audit"
        )
    lock = tmp_path / "worker.lock"
    lock.touch(mode=0o666)
    subprocess.run(["/usr/bin/sudo", "-n", "chown", "0:0", str(lock)], check=True)
    subprocess.run(["/usr/bin/sudo", "-n", "chmod", "0666", str(lock)], check=True)
    environment = {
        "PREFIX": "unit-test-carrier",
        "LOCK": str(lock),
        "REQUIRED_UID": "0",
        "REQUIRED_GID": "0",
        "REQUIRED_MODE": "0666",
    }
    command = [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/env",
        "-i",
        *(f"{name}={value}" for name, value in environment.items()),
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        CONTROLLER._quiescence_audit_code(),
    ]
    clear = subprocess.run(command, check=False, capture_output=True, text=True)
    assert clear.returncode == 0, clear.stderr
    assert clear.stdout.startswith("RECOVERY_QUIESCENT ")

    fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = subprocess.run(command, check=False, capture_output=True, text=True)
        assert busy.returncode == 3, busy.stderr
        assert "holders=" in busy.stdout
    finally:
        os.close(fd)


def test_production_quiescence_launcher_ignores_hostile_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim_marker = tmp_path / "shim-ran"
    shim = tmp_path / "sudo"
    shim.write_text(f"#!/usr/bin/bash\n/usr/bin/touch {shim_marker}\nexit 0\n")
    shim.chmod(0o755)
    captured: list[str] = []

    def fake_run(command: list[str], **_kwargs):
        captured.extend(command)
        output = b"".join(
            f"RECOVERY_QUIESCENT host-{worker}\n".encode() for worker in range(8)
        )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(CONTROLLER, "_run", fake_run)
    assert (
        CONTROLLER._remote_quiescence_once("unit-tag").count(b"RECOVERY_QUIESCENT") == 8
    )
    remote_argument = next(item for item in captured if item.startswith("--command="))
    production_tokens = shlex.split(remote_argument.removeprefix("--command="))
    assert production_tokens[0] == "/usr/bin/sudo"
    assert production_tokens[2:5] == ["/usr/bin/env", "-i", "PREFIX=unit-tag"]

    sudo = subprocess.run(
        ["/usr/bin/sudo", "-n", "true"], check=False, capture_output=True, text=True
    )
    if sudo.returncode == 0:
        subprocess.run(
            production_tokens,
            check=False,
            capture_output=True,
            env={"PATH": str(tmp_path)},
            text=True,
        )
        assert not shim_marker.exists()


def test_quiescence_wait_ignores_signals_until_audit_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def audit(_tag: str) -> bytes:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise CONTROLLER.RecoveryError("not yet")
        return b"eight-host proof"

    def no_wait(_seconds: float) -> None:
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN

    monkeypatch.setattr(CONTROLLER, "_remote_quiescence_once", audit)
    monkeypatch.setattr(CONTROLLER.time, "sleep", no_wait)
    assert CONTROLLER._wait_for_remote_quiescence("unit-tag") == b"eight-host proof"
    assert attempts == 2
    for signum, handler in previous.items():
        assert signal.getsignal(signum) == handler


def test_lock_symlinks_and_post_terminal_writes_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.write_bytes(b"unchanged")
    (tmp_path / "unsafe.lock").symlink_to(target)
    with pytest.raises(OSError):
        CONTROLLER._open_lock(
            tmp_path,
            "unsafe.lock",
            required_uid=os.getuid(),
            required_gid=os.getgid(),
            required_mode=stat.S_IMODE(target.stat().st_mode),
            required_parent_uid=os.getuid(),
            required_parent_gid=os.getgid(),
            forbid_parent_group_other_write=False,
        )

    evidence = CONTROLLER.EvidenceDirectory("test-terminal", home=tmp_path)
    evidence.write("RECOVERY_COMPLETE", b"terminal\n")
    evidence.close_writes()
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    with pytest.raises(CONTROLLER.RecoveryError, match="writes are closed"):
        evidence.write("FAILURE.json", b"forbidden\n")
    evidence.log("read-only post-terminal message")
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert target.read_bytes() == b"unchanged"
    evidence.close()
