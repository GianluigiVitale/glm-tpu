from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True

ROOT = Path(__file__).parents[3]
CONTROLLER_PATH = (
    ROOT / "scripts/greenfield/gate_d_repository_prerequisites/"
    "provision_gate_d_repository_prerequisites.py"
)
ROOT_HELPER_PATH = (
    ROOT / "scripts/greenfield/gate_d_repository_prerequisites/"
    "gate_d_repository_prerequisite_root.py"
)
RECOVERY_PATH = ROOT / "scripts/greenfield/recover_gate_d_worker_repositories.py"


def _load(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


ROOT_HELPER = _load(ROOT_HELPER_PATH, "gate_d_prerequisite_root_test")
CONTROLLER = _load(CONTROLLER_PATH, "gate_d_prerequisite_controller_test")


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    opt = tmp_path / "opt"
    opt.mkdir(mode=0o755)
    opt.chmod(0o755)
    return opt, opt / "glm-tpu", opt / "glm-tpu" / "locks"


def test_lock_provision_is_idempotent_exact_and_no_replace(tmp_path: Path) -> None:
    opt, glm_root, lock_root = _roots(tmp_path)
    names = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")
    first = ROOT_HELPER.ensure_locks(
        opt,
        glm_root,
        lock_root,
        names,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    second = ROOT_HELPER.ensure_locks(
        opt,
        glm_root,
        lock_root,
        names,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    assert [(item["name"], item["ino"]) for item in first] == [
        (item["name"], item["ino"]) for item in second
    ]
    for name in names:
        value = (lock_root / name).lstat()
        assert stat.S_ISREG(value.st_mode)
        assert stat.S_IMODE(value.st_mode) == 0o666
        assert value.st_nlink == 1 and value.st_size == 0

    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    (lock_root / "gate_d_repo_recovery.lock").symlink_to(target)
    with pytest.raises(OSError):
        ROOT_HELPER.ensure_locks(
            opt,
            glm_root,
            lock_root,
            ("gate_d_repo_recovery.lock",),
            uid=os.getuid(),
            gid=os.getgid(),
        )
    assert target.read_bytes() == b"unchanged"


def test_existing_wrong_lock_identity_refuses_without_repair(tmp_path: Path) -> None:
    opt, glm_root, lock_root = _roots(tmp_path)
    glm_root.mkdir(mode=0o755)
    glm_root.chmod(0o755)
    lock_root.mkdir(mode=0o755)
    lock_root.chmod(0o755)
    lock = lock_root / "gate_d_repo_recovery.lock"
    lock.touch(mode=0o600)
    with pytest.raises(ROOT_HELPER.PrerequisiteError, match="unsafe lock identity"):
        ROOT_HELPER.ensure_locks(
            opt,
            glm_root,
            lock_root,
            (lock.name,),
            uid=os.getuid(),
            gid=os.getgid(),
        )
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_controller_install_is_atomic_idempotent_and_never_replaces(
    tmp_path: Path,
) -> None:
    bin_root = tmp_path / "bin"
    bin_root.mkdir(mode=0o755)
    bin_root.chmod(0o755)
    raw = RECOVERY_PATH.read_bytes()
    digest = CONTROLLER._sha256(raw)
    assert digest == ROOT_HELPER.CONTROLLER_SHA256
    assert len(raw) == ROOT_HELPER.CONTROLLER_BYTES
    first = ROOT_HELPER.install_controller(
        bin_root,
        raw,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    second = ROOT_HELPER.install_controller(
        bin_root,
        raw,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    target = bin_root / ROOT_HELPER.CONTROLLER_NAME
    assert first["ino"] == second["ino"] == target.stat().st_ino
    assert target.read_bytes() == raw
    assert stat.S_IMODE(target.stat().st_mode) == 0o555

    replacement = b"different reviewed-looking bytes\n"
    with pytest.raises(ROOT_HELPER.PrerequisiteError, match="identity mismatch"):
        ROOT_HELPER.install_controller(
            bin_root,
            replacement,
            uid=os.getuid(),
            gid=os.getgid(),
        )
    assert target.read_bytes() == raw


def test_exact_crontab_transition_and_unknown_state_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = CONTROLLER.OLD_CRONTAB.encode()

    def fake_run(
        command: list[str],
        *,
        input_bytes: bytes | None = None,
        **_kwargs,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal state
        if command == ["/usr/bin/crontab", "-"]:
            assert input_bytes is not None
            state = input_bytes
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")
        assert command == ["/usr/bin/crontab", "-l"]
        return subprocess.CompletedProcess(command, 0, stdout=state, stderr=b"")

    monkeypatch.setattr(CONTROLLER, "_run", fake_run)
    monkeypatch.setattr(CONTROLLER, "_revalidate_locks", lambda _locks: None)
    before, after, disposition = CONTROLLER._set_crontab(())
    assert before == CONTROLLER.OLD_CRONTAB
    assert after == CONTROLLER.NEW_CRONTAB
    assert disposition == "updated"
    assert CONTROLLER._set_crontab(())[2] == "existing"

    state = b"@reboot /unreviewed\n"
    with pytest.raises(CONTROLLER.ProvisionError, match="exact reviewed old/new"):
        CONTROLLER._set_crontab(())
    assert state == b"@reboot /unreviewed\n"


def test_production_root_boundary_and_scope_are_exact() -> None:
    command = CONTROLLER._root_command(
        "reviewed-source", "gate_d_repo_prerequisite_unit", ["locks", "--lock", "x"]
    )
    assert command[:4] == ["/usr/bin/sudo", "-n", "/usr/bin/env", "-i"]
    assert command[8:14] == [
        "/usr/bin/python3",
        "-I",
        "-S",
        "-B",
        "-c",
        "reviewed-source",
    ]
    root_source = ROOT_HELPER_PATH.read_text()
    controller_source = CONTROLLER_PATH.read_text()
    assert "import jax" not in root_source + controller_source
    assert "libtpu" not in root_source + controller_source
    assert "checkout" not in root_source + controller_source
    assert "--expected-sha256" not in root_source
    assert CONTROLLER.INSTALLED_ROOT == Path(
        "/opt/glm-tpu/gate-d-repository-prerequisites-v2"
    )
    assert str(CONTROLLER.INSTALLED_ROOT) in controller_source
    assert ROOT_HELPER.CONTROLLER_SHA256 == CONTROLLER.RECOVERY_CONTROLLER_SHA256
    assert ROOT_HELPER.CONTROLLER_BYTES == CONTROLLER.RECOVERY_CONTROLLER_BYTES
    assert "for worker in range(8):" in controller_source
    assert controller_source.index("cron_before, cron_after") < controller_source.index(
        "_write_evidence(tag, evidence)"
    )


def test_privileged_boundary_rejects_missing_bytecode_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flags = SimpleNamespace(isolated=1, no_site=1, ignore_environment=1)
    fake_sys = SimpleNamespace(
        dont_write_bytecode=False,
        executable="/usr/bin/python3",
        flags=flags,
    )
    monkeypatch.setattr(ROOT_HELPER, "sys", fake_sys)
    monkeypatch.setattr(ROOT_HELPER.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        ROOT_HELPER.os,
        "environ",
        {
            "GLM_GATE_D_PREREQUISITE_CARRIER": (
                "gate_d_repo_prerequisite_20260901T010203123456789Z"
            ),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    with pytest.raises(
        ROOT_HELPER.PrerequisiteError, match="invalid privileged execution boundary"
    ):
        ROOT_HELPER._validate_privileged_boundary()
    fake_sys.dont_write_bytecode = True
    ROOT_HELPER._validate_privileged_boundary()


def test_remote_branch_pin_accepts_exact_git_oid_and_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oid = "f" * 40
    expected_ref = f"refs/heads/{CONTROLLER.BRANCH}"

    def fake_run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[bytes]:
        assert command[-2:] == [CONTROLLER.ORIGIN, expected_ref]
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{oid}\t{expected_ref}\n".encode(), stderr=b""
        )

    monkeypatch.setattr(CONTROLLER, "_run", fake_run)
    assert CONTROLLER._remote_branch_pin() == oid


@pytest.mark.parametrize(
    "raw",
    [
        f"{'f' * 64}\trefs/heads/{CONTROLLER.BRANCH}\n".encode(),
        f"{'f' * 40}\trefs/heads/wrong-branch\n".encode(),
        f"{'f' * 40} refs/heads/{CONTROLLER.BRANCH}\n".encode(),
        f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}".encode(),
        f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}\r\n".encode(),
        f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}\r".encode(),
        f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}\v".encode(),
        f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}\f".encode(),
        (
            f"{'f' * 40}\trefs/heads/{CONTROLLER.BRANCH}\n"
            f"{'e' * 40}\trefs/heads/{CONTROLLER.BRANCH}\n"
        ).encode(),
        b"\xff\n",
        b"",
    ],
)
def test_remote_branch_pin_rejects_sha256_wrong_ref_and_malformed_output(
    monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    monkeypatch.setattr(
        CONTROLLER,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, stdout=raw, stderr=b""
        ),
    )
    with pytest.raises(
        CONTROLLER.ProvisionError,
        match=rf"invalid origin pin response bytes={len(raw)} sha256=[0-9a-f]{{64}}",
    ):
        CONTROLLER._remote_branch_pin()


def test_privileged_receipts_fail_closed_on_semantic_drift() -> None:
    lock = {
        "dev": 1,
        "gid": 0,
        "ino": 2,
        "mode": "0666",
        "name": "gate_d_repo_recovery.lock",
        "nlink": 1,
        "size": 0,
        "uid": 0,
    }
    receipt = {
        "host": "worker-0",
        "locks": [lock],
        "marker": "PREREQUISITE_LOCKS_OK",
    }
    assert (
        CONTROLLER._validate_lock_receipt(receipt, ("gate_d_repo_recovery.lock",))
        == "worker-0"
    )
    lock["mode"] = "0777"
    with pytest.raises(CONTROLLER.ProvisionError, match="unsafe lock record"):
        CONTROLLER._validate_lock_receipt(receipt, ("gate_d_repo_recovery.lock",))

    controller = {
        "bytes": 10,
        "dev": 1,
        "gid": 0,
        "ino": 2,
        "mode": "0555",
        "name": "recover_gate_d_worker_repositories.py",
        "sha256": "a" * 64,
        "uid": 0,
    }
    controller_receipt = {
        "controller": controller,
        "host": "worker-0",
        "marker": "PREREQUISITE_CONTROLLER_OK",
    }
    assert (
        CONTROLLER._validate_controller_receipt(
            controller_receipt, expected_sha="a" * 64, expected_bytes=10
        )
        == "worker-0"
    )
    controller["uid"] = False
    with pytest.raises(CONTROLLER.ProvisionError, match="unsafe controller record"):
        CONTROLLER._validate_controller_receipt(
            controller_receipt, expected_sha="a" * 64, expected_bytes=10
        )


def test_fleet_receipts_require_eight_unique_hosts_and_local_worker_zero() -> None:
    records = {}
    for worker in range(8):
        records[worker] = {
            "host": f"worker-{worker}",
            "locks": [
                {
                    "dev": 1,
                    "gid": 0,
                    "ino": worker + 10,
                    "mode": "0666",
                    "name": "gate_d_repo_recovery.lock",
                    "nlink": 1,
                    "size": 0,
                    "uid": 0,
                }
            ],
            "marker": "PREREQUISITE_LOCKS_OK",
        }
    CONTROLLER._validate_fleet_receipts(records, "worker-0")
    records[7]["host"] = "worker-6"
    with pytest.raises(CONTROLLER.ProvisionError, match="authenticated 8/8"):
        CONTROLLER._validate_fleet_receipts(records, "worker-0")
    records[7]["host"] = "worker-7"
    with pytest.raises(CONTROLLER.ProvisionError, match="authenticated 8/8"):
        CONTROLLER._validate_fleet_receipts(records, "other-host")


def test_held_lock_revalidation_rejects_name_and_parent_substitution(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "locks"
    parent.mkdir(mode=0o755)
    lock_path = parent / "lease.lock"
    lock_path.touch(mode=0o664)
    lock_path.chmod(0o664)
    held = CONTROLLER._open_lock(
        parent,
        lock_path.name,
        uid=os.getuid(),
        gid=os.getgid(),
        mode=0o664,
        parent_uid=os.getuid(),
        parent_gid=os.getgid(),
        forbid_parent_write=False,
    )
    try:
        CONTROLLER._revalidate_lock(held)
        lock_path.rename(parent / "preserved.lock")
        lock_path.touch(mode=0o664)
        lock_path.chmod(0o664)
        with pytest.raises(CONTROLLER.ProvisionError, match="pathname drift"):
            CONTROLLER._revalidate_lock(held)
    finally:
        held.close()

    parent = tmp_path / "second-locks"
    parent.mkdir(mode=0o755)
    lock_path = parent / "lease.lock"
    lock_path.touch(mode=0o664)
    lock_path.chmod(0o664)
    held = CONTROLLER._open_lock(
        parent,
        lock_path.name,
        uid=os.getuid(),
        gid=os.getgid(),
        mode=0o664,
        parent_uid=os.getuid(),
        parent_gid=os.getgid(),
        forbid_parent_write=False,
    )
    try:
        parent.rename(tmp_path / "second-locks-preserved")
        parent.mkdir(mode=0o755)
        replacement = parent / "lease.lock"
        replacement.touch(mode=0o664)
        replacement.chmod(0o664)
        with pytest.raises(CONTROLLER.ProvisionError, match="parent pathname drift"):
            CONTROLLER._revalidate_lock(held)
    finally:
        held.close()


def test_evidence_publication_is_fd_bound_exclusive_and_replayable(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(mode=0o755)
    artifact_root.chmod(0o755)
    tag = "gate_d_repo_prerequisite_20260901T010203123456789Z"
    path, receipt = CONTROLLER._write_evidence(
        tag, {"status": "unit"}, artifact_root=artifact_root
    )
    raw = path.read_bytes()
    assert receipt["sha256"] == CONTROLLER._sha256(raw)
    assert receipt["bytes"] == len(raw)
    assert receipt["mode"] == "0600"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        CONTROLLER._write_evidence(
            tag, {"status": "replacement"}, artifact_root=artifact_root
        )
    with pytest.raises(CONTROLLER.ProvisionError, match="invalid evidence tag"):
        CONTROLLER._write_evidence(
            "../escape", {"status": "unsafe"}, artifact_root=artifact_root
        )


def test_default_off_and_static_syntax() -> None:
    bundle_root = CONTROLLER_PATH.parent
    assert sorted(
        path.relative_to(bundle_root).as_posix() for path in bundle_root.iterdir()
    ) == [
        "gate_d_repository_prerequisite_root.py",
        "provision_gate_d_repository_prerequisites.py",
    ]
    for path in (ROOT_HELPER_PATH, CONTROLLER_PATH):
        compile(path.read_bytes(), str(path), "exec")
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(CONTROLLER_PATH)],
        check=False,
        capture_output=True,
        env={},
        text=True,
    )
    assert completed.returncode != 0
    assert "invalid installed provisioner entry boundary" in completed.stderr
