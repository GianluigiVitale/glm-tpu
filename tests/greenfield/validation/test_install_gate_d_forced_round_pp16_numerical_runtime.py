from __future__ import annotations

import importlib.util
import json
import os
import stat
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/install_gate_d_forced_round_pp16_numerical_runtime.py"
SOURCE_CERTIFICATE = ROOT / (
    "docs/artifacts/"
    "gate-d-forced-round-pp16-numerical-orchestration-install-source-v2.json"
)
SPEC = importlib.util.spec_from_file_location(
    "gate_d_forced_round_hlo_installer", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _directory(path: Path, mode: int = 0o755) -> Path:
    path.mkdir(mode=mode)
    path.chmod(mode)
    return path


def test_payload_hashes_and_install_targets_are_exact() -> None:
    expected = {
        "run_gate_d_forced_round_pp16_numerical.py",
        "launch_gate_d_forced_round_pp16_numerical.py",
        "publish_gate_d_forced_round_pp16_numerical.py",
        "verify_gate_d_same_region_git_mirror.py",
    }
    assert set(MODULE.PAYLOADS) == expected
    for name, expected_sha256 in MODULE.PAYLOADS.items():
        assert (
            expected_sha256
            == sha256((ROOT / "scripts/greenfield" / name).read_bytes()).hexdigest()
        )
    assert set(MODULE.CAPSULE_NAMES) == expected - {
        "launch_gate_d_forced_round_pp16_numerical.py"
    }
    assert str(MODULE.SOURCE_ROOT) == (
        "/opt/glm-tpu/gate-d-forced-round-numerical-install-v2"
    )
    assert str(MODULE.LAUNCHER_TARGET) == (
        "/opt/glm-tpu/bin/launch_gate_d_forced_round_pp16_numerical_v2.py"
    )
    assert str(MODULE.CAPSULE_TARGET) == (
        "/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-numerical-v2"
    )


def test_source_certificate_binds_every_runtime_source_without_authority() -> None:
    certificate = json.loads(SOURCE_CERTIFICATE.read_text(encoding="ascii"))
    assert certificate["authorization"] == {
        "cloud_write": False,
        "launcher_invocation": False,
        "persistence_only": True,
        "privileged_install": False,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    assert certificate["predecessors"]["code_pin"] == (
        "2e401e3ed4afefdc5cc748270895f374415bf9c1"
    )
    for record in certificate["sources"].values():
        path = ROOT / record["path"]
        assert record["byte_count"] == path.stat().st_size
        assert record["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert certificate["verification"]["focused_tests_passed"] == 76
    assert certificate["verification"]["adjacent_tests_passed"] == 0
    assert certificate["verification"]["historical_v1_adjacent_tests_reused"] == 273
    assert certificate["verification"]["sealed_history_tests_deselected"] == 0
    assert certificate["verification"]["ruff_version"] == "0.16.5"


def test_capsule_and_launcher_publish_exactly_and_idempotently(tmp_path: Path) -> None:
    uid = os.getuid()
    gid = os.getgid()
    capsule_parent = _directory(tmp_path / "capsules")
    launcher_parent = _directory(tmp_path / "bin")
    capsule = capsule_parent / "runtime"
    launcher = launcher_parent / "launcher.py"
    payloads = {"driver.py": b"driver\n", "publisher.py": b"publisher\n"}
    launcher_raw = b"launcher\n"

    MODULE._publish_capsule(capsule_parent, capsule, payloads, uid=uid, gid=gid)
    MODULE._publish_launcher(launcher_parent, launcher, launcher_raw, uid=uid, gid=gid)
    first_capsule_inode = capsule.stat().st_ino
    first_launcher_inode = launcher.stat().st_ino

    MODULE._publish_capsule(capsule_parent, capsule, payloads, uid=uid, gid=gid)
    MODULE._publish_launcher(launcher_parent, launcher, launcher_raw, uid=uid, gid=gid)

    assert capsule.stat().st_ino == first_capsule_inode
    assert launcher.stat().st_ino == first_launcher_inode
    assert stat.S_IMODE(capsule.stat().st_mode) == 0o555
    assert stat.S_IMODE(launcher.stat().st_mode) == 0o555
    assert {item.name for item in capsule.iterdir()} == set(payloads)
    for name, raw in payloads.items():
        child = capsule / name
        assert child.read_bytes() == raw
        assert stat.S_IMODE(child.stat().st_mode) == 0o555
    assert launcher.read_bytes() == launcher_raw


@pytest.mark.parametrize("hostile_kind", ["regular", "symlink"])
def test_capsule_never_replaces_hostile_existing_target(
    tmp_path: Path, hostile_kind: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "capsules")
    target = parent / "runtime"
    hostile = b"hostile\n"
    if hostile_kind == "regular":
        target.write_bytes(hostile)
    else:
        destination = tmp_path / "destination"
        destination.write_bytes(hostile)
        target.symlink_to(destination)

    with pytest.raises((RuntimeError, OSError)):
        MODULE._publish_capsule(
            parent, target, {"driver.py": b"expected\n"}, uid=uid, gid=gid
        )

    assert (
        target.is_symlink()
        if hostile_kind == "symlink"
        else target.read_bytes() == hostile
    )


@pytest.mark.parametrize("hostile_kind", ["regular", "symlink"])
def test_launcher_never_replaces_hostile_existing_target(
    tmp_path: Path, hostile_kind: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "bin")
    target = parent / "launcher.py"
    hostile = b"hostile\n"
    if hostile_kind == "regular":
        target.write_bytes(hostile)
    else:
        destination = tmp_path / "destination"
        destination.write_bytes(hostile)
        target.symlink_to(destination)

    with pytest.raises((RuntimeError, OSError)):
        MODULE._publish_launcher(parent, target, b"expected\n", uid=uid, gid=gid)

    assert (
        target.is_symlink()
        if hostile_kind == "symlink"
        else target.read_bytes() == hostile
    )


@pytest.mark.parametrize("hostile_kind", ["regular", "symlink", "directory"])
def test_capsule_preserves_preexisting_pid_staging_target(
    tmp_path: Path, hostile_kind: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "capsules")
    target = parent / "runtime"
    staging = parent / f".{target.name}.tmp-{os.getpid()}"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel\n")
    if hostile_kind == "regular":
        staging.write_bytes(b"preserve\n")
    elif hostile_kind == "symlink":
        staging.symlink_to(sentinel)
    else:
        staging.mkdir()
        (staging / "preserve").write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError):
        MODULE._publish_capsule(
            parent, target, {"driver.py": b"expected\n"}, uid=uid, gid=gid
        )

    if hostile_kind == "regular":
        assert staging.read_bytes() == b"preserve\n"
    elif hostile_kind == "symlink":
        assert staging.is_symlink()
        assert sentinel.read_bytes() == b"sentinel\n"
    else:
        assert (staging / "preserve").read_bytes() == b"preserve\n"


@pytest.mark.parametrize("hostile_kind", ["regular", "symlink", "directory"])
def test_launcher_preserves_preexisting_pid_staging_target(
    tmp_path: Path, hostile_kind: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "bin")
    target = parent / "launcher.py"
    staging = parent / f".{target.name}.tmp-{os.getpid()}"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel\n")
    if hostile_kind == "regular":
        staging.write_bytes(b"preserve\n")
    elif hostile_kind == "symlink":
        staging.symlink_to(sentinel)
    else:
        staging.mkdir()
        (staging / "preserve").write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError):
        MODULE._publish_launcher(parent, target, b"expected\n", uid=uid, gid=gid)

    if hostile_kind == "regular":
        assert staging.read_bytes() == b"preserve\n"
    elif hostile_kind == "symlink":
        assert staging.is_symlink()
        assert sentinel.read_bytes() == b"sentinel\n"
    else:
        assert (staging / "preserve").read_bytes() == b"preserve\n"


def test_installer_is_install_only_and_requires_exact_root_invocation() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert source.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert "os.geteuid() != 0" in source
    assert "dict(os.environ) != EXPECTED_ENVIRONMENT" in source
    assert "sys.argv != [str(INSTALLER_PATH)]" in source
    assert "_RENAME_NOREPLACE" in source
    assert 'launcher_invoked": False' in source
    assert "subprocess" not in source
    assert "execve" not in source
    assert "jax" not in source.lower()
    assert "google.cloud" not in source
    assert "gs://" not in source


def test_stable_reader_rejects_links_and_wrong_hash(tmp_path: Path) -> None:
    uid = os.getuid()
    gid = os.getgid()
    source = tmp_path / "source.py"
    source.write_bytes(b"payload\n")
    source.chmod(0o555)
    link = tmp_path / "link.py"
    link.symlink_to(source)

    with pytest.raises(OSError):
        MODULE._read_regular(
            link,
            expected_sha256=sha256(b"payload\n").hexdigest(),
            mode=0o555,
            uid=uid,
            gid=gid,
        )
    with pytest.raises(RuntimeError, match="identity drifted"):
        MODULE._read_regular(
            source,
            expected_sha256="0" * 64,
            mode=0o555,
            uid=uid,
            gid=gid,
        )
