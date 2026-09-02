from __future__ import annotations

import importlib.util
import os
import stat
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = (
    ROOT
    / "scripts/greenfield/install_gate_d_layer1_rms_schedule_runtime.py"
)
ANALYZER = ROOT / (
    "scripts/greenfield/"
    "analyze_gate_d_layer1_rms_schedule_orchestration_source.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gate_d_projection_contraction_numerical_installer", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "gate_d_projection_contraction_numerical_source_analyzer", ANALYZER
)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
ANALYZER_MODULE = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(ANALYZER_MODULE)


def _directory(path: Path, mode: int = 0o755) -> Path:
    path.mkdir(mode=mode)
    path.chmod(mode)
    return path


def test_payload_hashes_and_install_targets_are_exact() -> None:
    expected = {
        "run_gate_d_layer1_rms_schedule.py",
        "launch_gate_d_layer1_rms_schedule.py",
        "publish_gate_d_layer1_rms_schedule.py",
        "verify_gate_d_same_region_git_mirror.py",
    }
    assert set(MODULE.PAYLOADS) == expected
    for name, expected_sha256 in MODULE.PAYLOADS.items():
        assert (
            expected_sha256
            == sha256((ROOT / "scripts/greenfield" / name).read_bytes()).hexdigest()
        )
    assert set(MODULE.CAPSULE_NAMES) == expected - {
        "launch_gate_d_layer1_rms_schedule.py"
    }
    assert str(MODULE.SOURCE_ROOT) == (
        "/opt/glm-tpu/gate-d-layer1-rms-schedule-install-v2"
    )
    assert str(MODULE.LAUNCHER_TARGET) == (
        "/opt/glm-tpu/bin/launch_gate_d_layer1_rms_schedule_v2.py"
    )
    assert str(MODULE.CAPSULE_TARGET) == (
        "/usr/local/libexec/glm-tpu/gate-d-layer1-rms-schedule-v2"
    )


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


def test_orchestration_analyzer_audits_every_exact_source_and_predecessor() -> None:
    snapshots = {
        relative: (ROOT / relative).read_bytes()
        for relative in ANALYZER_MODULE.AUDITED_SOURCE_PATHS
    }
    assert (
        ANALYZER_MODULE._audit_installer(snapshots[ANALYZER_MODULE.INSTALLER_PATH])[
            "launcher_invocation_count"
        ]
        == 0
    )
    assert ANALYZER_MODULE._audit_launcher(
        snapshots[ANALYZER_MODULE.LAUNCHER_PATH],
        snapshots[ANALYZER_MODULE.WRAPPER_PATH],
    )["retained_lock_fds"] == [11, 12]
    assert (
        ANALYZER_MODULE._audit_publisher(snapshots[ANALYZER_MODULE.PUBLISHER_PATH])[
            "sha256"
        ]
        == sha256(snapshots[ANALYZER_MODULE.PUBLISHER_PATH]).hexdigest()
    )
    assert (
        ANALYZER_MODULE._audit_wrapper(snapshots[ANALYZER_MODULE.WRAPPER_PATH])[
            "protected_numerical_process_count"
        ]
        == 1
    )
    assert ANALYZER_MODULE._predecessor_authority() == ({**ANALYZER_MODULE.REFERENCE_PATHS, **ANALYZER_MODULE.SEALED_INPUTS})


@pytest.mark.parametrize(
    ("auditor", "path", "needle", "replacement"),
    [
        (
            "publisher",
            "PUBLISHER_PATH",
            b"if control != db548:",
            b"if control == db548:",
        ),
        ("wrapper", "WRAPPER_PATH", b"--execute-once 1", b"--execute-once 2"),
        ("launcher", "LAUNCHER_PATH", b"F_ADD_SEALS", b"F_ADD_WEAKS"),
        (
            "installer",
            "INSTALLER_PATH",
            b'"launcher_invoked": False',
            b'"launcher_invoked": True',
        ),
    ],
)
def test_orchestration_analyzer_rejects_security_boundary_mutations(
    auditor: str,
    path: str,
    needle: bytes,
    replacement: bytes,
) -> None:
    raw = (ROOT / getattr(ANALYZER_MODULE, path)).read_bytes()
    assert needle in raw
    mutation = raw.replace(needle, replacement, 1)
    with pytest.raises((RuntimeError, KeyError, ValueError, SyntaxError)):
        if auditor == "launcher":
            ANALYZER_MODULE._audit_launcher(
                mutation,
                (ROOT / ANALYZER_MODULE.WRAPPER_PATH).read_bytes(),
            )
        else:
            getattr(ANALYZER_MODULE, f"_audit_{auditor}")(mutation)
