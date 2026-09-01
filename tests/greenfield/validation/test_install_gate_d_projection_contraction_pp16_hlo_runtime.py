from __future__ import annotations

import importlib.util
import os
import stat
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / (
    "scripts/greenfield/install_gate_d_projection_contraction_pp16_hlo_runtime.py"
)
ANALYZER = ROOT / (
    "scripts/greenfield/"
    "analyze_gate_d_projection_contraction_pp16_hlo_orchestration_source.py"
)


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load(SOURCE, "gate_d_projection_contraction_hlo_installer")
ANALYZER_MODULE = _load(ANALYZER, "gate_d_projection_contraction_hlo_source_analyzer")


def _directory(path: Path, mode: int = 0o755) -> Path:
    path.mkdir(mode=mode)
    path.chmod(mode)
    return path


def test_payload_hashes_and_install_targets_are_exact() -> None:
    expected = {
        "acquire_gate_d_projection_contraction_pp16_hlo.py",
        "launch_gate_d_projection_contraction_pp16_hlo.py",
        "publish_gate_d_projection_contraction_pp16_hlo.py",
        "verify_gate_d_same_region_git_mirror.py",
    }
    assert set(MODULE.PAYLOADS) == expected
    for name, expected_sha256 in MODULE.PAYLOADS.items():
        assert (
            expected_sha256
            == sha256((ROOT / "scripts/greenfield" / name).read_bytes()).hexdigest()
        )
    assert set(MODULE.CAPSULE_NAMES) == expected - {
        "launch_gate_d_projection_contraction_pp16_hlo.py"
    }
    assert str(MODULE.SOURCE_ROOT) == (
        "/opt/glm-tpu/gate-d-projection-contraction-hlo-install-v1"
    )
    assert str(MODULE.LAUNCHER_TARGET) == (
        "/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v1.py"
    )
    assert str(MODULE.CAPSULE_TARGET) == (
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v1"
    )


def test_installer_is_install_only_and_requires_exact_root_invocation() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert source.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert "os.geteuid() != 0" in source
    assert "dict(os.environ) != EXPECTED_ENVIRONMENT" in source
    assert "sys.argv != [str(INSTALLER_PATH)]" in source
    assert "_RENAME_NOREPLACE" in source
    assert '"launcher_invoked": False' in source
    assert "subprocess" not in source
    assert "execve" not in source
    assert "jax" not in source.lower()
    assert "google.cloud" not in source
    assert "gs://" not in source


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
    assert launcher.read_bytes() == launcher_raw


@pytest.mark.parametrize("target_kind", ["regular", "symlink"])
@pytest.mark.parametrize("publisher", ["capsule", "launcher"])
def test_publish_never_replaces_hostile_existing_target(
    tmp_path: Path, target_kind: str, publisher: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "parent")
    target = parent / "target"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"hostile\n")
    if target_kind == "regular":
        target.write_bytes(b"hostile\n")
    else:
        target.symlink_to(sentinel)

    with pytest.raises((RuntimeError, OSError)):
        if publisher == "capsule":
            MODULE._publish_capsule(
                parent, target, {"driver.py": b"expected\n"}, uid=uid, gid=gid
            )
        else:
            MODULE._publish_launcher(parent, target, b"expected\n", uid=uid, gid=gid)

    if target_kind == "regular":
        assert target.read_bytes() == b"hostile\n"
    else:
        assert target.is_symlink()
        assert sentinel.read_bytes() == b"hostile\n"


@pytest.mark.parametrize("target_kind", ["regular", "symlink", "directory"])
@pytest.mark.parametrize("publisher", ["capsule", "launcher"])
def test_publish_preserves_preexisting_pid_staging_target(
    tmp_path: Path, target_kind: str, publisher: str
) -> None:
    uid = os.getuid()
    gid = os.getgid()
    parent = _directory(tmp_path / "parent")
    target = parent / "target"
    staging = parent / f".{target.name}.tmp-{os.getpid()}"
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel\n")
    if target_kind == "regular":
        staging.write_bytes(b"preserve\n")
    elif target_kind == "symlink":
        staging.symlink_to(sentinel)
    else:
        staging.mkdir()
        (staging / "preserve").write_bytes(b"preserve\n")

    with pytest.raises(FileExistsError):
        if publisher == "capsule":
            MODULE._publish_capsule(
                parent, target, {"driver.py": b"expected\n"}, uid=uid, gid=gid
            )
        else:
            MODULE._publish_launcher(parent, target, b"expected\n", uid=uid, gid=gid)

    if target_kind == "regular":
        assert staging.read_bytes() == b"preserve\n"
    elif target_kind == "symlink":
        assert staging.is_symlink()
        assert sentinel.read_bytes() == b"sentinel\n"
    else:
        assert (staging / "preserve").read_bytes() == b"preserve\n"


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


def test_installer_source_audit_accepts_exact_source_and_rejects_mutations() -> None:
    raw = SOURCE.read_bytes()
    audit = ANALYZER_MODULE._audit_installer(raw)
    assert audit["payload_count"] == 4
    assert audit["launcher_invocation_count"] == 0
    mutations = (
        raw.replace(b"c660d50e", b"d660d50e", 1),
        raw.replace(b"install-v1", b"install-v9", 1),
        raw.replace(b"_publish_capsule(", b"_publish_capsule_removed(", 1),
        raw.replace(b'"launcher_invoked": False', b'"launcher_invoked": True', 1),
    )
    for mutation in mutations:
        with pytest.raises((RuntimeError, KeyError, ValueError, SyntaxError)):
            ANALYZER_MODULE._audit_installer(mutation)
