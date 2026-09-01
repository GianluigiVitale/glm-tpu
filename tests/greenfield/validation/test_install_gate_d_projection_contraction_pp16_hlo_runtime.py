from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
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
ARTIFACT = ROOT / (
    "docs/artifacts/"
    "gate-d-projection-contraction-pp16-hlo-orchestration-install-source.json"
)
ARTIFACT_SHA256 = "f6d1736105e67dbbe9b336b727ca015a3bf7506231aa8ae7dcd407a81cbe666e"
V2_ARTIFACT = ROOT / (
    "docs/artifacts/"
    "gate-d-projection-contraction-pp16-hlo-orchestration-install-source-v2.json"
)
V2_ARTIFACT_SHA256 = "b0e58ea7c5a5e3bb0158f936442a67336d279b9f991c2247adfe9f2629759c79"
INSTALL_ARTIFACT = ROOT / (
    "docs/artifacts/gate-d-projection-contraction-pp16-hlo-runtime-install.json"
)
INSTALL_ARTIFACT_SHA256 = (
    "99a134c400629c936b761bf738873f2a81a41512f8c1823d26d5232b0648abc6"
)
V2_INSTALL_ARTIFACT = ROOT / (
    "docs/artifacts/gate-d-projection-contraction-pp16-hlo-runtime-install-v2.json"
)
V2_INSTALL_ARTIFACT_SHA256 = (
    "6a10743f80b924e5fe025f97b51f62359193653f73b83b65809cfbd8c42a951c"
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
        "/opt/glm-tpu/gate-d-projection-contraction-hlo-install-v3"
    )
    assert str(MODULE.LAUNCHER_TARGET) == (
        "/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v3.py"
    )
    assert str(MODULE.CAPSULE_TARGET) == (
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v3"
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
        raw.replace(b"install-v3", b"install-v9", 1),
        raw.replace(b"_publish_capsule(", b"_publish_capsule_removed(", 1),
        raw.replace(b'"launcher_invoked": False', b'"launcher_invoked": True', 1),
    )
    for mutation in mutations:
        with pytest.raises((RuntimeError, KeyError, ValueError, SyntaxError)):
            ANALYZER_MODULE._audit_installer(mutation)


def test_orchestration_source_audits_all_exact_sources_and_predecessors() -> None:
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
            "input_spec_count"
        ]
        == 4
    )
    assert (
        ANALYZER_MODULE._audit_wrapper(snapshots[ANALYZER_MODULE.WRAPPER_PATH])[
            "protected_compile_process_count"
        ]
        == 1
    )
    assert ANALYZER_MODULE._predecessor_authority() == (ANALYZER_MODULE.REFERENCE_PATHS)


def test_historical_orchestration_source_artifact_binds_its_exact_commit() -> None:
    artifact = ARTIFACT.read_bytes()
    assert len(artifact) == 3026
    assert sha256(artifact).hexdigest() == ARTIFACT_SHA256
    parsed = json.loads(artifact)
    assert parsed["code_hash"] == "d4288831edbd4f08ea61ef8faf6edbe65eabfb4d"
    assert parsed["authorization"] == {
        "cloud_write": False,
        "compile_only_hlo_acquisition": False,
        "full_dsa_or_8k": False,
        "launcher_invocation": False,
        "persistence_only": True,
        "privileged_install": False,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    assert parsed["gate_d_closed"] is False
    current_pin = subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ancestry = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            parsed["code_hash"],
            current_pin,
        ],
        check=False,
    )
    assert ancestry.returncode == 0
    for relative, expected_sha256 in parsed["source_sha256s"].items():
        raw = subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                str(ROOT),
                "show",
                f"{parsed['code_hash']}:{relative}",
            ]
        )
        assert sha256(raw).hexdigest() == expected_sha256


def test_v2_orchestration_source_artifact_binds_its_exact_commit() -> None:
    artifact = V2_ARTIFACT.read_bytes()
    assert len(artifact) == 3026
    assert sha256(artifact).hexdigest() == V2_ARTIFACT_SHA256
    parsed = json.loads(artifact)
    assert parsed["code_hash"] == "efe99ba87c1e1a7163f436fb7cc55bcd46e395ae"
    current_pin = subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    ancestry = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            parsed["code_hash"],
            current_pin,
        ],
        check=False,
    )
    assert ancestry.returncode == 0
    for relative, expected_sha256 in parsed["source_sha256s"].items():
        raw = subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                str(ROOT),
                "show",
                f"{parsed['code_hash']}:{relative}",
            ]
        )
        assert sha256(raw).hexdigest() == expected_sha256


def test_runtime_install_artifact_binds_exact_persisted_sources() -> None:
    artifact = INSTALL_ARTIFACT.read_bytes()
    assert len(artifact) == 3761
    assert sha256(artifact).hexdigest() == INSTALL_ARTIFACT_SHA256
    parsed = json.loads(artifact)
    assert parsed["authorization"] == {
        "cloud_write": False,
        "full_dsa_or_8k": False,
        "hlo_acquisition": False,
        "launcher_invocation": False,
        "persistence_only": True,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    assert parsed["install"]["launcher_invoked"] is False
    postconditions = parsed["postconditions"]
    assert postconditions["launcher_process_present"] is False
    lease_recheck = postconditions["lease_recheck"]
    assert lease_recheck["simultaneously_held_by_auditor"] is True
    assert [record["path"] for record in lease_recheck["leases"]] == [
        "/opt/glm-tpu/locks/glm_pod_workload.lock",
        "/opt/glm-tpu/locks/glm_tpu_rsync.lock",
        "/home/gianl/glm-run/.glm_pod_workload.lock",
        "/home/gianl/.glm-tpu-rsync.lock",
    ]
    assert all(record["free"] is True for record in lease_recheck["leases"])
    assert parsed["status"] == "IMMUTABLE_RUNTIME_INSTALLED_NOT_INVOKED"
    source_commit = parsed["install"]["source_commit"]
    expected_members = {}
    for name in parsed["installed_objects"]["source_capsule"]["members"]:
        raw = subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                str(ROOT),
                "show",
                f"{source_commit}:scripts/greenfield/{name}",
            ]
        )
        expected_members[name] = sha256(raw).hexdigest()
    assert parsed["installed_objects"]["source_capsule"]["members"] == (
        expected_members
    )
    assert parsed["installed_objects"]["runtime_capsule"]["members"] == {
        name: expected_members[name]
        for name in parsed["installed_objects"]["runtime_capsule"]["members"]
    }
    assert (
        parsed["installed_objects"]["launcher"]["sha256"]
        == expected_members["launch_gate_d_projection_contraction_pp16_hlo.py"]
    )
    ancestry = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            parsed["install"]["source_commit"],
            parsed["install"]["authority_commit"],
        ],
        check=False,
    )
    assert ancestry.returncode == 0


def test_v2_runtime_install_artifact_binds_exact_persisted_sources() -> None:
    artifact = V2_INSTALL_ARTIFACT.read_bytes()
    assert len(artifact) == 3699
    assert sha256(artifact).hexdigest() == V2_INSTALL_ARTIFACT_SHA256
    parsed = json.loads(artifact)
    assert parsed["authorization"] == {
        "cloud_write": False,
        "full_dsa_or_8k": False,
        "hlo_acquisition": False,
        "launcher_invocation": False,
        "persistence_only": True,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    assert parsed["install"]["launcher_invoked"] is False
    assert parsed["install"]["source_commit"] == (
        "efe99ba87c1e1a7163f436fb7cc55bcd46e395ae"
    )
    assert parsed["install"]["authority_commit"] == (
        "f9fa964b61ce51a8e2e0452a3603b7e9c9d30093"
    )
    assert parsed["preconditions"]["install_source_certificate_sha256"] == (
        V2_ARTIFACT_SHA256
    )
    postconditions = parsed["postconditions"]
    assert postconditions["launcher_process_present"] is False
    lease_recheck = postconditions["lease_recheck"]
    assert lease_recheck["simultaneously_held_by_auditor"] is True
    assert [record["path"] for record in lease_recheck["leases"]] == [
        "/opt/glm-tpu/locks/glm_pod_workload.lock",
        "/opt/glm-tpu/locks/glm_tpu_rsync.lock",
        "/home/gianl/glm-run/.glm_pod_workload.lock",
        "/home/gianl/.glm-tpu-rsync.lock",
    ]
    assert all(record["free"] is True for record in lease_recheck["leases"])
    assert parsed["status"] == "IMMUTABLE_RUNTIME_V2_INSTALLED_NOT_INVOKED"
    source_commit = parsed["install"]["source_commit"]
    expected_members = {}
    for name in parsed["installed_objects"]["source_capsule"]["members"]:
        raw = subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                str(ROOT),
                "show",
                f"{source_commit}:scripts/greenfield/{name}",
            ]
        )
        expected_members[name] = sha256(raw).hexdigest()
    assert parsed["installed_objects"]["source_capsule"]["members"] == (
        expected_members
    )
    assert parsed["installed_objects"]["source_capsule"]["path"].endswith("install-v2")
    assert parsed["installed_objects"]["runtime_capsule"]["members"] == {
        name: expected_members[name]
        for name in parsed["installed_objects"]["runtime_capsule"]["members"]
    }
    assert parsed["installed_objects"]["runtime_capsule"]["path"].endswith("hlo-v2")
    assert (
        parsed["installed_objects"]["launcher"]["sha256"]
        == expected_members["launch_gate_d_projection_contraction_pp16_hlo.py"]
    )
    assert parsed["installed_objects"]["launcher"]["path"].endswith("hlo_v2.py")
    ancestry = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            parsed["install"]["source_commit"],
            parsed["install"]["authority_commit"],
        ],
        check=False,
    )
    assert ancestry.returncode == 0
