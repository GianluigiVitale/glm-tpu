from __future__ import annotations

import fcntl
import importlib.util
import os
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = (
    ROOT / "scripts/greenfield/launch_gate_d_layer1_rms_schedule.py"
)
SPEC = importlib.util.spec_from_file_location("gate_d_pp16_numerical_launcher", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_launcher_binds_committed_sources_and_root_owned_installation() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert str(MODULE.INSTALL_PATH) == (
        "/opt/glm-tpu/bin/launch_gate_d_layer1_rms_schedule_v1.py"
    )
    assert "expected_uid=0, expected_gid=0, expected_mode=0o555" in source
    assert source.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert "sys.flags.isolated != 1" in source
    assert "PYTHON_SHA256" in source
    assert 'launcher_raw != _git("show", f"{pin}:{SOURCE_PATH}")' in source
    assert 'f"{pin}:{WRAPPER_SOURCE_PATH}"' in source
    assert "os.O_NOFOLLOW" in source
    assert "os.execve(" in source
    assert 'f"/proc/self/fd/{WRAPPER_FD}"' in source
    wrapper = ROOT / MODULE.WRAPPER_SOURCE_PATH
    assert MODULE.WRAPPER_SHA256 == sha256(wrapper.read_bytes()).hexdigest()


def test_sealed_snapshot_survives_between_pass_path_substitution(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "wrapper.sh"
    original = b"#!/usr/bin/bash\nprintf 'ORIGINAL\\n'\n"
    malicious = b"#!/usr/bin/bash\nprintf 'SUBSTITUTED\\n'\n"
    wrapper.write_bytes(original)

    snapshot = MODULE._read_stable_regular(wrapper)
    descriptor = MODULE._create_sealed_wrapper(snapshot)
    try:
        replacement = tmp_path / "replacement.sh"
        replacement.write_bytes(malicious)
        os.replace(replacement, wrapper)

        assert wrapper.read_bytes() == malicious
        assert os.pread(descriptor, len(original) + 1, 0) == original
        assert (
            sha256(os.pread(descriptor, len(original), 0)).hexdigest()
            == sha256(original).hexdigest()
        )
        assert fcntl.fcntl(descriptor, MODULE.F_GET_SEALS) == MODULE.REQUIRED_SEALS
        with pytest.raises(PermissionError):
            os.pwrite(descriptor, b"X", 0)
    finally:
        os.close(descriptor)


def test_launcher_inherits_only_sealed_wrapper_and_lock_descriptors() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert "os.MFD_ALLOW_SEALING" in source
    assert "F_ADD_SEALS" in source
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert "WRAPPER_FD = 10" in source
    assert "LOCK_FDS = (11, 12)" in source
    assert source.count("_bind_inherited_fd(") == 3
    assert "GLM_GATE_D_WRAPPER_SHA256" in source
    assert "GLM_GATE_D_LAUNCHER_SHA256" in source


def test_launcher_binds_all_python_children_to_root_owned_immutable_capsule(
    tmp_path: Path,
) -> None:
    assert str(MODULE.IMMUTABLE_CAPSULE_ROOT) == (
        "/usr/local/libexec/glm-tpu/gate-d-layer1-rms-schedule-v1"
    )
    assert Path("/usr") in MODULE.IMMUTABLE_CAPSULE_ROOT.parents
    assert Path("/opt") not in MODULE.IMMUTABLE_CAPSULE_ROOT.parents
    expected = {
        "DRIVER": "scripts/greenfield/run_gate_d_layer1_rms_schedule.py",
        "PUBLISHER": (
            "scripts/greenfield/publish_gate_d_layer1_rms_schedule.py"
        ),
        "MIRROR_VERIFIER": (
            "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
        ),
    }
    for label, installed, repository_path, expected_sha256 in MODULE.IMMUTABLE_CHILDREN:
        assert repository_path == expected[label]
        assert installed.parent == MODULE.IMMUTABLE_CAPSULE_ROOT
        assert (
            expected_sha256 == sha256((ROOT / repository_path).read_bytes()).hexdigest()
        )

    hostile = tmp_path / "same-uid-capsule" / "child.py"
    hostile.parent.mkdir()
    hostile.write_text("raise SystemExit('substituted')\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="unsafe root-owned launcher parent"):
        MODULE._require_root_boundary(hostile)
