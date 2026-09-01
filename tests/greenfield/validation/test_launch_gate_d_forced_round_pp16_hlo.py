from __future__ import annotations

import fcntl
import importlib.util
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/launch_gate_d_forced_round_pp16_hlo.py"
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_forced_round_pp16_hlo.sh"
SPEC = importlib.util.spec_from_file_location(
    "gate_d_forced_round_hlo_launcher", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_launcher_binds_committed_sources_and_root_owned_installation() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert str(MODULE.INSTALL_PATH) == (
        "/opt/glm-tpu/bin/launch_gate_d_forced_round_pp16_hlo.py"
    )
    assert source.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert "expected_uid=0, expected_gid=0, expected_mode=0o555" in source
    assert "sys.flags.isolated != 1" in source
    assert 'launcher_raw != _git("show", f"{pin}:{SOURCE_PATH}")' in source
    assert 'f"{pin}:{WRAPPER_SOURCE_PATH}"' in source
    assert "os.O_NOFOLLOW" in source
    assert "os.execve(" in source
    assert 'f"/proc/self/fd/{WRAPPER_FD}"' in source
    assert MODULE.WRAPPER_SHA256 == sha256(WRAPPER.read_bytes()).hexdigest()


def test_sealed_snapshot_executes_original_after_named_path_substitution(
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
        completed = subprocess.run(
            [
                "/usr/bin/bash",
                "--noprofile",
                "--norc",
                f"/proc/self/fd/{descriptor}",
            ],
            check=True,
            capture_output=True,
            text=True,
            pass_fds=(descriptor,),
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        assert wrapper.read_bytes() == malicious
        assert completed.stdout == "ORIGINAL\n"
        assert fcntl.fcntl(descriptor, MODULE.F_GET_SEALS) == MODULE.REQUIRED_SEALS
        with pytest.raises(PermissionError):
            os.pwrite(descriptor, b"X", 0)
    finally:
        os.close(descriptor)


def test_launcher_to_bash_environment_has_no_oldpwd_before_wrapper_cd() -> None:
    script = """
import os
environment = dict(os.environ)
environment.update({
    "GLM_GATE_D_FORCED_ROUND_HLO_WRAPPER_SANITIZED": "1",
    "GLM_GATE_D_IMMUTABLE_LOCKS_HELD": "1",
    "GLM_GATE_D_LAUNCHER_PIN": "0" * 40,
    "GLM_GATE_D_LAUNCHER_SHA256": "1" * 64,
    "GLM_GATE_D_WRAPPER_MEMFD": "10",
    "GLM_GATE_D_WRAPPER_SHA256": "2" * 64,
})
os.execve(
    "/usr/bin/bash",
    [
        "/usr/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        "[[ -z ${OLDPWD+x} ]]; cd /; [[ -n ${OLDPWD+x} ]]; printf BOUNDARY_OK",
    ],
    environment,
)
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={
            "GLM_GATE_D_FORCED_ROUND_PP16_HLO_ACQUIRE": "1",
            "GLM_GATE_D_FORCED_ROUND_PP16_MODE": "compile_only",
            "GLM_GATE_D_FORCED_ROUND_PP16_TAG": (
                "gate_d_forced_round_pp16_hlo_20260901T120000123456789Z"
            ),
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.stdout == "BOUNDARY_OK"
    wrapper = WRAPPER.read_text(encoding="ascii")
    assert wrapper.index('python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"') < (
        wrapper.index("cd /")
    )
    assert "IMMUTABLE_LOCK_BROKER" not in wrapper
    assert "WRAPPER_ABS" not in wrapper


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


def test_launcher_binds_all_executed_python_to_immutable_capsule(
    tmp_path: Path,
) -> None:
    assert str(MODULE.IMMUTABLE_CAPSULE_ROOT) == (
        "/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-hlo"
    )
    expected = {
        "DRIVER": "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py",
        "PUBLISHER": "scripts/greenfield/publish_gate_d_forced_round_pp16_hlo.py",
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


def test_launcher_environment_is_exact_and_excludes_oldpwd() -> None:
    assert "OLDPWD" not in MODULE.INPUT_ENVIRONMENT_KEYS
    assert "OLDPWD" not in MODULE.EXPECTED_FIXED_ENVIRONMENT
    assert (
        MODULE.EXPECTED_FIXED_ENVIRONMENT["GLM_GATE_D_FORCED_ROUND_PP16_MODE"]
        == "compile_only"
    )
    assert (
        MODULE.EXPECTED_FIXED_ENVIRONMENT["GLM_GATE_D_FORCED_ROUND_PP16_HLO_ACQUIRE"]
        == "1"
    )
    source = SOURCE.read_text(encoding="ascii")
    assert "set(os.environ) != INPUT_ENVIRONMENT_KEYS" in source
