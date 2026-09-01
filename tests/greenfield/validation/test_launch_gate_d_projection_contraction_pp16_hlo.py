from __future__ import annotations

import ast
import fcntl
import importlib.util
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "scripts/greenfield/launch_gate_d_projection_contraction_pp16_hlo.py"
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_hlo.sh"
SPEC = importlib.util.spec_from_file_location(
    "gate_d_projection_contraction_hlo_launcher", SOURCE
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _embedded_wrapper_python(name: str) -> str:
    source = WRAPPER.read_text(encoding="ascii")
    marker = f"read -r -d '' {name} <<'{name}_EOF' || true\n"
    terminal = f"{name}_EOF\n"
    assert source.count(marker) == 1
    return source.split(marker, 1)[1].split(terminal, 1)[0]


GIT_AUTHORITY_VERIFIER = _embedded_wrapper_python("GIT_AUTHORITY_VERIFIER")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), *arguments],
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(repository.parent),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_authority_helper_namespace() -> dict[str, object]:
    tree = ast.parse(GIT_AUTHORITY_VERIFIER)
    selected: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id in {"git_environment", "git_command"}
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name == "git")
        ):
            selected.append(node)
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    exec(  # noqa: S102 - execute only the parsed, selected sealed helper nodes.
        compile(ast.Module(selected, type_ignores=[]), "<git-helper>", "exec"),
        namespace,
    )
    return namespace


def _repository_pair(tmp_path: Path) -> tuple[Path, str, str, str]:
    branch = "test-authority"
    bare = tmp_path / "remote.git"
    worktree = tmp_path / "worktree"
    bare.mkdir()
    worktree.mkdir()
    _git(bare, "init", "--bare", "-q")
    _git(worktree, "init", "-q", "-b", branch)
    tracked = worktree / "tracked"
    tracked.write_text("one\n", encoding="ascii")
    _git(worktree, "add", "tracked")
    _git(
        worktree,
        "-c",
        "user.name=Gate D",
        "-c",
        "user.email=gate-d@example.invalid",
        "commit",
        "-qm",
        "one",
    )
    tracked.write_text("two\n", encoding="ascii")
    _git(worktree, "add", "tracked")
    _git(
        worktree,
        "-c",
        "user.name=Gate D",
        "-c",
        "user.email=gate-d@example.invalid",
        "commit",
        "-qm",
        "two",
    )
    origin = f"file://{bare}"
    _git(worktree, "remote", "add", "origin", origin)
    _git(worktree, "push", "-qu", "origin", branch)
    return worktree, branch, origin, _git(worktree, "rev-parse", "HEAD")


def _install_hostile_local_git_config(worktree: Path, tmp_path: Path) -> Path:
    marker = tmp_path / "hostile-git-command-ran"
    hostile = tmp_path / "hostile-git-command.sh"
    hostile.write_text(
        f"#!/usr/bin/bash\n/usr/bin/printf HIT >> {marker}\nexit 0\n",
        encoding="ascii",
    )
    hostile.chmod(0o755)
    _git(worktree, "config", "--local", "core.fsmonitor", str(hostile))
    _git(worktree, "config", "--local", "core.sshCommand", str(hostile))
    _git(
        worktree,
        "config",
        "--local",
        "url.file:///definitely-not-the-git-origin/.insteadOf",
        _git(worktree, "config", "--local", "remote.origin.url"),
    )
    return marker


def test_launcher_binds_committed_sources_and_root_owned_installation() -> None:
    source = SOURCE.read_text(encoding="ascii")
    assert str(MODULE.INSTALL_PATH) == (
        "/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v1.py"
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


def test_hardened_git_helper_ignores_hostile_local_execution_and_url_rewrite(
    tmp_path: Path,
) -> None:
    worktree, branch, origin, pin = _repository_pair(tmp_path)
    marker = _install_hostile_local_git_config(worktree, tmp_path)
    namespace = _git_authority_helper_namespace()
    git_helper = namespace["git"]
    assert callable(git_helper)

    assert (
        git_helper(worktree, "status", "--porcelain=v1", "--untracked-files=all") == b""
    )
    expected_ref = f"refs/heads/{branch}"
    assert git_helper(
        Path("/"),
        "-c",
        "protocol.file.allow=always",
        "ls-remote",
        "--refs",
        origin,
        expected_ref,
    ) == (f"{pin}\t{expected_ref}\n".encode())
    assert not marker.exists()


def test_git_authority_verifier_rejects_replacement_ref_before_remote(
    tmp_path: Path,
) -> None:
    worktree, branch, _origin, pin = _repository_pair(tmp_path)
    marker = _install_hostile_local_git_config(worktree, tmp_path)
    _git(worktree, "replace", pin, f"{pin}^")
    production_origin = "git@github.com:GianluigiVitale/glm-tpu.git"
    _git(worktree, "remote", "set-url", "origin", production_origin)

    completed = subprocess.run(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            GIT_AUTHORITY_VERIFIER,
            str(worktree),
            branch,
            production_origin,
            pin,
        ],
        cwd="/",
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "forbidden replacement refs" in completed.stderr
    assert not marker.exists()


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
    "GLM_GATE_D_PROJECTION_CONTRACTION_HLO_WRAPPER_SANITIZED": "1",
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
            "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE": "1",
            "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE": "compile_only",
            "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG": (
                "gate_d_projection_contraction_pp16_hlo_20260901T120000123456789Z"
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
        "/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v1"
    )
    expected = {
        "DRIVER": "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py",
        "PUBLISHER": "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py",
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
        MODULE.EXPECTED_FIXED_ENVIRONMENT["GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE"]
        == "compile_only"
    )
    assert (
        MODULE.EXPECTED_FIXED_ENVIRONMENT[
            "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE"
        ]
        == "1"
    )
    source = SOURCE.read_text(encoding="ascii")
    assert "set(os.environ) != INPUT_ENVIRONMENT_KEYS" in source
