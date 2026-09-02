#!/usr/bin/env python3
"""Replay the locked US-CENTRAL2 Git mirror and prove one checkout closure."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
SOURCE_PATH = "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
PYTHON = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12")
PYTHON_SHA256 = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
MIRROR_URI = "gs://driftbench-dsv4-uc/repos/glm-tpu/.git"
BRANCH = "tooling/gate-d-compensated-pp16-numerical"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
TEMP_ROOT = Path("/home/gianl/glm-run")
BOUND_PATHS = (
    "docs/artifacts/gate-d-pp16-numerical-host-materialization-equivalence.json",
    "docs/artifacts/gate-d-compensated-pp16-hlo-source-location-bridge.json",
    "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json",
    "docs/artifacts/gate-d-runtime-locality-authority.json",
    "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py",
    "scripts/greenfield/launch_gate_d_compensated_pp16_numerical.py",
    "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py",
    "scripts/greenfield/publish_gate_d_compensated_pp16_numerical.py",
    "scripts/greenfield/run_gate_d_compensated_pp16_numerical.py",
    "scripts/greenfield/run_gate_d_compensated_pp16_numerical.sh",
    "scripts/greenfield/verify_gate_d_same_region_git_mirror.py",
)
EXPECTED_ENVIRONMENT = {
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
}
GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}
REMOTE_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_SSH_COMMAND": "/usr/bin/ssh -oBatchMode=yes -oConnectTimeout=30",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/home/gianl",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _snapshot_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe mirror-verifier file: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"mirror-verifier file changed while reading: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git(
    git_dir: Path,
    *arguments: str,
    worktree: Path | None = None,
    timeout: int = 180,
) -> bytes:
    command = ["/usr/bin/git"]
    if worktree is None:
        command.extend([f"--git-dir={git_dir}"])
    else:
        command.extend(["-C", str(worktree)])
    command.extend(arguments)
    try:
        return subprocess.check_output(
            command,
            env=GIT_ENVIRONMENT,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"Git mirror verification command failed: {arguments[0]}"
        ) from error


def validate_remote_pin_output(raw: bytes, pin: str, branch: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise RuntimeError("expected remote commit pin is invalid")
    if re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", branch) is None:
        raise RuntimeError("expected remote branch is invalid")
    expected = f"{pin}\trefs/heads/{branch}\n".encode("ascii")
    if raw != expected:
        raise RuntimeError("origin ls-remote output is stale or malformed")


def verify_origin_remote_pin(
    local_repo: Path,
    pin: str,
    *,
    branch: str,
    timeout: int = 60,
) -> dict[str, object]:
    command = [
        "/usr/bin/git",
        "-C",
        str(local_repo),
        "ls-remote",
        "--refs",
        "origin",
        f"refs/heads/{branch}",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=REMOTE_GIT_ENVIRONMENT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("origin ls-remote transport failed") from error
    if completed.returncode != 0:
        raise RuntimeError("origin ls-remote transport failed")
    validate_remote_pin_output(completed.stdout, pin, branch)
    return {
        "origin_remote_exact_record": True,
        "origin_remote_ref": f"refs/heads/{branch}",
        "origin_remote_sha256": sha256(completed.stdout).hexdigest(),
    }


def _require_no_symlinks(root: Path) -> None:
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise RuntimeError("downloaded Git mirror root is unsafe")
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise RuntimeError(f"downloaded Git mirror contains symlink: {path}")


def verify_mirror_directory(
    mirror_git_dir: Path,
    local_repo: Path,
    pin: str,
    *,
    branch: str,
    origin: str,
    bound_paths: tuple[str, ...],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", pin) is None:
        raise RuntimeError("expected mirror commit pin is invalid")
    if re.fullmatch(r"[a-z0-9][a-z0-9/_-]*", branch) is None:
        raise RuntimeError("expected mirror branch is invalid")
    _require_no_symlinks(mirror_git_dir)
    ref_path = mirror_git_dir / "refs/heads" / Path(branch)
    ref_raw = _snapshot_regular(ref_path)
    if ref_raw != (pin + "\n").encode("ascii"):
        raise RuntimeError("same-region Git mirror ref is stale or malformed")
    if (
        _git(mirror_git_dir, "config", "--local", "--get", "remote.origin.url").strip()
        != origin.encode()
    ):
        raise RuntimeError("same-region Git mirror origin drifted")
    if (mirror_git_dir / "objects/info/alternates").exists():
        raise RuntimeError("same-region Git mirror has object alternates")
    if (mirror_git_dir / "info/grafts").exists():
        raise RuntimeError("same-region Git mirror has grafts")
    if _git(
        mirror_git_dir, "for-each-ref", "--format=%(refname)", "refs/replace"
    ).strip():
        raise RuntimeError("same-region Git mirror has replacement refs")
    if _git(mirror_git_dir, "cat-file", "-t", pin).strip() != b"commit":
        raise RuntimeError("same-region Git mirror pin is not a commit")
    _git(mirror_git_dir, "fsck", "--strict", "--connectivity-only", pin, timeout=600)
    mirror_archive = _git(mirror_git_dir, "archive", "--format=tar", pin, timeout=600)
    local_archive = _git(
        local_repo / ".git",
        "archive",
        "--format=tar",
        pin,
        worktree=local_repo,
        timeout=600,
    )
    if mirror_archive != local_archive:
        raise RuntimeError("same-region Git mirror checkout archive drifted")
    tree_names = set(
        _git(mirror_git_dir, "ls-tree", "-r", "--name-only", pin)
        .decode("utf-8", errors="strict")
        .splitlines()
    )
    if not bound_paths or len(bound_paths) != len(set(bound_paths)):
        raise RuntimeError("same-region Git mirror bound path catalogue is invalid")
    blobs = []
    for path in bound_paths:
        if path not in tree_names or path.startswith("/") or ".." in Path(path).parts:
            raise RuntimeError(f"bound runtime blob is absent from mirror tree: {path}")
        mirror_object = _git(mirror_git_dir, "rev-parse", f"{pin}:{path}").strip()
        local_object = _git(
            local_repo / ".git",
            "rev-parse",
            f"{pin}:{path}",
            worktree=local_repo,
        ).strip()
        mirror_raw = _git(mirror_git_dir, "show", f"{pin}:{path}")
        local_raw = _git(
            local_repo / ".git", "show", f"{pin}:{path}", worktree=local_repo
        )
        if mirror_object != local_object or mirror_raw != local_raw:
            raise RuntimeError(f"same-region bound runtime blob drifted: {path}")
        blobs.append(
            {
                "git_object": mirror_object.decode("ascii", errors="strict"),
                "path": path,
                "sha256": sha256(mirror_raw).hexdigest(),
            }
        )
    return {
        "artifact_kind": "gate_d_same_region_git_mirror_replay",
        "bound_blob_count": len(blobs),
        "bound_blobs": blobs,
        "branch": branch,
        "checkout_archive_sha256": sha256(mirror_archive).hexdigest(),
        "commit": pin,
        "commit_connectivity_fsck": True,
        "mirror_uri": MIRROR_URI,
        "origin": origin,
        "schema_version": 1,
    }


def _verify_running_source(code_pin: str, expected_sha256: str) -> None:
    raw = _snapshot_regular(Path(__file__))
    if (
        _git(
            WORKTREE / ".git",
            "rev-parse",
            "HEAD",
            worktree=WORKTREE,
        )
        .decode("ascii")
        .strip()
        != code_pin
        or raw
        != _git(
            WORKTREE / ".git",
            "show",
            f"{code_pin}:{SOURCE_PATH}",
            worktree=WORKTREE,
        )
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError("same-region Git mirror verifier is not the committed blob")


def _download_mirror(destination: Path) -> None:
    environment = {
        **EXPECTED_ENVIRONMENT,
        "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
    }
    try:
        subprocess.run(
            [
                "/snap/bin/gcloud",
                "storage",
                "rsync",
                "--recursive",
                MIRROR_URI,
                str(destination),
            ],
            check=True,
            capture_output=True,
            env=environment,
            timeout=900,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("same-region Git mirror replay download failed") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if (
        Path(sys.executable) != PYTHON
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or dict(os.environ) != EXPECTED_ENVIRONMENT
        or sha256(_snapshot_regular(PYTHON)).hexdigest() != PYTHON_SHA256
    ):
        raise RuntimeError("same-region Git mirror verifier runtime drifted")
    _verify_running_source(
        arguments.expected_code_hash, arguments.expected_source_sha256
    )
    remote_evidence = verify_origin_remote_pin(
        WORKTREE,
        arguments.expected_code_hash,
        branch=BRANCH,
    )
    with tempfile.TemporaryDirectory(
        prefix="gate-d-git-mirror-", dir=TEMP_ROOT
    ) as root:
        mirror_git_dir = Path(root) / "git"
        _download_mirror(mirror_git_dir)
        evidence = verify_mirror_directory(
            mirror_git_dir,
            WORKTREE,
            arguments.expected_code_hash,
            branch=BRANCH,
            origin=ORIGIN,
            bound_paths=BOUND_PATHS,
        )
        evidence.update(remote_evidence)
    print(
        json.dumps(
            evidence,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
