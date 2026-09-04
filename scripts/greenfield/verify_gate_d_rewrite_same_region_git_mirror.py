#!/usr/bin/env python3
"""Replay and authenticate the rewrite branch from the US-CENTRAL2 Git mirror."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import types
from hashlib import sha256
from pathlib import Path

WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
BRANCH = "rewrite/topology-first-decode"
ORIGIN = "git@github.com:GianluigiVitale/glm-tpu.git"
SOURCE_PATH = "scripts/greenfield/verify_gate_d_rewrite_same_region_git_mirror.py"
INSTALL_PATH = Path(
    "/usr/local/libexec/glm-tpu/gate-d-m2048-strategy-nd-v2/"
    "verify_gate_d_rewrite_same_region_git_mirror.py"
)
BASE_PATH = "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
BASE_PIN = "986378238ac6458307aea69ef1f5e12bf82bc020"
BASE_SHA256 = "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b"
BOUND_PATHS = (
    "configs/greenfield-reuse-inventory.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v2-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v3-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v4-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v5-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v6-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v7-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v8-source.json",
    "docs/artifacts/gate-d-m2048-strategy-nd-v9-source.json",
    "docs/artifacts/gate-d-m2048-v2-install-repository-prestate-failure.json",
    "docs/artifacts/gate-d-m2048-v3-install-loader-quoting-failure.json",
    "docs/artifacts/gate-d-m2048-v4-install-runtime-loader-quoting-failure.json",
    "docs/artifacts/gate-d-m2048-v5-install-missing-libexec-parent-failure.json",
    "docs/artifacts/gate-d-m2048-v6-install-scp-symlink-dereference-failure.json",
    "docs/artifacts/gate-d-m2048-v7-missing-requests-auto-detection-failure.json",
    "docs/artifacts/gate-d-m2048-v8-zero-retention-preflight-incompatibility.json",
    "docs/greenfield/REUSE_INVENTORY.md",
    "glm_tpu/greenfield/benchmarking/__init__.py",
    "glm_tpu/greenfield/benchmarking/m2048_association_fingerprint.py",
    "scripts/greenfield/bootstrap_gate_d_provisioner.py",
    "scripts/greenfield/install_gate_d_m2048_strategy_nd_fleet.sh",
    "scripts/greenfield/install_gate_d_m2048_strategy_nd_runtime.py",
    "scripts/greenfield/launch_gate_d_m2048_strategy_nd_association.py",
    "scripts/greenfield/probe_m2048_strategy_nd_association.py",
    "scripts/greenfield/provision_gate_d_runtime_archive.py",
    "scripts/greenfield/publish_gate_d_m2048_strategy_nd_association.py",
    "scripts/greenfield/refresh_gate_d_m2048_worker_repository.py",
    "scripts/greenfield/run_gate_d_m2048_strategy_nd_association.sh",
    SOURCE_PATH,
    "tests/greenfield/benchmarking/test_m2048_association_fingerprint.py",
    "tests/greenfield/validation/test_m2048_strategy_nd_protected_harness.py",
)
GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
        env=GIT_ENVIRONMENT,
        stderr=subprocess.PIPE,
        timeout=60,
    )


def _snapshot(path: Path, *, root_owned: bool) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                root_owned
                and (
                    before.st_uid != 0
                    or before.st_gid != 0
                    or stat.S_IMODE(before.st_mode) != 0o555
                    or os.listxattr(path, follow_symlinks=False)
                )
            )
        ):
            raise RuntimeError(f"unsafe rewrite mirror verifier source: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
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
            raise RuntimeError(f"rewrite mirror verifier source changed: {path}")
        return raw
    finally:
        os.close(descriptor)


def _require_root_boundary(path: Path) -> None:
    if path != INSTALL_PATH:
        raise RuntimeError("rewrite mirror verifier is outside its immutable capsule")
    for parent in (path.parent, *path.parent.parents):
        metadata = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError(f"unsafe rewrite mirror verifier parent: {parent}")


def _verify_running_source(code_pin: str, expected_sha256: str) -> bytes:
    _require_root_boundary(Path(__file__))
    raw = _snapshot(INSTALL_PATH, root_owned=True)
    if (
        _git_bytes("rev-parse", "HEAD").decode("ascii").strip() != code_pin
        or _git_bytes("rev-parse", f"{code_pin}^{{commit}}").decode("ascii").strip()
        != code_pin
        or _git_bytes("for-each-ref", "--format=%(refname)", "refs/replace")
        or raw != _git_bytes("show", f"{code_pin}:{SOURCE_PATH}")
        or sha256(raw).hexdigest() != expected_sha256
    ):
        raise RuntimeError(
            "rewrite mirror verifier is not the committed immutable blob"
        )
    return raw


def _load_base(code_pin: str) -> types.ModuleType:
    path = WORKTREE / BASE_PATH
    raw = _snapshot(path, root_owned=False)
    if (
        raw != _git_bytes("show", f"{code_pin}:{BASE_PATH}")
        or raw != _git_bytes("show", f"{BASE_PIN}:{BASE_PATH}")
        or sha256(raw).hexdigest() != BASE_SHA256
    ):
        raise RuntimeError("rewrite mirror verifier base bytes drifted")
    module = types.ModuleType("_gate_d_rewrite_mirror_base")
    module.__file__ = str(INSTALL_PATH)
    module.__package__ = None
    exec(compile(raw, str(path), "exec"), module.__dict__)  # noqa: S102
    module.WORKTREE = WORKTREE
    module.BRANCH = BRANCH
    module.ORIGIN = ORIGIN
    module.SOURCE_PATH = SOURCE_PATH
    module.BOUND_PATHS = BOUND_PATHS
    return module


def main() -> int:
    arguments = sys.argv[1:]
    if (
        len(arguments) != 4
        or arguments[0] != "--expected-code-hash"
        or arguments[2] != "--expected-source-sha256"
    ):
        raise RuntimeError("rewrite mirror verifier invocation drifted")
    code_pin = arguments[1]
    _verify_running_source(code_pin, arguments[3])
    return _load_base(code_pin).main()


if __name__ == "__main__":
    raise SystemExit(main())
