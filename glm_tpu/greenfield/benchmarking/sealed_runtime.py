"""Sealed Gate-D execution runtime: immutable interpreter and dependency sites.

Reproduces the accepted acquisition drivers' provenance boundary for a bounded
TPU probe: the root-owned immutable Python (``-I -S -B``), the sealed JAX and
libtpu dependency sites verified by tree digest and capsule manifest digest,
an explicit ``sys.path`` containing only the detached committed source and the
sealed sites, and a closure check that every ``glm_tpu`` module was imported
from that source.  No mutable virtualenv is ever on the import path.
"""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any

PYTHON_RUNTIME_ROOT = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
PYTHON = PYTHON_RUNTIME_ROOT / "bin/python3.12"
PYTHON_SHA256 = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
PYTHON_RUNTIME_TREE_SHA256 = "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
JAX_SITE_ROOT = Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e")
JAX_SITE_TREE_SHA256 = "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
JAX_SITE_MANIFEST_SHA256 = "ef454caafd2e4ba5da4bc7f7f73bef4e8f157afd6c795a91b09e319961941eff"
LIBTPU_SITE_ROOT = Path("/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3")
LIBTPU_SITE_TREE_SHA256 = "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca"
LIBTPU_SITE_MANIFEST_SHA256 = "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa"
EXPECTED_RUNTIME_PATH = (
    str(PYTHON_RUNTIME_ROOT / "lib/python312.zip"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12/lib-dynload"),
)


def safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def sha_file(path: Path) -> str:
    """SHA-256 of a regular, non-linked file, refusing changes during the read."""

    digest = sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(f"sealed runtime dependency is unsafe: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns) != (
            metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns
        ):
            raise RuntimeError(f"sealed runtime dependency changed: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def runtime_tree_sha256(root: Path, *, require_sealed: bool) -> str:
    """Order-independent digest of a directory tree (names, modes, sizes, contents, links)."""

    digest = sha256()
    entries = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if require_sealed and (
            metadata.st_uid != 0
            or metadata.st_gid != 0
            or os.listxattr(entry, follow_symlinks=False)
            or (not stat.S_ISLNK(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) != safe_mode(metadata.st_mode))
        ):
            raise RuntimeError(f"sealed runtime entry is not root-owned sealed: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind, payload = b"D", b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(sha_file(entry))
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise RuntimeError(f"sealed runtime symlink escapes root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise RuntimeError(f"unsupported sealed runtime entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", safe_mode(metadata.st_mode)))
        digest.update(payload)
    return digest.hexdigest()


def validate_python_runtime() -> dict[str, str]:
    """Require the immutable interpreter under ``-I -S`` before any project import."""

    if Path(sys.executable) != PYTHON:
        raise RuntimeError("sealed Python executable boundary drifted")
    if sys.flags.isolated != 1 or sys.flags.no_site != 1 or not sys.flags.ignore_environment:
        raise RuntimeError("sealed Python must run with -I -S")
    if sha_file(PYTHON) != PYTHON_SHA256:
        raise RuntimeError("sealed Python executable bytes drifted")
    if runtime_tree_sha256(PYTHON_RUNTIME_ROOT, require_sealed=True) != PYTHON_RUNTIME_TREE_SHA256:
        raise RuntimeError("sealed Python runtime tree drifted")
    return {
        "python_executable": str(PYTHON),
        "python_sha256": PYTHON_SHA256,
        "python_runtime_tree_sha256": PYTHON_RUNTIME_TREE_SHA256,
    }


def validate_dependency_sites() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, root, tree, manifest in (
        ("jax", JAX_SITE_ROOT, JAX_SITE_TREE_SHA256, JAX_SITE_MANIFEST_SHA256),
        ("libtpu", LIBTPU_SITE_ROOT, LIBTPU_SITE_TREE_SHA256, LIBTPU_SITE_MANIFEST_SHA256),
    ):
        if runtime_tree_sha256(root, require_sealed=True) != tree or (
            sha_file(root / "CAPSULE_MANIFEST.json") != manifest
        ):
            raise RuntimeError(f"sealed {label} dependency site drifted")
        result[label] = {"root": str(root), "tree_sha256": tree, "manifest_sha256": manifest}
    return result


def install_sealed_source_path(source_root: Path) -> tuple[str, ...]:
    """Replace ``sys.path`` with the detached source and the sealed sites only."""

    expected = (str(source_root), str(JAX_SITE_ROOT), str(LIBTPU_SITE_ROOT), *EXPECTED_RUNTIME_PATH)
    sys.path[:] = list(expected)
    return expected


def verify_import_closure(source_root: Path) -> list[dict[str, str]]:
    """Every imported ``glm_tpu`` module must live under the detached source."""

    prefix = str(source_root) + "/"
    records = []
    for name, module in sorted(sys.modules.items()):
        if name != "glm_tpu" and not name.startswith("glm_tpu."):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path.startswith(prefix):
            raise RuntimeError(f"mutable project import escaped the sealed source: {name}")
        records.append({"module": name, "path": raw_path})
    if not records:
        raise RuntimeError("sealed project import closure is empty")
    return records


def runtime_provenance(source_root: Path) -> dict[str, Any]:
    return {
        "python": validate_python_runtime(),
        "sites": validate_dependency_sites(),
        "sys_path": list(sys.path),
        "source_root": str(source_root),
    }
