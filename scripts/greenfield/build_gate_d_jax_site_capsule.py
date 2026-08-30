#!/usr/bin/env python3
"""Build the minimal JAX CPU-lowering site tree for trusted provisioning.

This stdlib-only builder copies an explicit allowlist from the pinned venv,
omits bytecode caches, writes a source manifest, and prints the complete
canonical-safe-mode tree hash consumed by the fixed root provisioner. It does
not import JAX, initialize a backend, lower, compile, or execute anything.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import struct
from typing import Any


SOURCE_ROOT = Path("/home/gianl/vllm-env/lib/python3.12/site-packages")
ALLOWLIST = (
    "absl",
    "absl_py-2.4.0.dist-info",
    "cloudpickle",
    "cloudpickle-3.1.2.dist-info",
    "jax",
    "jax-0.10.1.dist-info",
    "jaxlib",
    "jaxlib-0.10.1.dist-info",
    "ml_dtypes",
    "ml_dtypes-0.5.4.dist-info",
    "numpy",
    "numpy-2.3.5.dist-info",
    "numpy.libs",
    "opt_einsum",
    "opt_einsum-3.4.0.dist-info",
    "packaging",
    "packaging-26.2.dist-info",
    "scipy",
    "scipy-1.17.1.dist-info",
    "scipy.libs",
    "typing_extensions.py",
    "typing_extensions-4.15.0.dist-info",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = sha256()
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(_sha_file(entry))
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise SystemExit(f"site-capsule symlink escapes root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise SystemExit(f"unsupported site-capsule entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", stat.S_IMODE(metadata.st_mode) & ~0o7022))
        digest.update(payload)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    if path.is_dir() and not path.is_symlink():
        return {"kind": "directory", "tree_sha256": _tree_sha256(path)}
    if path.is_file() and not path.is_symlink():
        return {
            "bytes": path.stat().st_size,
            "kind": "file",
            "sha256": _sha_file(path),
        }
    raise SystemExit(f"allowlisted dependency is unsupported: {path}")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o444,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_future_unprivileged_access(root: Path) -> None:
    """Require access that survives the provisioner's root:root ownership seal."""

    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            if mode & 0o005 != 0o005:
                raise SystemExit(
                    f"site-capsule directory will be unreadable after provisioning: {path}"
                )
        elif stat.S_ISREG(metadata.st_mode):
            if not mode & stat.S_IROTH:
                raise SystemExit(
                    f"site-capsule file will be unreadable after provisioning: {path}"
                )
        elif not stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"unsupported site-capsule entry: {path}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.geteuid() == 0:
        raise SystemExit("site-capsule builder must run unprivileged")
    output = Path(os.path.abspath(arguments.output))
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise SystemExit("site-capsule output parent is unsafe")
    sources = {name: _source_record(SOURCE_ROOT / name) for name in ALLOWLIST}
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ALLOWLIST:
        source = SOURCE_ROOT / name
        destination = output / name
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True, ignore=ignore)
        else:
            shutil.copy2(source, destination, follow_symlinks=False)
    if any(path.name == "__pycache__" or path.suffix == ".pyc" for path in output.rglob("*")):
        raise SystemExit("site capsule unexpectedly contains bytecode")
    if {name: _source_record(SOURCE_ROOT / name) for name in ALLOWLIST} != sources:
        raise SystemExit("site dependencies changed during capsule build")
    manifest = {
        "allowlist": list(ALLOWLIST),
        "claim_scope": "Copied dependency bytes only; no JAX import/backend/lowering/compile/execute claim.",
        "schema_version": 1,
        "source_entries": sources,
        "source_root": str(SOURCE_ROOT),
    }
    _write_exclusive(
        output / "CAPSULE_MANIFEST.json",
        (_canonical(manifest) + "\n").encode("ascii"),
    )
    _verify_future_unprivileged_access(output)
    result = {
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "files": sum(1 for path in output.rglob("*") if path.is_file()),
        "output": str(output),
        "tree_sha256": _tree_sha256(output),
    }
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
