#!/usr/bin/env python3
"""Build the minimal libtpu supplement for trusted Gate-D provisioning.

The existing sealed JAX site deliberately omits TPU runtime packages because it
was built for CPU lowering.  This stdlib-only builder copies exactly the
``libtpu`` package and its distribution metadata from the pinned compiler venv,
omits bytecode, writes a source manifest, and reports the complete canonical
safe-mode tree hash consumed by the fixed root-owned provisioner.  It does not
import JAX or libtpu, initialize a backend, lower, compile, execute, or use the
network.
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
    "libtpu",
    "libtpu-0.0.41.dist-info",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_file(path: Path) -> str:
    digest = sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SystemExit(f"libtpu source file is unsafe: {path}")
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        final = os.fstat(descriptor)
        if (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise SystemExit(f"libtpu source file changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


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
            payload = struct.pack(">Q", metadata.st_size) + bytes.fromhex(
                _sha_file(entry)
            )
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"L"
            target = os.readlink(entry).encode("utf-8")
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise SystemExit(
                    f"libtpu capsule symlink escapes root: {entry}"
                ) from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise SystemExit(f"unsupported libtpu capsule entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", stat.S_IMODE(metadata.st_mode) & ~0o7022))
        digest.update(payload)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    if path.is_dir() and not path.is_symlink():
        return {"kind": "directory", "tree_sha256": _tree_sha256(path)}
    raise SystemExit(f"allowlisted libtpu dependency is unsupported: {path}")


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
            if written <= 0:
                raise SystemExit("libtpu capsule manifest write stalled")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_future_unprivileged_access(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            if mode & 0o005 != 0o005:
                raise SystemExit(
                    f"libtpu capsule directory will be unreadable: {path}"
                )
        elif stat.S_ISREG(metadata.st_mode):
            if not mode & stat.S_IROTH:
                raise SystemExit(f"libtpu capsule file will be unreadable: {path}")
        elif not stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"unsupported libtpu capsule entry: {path}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.geteuid() == 0:
        raise SystemExit("libtpu capsule builder must run unprivileged")
    output = Path(os.path.abspath(arguments.output))
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise SystemExit("libtpu capsule output parent is unsafe")
    sources = {name: _source_record(SOURCE_ROOT / name) for name in ALLOWLIST}
    output.mkdir(mode=0o755, parents=False, exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ALLOWLIST:
        shutil.copytree(
            SOURCE_ROOT / name,
            output / name,
            symlinks=True,
            ignore=ignore,
        )
    if any(
        path.name == "__pycache__" or path.suffix == ".pyc"
        for path in output.rglob("*")
    ):
        raise SystemExit("libtpu capsule unexpectedly contains bytecode")
    if {name: _source_record(SOURCE_ROOT / name) for name in ALLOWLIST} != sources:
        raise SystemExit("libtpu dependencies changed during capsule build")
    manifest = {
        "allowlist": list(ALLOWLIST),
        "claim_scope": (
            "Copied libtpu dependency bytes only; no import, backend, lowering, "
            "compile, execution, cloud, network, numerical, performance or "
            "Gate-D claim."
        ),
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
        "bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
        "files": sum(1 for path in output.rglob("*") if path.is_file()),
        "output": str(output),
        "tree_sha256": _tree_sha256(output),
    }
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
