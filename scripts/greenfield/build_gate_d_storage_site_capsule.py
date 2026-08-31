#!/usr/bin/env python3
"""Build the pinned Google Cloud Storage publication dependency capsule.

This stdlib-only builder copies an explicit dependency closure from the pinned
development environment.  The result is later installed, without replacement,
as a root-owned read-only tree by the already reviewed Gate-D runtime
provisioner.  Building the tree imports no cloud client and performs no network,
JAX, backend, model, or TPU work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path("/home/gianl/vllm-env/lib/python3.12/site-packages")

# Complete top-level import closure observed for google-cloud-storage 3.10.1
# client construction under Python 3.12, plus each owning distribution's
# metadata.  Namespace packages are copied as complete trees so lazy imports
# during authenticated HTTP operations cannot escape this capsule.
ALLOWLIST = (
    "81d243bd2c585b0f4821__mypyc.cpython-312-x86_64-linux-gnu.so",
    "_cffi_backend.cpython-312-x86_64-linux-gnu.so",
    "cachetools",
    "cachetools-7.1.4.dist-info",
    "certifi",
    "certifi-2026.5.20.dist-info",
    "cffi",
    "cffi-2.0.0.dist-info",
    "charset_normalizer",
    "charset_normalizer-3.4.7.dist-info",
    "cryptography",
    "cryptography-48.0.0.dist-info",
    "google",
    "google_api_core-2.30.3.dist-info",
    "google_auth-2.53.0.dist-info",
    "google_cloud_core-2.6.0.dist-info",
    "google_cloud_storage-3.10.1.dist-info",
    "google_crc32c",
    "google_crc32c-1.8.0.dist-info",
    "google_crc32c.libs",
    "google_resumable_media-2.9.0.dist-info",
    "googleapis_common_protos-1.75.0.dist-info",
    "grpc",
    "grpc_status",
    "grpcio-1.80.0.dist-info",
    "grpcio_status-1.80.0.dist-info",
    "idna",
    "idna-3.17.dist-info",
    "opentelemetry",
    "opentelemetry_api-1.42.1.dist-info",
    "proto",
    "proto_plus-1.28.0.dist-info",
    "protobuf-6.33.6.dist-info",
    "pyasn1",
    "pyasn1-0.6.3.dist-info",
    "pyasn1_modules",
    "pyasn1_modules-0.4.2.dist-info",
    "pycparser",
    "pycparser-3.0.dist-info",
    "requests",
    "requests-2.34.2.dist-info",
    "simplejson",
    "simplejson-4.1.1.dist-info",
    "typing_extensions.py",
    "typing_extensions-4.15.0.dist-info",
    "urllib3",
    "urllib3-2.7.0.dist-info",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


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
                raise SystemExit(f"storage-capsule symlink escapes root: {entry}") from error
            payload = struct.pack(">I", len(target)) + target
        else:
            raise SystemExit(f"unsupported storage-capsule entry: {entry}")
        digest.update(kind)
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", _safe_mode(metadata.st_mode)))
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
    raise SystemExit(f"allowlisted storage dependency is unsupported: {path}")


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
                raise SystemExit("storage-capsule manifest write stalled")
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
                    f"storage-capsule directory will be unreadable after provisioning: {path}"
                )
        elif stat.S_ISREG(metadata.st_mode):
            if not mode & stat.S_IROTH:
                raise SystemExit(
                    f"storage-capsule file will be unreadable after provisioning: {path}"
                )
        elif not stat.S_ISLNK(metadata.st_mode):
            raise SystemExit(f"unsupported storage-capsule entry: {path}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.geteuid() == 0:
        raise SystemExit("storage-capsule builder must run unprivileged")
    output = Path(os.path.abspath(arguments.output))
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise SystemExit("storage-capsule output parent is unsafe")
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
        raise SystemExit("storage capsule unexpectedly contains bytecode")
    if {name: _source_record(SOURCE_ROOT / name) for name in ALLOWLIST} != sources:
        raise SystemExit("storage dependencies changed during capsule build")
    manifest = {
        "allowlist": list(ALLOWLIST),
        "claim_scope": (
            "Pinned Google Cloud Storage publication dependencies only; no network, "
            "JAX, backend, compile, model, or TPU claim."
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
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "files": sum(1 for path in output.rglob("*") if path.is_file()),
        "output": str(output),
        "tree_sha256": _tree_sha256(output),
    }
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
