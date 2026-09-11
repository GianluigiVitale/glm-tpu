#!/usr/bin/env python3
"""Evict the reviewed DB602/DB609 rank1..7 collected compiler copies only.

Dry run is the default. Both existing leases cover verification and eviction.
Apply verifies all files first, then rechecks each file and its cloud generation
immediately before its descriptor-relative unlink. A durable receipt records
each unlink intent and outcome, including partial failure. An interrupted intent
is deliberately ambiguous: inspect the named local file before recovery.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager, ExitStack
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.microbench_fp8_matmul import _atomic_json

WORKLOAD_LEASE = Path("/home/gianl/glm-run/.glm_pod_workload.lock")
RSYNC_LEASE = Path("/home/gianl/.glm-tpu-rsync.lock")
LOCAL_ROOT = Path("/home/gianl/glm-run")
BUCKET = "driftbench-dsv4-uc"
TAGS = {
    602: "greenfield_fp8_ws32_prefill_rolled_model_compile_20260909T054150449575877Z",
    609: "greenfield_fp8_ws32_prefill_canonical_model_compile_20260909T202846089848089Z",
}
FILENAMES = frozenset((
    "compile_journal.jsonl", "prefill_chunk.optimized_hlo.txt",
    "prefill_chunk.stablehlo.mlir", "prefill_tail.optimized_hlo.txt",
    "prefill_tail.stablehlo.mlir", "runner.json", "worker.log",
))
TOTAL_BYTES = 3_411_885_438


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if (manifest.get("artifact_kind") != "ws32_local_collected_copy_eviction_review_v1"
            or manifest.get("status") != "VERIFIED_NOT_DELETED"
            or manifest.get("problems") != []
            or manifest.get("cloud_objects_deleted") != 0):
        raise ValueError("manifest is not a clean verify-only review")
    entries = manifest["files"]
    expected = {
        str(LOCAL_ROOT / tag / "fleet" / f"rank{rank}" / name)
        for tag in TAGS.values() for rank in range(1, 8) for name in FILENAMES
    }
    if (len(entries) != 98 or {e["path"] for e in entries} != expected
            or sum(e["size"] for e in entries) != TOTAL_BYTES
            or manifest["total_bytes"] != TOTAL_BYTES):
        raise ValueError("manifest must name exactly the 98 reviewed local copies")
    for entry in entries:
        db, rank = entry["db"], entry["rank"]
        if type(db) is not int or db not in TAGS or type(rank) is not int or rank not in range(1, 8):
            raise ValueError("unreviewed DB or rank")
        path = Path(entry["path"])
        local = LOCAL_ROOT / TAGS[db] / "fleet" / f"rank{rank}" / path.name
        name = f"results/{TAGS[db]}/workers/rank{rank}/{path.name}"
        if (str(path) != entry["path"] or path != local or path.name not in FILENAMES
                or entry["name"] != name
                or entry["restore_uri"] != f"gs://{BUCKET}/{name}#{entry['generation']}"):
            raise ValueError("local/cloud scope or restore mapping differs")
        for key in ("inode", "size", "mtime_ns", "nlink"):
            if type(entry[key]) is not int or entry[key] <= 0:
                raise ValueError(f"invalid reviewed {key}")
        if (entry["nlink"] != 1 or entry["is_symlink"] is not False
                or entry["local_matches_receipt"] is not True
                or entry["cloud_generation_verified"] is not True
                or not re.fullmatch(r"[1-9][0-9]*", entry["generation"])
                or not re.fullmatch(r"[1-9][0-9]*", entry["ledger_source_generation"])
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or len(base64.b64decode(entry["crc32c"], validate=True)) != 4
                or (entry["size"], entry["sha256"], entry["crc32c"])
                != (entry["receipt_size"], entry["receipt_sha256"], entry["receipt_crc32c"])):
            raise ValueError("reviewed identity or receipt binding differs")
    return entries


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Anchor every component without following symlinks, including ancestors."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"refusing relative/traversal directory: {path}")
    with ExitStack() as stack:
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, descriptor)
        for part in path.parts[1:]:
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                 dir_fd=descriptor)
            stack.callback(os.close, descriptor)
        yield descriptor


def _identity(st: os.stat_result) -> tuple[int, ...]:
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink


def _require_local(st: os.stat_result, entry: dict[str, Any]) -> None:
    if (not stat.S_ISREG(st.st_mode)
            or (st.st_ino, st.st_size, st.st_mtime_ns, st.st_nlink)
            != (entry["inode"], entry["size"], entry["mtime_ns"], entry["nlink"])):
        raise ValueError(f"local identity changed since review: {entry['path']}")


def _holders(path: Path) -> bool:
    # PSmisc 23.4 does not accept '--' here. Absolute reviewed paths cannot
    # become command options, so do not add an unsupported option separator.
    if not path.is_absolute():
        raise ValueError("fuser requires an absolute reviewed path")
    result = subprocess.run(["fuser", str(path)], capture_output=True,
                            text=True, timeout=30)
    if result.returncode == 0:
        return True
    if result.returncode != 1 or result.stdout or result.stderr:
        raise RuntimeError(f"fuser could not prove no holders: {path}: "
                           f"rc={result.returncode} {result.stderr.strip()}")
    return False


@contextmanager
def _verified_file(entry: dict[str, Any], bucket: Any) -> Iterator[int]:
    import google_crc32c

    path = Path(entry["path"])
    with _directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require_local(before, entry)
            digest, crc = sha256(), google_crc32c.Checksum()
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                crc.update(chunk)
            if (_identity(os.fstat(stream.fileno())) != _identity(before)
                    or digest.hexdigest() != entry["sha256"]
                    or base64.b64encode(crc.digest()).decode("ascii") != entry["crc32c"]):
                raise ValueError(f"local bytes changed since review: {path}")
        generation = int(entry["generation"])
        blob = bucket.blob(entry["name"], generation=generation)
        blob.reload(if_generation_match=generation)
        if (int(blob.generation) != generation or int(blob.size) != entry["size"]
                or blob.crc32c != entry["crc32c"]):
            raise ValueError(f"cloud original differs from review: {entry['name']}")
        # Our reader must be closed before fuser, or it would see our own fd.
        if _holders(path):
            raise ValueError(f"file is held open: {path}")
        with _directory(path.parent) as current:
            if ((os.fstat(parent).st_dev, os.fstat(parent).st_ino)
                    != (os.fstat(current).st_dev, os.fstat(current).st_ino)):
                raise ValueError(f"parent directory changed: {path}")
        if _identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) != _identity(before):
            raise ValueError(f"identity changed immediately before unlink: {path}")
        yield parent


@contextmanager
def _leases() -> Iterator[None]:
    with ExitStack() as stack:
        for path in (WORKLOAD_LEASE, RSYNC_LEASE):
            parent = stack.enter_context(_directory(path.parent))
            descriptor = os.open(path.name, os.O_RDWR | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, descriptor)
            st = os.fstat(descriptor)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError(f"invalid lease file: {path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _durable_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Reuse atomic JSON publication, adding exclusive scratch and fsyncs."""
    deleted = [row for row in receipt["files"] if row["state"] == "DELETED"]
    receipt["local_files_deleted"] = len(deleted)
    receipt["local_payload_bytes_removed"] = sum(row["size"] for row in deleted)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    inode = os.fstat(descriptor).st_ino
    os.close(descriptor)
    try:
        _atomic_json(path, receipt)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        # Only our exact scratch inode; never remove a pre-existing temporary.
        try:
            st = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if st.st_ino == inode and stat.S_ISREG(st.st_mode):
                temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="unlink; default is a dry run")
    args = parser.parse_args(argv)
    raw = args.manifest.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != args.manifest_sha256:
        raise ValueError("manifest digest differs from the reviewed one")
    entries = _validate_manifest(json.loads(raw))
    from google.cloud import storage

    with _leases(), _directory(args.receipt.absolute().parent) as receipt_parent:
        # Reserve a fresh receipt before any cloud request or candidate unlink.
        descriptor = os.open(args.receipt.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=receipt_parent)
        os.close(descriptor)
        receipt_path = Path(f"/proc/self/fd/{receipt_parent}") / args.receipt.name
        receipt: dict[str, Any] = dict(
            checks="Exact reviewed scope; both leases; per-file inode/size/mtime/nlink, "
                   "streamed SHA256/CRC32C, cloud generation/size/CRC and fuser rechecks; "
                   "descriptor-relative unlink with no symlink ancestors.",
            cloud_objects_deleted=0,
            free_bytes_before=shutil.disk_usage(LOCAL_ROOT).free,
            local_files_deleted=0,
            local_files_planned=len(entries),
            local_payload_bytes_removed=0,
            manifest_path=str(args.manifest), manifest_sha256=digest,
            prior_attempt=None,
            recovery="Cloud originals retained. Restore exact generation URIs in manifest. "
                     "An interrupted UNLINK_PENDING entry requires inspecting its local path.",
            files=[dict(path=e["path"], size=e["size"], restore_uri=e["restore_uri"], state="PLANNED") for e in entries],
            status="VERIFYING", utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        _durable_receipt(receipt_path, receipt)
        try:
            bucket = storage.Client().bucket(BUCKET)
            bucket.reload()
            if str(bucket.location).upper() != "US-CENTRAL2":
                raise ValueError("bucket is not US-CENTRAL2")
            for entry, row in zip(entries, receipt["files"], strict=True):
                with _verified_file(entry, bucket):
                    row["state"] = "VERIFIED"
            _durable_receipt(receipt_path, receipt)
            if args.apply:
                for entry, row in zip(entries, receipt["files"], strict=True):
                    row["state"] = "UNLINK_PENDING"
                    _durable_receipt(receipt_path, receipt)
                    with _verified_file(entry, bucket) as parent:
                        os.unlink(Path(entry["path"]).name, dir_fd=parent)
                        row["state"] = "DELETED"
                        os.fsync(parent)
                    _durable_receipt(receipt_path, receipt)
            receipt["status"] = "LOCAL_COPIES_EVICTED" if args.apply else "DRY_RUN_VERIFIED_NOT_DELETED"
        except BaseException as error:
            pending = [row["path"] for row in receipt["files"] if row["state"] == "UNLINK_PENDING"]
            confirmed = any(row["state"] == "DELETED" for row in receipt["files"])
            receipt["unconfirmed_unlink_paths"] = pending
            if pending:
                receipt["status"] = "FAILED_PARTIAL_UNCERTAIN" if confirmed else "FAILED_UNCERTAIN"
            else:
                receipt["status"] = "FAILED_PARTIAL" if confirmed else "FAILED_NOT_DELETED"
            receipt["deletion_counts_are_confirmed_only"] = True
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            receipt["free_bytes_after"] = shutil.disk_usage(LOCAL_ROOT).free
            _durable_receipt(receipt_path, receipt)
        print(json.dumps({k: receipt[k] for k in ("status", "local_files_planned", "local_files_deleted",
                                                 "local_payload_bytes_removed", "free_bytes_after")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
