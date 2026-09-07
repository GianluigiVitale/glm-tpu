#!/usr/bin/env python3
"""Collect ORIGINAL files from an ended WS32 worker after controller loss.

Invoke on all hosts for inventory first. Publication requires a reviewed inventory
digest for that host and the operator's protected 8/8 zero-work census under both
leases. This tool additionally refuses local live runners/libtpu holders. It never
creates numerical records, SUCCESS, completion markers, or a correctness verdict.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile

import google_crc32c


GRAPHS = ("exact_materialize", "exact_promote", "prefill_chunk", "prefill_tail",
          "observer", "decode", "cache_probe")
FORMS = (("stablehlo.mlir", "stablehlo_sha256"), ("optimized_hlo.txt", "optimized_hlo_sha256"))
BUCKET = "driftbench-dsv4-uc"


def digest_file(path: Path) -> dict:
    """Hash one closed original file and refuse modification while reading."""
    digest, crc = sha256(), google_crc32c.Checksum()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            crc.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("original file changed while hashing: " + str(path))
    return dict(size=after.st_size, sha256=digest.hexdigest(),
                crc32c=base64.b64encode(crc.digest()).decode())


def original_inventory(root: Path, tag: str, pin: str, rank: int, capture: dict) -> dict:
    """Validate original file bindings and declare only the sealer's object names."""
    record_path = root / f"runner.rank{rank}.json"
    original_json = record_path.read_bytes()
    record = json.loads(original_json)
    if (record["code_hash"] != pin or record["launch_process_id"] != rank
            or record["jax_process_index"] != capture["jax_process_index"]
            or record["hostname"] != capture["hostname"]
            or record["status"] != "SUCCESS" or record["compile_only"] is not False
            or record["artifact_kind"] != "greenfield_ws32_short_decoder"
            or record["evidence_layout"] != "hlo_single_gzip_v2"
            or record["exact_dsa"] is not True or set(record["graphs"]) != set(GRAPHS)):
        raise ValueError("original numerical record identity/layout mismatch")
    files = []

    def add(relative: str, remote: str, expected_sha: str | None = None,
            expected_size: int | None = None, compressed: bool = False) -> None:
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("original evidence is missing or escapes run directory: " + relative)
        facts = digest_file(path)
        if ((expected_sha is not None and facts["sha256"] != expected_sha)
                or (expected_size is not None and facts["size"] != expected_size)):
            raise ValueError("original evidence disagrees with runner record: " + relative)
        files.append(dict(local=relative, remote=remote, compressed=compressed, **facts))

    add(record_path.name, "host_records/" + record_path.name, sha256(original_json).hexdigest())
    tensor = record["numerical_tensors"]
    if tensor["filename"] != f"runner.rank{rank}.npz":
        raise ValueError("numerical filename drifted")
    add(tensor["filename"], "host_records/" + tensor["filename"], tensor["sha256"], tensor["byte_count"])
    add(f"runner.rank{rank}.log", f"host_records/runner.rank{rank}.log")
    traces = record["trace"]["files"]
    if len(traces) != 1:
        raise ValueError("one original trace required")
    trace = traces[0]
    trace_path = Path("trace") / trace["relative_path"]
    if trace_path.is_absolute() or ".." in trace_path.parts:
        raise ValueError("trace path escapes run directory")
    add(str(trace_path), f"traces/trace.rank{rank}.xplane.pb", trace["sha256"], trace["byte_count"])
    for graph in GRAPHS:
        for suffix, key in FORMS:
            relative = f"hlo/{graph}.{suffix}"
            add(relative, relative + ".gz", record["graphs"][graph][key], compressed=True)
    inventory = dict(tag=tag, code_hash=pin, rank=rank, files=files)
    inventory["inventory_sha256"] = sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return inventory


def require_fleet_inventories(rows: list[dict], tag: str, pin: str) -> None:
    """Before any shared publication, require eight inventories of the same HLO."""
    if len(rows) != 8 or {row["rank"] for row in rows} != set(range(8)):
        raise ValueError("eight unique inventory ranks required")
    shared = None
    for row in rows:
        if row["tag"] != tag or row["code_hash"] != pin:
            raise ValueError("fleet inventory run identity drifted")
        unsigned = {k: v for k, v in row.items() if k != "inventory_sha256"}
        if sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != row["inventory_sha256"]:
            raise ValueError("fleet inventory digest drifted")
        graphs = {item["remote"]: (item["size"], item["sha256"]) for item in row["files"] if item["compressed"]}
        expected_graphs = {f"hlo/{g}.{s}.gz" for g in GRAPHS for s, _ in FORMS}
        if set(graphs) != expected_graphs or len(row["files"]) != 18:
            raise ValueError("fleet inventory object set drifted")
        if shared is not None and graphs != shared:
            raise ValueError("ranks disagree on shared HLO")
        shared = graphs


def require_local_idle() -> None:
    """This is a local guard; the outer workflow still needs its eight-host census."""
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            argv = (path / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if any(arg.endswith(b"/run_short_decoder_ws32.py") for arg in argv[1:3]):
            raise ValueError("WS32 runner remains live; no collection")
    result = subprocess.run(["sudo", "-n", "fuser", "/tmp/libtpu_lockfile"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode != 1 or result.stdout.strip() or result.stderr.strip():
        raise ValueError("libtpu idle state is not established")


def publish_exact(bucket, name: str, source: Path, expected: dict, *, compressed: bool) -> dict:
    """Create absent payloads conditionally; generation-readback every payload.

    Shared HLO is compared by its inflated bytes because the original uploader's
    gzip container may differ while carrying the identical authenticated graph.
    """
    import gzip
    from google.api_core.exceptions import PreconditionFailed

    blob = bucket.get_blob(name)
    if blob is None:
        with source.open("rb") as stream, tempfile.TemporaryFile() as packed:
            if digest_file(source) != expected:
                raise ValueError("source changed since inventory")
            if compressed:
                with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0, compresslevel=9) as compressor:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        compressor.write(block)
                packed.seek(0)
                upload = packed
            else:
                upload = stream
            target = bucket.blob(name)
            try:
                target.upload_from_file(upload, if_generation_match=0, checksum="crc32c")
            except PreconditionFailed:
                pass  # A concurrent shared-HLO upload is accepted only by readback below.
        blob = bucket.get_blob(name)
    if blob is None or not blob.generation:
        raise ValueError("published object absent")
    limit = expected["size"] + max(65536, expected["size"] // 100) if compressed else expected["size"]
    if int(blob.size) > limit or (not compressed and int(blob.size) != expected["size"]):
        raise ValueError("remote object size differs from original payload budget")
    # Read only this generation, never whatever a mutable name resolves to later.
    pinned = bucket.blob(name, generation=int(blob.generation))
    with tempfile.TemporaryFile() as downloaded:
        pinned.download_to_file(downloaded, if_generation_match=int(blob.generation), checksum="crc32c")
        stored_bytes = downloaded.tell()
        downloaded.seek(0)
        stored_crc = google_crc32c.Checksum()
        for block in iter(lambda: downloaded.read(8 * 1024 * 1024), b""):
            stored_crc.update(block)
        if stored_bytes != int(blob.size) or base64.b64encode(stored_crc.digest()).decode() != blob.crc32c:
            raise ValueError("generation readback size/CRC mismatch")
        downloaded.seek(0)
        reader = gzip.GzipFile(fileobj=downloaded, mode="rb") if compressed else downloaded
        digest, size = sha256(), 0
        while block := reader.read(min(8 * 1024 * 1024, expected["size"] - size + 1)):
            size += len(block)
            if size > expected["size"]:
                raise ValueError("remote payload exceeds original size")
            digest.update(block)
        if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError("existing remote payload differs from original")
    return dict(name=name, generation=str(blob.generation), size=int(blob.size),
                crc32c=blob.crc32c, original_sha256=expected["sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--topology-capture-sha256", required=True)
    parser.add_argument("--publish-inventory-sha256", help="default off; exact prior local inventory required")
    args = parser.parse_args()
    if (not re.fullmatch(r"greenfield_ws32_short_decoder_[a-z0-9_]+_[0-9]{8}T[0-9]{15}Z", args.tag)
            or not re.fullmatch(r"[0-9a-f]{40}", args.code_hash)):
        parser.error("invalid tag/pin")
    require_local_idle()
    rank = int(socket.gethostname().rsplit("-w-", 1)[1])
    if rank not in range(8):
        parser.error("not an eight-host worker")
    root = Path("/home/gianl/glm-run") / args.tag
    capture_path = Path("/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records") / f"topology.rank{rank}.json"
    capture_bytes = capture_path.read_bytes()
    if sha256(capture_bytes).hexdigest() != args.topology_capture_sha256:
        parser.error("topology capture digest drifted")
    capture = json.loads(capture_bytes)
    if capture["hostname"] != socket.gethostname():
        parser.error("topology capture hostname differs from this worker")
    inventory = original_inventory(root, args.tag, args.code_hash, rank, capture)
    if args.publish_inventory_sha256:
        if inventory["inventory_sha256"] != args.publish_inventory_sha256:
            parser.error("inventory changed since review")
        from google.cloud import storage
        bucket = storage.Client().bucket(BUCKET)
        if bucket.get_blob(f"results/{args.tag}/SUCCESS") is not None:
            parser.error("terminal SUCCESS already exists; no publication")
        receipts = []
        for item in inventory["files"]:
            require_local_idle()
            facts = {key: item[key] for key in ("size", "sha256", "crc32c")}
            receipts.append(publish_exact(bucket, f"results/{args.tag}/{item['remote']}",
                                          root / item["local"], facts, compressed=item["compressed"]))
        inventory["published_objects"] = receipts
    print("WS32_COLLECT " + json.dumps(inventory, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
