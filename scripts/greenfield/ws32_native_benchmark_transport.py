"""Bounded original native cold-worker evidence; no success/performance verdict.

Reuses the protected conditional publisher and exact-generation downloader.
Compiler text is shared across ranks by inflated identity. Partial workers keep
their exact missing-file list; neither a manifest nor successful transport is
permission to execute, a memory-fit verdict or an official benchmark score.
"""
from __future__ import annotations

import gzip
import base64
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping

from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.ws32_delivery_phase_transport import _bucket, _download
from scripts.greenfield.ws32_history_preflight import _plain_path

SCHEMA = "ws32_native_cold_originals_v1"
PROFILE = "ws32_native_sampled_request_v1"
TAG = re.compile(r"greenfield_ws32_native_benchmark_[0-9]{8}T[0-9]{15}Z")
USER_TAG = re.compile(r"greenfield_ws32_user_request_[0-9]{8}T[0-9]{15}Z")
RANK = re.compile(r"native[.]rank([0-7])")
# Retained production optimized texts are ~103MiB prefill/~76MB decode.
# The old blanket64MiB limit would refuse valid graphs after compilation.
# Bound individual roles below and the combined original set (including the
# replacement temporary) here. Shared compressed HLO keeps archive<10GiB.
COLD_CAP = 576 << 20
MANIFEST_CAP = 128 << 10
DISK_RESERVE = 1 << 30
FORMS = ("stablehlo.mlir", "optimized_hlo.txt")
GRAPHS = {
    "wk_decode": "wk", "wk_promote": "wk",
    "exact_materialize": "exact", "exact_promote": "exact",
    "prefill_chunk": "", "prefill_tail": "", "decode": "",
    "observer": "", "cache_probe": "", "cache_init": "",
}


def file_limits() -> dict[str, int]:
    return {
        "runner.json": 16 << 20, "initial_memory.json": 4 << 20,
        "journal.jsonl": 2 << 20, "wk/runner.json": 4 << 20,
        "exact/runner.json": 16 << 20,
        **{f"wk/call_records/call{i:03d}.json": 2 << 20 for i in range(42)},
        **{str(Path(folder) / f"{graph}.{form}"):
           (128 << 20 if graph in ("prefill_chunk", "prefill_tail") and form == "optimized_hlo.txt"
            else 96 << 20 if graph in ("decode", "observer") and form == "optimized_hlo.txt"
            else 64 << 20)
           for graph, folder in GRAPHS.items() for form in FORMS},
    }


def native_root(path: Path) -> Path | None:
    """Route only the new tag/rank namespace; historical writers unchanged."""
    for parent in path.parents:
        if RANK.fullmatch(parent.name) and (
                TAG.fullmatch(parent.parent.name) or USER_TAG.fullmatch(parent.parent.name)):
            return parent
    return None


def require_write_size(path: Path, size: int, *, append: bool = False) -> None:
    """Refuse unknown payloads and full-rank overflow BEFORE replacing originals."""
    root = native_root(path)
    if root is None:
        raise ValueError("native evidence path is outside its registered namespace")
    _plain_path(path)
    limits = file_limits()
    relative = str(path.relative_to(root))
    if relative not in limits or type(size) is not int or size < 0:
        raise ValueError("native evidence write name/size differs")
    previous = path.stat().st_size if path.exists() else 0
    target = size + (previous if append else 0)
    if target > limits[relative]:
        raise ValueError("native evidence file would exceed byte cap")
    # Account the original AND replacement temporary. Read metadata only, not
    # checkpoint payload or file contents. Unknown files are rejected as well.
    used = 0
    for existing in root.rglob("*"):
        _plain_path(existing)
        if existing.is_file():
            name = str(existing.relative_to(root))
            if name not in limits:
                raise ValueError("native cold evidence contains an unregistered file")
            used += existing.stat().st_size
    if used + size > COLD_CAP:
        raise ValueError("native cold evidence exceeds complete-rank byte cap")
    if shutil.disk_usage(root).free < size + DISK_RESERVE:
        raise ValueError("native evidence write would consume disk reserve")


class BoundedJournalStream:
    """Same append/fsync journal, with its exact serialized byte count bounded."""
    def __init__(self, stream: Any):
        self.stream = stream

    def write(self, text: str) -> Any:
        path = Path(self.stream.name)
        if native_root(path) is not None:
            require_write_size(path, len(text.encode()), append=True)
        return self.stream.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def _identity(tag: str, pin: str, rank: int) -> None:
    if (not isinstance(tag, str) or TAG.fullmatch(tag) is None
            or not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None
            or type(rank) is not int or not 0 <= rank < 8):
        raise ValueError("native evidence tag/pin/rank differs")


def prefix(tag: str, rank: int) -> str:
    return f"results/{tag}/native_cold/rank{rank}/"


def object_name(tag: str, rank: int, relative: str) -> str:
    if relative not in file_limits():
        raise ValueError("unregistered native evidence filename")
    if any(relative.endswith("." + form) for form in FORMS):
        return f"results/{tag}/native_cold/hlo/{Path(relative).name}.gz"
    return prefix(tag, rank) + relative + ".gz"


def inventory(root: Path) -> tuple[dict[str, dict], list[str]]:
    _plain_path(root)
    if not root.is_dir():
        raise ValueError("native cold evidence root missing")
    limits = file_limits()
    present = {}
    for path in root.rglob("*"):
        _plain_path(path)
        if path.is_dir():
            continue
        relative = str(path.relative_to(root))
        if (relative not in limits or not path.is_file()
                or not 0 < path.stat().st_size <= limits[relative]):
            raise ValueError("native cold original type/name/size differs")
        present[relative] = digest_file(path)
    if sum(row["size"] for row in present.values()) > COLD_CAP:
        raise ValueError("native cold originals exceed whole-rank byte cap")
    return present, sorted(set(limits) - set(present))


def publish_rank(*, run_root: Path, tag: str, pin: str, rank: int, client: Any) -> dict:
    """Publish an ended worker's originals, never overwrite a completed result.

    Caller authenticates process completion/idle first. Same-byte retries are
    idempotent; any conflicting remote generation remains untouched and refuses.
    No model/checkpoint objects, tarballs or full-size safety copies are created.
    """
    _identity(tag, pin, rank)
    _plain_path(run_root)
    if run_root.name != tag:
        raise ValueError("native run path/tag differs")
    root = run_root / f"native.rank{rank}"
    originals, missing = inventory(root)
    if not originals:
        raise ValueError("native cold worker produced no originals")
    largest = max(row["size"] for row in originals.values())
    if shutil.disk_usage(run_root).free < 2 * (largest + 65536) + DISK_RESERVE:
        raise ValueError("native publisher temporary/readback disk reserve insufficient")
    bucket = _bucket(client)
    if bucket.get_blob(f"results/{tag}/SUCCESS") is not None:
        raise ValueError("native terminal SUCCESS exists; publication closed")
    rows = []
    for relative, facts in sorted(originals.items()):
        receipt = publish_exact(bucket, object_name(tag, rank, relative), root / relative,
                                facts, compressed=True)
        rows.append(dict(relative_path=relative, original_bytes=facts["size"], **receipt))
    value = dict(schema=SCHEMA, profile=PROFILE, tag=tag, code_hash=pin, rank=rank,
        files=rows, missing=missing, original_bytes=sum(v["size"] for v in originals.values()),
        numerical_claim=False, performance_claim=False)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > MANIFEST_CAP:
        raise ValueError("native original manifest exceeds byte cap")
    local = run_root / f"native_cold_manifest.rank{rank}.json"
    _plain_path(local)
    if local.exists():
        if not local.is_file() or local.read_bytes() != raw:
            raise ValueError("native original manifest already differs")
    else:
        with local.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    publish_exact(bucket, prefix(tag, rank) + "manifest.json", local,
                  digest_file(local), compressed=False)
    return value


def validate_manifest(value: Mapping[str, Any], *, tag: str, pin: str, rank: int) -> list[dict]:
    _identity(tag, pin, rank)
    if (set(value) != {"schema", "profile", "tag", "code_hash", "rank", "files", "missing",
                       "original_bytes", "numerical_claim", "performance_claim"}
            or value.get("schema") != SCHEMA or value.get("profile") != PROFILE
            or value.get("tag") != tag or value.get("code_hash") != pin
            or type(value.get("rank")) is not int or value["rank"] != rank
            or value.get("numerical_claim") is not False or value.get("performance_claim") is not False
            or not isinstance(value.get("files"), list) or not value["files"]):
        raise ValueError("native original manifest identity differs")
    limits = file_limits()
    seen, total = set(), 0
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"relative_path", "original_bytes", "name",
                "generation", "size", "crc32c", "original_sha256"}:
            raise ValueError("native original row schema differs")
        relative = row["relative_path"]
        if (relative in seen or relative not in limits
                or row["name"] != object_name(tag, rank, relative)
                or type(row["original_bytes"]) is not int or not 0 < row["original_bytes"] <= limits[relative]
                or type(row["size"]) is not int or not 0 < row["size"] <= row["original_bytes"] + max(65536, row["original_bytes"] // 100)
                or not isinstance(row["generation"], str) or not row["generation"].isdecimal()
                or int(row["generation"]) <= 0 or not isinstance(row["crc32c"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"]) is None):
            raise ValueError("native original object binding differs")
        try:
            crc = base64.b64decode(row["crc32c"], validate=True)
        except Exception as exc:
            raise ValueError("native original CRC is not base64") from exc
        if len(crc) != 4:
            raise ValueError("native original CRC width differs")
        total += row["original_bytes"]
        seen.add(relative)
    if (total > COLD_CAP or type(value.get("original_bytes")) is not int
            or value["original_bytes"] != total or value.get("missing") != sorted(set(limits) - seen)):
        raise ValueError("native original inventory/byte total differs")
    return value["files"]


def collect_rank(*, destination: Path, tag: str, pin: str, rank: int,
                 client: Any, blobs: Mapping[str, Any]) -> dict:
    """Restore bounded originals by exact generation, size, CRC and inflated SHA.

    Shared graphs may also belong to peers; the outer fleet collector checks
    their complete union. This rejects missing/extra objects in this rank's own
    prefix. Complete manifest ≠ cold readiness; replay remains mandatory.
    """
    _identity(tag, pin, rank)
    _plain_path(destination)
    if not destination.is_dir():
        raise ValueError("native collection destination missing")
    _bucket(client)
    manifest_name = prefix(tag, rank) + "manifest.json"
    manifest_blob = blobs.get(manifest_name)
    raw = _download(manifest_blob, cap=MANIFEST_CAP)
    value = json.loads(raw)
    rows = validate_manifest(value, tag=tag, pin=pin, rank=rank)
    rank_names = {row["name"] for row in rows if row["name"].startswith(prefix(tag, rank))}
    if {name for name in blobs if name.startswith(prefix(tag, rank))} != rank_names | {manifest_name}:
        raise ValueError("native rank prefix contains missing/extra objects")
    needed = value["original_bytes"] + 2 * max(row["size"] for row in rows) + DISK_RESERVE
    if shutil.disk_usage(destination).free < needed:
        raise ValueError("native collection disk reserve insufficient")
    for row in rows:
        blob = blobs.get(row["name"])
        if blob is None or (str(blob.generation), int(blob.size), str(blob.crc32c)) != (
                row["generation"], row["size"], row["crc32c"]):
            raise ValueError("native original generation/size/CRC changed")
        packed = _download(blob, cap=row["size"])
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as source:
            data = source.read(row["original_bytes"] + 1)
        if len(data) != row["original_bytes"] or sha256(data).hexdigest() != row["original_sha256"]:
            raise ValueError("native inflated original size/hash differs")
        target = destination / f"native.rank{rank}" / row["relative_path"]
        _plain_path(target)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise ValueError("native existing local original differs")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shared = any(row["relative_path"].endswith("." + form) for form in FORMS)
            original = destination / "native_hlo" / target.name if shared else target
            _plain_path(original)
            original.parent.mkdir(parents=True, exist_ok=True)
            if original.exists():
                if not original.is_file() or original.read_bytes() != data:
                    raise ValueError("native shared compiler original differs between ranks")
            else:
                with original.open("xb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            if shared:
                os.link(original, target)
    return dict(manifest=value, manifest_sha256=sha256(raw).hexdigest(),
                generation=str(manifest_blob.generation), size=len(raw), crc32c=str(manifest_blob.crc32c))


def collect_fleet(*, destination: Path, tag: str, pin: str, client: Any,
                  blobs: Mapping[str, Any]) -> list[dict]:
    """Bind the exact eight-manifest union, including the shared graph namespace."""
    result = [collect_rank(destination=destination, tag=tag, pin=pin, rank=rank,
                          client=client, blobs=blobs) for rank in range(8)]
    expected = {prefix(tag, rank) + "manifest.json" for rank in range(8)}
    expected.update(row["name"] for value in result for row in value["manifest"]["files"])
    if {name for name in blobs if name.startswith(f"results/{tag}/native_cold/")} != expected:
        raise ValueError("native fleet original union contains missing/extra objects")
    return result
