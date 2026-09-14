"""Bounded native request originals and ended-worker publication/collection.

Reuses conditional creation and generation/CRC/SHA readback. This is transport,
not a quality verdict or permission to create SUCCESS. Partial failures remain
partial; old long-context transport and result prefixes are untouched.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield.ws32_native_benchmark_protocol import canonical
from scripts.greenfield.ws32_native_benchmark_requests import RANK_CAP, RESERVE
from scripts.greenfield.ws32_native_benchmark_observability import (
    TRACE_CAP, TRACE_STORED_CAP, trace_storage_bytes,
)
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact, require_local_idle
from scripts.greenfield.ws32_delivery_phase_transport import _bucket, _download
from glm_tpu.host_paths import _plain_path

SCHEMA = "ws32_native_request_originals_v1"
USER_SCHEMA = "glm_ws32_user_request_originals_v1"
MANIFEST_CAP = 2 << 20
ITEM_LIMITS = {"identity.json":4096, "before_cache.json.gz":2<<20,
    "cache_ready.json.gz":2<<20, "prefill_done.json.gz":2<<20,
    "tokens.jsonl":32<<20, "answer.txt.gz":16<<20, "result.json.gz":4<<20,
    "failure.json":4<<20, "dsa.npz":2<<20, "cache.npz":2<<20, "final_memory.json":64<<10}


def limit(relative: str, rank: int, *, user_request: bool = False) -> int:
    if relative in (f"runner.rank{rank}.json", f"runner.rank{rank}.log", f"ended.rank{rank}.json"):
        return 2 << 20
    parts = Path(relative).parts
    if len(parts) == 3 and parts[0] == f"sessions.rank{rank}" and re.fullmatch(r"item[0-9]{3}", parts[1]):
        if int(parts[1][4:]) >= (1 if user_request else 228) or parts[2] not in ITEM_LIMITS:
            raise ValueError("native request original item/name differs")
        if rank != 0 and parts[2] in ("tokens.jsonl", "answer.txt.gz"):
            raise ValueError("only rank0 may publish raw output")
        return ITEM_LIMITS[parts[2]]
    if (len(parts) >= 2 and parts[0] == f"native_trace.rank{rank}"
            and relative.endswith((".xplane.pb", ".trace.json.gz"))
            and ".." not in parts and not Path(relative).is_absolute()):
        return TRACE_CAP
    raise ValueError("unknown native request original")


def prefix(tag: str, rank: int, *, user_request: bool = False) -> str:
    channel = "user_requests" if user_request else "native_requests"
    return f"results/{tag}/{channel}/rank{rank}/"


def _write_once(path: Path, data: bytes) -> None:
    _plain_path(path)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("existing native original differs; refusing overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as out:
        out.write(data); out.flush(); os.fsync(out.fileno())


def publish(root: Path, tag: str, pin: str, rank: int, client: Any,
            *, user_request: bool = False) -> dict:
    cold._identity(tag, pin, rank, user_request=user_request)
    require_local_idle()
    _plain_path(root)
    if root.name != tag or not (root / f"ended.rank{rank}.json").is_file():
        raise ValueError("native publication requires original ended marker")
    bucket = _bucket(client)
    if bucket.get_blob(f"results/{tag}/SUCCESS") is not None:
        raise ValueError("native terminal SUCCESS exists; publication closed")
    originals = [root / name for name in (f"runner.rank{rank}.json", f"runner.rank{rank}.log", f"ended.rank{rank}.json")
                 if (root / name).exists()]
    for directory in (root / f"sessions.rank{rank}", root / f"native_trace.rank{rank}"):
        _plain_path(directory)
        for path in directory.rglob("*"):
            _plain_path(path)
            if not path.is_dir(): originals.append(path)
    entries, totals = [], dict(requests=0, trace=0)
    for path in sorted(originals):
        _plain_path(path)
        relative = str(path.relative_to(root))
        size = path.stat().st_size
        if not path.is_file() or not 0 <= size <= limit(relative, rank, user_request=user_request):
            raise ValueError("native original missing/oversized")
        # An empty log/token file is a real partial original; publish_exact and
        # collector retain it, never reinterpret it as a completed response.
        group = "trace" if relative.startswith("native_trace.") else "requests"
        totals[group] += size
        if totals["requests"] > RANK_CAP + (6<<20) or totals["trace"] > TRACE_CAP:
            raise ValueError("native original aggregate exceeds rank budget")
        entries.append((path, relative, size))
    # Screen compressed trace storage BEFORE publishing any request object.
    # Keep the original 128MiB/rank regional allowance and whole-run 10GiB cap.
    if sum(trace_storage_bytes(path) for path, relative, _ in entries
           if relative.startswith("native_trace.")) > TRACE_STORED_CAP:
        raise ValueError("native trace compressed storage exceeds rank budget")
    if shutil.disk_usage(root).free < 2*max((s for _,_,s in entries), default=0) + RESERVE:
        raise ValueError("native request publisher reserve insufficient")
    rows = []
    stored_trace = 0
    for path, relative, size in entries:
        facts = digest_file(path)
        if facts["size"] != size: raise ValueError("native original changed during publication")
        compressed = relative.endswith(("tokens.jsonl", ".xplane.pb")) or size == 0
        name = prefix(tag, rank, user_request=user_request) + relative + (".gz" if compressed else "")
        receipt = publish_exact(bucket, name, path, facts, compressed=compressed)
        if relative.startswith("native_trace."):
            stored_trace += receipt["size"]
            if stored_trace > TRACE_STORED_CAP:
                raise ValueError("native published trace exceeds storage budget")
        rows.append(dict(relative_path=relative, original_bytes=size,
                         encoding="gzip" if compressed else "identity", **receipt))
    manifest = dict(schema=USER_SCHEMA if user_request else SCHEMA, tag=tag, code_hash=pin, rank=rank,
                    files=rows, original_bytes=sum(totals.values()), quality_proven=False)
    raw = canonical(manifest)+b"\n"
    if len(raw) > MANIFEST_CAP: raise ValueError("native request manifest exceeds cap")
    local = root / f"native_request_manifest.rank{rank}.json"
    _write_once(local, raw)
    publish_exact(bucket, prefix(tag, rank, user_request=user_request)+"manifest.json", local, digest_file(local), compressed=False)
    return manifest


def collect(root: Path, tag: str, pin: str, rank: int, client: Any, blobs: dict,
            *, user_request: bool = False) -> dict:
    cold._identity(tag, pin, rank, user_request=user_request)
    bucket = _bucket(client)
    _plain_path(root)
    manifest_blob = blobs.get(prefix(tag, rank, user_request=user_request)+"manifest.json")
    manifest_raw = _download(manifest_blob, cap=MANIFEST_CAP)
    value = json.loads(manifest_raw)
    if (set(value) != {"schema","tag","code_hash","rank","files","original_bytes","quality_proven"}
            or value["schema"] != (USER_SCHEMA if user_request else SCHEMA) or value["tag"] != tag or value["code_hash"] != pin
            or type(value["rank"]) is not int or value["rank"] != rank or value["quality_proven"] is not False
            or not isinstance(value["files"], list) or not value["files"]):
        raise ValueError("native request manifest identity differs")
    seen, total, trace_total, trace_stored = set(), 0, 0, 0
    for row in value["files"]:
        if set(row) != {"relative_path","original_bytes","encoding","name","generation","size","crc32c","original_sha256"}:
            raise ValueError("native request manifest row fields differ")
        relative = row["relative_path"]
        if (relative in seen or type(row["original_bytes"]) is not int
                or not 0 <= row["original_bytes"] <= limit(relative, rank, user_request=user_request)
                or row["encoding"] not in ("identity", "gzip")
                or type(row["size"]) is not int or row["size"] < 0
                or row["name"] != prefix(tag, rank, user_request=user_request)+relative+(".gz" if row["encoding"]=="gzip" else "")
                or not re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"])):
            raise ValueError("native request original row scope/size differs")
        seen.add(relative); total += row["original_bytes"]
        if relative.startswith("native_trace."):
            trace_total += row["original_bytes"]
            trace_stored += row["size"]
    if (total != value["original_bytes"] or total-trace_total > RANK_CAP+(6<<20) or trace_total > TRACE_CAP
            or trace_stored > TRACE_STORED_CAP or f"ended.rank{rank}.json" not in seen):
        raise ValueError("native request total/ended original differs")
    expected = {r["name"] for r in value["files"]} | {manifest_blob.name}
    if expected != {name for name in blobs if name.startswith(prefix(tag, rank, user_request=user_request))}:
        raise ValueError("native request remote object inventory differs")
    if shutil.disk_usage(root).free < total + 2*max(r["original_bytes"] for r in value["files"]) + RESERVE:
        raise ValueError("native request collector reserve insufficient")
    for row in value["files"]:
        blob = blobs.get(row["name"])
        if (blob is None or str(blob.generation) != row["generation"]
                or blob.size != row["size"] or blob.crc32c != row["crc32c"]):
            raise ValueError("native request exact object generation differs")
        data = _download(blob, cap=limit(row["relative_path"], rank, user_request=user_request)+(1<<20))
        if row["encoding"] == "gzip":
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                data = stream.read(row["original_bytes"]+1)
        if len(data) != row["original_bytes"] or sha256(data).hexdigest() != row["original_sha256"]:
            raise ValueError("native request original bytes differ")
        _write_once(root / row["relative_path"], data)
    _write_once(root / f"native_request_manifest.rank{rank}.json", manifest_raw)
    return value


def publish_all(root: Path, tag: str, pin: str, rank: int, client: Any) -> dict:
    """Attempt both independent channels; a cold failure must not hide answers."""
    errors = {}
    result = None
    if (root / f"native.rank{rank}").is_dir():
        try:
            cold.publish_rank(run_root=root, tag=tag, pin=pin, rank=rank, client=client)
        except Exception as exc:
            errors["cold"] = f"{type(exc).__name__}: {exc}"
    try:
        result = publish(root, tag, pin, rank, client)
    except Exception as exc:
        errors["requests"] = f"{type(exc).__name__}: {exc}"
    if errors:
        raise RuntimeError("native publication incomplete: " + json.dumps(errors, sort_keys=True))
    return result


def main() -> None:
    from google.cloud import storage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    root = Path("/home/gianl/glm-run") / args.tag
    client = storage.Client()
    require_local_idle()
    result = publish_all(root, args.tag, args.code_hash, args.rank, client)
    print(json.dumps(dict(rank=args.rank, files=len(result["files"]), originals_published=True)), flush=True)


if __name__ == "__main__":
    main()
