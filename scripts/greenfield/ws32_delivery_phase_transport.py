"""Bounded original long-preparation evidence, not a numerical validator.

Reuse the existing conditional publisher and exact generation readback. Each
JSON/HLO original is gzip-compressed; no tar extraction, tensor archive, weight
copy or change to the historical256MiB failure-diagnostic allowance. The outer
collector/sealer must validate the reconstructed originals before any SUCCESS.
"""
from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
import socket
from typing import Any

import google_crc32c

from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact, require_local_idle
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from glm_tpu.host_paths import _plain_path
from scripts.greenfield.ws32_delivery_runtime import PROFILE, programs

BUCKET = "driftbench-dsv4-uc"
RUN_ROOT = Path("/home/gianl/glm-run")
SCHEMA = "ws32_delivery_phase_originals_v1"
MANIFEST = "manifest.json"
MANIFEST_CAP = 128 << 10
RANK_CAP = 160 << 20
DISK_RESERVE = 1 << 30
WK_GRAPHS = ("wk_decode", "wk_promote")
FORMS = ("stablehlo.mlir", "optimized_hlo.txt")


def file_limits(rank: int) -> dict[str, int]:
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("delivery phase rank differs")
    wk, decode = f"delivery_wk.rank{rank}", f"delivery_decode.rank{rank}"
    return {f"{wk}/runner.json": 4 << 20, f"{decode}/runner.json": 16 << 20,
            **{f"{wk}/{g}.{form}": 8 << 20 for g in WK_GRAPHS for form in FORMS},
            **{f"{wk}/call_records/call{i:03d}.json": 2 << 20 for i in range(42)}}


def require_write_size(path: Path, size: int) -> None:
    """Bound preparation metadata/HLO before the existing writers touch disk.

    Only the two explicit phase directories are routed here by those writers.
    Atomic runner replacement needs at most twice its per-file cap; previous
    originals survive a refusal. This is not a bound on unrelated trace files.
    """
    match = re.fullmatch(r"delivery_(?:wk|decode)\.rank([0-7])", path.parent.name)
    if match is None:
        raise ValueError("delivery writer phase directory differs")
    limit = file_limits(int(match[1])).get(f"{path.parent.name}/{path.name}")
    if limit is None or type(size) is not int or not 0 < size <= limit:
        raise ValueError("delivery preparation exceeds pre-write file budget")
    _plain_path(path)
    if shutil.disk_usage(path.parent).free < size + DISK_RESERVE:
        raise ValueError("delivery preparation write would consume disk reserve")


def _identity(tag: str, pin: str, label: str, rank: int) -> None:
    plan = programs.long_plan(label)
    if (not isinstance(tag, str) or re.fullmatch(
            rf"greenfield_ws32_short_decoder_{label}_numerical_c128_cap{plan.context_capacity}_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_[0-9]{{8}}T[0-9]{{15}}Z", tag) is None
            or not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{40}", pin) is None):
        raise ValueError("delivery phase tag/pin/workload differs")
    file_limits(rank)


def prefix(tag: str, rank: int) -> str:
    return f"results/{tag}/delivery_phase/rank{rank}/"


def _bucket(client: Any) -> Any:
    bucket = client.bucket(BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("delivery phase bucket is not US-CENTRAL2")
    return bucket


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _regular(path: Path, cap: int) -> None:
    _plain_path(path)
    if not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError(f"delivery phase original type/size differs: {path}")


def publish_rank(*, root: Path, tag: str, pin: str, label: str, rank: int,
                 client: Any) -> dict[str, Any]:
    """Publish ended-worker originals, including partial failures; no deletion.

    Caller proves local worker exit and serialized ownership. Missing files are
    explicit and cannot be collected as complete evidence. Unknown files refuse
    before publication rather than getting silently omitted.
    """
    _identity(tag, pin, label, rank)
    _plain_path(root)
    if root != RUN_ROOT / tag or not root.is_dir():
        raise ValueError("delivery phase publication root differs")
    limits = file_limits(rank)
    present = []
    for phase in (f"delivery_wk.rank{rank}", f"delivery_decode.rank{rank}"):
        directory = root / phase
        _plain_path(directory)
        if directory.exists():
            for path in directory.rglob("*"):
                _plain_path(path)
                if path.is_dir():
                    if path != directory / "call_records":
                        raise ValueError("delivery unexpected phase directory")
                else:
                    name = str(path.relative_to(root))
                    if name not in limits:
                        raise ValueError("delivery unexpected phase original")
                    _regular(path, limits[name])
                    present.append(name)
    sizes = {name: (root / name).stat().st_size for name in present}
    if sum(sizes.values()) > RANK_CAP:
        raise ValueError("delivery phase originals exceed rank budget")
    largest = max(sizes.values(), default=0)
    if shutil.disk_usage(root).free < 2 * (largest + max(65536, largest // 100)) + DISK_RESERVE:
        raise ValueError("delivery compression/readback disk reserve insufficient")
    bucket = _bucket(client)
    if bucket.get_blob(f"results/{tag}/SUCCESS") is not None:
        raise ValueError("delivery terminal SUCCESS already exists; no publication")
    rows = []
    for name in sorted(present):
        facts = digest_file(root / name)
        if facts["size"] != sizes[name]:
            raise ValueError("delivery original changed since inventory")
        receipt = publish_exact(bucket, prefix(tag, rank) + name + ".gz",
                                root / name, facts, compressed=True)
        rows.append(dict(relative_path=name, original_bytes=facts["size"], **receipt))
        if any(name.endswith(f"/{g}.{form}") for g in WK_GRAPHS for form in FORMS):
            # Same existing shared gzip layout as the seven model graphs. All
            # ranks must generation-readback equal inflated bytes; no overwrite.
            publish_exact(bucket, f"results/{tag}/hlo/{Path(name).name}.gz",
                          root / name, facts, compressed=True)
    value = dict(schema=SCHEMA, profile=PROFILE, tag=tag, code_hash=pin,
        context_label=label, rank=rank, files=rows, original_bytes=sum(sizes.values()),
        missing=sorted(set(limits) - set(present)), numerical_claim=False)
    local = root / f"delivery_phase_manifest.rank{rank}.json"
    _plain_path(local)
    if local.exists():
        _regular(local, MANIFEST_CAP)
        if json.loads(local.read_bytes()) != value:
            raise ValueError("delivery phase manifest already differs; originals retained")
    else:
        if len(_json(value)) * 2 > MANIFEST_CAP:
            raise ValueError("delivery phase manifest exceeds budget")
        _atomic_json(local, value)
    _regular(local, MANIFEST_CAP)
    publish_exact(bucket, prefix(tag, rank) + MANIFEST, local, digest_file(local), compressed=False)
    return value


def _download(blob: Any, *, cap: int) -> bytes:
    if (blob is None or not blob.generation or not blob.crc32c
            or blob.size is None or not 0 < int(blob.size) <= cap):
        raise ValueError("delivery remote object identity/size differs")
    data = blob.download_as_bytes(if_generation_match=int(blob.generation))
    crc = base64.b64encode(google_crc32c.Checksum(data).digest()).decode()
    if len(data) != int(blob.size) or crc != str(blob.crc32c):
        raise ValueError("delivery generation readback size/CRC differs")
    return data


def validate_manifest(value: Any, *, tag: str, pin: str, label: str, rank: int) -> list[dict]:
    _identity(tag, pin, label, rank)
    fixed = dict(schema=SCHEMA, profile=PROFILE, tag=tag, code_hash=pin,
                 context_label=label, rank=rank, missing=[], numerical_claim=False)
    if (not isinstance(value, dict) or set(value) != set(fixed) | {"files", "original_bytes"}
            or any(type(value.get(k)) is not type(v) or value.get(k) != v for k, v in fixed.items())
            or not isinstance(value.get("files"), list)):
        raise ValueError("delivery manifest incomplete or identity differs")
    limits, seen, total = file_limits(rank), set(), 0
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"relative_path", "original_bytes", "name", "generation", "size", "crc32c", "original_sha256"}:
            raise ValueError("delivery receipt schema differs")
        name = row["relative_path"]
        if (not isinstance(name, str) or name not in limits or name in seen
                or row["name"] != prefix(tag, rank) + name + ".gz"
                or type(row["original_bytes"]) is not int or not 0 < row["original_bytes"] <= limits[name]
                or type(row["size"]) is not int or not 0 < row["size"] <= row["original_bytes"] + max(65536, row["original_bytes"] // 100)
                or not isinstance(row["generation"], str) or not row["generation"].isdecimal() or int(row["generation"]) <= 0
                or not isinstance(row["crc32c"], str)
                or not isinstance(row["original_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["original_sha256"]) is None):
            raise ValueError("delivery receipt path/size/generation/hash differs")
        seen.add(name)
        total += row["original_bytes"]
    if seen != set(limits) or total > RANK_CAP or type(value["original_bytes"]) is not int or value["original_bytes"] != total:
        raise ValueError("delivery original inventory/total differs")
    return value["files"]


def collect_rank(*, destination: Path, tag: str, pin: str, label: str,
                 rank: int, client: Any, blobs: dict[str, Any]) -> list[dict]:
    """Reconstruct exact originals; records returned join the outer source ledger.

    `blobs` is the parent's complete prefix listing with full bucket object keys.
    No trust in worker verdicts, no model/memory validation claimed here. Existing
    local files are reused only after exact remote inflated bytes match them.
    """
    _identity(tag, pin, label, rank)
    _plain_path(destination)
    if not destination.is_dir():
        raise ValueError("delivery collection destination missing")
    _bucket(client)
    name = prefix(tag, rank) + MANIFEST
    blob = blobs.get(name)
    raw = _download(blob, cap=MANIFEST_CAP)
    value = json.loads(raw)
    rows = validate_manifest(value, tag=tag, pin=pin, label=label, rank=rank)
    expected = {name, *(row["name"] for row in rows)}
    if {key for key in blobs if key.startswith(prefix(tag, rank))} != expected:
        raise ValueError("delivery remote phase has missing/extra objects")
    needed = value["original_bytes"] + 2 * max(row["size"] for row in rows) + DISK_RESERVE
    if shutil.disk_usage(destination).free < needed:
        raise ValueError("delivery materialization disk reserve insufficient")
    records = [dict(name=name, generation=str(blob.generation), size=len(raw),
                    crc32c=str(blob.crc32c), sha256=sha256(raw).hexdigest())]
    for row in rows:
        obj = blobs.get(row["name"])
        if obj is None or (str(obj.generation), int(obj.size), str(obj.crc32c)) != (row["generation"], row["size"], row["crc32c"]):
            raise ValueError("delivery original generation metadata differs")
        packed = _download(obj, cap=row["size"])
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
            data = stream.read(row["original_bytes"] + 1)
        if len(data) != row["original_bytes"] or sha256(data).hexdigest() != row["original_sha256"]:
            raise ValueError("delivery inflated original bytes/hash differ")
        path = destination / row["relative_path"]
        _plain_path(path)
        if path.exists():
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError("delivery existing local original differs")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        records.append(dict(name=row["name"], generation=row["generation"], size=len(packed),
            crc32c=row["crc32c"], sha256=sha256(packed).hexdigest(),
            inflated_sha256=row["original_sha256"], inflated_bytes=len(data)))
    return records


def main() -> None:
    from google.cloud import storage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--context-label", required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    _identity(args.tag, args.code_hash, args.context_label, args.rank)
    if not socket.gethostname().endswith(f"-w-{args.rank}"):
        parser.error("delivery launch rank differs from this host")
    require_local_idle()
    value = publish_rank(root=RUN_ROOT / args.tag, tag=args.tag, pin=args.code_hash,
        label=args.context_label, rank=args.rank, client=storage.Client())
    print(json.dumps(dict(files=len(value["files"]), missing=value["missing"], numerical_claim=False)))


if __name__ == "__main__":
    main()
