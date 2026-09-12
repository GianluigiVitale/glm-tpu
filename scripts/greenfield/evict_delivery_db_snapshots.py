"""Recover launch space from eleven archived per-run DB copies, never primaryDB.

Fixed scope only. Cloud originals remain, and the unchanged eviction engine
enforces both leases, exact generation/bytes/CRC, no holders and durable intents.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

from scripts.greenfield import evict_reviewed_local_copies as engine
from scripts.greenfield.microbench_fp8_matmul import _atomic_json

TAGS = tuple("greenfield_fp8_" + s for s in (
    "ws32_prefill_prefix_mlp_diagnostic_l3_20260908T013056476665184Z",
    "ws32_prefill_router_boundary_diagnostic_l3_20260908T003156848804906Z",
    "ws32_prefill_layer_admission_l0_20260907T233201686218983Z",
    "ws32_prefill_moe_bounded_admission_20260907T210616212681016Z",
    "ws32_prefill_moe_boundary_diagnostic_20260907T204127593566771Z",
    "ws32_grouped_down_admission_20260907T195100892469765Z",
    "ws32_grouped_admission_20260907T194202909841356Z",
    "ws32_prefill_baseline_m256_20260907T190950865543399Z",
    "ws32_prefill_baseline_m128_20260907T190903501964517Z",
    "ws32_prefill_baseline_m32_20260907T190821551622724Z",
    "ws32_prefill_baseline_m8_20260907T190714689540075Z",
))
SIZES = (42438656, 41897984, 41279488, 35966976, 34816000, 34316288,
         34299904, 34283520, 34267136, 34254848, 34242560)
KIND = "ws32_delivery_archived_db_copy_review_v1"


def validate_manifest(manifest: dict) -> list[dict]:
    entries = manifest.get("files", [])
    expected = {str(engine.LOCAL_ROOT / tag / "results_ckpt.db"):
                (f"results/{tag}/results_ckpt.db", size)
                for tag, size in zip(TAGS, SIZES, strict=True)}
    if (manifest.get("artifact_kind") != KIND
            or manifest.get("status") != "VERIFIED_NOT_DELETED"
            or manifest.get("bucket_location") != "US-CENTRAL2"
            or type(manifest.get("cloud_objects_deleted")) is not int
            or manifest.get("cloud_objects_deleted") != 0
            or manifest.get("total_bytes") != sum(SIZES)
            or len(entries) != len(TAGS)
            or {e["path"] for e in entries} != set(expected)):
        raise ValueError("requires exactly eleven archived per-run DB copies")
    for entry in entries:
        if ((entry["name"], entry["size"]) != expected[entry["path"]]
                or entry["nlink"] != 1
                or entry["restore_uri"] != f"gs://{engine.BUCKET}/{entry['name']}#{entry['generation']}"):
            raise ValueError("snapshot scope/size/recovery differs")
        for key in ("size", "inode", "mtime_ns", "ctime_ns", "device", "nlink"):
            if type(entry[key]) is not int or entry[key] <= 0:
                raise ValueError("snapshot stat must use exact positive integers")
        if (not isinstance(entry["generation"], str)
                or not re.fullmatch(r"[1-9][0-9]*", entry["generation"])
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or len(base64.b64decode(entry["crc32c"], validate=True)) != 4):
            raise ValueError("snapshot cloud identity differs")
    return entries


def review(output: Path) -> None:
    from google.cloud import storage

    if output.exists():
        raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2":
            raise ValueError("wrong bucket region")
        entries, ledgers = [], []
        for tag in TAGS:
            root = engine.LOCAL_ROOT / tag
            raw = (root / "archive_receipts.json").read_bytes()
            name = f"results/{tag}/archive_receipts.json"
            blob = bucket.get_blob(name)
            if (blob is None or len(raw) > 2 << 20 or int(blob.size) != len(raw)
                    or blob.download_as_bytes(if_generation_match=int(blob.generation)) != raw):
                raise ValueError("original archive ledger differs")
            rows = [r for r in json.loads(raw) if r["name"] == f"results/{tag}/results_ckpt.db"]
            if len(rows) != 1:
                raise ValueError("snapshot must have exactly one archive receipt")
            r, path = rows[0], root / "results_ckpt.db"
            st = path.lstat()
            entry = dict(path=str(path), name=r["name"], generation=str(r["generation"]),
                size=r["size"], sha256=r["original_sha256"], crc32c=r["crc32c"],
                inode=st.st_ino, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                device=st.st_dev, nlink=st.st_nlink,
                restore_uri=f"gs://{engine.BUCKET}/{r['name']}#{r['generation']}")
            with engine._verified_file(entry, bucket):
                pass
            entries.append(entry)
            ledgers.append(dict(name=name, generation=str(blob.generation),
                                sha256=sha256(raw).hexdigest(), size=len(raw)))
        result = dict(artifact_kind=KIND, status="VERIFIED_NOT_DELETED",
            bucket_location=bucket.location, cloud_objects_deleted=0, total_bytes=sum(SIZES),
            files=entries, ledger_sources=ledgers)
        validate_manifest(result)
        _atomic_json(output, result)
        print(json.dumps(dict(files=len(entries), bytes=sum(SIZES),
                              sha256=sha256(output.read_bytes()).hexdigest())))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "review":
        parser = argparse.ArgumentParser()
        parser.add_argument("output", type=Path)
        review(parser.parse_args(args[1:]).output)
        return 0
    return engine.main(args, validate_manifest=validate_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
