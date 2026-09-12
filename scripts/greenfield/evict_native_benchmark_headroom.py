"""Fixed recoverable local copies only; keep all cloud originals and weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scripts.greenfield import evict_reviewed_local_copies as engine
from scripts.greenfield.evict_delivery_256k_headroom import verified_file
from scripts.greenfield.collect_ws32_worker_evidence import digest_file
from scripts.greenfield.microbench_fp8_matmul import _atomic_json

SCOPE = engine.REPO / "configs/greenfield-native-headroom-scope.json"
KIND = "ws32_native_benchmark_headroom_review_v1"


def validate_manifest(value: dict) -> list[dict]:
    scope = json.loads(SCOPE.read_bytes())
    expected = {r["path"]:r for r in scope}
    entries = value.get("files", [])
    if (value.get("artifact_kind") != KIND or value.get("status") != "VERIFIED_NOT_DELETED"
            or value.get("cloud_objects_deleted") != 0 or value.get("bucket_location") != "US-CENTRAL2"
            or len(entries) != len(expected) or {r["path"] for r in entries} != set(expected)
            or value.get("total_bytes") != sum(r["size"] for r in scope)):
        raise ValueError("native headroom requires exact reviewed scope")
    for r in entries:
        if (any(r.get(k) != v for k,v in expected[r["path"]].items()) or r["nlink"] != 1
                or r["restore_uri"] != f"gs://{engine.BUCKET}/{r['name']}#{r['generation']}"):
            raise ValueError("native headroom target/size/restore differs")
    return entries


def review(output: Path) -> None:
    from google.cloud import storage
    if output.exists(): raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2": raise ValueError("wrong region")
        rows = []
        for target in json.loads(SCOPE.read_bytes()):
            path = Path(target["path"])
            st = path.lstat()
            blob = bucket.get_blob(target["name"])
            if blob is None or st.st_size != target["size"] or st.st_nlink != 1:
                raise ValueError(f"missing/changed local or cloud original: {path}")
            facts = digest_file(path)
            r = dict(target, inode=st.st_ino, mtime_ns=st.st_mtime_ns, nlink=st.st_nlink,
                sha256=facts["sha256"], crc32c=facts["crc32c"], generation=str(blob.generation),
                restore_uri=f"gs://{engine.BUCKET}/{target['name']}#{blob.generation}")
            if target["restore_transform"] == "gunzip":
                r.update(compressed_size=int(blob.size), compressed_crc32c=blob.crc32c)
            with verified_file(r,bucket): pass
            rows.append(r)
        value = dict(artifact_kind=KIND,status="VERIFIED_NOT_DELETED",cloud_objects_deleted=0,
            bucket_location=bucket.location,files=rows,total_bytes=sum(r["size"] for r in rows))
        validate_manifest(value)
        _atomic_json(output,value)
        print(json.dumps(dict(files=len(rows),bytes=value["total_bytes"])),flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["review"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("output",type=Path)
        review(parser.parse_args(sys.argv[2:]).output)
    else:
        raise SystemExit(engine.main(validate_manifest=validate_manifest,verify_file=verified_file))
