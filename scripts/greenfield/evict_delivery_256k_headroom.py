"""Fixed archived local evidence scope for the full256K launch, never cloud deletion.

Gzip objects must decompress to the exact local SHA/size. Restore such entries
by downloading the specified generation and gunzipping, not by copying gzip bytes
to the raw filename. Existing lease/holder/identity/receipt/unlink engine is reused.
"""
from __future__ import annotations

import argparse
import base64
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import google_crc32c

from scripts.greenfield import evict_reviewed_local_copies as engine
from scripts.greenfield.microbench_fp8_matmul import _atomic_json

KIND = "ws32_delivery_256k_headroom_review_v1"
TAG = "greenfield_ws32_short_decoder_128k_d0_95_numerical_c128_cap131072_hrope_bp1_ps1_rp1_ep1_lm1_cd1_s26long_20260912T142548940670244Z"
PEERS = (298430693, 298430708, 300691600, 300691165, 300690477, 298430251, 300691555)
NAMES = ("cache_probe", "decode", "exact_materialize", "exact_promote", "observer", "prefill_chunk", "prefill_tail")
OLD = {
    "2k_acquire_c17_hrope_bp1_20260908T034734466845101Z": (57898, 75246059, 1442775, 254763, 76586150, 143943180, 128445529),
    "2k_numerical_c17_hrope_bp1_ps1_20260908T225358777672108Z": (0, 0, 1442775, 254763, 76586154, 142860881, 127276044),
    "2k_numerical_c17_hrope_bp1_ps1_20260908T221354548152364Z": (0, 0, 1442775, 254763, 0, 142860881, 127276044),
    "8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_cd1_20260909T225859315457683Z": (57898, 75246063, 1442775, 254763, 76586154, 100311057, 102219900),
    "8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_live32_20260909T103047459508942Z": (57898, 75246063, 1442775, 254763, 76586154, 100207508, 102123411),
}
COMPRESSED_CAP = 16 << 20


def scope() -> dict[str, tuple[str, int, str]]:
    pairs = {f"{TAG}/traces/trace.rank{r}.xplane.pb":
             (f"results/{TAG}/traces/trace.rank{r}.xplane.pb", s, "identity")
             for r, s in enumerate(PEERS, 1)}
    pairs[f"{TAG}/results.db"] = (f"results/{TAG}/orchestrator/results.db", 200810496, "identity")
    for suffix, sizes in OLD.items():
        tag = "greenfield_ws32_short_decoder_" + suffix
        for graph, size in zip(NAMES, sizes, strict=True):
            if size:
                p = f"{tag}/hlo/{graph}.optimized_hlo.txt"
                pairs[p] = (f"results/{p}.gz", size, "gunzip")
    return {str(engine.LOCAL_ROOT / p): values for p, values in pairs.items()}


def crc(raw: bytes) -> str:
    return base64.b64encode(google_crc32c.Checksum(raw).digest()).decode("ascii")


def verify_gzip(entry: dict, blob: object) -> None:
    generation = int(entry["generation"])
    size = entry["compressed_size"]
    if (not 0 < size <= COMPRESSED_CAP or int(blob.generation) != generation
            or int(blob.size) != size or blob.crc32c != entry["compressed_crc32c"]):
        raise ValueError("compressed cloud identity differs")
    # A bounded exact-generation byte range also avoids transparent gzip decoding.
    raw = blob.download_as_bytes(start=0, end=size - 1, raw_download=True,
                                 if_generation_match=generation, checksum=None, timeout=60)
    if len(raw) != size or crc(raw) != entry["compressed_crc32c"]:
        raise ValueError("compressed payload differs")
    digest, total = sha256(), 0
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        while chunk := stream.read(min(8 << 20, entry["size"] - total + 1)):
            total += len(chunk)
            if total > entry["size"]:
                raise ValueError("inflated size exceeds local original")
            digest.update(chunk)
    if total != entry["size"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("inflated bytes differ from local original")


def verified_file(entry: dict, bucket: object):
    return engine._verified_file(entry, bucket, cloud_verifier=(
        verify_gzip if entry["restore_transform"] == "gunzip" else None))


def validate_manifest(value: dict) -> list[dict]:
    expected, entries = scope(), value.get("files", [])
    if (value.get("artifact_kind") != KIND or value.get("status") != "VERIFIED_NOT_DELETED"
            or value.get("cloud_objects_deleted") != 0 or value.get("bucket_location") != "US-CENTRAL2"
            or len(entries) != len(expected) or {r["path"] for r in entries} != set(expected)
            or value.get("total_bytes") != sum(v[1] for v in expected.values())):
        raise ValueError("requires exact reviewed headroom scope")
    for r in entries:
        if ((r["name"], r["size"], r["restore_transform"]) != expected[r["path"]]
                or r["nlink"] != 1
                or r["restore_uri"] != f"gs://{engine.BUCKET}/{r['name']}#{r['generation']}"):
            raise ValueError("target/size/restore differs")
        if r["restore_transform"] == "gunzip" and not 0 < r["compressed_size"] <= COMPRESSED_CAP:
            raise ValueError("compressed cap exceeded")
    return entries


def review(output: Path) -> None:
    from google.cloud import storage
    if output.exists():
        raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2":
            raise ValueError("wrong region")
        rows = []
        for local, (name, size, transform) in scope().items():
            p = Path(local)
            st = p.lstat()
            blob = bucket.get_blob(name)
            if blob is None or st.st_size != size or st.st_nlink != 1:
                raise ValueError(f"missing/changed original: {p}")
            digest, checksum = sha256(), google_crc32c.Checksum()
            with p.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
                    checksum.update(chunk)
            r = dict(path=local, name=name, size=size, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
                     nlink=st.st_nlink, sha256=digest.hexdigest(),
                     crc32c=base64.b64encode(checksum.digest()).decode("ascii"),
                     generation=str(blob.generation), restore_transform=transform,
                     restore_uri=f"gs://{engine.BUCKET}/{name}#{blob.generation}")
            if transform == "gunzip":
                r.update(compressed_size=int(blob.size), compressed_crc32c=blob.crc32c)
            with verified_file(r, bucket):
                pass
            rows.append(r)
        value = dict(artifact_kind=KIND, status="VERIFIED_NOT_DELETED", cloud_objects_deleted=0,
                     bucket_location=bucket.location, files=rows, total_bytes=sum(r["size"] for r in rows))
        validate_manifest(value)
        _atomic_json(output, value)
        print(json.dumps(dict(files=len(rows), bytes=value["total_bytes"], sha256=sha256(output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    if sys.argv[1:2] == ["review"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("output", type=Path)
        review(parser.parse_args(sys.argv[2:]).output)
    else:
        raise SystemExit(engine.main(validate_manifest=validate_manifest, verify_file=verified_file))
