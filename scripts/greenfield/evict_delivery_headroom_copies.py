"""Review/evict only fixed archived local copies for the long-capacity launch.

Cloud objects, primary DB, active weights, rank0 originals and compact scientific
evidence remain. Reuse the existing leased, descriptor-relative eviction engine.
Review writes a fresh exact-path/generation manifest; apply requires its SHA.
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

ROOT = Path("/home/gianl/glm-run")
LEGACY = "greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z"
LEGACY_PREFIX = f"oracles/greenfield/glm52/prompt_index_cache/8k/{LEGACY}"
HISTORY = "greenfield_fp8_ws32_history_frontier_l06_20260911T212325931768009Z"
DB_TAGS = (HISTORY, *["greenfield_fp8_" + suffix for suffix in (
    "ws32_prefill_window_boundary_acquisition_l6_20260908T143917951158196Z",
    "ws32_prefill_layer_window_acquisition_l6_20260908T115105929011946Z",
    "ws32_prefill_moe_scaling_baseline_20260908T103951296307597Z",
    "ws32_prefill_layer_observed_admission_l3_20260908T014622122036216Z",
)])
HISTORY_NAMES = (
    *(f"{p}.{form}" for p in ("candidate_b128", "candidate_b114", "control_b128", "control_b114", "observer")
      for form in ("stablehlo.mlir", "optimized_hlo.txt")),
    *(f"materializers/layer{layer}_wk_{kind}.npz" for layer in (0, 1, 2, 6) for kind in ("decode", "promote")),
    "first_difference_group0.npz",
)
TOTAL_BYTES = 4_455_299_027
KIND = "ws32_delivery_exact_local_headroom_review_v1"


def scope() -> dict[str, str]:
    pairs = {}
    for step in range(1, 5):
        for event in range(21):
            rel = f"source_dumps/w4/topk.step{step:04d}.evt{event:02d}.proc0.npz"
            pairs[str(ROOT / LEGACY / rel)] = f"{LEGACY_PREFIX}/{rel}"
    for rank in range(1, 8):
        for name in HISTORY_NAMES:
            pairs[str(ROOT / HISTORY / "fleet" / f"rank{rank}" / name)] = f"results/{HISTORY}/workers/rank{rank}/{name}"
    for tag in DB_TAGS:
        pairs[str(ROOT / tag / "results_ckpt.db")] = f"results/{tag}/results_ckpt.db"
    return pairs


def validate_manifest(manifest: dict) -> list[dict]:
    entries = manifest.get("files", [])
    expected = scope()
    if (manifest.get("artifact_kind") != KIND or manifest.get("status") != "VERIFIED_NOT_DELETED"
            or manifest.get("bucket_location") != "US-CENTRAL2"
            or type(manifest.get("cloud_objects_deleted")) is not int
            or manifest.get("cloud_objects_deleted") != 0
            or manifest.get("file_count") != 222
            or manifest.get("total_bytes") != TOTAL_BYTES
            or len(entries) != 222 or {e["path"] for e in entries} != set(expected)
            or sum(e["size"] for e in entries) != TOTAL_BYTES):
        raise ValueError("delivery eviction requires exactly the 222 reviewed local copies")
    for e in entries:
        if (e["name"] != expected[e["path"]] or e["nlink"] != 1
                or type(e["size"]) is not int or e["size"] <= 0
                or e["restore_uri"] != f"gs://{engine.BUCKET}/{e['name']}#{e['generation']}"):
            raise ValueError("delivery eviction local/cloud identity differs")
        for key in ("inode", "mtime_ns", "ctime_ns", "device", "nlink"):
            if type(e[key]) is not int or e[key] <= 0:
                raise ValueError(f"invalid reviewed {key}")
        if (not isinstance(e["generation"], str)
                or not re.fullmatch(r"[1-9][0-9]*", e["generation"])
                or not re.fullmatch(r"[0-9a-f]{64}", e["sha256"])
                or len(base64.b64decode(e["crc32c"], validate=True)) != 4):
            raise ValueError("invalid cloud generation or byte digest")
    return entries


def review(output: Path) -> None:
    """Authenticate original small cloud ledgers, then every selected local file."""
    from google.cloud import storage
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json

    if output.exists():
        raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2":
            raise ValueError("wrong bucket region")
        ledgers = []
        def ledger(local: Path, remote: str):
            raw = local.read_bytes()
            obj = bucket.get_blob(remote)
            if obj is None or len(raw) > 2 << 20 or int(obj.size) != len(raw):
                raise ValueError(f"missing/oversized original ledger: {remote}")
            if obj.download_as_bytes(if_generation_match=int(obj.generation)) != raw:
                raise ValueError(f"original ledger differs: {remote}")
            ledgers.append(dict(path=str(local), name=remote, generation=str(obj.generation),
                                size=len(raw), sha256=sha256(raw).hexdigest(), crc32c=obj.crc32c))
            return json.loads(raw)

        remote = ledger(ROOT / LEGACY / "remote_objects.json", LEGACY_PREFIX + "/remote_objects.json")
        hashes = ledger(ROOT / LEGACY / "evidence_sha256.json", LEGACY_PREFIX + "/evidence_sha256.json")
        hashes = {r["path"]: r for r in hashes["files"]}
        receipts = {LEGACY_PREFIX + "/" + r["path"]: dict(r, original_sha256=hashes[r["path"]]["sha256"])
                    for r in remote["objects"] if r["path"] in hashes}
        for rank in range(1, 8):
            name = f"results/{HISTORY}/workers/rank{rank}/worker_receipts.json"
            rows = ledger(ROOT / HISTORY / "fleet" / f"rank{rank}" / "worker_receipts.json", name)
            receipts.update({r["name"]: r for r in rows})
        for tag in DB_TAGS:
            rows = ledger(ROOT / tag / "archive_receipts.json", f"results/{tag}/archive_receipts.json")
            receipts.update({r["name"]: r for r in rows})
        entries = []
        for local, name in scope().items():
            path, r = Path(local), receipts[name]
            st = path.lstat()
            entry = dict(path=local, name=name, generation=str(r["generation"]), size=r["size"],
                sha256=r["original_sha256"], crc32c=r["crc32c"], inode=st.st_ino, mtime_ns=st.st_mtime_ns,
                ctime_ns=st.st_ctime_ns, device=st.st_dev, nlink=st.st_nlink,
                restore_uri=f"gs://{engine.BUCKET}/{name}#{r['generation']}")
            if st.st_nlink != 1:
                raise ValueError(f"not a single-link disposable copy: {local}")
            with engine._verified_file(entry, bucket):
                pass
            entries.append(entry)
        result = dict(artifact_kind=KIND, status="VERIFIED_NOT_DELETED", bucket_location=bucket.location,
                      cloud_objects_deleted=0, files=entries, ledger_sources=ledgers,
                      total_bytes=TOTAL_BYTES, file_count=len(entries))
        validate_manifest(result)
        _atomic_json(output, result)
        print(json.dumps(dict(status=result["status"], file_count=len(entries), total_bytes=TOTAL_BYTES,
                              manifest_sha256=sha256(output.read_bytes()).hexdigest())))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "review":
        parser = argparse.ArgumentParser()
        parser.add_argument("output", type=Path)
        parsed = parser.parse_args(args[1:])
        review(parsed.output)
        return 0
    return engine.main(args, validate_manifest=validate_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
