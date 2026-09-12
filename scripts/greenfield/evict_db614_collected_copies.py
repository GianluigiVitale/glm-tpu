"""Fixed DB614 collected graph copies and archived DB snapshot; cloud retained.

Reuse the existing leased verification/unlink engine. Keep fleet/rank0 originals,
all compact rank evidence and the primary database. Default is dry-run.
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

TAG = "greenfield_fp8_ws32_flat_rows_prefill_compile_20260912T020453121038257Z"
KIND = "ws32_db614_collected_copy_review_v1"
GRAPH_SIZES = {"prefill_256k_flat_rows.stablehlo.mlir": 21100974,
               "prefill_256k_flat_rows.optimized_hlo.txt": 102947956}
SNAPSHOT_BYTES = 193433600
OLD_TAG = "greenfield_fp8_ws32_owned_state_prefill_compile_20260912T003527251082976Z"
OLD_GRAPH_SIZES = {"prefill_256k_state_donated.stablehlo.mlir": 20859926,
                   "prefill_256k_state_donated.optimized_hlo.txt": 102679115}
TOTAL_BYTES = (8 * sum(GRAPH_SIZES.values()) + SNAPSHOT_BYTES
               + GRAPH_SIZES["prefill_256k_flat_rows.optimized_hlo.txt"]
               + sum(OLD_GRAPH_SIZES.values()) + 102679115)


def scope() -> dict[str, tuple[str, int, Path, str]]:
    root = engine.LOCAL_ROOT / TAG
    result = {}
    for rank in range(1, 8):
        prefix = f"results/{TAG}/workers/rank{rank}"
        parent = root / "fleet" / f"rank{rank}"
        for name, size in GRAPH_SIZES.items():
            result[str(parent / name)] = (f"{prefix}/{name}", size,
                parent / "worker_receipts.json", f"{prefix}/worker_receipts.json")
    result[str(root / "results_ckpt.db")] = (f"results/{TAG}/results_ckpt.db",
        SNAPSHOT_BYTES, root / "archive_receipts.json", f"results/{TAG}/archive_receipts.json")
    for tag, graphs in ((TAG, GRAPH_SIZES), (OLD_TAG, OLD_GRAPH_SIZES)):
        run = engine.LOCAL_ROOT / tag
        for name, size in graphs.items():
            prefix = f"results/{tag}/workers/rank0"
            result[str(run / "rank0" / name)] = (f"{prefix}/{name}", size,
                run / "fleet/rank0/worker_receipts.json", f"{prefix}/worker_receipts.json")
        name = "hlo/candidate.optimized_hlo.txt"
        size = next(size for name, size in graphs.items() if name.endswith(".txt"))
        result[str(run / name)] = (f"results/{tag}/{name}", size,
            run / "archive_receipts.json", f"results/{tag}/archive_receipts.json")
    return result


def validate_manifest(manifest: dict) -> list[dict]:
    entries = manifest.get("files", [])
    expected = scope()
    if (manifest.get("artifact_kind") != KIND
            or manifest.get("status") != "VERIFIED_NOT_DELETED"
            or manifest.get("bucket_location") != "US-CENTRAL2"
            or type(manifest.get("cloud_objects_deleted")) is not int
            or manifest["cloud_objects_deleted"] != 0
            or type(manifest.get("total_bytes")) is not int
            or manifest["total_bytes"] != TOTAL_BYTES
            or len(entries) != 21 or {e["path"] for e in entries} != set(expected)):
        raise ValueError("requires exactly twenty-one reviewed DB613/614 duplicates")
    for entry in entries:
        name, size, ledger_path, ledger_name = expected[entry["path"]]
        if ((entry["name"], entry["size"]) != (name, size)
                or entry["nlink"] != 1
                or entry["restore_uri"] != f"gs://{engine.BUCKET}/{name}#{entry['generation']}"
                or entry.get("ledger_path") != str(ledger_path)
                or entry.get("ledger_name") != ledger_name):
            raise ValueError("collected copy scope/size/recovery differs")
        for key in ("size", "inode", "mtime_ns", "ctime_ns", "device", "nlink"):
            if type(entry[key]) is not int or entry[key] <= 0:
                raise ValueError("copy stat must use exact positive integers")
        if (not isinstance(entry["generation"], str)
                or not re.fullmatch(r"[1-9][0-9]*", entry["generation"])
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                or len(base64.b64decode(entry["crc32c"], validate=True)) != 4):
            raise ValueError("copy cloud identity differs")
    return entries


def review(output: Path) -> None:
    from google.cloud import storage

    if output.exists():
        raise FileExistsError(output)
    with engine._leases():
        bucket = storage.Client().get_bucket(engine.BUCKET)
        if bucket.location != "US-CENTRAL2":
            raise ValueError("wrong bucket region")
        entries, ledgers = [], {}
        for local, (name, size, ledger_path, ledger_name) in scope().items():
            if ledger_name not in ledgers:
                raw = ledger_path.read_bytes()
                blob = bucket.get_blob(ledger_name)
                if (blob is None or len(raw) > 2 << 20 or int(blob.size) != len(raw)
                        or blob.download_as_bytes(if_generation_match=int(blob.generation)) != raw):
                    raise ValueError("original archive ledger differs")
                ledgers[ledger_name] = (json.loads(raw), dict(name=ledger_name,
                    generation=str(blob.generation), sha256=sha256(raw).hexdigest(), size=len(raw)))
            rows = [r for r in ledgers[ledger_name][0] if r["name"] == name]
            if len(rows) != 1 or rows[0]["size"] != size:
                raise ValueError("copy needs one exact-size original receipt")
            r, path = rows[0], Path(local)
            st = path.lstat()
            entry = dict(path=local, name=name, size=size, generation=str(r["generation"]),
                sha256=r["original_sha256"], crc32c=r["crc32c"], inode=st.st_ino,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns, device=st.st_dev,
                nlink=st.st_nlink, ledger_path=str(ledger_path), ledger_name=ledger_name,
                restore_uri=f"gs://{engine.BUCKET}/{name}#{r['generation']}")
            with engine._verified_file(entry, bucket):
                pass
            if path.name in GRAPH_SIZES or path.name in OLD_GRAPH_SIZES:
                retained_tag = OLD_TAG if path.name in OLD_GRAPH_SIZES else TAG
                retained = engine.LOCAL_ROOT / retained_tag / "fleet/rank0" / path.name
                if retained.is_symlink() or sha256(retained.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("retained rank0 graph differs")
            entries.append(entry)
        result = dict(artifact_kind=KIND, status="VERIFIED_NOT_DELETED",
            bucket_location=bucket.location, cloud_objects_deleted=0, total_bytes=TOTAL_BYTES,
            files=entries, ledger_sources=[r[1] for r in ledgers.values()])
        validate_manifest(result)
        _atomic_json(output, result)
        print(json.dumps(dict(files=len(entries), bytes=TOTAL_BYTES,
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
