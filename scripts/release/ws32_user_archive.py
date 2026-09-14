"""Seal one original user response after replay, DB linkage and regional readback.

No benchmark score, public upload, model retry, checkpoint copy or full DB copy.
The controller owns both leases through this function. Failures retain originals
and any committed DB rows; retry is idempotent and never overwrites them.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from glm_tpu import user_request
from scripts.release import ws32_user_database as database
from scripts.release import ws32_user_transport as transport
from scripts.release import ws32_user_worker as worker
from scripts.release.ws32_user_result import read, same
from scripts.greenfield import ws32_native_benchmark_archive as original
from scripts.greenfield.collect_ws32_worker_evidence import digest_file, publish_exact
from scripts.greenfield.fp8_baseline_guard import validate_fleet

FILE_CAP, CONTROLLER_CAP, ARCHIVE_CAP = 64 << 20, 128 << 20, 10 << 30
FILES = ("launch.json", "census_pre.txt", "census_post.txt", "sync.txt", "prepared.txt",
         "final_watch.jsonl", "request.json", "user_replay.json", "user_collection.json", "db_link.json")


def persist(path: Path, value: dict) -> None:
    data = user_request.canonical(value) + b"\n"
    if len(data) > FILE_CAP:
        raise ValueError("user controller record exceeds cap")
    with transport.private_writes():
        transport.outputs._write_once(path, data)


def seal(*, root: Path, tag: str, pin: str, report: dict, collection: dict,
         blobs: dict, client: Any, db_path: Path = database.PRIMARY_DB) -> dict:
    database.validate_report(report, tag, pin)
    if root != worker.RUN_ROOT / tag:
        raise ValueError("user archive requires original run root")
    transport.private_directory(root)
    same(json.loads(read(root / "user_replay.json", FILE_CAP)), report, "saved user replay")
    same(json.loads(read(root / "user_collection.json", FILE_CAP)), collection, "saved user collection")
    launch = json.loads(read(root / "launch.json", FILE_CAP))
    expected = dict(tag=tag, code_hash=pin, request_file_sha256=report["request_file_sha256"], benchmark=False)
    same({key: launch[key] for key in expected}, expected, "archive launch")
    request_raw = read(root / "request.json", user_request.PAYLOAD_CAP)
    same(sha256(request_raw).hexdigest(), report["request_file_sha256"], "archive request bytes")
    for phase in ("pre", "post"):
        validate_fleet(read(root / f"census_{phase}.txt", FILE_CAP).decode())
    original.verify_ownership(root, tag, pin)
    expected = dict(schema="glm_ws32_user_collection_v1", tag=tag, code_hash=pin,
        request_file_sha256=report["request_file_sha256"], missing_cold_ranks=[], benchmark=False,
        protected_result_sealed=False)
    same({key: collection[key] for key in expected}, expected, "archive collection scope")
    if len(collection["requests"]) != 8 or len(collection["cold"]) != 8:
        raise ValueError("user archive requires both original eight-rank channels")
    bucket = transport.approved_bucket(client)
    expected_objects = {}
    for rank in range(8):
        for manifest, schema, prefix in (
            (collection["cold"][rank]["manifest"], transport.cold.USER_SCHEMA, transport.cold.prefix(tag, rank)),
            (collection["requests"][rank], transport.outputs.USER_SCHEMA,
             transport.outputs.prefix(tag, rank, user_request=True)),
        ):
            expected = dict(schema=schema, tag=tag, code_hash=pin, rank=rank)
            same({key: manifest[key] for key in expected}, expected, "user archive manifest")
            for row in manifest["files"]:
                facts = (str(row["generation"]), int(row["size"]), row["crc32c"])
                if row["name"] in expected_objects and expected_objects[row["name"]] != facts:
                    raise ValueError("user shared object identities disagree")
                expected_objects[row["name"]] = facts
            blob = blobs[prefix + "manifest.json"]
            expected_objects[blob.name] = (str(blob.generation), int(blob.size), blob.crc32c)
    names = {name for name in blobs if name.startswith((f"results/{tag}/native_cold/", f"results/{tag}/user_requests/"))}
    if names != set(expected_objects):
        raise ValueError("user archive worker union differs from collection")
    input_object = launch["input_object"]
    if input_object["name"] != f"results/{tag}/private_input/request.json":
        raise ValueError("user archive input namespace differs")
    same(input_object["original_sha256"], report["request_file_sha256"], "archived input hash")
    same(int(input_object["size"]), len(request_raw), "archived input size")
    expected_objects[input_object["name"]] = (str(input_object["generation"]), int(input_object["size"]), input_object["crc32c"])
    for name, facts in expected_objects.items():
        blob = bucket.get_blob(name)
        if blob is None or (str(blob.generation), int(blob.size), blob.crc32c) != facts:
            raise ValueError("user archive generation changed after collection")
    if sum(facts[1] for facts in expected_objects.values()) + CONTROLLER_CAP + (8 << 20) > ARCHIVE_CAP:
        raise ValueError("user archive exceeds 10GiB cap")
    # Refuse known controller oversize before adding rows; then export only this run.
    paths = [root / name for name in FILES if name != "db_link.json"]
    for path in paths:
        read(path, FILE_CAP)
    if sum(path.stat().st_size for path in paths) + FILE_CAP > CONTROLLER_CAP:
        raise ValueError("user controller originals exceed archive reserve")
    with transport.private_writes():
        link = database.record_result(database=db_path, tag=tag, pin=pin, report=report)
    persist(root / "db_link.json", link)
    paths.append(root / "db_link.json")
    controller = [publish_exact(bucket, f"results/{tag}/controller/{path.name}", path,
                               digest_file(path), compressed=False) for path in paths]
    ledger = dict(schema="glm_ws32_user_archive_ledger_v1", tag=tag, code_hash=pin,
        originals=[dict(name=name, generation=facts[0], size=facts[1], crc32c=facts[2])
                   for name, facts in sorted(expected_objects.items())], controller=controller)
    persist(root / "archive_ledger.json", ledger)
    if (root / "archive_ledger.json").stat().st_size > 4 << 20:
        raise ValueError("user archive ledger exceeds cap")
    ledger_receipt = publish_exact(bucket, f"results/{tag}/controller/archive_ledger.json",
        root / "archive_ledger.json", digest_file(root / "archive_ledger.json"), compressed=False)
    receipt = dict(schema="glm_ws32_user_response_sealed_v1", tag=tag, code_hash=pin,
        request_sha256=report["request_sha256"], request_file_sha256=report["request_file_sha256"],
        token_ids_sha256=report["token_ids_sha256"], generated_tokens=report["generated_tokens"],
        finish_reason=report["finish_reason"], run_id=link["run_id"], db_rows_sha256=link["rows_sha256"],
        replay_sha256=sha256(user_request.canonical(report)).hexdigest(), ledger=ledger_receipt,
        protected_result_sealed=True, evidence_archived=True, benchmark=False, quality_score=None,
        project_complete=False, cleanup="authenticated8host_idle",
        delivery_boundary=report["delivery_boundary"], durable_resume_verified=False,
        decode_trace_verified=report["trace_physical_coverage_verified"])
    persist(root / "USER_RESPONSE_SEALED.json", receipt)
    publish_exact(bucket, f"results/{tag}/USER_RESPONSE_SEALED.json", root / "USER_RESPONSE_SEALED.json",
                  digest_file(root / "USER_RESPONSE_SEALED.json"), compressed=False)
    return receipt
