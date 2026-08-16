#!/usr/bin/env python3
"""Seal or roll back the protected bounded WS32 DSA discriminator.

The numerical runner is intentionally small, but its evidence contract is not:
terminal validation is independent, the canonical results-DB mutation is one
SQLite transaction, every remote payload is generation/CRC/SHA bound, and
``SUCCESS`` is created last.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from typing import Any

import google_crc32c
from google.cloud import storage

from glm_tpu.greenfield.validation.ws32_dsa_association import (
    ARTIFACT_KIND,
    validate_ws32_layer0_dsa_association,
)


MODEL = "zai-org/GLM-5.2-FP8:greenfield-ws32-layer0-dsa-association"
REVISION = "ws32-layer0-dsa-association-v1"
BENCHMARK = "greenfield_ws32_layer0_dsa_association"
ITEM_ID = "layer0_position8155_tuple4_tuple8"
PROMPT = "Sealed position-8155 layer-0 DSA inputs and 8,155-row prompt cache."
GOLD = (
    "Exact accepted query/head/current-key, full score row, top-2048 set, "
    "order, and scores."
)
RUN_NOTE = "Protected bounded WS32 layer-0 DSA ownership/arithmetic diagnostic."
SUMMARY_NOTE = "Diagnostic only; no decoder, Gate-D, latency, or token-rate claim."


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical(value).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32c_file(path: Path) -> str:
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite append-only evidence: {path}")
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _require_sha(
    value: str, *, field: str, lengths: tuple[int, ...] = (64,)
) -> None:
    if len(value) not in lengths or not re.fullmatch(r"[0-9a-f]+", value):
        raise SystemExit(f"{field} is not a SHA-256 digest")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_short(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise SystemExit("WS32 DSA remote prefix is not a GCS URI")
    bucket, separator, prefix = uri[5:].partition("/")
    if not separator or bucket != "driftbench-dsv4-uc" or not prefix:
        raise SystemExit("WS32 DSA remote prefix is outside the approved bucket")
    return bucket, prefix.rstrip("/")


def _validation(args: argparse.Namespace) -> dict[str, Any]:
    return validate_ws32_layer0_dsa_association(
        args.run_dir,
        expected_code_hash=args.code_hash,
        association_manifest_sha256=args.association_manifest_sha256,
        prompt_cache_manifest_sha256=args.prompt_cache_manifest_sha256,
        internal_contract_sha256=args.internal_contract_sha256,
        internal_tensor_sha256=args.internal_tensor_sha256,
        source_summary_sha256=args.source_summary_sha256,
    )


def _expected_env(
    *,
    args: argparse.Namespace,
    runner: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "GLM_ENGINE": "greenfield_ws32_layer0_dsa_association",
        "GLM_EXECUTION_PLAN": "WS32_2D",
        "GLM_GREENFIELD_WS32_DSA_ASSOCIATION": 1,
        "association_manifest_sha256": args.association_manifest_sha256,
        "association_restored": validation["association_restored"],
        "candidate_mechanisms_proven": validation[
            "candidate_mechanisms_proven"
        ],
        "diagnostic_only": True,
        "exact_arms": validation["exact_arms"],
        "greenfield_code_hash": args.code_hash,
        "internal_contract_sha256": args.internal_contract_sha256,
        "internal_tensor_sha256": args.internal_tensor_sha256,
        "performance_claim": False,
        "prompt_cache_manifest_sha256": args.prompt_cache_manifest_sha256,
        "remote_prefix": args.remote_prefix,
        "run_tag": args.tag,
        "runner_sha256": validation["runner_sha256"],
        "tensor_sha256": runner["tensor_file"]["sha256"],
        "source_summary_sha256": args.source_summary_sha256,
    }


def _expected_item(
    runner: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    restored = validation["association_restored"]
    return {
        "benchmark": BENCHMARK,
        "item_id": ITEM_ID,
        "prompt": PROMPT,
        "gold": GOLD,
        "raw_output": _canonical(runner),
        "extracted": _canonical(
            {
                "association_restored": restored,
                "candidate_mechanisms_proven": validation[
                    "candidate_mechanisms_proven"
                ],
                "exact_arms": validation["exact_arms"],
            }
        ),
        "correct": int(restored),
        "score": float(restored),
        "n_prompt_tokens": None,
        "n_gen_tokens": None,
        "latency_ms": None,
        "seed": None,
        "finish_reason": None,
        "truncated": None,
    }


def _expected_summary() -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK,
        "n": 1,
        "metric": "diagnostic_completed",
        "value": 1.0,
        "card_value": None,
        "delta": None,
        "note": SUMMARY_NOTE,
    }


def _validate_command(args: argparse.Namespace) -> int:
    value = _validation(args)
    _atomic_json(args.output, value)
    print(_canonical(value))
    return 0


def _load_terminal_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner = json.loads((args.run_dir / "runner.json").read_text())
    validation = json.loads((args.run_dir / "validation.json").read_text())
    recomputed = _validation(args)
    if validation != recomputed:
        raise SystemExit("WS32 DSA validation record drifted")
    return runner, validation


def _existing_tag_runs(
    connection: sqlite3.Connection, *, tag: str
) -> list[int]:
    result = []
    for run_id, raw in connection.execute(
        "SELECT run_id,env_json FROM runs WHERE model=?", (MODEL,)
    ):
        try:
            environment = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(environment, dict) and environment.get("run_tag") == tag:
            result.append(int(run_id))
    return result


def _publish_db(args: argparse.Namespace) -> int:
    runner, validation = _load_terminal_inputs(args)
    environment = _expected_env(args=args, runner=runner, validation=validation)
    item = _expected_item(runner, validation)
    summary_row = _expected_summary()
    harness_short = _git_short(args.worktree)
    fork_short = _git_short(args.legacy_repo)
    timestamp = _utc()
    connection = sqlite3.connect(args.results_db)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SystemExit("canonical results DB failed integrity check")
        connection.execute("BEGIN IMMEDIATE")
        if _existing_tag_runs(connection, tag=args.tag):
            raise SystemExit("WS32 DSA run tag already exists in results DB")
        cursor = connection.execute(
            "INSERT INTO runs(created_utc,model,model_revision,harness_git,"
            "fork_git,env_json,pod,note) VALUES(?,?,?,?,?,?,?,?)",
            (
                timestamp,
                MODEL,
                REVISION,
                harness_short,
                fork_short,
                _canonical(environment),
                "db-v4-64-od",
                RUN_NOTE,
            ),
        )
        run_id = int(cursor.lastrowid)
        item_cursor = connection.execute(
            "INSERT INTO items(run_id,benchmark,item_id,asked_utc,prompt,gold,"
            "raw_output,extracted,correct,score,n_prompt_tokens,n_gen_tokens,"
            "latency_ms,seed,finish_reason,truncated) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                item["benchmark"],
                item["item_id"],
                timestamp,
                item["prompt"],
                item["gold"],
                item["raw_output"],
                item["extracted"],
                item["correct"],
                item["score"],
                item["n_prompt_tokens"],
                item["n_gen_tokens"],
                item["latency_ms"],
                item["seed"],
                item["finish_reason"],
                item["truncated"],
            ),
        )
        item_id = int(item_cursor.lastrowid)
        summary_cursor = connection.execute(
            "INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,"
            "card_value,delta,note) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                summary_row["benchmark"],
                timestamp,
                summary_row["n"],
                summary_row["metric"],
                summary_row["value"],
                summary_row["card_value"],
                summary_row["delta"],
                summary_row["note"],
            ),
        )
        summary_id = int(summary_cursor.lastrowid)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()

    db_link = {
        "benchmark": BENCHMARK,
        "canonical_results_db": str(args.results_db),
        "item_row_id": item_id,
        "results_db_run_id": run_id,
        "summary_row_id": summary_id,
    }
    terminal_summary = {
        "artifact_kind": ARTIFACT_KIND,
        "association_restored": validation["association_restored"],
        "candidate_mechanisms_proven": validation[
            "candidate_mechanisms_proven"
        ],
        "claim_scope": runner["claim_scope"],
        "code_hash": args.code_hash,
        "diagnostic_only": True,
        "elapsed_seconds": args.elapsed_seconds,
        "exact_arms": validation["exact_arms"],
        "performance_claim": False,
        "remote_prefix": args.remote_prefix,
        "results_db_run_id": run_id,
        "run_tag": args.tag,
        "status": "SUCCESS",
        "validation": validation,
    }
    _atomic_json(args.run_dir / "db_link.json", db_link)
    _atomic_json(args.run_dir / "summary.json", terminal_summary)
    source = sqlite3.connect(args.results_db)
    snapshot = sqlite3.connect(args.run_dir / "results_ckpt.db")
    try:
        source.backup(snapshot)
    finally:
        snapshot.close()
        source.close()
    snapshot = sqlite3.connect(args.run_dir / "results_ckpt.db")
    try:
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SystemExit("WS32 DSA DB snapshot failed integrity check")
    finally:
        snapshot.close()
    print(_canonical(db_link))
    return 0


def _row_dict(cursor: sqlite3.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    return {description[0]: value for description, value in zip(cursor.description, row)}


def _rollback_db(args: argparse.Namespace) -> int:
    runner, validation = _load_terminal_inputs(args)
    environment = _expected_env(args=args, runner=runner, validation=validation)
    expected_item = _expected_item(runner, validation)
    expected_summary = _expected_summary()
    connection = sqlite3.connect(args.results_db)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        matches = _existing_tag_runs(connection, tag=args.tag)
        if not matches:
            connection.rollback()
            print("WS32_DSA_DB_ROLLBACK_NOOP")
            return 0
        if len(matches) != 1:
            raise SystemExit("refusing ambiguous WS32 DSA DB rollback")
        run_id = matches[0]
        cursor = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        run = _row_dict(cursor, cursor.fetchone())
        if (
            run["model"] != MODEL
            or run["model_revision"] != REVISION
            or run["harness_git"] != _git_short(args.worktree)
            or run["fork_git"] != _git_short(args.legacy_repo)
            or json.loads(run["env_json"]) != environment
            or run["pod"] != "db-v4-64-od"
            or run["note"] != RUN_NOTE
        ):
            raise SystemExit("refusing non-identical WS32 DSA run rollback")
        item_cursor = connection.execute("SELECT * FROM items WHERE run_id=?", (run_id,))
        item_rows = item_cursor.fetchall()
        summary_cursor = connection.execute(
            "SELECT * FROM summary WHERE run_id=?", (run_id,)
        )
        summary_rows = summary_cursor.fetchall()
        if len(item_rows) != 1 or len(summary_rows) != 1:
            raise SystemExit("refusing nonterminal WS32 DSA DB rollback")
        item = _row_dict(item_cursor, item_rows[0])
        summary = _row_dict(summary_cursor, summary_rows[0])
        for name, value in expected_item.items():
            if item[name] != value:
                raise SystemExit("refusing non-identical WS32 DSA item rollback")
        for name, value in expected_summary.items():
            if summary[name] != value:
                raise SystemExit("refusing non-identical WS32 DSA summary rollback")
        link_path = args.run_dir / "db_link.json"
        if link_path.exists():
            link = json.loads(link_path.read_text())
            if link != {
                "benchmark": BENCHMARK,
                "canonical_results_db": str(args.results_db),
                "item_row_id": item["id"],
                "results_db_run_id": run_id,
                "summary_row_id": summary["id"],
            }:
                raise SystemExit("WS32 DSA DB link refuses rollback")
        connection.execute("DELETE FROM summary WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM items WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    print(f"WS32_DSA_DB_ROLLED_BACK run_id={run_id}")
    return 0


def _expected_payload_paths(run_dir: Path) -> list[Path]:
    roots = {
        "census_post.txt",
        "census_pre.txt",
        "db_link.json",
        "orchestrator.sealed.log",
        "remote_vacancy.txt",
        "results_ckpt.db",
        "runner.json",
        "runner.log",
        "summary.json",
        "sync.txt",
        "validation.json",
        "ws32_layer0_dsa_association.npz",
    }
    expected = [run_dir / name for name in sorted(roots)]
    expected.extend(sorted((run_dir / "hlo").glob("*")))
    for directory in ("association", "internal", "prompt_cache", "source"):
        expected.extend(sorted((run_dir / "inputs" / directory).glob("*")))
    expected = [path for path in expected if path.is_file()]
    validation = json.loads((run_dir / "validation.json").read_text())
    expected_hlo_count = validation.get("hlo_file_count")
    if type(expected_hlo_count) is not int or expected_hlo_count not in {
        16,
        18,
        20,
    } or len(list((run_dir / "hlo").glob("*"))) != expected_hlo_count or (
        len(expected) != 19 + expected_hlo_count
    ):
        raise SystemExit("WS32 DSA terminal local object set drifted")
    for path in expected:
        if not path.is_file() or ".partial" in path.name:
            raise SystemExit("WS32 DSA terminal payload contains invalid object")
    return expected


def _check_vacant(args: argparse.Namespace) -> int:
    bucket_name, prefix = _split_gs(args.remote_prefix)
    objects = list(storage.Client().list_blobs(bucket_name, prefix=prefix + "/"))
    if objects:
        raise SystemExit("WS32 DSA remote prefix is not vacant")
    print("WS32_DSA_REMOTE_VACANT")
    return 0


def _publish_archive(args: argparse.Namespace) -> int:
    bucket_name, prefix = _split_gs(args.remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if list(client.list_blobs(bucket_name, prefix=prefix + "/")):
        raise SystemExit("WS32 DSA remote prefix changed before publication")
    payload = _expected_payload_paths(args.run_dir)
    evidence_path = args.run_dir / "evidence.sha256"
    evidence = "".join(
        f"{_sha256_file(path)}  {path.relative_to(args.run_dir).as_posix()}\n"
        for path in payload
    )
    evidence_path.write_text(evidence)
    payload.append(evidence_path)
    records = []
    for path in payload:
        name = path.relative_to(args.run_dir).as_posix()
        blob = bucket.blob(f"{prefix}/{name}")
        blob.upload_from_filename(
            str(path), if_generation_match=0, checksum="crc32c"
        )
        blob.reload()
        expected_crc = _crc32c_file(path)
        if (
            blob.size != path.stat().st_size
            or blob.crc32c != expected_crc
            or not blob.generation
        ):
            raise SystemExit(f"WS32 DSA remote payload drifted: {name}")
        records.append(
            {
                "crc32c": expected_crc,
                "generation": int(blob.generation),
                "name": name,
                "sha256": _sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    observed = {
        blob.name.removeprefix(prefix + "/")
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    expected_names = {record["name"] for record in records}
    if observed != expected_names:
        raise SystemExit("WS32 DSA remote payload object set drifted")
    ledger: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_layer0_dsa_remote_objects",
        "code_hash": args.code_hash,
        "objects": records,
        "remote_prefix": args.remote_prefix,
        "run_tag": args.tag,
    }
    ledger["ledger_sha256"] = sha256(_canonical_bytes(ledger)).hexdigest()
    ledger_path = args.run_dir / "remote_objects.json"
    _atomic_json(ledger_path, ledger)
    ledger_blob = bucket.blob(f"{prefix}/remote_objects.json")
    ledger_blob.upload_from_filename(
        str(ledger_path), if_generation_match=0, checksum="crc32c"
    )
    ledger_blob.reload()
    if (
        ledger_blob.size != ledger_path.stat().st_size
        or ledger_blob.crc32c != _crc32c_file(ledger_path)
        or not ledger_blob.generation
    ):
        raise SystemExit("WS32 DSA remote ledger identity drifted")
    observed = {
        blob.name.removeprefix(prefix + "/")
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    if observed != expected_names | {"remote_objects.json"}:
        raise SystemExit("WS32 DSA remote ledger object set drifted")
    print(_canonical(ledger))
    return 0


def _publish_success(args: argparse.Namespace) -> int:
    bucket_name, prefix = _split_gs(args.remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    summary = json.loads((args.run_dir / "summary.json").read_text())
    validation = json.loads((args.run_dir / "validation.json").read_text())
    db_link = json.loads((args.run_dir / "db_link.json").read_text())
    ledger = json.loads((args.run_dir / "remote_objects.json").read_text())
    for record in ledger.get("objects", []):
        if not isinstance(record, dict) or set(record) != {
            "crc32c", "generation", "name", "sha256", "size"
        }:
            raise SystemExit("WS32 DSA remote ledger record drifted")
        payload_blob = bucket.blob(f"{prefix}/{record['name']}")
        payload_blob.reload()
        if (
            int(payload_blob.generation) != record["generation"]
            or payload_blob.size != record["size"]
            or payload_blob.crc32c != record["crc32c"]
        ):
            raise SystemExit(
                f"WS32 DSA payload changed before SUCCESS: {record['name']}"
            )
    ledger_blob = bucket.blob(f"{prefix}/remote_objects.json")
    ledger_blob.reload()
    ledger_path = args.run_dir / "remote_objects.json"
    if (
        ledger_blob.size != ledger_path.stat().st_size
        or ledger_blob.crc32c != _crc32c_file(ledger_path)
        or ledger_blob.download_as_bytes(
            if_generation_match=int(ledger_blob.generation)
        ) != ledger_path.read_bytes()
    ):
        raise SystemExit("WS32 DSA remote ledger changed before SUCCESS")
    success = {
        "artifact_kind": ARTIFACT_KIND,
        "association_restored": validation["association_restored"],
        "candidate_mechanisms_proven": validation[
            "candidate_mechanisms_proven"
        ],
        "code_hash": args.code_hash,
        "db_link_sha256": _sha256_file(args.run_dir / "db_link.json"),
        "diagnostic_only": True,
        "evidence_sha256": _sha256_file(args.run_dir / "evidence.sha256"),
        "exact_arms": validation["exact_arms"],
        "performance_claim": False,
        "post_census_sha256": _sha256_file(args.run_dir / "census_post.txt"),
        "remote_objects_sha256": _sha256_file(
            args.run_dir / "remote_objects.json"
        ),
        "remote_prefix": args.remote_prefix,
        "results_db_run_id": db_link["results_db_run_id"],
        "run_tag": args.tag,
        "status": "SUCCESS",
        "summary_sha256": _sha256_file(args.run_dir / "summary.json"),
        "validation_sha256": _sha256_file(args.run_dir / "validation.json"),
    }
    if summary["results_db_run_id"] != db_link["results_db_run_id"] or (
        ledger["code_hash"] != args.code_hash
    ):
        raise SystemExit("WS32 DSA terminal publication binding drifted")
    success_path = args.run_dir / "SUCCESS"
    _atomic_json(success_path, success)
    blob = bucket.blob(f"{prefix}/SUCCESS")
    blob.upload_from_filename(
        str(success_path), if_generation_match=0, checksum="crc32c"
    )
    blob.reload()
    identity = {
        "crc32c": _crc32c_file(success_path),
        "generation": int(blob.generation),
        "remote": args.remote_prefix + "/SUCCESS",
        "size": success_path.stat().st_size,
    }
    if blob.size != identity["size"] or blob.crc32c != identity["crc32c"]:
        raise SystemExit("WS32 DSA remote SUCCESS identity drifted")
    raw = blob.download_as_bytes(if_generation_match=identity["generation"])
    if raw != success_path.read_bytes():
        raise SystemExit("WS32 DSA remote SUCCESS content drifted")
    names = {
        item.name.removeprefix(prefix + "/")
        for item in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    expected = {item["name"] for item in ledger["objects"]} | {
        "remote_objects.json", "SUCCESS"
    }
    if names != expected:
        raise SystemExit("WS32 DSA terminal remote object set drifted")
    _atomic_json(args.run_dir / "success_upload.json", identity)
    print(_canonical(identity))
    return 0


def _rollback_success(args: argparse.Namespace) -> int:
    success_path = args.run_dir / "SUCCESS"
    if not success_path.is_file():
        print("WS32_DSA_SUCCESS_ROLLBACK_NOOP")
        return 0
    bucket_name, prefix = _split_gs(args.remote_prefix)
    blob = storage.Client().bucket(bucket_name).blob(f"{prefix}/SUCCESS")
    if not blob.exists():
        print("WS32_DSA_SUCCESS_ROLLBACK_NOOP")
        return 0
    blob.reload()
    if (
        blob.size != success_path.stat().st_size
        or blob.crc32c != _crc32c_file(success_path)
        or not blob.generation
    ):
        raise SystemExit("refusing unauthenticated WS32 DSA SUCCESS rollback")
    generation = int(blob.generation)
    raw = blob.download_as_bytes(if_generation_match=generation)
    if raw != success_path.read_bytes():
        raise SystemExit("refusing changed WS32 DSA SUCCESS rollback")
    blob.delete(if_generation_match=generation)
    print(f"WS32_DSA_SUCCESS_ROLLED_BACK generation={generation}")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--association-manifest-sha256", required=True)
    parser.add_argument("--prompt-cache-manifest-sha256", required=True)
    parser.add_argument("--internal-contract-sha256", required=True)
    parser.add_argument("--internal-tensor-sha256", required=True)
    parser.add_argument("--source-summary-sha256", required=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    _common(validate)
    validate.add_argument("--output", type=Path, required=True)
    publish_db = commands.add_parser("publish-db")
    _common(publish_db)
    publish_db.add_argument("--results-db", type=Path, required=True)
    publish_db.add_argument("--worktree", type=Path, required=True)
    publish_db.add_argument("--legacy-repo", type=Path, required=True)
    publish_db.add_argument("--elapsed-seconds", type=int, required=True)
    rollback_db = commands.add_parser("rollback-db")
    _common(rollback_db)
    rollback_db.add_argument("--results-db", type=Path, required=True)
    rollback_db.add_argument("--worktree", type=Path, required=True)
    rollback_db.add_argument("--legacy-repo", type=Path, required=True)
    for name in ("check-vacant", "publish-archive", "publish-success", "rollback-success"):
        command = commands.add_parser(name)
        _common(command)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    _require_sha(args.code_hash, field="code_hash", lengths=(40, 64))
    for field in (
        "association_manifest_sha256",
        "prompt_cache_manifest_sha256",
        "internal_contract_sha256",
        "internal_tensor_sha256",
        "source_summary_sha256",
    ):
        _require_sha(getattr(args, field), field=field)
    if args.command == "validate":
        return _validate_command(args)
    if args.command == "publish-db":
        return _publish_db(args)
    if args.command == "rollback-db":
        return _rollback_db(args)
    if args.command == "check-vacant":
        return _check_vacant(args)
    if args.command == "publish-archive":
        return _publish_archive(args)
    if args.command == "publish-success":
        return _publish_success(args)
    if args.command == "rollback-success":
        return _rollback_success(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
