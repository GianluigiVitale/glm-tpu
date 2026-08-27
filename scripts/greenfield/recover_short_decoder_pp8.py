#!/usr/bin/env python3
"""Seal an already validated PP8 short-decoder run without TPU execution."""

from __future__ import annotations

import argparse
import base64
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sqlite3
from typing import Any, BinaryIO

import google_crc32c
from google.cloud import storage


APPROVED_BUCKET = "driftbench-dsv4-uc"
APPROVED_PREFIX = "results/"
TERMINAL_FILES = {"SUCCESS", "remote_objects.json"}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


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


def _parse_results_uri(uri: str) -> tuple[str, str]:
    root = f"gs://{APPROVED_BUCKET}/{APPROVED_PREFIX}"
    if not uri.startswith(root):
        raise ValueError("recovery must remain below the approved results prefix")
    prefix = uri.removeprefix(f"gs://{APPROVED_BUCKET}/").rstrip("/") + "/"
    if prefix == APPROVED_PREFIX:
        raise ValueError("recovery prefix must name one run")
    return APPROVED_BUCKET, prefix


def _require_census(path: Path) -> None:
    workers = re.findall(
        r"^CENSUS_OK\s+(\S*?-w-[0-7])$", path.read_text(), re.MULTILINE
    )
    if len(workers) != 8 or len(set(workers)) != 8:
        raise ValueError(f"census is not authenticated 8/8 zero work: {path}")


def _files(root: Path, *, include_evidence: bool = True) -> dict[str, Path]:
    result = {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in TERMINAL_FILES
    }
    if not include_evidence:
        result.pop("evidence.sha256", None)
    return result


def _verify_hash_manifest(root: Path) -> None:
    seen: set[str] = set()
    for line in (root / "evidence.sha256").read_text().splitlines():
        expected, separator, relative = line.partition("  ")
        if (
            not separator
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or relative in seen
        ):
            raise ValueError("evidence manifest is malformed")
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"evidence hash drifted: {relative}")
        seen.add(relative)
    expected = set(_files(root, include_evidence=False))
    if seen != expected:
        raise ValueError("evidence manifest file set drifted")


def _table_rows(
    connection: sqlite3.Connection, table: str, *, exclude_run: int | None = None
) -> list[tuple[Any, ...]]:
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    query = f"SELECT * FROM {table}"
    values: tuple[object, ...] = ()
    if exclude_run is not None and "run_id" in columns:
        query += " WHERE run_id != ?"
        values = (exclude_run,)
    query += " ORDER BY rowid"
    return connection.execute(query, values).fetchall()


def restore_db(snapshot_path: Path, live_path: Path, run_id: int) -> None:
    """Restore only one rolled-back run after proving an exact DB prefix."""
    snapshot = sqlite3.connect(snapshot_path)
    live = sqlite3.connect(live_path)
    try:
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("snapshot DB integrity failed")
        if live.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("live DB integrity failed")
        tables = [
            row[0]
            for row in snapshot.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        live_tables = [
            row[0]
            for row in live.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        if tables != live_tables:
            raise ValueError("snapshot/live DB schemas drifted")
        live.execute("BEGIN IMMEDIATE")
        for table in tables:
            if _table_rows(snapshot, table, exclude_run=run_id) != _table_rows(
                live, table
            ):
                raise ValueError(f"live DB is not the exact snapshot prefix: {table}")
        if live.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]:
            raise ValueError("provisional run unexpectedly remains in live DB")
        for table in ("runs", "items", "summary"):
            columns = [
                row[1] for row in snapshot.execute(f"PRAGMA table_info({table})")
            ]
            if "run_id" not in columns:
                raise ValueError(f"missing run_id column: {table}")
            rows = snapshot.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            if not rows:
                raise ValueError(f"snapshot has no provisional rows: {table}")
            placeholders = ",".join("?" for _ in columns)
            live.executemany(
                f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                rows,
            )
        for table in ("runs", "items", "summary"):
            expected = snapshot.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            actual = live.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            if actual != expected:
                raise ValueError(f"restored DB rows drifted: {table}")
        live.commit()
    except Exception:
        live.rollback()
        raise
    finally:
        snapshot.close()
        live.close()


def rollback_db(snapshot_path: Path, live_path: Path, run_id: int) -> None:
    """Remove an exactly matching recovery publication after a seal failure."""
    snapshot = sqlite3.connect(snapshot_path)
    live = sqlite3.connect(live_path)
    try:
        live.execute("BEGIN IMMEDIATE")
        for table in ("runs", "items", "summary"):
            expected = snapshot.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            actual = live.execute(
                f"SELECT * FROM {table} WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            if actual != expected:
                raise ValueError(f"refusing non-identical recovery rollback: {table}")
        live.execute("DELETE FROM summary WHERE run_id = ?", (run_id,))
        live.execute("DELETE FROM items WHERE run_id = ?", (run_id,))
        live.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        live.commit()
    except Exception:
        live.rollback()
        raise
    finally:
        snapshot.close()
        live.close()


def _verify_source_objects(
    *,
    client: storage.Client,
    source_dir: Path,
    source_uri: str,
    source_tag: str,
) -> list[dict[str, object]]:
    bucket_name, prefix = _parse_results_uri(source_uri)
    if not prefix.endswith(f"/{source_tag}/"):
        raise ValueError("source URI and run tag disagree")
    bucket = client.bucket(bucket_name)
    records: list[dict[str, object]] = []
    for relative, path in sorted(_files(source_dir).items()):
        crc = _crc32c_file(path)
        candidates = (
            prefix + relative,
            prefix + f"diagnostic_local/{source_tag}/" + relative,
        )
        blob = next(
            (
                candidate
                for name in candidates
                if (candidate := bucket.get_blob(name)) is not None
                and int(candidate.size) == path.stat().st_size
                and candidate.crc32c == crc
                and candidate.generation
            ),
            None,
        )
        if blob is None:
            raise ValueError(f"preserved source object is missing: {relative}")
        records.append(
            {
                "crc32c": crc,
                "generation": int(blob.generation),
                "name": blob.name,
                "recovered_as": relative,
                "sha256": _sha256_file(path),
                "size": int(blob.size),
            }
        )
    return records


def prepare(
    *,
    source_dir: Path,
    run_dir: Path,
    recovery_census: Path,
    source_uri: str,
    remote_uri: str,
    source_tag: str,
    recovery_tag: str,
    workload_hash: str,
    recovery_hash: str,
    client: storage.Client,
) -> dict[str, Any]:
    if run_dir.exists():
        raise ValueError(f"recovery directory already exists: {run_dir}")
    if len(workload_hash) != 40 or len(recovery_hash) != 40:
        raise ValueError("full workload and recovery Git hashes are required")
    _, source_prefix = _parse_results_uri(source_uri)
    _, remote_prefix = _parse_results_uri(remote_uri)
    if source_prefix == remote_prefix or not remote_prefix.endswith(
        f"/{recovery_tag}/"
    ):
        raise ValueError("recovery requires a distinct tag-matching prefix")
    for name in ("census_pre.txt", "census_post.txt"):
        _require_census(source_dir / name)
    _require_census(recovery_census)
    summary = json.loads((source_dir / "summary.json").read_text())
    if (
        summary.get("gate_d_passed") is not True
        or summary.get("raw_token_claim") is not True
        or summary.get("code_hash") != workload_hash
        or summary.get("host_count") != 8
        or summary.get("context_capacity") != 2048
    ):
        raise ValueError("source Gate-D summary contract failed")
    run_id = summary.get("results_db_run_id")
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError("source summary has no DB run id")
    rollback = (source_dir / "provisional_db_rollback.txt").read_text().strip()
    if rollback != f"ROLLED_BACK_PROVISIONAL_DB_RUN={run_id}":
        raise ValueError("source provisional DB rollback receipt drifted")
    source_objects = _verify_source_objects(
        client=client,
        source_dir=source_dir,
        source_uri=source_uri,
        source_tag=source_tag,
    )
    run_dir.mkdir(parents=True)
    for relative, source in sorted(_files(source_dir).items()):
        if relative == "evidence.sha256":
            continue
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(recovery_census, run_dir / "census_recovery.txt")
    recovery: dict[str, Any] = {
        "artifact_kind": "greenfield_pp8_short_decoder_gate_d_recovery",
        "gate_d_passed": True,
        "model_workload_rerun": False,
        "numerical_claim": True,
        "performance_claim": True,
        "recovery_code_hash": recovery_hash,
        "recovery_run_tag": recovery_tag,
        "remote_prefix": remote_uri,
        "results_db_run_id": run_id,
        "source_objects": source_objects,
        "source_remote_prefix": source_uri,
        "source_run_tag": source_tag,
        "tpu_initialized_by_recovery": False,
        "workload_code_hash": workload_hash,
    }
    recovery["recovery_sha256"] = sha256(_canonical(recovery)).hexdigest()
    (run_dir / "recovery.json").write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n"
    )
    evidence = _files(run_dir, include_evidence=False)
    (run_dir / "evidence.sha256").write_text(
        "".join(
            f"{_sha256_file(path)}  {name}\n"
            for name, path in sorted(evidence.items())
        )
    )
    _verify_hash_manifest(run_dir)
    return {"recovery": recovery, "summary": summary}


def publish(
    *,
    run_dir: Path,
    remote_uri: str,
    recovery: dict[str, Any],
    summary: dict[str, Any],
    client: storage.Client,
) -> dict[str, Any]:
    bucket_name, prefix = _parse_results_uri(remote_uri)
    if list(client.list_blobs(bucket_name, prefix=prefix, max_results=1)):
        raise ValueError("recovery prefix is not vacant")
    bucket = client.bucket(bucket_name)
    _verify_hash_manifest(run_dir)
    records: list[dict[str, object]] = []
    for name, path in sorted(_files(run_dir).items()):
        blob = bucket.blob(prefix + name)
        blob.upload_from_filename(
            path, if_generation_match=0, checksum="crc32c", timeout=600
        )
        blob.reload()
        crc = _crc32c_file(path)
        if (
            int(blob.size) != path.stat().st_size
            or blob.crc32c != crc
            or not blob.generation
        ):
            raise ValueError(f"remote identity drifted: {name}")
        records.append(
            {
                "crc32c": crc,
                "generation": int(blob.generation),
                "name": name,
                "sha256": _sha256_file(path),
                "size": int(blob.size),
            }
        )
    ledger: dict[str, Any] = {"objects": records, "remote_prefix": remote_uri}
    ledger["ledger_sha256"] = sha256(_canonical(ledger)).hexdigest()
    ledger_path = run_dir / "remote_objects.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    ledger_blob = bucket.blob(prefix + "remote_objects.json")
    ledger_blob.upload_from_filename(
        ledger_path, if_generation_match=0, checksum="crc32c", timeout=600
    )
    ledger_blob.reload()
    ledger_crc = _crc32c_file(ledger_path)
    if int(ledger_blob.size) != ledger_path.stat().st_size or (
        ledger_blob.crc32c != ledger_crc
    ):
        raise ValueError("remote ledger identity drifted")
    success: dict[str, Any] = {
        "artifact_kind": "greenfield_pp8_short_decoder_gate_d_recovery_SUCCESS",
        "code_hash": recovery["workload_code_hash"],
        "gate_d_passed": True,
        "model_workload_rerun": False,
        "numerical_claim": True,
        "performance_claim": True,
        "recovery_code_hash": recovery["recovery_code_hash"],
        "remote_ledger_crc32c": ledger_crc,
        "remote_ledger_generation": int(ledger_blob.generation),
        "remote_ledger_sha256": ledger["ledger_sha256"],
        "results_db_run_id": recovery["results_db_run_id"],
        "summary_sha256": _sha256_file(run_dir / "summary.json"),
    }
    success["success_sha256"] = sha256(_canonical(success)).hexdigest()
    success_path = run_dir / "SUCCESS"
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True) + "\n")
    success_blob = bucket.blob(prefix + "SUCCESS")
    success_blob.upload_from_filename(
        success_path, if_generation_match=0, checksum="crc32c", timeout=600
    )
    success_blob.reload()
    if int(success_blob.size) != success_path.stat().st_size or (
        success_blob.crc32c != _crc32c_file(success_path)
    ):
        raise ValueError("terminal SUCCESS identity drifted")
    observed = {
        blob.name.removeprefix(prefix)
        for blob in client.list_blobs(bucket_name, prefix=prefix)
    }
    expected = set(_files(run_dir)) | {"remote_objects.json", "SUCCESS"}
    if observed != expected:
        raise ValueError("terminal remote object set drifted")
    return {
        "ledger_sha256": ledger["ledger_sha256"],
        "object_count": len(expected),
        "success_sha256": success["success_sha256"],
    }


def _lock(path: Path) -> BinaryIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    return stream


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--recovery-census", required=True, type=Path)
    parser.add_argument("--source-remote-prefix", required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--source-run-tag", required=True)
    parser.add_argument("--recovery-run-tag", required=True)
    parser.add_argument("--workload-code-hash", required=True)
    parser.add_argument("--recovery-code-hash", required=True)
    parser.add_argument(
        "--results-db",
        default=Path("/home/gianl/glm-tpu/bench/results.db"),
        type=Path,
    )
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    # Match the protected wrapper's global lock order.
    locks = (
        _lock(Path("/home/gianl/glm-run/.glm_pod_workload.lock")),
        _lock(Path("/home/gianl/.glm-tpu-rsync.lock")),
    )
    try:
        client = storage.Client()
        prepared = prepare(
            source_dir=args.source_dir,
            run_dir=args.run_dir,
            recovery_census=args.recovery_census,
            source_uri=args.source_remote_prefix,
            remote_uri=args.remote_prefix,
            source_tag=args.source_run_tag,
            recovery_tag=args.recovery_run_tag,
            workload_hash=args.workload_code_hash,
            recovery_hash=args.recovery_code_hash,
            client=client,
        )
        if not args.publish:
            print(
                "PP8_GATE_D_RECOVERY_PREPARED "
                f"sha={prepared['recovery']['recovery_sha256']}"
            )
            return 0
        run_id = int(prepared["recovery"]["results_db_run_id"])
        restore_db(args.run_dir / "results_ckpt.db", args.results_db, run_id)
        try:
            result = publish(
                run_dir=args.run_dir,
                remote_uri=args.remote_prefix,
                recovery=prepared["recovery"],
                summary=prepared["summary"],
                client=client,
            )
        except Exception:
            rollback_db(args.run_dir / "results_ckpt.db", args.results_db, run_id)
            raise
        print(
            "PP8_GATE_D_RECOVERY_SEALED "
            f"objects={result['object_count']} ledger={result['ledger_sha256']}"
        )
        return 0
    finally:
        for stream in reversed(locks):
            stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
