#!/usr/bin/env python3
"""Restore exact checkpoint objects from a generation-pinned capsule.

The restore is fail-closed, resumable, and writes ``SUCCESS`` last.  It never
copies object payloads or scans unrelated bucket prefixes.  Root-metadata mode
restores only the artifact's top-level control files and terminal marker.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


APPROVED_BUCKET = "driftbench-dsv4-uc"
APPROVED_LOCATION = "US-CENTRAL2"
TPU_LOCK = Path("/home/gianl/glm-run/.glm_pod_workload.lock")
RSYNC_LOCK = Path("/home/gianl/.glm-tpu-rsync.lock")


@dataclass(frozen=True)
class ObjectRecord:
    name: str
    generation: int
    size: int
    crc32c: str


def _canonical_sha256(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _expected_artifact(
    capsule: Mapping[str, Any], artifact_id: str
) -> tuple[str, dict[str, ObjectRecord]]:
    artifacts = capsule.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("capsule artifact inventory is missing")
    matches = [item for item in artifacts if item.get("id") == artifact_id]
    if len(matches) != 1:
        raise ValueError(f"expected one capsule artifact id {artifact_id!r}")
    artifact = matches[0]
    prefix = artifact.get("prefix")
    objects = artifact.get("objects")
    if not isinstance(prefix, str) or not prefix or not prefix.endswith("/"):
        raise ValueError("capsule artifact prefix is invalid")
    if not isinstance(objects, list) or not objects:
        raise ValueError("capsule artifact objects are missing")
    expected: dict[str, ObjectRecord] = {}
    for item in objects:
        record = ObjectRecord(
            name=item.get("name"),
            generation=int(item.get("generation", 0)),
            size=int(item.get("size", -1)),
            crc32c=item.get("crc32c"),
        )
        if (
            not isinstance(record.name, str)
            or not record.name.startswith(prefix)
            or record.generation <= 0
            or record.size < 0
            or not isinstance(record.crc32c, str)
            or not record.crc32c
            or record.name in expected
        ):
            raise ValueError("capsule object identity is invalid")
        expected[record.name] = record
    if f"{prefix}SUCCESS" not in expected:
        raise ValueError("capsule artifact lacks terminal SUCCESS")
    return prefix, expected


def _records(blobs: Iterable[Any]) -> dict[str, list[ObjectRecord]]:
    result: dict[str, list[ObjectRecord]] = {}
    for blob in blobs:
        record = ObjectRecord(
            name=str(blob.name),
            generation=int(blob.generation),
            size=int(blob.size),
            crc32c=str(blob.crc32c),
        )
        result.setdefault(record.name, []).append(record)
    return result


def _select_expected(
    prefix: str,
    expected: Mapping[str, ObjectRecord],
    *,
    root_metadata_only: bool,
) -> dict[str, ObjectRecord]:
    if not root_metadata_only:
        return dict(expected)
    selected = {
        name: record
        for name, record in expected.items()
        if "/" not in name.removeprefix(prefix)
    }
    if f"{prefix}SUCCESS" not in selected or len(selected) < 2:
        raise ValueError("artifact root metadata selection is incomplete")
    return selected


def _reconcile(
    expected: Mapping[str, ObjectRecord],
    active: Mapping[str, list[ObjectRecord]],
    soft_deleted: Mapping[str, list[ObjectRecord]],
) -> tuple[list[ObjectRecord], list[ObjectRecord]]:
    unexpected_active = set(active) - set(expected)
    if unexpected_active:
        raise ValueError(
            f"active prefix contains unexpected objects: {sorted(unexpected_active)}"
        )
    already_active: list[ObjectRecord] = []
    pending: list[ObjectRecord] = []
    for name, wanted in expected.items():
        live = active.get(name, [])
        if live:
            if len(live) != 1 or (
                live[0].size != wanted.size or live[0].crc32c != wanted.crc32c
            ):
                raise ValueError(f"active object does not match capsule: {name}")
            already_active.append(live[0])
            continue
        candidates = [
            item
            for item in soft_deleted.get(name, [])
            if item.generation == wanted.generation
        ]
        if len(candidates) != 1 or (
            candidates[0].size != wanted.size
            or candidates[0].crc32c != wanted.crc32c
        ):
            raise ValueError(
                f"exact soft-deleted generation is unavailable: {name}"
            )
        pending.append(wanted)
    success_name = next(name for name in expected if name.endswith("/SUCCESS"))
    if success_name in active and pending:
        raise ValueError("terminal SUCCESS is live while payload restoration is incomplete")
    return already_active, pending


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def restore(
    *,
    capsule_path: Path,
    artifact_id: str,
    output: Path,
    execute: bool,
    root_metadata_only: bool,
) -> dict[str, Any]:
    capsule = json.loads(capsule_path.read_text())
    if capsule.get("bucket") != APPROVED_BUCKET or capsule.get(
        "bucket_location"
    ) != APPROVED_LOCATION:
        raise ValueError("capsule is not bound to the approved bucket and region")
    if capsule.get("capsule_sha256") != _canonical_sha256(
        capsule, "capsule_sha256"
    ):
        raise ValueError("capsule self hash failed")
    prefix, artifact_objects = _expected_artifact(capsule, artifact_id)
    expected = _select_expected(
        prefix,
        artifact_objects,
        root_metadata_only=root_metadata_only,
    )

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(APPROVED_BUCKET)
    bucket.reload()
    if bucket.location != APPROVED_LOCATION:
        raise ValueError(f"approved bucket location drifted: {bucket.location}")

    def inventories() -> tuple[
        dict[str, list[ObjectRecord]], dict[str, list[ObjectRecord]]
    ]:
        return (
            _records(client.list_blobs(bucket, prefix=prefix)),
            _records(
                client.list_blobs(bucket, prefix=prefix, soft_deleted=True)
            ),
        )

    active, soft = inventories()
    already_active, pending = _reconcile(expected, active, soft)
    plan = {
        "artifact_id": artifact_id,
        "artifact_prefix": prefix,
        "bucket": f"gs://{APPROVED_BUCKET}",
        "capsule_sha256": capsule["capsule_sha256"],
        "expected_bytes": sum(item.size for item in expected.values()),
        "expected_objects": len(expected),
        "live_objects_before": len(already_active),
        "pending_objects": len(pending),
        "selection": "root_metadata" if root_metadata_only else "complete_artifact",
    }
    if not execute:
        return {"status": "DRY_RUN", **plan}
    if output.exists():
        raise FileExistsError(f"restore receipt is append-only: {output}")

    restored: list[dict[str, Any]] = []
    with ExitStack() as stack:
        tpu_lock = stack.enter_context(TPU_LOCK.open("a+"))
        fcntl.flock(tpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rsync_lock = stack.enter_context(RSYNC_LOCK.open("a+"))
        fcntl.flock(rsync_lock, fcntl.LOCK_EX)

        active, soft = inventories()
        already_active, pending = _reconcile(expected, active, soft)
        success = expected[f"{prefix}SUCCESS"]
        ordered = sorted(
            (item for item in pending if item.name != success.name),
            key=lambda item: item.name,
        )
        if success in pending:
            ordered.append(success)
        for wanted in ordered:
            blob = bucket.restore_blob(
                wanted.name,
                generation=wanted.generation,
                if_generation_match=0,
            )
            observed = ObjectRecord(
                name=str(blob.name),
                generation=int(blob.generation),
                size=int(blob.size),
                crc32c=str(blob.crc32c),
            )
            if (
                observed.name != wanted.name
                or observed.size != wanted.size
                or observed.crc32c != wanted.crc32c
            ):
                raise ValueError(f"restored object identity drifted: {wanted.name}")
            restored.append(
                {
                    "crc32c": observed.crc32c,
                    "name": observed.name,
                    "restored_generation": observed.generation,
                    "size": observed.size,
                    "source_generation": wanted.generation,
                }
            )

        active, soft = inventories()
        final_active, final_pending = _reconcile(expected, active, soft)
        if final_pending or len(final_active) != len(expected):
            raise ValueError("final active checkpoint inventory is incomplete")

        final_inventory = [
            {
                "crc32c": item.crc32c,
                "generation": item.generation,
                "name": item.name,
                "size": item.size,
            }
            for item in sorted(final_active, key=lambda item: item.name)
        ]

    receipt: dict[str, Any] = {
        "artifact_kind": "greenfield_soft_deleted_checkpoint_restore_receipt",
        **plan,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_active_bytes": sum(item["size"] for item in final_inventory),
        "final_active_objects": final_inventory,
        "restored_objects": restored,
        "status": "SUCCESS",
        "terminal_success_restored_last_this_execution": (
            restored[-1]["name"].endswith("/SUCCESS") if restored else None
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt, "receipt_sha256")
    _write_json(output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-metadata-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = restore(
        capsule_path=args.capsule,
        artifact_id=args.artifact_id,
        output=args.output,
        execute=args.execute,
        root_metadata_only=args.root_metadata_only,
    )
    print(
        "CHECKPOINT_RESTORE "
        f"status={result['status']} artifact={result['artifact_id']} "
        f"objects={result['expected_objects']} bytes={result['expected_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
