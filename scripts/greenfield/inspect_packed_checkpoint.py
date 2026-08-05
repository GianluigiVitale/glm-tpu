#!/usr/bin/env python3
"""Independently inspect a complete final-layout checkpoint in approved GCS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FullCheckpointLoadExpectation,
    verify_full_packed_checkpoint,
)


APPROVED_BUCKET = "driftbench-dsv4-uc"
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
EXPECTED_BRANCH = "rewrite/topology-first-decode"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def _parse_destination(value: str) -> tuple[str, str]:
    if not value.startswith("gs://"):
        raise ValueError("checkpoint destination is not a GCS URI")
    bucket, separator, prefix = value[5:].partition("/")
    if bucket != APPROVED_BUCKET or not separator or not prefix.strip("/"):
        raise ValueError("checkpoint destination is outside the approved prefix")
    return bucket, prefix.strip("/")


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"inspection output is append-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale inspection temporary exists: {temporary}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--packed-manifest-sha256", required=True)
    parser.add_argument("--layout-manifest-sha256", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--plan-group-sha256", required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--layout-code-hash", required=True)
    parser.add_argument("--pack-code-hash", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--plan-id", default="PP8_LP4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("checkpoint inspection requires the isolated branch")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale inspector code: expected={args.expected_code_hash} found={code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("checkpoint inspection requires a clean worktree")
    expectation = FullCheckpointLoadExpectation(
        packed_manifest_sha256=args.packed_manifest_sha256,
        layout_manifest_sha256=args.layout_manifest_sha256,
        source_inventory_sha256=args.source_inventory_sha256,
        source_revision=args.source_revision,
        topology_hash=args.topology_sha256,
        plan_group_hash=args.plan_group_sha256,
        plan_manifest_sha256=args.plan_manifest_sha256,
        execution_plan_sha256=args.execution_plan_sha256,
        layout_code_hash=args.layout_code_hash,
        pack_code_hash=args.pack_code_hash,
        destination=args.destination,
        plan_id=args.plan_id,
    )
    checkpoint = verify_full_packed_checkpoint(
        args.checkpoint_root, expectation
    )
    bucket_name, prefix = _parse_destination(args.destination)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    expected_objects = {
        f"{prefix}/control.json",
        f"{prefix}/layout_manifest.json",
        f"{prefix}/packed_manifest.json",
        f"{prefix}/SUCCESS",
    }
    for plan in checkpoint.plans:
        expected_objects.add(f"{prefix}/{plan.filename}")
        expected_objects.add(f"{prefix}/evidence/{plan.filename}.json")
    observed_blobs = tuple(client.list_blobs(bucket, prefix=f"{prefix}/"))
    observed_names = {blob.name for blob in observed_blobs}
    if observed_names != expected_objects:
        raise RuntimeError(
            "remote checkpoint object set drifted: "
            f"missing={sorted(expected_objects - observed_names)} "
            f"extra={sorted(observed_names - expected_objects)}"
        )
    for plan in checkpoint.plans:
        record = checkpoint.evidence_by_filename[plan.filename]
        data = bucket.blob(f"{prefix}/{plan.filename}")
        sidecar = bucket.blob(f"{prefix}/evidence/{plan.filename}.json")
        data.reload()
        remote_sidecar = json.loads(sidecar.download_as_bytes(timeout=300))
        if remote_sidecar != dict(record):
            raise RuntimeError(f"remote sidecar drift for {plan.filename}")
        if (
            int(data.size) != record["file_bytes"]
            or int(data.generation) != record["generation"]
            or data.crc32c != record["crc32c"]
        ):
            raise RuntimeError(
                f"remote generation/size/CRC drift for {plan.filename}"
            )
        metadata = data.metadata or {}
        if (
            metadata.get("sha256") != record["sha256"]
            or metadata.get("layout_manifest_sha256")
            != expectation.layout_manifest_sha256
            or metadata.get("greenfield_pack_code_hash")
            != expectation.pack_code_hash
            or metadata.get("plan_id") != expectation.plan_id
            or metadata.get("file_bytes") != str(record["file_bytes"])
        ):
            raise RuntimeError(f"remote metadata drift for {plan.filename}")
    summary = {
        "artifact_kind": "greenfield_full_checkpoint_remote_inspection",
        "checkpoint_destination": args.destination,
        "code_hash": code_hash,
        "data_file_count": len(checkpoint.plans),
        "inspected_utc": datetime.now(timezone.utc).isoformat(),
        "layout_manifest_sha256": expectation.layout_manifest_sha256,
        "packed_file_bytes": checkpoint.packed_manifest["packed_file_bytes"],
        "packed_manifest_sha256": expectation.packed_manifest_sha256,
        "packed_payload_bytes": checkpoint.packed_manifest[
            "packed_payload_bytes"
        ],
        "payload_sha256_contract": (
            "packer streaming SHA-256 + immutable GCS generation/CRC32C; "
            "direct loader performs the independent full byte rehash"
        ),
        "plan_id": expectation.plan_id,
        "remote_metadata_verified": True,
        "remote_object_count": len(observed_names),
        "remote_sidecars_verified": True,
        "source_inventory_sha256": expectation.source_inventory_sha256,
        "source_revision": expectation.source_revision,
        "status": "SUCCESS",
    }
    _write_json_once(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
