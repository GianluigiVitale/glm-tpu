#!/usr/bin/env python3
"""Stream the complete final-layout checkpoint directly to approved GCS.

No destination payload is staged on the 97 GB boot disk.  Each topology-local
stage is packed as a group so source tensors are read once and split to their
2/4 final owners with bounded host buffers.  Per-object SHA-256, GCS CRC32C,
generation, size, and an append-only sidecar make interrupted runs resumable
without overwriting accepted objects.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    DestinationFilePlan,
    build_destination_file_plans,
    destination_groups,
    stream_pack_group,
)
from glm_tpu.greenfield.partitioning import (  # noqa: E402
    inspect_layout_manifest,
    read_source_inventory,
)


DEFAULT_SOURCE = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
APPROVED_BUCKET = "driftbench-dsv4-uc"
CHUNK_BYTES = 64 * 1024 * 1024


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("destination must be a gs:// URI")
    bucket, separator, prefix = uri[5:].partition("/")
    if bucket != APPROVED_BUCKET or not separator or not prefix.strip("/"):
        raise ValueError(
            "destination must be a non-root prefix in the approved bucket"
        )
    return bucket, prefix.strip("/")


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"append-only local evidence mismatch at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise RuntimeError(f"append-only local evidence mismatch at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _upload_json_once(blob: Any, value: Mapping[str, Any]) -> None:
    blob.upload_from_string(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )


def _download_json(blob: Any) -> dict[str, Any]:
    decoded = json.loads(blob.download_as_bytes(timeout=300))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"remote JSON object is not a mapping: {blob.name}")
    return decoded


def _evidence_blob_name(prefix: str, filename: str) -> str:
    return f"{prefix}/evidence/{filename}.json"


def _local_evidence_path(run_dir: Path, filename: str) -> Path:
    return run_dir / "file_evidence" / f"{filename}.json"


def _verify_remote_evidence(
    *,
    bucket: Any,
    prefix: str,
    plan: DestinationFilePlan,
    layout_hash: str,
    run_dir: Path,
    code_hash: str,
) -> dict[str, Any] | None:
    data_blob = bucket.blob(f"{prefix}/{plan.filename}")
    evidence_blob = bucket.blob(_evidence_blob_name(prefix, plan.filename))
    data_exists = data_blob.exists()
    evidence_exists = evidence_blob.exists()
    if not data_exists and not evidence_exists:
        return None
    if evidence_exists and not data_exists:
        raise RuntimeError(
            f"remote evidence exists without its payload for {plan.filename}"
        )
    data_blob.reload()
    if data_exists and not evidence_exists:
        if int(data_blob.size) != plan.file_bytes or not data_blob.crc32c:
            raise RuntimeError(
                f"incomplete uncommitted payload for {plan.filename}; preserving it"
            )
        generation = int(data_blob.generation)
        metadata = dict(data_blob.metadata or {})
        object_sha256 = metadata.get("sha256")
        if not isinstance(object_sha256, str) or len(object_sha256) != 64:
            digest = sha256()
            with data_blob.open(
                "rb",
                chunk_size=CHUNK_BYTES,
                if_generation_match=generation,
                timeout=300,
            ) as stream:
                for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    digest.update(chunk)
            object_sha256 = digest.hexdigest()
            metadata["sha256"] = object_sha256
            metadata["file_bytes"] = str(plan.file_bytes)
            metadata["layout_manifest_sha256"] = layout_hash
            data_blob.metadata = metadata
            data_blob.patch(if_generation_match=generation, timeout=300)
            data_blob.reload()
        recovered = {
            "crc32c": data_blob.crc32c,
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "generation": int(data_blob.generation),
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "layout_manifest_sha256": layout_hash,
            "load_set": plan.load_set,
            "pack_code_hash": code_hash,
            "payload_bytes": plan.payload_bytes,
            "sha256": object_sha256,
            "stage_id": plan.stage_id,
        }
        _write_json_once(_local_evidence_path(run_dir, plan.filename), recovered)
        _upload_json_once(evidence_blob, recovered)
        evidence = recovered
    else:
        evidence = _download_json(evidence_blob)
    expected = {
        "destination_filename": plan.filename,
        "device_id": plan.device_id,
        "device_slot": plan.device_slot,
        "file_bytes": plan.file_bytes,
        "header_bytes": len(plan.header),
        "header_sha256": sha256(plan.header).hexdigest(),
        "layout_manifest_sha256": layout_hash,
        "load_set": plan.load_set,
        "payload_bytes": plan.payload_bytes,
        "stage_id": plan.stage_id,
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise RuntimeError(
                f"remote evidence drift for {plan.filename}: field {key}"
            )
    if (
        evidence.get("generation") != int(data_blob.generation)
        or evidence.get("crc32c") != data_blob.crc32c
        or evidence.get("file_bytes") != int(data_blob.size)
        or not isinstance(evidence.get("sha256"), str)
        or len(evidence["sha256"]) != 64
    ):
        raise RuntimeError(
            f"remote object generation/size/checksum drift for {plan.filename}"
        )
    metadata = data_blob.metadata or {}
    if (
        metadata.get("sha256") != evidence["sha256"]
        or metadata.get("layout_manifest_sha256") != layout_hash
    ):
        raise RuntimeError(f"remote object metadata drift for {plan.filename}")
    return evidence


def _stream_pending_group(
    *,
    bucket: Any,
    prefix: str,
    layout: Mapping[str, Any],
    plans: tuple[DestinationFilePlan, ...],
    source_root: Path,
    run_dir: Path,
    code_hash: str,
) -> list[dict[str, Any]]:
    def upload_one(plan: DestinationFilePlan) -> dict[str, Any]:
        blob = bucket.blob(
            f"{prefix}/{plan.filename}", chunk_size=CHUNK_BYTES
        )
        if blob.exists():
            raise RuntimeError(
                f"destination appeared without accepted evidence: {plan.filename}"
            )
        blob.content_type = "application/octet-stream"
        blob.metadata = {
            "greenfield_pack_code_hash": code_hash,
            "layout_manifest_sha256": layout["manifest_sha256"],
            "plan_id": layout["plan_id"],
        }
        stream = blob.open(
            "wb",
            chunk_size=CHUNK_BYTES,
            ignore_flush=True,
            if_generation_match=0,
            checksum="crc32c",
            timeout=300,
        )
        try:
            streamed = stream_pack_group(
                layout=layout,
                plans=(plan,),
                source_root=source_root,
                outputs={plan.filename: stream},
                validate_layout_contract=False,
            )
            stream.close()
        except BaseException:
            try:
                stream.terminate()
            except BaseException:
                pass
            raise
        item = streamed[0]
        blob.reload()
        if int(blob.size) != item.file_bytes or not blob.crc32c:
            raise RuntimeError(
                f"uploaded object size/CRC missing for {plan.filename}"
            )
        generation = int(blob.generation)
        metadata = dict(blob.metadata or {})
        metadata.update(
            {
                "file_bytes": str(item.file_bytes),
                "sha256": item.sha256,
            }
        )
        blob.metadata = metadata
        blob.patch(if_generation_match=generation, timeout=300)
        blob.reload()
        evidence = {
            "crc32c": blob.crc32c,
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": item.file_bytes,
            "generation": int(blob.generation),
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "layout_manifest_sha256": layout["manifest_sha256"],
            "load_set": plan.load_set,
            "pack_code_hash": code_hash,
            "payload_bytes": plan.payload_bytes,
            "sha256": item.sha256,
            "stage_id": plan.stage_id,
        }
        _write_json_once(
            _local_evidence_path(run_dir, plan.filename), evidence
        )
        evidence_blob = bucket.blob(
            _evidence_blob_name(prefix, plan.filename)
        )
        _upload_json_once(evidence_blob, evidence)
        return evidence

    result = []
    with ThreadPoolExecutor(max_workers=len(plans)) as executor:
        futures = {executor.submit(upload_one, plan): plan for plan in plans}
        for future in as_completed(futures):
            plan = futures[future]
            evidence = future.result()
            result.append(evidence)
            print(
                json.dumps(
                    {
                        "event": "file_complete",
                        "file_bytes": evidence["file_bytes"],
                        "filename": plan.filename,
                        "sha256": evidence["sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") != "rewrite/topology-first-decode":
        raise RuntimeError("streaming pack must run on the isolated rewrite branch")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("streaming pack requires a clean worktree")
    bucket_name, prefix = _parse_gs_uri(args.destination)
    if args.resume:
        if not args.run_dir.is_dir():
            raise RuntimeError("resume requires the original local run directory")
    else:
        if args.run_dir.exists():
            raise RuntimeError(
                f"append-only local run already exists: {args.run_dir}"
            )
        args.run_dir.mkdir(parents=True)

    layout = inspect_layout_manifest(args.layout_manifest)
    plans = build_destination_file_plans(layout)
    source_inventory = read_source_inventory(
        args.source_root,
        model_id=layout["source"]["model_id"],
        source_revision=layout["source"]["revision"],
        config_filename=layout["source"]["config_filename"],
    )
    if source_inventory.inventory_sha256 != layout["source"]["inventory_sha256"]:
        raise RuntimeError("live source inventory differs from the layout manifest")
    del source_inventory
    gc.collect()

    control: dict[str, Any] = {
        "artifact_kind": "greenfield_streaming_full_checkpoint_pack",
        "code_hash": code_hash,
        "destination": args.destination.rstrip("/"),
        "destination_file_count": len(plans),
        "expected_file_bytes": sum(plan.file_bytes for plan in plans),
        "layout_file_sha256": _sha256_file(args.layout_manifest),
        "layout_manifest_sha256": layout["manifest_sha256"],
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_revision": layout["source"]["revision"],
    }
    control["control_sha256"] = _hash_mapping(control)
    control_path = args.run_dir / "control.json"
    _write_json_once(control_path, control)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    control_blob = bucket.blob(f"{prefix}/control.json")
    layout_blob = bucket.blob(f"{prefix}/layout_manifest.json")
    success_blob = bucket.blob(f"{prefix}/SUCCESS")
    if args.resume:
        if not control_blob.exists() or _download_json(control_blob) != control:
            raise RuntimeError("remote resume control is absent or mismatched")
        if success_blob.exists():
            raise RuntimeError("remote checkpoint pack is already complete")
        if not layout_blob.exists():
            raise RuntimeError("remote resume layout manifest is missing")
        layout_blob.reload()
        if (
            int(layout_blob.size) != args.layout_manifest.stat().st_size
            or (layout_blob.metadata or {}).get("semantic_sha256")
            != layout["manifest_sha256"]
        ):
            raise RuntimeError("remote resume layout manifest drifted")
    else:
        if next(client.list_blobs(bucket, prefix=f"{prefix}/", max_results=1), None):
            raise RuntimeError("remote destination prefix already exists")
        _upload_json_once(control_blob, control)
        layout_blob.metadata = {
            "semantic_sha256": layout["manifest_sha256"],
            "file_sha256": control["layout_file_sha256"],
        }
        layout_blob.upload_from_filename(
            args.layout_manifest,
            content_type="application/json",
            if_generation_match=0,
            checksum="crc32c",
            timeout=300,
        )

    started = time.monotonic()
    all_evidence: list[dict[str, Any]] = []
    groups = destination_groups(plans)
    for group_index, group in enumerate(groups, start=1):
        accepted = []
        pending = []
        for plan in group:
            evidence = _verify_remote_evidence(
                bucket=bucket,
                prefix=prefix,
                plan=plan,
                layout_hash=layout["manifest_sha256"],
                run_dir=args.run_dir,
                code_hash=code_hash,
            )
            if evidence is None:
                pending.append(plan)
            else:
                _write_json_once(
                    _local_evidence_path(args.run_dir, plan.filename), evidence
                )
                accepted.append(evidence)
        print(
            json.dumps(
                {
                    "event": "group_start",
                    "group": group_index,
                    "groups": len(groups),
                    "load_set": group[0].load_set,
                    "pending_files": len(pending),
                    "resumed_files": len(accepted),
                    "stage_id": group[0].stage_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if pending:
            accepted.extend(
                _stream_pending_group(
                    bucket=bucket,
                    prefix=prefix,
                    layout=layout,
                    plans=tuple(pending),
                    source_root=args.source_root,
                    run_dir=args.run_dir,
                    code_hash=code_hash,
                )
            )
        all_evidence.extend(accepted)
        print(
            json.dumps(
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "event": "group_complete",
                    "group": group_index,
                    "groups": len(groups),
                    "stage_id": group[0].stage_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    by_filename = {record["destination_filename"]: record for record in all_evidence}
    if set(by_filename) != {plan.filename for plan in plans}:
        raise RuntimeError("final packed file evidence is incomplete")
    packed_manifest: dict[str, Any] = {
        "artifact_kind": "greenfield_full_packed_checkpoint",
        "code_hash": code_hash,
        "control_sha256": control["control_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "destination": args.destination.rstrip("/"),
        "elapsed_seconds": time.monotonic() - started,
        "file_count": len(plans),
        "files": [by_filename[plan.filename] for plan in plans],
        "layout_manifest_sha256": layout["manifest_sha256"],
        "packed_file_bytes": sum(record["file_bytes"] for record in all_evidence),
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_payload_bytes": layout["source"]["payload_bytes"],
        "source_revision": layout["source"]["revision"],
    }
    packed_manifest["manifest_sha256"] = _hash_mapping(packed_manifest)
    packed_blob = bucket.blob(f"{prefix}/packed_manifest.json")
    if packed_blob.exists():
        remote_manifest = _download_json(packed_blob)
        remote_unhashed = dict(remote_manifest)
        remote_hash = remote_unhashed.pop("manifest_sha256", None)
        if (
            remote_hash != _hash_mapping(remote_unhashed)
            or remote_manifest.get("control_sha256") != control["control_sha256"]
            or remote_manifest.get("layout_manifest_sha256")
            != layout["manifest_sha256"]
            or remote_manifest.get("files") != packed_manifest["files"]
        ):
            raise RuntimeError("existing remote packed manifest is inconsistent")
        packed_manifest = remote_manifest
    else:
        _upload_json_once(packed_blob, packed_manifest)
    packed_path = args.run_dir / "packed_manifest.json"
    _write_json_once(packed_path, packed_manifest)
    summary = {
        key: value
        for key, value in packed_manifest.items()
        if key != "files"
    }
    summary_path = args.run_dir / "summary.json"
    _write_json_once(summary_path, summary)
    evidence_path = args.run_dir / "evidence.sha256"
    _write_text_once(
        evidence_path,
        "".join(
            f"{_sha256_file(path)}  {path.name}\n"
            for path in (control_path, packed_path, summary_path)
        ),
    )
    success_blob.upload_from_string(
        f"{packed_manifest['manifest_sha256']}  packed_manifest.json\n",
        content_type="text/plain",
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    _write_text_once(
        args.run_dir / "SUCCESS",
        f"{packed_manifest['manifest_sha256']}  packed_manifest.json\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
