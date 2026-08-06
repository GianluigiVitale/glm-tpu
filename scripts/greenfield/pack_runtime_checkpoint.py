#!/usr/bin/env python3
"""Build the executable-ready PP8 decoder checkpoint as an append-only artifact.

The command is intentionally split into ``prepare``, ``pack-stage``, and
``finalize`` modes.  A protected harness can prepare once, run one stage on
each of the eight existing hosts, and commit the manifest only after all 32
device files and sidecars reconcile.  Payloads stream from the protected
final-owner checkpoint to GCS; no 26 GB destination file is staged on disk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FullCheckpointLoadExpectation,
    RUNTIME_FORMAT_VERSION,
    RUNTIME_PACK_CONTROL_KIND,
    RUNTIME_PACKED_ARTIFACT_KIND,
    RuntimeDestinationFilePlan,
    VerifiedPackedCheckpoint,
    build_runtime_destination_file_plans,
    build_runtime_layout_document,
    stream_runtime_weight_file,
    verify_full_packed_checkpoint,
    verify_source_file_sha256,
)
from glm_tpu.greenfield.model import (  # noqa: E402
    DecoderRuntimeWeightLayout,
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.partitioning import inspect_layout_manifest  # noqa: E402
from glm_tpu.greenfield.types import ExecutionPlan  # noqa: E402


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


def _mapping_hash(value: Mapping[str, Any], *, hash_field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(hash_field, None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


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


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("destination must be a gs:// URI")
    bucket, separator, prefix = uri[5:].partition("/")
    if bucket != APPROVED_BUCKET or not separator or not prefix.strip("/"):
        raise ValueError("destination must be a non-root approved-bucket prefix")
    if not prefix.startswith("checkpoints/greenfield/"):
        raise ValueError("runtime destination must remain under checkpoints/greenfield")
    return bucket, prefix.strip("/")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args],
        text=True,
    ).strip()


def _verify_repo(expected_code_hash: str) -> str:
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {expected_code_hash}, found {code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("runtime pack requires a clean worktree")
    return code_hash


def _upload_json_once(blob: Any, value: Mapping[str, Any]) -> None:
    blob.upload_from_string(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )


def _download_json(blob: Any) -> dict[str, Any]:
    value = json.loads(blob.download_as_bytes(timeout=300))
    if not isinstance(value, dict):
        raise RuntimeError(f"remote JSON object is not a mapping: {blob.name}")
    return value


@dataclass(frozen=True, slots=True)
class PackContext:
    source_checkpoint: VerifiedPackedCheckpoint
    layout: DecoderRuntimeWeightLayout
    layout_document: Mapping[str, Any]
    layout_bytes: bytes
    plans: tuple[RuntimeDestinationFilePlan, ...]
    common: Mapping[str, Any]
    control: Mapping[str, Any]


def _source_expectation(
    root: Path,
    expected_manifest_sha256: str,
) -> FullCheckpointLoadExpectation:
    packed = _read_json(root / "packed_manifest.json")
    if (
        _mapping_hash(packed, hash_field="manifest_sha256")
        != expected_manifest_sha256
    ):
        raise RuntimeError("protected source packed manifest hash drifted")
    layout = inspect_layout_manifest(root / "layout_manifest.json")
    plan_manifest = layout["plan_manifest"]
    return FullCheckpointLoadExpectation(
        packed_manifest_sha256=expected_manifest_sha256,
        layout_manifest_sha256=layout["manifest_sha256"],
        source_inventory_sha256=packed["source_inventory_sha256"],
        source_revision=packed["source_revision"],
        topology_hash=layout["topology_hash"],
        plan_group_hash=layout["plan_group_hash"],
        plan_manifest_sha256=layout["plan_manifest_sha256"],
        execution_plan_sha256=plan_manifest["execution_plan_sha256"],
        layout_code_hash=layout["code_hash"],
        pack_code_hash=packed["code_hash"],
        destination=packed["destination"],
        plan_id=packed["plan_id"],
        model_id=layout["source"]["model_id"],
    )


def _build_context(args: argparse.Namespace, code_hash: str) -> PackContext:
    source_expectation = _source_expectation(
        args.source_checkpoint_root,
        args.source_packed_manifest_sha256,
    )
    source_checkpoint = verify_full_packed_checkpoint(
        args.source_checkpoint_root,
        source_expectation,
    )
    plan_value = source_checkpoint.layout["plan_manifest"]["execution_plan"]
    plan = ExecutionPlan.from_dict(plan_value)
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    layout_document = build_runtime_layout_document(layout)
    layout_bytes = (
        json.dumps(layout_document, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    source_plans = tuple(
        item for item in source_checkpoint.plans if item.load_set == "base_decoder"
    )
    plans = build_runtime_destination_file_plans(
        layout,
        source_plans,
        source_packed_manifest_sha256=args.source_packed_manifest_sha256,
    )
    common: dict[str, Any] = {
        "destination": args.destination.rstrip("/"),
        "file_count": len(plans),
        "format_version": RUNTIME_FORMAT_VERSION,
        "model_id": plan.geometry.model_id,
        "pack_code_hash": code_hash,
        "padding_bytes": sum(device.padding_bytes for device in layout.devices),
        "plan_hash": plan.plan_hash,
        "plan_id": plan.name.value,
        "runtime_file_bytes": sum(item.file_bytes for item in plans),
        "runtime_layout_hash": layout.layout_hash,
        "runtime_layout_manifest_sha256": layout_document["manifest_sha256"],
        "runtime_payload_bytes": sum(item.payload_bytes for item in plans),
        "schedule_hash": schedule.schedule_hash,
        "source_checkpoint_destination": source_expectation.destination,
        "source_layout_manifest_sha256": (
            source_expectation.layout_manifest_sha256
        ),
        "source_leaf_count": layout.source_leaf_count,
        "source_packed_manifest_sha256": args.source_packed_manifest_sha256,
        "source_payload_bytes": sum(
            device.source_bytes for device in layout.devices
        ),
        "tensor_count": len(layout.specs) * len(layout.devices),
    }
    control: dict[str, Any] = {
        "artifact_kind": RUNTIME_PACK_CONTROL_KIND,
        **common,
        "runtime_layout_file_sha256": sha256(layout_bytes).hexdigest(),
    }
    control["control_sha256"] = _mapping_hash(
        control,
        hash_field="control_sha256",
    )
    return PackContext(
        source_checkpoint=source_checkpoint,
        layout=layout,
        layout_document=layout_document,
        layout_bytes=layout_bytes,
        plans=plans,
        common=common,
        control=control,
    )


def _remote_prerequisites(
    *,
    bucket: Any,
    prefix: str,
    context: PackContext,
) -> None:
    control_blob = bucket.blob(f"{prefix}/control.json")
    layout_blob = bucket.blob(f"{prefix}/runtime_layout.json")
    if not control_blob.exists() or not layout_blob.exists():
        raise RuntimeError("runtime pack has not been prepared")
    if _download_json(control_blob) != context.control:
        raise RuntimeError("remote runtime-pack control drifted")
    layout_blob.reload()
    if (
        int(layout_blob.size) != len(context.layout_bytes)
        or (layout_blob.metadata or {}).get("semantic_sha256")
        != context.layout_document["manifest_sha256"]
        or (layout_blob.metadata or {}).get("file_sha256")
        != context.control["runtime_layout_file_sha256"]
    ):
        raise RuntimeError("remote runtime layout identity drifted")


def _prepare(
    args: argparse.Namespace,
    context: PackContext,
    bucket: Any,
    prefix: str,
) -> None:
    if args.run_dir.exists():
        raise RuntimeError(f"append-only local run already exists: {args.run_dir}")
    if next(bucket.list_blobs(prefix=f"{prefix}/", max_results=1), None):
        raise RuntimeError("remote runtime destination prefix already exists")
    args.run_dir.mkdir(parents=True)
    control_path = args.run_dir / "control.json"
    layout_path = args.run_dir / "runtime_layout.json"
    _write_json_once(control_path, context.control)
    _write_json_once(layout_path, context.layout_document)
    control_blob = bucket.blob(f"{prefix}/control.json")
    layout_blob = bucket.blob(f"{prefix}/runtime_layout.json")
    _upload_json_once(control_blob, context.control)
    layout_blob.metadata = {
        "file_sha256": context.control["runtime_layout_file_sha256"],
        "runtime_layout_hash": context.layout.layout_hash,
        "semantic_sha256": context.layout_document["manifest_sha256"],
    }
    layout_blob.upload_from_filename(
        layout_path,
        content_type="application/json",
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    summary = {
        "control_sha256": context.control["control_sha256"],
        **context.common,
    }
    _write_json_once(args.run_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _static_file_record(
    plan: RuntimeDestinationFilePlan,
    context: PackContext,
) -> dict[str, Any]:
    source_record = context.source_checkpoint.evidence_by_filename[
        plan.source_filename
    ]
    return {
        "destination_filename": plan.filename,
        "device_id": plan.device_id,
        "device_slot": plan.device_slot,
        "file_bytes": plan.file_bytes,
        "header_bytes": len(plan.header),
        "header_sha256": sha256(plan.header).hexdigest(),
        "padding_bytes": plan.device_layout.padding_bytes,
        "payload_bytes": plan.payload_bytes,
        "runtime_layout_hash": plan.runtime_layout_hash,
        "source_file_sha256": source_record["sha256"],
        "source_filename": plan.source_filename,
        "source_leaf_count": plan.device_layout.source_leaf_count,
        "source_packed_manifest_sha256": (
            plan.source_packed_manifest_sha256
        ),
        "source_payload_bytes": plan.device_layout.source_bytes,
        "stage_id": plan.stage_id,
        "tensor_count": len(plan.tensors),
    }


def _validate_remote_record(
    *,
    bucket: Any,
    prefix: str,
    plan: RuntimeDestinationFilePlan,
    context: PackContext,
) -> dict[str, Any] | None:
    payload_blob = bucket.blob(f"{prefix}/{plan.filename}")
    evidence_blob = bucket.blob(f"{prefix}/evidence/{plan.filename}.json")
    payload_exists = payload_blob.exists()
    evidence_exists = evidence_blob.exists()
    if not payload_exists and not evidence_exists:
        return None
    if payload_exists != evidence_exists:
        raise RuntimeError(
            f"partial runtime destination exists for {plan.filename}; preserving it"
        )
    payload_blob.reload()
    record = _download_json(evidence_blob)
    for field, expected in _static_file_record(plan, context).items():
        if record.get(field) != expected:
            raise RuntimeError(
                f"remote runtime evidence drift for {plan.filename}: {field}"
            )
    if (
        record.get("generation") != int(payload_blob.generation)
        or record.get("crc32c") != payload_blob.crc32c
        or record.get("file_bytes") != int(payload_blob.size)
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
        or not isinstance(record.get("tensors"), list)
        or len(record["tensors"]) != len(plan.tensors)
    ):
        raise RuntimeError(f"remote runtime object drift for {plan.filename}")
    bindings = {
        tensor.spec.name: tensor for tensor in plan.device_layout.tensors
    }
    tensor_records = record["tensors"]
    by_name = {
        item.get("name"): item
        for item in tensor_records
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(by_name) != len(tensor_records):
        raise RuntimeError(
            f"remote runtime tensor names drift for {plan.filename}"
        )
    for tensor in plan.tensors:
        tensor_record = by_name.get(tensor.spec.name)
        if (
            tensor_record is None
            or tensor_record.get("byte_count") != tensor.byte_count
            or tensor_record.get("padding")
            != bindings[tensor.spec.name].is_padding
            or not isinstance(tensor_record.get("sha256"), str)
            or len(tensor_record["sha256"]) != 64
        ):
            raise RuntimeError(
                f"remote runtime tensor evidence drift for "
                f"{plan.filename}:{tensor.spec.name}"
            )
    metadata = payload_blob.metadata or {}
    if (
        metadata.get("sha256") != record["sha256"]
        or metadata.get("runtime_layout_hash") != context.layout.layout_hash
        or metadata.get("source_file_sha256") != record["source_file_sha256"]
    ):
        raise RuntimeError(f"remote runtime metadata drift for {plan.filename}")
    return record


def _pack_one(
    *,
    bucket: Any,
    prefix: str,
    plan: RuntimeDestinationFilePlan,
    context: PackContext,
    run_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    existing = _validate_remote_record(
        bucket=bucket,
        prefix=prefix,
        plan=plan,
        context=context,
    )
    if existing is not None:
        if not resume:
            raise RuntimeError(f"runtime file already exists: {plan.filename}")
        _write_json_once(
            run_dir / "file_evidence" / f"{plan.filename}.json",
            existing,
        )
        return existing
    source_plans = {
        item.filename: item for item in context.source_checkpoint.plans
    }
    source_plan = source_plans[plan.source_filename]
    source_path = context.source_checkpoint.root / plan.source_filename
    source_file_sha256 = context.source_checkpoint.evidence_by_filename[
        plan.source_filename
    ]["sha256"]
    verify_source_file_sha256(source_path, source_file_sha256)
    payload_blob = bucket.blob(
        f"{prefix}/{plan.filename}",
        chunk_size=CHUNK_BYTES,
    )
    payload_blob.content_type = "application/octet-stream"
    payload_blob.metadata = {
        "runtime_layout_hash": context.layout.layout_hash,
        "source_file_sha256": source_file_sha256,
        "source_packed_manifest_sha256": plan.source_packed_manifest_sha256,
    }
    output = payload_blob.open(
        "wb",
        chunk_size=CHUNK_BYTES,
        ignore_flush=True,
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    try:
        with source_path.open("rb", buffering=0) as source:
            evidence = stream_runtime_weight_file(
                source_plan=source_plan,
                runtime_plan=plan,
                source=source,
                output=output,
                verified_source_file_sha256=source_file_sha256,
                chunk_bytes=CHUNK_BYTES,
            )
        output.close()
    except BaseException:
        try:
            output.terminate()
        except BaseException:
            pass
        raise
    payload_blob.reload()
    if int(payload_blob.size) != evidence.file_bytes or not payload_blob.crc32c:
        raise RuntimeError(f"runtime upload did not seal for {plan.filename}")
    metadata = dict(payload_blob.metadata or {})
    metadata.update(
        {
            "file_bytes": str(evidence.file_bytes),
            "sha256": evidence.sha256,
        }
    )
    payload_blob.metadata = metadata
    payload_blob.patch(
        if_generation_match=int(payload_blob.generation),
        timeout=300,
    )
    payload_blob.reload()
    record = {
        **evidence.to_dict(),
        **_static_file_record(plan, context),
        "crc32c": payload_blob.crc32c,
        "generation": int(payload_blob.generation),
        "sha256": evidence.sha256,
        "tensors": [item.to_dict() for item in evidence.tensors],
    }
    record.pop("filename", None)
    evidence_path = run_dir / "file_evidence" / f"{plan.filename}.json"
    _write_json_once(evidence_path, record)
    evidence_blob = bucket.blob(f"{prefix}/evidence/{plan.filename}.json")
    _upload_json_once(evidence_blob, record)
    return record


def _pack_stage(
    args: argparse.Namespace,
    context: PackContext,
    bucket: Any,
    prefix: str,
) -> None:
    _remote_prerequisites(bucket=bucket, prefix=prefix, context=context)
    assignments = {
        stage.assignment.process_index: stage.assignment.stage_id
        for stage in build_pipeline_schedule(
            ExecutionPlan.from_dict(
                context.source_checkpoint.layout["plan_manifest"][
                    "execution_plan"
                ]
            )
        ).stages
    }
    if args.process_index not in assignments:
        raise RuntimeError(
            f"process index {args.process_index} owns no PP8 runtime stage"
        )
    stage_id = assignments[args.process_index]
    plans = tuple(plan for plan in context.plans if plan.stage_id == stage_id)
    if len(plans) != 4:
        raise RuntimeError("PP8 runtime stage does not contain four files")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _pack_one,
                bucket=bucket,
                prefix=prefix,
                plan=plan,
                context=context,
                run_dir=args.run_dir,
                resume=args.resume,
            ): plan
            for plan in plans
        }
        for future in as_completed(futures):
            plan = futures[future]
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "runtime_file_complete",
                        "file_bytes": record["file_bytes"],
                        "filename": plan.filename,
                        "sha256": record["sha256"],
                        "stage_id": stage_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "artifact_kind": "greenfield_runtime_checkpoint_stage_pack",
        "file_count": len(records),
        "process_index": args.process_index,
        "runtime_layout_hash": context.layout.layout_hash,
        "stage_id": stage_id,
        "total_file_bytes": sum(record["file_bytes"] for record in records),
    }
    _write_json_once(args.run_dir / f"stage_{stage_id:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _finalize(
    args: argparse.Namespace,
    context: PackContext,
    bucket: Any,
    prefix: str,
) -> None:
    _remote_prerequisites(bucket=bucket, prefix=prefix, context=context)
    if bucket.blob(f"{prefix}/SUCCESS").exists():
        raise RuntimeError("runtime checkpoint is already complete")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for plan in context.plans:
        record = _validate_remote_record(
            bucket=bucket,
            prefix=prefix,
            plan=plan,
            context=context,
        )
        if record is None:
            raise RuntimeError(f"runtime file is incomplete: {plan.filename}")
        records.append(record)
        _write_json_once(
            args.run_dir / "file_evidence" / f"{plan.filename}.json",
            record,
        )
    manifest: dict[str, Any] = {
        "artifact_kind": RUNTIME_PACKED_ARTIFACT_KIND,
        **context.common,
        "control_sha256": context.control["control_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest["manifest_sha256"] = _mapping_hash(
        manifest,
        hash_field="manifest_sha256",
    )
    manifest_blob = bucket.blob(f"{prefix}/runtime_manifest.json")
    _upload_json_once(manifest_blob, manifest)
    success = f"{manifest['manifest_sha256']}  runtime_manifest.json\n"
    bucket.blob(f"{prefix}/SUCCESS").upload_from_string(
        success,
        content_type="text/plain",
        if_generation_match=0,
        checksum="crc32c",
        timeout=300,
    )
    _write_json_once(args.run_dir / "control.json", context.control)
    _write_json_once(args.run_dir / "runtime_layout.json", context.layout_document)
    _write_json_once(args.run_dir / "runtime_manifest.json", manifest)
    _write_text_once(args.run_dir / "SUCCESS", success)
    summary = {key: value for key, value in manifest.items() if key != "files"}
    _write_json_once(args.run_dir / "summary.json", summary)
    _write_text_once(
        args.run_dir / "evidence.sha256",
        "".join(
            f"{_sha256_file(path)}  {path.name}\n"
            for path in (
                args.run_dir / "control.json",
                args.run_dir / "runtime_layout.json",
                args.run_dir / "runtime_manifest.json",
                args.run_dir / "summary.json",
            )
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("prepare", "pack-stage", "finalize"),
    )
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--process-index", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "pack-stage" and args.process_index is None:
        raise RuntimeError("pack-stage requires --process-index")
    if args.mode != "pack-stage" and args.process_index is not None:
        raise RuntimeError("--process-index is valid only for pack-stage")
    code_hash = _verify_repo(args.expected_code_hash)
    bucket_name, prefix = _parse_gs_uri(args.destination)
    context = _build_context(args, code_hash)
    from google.cloud import storage

    bucket = storage.Client().bucket(bucket_name)
    if args.mode == "prepare":
        _prepare(args, context, bucket, prefix)
    elif args.mode == "pack-stage":
        _pack_stage(args, context, bucket, prefix)
    else:
        _finalize(args, context, bucket, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
