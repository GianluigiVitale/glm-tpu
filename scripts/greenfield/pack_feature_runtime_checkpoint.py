#!/usr/bin/env python3
"""Build a complete PP8/PP16 expert-feature runtime artifact in GCS."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FEATURE_RUNTIME_FORMAT_VERSION,
    FEATURE_RUNTIME_PACK_CONTROL_KIND,
    FEATURE_RUNTIME_PACKED_ARTIFACT_KIND,
    FeatureRuntimeCheckpointLoadExpectation,
    FeatureRuntimeDestinationFilePlan,
    FullCheckpointLoadExpectation,
    RuntimeCheckpointLoadExpectation,
    VerifiedPackedCheckpoint,
    VerifiedRuntimeCheckpoint,
    build_feature_runtime_destination_file_plans,
    build_feature_runtime_layout_document,
    resolve_runtime_pack_stage,
    stream_feature_runtime_stage,
    verify_full_packed_checkpoint,
    verify_runtime_packed_checkpoint,
    verify_source_file_sha256,
)
from glm_tpu.greenfield.checkpoint.runtime_feature import (  # noqa: E402
    _source_evidence,
)
from glm_tpu.greenfield.model import (  # noqa: E402
    LEGACY_DENSE_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
    build_decoder_feature_fused_qkv_dense_runtime_weight_layout,
    build_decoder_feature_fused_qkv_runtime_weight_layout,
    build_decoder_feature_runtime_weight_layout,
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
    feature_expert_runtime_layout,
)
from glm_tpu.greenfield.partitioning import inspect_layout_manifest  # noqa: E402
from glm_tpu.greenfield.types import ExecutionPlan  # noqa: E402
from scripts.greenfield.pack_runtime_checkpoint import (  # noqa: E402
    CHUNK_BYTES,
    _download_json,
    _mapping_hash,
    _parse_gs_uri,
    _read_json,
    _sha256_file,
    _upload_json_once,
    _verify_repo,
    _write_json_once,
    _write_text_once,
)


@dataclass(frozen=True, slots=True)
class PackContext:
    source_full_checkpoint: VerifiedPackedCheckpoint
    source_runtime_checkpoint: VerifiedRuntimeCheckpoint
    source_layout: DecoderRuntimeWeightLayout
    target_plan: ExecutionPlan
    layout: DecoderRuntimeWeightLayout
    layout_document: Mapping[str, Any]
    layout_bytes: bytes
    plans: tuple[FeatureRuntimeDestinationFilePlan, ...]
    source_file_hashes: Mapping[str, str]
    source_tensor_hashes: Mapping[tuple[str, str], str]
    common: Mapping[str, Any]
    control: Mapping[str, Any]


def _source_full_expectation(
    root: Path,
    expected_manifest_sha256: str,
) -> FullCheckpointLoadExpectation:
    packed = _read_json(root / "packed_manifest.json")
    if _mapping_hash(packed, hash_field="manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError("protected full source packed manifest hash drifted")
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


def _source_runtime_expectation(
    root: Path,
    expected_manifest_sha256: str,
) -> RuntimeCheckpointLoadExpectation:
    manifest = _read_json(root / "runtime_manifest.json")
    if (
        _mapping_hash(manifest, hash_field="manifest_sha256")
        != expected_manifest_sha256
    ):
        raise RuntimeError("protected source runtime manifest hash drifted")
    return RuntimeCheckpointLoadExpectation(
        runtime_manifest_sha256=expected_manifest_sha256,
        runtime_layout_manifest_sha256=manifest["runtime_layout_manifest_sha256"],
        runtime_layout_hash=manifest["runtime_layout_hash"],
        source_packed_manifest_sha256=manifest["source_packed_manifest_sha256"],
        source_layout_manifest_sha256=manifest["source_layout_manifest_sha256"],
        plan_hash=manifest["plan_hash"],
        schedule_hash=manifest["schedule_hash"],
        pack_code_hash=manifest["pack_code_hash"],
        destination=manifest["destination"],
        source_destination=manifest["source_checkpoint_destination"],
        plan_id=manifest["plan_id"],
        model_id=manifest["model_id"],
    )


def _source_hashes(
    checkpoint: VerifiedRuntimeCheckpoint,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    files = {}
    tensors = {}
    for filename, record in checkpoint.evidence_by_filename.items():
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"source runtime file digest drifted: {filename}")
        files[filename] = digest
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, list):
            raise RuntimeError(f"source runtime tensor ledger missing: {filename}")
        for tensor in tensor_records:
            if not isinstance(tensor, Mapping):
                raise RuntimeError("source runtime tensor record is not an object")
            name = tensor.get("name")
            tensor_digest = tensor.get("sha256")
            if (
                not isinstance(name, str)
                or not isinstance(tensor_digest, str)
                or len(tensor_digest) != 64
                or (filename, name) in tensors
            ):
                raise RuntimeError("source runtime tensor identity drifted")
            tensors[(filename, name)] = tensor_digest
    return files, tensors


def _build_context(args: argparse.Namespace, code_hash: str) -> PackContext:
    full_expectation = _source_full_expectation(
        args.source_checkpoint_root,
        args.source_packed_manifest_sha256,
    )
    source_full = verify_full_packed_checkpoint(
        args.source_checkpoint_root,
        full_expectation,
    )
    source_plan = ExecutionPlan.from_dict(
        source_full.layout["plan_manifest"]["execution_plan"]
    )
    source_schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    runtime_expectation = _source_runtime_expectation(
        args.source_runtime_root,
        args.source_runtime_manifest_sha256,
    )
    source_runtime = verify_runtime_packed_checkpoint(
        args.source_runtime_root,
        runtime_expectation,
        source_layout,
        source_full,
    )
    target_plan = replace(
        source_plan,
        expert_layout=feature_expert_runtime_layout(
            source_plan.local_parallel_size
        ),
    )
    target_schedule = build_pipeline_schedule(target_plan)
    if getattr(args, "dense_convolution", False):
        if not getattr(args, "fused_qkv_a", False):
            raise RuntimeError(
                "dense convolution derivative requires fused qkv-a"
            )
        layout = build_decoder_feature_fused_qkv_dense_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )
    elif getattr(args, "fused_qkv_a", False):
        layout = build_decoder_feature_fused_qkv_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )
    else:
        layout = build_decoder_feature_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )
    layout_document = build_feature_runtime_layout_document(layout)
    layout_bytes = (
        json.dumps(layout_document, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    plans = build_feature_runtime_destination_file_plans(
        layout,
        source_runtime.plans,
        source_runtime_manifest_sha256=args.source_runtime_manifest_sha256,
    )
    source_file_hashes, source_tensor_hashes = _source_hashes(source_runtime)
    common: dict[str, Any] = {
        "destination": args.destination.rstrip("/"),
        "file_count": len(plans),
        "format_version": FEATURE_RUNTIME_FORMAT_VERSION,
        "model_id": target_plan.geometry.model_id,
        "pack_code_hash": code_hash,
        "padding_bytes": sum(device.padding_bytes for device in layout.devices),
        "plan_hash": target_plan.plan_hash,
        "plan_id": target_plan.name.value,
        "routed_expert_layout": layout.routed_expert_layout,
        "runtime_file_bytes": sum(plan.file_bytes for plan in plans),
        "runtime_layout_hash": layout.layout_hash,
        "runtime_layout_manifest_sha256": layout_document["manifest_sha256"],
        "runtime_payload_bytes": sum(plan.payload_bytes for plan in plans),
        "schedule_hash": target_schedule.schedule_hash,
        "source_checkpoint_destination": runtime_expectation.destination,
        "source_payload_bytes": sum(device.source_bytes for device in layout.devices),
        "source_runtime_layout_hash": runtime_expectation.runtime_layout_hash,
        "source_runtime_layout_manifest_sha256": (
            runtime_expectation.runtime_layout_manifest_sha256
        ),
        "source_runtime_manifest_sha256": args.source_runtime_manifest_sha256,
        "source_tensor_count": layout.source_leaf_count,
        "tensor_count": len(layout.specs) * len(layout.devices),
    }
    if layout.attention_projection_layout != SEPARATE_QKV_A_RUNTIME_LAYOUT:
        common["attention_projection_layout"] = (
            layout.attention_projection_layout
        )
    if layout.dense_projection_layout != LEGACY_DENSE_RUNTIME_LAYOUT:
        common["dense_projection_layout"] = layout.dense_projection_layout
    control: dict[str, Any] = {
        "artifact_kind": FEATURE_RUNTIME_PACK_CONTROL_KIND,
        **common,
        "runtime_layout_file_sha256": sha256(layout_bytes).hexdigest(),
    }
    control["control_sha256"] = _mapping_hash(
        control,
        hash_field="control_sha256",
    )
    return PackContext(
        source_full_checkpoint=source_full,
        source_runtime_checkpoint=source_runtime,
        source_layout=source_layout,
        target_plan=target_plan,
        layout=layout,
        layout_document=layout_document,
        layout_bytes=layout_bytes,
        plans=plans,
        source_file_hashes=source_file_hashes,
        source_tensor_hashes=source_tensor_hashes,
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
        raise RuntimeError("feature runtime pack has not been prepared")
    if _download_json(control_blob) != context.control:
        raise RuntimeError("remote feature-runtime control drifted")
    layout_blob.reload()
    if (
        int(layout_blob.size) != len(context.layout_bytes)
        or (layout_blob.metadata or {}).get("semantic_sha256")
        != context.layout_document["manifest_sha256"]
        or (layout_blob.metadata or {}).get("file_sha256")
        != context.control["runtime_layout_file_sha256"]
    ):
        raise RuntimeError("remote feature-runtime layout identity drifted")


def _prepare(
    args: argparse.Namespace,
    context: PackContext,
    bucket: Any,
    prefix: str,
) -> None:
    if args.resume:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        _remote_prerequisites(bucket=bucket, prefix=prefix, context=context)
        if bucket.blob(f"{prefix}/SUCCESS").exists():
            raise RuntimeError("feature runtime checkpoint is already complete")
    else:
        if args.run_dir.exists():
            raise RuntimeError(f"append-only local run already exists: {args.run_dir}")
        if next(bucket.list_blobs(prefix=f"{prefix}/", max_results=1), None):
            raise RuntimeError("remote feature-runtime prefix already exists")
        args.run_dir.mkdir(parents=True)
    layout_path = args.run_dir / "runtime_layout.json"
    _write_json_once(args.run_dir / "control.json", context.control)
    _write_json_once(layout_path, context.layout_document)
    control_blob = bucket.blob(f"{prefix}/control.json")
    layout_blob = bucket.blob(f"{prefix}/runtime_layout.json")
    if not args.resume:
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
    plan: FeatureRuntimeDestinationFilePlan,
    context: PackContext,
) -> dict[str, Any]:
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
        "source_files": [
            {
                "filename": filename,
                "sha256": context.source_file_hashes[filename],
            }
            for filename in plan.source_filenames
        ],
        "source_payload_bytes": plan.device_layout.source_bytes,
        "source_runtime_manifest_sha256": (plan.source_runtime_manifest_sha256),
        "source_tensor_count": plan.device_layout.source_leaf_count,
        "stage_id": plan.stage_id,
        "tensor_count": len(plan.tensors),
    }


def _expected_tensor_record(
    plan: FeatureRuntimeDestinationFilePlan,
    context: PackContext,
    name: str,
) -> dict[str, Any]:
    destination = next(tensor for tensor in plan.tensors if tensor.spec.name == name)
    binding = next(
        tensor for tensor in plan.device_layout.tensors if tensor.spec.name == name
    )
    return {
        "byte_count": destination.byte_count,
        "name": name,
        "padding": binding.is_padding,
        "sources": [
            _source_evidence(
                source=source,
                transform=binding.transform,
                destination_slot=plan.device_slot,
                source_filenames=plan.source_filenames,
                source_tensor_sha256=context.source_tensor_hashes,
            ).to_dict()
            for source in binding.sources
        ],
        "transform": binding.transform,
    }


def _validate_remote_record(
    *,
    bucket: Any,
    prefix: str,
    plan: FeatureRuntimeDestinationFilePlan,
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
            f"partial feature-runtime destination exists for {plan.filename}"
        )
    payload_blob.reload()
    record = _download_json(evidence_blob)
    for field, expected in _static_file_record(plan, context).items():
        if record.get(field) != expected:
            raise RuntimeError(
                f"remote feature-runtime evidence drift for {plan.filename}: {field}"
            )
    if (
        record.get("generation") != int(payload_blob.generation)
        or record.get("crc32c") != payload_blob.crc32c
        or record.get("file_bytes") != int(payload_blob.size)
        or not isinstance(record.get("sha256"), str)
        or len(record["sha256"]) != 64
    ):
        raise RuntimeError(f"remote feature-runtime object drift: {plan.filename}")
    tensor_records = record.get("tensors")
    if not isinstance(tensor_records, list) or len(tensor_records) != len(plan.tensors):
        raise RuntimeError(
            f"remote feature-runtime tensor ledger drift: {plan.filename}"
        )
    by_name = {
        item.get("name"): item
        for item in tensor_records
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(by_name) != len(tensor_records):
        raise RuntimeError(
            f"remote feature-runtime tensor names drift: {plan.filename}"
        )
    for tensor in plan.tensors:
        record_tensor = by_name.get(tensor.spec.name)
        expected = _expected_tensor_record(plan, context, tensor.spec.name)
        if record_tensor is None or any(
            record_tensor.get(field) != value for field, value in expected.items()
        ):
            raise RuntimeError(
                f"remote feature-runtime tensor drift: "
                f"{plan.filename}:{tensor.spec.name}"
            )
        digest = record_tensor.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("remote feature-runtime tensor digest drifted")
    metadata = payload_blob.metadata or {}
    if (
        metadata.get("sha256") != record["sha256"]
        or metadata.get("runtime_layout_hash") != context.layout.layout_hash
        or metadata.get("source_runtime_manifest_sha256")
        != context.common["source_runtime_manifest_sha256"]
    ):
        raise RuntimeError(f"remote feature-runtime metadata drift: {plan.filename}")
    return record


def _pack_stage(
    args: argparse.Namespace,
    context: PackContext,
    bucket: Any,
    prefix: str,
) -> None:
    _remote_prerequisites(bucket=bucket, prefix=prefix, context=context)
    process_index = args.process_index
    if args.topology_capture is not None:
        capture = _read_json(args.topology_capture)
        process_index = capture.get("jax_process_index")
        if not isinstance(process_index, int) or isinstance(process_index, bool):
            raise RuntimeError("topology capture lacks a valid JAX process index")
    stage_id = resolve_runtime_pack_stage(
        build_pipeline_schedule(context.target_plan),
        process_index=process_index,
        requested_stage_id=args.stage_id,
    )
    plans = tuple(plan for plan in context.plans if plan.stage_id == stage_id)
    source_plans = tuple(
        plan
        for plan in context.source_runtime_checkpoint.plans
        if plan.stage_id == stage_id
    )
    stage_size = context.target_plan.local_parallel_size
    if len(plans) != stage_size or len(source_plans) != stage_size:
        raise RuntimeError(
            "feature-runtime stage does not contain its exact local files"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    existing = tuple(
        _validate_remote_record(
            bucket=bucket,
            prefix=prefix,
            plan=plan,
            context=context,
        )
        for plan in plans
    )
    if any(record is not None for record in existing):
        if not args.resume or not all(record is not None for record in existing):
            raise RuntimeError(
                f"feature-runtime stage {stage_id} is partially present; preserving it"
            )
        records = tuple(record for record in existing if record is not None)
    else:
        for source_plan in source_plans:
            verify_source_file_sha256(
                context.source_runtime_checkpoint.root / source_plan.filename,
                context.source_file_hashes[source_plan.filename],
            )
        source_streams = {
            plan.device_slot: (
                context.source_runtime_checkpoint.root / plan.filename
            ).open("rb", buffering=0)
            for plan in source_plans
        }
        output_streams = {}
        payload_blobs = {}
        closed_slots: set[int] = set()
        try:
            for plan in plans:
                blob = bucket.blob(
                    f"{prefix}/{plan.filename}",
                    chunk_size=CHUNK_BYTES,
                )
                blob.content_type = "application/octet-stream"
                blob.metadata = {
                    "runtime_layout_hash": context.layout.layout_hash,
                    "source_runtime_manifest_sha256": context.common[
                        "source_runtime_manifest_sha256"
                    ],
                }
                payload_blobs[plan.device_slot] = blob
                output_streams[plan.device_slot] = blob.open(
                    "wb",
                    chunk_size=CHUNK_BYTES,
                    ignore_flush=True,
                    if_generation_match=0,
                    checksum="crc32c",
                    timeout=300,
                )
            streamed = stream_feature_runtime_stage(
                source_plans=source_plans,
                destination_plans=plans,
                sources=source_streams,
                outputs=output_streams,
                verified_source_file_sha256=context.source_file_hashes,
                source_tensor_sha256=context.source_tensor_hashes,
                chunk_bytes=CHUNK_BYTES,
            )
            for slot in range(stage_size):
                output_streams[slot].close()
                closed_slots.add(slot)
        except BaseException:
            for slot, output in output_streams.items():
                if slot not in closed_slots:
                    try:
                        output.terminate()
                    except BaseException:
                        pass
            raise
        finally:
            for source in source_streams.values():
                source.close()
        evidence_by_slot = {
            plan.device_slot: evidence
            for plan, evidence in zip(plans, streamed, strict=True)
        }
        record_values = []
        for plan in plans:
            evidence = evidence_by_slot[plan.device_slot]
            blob = payload_blobs[plan.device_slot]
            blob.reload()
            if int(blob.size) != evidence.file_bytes or not blob.crc32c:
                raise RuntimeError(
                    f"feature-runtime upload did not seal for {plan.filename}"
                )
            metadata = dict(blob.metadata or {})
            metadata.update(
                {"file_bytes": str(evidence.file_bytes), "sha256": evidence.sha256}
            )
            blob.metadata = metadata
            blob.patch(
                if_generation_match=int(blob.generation),
                timeout=300,
            )
            blob.reload()
            record = {
                **evidence.to_dict(),
                **_static_file_record(plan, context),
                "crc32c": blob.crc32c,
                "generation": int(blob.generation),
                "sha256": evidence.sha256,
                "tensors": [item.to_dict() for item in evidence.tensors],
            }
            record.pop("filename", None)
            _write_json_once(
                args.run_dir / "file_evidence" / f"{plan.filename}.json",
                record,
            )
            _upload_json_once(
                bucket.blob(f"{prefix}/evidence/{plan.filename}.json"),
                record,
            )
            record_values.append(record)
        records = tuple(record_values)
    for plan, record in zip(plans, records, strict=True):
        _write_json_once(
            args.run_dir / "file_evidence" / f"{plan.filename}.json",
            record,
        )
        print(
            json.dumps(
                {
                    "event": "feature_runtime_file_complete",
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
        "artifact_kind": "greenfield_feature_runtime_checkpoint_stage_pack",
        "file_count": len(records),
        "process_index": process_index,
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
        raise RuntimeError("feature runtime checkpoint is already complete")
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
            raise RuntimeError(f"feature runtime file incomplete: {plan.filename}")
        records.append(record)
        _write_json_once(
            args.run_dir / "file_evidence" / f"{plan.filename}.json",
            record,
        )
    manifest: dict[str, Any] = {
        "artifact_kind": FEATURE_RUNTIME_PACKED_ARTIFACT_KIND,
        **context.common,
        "control_sha256": context.control["control_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    manifest["manifest_sha256"] = _mapping_hash(
        manifest,
        hash_field="manifest_sha256",
    )
    _upload_json_once(
        bucket.blob(f"{prefix}/runtime_manifest.json"),
        manifest,
    )
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


def build_load_expectation(
    manifest: Mapping[str, Any],
) -> FeatureRuntimeCheckpointLoadExpectation:
    """Construct the pinned verifier expectation from a hash-checked manifest."""

    return FeatureRuntimeCheckpointLoadExpectation(
        runtime_manifest_sha256=manifest["manifest_sha256"],
        runtime_layout_manifest_sha256=manifest["runtime_layout_manifest_sha256"],
        runtime_layout_hash=manifest["runtime_layout_hash"],
        source_runtime_manifest_sha256=manifest["source_runtime_manifest_sha256"],
        source_runtime_layout_manifest_sha256=manifest[
            "source_runtime_layout_manifest_sha256"
        ],
        source_runtime_layout_hash=manifest["source_runtime_layout_hash"],
        plan_hash=manifest["plan_hash"],
        schedule_hash=manifest["schedule_hash"],
        pack_code_hash=manifest["pack_code_hash"],
        destination=manifest["destination"],
        source_destination=manifest["source_checkpoint_destination"],
        plan_id=manifest["plan_id"],
        model_id=manifest["model_id"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "pack-stage", "finalize"))
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--source-runtime-manifest-sha256", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--process-index", type=int)
    parser.add_argument("--topology-capture", type=Path)
    parser.add_argument("--stage-id", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fused-qkv-a", action="store_true")
    parser.add_argument("--dense-convolution", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selectors = int(args.process_index is not None) + int(
        args.topology_capture is not None
    )
    if args.mode == "pack-stage" and selectors != 1:
        raise RuntimeError(
            "pack-stage requires exactly one of --process-index or --topology-capture"
        )
    if args.mode != "pack-stage" and selectors:
        raise RuntimeError("process/topology selectors are valid only for pack-stage")
    if args.mode != "pack-stage" and args.stage_id is not None:
        raise RuntimeError("stage selector is valid only for pack-stage")
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
