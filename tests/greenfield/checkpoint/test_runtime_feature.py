from __future__ import annotations

import json
import struct
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint import (
    FEATURE_RUNTIME_FORMAT_VERSION,
    FEATURE_RUNTIME_PACK_CONTROL_KIND,
    FEATURE_RUNTIME_PACKED_ARTIFACT_KIND,
    FeatureRuntimeCheckpointLoadExpectation,
    build_feature_runtime_destination_file_plans,
    build_feature_runtime_layout_document,
    build_runtime_destination_file_plans,
    stream_feature_runtime_stage,
    stream_runtime_weight_file,
    verify_feature_runtime_packed_checkpoint,
    verify_runtime_packed_checkpoint,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.checkpoint.runtime_feature import _transform_qkv_a
from glm_tpu.greenfield.model import (
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    build_decoder_feature_fused_qkv_runtime_weight_layout,
    build_decoder_feature_runtime_weight_layout,
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
)
from tests.greenfield.checkpoint.test_runtime_loader import (
    _build_artifact,
    _mapping_hash,
    _write_json,
)
from tests.greenfield.checkpoint.test_runtime_pack import (
    _small_feature_source_plan,
    _source_file,
    _source_plans,
)

FEATURE_PACK_CODE_HASH = "5" * 40
FEATURE_DESTINATION = (
    "gs://driftbench-dsv4-uc/checkpoints/greenfield/test/runtime-feature"
)


def test_fused_qkv_a_transform_matches_shard_major_n82_semantics() -> None:
    q = np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)
    kv = (np.arange(32 * 128, dtype=np.uint8) + np.uint8(17)).reshape(
        32, 128
    )
    observed = np.frombuffer(
        _transform_qkv_a(
            bytearray(q.tobytes()),
            bytearray(kv.tobytes()),
            q_shape=q.shape,
            kv_shape=kv.shape,
            destination_shape=(32, 128, 3),
            dtype="F8_E4M3",
            transform="fuse_qkv_a_output_shards",
        ),
        dtype=np.uint8,
    ).reshape(32, 128, 3)
    expected = np.concatenate(
        (
            q.reshape(32, 2, 128).transpose(0, 2, 1),
            kv.reshape(32, 1, 128).transpose(0, 2, 1),
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(observed, expected)

    q_scale = np.asarray([[2.0]], dtype="<f4")
    kv_scale = np.asarray([[5.0]], dtype="<f4")
    observed_scale = np.frombuffer(
        _transform_qkv_a(
            bytearray(q_scale.tobytes()),
            bytearray(kv_scale.tobytes()),
            q_shape=q_scale.shape,
            kv_shape=kv_scale.shape,
            destination_shape=(32, 1, 5),
            dtype="F32",
            transform="fuse_qkv_a_expanded_scales",
        ),
        dtype="<f4",
    ).reshape(32, 1, 5)
    np.testing.assert_array_equal(
        observed_scale,
        np.broadcast_to(
            np.asarray([2.0, 2.0, 2.0, 2.0, 5.0], dtype=np.float32),
            (32, 1, 5),
        ),
    )


def _runtime_stage() -> tuple[object, tuple[object, ...], dict[int, bytes], dict]:
    source_plan = _small_feature_source_plan()
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        build_pipeline_schedule(source_plan),
    )
    packed_plans = _source_plans(source_layout)
    runtime_plans = build_runtime_destination_file_plans(
        source_layout,
        packed_plans,
        source_packed_manifest_sha256="a" * 64,
    )
    packed_by_owner = {(plan.stage_id, plan.device_slot): plan for plan in packed_plans}
    runtime_bytes = {}
    runtime_evidence = {}
    stage_plans = tuple(plan for plan in runtime_plans if plan.stage_id == 3)
    for runtime_plan in stage_plans:
        packed = packed_by_owner[(runtime_plan.stage_id, runtime_plan.device_slot)]
        packed_bytes = _source_file(packed)
        output = BytesIO()
        evidence = stream_runtime_weight_file(
            source_plan=packed,
            runtime_plan=runtime_plan,
            source=BytesIO(packed_bytes),
            output=output,
            verified_source_file_sha256=sha256(packed_bytes).hexdigest(),
            chunk_bytes=7,
        )
        runtime_bytes[runtime_plan.device_slot] = output.getvalue()
        runtime_evidence[runtime_plan.filename] = evidence
    return source_plan, stage_plans, runtime_bytes, runtime_evidence


def _tensor_bytes(file_bytes: bytes, plan: object, name: str) -> bytes:
    tensor = next(tensor for tensor in plan.tensors if tensor.spec.name == name)
    start = len(plan.header) + tensor.data_offset_start
    return file_bytes[start : len(plan.header) + tensor.data_offset_end]


def test_feature_runtime_stage_streams_exact_final_owner_transforms() -> None:
    source_plan, source_stage, source_bytes, source_evidence = _runtime_stage()
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        build_pipeline_schedule(source_plan),
    )
    target_layout = build_decoder_feature_runtime_weight_layout(
        target_plan,
        target_schedule,
        source_layout,
    )
    all_source_plans = build_runtime_destination_file_plans(
        source_layout,
        _source_plans(source_layout),
        source_packed_manifest_sha256="a" * 64,
    )
    all_destination_plans = build_feature_runtime_destination_file_plans(
        target_layout,
        all_source_plans,
        source_runtime_manifest_sha256="b" * 64,
    )
    destination_stage = tuple(
        plan for plan in all_destination_plans if plan.stage_id == 3
    )
    outputs = {slot: BytesIO() for slot in range(4)}
    source_file_hashes = {
        plan.filename: source_evidence[plan.filename].sha256 for plan in source_stage
    }
    source_tensor_hashes = {
        (plan.filename, tensor.name): tensor.sha256
        for plan in source_stage
        for tensor in source_evidence[plan.filename].tensors
    }
    evidence = stream_feature_runtime_stage(
        source_plans=source_stage,
        destination_plans=destination_stage,
        sources={slot: BytesIO(value) for slot, value in source_bytes.items()},
        outputs=outputs,
        verified_source_file_sha256=source_file_hashes,
        source_tensor_sha256=source_tensor_hashes,
        chunk_bytes=7,
    )
    assert len(evidence) == 4
    assert all(
        len(outputs[slot].getvalue()) == destination_stage[slot].file_bytes
        for slot in range(4)
    )
    assert all(
        sha256(outputs[slot].getvalue()).hexdigest() == evidence[slot].sha256
        for slot in range(4)
    )

    destination_slot = 2
    destination = destination_stage[destination_slot]
    output = outputs[destination_slot].getvalue()
    gate_name = "sparse.slot_00.experts.gate_proj.weight_bits"
    down_name = "sparse.slot_00.experts.down_proj.weight_bits"
    gate_scale_name = "sparse.slot_00.experts.gate_proj.scale_inv"
    down_scale_name = "sparse.slot_00.experts.down_proj.scale_inv"
    router_name = "sparse.slot_00.router_weight"

    gate_sources = [
        np.frombuffer(
            _tensor_bytes(source_bytes[slot], source_stage[slot], gate_name),
            dtype=np.uint8,
        ).reshape(2, 8, 8)
        for slot in range(4)
    ]
    expected_gate = np.concatenate(
        [value[:, 4:6, :].transpose(0, 2, 1) for value in gate_sources],
        axis=0,
    )
    observed_gate = np.frombuffer(
        _tensor_bytes(output, destination, gate_name), dtype=np.uint8
    ).reshape(8, 8, 2)
    np.testing.assert_array_equal(observed_gate, expected_gate)

    down_sources = [
        np.frombuffer(
            _tensor_bytes(source_bytes[slot], source_stage[slot], down_name),
            dtype=np.uint8,
        ).reshape(2, 8, 8)
        for slot in range(4)
    ]
    expected_down = np.concatenate(
        [value[:, :, 4:6].transpose(0, 2, 1) for value in down_sources],
        axis=0,
    )
    observed_down = np.frombuffer(
        _tensor_bytes(output, destination, down_name), dtype=np.uint8
    ).reshape(8, 2, 8)
    np.testing.assert_array_equal(observed_down, expected_down)

    gate_scale_sources = [
        np.frombuffer(
            _tensor_bytes(source_bytes[slot], source_stage[slot], gate_scale_name),
            dtype="<f4",
        ).reshape(2, 4, 4)
        for slot in range(4)
    ]
    expected_gate_scale = np.concatenate(
        [value[:, 2:3, :] for value in gate_scale_sources], axis=0
    )
    observed_gate_scale = np.frombuffer(
        _tensor_bytes(output, destination, gate_scale_name), dtype="<f4"
    ).reshape(8, 1, 4)
    np.testing.assert_array_equal(observed_gate_scale, expected_gate_scale)

    down_scale_sources = [
        np.frombuffer(
            _tensor_bytes(source_bytes[slot], source_stage[slot], down_scale_name),
            dtype="<f4",
        ).reshape(2, 4, 4)
        for slot in range(4)
    ]
    expected_down_scale = np.concatenate(
        [value[:, :, 2:3] for value in down_scale_sources], axis=0
    )
    observed_down_scale = np.frombuffer(
        _tensor_bytes(output, destination, down_scale_name), dtype="<f4"
    ).reshape(8, 4, 1)
    np.testing.assert_array_equal(observed_down_scale, expected_down_scale)

    assert _tensor_bytes(output, destination, router_name) == _tensor_bytes(
        source_bytes[destination_slot],
        source_stage[destination_slot],
        router_name,
    )
    gate_record = next(
        tensor
        for tensor in evidence[destination_slot].tensors
        if tensor.name == gate_name
    )
    assert gate_record.transform == "concat_experts_slice_output_transpose"
    assert len(gate_record.sources) == 4
    assert all(source.feature_axis == 1 for source in gate_record.sources)
    assert all(source.feature_start == 4 for source in gate_record.sources)
    assert all(source.feature_stop == 6 for source in gate_record.sources)
    assert all(source.transpose_axes == (0, 2, 1) for source in gate_record.sources)


def test_feature_runtime_stage_streams_fused_qkv_a_from_base_files() -> None:
    source_plan = _small_feature_source_plan()
    source_plan = replace(
        source_plan,
        geometry=replace(
            source_plan.geometry,
            hidden_size=128,
            q_lora_rank=128,
            kv_lora_rank=30,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            moe_intermediate_size=512,
            fp8_block_shape=(128, 128),
        ),
    )
    source_schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    packed_plans = _source_plans(source_layout)
    runtime_plans = build_runtime_destination_file_plans(
        source_layout,
        packed_plans,
        source_packed_manifest_sha256="a" * 64,
    )
    packed_by_owner = {
        (plan.stage_id, plan.device_slot): plan for plan in packed_plans
    }
    source_stage = tuple(
        plan for plan in runtime_plans if plan.stage_id == 3
    )
    source_bytes = {}
    source_evidence = {}
    for runtime_plan in source_stage:
        packed = packed_by_owner[(3, runtime_plan.device_slot)]
        packed_bytes = _source_file(packed)
        output = BytesIO()
        evidence = stream_runtime_weight_file(
            source_plan=packed,
            runtime_plan=runtime_plan,
            source=BytesIO(packed_bytes),
            output=output,
            verified_source_file_sha256=sha256(packed_bytes).hexdigest(),
            chunk_bytes=4096,
        )
        source_bytes[runtime_plan.device_slot] = output.getvalue()
        source_evidence[runtime_plan.filename] = evidence

    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    target_layout = build_decoder_feature_fused_qkv_runtime_weight_layout(
        target_plan,
        target_schedule,
        source_layout,
    )
    destination_stage = tuple(
        plan
        for plan in build_feature_runtime_destination_file_plans(
            target_layout,
            runtime_plans,
            source_runtime_manifest_sha256="b" * 64,
        )
        if plan.stage_id == 3
    )
    outputs = {slot: BytesIO() for slot in range(4)}
    evidence = stream_feature_runtime_stage(
        source_plans=source_stage,
        destination_plans=destination_stage,
        sources={
            slot: BytesIO(value) for slot, value in source_bytes.items()
        },
        outputs=outputs,
        verified_source_file_sha256={
            plan.filename: source_evidence[plan.filename].sha256
            for plan in source_stage
        },
        source_tensor_sha256={
            (plan.filename, tensor.name): tensor.sha256
            for plan in source_stage
            for tensor in source_evidence[plan.filename].tensors
        },
        chunk_bytes=4096,
    )
    destination = destination_stage[0]
    output = outputs[0].getvalue()
    q_name = "attention.slot_00.q_a.weight_bits"
    kv_name = "attention.slot_00.kv_a.weight_bits"
    packed_name = "attention.slot_00.qkv_a.weight_bits"
    q = np.frombuffer(
        _tensor_bytes(source_bytes[0], source_stage[0], q_name),
        dtype=np.uint8,
    ).reshape(128, 128)
    kv = np.frombuffer(
        _tensor_bytes(source_bytes[0], source_stage[0], kv_name),
        dtype=np.uint8,
    ).reshape(32, 128)
    expected = np.concatenate(
        (
            q.reshape(32, 4, 128).transpose(0, 2, 1),
            kv.reshape(32, 1, 128).transpose(0, 2, 1),
        ),
        axis=-1,
    )
    observed = np.frombuffer(
        _tensor_bytes(output, destination, packed_name),
        dtype=np.uint8,
    ).reshape(32, 128, 5)
    np.testing.assert_array_equal(observed, expected)
    record = next(item for item in evidence[0].tensors if item.name == packed_name)
    assert record.transform == "fuse_qkv_a_output_shards"
    assert tuple(source.source_tensor_name for source in record.sources) == (
        q_name,
        kv_name,
    )


def test_feature_runtime_layout_document_and_header_bind_source() -> None:
    source_plan, _, _, _ = _runtime_stage()
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        build_pipeline_schedule(source_plan),
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_layout = build_decoder_feature_runtime_weight_layout(
        target_plan,
        build_pipeline_schedule(target_plan),
        source_layout,
    )
    document = build_feature_runtime_layout_document(target_layout)
    unhashed = dict(document)
    observed = unhashed.pop("manifest_sha256")
    assert (
        observed
        == sha256(
            json.dumps(
                unhashed,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
    )
    source_plans = build_runtime_destination_file_plans(
        source_layout,
        _source_plans(source_layout),
        source_packed_manifest_sha256="a" * 64,
    )
    destinations = build_feature_runtime_destination_file_plans(
        target_layout,
        source_plans,
        source_runtime_manifest_sha256="b" * 64,
    )
    header_size = struct.unpack("<Q", destinations[0].header[:8])[0]
    header = json.loads(destinations[0].header[8 : 8 + header_size])
    assert (
        header["__metadata__"]["greenfield_source_runtime_manifest_sha256"] == "b" * 64
    )
    assert header["sparse.slot_00.experts.gate_proj.weight_bits"]["shape"] == [
        8,
        8,
        2,
    ]


def test_feature_runtime_stage_refuses_truncated_authenticated_source() -> None:
    source_plan, source_stage, source_bytes, source_evidence = _runtime_stage()
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        build_pipeline_schedule(source_plan),
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_layout = build_decoder_feature_runtime_weight_layout(
        target_plan,
        build_pipeline_schedule(target_plan),
        source_layout,
    )
    source_plans = build_runtime_destination_file_plans(
        source_layout,
        _source_plans(source_layout),
        source_packed_manifest_sha256="a" * 64,
    )
    destinations = tuple(
        plan
        for plan in build_feature_runtime_destination_file_plans(
            target_layout,
            source_plans,
            source_runtime_manifest_sha256="b" * 64,
        )
        if plan.stage_id == 3
    )
    source_file_hashes = {
        plan.filename: source_evidence[plan.filename].sha256 for plan in source_stage
    }
    source_tensor_hashes = {
        (plan.filename, tensor.name): tensor.sha256
        for plan in source_stage
        for tensor in source_evidence[plan.filename].tensors
    }
    truncated = dict(source_bytes)
    consumed = next(
        tensor
        for tensor in source_stage[0].tensors
        if tensor.spec.name == "sparse.slot_00.experts.down_proj.weight_bits"
    )
    truncated[0] = truncated[0][
        : len(source_stage[0].header) + consumed.data_offset_end - 1
    ]
    with pytest.raises(CheckpointValidationError, match="truncated"):
        stream_feature_runtime_stage(
            source_plans=source_stage,
            destination_plans=destinations,
            sources={slot: BytesIO(value) for slot, value in truncated.items()},
            outputs={slot: BytesIO() for slot in range(4)},
            verified_source_file_sha256=source_file_hashes,
            source_tensor_sha256=source_tensor_hashes,
            chunk_bytes=7,
        )


def _build_feature_artifact(
    tmp_path: Path,
    *,
    fused_qkv_a: bool = False,
) -> tuple[object, object, object]:
    source_plan = _small_feature_source_plan()
    if fused_qkv_a:
        source_plan = replace(
            source_plan,
            geometry=replace(
                source_plan.geometry,
                hidden_size=128,
                q_lora_rank=128,
                kv_lora_rank=30,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                v_head_dim=2,
                moe_intermediate_size=512,
                fp8_block_shape=(128, 128),
            ),
        )
    source_root = tmp_path / "source-runtime"
    source_layout, source_packed, source_expectation, _ = _build_artifact(
        source_root,
        source_plan,
    )
    source_checkpoint = verify_runtime_packed_checkpoint(
        source_root,
        source_expectation,
        source_layout,
        source_packed,
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    if fused_qkv_a:
        target_layout = build_decoder_feature_fused_qkv_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )
    else:
        target_layout = build_decoder_feature_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )
    plans = build_feature_runtime_destination_file_plans(
        target_layout,
        source_checkpoint.plans,
        source_runtime_manifest_sha256=source_expectation.runtime_manifest_sha256,
    )
    target_root = tmp_path / "feature-runtime"
    target_root.mkdir()
    layout_document = build_feature_runtime_layout_document(target_layout)
    layout_path = target_root / "runtime_layout.json"
    _write_json(layout_path, layout_document)
    common: dict[str, object] = {
        "destination": FEATURE_DESTINATION,
        "file_count": len(plans),
        "format_version": FEATURE_RUNTIME_FORMAT_VERSION,
        "model_id": "zai-org/GLM-5.2-FP8",
        "pack_code_hash": FEATURE_PACK_CODE_HASH,
        "padding_bytes": sum(device.padding_bytes for device in target_layout.devices),
        "plan_hash": target_layout.plan_hash,
        "plan_id": "PP8_LP4",
        "routed_expert_layout": target_layout.routed_expert_layout,
        "runtime_file_bytes": sum(plan.file_bytes for plan in plans),
        "runtime_layout_hash": target_layout.layout_hash,
        "runtime_layout_manifest_sha256": layout_document["manifest_sha256"],
        "runtime_payload_bytes": sum(plan.payload_bytes for plan in plans),
        "schedule_hash": target_layout.schedule_hash,
        "source_checkpoint_destination": source_expectation.destination,
        "source_payload_bytes": sum(
            device.source_bytes for device in target_layout.devices
        ),
        "source_runtime_layout_hash": source_expectation.runtime_layout_hash,
        "source_runtime_layout_manifest_sha256": (
            source_expectation.runtime_layout_manifest_sha256
        ),
        "source_runtime_manifest_sha256": (source_expectation.runtime_manifest_sha256),
        "source_tensor_count": target_layout.source_leaf_count,
        "tensor_count": len(target_layout.specs) * len(target_layout.devices),
    }
    if (
        target_layout.attention_projection_layout
        != SEPARATE_QKV_A_RUNTIME_LAYOUT
    ):
        common["attention_projection_layout"] = (
            target_layout.attention_projection_layout
        )
    control: dict[str, object] = {
        "artifact_kind": FEATURE_RUNTIME_PACK_CONTROL_KIND,
        **common,
        "runtime_layout_file_sha256": sha256(layout_path.read_bytes()).hexdigest(),
    }
    control["control_sha256"] = _mapping_hash(control, "control_sha256")
    _write_json(target_root / "control.json", control)

    source_file_hashes = {
        filename: record["sha256"]
        for filename, record in source_checkpoint.evidence_by_filename.items()
    }
    source_tensor_hashes = {
        (filename, tensor["name"]): tensor["sha256"]
        for filename, record in source_checkpoint.evidence_by_filename.items()
        for tensor in record["tensors"]
    }
    records = []
    for stage_id in range(8):
        source_stage = tuple(
            plan for plan in source_checkpoint.plans if plan.stage_id == stage_id
        )
        destination_stage = tuple(plan for plan in plans if plan.stage_id == stage_id)
        source_streams = {
            plan.device_slot: (source_checkpoint.root / plan.filename).open("rb")
            for plan in source_stage
        }
        output_streams = {}
        try:
            for plan in destination_stage:
                path = target_root / plan.filename
                path.parent.mkdir(parents=True, exist_ok=True)
                output_streams[plan.device_slot] = path.open("wb")
            stage_evidence = stream_feature_runtime_stage(
                source_plans=source_stage,
                destination_plans=destination_stage,
                sources=source_streams,
                outputs=output_streams,
                verified_source_file_sha256=source_file_hashes,
                source_tensor_sha256=source_tensor_hashes,
                chunk_bytes=31,
            )
        finally:
            for stream in (*source_streams.values(), *output_streams.values()):
                stream.close()
        for plan, evidence in zip(destination_stage, stage_evidence, strict=True):
            record = evidence.to_dict()
            record["destination_filename"] = record.pop("filename")
            record.update(
                {
                    "crc32c": "AAAAAA==",
                    "device_id": plan.device_id,
                    "device_slot": plan.device_slot,
                    "generation": plan.device_id + 1,
                    "header_bytes": len(plan.header),
                    "header_sha256": sha256(plan.header).hexdigest(),
                    "runtime_layout_hash": target_layout.layout_hash,
                    "source_runtime_manifest_sha256": (
                        source_expectation.runtime_manifest_sha256
                    ),
                    "stage_id": plan.stage_id,
                    "tensor_count": len(plan.tensors),
                }
            )
            records.append(record)
            _write_json(
                target_root / "evidence" / f"{plan.filename}.json",
                record,
            )
    manifest: dict[str, object] = {
        "artifact_kind": FEATURE_RUNTIME_PACKED_ARTIFACT_KIND,
        **common,
        "control_sha256": control["control_sha256"],
        "files": records,
    }
    manifest["manifest_sha256"] = _mapping_hash(manifest, "manifest_sha256")
    _write_json(target_root / "runtime_manifest.json", manifest)
    (target_root / "SUCCESS").write_text(
        f"{manifest['manifest_sha256']}  runtime_manifest.json\n"
    )
    expectation = FeatureRuntimeCheckpointLoadExpectation(
        runtime_manifest_sha256=manifest["manifest_sha256"],
        runtime_layout_manifest_sha256=layout_document["manifest_sha256"],
        runtime_layout_hash=target_layout.layout_hash,
        source_runtime_manifest_sha256=(source_expectation.runtime_manifest_sha256),
        source_runtime_layout_manifest_sha256=(
            source_expectation.runtime_layout_manifest_sha256
        ),
        source_runtime_layout_hash=source_expectation.runtime_layout_hash,
        plan_hash=target_layout.plan_hash,
        schedule_hash=target_layout.schedule_hash,
        pack_code_hash=FEATURE_PACK_CODE_HASH,
        destination=FEATURE_DESTINATION,
        source_destination=source_expectation.destination,
    )
    return target_layout, source_checkpoint, expectation


def test_feature_runtime_artifact_verifier_pins_every_transform(
    tmp_path: Path,
) -> None:
    target_layout, source_checkpoint, expectation = _build_feature_artifact(tmp_path)
    target_root = tmp_path / "feature-runtime"
    verified = verify_feature_runtime_packed_checkpoint(
        target_root,
        expectation,
        target_layout,
        source_checkpoint,
    )
    assert len(verified.plans) == 32
    assert len(verified.evidence_by_filename) == 32

    plan = verified.plans[12]
    sidecar = target_root / "evidence" / f"{plan.filename}.json"
    value = json.loads(sidecar.read_text())
    gate = next(
        tensor
        for tensor in value["tensors"]
        if tensor["name"] == "sparse.slot_00.experts.gate_proj.weight_bits"
    )
    gate["sources"][0]["feature_slice"]["start"] += 1
    _write_json(sidecar, value)
    with pytest.raises(
        CheckpointValidationError,
        match="tensor evidence drifted|sidecar disagrees",
    ):
        verify_feature_runtime_packed_checkpoint(
            target_root,
            expectation,
            target_layout,
            source_checkpoint,
        )


def test_fused_qkv_feature_runtime_artifact_roundtrips(
    tmp_path: Path,
) -> None:
    target_layout, source_checkpoint, expectation = _build_feature_artifact(
        tmp_path,
        fused_qkv_a=True,
    )
    target_root = tmp_path / "feature-runtime"
    verified = verify_feature_runtime_packed_checkpoint(
        target_root,
        expectation,
        target_layout,
        source_checkpoint,
    )
    assert len(verified.plans) == 32
    manifest = verified.runtime_manifest
    assert manifest["attention_projection_layout"] == (
        target_layout.attention_projection_layout
    )
    first = verified.evidence_by_filename[verified.plans[0].filename]
    fused = next(
        tensor
        for tensor in first["tensors"]
        if tensor["name"] == "attention.slot_00.qkv_a.scale_inv"
    )
    assert fused["transform"] == "fuse_qkv_a_expanded_scales"
    assert len(fused["sources"]) == 2
