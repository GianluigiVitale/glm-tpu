from __future__ import annotations

import json
import struct
from dataclasses import replace
from hashlib import sha256
from io import BytesIO

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint import (
    build_feature_runtime_destination_file_plans,
    build_feature_runtime_layout_document,
    build_runtime_destination_file_plans,
    stream_feature_runtime_stage,
    stream_runtime_weight_file,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.model import (
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    build_decoder_feature_runtime_weight_layout,
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
)
from tests.greenfield.checkpoint.test_runtime_pack import (
    _small_feature_source_plan,
    _source_file,
    _source_plans,
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
