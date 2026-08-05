from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct

import pytest

from glm_tpu.greenfield.checkpoint import (
    DestinationFilePlan,
    DestinationTensorPlan,
    build_runtime_destination_file_plans,
    stream_runtime_weight_file,
    verify_source_file_sha256,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.model import (
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.types import ExecutionPlan, PlanName, StageAssignment
from tests.greenfield.model.test_schedule import _geometry, _topology


def _small_plan() -> ExecutionPlan:
    geometry = replace(
        _geometry(),
        num_layers=8,
        first_dense_layers=3,
        hidden_size=8,
        dense_intermediate_size=16,
        num_routed_experts=8,
        num_shared_experts=1,
        routed_top_k=2,
        moe_intermediate_size=4,
        dsa_top_k=4,
        dsa_indexer_heads=4,
        dsa_indexer_head_dim=2,
        index_share_group_size=4,
        attention_heads=4,
        kv_heads=1,
        kv_lora_rank=4,
        q_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=16,
        vocab_size=32,
        fp8_block_shape=(2, 2),
        mlp_layer_types=("dense", "dense", "dense", *("sparse",) * 5),
        indexer_types=("full",) * 8,
    )
    assignments = tuple(
        StageAssignment(
            stage_id=stage,
            process_index=stage,
            device_ids=tuple(range(stage * 4, stage * 4 + 4)),
            layer_start=stage,
            layer_end_exclusive=stage + 1,
            persistent_weight_bytes=1,
            fp8_scale_bytes=1,
            kv_bytes_at_target_context=1,
            dsa_state_bytes=1,
            temporary_bytes=1,
            reserved_overlay_bytes=0,
        )
        for stage in range(8)
    )
    return ExecutionPlan(
        name=PlanName.PP8_LP4,
        geometry=geometry,
        topology=_topology(),
        target_context_length=16,
        pipeline_stages=8,
        local_parallel_size=4,
        stage_assignments=assignments,
        local_mesh_shape=(4,),
        residual_layout="stage_local_replicated",
        expert_layout="complete_expert_identity_lp4",
        kv_layout="stage_layer_context_sharded_lp4",
        transport="collective_permute_stage_ring",
    )


def _source_plans(layout: object) -> tuple[DestinationFilePlan, ...]:
    plans = []
    for device in layout.devices:
        offset = 0
        tensors = []
        sources = sorted(
            (
                source
                for tensor in device.tensors
                for source in tensor.sources
            ),
            key=lambda source: source.name,
        )
        for source in sources:
            tensors.append(
                DestinationTensorPlan(
                    name=source.name,
                    dtype=source.dtype,
                    shape=source.shape,
                    data_offset_start=offset,
                    data_offset_end=offset + source.byte_count,
                )
            )
            offset += source.byte_count
        plans.append(
            DestinationFilePlan(
                filename=(
                    f"base_decoder/stage_{device.stage_id:02d}/"
                    f"device_slot_{device.device_slot:02d}.safetensors"
                ),
                load_set="base_decoder",
                stage_id=device.stage_id,
                device_slot=device.device_slot,
                device_id=device.device_id,
                header=b"SOURCE_HEADER",
                payload_bytes=offset,
                tensors=tuple(tensors),
            )
        )
    return tuple(plans)


def _leaf_bytes(name: str, byte_count: int) -> bytes:
    seed = sha256(name.encode("utf-8")).digest()
    return (seed * ((byte_count + len(seed) - 1) // len(seed)))[:byte_count]


def _source_file(plan: DestinationFilePlan) -> bytes:
    return plan.header + b"".join(
        _leaf_bytes(tensor.name, tensor.byte_count) for tensor in plan.tensors
    )


def test_runtime_derivative_streams_every_leaf_and_explicit_padding() -> None:
    plan = _small_plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    sources = _source_plans(layout)
    runtime = build_runtime_destination_file_plans(
        layout,
        sources,
        source_packed_manifest_sha256="a" * 64,
    )
    assert len(runtime) == 32
    assert len({item.payload_bytes for item in runtime}) == 1

    source_plan = sources[2]
    runtime_plan = runtime[2]
    output = BytesIO()
    evidence = stream_runtime_weight_file(
        source_plan=source_plan,
        runtime_plan=runtime_plan,
        source=BytesIO(_source_file(source_plan)),
        output=output,
        chunk_bytes=7,
    )
    observed = output.getvalue()
    assert len(observed) == runtime_plan.file_bytes == evidence.file_bytes
    assert sha256(observed).hexdigest() == evidence.sha256
    assert evidence.source_payload_bytes + evidence.padding_bytes == (
        evidence.payload_bytes
    )
    assert evidence.source_leaf_count == len(source_plan.tensors)

    header_size = struct.unpack("<Q", observed[:8])[0]
    header = json.loads(observed[8 : 8 + header_size])
    assert header["__metadata__"]["greenfield_runtime_layout_hash"] == (
        layout.layout_hash
    )
    assert header["attention.slot_00.q_a.weight_bits"]["dtype"] == "U8"
    payload_start = len(runtime_plan.header)
    binding_by_name = {
        tensor.spec.name: tensor
        for tensor in runtime_plan.device_layout.tensors
    }
    source_by_name = {tensor.name: tensor for tensor in source_plan.tensors}
    for tensor_plan in runtime_plan.tensors:
        binding = binding_by_name[tensor_plan.spec.name]
        actual = observed[
            payload_start + tensor_plan.data_offset_start :
            payload_start + tensor_plan.data_offset_end
        ]
        if binding.is_padding:
            assert actual == bytes(tensor_plan.byte_count)
        else:
            expected = b"".join(
                _leaf_bytes(
                    source.name,
                    source_by_name[source.name].byte_count,
                )
                for source in binding.sources
            )
            assert actual == expected


def test_runtime_derivative_refuses_source_contract_drift() -> None:
    plan = _small_plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    sources = _source_plans(layout)
    runtime = build_runtime_destination_file_plans(
        layout,
        sources,
        source_packed_manifest_sha256="b" * 64,
    )
    source = sources[0]
    missing = replace(
        source,
        tensors=source.tensors[:-1],
        payload_bytes=source.payload_bytes - source.tensors[-1].byte_count,
    )
    with pytest.raises(CheckpointValidationError, match="exact source leaf set"):
        stream_runtime_weight_file(
            source_plan=missing,
            runtime_plan=runtime[0],
            source=BytesIO(_source_file(missing)),
            output=BytesIO(),
            chunk_bytes=7,
        )
    corrupt_header = bytearray(_source_file(source))
    corrupt_header[0] ^= 1
    with pytest.raises(CheckpointValidationError, match="header drifted"):
        stream_runtime_weight_file(
            source_plan=source,
            runtime_plan=runtime[0],
            source=BytesIO(corrupt_header),
            output=BytesIO(),
            chunk_bytes=7,
        )


def test_runtime_source_file_authentication_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "owner.safetensors"
    path.write_bytes(b"owner-payload")
    expected = sha256(path.read_bytes()).hexdigest()
    assert verify_source_file_sha256(path, expected) == expected
    path.write_bytes(b"corrupt")
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        verify_source_file_sha256(path, expected)
