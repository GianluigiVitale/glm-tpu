from __future__ import annotations

import json
import struct
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

from glm_tpu.greenfield.checkpoint import (
    DestinationFilePlan,
    DestinationTensorPlan,
    build_runtime_destination_file_plans,
    build_runtime_layout_document,
    stream_runtime_weight_file,
    verify_source_file_sha256,
)
from glm_tpu.greenfield.errors import CheckpointValidationError, PlanValidationError
from glm_tpu.greenfield.model import (
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    build_decoder_feature_runtime_weight_layout,
    build_decoder_feature_fused_qkv_runtime_weight_layout,
    build_decoder_fused_qkv_runtime_weight_layout,
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


def _small_feature_source_plan() -> ExecutionPlan:
    plan = _small_plan()
    return replace(
        plan,
        geometry=replace(plan.geometry, moe_intermediate_size=8),
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
        verified_source_file_sha256=sha256(
            _source_file(source_plan)
        ).hexdigest(),
        chunk_bytes=7,
    )
    observed = output.getvalue()
    assert len(observed) == runtime_plan.file_bytes == evidence.file_bytes
    assert sha256(observed).hexdigest() == evidence.sha256
    assert evidence.source_payload_bytes + evidence.padding_bytes == (
        evidence.payload_bytes
    )
    assert evidence.source_leaf_count == len(source_plan.tensors)
    assert evidence.source_file_sha256 == sha256(
        _source_file(source_plan)
    ).hexdigest()

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
            verified_source_file_sha256="c" * 64,
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
            verified_source_file_sha256="d" * 64,
            chunk_bytes=7,
        )


def test_runtime_layout_document_is_self_authenticating() -> None:
    plan = _small_plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    document = build_runtime_layout_document(layout)
    unhashed = dict(document)
    observed = unhashed.pop("manifest_sha256")
    encoded = json.dumps(
        unhashed,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert observed == sha256(encoded).hexdigest()
    assert document["runtime_layout_hash"] == layout.layout_hash


def test_feature_runtime_layout_redistributes_only_routed_expert_features() -> None:
    source_plan = _small_feature_source_plan()
    source_schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    target_layout = build_decoder_feature_runtime_weight_layout(
        target_plan,
        target_schedule,
        source_layout,
    )

    assert "routed_expert_layout" not in source_layout.to_dict()
    assert all(
        "transform" not in tensor.to_dict()
        for device in source_layout.devices
        for tensor in device.tensors
    )
    assert target_layout.plan_hash == target_plan.plan_hash
    assert target_layout.schedule_hash == target_schedule.schedule_hash
    assert target_layout.routed_expert_layout == FEATURE_EXPERT_RUNTIME_LAYOUT
    assert target_layout.runtime_bytes_per_chip == source_layout.runtime_bytes_per_chip

    specs = {spec.name: spec for spec in target_layout.specs}
    assert specs["sparse.slot_00.experts.gate_proj.weight_bits"].shape == (
        8,
        8,
        2,
    )
    assert specs["sparse.slot_00.experts.up_proj.scale_inv"].shape == (
        8,
        1,
        4,
    )
    assert specs["sparse.slot_00.experts.down_proj.weight_bits"].shape == (
        8,
        2,
        8,
    )
    assert specs["sparse.slot_00.experts.down_proj.scale_inv"].shape == (
        8,
        4,
        1,
    )

    sparse_device = target_layout.devices[3 * 4]
    bindings = {
        tensor.spec.name: tensor for tensor in sparse_device.tensors
    }
    gate = bindings["sparse.slot_00.experts.gate_proj.weight_bits"]
    assert gate.transform == "concat_experts_slice_output_transpose"
    assert {source.source_device_slot for source in gate.sources} == {0, 1, 2, 3}
    assert all(source.shape == (2, 8, 8) for source in gate.sources)
    assert all(source.selected_shape == (2, 8, 2) for source in gate.sources)
    assert gate.source_byte_count == gate.spec.byte_count

    router = bindings["sparse.slot_00.router_weight"]
    assert router.transform == "identity_runtime_tensor"
    assert router.sources[0].source_device_slot == 0
    assert router.sources[0].shape == router.spec.shape


def test_feature_runtime_layout_rejects_partial_fp8_scale_blocks() -> None:
    source_plan = _small_plan()
    source_schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    with pytest.raises(PlanValidationError, match="complete FP8 scale blocks"):
        build_decoder_feature_runtime_weight_layout(
            target_plan,
            target_schedule,
            source_layout,
        )


def test_fused_qkv_runtime_layout_replaces_separate_projection_state() -> None:
    source_plan = _small_plan()
    source_plan = replace(
        source_plan,
        geometry=replace(
            source_plan.geometry,
            hidden_size=6144,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            moe_intermediate_size=512,
            fp8_block_shape=(128, 128),
        ),
    )
    schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(source_plan, schedule)
    target_layout = build_decoder_fused_qkv_runtime_weight_layout(
        source_plan,
        schedule,
        source_layout,
    )

    specs = {spec.name: spec for spec in target_layout.specs}
    assert target_layout.attention_projection_layout == (
        FUSED_QKV_A_N82_RUNTIME_LAYOUT
    )
    assert specs["attention.slot_00.qkv_a.weight_bits"].shape == (
        32,
        6144,
        82,
    )
    assert specs["attention.slot_00.qkv_a.scale_inv"].shape == (
        32,
        48,
        82,
    )
    assert not any(
        name.endswith(
            (
                ".q_a.weight_bits",
                ".q_a.scale_inv",
                ".kv_a.weight_bits",
                ".kv_a.scale_inv",
            )
        )
        for name in specs
    )
    bindings = {
        tensor.spec.name: tensor for tensor in target_layout.devices[0].tensors
    }
    weight = bindings["attention.slot_00.qkv_a.weight_bits"]
    scale = bindings["attention.slot_00.qkv_a.scale_inv"]
    assert weight.transform == "fuse_qkv_a_output_shards"
    assert scale.transform == "fuse_qkv_a_expanded_scales"
    assert tuple(source.name for source in weight.sources) == (
        "attention.slot_00.q_a.weight_bits",
        "attention.slot_00.kv_a.weight_bits",
    )
    assert weight.derived_bytes == 0
    assert scale.derived_bytes == scale.spec.byte_count - scale.source_byte_count
    assert target_layout.runtime_bytes_per_chip > source_layout.runtime_bytes_per_chip
    assert target_layout.to_dict()["attention_projection_layout"] == (
        FUSED_QKV_A_N82_RUNTIME_LAYOUT
    )


def test_feature_and_fused_qkv_layout_share_one_source_pass() -> None:
    source_plan = _small_feature_source_plan()
    source_plan = replace(
        source_plan,
        geometry=replace(
            source_plan.geometry,
            hidden_size=6144,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            moe_intermediate_size=512,
            fp8_block_shape=(128, 128),
        ),
    )
    source_schedule = build_pipeline_schedule(source_plan)
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    target_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    target_schedule = build_pipeline_schedule(target_plan)
    target = build_decoder_feature_fused_qkv_runtime_weight_layout(
        target_plan,
        target_schedule,
        source_layout,
    )

    assert target.routed_expert_layout == FEATURE_EXPERT_RUNTIME_LAYOUT
    assert target.attention_projection_layout == FUSED_QKV_A_N82_RUNTIME_LAYOUT
    sparse_bindings = {
        tensor.spec.name: tensor for tensor in target.devices[3 * 4].tensors
    }
    assert sparse_bindings[
        "sparse.slot_00.experts.gate_proj.weight_bits"
    ].transform == "concat_experts_slice_output_transpose"
    qkv = sparse_bindings["attention.slot_00.qkv_a.weight_bits"]
    assert qkv.transform == "fuse_qkv_a_output_shards"
    assert tuple(source.name for source in qkv.sources) == (
        "attention.slot_00.q_a.weight_bits",
        "attention.slot_00.kv_a.weight_bits",
    )


def test_runtime_source_file_authentication_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "owner.safetensors"
    path.write_bytes(b"owner-payload")
    expected = sha256(path.read_bytes()).hexdigest()
    assert verify_source_file_sha256(path, expected) == expected
    path.write_bytes(b"corrupt")
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        verify_source_file_sha256(path, expected)
