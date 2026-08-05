from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PartitioningValidationError
from glm_tpu.greenfield.partitioning import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    SourceFile,
    SourceInventory,
    SourceTensor,
    build_placement_ledger,
    placement_recipe,
)
from glm_tpu.greenfield.types import ModelGeometry


REPO = Path(__file__).resolve().parents[3]
DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1}


@pytest.fixture(scope="module")
def geometry() -> ModelGeometry:
    config = json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    return ModelGeometry.from_hf_config(config)


def source_tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: str = "F8_E4M3",
    *,
    start: int = 0,
) -> SourceTensor:
    count = DTYPE_BYTES[dtype]
    for dimension in shape:
        count *= dimension
    return SourceTensor(
        name=name,
        filename="model.safetensors",
        dtype=dtype,
        shape=shape,
        data_offset_start=start,
        data_offset_end=start + count,
    )


def test_complete_experts_map_whole_by_contiguous_identity(
    geometry: ModelGeometry,
) -> None:
    tensor = source_tensor(
        "model.layers.3.mlp.experts.128.gate_proj.weight", (2048, 6144)
    )
    pp8 = placement_recipe(tensor, geometry, 4)
    pp16 = placement_recipe(tensor, geometry, 2)
    assert pp8.layout == pp16.layout == "expert_identity"
    assert [shard.device_slot for shard in pp8.shards] == [2]
    assert [shard.device_slot for shard in pp16.shards] == [1]
    assert pp8.packed_byte_count == tensor.byte_count


@pytest.mark.parametrize(
    ("suffix", "shape", "expected_axis"),
    [
        ("gate_proj.weight", (2048, 6144), 0),
        ("up_proj.weight_scale_inv", (16, 48), 0),
        ("down_proj.weight", (6144, 2048), 1),
        ("down_proj.weight_scale_inv", (48, 16), 1),
    ],
)
def test_shared_expert_weight_and_scale_shard_on_same_semantic_axis(
    geometry: ModelGeometry,
    suffix: str,
    shape: tuple[int, ...],
    expected_axis: int,
) -> None:
    dtype = "F32" if suffix.endswith("scale_inv") else "F8_E4M3"
    tensor = source_tensor(
        f"model.layers.3.mlp.shared_experts.{suffix}", shape, dtype
    )
    recipe = placement_recipe(tensor, geometry, 4)
    assert recipe.layout == "axis_sharded"
    assert {shard.axis for shard in recipe.shards} == {expected_axis}
    assert len(recipe.shards) == 4
    assert recipe.packed_byte_count == tensor.byte_count


def test_router_and_q_a_are_local_stage_replicas(geometry: ModelGeometry) -> None:
    for tensor in (
        source_tensor(
            "model.layers.3.mlp.gate.weight", (256, 6144), "BF16"
        ),
        source_tensor(
            "model.layers.3.self_attn.q_a_proj.weight", (2048, 6144)
        ),
    ):
        recipe = placement_recipe(tensor, geometry, 4)
        assert recipe.layout == "replicated"
        assert len(recipe.shards) == 4
        assert recipe.packed_byte_count == tensor.byte_count * 4


def test_attention_head_and_output_axes_are_explicit(
    geometry: ModelGeometry,
) -> None:
    q_b = placement_recipe(
        source_tensor(
            "model.layers.3.self_attn.q_b_proj.weight", (16384, 2048)
        ),
        geometry,
        4,
    )
    output = placement_recipe(
        source_tensor(
            "model.layers.3.self_attn.o_proj.weight", (6144, 16384)
        ),
        geometry,
        4,
    )
    assert {shard.axis for shard in q_b.shards} == {0}
    assert {shard.axis for shard in output.shards} == {1}


def test_non_layer_and_mtp_load_sets_are_not_implicit(
    geometry: ModelGeometry,
) -> None:
    embedding = placement_recipe(
        source_tensor("model.embed_tokens.weight", (154880, 6144), "BF16"),
        geometry,
        4,
    )
    mtp = placement_recipe(
        source_tensor("model.layers.78.eh_proj.weight", (6144, 12288), "BF16"),
        geometry,
        4,
    )
    assert embedding.load_set == BASE_LOAD_SET
    assert {shard.axis for shard in embedding.shards} == {0}
    assert mtp.load_set == MTP_LOAD_SET
    assert {shard.axis for shard in mtp.shards} == {1}


def test_unknown_leaf_or_nondivisible_axis_fails_closed(
    geometry: ModelGeometry,
) -> None:
    with pytest.raises(PartitioningValidationError, match="no layer ownership"):
        placement_recipe(
            source_tensor("model.layers.3.unknown.weight", (4, 4)),
            geometry,
            4,
        )
    with pytest.raises(PartitioningValidationError, match="does not divide"):
        placement_recipe(
            source_tensor("lm_head.weight", (3, 4), "BF16"), geometry, 4
        )


def test_ledger_reconciles_source_and_separates_optional_mtp(
    geometry: ModelGeometry,
) -> None:
    tiny_geometry = replace(
        geometry,
        num_layers=1,
        first_dense_layers=1,
        mlp_layer_types=("dense",),
        indexer_types=("shared",),
    )
    tensors = []
    offset = 0
    for name, shape, dtype in (
        ("model.layers.0.mlp.gate_proj.weight", (8, 8), "F8_E4M3"),
        ("model.layers.1.eh_proj.weight", (8, 8), "BF16"),
        ("model.norm.weight", (8,), "BF16"),
    ):
        tensor = source_tensor(name, shape, dtype, start=offset)
        tensors.append(tensor)
        offset = tensor.data_offset_end
    source_file = SourceFile(
        filename="model.safetensors",
        file_bytes=offset,
        header_bytes=0,
        payload_bytes=offset,
        tensor_count=len(tensors),
        header_sha256="a" * 64,
    )
    source = SourceInventory(
        model_id=tiny_geometry.model_id,
        source_revision="fixture",
        index_filename="model.safetensors.index.json",
        index_sha256="b" * 64,
        config_filename=None,
        config_sha256=None,
        declared_payload_bytes=offset,
        files=(source_file,),
        tensors=tuple(tensors),
    )
    ledger = build_placement_ledger(
        source,
        tiny_geometry,
        local_parallel_size=4,
        require_complete_model=False,
    )
    assert ledger.source_payload_bytes == source.payload_bytes
    assert ledger.packed_bytes(load_set=MTP_LOAD_SET) == tensors[1].byte_count
    assert ledger.packed_bytes(load_set=BASE_LOAD_SET) > tensors[0].byte_count

    with pytest.raises(PartitioningValidationError, match="leaf set"):
        build_placement_ledger(source, tiny_geometry, local_parallel_size=4)
