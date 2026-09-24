from __future__ import annotations

from math import prod

import pytest

from glm_tpu.model_loader.placement import placements_for_source_tensor
from glm_tpu.exceptions import PlanValidationError
from glm_tpu.model_loader.source_inventory import SourceTensor
from glm_tpu.config.model import ModelGeometry

# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (tests/config/test_model.py).
from tools.equivalence.fixture import config_json


_BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}


def _geometry() -> ModelGeometry:
    return ModelGeometry.from_hf_config(config_json())


def _tensor(name: str, dtype: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(
        name=name,
        filename="model-00001-of-00141.safetensors",
        dtype=dtype,
        shape=shape,
        data_offset_start=0,
        data_offset_end=prod(shape) * _BYTES[dtype],
    )


def test_ws32_routed_sources_form_exact_local_expert_tables() -> None:
    source = _tensor(
        "model.layers.3.mlp.experts.130.gate_proj.weight",
        "F8_E4M3",
        (2048, 6144),
    )
    placements = placements_for_source_tensor(source, _geometry())
    assert [placement.slot for placement in placements] == [16, 17, 18, 19]
    assert {placement.destination_name for placement in placements} == {
        "model.layers.3.mlp.experts.gate_proj.weight_bits"
    }
    for feature, placement in enumerate(placements):
        assert placement.source_starts == (0, feature * 1536)
        assert placement.source_stops == (2048, (feature + 1) * 1536)
        assert placement.destination_shape == (32, 2048, 1536)
        assert placement.destination_starts == (2, 0, 0)
        assert placement.destination_stops == (3, 2048, 1536)
        assert placement.partition_spec == ("expert", None, "feature")
        assert placement.destination_dtype == "U8"
        assert placement.transform == "fp8_bits"
    assert sum(placement.byte_count for placement in placements) == source.byte_count


def test_ws32_attention_layout_uses_only_feature_or_expert_slices() -> None:
    geometry = _geometry()
    cases = (
        (
            _tensor(
                "model.layers.0.self_attn.q_a_proj.weight",
                "F8_E4M3",
                (2048, 6144),
            ),
            (2048, 1536),
            (None, "feature"),
            8,
        ),
        (
            _tensor(
                "model.layers.0.self_attn.q_b_proj.weight",
                "F8_E4M3",
                (16384, 2048),
            ),
            (2048, 2048),
            ("expert", None),
            4,
        ),
        (
            _tensor(
                "model.layers.0.self_attn.kv_b_proj.weight",
                "F8_E4M3",
                (28672, 512),
            ),
            (3584, 512),
            ("expert", None),
            4,
        ),
        (
            _tensor(
                "model.layers.0.self_attn.o_proj.weight",
                "F8_E4M3",
                (6144, 16384),
            ),
            (1536, 2048),
            ("feature", "expert"),
            1,
        ),
    )
    for source, local_shape, partition_spec, replication in cases:
        placements = placements_for_source_tensor(source, geometry)
        assert len(placements) == 32
        assert {placement.destination_shape for placement in placements} == {local_shape}
        assert {placement.partition_spec for placement in placements} == {partition_spec}
        assert sum(placement.byte_count for placement in placements) == (source.byte_count * replication)


def test_ws32_embedding_dense_shared_and_compact_state_placement() -> None:
    geometry = _geometry()
    embedding = _tensor("model.embed_tokens.weight", "BF16", (154880, 6144))
    embedding_placements = placements_for_source_tensor(embedding, geometry)
    assert {placement.destination_shape for placement in embedding_placements} == {(19360, 1536)}
    assert sum(item.byte_count for item in embedding_placements) == embedding.byte_count

    dense = _tensor(
        "model.layers.0.mlp.down_proj.weight",
        "F8_E4M3",
        (6144, 12288),
    )
    dense_placements = placements_for_source_tensor(dense, geometry)
    assert {item.destination_shape for item in dense_placements} == {(1536, 1536)}
    assert {item.partition_spec for item in dense_placements} == {("feature", "expert")}
    assert sum(item.byte_count for item in dense_placements) == dense.byte_count

    shared = _tensor(
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        "F8_E4M3",
        (6144, 2048),
    )
    shared_placements = placements_for_source_tensor(shared, geometry)
    assert {item.destination_shape for item in shared_placements} == {(1536, 2048)}
    assert sum(item.byte_count for item in shared_placements) == 8 * shared.byte_count

    norm = _tensor("model.layers.0.self_attn.q_a_layernorm.weight", "BF16", (2048,))
    norm_placements = placements_for_source_tensor(norm, geometry)
    assert {item.destination_shape for item in norm_placements} == {(2048,)}
    assert sum(item.byte_count for item in norm_placements) == 32 * norm.byte_count


def test_ws32_mtp_sources_are_outside_the_base_decoder_layout() -> None:
    mtp = _tensor("model.layers.78.input_layernorm.weight", "BF16", (6144,))
    assert placements_for_source_tensor(mtp, _geometry()) == ()


def test_ws32_source_contract_refuses_divisible_shape_and_layer_drift() -> None:
    geometry = _geometry()
    mutations = (
        _tensor(
            "model.layers.0.self_attn.q_a_proj.weight",
            "F8_E4M3",
            (2176, 6144),
        ),
        _tensor(
            "model.layers.3.mlp.gate_proj.weight",
            "F8_E4M3",
            (12288, 6144),
        ),
        _tensor(
            "model.layers.3.self_attn.indexer.wk.weight",
            "F8_E4M3",
            (128, 6144),
        ),
        _tensor(
            "model.layers.3.mlp.experts.256.gate_proj.weight",
            "F8_E4M3",
            (2048, 6144),
        ),
    )
    for source in mutations:
        with pytest.raises(PlanValidationError):
            placements_for_source_tensor(source, geometry)
