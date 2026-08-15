from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderWeights,
    bind_ws32_decoder_weights,
    ws32_decode_result_specs,
    ws32_decoder_state_specs,
    ws32_decoder_weight_names,
    ws32_decoder_weight_specs,
)
from glm_tpu.greenfield.types import ModelGeometry


ROOT = Path(__file__).resolve().parents[3]


def _geometry() -> ModelGeometry:
    config = json.loads(
        (ROOT / "configs" / "glm-5.2-fp8-config.json").read_text(
            encoding="utf-8"
        )
    )
    return ModelGeometry.from_hf_config(config)


def test_ws32_decoder_contract_covers_exact_78_layer_model() -> None:
    geometry = _geometry()
    config = Ws32DecoderConfig(
        geometry=geometry, context_capacity=262_144
    )
    weights = ws32_decoder_weight_specs(config)
    state = ws32_decoder_state_specs()
    result = ws32_decode_result_specs()

    assert config.page_count == 512
    assert config.local_rows_per_page == 64
    assert config.kv_cache_shape == (78, 512, 512, 640)
    assert config.index_cache_shape == (21, 512, 512, 128)
    assert config.full_index_slots == (0, 1, 2, *range(6, 75, 4))
    assert config.full_index_slot_by_layer[:11] == (
        0,
        1,
        2,
        None,
        None,
        None,
        3,
        None,
        None,
        None,
        4,
    )
    assert len(weights.layers) == 78
    assert sum(layer.dense is not None for layer in weights.layers) == 3
    assert sum(layer.moe is not None for layer in weights.layers) == 75
    assert sum(layer.dsa is not None for layer in weights.layers) == 21
    assert isinstance(weights, Ws32DecoderWeights)
    assert str(weights.embedding_local) == "P('expert', 'feature')"
    assert str(weights.final_norm_weight_local) == "P('feature',)"
    assert str(weights.lm_head_local) == "P('expert', 'feature')"
    assert str(state.kv_cache_local) == "P(None, None, 'expert', None)"
    assert str(state.index_cache_local) == "P(None, None, 'expert', None)"
    assert str(result.final_residual_local) == "P(None, 'feature')"

    local_kv_bytes = 78 * 512 * 64 * 640 * 2
    local_index_bytes = 21 * 512 * 64 * 128 * 2
    assert local_kv_bytes == 3_271_557_120
    assert local_index_bytes == 176_160_768
    assert local_kv_bytes + local_index_bytes == 3_447_717_888


def test_ws32_decoder_contract_refuses_schedule_and_cache_drift() -> None:
    geometry = _geometry()
    with pytest.raises(PlanValidationError, match="layer zero"):
        Ws32DecoderConfig(
            geometry=replace(
                geometry,
                indexer_types=("shared", *geometry.indexer_types[1:]),
            ),
            context_capacity=8192,
        )
    with pytest.raises(PlanValidationError, match="packed cache"):
        Ws32DecoderConfig(
            geometry=geometry,
            context_capacity=8192,
            packed_cache_width=576,
        )
    with pytest.raises(PlanValidationError, match="context capacity"):
        Ws32DecoderConfig(geometry=geometry, context_capacity=0)


def test_ws32_decoder_names_bind_every_exact_final_layout_tensor() -> None:
    config = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=8192
    )
    names = ws32_decoder_weight_names(config)

    def leaves(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        assert isinstance(value, tuple)
        return tuple(name for item in value for name in leaves(item))

    exact_names = leaves(names)
    assert len(exact_names) == len(set(exact_names)) == 2310
    arrays = {name: object() for name in exact_names}
    bound = bind_ws32_decoder_weights(arrays, config)
    assert bound.embedding_local is arrays["model.embed_tokens.weight"]
    assert bound.layers[0].qkv_a.q_a_bits_local is arrays[
        "model.layers.0.self_attn.q_a_proj.weight_bits"
    ]
    assert bound.layers[77].moe is not None
    assert bound.layers[77].moe.expert_down_bits_local is arrays[
        "model.layers.77.mlp.experts.down_proj.weight_bits"
    ]
    assert bound.final_norm_weight_local is arrays["model.norm.weight"]
    assert bound.lm_head_local is arrays["lm_head.weight"]

    missing = dict(arrays)
    missing.pop("model.norm.weight")
    with pytest.raises(ValueError, match="tensor set drifted"):
        bind_ws32_decoder_weights(missing, config)
    with pytest.raises(ValueError, match="tensor set drifted"):
        bind_ws32_decoder_weights({**arrays, "rogue": object()}, config)
