from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.errors import PlanValidationError
from glm_tpu.optimized.ws32_decoder import (
    Ws32DecoderConfig,
    bind_ws32_decoder_weights,
    ws32_decoder_weight_names,
    ws32_decoder_weight_specs,
)
from glm_tpu.optimized.ws32_layer import Ws32StrategyNdDenseWeights
from glm_tpu.optimized.geometry import ModelGeometry
# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (test_glm53_model).
from tools.equivalence.fixture import config_json


ROOT = Path(__file__).resolve().parents[3]


def _geometry() -> ModelGeometry:
    return ModelGeometry.from_hf_config(config_json())


def test_ws32_strategy_nd_dense_contract_is_default_off_and_final_layout() -> None:
    geometry = _geometry()
    default = Ws32DecoderConfig(geometry=geometry, context_capacity=8192)
    assert not default.strategy_nd_dense
    config = Ws32DecoderConfig(
        geometry=geometry,
        context_capacity=8192,
        strategy_nd_dense=True,
    )
    specs = ws32_decoder_weight_specs(config)
    names = ws32_decoder_weight_names(config)
    for layer_id in range(3):
        dense_specs = specs.layers[layer_id].dense
        dense_names = names.layers[layer_id].dense
        assert isinstance(dense_specs, Ws32StrategyNdDenseWeights)
        assert isinstance(dense_names, Ws32StrategyNdDenseWeights)
        assert tuple(map(str, dense_specs)) == (
            "P('expert', None, None)",
            "P('expert', None, None)",
            "P('expert', None, 'feature')",
            "P('expert', None, 'feature')",
        )
        assert dense_names.merged_bits_in_out_local == (
            f"model.layers.{layer_id}.mlp.strategy_nd."
            "merged_gate_up.weight_bits_in_out"
        )
    assert all(layer.dense is None for layer in specs.layers[3:])
    with pytest.raises(PlanValidationError, match="exact GLM-5.2 geometry"):
        Ws32DecoderConfig(
            geometry=replace(geometry, hidden_size=3072),
            context_capacity=8192,
            strategy_nd_dense=True,
        )


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
    with pytest.raises(PlanValidationError, match="exact DSA flag"):
        Ws32DecoderConfig(
            geometry=geometry,
            context_capacity=8192,
            exact_dsa=1,  # type: ignore[arg-type]
        )


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


def test_ws32_main_rope_table_is_the_accepted_legacy_construction() -> None:
    """Spec §23.8: the host main-attention rotary table is the accepted GLM
    runtime's own construction, sized by the run's context capacity."""
    import ml_dtypes
    import numpy as np

    from glm_tpu.optimized.reference.rotary import build_rotary_table_host, rotary_table_sha256
    from glm_tpu.optimized.ws32_decoder import WS32_MAIN_ROPE_THETA, build_ws32_main_rope_table

    assert WS32_MAIN_ROPE_THETA == 8_000_000.0
    config = Ws32DecoderConfig(geometry=_geometry(), context_capacity=8192)
    assert config.main_rope_table_shape == (8192, 64)
    assert config.host_main_rope_table is False
    table = build_ws32_main_rope_table(config)
    assert table.shape == (8192, 64) and table.dtype == ml_dtypes.bfloat16
    assert rotary_table_sha256(table) == rotary_table_sha256(
        build_rotary_table_host(8192, rotary_dim=64, theta=8_000_000.0)
    )
    # Row 0 is cos=1, sin=0 for every pair; rows are within BF16 of FP64 truth.
    rows = np.asarray(table, dtype=np.float32)
    assert np.array_equal(rows[0], np.concatenate([np.ones(32), np.zeros(32)]).astype(np.float32))
    frequencies = np.power(
        np.float64(8_000_000.0), -np.arange(0, 64, 2, dtype=np.float64) / np.float64(64)
    )
    positions = np.arange(8192, dtype=np.float64)
    angles = positions[:, None] * frequencies[None, :]
    truth = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    assert np.max(np.abs(rows - truth)) <= 2 ** -8

    capacity = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=262_656, host_main_rope_table=True
    )
    assert capacity.main_rope_table_shape == (262_656, 64)
    assert capacity.host_main_rope_table is True
    with pytest.raises(PlanValidationError, match="host main-rotary table flag"):
        Ws32DecoderConfig(
            geometry=_geometry(), context_capacity=8192, host_main_rope_table=1
        )


def test_ws32_main_rope_row_selection_and_rotation_match_the_device_form_at_low_positions() -> None:
    """Execute the A′ math: row gather, cos|sin split and FP32-final-round
    rotation. At small positions it must agree with the on-device form to BF16;
    at long positions it must not, which is why §23.9 adopted it."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from glm_tpu.optimized.reference.rotary import apply_rotary, apply_rotary_fp32_final_round, rotary_cos_sin
    from glm_tpu.optimized.ws32_decoder import WS32_MAIN_ROPE_THETA, build_ws32_main_rope_table

    capacity = 262_656
    config = Ws32DecoderConfig(
        geometry=_geometry(), context_capacity=capacity, host_main_rope_table=True
    )
    table = jnp.asarray(build_ws32_main_rope_table(config))
    rotary_dim = config.geometry.qk_rope_head_dim
    half = rotary_dim // 2
    probe = jnp.asarray(
        np.tile(np.asarray([1.0, 0.0], dtype=np.float32), rotary_dim // 2)[None, None, :],
        dtype=jnp.bfloat16,
    )

    def table_form(position: int):
        row = jnp.take(table, jnp.asarray([position], dtype=jnp.int32), axis=0, mode="clip")[0]
        cos = row[:half][None, :]
        sin = row[half:][None, :]
        return apply_rotary_fp32_final_round(
            probe, cos[:, None, :], sin[:, None, :], interleaved=True
        )

    def device_form(position: int):
        cos, sin = rotary_cos_sin(
            jnp.asarray([position], dtype=jnp.int32),
            rotary_dim=rotary_dim,
            theta=WS32_MAIN_ROPE_THETA,
            dtype=jnp.bfloat16,
        )
        return apply_rotary(probe, cos[:, None, :], sin[:, None, :], interleaved=True)

    for position in (0, 1, 17, 1024):
        got = np.asarray(jax.jit(table_form, static_argnums=0)(position), dtype=np.float32)
        expected = np.asarray(jax.jit(device_form, static_argnums=0)(position), dtype=np.float32)
        assert np.max(np.abs(got - expected)) <= 2 ** -7, position
    # Row 0 rotates the probe by the identity.
    assert np.array_equal(
        np.asarray(jax.jit(table_form, static_argnums=0)(0), dtype=np.float32),
        np.asarray(probe, dtype=np.float32),
    )
    # The clamp is defined and finite at and beyond the capacity.
    for position in (capacity - 1, capacity, capacity + 5):
        assert np.all(np.isfinite(np.asarray(jax.jit(table_form, static_argnums=0)(position), dtype=np.float32)))
    # Against FP64 the table row is the accurate one at a long position.
    position = 262_000
    frequencies = np.power(
        np.float64(WS32_MAIN_ROPE_THETA),
        -np.arange(0, rotary_dim, 2, dtype=np.float64) / np.float64(rotary_dim),
    )
    truth_cos = np.cos(np.float64(position) * frequencies)
    row = np.asarray(table[position], dtype=np.float32)
    device_cos = np.asarray(
        rotary_cos_sin(
            jnp.asarray([position], dtype=jnp.int32),
            rotary_dim=rotary_dim,
            theta=WS32_MAIN_ROPE_THETA,
            dtype=jnp.bfloat16,
        )[0],
        dtype=np.float32,
    )[0]
    assert np.max(np.abs(row[:half] - truth_cos)) <= np.max(np.abs(device_cos - truth_cos)) + 1e-9
