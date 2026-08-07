from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    LegacyScoreGeometry,
    affine_key_layer_norm,
    bfloat16_from_uint16_bits,
    layer0_dsa_state,
    legacy_pagewise_dcp_scores,
    one_row_pagewise_scores,
    pack_legacy_fused_qkv_runtime_weights,
)
from glm_tpu.greenfield.validation.layer0_dsa_association import (
    compare_dsa_association_scores,
)


def test_bfloat16_artifact_bits_roundtrip() -> None:
    expected = np.asarray([0.0, 1.0, -2.5, 0.125], dtype=ml_dtypes.bfloat16)
    bits = expected.view(np.uint16)
    actual = np.asarray(
        bfloat16_from_uint16_bits(jnp.asarray(bits)),
        dtype=ml_dtypes.bfloat16,
    )
    assert np.array_equal(actual.view(np.uint16), bits)


def test_key_norm_modes_are_explicit_and_finite() -> None:
    value = jnp.asarray(
        [[0.125, -1.75, 2.5, 0.375], [4.0, -0.25, 0.75, -3.0]],
        dtype=jnp.float32,
    )
    weight = jnp.asarray([1.0, 0.75, -0.5, 1.25], dtype=jnp.bfloat16)
    bias = jnp.asarray([0.0, -0.125, 0.25, 0.5], dtype=jnp.bfloat16)
    divided = affine_key_layer_norm(
        value,
        weight,
        bias,
        epsilon=1e-6,
        mode="divide_sqrt",
    )
    multiplied = affine_key_layer_norm(
        value,
        weight,
        bias,
        epsilon=1e-6,
        mode="multiply_rsqrt",
    )
    assert divided.shape == value.shape
    assert divided.dtype == jnp.float32
    assert np.isfinite(np.asarray(divided)).all()
    assert np.isfinite(np.asarray(multiplied)).all()


def test_layer0_state_preserves_chunk_and_decode_geometry() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=3,
        hidden_size=8,
        q_lora_rank=4,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(11)

    def bf16(shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.asarray(
            rng.normal(size=shape).astype(ml_dtypes.bfloat16)
        )

    state = layer0_dsa_state(
        bf16((3, 8)),
        jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
        bf16((8,)),
        bf16((4, 8)),
        bf16((4,)),
        jnp.asarray(rng.normal(size=(8, 4)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.float32),
        bf16((4,)),
        bf16((4,)),
        bf16((2, 8)),
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    assert state.query.shape == (3, 2, 4)
    assert state.index_keys.shape == (6, 4)
    assert state.head_weights.shape == (3, 2)
    assert state.query.dtype == jnp.float32
    assert state.index_keys.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(state.query)).all()


def test_layer0_state_keeps_legacy_fused_qkv_companion_live() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=3,
        hidden_size=8,
        q_lora_rank=4,
        qkv_a_companion_rank=2,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(13)

    def bf16(shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.asarray(
            rng.normal(size=shape).astype(ml_dtypes.bfloat16)
        )

    state = layer0_dsa_state(
        bf16((3, 8)),
        jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
        bf16((8,)),
        bf16((6, 8)),
        bf16((4,)),
        jnp.asarray(rng.normal(size=(8, 4)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.float32),
        bf16((4,)),
        bf16((4,)),
        bf16((2, 8)),
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        q_a_projection_mode="legacy_fused_qkv_a",
    )
    assert state.query.shape == (3, 2, 4)
    assert state.qkv_a_companion.shape == (3, 2)
    assert state.qkv_a_companion.dtype == jnp.bfloat16
    assert np.any(np.asarray(state.qkv_a_companion) != 0)


def test_legacy_runtime_fused_qkv_pack_preserves_part_per_shard_order() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=3,
        hidden_size=128,
        q_lora_rank=4,
        qkv_a_companion_rank=2,
        legacy_tensor_shards=2,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    q_bits = np.arange(4 * 128, dtype=np.uint8).reshape(4, 128) % 0x70
    kv_bits = (
        np.arange(2 * 128, dtype=np.uint8).reshape(2, 128) + 3
    ) % 0x70
    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.asarray(q_bits),
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray(kv_bits),
        jnp.asarray([[5.0]], dtype=jnp.float32),
        geometry=geometry,
    )
    assert packed.global_weight.shape == (128, 6)
    assert packed.global_scale.shape == (1, 6)
    assert packed.sharded_weight.shape == (2, 128, 3)
    assert packed.sharded_scale.shape == (2, 1, 3)
    assert packed.global_weight.dtype == jnp.float8_e4m3fn
    assert np.array_equal(
        np.asarray(packed.global_weight).view(np.uint8),
        np.asarray(packed.sharded_weight)
        .transpose(1, 0, 2)
        .reshape(128, 6)
        .view(np.uint8),
    )
    assert np.array_equal(
        np.asarray(packed.sharded_weight[0]).view(np.uint8),
        np.concatenate((q_bits[:2].T, kv_bits[:1].T), axis=1),
    )
    assert np.array_equal(
        np.asarray(packed.sharded_scale[:, 0]),
        np.asarray([[2.0, 2.0, 5.0], [2.0, 2.0, 5.0]], dtype=np.float32),
    )


def test_layer0_runtime_fp8_global_and_sharded_fused_modes_are_live() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=3,
        hidden_size=128,
        q_lora_rank=4,
        qkv_a_companion_rank=2,
        legacy_tensor_shards=2,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(19)

    def bf16(shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.asarray(rng.normal(size=shape).astype(ml_dtypes.bfloat16))

    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.full((4, 128), 0x38, dtype=jnp.uint8),
        jnp.asarray([[0.5]], dtype=jnp.float32),
        jnp.full((2, 128), 0x30, dtype=jnp.uint8),
        jnp.asarray([[0.25]], dtype=jnp.float32),
        geometry=geometry,
    )
    prefix = (
        bf16((3, 128)),
        jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
        bf16((128,)),
    )
    tail = (
        bf16((4,)),
        jnp.asarray(rng.normal(size=(8, 4)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(4, 128)), dtype=jnp.float32),
        bf16((4,)),
        bf16((4,)),
        bf16((2, 128)),
    )
    global_state = layer0_dsa_state(
        *prefix,
        packed.global_weight,
        *tail,
        packed.global_scale,
        geometry=geometry,
        q_a_projection_mode="legacy_runtime_fused_qkv_a_global",
    )
    sharded_state = layer0_dsa_state(
        *prefix,
        packed.sharded_weight,
        *tail,
        packed.sharded_scale,
        geometry=geometry,
        q_a_projection_mode="legacy_runtime_fused_qkv_a_sharded",
    )
    for state in (global_state, sharded_state):
        assert state.query.shape == (3, 2, 4)
        assert state.qkv_a_companion.shape == (3, 2)
        assert state.qkv_a_companion.dtype == jnp.bfloat16
        assert np.isfinite(np.asarray(state.query)).all()
        assert np.any(np.asarray(state.qkv_a_companion) != 0)


def test_legacy_pagewise_geometry_reconstructs_direct_row() -> None:
    rng = np.random.default_rng(17)
    query = rng.normal(size=(3, 2, 4)).astype(np.float32)
    keys = rng.normal(size=(7, 4)).astype(ml_dtypes.bfloat16)
    head_weights = rng.normal(size=(3, 2)).astype(np.float32)
    geometry = LegacyScoreGeometry(
        dcp_size=2,
        local_page_size=2,
        local_score_width=6,
    )
    actual = np.asarray(
        legacy_pagewise_dcp_scores(
            jnp.asarray(query),
            jnp.asarray(keys),
            jnp.asarray(head_weights),
            geometry=geometry,
        ),
        dtype=np.float32,
    )
    per_head = np.maximum(
        np.einsum("hd,sd->hs", query[0], keys.astype(np.float32)) * 0.5,
        0.0,
    )
    expected = np.einsum("h,hs->s", head_weights[0], per_head)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_one_row_pagewise_matches_direct_small_score() -> None:
    rng = np.random.default_rng(23)
    query = rng.normal(size=(1, 2, 4)).astype(np.float32)
    keys = rng.normal(size=(7, 4)).astype(ml_dtypes.bfloat16)
    head_weights = rng.normal(size=(1, 2)).astype(np.float32)
    actual = np.asarray(
        one_row_pagewise_scores(
            jnp.asarray(query),
            jnp.asarray(keys),
            jnp.asarray(head_weights),
            page_size=2,
        ),
        dtype=np.float32,
    )
    per_head = np.maximum(
        np.einsum("hd,sd->hs", query[0], keys.astype(np.float32)) * 0.5,
        0.0,
    )
    expected = np.einsum("h,hs->s", head_weights[0], per_head)
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_association_comparison_pins_lowest_position_ties() -> None:
    scores = np.asarray([0.5, 2.0, 2.0, 1.0], dtype=np.float32)
    expected_positions = np.asarray([1, 2], dtype=np.int32)
    expected_scores = scores[expected_positions]
    comparison = compare_dsa_association_scores(
        scores,
        expected_positions,
        expected_scores,
    )
    assert comparison["passed"] is True
    assert comparison["actual_top_positions"] == [1, 2]


def test_association_comparison_reports_cutoff_swap() -> None:
    scores = np.asarray([0.5, 2.0, 2.0, 1.0], dtype=np.float32)
    comparison = compare_dsa_association_scores(
        scores,
        np.asarray([1, 3], dtype=np.int32),
        np.asarray([2.0, 1.0], dtype=np.float32),
    )
    assert comparison["passed"] is False
    assert comparison["swapped_position_count"] == 1


def test_association_comparison_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        compare_dsa_association_scores(
            np.asarray([1.0, np.nan], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
        )
