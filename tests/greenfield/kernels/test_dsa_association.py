from __future__ import annotations

from functools import partial
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.reference.dsa_association import (
    LegacyDcpXlaScoreGeometry,
    Layer0DsaProbeGeometry,
    LegacyTp32QaNormOutput,
    LegacyScoreGeometry,
    affine_key_layer_norm,
    bfloat16_from_uint16_bits,
    layer0_dsa_state,
    layer0_prompt_index_key_chunk,
    layer0_prompt_index_key_gather_cache_chunk,
    layer0_prompt_index_key_gather_cache_states_chunk,
    layer0_prompt_index_key_gather_chunk,
    layer0_prompt_index_keys_chunked,
    layer0_dsa_state_from_q_residual,
    legacy_local_dcp_score_inputs,
    legacy_local_dcp_xla_scores,
    legacy_tp32_fused_qkv_a_rms_norm,
    legacy_tp32_gspmd_fused_qkv_a_rms_norm,
    legacy_pagewise_dcp_scores,
    one_row_pagewise_scores,
    one_row_virtual_tp32_fused_qkv_a_rms_norm,
    pack_legacy_fused_qkv_runtime_weights,
)
from glm_tpu.greenfield.kernels.reference.qkv_a import (
    FusedQkvAContract,
    one_row_fused_qkv_a_convolution,
)
from glm_tpu.greenfield.validation.layer0_dsa_association import (
    compare_dsa_association_scores,
)


def test_layer0_probe_geometry_pins_distinct_model_and_key_epsilons() -> None:
    geometry = Layer0DsaProbeGeometry()
    assert geometry.rms_norm_epsilon == 1e-5
    assert geometry.q_norm_epsilon == 1e-5
    assert geometry.key_norm_epsilon == 1e-6


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


def test_chunked_prompt_keys_keep_only_live_rows() -> None:
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
    rng = np.random.default_rng(29)

    def bf16(shape: tuple[int, ...]) -> jnp.ndarray:
        return jnp.asarray(
            rng.normal(size=shape).astype(ml_dtypes.bfloat16)
        )

    arguments = (
        bf16((3, 8)),
        jnp.asarray([0, 1, 2, 1, 0], dtype=jnp.int32),
        bf16((8,)),
        jnp.asarray(rng.normal(size=(4, 8)), dtype=jnp.float32),
        bf16((4,)),
        bf16((4,)),
    )
    divided = layer0_prompt_index_keys_chunked(
        *arguments,
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    multiplied = layer0_prompt_index_keys_chunked(
        *arguments,
        geometry=geometry,
        key_norm_mode="multiply_rsqrt",
    )
    assert divided.shape == (5, 4)
    assert divided.dtype == jnp.bfloat16
    assert multiplied.shape == divided.shape
    assert multiplied.dtype == divided.dtype
    assert np.isfinite(np.asarray(divided, dtype=np.float32)).all()
    assert np.isfinite(np.asarray(multiplied, dtype=np.float32)).all()
    state = layer0_dsa_state(
        arguments[0],
        arguments[1],
        jnp.asarray([2], dtype=jnp.int32),
        arguments[2],
        bf16((4, 8)),
        bf16((4,)),
        jnp.asarray(rng.normal(size=(8, 4)), dtype=jnp.float32),
        arguments[3],
        arguments[4],
        arguments[5],
        bf16((2, 8)),
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    np.testing.assert_array_equal(
        np.asarray(divided), np.asarray(state.index_keys[:5])
    )
    chunk = layer0_prompt_index_key_chunk(
        jnp.take(arguments[0], arguments[1][:4], axis=0),
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    assert chunk.shape == (4, 4)
    assert chunk.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(chunk, dtype=np.float32)).all()
    bf16_weight_chunk = layer0_prompt_index_key_chunk(
        jnp.take(arguments[0], arguments[1][:4], axis=0),
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        projection_weight_mode="adapted_bf16",
    )
    assert bf16_weight_chunk.shape == chunk.shape
    assert bf16_weight_chunk.dtype == chunk.dtype
    assert np.isfinite(
        np.asarray(bf16_weight_chunk, dtype=np.float32)
    ).all()
    gathered_bf16_weight_chunk = layer0_prompt_index_key_gather_chunk(
        arguments[0],
        arguments[1][:4],
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        projection_weight_mode="adapted_bf16",
    )
    np.testing.assert_array_equal(
        np.asarray(gathered_bf16_weight_chunk),
        np.asarray(bf16_weight_chunk),
    )

    cache = jnp.zeros((3, 1, 4, 4), dtype=jnp.bfloat16)
    live_block_table = jnp.asarray([1, 2], dtype=jnp.int32)
    cache = layer0_prompt_index_key_gather_cache_chunk(
        cache,
        live_block_table,
        arguments[0],
        arguments[1][:4],
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    source_rope_cache = layer0_prompt_index_key_gather_cache_chunk(
        jnp.zeros((3, 1, 4, 4), dtype=jnp.bfloat16),
        live_block_table,
        arguments[0],
        arguments[1][:4],
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
    )
    np.testing.assert_array_equal(
        np.asarray(source_rope_cache), np.asarray(cache)
    )
    states = layer0_prompt_index_key_gather_cache_states_chunk(
        jnp.zeros((3, 1, 4, 4), dtype=jnp.bfloat16),
        live_block_table,
        arguments[0],
        arguments[1][:4],
        jnp.arange(4, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
    )
    np.testing.assert_array_equal(
        np.asarray(states.index_cache), np.asarray(source_rope_cache)
    )
    assert states.pre_layer_norm_key.shape == (4, 4)
    assert states.pre_rope_key.shape == (4, 4)
    assert states.post_rope_key.shape == (4, 4)
    assert states.pre_layer_norm_key.dtype == jnp.float32
    assert states.pre_rope_key.dtype == jnp.float32
    assert states.post_rope_key.dtype == jnp.float32
    with pytest.raises(ValueError, match="rotary association"):
        layer0_prompt_index_key_gather_cache_chunk(
            jnp.zeros((3, 1, 4, 4), dtype=jnp.bfloat16),
            live_block_table,
            arguments[0],
            arguments[1][:4],
            jnp.arange(4, dtype=jnp.int32),
            *arguments[2:],
            geometry=geometry,
            key_norm_mode="divide_sqrt",
            rotary_mode="unsupported",  # type: ignore[arg-type]
        )
    cache = layer0_prompt_index_key_gather_cache_chunk(
        cache,
        live_block_table,
        arguments[0],
        jnp.asarray([0, 0, 0, 0], dtype=jnp.int32),
        jnp.arange(4, 8, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    cache_rows = np.asarray(cache).reshape(3, 4, 4)
    np.testing.assert_array_equal(
        cache_rows[1], np.asarray(gathered_bf16_weight_chunk)
    )
    expected_tail = layer0_prompt_index_key_gather_chunk(
        arguments[0],
        jnp.asarray([0, 0, 0, 0], dtype=jnp.int32),
        jnp.arange(4, 8, dtype=jnp.int32),
        *arguments[2:],
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        projection_weight_mode="adapted_bf16",
    )
    np.testing.assert_array_equal(
        cache_rows[2, 0], np.asarray(expected_tail[0])
    )
    np.testing.assert_array_equal(cache_rows[2, 1:], np.zeros((3, 4)))
    np.testing.assert_array_equal(cache_rows[0], np.zeros((4, 4)))


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


def test_one_row_virtual_tp32_q_a_preserves_shard_major_n82_contract() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=32,
        hidden_size=128,
        q_lora_rank=64,
        qkv_a_companion_rank=32,
        legacy_tensor_shards=32,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(23)
    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.asarray(
            rng.integers(0x20, 0x48, size=(64, 128), dtype=np.uint8)
        ),
        jnp.asarray([[0.0125]], dtype=jnp.float32),
        jnp.asarray(
            rng.integers(0x20, 0x48, size=(32, 128), dtype=np.uint8)
        ),
        jnp.asarray([[0.0075]], dtype=jnp.float32),
        geometry=geometry,
    )
    hidden = jnp.asarray(
        rng.normal(size=(1, 128)).astype(ml_dtypes.bfloat16)
    )
    norm_weight = jnp.asarray(
        rng.uniform(0.75, 1.25, size=(64,)).astype(ml_dtypes.bfloat16)
    )
    outputs = []
    for projection_mode in (
        "lax_map",
        "vmap",
        "unrolled",
        "lax_map_convolution",
    ):
        output = one_row_virtual_tp32_fused_qkv_a_rms_norm(
            hidden,
            packed.sharded_weight,
            packed.sharded_scale,
            norm_weight,
            projection_mode=projection_mode,
            norm_mode="shard_sum",
            geometry=geometry,
        )
        assert output.q_residual.shape == (1, 64)
        assert output.qkv_a_companion.shape == (1, 32)
        assert output.q_residual.dtype == jnp.bfloat16
        assert output.qkv_a_companion.dtype == jnp.bfloat16
        outputs.append(np.asarray(output.q_residual).view(np.uint16))
    for output in outputs[1:]:
        np.testing.assert_array_equal(output, outputs[0])


def test_production_fused_qkv_a_matches_proven_convolution_association() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=2,
        hidden_size=128,
        q_lora_rank=4,
        qkv_a_companion_rank=2,
        legacy_tensor_shards=2,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(29)
    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.asarray(rng.integers(0x20, 0x48, size=(4, 128), dtype=np.uint8)),
        jnp.asarray([[0.0125]], dtype=jnp.float32),
        jnp.asarray(rng.integers(0x20, 0x48, size=(2, 128), dtype=np.uint8)),
        jnp.asarray([[0.0075]], dtype=jnp.float32),
        geometry=geometry,
    )
    hidden = jnp.asarray(
        rng.normal(size=(1, 128)).astype(ml_dtypes.bfloat16)
    )
    norm_weight = jnp.asarray(
        rng.uniform(0.75, 1.25, size=(4,)).astype(ml_dtypes.bfloat16)
    )
    expected = one_row_virtual_tp32_fused_qkv_a_rms_norm(
        hidden,
        packed.sharded_weight,
        packed.sharded_scale,
        norm_weight,
        projection_mode="lax_map_convolution",
        norm_mode="shard_sum",
        geometry=geometry,
    )
    production = one_row_fused_qkv_a_convolution(
        hidden,
        jax.lax.bitcast_convert_type(packed.sharded_weight, jnp.uint8),
        packed.sharded_scale,
        norm_weight,
        contract=FusedQkvAContract(
            hidden_size=128,
            q_lora_rank=4,
            kv_a_width=2,
            virtual_shards=2,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(production.q_residual).view(np.uint16),
        np.asarray(expected.q_residual).view(np.uint16),
    )
    np.testing.assert_array_equal(
        np.asarray(production.kv_a_projection).view(np.uint16),
        np.asarray(expected.qkv_a_companion).view(np.uint16),
    )


def test_one_row_virtual_tp32_q_a_rejects_dead_decode_rows() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=32,
        hidden_size=128,
        q_lora_rank=64,
        qkv_a_companion_rank=32,
        legacy_tensor_shards=32,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.zeros((64, 128), dtype=jnp.uint8),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.zeros((32, 128), dtype=jnp.uint8),
        jnp.ones((1, 1), dtype=jnp.float32),
        geometry=geometry,
    )
    with pytest.raises(ValueError, match="normalized_hidden shape drifted"):
        one_row_virtual_tp32_fused_qkv_a_rms_norm(
            jnp.zeros((32, 128), dtype=jnp.bfloat16),
            packed.sharded_weight,
            packed.sharded_scale,
            jnp.ones((64,), dtype=jnp.bfloat16),
            projection_mode="vmap",
            norm_mode="logical_mean",
            geometry=geometry,
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


def _run_forced_32_distributed_q_a_norm_case() -> None:
    if len(jax.devices()) != 32:
        raise AssertionError(f"expected 32 CPU devices, got {jax.devices()}")
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=3,
        hidden_size=128,
        q_lora_rank=64,
        qkv_a_companion_rank=32,
        legacy_tensor_shards=32,
        heads=2,
        head_dim=4,
        rotary_dim=2,
        theta=64.0,
    )
    rng = np.random.default_rng(29)
    q_bits = rng.integers(
        0x20, 0x48, size=(geometry.q_lora_rank, geometry.hidden_size), dtype=np.uint8
    )
    kv_bits = rng.integers(
        0x20,
        0x48,
        size=(geometry.qkv_a_companion_rank, geometry.hidden_size),
        dtype=np.uint8,
    )
    packed = pack_legacy_fused_qkv_runtime_weights(
        jnp.asarray(q_bits),
        jnp.asarray(rng.uniform(0.005, 0.02, size=(1, 1)), dtype=jnp.float32),
        jnp.asarray(kv_bits),
        jnp.asarray(rng.uniform(0.005, 0.02, size=(1, 1)), dtype=jnp.float32),
        geometry=geometry,
    )
    normalized = jnp.asarray(
        rng.normal(size=(geometry.decode_rows, geometry.hidden_size)).astype(
            ml_dtypes.bfloat16
        )
    )
    norm_weight = jnp.asarray(
        rng.uniform(0.75, 1.25, size=(geometry.q_lora_rank,)).astype(
            ml_dtypes.bfloat16
        )
    )
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("legacy_model",))
    mapped = jax.shard_map(
        partial(
            legacy_tp32_fused_qkv_a_rms_norm,
            axis_name="legacy_model",
            geometry=geometry,
        ),
        mesh=mesh,
        in_specs=(P(), P(None, "legacy_model"), P(None, "legacy_model"), P()),
        out_specs=LegacyTp32QaNormOutput(
            P(),
            P(None, "legacy_model"),
        ),
        check_vma=False,
    )
    arguments = (
        jax.device_put(normalized, NamedSharding(mesh, P())),
        jax.device_put(
            packed.global_weight,
            NamedSharding(mesh, P(None, "legacy_model")),
        ),
        jax.device_put(
            packed.global_scale,
            NamedSharding(mesh, P(None, "legacy_model")),
        ),
        jax.device_put(norm_weight, NamedSharding(mesh, P())),
    )
    lowered = jax.jit(mapped).lower(*arguments)
    compiled = lowered.compile()
    actual = compiled(*arguments)
    jax.block_until_ready(actual)

    local_output = (
        geometry.q_lora_rank + geometry.qkv_a_companion_rank
    ) // geometry.legacy_tensor_shards
    q_local = geometry.q_lora_rank // geometry.legacy_tensor_shards
    expanded_scale = jnp.repeat(
        packed.global_scale,
        geometry.hidden_size // packed.global_scale.shape[0],
        axis=0,
    )
    decoded = (
        packed.global_weight.astype(jnp.float32) * expanded_scale
    ).astype(jnp.bfloat16)
    projected = jax.lax.dot_general(
        normalized,
        decoded,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    projected = projected.reshape(
        geometry.decode_rows,
        geometry.legacy_tensor_shards,
        local_output,
    )
    logical_q = projected[:, :, :q_local].reshape(
        geometry.decode_rows, geometry.q_lora_rank
    )
    logical_companion = projected[:, :, q_local:].reshape(
        geometry.decode_rows, geometry.qkv_a_companion_rank
    )
    value_f32 = logical_q.astype(jnp.float32)
    reference = (
        value_f32
        * jax.lax.rsqrt(
            jnp.mean(jnp.square(value_f32), axis=-1, keepdims=True)
            + jnp.float32(geometry.q_norm_epsilon)
        )
    ).astype(jnp.bfloat16)
    reference = (reference * norm_weight).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(actual.q_residual, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        rtol=0,
        atol=2**-7,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.qkv_a_companion_shard).view(np.uint16),
        np.asarray(logical_companion).view(np.uint16),
    )

    from glm_tpu.greenfield.benchmarking.dsa_association import (
        validate_dsa_association_hlo,
    )

    contract = validate_dsa_association_hlo(
        compiled.as_text(),
        phase="legacy_tp32_distributed_q_a_norm",
        context=6,
        decode_rows=geometry.decode_rows,
        heads=geometry.heads,
        head_dim=geometry.head_dim,
        page_size=2,
        hidden_size=geometry.hidden_size,
        q_lora_rank=geometry.q_lora_rank,
        qkv_a_companion_rank=geometry.qkv_a_companion_rank,
        tensor_shards=geometry.legacy_tensor_shards,
        allow_cpu_bf16_collective_promotion=True,
    )
    assert contract["passed"], contract
    assert contract["distributed_collective_contract"][
        "collective_counts"
    ] == {"all-gather": 1, "all-reduce": 1}

    replicated = NamedSharding(mesh, P())
    output_sharded = NamedSharding(mesh, P(None, "legacy_model"))
    gspmd_function = jax.jit(
        partial(
            legacy_tp32_gspmd_fused_qkv_a_rms_norm,
            geometry=geometry,
        ),
        in_shardings=(
            replicated,
            output_sharded,
            output_sharded,
            replicated,
        ),
        out_shardings=LegacyTp32QaNormOutput(
            replicated,
            output_sharded,
        ),
    )
    gspmd_lowered = gspmd_function.lower(*arguments)
    gspmd_compiled = gspmd_lowered.compile()
    gspmd_actual = gspmd_compiled(*arguments)
    jax.block_until_ready(gspmd_actual)
    np.testing.assert_allclose(
        np.asarray(gspmd_actual.q_residual, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        rtol=0,
        atol=2**-7,
    )
    np.testing.assert_array_equal(
        np.asarray(gspmd_actual.qkv_a_companion_shard).view(np.uint16),
        np.asarray(logical_companion).view(np.uint16),
    )
    np.testing.assert_array_equal(
        np.asarray(gspmd_actual.q_residual).view(np.uint16),
        np.asarray(actual.q_residual).view(np.uint16),
    )
    gspmd_hlo = gspmd_compiled.as_text()
    gspmd_contract = validate_dsa_association_hlo(
        gspmd_hlo,
        phase="legacy_tp32_gspmd_q_a_norm",
        context=6,
        decode_rows=geometry.decode_rows,
        heads=geometry.heads,
        head_dim=geometry.head_dim,
        page_size=2,
        hidden_size=geometry.hidden_size,
        q_lora_rank=geometry.q_lora_rank,
        qkv_a_companion_rank=geometry.qkv_a_companion_rank,
        tensor_shards=geometry.legacy_tensor_shards,
        allow_cpu_bf16_collective_promotion=True,
    )
    assert gspmd_contract["passed"], (
        gspmd_contract,
        [
            line.strip()
            for line in gspmd_hlo.splitlines()
            if "all-reduce" in line or "all-gather" in line
        ],
    )
    assert gspmd_contract["distributed_collective_contract"][
        "collective_counts"
    ] == {"all-gather": 1, "all-reduce": 1}

    full = Layer0DsaProbeGeometry()
    full_arguments = (
        jax.device_put(
            jnp.zeros(
                (full.decode_rows, full.hidden_size), dtype=jnp.bfloat16
            ),
            NamedSharding(mesh, P()),
        ),
        jax.device_put(
            jnp.zeros(
                (
                    full.hidden_size,
                    full.q_lora_rank + full.qkv_a_companion_rank,
                ),
                dtype=jnp.float8_e4m3fn,
            ),
            NamedSharding(mesh, P(None, "legacy_model")),
        ),
        jax.device_put(
            jnp.ones(
                (
                    full.hidden_size // 128,
                    full.q_lora_rank + full.qkv_a_companion_rank,
                ),
                dtype=jnp.float32,
            ),
            NamedSharding(mesh, P(None, "legacy_model")),
        ),
        jax.device_put(
            jnp.ones((full.q_lora_rank,), dtype=jnp.bfloat16),
            NamedSharding(mesh, P()),
        ),
    )
    full_mapped = jax.shard_map(
        partial(
            legacy_tp32_fused_qkv_a_rms_norm,
            axis_name="legacy_model",
            geometry=full,
        ),
        mesh=mesh,
        in_specs=(P(), P(None, "legacy_model"), P(None, "legacy_model"), P()),
        out_specs=LegacyTp32QaNormOutput(
            P(),
            P(None, "legacy_model"),
        ),
        check_vma=False,
    )
    full_hlo = jax.jit(full_mapped).lower(*full_arguments).compile().as_text()
    full_contract = validate_dsa_association_hlo(
        full_hlo,
        phase="legacy_tp32_distributed_q_a_norm",
        allow_cpu_bf16_collective_promotion=True,
    )
    assert full_contract["passed"], full_contract
    assert full_contract["missing_shapes"] == []

    full_gspmd_function = jax.jit(
        partial(
            legacy_tp32_gspmd_fused_qkv_a_rms_norm,
            geometry=full,
        ),
        in_shardings=(
            replicated,
            output_sharded,
            output_sharded,
            replicated,
        ),
        out_shardings=LegacyTp32QaNormOutput(
            replicated,
            output_sharded,
        ),
    )
    full_gspmd_hlo = full_gspmd_function.lower(
        *full_arguments
    ).compile().as_text()
    full_gspmd_contract = validate_dsa_association_hlo(
        full_gspmd_hlo,
        phase="legacy_tp32_gspmd_q_a_norm",
        allow_cpu_bf16_collective_promotion=True,
    )
    assert full_gspmd_contract["passed"], full_gspmd_contract
    assert full_gspmd_contract["missing_shapes"] == []
    assert "0.00048828125" in full_gspmd_hlo


def test_distributed_q_a_norm_exactness_and_hlo_on_forced_32_cpu() -> None:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    code = (
        "from tests.greenfield.kernels.test_dsa_association import "
        "_run_forced_32_distributed_q_a_norm_case; "
        "_run_forced_32_distributed_q_a_norm_case()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_layer0_state_replacement_changes_only_query() -> None:
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
    rng = np.random.default_rng(31)
    q_residual = jnp.asarray(
        rng.normal(size=(3, 4)).astype(ml_dtypes.bfloat16)
    )
    keys = jnp.asarray(rng.normal(size=(6, 4)).astype(ml_dtypes.bfloat16))
    head_weights = jnp.asarray(rng.normal(size=(3, 2)), dtype=jnp.float32)
    companion = jnp.asarray(
        rng.normal(size=(3, 2)).astype(ml_dtypes.bfloat16)
    )
    wq_b = jnp.asarray(rng.normal(size=(8, 4)), dtype=jnp.float32)
    state = layer0_dsa_state_from_q_residual(
        q_residual,
        keys,
        head_weights,
        companion,
        wq_b,
        geometry=geometry,
    )
    assert state.query.shape == (3, 2, 4)
    np.testing.assert_array_equal(np.asarray(state.index_keys), np.asarray(keys))
    np.testing.assert_array_equal(
        np.asarray(state.head_weights), np.asarray(head_weights)
    )
    np.testing.assert_array_equal(
        np.asarray(state.qkv_a_companion), np.asarray(companion)
    )

    bf16_origin_keys = keys.at[0, 0].set(keys[0, 0] + jnp.bfloat16(0.25))
    bf16_wk_state = layer0_dsa_state_from_q_residual(
        q_residual,
        bf16_origin_keys,
        head_weights,
        companion,
        wq_b,
        geometry=geometry,
    )
    np.testing.assert_array_equal(
        np.asarray(bf16_wk_state.query), np.asarray(state.query)
    )
    assert not np.array_equal(
        np.asarray(bf16_wk_state.index_keys), np.asarray(state.index_keys)
    )
    np.testing.assert_array_equal(
        np.asarray(bf16_wk_state.head_weights), np.asarray(state.head_weights)
    )
    np.testing.assert_array_equal(
        np.asarray(bf16_wk_state.qkv_a_companion),
        np.asarray(state.qkv_a_companion),
    )


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


def test_legacy_dcp_xla_geometry_pins_sealed_8k_shapes() -> None:
    geometry = LegacyDcpXlaScoreGeometry()
    assert geometry.global_page_size == 4096
    assert geometry.owned_block_count == 3
    assert geometry.local_score_width == 1536
    with pytest.raises(ValueError, match="cannot hold"):
        LegacyDcpXlaScoreGeometry(cache_pages=2)


def test_legacy_local_dcp_xla_scores_reconstruct_direct_row() -> None:
    geometry = LegacyDcpXlaScoreGeometry(
        dcp_size=2,
        local_page_size=2,
        max_model_len=9,
        cache_pages=4,
        decode_rows=3,
        heads=2,
        head_dim=4,
    )
    rng = np.random.default_rng(37)
    query = rng.normal(size=(3, 2, 4)).astype(np.float32)
    keys = rng.normal(size=(7, 4)).astype(ml_dtypes.bfloat16)
    head_weights = rng.normal(size=(3, 2)).astype(np.float32)
    stitched = np.full((7,), -np.inf, dtype=np.float32)

    for shard in range(geometry.dcp_size):
        packed = legacy_local_dcp_score_inputs(
            jnp.asarray(keys), shard, geometry=geometry
        )
        assert packed.cache.shape == (4, 2, 4)
        assert packed.block_tables.shape == (3, 3)
        assert packed.kv_lens.shape == (3,)
        local = np.asarray(
            legacy_local_dcp_xla_scores(
                jnp.asarray(query),
                packed.cache,
                jnp.asarray(head_weights),
                packed.block_tables,
                packed.kv_lens,
                geometry=geometry,
            )[0],
            dtype=np.float32,
        )
        positions = np.asarray(packed.local_positions)
        valid = positions < len(keys)
        stitched[positions[valid]] = local[valid]

    per_head = np.maximum(
        np.einsum("hd,sd->hs", query[0], keys.astype(np.float32)) * 0.5,
        0.0,
    )
    expected = np.einsum("h,hs->s", head_weights[0], per_head)
    np.testing.assert_allclose(stitched, expected, rtol=1e-6, atol=1e-6)


def test_legacy_local_dcp_xla_scores_reject_contract_drift() -> None:
    geometry = LegacyDcpXlaScoreGeometry(
        dcp_size=2,
        local_page_size=2,
        max_model_len=8,
        cache_pages=3,
        decode_rows=3,
        heads=2,
        head_dim=4,
    )
    query = jnp.ones((3, 2, 4), dtype=jnp.float32)
    keys = jnp.ones((7, 4), dtype=jnp.bfloat16)
    weights = jnp.ones((3, 2), dtype=jnp.float32)
    packed = legacy_local_dcp_score_inputs(keys, 0, geometry=geometry)

    with pytest.raises(ValueError, match="out of range"):
        legacy_local_dcp_score_inputs(keys, 2, geometry=geometry)
    with pytest.raises(ValueError, match="must remain BF16"):
        legacy_local_dcp_score_inputs(
            keys.astype(jnp.float32), 0, geometry=geometry
        )
    with pytest.raises(ValueError, match="must remain FP32"):
        legacy_local_dcp_xla_scores(
            query.astype(jnp.bfloat16),
            packed.cache,
            weights,
            packed.block_tables,
            packed.kv_lens,
            geometry=geometry,
        )


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
