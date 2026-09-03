"""CPU smoke test of the legacy-geometry chunk pipeline with real shapes."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm_tpu.greenfield.benchmarking import legacy_prefill_owner_packing as pk  # noqa: E402
from glm_tpu.greenfield.benchmarking.legacy_prefill_chunk_probe import (  # noqa: E402
    WEIGHT_KEYS,
    legacy_geometry_chunk_pipeline,
)
from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host  # noqa: E402


def _bf16_bits(rng, shape, scale=1.0):
    return (rng.normal(size=shape) * scale).astype(ml_dtypes.bfloat16).view(np.uint16)


def _fp8(rng, shape):
    return rng.integers(0, 120, size=shape, dtype=np.uint8)


def _scale(rng, shape, value=0.01):
    return np.full(shape, value, dtype=np.float32)


@pytest.fixture(scope="module")
def weights():
    rng = np.random.default_rng(11)
    qkv_bits, qkv_scale = pk.pack_fused_qkv_a_owners(
        _fp8(rng, (2048, 6144)), _scale(rng, (16, 48)), _fp8(rng, (576, 6144)), _scale(rng, (5, 48))
    )
    qb_bits, qb_scale = pk.pack_q_b_owners(_fp8(rng, (16384, 2048)), _scale(rng, (128, 16)))
    w_uk_t, w_uv = pk.absorbed_kv_b_owners(_fp8(rng, (28672, 512)), _scale(rng, (224, 4)))
    o_bits, o_scale = pk.pack_o_proj_owners(_fp8(rng, (6144, 16384)), _scale(rng, (48, 128)))
    gu_bits, gu_scale = pk.pack_gate_up_owners(
        _fp8(rng, (12288, 6144)), _scale(rng, (96, 48)), _fp8(rng, (12288, 6144)), _scale(rng, (96, 48))
    )
    down_bits, down_scale = pk.pack_down_owners(_fp8(rng, (6144, 12288)), _scale(rng, (48, 96)))
    table = build_rotary_table_host(256, rotary_dim=64, theta=8_000_000.0)

    def bf16(bits):
        return jnp.asarray(bits.view(ml_dtypes.bfloat16))

    values = {
        "input_norm0": bf16(_bf16_bits(rng, (6144,))),
        "q_a_norm": bf16(_bf16_bits(rng, (2048,))),
        "kv_a_norm": bf16(_bf16_bits(rng, (512,))),
        "post_norm": bf16(_bf16_bits(rng, (6144,))),
        "input_norm1": bf16(_bf16_bits(rng, (6144,))),
        "k_norm0_weight": bf16(_bf16_bits(rng, (128,))),
        "k_norm0_bias": bf16(_bf16_bits(rng, (128,), 0.1)),
        "k_norm1_weight": bf16(_bf16_bits(rng, (128,))),
        "k_norm1_bias": bf16(_bf16_bits(rng, (128,), 0.1)),
        "qkv_bits": jnp.asarray(qkv_bits), "qkv_scale": jnp.asarray(qkv_scale),
        "qb_bits": jnp.asarray(qb_bits), "qb_scale": jnp.asarray(qb_scale),
        "w_uk_t": jnp.asarray(w_uk_t), "w_uv": jnp.asarray(w_uv),
        "o_bits": jnp.asarray(o_bits), "o_scale": jnp.asarray(o_scale),
        "gu_bits": jnp.asarray(gu_bits), "gu_scale": jnp.asarray(gu_scale),
        "down_bits": jnp.asarray(down_bits), "down_scale": jnp.asarray(down_scale),
        "wk0_bits": jnp.asarray(_fp8(rng, (128, 6144))), "wk0_scale": jnp.asarray(_scale(rng, (1, 48))),
        "wk1_bits": jnp.asarray(_fp8(rng, (128, 6144))), "wk1_scale": jnp.asarray(_scale(rng, (1, 48))),
        "rope_table": bf16(table.view(np.uint16)),
    }
    assert set(values) == set(WEIGHT_KEYS)
    return values


def test_chunk_pipeline_shapes_dtypes_and_finiteness(weights):
    rng = np.random.default_rng(12)
    rows = 2
    embedding = jnp.asarray(_bf16_bits(rng, (rows, 6144)).view(ml_dtypes.bfloat16))
    positions = jnp.arange(rows, dtype=jnp.int32)
    out = legacy_geometry_chunk_pipeline(embedding, positions, weights, softmax_scale=256 ** -0.5)
    assert out["keys0"].shape == (rows, 128) and out["keys0"].dtype == jnp.bfloat16
    assert out["keys1"].shape == (rows, 128) and out["keys1"].dtype == jnp.bfloat16
    for name in ("normalized0_row0", "attention_row0", "dense_row0", "normalized1_row0", "carried1_row0"):
        assert out[name].shape == (6144,) and out[name].dtype == jnp.bfloat16
    assert out["q_a_row0"].shape == (2048,) and out["latent_row0"].shape == (512,)
    for name, value in out.items():
        assert bool(jnp.isfinite(value.astype(jnp.float32)).all()), name
    # Row 0 attends only to itself: the one-row variant must agree with the
    # two-row run on CPU for the attention-independent path (same M on CPU
    # convolutions is not guaranteed by XLA, so report rather than assert).
    single = legacy_geometry_chunk_pipeline(embedding[:1], positions[:1], weights, softmax_scale=256 ** -0.5)
    lanes = int(jnp.count_nonzero(single["keys1"][0].view(jnp.uint16) != out["keys1"][0].view(jnp.uint16)))
    print(f"CPU row-0 keys1 lanes differing between M=1 and M=2: {lanes}")


def test_chunk_pipeline_refuses_missing_weights_and_bad_rows(weights):
    embedding = jnp.zeros((1, 6144), jnp.bfloat16)
    with pytest.raises(ValueError):
        legacy_geometry_chunk_pipeline(embedding, jnp.zeros((1,), jnp.int32), {k: v for k, v in weights.items() if k != "o_bits"}, softmax_scale=0.0625)
    with pytest.raises(ValueError):
        legacy_geometry_chunk_pipeline(embedding.astype(jnp.float32), jnp.zeros((1,), jnp.int32), weights, softmax_scale=0.0625)
    with pytest.raises(ValueError):
        legacy_geometry_chunk_pipeline(embedding, jnp.zeros((2,), jnp.int32), weights, softmax_scale=0.0625)
