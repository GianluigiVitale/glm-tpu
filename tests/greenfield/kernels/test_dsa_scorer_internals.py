from __future__ import annotations

import jax.numpy as jnp

from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    layer0_dsa_scorer_internals,
)


def test_layer0_scorer_internal_shapes_and_dtypes() -> None:
    geometry = Layer0DsaProbeGeometry(
        prompt_tokens=5,
        prompt_chunk=4,
        decode_rows=2,
        hidden_size=8,
        q_lora_rank=4,
        qkv_a_companion_rank=2,
        legacy_tensor_shards=2,
        heads=2,
        head_dim=4,
        rotary_dim=2,
    )
    result = layer0_dsa_scorer_internals(
        jnp.arange(16, dtype=jnp.float32).reshape(2, 8).astype(jnp.bfloat16),
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4).astype(jnp.bfloat16),
        jnp.asarray([5, 0], dtype=jnp.int32),
        jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 16,
        jnp.arange(32, dtype=jnp.float32).reshape(4, 8) / 32,
        jnp.ones((4,), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 16,
        geometry=geometry,
    )
    assert result.query.shape == (2, 2, 4)
    assert result.head_weights.shape == (2, 2)
    assert result.current_keys.shape == (2, 4)
    assert all(value.dtype == jnp.float32 for value in result)
