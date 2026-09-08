"""Small real weight-schema fixture for layer-major composition, CPU only."""

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding

from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
)
from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderWeights,
    Ws32LayerWeights,
    ws32_decoder_weight_specs,
)
from glm_tpu.greenfield.types import ModelGeometry


def fixture(mesh):
    raw = json.loads(Path("configs/glm-5.2-fp8-config.json").read_text())
    geometry = replace(
        ModelGeometry.from_hf_config(raw),
        num_layers=8,
        hidden_size=512,
        attention_heads=16,
        kv_heads=16,
        q_lora_rank=128,
        num_routed_experts=64,
        moe_intermediate_size=128,
        dense_intermediate_size=1024,
        vocab_size=256,
        dsa_top_k=128,
        mlp_layer_types=("dense",) * 3 + ("sparse",) * 5,
        indexer_types=(
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
            "shared",
        ),
    )
    config = Ws32DecoderConfig(
        geometry, 1536, sparse_segment_block=128, host_main_rope_table=True
    )
    rng = np.random.default_rng(921)

    def bf(shape):
        return jnp.asarray(rng.normal(0, 0.1, shape), jnp.bfloat16)

    def bits(shape):
        return jnp.asarray(
            np.asarray(rng.normal(0, 0.02, shape), ml_dtypes.float8_e4m3fn).view(
                np.uint8
            )
        )

    def scale(shape):
        return jnp.asarray(rng.uniform(0.5, 1.5, shape), jnp.float32)

    layers = []
    for i in range(8):
        q = Ws32QkvAWeights(
            jnp.ones(512, jnp.bfloat16),
            bits((128, 512)),
            scale((1, 4)),
            jnp.ones(128, jnp.bfloat16),
            bits((576, 512)),
            scale((5, 4)),
            jnp.ones(512, jnp.bfloat16),
        )
        a = Ws32AttentionWeights(
            bits((4096, 128)),
            scale((32, 1)),
            bits((7168, 512)),
            scale((56, 4)),
            bits((512, 4096)),
            scale((4, 32)),
        )
        d = (
            Ws32DsaWeights(
                bits((4096, 128)),
                scale((32, 1)),
                bits((128, 512)),
                scale((1, 4)),
                jnp.ones(128, jnp.bfloat16),
                bf((128,)),
                bf((32, 512)),
            )
            if i in (0, 1, 2, 6)
            else None
        )
        dense = (
            Ws32DenseWeights(
                bits((1024, 512)),
                scale((8, 4)),
                bits((1024, 512)),
                scale((8, 4)),
                bits((512, 1024)),
                scale((4, 8)),
            )
            if i < 3
            else None
        )
        moe = (
            Ws32MoeWeights(
                bf((64, 512)),
                jnp.zeros(64, jnp.float32),
                bits((64, 128, 512)),
                scale((64, 1, 4)),
                bits((64, 128, 512)),
                scale((64, 1, 4)),
                bits((64, 512, 128)),
                scale((64, 4, 1)),
                bits((128, 512)),
                scale((1, 4)),
                bits((128, 512)),
                scale((1, 4)),
                bits((512, 128)),
                scale((4, 1)),
            )
            if i >= 3
            else None
        )
        layers.append(
            Ws32LayerWeights(q, a, d, jnp.ones(512, jnp.bfloat16), dense, moe)
        )
    weights = Ws32DecoderWeights(
        bf((256, 512)), tuple(layers), jnp.ones(512, jnp.bfloat16), bf((256, 512))
    )
    weights = jax.tree.map(
        lambda v, s: jax.device_put(v, NamedSharding(mesh, s)),
        weights,
        ws32_decoder_weight_specs(config),
    )
    wk = tuple(bf((128, 512)).astype(jnp.float32) for _ in range(4))
    return config, weights, wk
