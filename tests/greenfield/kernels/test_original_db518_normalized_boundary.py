from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    layer0_prompt_index_key_from_normalized_boundary_chunk,
    layer0_prompt_index_key_gather_cache_states_chunk,
    layer0_prompt_normalized_hidden_boundary_chunk,
)
from glm_tpu.greenfield.validation.original_db518_normalized_boundary_hlo import (
    require_completed_normalization_boundary_hlo,
    require_normalized_key_control_boundary_hlo,
)


GEOMETRY = Layer0DsaProbeGeometry(
    prompt_tokens=129,
    prompt_chunk=128,
    decode_rows=2,
    hidden_size=8,
    q_lora_rank=8,
    qkv_a_companion_rank=4,
    legacy_tensor_shards=2,
    heads=2,
    head_dim=4,
    rotary_dim=2,
)


def _inputs():
    unique = jnp.asarray(
        np.linspace(-1.0, 1.0, 5 * 8, dtype=np.float32).reshape(5, 8),
        dtype=jnp.bfloat16,
    )
    rows = jnp.asarray(np.arange(128) % 5, dtype=jnp.int32)
    positions = jnp.arange(128, dtype=jnp.int32)
    input_norm = jnp.asarray(
        np.linspace(0.75, 1.25, 8, dtype=np.float32), dtype=jnp.bfloat16
    )
    wk = jnp.asarray(
        np.linspace(-0.25, 0.25, 4 * 8, dtype=np.float32).reshape(4, 8),
        dtype=jnp.float32,
    )
    key_weight = jnp.asarray([1.0, 0.75, 1.25, 0.5], dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.125, -0.125, 0.25], dtype=jnp.bfloat16)
    return unique, rows, positions, input_norm, wk, key_weight, key_bias


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def test_completed_boundary_composes_to_original_producer_keys():
    unique, rows, positions, input_norm, wk, key_weight, key_bias = _inputs()
    normalized = layer0_prompt_normalized_hidden_boundary_chunk(
        unique, rows, input_norm, geometry=GEOMETRY
    )
    boundary_keys = layer0_prompt_index_key_from_normalized_boundary_chunk(
        normalized,
        positions,
        wk,
        key_weight,
        key_bias,
        geometry=GEOMETRY,
    )
    states = layer0_prompt_index_key_gather_cache_states_chunk(
        jnp.zeros((1, 1, 128, 4), dtype=jnp.bfloat16),
        jnp.asarray([0], dtype=jnp.int32),
        unique,
        rows,
        positions,
        input_norm,
        wk,
        key_weight,
        key_bias,
        geometry=GEOMETRY,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
        projection_weight_mode="adapted_fp32",
        projection_mapping_mode="physical_m64_projection_keynorm_lax_map",
    )
    assert normalized.shape == (128, 8)
    assert normalized.dtype == jnp.bfloat16
    assert boundary_keys.shape == (128, 4)
    assert boundary_keys.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(boundary_keys).view(np.uint16),
        np.asarray(states.index_cache).reshape(128, 4).view(np.uint16),
    )


def test_boundary_rejects_dtype_and_geometry_drift():
    unique, rows, positions, input_norm, wk, key_weight, key_bias = _inputs()
    with pytest.raises(ValueError, match="values must be BF16"):
        layer0_prompt_normalized_hidden_boundary_chunk(
            unique.astype(jnp.float32), rows, input_norm, geometry=GEOMETRY
        )
    normalized = layer0_prompt_normalized_hidden_boundary_chunk(
        unique, rows, input_norm, geometry=GEOMETRY
    )
    with pytest.raises(ValueError, match="positions must be int32"):
        layer0_prompt_index_key_from_normalized_boundary_chunk(
            normalized,
            positions.astype(jnp.int16),
            wk,
            key_weight,
            key_bias,
            geometry=GEOMETRY,
        )
    with pytest.raises(ValueError, match="wk must remain FP32"):
        layer0_prompt_index_key_from_normalized_boundary_chunk(
            normalized,
            positions,
            wk.astype(jnp.bfloat16),
            key_weight,
            key_bias,
            geometry=GEOMETRY,
        )


@pytest.fixture(scope="module")
def real_boundary_hlo():
    geometry = Layer0DsaProbeGeometry()
    normalizer = partial(
        layer0_prompt_normalized_hidden_boundary_chunk, geometry=geometry
    )
    normalizer_lowered = jax.jit(normalizer).lower(
        jax.ShapeDtypeStruct((37, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048,), jnp.int32),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16),
    )
    normalizer_optimized = normalizer_lowered.compile().as_text()
    key_control = partial(
        layer0_prompt_index_key_from_normalized_boundary_chunk,
        geometry=geometry,
    )
    key_lowered = jax.jit(key_control).lower(
        jax.ShapeDtypeStruct((2048, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048,), jnp.int32),
        jax.ShapeDtypeStruct((128, 6144), jnp.float32),
        jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        jax.ShapeDtypeStruct((128,), jnp.bfloat16),
    )
    key_optimized = key_lowered.compile().as_text()
    return (
        normalizer_optimized,
        normalizer_lowered.as_text(),
        key_optimized,
        key_lowered.as_text(),
    )


def test_real_boundary_hlo_is_admitted(real_boundary_hlo):
    normalizer_optimized, normalizer_stable, key_optimized, key_stable = (
        real_boundary_hlo
    )
    assert require_completed_normalization_boundary_hlo(
        normalizer_optimized, normalizer_stable
    )["passed"]
    assert require_normalized_key_control_boundary_hlo(
        key_optimized, key_stable
    )["passed"]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda optimized, stable: (
            optimized.replace("bf16[37,6144]", "bf16[38,6144]"), stable
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.gather", "stablehlo.add"),
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.rsqrt", "stablehlo.sqrt"),
        ),
        lambda optimized, stable: (optimized, stable + "\nstablehlo.all_reduce"),
        lambda optimized, stable: (
            optimized.replace(
                "fusion(%input_norm_weight.1,",
                "fusion(%gather_convert_fusion,",
                1,
            ),
            stable,
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%square.0 = f32[2048,6144]{1,0} multiply(",
                "%square.0 = f32[2048,6144]{1,0} add(",
            ),
            stable,
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(stable, "%2 = chlo.square %1", "%2 = chlo.square %11"),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%15 = stablehlo.multiply %12, %14",
                "%15 = stablehlo.multiply %12, %12",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(stable, "dense<6.144000e+03>", "dense<6.143000e+03>"),
        ),
    ),
)
def test_normalization_boundary_hlo_rejects_hostile_mutations(
    real_boundary_hlo, mutation
):
    optimized, stable, _, _ = real_boundary_hlo
    with pytest.raises(RuntimeError):
        require_completed_normalization_boundary_hlo(*mutation(optimized, stable))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda optimized, stable: (
            optimized.replace("bf16[2048,6144]", "bf16[2047,6144]"), stable
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.while", "stablehlo.case"),
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.dot_general", "stablehlo.multiply"),
        ),
        lambda optimized, stable: (optimized, stable + "\nstablehlo.rsqrt"),
        lambda optimized, stable: (optimized, stable + "\nall-reduce("),
        lambda optimized, stable: (
            optimized.replace(
                "fusion(%key_norm_bias.1)",
                "fusion(%key_norm_weight.1)",
                1,
            ),
            stable,
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%dot_general.4 = f32[64,128]{1,0} dot(",
                "%dot_general.4 = f32[64,128]{1,0} add(",
            ),
            stable,
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%1 = stablehlo.dot_general %0, %arg0",
                "%1 = stablehlo.dot_general %0, %0",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%25 = stablehlo.add %21, %24",
                "%25 = stablehlo.add %21, %20",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%27 = stablehlo.subtract %25, %26",
                "%27 = stablehlo.subtract %25, %25",
            ),
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%concatenate.1 = f32[2048,128]{1,0} concatenate(",
                "%concatenate.1 = f32[2048,128]{1,0} add(",
            ),
            stable,
        ),
    ),
)
def test_key_control_boundary_hlo_rejects_hostile_mutations(
    real_boundary_hlo, mutation
):
    _, _, optimized, stable = real_boundary_hlo
    with pytest.raises(RuntimeError):
        require_normalized_key_control_boundary_hlo(*mutation(optimized, stable))
