from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas.rmsnorm import (
    fused_add_rms_norm_m1,
    source_fused_output_m1_feature_tiled_m8,
    source_fused_output_m1_m8_scratch,
    weighted_output_m1_m8_scratch,
)
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm


def test_fused_add_rms_norm_m1_interpret_matches_reference() -> None:
    hidden = jnp.linspace(-2.0, 3.0, 128, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    residual = jnp.linspace(0.5, -1.5, 128, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    weight = jnp.linspace(0.25, 1.25, 128, dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    expected_output, expected_carried = fused_add_rms_norm(
        hidden, residual, weight, epsilon=1e-5
    )
    output, carried = fused_add_rms_norm_m1(
        hidden, residual, weight, epsilon=1e-5, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected_output))
    np.testing.assert_array_equal(
        np.asarray(carried), np.asarray(expected_carried)
    )


@pytest.mark.parametrize(
    ("hidden_shape", "residual_shape", "weight_shape"),
    [
        ((2, 128), (2, 128), (128,)),
        ((1, 127), (1, 127), (127,)),
        ((1, 128), (1, 64), (128,)),
        ((1, 128), (1, 128), (64,)),
    ],
)
def test_fused_add_rms_norm_m1_refuses_geometry_drift(
    hidden_shape: tuple[int, ...],
    residual_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            jnp.ones(hidden_shape, jnp.bfloat16),
            jnp.ones(residual_shape, jnp.bfloat16),
            jnp.ones(weight_shape, jnp.bfloat16),
            epsilon=1e-5,
            interpret=True,
        )


def test_fused_add_rms_norm_m1_refuses_dtype_and_epsilon_drift() -> None:
    hidden = jnp.ones((1, 128), jnp.bfloat16)
    residual = jnp.ones((1, 128), jnp.bfloat16)
    weight = jnp.ones((128,), jnp.bfloat16)
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            hidden.astype(jnp.float32),
            residual,
            weight,
            epsilon=1e-5,
            interpret=True,
        )
    with pytest.raises(ValueError):
        fused_add_rms_norm_m1(
            hidden,
            residual,
            weight,
            epsilon=0.0,
            interpret=True,
        )


def test_weighted_output_m1_m8_scratch_interpret_matches_reference() -> None:
    hidden = jnp.linspace(-2.0, 3.0, 256, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    residual = jnp.linspace(0.5, -1.5, 256, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    inverse = jnp.asarray([0.9375], dtype=jnp.float32)
    weight = jnp.linspace(0.25, 1.25, 256, dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    expected = (
        (
            (hidden.astype(jnp.float32) + residual.astype(jnp.float32))
            * inverse[:, None]
        ).astype(jnp.bfloat16)
        * weight[None, :]
    ).astype(jnp.bfloat16)
    observed = weighted_output_m1_m8_scratch(
        hidden,
        residual,
        inverse,
        weight,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_weighted_output_m1_m8_scratch_refuses_contract_drift() -> None:
    hidden = jnp.ones((1, 128), jnp.bfloat16)
    residual = jnp.ones((1, 128), jnp.bfloat16)
    inverse = jnp.ones((1,), jnp.float32)
    weight = jnp.ones((128,), jnp.bfloat16)
    for values in (
        (jnp.ones((2, 128), jnp.bfloat16), residual, inverse, weight),
        (hidden, residual, jnp.ones((1,), jnp.bfloat16), weight),
        (hidden, residual, inverse, jnp.ones((64,), jnp.bfloat16)),
    ):
        with pytest.raises(ValueError):
            weighted_output_m1_m8_scratch(*values, interpret=True)


def test_weighted_output_m1_m8_scratch_jaxpr_has_true_m1_io() -> None:
    traced = str(
        jax.make_jaxpr(weighted_output_m1_m8_scratch)(
            jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        )
    )
    assert "greenfield_weighted_output_m1_m8_scratch_h128" in traced
    assert "Ref<vmem>{bf16[8,128]}" in traced
    assert "out_avals=(ShapedArray(bfloat16[1,128]),)" in traced
    rank_two_block = (
        "BlockMapping(block_shape=(Blocked(block_size=1), "
        "Blocked(block_size=128)))"
    )
    assert traced.count(rank_two_block) == 4
    assert "BlockMapping(block_shape=(Blocked(block_size=128)))" not in traced


def test_source_fused_output_m1_m8_scratch_matches_exact_source_order() -> None:
    dense = jnp.linspace(-1.0, 2.0, 256, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    attention = jnp.linspace(0.5, -0.5, 256, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    embedding = jnp.linspace(-0.25, 0.75, 256, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    validity = jnp.asarray([True], dtype=jnp.bool_)
    inverse = jnp.asarray([0.9375], dtype=jnp.float32)
    weight = jnp.linspace(0.25, 1.25, 256, dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    carried = (
        attention.astype(jnp.float32) + embedding.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = (
        (
            (dense.astype(jnp.float32) + carried.astype(jnp.float32))
            * inverse[:, None]
        ).astype(jnp.bfloat16)
        * weight[None, :]
    ).astype(jnp.bfloat16)
    observed = source_fused_output_m1_m8_scratch(
        dense,
        attention,
        embedding,
        validity,
        inverse,
        weight,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))


def test_source_fused_output_m1_m8_scratch_has_true_m1_io() -> None:
    traced = str(
        jax.make_jaxpr(source_fused_output_m1_m8_scratch)(
            jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.bool_),
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        )
    )
    assert "greenfield_source_fused_output_m1_m8_scratch_h128" in traced
    assert "Ref<vmem>{bf16[8,128]}" in traced
    assert "out_avals=(ShapedArray(bfloat16[1,128]),)" in traced
    assert "bf16[32,128]" not in traced


def test_source_fused_output_m1_feature_tiled_m8_is_all_live() -> None:
    width = 1024
    dense = jnp.linspace(-1.0, 2.0, width, dtype=jnp.float32)[None].astype(
        jnp.bfloat16
    )
    attention = jnp.linspace(
        0.5, -0.5, width, dtype=jnp.float32
    )[None].astype(jnp.bfloat16)
    embedding = jnp.linspace(
        -0.25, 0.75, width, dtype=jnp.float32
    )[None].astype(jnp.bfloat16)
    validity = jnp.asarray([True], dtype=jnp.bool_)
    inverse = jnp.asarray([0.9375], dtype=jnp.float32)
    weight = jnp.linspace(0.25, 1.25, width, dtype=jnp.float32).astype(
        jnp.bfloat16
    )
    carried = (
        attention.astype(jnp.float32) + embedding.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = (
        (
            (dense.astype(jnp.float32) + carried.astype(jnp.float32))
            * inverse[:, None]
        ).astype(jnp.bfloat16)
        * weight[None, :]
    ).astype(jnp.bfloat16)
    observed = source_fused_output_m1_feature_tiled_m8(
        dense,
        attention,
        embedding,
        validity,
        inverse,
        weight,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))
    assert observed.shape == (1, width)


def test_source_fused_output_m1_feature_tiled_m8_jaxpr() -> None:
    width = 1024
    traced = str(
        jax.make_jaxpr(source_fused_output_m1_feature_tiled_m8)(
            jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.bool_),
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct((width,), jnp.bfloat16),
        )
    )
    assert "greenfield_source_fused_output_m1_feature_tiled_m8_h1024" in traced
    assert traced.count(
        "Blocked(block_size=8), Blocked(block_size=128)"
    ) == 5
    assert "out_avals=(ShapedArray(bfloat16[8,128]),)" in traced
    assert "bf16[1,1024] = reshape" in traced
    assert "Ref<vmem>{bf16[8,128]}" not in traced
    assert "bf16[32,128]" not in traced


def test_source_fused_output_m1_feature_tiled_m8_refuses_dead_rows() -> None:
    width = 1024
    valid = (
        jnp.ones((1, width), jnp.bfloat16),
        jnp.ones((1, width), jnp.bfloat16),
        jnp.ones((1, width), jnp.bfloat16),
        jnp.ones((1,), jnp.bool_),
        jnp.ones((1,), jnp.float32),
        jnp.ones((width,), jnp.bfloat16),
    )
    mutations = (
        (jnp.ones((8, 128), jnp.bfloat16), *valid[1:]),
        (*valid[:5], jnp.ones((512,), jnp.bfloat16)),
        tuple(
            jnp.ones((1, 128), jnp.bfloat16) if index < 3 else value
            for index, value in enumerate(valid)
        ),
    )
    for values in mutations:
        with pytest.raises(ValueError):
            source_fused_output_m1_feature_tiled_m8(
                *values, interpret=True
            )
