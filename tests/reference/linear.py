"""Linear algebra of the reference: FP8 block dequantization, projections, SwiGLU.

Numerical contract (the engine's, not Hugging Face's eager BF16 one):

* FP8 E4M3FN checkpoint tensors are dequantized once, block by block:
  ``bf16(f32(e4m3(bits)) * scale_inv[block])`` with 128x128 blocks over the
  final ``[out, in]`` dimensions. This is exactly the tile decode of the FP8
  kernels and the load-time BF16 tables of production.
* A projection multiplies BF16-valued operands exactly in FP32, accumulates in
  FP32 and rounds once to its result dtype (BF16 unless stated otherwise).
  Operands are widened to FP32 before the dot, so no backend-specific BF16 dot
  path is involved; the products of two BF16 values are exact in FP32.
* Elementwise BF16 expressions (SiLU gating) round at each operation, because
  the reference runs op by op (no ``jit``; only the exact elementwise
  dequantization below is compiled, once per shape).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from glm_tpu.optimized.reference.fp8 import dequantize_fp8_bits_block_weight

FP8_BLOCK = (128, 128)
_HIGHEST = lax.Precision.HIGHEST


@partial(jax.jit, static_argnames=("block_shape",))
def _dequantize(
    bits: jax.Array, scale_inv: jax.Array, *, block_shape: tuple[int, int]
) -> jax.Array:
    return dequantize_fp8_bits_block_weight(
        bits, scale_inv, block_shape=block_shape, output_dtype=jnp.bfloat16
    )


def dequantize(
    bits: Any, scale_inv: Any, *, block_shape: tuple[int, int] = FP8_BLOCK
) -> jax.Array:
    """Dequantize raw E4M3FN bits (``uint8``) with FP32 block scales to BF16.

    One compiled program per shape (a table lookup and one FP32 product per element,
    no accumulation), so compiling changes no value.
    """
    return _dequantize(
        jnp.asarray(bits), jnp.asarray(scale_inv), block_shape=tuple(block_shape)
    )


def project(
    x: jax.Array, weight_out_in: jax.Array, *, dtype: Any = jnp.bfloat16
) -> jax.Array:
    """``x @ weight.T`` (checkpoint ``[out, in]`` weight), FP32 accumulation, one rounding."""
    result = lax.dot_general(
        x.astype(jnp.float32),
        weight_out_in.astype(jnp.float32),
        dimension_numbers=(((x.ndim - 1,), (1,)), ((), ())),
        precision=_HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return result.astype(dtype)


def einsum(spec: str, *operands: jax.Array, dtype: Any = jnp.bfloat16) -> jax.Array:
    """``jnp.einsum`` with the projection contract (FP32 operands and accumulation)."""
    result = jnp.einsum(
        spec,
        *(operand.astype(jnp.float32) for operand in operands),
        precision=_HIGHEST,
        preferred_element_type=jnp.float32,
    )
    return result.astype(dtype)


def swiglu_activation(gate: jax.Array, up: jax.Array) -> jax.Array:
    """``silu(gate) * up`` in BF16: ``gate * sigmoid(gate)`` then ``* up``."""
    return (gate * jax.nn.sigmoid(gate) * up).astype(jnp.bfloat16)


def swiglu(x: jax.Array, gate: jax.Array, up: jax.Array, down: jax.Array) -> jax.Array:
    """Dense GLM MLP: BF16 gate/up projections, BF16 SwiGLU, BF16 down projection."""
    return project(swiglu_activation(project(x, gate), project(x, up)), down)


def embed(token_ids: jax.Array, table: jax.Array) -> jax.Array:
    """Embedding rows ``[rows, hidden]`` (BF16, exact copies)."""
    return jnp.take(table, token_ids, axis=0)


def greedy_token(logits: jax.Array) -> jax.Array:
    """Greedy id per row: the lowest vocabulary id among tied maximal logits."""
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)
