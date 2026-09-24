"""Owner-local partial attention and LSE merge used by retained prefill only.

This is a numerical boundary, not a bitwise replacement: owner-local
softmax maxima change BF16 probability rounding and partial outputs round once
before the FP32 LSE merge. Empty owners contribute zero, including an entirely
empty selection. Decode retains the frozen selected-KV exchange.
"""

import jax
from jax import lax
import jax.numpy as jnp

from glm_tpu.kernels.names import KERNEL_NAMES


def gathered_partial_attention(queries, cache, counts, *, contract, config, interpret=False):
    """Contiguous local tiles and online softmax, returning a partial and LSE.

    Gather sanitizes padding before the PV dot. Empty owners return exact zero
    and -inf. Each row skips tiles beyond its owned count. The row grid also
    serves the <=32-row prefill tiles without a selected-KV collective.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows, heads, _ = queries.shape
    latent, width = contract.kv_lora_rank, contract.packed_cache_width
    if (not 1 <= rows <= 32 or heads != contract.num_heads
            or cache.shape != (rows, contract.top_k, width)
            or counts.shape != (rows,) or counts.dtype != jnp.int32):
        raise ValueError("partial attention row/cache/count geometry drifted")
    block = min(config.segment_block, contract.top_k)
    if contract.top_k % block:
        raise ValueError("partial attention block must divide top_k")
    if queries.shape[-1] != latent + contract.qk_rope_head_dim:
        raise ValueError("partial attention packed query width drifted")
    queries = jnp.pad(queries, ((0, 0), (0, 0), (0, width-queries.shape[-1])))
    blocks = contract.top_k // block
    cache = cache.reshape(rows, blocks, block, width)

    def kernel(count, q, kv, out, lse, maximum, denominator, accumulator):
        row, tile = pl.program_id(0), pl.program_id(1)
        @pl.when(tile == 0)
        def init():
            maximum[...] = jnp.full(maximum.shape, -jnp.inf, jnp.float32)
            denominator[...] = jnp.zeros(denominator.shape, jnp.float32)
            accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
            out[0] = jnp.zeros(out.shape[1:], jnp.bfloat16)
            lse[0] = jnp.full(lse.shape[1:], -jnp.inf, jnp.float32)

        @pl.when(tile * block < count[row])
        def attend():
            scores = lax.dot_general(q[0], kv[0, 0], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32) * jnp.float32(contract.softmax_scale)
            live = jnp.arange(block)[None, :] + tile * block < count[row]
            scores = jnp.where(live, scores, -jnp.inf)
            m = jnp.maximum(maximum[...], jnp.max(scores, axis=1, keepdims=True))
            correction = jnp.exp(maximum[...] - m)
            probs = jnp.exp(scores-m)
            den = denominator[...] * correction + jnp.sum(probs, axis=1, keepdims=True)
            acc = accumulator[...] * correction + lax.dot_general(probs.astype(jnp.bfloat16), kv[0, 0, :, :latent], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            maximum[...], denominator[...], accumulator[...] = m, den, acc
            @pl.when(tile == (count[row]-1)//block)
            def finish():
                out[0] = (acc/den).astype(jnp.bfloat16)
                lse[0] = jnp.broadcast_to(m+jnp.log(den), (heads, 128))

    call = pl.pallas_call(kernel, grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(rows, blocks),
        in_specs=(pl.BlockSpec((1, heads, width), lambda r, b, c: (r, 0, 0)),
                  pl.BlockSpec((1, 1, block, width), lambda r, b, c: (r, b, 0, 0))),
        out_specs=(pl.BlockSpec((1, heads, latent), lambda r, b, c: (r, 0, 0)),
                   pl.BlockSpec((1, heads, 128), lambda r, b, c: (r, 0, 0))),
        scratch_shapes=(pltpu.VMEM((heads, 1), jnp.float32), pltpu.VMEM((heads, 1), jnp.float32), pltpu.VMEM((heads, latent), jnp.float32))),
        out_shape=(jax.ShapeDtypeStruct((rows, heads, latent), jnp.bfloat16), jax.ShapeDtypeStruct((rows, heads, 128), jnp.float32)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"), vmem_limit_bytes=config.vmem_limit_bytes),
        interpret=interpret, name=KERNEL_NAMES["sparse_mla_partial_attention"])
    output, lse = call(counts, queries, cache)
    return output, lse[..., 0]
