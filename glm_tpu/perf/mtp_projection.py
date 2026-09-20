"""Native MTP input projection candidate; not a complete or admitted drafter.

The caller supplies embedding(x[t+1]) and post-final-norm hidden h[t] at
position t. Position zero masks only the embedding. The concatenation order is
all embedding features, then all hidden features, before the BF16 projection.
Reference: pinned vLLM DeepSeek MTP and IR RMSNorm in MTP_STATE_CONTRACT.
"""
from functools import partial
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from .bf16_resident import _dot_f32


class MtpProjectionWeights(NamedTuple):
    enorm_local: Any
    hnorm_local: Any
    # Output-feature sharded [H/4, 2H], replicated across expert8.
    eh_projection_local: Any


class MtpProjectionResult(NamedTuple):
    hidden_local: Any
    contract_valid: Any


def project_mapped(embedding, previous_hidden, positions, weights, *, hidden_size, epsilon):
    """Prepare a bounded row block; weights/inputs are already on WS32 owners.

    These RMS norms consume BF16 inputs, sum squares in FP32 over feature4,
    then round the normalized input before multiplying its BF16 norm weight.
    This matches the reference expression, not a claim of GPU/TPU bitwise parity.
    """
    if (type(hidden_size) is not int or hidden_size <= 0 or hidden_size % 4
            or type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon <= 0
            or embedding.ndim != 2 or not 1 <= embedding.shape[0] <= 128
            or embedding.shape[1] != hidden_size//4 or embedding.dtype != jnp.bfloat16
            or previous_hidden.shape != embedding.shape or previous_hidden.dtype != jnp.bfloat16
            or positions.shape != (embedding.shape[0],) or positions.dtype != jnp.int32
            or weights.enorm_local.shape != (hidden_size//4,) or weights.enorm_local.dtype != jnp.bfloat16
            or weights.hnorm_local.shape != (hidden_size//4,) or weights.hnorm_local.dtype != jnp.bfloat16
            or weights.eh_projection_local.shape != (hidden_size//4,2*hidden_size)
            or weights.eh_projection_local.dtype != jnp.bfloat16):
        raise ValueError('MTP projection requires BF16 feature4 inputs/weights and int32 positions')
    # Do not zero h[0]: it still carries the target's first prompt hidden state.
    masked = jnp.where((positions == 0)[:,None],jnp.bfloat16(0),embedding)
    def norm(x,w):
        return ws32_fused_add_rms_norm_mapped(x,jnp.zeros_like(x),w,
            global_hidden_size=hidden_size,epsilon=epsilon)[0]
    e = norm(masked,weights.enorm_local)
    h = norm(previous_hidden,weights.hnorm_local)
    # Gathering a local concat directly would incorrectly interleave e/h shards.
    e = lax.all_gather(e,'feature',axis=1,tiled=True)
    h = lax.all_gather(h,'feature',axis=1,tiled=True)
    output = _dot_f32(jnp.concatenate((e,h),axis=1),weights.eh_projection_local).astype(jnp.bfloat16)
    valid = ((positions >= 0) & jnp.all(jnp.isfinite(embedding),axis=1)
             & jnp.all(jnp.isfinite(previous_hidden),axis=1)
             & jnp.all(jnp.isfinite(output),axis=1))
    # Even an expert owner whose output remains finite cannot hide another
    # owner's invalid input or overflowing projection.
    valid = lax.pmin(valid.astype(jnp.int32),('expert','feature')) != 0
    return MtpProjectionResult(output,valid)


def build_mtp_projection(mesh, *, hidden_size, epsilon=1e-5):
    if tuple(mesh.axis_names) != ('expert','feature') or tuple(mesh.devices.shape) != (8,4):
        raise ValueError('MTP projection requires expert8 by feature4 mesh')
    return jax.jit(jax.shard_map(partial(project_mapped,hidden_size=hidden_size,epsilon=epsilon),
        mesh=mesh,in_specs=(P(None,'feature'),P(None,'feature'),P(),
            MtpProjectionWeights(P('feature'),P('feature'),P('feature',None))),
        out_specs=MtpProjectionResult(P(None,'feature'),P()),check_vma=False))
