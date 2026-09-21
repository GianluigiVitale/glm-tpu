"""One vectorized decode step for independent conversations sharing weights.

The batch axis belongs to tokens and request state, never model weights. The
existing single-conversation numerical body is transformed inside one compiled
call; the host does not run a separate model call for each conversation.
"""
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .ws32_decoder_challenger import build_ws32_challenger_decoder_program


class BatchedDecodeResult(NamedTuple):
    state: Any
    next_token: Any
    metadata: Any  # [conversation, token/health/position/context-length]


def build_batched_decoder_program(mesh, config, *, batch_size=8, donate_state=True,
                                  sparse_attention_interpret=False,
                                  linear_interpret=False):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError('batched decode requires one to eight conversations')
    if config.host_main_rope_table is not True:
        raise ValueError('batched decode requires the shared host rotary table')
    single = build_ws32_challenger_decoder_program(
        mesh, config, sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret)
    mapped = jax.vmap(single.execute, in_axes=(0, 0, None, None))

    def pack(tokens, health, position, lengths):
        healthy=jax.lax.pmin(health.astype(jnp.int32),('expert','feature'))
        return jnp.stack((tokens[:,0],healthy[:,0],position[:,0],lengths[:,0]),axis=1)
    pack=jax.shard_map(pack,mesh=mesh,in_specs=(P(),)*4,out_specs=P(),check_vma=False)

    def execute(tokens, state, weights, rope, active):
        if tokens.shape != (batch_size, 1) or tokens.dtype != jnp.int32:
            raise ValueError('batched decoder tokens must be int32[batch,1]')
        if active.shape != (batch_size,) or active.dtype != jnp.bool_:
            raise ValueError('batched decoder active mask must be bool[batch]')
        out = mapped(tokens, state, weights, rope)

        def retain_finished(previous, updated):
            mask = active.reshape((batch_size,) + (1,) * (previous.ndim - 1))
            return jnp.where(mask, updated, previous)

        next_state = jax.tree.map(retain_finished, state, out.state)
        next_token = jnp.where(active[:, None], out.next_token, tokens)
        metadata = pack(next_token,next_state.contract_valid,
                        next_state.position,next_state.context_lengths)
        return BatchedDecodeResult(next_state, next_token, metadata)

    return jax.jit(execute, donate_argnums=(1,) if donate_state else ())
