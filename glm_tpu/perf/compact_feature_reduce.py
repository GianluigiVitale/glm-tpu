"""Bounded owner-row feature reduction, with an exact full-buffer fallback."""
import jax.numpy as jnp
from jax import lax


def owner_compact_feature_sum(gate_up, start, count, *, capacity):
    """Reduce FP32 gate/up contributions and return the original BF16 shape.

    Internal precondition: every row outside [start,start+count) is positive
    zero, as produced by the routed expert panel unpacker. Routing metadata is
    replicated within each feature4 group; different expert owners may choose
    different branches. No expert-axis collective occurs inside either branch.
    This primitive is not wired into a model or responsible for route health.
    """
    if (gate_up.ndim!=3 or gate_up.shape[0]!=2 or gate_up.dtype!=jnp.float32
            or start.shape!=() or count.shape!=() or start.dtype!=jnp.int32 or count.dtype!=jnp.int32
            or type(capacity) is not int or capacity<1 or lax.axis_size('feature')!=4):
        raise ValueError('compact feature sum requires FP32 gate/up, int32 span and feature4')
    rows=gate_up.shape[1]
    def full(_):return lax.psum(gate_up,'feature').astype(jnp.bfloat16)
    if capacity>=rows:return full(None)
    safe_start=jnp.clip(start,0,rows)
    fits=(start==safe_start)&(count>=0)&(count<=capacity)&(count<=rows-safe_start)
    def compact(_):
        # Padding prevents dynamic_slice/update_slice from clamping a final
        # owner's start backwards when it has fewer rows than the bound.
        source=jnp.pad(gate_up,((0,0),(0,capacity),(0,0)))
        owned=lax.dynamic_slice_in_dim(source,safe_start,capacity,axis=1)
        owned=jnp.where(jnp.arange(capacity)[None,:,None]<count,owned,0)
        reduced=lax.psum(owned,'feature').astype(jnp.bfloat16)
        restored=lax.dynamic_update_slice_in_dim(
            jnp.zeros((2,rows+capacity,gate_up.shape[2]),jnp.bfloat16),reduced,safe_start,axis=1)
        return restored[:,:rows]
    return lax.cond(fits,compact,full,None)
