"""CPU32 numerical and collective-count proof for the fused input reductions."""
import os
import subprocess
import sys


def test_fused_projection_values_and_collective_count_cpu32():
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.perf.bf16_resident import (
    Bf16QkvAWeights, Bf16DsaWeights, _dot_f32, _head_weight_partial,
    fused_input_projections, prepare_attention_bf16,
)
mesh = Mesh(np.asarray(jax.devices()).reshape(8,4), ('expert','feature'))
rng = np.random.default_rng(20919)
def random(shape, dtype=jnp.bfloat16):
    return jnp.asarray(rng.normal(size=shape), dtype)
# Deliberately use nonaligned output widths, unequal feature shards and signed
# inputs; cover one-row decode and multi-row arithmetic, full/shared indexers.
for rows in (1, 3):
    x = random((rows, 4*32))
    q, kv = random((47, 4*32)), random((29, 4*32))
    wk, head = random((19, 4*32)), random((8*3, 4*32), jnp.float32)
    for full in (False, True):
        def body(x, q, kv, wk, head):
            weights = Bf16QkvAWeights(None, q, jnp.ones((47,),jnp.bfloat16),
                                     kv, jnp.ones((19,),jnp.bfloat16))
            dsa = Bf16DsaWeights(None, wk, None, None, head) if full else None
            partials = [_dot_f32(x,q), _dot_f32(x,kv)]
            if full: partials += [_dot_f32(x,wk), _head_weight_partial(x,dsa)]
            separate = tuple(lax.psum(v,'feature') for v in partials)
            fused = fused_input_projections(x,weights,dsa,feature_axis='feature')
            # Compare pre-norm sums and downstream BF16 rounding/norm boundaries.
            a = prepare_attention_bf16(x,weights,normalized=x,feature_axis='feature')
            b = prepare_attention_bf16(x,weights,normalized=x,feature_axis='feature',projected=fused[:2])
            return separate, fused, a.q_residual, b.q_residual, a.current_kv, b.current_kv
        projection_specs = (P(),P(),P(),P(None,'expert')) if full else (P(),P())
        fn = jax.jit(jax.shard_map(body,mesh=mesh,
            in_specs=(P(None,'feature'),P(None,'feature'),P(None,'feature'),
                      P(None,'feature'),P('expert','feature')),
            out_specs=(projection_specs, projection_specs, P(),P(),P(),P()),check_vma=False))
        separate, fused, qa,qb,kva,kvb = fn(x,q,kv,wk,head)
        # Inspect every owner/feature shard, including the sharded head weights.
        for a,b in zip(jax.tree.leaves((separate,qa,kva)),jax.tree.leaves((fused,qb,kvb))):
            for sa,sb in zip(a.addressable_shards,b.addressable_shards):
                np.testing.assert_array_equal(np.asarray(sa.data).view(np.uint8),np.asarray(sb.data).view(np.uint8))
        # The fused helper contains exactly one psum; q/kv/wk/head dots stay distinct.
        def only(x,q,kv,wk,head):
            return fused_input_projections(x,Bf16QkvAWeights(None,q,None,kv,None),
                Bf16DsaWeights(None,wk,None,None,head) if full else None,feature_axis='feature')
        isolated=jax.shard_map(only,mesh=mesh,in_specs=(P(None,'feature'),P(None,'feature'),
            P(None,'feature'),P(None,'feature'),P('expert','feature')),
            out_specs=projection_specs,check_vma=False)
        ir=str(jax.make_jaxpr(isolated)(x,q,kv,wk,head))
        assert ir.count('psum[')==1, ir
print('fused feature reductions: bitwise, 1 collective, full/shared, M1/M3')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip())
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
