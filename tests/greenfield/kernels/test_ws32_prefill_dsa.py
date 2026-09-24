"""Forced32 CPU collective/causal admission for the unwired prefill selector."""

import json
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize(
    "paired_position_sort,sorted_local_merge",
    [(False, False), (True, False), (True, True)],
)
def test_causal_multitoken_dsa_candidates_stay_in_expert8_groups(
    paired_position_sort, sorted_local_merge
):
    program = r"""
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.prefill_dsa import ws32_prefill_dsa_from_query_mapped
from glm_tpu.optimized.reference.dsa import dsa_scores, exact_topk
from glm_tpu.optimized.hlo_contract import parse_hlo_module
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng = np.random.default_rng(356)
q = jnp.asarray(rng.normal(0,.2,(5,32,128)),jnp.float32)
keys = jnp.asarray(rng.normal(0,.2,(8,256,128)),jnp.bfloat16)
weights = jnp.asarray(rng.normal(0,.2,(5,32)),jnp.float32).at[4].set(0)
pos = jnp.arange(2048,dtype=jnp.int32).reshape(256,8).T
lengths = jnp.asarray([0,1,129,1300,2048],jnp.int32)
specs=(P(),P('expert',None,None),P(),P('expert',None),P())
values=tuple(jax.device_put(v,NamedSharding(mesh,s)) for v,s in zip((q,keys,weights,pos,lengths),specs))
def body(q,k,w,p,l):
    selected,health=ws32_prefill_dsa_from_query_mapped(q,k[0],w,p[0],l,global_context_size=2048,top_k=64,key_tile=128,paired_position_sort=PAIRED_SORT,sorted_local_merge=SORTED_LOCAL)
    return selected.positions,selected.valid_counts,selected.scores,health[None,None]
mapped=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=specs,out_specs=(P(),P(),P(),P('expert','feature')),check_vma=False))
compiled=mapped.lower(*values).compile()
selected,counts,scores,health=compiled(*values)
assert np.asarray(health).all()
global_keys=keys.transpose(1,0,2).reshape(2048,128)
expected_scores=dsa_scores(q,global_keys,weights)
expected=exact_topk(expected_scores,lengths,top_k=64)
np.testing.assert_array_equal(selected,expected.positions)
np.testing.assert_array_equal(counts,expected.valid_counts)
np.testing.assert_array_equal(selected[4],np.arange(64))
module=parse_hlo_module(compiled.as_text())
ops=[x for x in module.instructions if x.is_collective]
groups=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert len(ops)==2 and all(x.opcode=='all-gather' and x.replica_groups==groups for x in ops)
# A repeated live position is unhealthy only on its context owner's replicas.
bad_pos=values[3].at[2,3].set(values[3][2,2])
bad=mapped(values[0],values[1],values[2],bad_pos,values[4])[3]
np.testing.assert_array_equal(np.asarray(bad)[2],False)
assert np.asarray(bad)[[0,1,3,4,5,6,7]].all()
# Merger counts are requested coverage, not proof of supplied cache coverage.
holes=jax.device_put(jnp.full((8,256),-1,jnp.int32),NamedSharding(mesh,specs[3]))
assert not np.asarray(mapped(values[0],values[1],values[2],holes,values[4])[3]).any()
# Each owner is locally increasing, but two owners overlap globally.
duplicate=values[3].at[1].set(values[3][0])
assert not np.asarray(mapped(values[0],values[1],values[2],duplicate,values[4])[3]).any()
print(json.dumps({'rows':5,'groups':[8,8],'all_chip_health':True}))
"""
    program = program.replace("PAIRED_SORT", repr(paired_position_sort))
    program = program.replace("SORTED_LOCAL", repr(sorted_local_merge))
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "rows": 5,
        "groups": [8, 8],
        "all_chip_health": True,
    }
