"""Bounded sampler semantics; no full-model quality or TPU speed claim."""
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig, nucleus_sample, request_uniform


@pytest.mark.parametrize('config', [dict(temperature=0.0), dict(temperature=-1.0),
    dict(temperature=float('inf')), dict(temperature=1e-100), dict(temperature=1e100),
    dict(top_p=1e-100), dict(top_p=0.0), dict(top_p=1.01), dict(top_p=True)])
def test_bad_configuration(config):
    with pytest.raises(ValueError): NucleusConfig(**config)


def test_ties_and_inclusive_cutoff():
    logits = jnp.zeros((1, 8), jnp.float32)
    fn = jax.jit(lambda u: nucleus_sample(logits, u, config=NucleusConfig(top_p=0.5)))
    for u, expected in ((0., 0), (.249, 0), (.25, 1), (.5, 2), (.75, 3), (.999999, 3)):
        result = fn(jnp.float32(u)); assert int(result.token_id[0]) == expected
        assert bool(result.contract_valid[0])


@pytest.mark.parametrize('temperature,top_p', [(1., .95), (.7, .8), (2., 1.), (1., .01)])
def test_matches_independent_numpy_distribution(temperature, top_p):
    scores = np.random.default_rng(9).normal(size=64).astype(np.float32)
    order = np.lexsort((np.arange(scores.size), -scores))
    weights = np.exp((scores[order].astype(np.float64) - scores.max()) / temperature)
    probs = weights / weights.sum()
    count = np.searchsorted(np.cumsum(probs), top_p, side='left') + 1
    cumulative = np.cumsum(weights[:count]); cumulative /= cumulative[-1]
    fn = jax.jit(lambda u: nucleus_sample(jnp.asarray(scores[None]), u,
                                        config=NucleusConfig(temperature, top_p)))
    for u in np.linspace(.001, .999, 41, dtype=np.float32):
        expected = order[np.searchsorted(cumulative, u, side='right')]
        assert int(fn(jnp.asarray(u)).token_id[0]) == expected


@pytest.mark.parametrize('uniform', [-.1, 1., float('nan'), float('inf')])
def test_invalid_uniform_fails_closed(uniform):
    result = nucleus_sample(jnp.zeros((1, 8)), jnp.float32(uniform), config=NucleusConfig())
    assert int(result.token_id[0]) == -1 and not bool(result.contract_valid[0])


def test_nonfinite_logit_fails_closed():
    for bad in (float('nan'), float('inf'), -float('inf')):
        result = nucleus_sample(jnp.array([[bad, 1.]]), jnp.float32(.2), config=NucleusConfig())
        assert int(result.token_id[0]) == -1 and not bool(result.contract_valid[0])


def test_single_mass_and_underflow_do_not_select_zero_probability_tail():
    for p in (.95, 1.):
        result = nucleus_sample(jnp.array([[0., -1000., -1001.]]),
                                jnp.float32(1 - 2**-24), config=NucleusConfig(top_p=p))
        assert int(result.token_id[0]) == 0


def test_uniform_replay_after_resume_and_request_isolation():
    draws = [request_uniform(seed=42, request_id='a', token_index=i) for i in range(10)]
    assert draws[5:] == [request_uniform(seed=42, request_id='a', token_index=i) for i in range(5, 10)]
    assert all(0 <= x < 1 and float(np.float32(x)) == x for x in draws)
    assert draws != [request_uniform(seed=42, request_id='b', token_index=i) for i in range(10)]
    with pytest.raises(ValueError): request_uniform(seed=True, request_id='a', token_index=0)
    with pytest.raises(ValueError): request_uniform(seed=1, request_id='a', token_index=-1)


def test_32_cpu_device_output_boundary():
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig, ws32_nucleus_sample_mapped, ws32_split_nucleus_sample_mapped, nucleus_sample
from glm_tpu.greenfield.kernels.ws32_io import Ws32GreedySampleResult, Ws32SplitGreedySampleResult, ws32_logits_mapped
from glm_tpu.greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
mesh = Mesh(np.array(jax.devices()).reshape(8,4), ('expert','feature'))
config = NucleusConfig()
sample = jax.jit(jax.shard_map(lambda x,u: ws32_nucleus_sample_mapped(x,u,vocab_size=64,config=config), mesh=mesh, in_specs=(P(None,'expert'),P()), out_specs=Ws32GreedySampleResult(P(),P()), check_vma=False))
logits=jnp.arange(64,dtype=jnp.bfloat16)[None]/8
for u in (.01,.5,.99):
    result=sample(logits,jnp.float32(u)); reference=nucleus_sample(logits,jnp.float32(u),config=config)
    np.testing.assert_array_equal(result.token_id,reference.token_id)
    assert bool(result.contract_valid[0])
hlo=sample.lower(logits,jnp.float32(.5)).compile().as_text()
assert 'all-gather' in hlo and 'all-to-all' not in hlo
collectives = parse_hlo_module(hlo).collectives
assert len(collectives) == 1
assert collectives[0].replica_groups == tuple(tuple(range(f,32,4)) for f in range(4))
rng=np.random.default_rng(5)
x=jnp.asarray(rng.normal(size=(1,128)),jnp.bfloat16); residual=x/4
norm=jnp.ones((128,),jnp.bfloat16); head=jnp.asarray(rng.normal(size=(64,128)),jnp.bfloat16)
def reference(x,r,n,h,u):
    normalized,final=ws32_fused_add_rms_norm_mapped(x,r,n,global_hidden_size=128)
    logits=ws32_logits_mapped(normalized,h,vocab_size=64)
    selected=ws32_nucleus_sample_mapped(logits,u,vocab_size=64,config=config)
    return Ws32SplitGreedySampleResult(selected.token_id,selected.contract_valid,final)
def candidate(x,r,n,h,u):
    return ws32_split_nucleus_sample_mapped(x,r,n,h,u,hidden_size=128,vocab_size=64,config=config)
args=(x,residual,norm,head,jnp.float32(.42))
kwargs=dict(mesh=mesh,in_specs=(P(None,'feature'),P(None,'feature'),P('feature'),P('expert','feature'),P()),out_specs=Ws32SplitGreedySampleResult(P(),P(),P(None,'feature')),check_vma=False)
a=jax.jit(jax.shard_map(candidate,**kwargs))(*args)
b=jax.jit(jax.shard_map(reference,**kwargs))(*args)
for left,right in zip(a,b): np.testing.assert_array_equal(left,right)
print('SAMPLER_32_CPU_OK')
'''
    env = dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32')
    result = subprocess.run([sys.executable, '-c', program], env=env,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SAMPLER_32_CPU_OK' in result.stdout
