"""Pooled prefill: frozen admission flow, canonical chunks and atomic refusal."""
import ast
import inspect
import os
import subprocess
import sys
import pytest


@pytest.mark.parametrize('options', [dict(pooled_moe=1), dict(pooled_moe=True),
    dict(pooled_moe=True,bf16_resident=True,mlp_window=True,rolled_prefix=True,
         expert_panels=True,canonical_dense=True,pending_cache_rows=True)])
def test_pooled_prefill_requires_explicit_supported_composition(options):
    from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
    with pytest.raises(ValueError):
        build_ws32_prefill_challenger_program(None, None, **options)


def test_pooled_runtime_changes_only_row_guards():
    from glm_tpu.greenfield.runtime import ws32_batched_prefill as frozen
    from glm_tpu.perf import pooled_prefill_runtime as pooled
    for name in ('ws32_batched_prefill_mapped', 'build_ws32_batched_prefill_program'):
        expected = inspect.getsource(getattr(frozen, name))
        expected = expected.replace('(128 if mlp_window else 32)', '(1024 if mlp_window else 32)')
        expected = expected.replace('token_ids.shape[0] not in (114, 128)', 'token_ids.shape[0] % 128 not in (0, 114)')
        expected = expected.replace('block_rows not in (114, 128)', 'block_rows % 128 not in (0, 114)')
        expected = expected.replace('canonical dense requires physical B114/B128', 'pooled dense requires B128 chunks and optional B114 tail')
        expected = expected.replace('Build raw prefill; <=128 MLP rows require explicit window opt-in.',
                                    'Private pooled prefill mirror; caller must bind the pooled layer window.')
        assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(getattr(pooled, name))))


def test_pooled_prefill_cpu32():
    code = r'''
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b,ws32_decoder as d
from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program as build
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(v):return jax.device_put(v,NamedSharding(mesh,P()))
config,raw,wk=fixture(mesh,panel_geometry=True)
weights=bf16_resident_weights(mesh,config,raw)
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
opts=dict(key_tile=128,mlp_window=True,rolled_prefix=True,expert_panels=True,canonical_dense=True,
    paired_position_sort=True,sorted_local_merge=True,sparse_attention_interpret=True,linear_interpret=True,
    lse_attention=True,bf16_resident=True)
small={n:build(mesh,config,block_rows=n,**opts) for n in (114,128)}
# Exercise multiple B128 chunks and a B114 final tail. Compare every cache,
# selected position, route-derived activation effect, token and frontier.
for rows in (256,498):
    large=build(mesh,config,block_rows=rows,pooled_moe=True,**opts)
    tokens=put(jnp.asarray(np.arange(rows,dtype=np.int32)%256))
    initial=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=rows)
    expected=initial
    for first in range(0,rows,128):
        n=min(128,rows-first)
        ref=small[n].execute(tokens[first:first+n],put(jnp.int32(n)),expected,weights,wk,rope)
        expected=ref.state
    actual=large.execute(tokens,put(jnp.int32(rows)),initial,weights,wk,rope)
    assert bool(np.asarray(actual.state.finished)) and bool(np.asarray(actual.state.decoder.contract_valid).all())
    for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(ref)):
        np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(e).view(np.uint8))
    # Rejection must preserve the entire committed state except the health latch.
    rejected=large.execute(tokens,put(jnp.int32(rows)),actual.state,weights,wk,rope)
    assert not bool(np.asarray(rejected.state.decoder.contract_valid).all())
    preserved=rejected.state._replace(decoder=rejected.state.decoder._replace(contract_valid=actual.state.decoder.contract_valid))
    for a,e in zip(jax.tree.leaves(preserved),jax.tree.leaves(actual.state)):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(e))
    assert int(np.asarray(rejected.next_token)[0])==-1
'''
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
