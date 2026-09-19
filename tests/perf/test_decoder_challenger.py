"""Challenger decode step equals the frozen step on the 8-layer CPU fixture.

Real native composition (fused norms, DSA, IndexShare, caches, router, MoE,
sampled head) with fixture weights on 32 forced CPU devices.  This is the
same equality contract the frozen sampled-request test uses between its
greedy and sampled programs; it is not TPU admission of the new graphs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions


def test_options_are_validated():
    with pytest.raises(ValueError):
        Ws32PerfOptions(sampler="beam")
    with pytest.raises(ValueError):
        Ws32PerfOptions(candidates_per_shard=0)
    with pytest.raises(ValueError):
        Ws32PerfOptions(grouped_routes="yes")  # type: ignore[arg-type]


def test_challenger_step_matches_frozen_step_cpu32():
    code = r'''
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_sampled_request import (
    build_ws32_sampled_prefill_program, build_ws32_sampled_decoder_program)
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig
from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions, build_ws32_challenger_decoder_program
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
def put(v, spec=P()): return jax.device_put(v, NamedSharding(mesh, spec))
config, weights, wk = fixture(mesh)
wk = tuple(put(v) for v in wk)
rope = put(jnp.asarray(d.build_ws32_main_rope_table(config), jnp.bfloat16))
sampling = NucleusConfig()
interpret = dict(sparse_attention_interpret=True, linear_interpret=True)
prefill = build_ws32_sampled_prefill_program(mesh, config, sampling=sampling, block_rows=2, key_tile=128, **interpret)
state = b.make_ws32_batched_prefill_state(mesh, config, prompt_length=3)
first = prefill.execute(put(jnp.array([30, 31], jnp.int32)), put(jnp.int32(2)), state, weights, wk, rope, put(jnp.float32(.5)))
last = prefill.execute(put(jnp.array([32, -1], jnp.int32)), put(jnp.int32(1)), first.state, weights, wk, rope, put(jnp.float32(.5)))
ds, token = b.finish_ws32_batched_prefill(last)
frozen = build_ws32_sampled_decoder_program(mesh, config, sampling=sampling, **interpret)
greedy = jax.jit(d.build_ws32_decoder_program(mesh, config, **interpret).execute)
tiles = RoutedProjectionConfig(block_shape=(128, 128), output_tile=128, contraction_tile=128)
challengers = {
    'nucleus': build_ws32_challenger_decoder_program(mesh, config, sampling=sampling,
        options=Ws32PerfOptions(sampler='nucleus', routed_projection=tiles), **interpret),
    'candidates': build_ws32_challenger_decoder_program(mesh, config, sampling=sampling,
        options=Ws32PerfOptions(sampler='nucleus_candidates', candidates_per_shard=8, routed_projection=tiles), **interpret),
    'greedy': build_ws32_challenger_decoder_program(mesh, config,
        options=Ws32PerfOptions(sampler='greedy', routed_projection=tiles), **interpret),
}
def leaves(r): return jax.tree.leaves(r.state) + [r.next_token, r.final_residual_local]
def mismatches(x, y): return sum(int(np.count_nonzero(np.asarray(a) != np.asarray(b))) for a, b in zip(leaves(x), leaves(y)))
report = {'tokens': {}, 'mismatches': {}, 'valid': {}}
for u in (.5, .999, .01):
    ref = frozen.execute(token, ds, weights, rope, put(jnp.float32(u)))
    report['tokens'][str(u)] = ref.next_token.tolist()
    report['valid'][str(u)] = bool(np.asarray(ref.state.contract_valid).all())
    for name in ('nucleus', 'candidates'):
        out = challengers[name].execute(token, ds, weights, rope, put(jnp.float32(u)))
        report['mismatches'][f'{name}/{u}'] = mismatches(ref, out)
ref = greedy(token, ds, weights, rope)
report['mismatches']['greedy'] = mismatches(ref, challengers['greedy'].execute(token, ds, weights, rope))
report['tokens']['greedy'] = ref.next_token.tolist()
# Poisoned state fails closed identically.
bad_state = ds._replace(contract_valid=jnp.zeros((1,), jnp.bool_))
out = challengers['candidates'].execute(token, bad_state, weights, rope, put(jnp.float32(.5)))
report['poisoned_valid'] = bool(np.asarray(out.state.contract_valid).any())
# A nonfinite uniform fails closed with token -1 like the frozen head.
out = challengers['candidates'].execute(token, ds, weights, rope, put(jnp.float32(float('nan'))))
report['nan_uniform'] = [out.next_token.tolist(), bool(np.asarray(out.state.contract_valid).any())]
# Collective census of the compiled step (CPU HLO; group geometry must stay local).
fg = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
eg = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
census = {}
for label, fn, args in (('frozen', frozen.execute, (token, ds, weights, rope, put(jnp.float32(.5)))),
                        ('challenger', challengers['candidates'].execute, (token, ds, weights, rope, put(jnp.float32(.5))))):
    ops = [x for x in parse_hlo_module(fn.lower(*args).compile().as_text()).instructions if x.is_collective]
    census[label] = dict(collectives=len(ops), local_groups=all(x.replica_groups in (fg, eg) for x in ops))
report['census'] = census
print(json.dumps(report, sort_keys=True))
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip())
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=1500)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert all(report["valid"].values())
    assert all(value == 0 for value in report["mismatches"].values()), report["mismatches"]
    assert not report["poisoned_valid"]
    assert report["nan_uniform"] == [[-1], False]
    assert report["census"]["challenger"]["local_groups"]
    assert report["census"]["challenger"]["collectives"] < report["census"]["frozen"]["collectives"]
