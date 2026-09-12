"""Actual native CPU composition, not checkpoint quality or TPU admission."""
import os
import subprocess
import sys


def test_sampled_prefill_decode_and_observer_cpu32():
    code = r'''
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_sampled_request import (
    build_ws32_sampled_prefill_program, build_ws32_sampled_decoder_program,
)
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
def put(v, spec=P()): return jax.device_put(v, NamedSharding(mesh, spec))
config, weights, wk = fixture(mesh)
# Uniform final logits make the sampled token independently predictable while
# all eight real layer/cache/DSA bodies still execute with nonzero weights.
weights = weights._replace(lm_head_local=jnp.zeros_like(weights.lm_head_local))
wk = tuple(put(v) for v in wk)
rope = put(jnp.asarray(d.build_ws32_main_rope_table(config), jnp.bfloat16))
sampling = NucleusConfig()
options = dict(key_tile=128, sparse_attention_interpret=True, linear_interpret=True)
program = build_ws32_sampled_prefill_program(mesh, config, sampling=sampling,
                                            block_rows=2, **options)
assert not program.template.mlp_window
state = b.make_ws32_batched_prefill_state(mesh, config, prompt_length=3)
tokens = put(jnp.array([30, 31], jnp.int32))
count = put(jnp.int32(2))
u = put(jnp.float32(.999))
args = (tokens, count, state, weights, wk, rope)
compiled = program.execute.lower(*args, u).compile()
ops = [x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
fg = tuple(tuple(range(e*4, e*4+4)) for e in range(8))
eg = tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert all(x.replica_groups in (fg, eg) for x in ops)
first = compiled(*args, put(jnp.float32(float('nan'))))
assert first.next_token.tolist() == [-1] and not bool(first.state.finished)
assert np.asarray(first.state.decoder.contract_valid).all()
tail = put(jnp.array([32, -1], jnp.int32))
tail_args = (tail, put(jnp.int32(1)), first.state, weights, wk, rope)
last = compiled(*tail_args, u)
assert last.next_token.tolist() == [243] and bool(last.state.finished)
assert np.asarray(last.state.decoder.contract_valid).all()
# Same executable, different draw: no hidden compile-time seed/token constant.
other = compiled(*tail_args, put(jnp.float32(0)))
assert other.next_token.tolist() == [0]
for x, y in zip(jax.tree.leaves(last.state), jax.tree.leaves(other.state)):
    np.testing.assert_array_equal(x, y)
bad = compiled(*tail_args, put(jnp.float32(1)))
assert bad.next_token.tolist() == [-1] and not bool(bad.state.finished)
assert not np.asarray(bad.state.decoder.contract_valid).any()
for name in state.decoder._fields:
    if name != 'contract_valid':
        np.testing.assert_array_equal(getattr(bad.state.decoder, name),
                                      getattr(first.state.decoder, name))
np.testing.assert_array_equal(bad.state.repaired_index_local, first.state.repaired_index_local)
greedy = b.build_ws32_batched_prefill_program(mesh, config, block_rows=2, **options)
greedy_last = greedy.execute(*tail_args)
assert greedy_last.next_token.tolist() == [0]
for x, y in zip(jax.tree.leaves(last.state), jax.tree.leaves(greedy_last.state)):
    np.testing.assert_array_equal(x, y)
ds, token = b.finish_ws32_batched_prefill(last)
decode = build_ws32_sampled_decoder_program(mesh, config, sampling=sampling,
                                           sparse_attention_interpret=True, linear_interpret=True)
step_args = (token, ds, weights, rope)
dec = decode.execute.lower(*step_args, u).compile()
steps = [x for x in parse_hlo_module(dec.as_text()).instructions if x.is_collective]
assert all(x.replica_groups in (fg, eg) for x in steps)
step = dec(*step_args, put(jnp.float32(.5)))
assert step.next_token.tolist() == [122]
assert step.state.position.tolist() == [4]
assert np.asarray(step.state.contract_valid).all()
assert dec(*step_args, u).next_token.tolist() == [243]
ref = d.build_ws32_decoder_program(mesh, config, sparse_attention_interpret=True,
                                    linear_interpret=True)
greedy_step = jax.jit(ref.execute)(*step_args)
assert greedy_step.next_token.tolist() == [0]
for x, y in zip(jax.tree.leaves(step.state), jax.tree.leaves(greedy_step.state)):
    np.testing.assert_array_equal(x, y)
np.testing.assert_array_equal(step.final_residual_local, greedy_step.final_residual_local)
observed = decode.observe(*step_args, put(jnp.float32(.5)))
for x, y in zip(jax.tree.leaves(step), jax.tree.leaves(observed.result)):
    np.testing.assert_array_equal(x, y)
assert observed.dsa.producer_layer_ids.tolist() == [0, 1, 2, 6]
invalid = dec(*step_args, put(jnp.float32(float('nan'))))
assert invalid.next_token.tolist() == [-1] and not np.asarray(invalid.state.contract_valid).any()
# The real compiled native programs also drive the request session. Pause by
# retaining the same object/cache, then resume; there is no host cache copy or
# durable/process-crash resume claim in this test.
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy, Ws32RequestSession
from time import perf_counter
emitted = []
calls = []
def request_decode(token, state, uniform):
    calls.append(1)
    return dec(token, state, weights, rope, uniform)
session = Ws32RequestSession(RequestPolicy('cpu-real', 5, 3, 3, 1536, 256, (254,)),
    decode_step=request_decode, replicate_uniform=put, fleet_all=lambda ok:ok,
    deliver=emitted.append, delivery_boundary='CPU test list append', request_started=perf_counter())
initial = compiled(*tail_args, session.next_uniform())
session.accept_prefill(initial)
assert len(emitted) == 1 and not calls and session.ttft_seconds >= 0
paused = session
paused.step()
paused.step()
assert session.finished and len(emitted) == 3 and len(calls) == 2
assert emitted[-1].finish_reason == 'length' and len(session.decode_seconds) == 2
assert session.delivered_request_seconds >= session.ttft_seconds
try:session.step(); raise AssertionError('generated beyond cap')
except RuntimeError:pass
assert len(calls) == 2
try:
    build_ws32_sampled_prefill_program(mesh, config, sampling=sampling, block_rows=33)
    raise AssertionError('original row guard bypassed')
except ValueError: pass
try:
    build_ws32_sampled_prefill_program(mesh, config, sampling=sampling, block_rows=2,
                                      flat_pending_rows=True)
    raise AssertionError('original option guard bypassed')
except ValueError: pass
try:
    build_ws32_sampled_decoder_program(mesh, config, sampling=None)
    raise AssertionError('implicit sampling accepted')
except ValueError: pass
print('CPU32_SAMPLED_PREFILL_DECODE_OBSERVER_PASS')
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, JAX_PLATFORMS="cpu",
                 XLA_FLAGS="--xla_force_host_platform_device_count=32"),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32_SAMPLED_PREFILL_DECODE_OBSERVER_PASS" in result.stdout


def test_optional_decode_input_abi_cpu32():
    code = r'''
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b, ws32_decoder as d
from glm_tpu.greenfield.runtime.ws32_sampled_request import build_ws32_sampled_decoder_program
from glm_tpu.greenfield.kernels.ws32_sampling import NucleusConfig
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
config, weights, wk = fixture(mesh)
sampling = NucleusConfig()
ds = b.make_ws32_batched_prefill_state(mesh, config, prompt_length=3).decoder
token = jnp.array([5], jnp.int32)
rope = jnp.zeros(config.main_rope_table_shape, jnp.bfloat16)
u = jnp.float32(.5)
# Cheap ABI checks for all optional-input combinations, including production's
# exact-DSA branch. Fake math here tests dispatch ONLY; the execution proof
# above is raw-DSA eight-layer CPU, not promoted exact-alias TPU arithmetic.
from glm_tpu.greenfield.kernels.ws32_layer import Ws32ExactDsaWeights
original_impl = d._ws32_decode_impl
try:
    for exact_on in (False, True):
        for rope_on in (False, True):
            variant = replace(config, exact_dsa=exact_on, host_main_rope_table=rope_on)
            seen = []
            def spy(t, s, w, *, config, exact_dsa_weights, main_rope_table,
                    observe_dsa, final_sample, **opts):
                assert (exact_dsa_weights is not None) == config.exact_dsa
                assert (main_rope_table is not None) == config.host_main_rope_table
                assert final_sample.keywords['config'] == sampling
                assert final_sample.keywords['uniform'].shape == ()
                seen.append(True)
                return d.Ws32DecodeStepResult(s, t, jnp.zeros((1, 128), jnp.bfloat16)), None, None
            d._ws32_decode_impl = spy
            optional = ()
            if exact_on:
                fake = d.Ws32ExactDsaWeights(jnp.zeros(1, jnp.uint8), jnp.zeros(1),
                    tuple(jnp.zeros((4, 4), jnp.bfloat16) for _ in range(4)),
                    jnp.zeros(1), jnp.zeros((4, 4), jnp.bfloat16))
                optional += (tuple(fake for _ in config.full_index_slots),)
            if rope_on: optional += (rope,)
            routed = build_ws32_sampled_decoder_program(mesh, variant, sampling=sampling)
            shaped = jax.eval_shape(routed.execute, token, ds, weights, *optional, u)
            assert shaped.next_token.shape == (1,) and seen
finally:
    d._ws32_decode_impl = original_impl
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, JAX_PLATFORMS="cpu",
                 XLA_FLAGS="--xla_force_host_platform_device_count=32"),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
