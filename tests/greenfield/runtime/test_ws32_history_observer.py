"""CPU loop parity against the production observer; not TPU or StrategyND.

The StrategyND dense kernel accepts production shapes only, so this test proves
the observer's LOOP reproduces ``ws32_decode_observed_mapped`` with plain dense
weights and interpreted kernels. Original step0 event reproduction on hardware,
with the StrategyND overlay, remains mandatory before any attribution.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def test_history_observer_cpu32_matches_production_observe_step_for_layers_0_to_6():
    source = r'''
from dataclasses import replace
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_history_frontier import LayerBoundary, PRODUCERS
from scripts.greenfield.ws32_history_observer import build_program, observer_config
from glm_tpu.greenfield.runtime import ws32_decoder as dec
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(), object)[::-1].reshape(8, 4), ('expert', 'feature'))
raw, weights, _ = fixture(mesh, panel_geometry=True)
raw = replace(raw, context_capacity=8192)            # the observer's fixed capacity
production = replace(raw, exact_dsa=True)             # eight layers plus head
observer = observer_config(raw, strategy_nd_dense=False)
assert observer == replace(production, geometry=observer.geometry) and observer.geometry.num_layers == 7
def put(v, s=P()): return jax.device_put(v, NamedSharding(mesh, s))
def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, a.dtype, b.shape, b.dtype)
    assert a.tobytes() == b.tobytes(), (a.shape, np.count_nonzero(a != b))
def owner(v):
    return put(np.asarray(v).reshape(observer.page_count, 8, 64, -1).transpose(1, 0, 2, 3), P('expert', None, None, None))

# Exact DSA owners exactly as the protected worker materializes them.
materializer = dec.build_ws32_exact_dsa_materializer_program(mesh, production)
exact = materializer.promote(materializer.decode(dec.select_ws32_exact_dsa_raw_weights(weights, production)))
assert len(exact) == 4
rope = put(jnp.asarray(dec.build_ws32_main_rope_table(production), jnp.bfloat16))
state = dec.make_ws32_initial_state(mesh, production)
# Populated, distinct histories: under zero caches producers0/1/6 emit identical
# observations, so equality could not discriminate producer order or cache reads.
rng = np.random.default_rng(8155)
def populated(shape):
    return put(jnp.asarray(rng.normal(0, 0.5, shape), jnp.bfloat16), P(None, None, 'expert', None))
state = state._replace(kv_cache_local=populated(production.kv_cache_shape),
                       index_cache_local=populated(production.index_cache_shape),
                       position=put(jnp.asarray([8155], jnp.int32)),
                       context_lengths=put(jnp.asarray([8156], jnp.int32)))

# Production observer with every layer result retained, the frontier test's pattern.
boundary = LayerBoundary(P(None, 'feature'), P(None, 'feature'), P(None, 'feature'), P(), P(), P('expert', 'feature', None))
def production_body(token, state, weights, exact, rope):
    original = dec.ws32_transformer_layer_mapped
    retained = []
    def retain(*args, **kwargs):
        result = original(*args, **kwargs)
        retained.append(result)
        return result
    with patch.object(dec, 'ws32_transformer_layer_mapped', retain):
        observed = dec.ws32_decode_observed_mapped(token, state, weights, config=production,
            exact_dsa_weights=exact, sparse_attention_interpret=True, linear_interpret=True,
            main_rope_table=rope)
    assert len(retained) == 8
    boundaries = tuple(LayerBoundary(v.output_local, v.carried_residual_local, v.normalized_input_local,
        v.route_indices, v.route_weights, v.contract_valid[None, None]) for v in retained[:7])
    return observed, boundaries
production_program = jax.jit(jax.shard_map(production_body, mesh=mesh,
    in_specs=(P(), dec.ws32_decoder_state_specs(), dec.ws32_decoder_weight_specs(production),
              dec.ws32_exact_dsa_specs(production), P()),
    out_specs=(dec.ws32_observed_decode_result_specs(), (boundary,) * 7), check_vma=False))
token = put(jnp.asarray([220], jnp.int32))
observed, expected = production_program(token, state, weights, exact, rope)
jax.block_until_ready((observed, expected))
assert bool(observed.result.state.contract_valid[0])

observe = build_program(mesh, observer, sparse_attention_interpret=True, linear_interpret=True)
kv = tuple(owner(state.kv_cache_local[i]) for i in range(7))
repaired = tuple(owner(state.index_cache_local[s]) for s in range(4))
args = (token, put(jnp.asarray([8155], jnp.int32)), put(jnp.asarray([8156], jnp.int32)),
        state.block_tables, kv, repaired, weights.embedding_local, weights.layers[:7], exact, rope,
        put(jnp.bool_(True)))
compiled = observe.lower(*args).compile()
actual = compiled(*args)
jax.block_until_ready(actual)
assert bool(actual.healthy)
assert len(actual.boundaries) == 7
for x, y in zip(jax.tree.leaves(actual.boundaries), jax.tree.leaves(expected), strict=True):
    equal(x, y)
for x, y in zip(jax.tree.leaves(actual.dsa), jax.tree.leaves(observed.dsa), strict=True):
    equal(x, y)
assert np.asarray(actual.dsa.producer_layer_ids).tolist() == list(PRODUCERS)
# The four observations are pairwise distinct, so a producer permutation or a
# duplicated producer would have been caught by the equality above.
selections = [np.asarray(actual.dsa.selected_positions[i]).tobytes() for i in range(4)]
assert len(set(selections)) == 4, 'observations not distinct; test cannot discriminate producers'
for i in range(7):
    assert np.isfinite(np.asarray(actual.boundaries[i].update, np.float32)).all()
print('OBSERVER_PARITY', flush=True)

# The witness identity is part of the program: any other token, position or
# context length is unhealthy, so a host cannot observe the wrong step.
for wrong in ((put(jnp.asarray([219], jnp.int32)),) + args[1:],
              args[:1] + (put(jnp.asarray([8154], jnp.int32)),) + args[2:],
              args[:2] + (put(jnp.asarray([8155], jnp.int32)),) + args[3:],
              args[:-1] + (put(jnp.bool_(False)),)):
    refused = compiled(*wrong)
    jax.block_until_ready(refused)
    assert not bool(refused.healthy)
    # A refused step carries sentinels, never a computed observation.
    assert (np.asarray(refused.dsa.selected_positions) == -1).all()
    assert (np.asarray(refused.dsa.selected_valid_counts) == 0).all()
    assert np.isneginf(np.asarray(refused.dsa.selected_scores)).all()
    for b in refused.boundaries:
        assert np.isnan(np.asarray(b.update, np.float32)).all()
        assert np.isnan(np.asarray(b.residual, np.float32)).all()
        assert np.isnan(np.asarray(b.normalized_input, np.float32)).all()
        assert (np.asarray(b.route_ids) == -1).all() and np.isnan(np.asarray(b.route_weights)).all()
        assert not np.asarray(b.health).any()
print('OBSERVER_REFUSALS', flush=True)

# Static contract: the raw preparation must be the ORIGINAL raw one, and the
# default StrategyND representation is production-geometry-only, so on this
# fixture the config itself refuses it before any program exists.
from glm_tpu.greenfield.errors import PlanValidationError
for bad in (replace(raw, exact_dsa=True), replace(raw, context_capacity=4096),
            replace(raw, host_main_rope_table=False)):
    try: observer_config(bad, strategy_nd_dense=False)
    except ValueError: pass
    else: raise AssertionError('non-original raw preparation accepted')
try: observer_config(raw, strategy_nd_dense=1)
except ValueError: pass
else: raise AssertionError('non-boolean dense representation accepted')
try: observer_config(raw)
except (ValueError, PlanValidationError) as error:
    # The geometry pin (segment block 128 here) or the config's own StrategyND
    # geometry check refuses; either way no StrategyND observer exists off production.
    assert 'geometry' in str(error), error
else: raise AssertionError('StrategyND observer constructed on fixture geometry')
# A program built for one weight branch refuses another. The two dense
# representations and the dense/MoE split all differ in pytree structure, so
# shard_map's specification check fires before the observer's own guard.
mismatched = tuple(l._replace(dense=None, moe=weights.layers[3].moe) if i < 3 else l
                   for i, l in enumerate(weights.layers[:7]))
try: observe.lower(*(args[:7] + (mismatched,) + args[8:]))
except ValueError as error:
    assert 'dense/indexer branch' in str(error) or 'pytree structure' in str(error), error
else: raise AssertionError('wrong dense/MoE branch accepted')
for kwargs in (dict(sparse_attention_interpret=1), dict(linear_interpret='yes'),
               dict(sparse_attention_interpret=True), dict(linear_interpret=True), {}):
    try: build_program(mesh, observer, **kwargs)
    except ValueError: pass
    else: raise AssertionError('non-boolean interpret flag or compilable plain-dense observer accepted')
# The StrategyND default pins the original numerical geometry.
for changed in (dict(logical_page_size=256), dict(sparse_segment_block=128),
                dict(rms_norm_epsilon=1e-6), dict(packed_cache_width=576)):
    try: observer_config(replace(raw, **changed))
    except (ValueError, PlanValidationError): pass
    else: raise AssertionError('changed numerical geometry accepted for the StrategyND observer')
print('HISTORY_OBSERVER_CPU_PASS', flush=True)
'''
    flags = f"{os.environ.get('XLA_FLAGS', '').strip()} --xla_force_host_platform_device_count=32".strip()
    result = subprocess.run([sys.executable, "-c", source], text=True, cwd=ROOT,
        capture_output=True, timeout=900,
        env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OBSERVER_PARITY" in result.stdout
    assert "OBSERVER_REFUSALS" in result.stdout
    assert "HISTORY_OBSERVER_CPU_PASS" in result.stdout
