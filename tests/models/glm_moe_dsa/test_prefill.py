"""The production prefill program on the CPU mesh: health, its one-pass selector, atomic refusal of a
finished state and exclusive state donation; and the B128/B114 programs over the real 78-layer checkpoint
schema, by shape evaluation (``tests/fixtures/prefill_layer_schema.json``, the tensor schema of the checkpoint
manifest, copied from ``archive/research-20260922``). The prefill block semantics are in
``test_prefill_semantics.py``.

Until S2f this test also compared every leaf with the independent frozen FP8 prefill program (raw
path bitwise, BF16-resident path within rtol 0.02 / atol 0.0625). That oracle is archived at
``archive/research-20260922``; its final green run is recorded in the S2f commit message, and the
reference model (``tests/reference``, ``VALIDATION.md``) plus the G1/G3 goldens carry the evidence.
"""

import os
import subprocess
import sys

import pytest


CODE = r"""
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.layers.attention import dsa_indexer  # the production DSA selects through its one-pass selector
seen=[]
one_pass=dsa_indexer.prefill_dsa_one_pass
def observed_selector(*args,**kwargs):
    seen.append(True)
    return one_pass(*args,**kwargs)
dsa_indexer.prefill_dsa_one_pass=observed_selector
from glm_tpu.models.glm_moe_dsa.prefill import build_prefill_program
from glm_tpu.runner.programs import build_program_set
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
mesh=cpu_mesh()
def put(v): return jax.device_put(v,NamedSharding(mesh,P()))
inputs=engine_inputs(mesh,panel_geometry=True)
config=inputs.config
initialize=build_program_set(mesh,config,interpret=True).cache_init.fn
# the production program hard-wires the admitted profile (window, rolled prefix, panels, canonical
# dense, one-pass selector); only the block size and the interpret flags are arguments
program=build_prefill_program(mesh,config,block_rows=128,sparse_attention_interpret=True,linear_interpret=True)
tokens,count=put(jnp.array([30,31,32]+[-1]*125,jnp.int32)),put(jnp.int32(3))
y=program.execute(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
assert bool(np.asarray(y.state.decoder.contract_valid).all())
assert bool(np.asarray(y.state.finished))
assert int(np.asarray(y.next_token)[0])>=0
assert len(seen) >= len(config.full_index_slots), 'the program did not use the one-pass selector'
# An already-finished state must refuse another append atomically.
committed=y.state
z=program.execute(put(jnp.array([33]+[-1]*127,jnp.int32)),put(jnp.int32(1)),committed,inputs.weights,inputs.wk,
                  inputs.rope)
for field in ('kv_cache_local','index_cache_local','position','context_lengths'):
    np.testing.assert_array_equal(np.asarray(getattr(committed.decoder,field)),np.asarray(getattr(z.state.decoder,field)))
assert not bool(np.asarray(z.state.decoder.contract_valid).all())
assert np.asarray(z.next_token).tolist()==[-1]
print('complete prefill: healthy, one-pass selector, atomic refusal of a finished state')
# Long-context ownership uses the SAME body with exclusive state donation.
owned=jax.jit(program.execute,donate_argnums=(2,))
plain=program.execute(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
donated=owned(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
jax.block_until_ready((plain,donated))
for expected,actual in zip(jax.tree.leaves(plain),jax.tree.leaves(donated)):
    np.testing.assert_array_equal(np.asarray(expected),np.asarray(actual))
print('Exclusive prefill state donation preserves every output leaf')
"""


@pytest.mark.cpu32
def test_prefill_program_health_selector_refusal_and_donation_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr


SCHEMA = r"""
import json
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from glm_tpu.config.cache import CacheConfig
from glm_tpu.config.model import geometry
from glm_tpu.exceptions import PlanValidationError
from glm_tpu.models.glm_moe_dsa.prefill import build_prefill_program
from glm_tpu.models.glm_moe_dsa.weights import (
    bf16_resident_weights, bind_decoder_weights, decoder_weight_names, decoder_weight_specs,
)
from glm_tpu.runner.programs import build_program_set
import tests.fixtures
from tests.fixtures.tiny_model import cpu_mesh
mesh = cpu_mesh()
config = CacheConfig(geometry(), 8192, host_main_rope_table=True)  # as the runtime builds it (8,192 slots)
g = config.geometry
assert g.num_layers == 78
schema = json.loads(Path(tests.fixtures.__file__).with_name('prefill_layer_schema.json').read_text())
source = {t['name']: t for t in schema['tensor_schema']}
types = {'BF16': jnp.bfloat16, 'F32': jnp.float32, 'U8': jnp.uint8}
def spec_tuple(spec):
    spec = list(spec)
    while spec and spec[-1] is None:
        spec.pop()
    return tuple(spec)
pairs = jax.tree.leaves(jax.tree.map(lambda n, s: (n, s), decoder_weight_names(config), decoder_weight_specs(config)),
                        is_leaf=lambda x: isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str))
arrays, used = {}, set()
for name, spec in pairs:
    if name.startswith('model.layers.'):
        layer, suffix = int(name.split('.')[2]), name.split('.', 3)[3]
        # dense layers and every indexer tensor have layer 0's schema, MoE layers layer 3's
        template = f'model.layers.{0 if layer < 3 or ".indexer." in name else 3}.{suffix}'
        t = source[template]
        used.add(template)
        assert spec_tuple(spec) == spec_tuple(t['partition_spec']), (name, spec, t['partition_spec'])
        shape, dtype = tuple(t['global_shape']), types[t['dtype']]
    elif name == 'model.norm.weight':
        shape, dtype = (g.hidden_size,), jnp.bfloat16
    else:
        shape, dtype = (g.vocab_size, g.hidden_size), jnp.bfloat16
    arrays[name] = jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))
assert used == set(source), sorted(set(source) - used)
raw = bind_decoder_weights(arrays, config)
resident = jax.eval_shape(lambda w: bf16_resident_weights(mesh, config, w), raw)
programs = build_program_set(mesh, config)
state = jax.eval_shape(programs.cache_init.fn, jax.ShapeDtypeStruct((), jnp.int32))
wk = tuple(jax.ShapeDtypeStruct((g.dsa_indexer_head_dim, g.hidden_size), jnp.float32) for _ in config.full_index_slots)
rope = jax.ShapeDtypeStruct(config.main_rope_table_shape, jnp.bfloat16)
for rows, spec in programs.prefill.items():
    out = jax.eval_shape(spec.fn, jax.ShapeDtypeStruct((rows,), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32),
                         state, resident, wk, rope)
    decoder = out.state.decoder
    assert decoder.kv_cache_local.shape == config.kv_cache_shape, rows
    assert decoder.index_cache_local.shape == out.state.repaired_index_local.shape == config.index_cache_shape, rows
    assert decoder.selected_positions.shape == (1, g.dsa_top_k) and out.next_token.shape == (1,), rows
    assert out.state.finished.shape == () and out.state.finished.dtype == jnp.bool_, rows
for changed in (replace(config, exact_dsa=True), replace(config, strategy_nd_dense=True),
                replace(config, host_main_rope_table=False)):
    try:
        build_prefill_program(mesh, changed, block_rows=128)
        raise AssertionError('unsupported config accepted')
    except PlanValidationError:
        pass
for rows in (0, 33, True, 129):
    try:
        build_prefill_program(mesh, config, block_rows=rows)
        raise AssertionError('invalid row count accepted')
    except PlanValidationError:
        pass
print('78-layer schema bound and both prefill programs evaluated')
"""


@pytest.mark.cpu32
def test_prefill_programs_accept_the_78_layer_checkpoint_schema_cpu32():
    """The production config's tensor names take the checkpoint's recorded shapes, dtypes and partition specs
    (``tests/fixtures/prefill_layer_schema.json``: layer 0 for the dense layers and every indexer tensor, layer 3
    for the MoE layers); bound, made resident and run through the B128 and B114 programs by shape evaluation only
    (no weight is allocated). Other block sizes and research configs are refused."""
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", SCHEMA], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
