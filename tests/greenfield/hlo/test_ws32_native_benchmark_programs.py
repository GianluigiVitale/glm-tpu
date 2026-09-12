"""Sampled production metadata/graph preparation, not actual TPU compilation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.greenfield import ws32_native_benchmark_programs as native
from scripts.greenfield import ws32_rolled_prefill_compile as original

ROOT = Path(__file__).resolve().parents[3]


def test_native_source_and_old_profile_guards_remain_distinct():
    native.require_source(ROOT)
    with pytest.raises(ValueError, match="source|frozen"):
        original.read_metadata(ROOT, full_canonical=True, pending_cache_rows=True,
                               flat_pending_rows=True, capture_barrier=True)
    with pytest.raises(ValueError, match="complete explicit"):
        original.read_metadata(ROOT, native_benchmark=True)
    with pytest.raises(ValueError, match="complete explicit"):
        original.prepare(None, None, repo=ROOT, native_benchmark=True)


@pytest.mark.parametrize('graph', ['decode', 'prefill_chunk', 'unregistered'])
def test_native_inspector_rejects_unregistered_raw_before_parsing(monkeypatch, graph):
    from glm_tpu.greenfield.sharding import hlo_contract
    monkeypatch.setattr(hlo_contract, 'parse_hlo_module', lambda _:pytest.fail('parsed unbound graph'))
    with pytest.raises(ValueError, match='RAW'):
        native.inspect_hlo('changed', 'not optimized evidence', repo=ROOT,
                           graph=graph, expected_optimized='0'*64)


def test_registered_materializers_and_probe_stay_original():
    from scripts.greenfield.ws32_delivery_companions import raw_registration
    originals = raw_registration('256k_e0')
    for name in ('exact_materialize', 'exact_promote', 'cache_probe'):
        assert native.RAW[name][1] == originals[name]


def test_compiled_cache_initializer_equals_original_and_production_lowering():
    code = r'''
from dataclasses import replace
from hashlib import sha256
from unittest.mock import patch
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scripts.greenfield.run_short_decoder_ws32 import _geometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from glm_tpu.greenfield.runtime.ws32_batched_prefill import make_ws32_batched_prefill_state
from scripts.greenfield.ws32_native_benchmark_programs import build_cache_initializer, RAW
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
g=replace(_geometry(),num_layers=3,mlp_layer_types=('dense',)*3,indexer_types=('full',)*3)
config=Ws32DecoderConfig(g,512,host_main_rope_table=True)
program=build_cache_initializer(mesh,config)
def prompt(n): return jax.device_put(jnp.int32(n),NamedSharding(mesh,P()))
compiled=program.lower(prompt(17)).compile()
assert compiled.memory_analysis().alias_size_in_bytes==0
for count in (17,19):
    actual=compiled(prompt(count))
    original=make_ws32_batched_prefill_state(mesh,config,prompt_length=count)
    pointers={}
    for leaf in jax.tree.leaves(actual):
        for shard in leaf.addressable_shards:
            pointer=shard.data.unsafe_buffer_pointer()
            key=(shard.device.id,pointer)
            assert key not in pointers, 'initializer aliased distinct donated state leaves'
            pointers[key]=True
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(original),strict=True):
        np.testing.assert_array_equal(a,b)
        assert a.sharding.is_equivalent_to(b.sharding,a.ndim)
bad=compiled(prompt(0))
assert not np.asarray(bad.decoder.contract_valid).any()
production=build_cache_initializer(mesh,Ws32DecoderConfig(_geometry(),262656,host_main_rope_table=True))
abstract=jax.ShapeDtypeStruct((),jnp.int32,sharding=NamedSharding(mesh,P()))
with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
    raw=str(production.trace(abstract).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
print('CACHE_INIT_PRODUCTION_RAW',len(raw),sha256(raw).hexdigest(),flush=True)
assert (len(raw),sha256(raw).hexdigest())==RAW['cache_init']
print('CACHE_INIT_CPU_VALUE_AND_SHARDING_PASS',flush=True)
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        text=True,capture_output=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
    assert 'CACHE_INIT_CPU_VALUE_AND_SHARDING_PASS' in result.stdout


def test_actual_78_layer_abstract_sampled_prefill_lowering():
    manifest = Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/manifest.json')
    if not manifest.is_file():
        pytest.skip('requires existing manifest only, never checkpoint payload')
    code = r'''
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
from unittest.mock import patch
import json
import ast
import re
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_native_benchmark_programs as native
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
# Source checkpoint metadata was independently authenticated by read_metadata.
# This test reuses its existing schema, not another full inventory verification.
metadata = SimpleNamespace(manifest=json.loads(Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/manifest.json').read_text()))
with patch('jax.device_put', side_effect=AssertionError('payload allocation')):
    prepared = native.prepare(mesh, metadata, repo=Path.cwd())
assert set(prepared.programs) == set(prepared.inputs) == {'prefill_chunk','prefill_tail'}
assert len(metadata.manifest['tensor_schema']) == 2310
assert all(isinstance(v,jax.ShapeDtypeStruct) for args in prepared.inputs.values() for v in jax.tree.leaves(args))
tpu_info.registry['cpu'] = lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
for role, rows in native.PLAN.graph_rows:
    program, args = prepared.programs[role], prepared.inputs[role]
    assert len(args)==7 and args[-1].shape==() and args[-1].dtype==jnp.float32
    assert args[0].shape==(rows,) and args[2].decoder.kv_cache_local.shape[1]==513
    assert program.ownership_contract==native.OWNED_STATE_CONTRACT
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        raw=str(program.execute.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
    assert (len(raw),sha256(raw).hexdigest())==native.RAW[role]
    # Normal RAW printing omits source debug scopes. Prove the actual output
    # collective shape/groups, not the presence of an optional profiler label.
    gathers=[line for line in raw.decode().splitlines()
             if 'stablehlo.all_gather' in line and 'tensor<1x154880xbf16>' in line]
    assert len(gathers)==1, gathers
    assert 'tensor<1x19360xbf16>' in gathers[0], gathers[0]
    groups=re.search(r'replica_groups = dense<(.*?)> : tensor<4x8xi64>',gathers[0])
    assert groups is not None, gathers[0]
    assert ast.literal_eval(groups.group(1))==[list(range(f,32,4)) for f in range(4)]
    print(json.dumps(dict(role=role,rows=rows,raw_bytes=len(raw),raw_sha256=sha256(raw).hexdigest(),
                         input_leaves=len(jax.tree.leaves(args)),tpu_execution=False)),flush=True)
print('NATIVE_PRODUCTION_ABSTRACT_SAMPLED_PAIR_PASS',flush=True)
'''
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=ROOT,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        text=True,capture_output=True,timeout=300,
    )
    assert result.returncode==0, result.stdout+result.stderr
    print(result.stdout)
    assert 'NATIVE_PRODUCTION_ABSTRACT_SAMPLED_PAIR_PASS' in result.stdout


def test_actual_production_sampled_decode_and_materializer_operands():
    code = r'''
from pathlib import Path
from types import SimpleNamespace
from hashlib import sha256
from unittest.mock import patch
import json, ast, re
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_native_benchmark_programs as native
assert jax.default_backend() == 'cpu'
mesh = Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
metadata = SimpleNamespace(manifest=json.loads(Path('/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z/manifest.json').read_text()))
tpu_info.registry['cpu'] = lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
with patch('jax.device_put',side_effect=AssertionError('payload allocation')), \
     patch('jax.stages.Lowered.compile',side_effect=AssertionError('TPU compilation')):
    pair = native.prepare(mesh, metadata, repo=Path.cwd())
    companion = native.prepare_companions(mesh, pair, repo=Path.cwd())
    assert set(companion.programs)==set(companion.inputs)=={'exact_materialize','exact_promote','decode','observer','cache_probe'}
    for role, args in companion.inputs.items():
        assert all(isinstance(v,jax.ShapeDtypeStruct) and v.sharding is not None for v in jax.tree.leaves(args))
        traced = companion.programs[role].trace(*args)
        donors = tuple(range(1,1+len(jax.tree.leaves(args[1])))) if role in ('decode','observer') else ()
        assert traced.donate_argnums==donors, (role,traced.donate_argnums,donors)
        with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
            raw=str(traced.lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
        assert (len(raw),sha256(raw).hexdigest())==native.RAW[role]
        if role in ('decode','observer'):
            assert args[0].shape==(1,) and args[-1].shape==() and args[-1].dtype==jnp.float32
            gathers=[line for line in raw.decode().splitlines() if 'stablehlo.all_gather' in line and 'tensor<1x154880xbf16>' in line]
            assert len(gathers)==1, gathers
            groups=re.search(r'replica_groups = dense<(.*?)> : tensor<4x8xi64>',gathers[0])
            assert groups is not None
            assert ast.literal_eval(groups.group(1))==[list(range(f,32,4)) for f in range(4)]
        print(json.dumps(dict(role=role,raw_bytes=len(raw),raw_sha256=sha256(raw).hexdigest(),
                             donated_leaves=len(donors),tpu_execution=False)),flush=True)
print('NATIVE_PRODUCTION_COMPANIONS_PASS',flush=True)
'''
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=ROOT,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        text=True,capture_output=True,timeout=240,
    )
    assert result.returncode==0, result.stdout+result.stderr
    print(result.stdout)
    assert 'NATIVE_PRODUCTION_COMPANIONS_PASS' in result.stdout
