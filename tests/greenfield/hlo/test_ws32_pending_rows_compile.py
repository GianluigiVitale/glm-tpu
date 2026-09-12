"""Preparation of changed E0 only. No TPU/worker or runtime-fit admission."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from scripts.greenfield import ws32_pending_rows_compile as candidate
from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_owned_state_compile as owned
from scripts.greenfield import ws32_rolled_prefill_compile as original


ROOT = Path(__file__).resolve().parents[3]


def test_new_source_recipe_and_historical_refusals():
    candidate.require_source(ROOT)
    for old in (canonical, owned):
        with pytest.raises(ValueError, match="source/prerequisite"):
            old.require_source(ROOT)


@pytest.mark.parametrize("path", list(candidate.MODEL_SOURCE_OVERRIDES) + list(candidate.PREREQUISITES))
def test_changed_source_or_prerequisite_refuses(tmp_path, path):
    # Read real current files except one mutated content; never edit the repo.
    for name in {**candidate.MODEL_SOURCE_OVERRIDES, **candidate.PREREQUISITES}:
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes() + (b"\nchanged" if name==path else b""))
    with pytest.raises(ValueError,match="source/prerequisite"):
        candidate.require_source(tmp_path)


def test_modes_refuse_before_metadata_or_backend():
    for bad in (1,None,"true"):
        with pytest.raises(ValueError,match="static bool"):
            original.read_metadata(ROOT,full_canonical=True,pending_cache_rows=bad)
        with pytest.raises(ValueError,match="static bool"):
            original.prepare(None,None,repo=ROOT,full_canonical=True,pending_cache_rows=bad)
    with pytest.raises(ValueError,match="full canonical source"):
        original.read_metadata(ROOT,pending_cache_rows=True)
    for label in (None,"128k_d1_0"):
        with pytest.raises(ValueError,match="restricted to canonical E0"):
            original.prepare(None,None,repo=ROOT,full_canonical=True,
                             pending_cache_rows=True,long_context_label=label)


def test_pending_option_reaches_both_uncompiled_builders(monkeypatch):
    from dataclasses import replace
    from scripts.greenfield import ws32_batched_prefill_runner as adapter
    from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config
    cfg=replace(config(),exact_dsa=False,strategy_nd_dense=False)
    plan=adapter.BatchedPrefillPlan(2034,128,8192,mlp_window=True)
    calls=[]
    monkeypatch.setattr(adapter,"build_ws32_batched_prefill_program",
                        lambda *a,**kw:calls.append(kw) or object())
    adapter.build_graph_pair(None,cfg,plan,pending_cache_rows=True)
    assert len(calls)==2 and all(c["pending_cache_rows"] is True for c in calls)
    for bad in (1,None,"true"):
        with pytest.raises(ValueError,match="static bool"):
            adapter.build_graph_pair(None,cfg,plan,pending_cache_rows=bad)


def test_production_e0_lowering_without_payload_compile_or_dispatch():
    source=r'''
from pathlib import Path
from hashlib import sha256
from unittest.mock import patch
import jax, numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_pending_rows_compile as candidate
assert jax.default_backend()=='cpu'
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
old=Path.open
def checked(path,*args,**kwargs):
    assert path.suffix not in ('.safetensors','.bin'),path
    if '/glm-ws32-runtime/' in str(path):assert path.name in ('manifest.json','SUCCESS'),path
    return old(path,*args,**kwargs)
with patch.object(Path,'open',checked), \
     patch('jax.device_put',side_effect=AssertionError('payload placement')), \
     patch('jax.stages.Compiled.__call__',side_effect=AssertionError('dispatch')), \
     patch('jax.stages.Lowered.compile',side_effect=AssertionError('compilation')):
    metadata=candidate.read_metadata(Path.cwd())
    pair=candidate.prepare(mesh,metadata,repo=Path.cwd())
    assert len(metadata.manifest['tensor_schema'])==2310
    assert set(pair.programs)==set(pair.inputs)=={candidate.PROGRAM}
    name=candidate.PROGRAM
    program=pair.programs[name]
    assert program.original_program.pending_cache_rows is True
    assert program.original_program.canonical_dense is True
    ids,count,state,weights,wk,rope=pair.inputs[name]
    assert ids.shape==(128,) and state.decoder.block_tables.shape==(1,513)
    assert rope.shape==(262656,64) and len(weights.layers)==78
    assert all(isinstance(v,jax.ShapeDtypeStruct) and v.sharding is not None
               for v in jax.tree.leaves(pair.inputs[name]))
    traced=program.execute.trace(*pair.inputs[name])
    assert traced.donate_argnums==tuple(range(2,2+len(jax.tree.leaves(state))))
    with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
        module=traced.lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')
        raw=str(module).encode()
    print('PENDING_ROWS_E0_RAW',len(raw),sha256(raw).hexdigest(),flush=True)
    assert (len(raw),sha256(raw).hexdigest())==candidate.RAW[name]
    assert b'jax.buffer_donor' in raw and b'stablehlo.scatter' in raw
    main=next(op for op in module.body.operations
              if op.operation.name=='func.func' and op.attributes['sym_name'].value=='main')
    donors=tuple(i for i,a in enumerate(main.attributes['arg_attrs'])
                 if 'jax.buffer_donor' in a or 'tf.aliasing_output' in a)
    assert donors==traced.donate_argnums,donors
print('PENDING_ROWS_E0_PREPARATION_PASS',flush=True)
'''
    result=subprocess.run([sys.executable,"-c",source],cwd=ROOT,capture_output=True,text=True,
        timeout=300,env=dict(os.environ,JAX_PLATFORMS="cpu",
                            XLA_FLAGS="--xla_force_host_platform_device_count=32"))
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
    assert "PENDING_ROWS_E0_PREPARATION_PASS" in result.stdout
