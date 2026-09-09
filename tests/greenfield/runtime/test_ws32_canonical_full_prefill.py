"""Full-runtime corrective wiring: CPU semantics, not TPU/8K admission."""

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[3]


def helpers():
    return runpy.run_path(
        str(ROOT / "tests/greenfield/runtime/test_ws32_paired_prefill.py")
    )


def test_canonical_full_runtime_cpu32():
    source = r"""
import jax,jax.numpy as jnp,numpy as np
from jax import lax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(x,s=P()):return jax.device_put(x,NamedSharding(mesh,s))
wk=tuple(put(x) for x in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
opts=dict(mlp_window=True,rolled_prefix=True,expert_panels=True,
    paired_position_sort=True,sorted_local_merge=True,key_tile=128,
    sparse_attention_interpret=True,linear_interpret=True)
programs={}
original=b.ws32_prefill_layer_window_mapped
seen=[]
def check_layer(*args,**kwargs):
    dense,moe=args[16:18]
    enabled=kwargs.get('canonical_dense',False)
    seen.append((dense is not None,moe is not None,enabled))
    assert enabled==(dense is not None)
    return original(*args,**kwargs)
for rows in (128,114):
    programs[False,rows]=b.build_ws32_batched_prefill_program(mesh,config,
        block_rows=rows,**opts)
    programs[True,rows]=b.build_ws32_batched_prefill_program(mesh,config,
        block_rows=rows,canonical_dense=True,**opts)
    assert not programs[False,rows].canonical_dense
    assert programs[True,rows].canonical_dense
def same(x,y):
    for a,c in zip(jax.tree.leaves(x),jax.tree.leaves(y),strict=True):
        a,c=np.asarray(a),np.asarray(c)
        assert a.shape==c.shape and a.dtype==c.dtype and a.tobytes()==c.tobytes()
def run(flag,rows,tokens,count,state,table):
    b.ws32_prefill_layer_window_mapped=check_layer if flag else original
    try:return programs[flag,rows].execute(tokens,put(jnp.int32(count)),state,weights,wk,table)
    finally:b.ws32_prefill_layer_window_mapped=original
def failed(out,old):
    assert not np.asarray(out.state.decoder.contract_valid).any()
    assert out.next_token.tolist()==[-1]
    same(out.state,old._replace(decoder=old.decoder._replace(
        contract_valid=put(jnp.zeros(1,jnp.bool_)))))
tokens=put(jnp.arange(128,dtype=jnp.int32)+30)
# Synthetic populated-prefix state crosses a page boundary; not a resume claim.
base=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=747)
base=base._replace(decoder=base.decoder._replace(
    position=put(jnp.array([505],jnp.int32)),
    context_lengths=put(jnp.array([506],jnp.int32))))
first=[run(flag,128,tokens,128,base,rope) for flag in (False,True)]
same(*first)
assert first[1].state.decoder.position.tolist()==[633]
assert not bool(first[1].state.finished) and first[1].next_token.tolist()==[-1]
assert np.asarray(first[1].state.decoder.contract_valid).all()
assert seen==[(True,False,True)]*3+[(False,True,False)]*5,seen
seen.clear()
# Physical B114: full2K tail, then own8K's91live+23padding and empty final tile.
for count in (114,91):
    states=[s.state._replace(prompt_length=put(jnp.int32(633+count))) for s in first]
    tail=put(jnp.concatenate((jnp.arange(count,dtype=jnp.int32)+70,
        jnp.full(114-count,-2147483648,jnp.int32))))
    poisoned=rope.at[633+count:747].set(jnp.nan)
    last=[run(flag,114,tail,count,s,poisoned) for flag,s in zip((False,True),states)]
    same(*last)
    assert bool(last[1].state.finished)
    assert last[1].state.decoder.position.tolist()==[633+count]
    b.finish_ws32_batched_prefill(last[1])
    np.testing.assert_array_equal(last[1].state.decoder.index_cache_local,last[1].state.repaired_index_local)
    clean=run(True,114,tail,count,states[1],rope)
    same(last[1],clean)
    # Candidate must not consume repaired history during prompt evaluation.
    altered=states[1]._replace(repaired_index_local=jnp.full_like(states[1].repaired_index_local,-9))
    changed=run(True,114,tail,count,altered,poisoned)
    for field in ('kv_cache_local','selected_positions','selected_scores'):
        same(getattr(changed.state.decoder,field),getattr(last[1].state.decoder,field))
    same(changed.next_token,last[1].next_token)
    failed(run(True,114,tail.at[count-1].set(-1),count,states[1],rope),states[1])
    failed(run(True,114,tail,count,last[1].state,rope),last[1].state)
assert seen==[(True,False,True)]*3+[(False,True,False)]*5,seen
# Late dense-layer2 single-owner failure must survive MoE and refuse atomically.
calls=[]
def poison_layer(*args,**kwargs):
    result=original(*args,**kwargs)
    layer=len(calls);calls.append(layer)
    if layer==2:
        bad=(lax.axis_index('expert')==3)&(lax.axis_index('feature')==2)
        result=result._replace(contract_valid=result.contract_valid.at[100].set(
            result.contract_valid[100]&~bad))
    return result
b.ws32_prefill_layer_window_mapped=poison_layer
poison=b.build_ws32_batched_prefill_program(mesh,config,block_rows=114,canonical_dense=True,**opts)
state=first[1].state
tail=put(jnp.arange(114,dtype=jnp.int32)+70)
failed(poison.execute(tail,put(jnp.int32(114)),state,weights,wk,rope),state)
assert len(calls)==8
print('CANONICAL_FULL_RUNTIME_CPU32_PASS',flush=True)
"""
    assert "CANONICAL_FULL_RUNTIME_CPU32_PASS" in helpers()["_cpu"](source, timeout=480)


@pytest.mark.parametrize("bad", [1, None, "true"])
def test_canonical_flag_requires_static_bool(bad):
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import _require_window_options

    with pytest.raises(ValueError, match="static booleans"):
        _require_window_options(True, True, True, True, bad)


@pytest.mark.parametrize("missing", ["mlp_window", "rolled_prefix", "expert_panels"])
def test_canonical_flag_requires_original_window(missing):
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import _require_window_options

    options = dict(
        mlp_window=True,
        rolled_prefix=True,
        expert_panels=True,
        sorted_local_merge=True,
        canonical_dense=True,
    )
    options[missing] = False
    with pytest.raises(ValueError):
        _require_window_options(**options)


@pytest.mark.parametrize("rows", [32, 113, 115])
def test_builder_refuses_unsupported_physical_rows(rows):
    from dataclasses import replace
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        build_ws32_batched_prefill_program,
    )
    from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config

    cfg = replace(config(), exact_dsa=False, strategy_nd_dense=False)
    with pytest.raises(ValueError, match="B114/B128"):
        build_ws32_batched_prefill_program(
            None,
            cfg,
            block_rows=rows,
            mlp_window=True,
            rolled_prefix=True,
            expert_panels=True,
            canonical_dense=True,
        )


def test_production_pair_lowering_and_original_false_identity():
    source = helpers()["_fixture_source"](
        "test_production_78_layer_schema_without_allocating_weights"
    ).split("plan=adapter.BatchedPrefillPlan", 1)[0]
    source += r"""
from hashlib import sha256
from unittest.mock import patch
from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield import ws32_canonical_prefill_compile as candidate
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
plan=admission.ROLLED_PLAN
options=admission.short_program_options(admission.ROLLED_SHORT_PROFILE)
original=admission.rolled_registration(Path.cwd())['graphs']
records={}
for enabled in (False,True):
    programs=adapter.build_graph_pair(mesh,config,plan,canonical_dense=enabled,**options)
    for graph,rows in plan.graph_rows:
        assert programs[graph].canonical_dense is enabled
        inputs=(abstract((rows,),jnp.int32),abstract((),jnp.int32),state,weights,wk,rope)
        with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
            raw=str(programs[graph].execute.trace(*inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
        item=dict(stablehlo_sha256=sha256(raw).hexdigest(),stablehlo_bytes=len(raw))
        if not enabled:assert item==original[graph],(graph,item,original[graph])
        else:
            assert item!=original[graph]
            assert (item['stablehlo_bytes'],item['stablehlo_sha256'])==candidate.RAW[graph]
        records[f'{enabled}:{graph}']=item
        print('CANONICAL_PRODUCTION_GRAPH',enabled,graph,json.dumps(item),flush=True)
print('CANONICAL_PRODUCTION_RECORDS='+json.dumps(records,sort_keys=True),flush=True)
"""
    output = helpers()["_cpu"](source, timeout=420)
    print(output)
    assert output.count("CANONICAL_PRODUCTION_GRAPH") == 4
