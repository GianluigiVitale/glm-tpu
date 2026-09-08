"""Paired full-engine opt-in: CPU wiring/exactness and offline TPU graph pins."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def _fixture_source(name):
    tree = ast.parse(
        (ROOT / "tests/greenfield/runtime/test_ws32_batched_prefill.py").read_text()
    )
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return next(
        ast.literal_eval(n.value)
        for n in fn.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "code" for t in n.targets)
    )


def _cpu(source, timeout=300):
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_default_and_paired_full_state_two_blocks_cpu32():
    source = _fixture_source(
        "test_layer_major_two_chunk_state_and_decode_handoff_cpu32"
    )
    # Reuse the actual eight-layer fixture (all four layer kinds), not a second
    # model. Two full trees compare before and after final repaired-cache swap.
    source = source.split("# Independently wire each layer", 1)[0]
    source += r"""
assert program.paired_position_sort is False and program.mlp_window is False
paired=b.build_ws32_batched_prefill_program(mesh,config,block_rows=17,key_tile=128,sparse_attention_interpret=True,linear_interpret=True,paired_position_sort=True)
assert paired.paired_position_sort is True and paired.mlp_window is False
actual=paired.execute(tokens,put(jnp.int32(17)),state,weights,wk,rope)
def same(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b),strict=True):
        x,y=np.asarray(x),np.asarray(y)
        assert x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()
same(actual,first)
tail=put(jnp.arange(11,dtype=jnp.int32)+60)
programs=[b.build_ws32_batched_prefill_program(mesh,config,block_rows=11,key_tile=128,sparse_attention_interpret=True,linear_interpret=True,paired_position_sort=flag) for flag in (False,True)]
last=[p.execute(tail,put(jnp.int32(11)),s,weights,wk,rope) for p,s in zip(programs,(first.state,actual.state))]
same(*last)
assert bool(last[1].state.finished)
assert np.asarray(last[1].state.decoder.contract_valid).all()
for bad in (1,None,"true"):
    try:b.build_ws32_batched_prefill_program(mesh,config,block_rows=17,paired_position_sort=bad)
    except ValueError:pass
    else:raise AssertionError('nonbool sort flag accepted')
print('PAIRED_FULL_STATE_CPU32_PASS')
"""
    assert "PAIRED_FULL_STATE_CPU32_PASS" in _cpu(source)


def test_production_offline_raw_stablehlo_preregistration():
    source = _fixture_source(
        "test_production_78_layer_schema_without_allocating_weights"
    )
    source = source.split("plan=adapter.BatchedPrefillPlan", 1)[0]
    source += r"""
from hashlib import sha256
from unittest.mock import patch
from glm_tpu.greenfield.validation.ws32_prefill_admission import short_acquisition
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
plan=adapter.BatchedPrefillPlan(2034,17,8192)
original=short_acquisition(Path.cwd())['graphs']
records={}
for flag in (False,True):
    programs=adapter.build_graph_pair(mesh,config,plan,paired_position_sort=flag)
    for graph,rows in plan.graph_rows:
        inputs=(abstract((rows,),jnp.int32),abstract((),jnp.int32),state,weights,wk,rope)
        # Offline only: CPU target lacks the live TPU version probe. Current
        # Mosaic serialization reproduces the original TPU-v4 raw graphs.
        with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
            raw=str(programs[graph].execute.trace(*inputs).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
        record=dict(stablehlo_sha256=sha256(raw).hexdigest(),stablehlo_bytes=len(raw))
        print(json.dumps(dict(paired=flag,graph=graph,**record)),flush=True)
        if not flag:assert record['stablehlo_sha256']==original[graph]['stablehlo_sha256']
        records[f'{flag}:{graph}']=record
print('PAIRED_FULL_PREREGISTRATION='+json.dumps(records,sort_keys=True),flush=True)
"""
    output = _cpu(source, timeout=420)
    print(output)
    records = json.loads(output.split("PAIRED_FULL_PREREGISTRATION=", 1)[1])
    for graph in ("prefill_chunk", "prefill_tail"):
        assert records[f"False:{graph}"] != records[f"True:{graph}"]
