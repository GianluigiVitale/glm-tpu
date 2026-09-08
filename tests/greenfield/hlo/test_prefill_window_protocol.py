"""Window schema/host refusal tests; synthetic arrays are not TPU evidence."""

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.greenfield import prefill_window_protocol as p
from scripts.greenfield.prefill_layer_evidence import owner_inputs, encode_arrays
from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host


def _fixture(case="competitive", slot=0):
    host = p.host_case(
        case, build_rotary_table_host(p.CAPACITY, rotary_dim=64, theta=8e6)
    )
    own = owner_inputs(host, slot)
    out = dict(
        output=np.zeros((128, 1536), p.BF16),
        residual=np.zeros((128, 1536), p.BF16),
        normalized=np.zeros((128, 1536), p.BF16),
        **{
            name: own[name].copy()
            for name in (
                "kv",
                "index",
                "repair",
                "positions",
                "counts",
                "scores",
                "health",
            )
        },
        routes=np.tile(np.arange(8, dtype=np.int32), (128, 1)),
        route_weights=np.zeros((128, 8), np.float32),
    )
    offset, count = p.CASES[case]
    out["route_weights"][:count] = 1 / 8
    for row in range(count):
        n = min(offset + row + 1, 2048)
        out["positions"][row, :n] = np.arange(n)
        out["counts"][row] = n
        out["scores"][row, :n] = 0
    for q, page, row in p.written_addresses(slot, offset, count):
        for name in ("kv", "index", "repair"):
            out[name][page, row] = 0
    return host, own, out


@pytest.mark.parametrize("case", p.CASES)
@pytest.mark.parametrize("slot", [0, 4, 8, 16, 28])
def test_fixed_case_and_written_owner_admission(case, slot):
    host, own, out = _fixture(case, slot)
    result = p.compare_case(out, out, own, slot=slot, case=case)
    assert result["passed"] and result["performance_claim"] is False
    assert host["table"].shape == (1, 8)
    assert result["written_rows_on_owner"] == len(
        p.written_addresses(slot, *p.CASES[case])
    )


@pytest.mark.parametrize(
    "failure",
    [
        "route_order",
        "selection",
        "tie_order",
        "score_order",
        "duplicate",
        "future",
        "count",
        "health",
        "output",
        "nonfinite",
        "untouched",
        "cache_row",
        "kv_padding",
        "weights",
    ],
)
def test_mutations_refuse(failure):
    _, own, reference = _fixture()
    out = {k: v.copy() for k, v in reference.items()}
    if failure == "route_order":
        out["routes"][0, :2] = out["routes"][0, 1::-1]
    elif failure == "selection":
        out["positions"][0, -1] = 2500
    elif failure == "tie_order":
        out["positions"][0, :2] = [1, 0]
    elif failure == "score_order":
        out["scores"][0, 1] = 1
    elif failure == "duplicate":
        out["positions"][0, 1] = 0
    elif failure == "future":
        out["positions"][0, -1] = 2554
    elif failure == "count":
        out["counts"][0] -= 1
    elif failure == "health":
        out["health"][110] = False
    elif failure == "output":
        out["output"][110, 0] = 100
    elif failure == "nonfinite":
        out["normalized"][110, 0] = np.nan
    elif failure == "untouched":
        out["repair"][7, 0, 0] = 1
    elif failure == "cache_row":
        _, page, row = p.written_addresses(0, *p.CASES["competitive"])[0]
        out["index"][page, row, 0] = 100
    elif failure == "kv_padding":
        _, page, row = p.written_addresses(0, *p.CASES["competitive"])[0]
        out["kv"][page, row, 639] = 1
    elif failure == "weights":
        out["route_weights"][0, 0] = 2
    try:
        verdict = p.compare_case(out, reference, own, slot=0, case="competitive")
    except ValueError:
        return
    assert not verdict["passed"]


def test_tail_padding_refuses_and_control_stacks_last_cache():
    _, own, out = _fixture("tail")
    bad = {k: v.copy() for k, v in out.items()}
    bad["output"][33, 0] = 1
    with pytest.raises(ValueError, match="padded"):
        p.compare_case(bad, out, own, slot=0, case="tail")
    blocks = []
    for i in range(4):
        blocks.append(
            {
                name: (
                    v.copy()
                    if name in ("kv", "index", "repair")
                    else v[i * 32 : (i + 1) * 32].copy()
                )
                for name, v in out.items()
            }
        )
        blocks[-1]["repair"].fill(i)
    joined = p.stack_control(blocks)
    assert np.all(joined["repair"] == 3)
    np.testing.assert_array_equal(joined["counts"], out["counts"])
    with pytest.raises(ValueError):
        p.stack_control(blocks[:3])


def test_per_row_failure_cannot_hide_in_whole_window_mean():
    _, own, reference = _fixture()
    out = {k: v.copy() for k, v in reference.items()}
    out["output"][110] = np.asarray(0.03125, dtype=p.BF16)
    verdict = p.compare_case(out, reference, own, slot=0, case="competitive")
    assert verdict["comparisons"]["output"]["aggregate"]["passed"]
    assert not verdict["comparisons"]["output"]["per_row"][110]["passed"]
    assert not verdict["passed"]


def test_actual_npz_replay_refuses_fitted_input_missing_owner_and_extra(tmp_path):
    slots = {101: 0, 109: 4, 117: 8, 125: 12}
    host, _, _ = _fixture()
    arrays = encode_arrays("input", host)
    for device, slot in slots.items():
        _, _, values = _fixture(slot=slot)
        for kind in ("actual", "control"):
            arrays.update(encode_arrays(f"{kind}_{device}", values))
    path = tmp_path / "original.npz"
    np.savez_compressed(path, **arrays)
    assert p.replay_case(path, case="competitive", slots_by_device=slots)["passed"]
    with pytest.raises(ValueError, match="four distinct"):
        p.replay_case(
            path, case="competitive", slots_by_device={101: 0, 109: 0, 117: 8, 125: 12}
        )
    arrays["extra"] = np.zeros(1)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="inventory"):
        p.replay_case(path, case="competitive", slots_by_device=slots)
    del arrays["extra"]
    omitted = arrays.pop("control_125__repair")
    np.savez_compressed(path, **arrays)
    with pytest.raises(KeyError):
        p.replay_case(path, case="competitive", slots_by_device=slots)
    arrays["control_125__repair"] = omitted
    arrays["input__update"] = arrays["input__update"].copy()
    arrays["input__update"][0, 0] ^= 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="original inputs"):
        p.replay_case(path, case="competitive", slots_by_device=slots)


def test_actual_window_layer6_programs_and_three_cache_carry_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from scripts.greenfield import prefill_window_protocol as p
from scripts.greenfield.probe_ws32_prefill_layer import input_specs,device_inputs
from scripts.greenfield.prefill_layer_evidence import local_observations
from glm_tpu.greenfield.runtime.ws32_batched_prefill import make_ws32_batched_prefill_state
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh);w=weights.layers[6]
assert w.dsa is not None and w.moe is not None and w.dense is None
put=lambda x,s=P():jax.device_put(x,NamedSharding(mesh,s))
k=put(wk[3]); specs=input_specs(w,k)
wide,small=p.build_programs(mesh,specs,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract,linear_interpret=True,sparse_attention_interpret=True)
rng=np.random.RandomState(77)
host={
 'update':np.asarray(rng.randn(128,512)*.01,dtype='bfloat16'),
 'residual':np.asarray(rng.randn(128,512)*.01,dtype='bfloat16'),
 'kv':np.asarray(rng.randn(8,3,64,640)*.01,dtype='bfloat16'),
 'index':np.asarray(rng.randn(8,3,64,128)*.01,dtype='bfloat16'),
 'repair':np.full((8,3,64,128),7,dtype='bfloat16'),
 'positions':np.full((128,128),-1,np.int32),'counts':np.zeros(128,np.int32),
 'scores':np.full((128,128),-np.inf,np.float32),'offset':np.asarray(505,np.int32),
 'count':np.asarray(128,np.int32),'table':np.asarray([[2,0,1]],np.int32),
 'health':np.ones((8,4,128),np.bool_),
 'rope':build_ws32_main_rope_table(config)[505:633],
}
values=device_inputs(host,specs,w,k,mesh)
actual=wide(*values);jax.block_until_ready(actual)
control=[];previous=None
for tile in range(4):
    inputs=p.control_inputs(values,tile,previous)
    if previous is not None:
        for i in (2,3,4):assert inputs[i] is previous[i]
    previous=small(*inputs);jax.block_until_ready(previous)
    control.append(local_observations(previous))
observed=local_observations(actual)
for device,fields in observed.items():
    expected=p.stack_control([c[device] for c in control])
    for name in fields:np.testing.assert_array_equal(fields[name],expected[name],err_msg=name)
assert len(observed)==32 and all(v['health'].all() for v in observed.values())
assert any(not np.array_equal(v['index'],v['repair']) for v in observed.values())
print('WINDOW_LAYER6_CPU32_API_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WINDOW_LAYER6_CPU32_API_PASS" in result.stdout


def test_production_layer6_window_schema_without_weight_allocation():
    code = r"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import prefill_window_protocol as p
from scripts.greenfield.probe_ws32_prefill_layer import input_specs,device_inputs
from scripts.greenfield.run_short_decoder_ws32 import _geometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig,ws32_decoder_weight_names,_bind_weight_name_tree
from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
schema=json.loads(Path('tests/greenfield/hlo/fixtures/prefill_layer_schema.json').read_text())['tensor_schema']
config=Ws32DecoderConfig(_geometry(),p.CAPACITY)
names=ws32_decoder_weight_names(config).layers[6]
source={t['name']:t for t in schema};arrays={};selected=[]
types={'BF16':jnp.bfloat16,'F32':jnp.float32,'U8':jnp.uint8}
for name in jax.tree.leaves(names):
    suffix=name.split('model.layers.6.',1)[1]
    t=source[f'model.layers.{0 if ".indexer." in name else 3}.{suffix}']
    selected.append(t)
    arrays[name]=jax.ShapeDtypeStruct(tuple(t['global_shape']),types[t['dtype']],sharding=NamedSharding(mesh,P(*t['partition_spec'])))
assert len(selected)==p.TENSORS_PER_CHIP==35
assert sum(t['byte_count'] for t in selected)==p.PAYLOAD_BYTES_PER_CHIP
w=_bind_weight_name_tree(names,arrays)
assert w.dsa is not None and w.moe is not None and w.dense is None
wk=jax.ShapeDtypeStruct((128,6144),jnp.float32,sharding=NamedSharding(mesh,P()))
specs=input_specs(w,wk)
host=p.host_case('competitive',build_rotary_table_host(p.CAPACITY,rotary_dim=64,theta=8e6))
values=device_inputs(host,specs,w,wk,mesh)
wide,small=p.build_programs(mesh,specs,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
for fn,args,rows in ((wide,values,128),(small,p.control_inputs(values,0),32)):
    assert callable(fn.lower)
    result=jax.eval_shape(fn,*args)
    assert len(result)==12 and result[0].shape==result[-1].shape==(rows,6144)
    assert result[2].shape==(8,8,64,640) and result[3].shape==result[4].shape==(8,8,64,128)
    assert result[5].shape==result[7].shape==(rows,2048)
    assert result[8].shape==result[9].shape==(rows,8) and result[10].shape==(8,4,rows)
print('PRODUCTION_LAYER6_WINDOW_SCHEMA_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRODUCTION_LAYER6_WINDOW_SCHEMA_PASS" in result.stdout
