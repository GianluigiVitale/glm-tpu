from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
    fp8_e4m3fn_lookup,
)
from glm_tpu.greenfield.kernels.stage_local import _stage_fp8_linear


def _bits(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)


def test_fp8_lookup_matches_ml_dtypes_for_every_finite_encoding() -> None:
    bits = np.arange(256, dtype=np.uint8)
    observed = np.asarray(fp8_e4m3fn_lookup(), dtype=np.float32)
    expected = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(observed[finite], expected[finite])
    assert np.array_equal(np.isnan(observed), np.isnan(expected))


def test_raw_bit_block_dequant_preserves_leading_expert_axis() -> None:
    values = np.arange(-24, 24, dtype=np.float32).reshape(2, 4, 6) / 16
    bits = jnp.asarray(_bits(values))
    scale = jnp.asarray(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.0], [1.5, 2.0]]],
        dtype=jnp.float32,
    )
    got = dequantize_fp8_bits_block_weight(
        bits, scale, block_shape=(2, 3), output_dtype=jnp.float32
    )
    expanded = np.repeat(np.repeat(np.asarray(scale), 2, axis=-2), 3, axis=-1)
    expected = _bits(values).view(ml_dtypes.float8_e4m3fn).astype(np.float32) * expanded
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_stage_fp8_linear_pallas_dispatch_matches_reference() -> None:
    hidden = jnp.asarray(
        np.linspace(-0.75, 0.5, 128, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    values = np.sin(
        np.arange(128 * 128, dtype=np.float32).reshape(128, 128) * 0.013
    )
    bits = jnp.asarray(_bits(values))
    scale = jnp.asarray([[0.625]], dtype=jnp.float32)
    expected = _stage_fp8_linear(
        hidden,
        bits,
        scale,
        block_shape=(128, 128),
        backend="reference",
        interpret=False,
    )
    actual = _stage_fp8_linear(
        hidden,
        bits,
        scale,
        block_shape=(128, 128),
        backend="pallas",
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_raw_fp8_dense_and_moe_match_bf16_reference_on_forced_cpu() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.fp8 import dequantize_fp8_bits_block_weight
from glm_tpu.greenfield.kernels.reference.linear import linear, residual_add, silu
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract, reference_moe_from_routes, route_glm_noaux_tc
from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm
from glm_tpu.greenfield.kernels.stage_local import stage_local_dense_fp8_mapped, stage_local_moe_fp8_mapped

def bits(x): return np.asarray(x, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)
def draw(seed, shape):
    return np.random.default_rng(seed).normal(0, 0.125, shape).astype(np.float32)

devices=np.asarray(jax.devices(), dtype=object)
mesh=Mesh(devices, ('stage',))
rep=NamedSharding(mesh,P())
contract=GlmMoeNumericalContract(hidden_size=8, intermediate_size=8, num_experts=16, top_k=4, stage_size=4, fp8_block_shape=(2,2))
hidden=jnp.asarray([[.5,-.25,.75,1,-1,.125,.25,-.5]],jnp.bfloat16)
norm=jnp.ones((8,),jnp.bfloat16)

gate=bits(draw(1,(8,8))); up=bits(draw(2,(8,8))); down=bits(draw(3,(8,8)))
gscale=np.ones((4,4),np.float32); uscale=np.ones((4,4),np.float32); dscale=np.ones((4,4),np.float32)
dense_expected=residual_add(hidden, linear((silu(linear(rms_norm(hidden,norm,epsilon=1e-5),dequantize_fp8_bits_block_weight(jnp.asarray(gate),jnp.asarray(gscale),block_shape=(2,2))))*linear(rms_norm(hidden,norm,epsilon=1e-5),dequantize_fp8_bits_block_weight(jnp.asarray(up),jnp.asarray(uscale),block_shape=(2,2)))).astype(jnp.bfloat16),dequantize_fp8_bits_block_weight(jnp.asarray(down),jnp.asarray(dscale),block_shape=(2,2))))
dense_map=jax.shard_map(lambda *x: stage_local_dense_fp8_mapped(*x,axis_name='stage',block_shape=(2,2)),mesh=mesh,in_specs=(P(),P(),P('stage',None),P('stage',None),P('stage',None),P('stage',None),P(None,'stage'),P(None,'stage')),out_specs=P(),check_vma=False)
dense_args=(jax.device_put(hidden,rep),jax.device_put(norm,rep),jax.device_put(gate,NamedSharding(mesh,P('stage',None))),jax.device_put(gscale,NamedSharding(mesh,P('stage',None))),jax.device_put(up,NamedSharding(mesh,P('stage',None))),jax.device_put(uscale,NamedSharding(mesh,P('stage',None))),jax.device_put(down,NamedSharding(mesh,P(None,'stage'))),jax.device_put(dscale,NamedSharding(mesh,P(None,'stage'))))
dense_compiled=jax.jit(dense_map).lower(*dense_args).compile(); dense_got=dense_compiled(*dense_args)

router=jnp.asarray(draw(4,(16,8)),jnp.bfloat16); bias=jnp.zeros((16,),jnp.float32)
eg=bits(draw(5,(16,8,8))); eu=bits(draw(6,(16,8,8))); ed=bits(draw(7,(16,8,8)))
es=np.ones((16,4,4),np.float32)
sg=bits(draw(8,(8,8))); su=bits(draw(9,(8,8))); sd=bits(draw(10,(8,8))); ss=np.ones((4,4),np.float32)
routes,weights=route_glm_noaux_tc(hidden,router,bias,top_k=4)
moe_expected=reference_moe_from_routes(hidden,routes,weights,dequantize_fp8_bits_block_weight(jnp.asarray(eg),jnp.asarray(es),block_shape=(2,2)),dequantize_fp8_bits_block_weight(jnp.asarray(eu),jnp.asarray(es),block_shape=(2,2)),dequantize_fp8_bits_block_weight(jnp.asarray(ed),jnp.asarray(es),block_shape=(2,2)),dequantize_fp8_bits_block_weight(jnp.asarray(sg),jnp.asarray(ss),block_shape=(2,2)),dequantize_fp8_bits_block_weight(jnp.asarray(su),jnp.asarray(ss),block_shape=(2,2)),dequantize_fp8_bits_block_weight(jnp.asarray(sd),jnp.asarray(ss),block_shape=(2,2)),contract=contract)
expert_sh=NamedSharding(mesh,P('stage',None,None)); shared_out=NamedSharding(mesh,P('stage',None)); shared_down=NamedSharding(mesh,P(None,'stage'))
moe_map=jax.shard_map(lambda *x: stage_local_moe_fp8_mapped(*x[:-1],x[-1][0],axis_name='stage',contract=contract),mesh=mesh,in_specs=(P(),P(),P(),P('stage',None,None),P('stage',None,None),P('stage',None,None),P('stage',None,None),P('stage',None,None),P('stage',None,None),P('stage',None),P('stage',None),P('stage',None),P('stage',None),P(None,'stage'),P(None,'stage'),P('stage')),out_specs=(P(),P(),P()),check_vma=False)
moe_args=(jax.device_put(hidden,rep),jax.device_put(router,rep),jax.device_put(bias,rep),jax.device_put(eg,expert_sh),jax.device_put(es,expert_sh),jax.device_put(eu,expert_sh),jax.device_put(es,expert_sh),jax.device_put(ed,expert_sh),jax.device_put(es,expert_sh),jax.device_put(sg,shared_out),jax.device_put(ss,shared_out),jax.device_put(su,shared_out),jax.device_put(ss,shared_out),jax.device_put(sd,shared_down),jax.device_put(ss,shared_down),jax.device_put(jnp.arange(4,dtype=jnp.int32),NamedSharding(mesh,P('stage'))))
moe_compiled=jax.jit(moe_map).lower(*moe_args).compile(); moe_got,got_routes,_=moe_compiled(*moe_args)
def counts(hlo): return {'ar':hlo.count(' all-reduce('),'ag':hlo.count(' all-gather('),'cp':hlo.count(' collective-permute(')}
print(json.dumps({'dense_error':float(jnp.max(jnp.abs(dense_got.astype(jnp.float32)-dense_expected.astype(jnp.float32)))),'dense_hlo':counts(dense_compiled.as_text()),'moe_error':float(jnp.max(jnp.abs(moe_got.astype(jnp.float32)-moe_expected.astype(jnp.float32)))),'moe_hlo':counts(moe_compiled.as_text()),'routes_exact':bool(jnp.array_equal(got_routes,routes))},sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = f"{existing} --xla_force_host_platform_device_count=4".strip()
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import json
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["dense_error"] <= 2**-12
    assert result["moe_error"] <= 2**-12
    assert result["routes_exact"]
    assert result["dense_hlo"] == {"ag": 0, "ar": 1, "cp": 0}
    assert result["moe_hlo"] == {"ag": 0, "ar": 1, "cp": 0}


def test_exact_dsa_query_keeps_four_entry_aliases_and_two_barriers() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.stage_local import _local_dsa_query, _local_dsa_query_tuple4_exact

devices = np.asarray(jax.devices(), dtype=object)
mesh = Mesh(devices, ('stage',))
replicated = NamedSharding(mesh, P())
weight_sharding = NamedSharding(mesh, P('stage', None))
contract = DsaNumericalContract(hidden_size=8, q_lora_rank=8, num_heads=8, head_dim=4, rotary_dim=2, top_k=4)
query = jax.device_put(jnp.asarray([[.5, -.25, .75, 1, -1, .125, .25, -.5]], jnp.bfloat16), replicated)
normalized = jax.device_put(jnp.asarray([[.25, -.5, .125, 1, -.75, .5, -.125, .75]], jnp.bfloat16), replicated)
weight = jax.device_put(jnp.arange(32 * 8, dtype=jnp.float32).reshape(32, 8) / 1024, weight_sharding)
head_weight = jax.device_put(jnp.arange(2 * 8, dtype=jnp.float32).reshape(2, 8) / 64, replicated)
position = jax.device_put(jnp.asarray([7], jnp.int32), replicated)

exact = jax.shard_map(
    lambda q, n, w0, w1, w2, w3, h, p: _local_dsa_query_tuple4_exact(q, n, (w0, w1, w2, w3), h, p, contract=contract)[0],
    mesh=mesh,
    in_specs=(P(), P(), P('stage', None), P('stage', None), P('stage', None), P('stage', None), P(), P()),
    out_specs=P('stage', None, None),
    check_vma=False,
)
ordinary = jax.shard_map(
    lambda q, n, w, h, p: _local_dsa_query(q, n, w, h, p, contract=contract)[0],
    mesh=mesh,
    in_specs=(P(), P(), P('stage', None), P(), P()),
    out_specs=P('stage', None, None),
    check_vma=False,
)
lowered = jax.jit(exact).lower(query, normalized, weight, weight, weight, weight, head_weight, position)
stablehlo = lowered.as_text()
compiled = lowered.compile()
actual = compiled(query, normalized, weight, weight, weight, weight, head_weight, position)
expected = jax.jit(ordinary)(query, normalized, weight, head_weight, position)
print(json.dumps({
    'barriers': stablehlo.count('stablehlo.optimization_barrier'),
    'dots': stablehlo.count('stablehlo.dot_general'),
    'exact': bool(jnp.array_equal(actual, expected)),
    'global_aliases': stablehlo.count('tensor<32x8xf32>'),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import json

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["barriers"] == 2
    assert result["dots"] == 4
    assert result["exact"]
    assert result["global_aliases"] >= 4
