"""Trace the ACTUAL worker input builder at real geometry, without weights/TPU."""

import os
import subprocess
import sys


def test_real_schema_twenty_input_worker_on_cpu32():
    code = r"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info._get_tpu_info=lambda:tpu_info.TpuInfo.from_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from scripts.greenfield.probe_ws32_prefill_layer import input_specs,device_inputs,scalar_inputs,build_wk_programs
from scripts.greenfield.prefill_layer_programs import build_layer_programs
from scripts.greenfield.prefill_router_boundary import build_router_prefix_program,scalar_prefix_inputs
from scripts.greenfield.prefill_materialized_reference import build_materialized_reference,assemble_reference_result
from scripts.greenfield import prefill_observed_reference as observed_ref
from scripts.greenfield.prefill_layer_evidence import host_case,mutate_case
from scripts.greenfield.run_short_decoder_ws32 import _geometry
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig,ws32_decoder_weight_names,_bind_weight_name_tree
from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
fixture=json.loads(Path('tests/greenfield/hlo/fixtures/prefill_layer_schema.json').read_text())
assert fixture['source_manifest_sha256']=='88df414301e0d303506163c078484ec14cdb15ff03972789df58960b091a7cbf'
config=Ws32DecoderConfig(geometry=_geometry(),context_capacity=1024)
rope=build_rotary_table_host(1024,rotary_dim=64,theta=8e6)
types={'BF16':jnp.bfloat16,'F32':jnp.float32,'U8':jnp.uint8}
for layer in (0,3):
    schema=[t for t in fixture['tensor_schema'] if t['name'].startswith(f'model.layers.{layer}.')]
    assert len(schema)==(27 if layer==0 else 28)
    assert sum(t['byte_count'] for t in schema)==(21557920 if layer==0 else 324821552)
    arrays={t['name']:jax.ShapeDtypeStruct(tuple(t['global_shape']),types[t['dtype']],sharding=NamedSharding(mesh,P(*t['partition_spec']))) for t in schema}
    weights=_bind_weight_name_tree(ws32_decoder_weight_names(config).layers[layer],arrays)
    wk=jax.ShapeDtypeStruct((128,6144),jnp.float32,sharding=NamedSharding(mesh,P())) if layer==0 else None
    specs=input_specs(weights,wk)
    host=host_case('tail',rope)
    assert host['table'].shape==mutate_case(host,'bad_page')['table'].shape==(1,2)
    values=device_inputs(host,specs,weights,wk,mesh)
    assert len(values)==20 and values[10].shape==(1,2)
    batch,scalar=build_layer_programs(mesh,specs,full_indexer=layer==0,sparse_mlp=layer==3,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
    for fn,args,rows in ((batch,values,17),(scalar,scalar_inputs(values,8),1)):
        result=jax.eval_shape(fn,*args)
        assert len(result)==12
        assert result[0].shape==result[1].shape==result[-1].shape==(rows,6144)
        assert result[2].shape==(8,2,64,640)
        assert result[3].shape==result[4].shape==(8,2,64,128)
        assert result[5].shape==result[7].shape==(rows,2048)
        assert result[6].shape==(rows,)
        assert result[8].shape==result[9].shape==(rows,8)
        assert result[10].shape==(8,4,rows)
    if layer==0:
        decode,promote=build_wk_programs(mesh,weights.dsa.wk_bits_local.sharding.spec,weights.dsa.wk_scale_local.sharding.spec,contract=config.dsa_contract)
        decoded=jax.eval_shape(decode,weights.dsa.wk_bits_local,weights.dsa.wk_scale_local)
        promoted=jax.eval_shape(promote,decoded)
        assert decoded.shape==promoted.shape==(128,6144)
        assert decoded.dtype==jnp.bfloat16 and promoted.dtype==jnp.float32
    else:
        pre,suffix=build_materialized_reference(mesh,specs,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
        scalar_values=scalar_inputs(values,8)
        boundary=jax.eval_shape(pre,*scalar_values)
        assert len(boundary)==5 and boundary[0].dtype==jnp.bfloat16
        assert boundary[0].shape==boundary[1].shape==boundary[3].shape==(1,6144)
        assert boundary[2].shape==(8,2,64,640) and boundary[4].shape==(8,4,1)
        mlp=jax.eval_shape(suffix,boundary[0],boundary[1],scalar_values[15],scalar_values[17])
        complete=assemble_reference_result(scalar_values,boundary,mlp)
        assert len(complete)==12 and complete[2] is boundary[2]
        assert complete[3] is scalar_values[3] and complete[4] is scalar_values[4]
        assert complete[10] is boundary[4] and complete[11] is boundary[3]
        # The next row must carry the materialized reference's KV, not old inputs.
        assert scalar_inputs(values,9,complete)[2] is boundary[2]
        assert 'greenfield_prefill_grouped_raw_fp8' not in str(jax.make_jaxpr(suffix)(boundary[0],boundary[1],scalar_values[15],scalar_values[17]))
        observed_pre,observed_suffix=observed_ref.build_materialized_reference(mesh,specs,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract,moe_contract=config.moe_contract)
        observed_boundary=jax.eval_shape(observed_pre,*scalar_values)
        assert len(observed_boundary)==13 and observed_boundary[12].shape==(1,6144)
        assert observed_boundary[12].dtype==jnp.bfloat16
        assert observed_ref.scalar_observed_inputs(values,9,observed_boundary)[2] is observed_boundary[10]
        observed_mlp=jax.eval_shape(observed_suffix,observed_boundary[0],observed_boundary[9],scalar_values[15],scalar_values[17])
        assembled=observed_ref.assemble_reference_result(scalar_values,observed_boundary,observed_mlp)
        assert assembled[-1] is observed_boundary[12]
        assert assembled[2] is observed_boundary[10]
        for batched,args,rows in ((True,values,17),(False,scalar_inputs(values,8),1)):
            fn=build_router_prefix_program(mesh,specs,batched=batched,dsa_contract=config.dsa_contract,attention_contract=config.attention_contract)
            result=jax.eval_shape(fn,*args)
            assert len(result)==12
            assert result[0].shape==(rows,6144)
            assert result[1].shape==(8,4,rows,32)
            assert result[2].shape==result[4].shape==(rows,256)
            assert result[3].shape==(256,)
            assert result[5].shape==result[6].shape==(rows,8)
            assert all(result[i].shape==(rows,6144) for i in (7,8,9))
            assert result[10].shape==(8,2,64,640)
            assert result[11].shape==(8,4,rows)
            if not batched:
                assert scalar_prefix_inputs(values,9,result)[2] is result[10]
                mlp_from_original=jax.eval_shape(suffix,result[0],result[9],values[15],values[17])
                assert len(mlp_from_original)==3
                assert mlp_from_original[0].shape==(1,6144)
                assert mlp_from_original[1].shape==mlp_from_original[2].shape==(1,8)
            graph=str(jax.make_jaxpr(fn)(*args))
            assert 'greenfield_prefill_grouped_raw_fp8' not in graph
print('REAL_SCHEMA_CPU32_WORKER_TREE_PASS')
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=32",
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_SCHEMA_CPU32_WORKER_TREE_PASS" in result.stdout
