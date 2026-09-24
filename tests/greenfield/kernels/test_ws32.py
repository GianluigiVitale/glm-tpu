from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.optimized.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import ws32_moe_fp8_from_routes_mapped
from glm_tpu.optimized.mesh import (
    Ws32MeshContract,
    Ws32PerChipMemory,
    build_ws32_base_capacity_report,
    build_ws32_mlp_capacity_report,
    build_ws32_execution_plan,
    build_ws32_physical_mesh,
    validate_ws32_repeated_hlo,
)
from glm_tpu.optimized.geometry import ModelGeometry, PhysicalDevice, PhysicalTopology, PlanName
from glm_tpu.optimized.source_inventory import inspect_source_inventory


REPO = Path(__file__).resolve().parents[3]


def _topology() -> PhysicalTopology:
    devices = []
    for device_id, coordinates in enumerate(
        (x, y, z) for x in range(2) for y in range(4) for z in range(4)
    ):
        devices.append(
            PhysicalDevice(
                device_id=device_id,
                process_index=device_id // 4,
                local_device_id=device_id % 4,
                coordinates=coordinates,
                core_on_chip=0,
                platform="tpu",
                device_kind="TPU v4",
            )
        )
    return PhysicalTopology(
        slice_name="db-v4-64-od",
        topology_shape=(2, 4, 4),
        devices=tuple(devices),
    )


def test_ws32_real_geometry_layout_is_one_row_and_reciprocal() -> None:
    exact_geometry = ModelGeometry.from_hf_config(
        json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    )
    contract = Ws32MeshContract()
    layout = contract.layout_summary(exact_geometry)
    assert layout["mesh_shape"] == [8, 4]
    assert layout["residual"] == {
        "global_shape": [1, 6144],
        "local_shape": [1, 1536],
        "partition_spec": [None, "feature"],
        "replicated_axis": "expert",
    }
    assert layout["dense_gate_up"]["local_shape"] == [1536, 1536]
    assert layout["dense_down"]["local_shape"] == [1536, 1536]
    assert layout["routed_gate_up"]["local_shape"] == [32, 2048, 1536]
    assert layout["routed_down"]["local_shape"] == [32, 1536, 2048]
    assert layout["attention"]["selected_cache"] == {
        "context_partition_axis": "expert",
        "replicated_axis": "feature",
        "exchange_axis": "expert",
        "physical_group_size": 8,
    }
    assert layout["attention"]["o_projection"]["reduction_axis"] == "expert"
    assert layout["dsa"]["score_head_reduction_axis"] == "expert"
    assert layout["forbidden"] == {
        "batch_32_decode_rows": True,
        "full_pod_hidden_reconstruction": True,
        "repeated_collective_group_size_32": True,
    }


def test_ws32_rejects_non_32_mesh() -> None:
    with pytest.raises(Exception, match="exactly 32"):
        Ws32MeshContract(expert_axis_size=4, feature_axis_size=4)


def test_ws32_physical_mesh_is_explicit_8x4_topology_mapping() -> None:
    mesh = build_ws32_physical_mesh(_topology())
    assert mesh.device_ids == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
        (16, 17, 18, 19),
        (20, 21, 22, 23),
        (24, 25, 26, 27),
        (28, 29, 30, 31),
    )
    assert mesh.feature_groups == mesh.device_ids
    assert mesh.expert_groups == (
        (0, 4, 8, 12, 16, 20, 24, 28),
        (1, 5, 9, 13, 17, 21, 25, 29),
        (2, 6, 10, 14, 18, 22, 26, 30),
        (3, 7, 11, 15, 19, 23, 27, 31),
    )
    assert len(mesh.mesh_hash) == 64


def test_ws32_execution_plan_requires_explicit_memory() -> None:
    geometry = ModelGeometry.from_hf_config(
        json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    )
    memory = Ws32PerChipMemory(
        persistent_weight_bytes=23_000_000_000,
        fp8_scale_bytes=700_000_000,
        kv_bytes_at_target_context=1_000_000_000,
        dsa_state_bytes=200_000_000,
        temporary_bytes=500_000_000,
        reserved_overlay_bytes=1_000_000_000,
    )
    plan, physical_mesh = build_ws32_execution_plan(
        geometry=geometry,
        topology=_topology(),
        target_context_length=262_144,
        memory=memory,
    )
    assert plan.name is PlanName.WS32_2D
    assert plan.local_mesh_shape == (8, 4)
    assert plan.stage_assignments[0].process_index is None
    assert plan.stage_assignments[0].device_ids == (
        physical_mesh.flattened_device_ids
    )
    assert plan.stage_assignments[0].accounted_bytes == 26_400_000_000
    assert "persistent_hidden_feature_shard" in plan.residual_layout


def test_ws32_moe_refuses_non_ws32_expert_axis_contract() -> None:
    with pytest.raises(ValueError, match="stage_size=8"):
        ws32_moe_fp8_from_routes_mapped(
            *(None,) * 15,
            contract=GlmMoeNumericalContract(stage_size=4),
        )


def test_ws32_real_inventory_mlp_capacity_reconciles() -> None:
    inventory_path = Path(
        "/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/"
        "greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/"
        "source_inventory.json"
    )
    if not inventory_path.is_file():
        pytest.skip("SHA-pinned real source inventory is not present")
    geometry = ModelGeometry.from_hf_config(
        json.loads((REPO / "configs/glm-5.2-fp8-config.json").read_text())
    )
    inventory = inspect_source_inventory(inventory_path)
    inventory_sha256 = (
        "a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4"
    )
    assert inventory.inventory_sha256 == inventory_sha256
    report = build_ws32_mlp_capacity_report(inventory, geometry)
    assert report.source_inventory_sha256 == inventory_sha256
    assert report.source_tensor_count == 115_818
    assert report.source_bytes == 728_700_174_336
    assert report.packed_bytes == 748_523_329_536
    assert report.shared_replication_extra_bytes == 19_822_924_800
    assert report.compact_replication_extra_bytes == 230_400
    assert report.maximum_persistent_bytes == 23_391_354_048
    assert report.minimum_persistent_bytes == 23_391_354_048
    assert set(report.fp8_scale_bytes_by_chip) == {5_707_584}
    base = build_ws32_base_capacity_report(inventory, geometry)
    assert base.source_inventory_sha256 == inventory_sha256
    assert base.source_tensor_count == 117_060
    assert base.source_bytes == 745_584_507_456
    assert base.packed_bytes == 786_172_488_192
    assert base.non_mlp_replication_extra_bytes == 20_764_825_536
    assert base.maximum_persistent_bytes == 24_567_890_256
    assert base.minimum_persistent_bytes == 24_567_890_256
    assert set(base.fp8_scale_bytes_by_chip) == {5_967_312}


def _synthetic_dense_hlo() -> str:
    feature_groups = ",".join(
        "{" + ",".join(str(item) for item in range(row * 4, row * 4 + 4)) + "}"
        for row in range(8)
    )
    expert_groups = ",".join(
        "{" + ",".join(str(row * 4 + column) for row in range(8)) + "}"
        for column in range(4)
    )
    return f'''HloModule ws32_dense, num_partitions=32

feature_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

expert_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

ENTRY main {{
  %gate_up = f32[2,1,2] parameter(0)
  %feature = f32[2,1,2] all-reduce(%gate_up), replica_groups={{{feature_groups}}}, to_apply=%feature_add, use_global_device_ids=true, metadata={{op_name="greenfield_ws32_dense/feature_gate_up_reduce/psum"}}
  %down = f32[1,2] parameter(1)
  ROOT %expert = f32[1,2] all-reduce(%down), replica_groups={{{expert_groups}}}, to_apply=%expert_add, use_global_device_ids=true, metadata={{op_name="greenfield_ws32_dense/expert_down_reduce/psum"}}
}}
'''


def test_ws32_hlo_contract_rejects_group_shape_async_and_reducer_drift() -> None:
    base = _synthetic_dense_hlo()
    report = validate_ws32_repeated_hlo(
        base, kind="dense", hidden_size=8, dense_intermediate_size=16
    )
    assert report.valid, report.violations
    mutations = (
        base.replace(
            "{0,1,2,3},{4,5,6,7}",
            "{0,1,2,4},{3,5,6,7}",
            1,
        ),
        base.replace("f32[2,1,2] all-reduce(%gate_up)", "f32[2,1,3] all-reduce(%gate_up)", 1),
        base.replace("all-reduce(%gate_up)", "all-reduce-start(%gate_up)", 1),
        base.replace("ROOT %sum = f32[] add(%a, %b)", "ROOT %sum = f32[] maximum(%a, %b)", 1),
    )
    for mutation in mutations:
        refused = validate_ws32_repeated_hlo(
            mutation,
            kind="dense",
            hidden_size=8,
            dense_intermediate_size=16,
        )
        assert not refused.valid


def test_ws32_forced_32_dense_and_moe_keep_residual_sharded() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.optimized.reference.fp8 import dequantize_fp8_bits_block_weight
from glm_tpu.optimized.reference.moe import GlmMoeNumericalContract, reference_moe_from_routes
from glm_tpu.greenfield.kernels.ws32 import ws32_dense_fp8_mapped, ws32_moe_fp8_from_routes_mapped
from glm_tpu.optimized.mesh import validate_ws32_repeated_hlo

def bits(x): return np.asarray(x, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)
def draw(seed, shape): return np.random.default_rng(seed).normal(0, 0.125, shape).astype(np.float32)
def scale(shape): return np.ones(shape, np.float32)

devices=np.asarray(jax.devices(), dtype=object).reshape(8,4)
mesh=Mesh(devices, ('expert','feature'))
hidden=jnp.asarray([[.5,-.25,.75,1,-1,.125,.25,-.5]], jnp.bfloat16)
dense=16; block=(2,2)
gate=bits(draw(1,(dense,8))); up=bits(draw(2,(dense,8))); down=bits(draw(3,(8,dense)))
gs=scale((dense//2,4)); ds=scale((4,dense//2))

hidden_sh=NamedSharding(mesh,P(None,'feature'))
dense_out_sh=NamedSharding(mesh,P('expert','feature'))
dense_down_sh=NamedSharding(mesh,P('feature','expert'))
dense_map=jax.shard_map(
    lambda *x: ws32_dense_fp8_mapped(*x, block_shape=block),
    mesh=mesh,
    in_specs=(P(None,'feature'),P('expert','feature'),P('expert','feature'),P('expert','feature'),P('expert','feature'),P('feature','expert'),P('feature','expert')),
    out_specs=P(None,'feature'), check_vma=False)
dense_args=(jax.device_put(hidden,hidden_sh),jax.device_put(gate,dense_out_sh),jax.device_put(gs,dense_out_sh),jax.device_put(up,dense_out_sh),jax.device_put(gs,dense_out_sh),jax.device_put(down,dense_down_sh),jax.device_put(ds,dense_down_sh))
dense_compiled=jax.jit(dense_map).lower(*dense_args).compile(); dense_got=dense_compiled(*dense_args)
dense_report=validate_ws32_repeated_hlo(dense_compiled.as_text(),kind='dense',hidden_size=8,dense_intermediate_size=16)

gate_bf=dequantize_fp8_bits_block_weight(jnp.asarray(gate),jnp.asarray(gs),block_shape=block)
up_bf=dequantize_fp8_bits_block_weight(jnp.asarray(up),jnp.asarray(gs),block_shape=block)
down_bf=dequantize_fp8_bits_block_weight(jnp.asarray(down),jnp.asarray(ds),block_shape=block)
def fp32_dot(lhs,rhs): return jax.lax.dot_general(lhs,rhs,dimension_numbers=(((lhs.ndim-1,),(rhs.ndim-1,)),((),())),preferred_element_type=jnp.float32)
hidden_chunks=hidden.reshape(1,4,2)
gate_chunks=gate_bf.reshape(8,2,4,2); up_chunks=up_bf.reshape(8,2,4,2)
dense_activated=[]
for expert_row in range(8):
    gate_parts=jnp.stack(tuple(fp32_dot(hidden_chunks[:,feature_row,:],gate_chunks[expert_row,:,feature_row,:]) for feature_row in range(4)))
    up_parts=jnp.stack(tuple(fp32_dot(hidden_chunks[:,feature_row,:],up_chunks[expert_row,:,feature_row,:]) for feature_row in range(4)))
    gate_sum=jnp.sum(gate_parts,axis=0,dtype=jnp.float32).astype(jnp.bfloat16)
    up_sum=jnp.sum(up_parts,axis=0,dtype=jnp.float32).astype(jnp.bfloat16)
    dense_activated.append((gate_sum*jax.nn.sigmoid(gate_sum)*up_sum).astype(jnp.bfloat16))
down_chunks=down_bf.reshape(4,2,8,2)
dense_expected=jnp.concatenate(tuple(jnp.sum(jnp.stack(tuple(fp32_dot(dense_activated[expert_row],down_chunks[feature_row,:,expert_row,:]) for expert_row in range(8))),axis=0,dtype=jnp.float32).astype(jnp.bfloat16) for feature_row in range(4)),axis=-1)

contract=GlmMoeNumericalContract(hidden_size=8,intermediate_size=8,num_experts=16,top_k=4,stage_size=8,fp8_block_shape=block)
routes=jnp.asarray([[0,5,9,15]],jnp.int32); weights=jnp.asarray([[.1,.2,.3,.4]],jnp.float32)
eg=bits(draw(4,(16,8,8))); eu=bits(draw(5,(16,8,8))); ed=bits(draw(6,(16,8,8)))
es=scale((16,4,4))
sg=bits(draw(7,(8,8))); su=bits(draw(8,(8,8))); sd=bits(draw(9,(8,8))); ss=scale((4,4))
expert_gate_sh=NamedSharding(mesh,P('expert',None,'feature')); expert_down_sh=NamedSharding(mesh,P('expert','feature',None))
shared_gate_sh=NamedSharding(mesh,P(None,'feature')); shared_down_sh=NamedSharding(mesh,P('feature',None)); rep=NamedSharding(mesh,P())
moe_map=jax.shard_map(
    lambda *x: ws32_moe_fp8_from_routes_mapped(*x,contract=contract),
    mesh=mesh,
    in_specs=(P(None,'feature'),P(),P(),P('expert',None,'feature'),P('expert',None,'feature'),P('expert',None,'feature'),P('expert',None,'feature'),P('expert','feature',None),P('expert','feature',None),P(None,'feature'),P(None,'feature'),P(None,'feature'),P(None,'feature'),P('feature',None),P('feature',None)),
    out_specs=P(None,'feature'),check_vma=False)
moe_args=(jax.device_put(hidden,hidden_sh),jax.device_put(routes,rep),jax.device_put(weights,rep),jax.device_put(eg,expert_gate_sh),jax.device_put(es,expert_gate_sh),jax.device_put(eu,expert_gate_sh),jax.device_put(es,expert_gate_sh),jax.device_put(ed,expert_down_sh),jax.device_put(es,expert_down_sh),jax.device_put(sg,shared_gate_sh),jax.device_put(ss,shared_gate_sh),jax.device_put(su,shared_gate_sh),jax.device_put(ss,shared_gate_sh),jax.device_put(sd,shared_down_sh),jax.device_put(ss,shared_down_sh))
moe_compiled=jax.jit(moe_map).lower(*moe_args).compile(); moe_got=moe_compiled(*moe_args)
moe_report=validate_ws32_repeated_hlo(moe_compiled.as_text(),kind='moe',hidden_size=8,moe_intermediate_size=8,top_k=4)
moe_expected=reference_moe_from_routes(hidden,routes,weights,dequantize_fp8_bits_block_weight(jnp.asarray(eg),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(eu),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(ed),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(sg),jnp.asarray(ss),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(su),jnp.asarray(ss),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(sd),jnp.asarray(ss),block_shape=block),contract=contract)
route_sensitive=[]
for route_position in range(contract.top_k):
    changed_weights=weights.at[0,route_position].set(jnp.float32(0))
    changed=reference_moe_from_routes(hidden,routes,changed_weights,dequantize_fp8_bits_block_weight(jnp.asarray(eg),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(eu),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(ed),jnp.asarray(es),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(sg),jnp.asarray(ss),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(su),jnp.asarray(ss),block_shape=block),dequantize_fp8_bits_block_weight(jnp.asarray(sd),jnp.asarray(ss),block_shape=block),contract=contract)
    route_sensitive.append(not np.array_equal(np.asarray(changed).view(np.uint16),np.asarray(moe_expected).view(np.uint16)))

def hlo_contract(text):
    lines=[line for line in text.splitlines() if ' all-reduce(' in line]
    groups=[]
    for line in lines:
        marker='replica_groups='
        if marker in line:
            groups.append(max(part.count(',')+1 for part in line.split(marker,1)[1].split('},')[0].split('{') if ',' in part))
    return {'all_reduce_count':len(lines),'maximum_group':max(groups) if groups else 0,'has_full_hidden':('bf16[32,8]' in text or 'f32[32,8]' in text)}
print(json.dumps({'dense_error':float(jnp.max(jnp.abs(dense_got.astype(jnp.float32)-dense_expected.astype(jnp.float32)))),'dense_hlo':hlo_contract(dense_compiled.as_text()),'dense_report_valid':dense_report.valid,'dense_report_violations':dense_report.violations,'dense_sharding':str(dense_got.sharding.spec),'moe_bitwise':bool(np.array_equal(np.asarray(moe_got).view(np.uint16),np.asarray(moe_expected).view(np.uint16))),'moe_hlo':hlo_contract(moe_compiled.as_text()),'moe_report_valid':moe_report.valid,'moe_report_violations':moe_report.violations,'moe_sharding':str(moe_got.sharding.spec),'route_oracle_sensitive':all(route_sensitive)},sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["dense_error"] <= 2**-14
    assert result["moe_bitwise"]
    assert result["route_oracle_sensitive"]
    assert result["dense_report_valid"], result["dense_report_violations"]
    assert result["moe_report_valid"], result["moe_report_violations"]
    assert result["dense_sharding"] == "P(None, 'feature')"
    assert result["moe_sharding"] == "P(None, 'feature')"
    assert result["dense_hlo"]["all_reduce_count"] == 2
    assert result["moe_hlo"]["all_reduce_count"] >= 3
    assert result["dense_hlo"]["maximum_group"] <= 8
    assert result["moe_hlo"]["maximum_group"] <= 8
    assert not result["dense_hlo"]["has_full_hidden"]
    assert not result["moe_hlo"]["has_full_hidden"]
