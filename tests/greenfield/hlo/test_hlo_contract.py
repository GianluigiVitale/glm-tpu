from __future__ import annotations

import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.errors import HloContractViolationError
from glm_tpu.greenfield.sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    lint_hlo,
    parse_hlo_module,
)


GOOD_HLO = r'''HloModule jit_pipeline, replica_count=1, num_partitions=8

add_region {
  a = bf16[] parameter(0)
  b = bf16[] parameter(1)
  ROOT sum = bf16[] add(a, b)
}

decode_repeated_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  local = bf16[1,6144]{1,0} all-reduce(x), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7}}, use_global_device_ids=true, to_apply=add_region, metadata={op_name="decode/layer/moe_combine" source_file="model/moe.py" source_line=91 source_stack="decoder.layer"}
  routed = bf16[1,2048]{1,0} slice(local), slice={[0:1],[0:2048]}, metadata={op_name="decode/layer/routed" source_file="model/moe.py" source_line=92}
  ROOT moved = bf16[1,6144]{1,0} collective-permute(local), channel_id=2, source_target_pairs={{0,4},{4,0},{1,5},{5,1},{2,6},{6,2},{3,7},{7,3}}, metadata={op_name="decode/pipeline/transport" source_file="runtime/transport.py" source_line=20}
}

ENTRY main {
  input = bf16[1,6144]{1,0} parameter(0)
  ROOT call = bf16[1,6144]{1,0} call(input), to_apply=decode_repeated_layer
}
'''


def good_policy() -> HloContractPolicy:
    return HloContractPolicy(
        name="PP8 synthetic",
        total_devices=8,
        repeated_region_patterns=(r"decode_repeated_layer", r"decode/"),
        maximum_repeated_collective_group_size=4,
        expected_repeated_replica_groups=((0, 1, 2, 3), (4, 5, 6, 7)),
        expected_collectives=(
            CollectiveExpectation("all-reduce", 1, repeated_only=True),
            CollectiveExpectation("collective-permute", 1),
        ),
        expected_collective_permute_pairs=(
            (0, 4),
            (4, 0),
            (1, 5),
            (5, 1),
            (2, 6),
            (6, 2),
            (3, 7),
            (7, 3),
        ),
    )


def test_parser_extracts_physical_collective_contract() -> None:
    module = parse_hlo_module(GOOD_HLO)
    assert module.name == "jit_pipeline"
    assert module.num_partitions == 8
    assert len(module.collectives) == 2
    reduction, permute = module.collectives
    assert reduction.replica_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert reduction.maximum_group_size == 4
    assert reduction.result_shapes[0].dimensions == (1, 6144)
    assert reduction.operand_shapes[0].dimensions == (1, 6144)
    assert reduction.source_file == "model/moe.py"
    assert reduction.source_line == 91
    assert reduction.source_stack == "decoder.layer"
    assert permute.source_target_pairs[0] == (0, 4)


def test_valid_local_contract_passes() -> None:
    report = lint_hlo(parse_hlo_module(GOOD_HLO), good_policy())
    assert report.valid, report.violations
    assert report.to_dict()["collective_counts"] == {
        "all-reduce": 1,
        "collective-permute": 1,
    }
    assert all(
        collective["inside_repeated_region"]
        for collective in report.to_dict()["collectives"]
    )
    report.raise_for_violations()


def test_full_pod_repeated_residual_is_rejected() -> None:
    bad = GOOD_HLO.replace(
        "{{0,1,2,3},{4,5,6,7}}", "{{0,1,2,3,4,5,6,7}}"
    )
    report = lint_hlo(parse_hlo_module(bad), good_policy())
    codes = {violation.code for violation in report.violations}
    assert "REPEATED_GROUP_TOO_LARGE" in codes
    assert "FULL_POD_REPEATED_COLLECTIVE" in codes
    assert "FULL_POD_RESIDUAL_RECONSTRUCTION" in codes
    assert "REPLICA_GROUPS_MISMATCH" in codes
    with pytest.raises(HloContractViolationError):
        report.raise_for_violations()


def test_full_pod_diagnostic_policy_can_measure_without_promoting() -> None:
    full_pod = GOOD_HLO.replace(
        "{{0,1,2,3},{4,5,6,7}}", "{{0,1,2,3,4,5,6,7}}"
    )
    policy = HloContractPolicy(
        name="explicit diagnostic comparator",
        total_devices=8,
        repeated_region_patterns=("decode",),
        maximum_repeated_collective_group_size=8,
        expected_repeated_replica_groups=((0, 1, 2, 3, 4, 5, 6, 7),),
        expected_collectives=(
            CollectiveExpectation("all-reduce", 1),
            CollectiveExpectation("collective-permute", 1),
        ),
        expected_collective_permute_pairs=good_policy().expected_collective_permute_pairs,
        allow_full_pod_repeated_collectives=True,
    )
    report = lint_hlo(parse_hlo_module(full_pod), policy)
    assert report.valid, report.violations


def test_missing_declared_local_group_is_rejected() -> None:
    bad = GOOD_HLO.replace(
        "{{0,1,2,3},{4,5,6,7}}", "{{0,1,2,3}}"
    )
    report = lint_hlo(parse_hlo_module(bad), good_policy())
    assert "REPLICA_GROUPS_MISMATCH" in {
        violation.code for violation in report.violations
    }


def test_dead_batch_32_rows_are_rejected() -> None:
    bad = GOOD_HLO.replace("bf16[1,6144]", "bf16[32,6144]")
    report = lint_hlo(parse_hlo_module(bad), good_policy())
    assert "DEAD_DECODE_ROWS" in {
        violation.code for violation in report.violations
    }


def test_wrong_collective_count_is_rejected() -> None:
    policy = HloContractPolicy(
        name="wrong count",
        total_devices=8,
        repeated_region_patterns=("decode",),
        maximum_repeated_collective_group_size=4,
        expected_collectives=(CollectiveExpectation("all-reduce", 75),),
    )
    report = lint_hlo(parse_hlo_module(GOOD_HLO), policy)
    assert any(
        violation.code == "COLLECTIVE_COUNT_MISMATCH"
        and "got 1" in violation.message
        for violation in report.violations
    )


def test_missing_global_device_ids_is_rejected() -> None:
    bad = GOOD_HLO.replace(", use_global_device_ids=true", "")
    report = lint_hlo(parse_hlo_module(bad), good_policy())
    assert "NON_GLOBAL_DEVICE_IDS" in {
        violation.code for violation in report.violations
    }


def test_tuple_collective_shapes_and_operands_are_preserved() -> None:
    text = r'''HloModule tuple_reduce, replica_count=1, num_partitions=4

decode_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  y = f32[1,8]{1,0} parameter(1)
  ROOT reduced = (bf16[1,6144]{1,0}, f32[1,8]{1,0}) all-reduce(x, y), channel_id=3, replica_groups={{0,1,2,3}}, use_global_device_ids=true, metadata={op_name="decode/layer/tuple"}
}
'''
    instruction = parse_hlo_module(text).collectives[0]
    assert [shape.dimensions for shape in instruction.result_shapes] == [
        (1, 6144),
        (1, 8),
    ]
    assert [shape.dtype for shape in instruction.operand_shapes] == ["bf16", "f32"]


def test_operand_shape_resolution_is_computation_local() -> None:
    text = r'''HloModule duplicate_names

first {
  x = f32[99]{0} parameter(0)
  ROOT copy = f32[99]{0} copy(x)
}

decode_layer {
  x = bf16[1,6144]{1,0} parameter(0)
  ROOT reduced = bf16[1,6144]{1,0} all-reduce(x), replica_groups={{0,1,2,3}}, use_global_device_ids=true
}
'''
    instruction = parse_hlo_module(text).collectives[0]
    assert instruction.operand_shapes == instruction.result_shapes


def test_policy_round_trip() -> None:
    policy = good_policy()
    assert HloContractPolicy.from_dict(policy.to_dict()) == policy


def test_current_jax_lowering_exposes_exact_local_groups() -> None:
    program = r'''
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(np.asarray(jax.devices()).reshape(2, 4), ("stage", "local"))
def repeated_layer(x):
    return lax.psum(x, "local")
lowered = jax.jit(jax.shard_map(
    repeated_layer,
    mesh=mesh,
    in_specs=P("stage", "local", None),
    out_specs=P("stage", "local", None),
    check_vma=False,
)).lower(jnp.ones((2, 4, 8), jnp.bfloat16))
print(lowered.compiler_ir(dialect="hlo").as_hlo_text())
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    text = subprocess.check_output(
        [sys.executable, "-c", program], env=env, text=True
    )
    policy = HloContractPolicy(
        name="forced CPU LP4 lowering",
        total_devices=8,
        repeated_region_patterns=(r"manual_computation_body",),
        maximum_repeated_collective_group_size=4,
        expected_repeated_replica_groups=(
            (0, 1, 2, 3),
            (4, 5, 6, 7),
        ),
        expected_collectives=(CollectiveExpectation("all-reduce", 1),),
    )
    report = lint_hlo(parse_hlo_module(text), policy)
    assert report.valid, report.violations
