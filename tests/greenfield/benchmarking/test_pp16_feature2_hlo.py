from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import (
    validate_feature2_main_optimized_hlo,
    validate_feature2_main_stablehlo,
    validate_feature2_materializer_optimized_hlo,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


def _materializer_hlo(phase: str) -> str:
    if phase == "query_fp32":
        parameters = """  %b0 = u8[1,2048,2048] parameter(0)
  %s0 = f32[1,16,16] parameter(1)
  %b1 = u8[1,2048,2048] parameter(2)
  %s1 = f32[1,16,16] parameter(3)
  %o0 = f32[1,2048,2048] convert(%b0)
  %o1 = f32[1,2048,2048] convert(%b1)
"""
        root = "(f32[1,2048,2048], f32[1,2048,2048]) tuple(%o0, %o1)"
    elif phase == "wk_decode_bf16":
        parameters = """  %b0 = u8[1,128,6144] parameter(0)
  %s0 = f32[1,1,48] parameter(1)
  %b1 = u8[1,128,6144] parameter(2)
  %s1 = f32[1,1,48] parameter(3)
  %o0 = bf16[1,128,6144] convert(%b0)
  %o1 = bf16[1,128,6144] convert(%b1)
"""
        root = "(bf16[1,128,6144], bf16[1,128,6144]) tuple(%o0, %o1)"
    else:
        parameters = """  %b0 = bf16[1,128,6144] parameter(0)
  %b1 = bf16[1,128,6144] parameter(1)
  %o0 = f32[1,128,6144] convert(%b0)
  %o1 = f32[1,128,6144] convert(%b1)
"""
        root = "(f32[1,128,6144], f32[1,128,6144]) tuple(%o0, %o1)"
    return (
        "HloModule feature2_materializer, num_partitions=2\n\n"
        "ENTRY %main {\n"
        f"{parameters}"
        f"  ROOT %root = {root}\n"
        "}\n"
    )


@pytest.mark.parametrize(
    "phase", ("query_fp32", "wk_decode_bf16", "wk_promote_fp32")
)
def test_feature2_materializer_hlo_is_owner_local(phase: str) -> None:
    report = validate_feature2_materializer_optimized_hlo(
        _materializer_hlo(phase), phase=phase
    )
    assert report["passed"] is True
    assert report["num_partitions"] == 2
    assert report["collective_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions=2", "num_partitions=32"),
        lambda value: value.replace(
            "  ROOT %root",
            "  %bad = f32[1] all-reduce(%o0), replica_groups={{0,1}}\n"
            "  ROOT %root",
        ),
        lambda value: value.replace(
            "  ROOT %root",
            '  %bad = f32[1] custom-call(), custom_call_target="tpu_custom_call"\n'
            "  ROOT %root",
        ),
        lambda value: value + "host_callback",
    ),
)
def test_feature2_materializer_hlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_materializer_optimized_hlo(
            mutation(_materializer_hlo("query_fp32")), phase="query_fp32"
        )


def _main_optimized_hlo() -> str:
    root = (
        "(s32[1,2048], s32[1], f32[1,2048], bf16[1,1,3072], "
        "bf16[1,1,32,256], bf16[1,576], bf16[1,16,256,640], "
        "bf16[1,16,256,128], bf16[1,16,256,128], u32[1,2], pred[1])"
    )
    lines = [
        "HloModule feature2_main, num_partitions=2",
        "",
        "ENTRY %main {",
        "  %tokens = s32[8156] parameter(0)",
        "  %positions = s32[8156] parameter(1)",
        "  %blocks = s32[1,16] parameter(2)",
        "  %context = s32[1] parameter(3)",
        "  %position = s32[1] parameter(4)",
        "  %rope = bf16[8192,64] parameter(5)",
        "  %g = bf16[1,3072] all-gather(%rope), replica_groups={{0,1}}, use_global_device_ids=true",
        "  %p = bf16[1,3072] collective-permute(%g), source_target_pairs={{0,1},{1,0}}",
    ]
    for index in range(8):
        lines.append(
            f"  %attn.{index} = bf16[1,16,256] custom-call(%rope), "
            'custom_call_target="tpu_custom_call", '
            'metadata={op_name="greenfield_pregathered_sparse_mla_'
            'h16_k2048_b512_w640/pallas_call"}'
        )
    lines.extend(
        [
            f"  ROOT %root = {root} tuple(%tokens)",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def test_feature2_main_optimized_hlo_is_parsed_and_lp2_only() -> None:
    report = validate_feature2_main_optimized_hlo(_main_optimized_hlo())
    assert report["passed"] is True
    assert report["h16_b512_attention_calls"] == 8
    assert report["physical_collective_count"] == 2
    assert report["collective_counts"] == {
        "all-gather": 1,
        "collective-permute": 1,
    }


def test_feature_shard_axis_survives_in_real_optimized_root() -> None:
    code = r'''
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

mesh = Mesh(np.asarray(jax.devices()), ("feature",))
mapped = jax.shard_map(
    lambda value: value,
    mesh=mesh,
    in_specs=P("feature", None, None),
    out_specs=P("feature", None, None),
    check_vma=False,
)
optimized = jax.jit(mapped).lower(
    jnp.zeros((2, 1, 3072), dtype=jnp.bfloat16)
).compile().as_text()
module = parse_hlo_module(optimized)
roots = [
    item for item in module.instructions
    if item.computation.startswith("ENTRY ")
    and item.raw_line.startswith("ROOT ")
]
assert len(roots) == 1
assert tuple(
    (shape.dtype.lower(), shape.dimensions)
    for shape in roots[0].result_shapes
) == (("bf16", (1, 1, 3072)),)
'''
    environment = os.environ.copy()
    existing = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=2".strip()
    )
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = str(Path.cwd())
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions=2", "num_partitions=32"),
        lambda value: value.replace("{{0,1}}", "{{0,1,2,3}}"),
        lambda value: value.replace(", use_global_device_ids=true", ""),
        lambda value: value.replace("{{0,1},{1,0}}", "{{0,2},{2,0}}"),
        lambda value: value.replace(
            "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640",
            "greenfield_pregathered_sparse_mla_h32_k2048_b512_w640",
            1,
        ),
        lambda value: value + "\nbf16[8156,6144] host_callback",
    ),
)
def test_feature2_main_optimized_hlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_optimized_hlo(mutation(_main_optimized_hlo()))


def _main_stablehlo() -> str:
    return '''module attributes {mhlo.num_partitions = 2 : i32} {
  func.func @main(%arg0: tensor<8156xi32>, %arg1: tensor<8156xi32>,
      %arg2: tensor<1x16xi32>, %arg3: tensor<8192x64xbf16>)
      -> (tensor<1x2048xi32>, tensor<1x2048xf32>,
          tensor<2x1x3072xbf16>, tensor<2x16x256x640xbf16>,
          tensor<2x16x256x128xbf16>, tensor<2x2xui32>) {
    %0 = stablehlo.all_gather %arg3, dim = 0,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    %1 = stablehlo.collective_permute %0,
      source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>
    %2 = stablehlo.custom_call @tpu_custom_call(%1)
      {backend_config = "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"}
    return
  }
}
'''


def test_feature2_main_stablehlo_pins_boundary_and_groups() -> None:
    assert validate_feature2_main_stablehlo(_main_stablehlo())["passed"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.replace("num_partitions = 2", "num_partitions = 32"),
        lambda value: value.replace("[[0, 1]]", "[[0, 1, 2, 3]]"),
        lambda value: value.replace("[[0, 1], [1, 0]]", "[[0, 2], [2, 0]]"),
        lambda value: value + "tensor<32x6144xbf16> outside_compilation",
    ),
)
def test_feature2_main_stablehlo_refuses_mutations(mutation) -> None:
    with pytest.raises(BenchmarkValidationError):
        validate_feature2_main_stablehlo(mutation(_main_stablehlo()))
