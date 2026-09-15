from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
REAL_WS32_PALLAS_HLO_ROOT = Path(
    os.environ.get(
        "GLM_TEST_WS32_PALLAS_REAL_LAYER_HLO_ROOT",
        "/home/gianl/glm-run/"
        "greenfield_ws32_pallas_real_layer_hlo_20260815T094203579842098Z/"
        "fleet_hlo",
    )
)


@pytest.mark.skipif(
    not (REAL_WS32_PALLAS_HLO_ROOT / "layer.rank0.optimized_hlo.txt").is_file(),
    reason="protected WS32 Pallas TPU HLO unavailable",
)
def test_ws32_pallas_real_tpu_hlo_is_exact_live_and_fail_closed() -> None:
    from jaxlib.xla_client import _xla

    from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
        WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256,
        WS32_PALLAS_REAL_LAYER_OPTIMIZED_HLO_SHA256,
        WS32_PALLAS_REAL_LAYER_STABLEHLO_SHA256,
        validate_ws32_pallas_one_layer_hlo,
    )

    stable = (REAL_WS32_PALLAS_HLO_ROOT / "layer.rank0.stablehlo.mlir").read_text()
    optimized = (
        REAL_WS32_PALLAS_HLO_ROOT / "layer.rank0.optimized_hlo.txt"
    ).read_text()

    def validate(stable_text: str, optimized_text: str):
        return validate_ws32_pallas_one_layer_hlo(
            stable_text,
            optimized_text,
            expected_stablehlo_sha256=sha256(
                stable_text.encode("utf-8")
            ).hexdigest(),
            expected_optimized_hlo_sha256=sha256(
                optimized_text.encode("utf-8")
            ).hexdigest(),
            expected_optimized_collective_result_dtype="bf16",
        )

    report = validate(stable, optimized)
    assert report.passed
    assert report.stablehlo_sha256 == WS32_PALLAS_REAL_LAYER_STABLEHLO_SHA256
    assert report.optimized_hlo_sha256 == WS32_PALLAS_REAL_LAYER_OPTIMIZED_HLO_SHA256
    assert report.live_closure_sha256 == WS32_PALLAS_REAL_LAYER_LIVE_CLOSURE_SHA256
    assert report.live_instruction_count == 1307
    assert (
        report.stablehlo_f32_pallas_count,
        report.stablehlo_bf16_pallas_count,
        report.stablehlo_exact_reducer_count,
        report.optimized_f32_pallas_count,
        report.optimized_bf16_pallas_count,
        report.live_pallas_count,
        report.exact_route_region_count,
        report.shared_pallas_count,
    ) == (18, 9, 10, 18, 9, 27, 8, 3)

    first_gate = next(
        line
        for line in optimized.splitlines()
        if "%greenfield_fp8_block_matmul_f32_m8_k1536_n2048.18 =" in line
    ).strip()
    first_up = next(
        line
        for line in optimized.splitlines()
        if "%greenfield_fp8_block_matmul_f32_m8_k1536_n2048.19 =" in line
    ).strip()
    crosswired = optimized.replace(
        "fusion(%greenfield_fp8_block_matmul_f32_m8_k1536_n2048.19), "
        "kind=kLoop, calls=%fused_computation.8",
        "fusion(%greenfield_fp8_block_matmul_f32_m8_k1536_n2048.18), "
        "kind=kLoop, calls=%fused_computation.8",
        1,
    )
    assert crosswired != optimized and first_gate in crosswired and first_up in crosswired
    _xla.hlo_module_from_text(crosswired)
    refused_crosswire = validate(stable, crosswired)
    assert not refused_crosswire.passed
    assert "optimized HLO exact live-root closure drifted" in refused_crosswire.violations

    root_bypass = optimized.replace(
        "ROOT %slice_add_fusion = bf16[1,1536]{1,0:T(2,128)(2,1)} "
        "fusion(%greenfield_fp8_block_matmul_m8_k2048_n1536.17, %psum.70)",
        "ROOT %slice_add_fusion = bf16[1,1536]{1,0:T(2,128)(2,1)} "
        "fusion(%greenfield_fp8_block_matmul_m8_k2048_n1536.17, %param.31)",
        1,
    )
    assert root_bypass != optimized
    _xla.hlo_module_from_text(root_bypass)
    refused_root = validate(stable, root_bypass)
    assert not refused_root.passed
    assert refused_root.live_pallas_count == 3

    wrong_target = optimized.replace(
        'custom_call_target="tpu_custom_call"',
        'custom_call_target="rogue_custom_call"',
        1,
    )
    _xla.hlo_module_from_text(wrong_target)
    refused_target = validate(stable, wrong_target)
    assert not refused_target.passed
    assert any("Pallas kernel contract drifted" in item for item in refused_target.violations)

    wrong_layout = optimized.replace(
        "operand_layout_constraints={bf16[8,1536]{1,0}, "
        "u8[2048,1536]{1,0}, f32[16,128]{1,0}}",
        "operand_layout_constraints={bf16[8,1536]{0,1}, "
        "u8[2048,1536]{1,0}, f32[16,128]{1,0}}",
        1,
    )
    _xla.hlo_module_from_text(wrong_layout)
    refused_layout = validate(stable, wrong_layout)
    assert not refused_layout.passed
    assert any("Pallas kernel contract drifted" in item for item in refused_layout.violations)

    wrong_group = optimized.replace(
        "replica_groups={{0,1,2,3},{4,5,6,7}",
        "replica_groups={{0,1,2,4},{3,5,6,7}",
        1,
    )
    _xla.hlo_module_from_text(wrong_group)
    refused_group = validate(stable, wrong_group)
    assert not refused_group.passed
    assert any("unknown replica group" in item for item in refused_group.violations)

    wrong_reducer = stable.replace("stablehlo.add %arg30, %arg31", "stablehlo.maximum %arg30, %arg31", 1)
    refused_reducer = validate(wrong_reducer, optimized)
    assert not refused_reducer.passed
    assert refused_reducer.stablehlo_exact_reducer_count == 9

    wrong_kernel = stable.replace(
        f'kernel_name = "greenfield_fp8_block_matmul_f32_m8_k1536_n2048"',
        '/* kernel_name = "greenfield_fp8_block_matmul_f32_m8_k1536_n2048" */ '
        'kernel_name = "rogue_kernel"',
        1,
    )
    refused_kernel = validate(wrong_kernel, optimized)
    assert not refused_kernel.passed
    assert any("StableHLO Pallas kernel cardinality drifted" in item for item in refused_kernel.violations)

    unreachable = optimized.replace(
        "ENTRY %main",
        "%rogue_unreachable () -> f32[] {\n"
        "  ROOT %rogue = f32[] constant(0)\n"
        "}\n\nENTRY %main",
        1,
    )
    _xla.hlo_module_from_text(unreachable)
    refused_unreachable = validate(stable, unreachable)
    assert not refused_unreachable.passed
    assert any(
        "unreachable computations" in item
        for item in refused_unreachable.violations
    )

    from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
        _called_computations,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    plain = next(
        item
        for item in parse_hlo_module(optimized).instructions
        if item.raw_opcode == "add" and "to_apply=" not in item.raw_line
    )
    with pytest.raises(ValueError, match="unsupported callee attributes"):
        _called_computations(replace(plain, raw_line=plain.raw_line + ", to_apply=%rogue"))
