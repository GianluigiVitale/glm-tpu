from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from functools import lru_cache
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_dense_convolution.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_projection_reduction_probe.sh"
REAL_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_convolution_"
        "20260813T000326337357270Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_OPTIMIZED_HLO_SHA256 = (
    "e3a2538f8d158f2113e93563ba3ba3a24e51a57b45981db33a7b3b7fdc857ca0"
)
REAL_M32_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_M32_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_m32_convolution_"
        "20260813T032542730910073Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_M32_OPTIMIZED_HLO_SHA256 = (
    "c9c9bf90c9528016846ccea48e04877e0ee0e44bc2d83f3ea0cbd9997162266a"
)
REAL_M32_LAYER1_ROOT = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_m32_cross_layer_"
    "20260813T065606056966514Z/hlo"
)
REAL_M32_LAYER1_OPTIMIZED_HLO = (
    REAL_M32_LAYER1_ROOT / "dense_convolution.optimized_hlo.txt"
)
REAL_M32_LAYER1_OPTIMIZED_HLO_SHA256 = (
    "41f9e6fb12b374b8d95fab39ad2e3d4a9a505f65c61ccb3ac4ee0f946132ae70"
)
REAL_M32_LAYER1_STABLEHLO = (
    REAL_M32_LAYER1_ROOT / "dense_convolution.stablehlo.mlir"
)
REAL_M32_LAYER1_STABLEHLO_SHA256 = (
    "c77c126ed2ceeead3f00e71c5467463612ab2f88b1ab58f7993ee09164bd6474"
)
ACCEPTED_M32_ROOT = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/"
    "decode_projection_lowering/8k/"
    "greenfield_accepted_decode_projection_lowering_20260811T184908676873350Z"
)
SPEC = importlib.util.spec_from_file_location("dense_convolution_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _synthetic_hlo() -> str:
    lines = [
        "HloModule dense_convolution, num_partitions=4",
        "%sum_reducer (lhs: f32[], rhs: f32[]) -> f32[] {",
        "  %lhs = f32[] parameter(0)",
        "  %rhs = f32[] parameter(1)",
        "  ROOT %sum_value = f32[] add(%lhs, %rhs)",
        "}",
        "ENTRY main {",
        "  %normalized = bf16[1,6144] parameter(0)",
        "  %residual = bf16[1,6144] parameter(100)",
        "  %norm = bf16[6144] parameter(101)",
    ]
    down_values = []
    parameter = 1
    for index in range(8):
        lines.extend(
            [
                f"  %gate_up_weight.{index} = bf16[6144,768] "
                f"parameter({parameter})",
                f"  %gate_up.{index} = f32[1,768] convolution("
                f"%normalized, %gate_up_weight.{index}), dim_labels=bf_io->bf, "
                'metadata={op_name="jit(probe)/'
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/gate_up"
                '"}',
                f"  %gate_up_bf16.{index} = bf16[1,768] "
                f"convert(%gate_up.{index})",
                f"  %gate.{index} = bf16[1,384] slice("
                f"%gate_up_bf16.{index}), slice={{[0:1], [0:384]}}",
                f"  %up.{index} = bf16[1,384] slice("
                f"%gate_up_bf16.{index}), slice={{[0:1], [384:768]}}",
                f"  %negate.{index} = bf16[1,384] negate(%gate.{index}), "
                'metadata={op_name="jit(probe)/'
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/neg"
                '"}',
                f"  %exp.{index} = bf16[1,384] exponential(%negate.{index}), "
                'metadata={op_name="jit(probe)/'
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/exp"
                '"}',
                f"  %one.{index} = bf16[1,384] constant(1)",
                f"  %denominator.{index} = bf16[1,384] add("
                f"%one.{index}, %exp.{index}), metadata={{op_name=\"jit(probe)/"
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/add\"}}",
                f"  %sigmoid.{index} = bf16[1,384] divide("
                f"%one.{index}, %denominator.{index}), metadata={{op_name=\"jit(probe)/"
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/div\"}}",
                f"  %silu.{index} = bf16[1,384] multiply("
                f"%gate.{index}, %sigmoid.{index}), metadata={{op_name=\"jit(probe)/"
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/mul\"}}",
                f"  %activated.{index} = bf16[1,384] multiply("
                f"%silu.{index}, %up.{index}), metadata={{op_name=\"jit(probe)/"
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/mul\"}}",
                f"  %down_weight.{index} = bf16[384,6144] "
                f"parameter({parameter + 1})",
                f"  %down.{index} = f32[1,6144] convolution("
                f"%activated.{index}, %down_weight.{index}), "
                "dim_labels=bf_io->bf, "
                'metadata={op_name="jit(probe)/'
                f"greenfield_dense_convolution_virtual_rank_{index:02d}/down"
                '"}',
                f"  %down_bf16.{index} = bf16[1,6144] "
                f"convert(%down.{index})",
                f"  %down_row.{index} = bf16[1,1,6144] "
                f"reshape(%down_bf16.{index})",
            ]
        )
        down_values.append(f"%down_row.{index}")
        parameter += 2
    lines.extend(
        [
            "  %stack = bf16[8,1,6144] concatenate("
            + ", ".join(down_values)
            + "), dimensions={0}",
            "  %gather = bf16[4,8,1,6144] all-gather(%stack), "
            "channel_id=1, replica_groups={{0,1,2,3}}, dimensions={0}, "
            "use_global_device_ids=true, "
            'metadata={op_name="jit(probe)/'
            "greenfield_strategy_nd_row0_dense_convolution_down/"
            'all-gather"}',
        ]
    )
    association_scope = (
        'metadata={op_name="jit(probe)/'
        "greenfield_strategy_nd_row0_dense_convolution_down/"
        'greenfield_strategy_nd_row0_association/add"}'
    )
    y_roots = []
    for band, cross in enumerate((False, True, False)):
        leaves = []
        for row in range(4):
            name = f"%association_y_leaf.{band}.{row}"
            lines.append(
                f"  {name} = bf16[2,4,1,2048] slice(%gather), "
                f"slice={{[{row}:{row + 1}], [0:2], [0:4], [0:1], "
                f"[{band * 2048}:{(band + 1) * 2048}]}}"
            )
            leaves.append(name)
        pairs = ((0, 3), (1, 2)) if cross else ((0, 1), (2, 3))
        left = f"%association_y.{band}.left"
        right = f"%association_y.{band}.right"
        root = f"%association_y.{band}.root"
        lines.extend(
            [
                f"  {left} = bf16[2,4,1,2048] add("
                f"{leaves[pairs[0][0]]}, {leaves[pairs[0][1]]}), "
                f"{association_scope}",
                f"  {right} = bf16[2,4,1,2048] add("
                f"{leaves[pairs[1][0]]}, {leaves[pairs[1][1]]}), "
                f"{association_scope}",
                f"  {root} = bf16[2,4,1,2048] add({left}, {right}), "
                f"{association_scope}",
            ]
        )
        y_roots.append(root)
    lines.extend(
        [
            "  %association_y_stack = bf16[2,4,1,6144] concatenate("
            + ", ".join(y_roots)
            + "), dimensions={3}",
            "  %association_x_left = bf16[4,1,6144] "
            "slice(%association_y_stack), slice={[0:1], [0:4], [0:1], [0:6144]}",
            "  %association_x_right = bf16[4,1,6144] "
            "slice(%association_y_stack), slice={[1:2], [0:4], [0:1], [0:6144]}",
            "  %association_x = bf16[4,1,6144] add("
            "%association_x_left, %association_x_right), "
            + association_scope,
        ]
    )
    z_roots = []
    for segment in range(24):
        leaves = []
        for row in range(4):
            name = f"%association_z_leaf.{segment}.{row}"
            lines.append(
                f"  {name} = bf16[1,256] slice(%association_x), "
                f"slice={{[{row}:{row + 1}], [0:1], "
                f"[{segment * 256}:{(segment + 1) * 256}]}}"
            )
            leaves.append(name)
        pairs = ((0, 3), (1, 2)) if segment % 2 else ((0, 1), (2, 3))
        left = f"%association_z.{segment}.left"
        right = f"%association_z.{segment}.right"
        root = f"%association_z.{segment}.root"
        lines.extend(
            [
                f"  {left} = bf16[1,256] add("
                f"{leaves[pairs[0][0]]}, {leaves[pairs[0][1]]}), "
                f"{association_scope}",
                f"  {right} = bf16[1,256] add("
                f"{leaves[pairs[1][0]]}, {leaves[pairs[1][1]]}), "
                f"{association_scope}",
                f"  {root} = bf16[1,256] add({left}, {right}), "
                f"{association_scope}",
            ]
        )
        z_roots.append(root)
    rms_scope = (
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_layer1_rmsnorm/'
    )
    lines.extend(
        [
            "  %update = bf16[1,6144] concatenate("
            + ", ".join(z_roots)
            + "), dimensions={1}",
            "  %update_f32 = f32[1,6144] convert(%update)",
            "  %residual_f32 = f32[1,6144] convert(%residual)",
            "  %combined = f32[1,6144] add(%update_f32, %residual_f32), "
            + rms_scope
            + 'add"}',
            "  %square = f32[1,6144] multiply(%combined, %combined), "
            + rms_scope
            + 'square"}',
            "  %zero = f32[] constant(0)",
            "  %sum = f32[1] reduce(%square, %zero), dimensions={1}, "
            "to_apply=%sum_reducer, "
            + rms_scope
            + 'reduce_sum"}',
            "  %sum_row = f32[1,1] reshape(%sum)",
            "  %width = f32[1,1] constant(6144)",
            "  %mean = f32[1,1] divide(%sum_row, %width), "
            + rms_scope
            + 'div"}',
            "  %epsilon = f32[1,1] constant(0.00001)",
            "  %variance = f32[1,1] add(%mean, %epsilon), "
            + rms_scope
            + 'add"}',
            "  %inverse = f32[1,1] rsqrt(%variance), "
            + rms_scope
            + 'rsqrt"}',
            "  %inverse_wide = f32[1,6144] broadcast(%inverse), dimensions={0,1}",
            "  %rms_normalized = f32[1,6144] multiply(%combined, %inverse_wide), "
            + rms_scope
            + 'mul"}',
            "  %rounded = bf16[1,6144] convert(%rms_normalized)",
            "  %norm_wide = bf16[1,6144] broadcast(%norm), dimensions={1}",
            "  %layer1 = bf16[1,6144] multiply(%rounded, %norm_wide), "
            + rms_scope
            + 'mul"}',
            "  ROOT %root = (bf16[1,6144], bf16[1,6144]) "
            "tuple(%update, %layer1)",
            "}",
        ]
    )
    return "\n".join(lines)


def _synthetic_m32_hlo() -> str:
    head, tail = _synthetic_hlo().split("  %stack =", 1)
    head = head.replace(
        "  %normalized = bf16[1,6144] parameter(0)",
        "  %normalized = bf16[1,6144] parameter(0)\n"
        "  %m32_zero = bf16[] constant(0)\n"
        "  %m32_padded = bf16[32,6144] pad(%normalized, %m32_zero), "
        "padding=0_31x0_0",
    )
    head = head.replace("convolution(%normalized,", "convolution(%m32_padded,")
    for before, after in (
        ("f32[1,768]", "f32[32,768]"),
        ("bf16[1,768]", "bf16[32,768]"),
        ("bf16[1,384]", "bf16[32,384]"),
        ("f32[1,6144]", "f32[32,6144]"),
    ):
        head = head.replace(before, after)
    head = head.replace(
        "slice={[0:1], [0:384]}", "slice={[0:32], [0:384]}"
    ).replace(
        "slice={[0:1], [384:768]}", "slice={[0:32], [384:768]}"
    )
    for index in range(8):
        head = head.replace(
            f"  %down_bf16.{index} = bf16[1,6144] ",
            f"  %down_bf16.{index} = bf16[32,6144] ",
        ).replace(
            f"  %down_row.{index} = bf16[1,1,6144] "
            f"reshape(%down_bf16.{index})",
            f"  %down_row.{index} = bf16[1,32,6144] "
            f"reshape(%down_bf16.{index})",
        )
    tail = tail.replace(
        "bf16[8,1,6144] concatenate(",
        "bf16[8,32,6144] concatenate(",
        1,
    ).replace(
        "  %gather = bf16[4,8,1,6144] all-gather(%stack)",
        "  %m32_anchor = bf16[8,32,6144] optimization-barrier(%stack)\n"
        "  %m32_live = bf16[8,1,6144] slice(%m32_anchor), "
        "slice={[0:8], [0:1], [0:6144]}\n"
        "  %gather = bf16[4,8,1,6144] all-gather(%m32_live)",
        1,
    )
    return head + "  %stack =" + tail


def _synthetic_m32_layer1_only_hlo() -> str:
    hlo = _synthetic_m32_hlo()
    rms_start = hlo.index("  %update_f32 =")
    root_start = hlo.index("  ROOT %root =", rms_start)
    rms = hlo[rms_start:root_start]
    rms = rms.replace(
        "  %update_f32 = f32[1,6144] convert(%update)",
        "  %rms_zero = bf16[] constant(0)\n"
        "  %update_m32 = bf16[32,6144] pad(%update, %rms_zero), "
        "padding=0_31x0_0\n"
        "  %residual_m32 = bf16[32,6144] pad(%residual, %rms_zero), "
        "padding=0_31x0_0\n"
        "  %update_f32 = f32[32,6144] convert(%update_m32)",
        1,
    ).replace(
        "  %residual_f32 = f32[1,6144] convert(%residual)",
        "  %residual_f32 = f32[32,6144] convert(%residual_m32)",
        1,
    )
    rms = rms.replace("f32[1,6144]", "f32[32,6144]")
    rms = rms.replace("f32[1,1]", "f32[32,1]")
    rms = rms.replace("f32[1] reduce", "f32[32] reduce")
    rms = rms.replace("bf16[1,6144] convert", "bf16[32,6144] convert")
    rms = rms.replace(
        "  %norm_wide = bf16[1,6144] broadcast(%norm), dimensions={1}",
        "  %norm_seed = bf16[1,6144] broadcast(%norm), dimensions={1}\n"
        "  %norm_wide = bf16[32,6144] broadcast(%norm_seed), "
        "dimensions={0,1}",
        1,
    ).replace(
        "  %layer1 = bf16[1,6144] multiply(%rounded, %norm_wide), ",
        "  %layer1_m32 = bf16[32,6144] multiply(%rounded, %norm_wide), ",
        1,
    )
    rms += (
        "  %layer1 = bf16[1,6144] slice(%layer1_m32), "
        "slice={[0:1], [0:6144]}\n"
    )
    hlo = hlo[:rms_start] + rms + hlo[root_start:]
    return hlo.replace(
        "  ROOT %root = (bf16[1,6144], bf16[1,6144]) "
        "tuple(%update, %layer1)",
        "  ROOT %layer1_result = bf16[1,6144] copy(%layer1)",
        1,
    )


def _synthetic_final_layout_optimized_hlo() -> str:
    hlo = _synthetic_m32_layer1_only_hlo().replace(
        "  %residual = bf16[1,6144] parameter(100)",
        "  %residual = bf16[1,6144] parameter(1)",
        1,
    ).replace(
        "  %norm = bf16[6144] parameter(101)",
        "  %norm = bf16[6144] parameter(6)\n"
        "  %packed_gate_bits = u8[1,8,6144,768] parameter(2)\n"
        "  %packed_gate_scale = f32[1,8,48,6] parameter(3)\n"
        "  %packed_down_bits = u8[1,8,384,6144] parameter(4)\n"
        "  %packed_down_scale = f32[1,8,3,48] parameter(5)",
        1,
    )
    for rank in range(8):
        scope = (
            'metadata={op_name="jit(probe)/'
            f"greenfield_dense_convolution_virtual_rank_{rank:02d}/"
        )
        gate_parameter = re.compile(
            rf"  %gate_up_weight\.{rank} = bf16\[6144,768\] parameter\([0-9]+\)"
        )
        gate_dequant = "\n".join(
            (
                f"  %gate_bits_slice.{rank} = u8[1,1,6144,768] "
                f"slice(%packed_gate_bits), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:6144], [0:768]}}",
                f"  %gate_bits.{rank} = u8[6144,768] "
                f"bitcast(%gate_bits_slice.{rank})",
                f"  %gate_fp8.{rank} = f8e4m3fn[6144,768] "
                f"bitcast-convert(%gate_bits.{rank}), {scope}bitcast_convert_type\"}}",
                f"  %gate_f32.{rank} = f32[6144,768] convert(%gate_fp8.{rank}), "
                f"{scope}convert_element_type\"}}",
                f"  %gate_scale_slice.{rank} = f32[1,1,48,6] "
                f"slice(%packed_gate_scale), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:48], [0:6]}}",
                f"  %gate_scale_seed.{rank} = f32[48,6] "
                f"reshape(%gate_scale_slice.{rank})",
                f"  %gate_scale_inner.{rank} = f32[48,128,6] "
                f"broadcast(%gate_scale_seed.{rank}), dimensions={{0,2}}",
                f"  %gate_scale_middle.{rank} = f32[6144,6] "
                f"reshape(%gate_scale_inner.{rank})",
                f"  %gate_scale_outer.{rank} = f32[6144,6,128] "
                f"broadcast(%gate_scale_middle.{rank}), dimensions={{0,1}}",
                f"  %gate_scale_wide.{rank} = f32[6144,768] "
                f"reshape(%gate_scale_outer.{rank})",
                f"  %gate_scaled.{rank} = f32[6144,768] multiply("
                f"%gate_f32.{rank}, %gate_scale_wide.{rank}), {scope}mul\"}}",
                f"  %gate_up_weight.{rank} = bf16[6144,768]{{1,0}} "
                f"convert(%gate_scaled.{rank}), {scope}convert_element_type\"}}",
            )
        )
        hlo, count = gate_parameter.subn(gate_dequant, hlo, count=1)
        assert count == 1
        down_parameter = re.compile(
            rf"  %down_weight\.{rank} = bf16\[384,6144\] parameter\([0-9]+\)"
        )
        down_dequant = "\n".join(
            (
                f"  %down_bits_slice.{rank} = u8[1,1,384,6144] "
                f"slice(%packed_down_bits), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:384], [0:6144]}}",
                f"  %down_bits.{rank} = u8[384,6144] "
                f"bitcast(%down_bits_slice.{rank})",
                f"  %down_fp8.{rank} = f8e4m3fn[384,6144] "
                f"bitcast-convert(%down_bits.{rank}), {scope}bitcast_convert_type\"}}",
                f"  %down_f32.{rank} = f32[384,6144] convert(%down_fp8.{rank}), "
                f"{scope}convert_element_type\"}}",
                f"  %down_scale_slice.{rank} = f32[1,1,3,48] "
                f"slice(%packed_down_scale), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:3], [0:48]}}",
                f"  %down_scale_seed.{rank} = f32[3,48] "
                f"reshape(%down_scale_slice.{rank})",
                f"  %down_scale_inner.{rank} = f32[3,128,48] "
                f"broadcast(%down_scale_seed.{rank}), dimensions={{0,2}}",
                f"  %down_scale_middle.{rank} = f32[384,48] "
                f"reshape(%down_scale_inner.{rank})",
                f"  %down_scale_outer.{rank} = f32[384,48,128] "
                f"broadcast(%down_scale_middle.{rank}), dimensions={{0,1}}",
                f"  %down_scale_wide.{rank} = f32[384,6144] "
                f"reshape(%down_scale_outer.{rank})",
                f"  %down_scaled.{rank} = f32[384,6144] multiply("
                f"%down_f32.{rank}, %down_scale_wide.{rank}), {scope}mul\"}}",
                f"  %down_weight.{rank} = bf16[384,6144]{{1,0}} "
                f"convert(%down_scaled.{rank}), {scope}convert_element_type\"}}",
            )
        )
        hlo, count = down_parameter.subn(down_dequant, hlo, count=1)
        assert count == 1
    return hlo


def _synthetic_m32_hlo_with_fused_live_row(*, rogue_return: bool) -> str:
    hlo = _synthetic_m32_hlo()
    start = hlo.index("  %stack =")
    end = hlo.index("\n  %gather =", start)
    body = hlo[start:end]
    parameters = []
    for index in range(8):
        parameters.append(f"%p{index}: bf16[1,32,6144]")
        body = body.replace(f"%down_row.{index}", f"%p{index}")
    if rogue_return:
        body = body.replace("  %m32_live =", "  %m32_exact =", 1)
        body += (
            "\n  ROOT %m32_live = bf16[8,1,6144] "
            "add(%m32_exact, %m32_exact)"
        )
    else:
        body = body.replace("  %m32_live =", "  ROOT %m32_live =", 1)
    computation = (
        "%fused_m32_live ("
        + ", ".join(parameters)
        + ") -> bf16[8,1,6144] {\n"
        + "".join(
            f"  %p{index} = bf16[1,32,6144] parameter({index})\n"
            for index in range(8)
        )
        + body
        + "\n}\n\n"
    )
    caller = (
        "  %m32_live = bf16[8,1,6144] fusion("
        + ", ".join(f"%down_row.{index}" for index in range(8))
        + "), kind=kLoop, calls=%fused_m32_live"
    )
    header = hlo.index("\n") + 1
    return hlo[:header] + computation + hlo[header:start] + caller + hlo[end:]


def _synthetic_m32_hlo_with_fused_rogue_down() -> str:
    hlo = _synthetic_m32_hlo()
    down_weight = (
        "  %down_weight.0 = bf16[384,6144] parameter(2)\n"
    )
    down = (
        "  %down.0 = f32[32,6144] convolution(%activated.0, "
        "%down_weight.0), dim_labels=bf_io->bf, "
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_virtual_rank_00/down"}\n'
    )
    rounded = "  %down_bf16.0 = bf16[32,6144] convert(%down.0)\n"
    row = (
        "  %down_row.0 = bf16[1,32,6144] reshape(%down_bf16.0)\n"
    )
    for line in (down_weight, down, rounded, row):
        assert line in hlo
    hlo = hlo.replace(down, "", 1).replace(rounded, "", 1).replace(row, "", 1)
    caller = (
        "  %down_row.0 = bf16[1,32,6144] fusion("
        "%activated.0, %down_weight.0), kind=kLoop, "
        "calls=%fused_rogue_down\n"
    )
    hlo = hlo.replace(down_weight, down_weight + caller, 1)
    computation = (
        "%fused_rogue_down (%p0: bf16[32,384], "
        "%p1: bf16[384,6144]) -> bf16[1,32,6144] {\n"
        "  %p0 = bf16[32,384] parameter(0)\n"
        "  %p1 = bf16[384,6144] parameter(1)\n"
        "  %down.0 = f32[32,6144] convolution(%p0, %p1), "
        "dim_labels=bf_io->bf, metadata={op_name=\"jit(probe)/"
        "greenfield_dense_convolution_virtual_rank_00/down\"}\n"
        "  %rounded.0 = bf16[32,6144] convert(%down.0)\n"
        "  %rogue.0 = bf16[32,6144] add(%rounded.0, %rounded.0)\n"
        "  ROOT %row.0 = bf16[1,32,6144] reshape(%rogue.0)\n"
        "}\n\n"
    )
    header = hlo.index("\n") + 1
    return hlo[:header] + computation + hlo[header:]


def _synthetic_hlo_with_fused_y0(*, rogue_return: str | None) -> str:
    hlo = _synthetic_hlo()
    start_marker = "  %association_y_leaf.0.0 ="
    end_marker = (
        "metadata={op_name=\"jit(probe)/"
        "greenfield_strategy_nd_row0_dense_convolution_down/"
        "greenfield_strategy_nd_row0_association/add\"}"
    )
    start = hlo.index(start_marker)
    root_start = hlo.index("  %association_y.0.root =", start)
    end = hlo.index(end_marker, root_start) + len(end_marker)
    body = hlo[start:end].replace("slice(%gather)", "slice(%fused_gather)")
    if rogue_return is not None:
        body = body.replace(
            "%association_y.0.root =",
            "%association_y.0.scoped_root =",
            1,
        )
        if rogue_return == "dead":
            body += (
                "\n  ROOT %association_y.0.root = bf16[2,4,1,2048] add("
                "%association_y_leaf.0.0, %association_y_leaf.0.0)"
            )
        elif rogue_return == "mixed":
            body += (
                "\n  ROOT %association_y.0.root = bf16[2,4,1,2048] add("
                "%association_y.0.scoped_root, %association_y_leaf.0.0)"
            )
        else:
            raise AssertionError(rogue_return)
    else:
        body = body.replace(
            "%association_y.0.root =",
            "ROOT %association_y.0.root =",
            1,
        )
    computation = (
        "%fused_y0 (%fused_gather: bf16[4,8,1,6144]) "
        "-> bf16[2,4,1,2048] {\n"
        "  %fused_gather = bf16[4,8,1,6144] parameter(0)\n"
        f"{body}\n"
        "}\n\n"
    )
    caller = (
        "  %association_y.0.root = bf16[2,4,1,2048] "
        "fusion(%gather), kind=kLoop, calls=%fused_y0"
    )
    header_end = hlo.index("\n") + 1
    return (
        hlo[:header_end]
        + computation
        + hlo[header_end:start]
        + caller
        + hlo[end:]
    )


@lru_cache(maxsize=4)
def _exact_stablehlo(
    compile_rows: int = 1, *, layer1_only: bool = False
) -> str:
    assert compile_rows in (1, 32)
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    _reduce_virtual_tp32_bf16_partials,
    _virtual_dense_convolution_down_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
replicated = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None))
contracts = (
    ((1, 6144), jnp.bfloat16, replicated),
    ((1, 6144), jnp.bfloat16, replicated),
    ((4, 3072, 6144), jnp.uint8, slot),
    ((4, 24, 48), jnp.float32, slot),
    ((4, 3072, 6144), jnp.uint8, slot),
    ((4, 24, 48), jnp.float32, slot),
    ((4, 6144, 3072), jnp.uint8, slot),
    ((4, 48, 24), jnp.float32, slot),
    ((6144,), jnp.bfloat16, replicated),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)

def local(normalized, residual, gate_bits, gate_scale, up_bits, up_scale,
          down_bits, down_scale, norm):
    partials = _virtual_dense_convolution_down_partials(
        normalized, gate_bits[0], gate_scale[0], up_bits[0], up_scale[0],
        down_bits[0], down_scale[0], block_shape=(128, 128),
    )
    with jax.named_scope("greenfield_strategy_nd_row0_dense_convolution_down"):
        update = _reduce_virtual_tp32_bf16_partials(
            partials,
            axis_name="lp4",
            groups=((0, 1, 2, 3),),
            association=STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        )
    layer1 = fused_add_rms_norm(update, residual, norm, epsilon=1e-5)[0]
    return update, layer1

mapped = jax.shard_map(
    local,
    mesh=mesh,
    in_specs=(P(), P(), P("lp4", None, None), P("lp4", None, None),
              P("lp4", None, None), P("lp4", None, None),
              P("lp4", None, None), P("lp4", None, None), P()),
    out_specs=(P(), P()),
    check_vma=False,
)
print(jax.jit(mapped).lower(*arguments).as_text())
'''
    if compile_rows == 32:
        program = program.replace(
            "def local(normalized, residual, gate_bits, gate_scale, up_bits, up_scale,\n"
            "          down_bits, down_scale, norm):\n"
            "    partials = _virtual_dense_convolution_down_partials(",
            "def local(normalized, residual, gate_bits, gate_scale, up_bits, up_scale,\n"
            "          down_bits, down_scale, norm):\n"
            "    normalized = jnp.pad(normalized, ((0, 31), (0, 0)), "
            "constant_values=jnp.bfloat16(0))\n"
            "    partials = _virtual_dense_convolution_down_partials(",
        ).replace(
            "down_bits[0], down_scale[0], block_shape=(128, 128),\n"
            "    )\n"
            "    with jax.named_scope",
            "down_bits[0], down_scale[0], block_shape=(128, 128), "
            "compile_rows=32,\n"
            "    )\n"
            "    partials = jax.lax.optimization_barrier(partials)\n"
            "    partials = partials[:, :1, :]\n"
            "    with jax.named_scope",
        )
    if layer1_only:
        program = program.replace(
            "    layer1 = fused_add_rms_norm(update, residual, norm, epsilon=1e-5)[0]",
            "    update = jnp.pad(update, ((0, 31), (0, 0)), "
            "constant_values=jnp.bfloat16(0))\n"
            "    residual = jnp.pad(residual, ((0, 31), (0, 0)), "
            "constant_values=jnp.bfloat16(0))\n"
            "    layer1 = fused_add_rms_norm(update, residual, norm, epsilon=1e-5)[0]\n"
            "    layer1 = layer1[:1, :]",
        ).replace(
            "    return update, layer1",
            "    return layer1",
        ).replace(
            "    out_specs=(P(), P()),",
            "    out_specs=P(),",
        )
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


@lru_cache(maxsize=1)
def _exact_final_layout_stablehlo() -> str:
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    _reduce_virtual_tp32_bf16_partials,
    _virtual_dense_final_layout_convolution_down_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
replicated = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None, None))
contracts = (
    ((1, 6144), jnp.bfloat16, replicated),
    ((1, 6144), jnp.bfloat16, replicated),
    ((4, 8, 6144, 768), jnp.uint8, slot),
    ((4, 8, 48, 6), jnp.float32, slot),
    ((4, 8, 384, 6144), jnp.uint8, slot),
    ((4, 8, 3, 48), jnp.float32, slot),
    ((6144,), jnp.bfloat16, replicated),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)

def local(normalized, residual, merged_bits, merged_scale,
          down_bits, down_scale, norm):
    normalized = jnp.pad(
        normalized, ((0, 31), (0, 0)),
        constant_values=jnp.bfloat16(0),
    )
    partials = _virtual_dense_final_layout_convolution_down_partials(
        normalized, merged_bits[0], merged_scale[0],
        down_bits[0], down_scale[0], block_shape=(128, 128),
        compile_rows=32,
    )
    partials = jax.lax.optimization_barrier(partials)[:, :1, :]
    with jax.named_scope("greenfield_strategy_nd_row0_dense_convolution_down"):
        update = _reduce_virtual_tp32_bf16_partials(
            partials,
            axis_name="lp4",
            groups=((0, 1, 2, 3),),
            association=STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
        )
    update = jnp.pad(
        update, ((0, 31), (0, 0)),
        constant_values=jnp.bfloat16(0),
    )
    residual = jnp.pad(
        residual, ((0, 31), (0, 0)),
        constant_values=jnp.bfloat16(0),
    )
    layer1 = fused_add_rms_norm(
        update, residual, norm, epsilon=1e-5,
    )[0]
    return layer1[:1, :]

mapped = jax.shard_map(
    local,
    mesh=mesh,
    in_specs=(P(), P(), P("lp4", None, None, None),
              P("lp4", None, None, None),
              P("lp4", None, None, None),
              P("lp4", None, None, None), P()),
    out_specs=P(),
    check_vma=False,
)
print(jax.jit(mapped).lower(*arguments).as_text())
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_dense_convolution_hlo_contract_accepts_exact_graph() -> None:
    stable = MODULE._validate_stablehlo(_exact_stablehlo())
    optimized = MODULE._validate_optimized_hlo(_synthetic_hlo())
    assert stable["passed"], stable["violations"]
    assert optimized["passed"], optimized["violations"]
    assert optimized["gate_up_convolution_count"] == 8
    assert optimized["down_convolution_count"] == 8


def test_dense_convolution_m32_hlo_contract_accepts_only_live_row() -> None:
    stablehlo = _exact_stablehlo(32)
    stable = MODULE._validate_stablehlo(stablehlo, compile_rows=32)
    optimized_hlo = _synthetic_m32_hlo()
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo, compile_rows=32
    )
    assert stable["passed"], stable["violations"]
    assert optimized["passed"], optimized["violations"]
    assert stable["compile_rows"] == optimized["compile_rows"] == 32
    assert stable["live_rows"] == optimized["live_rows"] == 1
    assert not MODULE._validate_stablehlo(stablehlo)["passed"]
    assert not MODULE._validate_optimized_hlo(optimized_hlo)["passed"]

    wrong_pad = stablehlo.replace("high = [31, 0]", "high = [30, 0]", 1)
    wrong_live_row = stablehlo.replace(
        "[0:8, 0:1, 0:6144]", "[0:8, 1:2, 0:6144]", 1
    )
    anchor_line = next(
        line
        for line in stablehlo.splitlines()
        if "stablehlo.optimization_barrier" in line
        and "tensor<8x32x6144xbf16>" in line
    )
    missing_anchor = stablehlo.replace(
        anchor_line,
        anchor_line.replace(
            "stablehlo.optimization_barrier", "stablehlo.reshape"
        ),
        1,
    )
    reordered_anchor = stablehlo.replace(
        anchor_line + "\n",
        "",
        1,
    ).replace(
        "stablehlo.slice %369 [0:8, 0:1, 0:6144]",
        "stablehlo.slice %368 [0:8, 0:1, 0:6144]",
        1,
    )
    assert (
        wrong_pad != stablehlo
        and wrong_live_row != stablehlo
        and missing_anchor != stablehlo
        and reordered_anchor != stablehlo
    )
    assert not MODULE._validate_stablehlo(
        wrong_pad, compile_rows=32
    )["passed"]
    assert not MODULE._validate_stablehlo(
        wrong_live_row, compile_rows=32
    )["passed"]
    assert not MODULE._validate_stablehlo(
        missing_anchor, compile_rows=32
    )["passed"]
    assert not MODULE._validate_stablehlo(
        reordered_anchor, compile_rows=32
    )["passed"]

    optimized_wrong_row = optimized_hlo.replace(
        "slice={[0:8], [0:1], [0:6144]}",
        "slice={[0:8], [1:2], [0:6144]}",
        1,
    )
    assert optimized_wrong_row != optimized_hlo
    assert not MODULE._validate_optimized_hlo(
        optimized_wrong_row, compile_rows=32
    )["passed"]
    optimized_extra_arithmetic = optimized_hlo.replace(
        "  %gather = bf16[4,8,1,6144] all-gather(%m32_live)",
        "  %m32_rogue = bf16[8,1,6144] add(%m32_live, %m32_live)\n"
        "  %gather = bf16[4,8,1,6144] all-gather(%m32_rogue)",
        1,
    )
    assert optimized_extra_arithmetic != optimized_hlo
    assert not MODULE._validate_optimized_hlo(
        optimized_extra_arithmetic, compile_rows=32
    )["passed"]

    fused = MODULE._validate_optimized_hlo(
        _synthetic_m32_hlo_with_fused_live_row(rogue_return=False),
        compile_rows=32,
    )
    assert fused["passed"], fused["violations"]
    fused_rogue = MODULE._validate_optimized_hlo(
        _synthetic_m32_hlo_with_fused_live_row(rogue_return=True),
        compile_rows=32,
    )
    assert not fused_rogue["passed"]
    rogue_down = MODULE._validate_optimized_hlo(
        _synthetic_m32_hlo_with_fused_rogue_down(),
        compile_rows=32,
    )
    assert not rogue_down["passed"]


def test_dense_cross_layer_hlo_contract_requires_layer1_only_result() -> None:
    stablehlo = _exact_stablehlo(32, layer1_only=True)
    stable = MODULE._validate_stablehlo(
        stablehlo, compile_rows=32, layer1_only=True
    )
    optimized_hlo = _synthetic_m32_layer1_only_hlo()
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo, compile_rows=32, layer1_only=True
    )
    assert stable["passed"], stable["violations"]
    assert optimized["passed"], optimized["violations"]
    assert stable["result_mode"] == optimized["result_mode"] == "layer1_only"


    wrong_pad = stablehlo.replace(
        "low = [0, 0], high = [31, 0], interior = [0, 0]",
        "low = [0, 0], high = [30, 0], interior = [0, 0]",
        1,
    )
    assert wrong_pad != stablehlo
    assert not MODULE._validate_stablehlo(
        wrong_pad, compile_rows=32, layer1_only=True
    )["passed"]
    residual_call = re.search(
        r"(%[0-9]+ = func\.call @_pad\()(%arg10)(, %[A-Za-z0-9_]+\))",
        stablehlo,
    )
    assert residual_call is not None
    dense_pad_sources = [
        source
        for source in re.findall(
            r"func\.call @_pad\((%[A-Za-z0-9_]+),", stablehlo
        )
        if source not in {"%arg9", "%arg10"}
    ]
    assert len(dense_pad_sources) == 1
    cross_wired_residual = (
        stablehlo[: residual_call.start(2)]
        + dense_pad_sources[0]
        + stablehlo[residual_call.end(2) :]
    )
    assert not MODULE._validate_stablehlo(
        cross_wired_residual, compile_rows=32, layer1_only=True
    )["passed"]
    assert not MODULE._validate_stablehlo(
        _exact_stablehlo(32), compile_rows=32, layer1_only=True
    )["passed"]
    assert not MODULE._validate_optimized_hlo(
        _synthetic_m32_hlo(), compile_rows=32, layer1_only=True
    )["passed"]
    doubled_result = optimized_hlo.replace(
        "ROOT %layer1_result = bf16[1,6144] copy(%layer1)",
        "ROOT %layer1_result = bf16[1,6144] add(%layer1, %layer1)",
        1,
    )
    assert doubled_result != optimized_hlo
    doubled_contract = MODULE._validate_optimized_hlo(
        doubled_result, compile_rows=32, layer1_only=True
    )
    assert not doubled_contract["passed"]
    assert not doubled_contract["lineage"]["rmsnorm_contract"][
        "exact_result_binding"
    ]
    scoped_mul = (
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_layer1_rmsnorm/mul"}'
    )
    valid_layer1 = (
        "  %layer1_m32 = bf16[32,6144] multiply(%rounded, %norm_wide), "
        + scoped_mul
    )
    reapplied_layer1 = (
        "  %layer1_m32 = bf16[32,6144] multiply(%rounded, %norm_wide)\n"
        "  %rogue_layer1_m32 = bf16[32,6144] multiply("
        "%layer1_m32, %norm_wide), "
        + scoped_mul
    )
    reapplied_weight = optimized_hlo.replace(
        valid_layer1, reapplied_layer1, 1
    ).replace(
        "%layer1 = bf16[1,6144] slice(%layer1_m32)",
        "%layer1 = bf16[1,6144] slice(%rogue_layer1_m32)",
        1,
    )
    assert reapplied_weight != optimized_hlo
    reapplied_contract = MODULE._validate_optimized_hlo(
        reapplied_weight, compile_rows=32, layer1_only=True
    )
    assert not reapplied_contract["passed"]
    assert not reapplied_contract["lineage"]["rmsnorm_contract"][
        "exact_weighted_operand_graph"
    ]
    rogue_variance = optimized_hlo.replace(
        "  %inverse = f32[32,1] rsqrt(%variance), ",
        "  %rogue_variance = f32[32,1] add(%variance, %variance)\n"
        "  %inverse = f32[32,1] rsqrt(%rogue_variance), ",
        1,
    )
    assert rogue_variance != optimized_hlo
    rogue_variance_contract = MODULE._validate_optimized_hlo(
        rogue_variance, compile_rows=32, layer1_only=True
    )
    assert not rogue_variance_contract["passed"]
    assert not rogue_variance_contract["lineage"]["rmsnorm_contract"][
        "exact_reduction_operand_graph"
    ]
    rogue_sum = optimized_hlo.replace(
        "  %sum_row = f32[32,1] reshape(%sum)",
        "  %rogue_sum = f32[32] add(%sum, %sum)\n"
        "  %sum_row = f32[32,1] reshape(%rogue_sum)",
        1,
    )
    assert rogue_sum != optimized_hlo
    rogue_sum_contract = MODULE._validate_optimized_hlo(
        rogue_sum, compile_rows=32, layer1_only=True
    )
    assert not rogue_sum_contract["passed"]
    assert not rogue_sum_contract["lineage"]["rmsnorm_contract"][
        "exact_reduction_operand_graph"
    ]
    rogue_mean = optimized_hlo.replace(
        "  %variance = f32[32,1] add(%mean, %epsilon), ",
        "  %rogue_mean = f32[32,1] add(%mean, %mean)\n"
        "  %variance = f32[32,1] add(%rogue_mean, %epsilon), ",
        1,
    )
    assert rogue_mean != optimized_hlo
    rogue_mean_contract = MODULE._validate_optimized_hlo(
        rogue_mean, compile_rows=32, layer1_only=True
    )
    assert not rogue_mean_contract["passed"]
    assert not rogue_mean_contract["lineage"]["rmsnorm_contract"][
        "exact_reduction_operand_graph"
    ]
    wrong_reducer = optimized_hlo.replace(
        "ROOT %sum_value = f32[] add(%lhs, %rhs)",
        "ROOT %sum_value = f32[] subtract(%lhs, %rhs)",
        1,
    )
    assert wrong_reducer != optimized_hlo
    wrong_reducer_contract = MODULE._validate_optimized_hlo(
        wrong_reducer, compile_rows=32, layer1_only=True
    )
    assert not wrong_reducer_contract["passed"]
    assert not wrong_reducer_contract["lineage"]["rmsnorm_contract"][
        "exact_reduction_operand_graph"
    ]
    wrong_live_row = optimized_hlo.replace(
        "slice={[0:1], [0:6144]}",
        "slice={[1:2], [0:6144]}",
        1,
    )
    assert wrong_live_row != optimized_hlo
    wrong_live_row_contract = MODULE._validate_optimized_hlo(
        wrong_live_row, compile_rows=32, layer1_only=True
    )
    assert not wrong_live_row_contract["passed"]
    assert not wrong_live_row_contract["lineage"]["rmsnorm_contract"][
        "exact_m32_reduction_geometry"
    ]
    wrong_reduction_axis = optimized_hlo.replace(
        "f32[32] reduce(%square, %zero), dimensions={1}",
        "f32[32] reduce(%square, %zero), dimensions={0}",
        1,
    )
    assert wrong_reduction_axis != optimized_hlo
    wrong_reduction_contract = MODULE._validate_optimized_hlo(
        wrong_reduction_axis, compile_rows=32, layer1_only=True
    )
    assert not wrong_reduction_contract["passed"]
    assert not wrong_reduction_contract["lineage"]["rmsnorm_contract"][
        "exact_m32_reduction_geometry"
    ]


def test_dense_final_layout_hlo_contract_requires_accepted_weight_layout() -> None:
    accepted = _synthetic_final_layout_optimized_hlo()
    contract = MODULE._validate_optimized_hlo(
        accepted,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert contract["passed"], contract
    assert contract["exact_accepted_weight_layout"] is True
    assert contract["exact_packed_weight_lineage"] is True
    no_owner_scale = accepted.replace(
        "  %gate_scale_slice.0 = f32[1,1,48,6] "
        "slice(%packed_gate_scale), "
        "slice={[0:1], [0:1], [0:48], [0:6]}",
        "  %gate_scale_no_owner.0 = f32[8,48,6] "
        "reshape(%packed_gate_scale)\n"
        "  %gate_scale_slice.0 = f32[1,48,6] "
        "slice(%gate_scale_no_owner.0), "
        "slice={[0:1], [0:48], [0:6]}",
        1,
    )
    assert no_owner_scale != accepted
    no_owner_contract = MODULE._validate_optimized_hlo(
        no_owner_scale,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert no_owner_contract["passed"], no_owner_contract
    assert no_owner_contract["exact_packed_weight_lineage"] is True
    for source, replacement in (
        (
            "bf16[6144,768]{1,0} convert",
            "bf16[6144,768]{0,1} convert",
        ),
        (
            "bf16[384,6144]{1,0} convert",
            "bf16[384,6144]{0,1} convert",
        ),
    ):
        rejected = MODULE._validate_optimized_hlo(
            accepted.replace(source, replacement, 1),
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
        )
        assert rejected["passed"] is False
        assert "accepted dense convolution weight layout drifted" in (
            rejected["violations"]
        )
    for weight, convolution, shape, live_edge in (
        (
            "%gate_up_weight.0",
            "%gate_up.0",
            "bf16[6144,768]{1,0}",
            "%m32_padded, %gate_up_weight.0",
        ),
        (
            "%down_weight.0",
            "%down.0",
            "bf16[384,6144]{1,0}",
            "%activated.0, %down_weight.0",
        ),
    ):
        marker = f"  {convolution} ="
        rogue = f"%rogue_{weight[1:]}"
        mutated = accepted.replace(
            marker,
            f"  {rogue} = {shape} add({weight}, {weight})\n{marker}",
            1,
        ).replace(live_edge, live_edge.replace(weight, rogue), 1)
        assert mutated != accepted
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
        )
        assert rejected["passed"] is False
        assert rejected["exact_packed_weight_lineage"] is False
        assert "packed dense weight lineage drifted" in rejected["violations"]
    source_mutations = (
        accepted.replace(
            "slice(%packed_gate_bits), slice={[0:1], [0:1], [0:6144], [0:768]}",
            "slice(%packed_gate_bits), slice={[0:1], [1:2], [0:6144], [0:768]}",
            1,
        ),
        accepted.replace(
            "  %gate_scaled.0 = f32[6144,768] multiply("
            "%gate_f32.0, %gate_scale_wide.0)",
            "  %rogue_gate_scale.0 = f32[6144,768] add("
            "%gate_scale_wide.0, %gate_scale_wide.0)\n"
            "  %gate_scaled.0 = f32[6144,768] multiply("
            "%gate_f32.0, %rogue_gate_scale.0)",
            1,
        ),
        accepted.replace(
            "  %down_fp8.0 = f8e4m3fn[384,6144] "
            "bitcast-convert(%down_bits.0)",
            "  %rogue_down_bits.0 = u8[384,6144] add("
            "%down_bits.0, %down_bits.0)\n"
            "  %down_fp8.0 = f8e4m3fn[384,6144] "
            "bitcast-convert(%rogue_down_bits.0)",
            1,
        ),
        accepted.replace(
            "  %gate_bits_slice.0 = u8[1,1,6144,768] "
            "slice(%packed_gate_bits), slice={[0:1], [0:1], [0:6144], [0:768]}",
            "  %reassociated_gate_bits.0 = u8[1,6144,8,768] "
            "reshape(%packed_gate_bits)\n"
            "  %gate_bits_slice.0 = u8[1,6144,1,768] "
            "slice(%reassociated_gate_bits.0), "
            "slice={[0:1], [0:6144], [0:1], [0:768]}",
            1,
        ),
        accepted.replace(
            "  %gate_scale_seed.0 = f32[48,6] "
            "reshape(%gate_scale_slice.0)\n"
            "  %gate_scale_inner.0 = f32[48,128,6] "
            "broadcast(%gate_scale_seed.0), dimensions={0,2}",
            "  %gate_scale_seed.0 = f32[6,48] "
            "reshape(%gate_scale_slice.0)\n"
            "  %gate_scale_inner.0 = f32[6,128,48] "
            "broadcast(%gate_scale_seed.0), dimensions={0,2}",
            1,
        ),
    )
    assert all(mutated != accepted for mutated in source_mutations)
    for mutated in source_mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
        )
        assert rejected["passed"] is False
        assert rejected["exact_packed_weight_lineage"] is False
        assert "packed dense weight lineage drifted" in rejected["violations"]


def test_dense_final_layout_stablehlo_binds_packed_shards_and_sources() -> None:
    stablehlo = _exact_final_layout_stablehlo()
    contract = MODULE._validate_stablehlo(
        stablehlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert contract["passed"], contract
    assert contract["matched_virtual_shards"] == list(range(8))
    dense_prefix = stablehlo.split('"stablehlo.all_gather"', 1)[0]
    assert "stablehlo.transpose" not in dense_prefix
    wrong_first_shard = stablehlo.replace(
        "stablehlo.slice %2 [0:1, 0:6144, 0:768]",
        "stablehlo.slice %2 [1:2, 0:6144, 0:768]",
        1,
    )
    assert wrong_first_shard != stablehlo
    assert not MODULE._validate_stablehlo(
        wrong_first_shard,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )["passed"]
    cross_wired_outer = stablehlo.replace(
        "manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)",
        "manual_computation(%arg0, %arg1, %arg3, %arg2, %arg4, %arg5, %arg6)",
        1,
    )
    assert cross_wired_outer != stablehlo
    assert not MODULE._validate_stablehlo(
        cross_wired_outer,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )["passed"]


@pytest.mark.skipif(
    not ACCEPTED_M32_ROOT.exists(),
    reason="accepted DB532 lowering evidence is not mounted",
)
def test_dense_convolution_m32_source_is_bound_to_db532() -> None:
    arguments = SimpleNamespace(
        accepted_m32_hlo=(
            ACCEPTED_M32_ROOT
            / "accepted_decode_projection_lowering/"
            "jit_step_fun_impl.m32.after_codegen_hlo.txt.gz"
        ),
        accepted_m32_hlo_sha256=(
            "25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c"
        ),
        accepted_m32_summary=(
            ACCEPTED_M32_ROOT / "accepted_decode_projection_lowering/summary.json"
        ),
        accepted_m32_summary_sha256=(
            "409c845c2c9d67a1d6de36f0cccd25d2982850ee86c35645b0839fc78a1507a3"
        ),
        accepted_m32_success=ACCEPTED_M32_ROOT / "SUCCESS",
        accepted_m32_success_sha256=(
            "6ef516dc42e046a996aa1fe542a4b11af5c2a14450e1aba7bfac98ddbf257278"
        ),
    )
    source = MODULE._load_accepted_m32_source(arguments)
    assert source["accepted_m32_hlo_raw_sha256"] == (
        "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
    )
    arguments.accepted_m32_summary_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="accepted M32 summary SHA-256 drifted"):
        MODULE._load_accepted_m32_source(arguments)


@pytest.mark.skipif(
    not REAL_M32_OPTIMIZED_HLO.exists(),
    reason="protected real M32 dense-convolution HLO is not mounted",
)
def test_dense_convolution_m32_contract_replays_fused_tpu_stack() -> None:
    assert MODULE._file_sha256(REAL_M32_OPTIMIZED_HLO) == (
        REAL_M32_OPTIMIZED_HLO_SHA256
    )
    hlo = REAL_M32_OPTIMIZED_HLO.read_text()
    exact = MODULE._validate_optimized_hlo(hlo, compile_rows=32)
    assert exact["passed"], exact["violations"]
    assert exact["lineage"]["ordered_stack_sources"] == [
        [rank] for rank in range(8)
    ]
    assert len(exact["lineage"]["m32_fused_stack_callers"]) == 8

    wrong_index = hlo.replace(
        "%constant.231 = s32[] constant(3)",
        "%constant.231 = s32[] constant(2)",
        1,
    )
    wrong_predecessor = hlo.replace(
        "fusion(%bitcast_dynamic-update-slice_fusion.5, %bitcast.416,",
        "fusion(%bitcast_dynamic-update-slice_fusion.6, %bitcast.416,",
        1,
    )
    missing_round = hlo.replace(
        "%bitcast.280 = bf16[1,32,6144]{2,1,0:T(8,128)(2,1)} "
        "bitcast(%convert_element_type.351)",
        "%bitcast.280 = bf16[1,32,6144]{2,1,0:T(8,128)(2,1)} "
        "bitcast(%conv_general_dilated.97)",
        1,
    )
    rogue_stack = hlo.replace(
        "  %slice.1043 = bf16[8,1,6144]",
        "  %rogue_m32_stack = bf16[8,32,6144] add("
        "%bitcast_dynamic-update-slice_fusion, "
        "%bitcast_dynamic-update-slice_fusion)\n"
        "  %slice.1043 = bf16[8,1,6144]",
        1,
    ).replace(
        "slice(%bitcast_dynamic-update-slice_fusion), ",
        "slice(%rogue_m32_stack), ",
        1,
    )
    wrong_live_row = hlo.replace(
        "slice={[0:8], [0:1], [0:6144]}",
        "slice={[0:8], [1:2], [0:6144]}",
        1,
    )
    mutations = (
        wrong_index,
        wrong_predecessor,
        missing_round,
        rogue_stack,
        wrong_live_row,
    )
    assert all(value != hlo for value in mutations)
    assert all(
        not MODULE._validate_optimized_hlo(value, compile_rows=32)["passed"]
        for value in mutations
    )


@pytest.mark.skipif(
    not REAL_M32_LAYER1_OPTIMIZED_HLO.exists()
    or not REAL_M32_LAYER1_STABLEHLO.exists(),
    reason="failed-closed M32 layer-1 TPU HLO is not mounted",
)
def test_dense_cross_layer_contract_replays_real_tpu_fusions() -> None:
    assert MODULE._file_sha256(REAL_M32_LAYER1_OPTIMIZED_HLO) == (
        REAL_M32_LAYER1_OPTIMIZED_HLO_SHA256
    )
    assert MODULE._file_sha256(REAL_M32_LAYER1_STABLEHLO) == (
        REAL_M32_LAYER1_STABLEHLO_SHA256
    )
    stable = MODULE._validate_stablehlo(
        REAL_M32_LAYER1_STABLEHLO.read_text(),
        compile_rows=32,
        layer1_only=True,
    )
    hlo = REAL_M32_LAYER1_OPTIMIZED_HLO.read_text()
    exact = MODULE._validate_optimized_hlo(
        hlo, compile_rows=32, layer1_only=True
    )
    assert stable["passed"], stable["violations"]
    assert exact["passed"], exact["violations"]
    contract = exact["lineage"]["rmsnorm_contract"]
    assert contract["exact_m32_reduction_geometry"] is True
    assert contract["exact_reduction_operand_graph"] is True
    assert contract["weighted_external_values"] == ["%fusion.143"]

    wrong_live_row = hlo.replace(
        "slice={[0:1], [0:6144]}, metadata={op_name=\"jit(local)/shard_map/"
        "greenfield_dense_convolution_layer1_m32_live_row/slice\"",
        "slice={[1:2], [0:6144]}, metadata={op_name=\"jit(local)/shard_map/"
        "greenfield_dense_convolution_layer1_m32_live_row/slice\"",
        1,
    )
    rogue_combined = hlo.replace(
        "  %slice.1302 = f32[1,6144]",
        "  %rogue_combined = f32[32,6144] add(%param_0.647, "
        "%param_0.647)\n  %slice.1302 = f32[1,6144]",
        1,
    ).replace(
        "slice(%param_0.647), slice={[0:1], [0:6144]}",
        "slice(%rogue_combined), slice={[0:1], [0:6144]}",
        1,
    )
    rogue_rsqrt = hlo.replace(
        "  %bitcast.445 = f32[1]",
        "  %rogue_rsqrt = f32[32] add(%add_rsqrt_fusion, "
        "%add_rsqrt_fusion)\n  %bitcast.445 = f32[1]",
        1,
    ).replace(
        "bitcast(%add_rsqrt_fusion)", "bitcast(%rogue_rsqrt)", 1
    )
    rogue_result = hlo.replace(
        "  ROOT %fusion.143 = bf16[1,6144]",
        "  %fusion.143 = bf16[1,6144]",
        1,
    )
    result_head, result_tail = rogue_result.rsplit("\n}", 1)
    rogue_result = (
        result_head
        + "\n  ROOT %rogue_result = bf16[1,6144] add(%fusion.143, "
        "%fusion.143)\n}"
        + result_tail
    )
    mutations = (wrong_live_row, rogue_combined, rogue_rsqrt, rogue_result)
    assert all(value != hlo for value in mutations)
    assert all(
        not MODULE._validate_optimized_hlo(
            value, compile_rows=32, layer1_only=True
        )["passed"]
        for value in mutations
    )


def test_dense_convolution_hlo_contract_binds_fused_component_root() -> None:
    live = MODULE._validate_optimized_hlo(
        _synthetic_hlo_with_fused_y0(rogue_return=None)
    )
    assert live["passed"], live["violations"]
    dead = MODULE._validate_optimized_hlo(
        _synthetic_hlo_with_fused_y0(rogue_return="dead")
    )
    assert not dead["passed"]
    assert not dead["lineage"]["association_edge_graph"]["y"]["exact"]
    mixed = MODULE._validate_optimized_hlo(
        _synthetic_hlo_with_fused_y0(rogue_return="mixed")
    )
    assert not mixed["passed"]
    assert not mixed["lineage"]["association_edge_graph"]["y"]["exact"]


@pytest.mark.skipif(
    not REAL_OPTIMIZED_HLO.exists(),
    reason="protected real dense-convolution HLO is not mounted",
)
def test_dense_convolution_hlo_contract_replays_real_tpu_fusions() -> None:
    assert MODULE._file_sha256(REAL_OPTIMIZED_HLO) == REAL_OPTIMIZED_HLO_SHA256
    hlo = REAL_OPTIMIZED_HLO.read_text()
    exact = MODULE._validate_optimized_hlo(hlo)
    assert exact["passed"], exact["violations"]

    missing_silu_round = hlo.splitlines()
    for index, line in enumerate(missing_silu_round):
        if line.lstrip().startswith("%mul.237 ="):
            missing_silu_round[index] = line.replace(
                '"original_type":"BF16"', '"original_type":"F32"'
            )
            break
    dead_silu_round = list(missing_silu_round)
    dead_silu_round.insert(
        index + 1,
        "  %dead_silu_round = bf16[1,384]{1,0:T(2,128)(2,1)} "
        "convert(%mul.237)",
    )
    rogue_gate = hlo.replace(
        "  %convert_bitcast_fusion.7 = bf16[1,1,6144]",
        "  %rogue_gate.0 = bf16[1,768]{1,0:T(2,128)(2,1)S(3)} "
        "add(%fusion.77, %fusion.77)\n"
        "  %convert_bitcast_fusion.7 = bf16[1,1,6144]",
        1,
    ).replace(
        "fusion(%bitcast.405, %bitcast.422, %fusion.77), kind=kOutput, "
        "calls=%fused_computation.162",
        "fusion(%bitcast.405, %bitcast.422, %rogue_gate.0), kind=kOutput, "
        "calls=%fused_computation.162",
        1,
    )
    rogue_activation = hlo.replace(
        "  %conv_general_dilated.95 = f32[1,6144]",
        "  %rogue_activation.0 = bf16[1,384]{1,0:T(2,128)(2,1)} "
        "add(%fusion.79, %fusion.79)\n"
        "  %conv_general_dilated.95 = f32[1,6144]",
        1,
    ).replace(
        "convolution(%fusion.79, %fusion.69)",
        "convolution(%rogue_activation.0, %fusion.69)",
        1,
    )
    mutations = (
        hlo.replace(
            "padding=0_0x0_0x0_0x0_4096",
            "padding=0_0x0_0x0_0x0_4095",
            1,
        ),
        hlo.replace(
            "%constant.136 = s32[] constant(5888)",
            "%constant.136 = s32[] constant(5632)",
            1,
        ),
        "\n".join(missing_silu_round) + "\n",
        "\n".join(dead_silu_round) + "\n",
        rogue_gate,
        rogue_activation,
        hlo.replace(
            "add(%get-tuple-element.19, %get-tuple-element.16)",
            "add(%get-tuple-element.19, %get-tuple-element.19)",
            1,
        ),
    )
    assert all(value != hlo for value in mutations)
    assert all(
        not MODULE._validate_optimized_hlo(value)["passed"]
        for value in mutations
    )


def test_dense_convolution_hlo_contract_rejects_wrong_geometry_and_group() -> None:
    hlo = _synthetic_hlo()
    wrong_width = hlo.replace("f32[1,768]", "f32[1,769]", 1)
    assert not MODULE._validate_optimized_hlo(wrong_width)["passed"]
    escaped = hlo.replace("{{0,1,2,3}}", "{{0,1,2,3,4,5,6,7}}")
    assert not MODULE._validate_optimized_hlo(escaped)["passed"]
    wrong_scope = hlo.replace(
        "greenfield_strategy_nd_row0_dense_convolution_down",
        "unscoped_dense_convolution",
    )
    assert not MODULE._validate_optimized_hlo(wrong_scope)["passed"]
    wrong_labels = hlo.replace("dim_labels=bf_io->bf", "dim_labels=fb_io->fb", 1)
    assert not MODULE._validate_optimized_hlo(wrong_labels)["passed"]
    duplicate_rank = hlo.replace(
        "greenfield_dense_convolution_virtual_rank_01",
        "greenfield_dense_convolution_virtual_rank_00",
    )
    assert not MODULE._validate_optimized_hlo(duplicate_rank)["passed"]


def test_dense_convolution_hlo_contract_rejects_bypass_and_cross_wiring() -> None:
    hlo = _synthetic_hlo()
    bypass = hlo.replace("all-gather(%stack)", "all-gather(%down_row.0)")
    assert not MODULE._validate_optimized_hlo(bypass)["passed"]
    cross_wired = hlo.replace(
        "%activated.1, %down_weight.1",
        "%activated.0, %down_weight.1",
    )
    assert not MODULE._validate_optimized_hlo(cross_wired)["passed"]
    dead_collective = hlo.replace(
        "tuple(%update, %layer1)",
        "tuple(%down_bf16.0, %down_bf16.0)",
    )
    assert not MODULE._validate_optimized_hlo(dead_collective)["passed"]
    missing_rmsnorm = hlo.replace(
        "tuple(%update, %layer1)",
        "tuple(%update, %update)",
    )
    assert not MODULE._validate_optimized_hlo(missing_rmsnorm)["passed"]
    activation_bypass = hlo.replace(
        "%activated.0, %down_weight.0", "%up.0, %down_weight.0"
    )
    assert not MODULE._validate_optimized_hlo(activation_bypass)["passed"]
    row_swap = hlo.replace(
        "concatenate(%down_row.0, %down_row.1",
        "concatenate(%down_row.1, %down_row.0",
    )
    assert not MODULE._validate_optimized_hlo(row_swap)["passed"]
    tree_bypass = hlo.replace(
        "greenfield_strategy_nd_row0_association/add",
        "rogue_strategy_nd_add",
        1,
    )
    assert not MODULE._validate_optimized_hlo(tree_bypass)["passed"]
    reassociated = hlo.replace(
        "%gate.0, %sigmoid.0", "%up.0, %sigmoid.0", 1
    ).replace("%silu.0, %up.0", "%silu.0, %gate.0", 1)
    assert not MODULE._validate_optimized_hlo(reassociated)["passed"]
    self_add = hlo.replace(
        "%association_y_leaf.0.0, %association_y_leaf.0.1",
        "%association_y_leaf.0.0, %association_y_leaf.0.0",
        1,
    )
    assert not MODULE._validate_optimized_hlo(self_add)["passed"]
    duplicate_physical_leaf = hlo.replace(
        "slice={[0:1], [0:2], [0:4], [0:1], [0:2048]}",
        "slice={[1:2], [0:2], [0:4], [0:1], [0:2048]}",
        1,
    )
    assert duplicate_physical_leaf != hlo
    assert not MODULE._validate_optimized_hlo(duplicate_physical_leaf)["passed"]
    wrong_middle_pairing = hlo.replace(
        "%association_y_leaf.1.0, %association_y_leaf.1.3",
        "%association_y_leaf.1.0, %association_y_leaf.1.1",
        1,
    ).replace(
        "%association_y_leaf.1.1, %association_y_leaf.1.2",
        "%association_y_leaf.1.3, %association_y_leaf.1.2",
        1,
    )
    assert wrong_middle_pairing != hlo
    assert not MODULE._validate_optimized_hlo(wrong_middle_pairing)["passed"]
    wrong_z_pairing = hlo.replace(
        "%association_z_leaf.1.0, %association_z_leaf.1.3",
        "%association_z_leaf.1.0, %association_z_leaf.1.1",
        1,
    ).replace(
        "%association_z_leaf.1.1, %association_z_leaf.1.2",
        "%association_z_leaf.1.3, %association_z_leaf.1.2",
        1,
    )
    assert wrong_z_pairing != hlo
    assert not MODULE._validate_optimized_hlo(wrong_z_pairing)["passed"]
    wrong_y_order = hlo.replace(
        "concatenate(%association_y.0.root, %association_y.1.root, "
        "%association_y.2.root)",
        "concatenate(%association_y.1.root, %association_y.0.root, "
        "%association_y.2.root)",
        1,
    )
    assert wrong_y_order != hlo
    assert not MODULE._validate_optimized_hlo(wrong_y_order)["passed"]
    wrong_z_order = hlo.replace(
        "concatenate(%association_z.0.root, %association_z.1.root",
        "concatenate(%association_z.1.root, %association_z.0.root",
        1,
    )
    assert wrong_z_order != hlo
    assert not MODULE._validate_optimized_hlo(wrong_z_order)["passed"]


def test_dense_convolution_stablehlo_contract_rejects_dead_rows_or_extra_op() -> None:
    stablehlo = _exact_stablehlo()
    dead_rows = stablehlo.replace("tensor<1x6144xbf16>", "tensor<32x6144xbf16>", 1)
    assert not MODULE._validate_stablehlo(dead_rows)["passed"]
    first_convolution = next(
        line for line in stablehlo.splitlines() if "stablehlo.convolution" in line
    )
    extra = stablehlo.replace(
        "sdy.return %886, %902",
        first_convolution.replace("%29 =", "%extra =")
        + "\n      sdy.return %886, %902",
        1,
    )
    assert not MODULE._validate_stablehlo(extra)["passed"]


def test_dense_convolution_stablehlo_contract_rejects_arithmetic_mutations() -> None:
    stablehlo = _exact_stablehlo()
    mutations = (
        stablehlo.replace(
            "stablehlo.multiply %39, %32", "stablehlo.multiply %39, %31", 1
        ),
        stablehlo.replace(
            "[0:384, 0:6144]", "[384:768, 0:6144]", 1
        ),
        stablehlo.replace(
            "dim_numbers = [b, f]x[i, o]->[b, f]",
            "dim_numbers = [f, b]x[i, o]->[f, b]",
            1,
        ),
        stablehlo.replace(
            "stablehlo.convolution(%40, %48)",
            "stablehlo.convolution(%84, %48)",
            1,
        ),
        stablehlo.replace(
            "stablehlo.concatenate %359, %360",
            "stablehlo.concatenate %360, %359",
            1,
        ),
        stablehlo.replace(
            '"stablehlo.all_gather"(%368)',
            '"stablehlo.all_gather"(%367)',
            1,
        ),
        stablehlo.replace(
            "stablehlo.add %474, %476",
            "stablehlo.add %474, %480",
            1,
        ),
        stablehlo.replace(
            "sdy.return %886, %902", "sdy.return %886, %886", 1
        ),
        stablehlo.replace(
            "manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8)",
            "manual_computation(%arg0, %arg1, %arg4, %arg5, %arg2, %arg3, %arg6, %arg7, %arg8)",
            1,
        ),
    )
    assert all(value != stablehlo for value in mutations)
    for mutated in mutations:
        result = MODULE._validate_stablehlo(mutated)
        assert not result["passed"], result


def test_dense_convolution_wrapper_pins_db538_and_protected_publication() -> None:
    text = WRAPPER.read_text()
    for marker in (
        "GLM_GREENFIELD_DENSE_CONVOLUTION_PROBE",
        "probe_layer0_dense_convolution.py",
        "DB538_RUN_ID=538",
        "DB538_CODE_HASH=e2a3a74a3b2ef1fa8f3b9cb1c5d7ec65f833eafc",
        "ACCEPTED_M32_HLO_SHA=25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c",
        "GLM_GREENFIELD_DENSE_CONVOLUTION_COMPILE_ROWS",
        "GLM_GREENFIELD_DENSE_LAYER1_ONLY",
        "GLM_GREENFIELD_DENSE_FINAL_LAYOUT",
        "01d019dfa316f9fc75cc64d293c3678c41d5425cbbd73b7e44d65f03fa6a4fba",
        'strict_census pre',
        'strict_census post',
        'rollback_provisional_db',
        'remote_prefix_listing=$(gcloud storage ls "$REMOTE_PREFIX/**"',
        'gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS"',
        'terminal_success_done=1',
    ):
        assert marker in text
    assert text.index("strict_census post") < text.index(
        'PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python -'
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "m32",
        "cross_layer",
        "cross_layer_mode",
        "final_layout",
        "final_layout_record",
        "final_layout_hlo",
        "final_layout_lineage",
        "classification",
        "comparison",
        "source",
        "hlo",
        "activation_graph",
        "association_graph",
        "rms_graph",
        "rms_reduction_geometry",
        "nonexact",
        "nonexact_negative_count",
        "nonexact_bad_index",
        "nonexact_negative_error",
        "nonexact_nan_mean",
    ),
)
def test_dense_convolution_wrapper_records_authenticated_diagnostic(
    tmp_path: Path,
    mutation: str,
) -> None:
    wrapper = WRAPPER.read_text()
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper, re.DOTALL)
    record_program = next(
        value for value in programs if "runner_valid" in value
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pin = "b" * 40
    tag = "dense_convolution_record_test"
    runner = {
        "artifact_kind": "glm52_layer0_dense_convolution_probe",
        "classification": "accepted_dense_convolution_exact",
        "code_hash": pin,
        "compile_rows": 1,
        "diagnostic_dead_rows": 0,
        "exact": True,
        "exact_arms": ["accepted_dense_convolution"],
        "final_dense_layout": False,
        "final_layout_records": {},
        "result_mode": "dense_and_layer1",
        "hlo": {
            "optimized_contract": {
                "async_collectives": [],
                "collective_count": 1,
                "compile_rows": 1,
                "convolution_count": 16,
                "down_convolution_count": 8,
                "final_dense_layout": False,
                "gate_up_convolution_count": 8,
                "live_rows": 1,
                "lineage": {
                    "activation_contract": {
                        str(rank): {
                            "add": [f"%add.{rank}"],
                            "divide": [f"%divide.{rank}"],
                            "exponential": [f"%exp.{rank}"],
                            "multiply": [f"%multiply.{rank}.0", f"%multiply.{rank}.1"],
                            "negate": [f"%negate.{rank}"],
                            "exact_operand_graph": True,
                        }
                        for rank in range(8)
                    },
                    "association_add_shapes": {"x": 1, "y": 9, "z": 72},
                    "association_edge_graph": {
                        "x": {
                            "component_count": 1,
                            "exact": True,
                            "leaf_rows": [0, 1],
                        },
                        "y": {
                            "component_count": 3,
                            "component_ids": list(range(3)),
                            "exact": True,
                            "leaf_pairings": {
                                str(component): (
                                    [[0, 1], [2, 3]]
                                    if component % 2 == 0
                                    else [[0, 3], [1, 2]]
                                )
                                for component in range(3)
                            },
                            "ordered_components": [
                                [component] for component in range(3)
                            ],
                        },
                        "z": {
                            "component_count": 24,
                            "component_ids": list(range(24)),
                            "exact": True,
                            "leaf_pairings": {
                                str(component): (
                                    [[0, 1], [2, 3]]
                                    if component % 2 == 0
                                    else [[0, 3], [1, 2]]
                                )
                                for component in range(24)
                            },
                            "ordered_components": [
                                [component] for component in range(24)
                            ],
                        },
                    },
                    "collective_convolution_sources": [
                        f"%down.{index}" for index in range(8)
                    ],
                    "down_virtual_ranks": list(range(8)),
                    "gate_up_virtual_ranks": list(range(8)),
                    "layer1_only_parameter_shapes": [
                        "bf16[1,6144]",
                        "bf16[6144]",
                    ],
                    "ordered_stack_sources": [[rank] for rank in range(8)],
                    "rmsnorm_contract": {
                        "add_count": 2,
                        "direct_exact_operand_graph": True,
                        "exact_cross_fusion_reduction_lineage": True,
                        "exact_m32_reduction_geometry": True,
                        "exact_reduction_operand_graph": True,
                        "exact_result_binding": True,
                        "exact_weighted_operand_graph": True,
                        "divide_count": 1,
                        "multiply_or_square_count": 3,
                        "reduce_count": 1,
                        "rsqrt_count": 1,
                        "semantic_counts": {
                            "add": 2,
                            "div": 1,
                            "mul": 2,
                            "reduce_sum": 1,
                            "rsqrt": 1,
                            "square": 1,
                        },
                    },
                },
                "num_partitions": 4,
                "num_replicas": 1,
                "passed": True,
                "result_mode": "dense_and_layer1",
                "unexpected_convolutions": [],
                "violations": [],
            },
            "optimized_sha256": "d" * 64,
            "stablehlo_contract": {
                "collective_counts": {
                    "all_gather": 1,
                    "all_reduce": 0,
                    "all_to_all": 0,
                    "collective_broadcast": 0,
                    "collective_permute": 0,
                    "reduce_scatter": 0,
                },
                "convolution_count": 16,
                "compile_rows": 1,
                "down_convolution_count": 8,
                "final_dense_layout": False,
                "gate_up_convolution_count": 8,
                "live_rows": 1,
                "matched_virtual_shards": list(range(8)),
                "passed": True,
                "result_mode": "dense_and_layer1",
                "violations": [],
            },
            "stablehlo_sha256": "e" * 64,
        },
        "layer1_comparison": {
            "elementwise_exact": True,
            "expected_sha256": (
                "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
            ),
            "first_mismatch_index": None,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "mismatch_count": 0,
            "observed_sha256": (
                "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
            ),
            "shape": [6144],
        },
        "performance_claim": False,
        "live_rows": 1,
        "position": 8155,
        "source": {
            "checkpoint_manifest_sha256": (
                "de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134"
            ),
            "db538_runner_sha256": (
                "303dd91eed1d75e0cd443645c5f1ef745f259c51596d0651646bb141db8f16f8"
            ),
            "db538_tensor_sha256": (
                "e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e"
            ),
            "db538_summary_sha256": (
                "90090ba9999812082727ed56b734163f07e3c27b4eb88fa049b9c2504516e782"
            ),
            "db538_success_sha256": (
                "7744356f63b67cc813901499d0828c029ea5a9c985a5dac65525700457f79985"
            ),
            "post_attention_residual_sha256": (
                "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e"
            ),
        },
        "status": "SUCCESS",
        "weight_records": [
            {
                "destination_filename": (
                    "base_decoder_runtime_feature/stage_00/"
                    f"device_slot_{slot:02d}.safetensors"
                ),
                "device_slot": slot,
                "evidence_file_sha256": "f" * 64,
                "header_sha256": "a" * 64,
                "tensors": [
                    {"name": name, "sha256": "b" * 64}
                    for name in (
                        "attention.slot_00.kv_b.weight_bits",
                        "attention.slot_00.kv_b.scale_inv",
                        "attention.slot_00.o.weight_bits",
                        "attention.slot_00.o.scale_inv",
                        "attention.slot_00.post_norm",
                        "dense.slot_00.gate.weight_bits",
                        "dense.slot_00.gate.scale_inv",
                        "dense.slot_00.up.weight_bits",
                        "dense.slot_00.up.scale_inv",
                        "dense.slot_00.down.weight_bits",
                        "dense.slot_00.down.scale_inv",
                        "attention.slot_01.input_norm",
                    )
                ],
            }
            for slot in range(4)
        ],
    }
    compile_rows = 1
    layer1_only = False
    final_layout = False
    if mutation in (
        "m32",
        "cross_layer",
        "cross_layer_mode",
        "final_layout",
            "final_layout_record",
            "final_layout_hlo",
            "final_layout_lineage",
        ):
        compile_rows = 32
        layer1_only = mutation.startswith(("cross_layer", "final_layout"))
        final_layout = mutation.startswith("final_layout")
        arm = (
            "accepted_m32_dense_final_layout_cross_layer"
            if final_layout
            else "accepted_m32_dense_cross_layer"
            if layer1_only
            else "accepted_m32_dense_convolution"
        )
        result_mode = "layer1_only" if layer1_only else "dense_and_layer1"
        runner.update(
            {
                "classification": f"{arm}_exact",
                "compile_rows": 32,
                "diagnostic_dead_rows": 31,
                "exact_arms": [arm],
                "final_dense_layout": final_layout,
                "result_mode": result_mode,
            }
        )
        runner["hlo"]["optimized_contract"]["compile_rows"] = 32
        runner["hlo"]["optimized_contract"][
            "final_dense_layout"
        ] = final_layout
        runner["hlo"]["optimized_contract"]["result_mode"] = result_mode
        runner["hlo"]["optimized_contract"]["lineage"][
            "m32_live_row_slices"
        ] = ["%m32_live"]
        if layer1_only:
            runner["hlo"]["optimized_contract"]["lineage"][
                "result_parameter_sources"
            ] = [
                [
                    f"%parameter.{index}"
                    for index in range(7 if final_layout else 9)
                ]
            ]
        runner["hlo"]["stablehlo_contract"]["compile_rows"] = 32
        runner["hlo"]["stablehlo_contract"][
            "final_dense_layout"
        ] = final_layout
        runner["hlo"]["stablehlo_contract"]["result_mode"] = result_mode
        runner["source"].update(
            {
                "accepted_m32_hlo_raw_sha256": (
                    "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
                ),
                "accepted_m32_hlo_sha256": (
                    "25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c"
                ),
                "accepted_m32_summary_sha256": (
                    "409c845c2c9d67a1d6de36f0cccd25d2982850ee86c35645b0839fc78a1507a3"
                ),
                "accepted_m32_success_sha256": (
                    "6ef516dc42e046a996aa1fe542a4b11af5c2a14450e1aba7bfac98ddbf257278"
                ),
            }
        )
        if final_layout:
            runner["final_layout_records"] = json.loads(
                json.dumps(MODULE._FINAL_DENSE_LAYOUT_RECORDS)
            )
            runner["hlo"]["optimized_contract"].update(
                {
                    "exact_accepted_weight_layout": True,
                    "exact_packed_weight_lineage": True,
                    "accepted_weight_layouts": {
                        label: [
                            {
                                "accepted": True,
                                "accepted_layout": True,
                                "convolution": f"%{label}.{index}",
                                "exact_packed_dequant": True,
                                "parameter_sources": [
                                    f"%{label}_bits",
                                    f"%{label}_scale",
                                ],
                                "virtual_rank": index,
                            }
                            for index in range(8)
                        ]
                        for label in ("gate_up", "down")
                    },
                }
            )
            if mutation == "final_layout_record":
                next(iter(runner["final_layout_records"].values()))[
                    "sha256"
                ] = "0" * 64
            elif mutation == "final_layout_hlo":
                runner["hlo"]["optimized_contract"][
                    "exact_accepted_weight_layout"
                ] = False
            elif mutation == "final_layout_lineage":
                runner["hlo"]["optimized_contract"][
                    "exact_packed_weight_lineage"
                ] = False
        if mutation == "cross_layer_mode":
            runner["result_mode"] = "dense_and_layer1"
    elif mutation == "classification":
        runner["classification"] = "accepted_dense_convolution_nonexact"
    elif mutation == "comparison":
        runner["layer1_comparison"]["mismatch_count"] = 1
    elif mutation == "source":
        runner["source"]["db538_tensor_sha256"] = "0" * 64
    elif mutation == "hlo":
        runner["hlo"]["stablehlo_contract"]["convolution_count"] = 15
    elif mutation == "activation_graph":
        runner["hlo"]["optimized_contract"]["lineage"][
            "activation_contract"
        ]["0"]["exact_operand_graph"] = False
    elif mutation == "association_graph":
        runner["hlo"]["optimized_contract"]["lineage"][
            "association_edge_graph"
        ]["z"]["exact"] = False
    elif mutation == "rms_graph":
        runner["hlo"]["optimized_contract"]["lineage"][
            "rmsnorm_contract"
        ]["direct_exact_operand_graph"] = False
    elif mutation == "rms_reduction_geometry":
        runner["hlo"]["optimized_contract"]["lineage"][
            "rmsnorm_contract"
        ]["exact_m32_reduction_geometry"] = False
    elif mutation.startswith("nonexact"):
        runner["classification"] = "accepted_dense_convolution_nonexact"
        runner["exact"] = False
        runner["exact_arms"] = []
        runner["layer1_comparison"].update(
            {
                "elementwise_exact": False,
                "first_mismatch_index": 1,
                "max_abs_error": 0.0078125,
                "mean_abs_error": 3.4686963772401214e-05,
                "mismatch_count": 1073,
                "observed_sha256": "2" * 64,
            }
        )
        if mutation == "nonexact_negative_count":
            runner["layer1_comparison"]["mismatch_count"] = -1
        elif mutation == "nonexact_bad_index":
            runner["layer1_comparison"]["first_mismatch_index"] = "rogue"
        elif mutation == "nonexact_negative_error":
            runner["layer1_comparison"]["max_abs_error"] = -1.0
        elif mutation == "nonexact_nan_mean":
            runner["layer1_comparison"]["mean_abs_error"] = float("nan")
    (run_dir / "runner.json").write_text(json.dumps(runner))
    database = tmp_path / "results.db"
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(run_dir),
            pin,
            str(database),
            str(REPO),
            "17",
            tag,
            "1",
            "303dd91eed1d75e0cd443645c5f1ef745f259c51596d0651646bb141db8f16f8",
            "e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e",
            "90090ba9999812082727ed56b734163f07e3c27b4eb88fa049b9c2504516e782",
            "7744356f63b67cc813901499d0828c029ea5a9c985a5dac65525700457f79985",
            "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e",
            "de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134",
            str(compile_rows),
            str(int(layer1_only)),
            str(int(final_layout)),
            "25041bfbcf319b6c6fc4c5888cb22548b246cccba784791796fe9e8f57199e4c",
            "409c845c2c9d67a1d6de36f0cccd25d2982850ee86c35645b0839fc78a1507a3",
            "6ef516dc42e046a996aa1fe542a4b11af5c2a14450e1aba7bfac98ddbf257278",
            "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775",
        ],
        input=record_program,
        text=True,
        capture_output=True,
        check=False,
    )
    if mutation not in (
        "none",
        "m32",
        "cross_layer",
        "final_layout",
        "nonexact",
    ):
        assert completed.returncode != 0
        return
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((run_dir / "summary.json").read_text())
    expected_exact = mutation in ("none", "m32", "cross_layer", "final_layout")
    expected_arm = (
        "accepted_m32_dense_final_layout_cross_layer"
        if mutation == "final_layout"
        else "accepted_m32_dense_cross_layer"
        if mutation == "cross_layer"
        else (
            "accepted_m32_dense_convolution"
            if mutation == "m32"
            else "accepted_dense_convolution"
        )
    )
    assert summary["exact_arms"] == (
        [expected_arm] if expected_exact else []
    )
    connection = sqlite3.connect(database)
    run = connection.execute(
        "SELECT model, model_revision, env_json, note FROM runs"
    ).fetchone()
    item = connection.execute(
        "SELECT benchmark, correct, score FROM items"
    ).fetchone()
    metric = connection.execute(
        "SELECT benchmark, metric, value FROM summary"
    ).fetchone()
    connection.close()
    assert run is not None
    assert run[0:2] == (
        (
            "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-final-layout-cross-layer"
            if mutation == "final_layout"
            else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-cross-layer"
            if mutation == "cross_layer"
            else (
                "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-m32-convolution"
                if mutation == "m32"
                else "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-convolution"
            )
        ),
        (
            "native-jax-accepted-dense-final-layout-v1"
            if mutation == "final_layout"
            else "native-jax-db542-dense-cross-layer-v1"
            if mutation == "cross_layer"
            else (
                "native-jax-db532-dense-m32-discriminator-v1"
                if mutation == "m32"
                else "native-jax-db538-dense-convolution-v1"
            )
        ),
    )
    environment = json.loads(run[2])
    assert environment["greenfield_run_tag"] == tag
    assert environment["compile_rows"] == compile_rows
    assert environment["final_dense_layout"] is final_layout
    if final_layout:
        assert environment["final_layout_records_sha256"] == (
            "01d019dfa316f9fc75cc64d293c3678c41d5425cbbd73b7e44d65f03fa6a4fba"
        )
        assert summary["final_layout_records"] == MODULE._FINAL_DENSE_LAYOUT_RECORDS
        assert summary["final_layout_records_sha256"] == (
            "01d019dfa316f9fc75cc64d293c3678c41d5425cbbd73b7e44d65f03fa6a4fba"
        )
    assert environment["result_mode"] == (
        "layer1_only"
        if mutation in ("cross_layer", "final_layout")
        else "dense_and_layer1"
    )
    assert run[3] == (
        "Protected layer-0 accepted final-layout dense cross-layer discriminator; no performance claim."
        if mutation == "final_layout"
        else "Protected layer-0 accepted-M32 dense cross-layer fusion discriminator; no performance claim."
        if mutation == "cross_layer"
        else (
            "Protected layer-0 accepted-M32 dense arithmetic discriminator; no performance claim."
            if mutation == "m32"
            else "Protected layer-0 dense convolution discriminator; no performance claim."
        )
    )
    assert item == (
        (
            "greenfield_layer0_dense_final_layout_cross_layer"
            if mutation == "final_layout"
            else "greenfield_layer0_dense_m32_cross_layer"
            if mutation == "cross_layer"
            else (
                "greenfield_layer0_dense_m32_convolution"
                if mutation == "m32"
                else "greenfield_layer0_dense_convolution"
            )
        ),
        int(expected_exact),
        float(expected_exact),
    )
    assert metric == (
        (
            "greenfield_layer0_dense_final_layout_cross_layer"
            if mutation == "final_layout"
            else "greenfield_layer0_dense_m32_cross_layer"
            if mutation == "cross_layer"
            else (
                "greenfield_layer0_dense_m32_convolution"
                if mutation == "m32"
                else "greenfield_layer0_dense_convolution"
            )
        ),
        "probe_contract_valid",
        1.0,
    )
