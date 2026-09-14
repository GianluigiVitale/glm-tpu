from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_dense_convolution.py"
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
REAL_FINAL_LAYOUT_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_FINAL_LAYOUT_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_final_layout_cross_layer_"
        "20260813T082823269638806Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_FINAL_LAYOUT_OPTIMIZED_HLO_SHA256 = (
    "caa2569ad56c953ae7cdbe8bf1583bcafd13bb8188435bca34ec818650a2e39e"
)
REAL_DENSE_ENVELOPE_ROOT = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_ENVELOPE_ROOT",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T094239645854705Z/hlo",
    )
)
REAL_DENSE_ENVELOPE_OPTIMIZED_HLO = (
    REAL_DENSE_ENVELOPE_ROOT / "dense_convolution.optimized_hlo.txt"
)
REAL_DENSE_ENVELOPE_OPTIMIZED_HLO_SHA256 = (
    "dbe7f3dbd82ebe832bcbd0c0d06cb85663fb1f63be202c893f07a1d98c288213"
)
REAL_DENSE_ENVELOPE_STABLEHLO = (
    REAL_DENSE_ENVELOPE_ROOT / "dense_convolution.stablehlo.mlir"
)
REAL_DENSE_ENVELOPE_STABLEHLO_SHA256 = (
    "74e1fe97c58cbf003dd0d4937bfa37ffc71e2ec7ef5a7317f2f6c509fcfa9fd3"
)
REAL_ACCEPTED_SCALE_ENVELOPE_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_ACCEPTED_SCALE_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T111019310055824Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_ACCEPTED_SCALE_ENVELOPE_OPTIMIZED_HLO_SHA256 = (
    "68b7ca3dba1d53d7172c4bc82cde9beb7abd464e3b748995ef261abe9d8e55ab"
)
REAL_SERIALIZED_SCALE_ENVELOPE_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_SERIALIZED_SCALE_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T113846748941458Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_SERIALIZED_SCALE_ENVELOPE_OPTIMIZED_HLO_SHA256 = (
    "0cff45d9ab9fb1428ea2c6483b33be8dc5416769db42800f8c3099ed1c16453b"
)
REAL_ACCEPTED_GEOMETRY_ENVELOPE_OPTIMIZED_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_ACCEPTED_GEOMETRY_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T115238978656998Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_ACCEPTED_GEOMETRY_ENVELOPE_OPTIMIZED_HLO_SHA256 = (
    "ac57c042ae591d99cebc982f37f06b3c335f3f3d1453fd9e2f7269a1c3b9094c"
)
REAL_DB548_OPTIMIZED_HLO = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_"
    "20260813T120703034434907Z/hlo/dense_convolution.optimized_hlo.txt"
)
REAL_DB548_OPTIMIZED_HLO_SHA256 = (
    "5f4dd83793da67be6a8c580949920e93f8c64fe8205816738e7d04890640a877"
)
REAL_PARTIAL_CAPTURE_OPTIMIZED_HLO = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_partial_capture_"
    "20260813T200306361654899Z/hlo/dense_convolution.optimized_hlo.txt"
)
REAL_PARTIAL_CAPTURE_OPTIMIZED_HLO_SHA256 = (
    "8426fbf24bece38cead562752ef87be07e7a2ab31bf94560e3065fb18277b290"
)
REAL_DB548_TENSOR = REAL_DB548_OPTIMIZED_HLO.parents[1] / (
    "dense_envelope_cross_layer.npz"
)
REAL_LAYER1_NORM_SLOT = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/PP8_LP4/"
    "greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z/"
    "base_decoder_runtime_feature/stage_00/device_slot_00.safetensors"
)
REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_split_rms_"
        "20260813T124723663516442Z/hlo/dense_convolution.optimized_hlo.txt",
    )
)
REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO_SHA256 = (
    "dbe6f797008722ae8b2b4a53ddc83d1ebcecf858c44682b9654cd7b207397103"
)
REAL_REDUCTION_BARRIER_SPLIT_RMS_ROOT = Path(
    os.environ.get(
        "GLM_DENSE_CONVOLUTION_REAL_REDUCTION_BARRIER_SPLIT_RMS_ROOT",
        "/home/gianl/glm-run/greenfield_layer0_dense_envelope_split_rms_"
        "20260813T130459893823080Z/hlo",
    )
)
REAL_REDUCTION_BARRIER_SPLIT_RMS_HLO = (
    REAL_REDUCTION_BARRIER_SPLIT_RMS_ROOT
    / "dense_convolution.optimized_hlo.txt"
)
REAL_REDUCTION_BARRIER_SPLIT_RMS_HLO_SHA256 = (
    "ad97e7f644fdd79fe726046b36d0cec0b6ca700402ff0b85f1ed814af27f7327"
)
REAL_REDUCTION_BARRIER_SPLIT_RMS_STABLEHLO = (
    REAL_REDUCTION_BARRIER_SPLIT_RMS_ROOT
    / "dense_convolution.stablehlo.mlir"
)
REAL_REDUCTION_BARRIER_SPLIT_RMS_STABLEHLO_SHA256 = (
    "2de54df491300f94bb2d600243439fe1c43c5614a9434588b3a896f19aab7f67"
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
        "  %packed_gate_bits = f8e4m3fn[1,8,6144,768]{3,2,1,0} parameter(2)\n"
        "  %packed_gate_scale = f32[1,8,48,768]{3,2,1,0} parameter(3)\n"
        "  %packed_down_bits = f8e4m3fn[1,8,384,6144]{3,2,1,0} parameter(4)\n"
        "  %packed_down_scale = f32[1,8,3,6144]{3,2,1,0} parameter(5)",
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
                f"  %gate_bits_slice.{rank} = f8e4m3fn[1,1,6144,768]{{3,2,1,0}} "
                f"slice(%packed_gate_bits), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:6144], [0:768]}}",
                f"  %gate_bits.{rank} = f8e4m3fn[6144,768]{{1,0}} "
                f"bitcast(%gate_bits_slice.{rank})",
                f"  %gate_f32.{rank} = f32[6144,768] convert(%gate_bits.{rank}), "
                f"{scope}convert_element_type\"}}",
                f"  %gate_scale_slice.{rank} = f32[1,1,48,768]{{3,2,1,0}} "
                f"slice(%packed_gate_scale), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:48], [0:768]}}",
                f"  %gate_scale_seed.{rank} = f32[48,768]{{1,0}} "
                f"reshape(%gate_scale_slice.{rank})",
                f"  %gate_scale_inner.{rank} = f32[48,128,768]{{2,1,0}} "
                f"broadcast(%gate_scale_seed.{rank}), dimensions={{0,2}}",
                f"  %gate_scale_wide.{rank} = f32[6144,768]{{1,0}} "
                f"reshape(%gate_scale_inner.{rank})",
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
                f"  %down_bits_slice.{rank} = f8e4m3fn[1,1,384,6144]{{3,2,1,0}} "
                f"slice(%packed_down_bits), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:384], [0:6144]}}",
                f"  %down_bits.{rank} = f8e4m3fn[384,6144]{{1,0}} "
                f"bitcast(%down_bits_slice.{rank})",
                f"  %down_f32.{rank} = f32[384,6144] convert(%down_bits.{rank}), "
                f"{scope}convert_element_type\"}}",
                f"  %down_scale_slice.{rank} = f32[1,1,3,6144]{{3,2,1,0}} "
                f"slice(%packed_down_scale), slice={{[0:1], [{rank}:{rank + 1}], "
                "[0:3], [0:6144]}}",
                f"  %down_scale_seed.{rank} = f32[3,6144]{{1,0}} "
                f"reshape(%down_scale_slice.{rank})",
                f"  %down_scale_inner.{rank} = f32[3,128,6144]{{2,1,0}} "
                f"broadcast(%down_scale_seed.{rank}), dimensions={{0,2}}",
                f"  %down_scale_wide.{rank} = f32[384,6144]{{1,0}} "
                f"reshape(%down_scale_inner.{rank})",
                f"  %down_scaled.{rank} = f32[384,6144] multiply("
                f"%down_f32.{rank}, %down_scale_wide.{rank}), {scope}mul\"}}",
                f"  %down_weight.{rank} = bf16[384,6144]{{1,0}} "
                f"convert(%down_scaled.{rank}), {scope}convert_element_type\"}}",
            )
        )
        hlo, count = down_parameter.subn(down_dequant, hlo, count=1)
        assert count == 1
    return hlo


def _synthetic_dense_envelope_optimized_hlo() -> str:
    hlo = _synthetic_final_layout_optimized_hlo()
    prefix = (
        "  %normalized = bf16[1,6144] parameter(0)\n"
        "  %m32_zero = bf16[] constant(0)\n"
        "  %m32_padded = bf16[32,6144] pad(%normalized, %m32_zero), "
        "padding=0_31x0_0\n"
        "  %residual = bf16[1,6144] parameter(1)\n"
        "  %norm = bf16[6144] parameter(6)\n"
        "  %packed_gate_bits = f8e4m3fn[1,8,6144,768]{3,2,1,0} parameter(2)\n"
        "  %packed_gate_scale = f32[1,8,48,768]{3,2,1,0} parameter(3)\n"
        "  %packed_down_bits = f8e4m3fn[1,8,384,6144]{3,2,1,0} parameter(4)\n"
        "  %packed_down_scale = f32[1,8,3,6144]{3,2,1,0} parameter(5)"
    )
    entry = "\n".join(
        (
            "  %attention = bf16[1,6144] parameter(0)",
            "  %combined_residual = bf16[1,6144] parameter(1)",
            "  %post_norm = bf16[6144] parameter(2)",
            "  %norm = bf16[6144] parameter(7)",
            "  %packed_gate_bits = f8e4m3fn[1,8,6144,768]{3,2,1,0} parameter(3)",
            "  %packed_gate_scale = f32[1,8,48,768]{3,2,1,0} parameter(4)",
            "  %packed_down_bits = f8e4m3fn[1,8,384,6144]{3,2,1,0} parameter(5)",
            "  %packed_down_scale = f32[1,8,3,6144]{3,2,1,0} parameter(6)",
        )
    )
    assert prefix in hlo
    hlo = hlo.replace(prefix, entry, 1)
    hlo = hlo.replace(
        "  %residual_m32 = bf16[32,6144] pad(%residual, %rms_zero), "
        "padding=0_31x0_0\n",
        "",
        1,
    ).replace(
        "  %residual_f32 = f32[32,6144] convert(%residual_m32)",
        "  %residual_f32 = f32[32,6144] convert(%predense_carried)",
        1,
    )
    gate_computations = []
    scope = (
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_predense_rmsnorm/'
    )
    for rank in range(8):
        gate_scope = (
            f"jit(probe)/greenfield_dense_convolution_virtual_rank_{rank:02d}/"
            "gate_up"
        )
        direct_gate = (
            f"  %gate_up.{rank} = f32[32,768] convolution("
            f"%m32_padded, %gate_up_weight.{rank}), dim_labels=bf_io->bf, "
            f'metadata={{op_name="{gate_scope}"}}'
        )
        gate_arguments = (
            f"%attention, %combined_residual, %post_norm, "
            f"%gate_up_weight.{rank}"
        )
        if rank == 0:
            fused_gate = "\n".join(
                (
                    "  %gate_pair.0 = (f32[32,768], bf16[32,6144]) "
                    f"fusion({gate_arguments}), kind=kOutput, "
                    "calls=%predense_gate_0",
                    "  %gate_up.0 = f32[32,768] get-tuple-element("
                    "%gate_pair.0), index=0",
                    "  %predense_carried = bf16[32,6144] "
                    "get-tuple-element(%gate_pair.0), index=1",
                )
            )
        else:
            fused_gate = (
                f"  %gate_up.{rank} = f32[32,768] fusion("
                f"{gate_arguments}), kind=kOutput, "
                f"calls=%predense_gate_{rank}"
            )
        assert direct_gate in hlo
        hlo = hlo.replace(direct_gate, fused_gate, 1)
        body = [
            f"%predense_gate_{rank} {{",
            "  %gate_attention = bf16[1,6144] parameter(0)",
            "  %gate_residual = bf16[1,6144] parameter(1)",
            "  %gate_norm = bf16[6144] parameter(2)",
            "  %gate_rhs = bf16[6144,768]{1,0} parameter(3)",
            "  %gate_zero = bf16[] constant(0)",
            "  %gate_f32_zero = f32[] constant(0)",
            "  %gate_attention_m32 = bf16[32,6144] pad("
            "%gate_attention, %gate_zero), padding=0_31x0_0",
            "  %gate_residual_m32 = bf16[32,6144] pad("
            "%gate_residual, %gate_zero), padding=0_31x0_0",
            "  %gate_attention_f32 = f32[32,6144] "
            "convert(%gate_attention_m32)",
            "  %gate_residual_f32 = f32[32,6144] "
            "convert(%gate_residual_m32)",
            "  %gate_sum = f32[32,6144] add("
            "%gate_attention_f32, %gate_residual_f32), " + scope + 'add"}',
            "  %gate_carried = bf16[32,6144] convert(%gate_sum)",
            "  %gate_square = f32[32,6144] multiply("
            "%gate_sum, %gate_sum), " + scope + 'square"}',
            "  %gate_sum_squares = f32[32] reduce("
            "%gate_square, %gate_f32_zero), dimensions={1}, "
            "to_apply=%sum_reducer, " + scope + 'reduce_sum"}',
            "  %gate_sum_row = f32[32,1] reshape(%gate_sum_squares)",
            "  %gate_width = f32[32,1] constant(6144)",
            "  %gate_mean = f32[32,1] divide("
            "%gate_sum_row, %gate_width), " + scope + 'div"}',
            "  %gate_epsilon = f32[32,1] constant(0.00001)",
            "  %gate_variance = f32[32,1] add("
            "%gate_mean, %gate_epsilon), " + scope + 'add"}',
            "  %gate_inverse = f32[32,1] rsqrt(%gate_variance), "
            + scope + 'rsqrt"}',
            "  %gate_inverse_wide = f32[32,6144] broadcast("
            "%gate_inverse), dimensions={0,1}",
            "  %gate_normalized = f32[32,6144] multiply("
            "%gate_sum, %gate_inverse_wide), " + scope + 'mul"}',
            "  %gate_rounded = bf16[32,6144] convert(%gate_normalized)",
            "  %gate_norm_seed = bf16[1,6144] broadcast("
            "%gate_norm), dimensions={1}",
            "  %gate_norm_wide = bf16[32,6144] broadcast("
            "%gate_norm_seed), dimensions={0,1}",
            "  %gate_weighted = bf16[32,6144] multiply("
            "%gate_rounded, %gate_norm_wide), " + scope + 'mul"}',
            f"  %gate_value.{rank} = f32[32,768] convolution("
            "%gate_weighted, %gate_rhs), dim_labels=bf_io->bf, "
            f'metadata={{op_name="{gate_scope}"}}',
        ]
        if rank == 0:
            body.append(
                "  ROOT %gate_root.0 = (f32[32,768], bf16[32,6144]) "
                "tuple(%gate_value.0, %gate_carried)"
            )
        else:
            body.append(f"  ROOT %gate_root.{rank} = f32[32,768] copy(%gate_value.{rank})")
        body.append("}")
        gate_computations.append("\n".join(body))
    hlo = hlo.replace(
        "ENTRY main {",
        "\n".join(gate_computations) + "\n\nENTRY main {",
        1,
    )
    return hlo


def _with_exact_scheduled_dense_tiling(hlo: str) -> str:
    gate = {
        "convolution_algorithm_config": {"emitter": "EmitAllBatchInSublanes"},
        "megacore_config": {
            "megacore_allreduce_bytes": "98304",
            "megacore_split_dim": "2",
        },
        "window_config": {
            "cost_model_type": "COST_MODEL_TYPE_CLASSIC",
            "input_window_bounds": ["4", "24"],
            "is_mask": False,
            "iteration_bounds": ["1", "1", "2"],
            "kernel_window_bounds": ["384", "6"],
            "output_window_bounds": ["4", "6"],
            "pad_input_on_minor_dim": "0",
            "pad_output_on_minor_dim": "0",
        },
    }
    down = {
        "convolution_algorithm_config": {"emitter": "EmitAllBatchInSublanes"},
        "megacore_config": {
            "megacore_allreduce_bytes": None,
            "megacore_split_dim": "0",
        },
        "window_config": {
            "cost_model_type": "COST_MODEL_TYPE_CLASSIC",
            "input_window_bounds": ["4", "3"],
            "is_mask": False,
            "iteration_bounds": ["8", "1", "1"],
            "kernel_window_bounds": ["48", "6"],
            "output_window_bounds": ["4", "6"],
            "pad_input_on_minor_dim": "0",
            "pad_output_on_minor_dim": "0",
        },
    }
    lines = []
    for line in hlo.splitlines():
        if " convolution(" in line and " = f32[32,768]" in line:
            line += ", backend_config=" + json.dumps(gate, separators=(",", ":"))
        elif " convolution(" in line and " = f32[32,6144]" in line:
            line += ", backend_config=" + json.dumps(down, separators=(",", ":"))
        lines.append(line)
    lines[0] = lines[0].replace(
        "HloModule dense_convolution,",
        "HloModule dense_convolution, is_scheduled=true,",
    )
    return "\n".join(lines)


def _with_exact_split_layer1_rms_schedule(hlo: str) -> str:
    reduction_config = {
        "megacore_config": {
            "megacore_allreduce_bytes": "4096",
            "megacore_split_dim": "0",
        },
        "window_config": {
            "cost_model_type": "COST_MODEL_TYPE_INVALID",
            "input_window_bounds": [],
            "is_mask": False,
            "iteration_bounds": ["2", "1"],
            "kernel_window_bounds": [],
            "output_window_bounds": ["2", "48"],
            "pad_input_on_minor_dim": "0",
            "pad_output_on_minor_dim": "0",
        },
    }
    scope = (
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_layer1_rmsnorm/'
    )
    reduction_computation = "\n".join(
        (
            "%split_rms_reduction {",
            "  %split_update = bf16[32,6144] parameter(0)",
            "  %split_residual = bf16[32,6144] parameter(1)",
            "  %split_update_f32 = f32[32,6144] convert(%split_update)",
            "  %split_residual_f32 = f32[32,6144] convert(%split_residual)",
            "  %split_combined = f32[32,6144] add("
            "%split_update_f32, %split_residual_f32), " + scope + 'add"}',
            "  %split_square = f32[32,6144] multiply("
            "%split_combined, %split_combined), " + scope + 'square"}',
            "  %split_zero = f32[] constant(0)",
            "  ROOT %split_sum = f32[32] reduce(%split_square, %split_zero), "
            "dimensions={1}, to_apply=%sum_reducer, " + scope + 'reduce_sum"}',
            "}",
        )
    )
    output_computation = "\n".join(
        (
            "%split_rms_output {",
            "  %output_update = bf16[32,6144] parameter(0)",
            "  %output_residual = bf16[32,6144] parameter(1)",
            "  %output_inverse = f32[32,1] parameter(2)",
            "  %output_norm = bf16[6144] parameter(3)",
            "  %output_update_f32 = f32[32,6144] convert(%output_update)",
            "  %output_residual_f32 = f32[32,6144] "
            "convert(%output_residual)",
            "  %output_combined = f32[32,6144] add("
            "%output_update_f32, %output_residual_f32), " + scope + 'add"}',
            "  %output_inverse_wide = f32[32,6144] broadcast("
            "%output_inverse), dimensions={0,1}",
            "  %output_normalized = f32[32,6144] multiply("
            "%output_combined, %output_inverse_wide), " + scope + 'mul"}',
            "  %output_rounded = bf16[32,6144] convert(%output_normalized)",
            "  %output_norm_seed = bf16[1,6144] broadcast("
            "%output_norm), dimensions={1}",
            "  %output_norm_wide = bf16[32,6144] broadcast("
            "%output_norm_seed), dimensions={0,1}",
            "  ROOT %output_weighted = bf16[32,6144] multiply("
            "%output_rounded, %output_norm_wide), " + scope + 'mul"}',
            "}",
        )
    )
    hlo = hlo.replace(
        "ENTRY main {",
        reduction_computation + "\n\n" + output_computation + "\n\nENTRY main {",
        1,
    )
    old = "\n".join(
        (
            "  %update_f32 = f32[32,6144] convert(%update_m32)",
            "  %residual_f32 = f32[32,6144] convert(%predense_carried)",
            "  %combined = f32[32,6144] add(%update_f32, %residual_f32), "
            + scope
            + 'add"}',
            "  %square = f32[32,6144] multiply(%combined, %combined), "
            + scope
            + 'square"}',
            "  %zero = f32[] constant(0)",
            "  %sum = f32[32] reduce(%square, %zero), dimensions={1}, "
            "to_apply=%sum_reducer, "
            + scope
            + 'reduce_sum"}',
        )
    )
    new = "\n".join(
        (
            "  %split_sum = f32[32] fusion(%update_m32, %predense_carried), "
            "kind=kLoop, calls=%split_rms_reduction, "
            + scope
            + 'reduce_sum"}, backend_config='
            + json.dumps(reduction_config, separators=(",", ":")),
        )
    )
    assert old in hlo
    hlo = hlo.replace(old, new, 1).replace(
        "  %sum_row = f32[32,1] reshape(%sum)",
        "  %sum_row = f32[32,1] reshape(%split_sum)",
        1,
    )
    old_output = "\n".join(
        (
            "  %inverse_wide = f32[32,6144] broadcast(%inverse), "
            "dimensions={0,1}",
            "  %rms_normalized = f32[32,6144] multiply("
            "%combined, %inverse_wide), " + scope + 'mul"}',
            "  %rounded = bf16[32,6144] convert(%rms_normalized)",
            "  %norm_seed = bf16[1,6144] broadcast(%norm), dimensions={1}",
            "  %norm_wide = bf16[32,6144] broadcast(%norm_seed), "
            "dimensions={0,1}",
            "  %layer1_m32 = bf16[32,6144] multiply(%rounded, %norm_wide), "
            + scope
            + 'mul"}',
        )
    )
    new_output = (
        "  %layer1_m32 = bf16[32,6144] fusion("
        "%update_m32, %predense_carried, %inverse, %norm), "
        "kind=kLoop, calls=%split_rms_output"
    )
    assert old_output in hlo
    return hlo.replace(old_output, new_output, 1)


@lru_cache(maxsize=1)
def _exact_dense_envelope_split_rms_stablehlo() -> str:
    stablehlo = _exact_dense_envelope_stablehlo()
    source = "\n".join(
        (
            "      %828 = stablehlo.convert %827 : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %829 = stablehlo.convert %6 : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %830 = stablehlo.add %828, %829 : tensor<32x6144xf32>",
        )
    )
    replacement = "\n".join(
        (
            "      %split_dense = stablehlo.optimization_barrier %827 : "
            "tensor<32x6144xbf16>",
            "      %split_residual = stablehlo.optimization_barrier %6 : "
            "tensor<32x6144xbf16>",
            "      %split_dense_f32 = stablehlo.convert %split_dense : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %split_residual_f32 = stablehlo.convert %split_residual : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %830 = stablehlo.add %split_dense_f32, "
            "%split_residual_f32 : tensor<32x6144xf32>",
            "      %828 = stablehlo.convert %827 : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %829 = stablehlo.convert %6 : "
            "(tensor<32x6144xbf16>) -> tensor<32x6144xf32>",
            "      %split_combined = stablehlo.add %828, %829 : "
            "tensor<32x6144xf32>",
        )
    )
    assert source in stablehlo
    return stablehlo.replace(source, replacement, 1).replace(
        "%840 = stablehlo.multiply %830, %839",
        "%840 = stablehlo.multiply %split_combined, %839",
        1,
    )


def _synthetic_dense_envelope_externalized_gate_hlo() -> str:
    """Move each exact RMS output across a fusion boundary before its gate."""

    hlo = _synthetic_dense_envelope_optimized_hlo()
    for rank in range(8):
        gate_scope = (
            f"jit(probe)/greenfield_dense_convolution_virtual_rank_{rank:02d}/"
            "gate_up"
        )
        internal_gate = (
            f"  %gate_value.{rank} = f32[32,768] convolution("
            "%gate_weighted, %gate_rhs), dim_labels=bf_io->bf, "
            f'metadata={{op_name="{gate_scope}"}}\n'
        )
        assert internal_gate in hlo
        hlo = hlo.replace(internal_gate, "", 1)
        if rank == 0:
            hlo = hlo.replace(
                "ROOT %gate_root.0 = (f32[32,768], bf16[32,6144]) "
                "tuple(%gate_value.0, %gate_carried)",
                "ROOT %gate_root.0 = (bf16[32,6144], bf16[32,6144]) "
                "tuple(%gate_weighted, %gate_carried)",
                1,
            ).replace(
                "  %gate_pair.0 = (f32[32,768], bf16[32,6144]) fusion(",
                "  %gate_pair.0 = (bf16[32,6144], bf16[32,6144]) fusion(",
                1,
            ).replace(
                "  %gate_up.0 = f32[32,768] get-tuple-element("
                "%gate_pair.0), index=0",
                "  %m32_padded.0 = bf16[32,6144] get-tuple-element("
                "%gate_pair.0), index=0\n"
                "  %gate_up.0 = f32[32,768] convolution("
                "%m32_padded.0, %gate_up_weight.0), dim_labels=bf_io->bf, "
                f'metadata={{op_name="{gate_scope}"}}',
                1,
            )
        else:
            hlo = hlo.replace(
                f"ROOT %gate_root.{rank} = f32[32,768] "
                f"copy(%gate_value.{rank})",
                f"ROOT %gate_root.{rank} = bf16[32,6144] "
                "copy(%gate_weighted)",
                1,
            ).replace(
                f"  %gate_up.{rank} = f32[32,768] fusion(",
                f"  %m32_padded.{rank} = bf16[32,6144] fusion(",
                1,
            )
            caller_end = f"calls=%predense_gate_{rank}"
            replacement = (
                caller_end
                + f"\n  %gate_up.{rank} = f32[32,768] convolution("
                + f"%m32_padded.{rank}, %gate_up_weight.{rank}), "
                + "dim_labels=bf_io->bf, "
                + f'metadata={{op_name="{gate_scope}"}}'
            )
            hlo = hlo.replace(caller_end, replacement, 1)
    return hlo


def _synthetic_dense_envelope_nested_rms_hlo() -> str:
    """Put exact RMS arithmetic in a nested producer inside each gate fusion."""

    hlo = _synthetic_dense_envelope_optimized_hlo()
    rms_computations = []
    for rank in range(8):
        computation_start = hlo.index(f"%predense_gate_{rank} {{")
        arithmetic_start = hlo.index("  %gate_zero =", computation_start)
        arithmetic_end = hlo.index("  %gate_value.", arithmetic_start)
        arithmetic = hlo[arithmetic_start:arithmetic_end]
        nested_arithmetic = (
            arithmetic.replace("%gate_attention", "%rms_attention")
            .replace("%gate_residual", "%rms_residual")
            .replace("%gate_norm", "%rms_norm")
            .replace("%gate_", "%rms_")
        )
        rms_computations.append(
            "\n".join(
                (
                    f"%predense_rms_{rank} {{",
                    "  %rms_attention = bf16[1,6144] parameter(0)",
                    "  %rms_residual = bf16[1,6144] parameter(1)",
                    "  %rms_norm = bf16[6144] parameter(2)",
                    nested_arithmetic.rstrip(),
                    "  ROOT %rms_root = (bf16[32,6144], bf16[32,6144]) "
                    "tuple(%rms_weighted, %rms_carried)",
                    "}",
                )
            )
        )
        nested_call = "\n".join(
            (
                "  %gate_rms_pair = (bf16[32,6144], bf16[32,6144]) "
                "fusion(%gate_attention, %gate_residual, %gate_norm), "
                f"kind=kLoop, calls=%predense_rms_{rank}",
                "  %gate_weighted = bf16[32,6144] get-tuple-element("
                "%gate_rms_pair), index=0",
                "  %gate_carried = bf16[32,6144] get-tuple-element("
                "%gate_rms_pair), index=1",
                "",
            )
        )
        hlo = hlo[:arithmetic_start] + nested_call + hlo[arithmetic_end:]
    hlo = hlo.replace(
        "%predense_gate_0 {",
        "\n".join(rms_computations) + "\n\n%predense_gate_0 {",
        1,
    )
    return hlo


def _synthetic_dense_envelope_tpu_corrections_hlo() -> str:
    """Model folded row scalars, async copies, and BF16 correction lowering."""

    hlo = _synthetic_dense_envelope_optimized_hlo()
    folded_replacements = (
        (
            "  %gate_sum_row = f32[32,1] reshape(%gate_sum_squares)\n"
            "  %gate_width = f32[32,1] constant(6144)\n"
            "  %gate_mean = f32[32,1] divide("
            "%gate_sum_row, %gate_width), ",
            "  %gate_width = f32[32] constant(6144)\n"
            "  %gate_mean = f32[32] divide("
            "%gate_sum_squares, %gate_width), ",
        ),
        ("  %gate_epsilon = f32[32,1]", "  %gate_epsilon = f32[32]"),
        ("  %gate_variance = f32[32,1]", "  %gate_variance = f32[32]"),
        ("  %gate_inverse = f32[32,1]", "  %gate_inverse = f32[32]"),
        (
            "broadcast(%gate_inverse), dimensions={0,1}",
            "broadcast(%gate_inverse), dimensions={0}",
        ),
    )
    for old, new in folded_replacements:
        assert old in hlo
        hlo = hlo.replace(old, new)

    weighted_scope = (
        'metadata={op_name="jit(probe)/'
        'greenfield_dense_convolution_predense_rmsnorm/mul"}'
    )
    old_weighted = (
        "  %gate_weighted = bf16[32,6144] multiply("
        "%gate_rounded, %gate_norm_wide), " + weighted_scope
    )
    corrected_weighted = (
        "  %gate_rounded_f32 = f32[32,6144] convert(%gate_rounded)\n"
        "  %gate_norm_f32 = f32[32,6144] convert(%gate_norm_wide)\n"
        "  %gate_weighted_f32 = f32[32,6144] multiply("
        "%gate_rounded_f32, %gate_norm_f32), "
        + weighted_scope
        + ', backend_config={"float_type_correction_info":'
        '{"original_type":"BF16"}}\n'
        "  %gate_weighted = bf16[32,6144] convert(%gate_weighted_f32)"
    )
    assert hlo.count(old_weighted) == 8
    hlo = hlo.replace(old_weighted, corrected_weighted)

    copy_anchor = (
        "  %packed_down_scale = f32[1,8,3,6144]{3,2,1,0} parameter(6)"
    )
    copy_values = "\n".join(
        (
            copy_anchor,
            "  %attention_copy_start = (bf16[1,6144], bf16[1,6144], "
            "u32[]) copy-start(%attention)",
            "  %attention_copy_done = bf16[1,6144] "
            "copy-done(%attention_copy_start)",
            "  %residual_copy_start = (bf16[1,6144], bf16[1,6144], "
            "u32[]) copy-start(%combined_residual)",
            "  %residual_copy_done = bf16[1,6144] "
            "copy-done(%residual_copy_start)",
        )
    )
    assert copy_anchor in hlo
    hlo = hlo.replace(copy_anchor, copy_values, 1)
    gate_arguments = "%attention, %combined_residual, %post_norm,"
    copied_arguments = (
        "%attention_copy_done, %residual_copy_done, %post_norm,"
    )
    assert hlo.count(gate_arguments) == 8
    return hlo.replace(gate_arguments, copied_arguments)


def _synthetic_final_layout_async_down_scale_hlo() -> str:
    hlo = _synthetic_final_layout_optimized_hlo()
    direct = (
        "  %down_scale_slice.0 = f32[1,1,3,6144]{3,2,1,0} "
        "slice(%packed_down_scale), "
        "slice={[0:1], [0:1], [0:3], [0:6144]}}"
    )
    asynchronous = "\n".join(
        (
            "  %down_scale_slice_start.0 = ((f32[1,8,3,6144]{3,2,1,0}), "
            "f32[1,1,3,6144]{3,2,1,0}, s32[]) slice-start(%packed_down_scale), "
            "slice={[0:1], [0:1], [0:3], [0:6144]}}",
            "  %down_scale_slice.0 = f32[1,1,3,6144]{3,2,1,0} "
            "slice-done(%down_scale_slice_start.0)",
        )
    )
    assert direct in hlo
    return hlo.replace(direct, asynchronous, 1)


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
    ((4, 8, 6144, 768), jnp.float8_e4m3fn, slot),
    ((4, 8, 48, 768), jnp.float32, slot),
    ((4, 8, 384, 6144), jnp.float8_e4m3fn, slot),
    ((4, 8, 3, 6144), jnp.float32, slot),
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


@lru_cache(maxsize=1)
def _exact_dense_envelope_stablehlo() -> str:
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
    ((6144,), jnp.bfloat16, replicated),
    ((4, 8, 6144, 768), jnp.float8_e4m3fn, slot),
    ((4, 8, 48, 768), jnp.float32, slot),
    ((4, 8, 384, 6144), jnp.float8_e4m3fn, slot),
    ((4, 8, 3, 6144), jnp.float32, slot),
    ((6144,), jnp.bfloat16, replicated),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)

def local(attention, residual, post_norm, merged_bits, merged_scale,
          down_bits, down_scale, layer1_norm):
    attention = jnp.pad(
        attention, ((0, 31), (0, 0)), constant_values=jnp.bfloat16(0),
    )
    residual = jnp.pad(
        residual, ((0, 31), (0, 0)), constant_values=jnp.bfloat16(0),
    )
    normalized, post_attention = fused_add_rms_norm(
        attention, residual, post_norm, epsilon=1e-5,
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
        update, ((0, 31), (0, 0)), constant_values=jnp.bfloat16(0),
    )
    layer1 = fused_add_rms_norm(
        update, post_attention, layer1_norm, epsilon=1e-5,
    )[0]
    return layer1[:1, :]

mapped = jax.shard_map(
    local,
    mesh=mesh,
    in_specs=(P(), P(), P(), P("lp4", None, None, None),
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


@lru_cache(maxsize=1)
def _exact_dense_partial_capture_stablehlo() -> str:
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm
from glm_tpu.greenfield.kernels.stage_local import (
    _virtual_dense_final_layout_convolution_down_partials,
)

mesh = Mesh(np.asarray(jax.devices()), ("lp4",))
replicated = NamedSharding(mesh, P())
slot = NamedSharding(mesh, P("lp4", None, None, None))
contracts = (
    ((1, 6144), jnp.bfloat16, replicated),
    ((1, 6144), jnp.bfloat16, replicated),
    ((6144,), jnp.bfloat16, replicated),
    ((4, 8, 6144, 768), jnp.float8_e4m3fn, slot),
    ((4, 8, 48, 768), jnp.float32, slot),
    ((4, 8, 384, 6144), jnp.float8_e4m3fn, slot),
    ((4, 8, 3, 6144), jnp.float32, slot),
)
arguments = tuple(
    jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    for shape, dtype, sharding in contracts
)

def local(attention, residual, post_norm, merged_bits, merged_scale,
          down_bits, down_scale):
    attention = jnp.pad(
        attention, ((0, 31), (0, 0)), constant_values=jnp.bfloat16(0),
    )
    residual = jnp.pad(
        residual, ((0, 31), (0, 0)), constant_values=jnp.bfloat16(0),
    )
    normalized, carried = fused_add_rms_norm(
        attention, residual, post_norm, epsilon=1e-5,
    )
    partials = _virtual_dense_final_layout_convolution_down_partials(
        normalized, merged_bits[0], merged_scale[0], down_bits[0],
        down_scale[0], block_shape=(128, 128), compile_rows=32,
    )
    partials = lax.optimization_barrier(partials)[:, :1, :]
    gathered = lax.all_gather(
        partials, "lp4", axis=0, tiled=False,
        axis_index_groups=((0, 1, 2, 3),),
    )
    return gathered, carried

mapped = jax.shard_map(
    local,
    mesh=mesh,
    in_specs=(P(), P(), P(), P("lp4", None, None, None),
              P("lp4", None, None, None), P("lp4", None, None, None),
              P("lp4", None, None, None)),
    out_specs=(P(), P()),
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
    accepted = _with_exact_scheduled_dense_tiling(
        _synthetic_final_layout_optimized_hlo()
    )
    contract = MODULE._validate_optimized_hlo(
        accepted,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert contract["passed"], contract
    assert contract["exact_accepted_weight_layout"] is True
    assert contract["exact_packed_weight_lineage"] is True
    assert contract["scheduled_kernel_geometry_required"] is True
    assert contract["exact_accepted_kernel_geometry"] is True
    for source, replacement in (
        ('"kernel_window_bounds":["384","6"]', '"kernel_window_bounds":["48","6"]'),
        ('"iteration_bounds":["8","1","1"]', '"iteration_bounds":["6","1","1"]'),
        ('"input_window_bounds":["4","24"]', '"input_window_bounds":["999","999"]'),
        ('"pad_input_on_minor_dim":"0"', '"pad_input_on_minor_dim":"1"'),
        ('"cost_model_type":"COST_MODEL_TYPE_CLASSIC"', '"cost_model_type":"ROGUE"'),
        ('"is_mask":false', '"is_mask":true'),
    ):
        wrong_tiling = accepted.replace(source, replacement, 1)
        assert wrong_tiling != accepted
        rejected = MODULE._validate_optimized_hlo(
            wrong_tiling,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
        )
        assert rejected["passed"] is False
        assert rejected["exact_accepted_kernel_geometry"] is False
        assert "accepted dense convolution tiling drifted" in rejected["violations"]
    no_owner_scale = accepted.replace(
        "  %gate_scale_slice.0 = f32[1,1,48,768]{3,2,1,0} "
        "slice(%packed_gate_scale), "
        "slice={[0:1], [0:1], [0:48], [0:768]}",
        "  %gate_scale_no_owner.0 = f32[8,48,768]{2,1,0} "
        "reshape(%packed_gate_scale)\n"
        "  %gate_scale_slice.0 = f32[1,48,768]{2,1,0} "
        "slice(%gate_scale_no_owner.0), "
        "slice={[0:1], [0:48], [0:768]}",
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
            "  %down_f32.0 = f32[384,6144] convert(%down_bits.0)",
            "  %rogue_down_bits.0 = f8e4m3fn[384,6144] add("
            "%down_bits.0, %down_bits.0)\n"
            "  %down_f32.0 = f32[384,6144] convert(%rogue_down_bits.0)",
            1,
        ),
        accepted.replace(
            "  %gate_bits_slice.0 = f8e4m3fn[1,1,6144,768]{3,2,1,0} "
            "slice(%packed_gate_bits), slice={[0:1], [0:1], [0:6144], [0:768]}",
            "  %reassociated_gate_bits.0 = f8e4m3fn[1,6144,8,768]{3,2,1,0} "
            "reshape(%packed_gate_bits)\n"
            "  %gate_bits_slice.0 = f8e4m3fn[1,6144,1,768]{3,2,1,0} "
            "slice(%reassociated_gate_bits.0), "
            "slice={[0:1], [0:6144], [0:1], [0:768]}",
            1,
        ),
        accepted.replace(
            "  %gate_scale_seed.0 = f32[48,768]{1,0} "
            "reshape(%gate_scale_slice.0)\n"
            "  %gate_scale_inner.0 = f32[48,128,768]{2,1,0} "
            "broadcast(%gate_scale_seed.0), dimensions={0,2}",
            "  %gate_scale_seed.0 = f32[768,48]{1,0} "
            "reshape(%gate_scale_slice.0)\n"
            "  %gate_scale_inner.0 = f32[768,128,48]{2,1,0} "
            "broadcast(%gate_scale_seed.0), dimensions={0,2}",
            1,
        ),
        accepted.replace(
            "%gate_bits.0 = f8e4m3fn[6144,768]{1,0} bitcast",
            "%gate_bits.0 = f8e4m3fn[6144,768]{0,1} bitcast",
            1,
        ),
        accepted.replace(
            "%down_bits.0 = f8e4m3fn[384,6144]{1,0} bitcast",
            "%down_bits.0 = f8e4m3fn[384,6144]{0,1} bitcast",
            1,
        ),
        accepted.replace(
            "%gate_scale_wide.0 = f32[6144,768]{1,0} reshape",
            "%gate_scale_wide.0 = f32[6144,768]{0,1} reshape",
            1,
        ),
        accepted.replace(
            "%down_scale_wide.0 = f32[384,6144]{1,0} reshape",
            "%down_scale_wide.0 = f32[384,6144]{0,1} reshape",
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


@pytest.mark.skipif(
    not REAL_FINAL_LAYOUT_OPTIMIZED_HLO.exists(),
    reason="protected final-layout optimized HLO is unavailable",
)
def test_dense_final_layout_real_hlo_records_rejected_compressed_scales() -> None:
    optimized_hlo = REAL_FINAL_LAYOUT_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_FINAL_LAYOUT_OPTIMIZED_HLO_SHA256
    )
    contract = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert contract["passed"] is False
    assert contract["scheduled_kernel_geometry_required"] is True
    assert contract["exact_packed_weight_lineage"] is False
    assert contract["exact_accepted_kernel_geometry"] is False
    assert "packed dense weight lineage drifted" in contract["violations"]
    assert "accepted dense convolution tiling drifted" in contract["violations"]


def test_dense_final_layout_accepts_only_exact_async_rank_slice() -> None:
    optimized_hlo = _synthetic_final_layout_async_down_scale_hlo()
    contract = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert contract["passed"], contract
    assert contract["exact_packed_weight_lineage"] is True
    wrong_rank = optimized_hlo.replace(
        "slice={[0:1], [0:1], [0:3], [0:6144]}}",
        "slice={[0:1], [1:2], [0:3], [0:6144]}}",
        1,
    )
    assert wrong_rank != optimized_hlo
    rejected = MODULE._validate_optimized_hlo(
        wrong_rank,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert rejected["passed"] is False
    assert rejected["exact_packed_weight_lineage"] is False


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
    assert contract["gate_up_layout_constraint_count"] == 8
    dependency_lines = [
        line
        for line in stablehlo.splitlines()
        if ":2 = stablehlo.optimization_barrier" in line
    ]
    assert len(dependency_lines) == 7
    rank_zero_barriers = [
        line
        for line in stablehlo.splitlines()
        if " = stablehlo.optimization_barrier " in line
        and ":2 =" not in line
        and "tensor<6144x768xbf16>" in line
    ]
    assert len(rank_zero_barriers) == 1
    rank_zero_match = re.search(
        r"(%[^ ]+) = stablehlo\.optimization_barrier (%[^ ]+) :",
        rank_zero_barriers[0],
    )
    assert rank_zero_match is not None
    missing_rank_zero_barrier = stablehlo.replace(
        rank_zero_barriers[0] + "\n", "", 1
    )
    missing_rank_zero_barrier = re.sub(
        re.escape(rank_zero_match.group(1)) + r"(?![A-Za-z0-9_.$#-])",
        rank_zero_match.group(2),
        missing_rank_zero_barrier,
    )
    assert missing_rank_zero_barrier != stablehlo
    rejected_rank_zero = MODULE._validate_stablehlo(
        missing_rank_zero_barrier,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert rejected_rank_zero["passed"] is False
    assert "accepted gate/up barrier drifted" in " ".join(
        rejected_rank_zero["violations"]
    )
    first_dependency = re.search(
        r"optimization_barrier\s+%[^,]+,\s*(%[^ ]+)",
        dependency_lines[0],
    )
    second_dependency = re.search(
        r"optimization_barrier\s+%[^,]+,\s*(%[^ ]+)",
        dependency_lines[1],
    )
    assert first_dependency is not None and second_dependency is not None
    crosswired_dependency = stablehlo.replace(
        dependency_lines[1],
        dependency_lines[1].replace(
            second_dependency.group(1), first_dependency.group(1), 1
        ),
        1,
    )
    assert crosswired_dependency != stablehlo
    rejected_dependency = MODULE._validate_stablehlo(
        crosswired_dependency,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert rejected_dependency["passed"] is False
    assert (
        "dense virtual shards lost the exact rank-ordered dependency"
        in rejected_dependency["violations"]
    )
    layout_lines = [
        line
        for line in stablehlo.splitlines()
        if "stablehlo.custom_call @LayoutConstraint" in line
    ]
    assert len(layout_lines) == 8
    result_match = re.search(
        r"^\s*(%[A-Za-z0-9_.$#-]+)\s*=", layout_lines[0]
    )
    source_match = re.search(
        r"@LayoutConstraint\((%[A-Za-z0-9_.$#-]+)\)", layout_lines[0]
    )
    assert result_match is not None and source_match is not None
    missing_layout = stablehlo.replace(layout_lines[0] + "\n", "", 1)
    missing_layout = re.sub(
        rf"{re.escape(result_match.group(1))}"
        r"(?![A-Za-z0-9_.$#-])",
        source_match.group(1),
        missing_layout,
    )
    missing_contract = MODULE._validate_stablehlo(
        missing_layout,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert missing_contract["passed"] is False
    wrong_layout = stablehlo.replace(
        "result_layouts = [dense<[1, 0]> : tensor<2xindex>]",
        "result_layouts = [dense<[0, 1]> : tensor<2xindex>]",
        1,
    )
    wrong_contract = MODULE._validate_stablehlo(
        wrong_layout,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
    )
    assert wrong_contract["passed"] is False
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


def test_dense_envelope_contract_binds_both_rmsnorm_boundaries() -> None:
    stablehlo = _exact_dense_envelope_stablehlo()
    stable = MODULE._validate_stablehlo(
        stablehlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert stable["passed"], stable
    assert stable["dense_envelope"] is True

    optimized_hlo = _with_exact_scheduled_dense_tiling(
        _synthetic_dense_envelope_optimized_hlo()
    )
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert optimized["passed"], optimized
    assert optimized["exact_accepted_kernel_geometry"] is True
    predense = optimized["lineage"]["predense_rmsnorm_contract"]
    assert predense["exact_operand_graph"] is True
    assert predense["exact_gate_input_binding"] is True
    assert predense["exact_fused_gate_ownership"] is True
    assert predense["fused_gate_binding_count"] == 8
    assert predense["exact_carried_residual_binding"] is True
    nested = MODULE._validate_optimized_hlo(
        _synthetic_dense_envelope_nested_rms_hlo(),
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert nested["passed"], nested
    assert nested["lineage"]["predense_rmsnorm_contract"][
        "exact_fused_gate_ownership"
    ] is True
    corrected_hlo = _synthetic_dense_envelope_tpu_corrections_hlo()
    corrected = MODULE._validate_optimized_hlo(
        corrected_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert corrected["passed"], corrected
    assert corrected["lineage"]["predense_rmsnorm_contract"][
        "weighted_value_count"
    ] == 8

    stable_mutations = (
        stablehlo.replace(
            "stablehlo.convolution(%20,",
            "stablehlo.convolution(%6,",
            1,
            ),
            stablehlo.replace(
                "stablehlo.convert %6 :",
                "stablehlo.convert %20 :",
                1,
            ),
        stablehlo.replace(
            "%14 = stablehlo.rsqrt %13",
            "%14 = stablehlo.rsqrt %11",
            1,
        ),
        stablehlo.replace(
            "applies stablehlo.add",
            "applies stablehlo.maximum",
            1,
        ),
    )
    assert all(value != stablehlo for value in stable_mutations)
    for mutated in stable_mutations:
        rejected = MODULE._validate_stablehlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
        )
        assert rejected["passed"] is False


    optimized_mutations = (
        optimized_hlo.replace(
            "convolution(%gate_weighted, %gate_rhs)",
            "convolution(%gate_carried, %gate_rhs)",
            1,
        ),
        optimized_hlo.replace(
            "  %gate_value.0 =",
            "  %rogue_predense = bf16[32,6144] add("
            "%gate_weighted, %gate_weighted)\n"
            "  %gate_value.0 =",
            1,
        ).replace(
            "convolution(%gate_weighted, %gate_rhs)",
            "convolution(%rogue_predense, %gate_rhs)",
            1,
        ),
        optimized_hlo.replace(
            "  %residual_f32 = f32[32,6144] "
            "convert(%predense_carried)",
            "  %residual_f32 = f32[32,6144] convert(%attention)",
            1,
        ),
        optimized_hlo.replace(
            "  %gate_inverse = f32[32,1] rsqrt(%gate_variance)",
            "  %gate_rogue_variance = f32[32,1] add("
            "%gate_variance, %gate_variance)\n"
            "  %gate_inverse = f32[32,1] rsqrt(%gate_rogue_variance)",
            1,
        ),
        optimized_hlo.replace(
            "  %gate_value.",
            "  %gate_materialized = bf16[32,6144] "
            "optimization-barrier(%gate_weighted)\n"
            "  %gate_value.",
        ).replace(
            "convolution(%gate_weighted, %gate_rhs)",
            "convolution(%gate_materialized, %gate_rhs)",
        ),
        optimized_hlo.replace(
            "  %gate_value.",
            "  %gate_s16 = s16[32,6144] convert(%gate_weighted)\n"
            "  %gate_reconverted = bf16[32,6144] convert(%gate_s16)\n"
            "  %gate_value.",
        ).replace(
            "convolution(%gate_weighted, %gate_rhs)",
            "convolution(%gate_reconverted, %gate_rhs)",
        ),
        _synthetic_dense_envelope_externalized_gate_hlo(),
    )
    assert all(value != optimized_hlo for value in optimized_mutations)
    for mutated in optimized_mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
        )
        assert rejected["passed"] is False
        assert "optimized pre-dense fused RMSNorm envelope drifted" in (
            rejected["violations"]
        )

    correction_mutations = (
        corrected_hlo.replace(
            '"original_type":"BF16"', '"original_type":"F32"', 1
        ),
        corrected_hlo.replace(
            "  %gate_rounded_f32 = f32[32,6144] convert(%gate_rounded)",
            "  %gate_rounded_s16 = s16[32,6144] convert(%gate_rounded)\n"
            "  %gate_rounded_f32 = f32[32,6144] "
            "convert(%gate_rounded_s16)",
            1,
        ),
        corrected_hlo.replace(
            "  %gate_weighted = bf16[32,6144] "
            "convert(%gate_weighted_f32)",
            "  %gate_weighted_rogue = f32[32,6144] add("
            "%gate_weighted_f32, %gate_weighted_f32)\n"
            "  %gate_weighted = bf16[32,6144] "
            "convert(%gate_weighted_rogue)",
            1,
        ),
        corrected_hlo.replace(
            "copy-done(%attention_copy_start)",
            "copy-done(%residual_copy_start)",
            1,
        ),
        corrected_hlo.replace(
            "  %gate_inverse_wide = f32[32,6144] broadcast("
            "%gate_inverse), dimensions={0}",
            "  %gate_inverse_transposed = f32[6144,32] broadcast("
            "%gate_inverse), dimensions={1}\n"
            "  %gate_inverse_wide = f32[32,6144] "
            "reshape(%gate_inverse_transposed)",
            1,
        ),
    )
    assert all(value != corrected_hlo for value in correction_mutations)
    for mutated in correction_mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
        )
        assert rejected["passed"] is False
        assert "optimized pre-dense fused RMSNorm envelope drifted" in (
            rejected["violations"]
        )


def test_dense_split_layer1_rms_requires_recompute_and_accepted_schedule() -> None:
    stablehlo = _exact_dense_envelope_split_rms_stablehlo()
    stable = MODULE._validate_stablehlo(
        stablehlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert stable["passed"], stable
    assert stable["split_layer1_rms"] is True

    optimized_hlo = _with_exact_split_layer1_rms_schedule(
        _with_exact_scheduled_dense_tiling(
            _synthetic_dense_envelope_optimized_hlo()
        )
    )
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert optimized["passed"], optimized
    assert optimized["split_layer1_rms"] is True
    rms = optimized["lineage"]["rmsnorm_contract"]
    assert rms["split_recompute_exact"] is True
    assert rms["exact_accepted_scheduled_reduction"] is True
    assert rms["accepted_scheduled_reduction_values"] == ["%split_sum"]

    stable_bypass = stablehlo.replace(
        "%840 = stablehlo.multiply %split_combined, %839",
        "%840 = stablehlo.multiply %830, %839",
        1,
    )
    assert stable_bypass != stablehlo
    assert not MODULE._validate_stablehlo(
        stable_bypass,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )["passed"]

    mutations = (
        optimized_hlo.replace(
            '"output_window_bounds":["2","48"]',
            '"output_window_bounds":["4","24"]',
            1,
        ),
        optimized_hlo.replace(
            '"megacore_allreduce_bytes":"4096","megacore_split_dim":"0"',
            '"megacore_allreduce_bytes":"4096","megacore_split_dim":"1"',
            1,
        ),
        optimized_hlo.replace(
            "%output_update_f32, %output_residual_f32",
            "%output_update_f32, %output_update_f32",
            1,
        ),
        optimized_hlo.replace(
            "  %layer1_m32 = bf16[32,6144] fusion("
            "%update_m32, %predense_carried, %inverse, %norm), "
            "kind=kLoop, calls=%split_rms_output",
            "\n".join(
                (
                    "  %main_update_barrier = bf16[32,6144] "
                    "optimization-barrier(%update_m32)",
                    "  %main_residual_barrier = bf16[32,6144] "
                    "optimization-barrier(%predense_carried)",
                    "  %main_update_f32 = f32[32,6144] "
                    "convert(%main_update_barrier)",
                    "  %main_residual_f32 = f32[32,6144] "
                    "convert(%main_residual_barrier)",
                    "  %main_combined = f32[32,6144] add("
                    "%main_update_f32, %main_residual_f32), "
                    'metadata={op_name="jit(probe)/'
                    'greenfield_dense_convolution_layer1_rmsnorm/add"}',
                    "  %main_inverse_wide = f32[32,6144] broadcast("
                    "%inverse), dimensions={0,1}",
                    "  %main_normalized = f32[32,6144] multiply("
                    "%main_combined, %main_inverse_wide), "
                    'metadata={op_name="jit(probe)/'
                    'greenfield_dense_convolution_layer1_rmsnorm/mul"}',
                    "  %main_rounded = bf16[32,6144] convert(%main_normalized)",
                    "  %main_norm_seed = bf16[1,6144] broadcast("
                    "%norm), dimensions={1}",
                    "  %main_norm_wide = bf16[32,6144] broadcast("
                    "%main_norm_seed), dimensions={0,1}",
                    "  %layer1_m32 = bf16[32,6144] multiply("
                    "%main_rounded, %main_norm_wide), "
                    'metadata={op_name="jit(probe)/'
                    'greenfield_dense_convolution_layer1_rmsnorm/mul"}',
                )
            ),
            1,
        ),
    )
    assert all(mutated != optimized_hlo for mutated in mutations)
    for mutated in mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
            split_layer1_rms=True,
        )
        assert rejected["passed"] is False
        assert "accepted split layer-1 RMS schedule drifted" in (
            rejected["violations"]
        )

    old_schedule = _with_exact_scheduled_dense_tiling(
        _synthetic_dense_envelope_optimized_hlo()
    )
    rejected_old = MODULE._validate_optimized_hlo(
        old_schedule,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert rejected_old["passed"] is False


@pytest.mark.skipif(
    not REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO.exists(),
    reason="protected output-barrier split-RMS HLO is unavailable",
)
def test_dense_split_layer1_rms_rejects_output_barrier_tuple_schedule() -> None:
    optimized_hlo = REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_OUTPUT_BARRIER_SPLIT_RMS_HLO_SHA256
    )
    rejected = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert rejected["passed"] is False
    assert "accepted split layer-1 RMS schedule drifted" in (
        rejected["violations"]
    )
    rms = rejected["lineage"]["rmsnorm_contract"]
    assert rms["exact_accepted_scheduled_reduction"] is False
    assert rms["accepted_scheduled_reduction_values"] == []


@pytest.mark.skipif(
    not (
        REAL_REDUCTION_BARRIER_SPLIT_RMS_HLO.exists()
        and REAL_REDUCTION_BARRIER_SPLIT_RMS_STABLEHLO.exists()
    ),
    reason="protected reduction-barrier split-RMS HLO is unavailable",
)
def test_dense_split_layer1_rms_replays_scalar_schedule_and_exact_output() -> None:
    optimized_hlo = REAL_REDUCTION_BARRIER_SPLIT_RMS_HLO.read_text()
    stablehlo = REAL_REDUCTION_BARRIER_SPLIT_RMS_STABLEHLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_REDUCTION_BARRIER_SPLIT_RMS_HLO_SHA256
    )
    assert sha256(stablehlo.encode()).hexdigest() == (
        REAL_REDUCTION_BARRIER_SPLIT_RMS_STABLEHLO_SHA256
    )
    stable = MODULE._validate_stablehlo(
        stablehlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert stable["passed"], stable
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
        split_layer1_rms=True,
    )
    assert optimized["passed"], optimized
    rms = optimized["lineage"]["rmsnorm_contract"]
    assert rms["exact_reduction_operand_graph"] is True
    assert rms["split_recompute_exact"] is True
    assert rms["split_output_fusion_exact"] is True
    assert rms["exact_accepted_scheduled_reduction"] is True
    assert rms["accepted_scheduled_reduction_values"] == [
        "%multiply_reduce_fusion"
    ]
    predense = optimized["lineage"]["predense_rmsnorm_contract"]
    assert predense["exact_layer1_source_identity"] is True

    rogue_arithmetic_lines = []
    pad_attribute_spoof_lines = []
    slice_attribute_spoof_lines = []
    pad_comment_spoof_lines = []
    slice_comment_spoof_lines = []
    association_scope_drift_lines = []
    for line in optimized_hlo.splitlines():
        if line.lstrip().startswith("%pad.12 ="):
            rogue_arithmetic_lines.append(
                "  %rogue_dense = bf16[1,6144]{1,0:T(2,128)(2,1)S(3)} "
                "add(%constant_dynamic-update-slice_fusion, "
                "%constant_dynamic-update-slice_fusion)"
            )
            rogue_arithmetic_lines.append(
                line.replace(
                    "pad(%constant_dynamic-update-slice_fusion,",
                    "pad(%rogue_dense,",
                    1,
                )
            )
            pad_attribute_spoof_lines.append(
                line.replace(
                    "padding=0_31x0_0, metadata={op_name=\"",
                    "padding=31_0x0_0, metadata={op_name=\""
                    "padding=0_31x0_0/",
                    1,
                )
            )
            pad_comment_spoof_lines.append(
                line.replace(
                    "padding=0_31x0_0,",
                    "/* padding=0_31x0_0 */ padding=31_0x0_0,",
                    1,
                )
            )
        else:
            rogue_arithmetic_lines.append(
                line.replace(
                    "fusion(%get-tuple-element.135, "
                    "%constant_dynamic-update-slice_fusion,",
                    "fusion(%get-tuple-element.135, %rogue_dense,",
                    1,
                )
            )
            pad_attribute_spoof_lines.append(line)
            pad_comment_spoof_lines.append(line)
        if line.lstrip().startswith("%slice.1263 ="):
            slice_attribute_spoof_lines.append(
                line.replace(
                    "slice={[0:1], [0:6144]}, metadata={op_name=\"",
                    "slice={[1:2], [0:6144]}, metadata={op_name=\""
                    "slice={[0:1],[0:6144]}/",
                    1,
                )
            )
            slice_comment_spoof_lines.append(
                line.replace(
                    "slice={[0:1], [0:6144]},",
                    "/* slice={[0:1],[0:6144]} */ "
                    "slice={[1:2], [0:6144]},",
                    1,
                )
            )
        else:
            slice_attribute_spoof_lines.append(line)
            slice_comment_spoof_lines.append(line)
        if line.lstrip().startswith("%add.633 ="):
            association_scope_drift_lines.append(
                line.replace(
                    "greenfield_strategy_nd_row0_dense_convolution_down/"
                    "greenfield_strategy_nd_row0_association/add",
                    "greenfield_strategy_nd_row0_dense_convolution_down/"
                    "rogue_association/add",
                    1,
                )
            )
        else:
            association_scope_drift_lines.append(line)
    rogue_arithmetic = "\n".join(rogue_arithmetic_lines)
    pad_attribute_spoof = "\n".join(pad_attribute_spoof_lines)
    slice_attribute_spoof = "\n".join(slice_attribute_spoof_lines)
    pad_comment_spoof = "\n".join(pad_comment_spoof_lines)
    slice_comment_spoof = "\n".join(slice_comment_spoof_lines)
    association_scope_drift = "\n".join(association_scope_drift_lines)

    mutations = (
        optimized_hlo.replace(
            "%add.608 = f32[32,6144]{1,0:T(8,128)} add("
            "%convert_element_type.357, %convert_element_type.356)",
            "%add.608 = f32[32,6144]{1,0:T(8,128)} add("
            "%convert_element_type.357, %convert_element_type.357)",
            1,
        ),
        optimized_hlo.replace(
            "%add.673 = f32[1,6144]{1,0:T(1,128)} add("
            "%convert_element_type.446, %convert_element_type.445)",
            "%add.673 = f32[1,6144]{1,0:T(1,128)} add("
            "%convert_element_type.446, %convert_element_type.446)",
            1,
        ),
        optimized_hlo.replace(
            "%slice.1263 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice("
            "%param_0.467), slice={[0:1], [0:6144]}",
            "%slice.1263 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice("
            "%param_0.467), slice={[1:2], [0:6144]}",
            1,
        ),
        optimized_hlo.replace(
            "%constant.2.clone.3 = bf16[]{:T(256)} constant(0)",
            "%constant.2.clone.3 = bf16[]{:T(256)} constant(1)",
            1,
        ),
        optimized_hlo.replace(
            "  %slice.1263 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice("
            "%param_0.467), slice={[0:1], [0:6144]}",
            "\n".join(
                (
                    "  %rogue_layout = bf16[32,6144]{0,1} bitcast("
                    "%param_0.467)",
                    "  %slice.1263 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
                    "slice(%rogue_layout), slice={[0:1], [0:6144]}",
                )
            ),
            1,
        ),
        rogue_arithmetic,
        pad_attribute_spoof,
        slice_attribute_spoof,
        pad_comment_spoof,
        slice_comment_spoof,
        association_scope_drift,
    )
    assert all(mutated != optimized_hlo for mutated in mutations)
    for mutated in mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
            split_layer1_rms=True,
        )
        assert rejected["passed"] is False
        assert "accepted split layer-1 RMS schedule drifted" in (
            rejected["violations"]
        )


@pytest.mark.skipif(
    not (
        REAL_DENSE_ENVELOPE_OPTIMIZED_HLO.exists()
        and REAL_DENSE_ENVELOPE_STABLEHLO.exists()
    ),
    reason="protected dense-envelope HLO is unavailable",
)
def test_dense_envelope_replays_db547_and_rejects_old_tiling() -> None:
    optimized_hlo = REAL_DENSE_ENVELOPE_OPTIMIZED_HLO.read_text()
    stablehlo = REAL_DENSE_ENVELOPE_STABLEHLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_DENSE_ENVELOPE_OPTIMIZED_HLO_SHA256
    )
    assert sha256(stablehlo.encode()).hexdigest() == (
        REAL_DENSE_ENVELOPE_STABLEHLO_SHA256
    )
    optimized = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    stable = MODULE._validate_stablehlo(
        stablehlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert optimized["passed"] is False
    assert stable["passed"] is False
    assert optimized["scheduled_kernel_geometry_required"] is True
    assert optimized["exact_packed_weight_lineage"] is False
    assert optimized["exact_accepted_kernel_geometry"] is False
    assert "packed dense weight lineage drifted" in optimized["violations"]
    assert "accepted dense convolution tiling drifted" in optimized["violations"]
    predense = optimized["lineage"]["predense_rmsnorm_contract"]
    assert predense["fused_gate_binding_count"] == 8
    assert predense["weighted_value_count"] == 8
    assert predense["exact_carried_residual_binding"] is True


@pytest.mark.skipif(
    not REAL_ACCEPTED_SCALE_ENVELOPE_OPTIMIZED_HLO.exists(),
    reason="protected accepted-scale envelope HLO is unavailable",
)
def test_dense_envelope_replays_accepted_scale_lowering_fail_closed() -> None:
    optimized_hlo = REAL_ACCEPTED_SCALE_ENVELOPE_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_ACCEPTED_SCALE_ENVELOPE_OPTIMIZED_HLO_SHA256
    )
    contract = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert contract["passed"] is False
    assert contract["exact_packed_weight_lineage"] is True
    assert contract["exact_accepted_kernel_geometry"] is False
    assert contract["violations"] == [
        "accepted dense convolution weight layout drifted",
        "accepted dense convolution tiling drifted",
    ]
    gate_rows = contract["accepted_weight_layouts"]["gate_up"]
    down_rows = contract["accepted_weight_layouts"]["down"]
    assert [
        row["virtual_rank"]
        for row in gate_rows
        if not row["exact_convolution_tiling"]
    ] == [1]
    assert all(row["exact_packed_dequant"] for row in gate_rows)
    assert all(
        row["exact_packed_dequant"]
        and row["exact_convolution_tiling"]
        for row in down_rows
    )

    wrong_rank_zero_copy = optimized_hlo.replace(
        "%copy.56 = f32[1,1,3,6144]{3,2,1,0:T(4,128)} copy",
        "%copy.56 = f32[1,1,3,6144]{3,1,2,0:T(4,128)} copy",
        1,
    )
    assert wrong_rank_zero_copy != optimized_hlo
    rejected = MODULE._validate_optimized_hlo(
        wrong_rank_zero_copy,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert rejected["exact_packed_weight_lineage"] is False
    assert rejected["accepted_weight_layouts"]["down"][0][
        "exact_packed_dequant"
    ] is False


@pytest.mark.skipif(
    not REAL_SERIALIZED_SCALE_ENVELOPE_OPTIMIZED_HLO.exists(),
    reason="protected serialized-scale envelope HLO is unavailable",
)
def test_dense_envelope_binds_materialized_gate_tuple_fail_closed() -> None:
    optimized_hlo = REAL_SERIALIZED_SCALE_ENVELOPE_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_SERIALIZED_SCALE_ENVELOPE_OPTIMIZED_HLO_SHA256
    )
    contract = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert contract["passed"] is False
    assert contract["exact_packed_weight_lineage"] is True
    assert contract["exact_accepted_kernel_geometry"] is False
    gate_rows = contract["accepted_weight_layouts"]["gate_up"]
    assert all(row["exact_packed_dequant"] for row in gate_rows)
    assert [
        row["virtual_rank"]
        for row in gate_rows
        if not row["exact_convolution_tiling"]
    ] == [0]
    assert all(
        row["exact_packed_dequant"]
        and row["exact_convolution_tiling"]
        for row in contract["accepted_weight_layouts"]["down"]
    )

    duplicated_piece = optimized_hlo.replace(
        "slice={[0:1536], [0:768]}",
        "slice={[1536:3072], [0:768]}",
        1,
    )
    assert duplicated_piece != optimized_hlo
    rejected_piece = MODULE._validate_optimized_hlo(
        duplicated_piece,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert rejected_piece["exact_packed_weight_lineage"] is False

    swapped_pieces = optimized_hlo.replace(
        "custom-call(%slice-done.24, %slice-done.25, "
        "%slice-done.26, %slice-done.27), "
        'custom_call_target="ConcatBitcast"',
        "custom-call(%slice-done.25, %slice-done.24, "
        "%slice-done.26, %slice-done.27), "
        'custom_call_target="ConcatBitcast"',
        1,
    )
    assert swapped_pieces != optimized_hlo
    rejected_order = MODULE._validate_optimized_hlo(
        swapped_pieces,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert rejected_order["exact_packed_weight_lineage"] is False

    wrong_tuple_result = optimized_hlo.replace(
        "get-tuple-element(%fusion), index=6, metadata={op_name=\"jit("
        "local_dense_envelope)/shard_map/greenfield_dense_convolution_"
        "virtual_rank_07/convert_element_type\"",
        "get-tuple-element(%fusion), index=5, metadata={op_name=\"jit("
        "local_dense_envelope)/shard_map/greenfield_dense_convolution_"
        "virtual_rank_07/convert_element_type\"",
        1,
    )
    assert wrong_tuple_result != optimized_hlo
    rejected_tuple = MODULE._validate_optimized_hlo(
        wrong_tuple_result,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert rejected_tuple["exact_packed_weight_lineage"] is False


@pytest.mark.skipif(
    not REAL_ACCEPTED_GEOMETRY_ENVELOPE_OPTIMIZED_HLO.exists(),
    reason="protected all-accepted dense geometry HLO is unavailable",
)
def test_dense_envelope_binds_split_m32_stack_fail_closed() -> None:
    optimized_hlo = REAL_ACCEPTED_GEOMETRY_ENVELOPE_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_ACCEPTED_GEOMETRY_ENVELOPE_OPTIMIZED_HLO_SHA256
    )
    contract = MODULE._validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        layer1_only=True,
        final_dense_layout=True,
        dense_envelope=True,
    )
    assert contract["passed"] is True, contract["violations"]
    assert contract["exact_packed_weight_lineage"] is True
    assert contract["exact_accepted_kernel_geometry"] is True
    assert contract["lineage"]["ordered_stack_sources"] == [
        [rank] for rank in range(8)
    ]
    assert len(contract["lineage"]["m32_fused_stack_callers"]) == 8

    mutations = (
        optimized_hlo.replace(
            "%constant.195 = s32[] constant(1)",
            "%constant.195 = s32[] constant(2)",
            1,
        ),
        optimized_hlo.replace(
            "fusion(%bitcast_dynamic-update-slice_fusion.7, %fusion.58), "
            "kind=kLoop, calls=%fused_computation.69",
            "fusion(%bitcast_dynamic-update-slice_fusion.7, %fusion.55), "
            "kind=kLoop, calls=%fused_computation.69",
            1,
        ),
        optimized_hlo.replace(
            "fusion(%bitcast_dynamic-update-slice_fusion.6, %fusion.55), "
            "kind=kLoop, calls=%fused_computation.68",
            "fusion(%bitcast_dynamic-update-slice_fusion.7, %fusion.55), "
            "kind=kLoop, calls=%fused_computation.68",
            1,
        ),
        optimized_hlo.replace(
            "%mul.190 = f32[6144,768]{1,0:T(8,128)} multiply(",
            "%mul.190 = f32[6144,768]{1,0:T(8,128)} add(",
            1,
        ),
    )
    assert all(mutated != optimized_hlo for mutated in mutations)
    for mutated in mutations:
        rejected = MODULE._validate_optimized_hlo(
            mutated,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
        )
        assert rejected["passed"] is False


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


@pytest.mark.skipif(
    not REAL_DB548_OPTIMIZED_HLO.exists(),
    reason="preserved DB548 TPU HLO is not available",
)
def test_dense_partial_capture_optimized_contract_pins_schedules_and_gather() -> None:
    optimized_hlo = REAL_DB548_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_DB548_OPTIMIZED_HLO_SHA256
    )
    captured = optimized_hlo.replace(
        "greenfield_strategy_nd_row0_dense_convolution_down/"
        "greenfield_strategy_nd_row0_association/"
        "greenfield_strategy_nd_row0_association_gather",
        "greenfield_dense_partial_capture_gather",
    )
    captured = re.sub(
        r"^  ROOT %fusion\.91 = .*?$",
        "  %capture_view = bf16[4,8,1,6144]{3,2,1,0} "
        "bitcast(%all-gather)\n"
        "  ROOT %capture_tuple = (bf16[4,8,1,6144]{3,2,1,0}, "
        "bf16[32,6144]{1,0}) tuple(%capture_view, %copy-done.1)",
        captured,
        count=1,
        flags=re.MULTILINE,
    )
    accepted = MODULE._validate_partial_capture_optimized_hlo(captured)
    assert accepted["passed"] is True
    assert accepted["accepted_gate_up_ranks"] == list(range(8))
    assert accepted["accepted_down_ranks"] == list(range(8))
    assert accepted["capture_gather_count"] == 1
    assert accepted["exact_capture_lineage"] is True
    assert accepted["exact_live_schedule_bijection"] is True
    assert accepted["exact_result_binding"] is True

    wrong_window = captured.replace(
        '"input_window_bounds":["4","24"]',
        '"input_window_bounds":["4","3"]',
        1,
    )
    assert MODULE._validate_partial_capture_optimized_hlo(wrong_window)[
        "passed"
    ] is False
    wrong_group = captured.replace(
        "replica_groups={{0,1,2,3}}",
        "replica_groups={{0,1,3,2}}",
        1,
    )
    assert MODULE._validate_partial_capture_optimized_hlo(wrong_group)[
        "passed"
    ] is False
    rogue_capture = captured.replace(
        "  %all-gather = ",
        "  %rogue_capture = bf16[8,1,6144]{2,1,0} constant(0)\n"
        "  %all-gather = ",
        1,
    ).replace(
        "all-gather(%copy.29)",
        "all-gather(%rogue_capture)",
        1,
    )
    assert MODULE._validate_partial_capture_optimized_hlo(rogue_capture)[
        "passed"
    ] is False
    wrong_live_row = captured.replace(
        "slice={[0:8], [0:1], [0:6144]}",
        "slice={[0:8], [1:2], [0:6144]}",
        1,
    )
    assert wrong_live_row != captured
    assert MODULE._validate_partial_capture_optimized_hlo(wrong_live_row)[
        "passed"
    ] is False
    rogue_arithmetic = captured.replace(
        "  %all-gather = ",
        "  %rogue_add = bf16[8,1,6144]{2,0,1:T(8,128)(2,1)S(3)} "
        "add(%copy.29, %copy.29)\n"
        "  %all-gather = ",
        1,
    ).replace(
        "all-gather(%copy.29)",
        "all-gather(%rogue_add)",
        1,
    )
    assert MODULE._validate_partial_capture_optimized_hlo(rogue_arithmetic)[
        "passed"
    ] is False
    rogue_result = captured.replace(
        "  %capture_view = ",
        "  %rogue_result = bf16[32,1,6144]{2,0,1:T(8,128)(2,1)S(3)} "
        "add(%all-gather, %all-gather)\n"
        "  %capture_view = ",
        1,
    ).replace(
        "bitcast(%all-gather)\n  ROOT %capture_tuple",
        "bitcast(%rogue_result)\n  ROOT %capture_tuple",
        1,
    )
    assert MODULE._validate_partial_capture_optimized_hlo(rogue_result)[
        "passed"
    ] is False

    live_down = next(
        line for line in captured.splitlines() if line.startswith("  %fusion.61 =")
    )
    wrong_live_down = live_down.replace(
        '/conv_general_dilated"', '/actual_live_down"', 1
    ).replace(
        '"iteration_bounds":["8","1","1"]',
        '"iteration_bounds":["7","1","1"]',
        1,
    )
    live_shape = re.match(r"  %fusion\.61 = (\S+) fusion\(", live_down)
    assert live_shape is not None
    dead_schedule = (
        f"  %dead_schedule = {live_shape.group(1)} copy(%fusion.61), metadata="
        + live_down.split("metadata=", 1)[1]
    )
    decoy_schedule = captured.replace(
        live_down, f"{wrong_live_down}\n{dead_schedule}", 1
    )
    decoy_result = MODULE._validate_partial_capture_optimized_hlo(
        decoy_schedule
    )
    assert decoy_result["accepted_down_ranks"] == list(range(8))
    assert decoy_result["exact_live_schedule_bijection"] is False
    assert decoy_result["passed"] is False

    async_collective = captured.replace(
        "  %capture_view = ",
        "  %rogue_async_start = ((f32[32]{0:T(128)S(3)}, "
        "f32[32]{0:T(128)S(3)}), u32[]) "
        "all-reduce-start(%add_rsqrt_fusion.1), channel_id=99, "
        "replica_groups={{0,1,2,3}}, use_global_device_ids=true, "
        "to_apply=%region_0.1\n"
        "  %rogue_async_done = f32[32]{0:T(128)S(3)} "
        "all-reduce-done(%rogue_async_start), channel_id=99\n"
        "  %capture_view = ",
        1,
    )
    async_result = MODULE._validate_partial_capture_optimized_hlo(
        async_collective
    )
    assert async_result["async_collectives"] == [
        "%rogue_async_start",
        "%rogue_async_done",
    ]
    assert async_result["passed"] is False


@pytest.mark.skipif(
    not REAL_PARTIAL_CAPTURE_OPTIMIZED_HLO.exists(),
    reason="preserved protected partial-capture HLO is not available",
)
def test_dense_partial_capture_replays_exact_protected_hlo() -> None:
    optimized_hlo = REAL_PARTIAL_CAPTURE_OPTIMIZED_HLO.read_text()
    assert sha256(optimized_hlo.encode()).hexdigest() == (
        REAL_PARTIAL_CAPTURE_OPTIMIZED_HLO_SHA256
    )
    result = MODULE._validate_partial_capture_optimized_hlo(optimized_hlo)
    assert result["passed"], result["violations"]
    assert result["exact_live_schedule_bijection"] is True
    assert result["exact_capture_lineage"] is True


def test_dense_partial_capture_stablehlo_pins_group_and_outer_return() -> None:
    stablehlo = _exact_dense_partial_capture_stablehlo()

    def validate(candidate: str) -> dict[str, object]:
        return MODULE._validate_stablehlo(
            candidate,
            compile_rows=32,
            layer1_only=True,
            final_dense_layout=True,
            dense_envelope=True,
            partials_only=True,
        )

    accepted = validate(stablehlo)
    assert accepted["passed"], accepted["violations"]

    exact_group = (
        "replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>"
    )
    wrong_group = stablehlo.replace(
        exact_group,
        "replica_groups = dense<[[0, 1, 3, 2]]> : tensor<1x4xi64> "
        f"/* {exact_group} */",
        1,
    )
    assert wrong_group != stablehlo
    assert not validate(wrong_group)["passed"]

    wrong_outer = stablehlo.replace(
        "%0:2 = sdy.manual_computation(",
        "%manual:2 = sdy.manual_computation(",
        1,
    )
    outer_return = (
        "    return %0#0, %0#1 : tensor<4x8x1x6144xbf16>, "
        "tensor<32x6144xbf16>"
    )
    decoy_return = (
        "    %capture_decoy = stablehlo.constant dense<0.000000e+00> : "
        "tensor<4x8x1x6144xbf16>\n"
        "    %carried_decoy = stablehlo.constant dense<0.000000e+00> : "
        "tensor<32x6144xbf16>\n"
        "    %0:2 = stablehlo.optimization_barrier %capture_decoy, "
        "%carried_decoy : tensor<4x8x1x6144xbf16>, tensor<32x6144xbf16>\n"
        f"{outer_return}"
    )
    wrong_outer = wrong_outer.replace(outer_return, decoy_return, 1)
    assert wrong_outer != stablehlo
    assert not validate(wrong_outer)["passed"]
