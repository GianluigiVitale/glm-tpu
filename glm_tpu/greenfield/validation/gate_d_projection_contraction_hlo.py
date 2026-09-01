"""Offline adjudication of the diagnostic Gate-D projection contraction HLO.

This module imports no JAX and never lowers, compiles, or invokes an executable.
It accepts the burned V2 run only as generation-bound diagnostic evidence.
"""

from __future__ import annotations

import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..sharding.hlo_contract import HloInstruction, HloShape, parse_hlo_module


class ProjectionContractionHloError(RuntimeError):
    """Raised when diagnostic HLO identity or causal structure drifts."""


EXPECTED_RUN_TAG = "gate_d_projection_contraction_pp16_hlo_20260901T204123954066730Z"
EXPECTED_RUN_DIR = Path("/home/gianl/gate-d-runs") / EXPECTED_RUN_TAG
EXPECTED_CODE_HASH = "d6a77b9ee7478bf0bb55d71ca43a3b5b0b0357bc"
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "7f678b10667252005d9c467496aeee44279bd55428e2d6ac5d9bb6051f384f4d"
)
EXPECTED_STABLEHLO_SHA256 = (
    "4b3fa252e837d381208453b06d7e369947fa4c6c58c795edc8ff5a8ea8c9e2e3"
)
EXPECTED_DIAGNOSTIC_LEDGER_SHA256 = (
    "16ce098c1d681b489cbd304c7cd5d213f8c1c378de31bb96e24d88f03e7540bc"
)
EXPECTED_DIAGNOSTIC_LEDGER_GENERATION = "1788295391457443"
EXPECTED_REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
    "gate_d_projection_contraction_pp16_hlo/" + EXPECTED_RUN_TAG + "/diagnostic"
)
EXPECTED_REMOTE_REPLAY = (
    Path(__file__).resolve().parents[3] / "docs/artifacts/"
    "gate-d-projection-contraction-pp16-hlo-diagnostic-remote-replay.json"
)
EXPECTED_REMOTE_REPLAY_SHA256 = (
    "5d6583beb2f52048754252053847fc3f5205553fa6244ad762181b7797578b61"
)
EXPECTED_FAILURE_ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "docs/artifacts/gate-d-projection-contraction-pp16-hlo-v2-publication-failure.json"
)
EXPECTED_FAILURE_ARTIFACT_SHA256 = (
    "d38c43afdd6c87d671c39e8f71448d73ac71290fee2bfc8fb207647d3e9010aa"
)

_ENTRY = (
    "ENTRY %main.5_spmd (param.5: bf16[1,1,6144], "
    "param.6: f32[1,128,6144], param.7: bf16[1,128], "
    "param.8: bf16[1,128]) -> (bf16[1,1,6144], f32[1,1,128], "
    "f32[1,1,128])"
)
_CONTRACTION = (
    "%fused_computation.1 (param_0.43: f32[1,128,6144], "
    "param_1.57: f32[6144]) -> (f32[], f32[128])"
)
_FORBIDDEN_OPCODES = {
    "all-gather",
    "all-reduce",
    "all-to-all",
    "collective-permute",
    "infeed",
    "outfeed",
    "recv",
    "recv-done",
    "reduce-scatter",
    "send",
    "send-done",
}
_FORBIDDEN_TEXT = (
    "host_callback",
    "xla_python_cpu_callback",
    "outside_compilation",
)


def _shape(dtype: str, *dimensions: int) -> HloShape:
    return HloShape(dtype=dtype, dimensions=dimensions)


def _instructions(
    instructions: tuple[HloInstruction, ...], computation: str
) -> dict[str, HloInstruction]:
    return {item.name: item for item in instructions if item.computation == computation}


def _require(
    instructions: dict[str, HloInstruction],
    name: str,
    opcode: str,
    operands: tuple[str, ...],
    results: tuple[HloShape, ...],
) -> HloInstruction:
    instruction = instructions.get(name)
    if instruction is None or (
        instruction.opcode,
        instruction.operand_names,
        instruction.result_shapes,
    ) != (opcode, operands, results):
        raise ProjectionContractionHloError(f"projection HLO edge drifted: {name}")
    return instruction


def _consumers(instructions: dict[str, HloInstruction], name: str) -> set[str]:
    return {
        instruction.name
        for instruction in instructions.values()
        if name in instruction.operand_names
    }


def _require_exact_consumers(
    instructions: dict[str, HloInstruction],
    expected: dict[str, set[str]],
    boundary: str,
) -> None:
    if any(
        _consumers(instructions, name) != consumers
        for name, consumers in expected.items()
    ):
        raise ProjectionContractionHloError(f"{boundary} consumer graph drifted")


def _require_root(instruction: HloInstruction, boundary: str) -> None:
    if not instruction.raw_line.startswith("ROOT "):
        raise ProjectionContractionHloError(f"{boundary} is not rooted")


def _audit_current_key_fusion_bodies(
    instructions: tuple[HloInstruction, ...],
) -> None:
    """Prove every called current-key fusion carries its parameters to its root."""

    normalize_name = (
        "%fused_computation.6 (param_0.26: bf16[1,128], "
        "param_1.32: bf16[1,128], param_2.46: f32[128], "
        "param_3.41: f32[], param_4.17: f32[]) -> f32[1,128]"
    )
    normalize = _instructions(instructions, normalize_name)
    if set(normalize) != {
        "%param_2.46",
        "%param_4.17",
        "%sub.23",
        "%sub.21",
        "%param_3.41",
        "%div.57",
        "%div.31",
        "%bitcast.17",
        "%param_1.32",
        "%convert_element_type.16",
        "%mul.44",
        "%param_0.26",
        "%convert_element_type.15",
        "%add.15",
    }:
        raise ProjectionContractionHloError(
            "normalization fusion instruction set drifted"
        )
    _require(normalize, "%param_2.46", "parameter", ("2",), (_shape("f32", 128),))
    _require(normalize, "%param_4.17", "parameter", ("4",), (_shape("f32"),))
    _require(normalize, "%sub.23", "broadcast", ("%param_4.17",), (_shape("f32", 128),))
    _require(
        normalize,
        "%sub.21",
        "subtract",
        ("%param_2.46", "%sub.23"),
        (_shape("f32", 128),),
    )
    _require(normalize, "%param_3.41", "parameter", ("3",), (_shape("f32"),))
    _require(normalize, "%div.57", "broadcast", ("%param_3.41",), (_shape("f32", 128),))
    _require(
        normalize,
        "%div.31",
        "divide",
        ("%sub.21", "%div.57"),
        (_shape("f32", 128),),
    )
    _require(
        normalize,
        "%bitcast.17",
        "bitcast",
        ("%div.31",),
        (_shape("f32", 1, 128),),
    )
    _require(
        normalize,
        "%param_1.32",
        "parameter",
        ("1",),
        (_shape("bf16", 1, 128),),
    )
    _require(
        normalize,
        "%convert_element_type.16",
        "convert",
        ("%param_1.32",),
        (_shape("f32", 1, 128),),
    )
    _require(
        normalize,
        "%mul.44",
        "multiply",
        ("%bitcast.17", "%convert_element_type.16"),
        (_shape("f32", 1, 128),),
    )
    _require(
        normalize,
        "%param_0.26",
        "parameter",
        ("0",),
        (_shape("bf16", 1, 128),),
    )
    _require(
        normalize,
        "%convert_element_type.15",
        "convert",
        ("%param_0.26",),
        (_shape("f32", 1, 128),),
    )
    normalize_root = _require(
        normalize,
        "%add.15",
        "add",
        ("%mul.44", "%convert_element_type.15"),
        (_shape("f32", 1, 128),),
    )
    _require_root(normalize_root, "normalization fusion output")
    _require_exact_consumers(
        normalize,
        {
            "%param_2.46": {"%sub.21"},
            "%param_4.17": {"%sub.23"},
            "%sub.23": {"%sub.21"},
            "%sub.21": {"%div.31"},
            "%param_3.41": {"%div.57"},
            "%div.57": {"%div.31"},
            "%div.31": {"%bitcast.17"},
            "%bitcast.17": {"%mul.44"},
            "%param_1.32": {"%convert_element_type.16"},
            "%convert_element_type.16": {"%mul.44"},
            "%mul.44": {"%add.15"},
            "%param_0.26": {"%convert_element_type.15"},
            "%convert_element_type.15": {"%add.15"},
            "%add.15": set(),
        },
        "normalization fusion",
    )

    split_name = (
        "%fused_computation.13 (param_0.45: f32[1,128]) -> (f32[1,64], f32[1,64])"
    )
    split = _instructions(instructions, split_name)
    if set(split) != {"%param_0.45", "%slice.16", "%slice.17", "%tuple.9"}:
        raise ProjectionContractionHloError("split fusion instruction set drifted")
    _require(split, "%param_0.45", "parameter", ("0",), (_shape("f32", 1, 128),))
    first_slice = _require(
        split,
        "%slice.16",
        "slice",
        ("%param_0.45",),
        (_shape("f32", 1, 64),),
    )
    second_slice = _require(
        split,
        "%slice.17",
        "slice",
        ("%param_0.45",),
        (_shape("f32", 1, 64),),
    )
    if (
        "slice={[0:1], [0:64]}" not in first_slice.raw_line
        or "slice={[0:1], [64:128]}" not in second_slice.raw_line
    ):
        raise ProjectionContractionHloError("split fusion slice indices drifted")
    split_root = _require(
        split,
        "%tuple.9",
        "tuple",
        ("%slice.16", "%slice.17"),
        (_shape("f32", 1, 64), _shape("f32", 1, 64)),
    )
    _require_root(split_root, "split fusion output")
    _require_exact_consumers(
        split,
        {
            "%param_0.45": {"%slice.16", "%slice.17"},
            "%slice.16": {"%tuple.9"},
            "%slice.17": {"%tuple.9"},
            "%tuple.9": set(),
        },
        "split fusion",
    )

    rotary_name = (
        "%fused_computation.8 (param_0.22: f32[1,32], "
        "param_1.53: f32[1,32]) -> (f32[1,32,1], f32[1,32,1])"
    )
    rotary = _instructions(instructions, rotary_name)
    if set(rotary) != {
        "%param_1.53",
        "%constant.41",
        "%mul.84",
        "%constant.17.clone.3",
        "%pow.26",
        "%iota.13",
        "%constant.18.clone.3",
        "%mul.83",
        "%mul.82",
        "%neg.13",
        "%constant.19.clone.3",
        "%div.48",
        "%div.47",
        "%pow.25",
        "%mul.81",
        "%cos.8",
        "%mul.48",
        "%param_0.22",
        "%sin.8",
        "%mul.47",
        "%sub.20",
        "%bitcast.19",
        "%mul.46.clone.1",
        "%mul.45.clone.1",
        "%add.16.clone.1",
        "%bitcast.18.clone.1",
        "%tuple.8",
    }:
        raise ProjectionContractionHloError("rotary fusion instruction set drifted")
    _require(rotary, "%param_1.53", "parameter", ("1",), (_shape("f32", 1, 32),))
    _require(rotary, "%param_0.22", "parameter", ("0",), (_shape("f32", 1, 32),))
    _require(rotary, "%cos.8", "cosine", ("%mul.81",), (_shape("f32", 1, 32),))
    _require(rotary, "%sin.8", "sine", ("%mul.81",), (_shape("f32", 1, 32),))
    _require(
        rotary,
        "%mul.48",
        "multiply",
        ("%param_1.53", "%cos.8"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%mul.47",
        "multiply",
        ("%param_0.22", "%sin.8"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%sub.20",
        "subtract",
        ("%mul.48", "%mul.47"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%bitcast.19",
        "bitcast",
        ("%sub.20",),
        (_shape("f32", 1, 32, 1),),
    )
    _require(
        rotary,
        "%mul.46.clone.1",
        "multiply",
        ("%param_0.22", "%cos.8"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%mul.45.clone.1",
        "multiply",
        ("%param_1.53", "%sin.8"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%add.16.clone.1",
        "add",
        ("%mul.46.clone.1", "%mul.45.clone.1"),
        (_shape("f32", 1, 32),),
    )
    _require(
        rotary,
        "%bitcast.18.clone.1",
        "bitcast",
        ("%add.16.clone.1",),
        (_shape("f32", 1, 32, 1),),
    )
    rotary_root = _require(
        rotary,
        "%tuple.8",
        "tuple",
        ("%bitcast.19", "%bitcast.18.clone.1"),
        (_shape("f32", 1, 32, 1), _shape("f32", 1, 32, 1)),
    )
    _require_root(rotary_root, "rotary fusion output")
    _require_exact_consumers(
        rotary,
        {
            "%param_1.53": {"%mul.48", "%mul.45.clone.1"},
            "%param_0.22": {"%mul.47", "%mul.46.clone.1"},
            "%cos.8": {"%mul.48", "%mul.46.clone.1"},
            "%sin.8": {"%mul.47", "%mul.45.clone.1"},
            "%mul.48": {"%sub.20"},
            "%mul.47": {"%sub.20"},
            "%sub.20": {"%bitcast.19"},
            "%bitcast.19": {"%tuple.8"},
            "%mul.46.clone.1": {"%add.16.clone.1"},
            "%mul.45.clone.1": {"%add.16.clone.1"},
            "%add.16.clone.1": {"%bitcast.18.clone.1"},
            "%bitcast.18.clone.1": {"%tuple.8"},
            "%tuple.8": set(),
        },
        "rotary fusion",
    )

    stack_name = (
        "%fused_computation.3 (param_0.8: f32[1,32,1], "
        "param_1.55: f32[1,32,1]) -> f32[1,32,2]"
    )
    stack = _instructions(instructions, stack_name)
    if set(stack) != {
        "%param_1.55",
        "%constant.39",
        "%pad.5",
        "%param_0.8",
        "%pad.4",
        "%maximum.2",
    }:
        raise ProjectionContractionHloError("stack fusion instruction set drifted")
    _require(stack, "%param_1.55", "parameter", ("1",), (_shape("f32", 1, 32, 1),))
    _require(
        stack,
        "%pad.5",
        "pad",
        ("%param_1.55", "%constant.39"),
        (_shape("f32", 1, 32, 2),),
    )
    _require(stack, "%param_0.8", "parameter", ("0",), (_shape("f32", 1, 32, 1),))
    _require(
        stack,
        "%pad.4",
        "pad",
        ("%param_0.8", "%constant.39"),
        (_shape("f32", 1, 32, 2),),
    )
    if (
        "padding=0_0x0_0x0_1" not in stack["%pad.5"].raw_line
        or "padding=0_0x0_0x1_0" not in stack["%pad.4"].raw_line
    ):
        raise ProjectionContractionHloError("stack fusion padding drifted")
    stack_root = _require(
        stack,
        "%maximum.2",
        "maximum",
        ("%pad.5", "%pad.4"),
        (_shape("f32", 1, 32, 2),),
    )
    _require_root(stack_root, "stack fusion output")
    _require_exact_consumers(
        stack,
        {
            "%param_1.55": {"%pad.5"},
            "%param_0.8": {"%pad.4"},
            "%constant.39": {"%pad.5", "%pad.4"},
            "%pad.5": {"%maximum.2"},
            "%pad.4": {"%maximum.2"},
            "%maximum.2": set(),
        },
        "stack fusion",
    )

    join_name = (
        "%fused_computation.4 (param_0.11: f32[1,64], "
        "param_1.54: f32[1,64]) -> f32[1,1,128]"
    )
    join = _instructions(instructions, join_name)
    if set(join) != {
        "%param_1.54",
        "%constant.38",
        "%pad.7",
        "%param_0.11",
        "%pad.6",
        "%maximum.3",
        "%bitcast.14",
    }:
        raise ProjectionContractionHloError("join fusion instruction set drifted")
    _require(join, "%param_1.54", "parameter", ("1",), (_shape("f32", 1, 64),))
    _require(
        join,
        "%pad.7",
        "pad",
        ("%param_1.54", "%constant.38"),
        (_shape("f32", 1, 128),),
    )
    _require(join, "%param_0.11", "parameter", ("0",), (_shape("f32", 1, 64),))
    _require(
        join,
        "%pad.6",
        "pad",
        ("%param_0.11", "%constant.38"),
        (_shape("f32", 1, 128),),
    )
    if (
        "padding=0_0x0_64" not in join["%pad.7"].raw_line
        or "padding=0_0x64_0" not in join["%pad.6"].raw_line
    ):
        raise ProjectionContractionHloError("join fusion padding drifted")
    _require(
        join,
        "%maximum.3",
        "maximum",
        ("%pad.7", "%pad.6"),
        (_shape("f32", 1, 128),),
    )
    join_root = _require(
        join,
        "%bitcast.14",
        "bitcast",
        ("%maximum.3",),
        (_shape("f32", 1, 1, 128),),
    )
    _require_root(join_root, "join fusion output")
    _require_exact_consumers(
        join,
        {
            "%param_1.54": {"%pad.7"},
            "%param_0.11": {"%pad.6"},
            "%constant.38": {"%pad.7", "%pad.6"},
            "%pad.7": {"%maximum.3"},
            "%pad.6": {"%maximum.3"},
            "%maximum.3": {"%bitcast.14"},
            "%bitcast.14": set(),
        },
        "join fusion",
    )


def audit_optimized_hlo_structure(optimized_hlo: str) -> dict[str, Any]:
    """Prove the FP32 6144 contraction, one-row owners, and zero communication."""

    try:
        module = parse_hlo_module(optimized_hlo)
    except (TypeError, ValueError) as error:
        raise ProjectionContractionHloError("optimized HLO is not parseable") from error
    if (
        module.name != "jit__projection_contraction_local"
        or module.num_partitions != 2
        or module.num_replicas not in (None, 1)
        or len(module.instructions) != 124
    ):
        raise ProjectionContractionHloError("optimized HLO module identity drifted")
    if tuple(module.collectives) or any(
        item.opcode in _FORBIDDEN_OPCODES for item in module.instructions
    ):
        raise ProjectionContractionHloError("projection HLO contains communication")
    lowered = optimized_hlo.lower()
    if any(token in lowered for token in _FORBIDDEN_TEXT):
        raise ProjectionContractionHloError("projection HLO contains a host effect")

    contraction = _instructions(module.instructions, _CONTRACTION)
    if set(contraction) != {
        "%param_1.57",
        "%dot_general.10",
        "%param_0.43",
        "%bitcast.11",
        "%multiply.12",
        "%constant.14.clone.3",
        "%reduce_sum.20",
        "%dot_general.9.clone.1",
        "%tuple.7",
    }:
        raise ProjectionContractionHloError("contraction instruction set drifted")
    _require(contraction, "%param_1.57", "parameter", ("1",), (_shape("f32", 6144),))
    _require(
        contraction,
        "%param_0.43",
        "parameter",
        ("0",),
        (_shape("f32", 1, 128, 6144),),
    )
    _require(
        contraction,
        "%dot_general.10",
        "broadcast",
        ("%param_1.57",),
        (_shape("f32", 128, 6144),),
    )
    _require(
        contraction,
        "%bitcast.11",
        "bitcast",
        ("%param_0.43",),
        (_shape("f32", 128, 6144),),
    )
    _require(
        contraction,
        "%multiply.12",
        "multiply",
        ("%dot_general.10", "%bitcast.11"),
        (_shape("f32", 128, 6144),),
    )
    reduction = _require(
        contraction,
        "%dot_general.9.clone.1",
        "reduce",
        ("%multiply.12", "%constant.14.clone.3"),
        (_shape("f32", 128),),
    )
    if "dimensions={1}" not in reduction.raw_line:
        raise ProjectionContractionHloError("contraction does not reduce width 6144")

    entry = _instructions(module.instructions, _ENTRY)
    expected_parameters = {
        "%param.5": ("bf16", (1, 1, 6144), "parameter(0)", "normalized_hidden_owner"),
        "%param.6": ("f32", (1, 128, 6144), "parameter(1)", "wk_weight_owner"),
        "%param.7": ("bf16", (1, 128), "parameter(2)", "key_norm_weight_owner"),
        "%param.8": ("bf16", (1, 128), "parameter(3)", "key_norm_bias_owner"),
    }
    for name, (dtype, dimensions, parameter, owner) in expected_parameters.items():
        instruction = entry.get(name)
        if (
            instruction is None
            or instruction.opcode != "parameter"
            or instruction.result_shapes != (_shape(dtype, *dimensions),)
            or parameter not in instruction.raw_line
            or owner not in instruction.raw_line
            or "sharding={devices=[2" not in instruction.raw_line
        ):
            raise ProjectionContractionHloError(f"owner input drifted: {name}")
    _require(
        entry,
        "%bitcast.20",
        "bitcast",
        ("%param.5",),
        (_shape("bf16", 1, 6144),),
    )
    _require(
        entry,
        "%convert.6",
        "convert",
        ("%bitcast.20",),
        (_shape("f32", 1, 6144),),
    )
    _require(
        entry,
        "%bitcast.21",
        "bitcast",
        ("%convert.6",),
        (_shape("f32", 6144),),
    )
    fusion = _require(
        entry,
        "%fusion",
        "fusion",
        ("%param.6", "%bitcast.21"),
        (_shape("f32"), _shape("f32", 128)),
    )
    if "calls=%fused_computation.1" not in fusion.raw_line:
        raise ProjectionContractionHloError("entry does not call contraction body")
    projected_path = (
        (
            "%get-tuple-element.4",
            "get-tuple-element",
            ("%fusion",),
            (_shape("f32", 128),),
            "index=1",
        ),
        (
            "%copy-start.1",
            "copy-start",
            ("%get-tuple-element.4",),
            (_shape("f32", 128), _shape("f32", 128), _shape("u32")),
            "copy-start",
        ),
        (
            "%copy-done.1",
            "copy-done",
            ("%copy-start.1",),
            (_shape("f32", 128),),
            "copy-done",
        ),
        (
            "%bitcast.22",
            "bitcast",
            ("%copy-done.1",),
            (_shape("f32", 1, 1, 128),),
            "bitcast",
        ),
    )
    for name, opcode, operands, results, raw_marker in projected_path:
        instruction = _require(entry, name, opcode, operands, results)
        if raw_marker not in instruction.raw_line:
            raise ProjectionContractionHloError(
                f"projected-key lineage drifted: {name}"
            )

    current_path = (
        (
            "%fusion.2",
            "fusion",
            (
                "%copy-done.2",
                "%copy-done.3",
                "%get-tuple-element.4",
                "%sqrt.4",
                "%div.23",
            ),
            (_shape("f32", 1, 128),),
        ),
        (
            "%fusion.5",
            "fusion",
            ("%fusion.2",),
            (_shape("f32", 1, 64), _shape("f32", 1, 64)),
        ),
        (
            "%get-tuple-element.7",
            "get-tuple-element",
            ("%fusion.5",),
            (_shape("f32", 1, 64),),
        ),
        ("%slice.13", "slice", ("%get-tuple-element.7",), (_shape("f32", 1, 32),)),
        ("%slice.14", "slice", ("%get-tuple-element.7",), (_shape("f32", 1, 32),)),
        (
            "%subtract_bitcast_fusion",
            "fusion",
            ("%slice.14", "%slice.13"),
            (_shape("f32", 1, 32, 1), _shape("f32", 1, 32, 1)),
        ),
        (
            "%get-tuple-element.5",
            "get-tuple-element",
            ("%subtract_bitcast_fusion",),
            (_shape("f32", 1, 32, 1),),
        ),
        (
            "%get-tuple-element.6",
            "get-tuple-element",
            ("%subtract_bitcast_fusion",),
            (_shape("f32", 1, 32, 1),),
        ),
        (
            "%pad_maximum_fusion",
            "fusion",
            ("%get-tuple-element.6", "%get-tuple-element.5"),
            (_shape("f32", 1, 32, 2),),
        ),
        ("%copy.9", "copy", ("%pad_maximum_fusion",), (_shape("f32", 1, 32, 2),)),
        ("%reshape.39", "reshape", ("%copy.9",), (_shape("f32", 1, 64),)),
        (
            "%get-tuple-element.8",
            "get-tuple-element",
            ("%fusion.5",),
            (_shape("f32", 1, 64),),
        ),
        (
            "%maximum_bitcast_fusion",
            "fusion",
            ("%get-tuple-element.8", "%reshape.39"),
            (_shape("f32", 1, 1, 128),),
        ),
    )
    for name, opcode, operands, results in current_path:
        _require(entry, name, opcode, operands, results)

    called_fusions = {
        "%fusion.2": "%fused_computation.6",
        "%fusion.5": "%fused_computation.13",
        "%subtract_bitcast_fusion": "%fused_computation.8",
        "%pad_maximum_fusion": "%fused_computation.3",
        "%maximum_bitcast_fusion": "%fused_computation.4",
    }
    if any(
        f"calls={body}," not in entry[name].raw_line
        for name, body in called_fusions.items()
    ):
        raise ProjectionContractionHloError("current-key fusion call target drifted")
    tuple_indices = {
        "%get-tuple-element.7": "index=0",
        "%get-tuple-element.8": "index=1",
        "%get-tuple-element.5": "index=0",
        "%get-tuple-element.6": "index=1",
    }
    if any(
        f", {marker}," not in entry[name].raw_line
        for name, marker in tuple_indices.items()
    ):
        raise ProjectionContractionHloError("current-key tuple index drifted")

    exact_consumers = {
        "%bitcast.20": {"%convert.6"},
        "%convert.6": {"%bitcast.21"},
        "%bitcast.21": {"%fusion"},
        "%fusion": {"%get-tuple-element.3", "%get-tuple-element.4"},
        "%copy-start.1": {"%copy-done.1"},
        "%copy-done.1": {"%bitcast.22"},
        "%bitcast.22": {"%tuple.11"},
        "%fusion.2": {"%fusion.5"},
        "%fusion.5": {"%get-tuple-element.7", "%get-tuple-element.8"},
        "%get-tuple-element.7": {"%slice.13", "%slice.14"},
        "%slice.13": {"%subtract_bitcast_fusion"},
        "%slice.14": {"%subtract_bitcast_fusion"},
        "%subtract_bitcast_fusion": {
            "%get-tuple-element.5",
            "%get-tuple-element.6",
        },
        "%get-tuple-element.5": {"%pad_maximum_fusion"},
        "%get-tuple-element.6": {"%pad_maximum_fusion"},
        "%pad_maximum_fusion": {"%copy.9"},
        "%copy.9": {"%reshape.39"},
        "%reshape.39": {"%maximum_bitcast_fusion"},
        "%get-tuple-element.8": {"%maximum_bitcast_fusion"},
        "%maximum_bitcast_fusion": {"%tuple.11"},
    }
    _require_exact_consumers(entry, exact_consumers, "rooted contraction")
    root = _require(
        entry,
        "%tuple.11",
        "tuple",
        ("%copy.11", "%bitcast.22", "%maximum_bitcast_fusion"),
        (
            _shape("bf16", 1, 1, 6144),
            _shape("f32", 1, 1, 128),
            _shape("f32", 1, 1, 128),
        ),
    )
    _require_root(root, "projection outputs")
    _audit_current_key_fusion_bodies(module.instructions)
    return {
        "collective_count": 0,
        "contraction_accumulation_dtype": "f32",
        "contraction_input_width": 6144,
        "entry_instruction_count": len(entry),
        "host_effect_count": 0,
        "instruction_count": len(module.instructions),
        "live_rows_per_owner": 1,
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "owner_count": 2,
        "output_shapes": [[1, 1, 6144], [1, 1, 128], [1, 1, 128]],
        "projection_width": 128,
    }


def audit_stablehlo_structure(stablehlo: str) -> dict[str, Any]:
    required = (
        "mhlo.num_partitions = 2",
        'sdy.mesh @mesh = <["feature"=2]>',
        "tensor<2x1x6144xbf16>",
        "tensor<2x128x6144xf32>",
        "tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
        "stablehlo.dot_general %4, %3, contracting_dims = [1] x [1]",
        "(tensor<1x6144xf32>, tensor<128x6144xf32>) -> tensor<1x128xf32>",
        "sdy.return %64, %65, %66",
    )
    lowered = stablehlo.lower()
    if any(item not in stablehlo for item in required) or any(
        token in lowered
        for token in (
            *_FORBIDDEN_TEXT,
            "stablehlo.all_",
            "stablehlo.collective_",
            "stablehlo.infeed",
            "stablehlo.outfeed",
        )
    ):
        raise ProjectionContractionHloError("StableHLO projection contract drifted")
    return {
        "collective_count": 0,
        "contraction_dtype": "f32",
        "contraction_width": 6144,
        "live_rows_per_owner": 1,
        "manual_axis": {"feature": 2},
        "owner_count": 2,
    }


def _read_regular(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise ProjectionContractionHloError(f"unsafe diagnostic file: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
        ):
            raise ProjectionContractionHloError(f"diagnostic file changed: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _audit_remote_replay(raw: bytes) -> dict[str, dict[str, Any]]:
    if sha256(raw).hexdigest() != EXPECTED_REMOTE_REPLAY_SHA256:
        raise ProjectionContractionHloError("remote replay artifact hash drifted")
    replay = json.loads(raw)
    objects = replay.get("objects")
    if (
        replay.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_hlo_independent_diagnostic_remote_replay"
        or replay.get("bucket")
        != {"location": "US-CENTRAL2", "name": "driftbench-dsv4-uc"}
        or replay.get("prefix") != EXPECTED_REMOTE_PREFIX
        or replay.get("run_tag") != EXPECTED_RUN_TAG
        or replay.get("claims")
        != {
            "cloud_mutation_performed": False,
            "diagnostic_terminal_last": True,
            "generation_downloads_verified": True,
            "read_only": True,
            "soft_deleted_query_exhaustive": True,
            "soft_deleted_query_matched_no_objects": True,
        }
        or replay.get("all_versions_catalogue")
        != {
            "live_generation_count": 15,
            "soft_deleted_generation_count": 0,
            "unique_path_count": 15,
        }
        or replay.get("soft_deleted_result")
        != {
            "exit_code": 1,
            "stderr": "ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n",
            "stdout_bytes": 0,
        }
        or not isinstance(objects, list)
        or len(objects) != 15
    ):
        raise ProjectionContractionHloError("remote replay claim boundary drifted")
    mapped = {item.get("path"): item for item in objects if isinstance(item, dict)}
    if len(mapped) != 15 or any(
        set(item) != {"crc32c", "generation", "path", "sha256", "size"}
        or not isinstance(item["generation"], str)
        or re.fullmatch(r"[0-9]+", item["generation"]) is None
        or not isinstance(item["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        or type(item["size"]) is not int
        or item["size"] < 0
        for item in mapped.values()
    ):
        raise ProjectionContractionHloError("remote replay object record drifted")
    terminal = mapped.get("diagnostic_objects.json")
    if (
        terminal is None
        or terminal["generation"] != EXPECTED_DIAGNOSTIC_LEDGER_GENERATION
        or int(terminal["generation"])
        != max(int(item["generation"]) for item in mapped.values())
    ):
        raise ProjectionContractionHloError("diagnostic terminal ordering drifted")
    return mapped


def adjudicate_diagnostic_run(run_dir: Path = EXPECTED_RUN_DIR) -> dict[str, Any]:
    if run_dir != EXPECTED_RUN_DIR:
        raise ProjectionContractionHloError("unexpected diagnostic run directory")
    optimized_raw = _read_regular(
        run_dir / "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt", 1 << 20
    )
    stable_raw = _read_regular(
        run_dir / "hlo/projection_contraction_pp16_stage0.stablehlo.mlir", 1 << 20
    )
    runner_raw = _read_regular(run_dir / "runner.json", 1 << 20)
    ledger_raw = _read_regular(run_dir / "diagnostic_objects.json", 1 << 20)
    receipt = json.loads(
        _read_regular(run_dir / "diagnostic_upload_receipt.json", 1 << 20)
    )
    failure_raw = _read_regular(EXPECTED_FAILURE_ARTIFACT, 1 << 20)
    remote_replay_raw = _read_regular(EXPECTED_REMOTE_REPLAY, 1 << 20)
    if (
        sha256(optimized_raw).hexdigest() != EXPECTED_OPTIMIZED_HLO_SHA256
        or sha256(stable_raw).hexdigest() != EXPECTED_STABLEHLO_SHA256
        or sha256(ledger_raw).hexdigest() != EXPECTED_DIAGNOSTIC_LEDGER_SHA256
        or sha256(failure_raw).hexdigest() != EXPECTED_FAILURE_ARTIFACT_SHA256
    ):
        raise ProjectionContractionHloError("diagnostic evidence hash drifted")
    runner = json.loads(runner_raw)
    ledger = json.loads(ledger_raw)
    remote_objects = _audit_remote_replay(remote_replay_raw)
    objects = {item["path"]: item for item in ledger.get("objects", [])}
    local_consumed = {
        "diagnostic_objects.json": ledger_raw,
        "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt": optimized_raw,
        "hlo/projection_contraction_pp16_stage0.stablehlo.mlir": stable_raw,
        "runner.json": runner_raw,
    }
    expected_input_spec = [
        {
            "dtype": "bfloat16",
            "name": "normalized_hidden_bf16",
            "shape": [2, 1, 6144],
        },
        {
            "dtype": "float32",
            "name": "wk_weight_fp32",
            "shape": [2, 128, 6144],
        },
        {
            "dtype": "bfloat16",
            "name": "key_norm_weight_bf16",
            "shape": [2, 128],
        },
        {
            "dtype": "bfloat16",
            "name": "key_norm_bias_bf16",
            "shape": [2, 128],
        },
    ]
    expected_output_spec = [
        {
            "dtype": "bfloat16",
            "name": "normalized_hidden_owners",
            "shape": [2, 1, 6144],
        },
        {
            "dtype": "float32",
            "name": "projected_key_owners",
            "shape": [2, 1, 128],
        },
        {
            "dtype": "float32",
            "name": "current_key_owners",
            "shape": [2, 1, 128],
        },
    ]
    if (
        runner.get("code_hash") != EXPECTED_CODE_HASH
        or runner.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or runner.get("compile_only") is not True
        or runner.get("compiled_executable_invocation_count") != 0
        or runner.get("tpu_numerical_execution_performed") is not False
        or runner.get("input_spec") != expected_input_spec
        or runner.get("output_spec") != expected_output_spec
        or runner.get("physical_group")
        != {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "local_device_count_visible": 4,
            "mesh_device_count": 2,
            "process_index": 0,
            "stage_id": 0,
        }
        or ledger.get("run_tag") != EXPECTED_RUN_TAG
        or objects.get(
            "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt", {}
        ).get("sha256")
        != EXPECTED_OPTIMIZED_HLO_SHA256
        or objects.get("hlo/projection_contraction_pp16_stage0.stablehlo.mlir", {}).get(
            "sha256"
        )
        != EXPECTED_STABLEHLO_SHA256
        or receipt.get("terminal", {}).get("generation")
        != EXPECTED_DIAGNOSTIC_LEDGER_GENERATION
        or receipt.get("terminal", {}).get("sha256")
        != EXPECTED_DIAGNOSTIC_LEDGER_SHA256
        or any(
            remote_objects.get(path, {}).get("sha256") != sha256(raw).hexdigest()
            or remote_objects.get(path, {}).get("size") != len(raw)
            for path, raw in local_consumed.items()
        )
        or set(objects) != set(remote_objects) - {"diagnostic_objects.json"}
        or any(objects[path] != remote_objects[path] for path in objects)
        or receipt.get("terminal")
        != {
            "crc32c": remote_objects["diagnostic_objects.json"]["crc32c"],
            "generation": remote_objects["diagnostic_objects.json"]["generation"],
            "path": "diagnostic_objects.json",
            "sha256": remote_objects["diagnostic_objects.json"]["sha256"],
            "size": remote_objects["diagnostic_objects.json"]["size"],
        }
        or Path(run_dir / "HLO_ACQUIRED").exists()
    ):
        raise ProjectionContractionHloError("diagnostic claim boundary drifted")
    optimized = audit_optimized_hlo_structure(optimized_raw.decode("ascii"))
    stable = audit_stablehlo_structure(stable_raw.decode("ascii"))
    if optimized["live_rows_per_owner"] != stable["live_rows_per_owner"]:
        raise ProjectionContractionHloError("HLO live-row contracts disagree")
    return {
        "artifact_kind": "gate_d_projection_contraction_pp16_diagnostic_hlo_adjudication",
        "authorization": {
            "full_8k": False,
            "numerical_execution": False,
            "performance_claim": False,
            "persistence_only": True,
        },
        "classification": (
            "DIAGNOSTIC_HLO_CAUSAL_STRUCTURE_ACCEPTED;"
            "PP16_OWNER_LOCALITY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;"
            "NO_HLO_ACQUIRED_TERMINAL;GATE_D_OPEN"
        ),
        "code_hash": EXPECTED_CODE_HASH,
        "diagnostic_ledger": {
            "generation": EXPECTED_DIAGNOSTIC_LEDGER_GENERATION,
            "sha256": EXPECTED_DIAGNOSTIC_LEDGER_SHA256,
        },
        "gate_d_closed": False,
        "optimized_hlo": {
            "sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "structure": optimized,
        },
        "run_tag": EXPECTED_RUN_TAG,
        "remote_replay": {
            "all_versions_live_generation_count": len(remote_objects),
            "generation_download_count": len(remote_objects),
            "sha256": EXPECTED_REMOTE_REPLAY_SHA256,
            "soft_deleted_generation_count": 0,
            "terminal_last": True,
        },
        "schema_version": 1,
        "stablehlo": {"sha256": EXPECTED_STABLEHLO_SHA256, "structure": stable},
    }
