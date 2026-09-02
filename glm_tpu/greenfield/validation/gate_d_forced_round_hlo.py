"""Fail-closed causal adjudication for the Gate-D forced-round PP16 HLO.

This module is deliberately offline: it imports no JAX and never lowers,
compiles, or invokes an executable.  The generic HLO surface linter is useful
for collective counts, but it cannot establish the operand lineage needed for
the Gate-D numerical challenger.  This validator binds the exact acquisition
and follows the rounded value through the optimized graph.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter
from collections.abc import Iterable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..sharding.hlo_contract import HloInstruction, HloShape, parse_hlo_module


class ForcedRoundHloAdjudicationError(RuntimeError):
    """Raised when acquired HLO or its append-only archive drifts."""


EXPECTED_CODE_HASH = "b8bdeb14c12a3742f3cc27ef91a4a2e406d412a5"
EXPECTED_RUN_TAG = "gate_d_forced_round_pp16_hlo_20260901T141137500138602Z"
EXPECTED_STABLEHLO_SHA256 = (
    "45eae705b60783bf8b65a1d3d209c79e8d43685cec0aa912b3141caf36fb19e1"
)
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "a0b87e2b43ba81bfe1549fbcc13434d2fbdca91b11c8b2f9a317ff9a9dbd9d45"
)
EXPECTED_REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
    "gate_d_forced_round_pp16_hlo/" + EXPECTED_RUN_TAG
)
EXPECTED_RUN_DIR = Path("/home/gianl/gate-d-runs") / EXPECTED_RUN_TAG
EXPECTED_REMOTE_REPLAY_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/artifacts/gate-d-forced-round-pp16-hlo-remote-replay.json"
)
EXPECTED_REMOTE_REPLAY_SHA256 = (
    "85bcd02c9b112ef3e21d065ebfae04f0d06e4a922f1d15c0f971fb669992e1d8"
)

_EXPECTED_PAYLOAD_OBJECTS = (
    (
        "census_post.txt",
        "1788272141214905",
        "sVYu6w==",
        "0ebc167eb854fed9a89bec35c351437d35d6c6a537857623bf3c05f777389fa3",
        653,
    ),
    (
        "census_pre.txt",
        "1788272141288061",
        "dRWkJg==",
        "6d7e87d498306c8f92ced3e6098f15da3021f71cfed304167b51f7a4cafe0322",
        653,
    ),
    (
        "dependencies.json",
        "1788272141366645",
        "vXoHtA==",
        "6530a9f625e29e89ce51d98fff6d1cd3c099cdd181af11d967244d860f7d7c1c",
        150096,
    ),
    (
        "evidence.json",
        "1788272141437437",
        "BvUoKw==",
        "8ecbe021d8233363948312368440de27f6b181a522743f034a266b8dfc97bdcd",
        2061,
    ),
    (
        "hlo/forced_round_pp16_stage0.optimized_hlo.txt",
        "1788272141517475",
        "wLqUVA==",
        EXPECTED_OPTIMIZED_HLO_SHA256,
        263868,
    ),
    (
        "hlo/forced_round_pp16_stage0.stablehlo.mlir",
        "1788272141591756",
        "puoOTw==",
        EXPECTED_STABLEHLO_SHA256,
        76529,
    ),
    (
        "mirror.sha256",
        "1788272141675968",
        "AmU8QA==",
        "bacbbda4465ef75c33c3d41fa4a09a9bc3a11fe0425401c629510f5404131e2a",
        2921,
    ),
    (
        "orchestrator.sealed.log",
        "1788272141756610",
        "EogDng==",
        "4fa0adaaf9ec6d65012783450c80b01fa95706ce0d560c03b0ff7c4f053ef6ea",
        645,
    ),
    (
        "publisher_runtime.json",
        "1788272141828963",
        "bk53fw==",
        "651ea242d98dda0c7941cda031a5d5a9c6c8f229e9bba415bdb17cfd1105efdd",
        878,
    ),
    (
        "remote_vacancy.raw.txt",
        "1788272141908990",
        "w8uC4g==",
        "06adceaeeeda14547ee896c26fe5d7a3f4ea81d2ff3348d36657ff9f9eb4f4da",
        346,
    ),
    (
        "remote_vacancy.txt",
        "1788272141994264",
        "+UtTww==",
        "d91aebebacffd492bee52552ee8d528f23e1c6b66293c2648795919886597673",
        451,
    ),
    (
        "runner.json",
        "1788272142084383",
        "y/0l+A==",
        "45a17944257b690291505095c667866ca887594540e545f7da81a72b8c7f9bc7",
        10174,
    ),
    (
        "runner.log",
        "1788272142169501",
        "amBVOg==",
        "6c1fc992e948cc59d2d540c36e018f4ded866f0d93f2e0320db354f8a54d92b1",
        156,
    ),
    (
        "summary.json",
        "1788272142256485",
        "r6G+ZA==",
        "bb39ba8b5aa767498e21e85bb81f806235579c8aa846321e5b5b44b434597f3c",
        920,
    ),
    (
        "sync.txt",
        "1788272142335996",
        "DYYcHA==",
        "06ade8c89c893147cb30c083066ae53f7f17e78469a79b001494719d5117edff",
        112,
    ),
)
_EXPECTED_LEDGER_OBJECT = (
    "remote_objects.json",
    "1788272142420384",
    "VNZ9OA==",
    "5b9f571d4c6c57ac3da2590014839e2e958d833eb52abf8e365021080790f402",
    2706,
)
_EXPECTED_TERMINAL_OBJECT = (
    "HLO_ACQUIRED",
    "1788272143585315",
    "z/dysw==",
    "aabab7d562d33893d89aea6e3f71ab93f59c0dbb36dc2b1df20a77869ba31f57",
    744,
)
_EXPECTED_REMOTE_OBJECTS = (
    *_EXPECTED_PAYLOAD_OBJECTS,
    _EXPECTED_LEDGER_OBJECT,
    _EXPECTED_TERMINAL_OBJECT,
)
_RUN_EVIDENCE_PATHS = tuple(item[0] for item in _EXPECTED_PAYLOAD_OBJECTS) + (
    "remote_objects.json",
    "HLO_ACQUIRED",
    "terminal_upload_receipt.json",
)

_BOUNDARY_COMPUTATION = (
    "%fused_computation.40 (param_0.98: f32[1,6144], param_1.249: f32[], "
    "param_2.205: bf16[1,6144], param_3.107: bf16[1,6144]) -> bf16[1,6144]"
)
_REDUCTION_COMPUTATION = (
    "%fused_computation.41 (param_0.279: bf16[1,6144], "
    "param_1.373: bf16[1,6144]) -> f32[]"
)
_QKV_BODY_COMPUTATION = (
    "%wide.region_1.2_spmd.sunk (wide.param.0: (s32[], bf16[32,1,82], "
    "u8[32,6144,82], f32[32,48,82], bf16[1,6144], /*index=5*/s32[], "
    "f32[])) -> (s32[], bf16[32,1,82], u8[32,6144,82], "
    "f32[32,48,82], bf16[1,6144], /*index=5*/s32[], f32[])"
)
_ENTRY_COMPUTATION = (
    "ENTRY %main.27_spmd (param.22: bf16[1,6144], param.23: bf16[1,6144], "
    "param.24: bf16[6144], param.25: bf16[1,16,256,128], "
    "param.26: u8[32,6144,82], param.27: f32[32,48,82], "
    "param.28: bf16[2048], param.29: f32[1,2048,2048], "
    "param.30: f32[1,128,6144], param.31: bf16[1,128], "
    "param.32: bf16[1,128], param.33: bf16[1,16,6144]) -> "
    "(bf16[1,1,6144], f32[1,1,32,128], f32[1,1,32], f32[1,1,128], "
    "bf16[1,16,256,128], /*index=5*/s32[1,1,2048], s32[1,1], "
    "f32[1,1,2048], u8[1])"
)

_FORBIDDEN_TEXT = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)
_FORBIDDEN_OPCODES = {
    "all-reduce",
    "all-to-all",
    "collective-broadcast",
    "collective-permute",
    "infeed",
    "outfeed",
    "recv",
    "recv-done",
    "reduce-scatter",
    "send",
    "send-done",
}


def _shape(dtype: str, *dimensions: int) -> HloShape:
    return HloShape(dtype, tuple(dimensions))


def _instructions(
    all_instructions: Iterable[HloInstruction], computation: str
) -> dict[str, HloInstruction]:
    selected = [item for item in all_instructions if item.computation == computation]
    result = {item.name: item for item in selected}
    if len(result) != len(selected):
        raise ForcedRoundHloAdjudicationError(
            f"duplicate HLO instruction names in {computation}"
        )
    return result


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
        raise ForcedRoundHloAdjudicationError(f"forced-round HLO edge drifted: {name}")
    return instruction


def _require_raw(instruction: HloInstruction, *needles: str) -> None:
    if any(needle not in instruction.raw_line for needle in needles):
        raise ForcedRoundHloAdjudicationError(
            f"forced-round HLO attributes drifted: {instruction.name}"
        )


def _consumers(instructions: dict[str, HloInstruction], name: str) -> set[str]:
    return {
        instruction.name
        for instruction in instructions.values()
        if name in instruction.operand_names
    }


def _require_exact_names(
    instructions: dict[str, HloInstruction], expected: set[str], label: str
) -> None:
    if set(instructions) != expected:
        raise ForcedRoundHloAdjudicationError(f"{label} instruction set drifted")


def audit_optimized_hlo_structure(optimized_hlo: str) -> dict[str, Any]:
    """Prove the optimized causal boundary, consumers, roots, and locality."""

    try:
        module = parse_hlo_module(optimized_hlo)
    except (TypeError, ValueError) as error:
        raise ForcedRoundHloAdjudicationError(
            "optimized HLO is not parseable"
        ) from error
    if (
        module.name != "jit_mapped"
        or module.num_partitions != 2
        or module.num_replicas not in (None, 1)
        or len(module.instructions) != 1114
    ):
        raise ForcedRoundHloAdjudicationError("optimized HLO module identity drifted")

    computations = {item.computation for item in module.instructions}
    required_computations = {
        _BOUNDARY_COMPUTATION,
        _REDUCTION_COMPUTATION,
        _QKV_BODY_COMPUTATION,
        _ENTRY_COMPUTATION,
    }
    if not required_computations.issubset(computations):
        raise ForcedRoundHloAdjudicationError(
            "optimized HLO required computation signature drifted"
        )

    boundary = _instructions(module.instructions, _BOUNDARY_COMPUTATION)
    _require_exact_names(
        boundary,
        {
            "%param_3.107",
            "%convert_element_type.96",
            "%param_2.205",
            "%convert_element_type.95",
            "%add.198",
            "%param_1.249",
            "%mul.175",
            "%mul.173",
            "%reduce_precision.4",
            "%param_0.98",
            "%mul.172",
            "%convert_element_type.87",
        },
        "forced-round boundary",
    )
    _require(boundary, "%param_3.107", "parameter", ("3",), (_shape("bf16", 1, 6144),))
    _require(boundary, "%param_2.205", "parameter", ("2",), (_shape("bf16", 1, 6144),))
    _require(boundary, "%param_1.249", "parameter", ("1",), (_shape("f32"),))
    _require(boundary, "%param_0.98", "parameter", ("0",), (_shape("f32", 1, 6144),))
    _require(
        boundary,
        "%convert_element_type.96",
        "convert",
        ("%param_3.107",),
        (_shape("f32", 1, 6144),),
    )
    _require(
        boundary,
        "%convert_element_type.95",
        "convert",
        ("%param_2.205",),
        (_shape("f32", 1, 6144),),
    )
    _require(
        boundary,
        "%add.198",
        "add",
        ("%convert_element_type.96", "%convert_element_type.95"),
        (_shape("f32", 1, 6144),),
    )
    _require(
        boundary, "%mul.175", "broadcast", ("%param_1.249",), (_shape("f32", 1, 6144),)
    )
    _require(
        boundary,
        "%mul.173",
        "multiply",
        ("%add.198", "%mul.175"),
        (_shape("f32", 1, 6144),),
    )
    rounded = _require(
        boundary,
        "%reduce_precision.4",
        "reduce-precision",
        ("%mul.173",),
        (_shape("f32", 1, 6144),),
    )
    _require_raw(rounded, "exponent_bits=8", "mantissa_bits=7")
    _require(
        boundary,
        "%mul.172",
        "multiply",
        ("%reduce_precision.4", "%param_0.98"),
        (_shape("f32", 1, 6144),),
    )
    root_boundary = _require(
        boundary,
        "%convert_element_type.87",
        "convert",
        ("%mul.172",),
        (_shape("bf16", 1, 6144),),
    )
    if not root_boundary.raw_line.startswith("ROOT "):
        raise ForcedRoundHloAdjudicationError("forced-round boundary is not rooted")

    all_reduce_precision = [
        item for item in module.instructions if item.opcode == "reduce-precision"
    ]
    if all_reduce_precision != [rounded]:
        raise ForcedRoundHloAdjudicationError(
            "optimized HLO has a missing or competing precision boundary"
        )

    reduction = _instructions(module.instructions, _REDUCTION_COMPUTATION)
    _require_exact_names(
        reduction,
        {
            "%param_1.373",
            "%convert_element_type.100",
            "%param_0.279",
            "%convert_element_type.99",
            "%add.200",
            "%square.13",
            "%constant.263.clone.8",
            "%reduce_sum.45",
        },
        "RMS reduction",
    )
    _require(reduction, "%param_1.373", "parameter", ("1",), (_shape("bf16", 1, 6144),))
    _require(reduction, "%param_0.279", "parameter", ("0",), (_shape("bf16", 1, 6144),))
    _require(
        reduction,
        "%convert_element_type.100",
        "convert",
        ("%param_1.373",),
        (_shape("f32", 1, 6144),),
    )
    _require(
        reduction,
        "%convert_element_type.99",
        "convert",
        ("%param_0.279",),
        (_shape("f32", 1, 6144),),
    )
    _require(
        reduction,
        "%add.200",
        "add",
        ("%convert_element_type.100", "%convert_element_type.99"),
        (_shape("f32", 1, 6144),),
    )
    _require(
        reduction,
        "%square.13",
        "multiply",
        ("%add.200", "%add.200"),
        (_shape("f32", 1, 6144),),
    )
    _require(reduction, "%constant.263.clone.8", "constant", ("0",), (_shape("f32"),))
    reduction_root = _require(
        reduction,
        "%reduce_sum.45",
        "reduce",
        ("%square.13", "%constant.263.clone.8"),
        (_shape("f32"),),
    )
    _require_raw(reduction_root, "dimensions={0,1}", "to_apply=%region_0.0")
    if not reduction_root.raw_line.startswith("ROOT "):
        raise ForcedRoundHloAdjudicationError("RMS reduction is not rooted")

    entry = _instructions(module.instructions, _ENTRY_COMPUTATION)
    if len(entry) != 236:
        raise ForcedRoundHloAdjudicationError("entry instruction count drifted")
    _require(entry, "%param.22", "parameter", ("0",), (_shape("bf16", 1, 6144),))
    _require(entry, "%param.23", "parameter", ("1",), (_shape("bf16", 1, 6144),))
    _require(entry, "%param.24", "parameter", ("2",), (_shape("bf16", 6144),))
    weight = _require(
        entry,
        "%broadcast_in_dim.198",
        "convert",
        ("%param.24",),
        (_shape("f32", 6144),),
    )
    _require_raw(weight, 'op_name="jit(mapped)/shard_map/broadcast_in_dim"')
    _require(
        entry,
        "%bitcast.135",
        "bitcast",
        ("%broadcast_in_dim.198",),
        (_shape("f32", 1, 6144),),
    )
    reduction_call = _require(
        entry,
        "%multiply_reduce_fusion.1",
        "fusion",
        ("%param.23", "%param.22"),
        (_shape("f32"),),
    )
    _require_raw(reduction_call, "calls=%fused_computation.41")
    _require(
        entry,
        "%constant.264.clone.1",
        "constant",
        ("0.000162760422",),
        (_shape("f32"),),
    )
    _require(entry, "%constant.265.clone.1", "constant", ("1e-05",), (_shape("f32"),))
    _require(
        entry,
        "%div.47",
        "multiply",
        ("%multiply_reduce_fusion.1", "%constant.264.clone.1"),
        (_shape("f32"),),
    )
    _require(
        entry, "%add.144", "add", ("%div.47", "%constant.265.clone.1"), (_shape("f32"),)
    )
    _require(entry, "%rsqrt.8", "rsqrt", ("%add.144",), (_shape("f32"),))
    boundary_call = _require(
        entry,
        "%multiply_convert_fusion",
        "fusion",
        ("%bitcast.135", "%rsqrt.8", "%param.23", "%param.22"),
        (_shape("bf16", 1, 6144),),
    )
    _require_raw(boundary_call, "calls=%fused_computation.40")

    if _consumers(entry, "%param.22") != {
        "%multiply_reduce_fusion.1",
        "%multiply_convert_fusion",
    } or _consumers(entry, "%param.23") != {
        "%multiply_reduce_fusion.1",
        "%multiply_convert_fusion",
    }:
        raise ForcedRoundHloAdjudicationError(
            "residual inputs have a competing unrounded primary path"
        )
    if _consumers(entry, "%multiply_convert_fusion") != {"%convert.21", "%tuple.72"}:
        raise ForcedRoundHloAdjudicationError("rounded boundary consumer set drifted")
    _require(
        entry,
        "%convert.21",
        "convert",
        ("%multiply_convert_fusion",),
        (_shape("f32", 1, 6144),),
    )
    _require(entry, "%bitcast.132", "bitcast", ("%convert.21",), (_shape("f32", 6144),))
    _require(entry, "%bitcast.133", "bitcast", ("%convert.21",), (_shape("f32", 6144),))
    if _consumers(entry, "%convert.21") != {"%bitcast.132", "%bitcast.133"}:
        raise ForcedRoundHloAdjudicationError("rounded DSA consumer fanout drifted")
    _require(
        entry,
        "%fusion.12",
        "fusion",
        ("%param.33", "%bitcast.132"),
        (_shape("f32", 16),),
    )
    _require_raw(entry["%fusion.12"], 'op_name="jit(mapped)/shard_map/dot_general"')
    _require(
        entry,
        "%fusion.8",
        "fusion",
        ("%copy-done.1", "%bitcast.133"),
        (_shape("f32"), _shape("f32", 128)),
    )
    _require_raw(entry["%fusion.8"], 'op_name="jit(mapped)/shard_map/reduce_sum"')

    qkv_tuple = _require(
        entry,
        "%tuple.72",
        "tuple",
        (
            "%copy.55",
            "%custom-call.14",
            "%copy.40",
            "%param.27",
            "%multiply_convert_fusion",
            "%constant.281.clone.1",
            "%constant.493",
        ),
        (
            _shape("s32"),
            _shape("bf16", 32, 1, 82),
            _shape("u8", 32, 6144, 82),
            _shape("f32", 32, 48, 82),
            _shape("bf16", 1, 6144),
            _shape("s32"),
            _shape("f32"),
        ),
    )
    if qkv_tuple.operand_names[4] != "%multiply_convert_fusion":
        raise ForcedRoundHloAdjudicationError("QKV tuple index-4 boundary drifted")
    qkv_while = _require(
        entry, "%while.21", "while", ("%tuple.72",), qkv_tuple.result_shapes
    )
    _require_raw(
        qkv_while, "condition=%wide.region_2.3_spmd", "body=%wide.region_1.2_spmd.sunk"
    )

    qkv_body = _instructions(module.instructions, _QKV_BODY_COMPUTATION)
    carried = _require(
        qkv_body,
        "%get-tuple-element.303",
        "get-tuple-element",
        ("%wide.param.0",),
        (_shape("bf16", 1, 6144),),
    )
    _require_raw(carried, "index=4")
    qkv_compute = qkv_body.get("%constant_dynamic-update-slice_fusion.2")
    body_root = qkv_body.get("%tuple.74")
    if (
        qkv_compute is None
        or qkv_compute.opcode != "fusion"
        or "%get-tuple-element.303" not in qkv_compute.operand_names
        or "one_row_fused_qkv_a_n82_convolution" not in qkv_compute.raw_line
        or body_root is None
        or body_root.opcode != "tuple"
        or not body_root.raw_line.startswith("ROOT ")
        or len(body_root.operand_names) != 7
        or body_root.operand_names[4] != "%get-tuple-element.303"
    ):
        raise ForcedRoundHloAdjudicationError(
            "QKV while index-4 preservation or consumption drifted"
        )

    extracted = _require(
        entry,
        "%get-tuple-element.351",
        "get-tuple-element",
        ("%while.21",),
        (_shape("bf16", 1, 6144),),
    )
    _require_raw(extracted, "index=4")
    _require(
        entry,
        "%copy-start.2",
        "copy-start",
        ("%get-tuple-element.351",),
        (_shape("bf16", 1, 6144), _shape("bf16", 1, 6144), _shape("u32")),
    )
    _require(
        entry,
        "%copy-done.2",
        "copy-done",
        ("%copy-start.2",),
        (_shape("bf16", 1, 6144),),
    )
    _require(
        entry,
        "%bitcast.130",
        "bitcast",
        ("%copy-done.2",),
        (_shape("bf16", 1, 1, 6144),),
    )
    root = _require(
        entry,
        "%tuple.79",
        "tuple",
        (
            "%bitcast.130",
            "%bitcast.139",
            "%broadcast_in_dim_reshape_transpose.7",
            "%fusion.25",
            "%bitcast.128",
            "%bitcast.143",
            "%copy.72",
            "%select_bitcast_fusion.1",
            "%bitcast.48",
        ),
        (
            _shape("bf16", 1, 1, 6144),
            _shape("f32", 1, 1, 32, 128),
            _shape("f32", 1, 1, 32),
            _shape("f32", 1, 1, 128),
            _shape("bf16", 1, 16, 256, 128),
            _shape("s32", 1, 1, 2048),
            _shape("s32", 1, 1),
            _shape("f32", 1, 1, 2048),
            _shape("u8", 1),
        ),
    )
    if not root.raw_line.startswith("ROOT "):
        raise ForcedRoundHloAdjudicationError("entry output tuple is not rooted")

    collectives = tuple(module.collectives)
    expected_collectives = (
        ("%all-gather.3", 2, _shape("f32", 2, 1, 2064)),
        ("%all-gather.4", 3, _shape("s32", 2, 1, 2048)),
        ("%all-gather.5", 4, _shape("f32", 2, 1, 2048)),
    )
    if len(collectives) != len(expected_collectives):
        raise ForcedRoundHloAdjudicationError("collective count drifted")
    for instruction, (name, channel, result_shape) in zip(
        collectives, expected_collectives, strict=True
    ):
        if (
            instruction.name != name
            or instruction.opcode != "all-gather"
            or instruction.channel_id != channel
            or instruction.replica_groups != ((0, 1),)
            or instruction.result_shapes != (result_shape,)
            or not instruction.use_global_device_ids
        ):
            raise ForcedRoundHloAdjudicationError(
                f"PP16 local collective drifted: {instruction.name}"
            )
    if any(item.opcode in _FORBIDDEN_OPCODES for item in module.instructions):
        raise ForcedRoundHloAdjudicationError("forbidden optimized-HLO operation found")
    lowered = optimized_hlo.lower()
    if any(token in lowered for token in _FORBIDDEN_TEXT):
        raise ForcedRoundHloAdjudicationError("host-staged optimized-HLO token found")

    custom_targets = Counter(
        match.group(1)
        for item in module.instructions
        if item.opcode == "custom-call"
        for match in [re.search(r'custom_call_target="([^"]+)"', item.raw_line)]
        if match is not None
    )
    if custom_targets != Counter(
        {"AssumeGatherIndicesInBound": 7, "AllocateBuffer": 2, "ConcatBitcast": 1}
    ):
        raise ForcedRoundHloAdjudicationError("custom-call target multiset drifted")

    return {
        "boundary_computation": _BOUNDARY_COMPUTATION.split(" (")[0],
        "boundary_output_shape": [1, 6144],
        "boundary_output_dtype": "bf16",
        "collective_channels": [2, 3, 4],
        "collective_count": 3,
        "collective_groups": [[0, 1]],
        "custom_call_targets": dict(sorted(custom_targets.items())),
        "dsa_consumers": ["head_weight_projection", "current_key_projection"],
        "entry_instruction_count": len(entry),
        "entry_output_count": len(root.result_shapes),
        "instruction_count": len(module.instructions),
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "qkv_loop_carried_index": 4,
        "reduce_precision": {"count": 1, "exponent_bits": 8, "mantissa_bits": 7},
        "rooted_primary_output_index": 0,
        "unrounded_primary_bypass_count": 0,
    }


def audit_stablehlo_structure(stablehlo: str) -> dict[str, Any]:
    """Prove the pre-optimization one-row forced boundary and local groups."""

    required = (
        "module @jit_mapped attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32}",
        'sdy.mesh @mesh = <["feature"=2]>',
        "%5 = stablehlo.convert %arg14 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
        "%6 = stablehlo.convert %arg15 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
        "%7 = stablehlo.add %5, %6 : tensor<1x6144xf32>",
        "%8 = chlo.square %7 : tensor<1x6144xf32> -> tensor<1x6144xf32>",
        "%9 = stablehlo.reduce(%8 init: %cst) applies stablehlo.add across dimensions = [1]",
        "%12 = stablehlo.divide %10, %11 : tensor<1x1xf32>",
        "%14 = stablehlo.add %12, %13 : tensor<1x1xf32>",
        "%15 = stablehlo.rsqrt %14 : tensor<1x1xf32>",
        "%17 = stablehlo.multiply %7, %16 : tensor<1x6144xf32>",
        "%18 = stablehlo.reduce_precision %17, format = e8m7 : tensor<1x6144xf32>",
        "%19 = stablehlo.convert %arg16 : (tensor<6144xbf16>) -> tensor<6144xf32>",
        "%21 = stablehlo.multiply %18, %20 : tensor<1x6144xf32>",
        "%22 = stablehlo.convert %21 : (tensor<1x6144xf32>) -> tensor<1x6144xbf16>",
        "%iterArg_97 = %22",
        "%114 = stablehlo.optimization_barrier %22 : tensor<1x6144xbf16>",
        "%121 = stablehlo.convert %114 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
        "%174 = stablehlo.convert %114 : (tensor<1x6144xbf16>) -> tensor<1x6144xf32>",
        "%346 = stablehlo.broadcast_in_dim %22, dims = [1, 2] : (tensor<1x6144xbf16>) -> tensor<1x1x6144xbf16>",
        "sdy.return %346, %347, %348, %349, %350, %351, %352, %353, %354",
    )
    if any(item not in stablehlo for item in required):
        raise ForcedRoundHloAdjudicationError("StableHLO causal lineage drifted")
    if stablehlo.count("stablehlo.reduce_precision") != 1:
        raise ForcedRoundHloAdjudicationError("StableHLO precision edge count drifted")
    stable_collectives = re.findall(
        r'"stablehlo\.(all_gather|all_reduce|all_to_all|collective_permute|reduce_scatter)"[^\n]*',
        stablehlo,
    )
    if stable_collectives != ["all_gather", "all_gather", "all_gather"]:
        raise ForcedRoundHloAdjudicationError("StableHLO collective set drifted")
    all_gather_lines = [
        line for line in stablehlo.splitlines() if '"stablehlo.all_gather"' in line
    ]
    if len(all_gather_lines) != 3 or any(
        "replica_groups = dense<[[0, 1]]>" not in line
        or "use_global_device_ids" not in line
        for line in all_gather_lines
    ):
        raise ForcedRoundHloAdjudicationError("StableHLO collective locality drifted")
    lowered = stablehlo.lower()
    if any(token in lowered for token in _FORBIDDEN_TEXT):
        raise ForcedRoundHloAdjudicationError("host-staged StableHLO token found")
    return {
        "collective_count": 3,
        "collective_groups": [[0, 1]],
        "input_count_after_dce": 12,
        "manual_axis": "feature",
        "mesh_size": 2,
        "output_count": 9,
        "reduce_precision": {"count": 1, "format": "e8m7"},
    }


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForcedRoundHloAdjudicationError(
            f"invalid JSON evidence: {label}"
        ) from error
    if not isinstance(value, dict):
        raise ForcedRoundHloAdjudicationError(
            f"JSON evidence is not an object: {label}"
        )
    return value


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_absolute_directory_nofollow(path: Path) -> int:
    if not path.is_absolute() or any(
        part in ("", ".", "..") for part in path.parts[1:]
    ):
        raise ForcedRoundHloAdjudicationError("unsafe absolute evidence directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        final = os.fstat(descriptor)
        if not stat.S_ISDIR(final.st_mode):
            raise ForcedRoundHloAdjudicationError("evidence directory is not regular")
        return descriptor
    except OSError as error:
        os.close(descriptor)
        raise ForcedRoundHloAdjudicationError(
            f"cannot open no-follow evidence directory: {path}"
        ) from error
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_at(directory_fd: int, relative: str) -> bytes:
    parts = Path(relative).parts
    if (
        not parts
        or relative.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ForcedRoundHloAdjudicationError("unsafe evidence-relative path")
    current = os.dup(directory_fd)
    file_fd: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        for component in parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=current,
        )
        before = os.fstat(file_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise ForcedRoundHloAdjudicationError(
                f"evidence file identity is unsafe: {relative}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(file_fd, min(remaining, 1024 * 1024))
            if not chunk:
                raise ForcedRoundHloAdjudicationError(
                    f"evidence file truncated during read: {relative}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            raise ForcedRoundHloAdjudicationError(
                f"evidence file grew during read: {relative}"
            )
        after = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(after):
            raise ForcedRoundHloAdjudicationError(
                f"evidence file changed during read: {relative}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise ForcedRoundHloAdjudicationError(
            f"cannot read no-follow evidence file: {relative}"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(current)


def _read_absolute_regular_nofollow(path: Path) -> bytes:
    parent_fd = _open_absolute_directory_nofollow(path.parent)
    try:
        return _read_regular_at(parent_fd, path.name)
    finally:
        os.close(parent_fd)


def _read_canonical_run(run_dir: Path) -> dict[str, bytes]:
    if str(run_dir) != str(EXPECTED_RUN_DIR):
        raise ForcedRoundHloAdjudicationError("acquired run canonical path drifted")
    run_fd = _open_absolute_directory_nofollow(EXPECTED_RUN_DIR)
    try:
        run_stat = os.fstat(run_fd)
        if run_stat.st_uid != os.getuid() or run_stat.st_mode & 0o077:
            raise ForcedRoundHloAdjudicationError(
                "acquired run directory authority drifted"
            )
        return {path: _read_regular_at(run_fd, path) for path in _RUN_EVIDENCE_PATHS}
    finally:
        os.close(run_fd)


def _remote_tuple(record: object) -> tuple[str, str, str, str, int]:
    if not isinstance(record, dict) or set(record) != {
        "crc32c",
        "generation",
        "path",
        "sha256",
        "size",
    }:
        raise ForcedRoundHloAdjudicationError("remote object schema drifted")
    path = record["path"]
    generation = record["generation"]
    crc32c = record["crc32c"]
    digest = record["sha256"]
    size = record["size"]
    if (
        not isinstance(path, str)
        or not isinstance(generation, str)
        or re.fullmatch(r"[1-9][0-9]*", generation) is None
        or not isinstance(crc32c, str)
        or not crc32c
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or type(size) is not int
        or size < 0
    ):
        raise ForcedRoundHloAdjudicationError("remote object field type drifted")
    return path, generation, crc32c, digest, size


def _audit_remote_replay(raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_REMOTE_REPLAY_SHA256:
        raise ForcedRoundHloAdjudicationError("independent remote replay bytes drifted")
    replay = _load_json_bytes(raw, "independent remote replay")
    objects = replay.get("objects")
    if (
        not isinstance(objects, list)
        or tuple(map(_remote_tuple, objects)) != _EXPECTED_REMOTE_OBJECTS
    ):
        raise ForcedRoundHloAdjudicationError(
            "independent remote replay catalogue drifted"
        )
    if (
        replay.get("artifact_kind")
        != "gate_d_forced_round_pp16_hlo_independent_remote_replay"
        or replay.get("schema_version") != 1
        or type(replay.get("schema_version")) is not int
        or replay.get("captured_at_utc") != "2026-09-01T15:00:01.666455978Z"
        or replay.get("gcloud_version") != "582.0.0"
        or replay.get("run_tag") != EXPECTED_RUN_TAG
        or replay.get("prefix") != EXPECTED_REMOTE_PREFIX
        or replay.get("bucket")
        != {"location": "US-CENTRAL2", "name": "driftbench-dsv4-uc"}
        or replay.get("all_versions_catalogue")
        != {
            "live_generation_count": 17,
            "soft_deleted_generation_count": 0,
            "unique_path_count": 17,
        }
        or replay.get("claims")
        != {
            "cloud_mutation_performed": False,
            "generation_downloads_verified": True,
            "read_only": True,
            "soft_deleted_query_exhaustive": True,
            "soft_deleted_query_matched_no_objects": True,
        }
        or replay.get("queries")
        != {
            "all_versions": "gcloud storage ls --all-versions --json PREFIX/**",
            "bucket": "gcloud storage buckets describe gs://driftbench-dsv4-uc",
            "generation_replay": (
                "gcloud storage cp PREFIX/PATH#GENERATION LOCAL_FILE"
            ),
            "soft_deleted": (
                "PYTHONWARNINGS=ignore gcloud storage ls --soft-deleted "
                "--exhaustive --json PREFIX/**"
            ),
        }
        or replay.get("soft_deleted_result")
        != {
            "exit_code": 1,
            "stderr": (
                "ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n"
            ),
            "stdout_bytes": 0,
        }
    ):
        raise ForcedRoundHloAdjudicationError("independent remote replay claim drifted")
    return {
        "all_versions_live_generation_count": 17,
        "bucket_location": "US-CENTRAL2",
        "generation_download_count": 17,
        "sha256": EXPECTED_REMOTE_REPLAY_SHA256,
        "soft_deleted_generation_count": 0,
    }


def _adjudicate_acquired_files(
    evidence: Mapping[str, bytes], remote_replay_raw: bytes
) -> dict[str, Any]:
    if set(evidence) != set(_RUN_EVIDENCE_PATHS) or any(
        not isinstance(value, bytes) for value in evidence.values()
    ):
        raise ForcedRoundHloAdjudicationError("acquired run file catalogue drifted")
    remote_replay = _audit_remote_replay(remote_replay_raw)
    stable_raw = evidence["hlo/forced_round_pp16_stage0.stablehlo.mlir"]
    optimized_raw = evidence["hlo/forced_round_pp16_stage0.optimized_hlo.txt"]
    if (
        sha256(stable_raw).hexdigest() != EXPECTED_STABLEHLO_SHA256
        or sha256(optimized_raw).hexdigest() != EXPECTED_OPTIMIZED_HLO_SHA256
    ):
        raise ForcedRoundHloAdjudicationError("acquired HLO byte identity drifted")
    try:
        stable_text = stable_raw.decode("utf-8", errors="strict")
        optimized_text = optimized_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ForcedRoundHloAdjudicationError("acquired HLO is not UTF-8") from error

    runner = _load_json_bytes(evidence["runner.json"], "runner.json")
    summary = _load_json_bytes(evidence["summary.json"], "summary.json")
    marker = _load_json_bytes(evidence["HLO_ACQUIRED"], "HLO_ACQUIRED")
    ledger = _load_json_bytes(evidence["remote_objects.json"], "remote_objects.json")
    receipt = _load_json_bytes(
        evidence["terminal_upload_receipt.json"], "terminal_upload_receipt.json"
    )
    local_evidence = _load_json_bytes(evidence["evidence.json"], "evidence.json")
    expected_false = (
        runner.get("compiled_executable_invocation_count") == 0
        and runner.get("compile_only") is True
        and runner.get("gate_d_closed") is False
        and runner.get("numerical_claim") is False
        and runner.get("performance_claim") is False
        and runner.get("tpu_numerical_execution_performed") is False
        and summary.get("status") == "HLO_ACQUIRED_UNADJUDICATED"
        and marker.get("adjudicated") is False
    )
    if not expected_false:
        raise ForcedRoundHloAdjudicationError("acquisition claim boundary drifted")
    if (
        runner.get("code_hash") != EXPECTED_CODE_HASH
        or summary.get("code_hash") != EXPECTED_CODE_HASH
    ):
        raise ForcedRoundHloAdjudicationError("acquisition code pin drifted")
    if (
        runner.get("hlo", {}).get("stablehlo", {}).get("sha256")
        != EXPECTED_STABLEHLO_SHA256
        or runner.get("hlo", {}).get("optimized", {}).get("sha256")
        != EXPECTED_OPTIMIZED_HLO_SHA256
    ):
        raise ForcedRoundHloAdjudicationError("runner HLO hashes drifted")
    if (
        len(runner.get("input_spec", [])) != 16
        or len(runner.get("output_spec", [])) != 9
    ):
        raise ForcedRoundHloAdjudicationError("abstract 16-to-9 signature drifted")
    physical = runner.get("physical_group", {})
    if (
        physical.get("device_ids") != [0, 1]
        or physical.get("coordinates") != [[0, 0, 0], [1, 0, 0]]
        or physical.get("mesh_device_count") != 2
        or physical.get("stage_id") != 0
    ):
        raise ForcedRoundHloAdjudicationError("physical PP16 group drifted")
    memory = runner.get("memory_analysis", {})
    if any(
        memory.get(name) != 0
        for name in (
            "host_alias_size_in_bytes",
            "host_argument_size_in_bytes",
            "host_generated_code_size_in_bytes",
            "host_output_size_in_bytes",
            "host_temp_size_in_bytes",
        )
    ):
        raise ForcedRoundHloAdjudicationError("host memory materialization is nonzero")

    objects = ledger.get("objects")
    if (
        ledger.get("run_tag") != EXPECTED_RUN_TAG
        or not isinstance(objects, list)
        or tuple(map(_remote_tuple, objects)) != _EXPECTED_PAYLOAD_OBJECTS
    ):
        raise ForcedRoundHloAdjudicationError("remote ledger identity drifted")
    for relative, _, _, digest, size in _EXPECTED_PAYLOAD_OBJECTS:
        raw = evidence[relative]
        if sha256(raw).hexdigest() != digest or len(raw) != size:
            raise ForcedRoundHloAdjudicationError(
                f"remote ledger bytes drifted: {relative}"
            )
    expected_local_files = [
        {"byte_count": size, "path": path, "sha256": digest}
        for path, _, _, digest, size in _EXPECTED_PAYLOAD_OBJECTS
        if path != "evidence.json"
    ]
    if (
        local_evidence.get("artifact_kind")
        != "gate_d_forced_round_pp16_hlo_local_evidence"
        or local_evidence.get("code_hash") != EXPECTED_CODE_HASH
        or local_evidence.get("run_tag") != EXPECTED_RUN_TAG
        or local_evidence.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or local_evidence.get("gate_d_closed") is not False
        or local_evidence.get("numerical_claim") is not False
        or local_evidence.get("performance_claim") is not False
        or local_evidence.get("files") != expected_local_files
    ):
        raise ForcedRoundHloAdjudicationError("local evidence manifest drifted")

    remote_ledger = marker.get("remote_ledger", {})
    terminal = receipt.get("terminal", {})
    ledger_path, ledger_generation, ledger_crc32c, ledger_sha, ledger_size = (
        _EXPECTED_LEDGER_OBJECT
    )
    terminal_path, terminal_generation, terminal_crc32c, terminal_sha, terminal_size = (
        _EXPECTED_TERMINAL_OBJECT
    )
    expected_ledger = {
        "crc32c": ledger_crc32c,
        "generation": ledger_generation,
        "path": ledger_path,
        "sha256": ledger_sha,
        "size": ledger_size,
    }
    expected_terminal = {
        "crc32c": terminal_crc32c,
        "generation": terminal_generation,
        "path": terminal_path,
        "sha256": terminal_sha,
        "size": terminal_size,
    }
    if (
        marker.get("run_tag") != EXPECTED_RUN_TAG
        or marker.get("artifact_kind") != "gate_d_forced_round_pp16_hlo_acquired"
        or marker.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or marker.get("adjudicated") is not False
        or marker.get("gate_d_closed") is not False
        or marker.get("numerical_claim") is not False
        or marker.get("performance_claim") is not False
        or marker.get("tpu_numerical_execution_performed") is not False
        or marker.get("evidence_sha256")
        != sha256(evidence["evidence.json"]).hexdigest()
        or marker.get("summary_sha256") != sha256(evidence["summary.json"]).hexdigest()
        or receipt.get("artifact_kind")
        != "gate_d_forced_round_pp16_hlo_terminal_receipt"
        or receipt.get("remote") != EXPECTED_REMOTE_PREFIX + "/HLO_ACQUIRED"
        or remote_ledger != expected_ledger
        or terminal != expected_terminal
        or sha256(evidence["remote_objects.json"]).hexdigest() != ledger_sha
        or len(evidence["remote_objects.json"]) != ledger_size
        or sha256(evidence["HLO_ACQUIRED"]).hexdigest() != terminal_sha
        or len(evidence["HLO_ACQUIRED"]) != terminal_size
        or not (
            int(_EXPECTED_PAYLOAD_OBJECTS[-1][1])
            < int(ledger_generation)
            < int(terminal_generation)
        )
    ):
        raise ForcedRoundHloAdjudicationError("terminal-last archive closure drifted")

    stable_report = audit_stablehlo_structure(stable_text)
    optimized_report = audit_optimized_hlo_structure(optimized_text)
    return {
        "artifact_kind": "gate_d_forced_round_pp16_hlo_causal_adjudication",
        "authorization": {
            "full_8k": False,
            "numerical_execution": False,
            "performance_claim": False,
            "persistence_only": True,
        },
        "classification": (
            "HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_LOCALITY_ACCEPTED;"
            "TPU_NUMERICAL_UNPROVEN;GATE_D_OPEN"
        ),
        "code_hash": EXPECTED_CODE_HASH,
        "gate_d_closed": False,
        "optimized_hlo": {
            "sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            **optimized_report,
        },
        "remote_archive": {
            "object_count_including_ledger_and_terminal": 17,
            "payload_generation_count": len(_EXPECTED_PAYLOAD_OBJECTS),
            "prefix": EXPECTED_REMOTE_PREFIX,
            "remote_replay": remote_replay,
            "terminal_generation": terminal_generation,
            "terminal_last": True,
        },
        "run_tag": EXPECTED_RUN_TAG,
        "schema_version": 1,
        "stablehlo": {"sha256": EXPECTED_STABLEHLO_SHA256, **stable_report},
        "status": "HLO_ACCEPTED_NUMERICAL_UNPROVEN",
    }


def adjudicate_acquired_run(run_dir: Path) -> dict[str, Any]:
    """Bind exact no-follow run bytes to independently replayed remote generations."""

    evidence = _read_canonical_run(run_dir)
    remote_replay_raw = _read_absolute_regular_nofollow(EXPECTED_REMOTE_REPLAY_PATH)
    return _adjudicate_acquired_files(evidence, remote_replay_raw)
