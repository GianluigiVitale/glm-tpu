"""Fail-closed HLO proof for the integrated layer-0 boundary diagnostic."""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    lint_hlo,
    parse_hlo_module,
)
from .association_fingerprint import validate_strategy_nd_reduction
from .dense_rms_replay import (
    _ExactDenseRmsGraph,
    _ResolvedHloValue,
    _called_computation,
    _computation_base,
    _exact_accepted_rms_schedule,
    _shape_signature,
    _validate_exact_dense_rms_value_flow,
)


INTEGRATED_DENSE_RMS_STABLEHLO_SHA256 = (
    "0ae728d6d6ddedbc7d2818975f07b03bc518f26950091bdb592c6574c07853b2"
)
INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256 = (
    "7086b1d0209ce509c7b513f7e4f8ce5c9a232ca658f784617f23ef710fc67568"
)
INTEGRATED_DENSE_ORDINAL_RMS_STABLEHLO_SHA256 = (
    "7a44e06a9a8c37b35e94477583f3be6934bf9aff6e304a56929f98d5b8ce85cf"
)
INTEGRATED_DENSE_PREDENSE_SPLIT_RMS_STABLEHLO_SHA256 = (
    "1fc33c9a12c1dfb2185c063962c56229258597be4ac60505f090e5044bc6f693"
)


def _validate_preceding_attention_input(
    report: Any,
    reduction: Any,
) -> Mapping[str, Any]:
    """Prove rank zero contributes sealed attention and every peer zero."""

    graph = _ExactDenseRmsGraph(report)
    bf16_m1 = (("bf16", (1, 6144)),)
    wide_shapes = {
        (("bf16", (32, 6144)),),
        (("f32", (32, 6144)),),
    }
    accepted_bf16_m32_layouts = (
        "bf16[32,6144]{1,0:T(8,128)(2,1)}",
        "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}",
    )
    accepted_bf16_m1_layouts = (
        "bf16[1,6144]{1,0:T(2,128)(2,1)}",
        "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}",
    )

    def exact_layout(value: Any, *, tpu_layout: bool) -> bool:
        if value is None:
            return False
        shape = graph.shape(value)
        if shape == (("f32", (32, 6144)),):
            return not tpu_layout and graph.exact_prefix(
                value.instruction, "f32[32,6144]{1,0}"
            )
        if shape == (("f32", (1, 6144)),):
            return not tpu_layout and graph.exact_prefix(
                value.instruction, "f32[1,6144]{1,0}"
            )
        if shape == (("bf16", (32, 6144)),):
            return bool(
                (not tpu_layout and graph.exact_prefix(
                    value.instruction, "bf16[32,6144]{1,0}"
                ))
                or (
                    tpu_layout
                    and any(
                        graph.exact_prefix(value.instruction, signature)
                        for signature in accepted_bf16_m32_layouts
                    )
                )
            )
        if shape == bf16_m1:
            return bool(
                (not tpu_layout and graph.exact_prefix(
                    value.instruction, "bf16[1,6144]{1,0}"
                ))
                or (
                    tpu_layout
                    and any(
                        graph.exact_prefix(value.instruction, signature)
                        for signature in accepted_bf16_m1_layouts
                    )
                )
            )
        return False

    def exact_partition_index(value: Any, depth: int = 0) -> bool:
        value = graph.semantic(value)
        if value is None or depth > 5:
            return False
        opcode = value.instruction.raw_opcode
        if opcode == "partition-id":
            return True
        if opcode == "convert":
            return exact_partition_index(graph.operand(value, 0), depth + 1)
        if opcode in {"and", "remainder", "divide"} and len(
            value.instruction.operand_names
        ) == 2:
            operands = [graph.operand(value, index) for index in range(2)]
            expected = {
                "and": np.float32(31.0),
                "remainder": np.float32(32.0),
                "divide": np.float32(1.0),
            }[opcode]
            if opcode == "and":
                return any(
                    exact_partition_index(operands[index], depth + 1)
                    and graph.exact_constant(operands[1 - index], expected)
                    for index in range(2)
                )
            return bool(
                exact_partition_index(operands[0], depth + 1)
                and graph.exact_constant(operands[1], expected)
            )
        return False

    def exact_attention_pad(
        value: Any, *, tpu_layout: bool, depth: int = 0
    ) -> bool:
        value = graph.semantic(value)
        if value is None or depth > 4:
            return False
        if not exact_layout(value, tpu_layout=tpu_layout):
            return False
        if value.instruction.raw_opcode == "convert":
            if graph.shape(value) not in wide_shapes | {
                (("f32", (1, 6144)),),
                bf16_m1,
            }:
                return False
            return exact_attention_pad(
                graph.operand(value, 0),
                tpu_layout=tpu_layout,
                depth=depth + 1,
            )
        if value.instruction.raw_opcode != "pad" or graph.shape(value) not in wide_shapes:
            return False
        attributes = re.findall(
            r"\bpadding=([^,\s}]+)", graph.clean(value.instruction)
        )
        source = graph.semantic(graph.operand(value, 0))
        if source is not None and source.instruction.raw_opcode == "convert":
            if not exact_layout(source, tpu_layout=tpu_layout):
                return False
            source = graph.semantic(graph.operand(source, 0))
        return bool(
            attributes == ["0_31x0_0"]
            and graph.exact_constant(
                graph.operand(value, 1), np.float32(0.0)
            )
            and graph.exact_parameter(source, 0, bf16_m1)
            and source is not None
            and exact_layout(source, tpu_layout=tpu_layout)
        )

    external_input = graph.operand(graph.value(reduction), 0)
    external_semantic = graph.resolve_parameter(external_input)
    tpu_layout = graph.shape(external_semantic) == (("bf16", (32, 6144)),)
    if not exact_layout(external_semantic, tpu_layout=tpu_layout):
        raise BenchmarkValidationError(
            "preceding attention collective external input layout drifted"
        )
    selected = graph.semantic(external_input)
    if (
        selected is None
        or selected.instruction.raw_opcode != "select"
        or graph.shape(selected) not in wide_shapes
        or len(selected.instruction.operand_names) != 3
        or not exact_layout(selected, tpu_layout=tpu_layout)
    ):
        raise BenchmarkValidationError(
            "preceding attention collective input is not the exact owner select"
        )
    predicate = graph.semantic(graph.operand(selected, 0))
    if predicate is not None and predicate.instruction.raw_opcode in {
        "broadcast",
        "broadcast-in-dim",
    }:
        dimensions = re.findall(
            r"\bdimensions=\{([^}]*)\}", graph.clean(predicate.instruction)
        )
        if dimensions != [""]:
            predicate = None
        else:
            predicate = graph.semantic(graph.operand(predicate, 0))
    if (
        predicate is None
        or predicate.instruction.raw_opcode != "compare"
        or len(predicate.instruction.operand_names) != 2
        or re.findall(
            r"\bdirection=([A-Z]+)", graph.clean(predicate.instruction)
        )
        not in (["EQ"], ["NE"])
    ):
        raise BenchmarkValidationError(
            "preceding attention owner predicate drifted"
        )
    predicate_operands = [graph.operand(predicate, index) for index in range(2)]
    if not any(
        exact_partition_index(predicate_operands[index])
        and graph.exact_constant(
            predicate_operands[1 - index], np.float32(0.0)
        )
        for index in range(2)
    ):
        raise BenchmarkValidationError(
            "preceding attention owner is not exact physical rank zero"
        )
    branches = [graph.operand(selected, index) for index in (1, 2)]
    direction = re.findall(
        r"\bdirection=([A-Z]+)", graph.clean(predicate.instruction)
    )[0]
    attention_index = 0 if direction == "EQ" else 1
    if not (
        exact_attention_pad(
            branches[attention_index], tpu_layout=tpu_layout
        )
        and exact_layout(
            graph.semantic(branches[1 - attention_index]),
            tpu_layout=tpu_layout,
        )
        and graph.exact_constant(
            branches[1 - attention_index], np.float32(0.0)
        )
    ):
        raise BenchmarkValidationError(
            "preceding attention owner/zero branches drifted"
        )
    result = {
        "exact_attention_collective_input": True,
        "exact_attention_owner_rank": 0,
    }
    return result


def _validate_split_predense_value_flow(
    report: Any,
    scheduled: Any,
    gate_convolution: Any,
    *,
    preceding_attention_reduction: Any | None = None,
) -> Mapping[str, Any]:
    """Bind the scalar pre-dense RMS schedule to the sole live gate input."""

    graph = _ExactDenseRmsGraph(report)
    f32_m32 = (("f32", (32, 6144)),)
    bf16_m32 = (("bf16", (32, 6144)),)
    f32_m1 = (("f32", (1, 6144)),)
    bf16_m1 = (("bf16", (1, 6144)),)
    f32_rows = (("f32", (32,)),)

    entry_parameters = {
        int(match.group(1)): _shape_signature(item)
        for item in report.module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_opcode == "parameter"
        and (match := re.search(r"\bparameter\(([0-9]+)\)", item.raw_line))
        is not None
    }
    if entry_parameters != {
        0: bf16_m1,
        1: bf16_m1,
        2: (("bf16", (6144,)),),
        3: (("f8e4m3fn", (1, 1, 6144, 768)),),
        4: (("f32", (1, 1, 48, 768)),),
        5: (("f8e4m3fn", (1, 1, 384, 6144)),),
        6: (("f32", (1, 1, 3, 6144)),),
        7: (("bf16", (6144,)),),
    }:
        raise BenchmarkValidationError(
            "integrated pre-dense split ENTRY inputs drifted"
        )

    def semantic_opcode(
        value: _ResolvedHloValue | None,
        opcode: str,
        shape: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> _ResolvedHloValue | None:
        value = graph.semantic(value)
        return (
            value
            if value is not None
            and value.instruction.raw_opcode == opcode
            and graph.shape(value) == shape
            else None
        )

    def exact_pad(
        value: _ResolvedHloValue | None,
        parameter_index: int,
        *,
        dtype: str,
    ) -> bool:
        value = semantic_opcode(value, "pad", ((dtype, (32, 6144)),))
        if value is None or len(value.instruction.operand_names) != 2:
            return False
        attributes = re.findall(
            r"\bpadding=([^,\s}]+)", graph.clean(value.instruction)
        )
        return bool(
            attributes == ["0_31x0_0"]
            and graph.exact_constant(graph.operand(value, 1), np.float32(0.0))
            and graph.exact_parameter(
                graph.operand(value, 0), parameter_index, ((dtype, (1, 6144)),)
            )
        )

    def exact_source_f32(
        value: _ResolvedHloValue | None, parameter_index: int
    ) -> bool:
        value = semantic_opcode(value, "convert", f32_m32)
        if value is None:
            return False
        source = graph.operand(value, 0)
        if parameter_index == 0 and preceding_attention_reduction is not None:
            return graph.exact_external(source, preceding_attention_reduction)
        if exact_pad(source, parameter_index, dtype="bf16"):
            return True
        padded = semantic_opcode(source, "pad", f32_m32)
        if padded is None or len(padded.instruction.operand_names) != 2:
            return False
        converted = semantic_opcode(graph.operand(padded, 0), "convert", f32_m1)
        attributes = re.findall(
            r"\bpadding=([^,\s}]+)", graph.clean(padded.instruction)
        )
        return bool(
            converted is not None
            and attributes == ["0_31x0_0"]
            and graph.exact_constant(
                graph.operand(padded, 1), np.float32(0.0)
            )
            and graph.exact_parameter(
                graph.operand(converted, 0), parameter_index, bf16_m1
            )
        )

    def exact_sum(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "add", f32_m32)
        if value is None or len(value.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        return any(
            exact_source_f32(operands[index], 0)
            and exact_source_f32(operands[1 - index], 1)
            for index in range(2)
        )

    def exact_square(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if value is None or graph.shape(value) != f32_m32:
            return False
        if value.instruction.raw_opcode == "square":
            return exact_sum(graph.operand(value, 0))
        return bool(
            value.instruction.raw_opcode == "multiply"
            and len(value.instruction.operand_names) == 2
            and exact_sum(graph.operand(value, 0))
            and exact_sum(graph.operand(value, 1))
        )

    scheduled_signature = _shape_signature(scheduled)
    scheduled_root = graph.semantic(graph.value(scheduled))
    if scheduled_signature == (f32_rows[0], bf16_m32[0]):
        if scheduled_root is None or scheduled_root.instruction.raw_opcode != "tuple":
            scheduled_root = None
        else:
            scheduled_root = graph.semantic(graph.operand(scheduled_root, 0))
    if (
        scheduled_root is None
        or scheduled_root.instruction.raw_opcode != "reduce"
        or graph.shape(scheduled_root) != f32_rows
        or len(scheduled_root.instruction.operand_names) != 2
        or re.findall(
            r"\bdimensions=\{([^}]*)\}", graph.clean(scheduled_root.instruction)
        )
        != ["1"]
        or not exact_square(graph.operand(scheduled_root, 0))
        or not graph.exact_constant(
            graph.operand(scheduled_root, 1), np.float32(0.0)
        )
        or not graph.exact_reducer(scheduled_root)
    ):
        raise BenchmarkValidationError(
            "integrated pre-dense scheduled reduction arithmetic drifted"
        )

    def exact_scheduled(value: _ResolvedHloValue | None) -> bool:
        if value is None:
            return False
        if scheduled_signature == f32_rows:
            return graph.exact_external(value, scheduled)
        value = graph.resolve_parameter(value)
        return bool(
            scheduled_signature == (f32_rows[0], bf16_m32[0])
            and value.instruction.raw_opcode == "get-tuple-element"
            and graph._tuple_index(value.instruction) == 0
            and graph.exact_external(graph.operand(value, 0), scheduled)
        )

    def exact_mean(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if (
            value is None
            or value.instruction.raw_opcode not in {"divide", "multiply"}
            or graph.shape(value) != f32_rows
            or len(value.instruction.operand_names) != 2
        ):
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        if value.instruction.raw_opcode == "divide":
            return exact_scheduled(operands[0]) and graph.exact_constant(
                operands[1], np.float32(6144.0)
            )
        return any(
            exact_scheduled(operands[index])
            and graph.exact_constant(
                operands[1 - index], np.float32(1.0 / 6144.0)
            )
            for index in range(2)
        )

    def exact_rsqrt(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "rsqrt", f32_rows)
        if value is None:
            return False
        variance = semantic_opcode(graph.operand(value, 0), "add", f32_rows)
        if variance is None or len(variance.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(variance, index) for index in range(2)]
        return any(
            exact_mean(operands[index])
            and graph.exact_constant(
                operands[1 - index], np.float32(1.0e-5)
            )
            for index in range(2)
        )

    def exact_rsqrt_broadcast(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode in {"broadcast", "broadcast-in-dim"}
            and graph.shape(value) == f32_m32
            and re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
            )
            == ["0"]
            and exact_rsqrt(graph.operand(value, 0))
        )

    def exact_normalized_round(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "convert", bf16_m32)
        normalized = (
            semantic_opcode(graph.operand(value, 0), "multiply", f32_m32)
            if value is not None
            else None
        )
        if normalized is None or len(normalized.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(normalized, index) for index in range(2)]
        return any(
            exact_sum(operands[index])
            and exact_rsqrt_broadcast(operands[1 - index])
            for index in range(2)
        )

    def exact_norm_weight_f32(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "convert", f32_m32)
        if value is None:
            return False
        broadcast = graph.semantic(graph.operand(value, 0))
        return bool(
            broadcast is not None
            and broadcast.instruction.raw_opcode
            in {"broadcast", "broadcast-in-dim"}
            and graph.shape(broadcast) == bf16_m32
            and re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(broadcast.instruction)
            )
            == ["1"]
            and graph.exact_parameter(
                graph.operand(broadcast, 0), 2, (("bf16", (6144,)),)
            )
        )

    def exact_weighted(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "convert", bf16_m32)
        weighted = (
            semantic_opcode(graph.operand(value, 0), "multiply", f32_m32)
            if value is not None
            else None
        )
        if weighted is None or len(weighted.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(weighted, index) for index in range(2)]
        for index in range(2):
            lifted = semantic_opcode(operands[index], "convert", f32_m32)
            if (
                lifted is not None
                and exact_normalized_round(graph.operand(lifted, 0))
                and exact_norm_weight_f32(operands[1 - index])
            ):
                return True
        return False

    gate_computation = _computation_base(gate_convolution.computation)
    callers = tuple(
        item
        for item in report.module.instructions
        if item.raw_opcode == "fusion"
        and _called_computation(item) == gate_computation
    )
    if (
        gate_convolution.computation.startswith("ENTRY ")
        or len(callers) != 1
    ):
        raise BenchmarkValidationError(
            "integrated pre-dense gate is not in one exact enclosing fusion"
        )
    caller = callers[0]
    caller_value = graph.value(caller)
    bindings = tuple(
        operand
        for index in range(len(caller.operand_names))
        if (operand := graph.operand(caller_value, index)) is not None
    )
    if len(bindings) != len(caller.operand_names):
        raise BenchmarkValidationError(
            "integrated pre-dense gate fusion bindings drifted"
        )
    bound_gate = _ResolvedHloValue(gate_convolution, bindings)
    gate_lhs = graph.operand(bound_gate, 0)
    exact_fused_ownership = bool(
        gate_lhs is not None
        and gate_lhs.instruction.computation == gate_convolution.computation
        and gate_lhs.instruction.raw_opcode != "parameter"
    )
    if not exact_fused_ownership or not exact_weighted(gate_lhs):
        raise BenchmarkValidationError(
            "integrated pre-dense scheduled RMS does not feed the fused gate LHS"
        )
    return {
        "exact_predense_gate_fusion_ownership": True,
        "exact_predense_gate_input": True,
        "exact_predense_recompute": True,
        "exact_predense_scheduled_reduction": True,
        "exact_predense_scheduled_rsqrt": True,
        "predense_gate_caller": caller.name,
        "predense_gate_convolution": gate_convolution.name,
    }


def integrated_dense_rms_hlo_policy(
    member_device_ids: Sequence[int],
    *,
    preceding_attention_collective: bool = False,
) -> HloContractPolicy:
    """Require the diagnostic's exact full-pod collective scope and group."""

    members = tuple(int(value) for value in member_device_ids)
    if members != tuple(range(32)):
        raise BenchmarkValidationError(
            "integrated dense RMS requires physical ids 0..31"
        )
    if not isinstance(preceding_attention_collective, bool):
        raise BenchmarkValidationError(
            "preceding-attention collective flag must be boolean"
        )
    patterns = (
        r"integrated_dense_rms_preceding_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    ) if preceding_attention_collective else (
        r"integrated_dense_rms_strategy_nd_collective",
    )
    return HloContractPolicy(
        name="strategy-nd-integrated-dense-rms",
        total_devices=32,
        repeated_region_patterns=patterns,
        maximum_repeated_collective_group_size=32,
        expected_repeated_replica_groups=(members,),
        expected_collectives=(
            CollectiveExpectation(
                "all-reduce", 2 if preceding_attention_collective else 1
            ),
        ),
        partition_id_to_device_id=members,
        forbidden_row_width_pairs=(),
        allow_full_pod_repeated_collectives=True,
    )


def validate_integrated_dense_rms_stablehlo(
    stablehlo: str,
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
) -> Mapping[str, Any]:
    """Pin the complete exact eight-input StableHLO graph byte-for-byte."""

    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    if not isinstance(preceding_attention_collective, bool):
        raise BenchmarkValidationError(
            "preceding-attention collective flag must be boolean"
        )
    if not isinstance(split_predense_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense pre-dense split flag must be boolean"
        )
    if preceding_attention_collective and not split_layer1_rms:
        raise BenchmarkValidationError(
            "preceding attention collective requires split RMS"
        )
    if split_predense_rms and not split_layer1_rms:
        raise BenchmarkValidationError(
            "pre-dense split RMS requires layer-1 split RMS"
        )
    if split_predense_rms and preceding_attention_collective:
        raise BenchmarkValidationError(
            "pre-dense split RMS and rejected ordinal arms are disjoint"
        )
    expected = (
        INTEGRATED_DENSE_PREDENSE_SPLIT_RMS_STABLEHLO_SHA256
        if split_predense_rms
        else INTEGRATED_DENSE_ORDINAL_RMS_STABLEHLO_SHA256
        if preceding_attention_collective
        else INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256
        if split_layer1_rms
        else INTEGRATED_DENSE_RMS_STABLEHLO_SHA256
    )
    digest = sha256(stablehlo.encode()).hexdigest()
    if digest != expected:
        raise BenchmarkValidationError(
            "integrated dense RMS StableHLO SHA-256 drifted: "
            f"expected={expected} found={digest}"
        )
    result = {
        "exact_graph_sha256": digest,
        "exact_input_schema": True,
        "exact_predense_rms": True,
        "exact_final_layout_dense": True,
        "exact_strategy_nd_collective": True,
        "exact_layer1_rms": True,
        "exact_live_result": True,
        "split_layer1_rms": split_layer1_rms,
        "passed": True,
        "performance_claim": False,
        "violations": [],
    }
    if preceding_attention_collective:
        result["preceding_attention_collective"] = True
    if split_predense_rms:
        result["split_predense_rms"] = True
    return result


def validate_integrated_dense_rms_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
) -> Mapping[str, Any]:
    """Bind real contractions through one BF16 StrategyND and layer-1 ROOT."""

    from scripts.greenfield.probe_layer0_dense_convolution import (
        _validate_optimized_hlo,
    )

    members = tuple(int(value) for value in member_device_ids)
    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    if not isinstance(preceding_attention_collective, bool):
        raise BenchmarkValidationError(
            "preceding-attention collective flag must be boolean"
        )
    if not isinstance(split_predense_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense pre-dense split flag must be boolean"
        )
    if preceding_attention_collective and not split_layer1_rms:
        raise BenchmarkValidationError(
            "preceding attention collective requires split RMS"
        )
    if split_predense_rms and not split_layer1_rms:
        raise BenchmarkValidationError(
            "pre-dense split RMS requires layer-1 split RMS"
        )
    if split_predense_rms and preceding_attention_collective:
        raise BenchmarkValidationError(
            "pre-dense split RMS and rejected ordinal arms are disjoint"
        )
    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        integrated_dense_rms_hlo_policy(
            members,
            preceding_attention_collective=preceding_attention_collective,
        ),
    )
    report.raise_for_violations()
    reductions = tuple(
        item
        for item in report.module.collectives
        if item.opcode == "all-reduce"
    )
    expected_reductions = 2 if preceding_attention_collective else 1
    if len(reductions) != expected_reductions:
        raise BenchmarkValidationError(
            f"integrated dense RMS requires exactly {expected_reductions} all-reduces"
        )
    attention_reductions = tuple(
        item
        for item in reductions
        if "integrated_dense_rms_preceding_attention_collective"
        in (item.op_name or "").split("/")
    )
    dense_reductions = tuple(
        item
        for item in reductions
        if "integrated_dense_rms_strategy_nd_collective"
        in (item.op_name or "").split("/")
    )
    if len(dense_reductions) != 1 or (
        preceding_attention_collective
        and (
            len(attention_reductions) != 1
            or attention_reductions[0].index >= dense_reductions[0].index
        )
    ) or (not preceding_attention_collective and attention_reductions):
        raise BenchmarkValidationError(
            "integrated dense collective scope/order drifted"
        )
    reduction = dense_reductions[0]
    attention_reduction = (
        attention_reductions[0] if preceding_attention_collective else None
    )
    algorithm = validate_strategy_nd_reduction(report, reduction)
    attention_algorithm = (
        validate_strategy_nd_reduction(report, attention_reduction)
        if attention_reduction is not None
        else None
    )
    attention_input = (
        _validate_preceding_attention_input(report, attention_reduction)
        if attention_reduction is not None
        else {}
    )
    entry = tuple(
        item
        for item in report.module.instructions
        if item.computation.startswith("ENTRY ")
    )
    roots = tuple(
        item for item in entry if item.raw_line.lstrip().startswith("ROOT ")
    )
    if len(roots) != 1 or _shape_signature(roots[0]) != (
        ("u16", (1, 6144)),
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS ENTRY result drifted"
        )
    scheduled = tuple(
        item
        for item in entry
        if "integrated_dense_rms_layer1_norm" in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if len(scheduled) != 1:
        raise BenchmarkValidationError(
            "integrated dense RMS accepted layer-1 schedule drifted"
        )
    expected_schedule_shape = (
        (("f32", (32,)),)
        if split_layer1_rms
        else (("f32", (32,)), ("f32", (32, 6144)))
    )
    if _shape_signature(scheduled[0]) != expected_schedule_shape:
        raise BenchmarkValidationError(
            "integrated dense RMS layer-1 schedule form drifted"
        )
    predense_scheduled = tuple(
        item
        for item in entry
        if "integrated_dense_rms_predense_norm" in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if split_predense_rms:
        if len(predense_scheduled) != 1 or _shape_signature(
            predense_scheduled[0]
        ) != (("f32", (32,)),):
            raise BenchmarkValidationError(
                "integrated dense accepted pre-dense scalar schedule drifted"
            )
    elif predense_scheduled:
        raise BenchmarkValidationError(
            "integrated dense unexpected pre-dense scalar schedule"
        )
    contraction = _validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        final_dense_layout=True,
        dense_envelope=True,
        integrated_dense_rms=True,
        accepted_gate_singleton=True,
        accepted_gate_dequant_fusion=True,
        preceding_attention_collective=preceding_attention_collective,
    )
    required_contraction = (
        "accepted_down_schedule",
        "accepted_gate_up_schedule",
        "exact_accepted_kernel_geometry",
        "exact_accepted_weight_layout",
        "exact_activation_graph",
        "exact_carried_residual_binding",
        "exact_collective_input_binding",
        "exact_gate_dequant_fusion_boundary",
        "exact_gate_singleton_external_boundary",
        "exact_packed_weight_lineage",
        "exact_result_binding",
    )
    if not contraction.get("passed") or not all(
        contraction.get(name) is True for name in required_contraction
    ):
        raise BenchmarkValidationError(
            f"integrated dense contraction HLO drifted: {contraction}"
        )
    predense_flow: Mapping[str, Any] = {}
    if split_predense_rms:
        gate_records = contraction.get("accepted_weight_layouts", {}).get(
            "gate_up", []
        )
        gate_name = (
            gate_records[0].get("convolution")
            if len(gate_records) == 1
            and gate_records[0].get("virtual_rank") == 0
            else None
        )
        gate_instructions = tuple(
            item
            for item in report.module.instructions
            if item.name == gate_name
        )
        if len(gate_instructions) != 1:
            raise BenchmarkValidationError(
                "integrated dense pre-dense split lacks one physical gate"
            )
        predense_flow = _validate_split_predense_value_flow(
            report,
            predense_scheduled[0],
            gate_instructions[0],
        )
    exact_rms = _validate_exact_dense_rms_value_flow(
        report,
        reduction,
        scheduled[0],
        roots[0],
        integrated_dense=True,
        require_split_output_fusion=split_layer1_rms,
        preceding_attention_reduction=attention_reduction,
    )
    if (
        exact_rms.get("residual_source_mode")
        != "integrated_attention_plus_combined_residual"
        or not all(
            exact_rms.get(name) is True
            for name in (
                "exact_collective_input",
                "exact_reduction_operand_graph",
                "exact_weighted_operand_graph",
                "exact_result_binding",
            )
        )
        or (
            split_layer1_rms
            and not all(
                exact_rms.get(name) is True
                for name in (
                    "exact_accepted_scheduled_reduction",
                    "split_output_fusion_exact",
                    "split_recompute_exact",
                )
            )
        )
    ):
        raise BenchmarkValidationError(
            f"integrated dense layer-1 HLO drifted: {exact_rms}"
        )
    if any(
        marker in optimized_hlo
        for marker in ("host_callback", "xla_python_cpu_callback", " outfeed(")
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS contains a host effect"
        )
    result = {
        "accepted_scheduled_reduction": scheduled[0].name,
        "collective_algorithm": dict(algorithm),
        "collective_count": len(report.module.collectives),
        "contraction": contraction,
        **exact_rms,
        "live_rows": 1,
        "num_partitions": report.module.num_partitions,
        "num_replicas": report.module.num_replicas,
        "passed": True,
        "performance_claim": False,
        "split_layer1_rms": split_layer1_rms,
        "violations": [],
    }
    if preceding_attention_collective:
        result.update(
            {
                "attention_collective": attention_reduction.name,
                "attention_collective_algorithm": dict(attention_algorithm),
                **attention_input,
                "preceding_attention_collective": True,
            }
        )
    if split_predense_rms:
        result.update(
            {
                "accepted_predense_scheduled_reduction": (
                    predense_scheduled[0].name
                ),
                **predense_flow,
                "split_predense_rms": True,
            }
        )
    return result
