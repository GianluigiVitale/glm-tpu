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
INTEGRATED_DENSE_ACCEPTED_SOURCE_STABLEHLO_SHA256 = (
    "b46a58b1cb7576b8124b02ac09b722e07ddebf6cf7663ac5e404f64414144fed"
)
# Captured once by the fail-closed protected acquisition at ``63e7017``.
# Both complete compiler graphs are pinned before arithmetic is authorized.
INTEGRATED_DENSE_NATIVE_SOURCE_STABLEHLO_SHA256 = (
    "0884c34e137688bb66c96dfeffbdd5c0bacfd4710d52c6be45d83b4fbe683d66"
)
INTEGRATED_DENSE_NATIVE_SOURCE_OPTIMIZED_HLO_SHA256 = (
    "4b13a9f123ae75ae2b196673d96788360ee5f6bd72ebd0d7cad9865a0feaaf63"
)

_NATIVE_SOURCE_COLLECTIVE_SCOPES = (
    "native_source_context_embedding_collective",
    "native_source_context_attention_collective",
    "integrated_dense_rms_strategy_nd_collective",
)
_NATIVE_SOURCE_INPUT_SCHEMA = (
    (0, "attended_latent", (("bf16", (64, 512)),)),
    (1, "token_ids", (("s32", (32,)),)),
    (2, "post_attention_norm", (("bf16", (6144,)),)),
    (3, "embedding_shard", (("bf16", (1, 4840, 6144)),)),
    (4, "kv_b_bits", (("u8", (1, 7168, 512)),)),
    (5, "kv_b_scale", (("f32", (1, 56, 4)),)),
    (6, "o_bits_in_out", (("u8", (1, 512, 6144)),)),
    (7, "o_scale_in_out", (("f32", (1, 4, 48)),)),
    (8, "merged_bits", (("f8e4m3fn", (1, 1, 6144, 768)),)),
    (9, "merged_scale", (("f32", (1, 1, 48, 768)),)),
    (10, "down_bits", (("f8e4m3fn", (1, 1, 384, 6144)),)),
    (11, "down_scale", (("f32", (1, 1, 3, 6144)),)),
    (12, "layer1_norm", (("bf16", (6144,)),)),
)


def _validate_preceding_attention_input(
    report: Any,
    reduction: Any,
    *,
    parameter_index: int = 0,
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
            and graph.exact_parameter(source, parameter_index, bf16_m1)
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
        "exact_attention_parameter_index": parameter_index,
    }
    return result


def _validate_accepted_attention_input(
    report: Any,
    reduction: Any,
    embedding_reduction: Any,
) -> Mapping[str, Any]:
    """Prove the sealed attention owner input is guarded by embedding row 0."""

    graph = _ExactDenseRmsGraph(report)
    m32_shapes = {
        (("bf16", (32, 6144)),),
        (("f32", (32, 6144)),),
    }
    bf16_m32 = (("bf16", (32, 6144)),)
    f32_m32 = (("f32", (32, 6144)),)

    def exact_layout(
        value: _ResolvedHloValue | None, *, tpu_layout: bool
    ) -> bool:
        if value is None:
            return False
        shape = graph.shape(value)
        cpu_prefixes = {
            (("bf16", (32, 6144)),): ("bf16[32,6144]{1,0}",),
            (("f32", (32, 6144)),): ("f32[32,6144]{1,0}",),
            (("bf16", (1, 6144)),): ("bf16[1,6144]{1,0}",),
            (("f32", (1, 6144)),): ("f32[1,6144]{1,0}",),
            (("bf16", (1, 1)),): ("bf16[1,1]{1,0}",),
            (("f32", (1, 1)),): ("f32[1,1]{1,0}",),
            (("pred", (32, 6144)),): ("pred[32,6144]{1,0}",),
            (("pred", (1, 1)),): ("pred[1,1]{1,0}",),
            (("pred", ()),): ("pred[]",),
        }
        tpu_prefixes = {
            (("bf16", (32, 6144)),): (
                "bf16[32,6144]{1,0:T(8,128)(2,1)}",
                "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}",
            ),
            (("f32", (32, 6144)),): (
                "f32[32,6144]{1,0:T(8,128)}",
                "f32[32,6144]{1,0:T(8,128)S(3)}",
            ),
            (("bf16", (1, 6144)),): (
                "bf16[1,6144]{1,0:T(2,128)(2,1)}",
                "bf16[1,6144]{1,0:T(2,128)(2,1)S(3)}",
            ),
            (("f32", (1, 6144)),): (
                "f32[1,6144]{1,0:T(1,128)}",
                "f32[1,6144]{1,0:T(1,128)S(3)}",
            ),
            (("bf16", (1, 1)),): (
                "bf16[1,1]{1,0:T(2,128)(2,1)}",
                "bf16[1,1]{1,0:T(2,128)(2,1)S(3)}",
            ),
            (("f32", (1, 1)),): (
                "f32[1,1]{1,0:T(1,128)}",
                "f32[1,1]{1,0:T(1,128)S(3)}",
            ),
            (("pred", (32, 6144)),): (
                "pred[32,6144]{1,0:T(8,128)(4,1)}",
            ),
            (("pred", (1, 1)),): (
                "pred[1,1]{1,0:T(4,128)(4,1)}",
            ),
            (("pred", ()),): (
                "pred[]{:T(512)}",
            ),
        }
        prefixes = (tpu_prefixes if tpu_layout else cpu_prefixes).get(
            shape, ()
        )
        return any(
            graph.exact_prefix(value.instruction, prefix)
            for prefix in prefixes
        )

    def exact_convert_chain(
        value: _ResolvedHloValue | None,
        *,
        count: int,
        tpu_layout: bool,
    ) -> _ResolvedHloValue | None:
        value = graph.semantic(value)
        for _ in range(count):
            if (
                value is None
                or value.instruction.raw_opcode != "convert"
                or len(value.instruction.operand_names) != 1
                or not exact_layout(value, tpu_layout=tpu_layout)
            ):
                return None
            source = graph.semantic(graph.operand(value, 0))
            if (
                source is None
                or not exact_layout(source, tpu_layout=tpu_layout)
                or graph.shape(source)[0][1] != graph.shape(value)[0][1]
                or {graph.shape(source)[0][0], graph.shape(value)[0][0]}
                != {"bf16", "f32"}
            ):
                return None
            value = source
        if value is not None and value.instruction.raw_opcode == "convert":
            return None
        return value

    def exact_partition_index(
        value: _ResolvedHloValue | None, depth: int = 0
    ) -> bool:
        value = graph.semantic(value)
        if value is None or depth > 5:
            return False
        opcode = value.instruction.raw_opcode
        if opcode == "partition-id":
            return True
        if opcode == "convert":
            return exact_partition_index(graph.operand(value, 0), depth + 1)
        if opcode == "and" and len(value.instruction.operand_names) == 2:
            operands = [graph.operand(value, index) for index in range(2)]
            return any(
                exact_partition_index(operands[index], depth + 1)
                and graph.exact_constant(
                    operands[1 - index], np.float32(31.0)
                )
                for index in range(2)
            )
        return False

    def exact_owner_predicate(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if value is not None and value.instruction.raw_opcode in {
            "broadcast",
            "broadcast-in-dim",
        }:
            if re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
            ) != [""]:
                return False
            value = graph.semantic(graph.operand(value, 0))
        if (
            value is None
            or value.instruction.raw_opcode != "compare"
            or len(value.instruction.operand_names) != 2
        ):
            return False
        direction = re.findall(
            r"\bdirection=([A-Z]+)", graph.clean(value.instruction)
        )
        operands = [graph.operand(value, index) for index in range(2)]
        return bool(
            direction in (["EQ"], ["NE"])
            and any(
                exact_partition_index(operands[index])
                and graph.exact_constant(
                    operands[1 - index], np.float32(0.0)
                )
                for index in range(2)
            )
        )

    def exact_padded_attention(
        value: _ResolvedHloValue | None, *, tpu_layout: bool
    ) -> bool:
        value = exact_convert_chain(
            value, count=0 if tpu_layout else 2, tpu_layout=tpu_layout
        )
        if (
            value is None
            or value.instruction.raw_opcode != "pad"
            or graph.shape(value) not in m32_shapes
            or not exact_layout(value, tpu_layout=tpu_layout)
            or len(value.instruction.operand_names) != 2
            or re.findall(
                r"\bpadding=([^,\s}]+)", graph.clean(value.instruction)
            )
            != ["0_31x0_0"]
            or not graph.exact_constant(
                graph.operand(value, 1), np.float32(0.0)
            )
        ):
            return False
        source = exact_convert_chain(
            graph.operand(value, 0),
            count=0 if tpu_layout else 1,
            tpu_layout=tpu_layout,
        )
        return bool(
            exact_layout(source, tpu_layout=tpu_layout)
            and graph.exact_parameter(
                source, 0, (("bf16", (1, 6144)),)
            )
        )

    def exact_zero(value: _ResolvedHloValue | None) -> bool:
        return graph.exact_constant(value, np.float32(0.0))

    def exact_owner_select(
        value: _ResolvedHloValue | None, *, tpu_layout: bool
    ) -> bool:
        value = exact_convert_chain(
            value, count=0 if tpu_layout else 2, tpu_layout=tpu_layout
        )
        if (
            value is None
            or value.instruction.raw_opcode != "select"
            or graph.shape(value) not in m32_shapes
            or not exact_layout(value, tpu_layout=tpu_layout)
            or len(value.instruction.operand_names) != 3
            or not exact_owner_predicate(graph.operand(value, 0))
        ):
            return False
        predicate = graph.semantic(graph.operand(value, 0))
        if predicate is not None and predicate.instruction.raw_opcode in {
            "broadcast",
            "broadcast-in-dim",
        }:
            predicate = graph.semantic(graph.operand(predicate, 0))
        direction = re.findall(
            r"\bdirection=([A-Z]+)", graph.clean(predicate.instruction)
        )[0]
        branches = [graph.operand(value, index) for index in (1, 2)]
        source_index = 0 if direction == "EQ" else 1
        return bool(
            exact_padded_attention(
                branches[source_index], tpu_layout=tpu_layout
            )
            and exact_layout(
                graph.semantic(branches[1 - source_index]),
                tpu_layout=tpu_layout,
            )
            and exact_zero(branches[1 - source_index])
        )

    def exact_embedding_scalar(
        value: _ResolvedHloValue | None, *, tpu_layout: bool
    ) -> bool:
        value = graph.semantic(value)
        if value is not None and value.instruction.raw_opcode == "bitcast":
            source = graph.semantic(graph.operand(value, 0))
            if (
                source is None
                or graph.shape(source)[0][0] != graph.shape(value)[0][0]
                or np.prod(graph.shape(source)[0][1], dtype=np.int64) != 1
                or np.prod(graph.shape(value)[0][1], dtype=np.int64) != 1
            ):
                return False
            value = graph.semantic(graph.operand(value, 0))
        value = exact_convert_chain(
            value, count=0 if tpu_layout else 2, tpu_layout=tpu_layout
        )
        if (
            value is None
            or value.instruction.raw_opcode != "slice"
            or not exact_layout(value, tpu_layout=tpu_layout)
            or len(value.instruction.operand_names) != 1
            or re.findall(
                r"\bslice=(\{\[[^}\n]+\]\})",
                re.sub(r"\s+", "", graph.clean(value.instruction)),
            )
            != ["{[0:1],[0:1]}"]
        ):
            return False
        source = exact_convert_chain(
            graph.operand(value, 0),
            count=0 if tpu_layout else 2,
            tpu_layout=tpu_layout,
        )
        return bool(
            exact_layout(source, tpu_layout=tpu_layout)
            and graph.exact_external(source, embedding_reduction)
        )

    def exact_finite_predicate(
        value: _ResolvedHloValue | None, *, tpu_layout: bool
    ) -> bool:
        value = graph.semantic(value)
        if value is not None and value.instruction.raw_opcode in {
            "broadcast",
            "broadcast-in-dim",
        }:
            if re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
            ) != [""]:
                return False
            value = graph.semantic(graph.operand(value, 0))
        if value is not None and value.instruction.raw_opcode == "bitcast":
            source = graph.semantic(graph.operand(value, 0))
            if (
                source is None
                or graph.shape(source) != (("pred", (1, 1)),)
                or graph.shape(value) != (("pred", ()),)
                or not exact_layout(source, tpu_layout=tpu_layout)
                or not exact_layout(value, tpu_layout=tpu_layout)
            ):
                return False
            value = source
        return bool(
            value is not None
            and value.instruction.raw_opcode == "is-finite"
            and len(value.instruction.operand_names) == 1
            and exact_embedding_scalar(
                graph.operand(value, 0), tpu_layout=tpu_layout
            )
        )

    selected = graph.semantic(graph.operand(graph.value(reduction), 0))
    tpu_layout = bool(selected is not None and graph.shape(selected) == bf16_m32)
    if (
        selected is None
        or selected.instruction.raw_opcode != "select"
        or graph.shape(selected) not in m32_shapes
        or not exact_layout(selected, tpu_layout=tpu_layout)
        or len(selected.instruction.operand_names) != 3
        or not exact_finite_predicate(
            graph.operand(selected, 0), tpu_layout=tpu_layout
        )
        or not exact_owner_select(
            graph.operand(selected, 1), tpu_layout=tpu_layout
        )
        or not exact_layout(
            graph.semantic(graph.operand(selected, 2)),
            tpu_layout=tpu_layout,
        )
        or not exact_zero(graph.operand(selected, 2))
    ):
        raise BenchmarkValidationError(
            "accepted attention collective input/embedding guard drifted"
        )
    return {
        "exact_accepted_attention_input": True,
        "exact_attention_embedding_guard": True,
        "exact_attention_owner_rank": 0,
    }


def _validate_split_predense_value_flow(
    report: Any,
    scheduled: Any,
    gate_convolution: Any,
    *,
    preceding_attention_reduction: Any | None = None,
    accepted_embedding_reduction: Any | None = None,
    accepted_attention_reduction: Any | None = None,
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
    accepted_source_context = bool(
        accepted_embedding_reduction is not None
        and accepted_attention_reduction is not None
    )
    if (accepted_embedding_reduction is None) != (
        accepted_attention_reduction is None
    ):
        raise BenchmarkValidationError(
            "integrated pre-dense source reductions are incomplete"
        )
    expected_entry_parameters = (
        {
            0: bf16_m1,
            1: bf16_m1,
            2: (("pred", (32,)),),
            3: (("bf16", (6144,)),),
            4: (("f8e4m3fn", (1, 1, 6144, 768)),),
            5: (("f32", (1, 1, 48, 768)),),
            6: (("f8e4m3fn", (1, 1, 384, 6144)),),
            7: (("f32", (1, 1, 3, 6144)),),
            8: (("bf16", (6144,)),),
        }
        if accepted_source_context
        else {
        0: bf16_m1,
        1: bf16_m1,
        2: (("bf16", (6144,)),),
        3: (("f8e4m3fn", (1, 1, 6144, 768)),),
        4: (("f32", (1, 1, 48, 768)),),
        5: (("f8e4m3fn", (1, 1, 384, 6144)),),
        6: (("f32", (1, 1, 3, 6144)),),
        7: (("bf16", (6144,)),),
        }
    )
    if entry_parameters != expected_entry_parameters:
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

    def exact_predicate_broadcast(
        value: _ResolvedHloValue | None,
    ) -> bool:
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode
            in {"broadcast", "broadcast-in-dim"}
            and graph.shape(value) == (("pred", (32, 6144)),)
            and re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
            )
            == ["0"]
            and graph.exact_parameter(
                graph.operand(value, 0), 2, (("pred", (32,)),)
            )
        )

    def exact_nan(value: _ResolvedHloValue | None) -> bool:
        value = graph.identity(value)
        while value is not None and value.instruction.raw_opcode in {
            "broadcast",
            "broadcast-in-dim",
        }:
            value = graph.identity(graph.operand(value, 0))
        return bool(
            value is not None
            and value.instruction.raw_opcode == "constant"
            and re.findall(
                r"\bconstant\(([^)]*)\)", graph.clean(value.instruction)
            )
            == ["nan"]
        )

    def exact_external_f32(
        value: _ResolvedHloValue | None, reduction: Any
    ) -> bool:
        value = graph.semantic(value)
        if value is None:
            return False
        if value.instruction.raw_opcode == "convert" and graph.shape(value) == f32_m32:
            return graph.exact_external(graph.operand(value, 0), reduction)
        return bool(
            graph.shape(value) == f32_m32
            and graph.exact_external(value, reduction)
        )

    def exact_selected_embedding_f32(
        value: _ResolvedHloValue | None,
    ) -> bool:
        value = graph.semantic(value)
        lifted_bf16 = False
        if (
            value is not None
            and value.instruction.raw_opcode == "convert"
            and graph.shape(value) == f32_m32
        ):
            value = graph.semantic(graph.operand(value, 0))
            lifted_bf16 = bool(
                value is not None and graph.shape(value) == bf16_m32
            )
        if (
            value is None
            or value.instruction.raw_opcode != "select"
            or graph.shape(value) not in {f32_m32, bf16_m32}
            or len(value.instruction.operand_names) != 3
            or not exact_predicate_broadcast(graph.operand(value, 0))
        ):
            return False
        branches = [graph.operand(value, index) for index in (1, 2)]
        return bool(
            (
                graph.exact_external(
                    branches[0], accepted_embedding_reduction
                )
                if lifted_bf16
                else exact_external_f32(
                    branches[0], accepted_embedding_reduction
                )
            )
            and exact_nan(branches[1])
        )

    def exact_sum(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "add", f32_m32)
        if value is None or len(value.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        if accepted_source_context:
            return any(
                exact_external_f32(
                    operands[index], accepted_attention_reduction
                )
                and exact_selected_embedding_f32(operands[1 - index])
                for index in range(2)
            )
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
                graph.operand(broadcast, 0),
                3 if accepted_source_context else 2,
                (("bf16", (6144,)),),
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
        "accepted_source_context": accepted_source_context,
        "predense_gate_caller": caller.name,
        "predense_gate_convolution": gate_convolution.name,
    }


def integrated_dense_rms_hlo_policy(
    member_device_ids: Sequence[int],
    *,
    preceding_attention_collective: bool = False,
    accepted_source_context: bool = False,
    native_source_context: bool = False,
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
    if not isinstance(accepted_source_context, bool):
        raise BenchmarkValidationError(
            "accepted-source-context flag must be boolean"
        )
    if not isinstance(native_source_context, bool):
        raise BenchmarkValidationError(
            "native-source-context flag must be boolean"
        )
    if sum(
        bool(value)
        for value in (
            preceding_attention_collective,
            accepted_source_context,
            native_source_context,
        )
    ) > 1:
        raise BenchmarkValidationError(
            "integrated source-context collective scopes are disjoint"
        )
    patterns = (
        r"native_source_context_embedding_collective",
        r"native_source_context_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    ) if native_source_context else (
        r"accepted_source_context_embedding_collective",
        r"accepted_source_context_attention_collective",
        r"integrated_dense_rms_strategy_nd_collective",
    ) if accepted_source_context else (
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
                "all-reduce",
                3
                if accepted_source_context or native_source_context
                else 2
                if preceding_attention_collective
                else 1,
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
    accepted_source_context: bool = False,
    native_source_context: bool = False,
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
    if not isinstance(accepted_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense accepted-source flag must be boolean"
        )
    if not isinstance(native_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense native-source flag must be boolean"
        )
    if accepted_source_context and native_source_context:
        raise BenchmarkValidationError(
            "accepted and native source contexts are disjoint"
        )
    if accepted_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "accepted source context is a distinct integrated discriminator"
        )
    if native_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "native source context is a distinct integrated discriminator"
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
        INTEGRATED_DENSE_NATIVE_SOURCE_STABLEHLO_SHA256
        if native_source_context
        else INTEGRATED_DENSE_ACCEPTED_SOURCE_STABLEHLO_SHA256
        if accepted_source_context
        else INTEGRATED_DENSE_PREDENSE_SPLIT_RMS_STABLEHLO_SHA256
        if split_predense_rms
        else INTEGRATED_DENSE_ORDINAL_RMS_STABLEHLO_SHA256
        if preceding_attention_collective
        else INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256
        if split_layer1_rms
        else INTEGRATED_DENSE_RMS_STABLEHLO_SHA256
    )
    digest = sha256(stablehlo.encode()).hexdigest()
    if native_source_context and not expected:
        raise BenchmarkValidationError(
            "native source StableHLO is not pinned: " f"found={digest}"
        )
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
    if accepted_source_context:
        result.update(
            {
                "accepted_source_context": True,
                "exact_embedding_collective": True,
                "exact_validity_predicate": True,
                "exact_attention_collective": True,
                "exact_direct_layer1_recompute": True,
            }
        )
    if native_source_context:
        result.update(
            {
                "native_source_context": True,
                "exact_embedding_collective": True,
                "exact_validity_predicate": True,
                "exact_attention_collective": True,
                "exact_native_embedding_lookup": True,
                "exact_native_attention_projection": True,
                "exact_direct_layer1_recompute": True,
            }
        )
    return result


def _validate_native_source_context_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
) -> Mapping[str, Any]:
    """Validate the one acquired native-source TPU executable exactly.

    The complete optimized graph is byte-pinned first.  The structural checks
    below make the accepted executable auditable; the full digest is what makes
    every source edge, fusion body, physical layout and backend configuration
    fail closed rather than admitting an unreviewed compiler variant.
    """

    digest = sha256(optimized_hlo.encode()).hexdigest()
    if digest != INTEGRATED_DENSE_NATIVE_SOURCE_OPTIMIZED_HLO_SHA256:
        raise BenchmarkValidationError(
            "integrated native-source optimized HLO SHA-256 drifted: "
            f"expected={INTEGRATED_DENSE_NATIVE_SOURCE_OPTIMIZED_HLO_SHA256} "
            f"found={digest}"
        )
    members = tuple(int(value) for value in member_device_ids)
    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        integrated_dense_rms_hlo_policy(
            members,
            native_source_context=True,
        ),
    )
    report.raise_for_violations()
    module = report.module
    if module.num_partitions != 32 or len(module.instructions) != 602:
        raise BenchmarkValidationError(
            "integrated native-source module geometry drifted"
        )

    async_collective_opcodes = {
        f"{opcode}-{suffix}"
        for opcode in (
            "all-gather",
            "all-reduce",
            "all-to-all",
            "collective-broadcast",
            "collective-permute",
            "reduce-scatter",
        )
        for suffix in ("start", "done")
    }
    if any(
        item.raw_opcode in async_collective_opcodes
        for item in module.instructions
    ):
        raise BenchmarkValidationError(
            "integrated native-source graph contains an async collective"
        )

    reductions = tuple(
        item for item in module.collectives if item.raw_opcode == "all-reduce"
    )
    reduction_scopes = tuple(
        next(
            (
                scope
                for scope in _NATIVE_SOURCE_COLLECTIVE_SCOPES
                if scope in (item.op_name or "").split("/")
            ),
            None,
        )
        for item in reductions
    )
    exact_reduction_shape = (("bf16", (32, 6144)),)
    if not (
        len(module.collectives) == 3
        and len(reductions) == 3
        and reduction_scopes == _NATIVE_SOURCE_COLLECTIVE_SCOPES
        and all(
            _shape_signature(item) == exact_reduction_shape
            and tuple(
                (shape.dtype, shape.dimensions)
                for shape in item.operand_shapes
            )
            == exact_reduction_shape
            and item.replica_groups == (members,)
            and item.use_global_device_ids is True
            for item in reductions
        )
    ):
        raise BenchmarkValidationError(
            "integrated native-source collective scope/value geometry drifted"
        )
    algorithms = tuple(
        dict(validate_strategy_nd_reduction(report, item))
        for item in reductions
    )

    entry = tuple(
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
    )
    inputs = tuple(
        item for item in entry if item.raw_opcode == "parameter"
    )
    observed_input_schema = tuple(
        sorted(
            (
                int(item.operand_names[0])
                if len(item.operand_names) == 1
                and item.operand_names[0].isdigit()
                else -1,
                item.op_name,
                _shape_signature(item),
            )
            for item in inputs
        )
    )
    if observed_input_schema != _NATIVE_SOURCE_INPUT_SCHEMA:
        raise BenchmarkValidationError(
            "integrated native-source input schema drifted"
        )

    def scoped(
        scope: str,
        opcode: str,
        shape: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> tuple[Any, ...]:
        return tuple(
            item
            for item in module.instructions
            if item.raw_opcode == opcode
            and scope in (item.op_name or "").split("/")
            and _shape_signature(item) == shape
        )

    embedding_gathers = scoped(
        "native_source_context_embedding_lookup",
        "gather",
        (("bf16", (32, 6144)),),
    )
    wuv_calls = tuple(
        item
        for item in module.instructions
        if item.raw_opcode == "custom-call"
        and _shape_signature(item) == (("bf16", (48, 8, 128)),)
        and item.name
        == "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.1"
        and (item.op_name or "").split("/")[-2:] == [
            "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
            "pallas_call",
        ]
        and 'custom_call_target="tpu_custom_call"' in item.raw_line
    )
    attention_convolutions = scoped(
        "native_source_context_attention_projection",
        "convolution",
        (("f32", (32, 6144)),),
    )
    dense_convolutions = tuple(
        item
        for item in module.instructions
        if item.raw_opcode == "convolution"
        and "integrated_dense_rms_contraction"
        in (item.op_name or "").split("/")
    )
    predense_reductions = scoped(
        "native_source_context_predense_reduction",
        "reduce",
        (("f32", (32,)),),
    )
    predense_rsqrt = scoped(
        "native_source_context_predense_reduction",
        "rsqrt",
        (("f32", (32,)),),
    )
    layer1_reductions = scoped(
        "native_source_context_layer1_reduction",
        "reduce",
        (("f32", (32,)),),
    )
    layer1_rsqrt = scoped(
        "native_source_context_layer1_reduction",
        "rsqrt",
        (("f32", (32,)),),
    )
    roots = tuple(
        item
        for item in entry
        if item.raw_line.lstrip().startswith("ROOT ")
    )
    if not (
        len(embedding_gathers) == 1
        and len(wuv_calls) == 1
        and len(attention_convolutions) == 1
        and tuple(_shape_signature(item) for item in dense_convolutions)
        == (
            (("f32", (32, 768)),),
            (("f32", (32, 6144)),),
        )
        and len(predense_reductions) == 1
        and len(predense_rsqrt) == 1
        and len(layer1_reductions) == 1
        and len(layer1_rsqrt) == 1
        and len(roots) == 1
        and roots[0].raw_opcode == "fusion"
        and _shape_signature(roots[0]) == (("u16", (1, 6144)),)
        and roots[0].op_name
        == "jit(integrated)/shard_map/integrated_dense_rms_live_row/bitcast_convert_type"
    ):
        raise BenchmarkValidationError(
            "integrated native-source live graph structure drifted"
        )
    if any(
        marker in optimized_hlo
        for marker in (
            "host_callback",
            "xla_python_cpu_callback",
            " outfeed(",
        )
    ):
        raise BenchmarkValidationError(
            "integrated native-source graph contains a host effect"
        )

    return {
        "collective_algorithms": list(algorithms),
        "collective_count": 3,
        "collective_scopes": list(_NATIVE_SOURCE_COLLECTIVE_SCOPES),
        "exact_attention_embedding_dependency": True,
        "exact_direct_layer1_recompute": True,
        "exact_final_layout_dense": True,
        "exact_graph_sha256": digest,
        "exact_input_schema": True,
        "exact_layer1_rms": True,
        "exact_live_result": True,
        "exact_native_attention_projection": True,
        "exact_native_embedding_lookup": True,
        "exact_predense_rms": True,
        "exact_strategy_nd_collectives": True,
        "exact_wuv_kernel": True,
        "live_rows": 1,
        "native_source_context": True,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": True,
        "performance_claim": False,
        "split_layer1_rms": False,
        "violations": [],
    }


def validate_integrated_dense_rms_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
    accepted_source_context: bool = False,
    native_source_context: bool = False,
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
    if not isinstance(accepted_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense accepted-source flag must be boolean"
        )
    if not isinstance(native_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense native-source flag must be boolean"
        )
    if accepted_source_context and native_source_context:
        raise BenchmarkValidationError(
            "accepted and native source contexts are disjoint"
        )
    if accepted_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "accepted source context is a distinct integrated discriminator"
        )
    if native_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "native source context is a distinct integrated discriminator"
        )
    if native_source_context:
        return _validate_native_source_context_hlo(
            optimized_hlo,
            members,
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
            accepted_source_context=accepted_source_context,
            native_source_context=native_source_context,
        ),
    )
    report.raise_for_violations()
    reductions = tuple(
        item
        for item in report.module.collectives
        if item.opcode == "all-reduce"
    )
    expected_reductions = (
        3
        if accepted_source_context
        else 2
        if preceding_attention_collective
        else 1
    )
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
    embedding_reductions = tuple(
        item
        for item in reductions
        if "accepted_source_context_embedding_collective"
        in (item.op_name or "").split("/")
    )
    accepted_attention_reductions = tuple(
        item
        for item in reductions
        if "accepted_source_context_attention_collective"
        in (item.op_name or "").split("/")
    )
    if len(dense_reductions) != 1 or (
        preceding_attention_collective
        and (
            len(attention_reductions) != 1
            or attention_reductions[0].index >= dense_reductions[0].index
        )
    ) or (not preceding_attention_collective and attention_reductions) or (
        accepted_source_context
        and (
            len(embedding_reductions) != 1
            or len(accepted_attention_reductions) != 1
            or not (
                embedding_reductions[0].index
                < accepted_attention_reductions[0].index
                < dense_reductions[0].index
            )
        )
    ) or (
        not accepted_source_context
        and (embedding_reductions or accepted_attention_reductions)
    ):
        raise BenchmarkValidationError(
            "integrated dense collective scope/order drifted"
        )
    reduction = dense_reductions[0]
    attention_reduction = (
        attention_reductions[0] if preceding_attention_collective else None
    )
    embedding_reduction = (
        embedding_reductions[0] if accepted_source_context else None
    )
    accepted_attention_reduction = (
        accepted_attention_reductions[0]
        if accepted_source_context
        else None
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
    accepted_source_inputs = (
        {
            **_validate_preceding_attention_input(
                report, embedding_reduction, parameter_index=1
            ),
            **_validate_accepted_attention_input(
                report,
                accepted_attention_reduction,
                embedding_reduction,
            ),
        }
        if accepted_source_context
        else {}
    )
    embedding_algorithm = (
        validate_strategy_nd_reduction(report, embedding_reduction)
        if embedding_reduction is not None
        else None
    )
    accepted_attention_algorithm = (
        validate_strategy_nd_reduction(report, accepted_attention_reduction)
        if accepted_attention_reduction is not None
        else None
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
        if (
            "accepted_source_context_layer1_norm"
            if accepted_source_context
            else "integrated_dense_rms_layer1_norm"
        )
        in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if len(scheduled) != 1:
        raise BenchmarkValidationError(
            "integrated dense RMS accepted layer-1 schedule drifted"
        )
    expected_schedule_shape = (
        (("f32", (32,)),)
        if split_layer1_rms or accepted_source_context
        else (("f32", (32,)), ("f32", (32, 6144)))
    )
    if _shape_signature(scheduled[0]) != expected_schedule_shape:
        raise BenchmarkValidationError(
            "integrated dense RMS layer-1 schedule form drifted"
        )
    predense_scheduled = tuple(
        item
        for item in entry
        if (
            "accepted_source_context_predense_norm"
            if accepted_source_context
            else "integrated_dense_rms_predense_norm"
        )
        in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if split_predense_rms or accepted_source_context:
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
        accepted_source_context=accepted_source_context,
    )
    required_contraction = (
        "accepted_down_schedule",
        "accepted_gate_up_schedule",
        "exact_accepted_kernel_geometry",
        "exact_accepted_weight_layout",
        "exact_activation_graph",
        "exact_collective_input_binding",
        "exact_gate_dequant_fusion_boundary",
        "exact_gate_singleton_external_boundary",
        "exact_packed_weight_lineage",
        "exact_result_binding",
    )
    if not accepted_source_context:
        required_contraction += ("exact_carried_residual_binding",)
    if not contraction.get("passed") or not all(
        contraction.get(name) is True for name in required_contraction
    ):
        raise BenchmarkValidationError(
            f"integrated dense contraction HLO drifted: {contraction}"
        )
    predense_flow: Mapping[str, Any] = {}
    if split_predense_rms or accepted_source_context:
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
            accepted_embedding_reduction=embedding_reduction,
            accepted_attention_reduction=accepted_attention_reduction,
        )
    exact_rms = _validate_exact_dense_rms_value_flow(
        report,
        reduction,
        scheduled[0],
        roots[0],
        integrated_dense=True,
        require_split_output_fusion=(
            split_layer1_rms or accepted_source_context
        ),
        preceding_attention_reduction=attention_reduction,
        accepted_embedding_reduction=embedding_reduction,
        accepted_attention_reduction=accepted_attention_reduction,
    )
    if (
        exact_rms.get("residual_source_mode")
        != (
            "accepted_embedding_predicate_plus_attention"
            if accepted_source_context
            else "integrated_attention_plus_combined_residual"
        )
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
            (split_layer1_rms or accepted_source_context)
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
    if accepted_source_context:
        result.update(
            {
                "accepted_source_context": True,
                "embedding_collective": embedding_reduction.name,
                "embedding_collective_algorithm": dict(
                    embedding_algorithm
                ),
                "accepted_attention_collective": (
                    accepted_attention_reduction.name
                ),
                "accepted_attention_collective_algorithm": dict(
                    accepted_attention_algorithm
                ),
                **accepted_source_inputs,
                "accepted_predense_scheduled_reduction": (
                    predense_scheduled[0].name
                ),
                **predense_flow,
            }
        )
    return result
