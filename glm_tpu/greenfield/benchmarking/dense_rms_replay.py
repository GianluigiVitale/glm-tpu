"""Model-free replay of the accepted dense combine through layer-1 RMSNorm."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.reference.rmsnorm import fused_add_rms_norm
from ..sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    HloInstruction,
    HloLintReport,
    lint_hlo,
    parse_hlo_module,
)
from ..sharding.stablehlo_dense_convolution import (
    validate_strategy_nd_dense_rms_stablehlo,
)
from .association_fingerprint import (
    ACCEPTED_DECODE_RESULT_LAYOUT,
    STRATEGY_ND_ALGORITHM,
    array_sha256,
    validate_strategy_nd_reduction,
)


DENSE_RMS_SOURCE_NPZ_SHA256 = (
    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
)
DENSE_RMS_SOURCE_RUNNER_SHA256 = (
    "d22b35f664409485b697fdb579665dc4b1ecf42683a2aaff9e9a4d7481ee8343"
)
DENSE_RMS_SOURCE_SUMMARY_SHA256 = (
    "c349b5fd458f34986d8cc59c0f026af6b0a4c4e998f8691d6f6aa83a9b55e416"
)
DENSE_RMS_SOURCE_SUCCESS_SHA256 = (
    "6cac897695fc1e78d0a10c0e36c993cffd281c6a88721d8955fa470bd44b0b85"
)
DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256 = (
    "5ffef6b346754dd7e8cc5953a81f1d4f7a14235fbd567d1b2302505f2e9d60d8"
)
DENSE_RMS_SOURCE_TAG = (
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z"
)
DENSE_RMS_SOURCE_CODE_HASH = "c0ef9525bcd893bdad377af2b56e5ebd5fb13b34"
DENSE_RMS_SOURCE_KEYS = (
    "accepted_layer1_normalized_bfloat16_bits",
    "compile_rows",
    "layer1_input_norm_bfloat16_bits",
    "normalized_mlp_bfloat16_bits",
    "post_attention_residual_bfloat16_bits",
    "dense_virtual_partials_bfloat16_bits",
    "post_attention_m32_bfloat16_bits",
    "attention_update_bfloat16_bits",
    "combined_residual_bfloat16_bits",
)
DENSE_RMS_ARRAY_RECORDS = {
    "accepted_layer1_normalized_bfloat16_bits": (
        (6144,), np.dtype(np.uint16),
        "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039",
    ),
    "compile_rows": (
        (1,), np.dtype(np.int32),
        "8d71b3faab8201459ad37ef499beb336ba88bdcfa0f51ee6f0a46ec3192d750a",
    ),
    "layer1_input_norm_bfloat16_bits": (
        (6144,), np.dtype(np.uint16),
        "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87",
    ),
    "normalized_mlp_bfloat16_bits": (
        (1, 6144), np.dtype(np.uint16),
        "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f",
    ),
    "post_attention_residual_bfloat16_bits": (
        (1, 6144), np.dtype(np.uint16),
        "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e",
    ),
    "dense_virtual_partials_bfloat16_bits": (
        (4, 8, 1, 6144), np.dtype(np.uint16),
        "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35",
    ),
    "post_attention_m32_bfloat16_bits": (
        (32, 6144), np.dtype(np.uint16),
        "f583581fe6cdd8f1cb437b7864070de9fdd39b83be01d2d4afe072a997042dc0",
    ),
    "attention_update_bfloat16_bits": (
        (1, 6144), np.dtype(np.uint16),
        "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7",
    ),
    "combined_residual_bfloat16_bits": (
        (1, 6144), np.dtype(np.uint16),
        "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3",
    ),
}


@dataclass(frozen=True, slots=True)
class DenseRmsInputs:
    post_attention_residual_bits: np.ndarray
    layer1_norm_bits: np.ndarray
    accepted_layer1_bits: np.ndarray


@dataclass(frozen=True, slots=True)
class CompiledStrategyNdDenseRmsReplay:
    compiled: Any
    input_sharding: Any
    replicated_sharding: Any
    member_device_ids: tuple[int, ...]
    stablehlo: str
    stablehlo_contract: Mapping[str, Any]
    optimized_hlo: str
    optimized_hlo_contract: Mapping[str, Any]
    collective_algorithm: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_dense_rms_inputs(path: Path) -> DenseRmsInputs:
    """Load the direct residual, norm, and target from the sealed bundle."""

    if not path.is_file() or _file_sha256(path) != DENSE_RMS_SOURCE_NPZ_SHA256:
        raise BenchmarkValidationError("dense RMS source NPZ SHA-256 drifted")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != DENSE_RMS_SOURCE_KEYS:
            raise BenchmarkValidationError("dense RMS source NPZ keys drifted")
        for name in payload.files:
            value = np.ascontiguousarray(payload[name])
            shape, dtype, digest = DENSE_RMS_ARRAY_RECORDS[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or _raw_sha256(value) != digest
            ):
                raise BenchmarkValidationError(
                    f"dense RMS source tensor drifted: {name}"
                )
            arrays[name] = value
    if arrays["compile_rows"].tolist() != [32]:
        raise BenchmarkValidationError("dense RMS source compile rows drifted")
    return DenseRmsInputs(
        post_attention_residual_bits=arrays[
            "post_attention_residual_bfloat16_bits"
        ],
        layer1_norm_bits=arrays["layer1_input_norm_bfloat16_bits"],
        accepted_layer1_bits=arrays[
            "accepted_layer1_normalized_bfloat16_bits"
        ],
    )


def strategy_nd_dense_rms_hlo_policy(
    member_device_ids: Sequence[int],
) -> HloContractPolicy:
    members = tuple(int(value) for value in member_device_ids)
    if members != tuple(range(32)):
        raise BenchmarkValidationError(
            "StrategyND dense RMS replay requires physical ids 0..31"
        )
    return HloContractPolicy(
        name="strategy-nd-dense-rms-replay",
        total_devices=32,
        repeated_region_patterns=(r"strategy_nd_dense_rms_collective",),
        maximum_repeated_collective_group_size=32,
        expected_repeated_replica_groups=(members,),
        expected_collectives=(CollectiveExpectation("all-reduce", 1),),
        partition_id_to_device_id=members,
        forbidden_row_width_pairs=(),
        allow_full_pod_repeated_collectives=True,
    )


def _shape_signature(value: HloInstruction) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((shape.dtype, shape.dimensions) for shape in value.result_shapes)


def _computation_base(value: str) -> str:
    return value.split(" ", 1)[0].lstrip("%")


def _called_computation(value: HloInstruction) -> str | None:
    sanitized = re.sub(r'"(?:\\.|[^"\\])*"', '""', value.raw_line)
    sanitized = re.sub(r"/\*.*?\*/", "", sanitized)
    match = re.search(r"\bcalls=%?([^,\s}\]]+)", sanitized)
    return None if match is None else match.group(1)


@dataclass(frozen=True, slots=True)
class _ResolvedHloValue:
    instruction: HloInstruction
    bindings: tuple["_ResolvedHloValue", ...] = ()


class _ExactDenseRmsGraph:
    """Fusion-aware exact matcher for the diagnostic's live arithmetic."""

    def __init__(self, report: HloLintReport) -> None:
        self.by_key = {
            (item.computation, item.name): item
            for item in report.module.instructions
        }
        self.computations: dict[str, tuple[HloInstruction, ...]] = {}
        for item in report.module.instructions:
            name = _computation_base(item.computation)
            self.computations[name] = self.computations.get(name, ()) + (item,)

    @staticmethod
    def shape(value: _ResolvedHloValue) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return _shape_signature(value.instruction)

    @staticmethod
    def _parameter_index(value: HloInstruction) -> int | None:
        match = re.search(r"\bparameter\(([0-9]+)\)", value.raw_line)
        return None if match is None else int(match.group(1))

    @staticmethod
    def _tuple_index(value: HloInstruction) -> int | None:
        clean = re.sub(r'"(?:\\.|[^"\\])*"', '""', value.raw_line)
        clean = re.sub(r"/\*.*?\*/", "", clean)
        matches = re.findall(r"\bindex=([0-9]+)", clean)
        return int(matches[0]) if len(matches) == 1 else None

    @staticmethod
    def clean(value: HloInstruction) -> str:
        result = re.sub(r'"(?:\\.|[^"\\])*"', '""', value.raw_line)
        return re.sub(r"/\*.*?\*/", "", result)

    @staticmethod
    def exact_prefix(value: HloInstruction, signature: str) -> bool:
        line = value.raw_line.lstrip()
        if line.startswith("ROOT "):
            line = line[len("ROOT ") :]
        return line.startswith(
            f"{value.name} = {signature} {value.raw_opcode}("
        )

    def value(self, instruction: HloInstruction) -> _ResolvedHloValue:
        return _ResolvedHloValue(instruction)

    def operand(
        self, value: _ResolvedHloValue, index: int
    ) -> _ResolvedHloValue | None:
        if index >= len(value.instruction.operand_names):
            return None
        item = self.by_key.get(
            (value.instruction.computation, value.instruction.operand_names[index])
        )
        if item is None:
            return None
        result = _ResolvedHloValue(item, value.bindings)
        return self.resolve_parameter(result)

    def resolve_parameter(
        self, value: _ResolvedHloValue
    ) -> _ResolvedHloValue:
        seen: set[tuple[str, str]] = set()
        while (
            value.instruction.raw_opcode == "parameter"
            and not value.instruction.computation.startswith("ENTRY ")
        ):
            key = (value.instruction.computation, value.instruction.name)
            if key in seen:
                break
            seen.add(key)
            index = self._parameter_index(value.instruction)
            if index is None or index >= len(value.bindings):
                break
            value = value.bindings[index]
        return value

    def expand_fusion(
        self, value: _ResolvedHloValue
    ) -> _ResolvedHloValue | None:
        value = self.resolve_parameter(value)
        if value.instruction.raw_opcode != "fusion":
            return value
        callee = self.computations.get(
            _called_computation(value.instruction) or "", ()
        )
        roots = tuple(
            item for item in callee if item.raw_line.lstrip().startswith("ROOT ")
        )
        bindings = tuple(
            operand
            for index in range(len(value.instruction.operand_names))
            if (operand := self.operand(value, index)) is not None
        )
        if len(roots) != 1 or len(bindings) != len(value.instruction.operand_names):
            return None
        return _ResolvedHloValue(roots[0], bindings)

    def semantic(self, value: _ResolvedHloValue | None) -> _ResolvedHloValue | None:
        """Cross only exact call/tuple/async-copy identity boundaries."""

        seen: set[tuple[str, str, int]] = set()
        while value is not None:
            value = self.resolve_parameter(value)
            key = (
                value.instruction.computation,
                value.instruction.name,
                len(value.bindings),
            )
            if key in seen:
                return None
            seen.add(key)
            if value.instruction.raw_opcode == "fusion":
                value = self.expand_fusion(value)
                continue
            if value.instruction.raw_opcode == "get-tuple-element":
                index = self._tuple_index(value.instruction)
                source = self.operand(value, 0)
                if index is None or source is None:
                    return None
                source = self.semantic(source)
                if source is None or source.instruction.raw_opcode != "tuple":
                    return None
                value = self.operand(source, index)
                continue
            if value.instruction.raw_opcode == "copy-done":
                start = self.operand(value, 0)
                if start is None or start.instruction.raw_opcode != "copy-start":
                    return None
                value = self.operand(start, 0)
                continue
            return value
        return None

    def identity(self, value: _ResolvedHloValue | None) -> _ResolvedHloValue | None:
        """Unwrap only logically shape-preserving, non-arithmetic identities."""

        seen: set[tuple[str, str, int]] = set()
        while (value := self.semantic(value)) is not None:
            key = (
                value.instruction.computation,
                value.instruction.name,
                len(value.bindings),
            )
            if key in seen:
                return None
            seen.add(key)
            opcode = value.instruction.raw_opcode
            if opcode not in {"copy", "optimization-barrier", "reshape"}:
                return value
            source = self.operand(value, 0)
            if source is None or len(value.instruction.operand_shapes) != 1:
                return None
            operand_shape = value.instruction.operand_shapes[0]
            result_shape = value.instruction.result_shapes[0]
            if operand_shape.dtype != result_shape.dtype:
                return None
            if opcode == "reshape" and tuple(
                dimension for dimension in operand_shape.dimensions if dimension != 1
            ) != tuple(
                dimension for dimension in result_shape.dimensions if dimension != 1
            ):
                return None
            if opcode != "reshape" and operand_shape.dimensions != result_shape.dimensions:
                return None
            value = source
        return None

    def exact_parameter(
        self,
        value: _ResolvedHloValue | None,
        index: int,
        shape: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> bool:
        value = self.identity(value)
        return bool(
            value is not None
            and value.instruction.computation.startswith("ENTRY ")
            and value.instruction.raw_opcode == "parameter"
            and self._parameter_index(value.instruction) == index
            and self.shape(value) == shape
        )

    def exact_external(
        self,
        value: _ResolvedHloValue | None,
        expected: HloInstruction,
    ) -> bool:
        seen: set[tuple[str, str, int]] = set()
        while value is not None:
            value = self.resolve_parameter(value)
            key = (
                value.instruction.computation,
                value.instruction.name,
                len(value.bindings),
            )
            if key in seen:
                return False
            seen.add(key)
            if (
                value.instruction.computation == expected.computation
                and value.instruction.name == expected.name
            ):
                return True
            if value.instruction.raw_opcode == "copy-done":
                start = self.operand(value, 0)
                if start is None or start.instruction.raw_opcode != "copy-start":
                    return False
                value = self.operand(start, 0)
                continue
            if value.instruction.raw_opcode not in {
                "copy",
                "optimization-barrier",
                "reshape",
            }:
                return False
            source = self.operand(value, 0)
            if source is None or len(value.instruction.operand_shapes) != 1:
                return False
            operand_shape = value.instruction.operand_shapes[0]
            result_shape = value.instruction.result_shapes[0]
            if (
                operand_shape.dtype != result_shape.dtype
                or tuple(
                    dimension
                    for dimension in operand_shape.dimensions
                    if dimension != 1
                )
                != tuple(
                    dimension
                    for dimension in result_shape.dimensions
                    if dimension != 1
                )
            ):
                return False
            value = source
        return False

    def exact_constant(
        self, value: _ResolvedHloValue | None, expected: np.float32
    ) -> bool:
        seen: set[tuple[str, str, int]] = set()
        while (value := self.identity(value)) is not None:
            key = (
                value.instruction.computation,
                value.instruction.name,
                len(value.bindings),
            )
            if key in seen:
                return False
            seen.add(key)
            if value.instruction.raw_opcode in {"broadcast", "broadcast-in-dim"}:
                value = self.operand(value, 0)
                continue
            if value.instruction.raw_opcode != "constant":
                return False
            match = re.search(
                r"\bconstant\(([-+]?(?:[0-9]+(?:\.[0-9]*)?|"
                r"\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\)",
                self.clean(value.instruction),
            )
            return bool(
                match is not None
                and np.float32(float(match.group(1))) == expected
            )
        return False

    def exact_reducer(self, value: _ResolvedHloValue) -> bool:
        match = re.search(
            r"\bto_apply=%?([^,\s}\]]+)", self.clean(value.instruction)
        )
        if match is None:
            return False
        computation = self.computations.get(match.group(1), ())
        parameters = tuple(
            item for item in computation if item.raw_opcode == "parameter"
        )
        roots = tuple(
            item
            for item in computation
            if item.raw_line.lstrip().startswith("ROOT ")
        )
        return bool(
            len(computation) == 3
            and len(parameters) == 2
            and {self._parameter_index(item) for item in parameters} == {0, 1}
            and len(roots) == 1
            and roots[0].raw_opcode == "add"
            and set(roots[0].operand_names) == {item.name for item in parameters}
            and _shape_signature(roots[0]) == (("f32", ()),)
        )

    def exact_row0(
        self,
        value: _ResolvedHloValue | None,
        source_match: Any,
        *,
        dtype: str,
    ) -> bool:
        value = self.semantic(value)
        if value is None or value.instruction.raw_opcode != "slice":
            return False
        attributes = re.findall(
            r"\bslice=(\{\[[^}\n]+\]\})",
            re.sub(r"\s+", "", self.clean(value.instruction)),
        )
        return bool(
            attributes == ["{[0:1],[0:6144]}"]
            and self.shape(value) == ((dtype, (1, 6144)),)
            and source_match(self.operand(value, 0), rows=32)
        )


def _exact_accepted_rms_schedule(value: HloInstruction) -> bool:
    signature = _shape_signature(value)
    if value.raw_opcode != "fusion" or signature not in {
        (("f32", (32,)),),
        (("f32", (32,)), ("f32", (32, 6144))),
    }:
        return False
    marker = "backend_config="
    if marker not in value.raw_line:
        return False
    try:
        config = json.loads(value.raw_line.split(marker, 1)[1])
    except json.JSONDecodeError:
        return False
    window = config.get("window_config", {})
    megacore = config.get("megacore_config", {})
    common = bool(
        window.get("kernel_window_bounds") == []
        and window.get("input_window_bounds") == []
        and window.get("cost_model_type") == "COST_MODEL_TYPE_INVALID"
        and window.get("is_mask") is False
        and window.get("pad_input_on_minor_dim") == "0"
        and window.get("pad_output_on_minor_dim") == "0"
        and megacore.get("megacore_allreduce_bytes") == "4096"
    )
    if not common:
        return False
    if signature == (("f32", (32,)),):
        return bool(
            window.get("output_window_bounds") == ["2", "48"]
            and window.get("iteration_bounds") == ["2", "1"]
            and megacore.get("megacore_split_dim") == "0"
        )
    return bool(
        window.get("output_window_bounds") == ["4", "24"]
        and window.get("iteration_bounds") == ["1", "2"]
        and megacore.get("megacore_split_dim") == "1"
    )


def _validate_exact_dense_rms_value_flow(
    report: HloLintReport,
    reduction: HloInstruction,
    scheduled: HloInstruction,
    root: HloInstruction,
    *,
    integrated_dense: bool = False,
    require_split_output_fusion: bool = False,
    preceding_attention_reduction: HloInstruction | None = None,
) -> Mapping[str, Any]:
    """Bind sealed inputs through the exact live collective/RMS result."""

    if not isinstance(require_split_output_fusion, bool):
        raise BenchmarkValidationError(
            "dense RMS split-output ownership flag must be boolean"
        )
    if preceding_attention_reduction is not None and not (
        integrated_dense and require_split_output_fusion
    ):
        raise BenchmarkValidationError(
            "preceding attention reduction requires integrated split RMS"
        )
    graph = _ExactDenseRmsGraph(report)
    f32_m32 = (("f32", (32, 6144)),)
    bf16_m32 = (("bf16", (32, 6144)),)
    bf16_m1 = (("bf16", (1, 6144)),)
    f32_m1 = (("f32", (1, 6144)),)
    entry_parameters = {
        int(match.group(1)): _shape_signature(item)
        for item in report.module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_opcode == "parameter"
        and (match := re.search(r"\bparameter\(([0-9]+)\)", item.raw_line))
        is not None
    }
    direct_residual = not integrated_dense and entry_parameters == {
        0: (("u16", (1, 32, 6144)),),
        1: (("bf16", (1, 6144)),),
        2: (("bf16", (6144,)),),
    }
    hybrid_control = not integrated_dense and entry_parameters == {
        0: (("u16", (1, 32, 6144)),),
        1: (("bf16", (1, 6144)),),
        2: (("bf16", (1, 6144)),),
        3: (("bf16", (6144,)),),
    }
    integrated_control = integrated_dense and entry_parameters == {
        0: (("bf16", (1, 6144)),),
        1: (("bf16", (1, 6144)),),
        2: (("bf16", (6144,)),),
        3: (("f8e4m3fn", (1, 1, 6144, 768)),),
        4: (("f32", (1, 1, 48, 768)),),
        5: (("f8e4m3fn", (1, 1, 384, 6144)),),
        6: (("f32", (1, 1, 3, 6144)),),
        7: (("bf16", (6144,)),),
    }
    if not (direct_residual or hybrid_control or integrated_control):
        raise BenchmarkValidationError("dense RMS replay ENTRY inputs drifted")
    pin_integrated_split_layouts = bool(
        integrated_control and require_split_output_fusion
    )
    accepted_bf16_m32_prefix = (
        "bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}"
    )
    accepted_bf16_m1_prefix = "bf16[1,6144]{1,0:T(2,128)(2,1)}"
    accepted_f32_m1_prefix = "f32[1,6144]{1,0:T(1,128)}"
    accepted_u16_m1_prefix = "u16[1,6144]{1,0:T(2,128)(2,1)}"

    def exact_split_prefix(
        value: _ResolvedHloValue, signature: str
    ) -> bool:
        return bool(
            not pin_integrated_split_layouts
            or graph.exact_prefix(value.instruction, signature)
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
                graph.operand(value, 0),
                parameter_index,
                ((dtype, (1, 6144)),),
            )
        )

    def exact_padded_f32(
        value: _ResolvedHloValue | None, parameter_index: int
    ) -> bool:
        value = graph.semantic(value)
        if value is None:
            return False
        if value.instruction.raw_opcode == "convert" and graph.shape(value) == f32_m32:
            return exact_pad(graph.operand(value, 0), parameter_index, "bf16")
        if value.instruction.raw_opcode == "pad" and graph.shape(value) == f32_m32:
            source = graph.semantic(graph.operand(value, 0))
            if (
                source is None
                or source.instruction.raw_opcode != "convert"
                or graph.shape(source) != f32_m1
                or not graph.exact_parameter(
                    graph.operand(source, 0), parameter_index, bf16_m1
                )
            ):
                return False
            attributes = re.findall(
                r"\bpadding=([^,\s}]+)", graph.clean(value.instruction)
            )
            return bool(
                attributes == ["0_31x0_0"]
                and graph.exact_constant(
                    graph.operand(value, 1), np.float32(0.0)
                )
            )
        return False

    def exact_hybrid_carried(value: _ResolvedHloValue | None) -> bool:
        value = semantic_opcode(value, "convert", bf16_m32)
        if value is None or not exact_split_prefix(
            value, accepted_bf16_m32_prefix
        ):
            return False
        addition = semantic_opcode(graph.operand(value, 0), "add", f32_m32)
        if addition is None or len(addition.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(addition, index) for index in range(2)]
        if integrated_control and preceding_attention_reduction is not None:
            def exact_attention_collective_f32(
                candidate: _ResolvedHloValue | None,
            ) -> bool:
                candidate = graph.semantic(candidate)
                return bool(
                    candidate is not None
                    and candidate.instruction.raw_opcode == "convert"
                    and graph.shape(candidate) == f32_m32
                    and graph.exact_external(
                        graph.operand(candidate, 0),
                        preceding_attention_reduction,
                    )
                )

            return bool(
                any(
                    exact_attention_collective_f32(operands[left])
                    and exact_padded_f32(operands[1 - left], 1)
                    for left in range(2)
                )
            )
        source_indexes = (0, 1) if integrated_control else (1, 2)
        return bool(
            any(
                exact_padded_f32(operands[left], source_indexes[0])
                and exact_padded_f32(
                    operands[1 - left], source_indexes[1]
                )
                for left in range(2)
            )
        )

    def exact_direct_residual_f32(
        value: _ResolvedHloValue | None,
    ) -> bool:
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode == "convert"
            and graph.shape(value) == f32_m32
            and exact_pad(graph.operand(value, 0), 1, "bf16")
        )

    reduction_value = graph.value(reduction)
    scheduled_signature = _shape_signature(scheduled)
    tuple_sum_consumed = [False]
    split_output_owners: dict[str, set[str]] = {
        "normalized": set(),
        "sum": set(),
        "weighted": set(),
    }

    def exact_schedule_projection(
        value: _ResolvedHloValue | None, index: int
    ) -> bool:
        if value is None:
            return False
        if scheduled_signature == (("f32", (32,)),):
            return index == 0 and graph.exact_external(value, scheduled)
        value = graph.resolve_parameter(value)
        return bool(
            value.instruction.raw_opcode == "get-tuple-element"
            and graph._tuple_index(value.instruction) == index
            and graph.exact_external(graph.operand(value, 0), scheduled)
        )

    def exact_collective_f32(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode == "convert"
            and graph.shape(value) == f32_m32
            and graph.exact_external(graph.operand(value, 0), reduction)
        )

    def exact_carried_f32(value: _ResolvedHloValue | None) -> bool:
        if direct_residual:
            return exact_direct_residual_f32(value)
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode == "convert"
            and graph.shape(value) == f32_m32
            and exact_hybrid_carried(graph.operand(value, 0))
        )

    def exact_row0_f32_source(
        value: _ResolvedHloValue | None,
        source_match: Any,
    ) -> bool:
        value = semantic_opcode(value, "convert", f32_m1)
        if value is None or not exact_split_prefix(value, accepted_f32_m1_prefix):
            return False
        row = graph.semantic(graph.operand(value, 0))
        if (
            row is None
            or row.instruction.raw_opcode != "slice"
            or graph.shape(row) != bf16_m1
            or not exact_split_prefix(row, accepted_bf16_m1_prefix)
            or re.findall(
                r"\bslice=(\{\[[^}\n]+\]\})",
                re.sub(r"\s+", "", graph.clean(row.instruction)),
            )
            != ["{[0:1],[0:6144]}"]
            or len(row.instruction.operand_names) != 1
        ):
            return False
        if pin_integrated_split_layouts:
            parameter = graph.by_key.get(
                (
                    row.instruction.computation,
                    row.instruction.operand_names[0],
                )
            )
            if not (
                parameter is not None
                and parameter.raw_opcode == "parameter"
                and graph.exact_prefix(parameter, accepted_bf16_m32_prefix)
            ):
                return False
        return bool(
            source_match(graph.operand(row, 0), rows=32)
        )

    def exact_collective_row0_f32(
        value: _ResolvedHloValue | None,
    ) -> bool:
        return exact_row0_f32_source(
            value,
            lambda candidate, *, rows: bool(
                rows == 32 and graph.exact_external(candidate, reduction)
            ),
        )

    def exact_carried_row0_f32(
        value: _ResolvedHloValue | None,
    ) -> bool:
        def exact_carried_bf16(
            candidate: _ResolvedHloValue | None, *, rows: int
        ) -> bool:
            if rows != 32:
                return False
            if direct_residual:
                return exact_pad(candidate, 1, "bf16")
            return exact_hybrid_carried(candidate)

        return exact_row0_f32_source(value, exact_carried_bf16)

    def exact_sum(
        value: _ResolvedHloValue | None,
        *,
        rows: int = 32,
        track_split_output: bool = False,
    ) -> bool:
        if rows == 1:
            if graph.exact_row0(
                value,
                lambda candidate, *, rows: exact_sum(
                    candidate,
                    rows=rows,
                    track_split_output=track_split_output,
                ),
                dtype="f32",
            ):
                return True
            if not require_split_output_fusion:
                return False
            recompute = semantic_opcode(value, "add", f32_m1)
            if (
                recompute is None
                or not exact_split_prefix(recompute, accepted_f32_m1_prefix)
                or len(recompute.instruction.operand_names) != 2
            ):
                return False
            operands = [graph.operand(recompute, index) for index in range(2)]
            exact = any(
                exact_collective_row0_f32(operands[left])
                and exact_carried_row0_f32(operands[1 - left])
                for left in range(2)
            )
            if exact and track_split_output:
                split_output_owners["sum"].add(
                    recompute.instruction.computation
                )
            return exact
        from_tuple_schedule = exact_schedule_projection(value, 1)
        value = semantic_opcode(value, "add", f32_m32)
        if value is None or len(value.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        exact = any(
            exact_collective_f32(operands[left])
            and exact_carried_f32(operands[1 - left])
            for left in range(2)
        )
        if exact and from_tuple_schedule:
            tuple_sum_consumed[0] = True
        if exact and track_split_output:
            split_output_owners["sum"].add(value.instruction.computation)
        return exact

    def exact_partial_input(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if (
            value is None
            or value.instruction.raw_opcode != "bitcast"
            or graph.shape(value) != bf16_m32
            or len(value.instruction.operand_names) != 1
        ):
            return False
        singleton = graph.semantic(graph.operand(value, 0))
        parameter = (
            graph.identity(graph.operand(singleton, 0))
            if singleton is not None
            and singleton.instruction.raw_opcode == "bitcast-convert"
            else None
        )
        return bool(
            singleton is not None
            and singleton.instruction.raw_opcode == "bitcast-convert"
            and graph.shape(singleton) == (("bf16", (1, 32, 6144)),)
            and graph.exact_prefix(
                singleton.instruction,
                "bf16[1,32,6144]{2,1,0:T(8,128)(2,1)}",
            )
            and graph.exact_parameter(
                graph.operand(singleton, 0),
                0,
                (("u16", (1, 32, 6144)),),
            )
            and parameter is not None
            and graph.exact_prefix(
                parameter.instruction,
                "u16[1,32,6144]{2,1,0:T(8,128)(2,1)}",
            )
            and graph.exact_prefix(
                value.instruction,
                f"bf16[32,6144]{ACCEPTED_DECODE_RESULT_LAYOUT}",
            )
        )

    if (
        len(reduction.operand_names) != 1
        or (
            not integrated_control
            and not exact_partial_input(graph.operand(reduction_value, 0))
        )
    ):
        raise BenchmarkValidationError(
            "dense RMS StrategyND input is not the exact sealed BF16 partial"
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

    scheduled_semantic = graph.identity(graph.value(scheduled))
    if (
        scheduled_semantic is not None
        and scheduled_semantic.instruction.raw_opcode == "tuple"
        and _shape_signature(scheduled)
        == (("f32", (32,)), ("f32", (32, 6144)))
    ):
        scheduled_semantic = graph.identity(
            graph.operand(scheduled_semantic, 0)
        )
    if (
        scheduled_semantic is None
        or scheduled_semantic.instruction.raw_opcode != "reduce"
        or graph.shape(scheduled_semantic) != (("f32", (32,)),)
        or len(scheduled_semantic.instruction.operand_names) != 2
        or re.findall(
            r"\bdimensions=\{([^}]*)\}",
            graph.clean(scheduled_semantic.instruction),
        )
        != ["1"]
        or not exact_square(graph.operand(scheduled_semantic, 0))
        or not graph.exact_constant(
            graph.operand(scheduled_semantic, 1), np.float32(0.0)
        )
        or not graph.exact_reducer(scheduled_semantic)
    ):
        raise BenchmarkValidationError(
            "dense RMS accepted scheduled reduction arithmetic drifted"
        )

    def exact_scheduled(value: _ResolvedHloValue | None) -> bool:
        return exact_schedule_projection(value, 0)

    def exact_mean(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if (
            value is None
            or graph.shape(value) not in {
                (("f32", (32,)),),
                (("f32", (32, 1)),),
            }
            or len(value.instruction.operand_names) != 2
        ):
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        if value.instruction.raw_opcode == "divide":
            return exact_scheduled(operands[0]) and graph.exact_constant(
                operands[1], np.float32(6144.0)
            )
        return bool(
            value.instruction.raw_opcode == "multiply"
            and any(
                exact_scheduled(operands[index])
                and graph.exact_constant(
                    operands[1 - index], np.float32(1.0 / 6144.0)
                )
                for index in range(2)
            )
        )

    def exact_rsqrt(value: _ResolvedHloValue | None) -> bool:
        value = graph.semantic(value)
        if (
            value is None
            or value.instruction.raw_opcode != "rsqrt"
            or graph.shape(value) != (("f32", (32,)),)
        ):
            return False
        variance = semantic_opcode(
            graph.operand(value, 0), "add", (("f32", (32,)),)
        )
        if variance is None or len(variance.instruction.operand_names) != 2:
            return False
        operands = [graph.operand(variance, index) for index in range(2)]
        return any(
            exact_mean(operands[index])
            and graph.exact_constant(operands[1 - index], np.float32(1.0e-5))
            for index in range(2)
        )

    def exact_rsqrt_broadcast(
        value: _ResolvedHloValue | None, *, rows: int
    ) -> bool:
        value = graph.semantic(value)
        expected_shape = (("f32", (rows, 6144)),)
        if (
            value is None
            or value.instruction.raw_opcode not in {"broadcast", "broadcast-in-dim"}
            or graph.shape(value) != expected_shape
            or (
                rows == 1
                and not exact_split_prefix(value, accepted_f32_m1_prefix)
            )
        ):
            return False
        dimensions = re.findall(
            r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
        )
        source = graph.semantic(graph.operand(value, 0))
        if rows == 32:
            return dimensions == ["0"] and exact_rsqrt(source)
        if (
            dimensions != ["0"]
            or source is None
            or source.instruction.raw_opcode != "bitcast"
            or graph.shape(source) != (("f32", (1,)),)
            or not graph.exact_prefix(
                source.instruction, "f32[1]{0:T(128)S(3)}"
            )
            or tuple(
                (shape.dtype, shape.dimensions)
                for shape in source.instruction.operand_shapes
            )
            != (("f32", (32,)),)
        ):
            return False
        producer = graph.operand(source, 0)
        if producer is None:
            return False
        producer = graph.resolve_parameter(producer)
        return bool(
            graph.exact_prefix(
                producer.instruction, "f32[32]{0:T(128)S(3)}"
            )
            and exact_rsqrt(producer)
        )

    def exact_normalized_f32(
        value: _ResolvedHloValue | None,
        *,
        rows: int,
        track_split_output: bool = False,
    ) -> bool:
        shape = (("f32", (rows, 6144)),)
        value = semantic_opcode(value, "multiply", shape)
        if (
            value is None
            or (
                rows == 1
                and not exact_split_prefix(value, accepted_f32_m1_prefix)
            )
            or len(value.instruction.operand_names) != 2
        ):
            return False
        operands = [graph.operand(value, index) for index in range(2)]
        for index in range(2):
            if exact_sum(
                operands[index],
                rows=rows,
                track_split_output=track_split_output,
            ) and exact_rsqrt_broadcast(operands[1 - index], rows=rows):
                if track_split_output:
                    split_output_owners["normalized"].add(
                        value.instruction.computation
                    )
                return True
        return False

    def exact_normalized_bf16(
        value: _ResolvedHloValue | None,
        *,
        rows: int,
        track_split_output: bool = False,
    ) -> bool:
        value = semantic_opcode(
            value, "convert", (("bf16", (rows, 6144)),)
        )
        return bool(
            value is not None
            and (
                rows != 1
                or exact_split_prefix(value, accepted_bf16_m1_prefix)
            )
            and exact_normalized_f32(
                graph.operand(value, 0),
                rows=rows,
                track_split_output=track_split_output,
            )
        )

    def exact_weight_f32(
        value: _ResolvedHloValue | None, *, rows: int
    ) -> bool:
        value = graph.semantic(value)
        if value is None:
            return False
        if value.instruction.raw_opcode == "convert":
            return bool(
                (
                    rows != 1
                    or exact_split_prefix(value, accepted_f32_m1_prefix)
                )
                and exact_weight_bf16(graph.operand(value, 0), rows=rows)
            )
        return False

    def exact_weight_bf16(
        value: _ResolvedHloValue | None, *, rows: int
    ) -> bool:
        value = graph.semantic(value)
        if (
            value is None
            or value.instruction.raw_opcode not in {"broadcast", "broadcast-in-dim"}
            or graph.shape(value) != (("bf16", (rows, 6144)),)
            or (
                rows == 1
                and not exact_split_prefix(value, accepted_bf16_m1_prefix)
            )
            or re.findall(
                r"\bdimensions=\{([^}]*)\}", graph.clean(value.instruction)
            )
            != ["1"]
        ):
            return False
        return graph.exact_parameter(
            graph.operand(value, 0),
            7 if integrated_control else 2 if direct_residual else 3,
            (("bf16", (6144,)),),
        )

    def exact_normalized_lift(
        value: _ResolvedHloValue | None,
        *,
        rows: int,
        track_split_output: bool = False,
    ) -> bool:
        value = graph.semantic(value)
        return bool(
            value is not None
            and value.instruction.raw_opcode == "convert"
            and graph.shape(value) == (("f32", (rows, 6144)),)
            and (
                rows != 1
                or exact_split_prefix(value, accepted_f32_m1_prefix)
            )
            and exact_normalized_bf16(
                graph.operand(value, 0),
                rows=rows,
                track_split_output=track_split_output,
            )
        )

    def exact_weighted_bf16(
        value: _ResolvedHloValue | None,
        *,
        rows: int,
        track_split_output: bool = False,
    ) -> bool:
        value = semantic_opcode(
            value, "convert", (("bf16", (rows, 6144)),)
        )
        if value is None or (
            rows == 1
            and not exact_split_prefix(value, accepted_bf16_m1_prefix)
        ):
            return False
        multiply = semantic_opcode(
            graph.operand(value, 0), "multiply", (("f32", (rows, 6144)),)
        )
        if (
            multiply is None
            or (
                rows == 1
                and not exact_split_prefix(multiply, accepted_f32_m1_prefix)
            )
            or len(multiply.instruction.operand_names) != 2
        ):
            return False
        operands = [graph.operand(multiply, index) for index in range(2)]
        for index in range(2):
            if exact_normalized_lift(
                operands[index],
                rows=rows,
                track_split_output=track_split_output,
            ) and exact_weight_f32(operands[1 - index], rows=rows):
                if track_split_output:
                    split_output_owners["weighted"].add(
                        multiply.instruction.computation
                    )
                return True
        return False

    def exact_final_bf16_row(value: _ResolvedHloValue | None) -> bool:
        if exact_weighted_bf16(
            value,
            rows=1,
            track_split_output=require_split_output_fusion,
        ):
            return True
        value = graph.semantic(value)
        if (
            value is not None
            and value.instruction.raw_opcode == "slice"
            and graph.shape(value) == bf16_m1
            and re.findall(
                r"\bslice=(\{\[[^}\n]+\]\})",
                re.sub(r"\s+", "", graph.clean(value.instruction)),
            )
            == ["{[0:1],[0:6144]}"]
            and exact_weighted_bf16(
                graph.operand(value, 0),
                rows=32,
                track_split_output=require_split_output_fusion,
            )
        ):
            return True
        # TPU/CPU may lift the already rounded M32 BF16 result, take row zero,
        # then round the row back to BF16 before the final bitcast-convert.
        value = semantic_opcode(value, "convert", bf16_m1)
        row = graph.semantic(graph.operand(value, 0)) if value is not None else None
        if (
            row is None
            or row.instruction.raw_opcode != "slice"
            or graph.shape(row) != f32_m1
            or re.findall(
                r"\bslice=(\{\[[^}\n]+\]\})",
                re.sub(r"\s+", "", graph.clean(row.instruction)),
            )
            != ["{[0:1],[0:6144]}"]
        ):
            return False
        lift = semantic_opcode(graph.operand(row, 0), "convert", f32_m32)
        return bool(
            lift is not None
            and exact_weighted_bf16(
                graph.operand(lift, 0),
                rows=32,
                track_split_output=require_split_output_fusion,
            )
        )

    if pin_integrated_split_layouts and not graph.exact_prefix(
        root, accepted_u16_m1_prefix
    ):
        raise BenchmarkValidationError(
            "dense RMS live ENTRY result layout drifted"
        )
    output = semantic_opcode(graph.value(root), "bitcast-convert", (("u16", (1, 6144)),))
    if output is not None and not exact_split_prefix(
        output, accepted_u16_m1_prefix
    ):
        output = None
    if output is None or not exact_final_bf16_row(graph.operand(output, 0)):
        output_operand = graph.semantic(
            graph.operand(output, 0) if output is not None else None
        )
        row_source = (
            graph.operand(output_operand, 0)
            if output_operand is not None
            and output_operand.instruction.raw_opcode == "slice"
            else None
        )
        weighted_round = graph.semantic(row_source)
        weighted_mul = (
            graph.semantic(graph.operand(weighted_round, 0))
            if weighted_round is not None
            and weighted_round.instruction.raw_opcode == "convert"
            else None
        )
        weighted_operands = (
            [graph.operand(weighted_mul, index) for index in range(2)]
            if weighted_mul is not None
            and weighted_mul.instruction.raw_opcode == "multiply"
            else [None, None]
        )
        normalized_lift = graph.semantic(weighted_operands[0])
        normalized_round = (
            graph.semantic(graph.operand(normalized_lift, 0))
            if normalized_lift is not None
            and normalized_lift.instruction.raw_opcode == "convert"
            else None
        )
        normalized_mul = (
            graph.semantic(graph.operand(normalized_round, 0))
            if normalized_round is not None
            and normalized_round.instruction.raw_opcode == "convert"
            else None
        )
        normalized_operands = (
            [graph.operand(normalized_mul, index) for index in range(2)]
            if normalized_mul is not None
            and normalized_mul.instruction.raw_opcode == "multiply"
            else [None, None]
        )
        raise BenchmarkValidationError(
            "dense RMS live output is not the exact weighted row-zero result: "
            f"output={None if output_operand is None else output_operand.instruction.raw_opcode} "
            f"m32_weighted={exact_weighted_bf16(row_source, rows=32)} "
            f"normalized={exact_normalized_lift(weighted_operands[0], rows=32)} "
            f"weight={exact_weight_f32(weighted_operands[1], rows=32)} "
            f"sum={exact_sum(normalized_operands[0])} "
            f"rsqrt={exact_rsqrt_broadcast(normalized_operands[1], rows=32)}"
        )
    if (
        scheduled_signature
        == (("f32", (32,)), ("f32", (32, 6144)))
        and not tuple_sum_consumed[0]
    ):
        raise BenchmarkValidationError(
            "dense RMS tuple schedule does not feed the live normalized result"
        )
    result = {
        "exact_collective_input": True,
        "exact_direct_residual": direct_residual,
        "exact_residual_round": hybrid_control or integrated_control,
        "exact_reduction_operand_graph": True,
        "exact_weighted_operand_graph": True,
        "exact_result_binding": True,
        "residual_source_mode": (
            "integrated_attention_plus_combined_residual"
            if integrated_control
            else
            "direct_post_attention_residual"
            if direct_residual
            else "hybrid_attention_plus_combined_control"
        ),
    }
    if preceding_attention_reduction is not None:
        result["preceding_attention_collective"] = True
    if require_split_output_fusion:
        exact_schedule = bool(
            scheduled_signature == (("f32", (32,)),)
            and _exact_accepted_rms_schedule(scheduled)
        )
        owner_sets = tuple(split_output_owners.values())
        exact_output_fusion = bool(
            all(len(values) == 1 for values in owner_sets)
            and len(set().union(*owner_sets)) == 1
            and not next(iter(split_output_owners["weighted"])).startswith(
                "ENTRY "
            )
        )
        if not (
            exact_schedule
            and len(split_output_owners["sum"]) == 1
            and exact_output_fusion
        ):
            raise BenchmarkValidationError(
                "dense RMS accepted split output-fusion ownership drifted: "
                f"schedule={exact_schedule} owners={split_output_owners}"
            )
        result.update(
            {
                "exact_accepted_scheduled_reduction": True,
                "split_output_fusion_exact": True,
                "split_recompute_exact": True,
            }
        )
    return result


def validate_strategy_nd_dense_rms_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
) -> tuple[HloLintReport, Mapping[str, Any], Mapping[str, Any]]:
    """Pin the physical collective and its live scheduled RMS consumer."""

    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        strategy_nd_dense_rms_hlo_policy(member_device_ids),
    )
    report.raise_for_violations()
    reductions = tuple(
        item
        for item in report.module.collectives
        if item.opcode == "all-reduce"
    )
    if len(reductions) != 1:
        raise BenchmarkValidationError(
            f"dense RMS replay requires one all-reduce, found {len(reductions)}"
        )
    reduction = reductions[0]
    algorithm = validate_strategy_nd_reduction(report, reduction)
    entry = tuple(
        item
        for item in report.module.instructions
        if item.computation.startswith("ENTRY ")
    )
    roots = tuple(
        item for item in entry if item.raw_line.lstrip().startswith("ROOT ")
    )
    parameters = tuple(item for item in entry if item.raw_opcode == "parameter")
    if len(roots) != 1 or _shape_signature(roots[0]) != (("u16", (1, 6144)),):
        raise BenchmarkValidationError("dense RMS replay ENTRY result drifted")
    root = roots[0]
    parameter_records = {}
    for item in parameters:
        match = re.search(r"\bparameter\(([0-9]+)\)", item.raw_line)
        if match is None:
            raise BenchmarkValidationError("dense RMS ENTRY parameter index drifted")
        parameter_records[int(match.group(1))] = _shape_signature(item)
    if parameter_records not in ({
        0: (("u16", (1, 32, 6144)),),
        1: (("bf16", (1, 6144)),),
        2: (("bf16", (6144,)),),
    }, {
        0: (("u16", (1, 32, 6144)),),
        1: (("bf16", (1, 6144)),),
        2: (("bf16", (1, 6144)),),
        3: (("bf16", (6144,)),),
    }):
        raise BenchmarkValidationError("dense RMS replay ENTRY inputs drifted")

    scheduled = tuple(
        item
        for item in entry
        if "strategy_nd_dense_rms_layer1" in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if len(scheduled) != 1:
        raise BenchmarkValidationError(
            "dense RMS accepted scheduled reduction/result binding drifted"
        )
    exact_flow = _validate_exact_dense_rms_value_flow(
        report, reduction, scheduled[0], root
    )
    if any(
        item.raw_opcode == "convolution"
        or (
            item.raw_opcode == "custom-call"
            and 'custom_call_target="tpu_custom_call"' in item.raw_line
        )
        for item in report.module.instructions
    ):
        raise BenchmarkValidationError("dense RMS replay contains model compute")
    if any(
        marker in optimized_hlo
        for marker in ("host_callback", "xla_python_cpu_callback", " outfeed(")
    ):
        raise BenchmarkValidationError("dense RMS replay contains a host effect")
    contract = {
        "accepted_scheduled_reduction": scheduled[0].name,
        "collective_count": len(report.module.collectives),
        **exact_flow,
        "live_rows": 1,
        "num_partitions": report.module.num_partitions,
        "num_replicas": report.module.num_replicas,
        "passed": True,
        "performance_claim": False,
        "violations": [],
    }
    return report, algorithm, contract


def _replay_function() -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    def replay(
        local_partial_bits: Any,
        post_attention_residual: Any,
        layer1_norm: Any,
    ) -> Any:
        with jax.named_scope("strategy_nd_dense_rms_collective"):
            local_partials = lax.bitcast_convert_type(
                local_partial_bits[0], jnp.bfloat16
            )
            dense_m32 = lax.psum(local_partials, "member")
        with jax.named_scope("strategy_nd_dense_rms_residual"):
            residual_m32 = jnp.pad(
                post_attention_residual,
                ((0, 31), (0, 0)),
                mode="constant",
                constant_values=jnp.bfloat16(0),
            )
        with jax.named_scope("strategy_nd_dense_rms_layer1"):
            layer1, _ = fused_add_rms_norm(
                dense_m32,
                residual_m32,
                layer1_norm,
                epsilon=1e-5,
            )
        with jax.named_scope("strategy_nd_dense_rms_live_row"):
            return lax.bitcast_convert_type(layer1[:1, :], jnp.uint16)

    return replay


def build_strategy_nd_dense_rms_replay(
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    enforce_optimized_hlo_contract: bool = True,
) -> CompiledStrategyNdDenseRmsReplay:
    """Compile the all-32-chip combine and RMS boundary as one program."""

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    members = tuple(int(value) for value in member_device_ids)
    by_id = {int(device.id): device for device in runtime_devices}
    if len(runtime_devices) != 32 or set(by_id) != set(range(32)):
        raise BenchmarkValidationError(
            "StrategyND dense RMS replay requires 32 global devices"
        )
    strategy_nd_dense_rms_hlo_policy(members)
    mesh = Mesh(
        np.asarray([by_id[value] for value in members], dtype=object),
        ("member",),
    )
    input_sharding = NamedSharding(mesh, P("member", None, None))
    replicated = NamedSharding(mesh, P())
    mapped = jax.shard_map(
        _replay_function(),
        mesh=mesh,
        in_specs=(P("member", None, None), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    examples = (
        jax.device_put(
            np.zeros((32, 32, 6144), dtype=np.uint16), input_sharding
        ),
        jax.device_put(
            np.zeros((1, 6144), dtype=np.uint16).view(ml_dtypes.bfloat16),
            replicated,
        ),
        jax.device_put(
            np.zeros((6144,), dtype=np.uint16).view(ml_dtypes.bfloat16),
            replicated,
        ),
    )
    lowered = jax.jit(mapped).lower(*examples)
    stablehlo = lowered.as_text()
    stablehlo_contract = validate_strategy_nd_dense_rms_stablehlo(stablehlo)
    if not (
        stablehlo_contract["passed"]
        and stablehlo_contract.get("residual_source_mode")
        == "direct_post_attention_residual"
    ):
        raise BenchmarkValidationError(
            f"dense RMS StableHLO failed: {stablehlo_contract}"
        )
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    if enforce_optimized_hlo_contract:
        report, algorithm, optimized_contract = validate_strategy_nd_dense_rms_hlo(
            optimized_hlo, members
        )
        if optimized_contract.get("residual_source_mode") != (
            "direct_post_attention_residual"
        ):
            raise BenchmarkValidationError(
                "dense RMS optimized HLO does not consume the direct residual"
            )
    else:
        report = lint_hlo(
            parse_hlo_module(optimized_hlo),
            strategy_nd_dense_rms_hlo_policy(members),
        )
        algorithm = {}
        optimized_contract = {
            "passed": False,
            "performance_claim": False,
            "violations": ["optimized TPU contract was not requested"],
        }
    return CompiledStrategyNdDenseRmsReplay(
        compiled=compiled,
        input_sharding=input_sharding,
        replicated_sharding=replicated,
        member_device_ids=members,
        stablehlo=stablehlo,
        stablehlo_contract=stablehlo_contract,
        optimized_hlo=optimized_hlo,
        optimized_hlo_contract=optimized_contract,
        collective_algorithm=algorithm,
    )


def execute_strategy_nd_dense_rms_replay(
    compiled: CompiledStrategyNdDenseRmsReplay,
    physical_partial_bits: np.ndarray,
    inputs: DenseRmsInputs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute twice and return the one live normalized row as raw BF16 bits."""

    import jax

    physical = np.ascontiguousarray(physical_partial_bits)
    if physical.shape != (32, 6144) or physical.dtype != np.uint16:
        raise BenchmarkValidationError(
            "physical dense partials must be uint16[32,6144]"
        )
    distributed = np.ascontiguousarray(
        np.broadcast_to(physical[:, None, :], (32, 32, 6144))
    )
    arguments = (
        jax.device_put(distributed, compiled.input_sharding),
        jax.device_put(
            inputs.post_attention_residual_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(
            inputs.layer1_norm_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
    )

    def execute_once() -> tuple[np.ndarray, tuple[str, ...]]:
        result = compiled.compiled(*arguments)
        jax.block_until_ready(result)
        local = tuple(
            np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(6144)
            for shard in sorted(
                result.addressable_shards,
                key=lambda shard: int(shard.device.id),
            )
        )
        hashes = tuple(array_sha256(value) for value in local)
        if len(local) != len(compiled.replicated_sharding.addressable_devices) or len(set(hashes)) != 1:
            raise BenchmarkValidationError(
                "dense RMS replay output differs across local replicas"
            )
        return np.ascontiguousarray(local[0]), hashes

    output, local_hashes = execute_once()
    repeated, repeated_hashes = execute_once()
    if not np.array_equal(output, repeated):
        raise BenchmarkValidationError("dense RMS replay is nondeterministic")
    return output, {
        "determinism_repeat_invocations": 1,
        "distributed_input_sha256": array_sha256(distributed),
        "invocation_count": 2,
        "local_replica_output_sha256": list(local_hashes),
        "output_bits_sha256": array_sha256(output),
        "repeated_local_replica_output_sha256": list(repeated_hashes),
        "repeated_output_bits_sha256": array_sha256(repeated),
    }
