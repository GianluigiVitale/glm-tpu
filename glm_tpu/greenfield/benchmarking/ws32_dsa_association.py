"""HLO contracts for the bounded WS32 layer-0 DSA discriminator."""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any

from .ws32_pallas_one_layer import _live_instruction_closure
from ..sharding.hlo_contract import HloShape, parse_hlo_module


# These hashes cover the complete whitespace-normalized ``func.func`` bodies
# generated on the pinned JAX build.  The enclosing module name and
# ``mhlo.num_partitions`` attributes are deliberately excluded because they
# differ between CPU preflight and TPU acquisition; every numerical operation,
# operand edge, dtype, shape, precision, slice and return remains covered.
_STABLEHLO_BODY_SHA256 = {
    "tuple4_materializer": (
        "bddbd21036174b090e9c424d49eaaafe06361434e59a74cf1af5c133c36b7806"
    ),
    "tuple4_query_head": (
        "0d95a07694579e0eeb7dce37ba0a533c8e1782c6edf08f676b0da6f398031d1a"
    ),
    "tuple8_materializer": (
        "a15ca911e55cd37738dd42cb227106b7e90cceb2814be2c4b3111471e608a421"
    ),
    "tuple8_query_head": (
        "fdd082fb21a39fff9f7a7f9e9d59c19d1708d165019eff379d24e2ffa4e314e1"
    ),
    "wk_decode": (
        "64c6619851d25f91cfae02734eb20caf769900ba2b7c413dd153f938cb943d10"
    ),
    "wk_promote": (
        "21a2b821e991c30d362735c4e9d0e7a66981952b974263470333afe8a9a73280"
    ),
    "exact_current_key": (
        "9ab347fc951b219f2d71e03249bc9b011d9ed2d7096ae26e6f795565d98ecf50"
    ),
    "default_score": (
        "a825c449b3f4e2574c694e98268c24e0dee8c0866ae0a6f0f937e54194f019e5"
    ),
}


def _stablehlo_body_sha256(stablehlo: str) -> str:
    lines = [line.strip() for line in stablehlo.splitlines() if line.strip()]
    if (
        len(lines) < 3
        or not lines[0].startswith("module @")
        or lines[-1] != "}"
        or not lines[1].startswith("func.func public @main(")
        or sum(
            line.startswith("func.func public @main(") for line in lines
        )
        != 1
    ):
        raise ValueError("WS32 DSA StableHLO module envelope drifted")
    return sha256("\n".join(lines[1:-1]).encode()).hexdigest()


def _attribute_positions(value: str, marker: str) -> tuple[int, ...]:
    positions: list[int] = []
    index = 0
    quoted = False
    escaped = False
    comment = False
    while index < len(value):
        if comment:
            if value.startswith("*/", index):
                comment = False
                index += 2
            else:
                index += 1
            continue
        if quoted:
            character = value[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            index += 1
            continue
        if value.startswith("/*", index):
            comment = True
            index += 2
            continue
        if value[index] == '"':
            quoted = True
            index += 1
            continue
        if value.startswith(marker, index):
            positions.append(index)
            index += len(marker)
            continue
        index += 1
    if comment or quoted:
        raise ValueError("WS32 DSA HLO contains unterminated comment/string")
    return tuple(positions)


def _actual_marker_values(raw_line: str, op_name: str | None) -> set[str]:
    values = set(() if op_name is None else op_name.split("/"))
    for attribute in ("custom_call_target", "kernel_name"):
        marker = f'{attribute}="'
        positions = _attribute_positions(raw_line, marker)
        for position in positions:
            tail = raw_line[position + len(marker) :]
            match = re.match(r'([^"\\]*(?:\\.[^"\\]*)*)"', tail)
            if match is None:
                raise ValueError(f"WS32 DSA {attribute} attribute drifted")
            values.add(match.group(1))
    return values


def _live_marker_present(instruction: Any, marker: str) -> bool:
    """Match exact attributes or a slash-delimited live op-name path."""

    if marker in _actual_marker_values(
        instruction.raw_line, instruction.op_name
    ):
        return True
    if "/" not in marker or instruction.op_name is None:
        return False
    path = f"/{instruction.op_name.strip('/')}/"
    component = f"/{marker.strip('/')}/"
    return component in path


def _backend_config(raw_line: str) -> dict[str, Any]:
    marker = "backend_config="
    positions = _attribute_positions(raw_line, marker)
    if not positions:
        return {}
    if len(positions) != 1:
        raise ValueError("WS32 DSA optimized HLO backend_config count drifted")
    start = positions[0]
    value = raw_line[start + len(marker) :].lstrip()
    parsed, end = json.JSONDecoder().raw_decode(value)
    if value[end:].strip().strip(",") or not isinstance(parsed, dict):
        raise ValueError("WS32 DSA optimized HLO backend_config drifted")
    return parsed


def validate_ws32_dsa_stablehlo(
    stablehlo: str,
    *,
    component: str,
) -> dict[str, Any]:
    """Require the exact compiler-independent numerical program body."""

    expected = _STABLEHLO_BODY_SHA256.get(component)
    if expected is None:
        raise ValueError("unknown WS32 DSA StableHLO component")
    observed = _stablehlo_body_sha256(stablehlo)
    return {
        "body_sha256": observed,
        "component": component,
        "exact": observed == expected,
        "expected_body_sha256": expected,
    }


def validate_ws32_dsa_component_hlo(
    optimized_hlo: str,
    *,
    expected_entry_parameters: dict[tuple[str, tuple[int, ...]], int],
    stablehlo: str | None = None,
    expected_stablehlo_component: str | None = None,
    expected_collective_groups: tuple[int, ...] = (),
    required_live_markers: tuple[str, ...] = (),
    required_16k_fusion_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Prove a small candidate's live inputs, groups, and TPU fusion bytes."""

    module = parse_hlo_module(optimized_hlo)
    live = _live_instruction_closure(module.instructions)
    entry_parameters: dict[tuple[str, tuple[int, ...]], int] = {}
    for instruction in live:
        if not instruction.computation.startswith("ENTRY ") or (
            instruction.raw_opcode != "parameter"
            or len(instruction.result_shapes) != 1
        ):
            continue
        shape = instruction.result_shapes[0]
        key = (shape.dtype, shape.dimensions)
        entry_parameters[key] = entry_parameters.get(key, 0) + 1
    collectives = tuple(
        instruction for instruction in live if instruction.is_collective
    )
    collective_groups = tuple(
        sorted(instruction.maximum_group_size for instruction in collectives)
    )
    async_collectives = sorted(
        instruction.raw_opcode
        for instruction in live
        if instruction.raw_opcode.endswith("-start")
        or instruction.raw_opcode.endswith("-done")
    )
    forbidden_opcodes = sorted(
        {
            instruction.raw_opcode
            for instruction in live
            if instruction.raw_opcode
            in {
                "all-gather",
                "all-to-all",
                "collective-permute",
                "outfeed",
                "reduce-scatter",
            }
        }
    )
    actual_markers = set().union(
        *(
            _actual_marker_values(instruction.raw_line, instruction.op_name)
            for instruction in live
        )
    )
    actual_markers.update(
        marker
        for marker in required_live_markers
        if any(
            _live_marker_present(instruction, marker) for instruction in live
        )
    )
    missing_markers = [
        marker for marker in required_live_markers if marker not in actual_markers
    ]
    sixteen_kib_fusions = []
    backend_parse_errors = []
    for instruction in live:
        try:
            backend = _backend_config(instruction.raw_line)
        except (json.JSONDecodeError, ValueError) as error:
            backend_parse_errors.append(f"{instruction.name}: {error}")
            continue
        megacore = backend.get("megacore_config", {})
        if isinstance(megacore, dict) and (
            megacore.get("megacore_allreduce_bytes") == "16384"
        ):
            sixteen_kib_fusions.append(instruction)
    required_16k_fusions = int(required_16k_fusion_shape is not None)
    fusion_shapes_exact = required_16k_fusion_shape is None
    fusion_parameter_binding = required_16k_fusion_shape is None
    if required_16k_fusion_shape is not None and len(sixteen_kib_fusions) == 1:
        alias_count, local_width = required_16k_fusion_shape
        fusion = sixteen_kib_fusions[0]
        fusion_shapes_exact = fusion.result_shapes == tuple(
            HloShape("f32", (local_width,))
            for _ in range(alias_count)
        )
        entry = {
            instruction.name: instruction
            for instruction in live
            if instruction.computation.startswith("ENTRY ")
        }
        weight_names = {
            name
            for name, instruction in entry.items()
            if instruction.raw_opcode == "parameter"
            and instruction.result_shapes
            and instruction.result_shapes[0].dtype == "f32"
            and instruction.result_shapes[0].dimensions == (local_width, 2048)
        }
        q_names = {
            name
            for name, instruction in entry.items()
            if instruction.raw_opcode == "parameter"
            and instruction.result_shapes
            and instruction.result_shapes[0].dtype == "bf16"
            and instruction.result_shapes[0].dimensions == (1, 2048)
        }

        def layout_origins(name: str, seen: set[str]) -> set[str]:
            if name in seen:
                return set()
            seen.add(name)
            instruction = entry.get(name)
            if instruction is None or instruction.raw_opcode == "parameter":
                return {name}
            if instruction.raw_opcode not in {
                "bitcast",
                "copy",
                "convert",
                "optimization-barrier",
                "reshape",
            }:
                return set()
            result = set()
            for operand in instruction.operand_names:
                result.update(layout_origins(operand, seen))
            return result

        non_weights = [
            name for name in fusion.operand_names if name not in weight_names
        ]
        fusion_parameter_binding = bool(
            fusion.computation.startswith("ENTRY ")
            and len(weight_names) == alias_count
            and weight_names.issubset(set(fusion.operand_names))
            and len(fusion.operand_names) == alias_count + 1
            and len(non_weights) == 1
            and layout_origins(non_weights[0], set()) == q_names
        )
    stablehlo_body_sha256 = None
    stablehlo_exact = expected_stablehlo_component is None
    expected_stablehlo_body_sha256 = None
    if stablehlo is not None:
        stablehlo_body_sha256 = _stablehlo_body_sha256(stablehlo)
    if expected_stablehlo_component is not None:
        if stablehlo is None:
            raise ValueError("missing WS32 DSA StableHLO component")
        stable_result = validate_ws32_dsa_stablehlo(
            stablehlo, component=expected_stablehlo_component
        )
        expected_stablehlo_body_sha256 = stable_result[
            "expected_body_sha256"
        ]
        stablehlo_exact = stable_result["exact"]
    all_parameter_keys = set(entry_parameters) | set(expected_entry_parameters)
    parameter_mismatches = {
        f"{dtype}{list(dimensions)}": {
            "expected": expected_entry_parameters.get((dtype, dimensions), 0),
            "observed": entry_parameters.get((dtype, dimensions), 0),
        }
        for dtype, dimensions in all_parameter_keys
        if entry_parameters.get((dtype, dimensions), 0)
        != expected_entry_parameters.get((dtype, dimensions), 0)
    }
    passed = bool(
        not parameter_mismatches
        and collective_groups == tuple(sorted(expected_collective_groups))
        and not async_collectives
        and not forbidden_opcodes
        and not missing_markers
        and not backend_parse_errors
        and len(sixteen_kib_fusions) == required_16k_fusions
        and fusion_shapes_exact
        and fusion_parameter_binding
        and stablehlo_exact
    )
    return {
        "async_collectives": async_collectives,
        "backend_parse_errors": backend_parse_errors,
        "collective_groups": list(collective_groups),
        "entry_parameters": {
            f"{dtype}{list(dimensions)}": count
            for (dtype, dimensions), count in sorted(entry_parameters.items())
        },
        "forbidden_opcodes": forbidden_opcodes,
        "live_instruction_count": len(live),
        "missing_markers": missing_markers,
        "parameter_mismatches": parameter_mismatches,
        "passed": passed,
        "required_16k_fusion_count": required_16k_fusions,
        "sixteen_kib_fusion_count": len(sixteen_kib_fusions),
        "sixteen_kib_fusion_parameter_binding": fusion_parameter_binding,
        "sixteen_kib_fusion_shapes_exact": fusion_shapes_exact,
        "stablehlo": {
            "body_sha256": stablehlo_body_sha256,
            "component": expected_stablehlo_component,
            "exact": stablehlo_exact,
            "expected_body_sha256": expected_stablehlo_body_sha256,
        },
    }


def classify_ws32_dsa_query_head_contract(
    contract: dict[str, Any],
) -> str:
    """Separate a tested 16-KiB fusion miss from an invalid graph."""

    base_valid = bool(
        not contract.get("parameter_mismatches")
        and contract.get("collective_groups") == []
        and not contract.get("async_collectives")
        and not contract.get("forbidden_opcodes")
        and not contract.get("missing_markers")
        and not contract.get("backend_parse_errors")
        and contract.get("required_16k_fusion_count") == 1
        and contract.get("stablehlo", {}).get("exact") is True
    )
    if not base_valid:
        return "INVALID"
    fusion_valid = bool(
        contract.get("sixteen_kib_fusion_count") == 1
        and contract.get("sixteen_kib_fusion_shapes_exact") is True
        and contract.get("sixteen_kib_fusion_parameter_binding") is True
    )
    if contract.get("passed") is fusion_valid:
        return "PASSED" if fusion_valid else "HYPOTHESIS_REJECTED"
    return "INVALID"
