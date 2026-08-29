"""Fail-closed HLO contracts for the bounded PP16 feature2 acquisition.

The source/JAXpr contract in :mod:`pp16_feature2_program` proves the exact
four-chunk semantic graph.  This module independently inspects the optimized
TPU executable so a compile-only acquisition cannot be admitted from source
structure alone.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import HloInstruction, HloShape, parse_hlo_module

Feature2MaterializerPhase = Literal["query_fp32", "wk_decode_bf16", "wk_promote_fp32"]

FEATURE2_OPTIMIZED_HLO_CANONICALIZER_VERSION = 1
FEATURE2_SEALED_STABLEHLO_SHA256 = (
    "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4"
)
FEATURE2_SEALED_OPTIMIZED_CANONICAL_SHA256 = (
    "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
)
FEATURE2_SEALED_OPTIMIZED_CANONICAL_BYTES = 6_558_627
FEATURE2_SEALED_OPTIMIZED_STACK_FRAME_REFERENCES = 14_561
_DEBUG_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")
_INSTRUCTION_LINE = re.compile(r"^\s*(?:ROOT )?%[A-Za-z0-9_.-]+ = ")
_DEBUG_ROW_PATTERNS = {
    "FileNames": re.compile(r'^([1-9][0-9]*) "(?:\\.|[^"\\])*"$'),
    "FunctionNames": re.compile(r'^([1-9][0-9]*) "(?:\\.|[^"\\])*"$'),
    "FileLocations": re.compile(
        r"^([1-9][0-9]*) \{file_name_id=[1-9][0-9]* "
        r"function_name_id=[1-9][0-9]* line=[0-9]+ end_line=[0-9]+ "
        r"column=[0-9]+ end_column=[0-9]+\}$"
    ),
    "StackFrames": re.compile(
        r"^([1-9][0-9]*) \{file_location_id=[1-9][0-9]* "
        r"parent_frame_id=[0-9]+\}$"
    ),
}

_HOST_MARKERS = (
    "debug_callback",
    "host_callback",
    "io_callback",
    "outside_compilation",
    "pure_callback",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)
_FORBIDDEN_TRANSPORT_OPCODES = frozenset({"infeed", "outfeed", "recv", "send"})
_HISTORICAL_LOCAL_MAIN_ROOT = (
    ("s32", (1, 2048)),
    ("s32", (1,)),
    ("f32", (1, 2048)),
    ("bf16", (1, 1, 3072)),
    ("bf16", (1, 1, 32, 256)),
    ("bf16", (1, 576)),
    ("bf16", (1, 16, 256, 640)),
    ("bf16", (1, 16, 256, 128)),
    ("bf16", (1, 16, 256, 128)),
    ("u32", (1, 2)),
    ("pred", (1,)),
)
_SEALED_BOUNDARY_LOCAL_MAIN_ROOT = (
    *_HISTORICAL_LOCAL_MAIN_ROOT[:6],
    ("bf16", (1, 1, 6144)),
    ("bf16", (1, 1, 2048)),
    ("f32", (1, 1, 32, 128)),
    ("f32", (1, 1, 32)),
    *_HISTORICAL_LOCAL_MAIN_ROOT[6:],
)
_HISTORICAL_GLOBAL_MAIN_RESULTS = (
    ("tensor<1x2048xi32>", "result.event1_positions"),
    ("tensor<1xi32>", "result.event1_valid_counts"),
    ("tensor<1x2048xf32>", "result.event1_scores"),
    ("tensor<2x1x3072xbf16>", "result.current_carried_halves"),
    ("tensor<2x1x32x256xbf16>", "result.current_attention_query_owners"),
    ("tensor<1x576xbf16>", "result.current_kv_a"),
    ("tensor<2x16x256x640xbf16>", "result.layer0_kv_cache_owners"),
    ("tensor<2x16x256x128xbf16>", "result.layer0_index_cache_owners"),
    ("tensor<2x16x256x128xbf16>", "result.layer1_index_cache_owners"),
    ("tensor<2x2xui32>", "result.carried_liveness_digest_owners"),
    ("tensor<1xi1>", "result.contract_valid"),
)
_SEALED_BOUNDARY_GLOBAL_MAIN_RESULTS = (
    *_HISTORICAL_GLOBAL_MAIN_RESULTS[:6],
    (
        "tensor<2x1x6144xbf16>",
        "result.current_normalized_hidden_owners",
    ),
    ("tensor<2x1x2048xbf16>", "result.current_q_a_state_owners"),
    ("tensor<2x1x32x128xf32>", "result.current_dsa_query_owners"),
    ("tensor<2x1x32xf32>", "result.current_dsa_head_weights_owners"),
    *_HISTORICAL_GLOBAL_MAIN_RESULTS[6:],
)
_SEALED_BOUNDARY_ROOT_MARKERS = {
    6: "greenfield_pp16_feature2_sealed_normalized_hidden",
    7: "greenfield_pp16_feature2_sealed_q_a_state",
    8: "greenfield_pp16_feature2_sealed_dsa_query",
    9: "greenfield_pp16_feature2_sealed_dsa_head_weights",
}
_H16_ATTENTION_KERNEL = "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"
_FP8_ATTENTION_O_PREFIX = "greenfield_fp8_strategy_nd_o_m8_k512_n"
_ATTENTION_REDUCER_SCOPE = "greenfield_strategy_nd_feature2_attention_output"
_DENSE_REDUCER_SCOPE = "greenfield_strategy_nd_feature2_dense_down"
_STABLE_FUNCTION = re.compile(
    r"^\s*func\.func (?:public |private )?@([A-Za-z0-9_.$-]+)"
)
_STABLE_DEFINITION = re.compile(r"^\s*(%[A-Za-z0-9_.$-]+)(?::[0-9]+)?\s*=")
_STABLE_VALUE = re.compile(r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?")
_HLO_COMPUTATION_REFERENCE = re.compile(r"%[A-Za-z0-9_.-]+")
_HLO_CALLED_COMPUTATION = re.compile(r"(?:^|[, ])calls=(%[A-Za-z0-9_.-]+)")
_HLO_PARAMETER_INDEX = re.compile(r"\bparameter\(([0-9]+)\)")
_HLO_TUPLE_INDEX = re.compile(r"(?:^|[, ])index=([0-9]+)")


def _strip_instruction_metadata_stack_frame_ids(line: str) -> tuple[str, int]:
    """Remove only terminal stack-frame keys in top-level instruction metadata."""

    output: list[str] = []
    index = 0
    quoted = False
    escaped = False
    brace_depth = 0
    metadata_depth: int | None = None
    metadata_open_index: int | None = None
    metadata_container_count = 0
    count = 0
    while index < len(line):
        character = line[index]
        if quoted:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            index += 1
            continue
        if character == '"':
            quoted = True
            output.append(character)
            index += 1
            continue
        if line.startswith("metadata={", index):
            previous = index - 1
            while previous >= 0 and line[previous].isspace():
                previous -= 1
            if brace_depth == 0 and previous >= 0 and line[previous] == ",":
                if _INSTRUCTION_LINE.match(line) is None:
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO metadata is not on an instruction"
                    )
                metadata_container_count += 1
                if metadata_container_count != 1:
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO has duplicate instruction metadata"
                    )
                metadata_open_index = index + len("metadata=")
        if line.startswith("stack_frame_id=", index):
            end = index + len("stack_frame_id=")
            while end < len(line) and line[end].isdigit():
                end += 1
            if (
                index == 0
                or line[index - 1] != " "
                or metadata_depth is None
                or brace_depth != metadata_depth
                or end == index + len("stack_frame_id=")
                or end >= len(line)
                or line[end] != "}"
            ):
                raise BenchmarkValidationError(
                    "feature2 optimized HLO has stack-frame data outside terminal "
                    "instruction metadata"
                )
            if not output or output[-1] != " ":
                raise BenchmarkValidationError(
                    "feature2 optimized HLO stack-frame separator drifted"
                )
            output.pop()
            count += 1
            index = end
            continue
        if character == "{":
            brace_depth += 1
            if index == metadata_open_index:
                metadata_depth = brace_depth
        elif character == "}":
            if metadata_depth == brace_depth:
                metadata_depth = None
            brace_depth -= 1
        output.append(character)
        index += 1
    if quoted:
        raise BenchmarkValidationError(
            "feature2 optimized HLO has an unterminated quoted string"
        )
    if metadata_depth is not None:
        raise BenchmarkValidationError(
            "feature2 optimized HLO has unterminated instruction metadata"
        )
    return "".join(output), count


def canonicalize_feature2_optimized_hlo(
    optimized_hlo: str,
) -> tuple[str, dict[str, Any]]:
    """Remove only validated non-executable TPU source-debug provenance."""

    if not isinstance(optimized_hlo, str) or not optimized_hlo.endswith("\n"):
        raise BenchmarkValidationError(
            "feature2 optimized HLO must be nonempty newline-terminated text"
        )
    lines = optimized_hlo.splitlines(keepends=True)
    if len(lines) < 8 or not lines[0].startswith("HloModule ") or lines[1] != "\n":
        raise BenchmarkValidationError("feature2 optimized HLO module header drifted")
    header_indices: dict[str, int] = {}
    for header in _DEBUG_TABLES:
        matches = tuple(
            index for index, line in enumerate(lines) if line == f"{header}\n"
        )
        if len(matches) != 1:
            raise BenchmarkValidationError(
                f"feature2 optimized HLO requires one {header} debug table"
            )
        header_indices[header] = matches[0]
    ordered = tuple(header_indices[name] for name in _DEBUG_TABLES)
    if ordered != tuple(sorted(ordered)) or ordered[0] != 2:
        raise BenchmarkValidationError(
            "feature2 optimized HLO debug tables are missing, duplicated, or relocated"
        )
    computation_start = next(
        (
            index
            for index, line in enumerate(lines[ordered[-1] + 1 :], ordered[-1] + 1)
            if line.startswith(("%", "ENTRY "))
        ),
        None,
    )
    if computation_start is None:
        raise BenchmarkValidationError("feature2 optimized HLO has no computation body")

    section_records: dict[str, Any] = {}
    boundaries = (*ordered[1:], computation_start)
    for header, start, end in zip(_DEBUG_TABLES, ordered, boundaries, strict=True):
        body = lines[start + 1 : end]
        trailing_blank_count = 2 if header == "StackFrames" else 1
        rows = body[:-trailing_blank_count]
        if (
            len(body) <= trailing_blank_count
            or body[-trailing_blank_count:] != ["\n"] * trailing_blank_count
            or any(line == "\n" for line in rows)
        ):
            raise BenchmarkValidationError(
                f"feature2 optimized HLO {header} table boundary drifted"
            )
        identifiers = []
        pattern = _DEBUG_ROW_PATTERNS[header]
        for line in rows:
            match = pattern.fullmatch(line.removesuffix("\n"))
            if match is None:
                raise BenchmarkValidationError(
                    f"feature2 optimized HLO {header} row syntax drifted"
                )
            identifiers.append(int(match.group(1)))
        if identifiers != list(range(1, len(identifiers) + 1)):
            raise BenchmarkValidationError(
                f"feature2 optimized HLO {header} row ids are not contiguous"
            )
        raw_section = "".join(lines[start:end]).encode()
        section_records[header] = {
            "row_count": len(identifiers),
            "sha256": sha256(raw_section).hexdigest(),
        }

    canonical_lines = [*lines[: ordered[0]], *lines[computation_start:]]
    stripped_stack_frame_references = 0
    for index, line in enumerate(canonical_lines):
        canonical_lines[index], removed = _strip_instruction_metadata_stack_frame_ids(
            line
        )
        stripped_stack_frame_references += removed
    canonical = "".join(canonical_lines)
    if "stack_frame_id=" in canonical:
        # Quoted text is preserved deliberately and will change the canonical pin,
        # but an unquoted key must never survive the lexical pass.
        for line in canonical.splitlines():
            _, removed = _strip_instruction_metadata_stack_frame_ids(line)
            if removed:
                raise BenchmarkValidationError(
                    "feature2 optimized HLO retained stack-frame metadata"
                )
    canonical_bytes = canonical.encode()
    return canonical, {
        "byte_count": len(canonical_bytes),
        "canonicalizer_version": FEATURE2_OPTIMIZED_HLO_CANONICALIZER_VERSION,
        "debug_sections": section_records,
        "sha256": sha256(canonical_bytes).hexdigest(),
        "stripped_stack_frame_references": stripped_stack_frame_references,
    }


def validate_feature2_sealed_hlo_archive_identity(
    stablehlo_path: str | Path,
    optimized_hlo_path: str | Path,
    canonical_hlo_path: str | Path,
    *,
    expected_stablehlo_sha256: str,
    expected_canonical_sha256: str,
    expected_canonical_bytes: int,
    expected_canonicalizer_version: int,
    expected_stripped_stack_frame_references: int,
) -> dict[str, Any]:
    """Recompute the sealed archive identity from its three actual HLO files."""

    expected_hashes = (expected_stablehlo_sha256, expected_canonical_sha256)
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in expected_hashes):
        raise ValueError("sealed HLO expected hashes must be lowercase SHA-256")
    if (
        expected_canonical_bytes <= 0
        or expected_canonicalizer_version <= 0
        or expected_stripped_stack_frame_references <= 0
    ):
        raise ValueError("sealed HLO expected identity counts must be positive")
    stable_path = Path(stablehlo_path)
    optimized_path = Path(optimized_hlo_path)
    canonical_path = Path(canonical_hlo_path)
    stable_bytes = stable_path.read_bytes()
    optimized_bytes = optimized_path.read_bytes()
    archived_canonical_bytes = canonical_path.read_bytes()
    stable_sha256 = sha256(stable_bytes).hexdigest()
    optimized_sha256 = sha256(optimized_bytes).hexdigest()
    if stable_sha256 != expected_stablehlo_sha256:
        raise BenchmarkValidationError(
            "sealed archive StableHLO identity drifted: "
            f"expected={expected_stablehlo_sha256} observed={stable_sha256}"
        )
    canonical, canonical_identity = canonicalize_feature2_optimized_hlo(
        optimized_bytes.decode()
    )
    expected_canonical_identity = {
        "byte_count": expected_canonical_bytes,
        "canonicalizer_version": expected_canonicalizer_version,
        "sha256": expected_canonical_sha256,
        "stripped_stack_frame_references": (expected_stripped_stack_frame_references),
    }
    observed_canonical_identity = {
        key: canonical_identity.get(key) for key in expected_canonical_identity
    }
    if observed_canonical_identity != expected_canonical_identity:
        raise BenchmarkValidationError(
            "sealed archive optimized canonical-HLO identity drifted: "
            f"expected={expected_canonical_identity} "
            f"observed={observed_canonical_identity}"
        )
    recomputed_canonical_bytes = canonical.encode()
    if archived_canonical_bytes != recomputed_canonical_bytes:
        raise BenchmarkValidationError(
            "sealed archive canonical-HLO file is not the recomputed optimized HLO"
        )
    return {
        "canonical_hlo_identity": canonical_identity,
        "optimized_hlo_bytes": len(optimized_bytes),
        "optimized_hlo_sha256": optimized_sha256,
        "stablehlo_bytes": len(stable_bytes),
        "stablehlo_sha256": stable_sha256,
    }


def _shape_signature(
    shapes: tuple[HloShape, ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple((shape.dtype.lower(), shape.dimensions) for shape in shapes)


def _entry_instructions(
    instructions: tuple[HloInstruction, ...],
) -> tuple[HloInstruction, ...]:
    return tuple(
        instruction
        for instruction in instructions
        if instruction.computation.startswith("ENTRY ")
    )


def _entry_root(instructions: tuple[HloInstruction, ...]) -> HloInstruction | None:
    roots = tuple(
        instruction
        for instruction in _entry_instructions(instructions)
        if instruction.raw_line.startswith("ROOT ")
    )
    return roots[0] if len(roots) == 1 else None


def _host_markers(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(marker for marker in _HOST_MARKERS if marker in lowered)


def _entry_parameter_count(
    instructions: tuple[HloInstruction, ...],
    dtype: str,
    dimensions: tuple[int, ...],
) -> int:
    return sum(
        instruction.raw_opcode == "parameter"
        and _shape_signature(instruction.result_shapes) == ((dtype, dimensions),)
        for instruction in _entry_instructions(instructions)
    )


def _raise(what: str, violations: list[str]) -> None:
    if violations:
        raise BenchmarkValidationError(f"{what} rejected: {violations}")


def _stablehlo_function_statements(stablehlo: str) -> dict[str, tuple[str, ...]]:
    """Split textual StableHLO into function-local SSA statements."""

    function_lines: dict[str, list[str]] = {}
    current: str | None = None
    for line in stablehlo.splitlines():
        match = _STABLE_FUNCTION.match(line)
        if match is not None:
            current = match.group(1)
            if current in function_lines:
                raise BenchmarkValidationError(
                    f"feature2 StableHLO duplicates function @{current}"
                )
            function_lines[current] = []
        if current is not None:
            function_lines[current].append(line)
    if "main" not in function_lines:
        raise BenchmarkValidationError("feature2 StableHLO has no public @main")

    result: dict[str, tuple[str, ...]] = {}
    for function, lines in function_lines.items():
        statements: list[str] = []
        current_lines: list[str] = []
        for line in lines:
            if _STABLE_DEFINITION.match(line) or re.match(r"^\s*return\b", line):
                if current_lines:
                    statements.append("\n".join(current_lines))
                current_lines = [line]
            elif current_lines:
                current_lines.append(line)
        if current_lines:
            statements.append("\n".join(current_lines))
        result[function] = tuple(statements)
    return result


def _stablehlo_reachable_functions(
    statements: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    calls = {
        function: frozenset(
            target
            for statement in body
            for target in re.findall(r"\bcall @([A-Za-z0-9_.$-]+)", statement)
        )
        for function, body in statements.items()
    }
    reachable: set[str] = set()
    pending = ["main"]
    while pending:
        function = pending.pop()
        if function in reachable:
            continue
        if function not in statements:
            raise BenchmarkValidationError(
                f"feature2 StableHLO calls missing function @{function}"
            )
        reachable.add(function)
        pending.extend(calls[function] - reachable)
    return frozenset(reachable)


def _stablehlo_main_boundary_contract(
    stablehlo: str,
    *,
    sealed_boundary_capture: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Bind public result names, types, and live return projections exactly."""

    expected = (
        _SEALED_BOUNDARY_GLOBAL_MAIN_RESULTS
        if sealed_boundary_capture
        else _HISTORICAL_GLOBAL_MAIN_RESULTS
    )
    signatures = tuple(
        re.finditer(
            r"^\s*func\.func public @main\(.*?\)\s*->\s*\((.*?)\)\s*\{",
            stablehlo,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    violations: list[str] = []
    if len(signatures) != 1:
        return (
            {"output_count": 0, "result_infos": [], "terminal_types": []},
            [f"expected one public @main signature, found {len(signatures)}"],
        )
    observed_signature = re.sub(r"\s+", "", signatures[0].group(1))
    expected_signature = ",".join(
        f'{tensor_type}{{jax.result_info="{result_info}"}}'
        for tensor_type, result_info in expected
    )
    if observed_signature != expected_signature:
        violations.append(
            "public @main result-info/type boundary drifted: "
            f"expected={expected_signature} observed={observed_signature}"
        )

    statements = _stablehlo_function_statements(stablehlo)
    returns = tuple(
        statement
        for statement in statements["main"]
        if re.match(r"^\s*return\b", statement)
    )
    return_operands: tuple[str, ...] = ()
    return_types: tuple[str, ...] = ()
    if len(returns) != 1:
        violations.append(f"public @main return count drifted: {len(returns)}")
    else:
        match = re.fullmatch(
            r"\s*return\s+(.+?)\s*:\s*(.+?)\s*",
            returns[0].splitlines()[0],
        )
        if match is None:
            violations.append("public @main return syntax drifted")
        else:
            return_operands = tuple(_STABLE_VALUE.findall(match.group(1)))
            return_types = tuple(item.strip() for item in match.group(2).split(","))
            expected_types = tuple(item[0] for item in expected)
            if return_types != expected_types:
                violations.append(
                    "public @main return types drifted: "
                    f"expected={expected_types} observed={return_types}"
                )
            projections = tuple(
                re.fullmatch(r"(%[A-Za-z0-9_.$-]+)#([0-9]+)", operand)
                for operand in return_operands
            )
            if (
                len(return_operands) != len(expected)
                or any(item is None for item in projections)
                or len({item.group(1) for item in projections if item is not None}) != 1
                or tuple(int(item.group(2)) for item in projections if item is not None)
                != tuple(range(len(expected)))
            ):
                violations.append(
                    "public @main return lost exact live ordered tuple projections: "
                    f"operands={return_operands}"
                )

    terminal_shapes = []
    for tensor_type, _ in expected:
        fields = tensor_type.removeprefix("tensor<").removesuffix(">").split("x")
        terminal_shapes.append([int(item) for item in fields[:-1]])
    return (
        {
            "output_count": len(return_operands),
            "result_infos": [item[1] for item in expected],
            "terminal_shapes": terminal_shapes,
            "terminal_types": [item[0] for item in expected],
        },
        violations,
    )


def _stablehlo_direct_users(
    statements: tuple[str, ...],
    definition_index: int,
    name: str,
) -> tuple[str, ...]:
    """Return users before the same SSA spelling is reused in another region."""

    users: list[str] = []
    for statement in statements[definition_index + 1 :]:
        match = _STABLE_DEFINITION.match(statement)
        if match is not None and match.group(1) == name:
            break
        if "=" not in statement:
            continue
        operands = {
            value.split("#", 1)[0]
            for value in _STABLE_VALUE.findall(statement.split("=", 1)[1])
        }
        if name in operands:
            users.append(statement)
    return tuple(users)


def _stablehlo_slice_ranges(
    statement: str,
    *,
    operand: str,
) -> tuple[tuple[int, int, int], ...] | None:
    """Parse one static StableHLO slice without accepting implicit drift."""

    match = re.search(
        rf"\bstablehlo\.slice\s+{re.escape(operand)}\s+\[([^\]]+)\]",
        statement,
    )
    if match is None:
        return None
    ranges: list[tuple[int, int, int]] = []
    for raw_dimension in match.group(1).split(","):
        fields = tuple(field.strip() for field in raw_dimension.split(":"))
        if len(fields) not in (2, 3) or any(
            re.fullmatch(r"[0-9]+", field) is None for field in fields
        ):
            return None
        start, limit = (int(field) for field in fields[:2])
        stride = 1 if len(fields) == 2 else int(fields[2])
        if stride <= 0:
            return None
        ranges.append((start, limit, stride))
    return tuple(ranges)


def _stablehlo_projection_contract(
    stablehlo: str,
    *,
    full_width_rounded_then_slice: bool,
) -> tuple[dict[str, Any], list[str]]:
    statements = _stablehlo_function_statements(stablehlo)
    reachable = _stablehlo_reachable_functions(statements)
    expected_width = 6144 if full_width_rounded_then_slice else 3072
    expected_attention_per_chunk = 16 if full_width_rounded_then_slice else 32
    expected_down_per_chunk = 16 if full_width_rounded_then_slice else 32
    expected_attention_total = expected_attention_per_chunk * 4
    expected_down_total = expected_down_per_chunk * 4
    expected_kernel = f"{_FP8_ATTENTION_O_PREFIX}{expected_width}"
    expected_attention_signature = (
        f": (tensor<8x512xbf16>, tensor<{expected_width}x512xui8>, "
        f"tensor<8x128xf32>) -> tensor<8x{expected_width}xbf16>"
    )
    half_down_signature = (
        "(tensor<1x384xbf16>, tensor<384x3072xbf16>) -> tensor<1x3072xf32>"
    )
    full_down_signature = (
        "(tensor<1x384xbf16>, tensor<384x6144xbf16>) -> tensor<1x6144xf32>"
    )
    gate_signature = "(tensor<1x6144xbf16>, tensor<6144x768xbf16>) -> tensor<1x768xf32>"

    all_attention: list[tuple[str, str, str]] = []
    down_by_function: Counter[str] = Counter()
    gate_by_function: Counter[str] = Counter()
    reducer_by_function: Counter[str] = Counter()
    violations: list[str] = []
    for function, body in statements.items():
        for statement_index, statement in enumerate(body):
            definition_match = _STABLE_DEFINITION.match(statement)
            if definition_match is None:
                continue
            name = definition_match.group(1)
            if (
                "stablehlo.custom_call @tpu_custom_call" in statement
                and f'kernel_name = "{_FP8_ATTENTION_O_PREFIX}' in statement
            ):
                all_attention.append((function, name, statement))
                direct_users = _stablehlo_direct_users(body, statement_index, name)
                row_ranges = (
                    None
                    if len(direct_users) != 1
                    else _stablehlo_slice_ranges(direct_users[0], operand=name)
                )
                if (
                    len(direct_users) != 1
                    or "stablehlo.slice" not in direct_users[0]
                    or f"-> tensor<1x{expected_width}xbf16>" not in direct_users[0]
                    or row_ranges != ((0, 1, 1), (0, expected_width, 1))
                ):
                    violations.append(
                        f"attention producer {function}:{name} lost its immediate "
                        "exact row-0 rounded slice"
                    )
                    continue
                if full_width_rounded_then_slice:
                    row_match = _STABLE_DEFINITION.match(direct_users[0])
                    row_name = None if row_match is None else row_match.group(1)
                    row_index = body.index(direct_users[0], statement_index + 1)
                    half_users = (
                        ()
                        if row_name is None
                        else _stablehlo_direct_users(body, row_index, row_name)
                    )
                    half_results = tuple(
                        (
                            user,
                            _stablehlo_slice_ranges(user, operand=row_name),
                        )
                        for user in half_users
                        if "stablehlo.slice" in user
                        and "-> tensor<1x3072xbf16>" in user
                    )
                    expected_halves = (
                        ((0, 1, 1), (0, 3072, 1)),
                        ((0, 1, 1), (3072, 6144, 1)),
                    )
                    if (
                        len(half_users) != 2
                        or len(half_results) != 2
                        or tuple(ranges for _, ranges in half_results)
                        != expected_halves
                    ):
                        violations.append(
                            f"attention producer {function}:{name} does not split "
                            "immediately into ordered complementary owner halves"
                        )
            if "stablehlo.convolution" in statement:
                if half_down_signature in statement or full_down_signature in statement:
                    down_by_function[function] += 1
                    direct_users = _stablehlo_direct_users(body, statement_index, name)
                    if (
                        len(direct_users) != 1
                        or "stablehlo.convert" not in direct_users[0]
                        or f"tensor<1x{expected_width}xbf16>" not in direct_users[0]
                    ):
                        violations.append(
                            f"dense-down producer {function}:{name} lost its "
                            "immediate BF16 rounding"
                        )
                    elif full_width_rounded_then_slice:
                        rounded_match = _STABLE_DEFINITION.match(direct_users[0])
                        rounded_name = (
                            None if rounded_match is None else rounded_match.group(1)
                        )
                        rounded_index = body.index(direct_users[0], statement_index + 1)
                        half_users = (
                            ()
                            if rounded_name is None
                            else _stablehlo_direct_users(
                                body, rounded_index, rounded_name
                            )
                        )
                        half_results = tuple(
                            (
                                user,
                                _stablehlo_slice_ranges(user, operand=rounded_name),
                            )
                            for user in half_users
                            if "stablehlo.slice" in user
                            and "-> tensor<1x3072xbf16>" in user
                        )
                        expected_halves = (
                            ((0, 1, 1), (0, 3072, 1)),
                            ((0, 1, 1), (3072, 6144, 1)),
                        )
                        if (
                            len(half_users) != 2
                            or len(half_results) != 2
                            or tuple(ranges for _, ranges in half_results)
                            != expected_halves
                        ):
                            violations.append(
                                f"dense-down producer {function}:{name} does not "
                                "split immediately into ordered complementary "
                                "owner halves"
                            )
                if gate_signature in statement:
                    gate_by_function[function] += 1
            if (
                "stablehlo.collective_permute" in statement
                and "(tensor<4x1x3072xbf16>) -> tensor<4x1x3072xbf16>" in statement
            ):
                reducer_by_function[function] += 1

    expected_attention = tuple(
        item
        for item in all_attention
        if f'kernel_name = "{expected_kernel}"' in item[2]
    )
    malformed_attention = tuple(
        f"{function}:{name}"
        for function, name, statement in expected_attention
        if expected_attention_signature not in statement
    )
    producer_functions = Counter(function for function, _, _ in expected_attention)
    if len(all_attention) != expected_attention_total:
        violations.append(
            "attention producer count/variant drifted: "
            f"expected={expected_attention_total} observed={len(all_attention)}"
        )
    if len(expected_attention) != expected_attention_total or malformed_attention:
        violations.append(
            "attention producer geometry drifted: "
            f"matched={len(expected_attention)} malformed={malformed_attention}"
        )
    expected_distribution = sorted([expected_attention_per_chunk] * 4)
    if sorted(producer_functions.values()) != expected_distribution:
        violations.append(
            "attention producer chunk distribution drifted: "
            f"observed={dict(producer_functions)}"
        )
    if set(producer_functions) - reachable:
        violations.append(
            "attention producer exists only in unreachable functions: "
            f"{sorted(set(producer_functions) - reachable)}"
        )

    down_signature = (
        full_down_signature if full_width_rounded_then_slice else half_down_signature
    )
    opposite_down_signature = (
        half_down_signature if full_width_rounded_then_slice else full_down_signature
    )
    all_convolutions = tuple(
        statement
        for body in statements.values()
        for statement in body
        if "stablehlo.convolution" in statement
    )
    observed_down = sum(down_signature in item for item in all_convolutions)
    opposite_down = sum(opposite_down_signature in item for item in all_convolutions)
    observed_gate = sum(gate_signature in item for item in all_convolutions)
    if observed_down != expected_down_total or opposite_down:
        violations.append(
            "dense-down convolution variant drifted: "
            f"expected={expected_down_total} observed={observed_down} "
            f"opposite={opposite_down}"
        )
    if observed_gate != 64:
        violations.append(
            f"dense gate convolution count drifted: expected=64 observed={observed_gate}"
        )
    expected_function_set = set(producer_functions)
    if any(
        down_by_function[function] != expected_down_per_chunk
        or gate_by_function[function] != 16
        or reducer_by_function[function] != 2
        for function in expected_function_set
    ):
        violations.append(
            "producer/reducer per-chunk lineage drifted: "
            f"down={dict(down_by_function)} gate={dict(gate_by_function)} "
            f"half_reducers={dict(reducer_by_function)}"
        )

    return (
        {
            "attention_kernel": expected_kernel,
            "attention_producer_count": len(expected_attention),
            "dense_down_convolution_count": observed_down,
            "dense_gate_convolution_count": observed_gate,
            "full_width_rounded_then_slice": full_width_rounded_then_slice,
            "half_reducer_count": sum(reducer_by_function.values()),
            "producer_chunk_count": len(producer_functions),
            "reachable_function_count": len(reachable),
        },
        violations,
    )


def _live_optimized_instruction_keys(
    instructions: tuple[HloInstruction, ...],
) -> tuple[frozenset[tuple[str, str]], int]:
    """Trace live instructions and called computations from the ENTRY root."""

    by_computation: dict[str, list[HloInstruction]] = defaultdict(list)
    for instruction in instructions:
        by_computation[instruction.computation].append(instruction)

    def symbol(computation: str) -> str:
        value = computation.removeprefix("ENTRY ")
        return value.split(" ", 1)[0]

    computation_by_symbol = {
        symbol(computation): computation for computation in by_computation
    }
    if len(computation_by_symbol) != len(by_computation):
        raise BenchmarkValidationError(
            "feature2 optimized HLO has ambiguous computation symbols"
        )
    entries = tuple(
        computation
        for computation in by_computation
        if computation.startswith("ENTRY ")
    )
    if len(entries) != 1:
        raise BenchmarkValidationError(
            f"feature2 optimized HLO requires one ENTRY computation, found {len(entries)}"
        )

    live: set[tuple[str, str]] = set()
    seen_computations: set[str] = set()
    pending_computations: deque[str] = deque(entries)
    while pending_computations:
        computation = pending_computations.popleft()
        if computation in seen_computations:
            continue
        seen_computations.add(computation)
        local = {
            instruction.name: instruction for instruction in by_computation[computation]
        }
        roots = tuple(
            instruction
            for instruction in by_computation[computation]
            if instruction.raw_line.startswith("ROOT ")
        )
        if len(roots) != 1:
            raise BenchmarkValidationError(
                "feature2 optimized HLO reachable computation has invalid root "
                f"count: computation={computation!r} roots={len(roots)}"
            )
        pending_instructions = [roots[0]]
        while pending_instructions:
            instruction = pending_instructions.pop()
            key = (computation, instruction.name)
            if key in live:
                continue
            live.add(key)
            pending_instructions.extend(
                local[name] for name in instruction.operand_names if name in local
            )
            for reference in _HLO_COMPUTATION_REFERENCE.findall(instruction.raw_line):
                called = computation_by_symbol.get(reference)
                if called is not None and called not in seen_computations:
                    pending_computations.append(called)
    return frozenset(live), len(seen_computations)


def _optimized_terminal_boundary_contract(
    instructions: tuple[HloInstruction, ...],
    root: HloInstruction | None,
    live_keys: frozenset[tuple[str, str]],
    *,
    sealed_boundary_capture: bool,
    sealed_canonical_identity_authenticated: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Require exact root arity/types and authenticated diagnostic producers.

    TPU optimization is allowed to erase identity barriers and their source
    names. The sealed successor therefore binds the complete executable
    producer graph with its separately validated canonical HLO identity rather
    than accepting weaker surviving-name or same-shape ancestry heuristics.
    """

    expected = (
        _SEALED_BOUNDARY_LOCAL_MAIN_ROOT
        if sealed_boundary_capture
        else _HISTORICAL_LOCAL_MAIN_ROOT
    )
    violations: list[str] = []
    if root is None or root.raw_opcode != "tuple":
        return (
            {"output_count": 0, "root_operands": [], "sealed_bindings": {}},
            ["optimized ENTRY requires one tuple root"],
        )
    if len(root.operand_names) != len(expected):
        violations.append(
            "optimized root operand count drifted: "
            f"expected={len(expected)} observed={len(root.operand_names)}"
        )
    local = {
        instruction.name: instruction
        for instruction in instructions
        if instruction.computation == root.computation
    }
    for index, (operand, expected_shape) in enumerate(
        zip(root.operand_names, expected, strict=False)
    ):
        source = local.get(operand)
        observed = () if source is None else _shape_signature(source.result_shapes)
        if observed != (expected_shape,):
            violations.append(
                "optimized root operand geometry drifted: "
                f"index={index} operand={operand} expected={expected_shape} "
                f"observed={observed}"
            )

    sealed_bindings: dict[str, str] = {}
    if sealed_boundary_capture:
        if not sealed_canonical_identity_authenticated:
            violations.append(
                "optimized sealed roots lack the authenticated canonical-HLO "
                "producer identity"
            )
        elif all(
            index < len(root.operand_names)
            and local.get(root.operand_names[index]) is not None
            and _shape_signature(local[root.operand_names[index]].result_shapes)
            == (expected[index],)
            and (root.computation, root.operand_names[index]) in live_keys
            for index in _SEALED_BOUNDARY_ROOT_MARKERS
        ):
            sealed_bindings = {
                str(index): marker
                for index, marker in _SEALED_BOUNDARY_ROOT_MARKERS.items()
            }
        else:
            violations.append(
                "optimized sealed canonical roots lost exact live operand geometry"
            )
    return (
        {
            "output_count": len(root.operand_names),
            "root_operands": list(root.operand_names),
            "sealed_bindings": sealed_bindings,
        },
        violations,
    )


def _optimized_projection_producer_kind(
    instruction: HloInstruction,
) -> Literal["attention", "dense_down"] | None:
    if (
        instruction.raw_opcode == "custom-call"
        and _FP8_ATTENTION_O_PREFIX in instruction.raw_line
        and 'custom_call_target="tpu_custom_call"' in instruction.raw_line
    ):
        return "attention"
    if (
        instruction.raw_opcode == "convolution"
        and instruction.op_name is not None
        and "greenfield_dense_feature2_virtual_rank_" in instruction.op_name
        and _shape_signature(instruction.result_shapes)
        in ((("f32", (1, 3072)),), (("f32", (1, 6144)),))
    ):
        return "dense_down"
    return None


def _optimized_slice_ranges(
    instruction: HloInstruction,
) -> tuple[tuple[int, int, int], ...] | None:
    """Parse one static optimized-HLO slice without normalizing ownership."""

    if instruction.raw_opcode != "slice":
        return None
    match = re.search(r"\bslice=\{([^}]*)\}", instruction.raw_line)
    if match is None:
        return None
    raw_ranges = match.group(1)
    if re.fullmatch(r"\s*\[[^]]+\](?:\s*,\s*\[[^]]+\])*\s*", raw_ranges) is None:
        return None
    ranges: list[tuple[int, int, int]] = []
    for raw_dimension in re.findall(r"\[([^]]+)\]", raw_ranges):
        fields = tuple(field.strip() for field in raw_dimension.split(":"))
        if len(fields) not in (2, 3) or any(
            re.fullmatch(r"[0-9]+", field) is None for field in fields
        ):
            return None
        start, limit = (int(field) for field in fields[:2])
        stride = 1 if len(fields) == 2 else int(fields[2])
        ranges.append((start, limit, stride))
    if any(stride <= 0 for _, _, stride in ranges):
        return None
    return tuple(ranges)


def _optimized_executable_prefix(instruction: HloInstruction) -> str:
    """Return opcode text before non-executable provenance/config fields."""

    prefix = instruction.raw_line
    boundaries = tuple(
        index
        for marker in (
            ", metadata=",
            ", backend_config=",
            ", frontend_attributes=",
        )
        if (index := prefix.find(marker)) >= 0
    )
    return prefix if not boundaries else prefix[: min(boundaries)]


def _optimized_compare_direction(instruction: HloInstruction) -> str | None:
    if instruction.raw_opcode != "compare":
        return None
    executable = _optimized_executable_prefix(instruction)
    match = re.search(r"(?:^|,\s*)direction=([A-Z]+)(?=,|$)", executable)
    return None if match is None else match.group(1)


def _optimized_constant_literal(instruction: HloInstruction) -> str | None:
    if instruction.raw_opcode != "constant":
        return None
    matches = re.findall(
        r"\bconstant\(([^()]*)\)",
        _optimized_executable_prefix(instruction),
    )
    if len(matches) != 1:
        return None
    return matches[0].strip()


def _optimized_reducer_lineage(
    instructions: tuple[HloInstruction, ...],
    reducer: HloInstruction,
    *,
    full_width_rounded_then_slice: bool,
) -> tuple[
    tuple[tuple[int, HloInstruction], ...],
    tuple[tuple[str, int, HloInstruction], ...],
    dict[str, Any] | None,
]:
    """Trace one reducer input through fusion parameters to producer leaves.

    The optimized TPU module represents the two half stacks and their local
    y-reductions with nested fusions.  A computation-local walk is therefore
    insufficient: it can count every producer and every reducer while still
    allowing the same-shaped attention and dense reducer operands to be
    exchanged.  Frames below bind each callee parameter to the exact caller
    operand and preserve tuple-result selection.
    """

    by_computation: dict[str, dict[str, HloInstruction]] = defaultdict(dict)
    roots: dict[str, HloInstruction] = {}
    for instruction in instructions:
        local = by_computation[instruction.computation]
        if instruction.name in local:
            raise BenchmarkValidationError(
                "feature2 optimized HLO duplicates an instruction name in "
                f"{instruction.computation}: {instruction.name}"
            )
        local[instruction.name] = instruction
        if instruction.raw_line.startswith("ROOT "):
            if instruction.computation in roots:
                raise BenchmarkValidationError(
                    "feature2 optimized HLO duplicates a computation root: "
                    f"{instruction.computation}"
                )
            roots[instruction.computation] = instruction

    def symbol(computation: str) -> str:
        return computation.removeprefix("ENTRY ").split(" ", 1)[0]

    computation_by_symbol = {
        symbol(computation): computation for computation in by_computation
    }
    if len(computation_by_symbol) != len(by_computation):
        raise BenchmarkValidationError(
            "feature2 optimized HLO has ambiguous computation symbols"
        )

    # frame -> (computation, ((parent frame, caller operand), ...))
    frames: dict[int, tuple[str, tuple[tuple[int, str], ...]]] = {
        0: (reducer.computation, ())
    }
    child_frames: dict[tuple[int, int, str], int] = {}

    def called_computation(instruction: HloInstruction) -> str | None:
        matches = tuple(_HLO_CALLED_COMPUTATION.findall(instruction.raw_line))
        if len(matches) > 1:
            raise BenchmarkValidationError(
                "feature2 optimized HLO instruction calls multiple computations: "
                f"{instruction.name}"
            )
        if not matches:
            return None
        called = computation_by_symbol.get(matches[0])
        if called is None or called not in roots:
            raise BenchmarkValidationError(
                "feature2 optimized HLO calls a missing/rootless computation: "
                f"{matches[0]}"
            )
        return called

    def descend_call(
        frame_id: int,
        instruction: HloInstruction,
        selected_index: int | None,
        pending: list[tuple[int, str, int | None]],
    ) -> None:
        called = called_computation(instruction)
        if called is None:
            raise BenchmarkValidationError(
                "feature2 optimized HLO tuple selection lost its called "
                f"computation: {instruction.name}"
            )
        child_key = (frame_id, instruction.index, called)
        child_id = child_frames.get(child_key)
        if child_id is None:
            child_id = len(frames)
            child_frames[child_key] = child_id
            frames[child_id] = (
                called,
                tuple((frame_id, operand) for operand in instruction.operand_names),
            )
        root = roots[called]
        if selected_index is not None:
            if root.raw_opcode == "tuple":
                if selected_index >= len(root.operand_names):
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO tuple result index exceeds root: "
                        f"call={instruction.name} index={selected_index}"
                    )
                pending.append((child_id, root.operand_names[selected_index], None))
                return
            if selected_index != 0:
                raise BenchmarkValidationError(
                    "feature2 optimized HLO selects a non-tuple call result: "
                    f"call={instruction.name} index={selected_index}"
                )
        pending.append((child_id, root.name, None))

    def walk(
        seeds: list[tuple[int, str, int | None]],
    ) -> tuple[
        dict[tuple[int, str], HloInstruction],
        dict[tuple[str, int, str], HloInstruction],
    ]:
        pending = list(seeds)
        visited: dict[tuple[int, str], HloInstruction] = {}
        frontiers: dict[tuple[str, int, str], HloInstruction] = {}
        while pending:
            if len(visited) > 100_000:
                raise BenchmarkValidationError(
                    "feature2 optimized reducer lineage exceeded its finite bound"
                )
            frame_id, name, selected_index = pending.pop()
            computation, bindings = frames[frame_id]
            instruction = by_computation[computation].get(name)
            if instruction is None:
                if not name.startswith("%"):
                    # Literal operands (for example ``-inf``) are not SSA edges.
                    continue
                raise BenchmarkValidationError(
                    "feature2 optimized reducer lineage references a missing local "
                    f"instruction: computation={computation} name={name}"
                )
            key = (frame_id, name)
            if key in visited and selected_index is None:
                continue
            visited[key] = instruction

            producer_kind = _optimized_projection_producer_kind(instruction)
            if producer_kind is not None:
                frontiers[(producer_kind, frame_id, name)] = instruction
                continue

            if instruction.raw_opcode == "parameter":
                match = _HLO_PARAMETER_INDEX.search(instruction.raw_line)
                if match is None:
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO has an unindexed parameter: "
                        f"{instruction.name}"
                    )
                parameter_index = int(match.group(1))
                if bindings:
                    if parameter_index >= len(bindings):
                        raise BenchmarkValidationError(
                            "feature2 optimized HLO callee parameter exceeds caller "
                            f"operands: computation={computation} "
                            f"index={parameter_index}"
                        )
                    pending.append((*bindings[parameter_index], None))
                continue

            if instruction.raw_opcode == "get-tuple-element":
                if len(instruction.operand_names) != 1:
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO tuple selection has invalid arity: "
                        f"{instruction.name}"
                    )
                match = _HLO_TUPLE_INDEX.search(instruction.raw_line)
                if match is None:
                    raise BenchmarkValidationError(
                        "feature2 optimized HLO tuple selection has no index: "
                        f"{instruction.name}"
                    )
                source = by_computation[computation].get(instruction.operand_names[0])
                if source is not None and called_computation(source) is not None:
                    visited[(frame_id, source.name)] = source
                    descend_call(
                        frame_id,
                        source,
                        int(match.group(1)),
                        pending,
                    )
                    continue

            called = called_computation(instruction)
            if called is not None:
                descend_call(frame_id, instruction, selected_index, pending)
                continue
            pending.extend(
                (frame_id, operand, None) for operand in instruction.operand_names
            )
        return visited, frontiers

    visited, frontiers = walk([(0, operand, None) for operand in reducer.operand_names])
    ownership: dict[str, Any] | None = None
    if full_width_rounded_then_slice:
        owner_selects = tuple(
            (frame_id, instruction)
            for (frame_id, _), instruction in visited.items()
            if instruction.raw_opcode == "select"
            and _shape_signature(instruction.result_shapes) == (("bf16", (4, 1, 3072)),)
        )
        if len(owner_selects) == 1 and len(owner_selects[0][1].operand_names) == 3:
            select_frame, owner_select = owner_selects[0]
            branch_walks = {
                "predicate": walk(
                    [(select_frame, owner_select.operand_names[0], None)]
                ),
                "true": walk([(select_frame, owner_select.operand_names[1], None)]),
                "false": walk([(select_frame, owner_select.operand_names[2], None)]),
            }
            branch_reports: dict[str, Any] = {}
            for branch in ("true", "false"):
                branch_visited, branch_frontiers = branch_walks[branch]
                half_slices = tuple(
                    (frame_id, instruction)
                    for (frame_id, _), instruction in branch_visited.items()
                    if instruction.raw_opcode == "slice"
                    and len(instruction.operand_shapes) == 1
                    and len(instruction.result_shapes) == 1
                    and instruction.operand_shapes[0].dimensions[-1:] == (6144,)
                    and instruction.result_shapes[0].dimensions[-1:] == (3072,)
                )
                slice_frontiers: list[tuple[tuple[str, int, str], ...]] = []
                for frame_id, half_slice in half_slices:
                    if len(half_slice.operand_names) != 1:
                        slice_frontiers.append(())
                        continue
                    _, traced = walk([(frame_id, half_slice.operand_names[0], None)])
                    slice_frontiers.append(tuple(sorted(traced)))
                branch_reports[branch] = {
                    "frontier_keys": tuple(sorted(branch_frontiers)),
                    "slice_frontiers": tuple(slice_frontiers),
                    "slice_ranges": tuple(
                        _optimized_slice_ranges(instruction)
                        for _, instruction in half_slices
                    ),
                    "stack_count": sum(
                        instruction.raw_opcode == "concatenate"
                        and _shape_signature(instruction.result_shapes)
                        == (("bf16", (16, 1, 3072)),)
                        for instruction in branch_visited.values()
                    ),
                }
            predicate_visited, predicate_frontiers = branch_walks["predicate"]
            predicate_compares = tuple(
                (frame_id, instruction)
                for (frame_id, _), instruction in predicate_visited.items()
                if instruction.raw_opcode == "compare"
                and _optimized_compare_direction(instruction) == "EQ"
            )

            def local_operand(
                frame_id: int,
                instruction: HloInstruction,
                operand_index: int,
            ) -> HloInstruction | None:
                if operand_index >= len(instruction.operand_names):
                    return None
                computation = frames[frame_id][0]
                return by_computation[computation].get(
                    instruction.operand_names[operand_index]
                )

            causal_eq_zero_partition_id = False
            if len(predicate_compares) == 1:
                compare_frame, compare = predicate_compares[0]
                axis_s32 = local_operand(compare_frame, compare, 0)
                zero_s32 = local_operand(compare_frame, compare, 1)
                if axis_s32 is not None and axis_s32.raw_opcode == "convert":
                    masked_u32 = local_operand(compare_frame, axis_s32, 0)
                else:
                    masked_u32 = None
                if masked_u32 is not None and masked_u32.raw_opcode == "and":
                    partition_id = local_operand(compare_frame, masked_u32, 0)
                    mask_one = local_operand(compare_frame, masked_u32, 1)
                else:
                    partition_id = None
                    mask_one = None
                causal_eq_zero_partition_id = bool(
                    _shape_signature(compare.operand_shapes)
                    == (("s32", ()), ("s32", ()))
                    and _shape_signature(compare.result_shapes) == (("pred", ()),)
                    and axis_s32 is not None
                    and _shape_signature(axis_s32.operand_shapes) == (("u32", ()),)
                    and _shape_signature(axis_s32.result_shapes) == (("s32", ()),)
                    and masked_u32 is not None
                    and _shape_signature(masked_u32.operand_shapes)
                    == (("u32", ()), ("u32", ()))
                    and _shape_signature(masked_u32.result_shapes) == (("u32", ()),)
                    and partition_id is not None
                    and partition_id.raw_opcode == "partition-id"
                    and _shape_signature(partition_id.result_shapes) == (("u32", ()),)
                    and mask_one is not None
                    and mask_one.raw_opcode == "constant"
                    and _shape_signature(mask_one.result_shapes) == (("u32", ()),)
                    and _optimized_constant_literal(mask_one) == "1"
                    and zero_s32 is not None
                    and zero_s32.raw_opcode == "constant"
                    and _shape_signature(zero_s32.result_shapes) == (("s32", ()),)
                    and _optimized_constant_literal(zero_s32) == "0"
                )
            branch_reports["predicate"] = {
                "causal_eq_zero_partition_id": causal_eq_zero_partition_id,
                "compare_eq_count": sum(
                    instruction.raw_opcode == "compare"
                    and _optimized_compare_direction(instruction) == "EQ"
                    for instruction in predicate_visited.values()
                ),
                "constant_zero_count": sum(
                    instruction.raw_opcode == "constant"
                    and _optimized_constant_literal(instruction) == "0"
                    for instruction in predicate_visited.values()
                ),
                "frontier_count": len(predicate_frontiers),
                "partition_id_count": sum(
                    instruction.raw_opcode == "partition-id"
                    for instruction in predicate_visited.values()
                ),
            }
            ownership = {
                "owner_select_count": 1,
                **branch_reports,
            }
        else:
            ownership = {
                "owner_select_count": len(owner_selects),
            }

    return (
        tuple(sorted(visited.items(), key=lambda item: item[0])),
        tuple(
            (kind, frame_id, instruction)
            for (kind, frame_id, _), instruction in sorted(frontiers.items())
        ),
        ownership,
    )


def validate_feature2_materializer_optimized_hlo(
    optimized_hlo: str,
    *,
    phase: Feature2MaterializerPhase,
) -> dict[str, Any]:
    """Require three separate owner-local weight adaptation executables."""

    if phase not in ("query_fp32", "wk_decode_bf16", "wk_promote_fp32"):
        raise ValueError(f"unknown feature2 materializer phase: {phase!r}")
    module = parse_hlo_module(optimized_hlo)
    instructions = module.instructions
    root = _entry_root(instructions)
    if phase == "query_fp32":
        expected_parameters = {
            ("u8", (1, 2048, 2048)): 2,
            ("f32", (1, 16, 16)): 2,
        }
        expected_root = (("f32", (1, 2048, 2048)),) * 2
    elif phase == "wk_decode_bf16":
        expected_parameters = {
            ("u8", (1, 128, 6144)): 2,
            ("f32", (1, 1, 48)): 2,
        }
        expected_root = (("bf16", (1, 128, 6144)),) * 2
    else:
        expected_parameters = {("bf16", (1, 128, 6144)): 2}
        expected_root = (("f32", (1, 128, 6144)),) * 2

    observed_parameters = {
        f"{dtype}{dimensions}": _entry_parameter_count(instructions, dtype, dimensions)
        for (dtype, dimensions) in expected_parameters
    }
    collectives = tuple(module.collectives)
    forbidden_transport = tuple(
        instruction.name
        for instruction in instructions
        if instruction.raw_opcode in _FORBIDDEN_TRANSPORT_OPCODES
    )
    custom_targets = tuple(
        sorted(set(re.findall(r'custom_call_target="([^"]+)"', optimized_hlo)))
    )
    allowed_metadata_targets = {
        "AssumeGatherIndicesInBound",
        "GatherScatterIndicesBitpacked",
    }
    forbidden_custom_targets = tuple(
        target for target in custom_targets if target not in allowed_metadata_targets
    )
    root_shapes = () if root is None else _shape_signature(root.result_shapes)
    violations: list[str] = []
    if module.num_partitions != 2:
        violations.append(f"expected two partitions, found {module.num_partitions}")
    for (dtype, dimensions), count in expected_parameters.items():
        observed = _entry_parameter_count(instructions, dtype, dimensions)
        if observed != count:
            violations.append(
                "entry parameter count drifted for "
                f"{dtype}{dimensions}: expected={count} observed={observed}"
            )
    if root_shapes != expected_root:
        violations.append(
            f"root boundary drifted: expected={expected_root} observed={root_shapes}"
        )
    if collectives:
        violations.append("materializer contains physical communication")
    if forbidden_transport:
        violations.append(
            f"materializer contains device/host transport: {forbidden_transport}"
        )
    markers = _host_markers(optimized_hlo)
    if markers:
        violations.append(f"materializer contains host markers: {markers}")
    if forbidden_custom_targets:
        violations.append(
            f"materializer contains unapproved custom calls: {forbidden_custom_targets}"
        )
    for marker in ("u8[2,128,6144]", "f32[2,128,6144]", "f32[2,2048,2048]"):
        if marker in optimized_hlo:
            violations.append(f"materializer reconstructs cross-owner state {marker}")
    _raise(f"feature2 {phase} optimized HLO", violations)
    return {
        "collective_count": len(collectives),
        "custom_call_targets": list(custom_targets),
        "entry_parameter_counts": observed_parameters,
        "host_markers": list(markers),
        "instruction_count": len(instructions),
        "num_partitions": module.num_partitions,
        "passed": True,
        "phase": phase,
        "root_shapes": [
            {"dtype": dtype, "shape": list(shape)} for dtype, shape in root_shapes
        ],
    }


def validate_feature2_main_stablehlo(
    stablehlo: str,
    *,
    full_width_rounded_then_slice: bool = False,
    sealed_boundary_capture: bool = False,
) -> dict[str, Any]:
    """Pin the source-level manual LP2 boundary before TPU optimization."""

    if not isinstance(stablehlo, str) or not stablehlo.strip():
        raise BenchmarkValidationError("feature2 main StableHLO is empty")
    if not isinstance(full_width_rounded_then_slice, bool):
        raise TypeError("feature2 StableHLO variant flag must be boolean")
    if not isinstance(sealed_boundary_capture, bool):
        raise TypeError("feature2 StableHLO sealed-boundary flag must be boolean")
    compact = re.sub(r"\s+", "", stablehlo)
    markers = _host_markers(stablehlo)
    violations: list[str] = []
    stablehlo_sha256 = sha256(stablehlo.encode()).hexdigest()
    if sealed_boundary_capture and stablehlo_sha256 != FEATURE2_SEALED_STABLEHLO_SHA256:
        violations.append(
            "sealed StableHLO identity drifted: "
            f"expected={FEATURE2_SEALED_STABLEHLO_SHA256} "
            f"observed={stablehlo_sha256}"
        )
    if "mhlo.num_partitions=2:i32" not in compact:
        violations.append("StableHLO lost the exact two-partition module")
    required = (
        "tensor<8156xi32>",
        "tensor<1x16xi32>",
        "tensor<8192x64xbf16>",
        "tensor<1x2048xi32>",
        "tensor<1x2048xf32>",
        "tensor<2x1x3072xbf16>",
        "tensor<2x16x256x640xbf16>",
        "tensor<2x16x256x128xbf16>",
        "tensor<2x2xui32>",
    )
    if sealed_boundary_capture:
        required += (
            "tensor<2x1x6144xbf16>",
            "tensor<2x1x2048xbf16>",
            "tensor<2x1x32x128xf32>",
            "tensor<2x1x32xf32>",
        )
    missing = tuple(marker for marker in required if marker not in compact)
    if missing:
        violations.append(f"StableHLO boundary shapes drifted: {missing}")
    forbidden = tuple(
        marker
        for marker in (
            "tensor<32x6144xbf16>",
            "tensor<8156x6144xbf16>",
            "stablehlo.infeed",
            "stablehlo.outfeed",
            "stablehlo.recv",
            "stablehlo.send",
        )
        if marker in compact
    )
    if forbidden:
        violations.append(f"StableHLO contains forbidden work/state: {forbidden}")
    if markers:
        violations.append(f"StableHLO contains host markers: {markers}")
    # Every explicitly stated collective group must be the single LP2 pair.
    bad_groups = tuple(
        match.group(0)
        for match in re.finditer(r"replica_groups=dense<([^>]*)>", compact)
        if match.group(1) not in ("[[0,1]]", "[0,1]")
    )
    if bad_groups:
        violations.append(f"StableHLO collective group escaped LP2: {bad_groups}")
    bad_pairs = tuple(
        match.group(0)
        for match in re.finditer(r"source_target_pairs=dense<([^>]*)>", compact)
        if match.group(1) != "[[0,1],[1,0]]"
    )
    if bad_pairs:
        violations.append(f"StableHLO permute pairs drifted: {bad_pairs}")
    if _H16_ATTENTION_KERNEL not in stablehlo:
        violations.append("StableHLO lost the exact H16/B512 attention kernel")
    terminal_contract, terminal_violations = _stablehlo_main_boundary_contract(
        stablehlo,
        sealed_boundary_capture=sealed_boundary_capture,
    )
    violations.extend(terminal_violations)
    projection_contract, projection_violations = _stablehlo_projection_contract(
        stablehlo,
        full_width_rounded_then_slice=full_width_rounded_then_slice,
    )
    violations.extend(projection_violations)
    if "tensor<16x1x6144xbf16>" in compact:
        violations.append("StableHLO stacks full-width producer leaves")
    _raise("feature2 main StableHLO", violations)
    return {
        "exact_h16_attention_present": True,
        "forbidden_markers": [],
        "host_markers": [],
        "num_partitions": 2,
        "passed": True,
        "projection_contract": projection_contract,
        "sealed_boundary_capture": sealed_boundary_capture,
        "stablehlo_sha256": stablehlo_sha256,
        **terminal_contract,
    }


def validate_feature2_main_optimized_hlo(
    optimized_hlo: str,
    *,
    full_width_rounded_then_slice: bool = False,
    sealed_boundary_capture: bool = False,
) -> dict[str, Any]:
    """Parse and bound the optimized two-device feature2 executable."""

    if not isinstance(full_width_rounded_then_slice, bool):
        raise TypeError("feature2 optimized-HLO variant flag must be boolean")
    if not isinstance(sealed_boundary_capture, bool):
        raise TypeError("feature2 optimized-HLO sealed-boundary flag must be boolean")
    sealed_canonical_identity: dict[str, Any] = {}
    sealed_canonical_identity_authenticated = False
    canonical_identity_violation: str | None = None
    if sealed_boundary_capture:
        try:
            _, sealed_canonical_identity = canonicalize_feature2_optimized_hlo(
                optimized_hlo
            )
        except BenchmarkValidationError as error:
            canonical_identity_violation = (
                f"sealed optimized HLO canonicalization refused: {error}"
            )
        else:
            expected_identity = {
                "byte_count": FEATURE2_SEALED_OPTIMIZED_CANONICAL_BYTES,
                "canonicalizer_version": FEATURE2_OPTIMIZED_HLO_CANONICALIZER_VERSION,
                "sha256": FEATURE2_SEALED_OPTIMIZED_CANONICAL_SHA256,
                "stripped_stack_frame_references": (
                    FEATURE2_SEALED_OPTIMIZED_STACK_FRAME_REFERENCES
                ),
            }
            observed_identity = {
                key: sealed_canonical_identity.get(key) for key in expected_identity
            }
            sealed_canonical_identity_authenticated = (
                observed_identity == expected_identity
            )
            if not sealed_canonical_identity_authenticated:
                canonical_identity_violation = (
                    "sealed optimized canonical-HLO identity drifted: "
                    f"expected={expected_identity} observed={observed_identity}"
                )
    module = parse_hlo_module(optimized_hlo)
    instructions = module.instructions
    live_keys, reachable_computation_count = _live_optimized_instruction_keys(
        instructions
    )
    root = _entry_root(instructions)
    root_shapes = () if root is None else _shape_signature(root.result_shapes)
    collectives = tuple(module.collectives)
    collective_counts = Counter(item.opcode for item in collectives)
    violations: list[str] = []
    if canonical_identity_violation is not None:
        violations.append(canonical_identity_violation)
    if module.num_partitions != 2:
        violations.append(f"expected two partitions, found {module.num_partitions}")
    expected_root = (
        _SEALED_BOUNDARY_LOCAL_MAIN_ROOT
        if sealed_boundary_capture
        else _HISTORICAL_LOCAL_MAIN_ROOT
    )
    if root_shapes != expected_root:
        violations.append(
            "optimized terminal boundary drifted: "
            f"expected={expected_root} observed={root_shapes}"
        )
    terminal_contract, terminal_violations = _optimized_terminal_boundary_contract(
        instructions,
        root,
        live_keys,
        sealed_boundary_capture=sealed_boundary_capture,
        sealed_canonical_identity_authenticated=(
            sealed_canonical_identity_authenticated
        ),
    )
    violations.extend(terminal_violations)
    allowed_collectives = {"all-gather", "all-reduce", "collective-permute"}
    for collective in collectives:
        if collective.opcode not in allowed_collectives:
            violations.append(f"forbidden optimized collective {collective.opcode}")
        if collective.opcode == "collective-permute":
            if collective.source_target_pairs != ((0, 1), (1, 0)):
                violations.append(
                    f"feature permute pairs drifted: {collective.source_target_pairs}"
                )
        elif collective.replica_groups != ((0, 1),):
            violations.append(f"collective escaped LP2: {collective.replica_groups}")
        elif not collective.use_global_device_ids:
            violations.append(
                f"collective lost explicit global device ids: {collective.name}"
            )
        if collective.maximum_group_size > 2:
            violations.append(
                f"collective group exceeds LP2: {collective.maximum_group_size}"
            )

    forbidden_transport = tuple(
        instruction.name
        for instruction in instructions
        if instruction.raw_opcode in _FORBIDDEN_TRANSPORT_OPCODES
    )
    if forbidden_transport:
        violations.append(f"optimized graph contains transport: {forbidden_transport}")
    markers = _host_markers(optimized_hlo)
    if markers:
        violations.append(f"optimized graph contains host markers: {markers}")
    forbidden_shapes = tuple(
        marker
        for marker in (
            "bf16[32,6144]",
            "bf16[8156,6144]",
            "bf16[16,1,6144]",
            "bf16[4,1,6144]",
        )
        if marker in optimized_hlo
    )
    if forbidden_shapes:
        violations.append(
            f"optimized graph reconstructs forbidden hidden state: {forbidden_shapes}"
        )
    h16_calls = tuple(
        instruction
        for instruction in instructions
        if instruction.raw_opcode == "custom-call"
        and _H16_ATTENTION_KERNEL in instruction.raw_line
        and 'custom_call_target="tpu_custom_call"' in instruction.raw_line
    )
    if len(h16_calls) != 8:
        violations.append(
            f"exact H16/B512 call count drifted: expected=8 observed={len(h16_calls)}"
        )
    if optimized_hlo.count("h32_k2048"):
        violations.append("optimized graph contains forbidden H32 attention")

    expected_width = 6144 if full_width_rounded_then_slice else 3072
    expected_attention_count = 64 if full_width_rounded_then_slice else 128
    expected_down_count = 64 if full_width_rounded_then_slice else 128
    expected_attention_kernel = f"{_FP8_ATTENTION_O_PREFIX}{expected_width}"
    attention_producers = tuple(
        instruction
        for instruction in instructions
        if instruction.raw_opcode == "custom-call"
        and _FP8_ATTENTION_O_PREFIX in instruction.raw_line
        and 'custom_call_target="tpu_custom_call"' in instruction.raw_line
    )
    expected_attention_geometry = (
        (
            ("bf16", (8, 512)),
            ("u8", (expected_width, 512)),
            ("f32", (8, 128)),
        ),
        (("bf16", (8, expected_width)),),
    )
    malformed_attention = tuple(
        instruction.name
        for instruction in attention_producers
        if (
            expected_attention_kernel not in instruction.raw_line
            or (
                _shape_signature(instruction.operand_shapes),
                _shape_signature(instruction.result_shapes),
            )
            != expected_attention_geometry
        )
    )
    live_attention_count = sum(
        (instruction.computation, instruction.name) in live_keys
        for instruction in attention_producers
    )
    if (
        len(attention_producers) != expected_attention_count
        or live_attention_count != expected_attention_count
        or malformed_attention
    ):
        violations.append(
            "optimized attention producer variant/geometry/liveness drifted: "
            f"expected={expected_attention_count} total={len(attention_producers)} "
            f"live={live_attention_count} malformed={malformed_attention}"
        )

    convolution_instructions = tuple(
        instruction
        for instruction in instructions
        if instruction.raw_opcode == "convolution"
    )
    gate_geometry = (
        (("bf16", (1, 6144)), ("bf16", (6144, 768))),
        (("f32", (1, 768)),),
    )
    down_geometry = (
        (("bf16", (1, 384)), ("bf16", (384, expected_width))),
        (("f32", (1, expected_width)),),
    )
    opposite_width = 3072 if full_width_rounded_then_slice else 6144
    opposite_down_geometry = (
        (("bf16", (1, 384)), ("bf16", (384, opposite_width))),
        (("f32", (1, opposite_width)),),
    )

    def convolution_geometry(instruction: HloInstruction) -> tuple[Any, Any]:
        return (
            _shape_signature(instruction.operand_shapes),
            _shape_signature(instruction.result_shapes),
        )

    gate_convolutions = tuple(
        instruction
        for instruction in convolution_instructions
        if convolution_geometry(instruction) == gate_geometry
        and instruction.op_name is not None
        and "greenfield_dense_feature2_virtual_rank_" in instruction.op_name
    )
    down_convolutions = tuple(
        instruction
        for instruction in convolution_instructions
        if convolution_geometry(instruction) == down_geometry
        and instruction.op_name is not None
        and "greenfield_dense_feature2_virtual_rank_" in instruction.op_name
    )
    opposite_down_convolutions = tuple(
        instruction
        for instruction in convolution_instructions
        if convolution_geometry(instruction) == opposite_down_geometry
        and instruction.op_name is not None
        and "greenfield_dense_feature2_virtual_rank_" in instruction.op_name
    )
    live_gate_count = sum(
        (instruction.computation, instruction.name) in live_keys
        for instruction in gate_convolutions
    )
    live_down_count = sum(
        (instruction.computation, instruction.name) in live_keys
        for instruction in down_convolutions
    )
    if len(gate_convolutions) != 64 or live_gate_count != 64:
        violations.append(
            "optimized dense gate geometry/liveness drifted: "
            f"expected=64 total={len(gate_convolutions)} live={live_gate_count}"
        )
    if (
        len(down_convolutions) != expected_down_count
        or live_down_count != expected_down_count
        or opposite_down_convolutions
    ):
        violations.append(
            "optimized dense-down variant/geometry/liveness drifted: "
            f"expected={expected_down_count} total={len(down_convolutions)} "
            f"live={live_down_count} opposite={len(opposite_down_convolutions)}"
        )

    reducer_counts: dict[str, int] = {}
    reducer_lineage: dict[str, list[dict[str, int]]] = {}
    for scope in (_ATTENTION_REDUCER_SCOPE, _DENSE_REDUCER_SCOPE):
        scoped = tuple(
            collective
            for collective in collectives
            if collective.opcode == "collective-permute"
            and collective.op_name is not None
            and scope in collective.op_name
            and _shape_signature(collective.operand_shapes) == (("bf16", (4, 1, 3072)),)
        )
        live_scoped = sum(
            (instruction.computation, instruction.name) in live_keys
            for instruction in scoped
        )
        reducer_counts[scope] = len(scoped)
        if len(scoped) != 4 or live_scoped != 4:
            violations.append(
                f"optimized {scope} half reducer drifted: "
                f"expected=4 total={len(scoped)} live={live_scoped}"
            )
        expected_kind = (
            "attention" if scope == _ATTENTION_REDUCER_SCOPE else "dense_down"
        )
        expected_frontier_count = (
            expected_attention_count // 4
            if expected_kind == "attention"
            else expected_down_count // 4
        )
        scope_reports: list[dict[str, int]] = []
        for reducer in scoped:
            (
                visited_instances,
                frontier_instances,
                ownership,
            ) = _optimized_reducer_lineage(
                instructions,
                reducer,
                full_width_rounded_then_slice=(full_width_rounded_then_slice),
            )
            frontier_counts = Counter(kind for kind, _, _ in frontier_instances)
            target_frontier = tuple(
                instruction
                for kind, _, instruction in frontier_instances
                if kind == expected_kind
            )
            opposite_count = sum(
                count
                for kind, count in frontier_counts.items()
                if kind != expected_kind
            )
            expected_geometry = (
                expected_attention_geometry
                if expected_kind == "attention"
                else down_geometry
            )
            malformed_frontier = tuple(
                instruction.name
                for instruction in target_frontier
                if (
                    _shape_signature(instruction.operand_shapes),
                    _shape_signature(instruction.result_shapes),
                )
                != expected_geometry
            )
            shape_opcode_counts = Counter(
                (
                    instruction.raw_opcode,
                    _shape_signature(instruction.result_shapes),
                )
                for _, instruction in visited_instances
            )
            stack_count = shape_opcode_counts[
                ("concatenate", (("bf16", (16, 1, 3072)),))
            ]
            half_reshape_count = shape_opcode_counts[
                ("reshape", (("bf16", (4, 4, 1, 3072)),))
            ]
            owner_select_count = shape_opcode_counts[
                ("select", (("bf16", (4, 1, 3072)),))
            ]
            y_add_1024_count = shape_opcode_counts[("add", (("bf16", (4, 1, 1024)),))]
            y_add_2048_count = shape_opcode_counts[("add", (("bf16", (4, 1, 2048)),))]
            if (
                len(target_frontier) != expected_frontier_count
                or opposite_count
                or malformed_frontier
            ):
                violations.append(
                    f"optimized {scope} reducer {reducer.name} producer "
                    "frontier drifted: "
                    f"expected={expected_kind}:{expected_frontier_count} "
                    f"observed={dict(frontier_counts)} "
                    f"malformed={malformed_frontier}"
                )
            if (
                stack_count != 2
                or half_reshape_count != 2
                or owner_select_count != 1
                or y_add_1024_count != 2
                or y_add_2048_count != 2
            ):
                violations.append(
                    f"optimized {scope} reducer {reducer.name} lost its exact "
                    "two-half stack/y-reduction/owner-select path: "
                    f"stacks={stack_count} reshapes={half_reshape_count} "
                    f"selects={owner_select_count} y1024={y_add_1024_count} "
                    f"y2048={y_add_2048_count}"
                )
            owner_half_contract_passed = not full_width_rounded_then_slice
            if full_width_rounded_then_slice:
                lower = ((0, 1, 1), (0, 3072, 1))
                upper = ((0, 1, 1), (3072, 6144, 1))
                true_report = {} if ownership is None else ownership.get("true", {})
                false_report = {} if ownership is None else ownership.get("false", {})
                predicate_report = (
                    {} if ownership is None else ownership.get("predicate", {})
                )
                true_frontiers = tuple(true_report.get("frontier_keys", ()))
                false_frontiers = tuple(false_report.get("frontier_keys", ()))
                true_slice_frontiers = tuple(true_report.get("slice_frontiers", ()))
                false_slice_frontiers = tuple(false_report.get("slice_frontiers", ()))
                true_mapped = tuple(
                    keys[0] for keys in true_slice_frontiers if len(keys) == 1
                )
                false_mapped = tuple(
                    keys[0] for keys in false_slice_frontiers if len(keys) == 1
                )
                expected_branch_frontier = {
                    (expected_kind, frame_id, instruction.name)
                    for kind, frame_id, instruction in frontier_instances
                    if kind == expected_kind
                }
                owner_half_contract_passed = bool(
                    ownership is not None
                    and ownership.get("owner_select_count") == 1
                    and predicate_report.get("causal_eq_zero_partition_id") is True
                    and predicate_report.get("compare_eq_count") == 1
                    and predicate_report.get("constant_zero_count") == 1
                    and predicate_report.get("frontier_count") == 0
                    and predicate_report.get("partition_id_count") == 1
                    and true_report.get("stack_count") == 1
                    and false_report.get("stack_count") == 1
                    and len(true_frontiers) == expected_frontier_count
                    and len(false_frontiers) == expected_frontier_count
                    and set(true_frontiers) == expected_branch_frontier
                    and set(false_frontiers) == expected_branch_frontier
                    and len(true_report.get("slice_ranges", ()))
                    == expected_frontier_count
                    and len(false_report.get("slice_ranges", ()))
                    == expected_frontier_count
                    and set(true_report.get("slice_ranges", ())) == {upper}
                    and set(false_report.get("slice_ranges", ())) == {lower}
                    and len(true_mapped) == expected_frontier_count
                    and len(false_mapped) == expected_frontier_count
                    and set(true_mapped) == expected_branch_frontier
                    and set(false_mapped) == expected_branch_frontier
                )
                if not owner_half_contract_passed:
                    violations.append(
                        f"optimized {scope} reducer {reducer.name} lost exact "
                        "owner-half lineage: lower must exclusively feed the "
                        "false/half-0 branch and upper the true/half-1 peer "
                        f"branch: ownership={ownership}"
                    )
            scope_reports.append(
                {
                    "half_reshape_count": half_reshape_count,
                    "owner_half_contract_passed": int(owner_half_contract_passed),
                    "owner_select_count": owner_select_count,
                    "producer_frontier_count": len(target_frontier),
                    "stack_count": stack_count,
                    "y_add_1024_count": y_add_1024_count,
                    "y_add_2048_count": y_add_2048_count,
                }
            )
        reducer_lineage[scope] = scope_reports
    # Runtime inputs must remain explicit at the executable boundary.  This
    # prevents a compile fixture from being mistaken for the authenticated run.
    expected_runtime_parameters = {
        ("s32", (8156,)): 2,
        ("s32", (1, 16)): 1,
        ("s32", (1,)): 2,
        ("bf16", (8192, 64)): 1,
    }
    observed_runtime_parameters = {}
    for (dtype, dimensions), expected in expected_runtime_parameters.items():
        observed = _entry_parameter_count(instructions, dtype, dimensions)
        observed_runtime_parameters[f"{dtype}{dimensions}"] = observed
        if observed != expected:
            violations.append(
                "runtime input boundary drifted for "
                f"{dtype}{dimensions}: expected={expected} observed={observed}"
            )
    _raise("feature2 main optimized HLO", violations)
    return {
        "collective_counts": dict(sorted(collective_counts.items())),
        "h16_b512_attention_calls": len(h16_calls),
        "host_markers": [],
        "instruction_count": len(instructions),
        "live_attention_producer_count": live_attention_count,
        "live_dense_down_convolution_count": live_down_count,
        "live_dense_gate_convolution_count": live_gate_count,
        "num_partitions": module.num_partitions,
        "passed": True,
        "physical_collective_count": len(collectives),
        "projection_width": expected_width,
        "reachable_computation_count": reachable_computation_count,
        "sealed_boundary_capture": sealed_boundary_capture,
        "reducer_counts": reducer_counts,
        "reducer_lineage": reducer_lineage,
        "root_shapes": [
            {"dtype": dtype, "shape": list(shape)} for dtype, shape in root_shapes
        ],
        "runtime_parameter_counts": observed_runtime_parameters,
        "sealed_canonical_hlo_identity": (
            sealed_canonical_identity if sealed_boundary_capture else {}
        ),
        **terminal_contract,
    }
