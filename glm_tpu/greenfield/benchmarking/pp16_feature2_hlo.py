"""Fail-closed HLO contracts for the bounded PP16 feature2 acquisition.

The source/JAXpr contract in :mod:`pp16_feature2_program` proves the exact
four-chunk semantic graph.  This module independently inspects the optimized
TPU executable so a compile-only acquisition cannot be admitted from source
structure alone.
"""

from __future__ import annotations

import re
from collections import Counter
from hashlib import sha256
from typing import Any, Literal

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import HloInstruction, HloShape, parse_hlo_module

Feature2MaterializerPhase = Literal["query_fp32", "wk_decode_bf16", "wk_promote_fp32"]

FEATURE2_OPTIMIZED_HLO_CANONICALIZER_VERSION = 1
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
_LOCAL_MAIN_ROOT = (
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
_H16_ATTENTION_KERNEL = "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"


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


def validate_feature2_main_stablehlo(stablehlo: str) -> dict[str, Any]:
    """Pin the source-level manual LP2 boundary before TPU optimization."""

    if not isinstance(stablehlo, str) or not stablehlo.strip():
        raise BenchmarkValidationError("feature2 main StableHLO is empty")
    compact = re.sub(r"\s+", "", stablehlo)
    markers = _host_markers(stablehlo)
    violations: list[str] = []
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
    _raise("feature2 main StableHLO", violations)
    return {
        "exact_h16_attention_present": True,
        "forbidden_markers": [],
        "host_markers": [],
        "num_partitions": 2,
        "passed": True,
    }


def validate_feature2_main_optimized_hlo(optimized_hlo: str) -> dict[str, Any]:
    """Parse and bound the optimized two-device feature2 executable."""

    module = parse_hlo_module(optimized_hlo)
    instructions = module.instructions
    root = _entry_root(instructions)
    root_shapes = () if root is None else _shape_signature(root.result_shapes)
    collectives = tuple(module.collectives)
    collective_counts = Counter(item.opcode for item in collectives)
    violations: list[str] = []
    if module.num_partitions != 2:
        violations.append(f"expected two partitions, found {module.num_partitions}")
    if root_shapes != _LOCAL_MAIN_ROOT:
        violations.append(
            "optimized terminal boundary drifted: "
            f"expected={_LOCAL_MAIN_ROOT} observed={root_shapes}"
        )
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
        for marker in ("bf16[32,6144]", "bf16[8156,6144]")
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
        "num_partitions": module.num_partitions,
        "passed": True,
        "physical_collective_count": len(collectives),
        "root_shapes": [
            {"dtype": dtype, "shape": list(shape)} for dtype, shape in root_shapes
        ],
        "runtime_parameter_counts": observed_runtime_parameters,
    }
