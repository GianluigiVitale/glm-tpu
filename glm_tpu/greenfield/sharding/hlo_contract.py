"""Parse textual XLA HLO and enforce physical collective contracts.

The parser targets ``XlaComputation.as_hlo_text()`` / JAX HLO text. It keeps
the raw instruction line alongside structured shapes, replica groups,
source-target pairs, channel id, and source metadata. Runtime XPlane evidence
remains mandatory; this static contract prevents known-bad executables from
reaching that stage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
import json
import math
import re
from typing import Any, Mapping, Sequence

from ..errors import HloContractViolationError


COLLECTIVE_OPCODES = frozenset(
    {
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-broadcast",
        "collective-permute",
        "reduce-scatter",
    }
)

_SHAPE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\[([0-9,]*)\]")
_OPCODE_RE = re.compile(r"\b([a-z][a-z0-9-]*)\(")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.%:-]+$")


@dataclass(frozen=True, slots=True)
class HloShape:
    dtype: str
    dimensions: tuple[int, ...]

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.dimensions:
            count *= dimension
        return count

    def to_dict(self) -> dict[str, Any]:
        return {"dimensions": list(self.dimensions), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class HloInstruction:
    index: int
    computation: str
    name: str
    raw_opcode: str
    opcode: str
    result_shapes: tuple[HloShape, ...]
    operand_names: tuple[str, ...]
    operand_shapes: tuple[HloShape, ...]
    replica_groups: tuple[tuple[int, ...], ...]
    source_target_pairs: tuple[tuple[int, int], ...]
    channel_id: int | None
    use_global_device_ids: bool
    op_name: str | None
    source_file: str | None
    source_line: int | None
    source_stack: str | None
    raw_line: str

    @property
    def is_collective(self) -> bool:
        return self.opcode in COLLECTIVE_OPCODES

    @property
    def maximum_group_size(self) -> int:
        if self.replica_groups:
            return max(len(group) for group in self.replica_groups)
        if self.source_target_pairs:
            return 2
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "computation": self.computation,
            "index": self.index,
            "maximum_group_size": self.maximum_group_size,
            "name": self.name,
            "op_name": self.op_name,
            "opcode": self.opcode,
            "operand_names": list(self.operand_names),
            "operand_shapes": [shape.to_dict() for shape in self.operand_shapes],
            "raw_line": self.raw_line,
            "raw_opcode": self.raw_opcode,
            "replica_groups": [list(group) for group in self.replica_groups],
            "result_shapes": [shape.to_dict() for shape in self.result_shapes],
            "source_file": self.source_file,
            "source_line": self.source_line,
            "source_stack": self.source_stack,
            "source_target_pairs": [list(pair) for pair in self.source_target_pairs],
            "use_global_device_ids": self.use_global_device_ids,
        }


@dataclass(frozen=True, slots=True)
class HloModule:
    name: str
    num_partitions: int | None
    num_replicas: int | None
    instructions: tuple[HloInstruction, ...]

    @property
    def collectives(self) -> tuple[HloInstruction, ...]:
        return tuple(
            instruction
            for instruction in self.instructions
            if instruction.is_collective
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": [
                instruction.to_dict() for instruction in self.instructions
            ],
            "name": self.name,
            "num_partitions": self.num_partitions,
            "num_replicas": self.num_replicas,
        }


def _parse_shapes(value: str) -> tuple[HloShape, ...]:
    result = []
    for match in _SHAPE_RE.finditer(value):
        dimensions = (
            ()
            if not match.group(2)
            else tuple(int(item) for item in match.group(2).split(","))
        )
        result.append(HloShape(match.group(1), dimensions))
    return tuple(result)


def _balanced_call_end(value: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(value)):
        if value[index] == "(":
            depth += 1
        elif value[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unbalanced HLO call syntax: {value!r}")


def _split_operands(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    result = []
    start = 0
    depth = 0
    for index, character in enumerate(value):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    names = []
    for operand in result:
        operand = re.sub(r"/\*.*?\*/", "", operand).strip()
        token = operand.split()[0] if operand else ""
        if _NAME_RE.match(token):
            names.append(token)
    return tuple(names)


def _braced_groups(attributes: str, name: str) -> tuple[tuple[int, ...], ...]:
    marker = f"{name}="
    start = attributes.find(marker)
    if start < 0:
        return ()
    value = attributes[start + len(marker) :].lstrip()
    mesh_match = re.match(r"mesh\[([^]]+)\]\s*\{([^}]*)\}", value)
    if mesh_match is not None:
        axis_items = re.findall(
            r"'([^']+)'\s*=\s*([0-9]+)", mesh_match.group(1)
        )
        selected_axes = re.findall(r"'([^']+)'", mesh_match.group(2))
        if not axis_items or not selected_axes:
            raise ValueError(f"malformed mesh {name} in HLO attributes")
        axis_names = tuple(item[0] for item in axis_items)
        axis_sizes = tuple(int(item[1]) for item in axis_items)
        if len(set(axis_names)) != len(axis_names) or any(
            size <= 0 for size in axis_sizes
        ):
            raise ValueError(f"invalid mesh {name} axes in HLO attributes")
        if len(set(selected_axes)) != len(selected_axes) or any(
            axis not in axis_names for axis in selected_axes
        ):
            raise ValueError(f"unknown mesh {name} group axis in HLO attributes")

        selected = frozenset(selected_axes)
        fixed_indices = tuple(
            index
            for index, axis in enumerate(axis_names)
            if axis not in selected
        )
        varied_indices = tuple(
            index
            for index, axis in enumerate(axis_names)
            if axis in selected
        )
        strides = tuple(
            math.prod(axis_sizes[index + 1 :])
            for index in range(len(axis_sizes))
        )
        groups = []
        for fixed_values in product(
            *(range(axis_sizes[index]) for index in fixed_indices)
        ):
            fixed = dict(zip(fixed_indices, fixed_values, strict=True))
            group = []
            for varied_values in product(
                *(range(axis_sizes[index]) for index in varied_indices)
            ):
                coordinates = [0] * len(axis_sizes)
                for index, coordinate in fixed.items():
                    coordinates[index] = coordinate
                for index, coordinate in zip(
                    varied_indices, varied_values, strict=True
                ):
                    coordinates[index] = coordinate
                group.append(
                    sum(
                        coordinate * stride
                        for coordinate, stride in zip(
                            coordinates, strides, strict=True
                        )
                    )
                )
            groups.append(tuple(group))
        return tuple(groups)
    start = attributes.find("{", start + len(marker))
    if start < 0:
        return ()
    depth = 0
    end = None
    for index in range(start, len(attributes)):
        if attributes[index] == "{":
            depth += 1
        elif attributes[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise ValueError(f"unbalanced {name} in HLO attributes")
    outer = attributes[start:end]
    groups = []
    for body in re.findall(r"\{([^{}]*)\}", outer):
        if not body.strip():
            groups.append(())
            continue
        groups.append(tuple(int(item.strip()) for item in body.split(",")))
    return tuple(groups)


def _quoted_attribute(value: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}="((?:[^"\\]|\\.)*)"', value)
    if not match:
        return None
    return bytes(match.group(1), "utf-8").decode("unicode_escape")


def _integer_attribute(value: str, name: str) -> int | None:
    match = re.search(rf"\b{re.escape(name)}=(-?[0-9]+)", value)
    return None if not match else int(match.group(1))


@dataclass(frozen=True, slots=True)
class _PendingInstruction:
    index: int
    computation: str
    name: str
    raw_opcode: str
    opcode: str
    result_shapes: tuple[HloShape, ...]
    operand_names: tuple[str, ...]
    attributes: str
    raw_line: str


def parse_hlo_module(text: str) -> HloModule:
    """Parse one textual HLO module without importing JAX or initializing TPU."""

    lines = text.splitlines()
    header = next((line.strip() for line in lines if line.strip()), "")
    if not header.startswith("HloModule "):
        raise ValueError("expected textual HLO beginning with 'HloModule '")
    name = header[len("HloModule ") :].split(",", 1)[0].strip()
    num_partitions = _integer_attribute(header, "num_partitions")
    num_replicas = _integer_attribute(header, "replica_count")
    if num_replicas is None:
        num_replicas = _integer_attribute(header, "num_replicas")

    computation = "<module>"
    pending = []
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("{") and " = " not in line:
            computation = line[:-1].strip()
            continue
        if line == "}" or " = " not in line:
            continue
        instruction_text = line
        if instruction_text.startswith("ROOT "):
            instruction_text = instruction_text[len("ROOT ") :]
        instruction_name, rhs = instruction_text.split(" = ", 1)
        opcode_match = _OPCODE_RE.search(rhs)
        if opcode_match is None:
            continue
        raw_opcode = opcode_match.group(1)
        opcode = raw_opcode
        if raw_opcode.endswith("-start"):
            candidate = raw_opcode[: -len("-start")]
            if candidate in COLLECTIVE_OPCODES:
                opcode = candidate
        call_open = opcode_match.end() - 1
        call_end = _balanced_call_end(rhs, call_open)
        pending.append(
            _PendingInstruction(
                index=len(pending),
                computation=computation,
                name=instruction_name,
                raw_opcode=raw_opcode,
                opcode=opcode,
                result_shapes=_parse_shapes(rhs[: opcode_match.start()]),
                operand_names=_split_operands(rhs[call_open + 1 : call_end]),
                attributes=rhs[call_end + 1 :],
                raw_line=line,
            )
        )

    shapes_by_name = {
        (instruction.computation, instruction.name): instruction.result_shapes
        for instruction in pending
    }
    instructions = []
    for instruction in pending:
        operand_shapes = tuple(
            shape
            for operand in instruction.operand_names
            for shape in shapes_by_name.get(
                (instruction.computation, operand), ()
            )
        )
        replica_groups = _braced_groups(
            instruction.attributes, "replica_groups"
        )
        source_target_pairs = _braced_groups(
            instruction.attributes, "source_target_pairs"
        )
        if any(len(pair) != 2 for pair in source_target_pairs):
            raise ValueError(
                f"collective-permute {instruction.name} has malformed source-target pairs"
            )
        instructions.append(
            HloInstruction(
                index=instruction.index,
                computation=instruction.computation,
                name=instruction.name,
                raw_opcode=instruction.raw_opcode,
                opcode=instruction.opcode,
                result_shapes=instruction.result_shapes,
                operand_names=instruction.operand_names,
                operand_shapes=operand_shapes,
                replica_groups=replica_groups,
                source_target_pairs=tuple(
                    (pair[0], pair[1]) for pair in source_target_pairs
                ),
                channel_id=_integer_attribute(
                    instruction.attributes, "channel_id"
                ),
                use_global_device_ids=(
                    "use_global_device_ids=true" in instruction.attributes
                ),
                op_name=_quoted_attribute(instruction.attributes, "op_name"),
                source_file=_quoted_attribute(
                    instruction.attributes, "source_file"
                ),
                source_line=_integer_attribute(
                    instruction.attributes, "source_line"
                ),
                source_stack=_quoted_attribute(
                    instruction.attributes, "source_stack"
                ),
                raw_line=instruction.raw_line,
            )
        )
    if not instructions:
        raise ValueError("HLO module contains no parseable instructions")
    return HloModule(
        name=name,
        num_partitions=num_partitions,
        num_replicas=num_replicas,
        instructions=tuple(instructions),
    )


@dataclass(frozen=True, slots=True)
class CollectiveExpectation:
    opcode: str
    count: int
    repeated_only: bool = False

    def __post_init__(self) -> None:
        if self.opcode not in COLLECTIVE_OPCODES:
            raise ValueError(f"unknown collective opcode {self.opcode!r}")
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 0:
            raise ValueError("collective expectation count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "opcode": self.opcode,
            "repeated_only": self.repeated_only,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CollectiveExpectation":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class HloContractPolicy:
    name: str
    total_devices: int
    repeated_region_patterns: tuple[str, ...]
    maximum_repeated_collective_group_size: int
    expected_repeated_replica_groups: tuple[tuple[int, ...], ...] = ()
    expected_collectives: tuple[CollectiveExpectation, ...] = ()
    expected_collective_permute_pairs: tuple[tuple[int, int], ...] = ()
    partition_id_to_device_id: tuple[int, ...] = ()
    forbidden_row_width_pairs: tuple[tuple[int, int], ...] = ((32, 6144),)
    residual_width: int = 6144
    require_global_device_ids: bool = True
    global_device_id_exempt_opcodes: tuple[str, ...] = ()
    require_repeated_region: bool = True
    allow_full_pod_repeated_collectives: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repeated_region_patterns", tuple(self.repeated_region_patterns)
        )
        object.__setattr__(
            self,
            "expected_repeated_replica_groups",
            tuple(tuple(group) for group in self.expected_repeated_replica_groups),
        )
        object.__setattr__(
            self, "expected_collectives", tuple(self.expected_collectives)
        )
        object.__setattr__(
            self,
            "expected_collective_permute_pairs",
            tuple(tuple(pair) for pair in self.expected_collective_permute_pairs),
        )
        object.__setattr__(
            self,
            "partition_id_to_device_id",
            tuple(self.partition_id_to_device_id),
        )
        object.__setattr__(
            self,
            "forbidden_row_width_pairs",
            tuple(tuple(pair) for pair in self.forbidden_row_width_pairs),
        )
        object.__setattr__(
            self,
            "global_device_id_exempt_opcodes",
            tuple(self.global_device_id_exempt_opcodes),
        )
        if not self.name.strip() or self.total_devices <= 0:
            raise ValueError("HLO policy requires a name and positive total_devices")
        if self.maximum_repeated_collective_group_size <= 0:
            raise ValueError("maximum repeated collective group size must be positive")
        if self.maximum_repeated_collective_group_size > self.total_devices:
            raise ValueError(
                "maximum repeated collective group size cannot exceed total_devices"
            )
        if self.require_repeated_region and not self.repeated_region_patterns:
            raise ValueError(
                "a required repeated region needs at least one match pattern"
            )
        for pattern in self.repeated_region_patterns:
            re.compile(pattern)
        if set(self.global_device_id_exempt_opcodes) - COLLECTIVE_OPCODES:
            raise ValueError(
                "global-device-id exemptions must name collective opcodes"
            )
        if self.partition_id_to_device_id and (
            len(self.partition_id_to_device_id) != self.total_devices
            or sorted(self.partition_id_to_device_id)
            != list(range(self.total_devices))
        ):
            raise ValueError(
                "partition_id_to_device_id must be a permutation of all device ids"
            )
        for group in self.expected_repeated_replica_groups:
            if (
                not group
                or len(set(group)) != len(group)
                or any(device < 0 or device >= self.total_devices for device in group)
            ):
                raise ValueError(
                    "expected replica groups must contain unique in-range devices"
                )
        if self.expected_repeated_replica_groups:
            members = [
                device
                for group in self.expected_repeated_replica_groups
                for device in group
            ]
            if sorted(members) != list(range(self.total_devices)):
                raise ValueError(
                    "expected replica groups must partition every device exactly once"
                )
        for pair in self.expected_collective_permute_pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(device < 0 or device >= self.total_devices for device in pair)
            ):
                raise ValueError(
                    "collective-permute pairs require distinct in-range source/target"
                )
        expectation_keys = [
            (expectation.opcode, expectation.repeated_only)
            for expectation in self.expected_collectives
        ]
        if len(expectation_keys) != len(set(expectation_keys)):
            raise ValueError("collective expectations must be unique by opcode/scope")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_repeated_replica_groups": [
                list(group) for group in self.expected_repeated_replica_groups
            ],
            "expected_collective_permute_pairs": [
                list(pair) for pair in self.expected_collective_permute_pairs
            ],
            "expected_collectives": [
                expectation.to_dict() for expectation in self.expected_collectives
            ],
            "forbidden_row_width_pairs": [
                list(pair) for pair in self.forbidden_row_width_pairs
            ],
            "global_device_id_exempt_opcodes": list(
                self.global_device_id_exempt_opcodes
            ),
            "maximum_repeated_collective_group_size": self.maximum_repeated_collective_group_size,
            "allow_full_pod_repeated_collectives": self.allow_full_pod_repeated_collectives,
            "name": self.name,
            "partition_id_to_device_id": list(self.partition_id_to_device_id),
            "repeated_region_patterns": list(self.repeated_region_patterns),
            "require_global_device_ids": self.require_global_device_ids,
            "require_repeated_region": self.require_repeated_region,
            "residual_width": self.residual_width,
            "total_devices": self.total_devices,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HloContractPolicy":
        data = dict(value)
        data["expected_collectives"] = tuple(
            CollectiveExpectation.from_dict(item)
            for item in data.get("expected_collectives", ())
        )
        return cls(**data)


@dataclass(frozen=True, slots=True)
class HloViolation:
    code: str
    message: str
    instruction_name: str | None = None
    computation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "computation": self.computation,
            "instruction_name": self.instruction_name,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class HloLintReport:
    module: HloModule
    policy: HloContractPolicy
    repeated_instruction_indices: tuple[int, ...]
    violations: tuple[HloViolation, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def raise_for_violations(self) -> None:
        if self.violations:
            detail = "\n".join(
                f"[{violation.code}] {violation.message}"
                for violation in self.violations
            )
            raise HloContractViolationError(
                f"HLO contract {self.policy.name!r} failed:\n{detail}"
            )

    def to_dict(self) -> dict[str, Any]:
        counts = Counter(
            instruction.opcode for instruction in self.module.collectives
        )
        repeated = set(self.repeated_instruction_indices)
        return {
            "collective_counts": dict(sorted(counts.items())),
            "collectives": [
                {
                    **instruction.to_dict(),
                    "inside_repeated_region": instruction.index in repeated,
                    "physical_replica_groups": [
                        list(group)
                        for group in _physical_replica_groups(
                            instruction, self.policy
                        )
                    ],
                    "physical_source_target_pairs": [
                        list(pair)
                        for pair in _physical_source_target_pairs(
                            instruction, self.policy
                        )
                    ],
                }
                for instruction in self.module.collectives
            ],
            "module_name": self.module.name,
            "policy": self.policy.to_dict(),
            "repeated_instruction_indices": list(self.repeated_instruction_indices),
            "valid": self.valid,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def _matches_repeated_region(
    instruction: HloInstruction, patterns: Sequence[str]
) -> bool:
    haystack = "\n".join(
        value
        for value in (
            instruction.computation,
            instruction.op_name,
            instruction.source_file,
            instruction.source_stack,
        )
        if value
    )
    return any(re.search(pattern, haystack) for pattern in patterns)


def _violation(
    code: str, message: str, instruction: HloInstruction | None = None
) -> HloViolation:
    return HloViolation(
        code=code,
        message=message,
        instruction_name=None if instruction is None else instruction.name,
        computation=None if instruction is None else instruction.computation,
    )


def _physical_device_id(partition_id: int, policy: HloContractPolicy) -> int:
    if not policy.partition_id_to_device_id:
        return partition_id
    if not 0 <= partition_id < len(policy.partition_id_to_device_id):
        raise ValueError(
            f"HLO partition id {partition_id} is outside the device assignment"
        )
    return policy.partition_id_to_device_id[partition_id]


def _physical_replica_groups(
    instruction: HloInstruction,
    policy: HloContractPolicy,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_physical_device_id(item, policy) for item in group)
        for group in instruction.replica_groups
    )


def _physical_source_target_pairs(
    instruction: HloInstruction,
    policy: HloContractPolicy,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            _physical_device_id(source, policy),
            _physical_device_id(target, policy),
        )
        for source, target in instruction.source_target_pairs
    )


def lint_hlo(module: HloModule, policy: HloContractPolicy) -> HloLintReport:
    repeated = tuple(
        instruction
        for instruction in module.instructions
        if _matches_repeated_region(instruction, policy.repeated_region_patterns)
    )
    repeated_indices = tuple(instruction.index for instruction in repeated)
    repeated_index_set = set(repeated_indices)
    forbidden_shapes = set(policy.forbidden_row_width_pairs)
    violations = []
    if policy.require_repeated_region and not repeated:
        violations.append(
            _violation(
                "MISSING_REPEATED_REGION",
                "no instruction matched repeated_region_patterns",
            )
        )

    expected_groups = {
        tuple(sorted(group)) for group in policy.expected_repeated_replica_groups
    }
    for instruction in repeated:
        for shape in instruction.result_shapes:
            if (
                len(shape.dimensions) >= 2
                and tuple(shape.dimensions[:2]) in forbidden_shapes
            ):
                violations.append(
                    _violation(
                        "DEAD_DECODE_ROWS",
                        f"{instruction.name} materializes forbidden decode shape "
                        f"{shape.dtype}{shape.dimensions}",
                        instruction,
                    )
                )
        if not instruction.is_collective:
            continue
        if instruction.opcode != "collective-permute" and not instruction.replica_groups:
            violations.append(
                _violation(
                    "MISSING_REPLICA_GROUPS",
                    f"{instruction.name} has no explicit replica_groups",
                    instruction,
                )
            )
        if instruction.maximum_group_size > policy.maximum_repeated_collective_group_size:
            violations.append(
                _violation(
                    "REPEATED_GROUP_TOO_LARGE",
                    f"{instruction.name} physical group size "
                    f"{instruction.maximum_group_size} exceeds "
                    f"{policy.maximum_repeated_collective_group_size}",
                    instruction,
                )
            )
        if (
            instruction.maximum_group_size >= policy.total_devices
            and not policy.allow_full_pod_repeated_collectives
        ):
            violations.append(
                _violation(
                    "FULL_POD_REPEATED_COLLECTIVE",
                    f"{instruction.name} synchronizes all {policy.total_devices} devices",
                    instruction,
                )
            )
            shapes = instruction.result_shapes + instruction.operand_shapes
            if any(
                shape.dimensions
                and shape.dimensions[-1] == policy.residual_width
                and shape.element_count >= policy.residual_width
                for shape in shapes
            ):
                violations.append(
                    _violation(
                        "FULL_POD_RESIDUAL_RECONSTRUCTION",
                        f"{instruction.name} reconstructs residual-width data over the full pod",
                        instruction,
                    )
                )
        if expected_groups and instruction.replica_groups:
            actual_groups = {
                tuple(sorted(group))
                for group in _physical_replica_groups(instruction, policy)
            }
            if actual_groups != expected_groups:
                violations.append(
                    _violation(
                        "REPLICA_GROUPS_MISMATCH",
                        f"{instruction.name} expected physical groups "
                        f"{sorted(expected_groups)}, got {sorted(actual_groups)}",
                        instruction,
                    )
                )
        if (
            policy.require_global_device_ids
            and instruction.opcode != "collective-permute"
            and instruction.opcode not in policy.global_device_id_exempt_opcodes
            and not instruction.use_global_device_ids
        ):
            violations.append(
                _violation(
                    "NON_GLOBAL_DEVICE_IDS",
                    f"{instruction.name} does not declare use_global_device_ids=true",
                    instruction,
                )
            )

    if policy.expected_collective_permute_pairs:
        actual_pairs = tuple(
            pair
            for instruction in module.collectives
            if instruction.opcode == "collective-permute"
            for pair in _physical_source_target_pairs(instruction, policy)
        )
        if sorted(actual_pairs) != sorted(policy.expected_collective_permute_pairs):
            violations.append(
                _violation(
                    "COLLECTIVE_PERMUTE_PAIRS_MISMATCH",
                    f"expected pairs {policy.expected_collective_permute_pairs}, "
                    f"got {actual_pairs}",
                )
            )

    for expectation in policy.expected_collectives:
        actual = sum(
            instruction.opcode == expectation.opcode
            and (
                not expectation.repeated_only
                or instruction.index in repeated_index_set
            )
            for instruction in module.collectives
        )
        if actual != expectation.count:
            scope = "repeated " if expectation.repeated_only else ""
            violations.append(
                _violation(
                    "COLLECTIVE_COUNT_MISMATCH",
                    f"expected {expectation.count} {scope}{expectation.opcode} "
                    f"instruction(s), got {actual}",
                )
            )

    return HloLintReport(
        module=module,
        policy=policy,
        repeated_instruction_indices=repeated_indices,
        violations=tuple(violations),
    )
