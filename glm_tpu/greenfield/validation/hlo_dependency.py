"""Exact HLO value-dependency helpers shared by bounded Gate-D discriminators.

Verbatim port of the helpers first written for the layer-0 projection/reduction
probe so that drivers executed from the sealed ``glm_tpu`` source archive can
prove result liveness without importing mutable ``scripts/`` sources.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


def _shape_signatures(shapes: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        f"{shape.dtype}[{','.join(map(str, shape.dimensions))}]"
        for shape in shapes
    )


def _instruction_key(instruction: Any) -> tuple[str, str]:
    return instruction.computation, instruction.name


def _computation_id(header: str) -> str:
    value = header.removeprefix("ENTRY ").split(None, 1)[0]
    return value.removeprefix("%").split("(", 1)[0]


def _tuple_index(instruction: Any) -> int | None:
    match = re.search(r"\bindex=([0-9]+)", instruction.raw_line)
    return None if match is None else int(match.group(1))


def _fusion_operand_indices(
    module: Any,
    fusion: Any,
    *,
    result_index: int | None = None,
    fusion_stack: frozenset[tuple[str, str]] = frozenset(),
    by_key: Mapping[tuple[str, str], Any] | None = None,
    instructions_by_computation: Mapping[str, tuple[Any, ...]] | None = None,
    fusion_memo: dict[
        tuple[tuple[str, str], int | None],
        tuple[frozenset[int], tuple[str, ...]],
    ]
    | None = None,
) -> tuple[set[int], list[str]]:
    """Map a fusion result to only caller operands used by its called root."""

    fusion_key = _instruction_key(fusion)
    memo_key = (fusion_key, result_index)
    if fusion_memo is not None and memo_key in fusion_memo:
        indices, errors = fusion_memo[memo_key]
        return set(indices), list(errors)
    if fusion_key in fusion_stack:
        return set(), [f"recursive fusion lineage: {fusion.name}"]
    match = re.search(r"\bcalls=%?([^,\s}\]]+)", fusion.raw_line)
    if match is None:
        return set(), [f"fusion lacks called computation: {fusion.name}"]
    callee = match.group(1)
    if by_key is None:
        by_key = {_instruction_key(item): item for item in module.instructions}
    if instructions_by_computation is None:
        grouped: dict[str, list[Any]] = {}
        for item in module.instructions:
            grouped.setdefault(_computation_id(item.computation), []).append(item)
        instructions_by_computation = {
            name: tuple(items) for name, items in grouped.items()
        }
    instructions = instructions_by_computation.get(callee, ())
    roots = tuple(
        item
        for item in instructions
        if item.raw_line.lstrip().startswith("ROOT ")
    )
    if len(roots) != 1:
        return set(), [f"fusion callee root drifted: {fusion.name}"]
    errors: list[str] = []
    visiting: set[tuple[tuple[str, str], int | None]] = set()
    dependency_memo: dict[
        tuple[tuple[str, str], int | None], frozenset[int]
    ] = {}

    def dependencies(instruction: Any, selected: int | None = None) -> set[int]:
        key = (_instruction_key(instruction), selected)
        if key in dependency_memo:
            return set(dependency_memo[key])
        if key in visiting:
            errors.append(f"cyclic fusion value: {instruction.name}")
            return set()
        visiting.add(key)
        try:
            result: set[int]
            if instruction.opcode == "parameter":
                parameter_match = re.search(
                    r"\bparameter\(([0-9]+)\)", instruction.raw_line
                )
                if parameter_match is None or selected is not None:
                    errors.append(
                        f"malformed fusion parameter: {instruction.name}"
                    )
                    return set()
                result = {int(parameter_match.group(1))}
            elif instruction.opcode == "constant":
                result = set()
            elif instruction.opcode == "get-tuple-element":
                index = _tuple_index(instruction)
                if index is None or len(instruction.operand_names) != 1:
                    errors.append(
                        f"malformed fusion tuple selection: {instruction.name}"
                    )
                    return set()
                producer = by_key.get(
                    (instruction.computation, instruction.operand_names[0])
                )
                if producer is None:
                    errors.append(
                        f"undefined fusion tuple producer: {instruction.name}"
                    )
                    return set()
                result = dependencies(producer, index)
            elif instruction.opcode == "tuple":
                if selected is None:
                    indices = range(len(instruction.operand_names))
                elif selected < len(instruction.operand_names):
                    indices = (selected,)
                else:
                    errors.append(f"fusion tuple index drifted: {instruction.name}")
                    return set()
                result = set()
                for index in indices:
                    operand = by_key.get(
                        (instruction.computation, instruction.operand_names[index])
                    )
                    if operand is None:
                        errors.append(
                            f"undefined fusion tuple operand: {instruction.name}"
                        )
                    else:
                        result.update(dependencies(operand))
            elif instruction.opcode == "fusion":
                nested_indices, nested_errors = _fusion_operand_indices(
                    module,
                    instruction,
                    result_index=selected,
                    fusion_stack=fusion_stack | {fusion_key},
                    by_key=by_key,
                    instructions_by_computation=instructions_by_computation,
                    fusion_memo=fusion_memo,
                )
                errors.extend(nested_errors)
                result = set()
                for index in nested_indices:
                    if index >= len(instruction.operand_names):
                        errors.append(
                            f"nested fusion operand index drifted: {instruction.name}"
                        )
                        continue
                    operand = by_key.get(
                        (instruction.computation, instruction.operand_names[index])
                    )
                    if operand is None:
                        errors.append(
                            f"undefined nested fusion operand: {instruction.name}"
                        )
                    else:
                        result.update(dependencies(operand))
            elif selected is not None:
                errors.append(
                    f"tuple index applied to non-tuple fusion value: "
                    f"{instruction.name}"
                )
                return set()
            else:
                result = set()
                for operand_name in instruction.operand_names:
                    operand = by_key.get((instruction.computation, operand_name))
                    if operand is None:
                        errors.append(
                            f"undefined fusion operand {operand_name}: "
                            f"{instruction.name}"
                        )
                    else:
                        result.update(dependencies(operand))
            dependency_memo[key] = frozenset(result)
            return result
        finally:
            visiting.remove(key)

    indices = dependencies(roots[0], result_index)
    invalid = sorted(
        index for index in indices if index >= len(fusion.operand_names)
    )
    if invalid:
        errors.append(f"fusion caller operand indices drifted: {invalid}")
        indices.difference_update(invalid)
    if fusion_memo is not None:
        fusion_memo[memo_key] = (frozenset(indices), tuple(errors))
    return indices, errors


def _value_depends_on(module: Any, value: Any, source: Any) -> bool:
    """Test exact HLO value dependency while respecting tuple element selection."""

    by_key = {_instruction_key(item): item for item in module.instructions}
    grouped: dict[str, list[Any]] = {}
    for item in module.instructions:
        grouped.setdefault(_computation_id(item.computation), []).append(item)
    instructions_by_computation = {
        name: tuple(items) for name, items in grouped.items()
    }
    fusion_memo: dict[
        tuple[tuple[str, str], int | None],
        tuple[frozenset[int], tuple[str, ...]],
    ] = {}
    source_key = _instruction_key(source)
    visiting: set[tuple[str, str]] = set()
    dependency_memo: dict[tuple[str, str], bool] = {}

    def depends(key: tuple[str, str]) -> bool:
        if key == source_key:
            return True
        if key in dependency_memo:
            return dependency_memo[key]
        if key in visiting:
            return False
        instruction = by_key.get(key)
        if instruction is None:
            return False
        visiting.add(key)
        try:
            operands = instruction.operand_names
            if instruction.opcode == "get-tuple-element":
                index = _tuple_index(instruction)
                if index is None or len(operands) != 1:
                    result = False
                    producer = None
                else:
                    producer = by_key.get((instruction.computation, operands[0]))
                if producer is None:
                    result = False
                elif producer.opcode == "tuple":
                    result = index < len(producer.operand_names) and depends(
                        (instruction.computation, producer.operand_names[index])
                    )
                elif producer.opcode == "fusion":
                    indices, errors = _fusion_operand_indices(
                        module,
                        producer,
                        result_index=index,
                        by_key=by_key,
                        instructions_by_computation=instructions_by_computation,
                        fusion_memo=fusion_memo,
                    )
                    result = not errors and any(
                        depends(
                            (
                                instruction.computation,
                                producer.operand_names[operand_index],
                            )
                        )
                        for operand_index in indices
                    )
                else:
                    result = False
            elif instruction.opcode == "fusion":
                indices, errors = _fusion_operand_indices(
                    module,
                    instruction,
                    by_key=by_key,
                    instructions_by_computation=instructions_by_computation,
                    fusion_memo=fusion_memo,
                )
                result = not errors and any(
                    depends((instruction.computation, operands[index]))
                    for index in indices
                )
            else:
                result = any(
                    depends((instruction.computation, operand))
                    for operand in operands
                )
        finally:
            visiting.remove(key)
        dependency_memo[key] = result
        return result

    return depends(_instruction_key(value))


shape_signatures = _shape_signatures
instruction_key = _instruction_key
value_depends_on = _value_depends_on
