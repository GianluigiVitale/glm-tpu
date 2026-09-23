"""Build and validate the bounded real WS32 layer-3 MoE executable."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Mapping

import jax
from jax.sharding import PartitionSpec as P

from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.ws32 import ws32_moe_fp8_from_routes_mapped
from ..sharding.hlo_contract import HloInstruction, parse_hlo_module
from ..sharding.ws32 import validate_ws32_repeated_hlo
# Moved verbatim to glm_tpu.optimized.topology_binding (S2a); re-exported for research importers.
from glm_tpu.optimized.topology_binding import (  # noqa: F401
    _TOPOLOGY_CAPTURE_KEYS,
    validate_ws32_topology_fleet,
)


WS32_ONE_LAYER_INPUT_SPECS = (
    P(None, "feature"),
    P(),
    P(),
    P("expert", None, "feature"),
    P("expert", None, "feature"),
    P("expert", None, "feature"),
    P("expert", None, "feature"),
    P("expert", "feature", None),
    P("expert", "feature", None),
    P(None, "feature"),
    P(None, "feature"),
    P(None, "feature"),
    P(None, "feature"),
    P("feature", None),
    P("feature", None),
)
WS32_ONE_LAYER_OUTPUT_SPEC = P(None, "feature")


@dataclass(frozen=True, slots=True)
class Ws32OneLayerHloReport:
    """Pre-execution proof and exact compiler identities for one layer."""

    stablehlo_sha256: str
    optimized_hlo_sha256: str
    stablehlo_collective_count: int
    optimized_hlo_collective_count: int
    feature_reduce_count: int
    expert_reduce_count: int
    f32_operand_reduce_count: int
    bf16_result_reduce_count: int
    f32_result_reduce_count: int
    maximum_group_size: int
    entry_parameter_count: int
    live_entry_parameter_count: int
    live_collective_count: int
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_parameter_count": self.entry_parameter_count,
            "expert_reduce_count": self.expert_reduce_count,
            "f32_operand_reduce_count": self.f32_operand_reduce_count,
            "feature_reduce_count": self.feature_reduce_count,
            "bf16_result_reduce_count": self.bf16_result_reduce_count,
            "f32_result_reduce_count": self.f32_result_reduce_count,
            "live_collective_count": self.live_collective_count,
            "live_entry_parameter_count": self.live_entry_parameter_count,
            "maximum_group_size": self.maximum_group_size,
            "optimized_hlo_collective_count": self.optimized_hlo_collective_count,
            "optimized_hlo_sha256": self.optimized_hlo_sha256,
            "passed": self.passed,
            "stablehlo_collective_count": self.stablehlo_collective_count,
            "stablehlo_sha256": self.stablehlo_sha256,
            "violations": list(self.violations),
        }


def build_ws32_one_layer_mapped(
    mesh: Any,
    *,
    contract: GlmMoeNumericalContract,
) -> Any:
    """Return the exact one-row, routes-supplied WS32 shard-map body."""

    if contract.stage_size != 8 or contract.local_experts != 32:
        raise ValueError("real WS32 layer requires expert8 ownership")

    def body(*values: Any) -> Any:
        with jax.named_scope("greenfield_ws32_real_layer3"):
            return ws32_moe_fp8_from_routes_mapped(
                *values,
                contract=contract,
                expert_axis="expert",
                feature_axis="feature",
            )

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=WS32_ONE_LAYER_INPUT_SPECS,
        out_specs=WS32_ONE_LAYER_OUTPUT_SPEC,
        check_vma=False,
    )


def ws32_one_layer_inputs(
    hidden: Any,
    route_indices: Any,
    route_weights: Any,
    arrays: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Bind checkpoint leaves to the immutable HLO argument order."""

    required = {
        "expert_gate_bits",
        "expert_gate_scale",
        "expert_up_bits",
        "expert_up_scale",
        "expert_down_bits",
        "expert_down_scale",
        "shared_gate_bits",
        "shared_gate_scale",
        "shared_up_bits",
        "shared_up_scale",
        "shared_down_bits",
        "shared_down_scale",
    }
    if not required <= set(arrays):
        raise ValueError("WS32 one-layer checkpoint inputs are incomplete")
    return (
        hidden,
        route_indices,
        route_weights,
        arrays["expert_gate_bits"],
        arrays["expert_gate_scale"],
        arrays["expert_up_bits"],
        arrays["expert_up_scale"],
        arrays["expert_down_bits"],
        arrays["expert_down_scale"],
        arrays["shared_gate_bits"],
        arrays["shared_gate_scale"],
        arrays["shared_up_bits"],
        arrays["shared_up_scale"],
        arrays["shared_down_bits"],
        arrays["shared_down_scale"],
    )


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _entry_instructions(
    instructions: tuple[HloInstruction, ...],
) -> tuple[HloInstruction, ...]:
    entries = tuple(
        instruction
        for instruction in instructions
        if instruction.computation.startswith("ENTRY ")
    )
    if entries:
        return entries
    return tuple(
        instruction
        for instruction in instructions
        if instruction.computation == "main"
    )


def _computation_base(value: str) -> str:
    return value.split(" ", 1)[0]


def _flatten_dependencies(value: Any) -> frozenset[str]:
    if isinstance(value, frozenset):
        return value
    if isinstance(value, tuple):
        return frozenset().union(
            *(_flatten_dependencies(item) for item in value)
        )
    raise ValueError("optimized HLO dependency value is invalid")


def _merge_dependencies(values: tuple[Any, ...]) -> Any:
    if values and all(isinstance(value, tuple) for value in values):
        widths = {len(value) for value in values}
        if len(widths) == 1:
            return tuple(
                _merge_dependencies(tuple(value[index] for value in values))
                for index in range(next(iter(widths)))
            )
    return frozenset().union(*(_flatten_dependencies(value) for value in values))


def _add_dependencies(value: Any, extra: frozenset[str]) -> Any:
    if isinstance(value, tuple):
        return tuple(_add_dependencies(item, extra) for item in value)
    return _flatten_dependencies(value) | extra


def _semantic_root_dependencies(
    instructions: tuple[HloInstruction, ...],
    entry_root: HloInstruction,
) -> frozenset[str]:
    """Resolve exact SSA dependencies through calls, tuples, and branches."""

    computations: dict[str, dict[str, HloInstruction]] = defaultdict(dict)
    roots: dict[str, HloInstruction] = {}
    for instruction in instructions:
        computation = _computation_base(instruction.computation)
        computations[computation][instruction.name] = instruction
        if instruction.raw_line.lstrip().startswith("ROOT "):
            if computation in roots:
                raise ValueError(f"multiple optimized HLO roots in {computation}")
            roots[computation] = instruction

    cache: dict[tuple[str, str, tuple[tuple[int, Any], ...]], Any] = {}
    active: set[tuple[str, str, tuple[tuple[int, Any], ...]]] = set()

    def evaluate(
        computation: str,
        name: str,
        environment: tuple[tuple[int, Any], ...],
    ) -> Any:
        key = (computation, name, environment)
        if key in cache:
            return cache[key]
        if key in active:
            raise ValueError("cyclic optimized HLO dependency graph")
        active.add(key)
        try:
            instruction = computations.get(computation, {}).get(name)
            if instruction is None:
                raise ValueError(f"missing optimized HLO value {computation}:{name}")
            environment_map = dict(environment)
            opcode = instruction.raw_opcode
            if opcode == "parameter":
                match = re.search(r"\bparameter\((\d+)\)", instruction.raw_line)
                if match is None:
                    raise ValueError("optimized HLO parameter number is missing")
                number = int(match.group(1))
                if computation == _computation_base(entry_root.computation):
                    value: Any = frozenset((f"parameter:{instruction.name}",))
                elif number in environment_map:
                    value = environment_map[number]
                else:
                    raise ValueError(
                        f"optimized HLO callee parameter {number} is unbound"
                    )
            elif opcode in {"constant", "iota", "partition-id", "replica-id"}:
                value = frozenset()
            elif opcode == "tuple":
                value = tuple(
                    evaluate(computation, operand, environment)
                    for operand in instruction.operand_names
                )
            elif opcode == "get-tuple-element":
                if len(instruction.operand_names) != 1:
                    raise ValueError("optimized HLO GTE arity drifted")
                match = re.search(r"\bindex=(\d+)", instruction.raw_line)
                source = evaluate(
                    computation, instruction.operand_names[0], environment
                )
                if match is None or not isinstance(source, tuple):
                    raise ValueError("optimized HLO GTE source/index drifted")
                index = int(match.group(1))
                if not 0 <= index < len(source):
                    raise ValueError("optimized HLO GTE index is out of range")
                value = source[index]
            elif opcode in {"fusion", "call"}:
                match = re.search(r"\bcalls=([^,\s]+)", instruction.raw_line)
                if match is None:
                    raise ValueError("optimized HLO caller target is missing")
                callee = match.group(1)
                if callee not in roots:
                    raise ValueError(f"optimized HLO callee {callee} has no root")
                operands = tuple(
                    evaluate(computation, operand, environment)
                    for operand in instruction.operand_names
                )
                value = evaluate(
                    callee,
                    roots[callee].name,
                    tuple(enumerate(operands)),
                )
            elif opcode == "conditional":
                match = re.search(
                    r"\bbranch_computations=\{([^}]*)\}", instruction.raw_line
                )
                if match is None:
                    raise ValueError("optimized HLO conditional branches are missing")
                branches = tuple(
                    item.strip() for item in match.group(1).split(",")
                )
                if len(instruction.operand_names) != len(branches) + 1 or any(
                    branch not in roots for branch in branches
                ):
                    raise ValueError("optimized HLO conditional arity drifted")
                predicate = _flatten_dependencies(
                    evaluate(
                        computation, instruction.operand_names[0], environment
                    )
                )
                branch_values = tuple(
                    evaluate(
                        branch,
                        roots[branch].name,
                        (
                            (
                                0,
                                evaluate(
                                    computation,
                                    instruction.operand_names[index + 1],
                                    environment,
                                ),
                            ),
                        ),
                    )
                    for index, branch in enumerate(branches)
                )
                value = _add_dependencies(
                    _merge_dependencies(branch_values), predicate
                )
            else:
                operands = tuple(
                    evaluate(computation, operand, environment)
                    for operand in instruction.operand_names
                )
                value = _merge_dependencies(operands)
                if opcode in {
                    "all-reduce",
                    "all-gather",
                    "all-to-all",
                    "collective-permute",
                    "reduce-scatter",
                }:
                    value = _add_dependencies(
                        value,
                        frozenset(
                            (
                                "collective:"
                                f"{computation}:{instruction.name}",
                            )
                        ),
                    )
            cache[key] = value
            return value
        finally:
            active.discard(key)

    entry_computation = _computation_base(entry_root.computation)
    return _flatten_dependencies(
        evaluate(entry_computation, entry_root.name, ())
    )


def validate_ws32_one_layer_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
    expected_optimized_collective_result_dtype: str | None,
) -> Ws32OneLayerHloReport:
    """Fail closed before arithmetic on graph, groups, liveness, and pins.

    The first protected invocation intentionally uses vacant pins and stops
    after atomically persisting both graphs.  Those immutable graphs are then
    reviewed offline and their exact hashes become required inputs to the
    numerical invocation.
    """

    stable_digest = _sha256_text(stablehlo)
    optimized_digest = _sha256_text(optimized_hlo)
    violations: list[str] = []
    for name, expected, observed in (
        ("StableHLO", expected_stablehlo_sha256, stable_digest),
        ("optimized HLO", expected_optimized_hlo_sha256, optimized_digest),
    ):
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            violations.append(f"{name} pin is not a lowercase SHA-256")
        elif expected == "0" * 64:
            violations.append(f"{name} pin is intentionally vacant")
        elif observed != expected:
            violations.append(f"{name} digest drifted")

    stable_collectives = stablehlo.count("stablehlo.all_reduce")
    if stable_collectives != 10:
        violations.append(
            f"StableHLO expected 10 all-reduces, found {stable_collectives}"
        )
    if stablehlo.count("sdy.manual_computation") != 1:
        violations.append("StableHLO requires one manual computation")
    if "mhlo.num_partitions = 32 : i32" not in stablehlo:
        violations.append("StableHLO omits exact 32-partition identity")
    forbidden_stable = (
        "stablehlo.all_gather",
        "stablehlo.all_to_all",
        "stablehlo.collective_broadcast",
        "stablehlo.collective_permute",
        "stablehlo.reduce_scatter",
        "stablehlo.all_reduce_start",
        "stablehlo.all_reduce_done",
    )
    present_forbidden = tuple(value for value in forbidden_stable if value in stablehlo)
    if present_forbidden:
        violations.append(f"StableHLO has forbidden collectives {present_forbidden}")

    repeated = validate_ws32_repeated_hlo(
        optimized_hlo,
        kind="moe",
        hidden_size=6144,
        moe_intermediate_size=2048,
        top_k=8,
        expected_result_dtype=expected_optimized_collective_result_dtype,
    )
    violations.extend(repeated.violations)
    module = parse_hlo_module(optimized_hlo)
    if module.num_partitions != 32:
        violations.append(
            f"optimized HLO expected 32 partitions, found {module.num_partitions}"
        )
    entry = _entry_instructions(module.instructions)
    roots = tuple(
        item for item in entry if item.raw_line.lstrip().startswith("ROOT ")
    )
    parameters = tuple(item for item in entry if item.raw_opcode == "parameter")
    if len(roots) != 1:
        violations.append(f"optimized HLO expected one ENTRY root, found {len(roots)}")
        live_parameters = 0
        live_collectives = 0
    else:
        root = roots[0]
        try:
            dependencies = _semantic_root_dependencies(
                module.instructions, root
            )
        except ValueError as exc:
            dependencies = frozenset()
            violations.append(f"optimized HLO semantic dependency failure: {exc}")
        live_parameters = sum(
            f"parameter:{item.name}" in dependencies for item in parameters
        )
        live_collectives = sum(
            (
                "collective:"
                f"{_computation_base(item.computation)}:{item.name}"
            )
            in dependencies
            for item in module.collectives
        )
        if live_parameters != len(parameters):
            violations.append(
                "optimized HLO contains dead ENTRY inputs: "
                f"live={live_parameters} total={len(parameters)}"
            )
        if live_collectives != len(module.collectives):
            violations.append(
                "optimized HLO contains a collective outside the live root path"
            )
    if len(parameters) != 15:
        violations.append(
            f"optimized HLO expected 15 live inputs, found {len(parameters)}"
        )
    parameter_shapes = Counter(
        (shape.dtype, shape.dimensions)
        for item in parameters
        for shape in item.result_shapes
    )
    expected_parameter_shapes = Counter(
        {
            ("bf16", (1, 1536)): 1,
            ("s32", (1, 8)): 1,
            ("f32", (1, 8)): 1,
            ("u8", (32, 2048, 1536)): 2,
            ("f32", (32, 16, 12)): 2,
            ("u8", (32, 1536, 2048)): 1,
            ("f32", (32, 12, 16)): 1,
            ("u8", (2048, 1536)): 2,
            ("f32", (16, 12)): 2,
            ("u8", (1536, 2048)): 1,
            ("f32", (12, 16)): 1,
        }
    )
    if parameter_shapes != expected_parameter_shapes:
        violations.append(
            "optimized HLO local input geometry drifted: "
            f"{dict(parameter_shapes)}"
        )
    forbidden_hidden = []
    for item in module.instructions:
        for shape in item.result_shapes:
            if shape.dimensions in ((32, 6144), (32, 1, 6144)):
                forbidden_hidden.append(item.name)
    if forbidden_hidden:
        violations.append(
            "optimized HLO reconstructs a forbidden full-pod hidden tensor: "
            f"{forbidden_hidden}"
        )
    return Ws32OneLayerHloReport(
        stablehlo_sha256=stable_digest,
        optimized_hlo_sha256=optimized_digest,
        stablehlo_collective_count=stable_collectives,
        optimized_hlo_collective_count=repeated.all_reduce_count,
        feature_reduce_count=repeated.feature_reduce_count,
        expert_reduce_count=repeated.expert_reduce_count,
        f32_operand_reduce_count=repeated.f32_operand_reduce_count,
        bf16_result_reduce_count=repeated.bf16_result_reduce_count,
        f32_result_reduce_count=repeated.f32_result_reduce_count,
        maximum_group_size=repeated.maximum_group_size,
        entry_parameter_count=len(parameters),
        live_entry_parameter_count=live_parameters,
        live_collective_count=live_collectives,
        violations=tuple(dict.fromkeys(violations)),
    )
