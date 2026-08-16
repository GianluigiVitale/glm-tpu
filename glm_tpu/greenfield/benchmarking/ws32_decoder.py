"""Fail-closed structural HLO contracts for the complete WS32 decoder."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from dataclasses import dataclass
from hashlib import sha256
import re
from typing import Any, Mapping

from .ws32_pallas_one_layer import _live_instruction_closure
from ..sharding.hlo_contract import HloInstruction, parse_hlo_module
from ..sharding.ws32 import _exact_scalar_add_reducer, _logical_groups


_HEX = re.compile(r"[0-9a-f]{64}")
_ASYNC_COLLECTIVE = re.compile(
    r"^(?:all-gather|all-reduce|all-to-all|collective-broadcast|"
    r"collective-permute|reduce-scatter)-(?:start|done)$"
)
_HOST_MARKERS = (
    "host_callback",
    "outside_compilation",
    "xla_ffi_python_cpu_callback",
    "xla_python_cpu_callback",
)


@dataclass(frozen=True, slots=True)
class Ws32DecoderHloReport:
    kind: str
    stablehlo_sha256: str
    optimized_hlo_sha256: str
    instruction_count: int
    live_instruction_count: int
    collective_count: int
    live_collective_count: int
    all_reduce_count: int
    all_gather_count: int
    feature_collective_count: int
    expert_collective_count: int
    fused_rmsnorm_collective_count: int
    rounded_first_rmsnorm_collective_count: int
    maximum_group_size: int
    async_collective_count: int
    forbidden_full_hidden_values: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_gather_count": self.all_gather_count,
            "all_reduce_count": self.all_reduce_count,
            "async_collective_count": self.async_collective_count,
            "collective_count": self.collective_count,
            "expert_collective_count": self.expert_collective_count,
            "feature_collective_count": self.feature_collective_count,
            "fused_rmsnorm_collective_count": (
                self.fused_rmsnorm_collective_count
            ),
            "forbidden_full_hidden_values": list(
                self.forbidden_full_hidden_values
            ),
            "instruction_count": self.instruction_count,
            "kind": self.kind,
            "live_collective_count": self.live_collective_count,
            "live_instruction_count": self.live_instruction_count,
            "maximum_group_size": self.maximum_group_size,
            "optimized_hlo_sha256": self.optimized_hlo_sha256,
            "passed": self.passed,
            "rounded_first_rmsnorm_collective_count": (
                self.rounded_first_rmsnorm_collective_count
            ),
            "stablehlo_sha256": self.stablehlo_sha256,
            "violations": list(self.violations),
        }


def _digest(value: str, *, field: str) -> str:
    if _HEX.fullmatch(value) is None:
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _group_family(instruction: HloInstruction) -> str | None:
    observed = frozenset(tuple(group) for group in instruction.replica_groups)
    feature, expert = _logical_groups()
    if observed == feature:
        return "feature"
    if observed == expert:
        return "expert"
    return None


def _exact_add_reducer(
    instruction: HloInstruction,
    *,
    module_instructions: tuple[HloInstruction, ...],
    instructions_by_computation: Mapping[
        str, tuple[HloInstruction, ...]
    ]
    | None = None,
) -> bool:
    """Accept an exact scalar add reducer, including tuple-combined psums."""

    arity = len(instruction.operand_shapes)
    if arity <= 0 or len(instruction.result_shapes) != arity:
        return False
    accumulator_dtype = instruction.operand_shapes[0].dtype
    if any(
        operand.dtype != accumulator_dtype
        or operand.dimensions != result.dimensions
        or result.dtype
        not in {accumulator_dtype, "bf16" if accumulator_dtype == "f32" else accumulator_dtype}
        for operand, result in zip(
            instruction.operand_shapes,
            instruction.result_shapes,
            strict=True,
        )
    ):
        return False
    # TPU XLA combines tuple all-reduces with one ordinary scalar reducer
    # shared by every operand, rather than a tuple-root reducer computation.
    representative = replace(
        instruction,
        operand_shapes=(instruction.operand_shapes[0],),
        result_shapes=(instruction.result_shapes[0],),
    )
    return _exact_scalar_add_reducer(
        representative,
        module_instructions=module_instructions,
        instructions_by_computation=instructions_by_computation,
    )


def _forbidden_full_hidden_values(
    instructions: tuple[HloInstruction, ...], *, hidden_size: int
) -> tuple[str, ...]:
    forbidden = []
    for instruction in instructions:
        shapes = (*instruction.result_shapes, *instruction.operand_shapes)
        if any(
            shape.dtype in {"bf16", "f32"}
            and shape.dimensions in {
                (32, hidden_size),
                (32, 1, hidden_size),
                (1, 32, hidden_size),
            }
            for shape in shapes
        ):
            forbidden.append(instruction.name)
    return tuple(sorted(set(forbidden)))


def _hidden_reconstructing_all_gathers(
    instructions: tuple[HloInstruction, ...], *, hidden_size: int
) -> tuple[str, ...]:
    """Reject a subgroup gather that physically rebuilds one hidden row."""

    forbidden = []
    for instruction in instructions:
        if (
            instruction.raw_opcode != "all-gather"
            or len(instruction.operand_shapes) != 1
            or len(instruction.result_shapes) != 1
        ):
            continue
        operand = instruction.operand_shapes[0]
        result = instruction.result_shapes[0]
        if (
            operand.dtype in {"bf16", "f32"}
            and result.dtype in {"bf16", "f32"}
            and result.element_count == hidden_size
            and instruction.maximum_group_size > 1
            and operand.element_count * instruction.maximum_group_size
            == result.element_count
        ):
            forbidden.append(instruction.name)
    return tuple(sorted(forbidden))


def validate_ws32_decoder_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
    hidden_size: int,
    kind: str = "decode",
    expected_split_rmsnorm_collective_count: int = 157,
) -> Ws32DecoderHloReport:
    """Validate one complete decoder/prefill graph before device execution.

    A first compile acquisition uses all-zero expected hashes.  Every
    structural rule still applies; only the two explicit identity violations
    remain admissible to that acquisition wrapper.  Numerical execution must
    use the exact acquired hashes.
    """

    if kind not in {"cache_probe", "decode", "observer", "prefill"}:
        raise ValueError("WS32 complete HLO kind is invalid")
    if hidden_size <= 0 or hidden_size % 4:
        raise ValueError("WS32 complete HLO hidden size is invalid")
    if (
        not isinstance(expected_split_rmsnorm_collective_count, int)
        or isinstance(expected_split_rmsnorm_collective_count, bool)
        or expected_split_rmsnorm_collective_count < 0
    ):
        raise ValueError("WS32 split RMSNorm count is invalid")
    expected_stablehlo_sha256 = _digest(
        expected_stablehlo_sha256, field="expected_stablehlo_sha256"
    )
    expected_optimized_hlo_sha256 = _digest(
        expected_optimized_hlo_sha256,
        field="expected_optimized_hlo_sha256",
    )
    stable_digest = sha256(stablehlo.encode("utf-8")).hexdigest()
    optimized_digest = sha256(optimized_hlo.encode("utf-8")).hexdigest()
    module = parse_hlo_module(optimized_hlo)
    mutable_computations: defaultdict[str, list[HloInstruction]] = defaultdict(
        list
    )
    for instruction in module.instructions:
        mutable_computations[
            instruction.computation.split(" ", 1)[0].lstrip("%")
        ].append(instruction)
    instructions_by_computation = {
        name: tuple(items) for name, items in mutable_computations.items()
    }
    live = _live_instruction_closure(module.instructions)
    live_keys = {(item.computation, item.name) for item in live}
    collectives = tuple(module.collectives)
    live_collectives = tuple(
        item
        for item in collectives
        if (item.computation, item.name) in live_keys
    )
    async_collectives = tuple(
        item
        for item in module.instructions
        if _ASYNC_COLLECTIVE.fullmatch(item.raw_opcode)
    )
    families = Counter(_group_family(item) for item in collectives)
    fused_rmsnorm_collectives = tuple(
        item
        for item in live_collectives
        if item.op_name is not None
        and "greenfield_ws32_fused_rmsnorm" in item.op_name.split("/")
    )
    rounded_first_rmsnorm_collectives = tuple(
        item
        for item in live_collectives
        if item.op_name is not None
        and "greenfield_ws32_rmsnorm" in item.op_name.split("/")
    )
    forbidden_hidden = _forbidden_full_hidden_values(
        module.instructions, hidden_size=hidden_size
    )
    forbidden_hidden = tuple(
        sorted(
            set(forbidden_hidden)
            | set(
                _hidden_reconstructing_all_gathers(
                    live, hidden_size=hidden_size
                )
            )
        )
    )
    violations: list[str] = []
    if stable_digest != expected_stablehlo_sha256:
        violations.append("StableHLO identity drifted")
    if optimized_digest != expected_optimized_hlo_sha256:
        violations.append("optimized HLO identity drifted")
    required_scope = {
        "decode": "greenfield_ws32_complete_decoder",
        "observer": "greenfield_ws32_complete_decoder_dsa_observer",
        "prefill": "greenfield_ws32_teacher_forced_prefill",
        "cache_probe": "greenfield_ws32_cache_probe",
    }[kind]
    if not any(
        item.op_name is not None
        and required_scope in item.op_name.split("/")
        for item in live
    ):
        violations.append(
            f"optimized HLO lost live exact {kind} source scope"
        )
    lowered_stable = stablehlo.lower()
    present_host = tuple(
        marker for marker in _HOST_MARKERS if marker in lowered_stable
    )
    if present_host:
        violations.append(
            f"StableHLO contains host execution markers {present_host}"
        )
    if async_collectives:
        violations.append("optimized HLO contains async collective start/done")
    if not collectives:
        violations.append("complete WS32 graph contains no physical collective")
    if len(live_collectives) != len(collectives):
        violations.append("optimized HLO contains a dead collective decoy")
    if any(
        item.raw_opcode not in {"all-reduce", "all-gather"}
        for item in collectives
    ):
        violations.append("complete WS32 graph contains a forbidden collective")
    if families[None]:
        violations.append("complete WS32 graph uses an unknown replica group")
    if any(not item.use_global_device_ids for item in collectives):
        violations.append("complete WS32 collective omits global device ids")
    for item in collectives:
        if item.raw_opcode == "all-reduce" and not _exact_add_reducer(
            item,
            module_instructions=module.instructions,
            instructions_by_computation=instructions_by_computation,
        ):
            violations.append(f"{item.name} reducer is not exact scalar add")
    if forbidden_hidden:
        violations.append("optimized HLO reconstructs a full-pod hidden value")
    maximum_group = max(
        (item.maximum_group_size for item in collectives), default=0
    )
    if maximum_group > 8:
        violations.append("complete WS32 collective exceeds subgroup size eight")
    if module.num_partitions != 32:
        violations.append("complete WS32 executable is not partitioned 32 ways")
    expected_split_boundaries = (
        0 if kind == "cache_probe" else expected_split_rmsnorm_collective_count
    )
    if len(fused_rmsnorm_collectives) != expected_split_boundaries:
        violations.append(
            "complete WS32 graph lacks the exact split-residual RMSNorm "
            "boundary count"
        )
    if rounded_first_rmsnorm_collectives:
        violations.append(
            "complete WS32 graph retains rounded-first RMSNorm boundaries"
        )
    if any(
        _group_family(item) != "feature"
        or item.raw_opcode != "all-reduce"
        or any(shape.dtype != "f32" for shape in item.operand_shapes)
        for item in fused_rmsnorm_collectives
    ):
        violations.append(
            "split-residual RMSNorm reduction geometry drifted"
        )
    return Ws32DecoderHloReport(
        kind=kind,
        stablehlo_sha256=stable_digest,
        optimized_hlo_sha256=optimized_digest,
        instruction_count=len(module.instructions),
        live_instruction_count=len(live),
        collective_count=len(collectives),
        live_collective_count=len(live_collectives),
        all_reduce_count=sum(
            item.raw_opcode == "all-reduce" for item in collectives
        ),
        all_gather_count=sum(
            item.raw_opcode == "all-gather" for item in collectives
        ),
        feature_collective_count=families["feature"],
        expert_collective_count=families["expert"],
        fused_rmsnorm_collective_count=len(fused_rmsnorm_collectives),
        rounded_first_rmsnorm_collective_count=len(
            rounded_first_rmsnorm_collectives
        ),
        maximum_group_size=maximum_group,
        async_collective_count=len(async_collectives),
        forbidden_full_hidden_values=forbidden_hidden,
        violations=tuple(violations),
    )
