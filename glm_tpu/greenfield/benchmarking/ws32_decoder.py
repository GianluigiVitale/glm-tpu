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
    strategy_nd_dense_hidden_gather_count: int
    strategy_nd_dense_expert_gather_count: int
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
            "strategy_nd_dense_expert_gather_count": (
                self.strategy_nd_dense_expert_gather_count
            ),
            "strategy_nd_dense_hidden_gather_count": (
                self.strategy_nd_dense_hidden_gather_count
            ),
            "violations": list(self.violations),
        }


@dataclass(frozen=True, slots=True)
class Ws32ExactDsaMaterializerHloReport:
    kind: str
    stablehlo_sha256: str
    optimized_hlo_sha256: str
    instruction_count: int
    live_instruction_count: int
    collective_count: int
    maximum_group_size: int
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "collective_count": self.collective_count,
            "instruction_count": self.instruction_count,
            "kind": self.kind,
            "live_instruction_count": self.live_instruction_count,
            "maximum_group_size": self.maximum_group_size,
            "optimized_hlo_sha256": self.optimized_hlo_sha256,
            "passed": self.passed,
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
    instructions: tuple[HloInstruction, ...],
    *,
    hidden_size: int,
    allowed_instruction_indices: frozenset[int] = frozenset(),
) -> tuple[str, ...]:
    forbidden = []
    for instruction in instructions:
        if instruction.index in allowed_instruction_indices:
            continue
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


def _exact_wk_feature_slice_instructions(
    instructions: tuple[HloInstruction, ...],
    *,
    hidden_size: int,
    kind: str,
) -> tuple[frozenset[int], int]:
    """Admit only XLA's closed local W_K quarter-slice scaffolding.

    TPU XLA may tile one already-local ``f32[128, hidden]`` W_K weight into
    four ``f32[32, hidden]`` values and immediately restore it with
    ``ConcatBitcast``.  The intermediate shape aliases the batch-32 dead-row
    sentinel, so recognize the exact producer/consumer chain rather than
    creating a shape-wide exception.
    """

    full_shape = ("f32", (128, hidden_size))
    slice_shape = ("f32", (32, hidden_size))
    scalar_shape = ("s32", ())

    def shapes(value: tuple[Any, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((shape.dtype, shape.dimensions) for shape in value)

    by_key = {
        (instruction.computation, instruction.name): instruction
        for instruction in instructions
    }
    consumers: defaultdict[tuple[str, str], list[HloInstruction]] = defaultdict(
        list
    )
    for instruction in instructions:
        for operand_name in instruction.operand_names:
            consumers[(instruction.computation, operand_name)].append(instruction)

    expected_spans = ((0, 32), (32, 64), (64, 96), (96, 128))
    span_pattern = re.compile(
        rf"slice=\{{\s*\[([0-9]+):([0-9]+)\]\s*,\s*\[0:{hidden_size}\]\s*\}}"
    )
    required_scope = {
        "decode": "greenfield_ws32_complete_decoder",
        "observer": "greenfield_ws32_complete_decoder_dsa_observer",
        "prefill": "greenfield_ws32_teacher_forced_prefill",
    }.get(kind)
    if required_scope is None:
        return frozenset(), 0

    allowed: set[int] = set()
    group_count = 0
    for concat in instructions:
        if (
            concat.raw_opcode != "custom-call"
            or 'custom_call_target="ConcatBitcast"' not in concat.raw_line
            or shapes(concat.operand_shapes) != (slice_shape,) * 4
            or shapes(concat.result_shapes) != (full_shape,)
            or len(concat.operand_names) != 4
            or len(set(concat.operand_names)) != 4
        ):
            continue
        done = tuple(
            by_key.get((concat.computation, name))
            for name in concat.operand_names
        )
        if any(item is None for item in done):
            continue
        if any(
            item.raw_opcode != "slice-done"
            or shapes(item.operand_shapes)
            != (full_shape, slice_shape, scalar_shape)
            or shapes(item.result_shapes) != (slice_shape,)
            or len(item.operand_names) != 1
            for item in done
            if item is not None
        ):
            continue
        starts = tuple(
            by_key.get((concat.computation, item.operand_names[0]))
            for item in done
            if item is not None
        )
        if len(starts) != 4 or any(item is None for item in starts):
            continue
        if any(
            item.raw_opcode != "slice-start"
            or shapes(item.operand_shapes) != (full_shape,)
            or shapes(item.result_shapes)
            != (full_shape, slice_shape, scalar_shape)
            or len(item.operand_names) != 1
            for item in starts
            if item is not None
        ):
            continue
        concrete_done = tuple(item for item in done if item is not None)
        concrete_starts = tuple(item for item in starts if item is not None)
        matches = tuple(span_pattern.search(item.raw_line) for item in concrete_starts)
        spans = tuple(
            (int(match.group(1)), int(match.group(2)))
            for match in matches
            if match is not None
        )
        # ConcatBitcast operand order is not semantic: the sealed DB567 decode
        # graph carries 226 groups with non-ascending operand order (FP8 weight
        # tiles) and produced exact tokens; placement is carried by each
        # slice-start's ``slice=`` attribute.  Require the exact set of four
        # disjoint quarter spans in any order (a rotated W_K group appeared once
        # in the 2026-09-05 chunked-prefill acquisition).
        if (
            len(matches) != 4
            or any(match is None for match in matches)
            or tuple(sorted(spans)) != expected_spans
            or len({item.operand_names[0] for item in concrete_starts}) != 1
        ):
            continue
        if any(
            consumers[(item.computation, item.name)] != [next_item]
            for item, next_item in zip(
                concrete_starts, concrete_done, strict=True
            )
        ) or any(
            consumers[(item.computation, item.name)] != [concat]
            for item in concrete_done
        ):
            continue
        concat_consumers = consumers[(concat.computation, concat.name)]
        if len(concat_consumers) != 1:
            continue
        consumer = concat_consumers[0]
        consumer_scopes = (
            () if consumer.op_name is None else consumer.op_name.split("/")
        )
        exact_current_key_consumer = (
            consumer.raw_opcode == "fusion"
            and required_scope in consumer_scopes
            and "greenfield_ws32_exact_dsa" in consumer_scopes
            and "exact_current_key" in consumer_scopes
        )
        prefill_weight_carry = kind == "prefill" and consumer.raw_opcode == "tuple"
        if not (exact_current_key_consumer or prefill_weight_carry):
            continue
        allowed.update(
            item.index for item in (concat, *concrete_done, *concrete_starts)
        )
        group_count += 1
    return frozenset(allowed), group_count


def _hidden_reconstructing_all_gathers(
    instructions: tuple[HloInstruction, ...],
    *,
    hidden_size: int,
    allow_exact_dsa_feature_gather: bool = False,
    allow_strategy_nd_dense_feature_gather: bool = False,
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
            if (
                allow_exact_dsa_feature_gather
                and instruction.op_name is not None
                and "greenfield_ws32_exact_dsa"
                in instruction.op_name.split("/")
                and _group_family(instruction) == "feature"
            ):
                continue
            if (
                allow_strategy_nd_dense_feature_gather
                and instruction.op_name is not None
                and "greenfield_ws32_strategy_nd_dense"
                in instruction.op_name.split("/")
                and "hidden_gather" in instruction.op_name.split("/")
                and _group_family(instruction) == "feature"
                and operand.dtype == result.dtype == "bf16"
                and operand.dimensions == (1, hidden_size // 4)
                and result.dimensions == (1, hidden_size)
                and instruction.maximum_group_size == 4
            ):
                continue
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
    exact_dsa: bool = False,
    strategy_nd_dense: bool = False,
    host_main_rope_table: bool = False,
    full_indexer_count: int = 21,
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
    if not isinstance(exact_dsa, bool):
        raise ValueError("WS32 exact DSA HLO flag must be boolean")
    if not isinstance(strategy_nd_dense, bool):
        raise ValueError("WS32 StrategyND dense HLO flag must be boolean")
    if not isinstance(full_indexer_count, int) or isinstance(
        full_indexer_count, bool
    ) or full_indexer_count <= 0:
        raise ValueError("WS32 full-indexer HLO count is invalid")
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
    strategy_hidden_gathers = tuple(
        item
        for item in live_collectives
        if item.op_name is not None
        and "greenfield_ws32_strategy_nd_dense"
        in item.op_name.split("/")
        and "hidden_gather" in item.op_name.split("/")
    )
    strategy_expert_gathers = tuple(
        item
        for item in live_collectives
        if item.op_name is not None
        and "greenfield_ws32_strategy_nd_dense"
        in item.op_name.split("/")
        and "expert_gather" in item.op_name.split("/")
    )
    allowed_exact_wk_slice_indices, _ = (
        _exact_wk_feature_slice_instructions(
            module.instructions,
            hidden_size=hidden_size,
            kind=kind,
        )
        if exact_dsa
        else (frozenset(), 0)
    )
    forbidden_hidden = _forbidden_full_hidden_values(
        module.instructions,
        hidden_size=hidden_size,
        allowed_instruction_indices=allowed_exact_wk_slice_indices,
    )
    forbidden_hidden = tuple(
        sorted(
            set(forbidden_hidden)
            | set(
                _hidden_reconstructing_all_gathers(
                    live,
                    hidden_size=hidden_size,
                    allow_exact_dsa_feature_gather=exact_dsa,
                    allow_strategy_nd_dense_feature_gather=(
                        strategy_nd_dense
                    ),
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
    expected_strategy_count = (
        3 if strategy_nd_dense and kind != "cache_probe" else 0
    )
    if len(strategy_hidden_gathers) != expected_strategy_count or any(
        item.raw_opcode != "all-gather"
        or _group_family(item) != "feature"
        or item.maximum_group_size != 4
        or len(item.operand_shapes) != 1
        or len(item.result_shapes) != 1
        or item.operand_shapes[0].dtype != "bf16"
        or item.operand_shapes[0].dimensions != (1, hidden_size // 4)
        or item.result_shapes[0].dtype != "bf16"
        or item.result_shapes[0].dimensions != (1, hidden_size)
        for item in strategy_hidden_gathers
    ):
        violations.append(
            "WS32 StrategyND dense hidden gather count/geometry drifted"
        )
    if len(strategy_expert_gathers) != expected_strategy_count or any(
        item.raw_opcode != "all-gather"
        or _group_family(item) != "expert"
        or item.maximum_group_size != 8
        or len(item.operand_shapes) != 1
        or len(item.result_shapes) != 1
        or item.operand_shapes[0].dtype != "bf16"
        or item.operand_shapes[0].dimensions
        != (4, 1, hidden_size // 4)
        or item.result_shapes[0].dtype != "bf16"
        or item.result_shapes[0].dimensions
        != (32, 1, hidden_size // 4)
        for item in strategy_expert_gathers
    ):
        violations.append(
            "WS32 StrategyND dense expert gather count/geometry drifted"
        )
    if exact_dsa and kind != "cache_probe":

        def scoped(item: HloInstruction, marker: str) -> bool:
            return item.op_name is not None and marker in item.op_name.split("/")

        normalized_gathers = tuple(
            item
            for item in live_collectives
            if item.raw_opcode == "all-gather"
            and scoped(item, "greenfield_ws32_exact_dsa")
            and scoped(item, "normalized_feature_gather")
        )
        query_gathers = tuple(
            item
            for item in live_collectives
            if item.raw_opcode == "all-gather"
            and scoped(item, "query_feature_gather")
        )
        repair_gathers = tuple(
            item
            for item in live_collectives
            if item.raw_opcode == "all-gather"
            and any(
                part.startswith("prompt_m64_repair_layer_")
                for part in (() if item.op_name is None else item.op_name.split("/"))
            )
        )
        expected_normalized = full_indexer_count
        if len(normalized_gathers) != expected_normalized or any(
            _group_family(item) != "feature"
            or item.maximum_group_size != 4
            for item in normalized_gathers
        ):
            violations.append("exact WS32 DSA normalized gather count/geometry drifted")
        if len(query_gathers) != full_indexer_count or any(
            _group_family(item) != "feature"
            or item.maximum_group_size != 4
            for item in query_gathers
        ):
            violations.append("exact WS32 DSA tuple4 query gather count drifted")
        expected_repairs = full_indexer_count if kind == "prefill" else 0
        if len(repair_gathers) != expected_repairs or any(
            _group_family(item) != "feature"
            or item.maximum_group_size != 4
            for item in repair_gathers
        ):
            violations.append("exact WS32 prompt M64 repair gather count drifted")
        exact_markers = (
            "one_row_fused_qkv_a_n82_convolution",
            "tuple4_query",
            "exact_current_key",
            "default_score",
        )
        for marker in exact_markers:
            if not any(scoped(item, marker) for item in live):
                violations.append(f"exact WS32 DSA lost live {marker} scope")
        if any(scoped(item, "query_expert_gather") for item in live):
            violations.append("exact WS32 DSA retained expert-owned query gather")
        sixteen_kib = tuple(
            item
            for item in live
            if scoped(item, "tuple4_query")
            and "megacore_allreduce_bytes" in item.raw_line
            and re.search(
                r'megacore_allreduce_bytes(?:\\?"|[^0-9]){1,32}16384',
                item.raw_line,
            )
        )
        if len(sixteen_kib) != full_indexer_count:
            violations.append("exact WS32 DSA tuple4 16-KiB fusion count drifted")
    # Spec §23.8: the legacy-faithful main-attention rotary consumes one
    # replicated host BF16 cos|sin row per step; it is present iff declared and
    # never appears in the cache probe.  This is a separate top-level block: it
    # must not guard, or be guarded by, the exact-DSA structure checks above.
    table_scopes = {
        "greenfield_ws32_main_rope_table",
        "greenfield_ws32_main_rope_table_lookup",
    }
    table_present = any(
        item.op_name is not None and table_scopes & set(item.op_name.split("/"))
        for item in live
    )
    if host_main_rope_table and kind != "cache_probe":
        if not table_present:
            violations.append("WS32 main rotary host table scope is missing")
    elif table_present:
        violations.append("WS32 main rotary host table appears without a declaration")
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
        strategy_nd_dense_hidden_gather_count=len(
            strategy_hidden_gathers
        ),
        strategy_nd_dense_expert_gather_count=len(
            strategy_expert_gathers
        ),
        maximum_group_size=maximum_group,
        async_collective_count=len(async_collectives),
        forbidden_full_hidden_values=forbidden_hidden,
        violations=tuple(violations),
    )


def validate_ws32_exact_dsa_materializer_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
    kind: str,
    full_indexer_count: int = 21,
) -> Ws32ExactDsaMaterializerHloReport:
    """Fail closed on the two one-shot exact-owner executables."""

    if kind not in {"exact_materialize", "exact_promote"}:
        raise ValueError("WS32 exact materializer HLO kind is invalid")
    if not isinstance(full_indexer_count, int) or isinstance(
        full_indexer_count, bool
    ) or full_indexer_count <= 0:
        raise ValueError("WS32 exact materializer layer count is invalid")
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
    live = _live_instruction_closure(module.instructions)
    live_keys = {(item.computation, item.name) for item in live}
    collectives = tuple(
        item
        for item in module.collectives
        if (item.computation, item.name) in live_keys
    )
    violations: list[str] = []
    if stable_digest != expected_stablehlo_sha256:
        violations.append("StableHLO identity drifted")
    if optimized_digest != expected_optimized_hlo_sha256:
        violations.append("optimized HLO identity drifted")
    if len(collectives) != len(module.collectives):
        violations.append("exact materializer contains a dead collective decoy")
    if module.num_partitions != 32:
        violations.append("exact materializer is not partitioned 32 ways")
    entry_parameters = Counter(
        (shape.dtype, shape.dimensions)
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_opcode == "parameter"
        for shape in item.result_shapes
    )
    materialize_parameters = Counter(
        {
            ("u8", (2048, 1536)): full_indexer_count,
            ("f32", (16, 12)): full_indexer_count,
            ("u8", (576, 1536)): full_indexer_count,
            ("f32", (5, 12)): full_indexer_count,
            ("u8", (512, 2048)): full_indexer_count,
            ("f32", (4, 16)): full_indexer_count,
            ("u8", (128, 1536)): full_indexer_count,
            ("f32", (1, 12)): full_indexer_count,
            ("bf16", (4, 1536)): full_indexer_count,
        }
    )
    promote_parameters = Counter(
        {
            ("u8", (32, 6144, 82)): full_indexer_count,
            ("f32", (32, 48, 82)): full_indexer_count,
            ("f32", (1024, 2048)): full_indexer_count,
            ("bf16", (128, 6144)): full_indexer_count,
            ("bf16", (8, 6144)): full_indexer_count,
        }
    )
    expected_parameters = (
        materialize_parameters
        if kind == "exact_materialize"
        else promote_parameters
    )
    if entry_parameters != expected_parameters:
        violations.append("exact materializer entry parameter geometry drifted")
    if any(marker in stablehlo.lower() for marker in _HOST_MARKERS):
        violations.append("exact materializer StableHLO contains host execution")
    tuple_all_reduces = tuple(
        item for item in collectives if item.raw_opcode == "all-reduce"
    )
    allowed_tuple_all_reduce = bool(
        kind == "exact_materialize"
        and len(tuple_all_reduces) == 1
        and len(tuple_all_reduces[0].operand_shapes) == full_indexer_count
        and len(tuple_all_reduces[0].result_shapes) == full_indexer_count
        and all(
            shape.dtype == "f32" and shape.dimensions == (48,)
            for shape in (
                *tuple_all_reduces[0].operand_shapes,
                *tuple_all_reduces[0].result_shapes,
            )
        )
        and _group_family(tuple_all_reduces[0]) == "feature"
        and tuple_all_reduces[0].maximum_group_size == 4
        and tuple_all_reduces[0].use_global_device_ids
        and _exact_add_reducer(
            tuple_all_reduces[0], module_instructions=module.instructions
        )
    )
    if any(
        (
            item.raw_opcode != "all-gather"
            and not (
                allowed_tuple_all_reduce
                and item is tuple_all_reduces[0]
            )
        )
        or _group_family(item) not in {"feature", "expert"}
        or not item.use_global_device_ids
        for item in collectives
    ):
        violations.append("exact materializer collective geometry drifted")
    maximum_group = max(
        (item.maximum_group_size for item in collectives), default=0
    )
    if maximum_group > 8:
        violations.append("exact materializer collective exceeds group eight")
    if any(
        _ASYNC_COLLECTIVE.fullmatch(item.raw_opcode)
        for item in module.instructions
    ):
        violations.append("exact materializer contains async collectives")

    def marker_present(marker: str) -> bool:
        return any(
            item.op_name is not None and marker in item.op_name.split("/")
            for item in live
        )

    if kind == "exact_materialize":
        if not collectives:
            violations.append("exact owner decode contains no physical gather")
        for marker in ("qkv_a", "tuple4_query", "wk_decode", "head_owner"):
            if not marker_present(marker):
                violations.append(f"exact owner decode lost live {marker} scope")
        if marker_present("wk_promote"):
            violations.append("exact owner decode crossed the BF16 wk boundary")
    else:
        if collectives:
            violations.append("exact wk promotion unexpectedly communicates")
        if not marker_present("wk_promote"):
            violations.append("exact wk promotion lost its live source scope")
        if marker_present("wk_decode"):
            violations.append("exact wk promotion rematerializes raw wk")
    return Ws32ExactDsaMaterializerHloReport(
        kind=kind,
        stablehlo_sha256=stable_digest,
        optimized_hlo_sha256=optimized_digest,
        instruction_count=len(module.instructions),
        live_instruction_count=len(live),
        collective_count=len(collectives),
        maximum_group_size=maximum_group,
        violations=tuple(violations),
    )
