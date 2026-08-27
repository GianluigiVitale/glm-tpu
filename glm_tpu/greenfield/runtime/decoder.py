"""One-step all-stage raw-FP8 decoder program over the global device mesh."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Literal, NamedTuple

from ..errors import PlanValidationError
from ..kernels.layer import (
    AttentionFp8Weights,
    AttentionProjectionBackend,
    DenseFp8Weights,
    DsaFp8Weights,
    MoeFp8Weights,
    SparseMoeBackend,
    stage_local_transformer_layer_fp8_mapped,
    stage_local_transformer_layer_fp8_split_mapped,
)
from ..kernels.pallas import Fp8BlockMatmulConfig
from ..kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.fp8 import dequantize_fp8_bits_block_weight
from ..kernels.reference.linear import vocabulary_logits
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    promote_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from ..kernels.reference.rmsnorm import final_norm, fused_add_rms_norm
from ..kernels.reference.rotary import (
    build_rotary_table_host,
    rotary_table_sha256,
)
from ..kernels.stage_local import (
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    VIRTUAL_TP32_REDUCTION_ASSOCIATIONS,
    StageLinearBackend,
    VirtualTp32ReductionAssociation,
)
from ..model.schedule import PipelineSchedule, StageExecution
from ..model.state import DecoderStateLayout
from ..model.weights import (
    COMPLETE_EXPERT_RUNTIME_LAYOUT,
    FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
    FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    LEGACY_DENSE_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
    feature_expert_runtime_layout,
)
from ..sharding.hlo_contract import (
    HloInstruction,
    HloModule,
    HloShape,
    parse_hlo_module,
)
from ..sharding.stablehlo_dense_convolution import (
    validate_dense_final_layout_decoder_stablehlo,
)
from ..sharding.stablehlo_strategy_nd import (
    validate_strategy_nd_attention_stablehlo,
)
from ..types import ExecutionPlan, PlanName
from .pipeline import (
    PipelineSkeletonConfig,
    _canonical_groups,
    _canonical_pairs,
    _partition_maps,
)

REFERENCE_FEATURE_OUTPUT_TILE = 128
PROMOTED_FEATURE_OUTPUT_TILE = 256
TOKEN_OBSERVATION_CANDIDATES = 16
_TPU_DECODER_BACKEND_CONTRACTS = frozenset(
    (
        "tpu_v4_pp8_reference",
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
        "tpu_v4_pp16_reference",
        "tpu_v4_pp16_pallas_feature",
    )
)
_FEATURE_DECODER_BACKEND_CONTRACTS = frozenset(
    (
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
        "tpu_v4_pp16_pallas_feature",
    )
)
LAYER0_RESIDUAL_DISCRIMINATOR_VARIANTS = (
    ("baseline_bf16", False, False),
    ("attention_output_fp32", True, False),
    ("dense_down_fp32", False, True),
    ("attention_output_and_dense_down_fp32", True, True),
)
Layer0ResidualDiscriminatorKind = Literal[
    "combine_precision",
    "virtual_tp32",
    "attention_schedule",
    "attention_output_association",
    "strategy_nd_row0_association",
]
LAYER0_VIRTUAL_TP32_DISCRIMINATOR_VARIANTS = (
    VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
)
LAYER0_ATTENTION_SCHEDULE_DISCRIMINATOR_VARIANTS = (
    "attention_schedule_control",
    "replicated_monolithic_attention",
)
LAYER0_ATTENTION_OUTPUT_ASSOCIATION_VARIANTS = (
    "attention_output_association_control",
    *tuple(
        f"attention_output_{association}"
        for association in VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
    ),
)
LAYER0_STRATEGY_ND_ROW0_ASSOCIATION_VARIANTS = (
    "strategy_nd_row0_control",
    "strategy_nd_row0_both",
)
LAYER0_INGREDIENT_NAMES = (
    "selected_positions",
    "selected_scores",
    "selected_valid_counts",
    "normalized_input",
    "combined_residual",
    "current_cache_row",
    "owner_selected_positions",
    "owner_selected_valid_counts",
    "owner_selected_cache_values",
    "owner_selected_cache_valid",
    "sparse_partial_output",
    "sparse_partial_logsumexp",
    "sparse_partial_valid",
    "combined_attention_output",
    "combined_attention_logsumexp",
    "combined_attention_valid",
    "value_states",
    "attention_output_input",
    "attention_virtual_partials",
    "attention_local_update",
    "attention_reduced_update",
    "normalized_mlp",
    "post_attention_residual",
    "dense_virtual_partials",
    "dense_local_update",
    "dense_reduced_update",
    "next_hidden",
    "layer1_normalized",
    "contract_valid",
)


def _layer0_residual_discriminator_specs(
    kind: Layer0ResidualDiscriminatorKind,
) -> tuple[
    tuple[
        str,
        bool,
        bool,
        VirtualTp32ReductionAssociation | None,
        bool,
        bool,
    ],
    ...,
]:
    if kind == "combine_precision":
        return tuple(
            (name, attention_fp32, dense_fp32, None, False, False)
            for name, attention_fp32, dense_fp32 in (
                LAYER0_RESIDUAL_DISCRIMINATOR_VARIANTS
            )
        )
    if kind == "virtual_tp32":
        return tuple(
            (name, False, False, name, False, False)
            for name in LAYER0_VIRTUAL_TP32_DISCRIMINATOR_VARIANTS
        )
    if kind == "attention_schedule":
        return tuple(
            (name, False, False, None, False, monolithic)
            for name, monolithic in (
                ("attention_schedule_control", False),
                ("replicated_monolithic_attention", True),
            )
        )
    if kind == "attention_output_association":
        return (
            (
                "attention_output_association_control",
                False,
                False,
                None,
                False,
                False,
            ),
            *tuple(
                (
                    f"attention_output_{association}",
                    False,
                    False,
                    association,
                    True,
                    False,
                )
                for association in VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
            ),
        )
    if kind == "strategy_nd_row0_association":
        return (
            (
                "strategy_nd_row0_control",
                False,
                False,
                None,
                False,
                False,
            ),
            (
                "strategy_nd_row0_both",
                False,
                False,
                STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
                False,
                False,
            ),
        )
    raise PlanValidationError(
        f"unknown layer-0 residual discriminator kind: {kind}"
    )
_PREFILL_INDEX_REPAIR_BRANCH_OP_NAME_PREFIX = (
    "jit(execute)/shard_map/cond/branch_"
)
_HLO_CALLEE_PATTERN = re.compile(
    r"\b(?:body|calls|condition|to_apply)=(%?[^,\s}\]]+)"
)


def _is_prefill_index_repair_op_name(op_name: str | None) -> bool:
    """Recognize the post-scan repair before and after XLA inlining."""

    if op_name is None:
        return False
    if "physical_m64_prompt_index_key_chunk" in op_name or (
        "repair_stage_local_prompt_index_cache" in op_name
    ):
        return True
    if not op_name.startswith(_PREFILL_INDEX_REPAIR_BRANCH_OP_NAME_PREFIX):
        return False
    branch_and_tail = op_name.removeprefix(
        _PREFILL_INDEX_REPAIR_BRANCH_OP_NAME_PREFIX
    )
    branch, separator, tail = branch_and_tail.partition("_fun")
    return bool(
        separator
        and branch.isdigit()
        and (not tail or tail.startswith("/"))
    )


def _hlo_computation_identifier(header: str) -> str:
    value = header.removeprefix("ENTRY ").split(None, 1)[0]
    return value.removeprefix("%").split("(", 1)[0]


def _prefill_index_repair_computations(module: HloModule) -> set[str]:
    """Return only computations rooted in exact repair operation metadata.

    TPU SPMD lowering emits unnamed slice/custom-call scaffolding and unnamed
    fusion callees around otherwise named repair operations.  Following only
    explicit HLO call edges from computations containing the pinned repair
    names admits that compiler scaffolding without creating a module-wide
    shape exception.
    """

    headers_by_identifier = {
        _hlo_computation_identifier(instruction.computation): (
            instruction.computation
        )
        for instruction in module.instructions
    }
    call_graph: dict[str, set[str]] = {}
    repair_identifiers: set[str] = set()
    for instruction in module.instructions:
        source = _hlo_computation_identifier(instruction.computation)
        call_graph.setdefault(source, set()).update(
            match.removeprefix("%")
            for match in _HLO_CALLEE_PATTERN.findall(instruction.raw_line)
        )
        if _is_prefill_index_repair_op_name(instruction.op_name):
            repair_identifiers.add(source)

    pending = list(repair_identifiers)
    while pending:
        source = pending.pop()
        for target in call_graph.get(source, ()):
            if target in headers_by_identifier and target not in repair_identifiers:
                repair_identifiers.add(target)
                pending.append(target)
    return {
        headers_by_identifier[identifier]
        for identifier in repair_identifiers
        if identifier in headers_by_identifier
    }


def _classify_scoped_shape_occurrences(
    module: HloModule,
    *,
    forbidden_signatures: set[str],
    repair_allowed_signatures: set[str],
    prefill_index_repair: bool,
    instruction_allowed_signatures: Mapping[int, set[str]] | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    """Count forbidden shapes, admitting exact repair-rooted occurrences."""

    repair_computations = (
        _prefill_index_repair_computations(module)
        if prefill_index_repair
        else set()
    )
    allowed_repair: Counter[str] = Counter()
    allowed_instruction: Counter[str] = Counter()
    forbidden: Counter[str] = Counter()
    instruction_allowed_signatures = instruction_allowed_signatures or {}
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            signature = (
                f"{shape.dtype}["
                + ",".join(str(value) for value in shape.dimensions)
                + "]"
            )
            if signature not in forbidden_signatures:
                continue
            if signature in instruction_allowed_signatures.get(
                instruction.index, set()
            ):
                allowed_instruction[signature] += 1
            elif (
                signature in repair_allowed_signatures
                and instruction.computation in repair_computations
            ):
                allowed_repair[signature] += 1
            else:
                forbidden[signature] += 1
    return (
        dict(sorted(allowed_repair.items())),
        dict(sorted(allowed_instruction.items())),
        dict(sorted(forbidden.items())),
        len(repair_computations),
    )


def _exact_wk_feature_slice_instructions(
    module: HloModule,
) -> tuple[set[int], dict[str, Any]]:
    """Recognize TPU's exact local ``wk`` feature-slice scaffolding.

    A local ``f32[128,6144]`` key-projection weight is physically tiled into
    four ``f32[32,6144]`` feature slices before ``ConcatBitcast`` restores the
    same local weight. The slice shape happens to equal the historical
    batch-32 dead-row sentinel. Admit only the complete compiler pattern;
    any parameter, activation, incomplete slice group, or other consumer with
    that shape remains forbidden.
    """

    full_shape = ("f32", (128, 6144))
    slice_shape = ("f32", (32, 6144))
    scalar_shape = ("s32", ())

    def shapes(
        value: Sequence[HloShape],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((shape.dtype, shape.dimensions) for shape in value)

    by_computation_and_name = {
        (instruction.computation, instruction.name): instruction
        for instruction in module.instructions
    }
    allowed_indices: set[int] = set()
    groups: list[dict[str, Any]] = []
    expected_spans = ((0, 32), (32, 64), (64, 96), (96, 128))
    span_pattern = re.compile(
        r"slice=\{\[([0-9]+):([0-9]+)\],\s*\[0:6144\]\}"
    )

    for concat in module.instructions:
        if (
            concat.opcode != "custom-call"
            or 'custom_call_target="ConcatBitcast"' not in concat.raw_line
            or shapes(concat.operand_shapes) != (slice_shape,) * 4
            or shapes(concat.result_shapes) != (full_shape,)
            or len(concat.operand_names) != 4
            or len(set(concat.operand_names)) != 4
        ):
            continue
        done_instructions = tuple(
            by_computation_and_name.get(
                (concat.computation, operand_name)
            )
            for operand_name in concat.operand_names
        )
        if any(instruction is None for instruction in done_instructions):
            continue
        if any(
            instruction.opcode != "slice-done"
            or shapes(instruction.operand_shapes)
            != (full_shape, slice_shape, scalar_shape)
            or shapes(instruction.result_shapes) != (slice_shape,)
            or len(instruction.operand_names) != 1
            for instruction in done_instructions
            if instruction is not None
        ):
            continue
        start_instructions = tuple(
            by_computation_and_name.get(
                (concat.computation, instruction.operand_names[0])
            )
            for instruction in done_instructions
            if instruction is not None
        )
        if len(start_instructions) != 4 or any(
            instruction is None for instruction in start_instructions
        ):
            continue
        if any(
            instruction.opcode != "slice-start"
            or shapes(instruction.operand_shapes) != (full_shape,)
            or shapes(instruction.result_shapes)
            != (full_shape, slice_shape, scalar_shape)
            or len(instruction.operand_names) != 1
            for instruction in start_instructions
            if instruction is not None
        ):
            continue
        starts = tuple(
            instruction
            for instruction in start_instructions
            if instruction is not None
        )
        matches = tuple(span_pattern.search(item.raw_line) for item in starts)
        if any(match is None for match in matches):
            continue
        spans = tuple(
            (int(match.group(1)), int(match.group(2)))
            for match in matches
            if match is not None
        )
        if (
            spans != expected_spans
            or len({item.operand_names[0] for item in starts}) != 1
        ):
            continue
        accepted = (concat, *done_instructions, *starts)
        allowed_indices.update(
            instruction.index
            for instruction in accepted
            if instruction is not None
        )
        groups.append(
            {
                "computation": _hlo_computation_identifier(
                    concat.computation
                ),
                "concat": concat.name,
                "input": starts[0].operand_names[0],
                "spans": [list(span) for span in spans],
            }
        )

    return allowed_indices, {
        "group_count": len(groups),
        "groups": groups,
        "instruction_count": len(allowed_indices),
        "slice_shape": "f32[32,6144]",
        "source_shape": "f32[128,6144]",
    }


class DsaInternalObservation(NamedTuple):
    """Per-stage slots for already-live full-indexer diagnostic values."""

    normalized_hidden: Any
    q_a_state: Any
    query: Any
    head_weights: Any
    current_key: Any
    producer_layer_ids: Any


@dataclass(frozen=True, slots=True)
class DecoderStepConfig:
    stage_count: int
    local_parallel_size: int
    hidden_size: int
    selected_width: int
    context_capacity: int
    maximum_layer_slots: int
    maximum_full_indexer_slots: int
    logical_page_size: int
    local_rows_per_page: int
    packed_cache_width: int
    index_key_width: int
    dsa_indexer_heads: int
    vocab_size: int
    dsa_score_default_precision: bool = False
    main_rope_table_width: int = 64

    def __post_init__(self) -> None:
        for field in (
            "stage_count",
            "local_parallel_size",
            "hidden_size",
            "selected_width",
            "context_capacity",
            "maximum_layer_slots",
            "maximum_full_indexer_slots",
            "logical_page_size",
            "local_rows_per_page",
            "packed_cache_width",
            "index_key_width",
            "dsa_indexer_heads",
            "vocab_size",
            "main_rope_table_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"decoder {field} must be positive")
        if self.logical_page_size != (
            self.local_parallel_size * self.local_rows_per_page
        ):
            raise PlanValidationError("decoder local page geometry is inconsistent")
        if self.context_capacity % self.local_parallel_size:
            raise PlanValidationError(
                "decoder context capacity must shard exactly over the local group"
            )
        if self.main_rope_table_width % 2:
            raise PlanValidationError(
                "decoder main-RoPE table width must be even"
            )
        if not isinstance(self.dsa_score_default_precision, bool):
            raise PlanValidationError(
                "decoder DSA score-precision flag must be boolean"
            )

    @property
    def total_devices(self) -> int:
        return self.stage_count * self.local_parallel_size

    @property
    def metadata_width(self) -> int:
        # Positions, count, producer, visited, health, active.
        return self.selected_width + 5

    @property
    def dsa_observation_width(self) -> int:
        # Position ids, bit-exact FP32 selected-score payload, count, producer.
        return 2 * self.selected_width + 2

    @property
    def token_observation_candidates(self) -> int:
        # Never ask local top-k for more entries than a vocabulary shard owns.
        # The real GLM geometry uses the full diagnostic width; the smaller
        # value exists only for reduced semantic-test geometries.
        local_vocab = self.vocab_size // self.local_parallel_size
        return min(TOKEN_OBSERVATION_CANDIDATES, local_vocab)

    @property
    def token_observation_width(self) -> int:
        # Candidate token ids followed by bit-exact FP32-cast logit payloads.
        return 2 * self.token_observation_candidates

    @property
    def count_index(self) -> int:
        return self.selected_width

    @property
    def producer_index(self) -> int:
        return self.selected_width + 1

    @property
    def visited_index(self) -> int:
        return self.selected_width + 2

    @property
    def health_index(self) -> int:
        return self.selected_width + 3

    @property
    def active_index(self) -> int:
        return self.selected_width + 4


@dataclass(frozen=True, slots=True)
class DecoderStepProgram:
    config: DecoderStepConfig
    execute: Any
    mesh: Any
    input_specs: tuple[Any, ...]
    groups: tuple[tuple[int, ...], ...]
    pairs: tuple[tuple[int, int], ...]
    plan_hash: str
    schedule_hash: str
    state_layout_hash: str
    weight_layout_hash: str
    sparse_moe_backend: SparseMoeBackend
    feature_output_tile: int
    feature_fuse_route_weighting: bool
    feature_reconstruct_down_fp32: bool
    linear_backend: StageLinearBackend
    dsa_query_backend: StageLinearBackend
    dsa_query_exact_association: bool
    dsa_head_key_exact_association: bool
    dsa_score_default_precision: bool
    attention_projection_backend: AttentionProjectionBackend
    main_rope_table_enabled: bool
    pregathered_b512_attention: bool
    strategy_nd_attention_projection: bool
    dense_final_layout_convolution: bool
    main_rope_table_host: Any | None
    main_rope_table_sha256: str | None
    main_rope_table_bytes_per_device: int
    complete_token_path: bool
    observe_dsa_events: bool
    observe_dsa_internals: bool
    observe_layer_residuals: bool
    observe_prefill_index_inputs: bool
    prefill_index_weight_names: tuple[tuple[str, str], ...]
    dsa_query_weight_names: tuple[tuple[str, str], ...]
    materialize_dsa_query_weights_fp32: Any | None
    decode_prefill_index_weights_bf16: Any | None
    promote_prefill_index_weights_fp32: Any | None
    repair_prefill_index_cache: Any | None
    layer0_residual_discriminators: tuple[tuple[str, Any], ...]
    layer0_residual_discriminator_kind: Layer0ResidualDiscriminatorKind
    layer0_ingredients_observer: Any | None
    split_residual_state: bool


def _validate_pallas_feature_decoder_calls(
    optimized_hlo: str,
    *,
    sparse_layers: int,
    feature_output_tile: int = PROMOTED_FEATURE_OUTPUT_TILE,
    fuse_route_weighting: bool = False,
    reconstruct_down_fp32: bool = False,
    local_parallel_size: int = 4,
) -> dict[str, Any]:
    """Pin every production feature-MoE kernel and reject weight overlays."""

    if feature_output_tile not in (128, 256):
        raise PlanValidationError("feature output tile must be 128 or 256")
    if not isinstance(reconstruct_down_fp32, bool):
        raise PlanValidationError(
            "feature FP32 reconstruction flag must be boolean"
        )
    if reconstruct_down_fp32 and fuse_route_weighting:
        raise PlanValidationError(
            "feature FP32 reconstruction is incompatible with fused route "
            "weighting"
        )
    if local_parallel_size not in (2, 4):
        raise PlanValidationError(
            "feature Pallas decoder supports only LP2 and LP4"
        )
    local_intermediate = 2048 // local_parallel_size
    selected_name = (
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i"
        f"{local_intermediate}"
    )
    if feature_output_tile != 128:
        selected_name += f"_ot{feature_output_tile}"
    if reconstruct_down_fp32:
        selected_name += "_downf32"
    if fuse_route_weighting:
        selected_name += "_wsum"
    kernel_names = (
        selected_name,
        "greenfield_fp8_block_up_gate_m8_k6144_n"
        f"{local_intermediate}",
        "greenfield_fp8_block_matmul_m8_k"
        f"{local_intermediate}_n6144",
    ) + (
        ("greenfield_fp32_to_bf16_r8_h6144",)
        if reconstruct_down_fp32
        else ()
    )
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]
    calls_by_kernel = {}
    for name in kernel_names:
        pattern = re.compile(re.escape(name) + r"(?=[^A-Za-z0-9_]|$)")
        calls_by_kernel[name] = [
            line for line in custom_calls if pattern.search(line)
        ]
    kernel_counts = {
        name: len(lines) for name, lines in calls_by_kernel.items()
    }
    expected_kernel_counts = {name: sparse_layers for name in kernel_names}
    violations = []
    if kernel_counts != expected_kernel_counts:
        violations.append(
            "decoder feature-Pallas kernel counts drifted: "
            f"expected={expected_kernel_counts} observed={kernel_counts}"
        )

    selected_lines = calls_by_kernel[selected_name]
    malformed_selected = [
        line
        for line in selected_lines
        if line.count(f"u8[256,6144,{local_intermediate}]") < 2
        or f"u8[256,{local_intermediate},6144]" not in line
    ]
    if malformed_selected:
        violations.append(
            "decoder feature-Pallas selected kernels lack exact raw-U8 tables"
        )
    selected_result_shape = (
        "bf16[8,6144]"
        if fuse_route_weighting
        else (
            "f32[8,8,6144]"
            if reconstruct_down_fp32
            else "bf16[8,8,6144]"
        )
    )
    malformed_selected_result = [
        line for line in selected_lines if selected_result_shape not in line
    ]
    if malformed_selected_result:
        violations.append(
            "decoder feature-Pallas selected result shape drifted: "
            f"expected={selected_result_shape}"
        )
    if fuse_route_weighting and "bf16[8,8,6144]" in optimized_hlo:
        violations.append(
            "decoder retains the unfused routed [8,8,6144] HBM result"
        )
    if reconstruct_down_fp32:
        boundary_name = "greenfield_fp32_to_bf16_r8_h6144"
        malformed_boundary = [
            line
            for line in calls_by_kernel[boundary_name]
            if "f32[8,6144]" not in line or "bf16[8,6144]" not in line
        ]
        if malformed_boundary:
            violations.append(
                "decoder FP32 reconstruction boundaries lack exact FP32 "
                "operands and BF16 results"
            )

    shared_raw_shapes = {
        kernel_names[1]: (f"u8[{local_intermediate},6144]",),
        kernel_names[2]: (f"u8[6144,{local_intermediate}]",),
    }
    malformed_shared = [
        name
        for name, shapes in shared_raw_shapes.items()
        if any(
            any(shape not in line for shape in shapes)
            for line in calls_by_kernel[name]
        )
    ]
    if malformed_shared:
        violations.append(
            "decoder feature-Pallas shared kernels lack raw-U8 operands: "
            f"{malformed_shared}"
        )

    forbidden_shapes = (
        f"bf16[256,6144,{local_intermediate}]",
        f"f32[256,6144,{local_intermediate}]",
        f"bf16[256,{local_intermediate},6144]",
        f"f32[256,{local_intermediate},6144]",
    )
    forbidden_overlays = [
        shape for shape in forbidden_shapes if shape in optimized_hlo
    ]
    if forbidden_overlays:
        violations.append(
            "decoder contains complete decoded routed-expert overlays: "
            f"{forbidden_overlays}"
        )
    forbidden_formatted_overlays = [
        shape
        for shape in (
            f"f8e4m3fn[{local_intermediate},6144]",
            f"f8e4m3fn[6144,{local_intermediate}]",
        )
        if shape in optimized_hlo
    ]
    if forbidden_formatted_overlays:
        violations.append(
            "decoder contains whole-table shared FP8 formatting: "
            f"{forbidden_formatted_overlays}"
        )
    return {
        "expected_kernel_counts": expected_kernel_counts,
        "forbidden_decoded_expert_overlays": forbidden_overlays,
        "forbidden_formatted_shared_overlays": forbidden_formatted_overlays,
        "feature_output_tile": feature_output_tile,
        "local_intermediate": local_intermediate,
        "local_parallel_size": local_parallel_size,
        "fuse_route_weighting": fuse_route_weighting,
        "reconstruct_down_fp32": reconstruct_down_fp32,
        "kernel_counts": kernel_counts,
        "passed": not violations,
        "violations": violations,
    }


def _validate_fused_qkv_a_decoder_association(
    optimized_hlo: str,
    *,
    layers: int,
    prefill_index_repair: bool = False,
    dsa_head_key_exact_association: bool = False,
    module: HloModule | None = None,
) -> dict[str, Any]:
    """Require DB502's physical one-row N82 primitive in every layer."""

    if module is None:
        module = parse_hlo_module(optimized_hlo)
    convolutions = [
        instruction
        for instruction in module.instructions
        if instruction.raw_opcode == "convolution"
        and any(
            shape.dtype == "f32" and shape.dimensions == (1, 82)
            for shape in instruction.result_shapes
        )
        and "dim_labels=bf_io->bf" in instruction.raw_line
    ]
    lowered = optimized_hlo.lower()
    required_shapes = {
        "packed_weight": "u8[32,6144,82]" in lowered,
        "expanded_scale": "f32[32,48,82]" in lowered,
        "one_row_input": "bf16[1,6144]" in lowered,
        "one_row_convolution": bool(convolutions),
    }
    forbidden_candidates = {
        "u8[2048,6144]",
        "u8[576,6144]",
        "f32[16,48]",
        "f32[5,48]",
        "bf16[32,6144]",
        "f32[32,6144]",
    }
    exact_wk_feature_slice_indices: set[int] = set()
    exact_wk_feature_slice_contract: dict[str, Any] = {
        "group_count": 0,
        "groups": [],
        "instruction_count": 0,
        "slice_shape": "f32[32,6144]",
        "source_shape": "f32[128,6144]",
    }
    if dsa_head_key_exact_association:
        (
            exact_wk_feature_slice_indices,
            exact_wk_feature_slice_contract,
        ) = _exact_wk_feature_slice_instructions(module)
    (
        allowed_repair_shapes,
        allowed_exact_wk_feature_slices,
        forbidden_shape_counts,
        repair_computation_count,
    ) = _classify_scoped_shape_occurrences(
        module,
        forbidden_signatures=forbidden_candidates,
        repair_allowed_signatures={"f32[32,6144]"},
        prefill_index_repair=prefill_index_repair,
        instruction_allowed_signatures={
            index: {"f32[32,6144]"}
            for index in exact_wk_feature_slice_indices
        },
    )
    forbidden_shapes = tuple(forbidden_shape_counts)
    violations = []
    if len(convolutions) != layers:
        violations.append(
            "fused qkv-a convolution count drifted: "
            f"expected={layers} observed={len(convolutions)}"
        )
    if not all(required_shapes.values()):
        violations.append(
            f"fused qkv-a HLO lacks required shapes: {required_shapes}"
        )
    if forbidden_shapes:
        violations.append(
            "fused qkv-a HLO retains separate/dead-row state: "
            f"{forbidden_shapes}"
        )
    return {
        "convolution_count": len(convolutions),
        "expected_convolution_count": layers,
        "allowed_prefill_index_repair_shape_counts": allowed_repair_shapes,
        "allowed_exact_wk_feature_slice_shape_counts": (
            allowed_exact_wk_feature_slices
        ),
        "exact_wk_feature_slice_contract": (
            exact_wk_feature_slice_contract
        ),
        "forbidden_shapes": list(forbidden_shapes),
        "prefill_index_repair": prefill_index_repair,
        "prefill_index_repair_computation_count": repair_computation_count,
        "passed": not violations,
        "required_shapes": required_shapes,
        "violations": violations,
    }


def _validate_dense_final_layout_convolution_hlo(
    optimized_hlo: str,
    stablehlo: str | None,
    *,
    dense_layers: int,
    enabled: bool,
    groups: Sequence[Sequence[int]] = ((0, 1, 2, 3),),
    module: HloModule | None = None,
) -> dict[str, Any]:
    """Pin the true-row-one production form of the DB548 dense arithmetic.

    The protected probe already proves the eight-shard arithmetic and packed
    source lineage in isolation.  The complete decoder must independently
    prove that all three dense layers select that primitive, retain one live
    row, use the accepted ``[in, out]`` RHS layout, and do not silently fall
    back to the old fused Pallas dense kernel.
    """

    if not isinstance(enabled, bool):
        raise PlanValidationError(
            "dense final-layout convolution HLO flag must be boolean"
        )
    if module is None:
        module = parse_hlo_module(optimized_hlo)
    stablehlo_available = stablehlo is not None
    stablehlo_text = stablehlo or ""
    by_key = {
        (instruction.computation, instruction.name): instruction
        for instruction in module.instructions
    }

    def exact_rhs_layout(
        instruction: HloInstruction, shape: str
    ) -> bool:
        if len(instruction.operand_names) != 2:
            return False
        producer = by_key.get(
            (instruction.computation, instruction.operand_names[1])
        )
        return bool(
            producer is not None
            and producer.opcode
            in {
                "bitcast",
                "copy",
                "convert",
                "fusion",
                "optimization-barrier",
                "parameter",
                "reshape",
            }
            and re.search(
                rf"=\s*{re.escape(shape)}\{{1,0(?::|\}})",
                producer.raw_line,
            )
        )

    expected_per_kind = 8 * dense_layers if enabled else 0
    expected_rank_count = dense_layers if enabled else 0
    gate: list[HloInstruction] = []
    down: list[HloInstruction] = []
    malformed_scoped: list[str] = []
    rank_counts = {
        "gate_up": Counter(),
        "down": Counter(),
    }
    scope = re.compile(
        r"(?:^|/)greenfield_dense_convolution_virtual_rank_(0[0-7])/"
    )
    for instruction in module.instructions:
        op_name = instruction.op_name or ""
        match = scope.search(op_name)
        if match is None or instruction.raw_opcode != "convolution":
            continue
        rank = int(match.group(1))
        operands = _hlo_shape_signature(instruction.operand_shapes)
        results = _hlo_shape_signature(instruction.result_shapes)
        exact_dimensions = "dim_labels=bf_io->bf" in re.sub(
            r"\s+", "", instruction.raw_line
        )
        if (
            operands == ("bf16[1,6144]", "bf16[6144,768]")
            and results == ("f32[1,768]",)
            and exact_dimensions
            and exact_rhs_layout(instruction, "bf16[6144,768]")
        ):
            gate.append(instruction)
            rank_counts["gate_up"][rank] += 1
        elif (
            operands == ("bf16[1,384]", "bf16[384,6144]")
            and results == ("f32[1,6144]",)
            and exact_dimensions
            and exact_rhs_layout(instruction, "bf16[384,6144]")
        ):
            down.append(instruction)
            rank_counts["down"][rank] += 1
        else:
            malformed_scoped.append(instruction.name)

    normalized_stable = tuple(
        re.sub(r"\s+", "", line)
        for line in stablehlo_text.splitlines()
        if "stablehlo.convolution" in line
    )
    stable_attributes = (
        "dim_numbers=[b,f]x[i,o]->[b,f]",
        "window={stride=[],pad=[],lhs_dilate=[],rhs_dilate=[],reverse=[]}",
        "batch_group_count=1:i64,feature_group_count=1:i64",
        "precision_config=[#stablehlo<precisionDEFAULT>,"
        "#stablehlo<precisionDEFAULT>]",
    )
    stable_gate = [
        line
        for line in normalized_stable
        if "(tensor<1x6144xbf16>,tensor<6144x768xbf16>)"
        "->tensor<1x768xf32>" in line
        and all(attribute in line for attribute in stable_attributes)
    ]
    stable_down = [
        line
        for line in normalized_stable
        if "(tensor<1x384xbf16>,tensor<384x6144xbf16>)"
        "->tensor<1x6144xf32>" in line
        and all(attribute in line for attribute in stable_attributes)
    ]
    stable_layout_constraints = sum(
        "@LayoutConstraint(" in line
        and "tensor<6144x768xbf16>" in line
        for line in stablehlo_text.splitlines()
    )
    expected_ranks = {
        rank: expected_rank_count for rank in range(8)
    }
    observed_rank_counts = {
        label: dict(sorted(counts.items()))
        for label, counts in rank_counts.items()
    }
    parameters: dict[str, dict[int, tuple[str, str]]] = {}
    roots: dict[str, tuple[str, str]] = {}

    def computation_symbol(value: str) -> str:
        parts = value.split()
        if parts and parts[0] == "ENTRY":
            parts = parts[1:]
        return parts[0] if parts else value

    for instruction in module.instructions:
        symbol = computation_symbol(instruction.computation)
        if instruction.opcode == "parameter":
            match = re.search(r"\bparameter\(([0-9]+)\)", instruction.raw_line)
            if match is not None:
                parameters.setdefault(symbol, {})[int(match.group(1))] = (
                    instruction.computation,
                    instruction.name,
                )
        if instruction.raw_line.lstrip().startswith("ROOT "):
            roots[symbol] = (instruction.computation, instruction.name)

    # Follow actual value flow through fusion/call boundaries.  Caller
    # operands enter only through the corresponding callee parameter, and a
    # callee reaches its call result only through its ROOT.  This prevents an
    # unused argument or a dead convolution from satisfying liveness.
    forward_edges: dict[
        tuple[str, str],
        list[tuple[tuple[str, str], str, int | None]],
    ] = {}

    def add_edge(
        source: tuple[str, str],
        target: tuple[str, str],
        kind: str,
        index: int | None = None,
    ) -> None:
        if source in by_key and target in by_key:
            forward_edges.setdefault(source, []).append((target, kind, index))

    call_opcodes = {"call", "conditional", "fusion", "while"}
    for instruction in module.instructions:
        target = (instruction.computation, instruction.name)
        structural_line = re.sub(
            r'"(?:\\.|[^"\\])*"',
            '""',
            re.sub(r"/\*.*?\*/", "", instruction.raw_line),
        )
        if instruction.opcode == "parameter":
            continue
        if instruction.opcode == "tuple":
            for index, operand_name in enumerate(instruction.operand_names):
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "tuple",
                    index,
                )
        elif instruction.opcode == "get-tuple-element":
            index_match = re.search(
                r"\bindex=([0-9]+)", instruction.raw_line
            )
            if instruction.operand_names and index_match is not None:
                add_edge(
                    (
                        instruction.computation,
                        instruction.operand_names[0],
                    ),
                    target,
                    "get-tuple-element",
                    int(index_match.group(1)),
                )
        elif instruction.opcode not in call_opcodes:
            for operand_name in instruction.operand_names:
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "ordinary",
                )
        called = re.search(
            r"\bcalls=(%[A-Za-z0-9_.:-]+)", structural_line
        )
        if called is None:
            while_body = re.search(
                r"\bbody=(%[A-Za-z0-9_.:-]+)",
                structural_line,
            )
            if (
                instruction.opcode == "while"
                and len(instruction.operand_names) == 1
                and while_body is not None
            ):
                symbol = while_body.group(1)
                parameter = parameters.get(symbol, {}).get(0)
                if parameter is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[0],
                        ),
                        parameter,
                        "identity",
                    )
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")
                continue
            branch_match = re.search(
                r"\bbranch_computations=\{([^}]*)\}",
                structural_line,
            )
            if branch_match is None:
                continue
            branch_symbols = tuple(
                item.strip()
                for item in branch_match.group(1).split(",")
            )
            if (
                instruction.opcode != "conditional"
                or len(instruction.operand_names) != len(branch_symbols) + 1
            ):
                continue
            for branch_index, symbol in enumerate(branch_symbols):
                parameter = parameters.get(symbol, {}).get(0)
                if parameter is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[branch_index + 1],
                        ),
                        parameter,
                        "identity",
                    )
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")
        else:
            symbol = called.group(1)
            for index, operand_name in enumerate(instruction.operand_names):
                parameter = parameters.get(symbol, {}).get(index)
                if parameter is not None:
                    add_edge(
                        (instruction.computation, operand_name),
                        parameter,
                        "identity",
                    )
            root = roots.get(symbol)
            if root is not None:
                add_edge(root, target, "identity")

    gate_keys = {(item.computation, item.name) for item in gate}
    down_keys = {(item.computation, item.name) for item in down}
    entry_root_keys = {
        (instruction.computation, instruction.name)
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    }

    def first_reached_targets(
        seed: tuple[str, str], targets: set[tuple[str, str]]
    ) -> set[tuple[str, str]]:
        ValuePath = tuple[int, ...] | None
        pending: list[tuple[tuple[str, str], ValuePath]] = [(seed, ())]
        visited: set[tuple[tuple[str, str], ValuePath]] = {(seed, ())}
        reached: set[tuple[str, str]] = set()
        while pending:
            current, path = pending.pop()
            for target, kind, index in forward_edges.get(current, ()):
                target_instruction = by_key[target]
                if kind == "identity":
                    target_path = path
                elif kind == "tuple":
                    assert index is not None
                    target_path = None if path is None else (index, *path)
                elif kind == "get-tuple-element":
                    assert index is not None
                    if path is None or path == ():
                        # The seed is never an unclassified tuple.  Reaching a
                        # GTE without a concrete tuple-element path means an
                        # intervening operation erased provenance; fail closed
                        # instead of allowing every index.
                        continue
                    elif path[0] != index:
                        continue
                    else:
                        target_path = path[1:]
                else:
                    exact_tuple_identity = (
                        target_instruction.opcode
                        in {"copy", "optimization-barrier"}
                        and len(target_instruction.operand_names) == 1
                        and len(target_instruction.result_shapes) > 1
                    )
                    target_path = (
                        path
                        if exact_tuple_identity
                        else None
                        if len(target_instruction.result_shapes) > 1
                        else ()
                    )
                if target in targets:
                    reached.add(target)
                    continue
                state = (target, target_path)
                if state not in visited:
                    visited.add(state)
                    pending.append(state)
        return reached

    gate_down_links = {
        key: first_reached_targets(key, down_keys) for key in gate_keys
    }
    down_root_links = {
        key: first_reached_targets(key, entry_root_keys) for key in down_keys
    }
    exact_gate_down_bijection = bool(
        enabled
        and len(gate_down_links) == expected_per_kind
        and all(len(targets) == 1 for targets in gate_down_links.values())
        and Counter(
            target
            for targets in gate_down_links.values()
            for target in targets
        )
        == Counter({key: 1 for key in down_keys})
    )
    exact_down_result_liveness = bool(
        enabled
        and len(down_root_links) == expected_per_kind
        and all(targets for targets in down_root_links.values())
    )
    exact_stablehlo = (
        validate_dense_final_layout_decoder_stablehlo(
            stablehlo_text,
            dense_layers=dense_layers,
            replica_groups=groups,
        )
        if enabled and stablehlo_available
        else {
            "dense_layer_group_count": 0,
            "exact_live_result_count": 0,
            "exact_runtime_u8_bitcast_count": 0,
            "gate_up_layout_constraint_count": 0,
            "matched_virtual_shard_count": 0,
            "dead_row_node_count": 0,
            "passed": not enabled,
            "violations": [],
        }
    )
    stable_dead_rows = int(exact_stablehlo["dead_row_node_count"])
    violations = []
    if enabled and not stablehlo_available:
        violations.append(
            "dense final-layout contract requires paired StableHLO"
        )
    if malformed_scoped:
        violations.append(
            "dense final-layout scoped convolution geometry drifted: "
            f"{malformed_scoped}"
        )
    if len(gate) != expected_per_kind or len(down) != expected_per_kind:
        violations.append(
            "dense final-layout optimized convolution count drifted: "
            f"expected={expected_per_kind}/{expected_per_kind} "
            f"observed={len(gate)}/{len(down)}"
        )
    if enabled and any(
        dict(counts) != expected_ranks for counts in rank_counts.values()
    ):
        violations.append(
            "dense final-layout virtual-rank bijection drifted: "
            f"{observed_rank_counts}"
        )
    if enabled and not exact_gate_down_bijection:
        violations.append(
            "dense final-layout gate/up and down convolutions are not live/bijective"
        )
    if enabled and not exact_down_result_liveness:
        violations.append(
            "dense final-layout down convolutions do not reach the decoder result"
        )
    if enabled and not exact_stablehlo["passed"]:
        violations.append(
            "dense final-layout exact StableHLO lineage drifted: "
            f"{exact_stablehlo['violations']}"
        )
    if len(stable_gate) != expected_per_kind or len(stable_down) != expected_per_kind:
        violations.append(
            "dense final-layout StableHLO convolution count/arithmetic drifted: "
            f"expected={expected_per_kind}/{expected_per_kind} "
            f"observed={len(stable_gate)}/{len(stable_down)}"
        )
    if stable_layout_constraints != expected_per_kind:
        violations.append(
            "dense final-layout StableHLO layout-constraint count drifted: "
            f"expected={expected_per_kind} observed={stable_layout_constraints}"
        )
    if enabled and stable_dead_rows:
        violations.append("dense final-layout decoder contains M32/dead-row state")
    return {
        "applicable": enabled,
        "expected_convolution_count_per_kind": expected_per_kind,
        "gate_up_convolution_count": len(gate),
        "down_convolution_count": len(down),
        "optimized_virtual_rank_counts": observed_rank_counts,
        "optimized_exact_gate_down_bijection": exact_gate_down_bijection,
        "optimized_exact_down_result_liveness": exact_down_result_liveness,
        "stablehlo_exact_arithmetic_contract": exact_stablehlo,
        "stablehlo_gate_up_convolution_count": len(stable_gate),
        "stablehlo_down_convolution_count": len(stable_down),
        "stablehlo_layout_constraint_count": stable_layout_constraints,
        "stablehlo_dead_row_signature_count": stable_dead_rows,
        "passed": not violations,
        "violations": violations,
    }


def _validate_dsa_query_decoder_association(
    optimized_hlo: str,
    *,
    full_indexer_layers: int,
    local_parallel_size: int,
    dsa_indexer_heads: int,
    index_key_width: int,
    backend: StageLinearBackend,
    exact_association: bool = False,
) -> dict[str, Any]:
    """Pin the exact query projection to a local owner, never a global table."""

    global_output_width = dsa_indexer_heads * index_key_width
    local_output_width = global_output_width // local_parallel_size
    local_shape = f"f32[{local_output_width},2048]"
    global_shapes = tuple(
        f"{dtype}[{global_output_width},2048]"
        for dtype in ("bf16", "f32", "f8e4m3fn")
    )
    local_shape_occurrences = sum(
        local_shape in line for line in optimized_hlo.splitlines()
    )
    forbidden_global_shapes = [
        shape for shape in global_shapes if shape in optimized_hlo
    ]
    tuple4_reduction_key = '"megacore_allreduce_bytes":"16384"'
    all_16k_reduction_fusion_count = optimized_hlo.count(
        tuple4_reduction_key
    )
    tuple4_reduction_fusion_count = sum(
        tuple4_reduction_key in line
        and line.split(" fusion(", 1)[0].count(
            f"f32[{local_output_width}]"
        )
        == 4
        for line in optimized_hlo.splitlines()
    )
    violations = []
    if not isinstance(exact_association, bool):
        raise PlanValidationError(
            "exact DSA query association flag must be boolean"
        )
    if global_output_width % local_parallel_size:
        violations.append("DSA query width does not shard over the local group")
    if forbidden_global_shapes:
        violations.append(
            "decoder reconstructs a global DSA query weight: "
            f"{forbidden_global_shapes}"
        )
    if backend == "reference":
        if local_shape_occurrences < full_indexer_layers:
            violations.append(
                "decoder lost the complete local FP32 DSA owner boundary: "
                f"shape={local_shape} expected_at_least={full_indexer_layers} "
                f"observed={local_shape_occurrences}"
            )
        if exact_association and (
            tuple4_reduction_fusion_count != full_indexer_layers
        ):
            violations.append(
                "decoder lost the exact four-reduction DSA query fusion: "
                f"expected={full_indexer_layers} "
                f"observed={tuple4_reduction_fusion_count}"
            )
        if not exact_association and tuple4_reduction_fusion_count:
            violations.append(
                "default decoder unexpectedly enables the exact DSA query "
                "association"
            )
    elif backend == "pallas":
        if local_shape_occurrences:
            violations.append(
                "Pallas DSA query path retains a decoded local weight overlay"
            )
    else:
        raise PlanValidationError("DSA query HLO backend is unknown")
    return {
        "backend": backend,
        "forbidden_global_shapes": forbidden_global_shapes,
        "local_owner_shape": local_shape,
        "local_owner_shape_occurrences": local_shape_occurrences,
        "exact_association": exact_association,
        "all_16k_reduction_fusion_count": (
            all_16k_reduction_fusion_count
        ),
        "tuple4_reduction_fusion_count": tuple4_reduction_fusion_count,
        "expected_tuple4_reduction_fusion_count": (
            full_indexer_layers if exact_association else 0
        ),
        "passed": not violations,
        "violations": violations,
    }


def _validate_dsa_head_key_decoder_association(
    optimized_hlo: str,
    *,
    full_indexer_layers: int,
    maximum_full_indexer_slots: int,
    exact_association: bool,
    prefill_index_repair: bool = False,
) -> dict[str, Any]:
    """Pin DB527's external wk, BF16 head boundary, and divide/sqrt path."""

    if not isinstance(exact_association, bool):
        raise PlanValidationError(
            "exact DSA head/key association flag must be boolean"
        )
    if not isinstance(prefill_index_repair, bool):
        raise PlanValidationError(
            "prefill index-repair HLO flag must be boolean"
        )
    module = parse_hlo_module(optimized_hlo)
    local_wk_shape = "f32[128,6144]"
    external_wk_parameters = sum(
        instruction.computation.startswith("ENTRY ")
        and instruction.raw_opcode == "parameter"
        and len(instruction.result_shapes) == 1
        and instruction.result_shapes[0].dtype == "f32"
        and instruction.result_shapes[0].dimensions[-2:] == (128, 6144)
        for instruction in module.instructions
    )
    key_reduction_marker = '"megacore_allreduce_bytes":"8192"'
    key_projection_signature_count = sum(
        key_reduction_marker in line
        and "f32[128]" in line.split(" fusion(", 1)[0]
        for line in optimized_hlo.splitlines()
    )
    key_sqrt_count = sum(
        " = f32[]" in line
        and " sqrt(" in line
        and "/shard_map/" in line
        for line in optimized_hlo.splitlines()
    )
    normalized_f32_convert_count = sum(
        " = f32[1,6144]" in line
        and " convert(" in line
        and "/shard_map/" in line
        for line in optimized_hlo.splitlines()
    )
    key_projection_count = (
        key_projection_signature_count
        if exact_association or external_wk_parameters or key_sqrt_count
        else 0
    )
    forbidden_global_shapes = [
        shape
        for shape in ("f32[32,128,6144]", "f32[4,128,6144]")
        if shape in optimized_hlo
    ]
    violations = []
    if forbidden_global_shapes:
        violations.append(
            "exact DSA head/key state escaped its stage-local owner: "
            f"{forbidden_global_shapes}"
        )
    if exact_association:
        if external_wk_parameters < maximum_full_indexer_slots:
            violations.append(
                "decoder lost external FP32 wk owners: "
                f"expected_at_least={maximum_full_indexer_slots} "
                f"observed={external_wk_parameters}"
            )
        if key_projection_count != full_indexer_layers:
            violations.append(
                "decoder lost exact recurrent key projections: "
                f"expected={full_indexer_layers} "
                f"observed={key_projection_count}"
            )
        if key_sqrt_count != full_indexer_layers:
            violations.append(
                "decoder lost divide-by-sqrt recurrent key norms: "
                f"expected={full_indexer_layers} observed={key_sqrt_count}"
            )
        if normalized_f32_convert_count < full_indexer_layers:
            violations.append(
                "decoder lost completed BF16 normalized head/key boundaries: "
                f"expected_at_least={full_indexer_layers} "
                f"observed={normalized_f32_convert_count}"
            )
    elif (
        (external_wk_parameters and not prefill_index_repair)
        or key_sqrt_count
    ):
        # ``f32[128]`` plus an 8-KiB megacore reduction is not sufficient to
        # identify the exact recurrent-key projection on LP2.  The ordinary
        # owner-split attention LSE has that same optimized fusion signature
        # when two owners each contribute 64 values.  Exact head/key state is
        # still unambiguously rejected by its external FP32 ``wk`` owner or
        # divide-by-sqrt normalization, and the Pallas/reference projection
        # counts are pinned independently by the stage-linear contract.
        violations.append(
            "default decoder unexpectedly enables exact DSA head/key state"
        )
    return {
        "exact_association": exact_association,
        "external_wk_parameter_count": external_wk_parameters,
        "expected_external_wk_parameter_count": (
            maximum_full_indexer_slots if exact_association else 0
        ),
        "forbidden_global_shapes": forbidden_global_shapes,
        "key_projection_count": key_projection_count,
        "key_projection_signature_count": (
            key_projection_signature_count
        ),
        "expected_key_projection_count": (
            full_indexer_layers if exact_association else 0
        ),
        "key_sqrt_count": key_sqrt_count,
        "expected_key_sqrt_count": (
            full_indexer_layers if exact_association else 0
        ),
        "local_wk_shape": local_wk_shape,
        "normalized_f32_convert_count": normalized_f32_convert_count,
        "prefill_index_repair": prefill_index_repair,
        "passed": not violations,
        "violations": violations,
    }


def validate_dsa_query_weight_materializer_hlo(
    optimized_hlo: str,
    *,
    full_indexer_slots: int,
    local_output_width: int,
    q_lora_rank: int,
    total_devices: int,
) -> dict[str, Any]:
    """Require a local raw-FP8-to-FP32 owner materializer only."""

    if full_indexer_slots <= 0:
        raise PlanValidationError("DSA query materializer needs local slots")
    module = parse_hlo_module(optimized_hlo)
    local_raw_shape = f"u8[{local_output_width},{q_lora_rank}]"
    scale_rows = (local_output_width + 127) // 128
    scale_columns = (q_lora_rank + 127) // 128
    local_scale_shape = f"f32[{scale_rows},{scale_columns}]"
    local_fp32_shape = f"f32[{local_output_width},{q_lora_rank}]"
    raw_occurrences = optimized_hlo.count(local_raw_shape)
    scale_occurrences = optimized_hlo.count(local_scale_shape)
    fp32_occurrences = optimized_hlo.count(local_fp32_shape)

    def entry_parameter_count(dtype: str, tail: tuple[int, ...]) -> int:
        return sum(
            instruction.computation.startswith("ENTRY ")
            and instruction.raw_opcode == "parameter"
            and len(instruction.result_shapes) == 1
            and instruction.result_shapes[0].dtype == dtype
            and instruction.result_shapes[0].dimensions[-len(tail) :] == tail
            for instruction in module.instructions
        )

    raw_parameter_count = entry_parameter_count(
        "u8", (local_output_width, q_lora_rank)
    )
    scale_parameter_count = entry_parameter_count(
        "f32", (scale_rows, scale_columns)
    )
    forbidden_operations = sorted(
        {
            instruction.opcode
            for instruction in module.instructions
            if instruction.opcode
            in {
                "all-gather",
                "all-reduce",
                "all-to-all",
                "collective-permute",
                "outfeed",
                "reduce-scatter",
            }
        }
    )
    custom_call_targets = sorted(
        set(
            re.findall(
                r'custom_call_target="([^"]+)"',
                optimized_hlo,
            )
        )
    )
    allowed_metadata_custom_call_targets = (
        "AssumeGatherIndicesInBound",
        "GatherScatterIndicesBitpacked",
    )
    forbidden_custom_call_targets = sorted(
        set(custom_call_targets)
        - set(allowed_metadata_custom_call_targets)
    )
    lowered = optimized_hlo.lower()
    host_markers = sorted(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in lowered
    )
    forbidden_global_shapes = [
        shape
        for shape in (
            f"u8[{local_output_width * 4},{q_lora_rank}]",
            f"f32[{local_output_width * 4},{q_lora_rank}]",
        )
        if shape in optimized_hlo
    ]
    violations = []
    if raw_parameter_count < full_indexer_slots:
        violations.append("DSA query materializer lost local raw owners")
    if scale_parameter_count < full_indexer_slots:
        violations.append("DSA query materializer lost local scale owners")
    if fp32_occurrences < full_indexer_slots:
        violations.append("DSA query materializer lost local FP32 outputs")
    if forbidden_operations:
        violations.append(
            "DSA query materializer contains communication/callbacks: "
            f"{forbidden_operations}"
        )
    if forbidden_custom_call_targets:
        violations.append(
            "DSA query materializer contains an unapproved custom call: "
            f"{forbidden_custom_call_targets}"
        )
    if host_markers:
        violations.append(
            "DSA query materializer contains a host callback: "
            f"{host_markers}"
        )
    if forbidden_global_shapes:
        violations.append(
            "DSA query materializer reconstructs global state: "
            f"{forbidden_global_shapes}"
        )
    if module.num_partitions not in (None, total_devices):
        violations.append(
            "DSA query materializer partition count drifted: "
            f"expected={total_devices} observed={module.num_partitions}"
        )
    return {
        "full_indexer_slots": full_indexer_slots,
        "local_raw_shape": local_raw_shape,
        "local_raw_shape_occurrences": raw_occurrences,
        "local_raw_parameter_count": raw_parameter_count,
        "local_scale_shape": local_scale_shape,
        "local_scale_shape_occurrences": scale_occurrences,
        "local_scale_parameter_count": scale_parameter_count,
        "local_fp32_shape": local_fp32_shape,
        "local_fp32_shape_occurrences": fp32_occurrences,
        "allowed_metadata_custom_call_targets": list(
            allowed_metadata_custom_call_targets
        ),
        "custom_call_targets": custom_call_targets,
        "forbidden_custom_call_targets": forbidden_custom_call_targets,
        "forbidden_operations": forbidden_operations,
        "forbidden_global_shapes": forbidden_global_shapes,
        "host_markers": host_markers,
        "num_partitions": module.num_partitions,
        "passed": not violations,
        "violations": violations,
    }


def _validate_pallas_stage_linear_decoder_calls(
    optimized_hlo: str,
    *,
    layers: int,
    dense_layers: int,
    full_indexer_layers: int,
    dsa_query_backend: StageLinearBackend = "pallas",
    dsa_head_key_exact_association: bool = False,
    attention_projection_backend: AttentionProjectionBackend = "separate",
    strategy_nd_attention_projection: bool = False,
    dense_final_layout_convolution: bool = False,
    prefill_index_repair: bool = False,
    module: HloModule | None = None,
) -> dict[str, Any]:
    """Pin every raw-FP8 attention/dense projection replacing an overlay."""

    if attention_projection_backend not in (
        "separate",
        "fused_n82_convolution",
    ):
        raise PlanValidationError("attention projection HLO backend is unknown")
    separate_qkv_a_calls = (
        layers if attention_projection_backend == "separate" else 0
    )
    expected_kernel_counts = {
        "greenfield_fp8_block_matmul_m8_k6144_n2048": separate_qkv_a_calls,
        "greenfield_fp8_block_matmul_m8_k2048_n4096": layers,
        "greenfield_fp8_block_matmul_m8_k6144_n640": separate_qkv_a_calls,
        "greenfield_fp8_block_matmul_m8_k4096_n6144": (
            0 if strategy_nd_attention_projection else layers
        ),
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144": (
            8 * layers if strategy_nd_attention_projection else 0
        ),
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": layers,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": layers,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": (
            full_indexer_layers if dsa_query_backend == "pallas" else 0
        ),
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128": (
            0 if dsa_head_key_exact_association else full_indexer_layers
        ),
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": (
            0 if dense_final_layout_convolution else dense_layers
        ),
    }
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]
    kernel_counts = {
        name: sum(name in line for line in custom_calls)
        for name in expected_kernel_counts
    }
    if dsa_query_backend not in ("reference", "pallas"):
        raise PlanValidationError("DSA query HLO backend is unknown")
    if not isinstance(dsa_head_key_exact_association, bool):
        raise PlanValidationError("exact DSA head/key HLO flag must be boolean")
    if not isinstance(strategy_nd_attention_projection, bool):
        raise PlanValidationError(
            "StrategyND attention-projection HLO flag must be boolean"
        )
    decoded_dimensions = (
        "2048,6144",
        "4096,2048",
        "576,6144",
        "6144,4096",
        "7168,512",
        "128,6144",
        "3072,6144",
        "6144,3072",
    )
    if dsa_query_backend != "reference":
        decoded_dimensions += ("1024,2048",)
    forbidden_signatures = {
        f"{dtype}[{shape}]"
        for dtype in ("bf16", "f32")
        for shape in decoded_dimensions
    }
    if dsa_head_key_exact_association:
        forbidden_signatures.discard("f32[128,6144]")
    formatted_dimensions = (
        "2048,6144",
        "4096,2048",
        "6144,4096",
        "7168,512",
        "1024,2048",
        "128,6144",
    )
    forbidden_formatted_signatures = {
        f"f8e4m3fn[{shape}]"
        for shape in formatted_dimensions
    }
    repair_allowed_signatures = {
        "bf16[2048,6144]",
        "f32[128,6144]",
    }
    if optimized_hlo.lstrip().startswith("HloModule "):
        if module is None:
            module = parse_hlo_module(optimized_hlo)
        (
            allowed_repair_overlays,
            _allowed_instruction_overlays,
            forbidden_overlay_counts,
            repair_computation_count,
        ) = _classify_scoped_shape_occurrences(
            module,
            forbidden_signatures=(
                forbidden_signatures | forbidden_formatted_signatures
            ),
            repair_allowed_signatures=repair_allowed_signatures,
            prefill_index_repair=prefill_index_repair,
        )
        forbidden_overlays = [
            shape
            for shape in forbidden_overlay_counts
            if shape in forbidden_signatures
        ]
        forbidden_formatted_overlays = [
            shape
            for shape in forbidden_overlay_counts
            if shape in forbidden_formatted_signatures
        ]
    else:
        # Tiny unit fixtures predate full module parsing. They receive no
        # repair exception because operation scope cannot be proven.
        allowed_repair_overlays = {}
        repair_computation_count = 0
        forbidden_overlays = sorted(
            shape for shape in forbidden_signatures if shape in optimized_hlo
        )
        forbidden_formatted_overlays = sorted(
            shape
            for shape in forbidden_formatted_signatures
            if shape in optimized_hlo
        )
    violations = []
    if kernel_counts != expected_kernel_counts:
        violations.append(
            "decoder stage-linear Pallas kernel counts drifted: "
            f"expected={expected_kernel_counts} observed={kernel_counts}"
        )
    if forbidden_overlays:
        violations.append(
            "decoder retains decoded attention/dense weight overlays: "
            f"{forbidden_overlays}"
        )
    if forbidden_formatted_overlays:
        violations.append(
            "decoder performs whole-table FP8 input formatting outside "
            f"Pallas: {forbidden_formatted_overlays}"
        )
    return {
        "attention_projection_backend": attention_projection_backend,
        "allowed_prefill_index_repair_shape_counts": (
            allowed_repair_overlays
        ),
        "prefill_index_repair": prefill_index_repair,
        "prefill_index_repair_computation_count": repair_computation_count,
        "expected_kernel_counts": expected_kernel_counts,
        "dsa_query_backend": dsa_query_backend,
        "dsa_head_key_exact_association": (
            dsa_head_key_exact_association
        ),
        "forbidden_decoded_weight_overlays": forbidden_overlays,
        "forbidden_formatted_weight_overlays": forbidden_formatted_overlays,
        "kernel_counts": kernel_counts,
        "passed": not violations,
        "violations": violations,
    }


def _hlo_shape_signature(shapes: Sequence[Any]) -> tuple[str, ...]:
    return tuple(
        f"{shape.dtype}["
        + ",".join(str(dimension) for dimension in shape.dimensions)
        + "]"
        for shape in shapes
    )


def _compact_collective_record(collective: Any) -> dict[str, Any]:
    return {
        "channel_id": collective.channel_id,
        "name": collective.name,
        "op_name": collective.op_name,
        "operand_shapes": list(_hlo_shape_signature(collective.operand_shapes)),
        "raw_opcode": collective.raw_opcode,
        "replica_groups": [list(group) for group in collective.replica_groups],
        "result_shapes": list(_hlo_shape_signature(collective.result_shapes)),
        "source_target_pairs": [
            list(pair) for pair in collective.source_target_pairs
        ],
        "use_global_device_ids": collective.use_global_device_ids,
    }


def _classify_decoder_live_tensor_shapes(
    module: HloModule,
    *,
    config: DecoderStepConfig,
    full_indexer_layers: int,
    backend_contract: str,
    prefill_index_repair: bool = False,
    dsa_head_key_exact_association: bool = False,
    strategy_nd_attention_projection: bool = False,
) -> dict[str, Any]:
    """Separate pinned DSA head scores from forbidden batch/full-pod tensors.

    The real GLM geometry has 32 DSA indexer heads and the protected pod has
    32 devices.  At 8K, each LP4 context shard is also 2,048 positions wide,
    so the legitimate local scorer matrix ``f32[32,2048]`` collides exactly
    with the old shape-only dead-row sentinel.  An exception based only on
    dtype and dimensions would weaken the one-live-row contract.  Instead,
    admit that matrix only inside the exact TPU DSA scorer dataflow already
    present in every full-indexer layer: head/key contraction, scale/clamp,
    bitcast, and head reduction.  Any different producer, dtype, operation,
    count, or computation remains forbidden.
    """

    local_context_width = (
        config.context_capacity // config.local_parallel_size
    )
    score_dimensions = (
        config.dsa_indexer_heads,
        local_context_width,
    )
    score_shape = ("f32", score_dimensions)
    query_shape = (
        "f32",
        (config.dsa_indexer_heads, config.index_key_width),
    )
    key_shape = (
        "bf16",
        (local_context_width, config.index_key_width),
    )
    expanded_score_shape = (
        "f32",
        (1, config.dsa_indexer_heads, local_context_width),
    )
    reduced_score_shape = ("f32", (local_context_width,))

    def shape_key(shape: HloShape) -> tuple[str, tuple[int, ...]]:
        return shape.dtype, shape.dimensions

    def has_result(
        instruction: HloInstruction,
        expected: tuple[str, tuple[int, ...]],
    ) -> bool:
        return any(shape_key(shape) == expected for shape in instruction.result_shapes)

    def has_operand(
        instruction: HloInstruction,
        expected: tuple[str, tuple[int, ...]],
    ) -> bool:
        return any(shape_key(shape) == expected for shape in instruction.operand_shapes)

    instructions_by_computation: dict[str, list[HloInstruction]] = {}
    for instruction in module.instructions:
        instructions_by_computation.setdefault(
            instruction.computation, []
        ).append(instruction)

    expected_score_opcode_counts = {
        "bitcast": 1,
        "broadcast": 2,
        "convolution": 1,
        "maximum": 1,
        "multiply": 1,
    }

    def accepted_score_instruction(instruction: HloInstruction) -> bool:
        op_name = instruction.op_name or ""
        if instruction.opcode == "convolution":
            return op_name.endswith(
                "/shard_map/cond/branch_1_fun/"
                "rhd,sd->rhs/dot_general"
            )
        if instruction.opcode == "broadcast":
            return re.search(r"/shard_map/broadcast\.[0-9]+$", op_name) is not None
        if instruction.opcode == "multiply":
            return op_name.endswith(
                "/shard_map/cond/branch_1_fun/mul"
            )
        if instruction.opcode in ("maximum", "bitcast"):
            return op_name.endswith(
                "/shard_map/cond/branch_1_fun/max"
            )
        return False

    valid_computations: set[str] = set()
    body_records = []
    if backend_contract != "cpu_reference":
        for computation, instructions in instructions_by_computation.items():
            contractions = [
                instruction
                for instruction in instructions
                if instruction.opcode == "convolution"
                and instruction.op_name is not None
                and instruction.op_name.endswith(
                    "/shard_map/cond/branch_1_fun/"
                    "rhd,sd->rhs/dot_general"
                )
                and has_result(instruction, score_shape)
                and has_operand(instruction, query_shape)
                and has_operand(instruction, key_shape)
            ]
            head_weight_broadcasts = [
                instruction
                for instruction in instructions
                if instruction.opcode == "broadcast"
                and instruction.op_name is not None
                and instruction.op_name.endswith(
                    "/shard_map/cond/branch_1_fun/"
                    "rh,rhs->rs/dot_general"
                )
                and has_operand(
                    instruction,
                    ("f32", (config.dsa_indexer_heads,)),
                )
                and has_result(instruction, expanded_score_shape)
            ]
            head_weight_multiplications = [
                instruction
                for instruction in instructions
                if instruction.opcode == "multiply"
                and instruction.op_name is not None
                and re.search(
                    r"/shard_map/multiply\.[0-9]+$",
                    instruction.op_name,
                )
                is not None
                and sum(
                    shape_key(shape) == expanded_score_shape
                    for shape in instruction.operand_shapes
                )
                == 2
                and has_result(instruction, expanded_score_shape)
            ]
            reductions = [
                instruction
                for instruction in instructions
                if instruction.opcode == "reduce"
                and instruction.op_name is not None
                and instruction.op_name.endswith(
                    "/shard_map/cond/branch_1_fun/"
                    "rh,rhs->rs/dot_general"
                )
                and has_operand(instruction, expanded_score_shape)
                and has_result(instruction, reduced_score_shape)
            ]
            score_instructions = [
                instruction
                for instruction in instructions
                if any(
                    shape_key(shape) == score_shape
                    for shape in (
                        instruction.operand_shapes
                        + instruction.result_shapes
                    )
                )
            ]
            if not contractions and not reductions and not score_instructions:
                continue
            opcode_counts = dict(
                sorted(Counter(item.opcode for item in score_instructions).items())
            )
            shape_occurrences = sum(
                sum(
                    shape_key(shape) == score_shape
                    for shape in (
                        instruction.operand_shapes
                        + instruction.result_shapes
                    )
                )
                for instruction in score_instructions
            )
            valid = (
                len(contractions) == 1
                and len(head_weight_broadcasts) == 1
                and len(head_weight_multiplications) == 1
                and len(reductions) == 1
                and opcode_counts == expected_score_opcode_counts
                and shape_occurrences == 10
                and all(
                    accepted_score_instruction(instruction)
                    for instruction in score_instructions
                )
                and sum(
                    "operand_precision={highest,highest}"
                    in instruction.raw_line
                    for instruction in contractions
                )
                == (0 if config.dsa_score_default_precision else 1)
            )
            body_records.append(
                {
                    "computation": computation,
                    "contraction_count": len(contractions),
                    "head_weight_broadcast_count": len(
                        head_weight_broadcasts
                    ),
                    "head_weight_multiply_count": len(
                        head_weight_multiplications
                    ),
                    "opcode_counts": opcode_counts,
                    "reduction_count": len(reductions),
                    "score_shape_occurrences": shape_occurrences,
                    "highest_precision_contraction_count": sum(
                        "operand_precision={highest,highest}"
                        in instruction.raw_line
                        for instruction in contractions
                    ),
                    "valid": valid,
                }
            )
            if valid:
                valid_computations.add(computation)

    repair_computations = (
        _prefill_index_repair_computations(module)
        if prefill_index_repair
        else set()
    )
    repair_shape = ("f32", (config.total_devices, config.hidden_size))
    exact_wk_feature_slice_indices: set[int] = set()
    exact_wk_feature_slice_contract: dict[str, Any] = {
        "group_count": 0,
        "groups": [],
        "instruction_count": 0,
        "slice_shape": "f32[32,6144]",
        "source_shape": "f32[128,6144]",
    }
    if dsa_head_key_exact_association:
        (
            exact_wk_feature_slice_indices,
            exact_wk_feature_slice_contract,
        ) = _exact_wk_feature_slice_instructions(module)
    allowed_prefill_index_repair_shapes = []
    allowed_exact_wk_feature_slices = []
    allowed_dsa_score_shapes = []
    allowed_strategy_nd_virtual_partials = []
    forbidden_shapes = []
    hard_forbidden_dimensions = {
        (config.total_devices, config.hidden_size),
        (config.total_devices, 1, config.hidden_size),
        (config.total_devices, 2, 1, config.hidden_size),
        (config.total_devices, config.selected_width),
        (config.total_devices, 1, config.selected_width),
        (
            config.total_devices,
            config.dsa_indexer_heads,
            config.index_key_width,
        ),
        (
            config.total_devices,
            1,
            config.dsa_indexer_heads,
            config.index_key_width,
        ),
    }
    strategy_nd_virtual_partial_shape = (
        "bf16",
        (config.total_devices, 1, config.hidden_size),
    )
    strategy_nd_virtual_partial_keys: set[tuple[str, str]] = set()
    if strategy_nd_attention_projection:
        gather_scope = "greenfield_strategy_nd_row0_association_gather"
        strategy_gathers = tuple(
            instruction
            for instruction in module.collectives
            if instruction.op_name is not None
            and gather_scope in instruction.op_name.split("/")
        )
        gather_keys = tuple(
            (instruction.computation, instruction.name)
            for instruction in strategy_gathers
        )
        strategy_nd_virtual_partial_keys = {
            key
            for key, bits in _HloValueFlow(module).provenance(
                gather_keys
            ).items()
            if bits
        }
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            if shape.dimensions not in hard_forbidden_dimensions:
                continue
            record = {
                "computation": instruction.computation,
                "instruction": instruction.name,
                "op_name": instruction.op_name,
                "opcode": instruction.opcode,
                "shape": shape.to_dict(),
            }
            is_pinned_dsa_score = (
                shape_key(shape) == score_shape
                and instruction.computation in valid_computations
                and accepted_score_instruction(instruction)
            )
            if is_pinned_dsa_score:
                allowed_dsa_score_shapes.append(record)
            elif (
                shape_key(shape) == strategy_nd_virtual_partial_shape
                and (instruction.computation, instruction.name)
                in strategy_nd_virtual_partial_keys
            ):
                # This is 32 virtual BF16 contraction partials resident on
                # each LP4 owner after an exact four-way local gather, not a
                # residual reconstructed over the physical 32-chip pod.
                allowed_strategy_nd_virtual_partials.append(record)
            elif (
                shape_key(shape) == repair_shape
                and instruction.index in exact_wk_feature_slice_indices
            ):
                allowed_exact_wk_feature_slices.append(record)
            elif (
                shape_key(shape) == repair_shape
                and instruction.computation in repair_computations
            ):
                allowed_prefill_index_repair_shapes.append(record)
            else:
                forbidden_shapes.append(record)

    expected_body_count = (
        0 if backend_contract == "cpu_reference" else full_indexer_layers
    )
    body_violations = []
    if len(valid_computations) != expected_body_count:
        body_violations.append(
            "decoder local DSA score-body contract drifted: "
            f"expected={expected_body_count} "
            f"observed={len(valid_computations)}"
        )
    return {
        "allowed_dsa_score_shapes": allowed_dsa_score_shapes,
        "allowed_prefill_index_repair_shapes": (
            allowed_prefill_index_repair_shapes
        ),
        "allowed_strategy_nd_virtual_partials": (
            allowed_strategy_nd_virtual_partials
        ),
        "allowed_exact_wk_feature_slices": (
            allowed_exact_wk_feature_slices
        ),
        "body_records": body_records,
        "expected_score_body_count": expected_body_count,
        "exact_wk_feature_slice_contract": (
            exact_wk_feature_slice_contract
        ),
        "forbidden_shapes": forbidden_shapes,
        "local_context_width": local_context_width,
        "passed": not body_violations and not forbidden_shapes,
        "prefill_index_repair": prefill_index_repair,
        "prefill_index_repair_computation_count": len(
            repair_computations
        ),
        "score_body_count": len(valid_computations),
        "score_dimensions": list(score_dimensions),
        "score_precision": (
            "default" if config.dsa_score_default_precision else "highest"
        ),
        "strategy_nd_attention_projection": (
            strategy_nd_attention_projection
        ),
        "violations": body_violations,
    }


def _validate_complete_token_collective_lowering(
    module: Any,
    *,
    expected_groups: Sequence[Sequence[int]],
    expected_pairs: Sequence[Sequence[int]],
    backend_contract: str,
    token_observation_candidates: int = 1,
    dsa_query_exact_association: bool = False,
    main_rope_table_enabled: bool = False,
    local_parallel_size: int = 4,
) -> dict[str, Any]:
    """Pin TPU's compact top-1 exchange and one-token return lowering."""

    if backend_contract == "cpu_reference":
        return {
            "applicable": False,
            "backend_contract": backend_contract,
            "lowering": "cpu_reference_all_gather",
            "passed": True,
            "violations": [],
        }

    canonical_groups = tuple(tuple(group) for group in expected_groups)
    canonical_pairs = tuple(tuple(pair) for pair in expected_pairs)
    reductions = tuple(
        item for item in module.collectives if item.opcode == "all-reduce"
    )
    permutes = tuple(
        item
        for item in module.collectives
        if item.opcode == "collective-permute"
    )
    score_exchange = tuple(
        item
        for item in reductions
        if len(item.result_shapes) == 1
        and item.result_shapes[0].dtype == "bf16"
        and prod(item.result_shapes[0].dimensions)
        == local_parallel_size * token_observation_candidates
    )
    token_id_exchange = tuple(
        item
        for item in reductions
        if len(item.result_shapes) == 1
        and item.result_shapes[0].dtype == "s32"
        and prod(item.result_shapes[0].dimensions)
        == local_parallel_size * token_observation_candidates
    )
    token_return = tuple(
        item
        for item in permutes
        if _hlo_shape_signature(item.operand_shapes) == ("s32[1]",)
    )
    violations = []
    if len(score_exchange) != 1:
        violations.append(
            "complete-token score exchange must be exactly one local "
            f"{token_observation_candidates}-candidate "
            f"all-reduce, found {len(score_exchange)}"
        )
    if len(token_id_exchange) != 1:
        violations.append(
            "complete-token id exchange must be exactly one local "
            f"{token_observation_candidates}-candidate "
            f"all-reduce, found {len(token_id_exchange)}"
        )
    for label, candidates, expected_dtype in (
        ("score", score_exchange, "bf16"),
        ("id", token_id_exchange, "s32"),
    ):
        for collective in candidates:
            if (
                len(collective.operand_shapes) != 1
                or collective.operand_shapes[0].dtype != expected_dtype
                or prod(collective.operand_shapes[0].dimensions)
                != local_parallel_size * token_observation_candidates
            ):
                violations.append(
                    f"complete-token {label} exchange operand shape drifted"
                )
            if collective.replica_groups != canonical_groups:
                violations.append(
                    "complete-token "
                    f"{label} exchange escaped pipeline-local groups"
                )
            if not collective.use_global_device_ids:
                violations.append(
                    f"complete-token {label} exchange lacks global device ids"
                )
            if re.search(
                r"\bto_apply=%?add(?:\.|,|\s|$)", collective.raw_line
            ) is None:
                violations.append(
                    f"complete-token {label} exchange is not the pinned "
                    "one-hot sum lowering"
                )
    if len(token_return) != 1:
        violations.append(
            "complete-token return must be exactly one s32[1] "
            f"collective-permute, found {len(token_return)}"
        )
    mapped_token_name = (
        "mapped_token_exact_query"
        if dsa_query_exact_association
        else "mapped_token"
    )
    if main_rope_table_enabled:
        mapped_token_name += "_main_rope"
    direct_token_return_op_name = (
        f"jit({mapped_token_name})/shard_map/ppermute"
    )
    accepted_token_return_op_names = (
        direct_token_return_op_name,
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    )
    for collective in token_return:
        if _hlo_shape_signature(collective.result_shapes) != (
            "s32[1]",
            "s32[1]",
            "u32[]",
            "u32[]",
        ):
            violations.append("complete-token return TPU result shape drifted")
        if collective.source_target_pairs != canonical_pairs:
            violations.append("complete-token return lane pairs drifted")
        if collective.op_name not in accepted_token_return_op_names:
            violations.append("complete-token return source operation drifted")
    return {
        "applicable": True,
        "backend_contract": backend_contract,
        "lowering": "local_one_hot_all_reduce",
        "token_observation_candidates": token_observation_candidates,
        "accepted_token_return_op_names": list(
            accepted_token_return_op_names
        ),
        "passed": not violations,
        "score_exchange": [
            _compact_collective_record(item) for item in score_exchange
        ],
        "token_id_exchange": [
            _compact_collective_record(item) for item in token_id_exchange
        ],
        "token_return": [
            _compact_collective_record(item) for item in token_return
        ],
        "violations": violations,
    }


def _validate_main_rope_table_hlo(
    module: HloModule,
    *,
    config: DecoderStepConfig,
    layers: int,
    enabled: bool,
    required_entry_root_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Pin the explicit table input and main-MLA-only FP32 rotary scope."""

    table_shape = (
        config.context_capacity,
        config.main_rope_table_width,
    )
    table_parameters = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.opcode == "parameter"
        and any(
            shape.dtype == "bf16" and shape.dimensions == table_shape
            for shape in instruction.result_shapes
        )
    )
    named_table_parameters = tuple(
        instruction
        for instruction in table_parameters
        if instruction.op_name is not None
        and "main_rope_table" in instruction.op_name
    )
    scoped = tuple(
        instruction
        for instruction in module.instructions
        if instruction.op_name is not None
        and "greenfield_main_rope_table" in instruction.op_name
    )
    forbidden_opcodes = {
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-permute",
        "cosine",
        "power",
        "reduce-scatter",
        "sine",
    }
    forbidden = tuple(
        instruction.raw_line
        for instruction in scoped
        if instruction.opcode in forbidden_opcodes
    )
    bf16_arithmetic = tuple(
        instruction.raw_line
        for instruction in scoped
        if instruction.opcode in {"add", "multiply", "subtract"}
        and any(
            shape.dtype == "bf16" for shape in instruction.result_shapes
        )
    )
    scoped_fp32_multiplies = tuple(
        instruction
        for instruction in scoped
        if instruction.opcode == "multiply"
        and any(shape.dtype == "f32" for shape in instruction.result_shapes)
    )
    fp32_multiply_count = len(scoped_fp32_multiplies)
    scoped_fp32_combines = tuple(
        instruction
        for instruction in scoped
        if instruction.opcode in {"add", "subtract"}
        and any(shape.dtype == "f32" for shape in instruction.result_shapes)
    )
    fp32_combine_count = len(scoped_fp32_combines)

    def computation_symbol(value: str) -> str:
        parts = value.split()
        if parts and parts[0] == "ENTRY":
            parts = parts[1:]
        return parts[0] if parts else value

    instruction_by_key = {
        (instruction.computation, instruction.name): instruction
        for instruction in module.instructions
    }
    parameters: dict[str, dict[int, tuple[str, str]]] = {}
    roots: dict[str, tuple[str, str]] = {}
    for instruction in module.instructions:
        symbol = computation_symbol(instruction.computation)
        if instruction.opcode == "parameter":
            parameter_match = re.search(
                r"\bparameter\(([0-9]+)\)", instruction.raw_line
            )
            if parameter_match is not None:
                parameters.setdefault(symbol, {})[
                    int(parameter_match.group(1))
                ] = (instruction.computation, instruction.name)
        if instruction.raw_line.lstrip().startswith("ROOT "):
            roots[symbol] = (instruction.computation, instruction.name)

    # Edges preserve tuple element paths through branch arguments and call
    # results.  This is deliberately a small HLO value-flow proof rather than
    # an op-name heuristic: optimized TPU HLO moves the table lookup and rotary
    # arithmetic into separate fusion/conditional computations.
    edge_map: dict[
        tuple[str, str],
        list[tuple[tuple[str, str], str, int | None]],
    ] = {}

    def add_edge(
        source: tuple[str, str],
        target: tuple[str, str],
        kind: str,
        index: int | None = None,
    ) -> None:
        if source in instruction_by_key and target in instruction_by_key:
            edge_map.setdefault(source, []).append((target, kind, index))

    call_opcodes = {"call", "conditional", "fusion", "while"}
    for instruction in module.instructions:
        target = (instruction.computation, instruction.name)
        if instruction.opcode == "parameter":
            continue
        if instruction.opcode == "tuple":
            for index, operand_name in enumerate(instruction.operand_names):
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "tuple",
                    index,
                )
        elif instruction.opcode == "get-tuple-element":
            tuple_index_match = re.search(
                r"\bindex=([0-9]+)", instruction.raw_line
            )
            if instruction.operand_names and tuple_index_match is not None:
                add_edge(
                    (
                        instruction.computation,
                        instruction.operand_names[0],
                    ),
                    target,
                    "get-tuple-element",
                    int(tuple_index_match.group(1)),
                )
        elif instruction.opcode not in call_opcodes:
            for operand_name in instruction.operand_names:
                add_edge(
                    (instruction.computation, operand_name),
                    target,
                    "ordinary",
                )

        called_match = re.search(
            r"\bcalls=(%[A-Za-z0-9_.:-]+)", instruction.raw_line
        )
        if called_match is not None:
            called_symbol = called_match.group(1)
            for index, operand_name in enumerate(instruction.operand_names):
                parameter_key = parameters.get(called_symbol, {}).get(index)
                if parameter_key is not None:
                    add_edge(
                        (instruction.computation, operand_name),
                        parameter_key,
                        "identity",
                    )
            root_key = roots.get(called_symbol)
            if root_key is not None:
                add_edge(root_key, target, "identity")

        branches_match = re.search(
            r"\bbranch_computations=\{([^}]*)\}",
            instruction.raw_line,
        )
        if branches_match is not None:
            branch_symbols = tuple(
                item.strip()
                for item in branches_match.group(1).split(",")
                if item.strip()
            )
            for branch_index, branch_symbol in enumerate(branch_symbols):
                operand_index = branch_index + 1
                if operand_index >= len(instruction.operand_names):
                    continue
                parameter_key = parameters.get(branch_symbol, {}).get(0)
                if parameter_key is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[operand_index],
                        ),
                        parameter_key,
                        "identity",
                    )
                root_key = roots.get(branch_symbol)
                if root_key is not None:
                    add_edge(root_key, target, "identity")

        body_match = re.search(
            r"\bbody=(%[A-Za-z0-9_.:-]+)", instruction.raw_line
        )
        if body_match is not None and instruction.operand_names:
            body_symbol = body_match.group(1)
            body_parameter = parameters.get(body_symbol, {}).get(0)
            if body_parameter is not None:
                add_edge(
                    (
                        instruction.computation,
                        instruction.operand_names[0],
                    ),
                    body_parameter,
                    "identity",
                )
                add_edge(target, body_parameter, "identity")
            body_root = roots.get(body_symbol)
            if body_root is not None:
                add_edge(body_root, target, "identity")

    ValuePath = tuple[int, ...] | None

    def forward_dataflow(
        seeds: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], set[ValuePath]]:
        reached: dict[tuple[str, str], set[ValuePath]] = {}
        pending: list[tuple[tuple[str, str], ValuePath]] = []
        for seed in seeds:
            if seed in instruction_by_key:
                reached.setdefault(seed, set()).add(())
                pending.append((seed, ()))
        while pending:
            source, path = pending.pop()
            for target, kind, index in edge_map.get(source, ()):
                target_instruction = instruction_by_key[target]
                if kind == "identity":
                    target_path = path
                elif kind == "tuple":
                    assert index is not None
                    target_path = (
                        None
                        if path is None
                        else (index, *path)
                    )
                elif kind == "get-tuple-element":
                    assert index is not None
                    if path is None or path == ():
                        target_path = (
                            None
                            if len(target_instruction.result_shapes) > 1
                            else ()
                        )
                    elif path[0] != index:
                        continue
                    else:
                        target_path = path[1:]
                else:
                    target_path = (
                        None
                        if len(target_instruction.result_shapes) > 1
                        else ()
                    )
                target_paths = reached.setdefault(target, set())
                if target_path not in target_paths:
                    target_paths.add(target_path)
                    pending.append((target, target_path))
        return reached

    def instruction_key(instruction: HloInstruction) -> tuple[str, str]:
        return instruction.computation, instruction.name

    table_flow = forward_dataflow(
        tuple(instruction_key(item) for item in table_parameters)
    )
    table_dependent_lookups = tuple(
        instruction
        for instruction in module.instructions
        if instruction_key(instruction) in table_flow
        and instruction.opcode == "dynamic-slice"
        and instruction.op_name is not None
        and "greenfield_main_rope_table_lookup" in instruction.op_name
    )
    lookup_flow = forward_dataflow(
        tuple(instruction_key(item) for item in table_dependent_lookups)
    )

    def lookup_dependent(instruction: HloInstruction) -> bool:
        return instruction_key(instruction) in lookup_flow

    table_dependent_fp32_multiplies = tuple(
        item for item in scoped_fp32_multiplies if lookup_dependent(item)
    )
    table_dependent_fp32_combines = tuple(
        item for item in scoped_fp32_combines if lookup_dependent(item)
    )

    def is_final_round(instruction: HloInstruction) -> bool:
        return (
            instruction.opcode == "convert"
            and any(
                shape.dtype == "bf16" for shape in instruction.result_shapes
            )
            and any(
                shape.dtype == "f32" for shape in instruction.operand_shapes
            )
        )

    scoped_final_rounds = tuple(
        instruction for instruction in scoped if is_final_round(instruction)
    )
    scoped_final_round_keys = {
        (instruction.computation, instruction.name)
        for instruction in scoped_final_rounds
    }
    scoped_multiply_keys = {
        (instruction.computation, instruction.name)
        for instruction in scoped_fp32_multiplies
    }
    scoped_combine_keys = {
        (instruction.computation, instruction.name)
        for instruction in scoped_fp32_combines
    }
    combines_with_direct_fp32_product_operands = sum(
        len(instruction.operand_names) == 2
        and all(
            (instruction.computation, operand_name)
            in scoped_multiply_keys
            for operand_name in instruction.operand_names
        )
        for instruction in scoped_fp32_combines
    )
    premature_product_rounds = tuple(
        instruction.raw_line
        for instruction in module.instructions
        if is_final_round(instruction)
        and scoped_multiply_keys.intersection(
            (instruction.computation, operand_name)
            for operand_name in instruction.operand_names
        )
    )
    combine_users: dict[tuple[str, str], list[HloInstruction]] = {
        key: [] for key in scoped_combine_keys
    }
    direct_dataflow_round_combine_keys: set[tuple[str, str]] = set()
    direct_dataflow_final_rounds: list[HloInstruction] = []
    for instruction in module.instructions:
        consumed_combine_keys = scoped_combine_keys.intersection(
            (instruction.computation, operand_name)
            for operand_name in instruction.operand_names
        )
        if not consumed_combine_keys:
            continue
        for combine_key in consumed_combine_keys:
            combine_users[combine_key].append(instruction)
        if (
            is_final_round(instruction)
            and (instruction.computation, instruction.name)
            not in scoped_final_round_keys
        ):
            direct_dataflow_round_combine_keys.update(
                consumed_combine_keys
            )
            direct_dataflow_final_rounds.append(instruction)
    scoped_final_round_count = len(scoped_final_rounds)
    direct_dataflow_final_round_count = len(
        direct_dataflow_round_combine_keys
    )
    final_round_count = (
        scoped_final_round_count + direct_dataflow_final_round_count
    )
    final_rounds = tuple(
        (*scoped_final_rounds, *direct_dataflow_final_rounds)
    )
    table_dependent_final_rounds = tuple(
        item for item in final_rounds if lookup_dependent(item)
    )
    table_dependent_direct_round_combine_keys: set[tuple[str, str]] = set()
    for instruction in table_dependent_final_rounds:
        if instruction_key(instruction) in scoped_final_round_keys:
            continue
        table_dependent_direct_round_combine_keys.update(
            scoped_combine_keys.intersection(
                (instruction.computation, operand_name)
                for operand_name in instruction.operand_names
            )
        )
    table_dependent_final_round_count = sum(
        lookup_dependent(item) for item in scoped_final_rounds
    ) + len(table_dependent_direct_round_combine_keys)
    final_round_flow = forward_dataflow(
        tuple(instruction_key(item) for item in table_dependent_final_rounds)
    )
    entry_roots = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )

    def dependent_root_indices(
        flow: Mapping[tuple[str, str], set[ValuePath]],
    ) -> tuple[int, ...]:
        indices: set[int] = set()
        for root in entry_roots:
            for path in flow.get(instruction_key(root), ()):
                if path is None:
                    indices.update(required_entry_root_indices)
                elif path:
                    indices.add(path[0])
        return tuple(sorted(indices))

    table_dependent_root_indices = dependent_root_indices(table_flow)
    final_round_dependent_root_indices = dependent_root_indices(
        final_round_flow
    )
    combines_with_sole_convert_user = sum(
        len(users) == 1 and is_final_round(users[0])
        for users in combine_users.values()
    )
    violations = []
    if enabled:
        if len(table_parameters) != 1 or len(named_table_parameters) != 1:
            violations.append("main-RoPE table parameter contract drifted")
        if not scoped:
            violations.append("main-RoPE table scope is absent")
        if not table_dependent_lookups:
            violations.append(
                "main-RoPE table does not feed its position lookup"
            )
        if forbidden:
            violations.append("main-RoPE table scope contains forbidden operations")
        if bf16_arithmetic:
            violations.append("main-RoPE table scope contains BF16 arithmetic")
        if premature_product_rounds:
            violations.append("main-RoPE FP32 products are rounded before combine")
        if fp32_multiply_count < layers * 8:
            violations.append("main-RoPE FP32 multiply count is too small")
        if (
            len(table_dependent_fp32_multiplies)
            != fp32_multiply_count
        ):
            violations.append(
                "main-RoPE FP32 products do not depend on the table lookup"
            )
        if fp32_combine_count < layers * 4:
            violations.append("main-RoPE FP32 combine count is too small")
        if len(table_dependent_fp32_combines) != fp32_combine_count:
            violations.append(
                "main-RoPE FP32 combines do not depend on the table lookup"
            )
        if (
            combines_with_direct_fp32_product_operands
            != fp32_combine_count
        ):
            violations.append(
                "main-RoPE FP32 combines do not directly consume scoped products"
            )
        if final_round_count < layers * 2:
            violations.append("main-RoPE final-round count is too small")
        if table_dependent_final_round_count != final_round_count:
            violations.append(
                "main-RoPE final rounds do not depend on the table lookup"
            )
        missing_table_roots = sorted(
            set(required_entry_root_indices).difference(
                table_dependent_root_indices
            )
        )
        if missing_table_roots:
            violations.append(
                "main-RoPE captured roots do not depend on the table: "
                f"{missing_table_roots}"
            )
        missing_round_roots = sorted(
            set(required_entry_root_indices).difference(
                final_round_dependent_root_indices
            )
        )
        if missing_round_roots:
            violations.append(
                "main-RoPE captured roots do not depend on final BF16 rounds: "
                f"{missing_round_roots}"
            )
    elif table_parameters or named_table_parameters or scoped:
        violations.append("default decoder unexpectedly contains main-RoPE table state")
    return {
        "applicable": enabled,
        "bf16_arithmetic": list(bf16_arithmetic),
        "combines_with_direct_fp32_product_operands": (
            combines_with_direct_fp32_product_operands
        ),
        "combines_with_sole_convert_user": (
            combines_with_sole_convert_user
        ),
        "direct_dataflow_final_round_count": (
            direct_dataflow_final_round_count
        ),
        "expected_table_shape": list(table_shape),
        "final_round_count": final_round_count,
        "forbidden_instructions": list(forbidden),
        "fp32_combine_count": fp32_combine_count,
        "fp32_multiply_count": fp32_multiply_count,
        "named_table_parameter_count": len(named_table_parameters),
        "passed": not violations,
        "premature_product_rounds": list(premature_product_rounds),
        "scoped_instruction_count": len(scoped),
        "scoped_final_round_count": scoped_final_round_count,
        "table_dependent_final_round_count": (
            table_dependent_final_round_count
        ),
        "table_dependent_fp32_combine_count": len(
            table_dependent_fp32_combines
        ),
        "table_dependent_fp32_multiply_count": len(
            table_dependent_fp32_multiplies
        ),
        "table_dependent_lookup_count": len(table_dependent_lookups),
        "table_dependent_root_indices": list(
            table_dependent_root_indices
        ),
        "table_parameter_count": len(table_parameters),
        "final_round_dependent_root_indices": list(
            final_round_dependent_root_indices
        ),
        "required_entry_root_indices": list(required_entry_root_indices),
        "violations": violations,
    }


def _validate_pregathered_b512_attention_hlo(
    optimized_hlo: str,
    *,
    module: HloModule,
    config: DecoderStepConfig,
    layers: int,
    enabled: bool,
) -> dict[str, Any]:
    """Pin the DB537-selected LP4 exchange and block-512 attention body."""

    exchange_scope = "greenfield_selected_cache_lp4_exchange"
    attention_scope = "greenfield_pregathered_b512_attention"
    kernel_name = "greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"
    old_scopes = (
        "greenfield_owner_split_attention_output_gather",
        "greenfield_owner_split_attention_lse_gather",
        "greenfield_owner_split_attention_validity_gather",
        "greenfield_replicated_monolithic_attention_cache_gather",
        "greenfield_replicated_monolithic_attention",
    )
    kernel_identifier = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(kernel_name)}(?![A-Za-z0-9_])"
    )

    def in_scope(op_name: str | None, scope: str) -> bool:
        return bool(op_name) and scope in op_name.split("/")

    exchanges = tuple(
        collective
        for collective in module.collectives
        if in_scope(collective.op_name, exchange_scope)
    )
    scoped_attention_instructions = tuple(
        instruction
        for instruction in module.instructions
        if in_scope(instruction.op_name, attention_scope)
    )
    old_scope_instructions = tuple(
        instruction.name
        for instruction in module.instructions
        if any(in_scope(instruction.op_name, scope) for scope in old_scopes)
    )
    custom_calls = tuple(
        instruction
        for instruction in module.instructions
        if instruction.opcode == "custom-call"
        and 'custom_call_target="tpu_custom_call"' in instruction.raw_line
        and kernel_identifier.search(instruction.raw_line) is not None
        and in_scope(instruction.op_name, attention_scope)
    )
    violations: list[str] = []
    if not enabled:
        if exchanges or scoped_attention_instructions or custom_calls:
            violations.append(
                "default decoder unexpectedly contains pregathered-B512 attention"
            )
        return {
            "applicable": False,
            "attention_instruction_count": len(scoped_attention_instructions),
            "exchange_count": len(exchanges),
            "kernel_count": len(custom_calls),
            "passed": not violations,
            "violations": violations,
        }

    expected_cache_shape = HloShape(
        "bf16", (1, config.selected_width, config.packed_cache_width)
    )
    folded_cache_shape = HloShape(
        "bf16", (1, 4, config.selected_width // 4, config.packed_cache_width)
    )
    if (
        config.local_parallel_size != 4
        or config.selected_width != 2048
        or config.packed_cache_width != 640
    ):
        violations.append("pregathered-B512 HLO geometry is not PP8 LP4 GLM-5.2")
    if len(exchanges) != layers:
        violations.append(
            "pregathered-B512 selected-cache exchange count drifted: "
            f"expected={layers} observed={len(exchanges)}"
        )
    malformed_exchanges = tuple(
        collective.name
        for collective in exchanges
        if collective.opcode != "all-reduce"
        or not (
            expected_cache_shape in collective.operand_shapes
            and expected_cache_shape in collective.result_shapes
        )
        and not (
            folded_cache_shape in collective.operand_shapes
            and folded_cache_shape in collective.result_shapes
        )
    )
    if malformed_exchanges:
        violations.append(
            "pregathered-B512 exchange lost exact BF16 selected-cache shape: "
            f"{malformed_exchanges}"
        )
    if len(custom_calls) != layers:
        violations.append(
            "pregathered-B512 Pallas kernel count drifted: "
            f"expected={layers} observed={len(custom_calls)}"
        )
    malformed_calls = tuple(
        instruction.name
        for instruction in custom_calls
        if instruction.operand_shapes
        not in (
            (
                HloShape("s32", (1,)),
                HloShape("bf16", (1, 16, 640)),
                expected_cache_shape,
            ),
            (
                HloShape("s32", (1,)),
                HloShape("bf16", (1, 16, 640)),
                folded_cache_shape,
            ),
        )
        or instruction.result_shapes != (HloShape("bf16", (1, 16, 512)),)
    )
    if malformed_calls:
        violations.append(
            "pregathered-B512 Pallas calls lost exact selected-cache/output shapes"
        )

    instruction_by_key = {
        (instruction.computation, instruction.name): instruction
        for instruction in module.instructions
    }
    exchange_keys = {
        (instruction.computation, instruction.name) for instruction in exchanges
    }
    cache_shapes = {expected_cache_shape, folded_cache_shape}
    allowed_cache_transforms = {"bitcast", "copy", "reshape"}
    kernel_exchange_links: list[dict[str, str]] = []
    cache_dataflow_failures: list[str] = []
    for custom_call in custom_calls:
        if len(custom_call.operand_names) < 3:
            cache_dataflow_failures.append(
                f"{custom_call.name}: missing cache operand"
            )
            continue
        current = (custom_call.computation, custom_call.operand_names[2])
        visited: set[tuple[str, str]] = set()
        while current not in exchange_keys:
            if current in visited:
                cache_dataflow_failures.append(
                    f"{custom_call.name}: cache dataflow contains a cycle"
                )
                break
            visited.add(current)
            source = instruction_by_key.get(current)
            if source is None:
                cache_dataflow_failures.append(
                    f"{custom_call.name}: cache operand source is absent"
                )
                break
            if (
                source.opcode not in allowed_cache_transforms
                or len(source.operand_names) != 1
                or len(source.operand_shapes) != 1
                or len(source.result_shapes) != 1
                or source.operand_shapes[0] not in cache_shapes
                or source.result_shapes[0] not in cache_shapes
                or source.operand_shapes[0].element_count
                != source.result_shapes[0].element_count
            ):
                cache_dataflow_failures.append(
                    f"{custom_call.name}: cache operand bypasses the scoped exchange"
                )
                break
            current = (source.computation, source.operand_names[0])
        else:
            kernel_exchange_links.append(
                {
                    "computation": custom_call.computation,
                    "exchange": current[1],
                    "kernel": custom_call.name,
                }
            )
    if cache_dataflow_failures:
        violations.append(
            "pregathered-B512 kernel cache operands do not depend on their "
            "scoped exchanges: "
            f"{tuple(cache_dataflow_failures)}"
        )
    linked_exchange_counts = Counter(
        (item["computation"], item["exchange"])
        for item in kernel_exchange_links
    )
    exchange_bijection = (
        len(kernel_exchange_links) == len(custom_calls)
        and set(linked_exchange_counts) == exchange_keys
        and all(count == 1 for count in linked_exchange_counts.values())
    )
    if not exchange_bijection:
        violations.append(
            "pregathered-B512 exchanges and kernel cache operands are not bijective"
        )
    if not scoped_attention_instructions:
        violations.append("pregathered-B512 attention scope is absent")
    if old_scope_instructions:
        violations.append(
            "pregathered-B512 decoder retained an old attention exchange path: "
            f"{old_scope_instructions[:8]}"
        )
    return {
        "applicable": True,
        "attention_instruction_count": len(scoped_attention_instructions),
        "cache_dataflow_failures": cache_dataflow_failures,
        "exchange_count": len(exchanges),
        "exchange_bijection": exchange_bijection,
        "exchange_names": [item.name for item in exchanges],
        "exchange_shapes": [
            [shape.to_dict() for shape in item.result_shapes]
            for item in exchanges
        ],
        "expected_exchange_count": layers,
        "expected_exchange_shape": "bf16[1,2048,640]",
        "kernel_count": len(custom_calls),
        "kernel_exchange_links": kernel_exchange_links,
        "kernel_name": kernel_name,
        "old_scope_instruction_count": len(old_scope_instructions),
        "passed": not violations,
        "violations": violations,
    }


class _HloValueFlow:
    """Track HLO value provenance across tuples and called computations."""

    def __init__(self, module: HloModule) -> None:
        self._instructions = {
            (instruction.computation, instruction.name): instruction
            for instruction in module.instructions
        }
        parameters: dict[str, dict[int, tuple[str, str]]] = {}
        roots: dict[str, tuple[str, str]] = {}

        def computation_symbol(value: str) -> str:
            parts = value.split()
            if parts and parts[0] == "ENTRY":
                parts = parts[1:]
            return parts[0] if parts else value

        for instruction in module.instructions:
            symbol = computation_symbol(instruction.computation)
            if instruction.opcode == "parameter":
                match = re.search(
                    r"\bparameter\(([0-9]+)\)", instruction.raw_line
                )
                if match is not None:
                    parameters.setdefault(symbol, {})[int(match.group(1))] = (
                        instruction.computation,
                        instruction.name,
                    )
            if instruction.raw_line.lstrip().startswith("ROOT "):
                roots[symbol] = (instruction.computation, instruction.name)

        self._edges: dict[
            tuple[str, str],
            list[tuple[tuple[str, str], str, int | None]],
        ] = {}
        self._reverse_edges: dict[
            tuple[str, str], set[tuple[str, str]]
        ] = {}

        def add_edge(
            source: tuple[str, str],
            target: tuple[str, str],
            kind: str,
            index: int | None = None,
        ) -> None:
            if source in self._instructions and target in self._instructions:
                self._edges.setdefault(source, []).append(
                    (target, kind, index)
                )
                self._reverse_edges.setdefault(target, set()).add(source)

        call_opcodes = {"call", "conditional", "fusion", "while"}
        for instruction in module.instructions:
            target = (instruction.computation, instruction.name)
            if instruction.opcode == "parameter":
                continue
            if instruction.opcode == "tuple":
                for index, operand_name in enumerate(
                    instruction.operand_names
                ):
                    add_edge(
                        (instruction.computation, operand_name),
                        target,
                        "tuple",
                        index,
                    )
            elif instruction.opcode == "get-tuple-element":
                match = re.search(r"\bindex=([0-9]+)", instruction.raw_line)
                if instruction.operand_names and match is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[0],
                        ),
                        target,
                        "get-tuple-element",
                        int(match.group(1)),
                    )
            elif instruction.opcode not in call_opcodes:
                for operand_name in instruction.operand_names:
                    add_edge(
                        (instruction.computation, operand_name),
                        target,
                        "ordinary",
                    )

            called = re.search(
                r"\bcalls=(%[A-Za-z0-9_.:-]+)", instruction.raw_line
            )
            if called is not None:
                symbol = called.group(1)
                for index, operand_name in enumerate(
                    instruction.operand_names
                ):
                    parameter = parameters.get(symbol, {}).get(index)
                    if parameter is not None:
                        add_edge(
                            (instruction.computation, operand_name),
                            parameter,
                            "identity",
                        )
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")

            branches = re.search(
                r"\bbranch_computations=\{([^}]*)\}",
                instruction.raw_line,
            )
            if branches is not None:
                symbols = tuple(
                    item.strip()
                    for item in branches.group(1).split(",")
                    if item.strip()
                )
                for branch_index, symbol in enumerate(symbols):
                    operand_index = branch_index + 1
                    if operand_index >= len(instruction.operand_names):
                        continue
                    parameter = parameters.get(symbol, {}).get(0)
                    if parameter is not None:
                        add_edge(
                            (
                                instruction.computation,
                                instruction.operand_names[operand_index],
                            ),
                            parameter,
                            "identity",
                        )
                    root = roots.get(symbol)
                    if root is not None:
                        add_edge(root, target, "identity")

            body = re.search(
                r"\bbody=(%[A-Za-z0-9_.:-]+)", instruction.raw_line
            )
            if body is not None and instruction.operand_names:
                symbol = body.group(1)
                parameter = parameters.get(symbol, {}).get(0)
                if parameter is not None:
                    add_edge(
                        (
                            instruction.computation,
                            instruction.operand_names[0],
                        ),
                        parameter,
                        "identity",
                    )
                    add_edge(target, parameter, "identity")
                root = roots.get(symbol)
                if root is not None:
                    add_edge(root, target, "identity")

    def provenance(
        self,
        seeds: Sequence[tuple[str, str]],
        *,
        allowed_target_opcodes: frozenset[str] | None = None,
    ) -> dict[tuple[str, str], int]:
        """Return one bit per seed at every reachable HLO value."""

        ValuePath = tuple[int, ...] | None
        reached: dict[
            tuple[str, str], dict[ValuePath, int]
        ] = {}
        pending: list[tuple[tuple[str, str], ValuePath, int]] = []
        for index, seed in enumerate(seeds):
            if seed not in self._instructions:
                continue
            bit = 1 << index
            reached.setdefault(seed, {})[()] = bit
            pending.append((seed, (), bit))
        while pending:
            source, path, bits = pending.pop()
            for target, kind, index in self._edges.get(source, ()):
                target_instruction = self._instructions[target]
                if (
                    allowed_target_opcodes is not None
                    and target_instruction.opcode
                    not in allowed_target_opcodes
                ):
                    continue
                if kind == "identity":
                    target_path = path
                elif kind == "tuple":
                    assert index is not None
                    target_path = (
                        None if path is None else (index, *path)
                    )
                elif kind == "get-tuple-element":
                    assert index is not None
                    if path is None or path == ():
                        target_path = (
                            None
                            if len(target_instruction.result_shapes) > 1
                            else ()
                        )
                    elif path[0] != index:
                        continue
                    else:
                        target_path = path[1:]
                else:
                    target_path = (
                        None
                        if len(target_instruction.result_shapes) > 1
                        else ()
                    )
                target_paths = reached.setdefault(target, {})
                old_bits = target_paths.get(target_path, 0)
                new_bits = old_bits | bits
                if new_bits != old_bits:
                    target_paths[target_path] = new_bits
                    pending.append((target, target_path, new_bits))
        merged: dict[tuple[str, str], int] = {}
        for key, path_bits in reached.items():
            bits = 0
            for value in path_bits.values():
                bits |= value
            merged[key] = bits
        return merged

    def exclusive_sources(
        self,
        targets: Sequence[tuple[str, str]],
        *,
        permitted_sources: frozenset[tuple[str, str]],
        allowed_opcodes: frozenset[str],
    ) -> tuple[frozenset[tuple[str, str]], tuple[str, ...]]:
        """Trace targets backward and reject every non-permitted leaf/opcode."""

        reached_sources: set[tuple[str, str]] = set()
        violations: list[str] = []
        pending = list(targets)
        visited: set[tuple[str, str]] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in permitted_sources:
                reached_sources.add(current)
                continue
            instruction = self._instructions.get(current)
            if instruction is None:
                violations.append(f"missing HLO value {current}")
                continue
            if instruction.opcode not in allowed_opcodes:
                violations.append(
                    f"{current[1]} uses non-shape opcode {instruction.opcode}"
                )
                continue
            sources = self._reverse_edges.get(current, set())
            if not sources:
                violations.append(
                    f"{current[1]} reaches an unapproved {instruction.opcode} leaf"
                )
                continue
            pending.extend(sources)
        return frozenset(reached_sources), tuple(sorted(set(violations)))


def _validate_strategy_nd_attention_projection_hlo(
    module: HloModule,
    *,
    layers: int,
    enabled: bool,
) -> dict[str, Any]:
    """Pin DB533's exact local replay for every attention projection."""

    scope = "greenfield_strategy_nd_row0_attention_output"
    gather_scope = "greenfield_strategy_nd_row0_association_gather"
    kernel_name = "greenfield_fp8_strategy_nd_o_m8_k512_n6144"

    def in_scope(instruction: HloInstruction, name: str) -> bool:
        return bool(instruction.op_name) and name in (
            instruction.op_name or ""
        ).split("/")

    scoped = tuple(
        instruction
        for instruction in module.instructions
        if in_scope(instruction, scope)
    )
    gathers = tuple(
        instruction
        for instruction in module.collectives
        if in_scope(instruction, scope)
        and in_scope(instruction, gather_scope)
    )
    kernel_identifier = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(kernel_name)}(?![A-Za-z0-9_])"
    )
    kernels = tuple(
        instruction
        for instruction in module.instructions
        if instruction.opcode == "custom-call"
        and 'custom_call_target="tpu_custom_call"' in instruction.raw_line
        and kernel_identifier.search(instruction.raw_line) is not None
    )
    violations: list[str] = []
    if not enabled:
        if scoped or gathers or kernels:
            violations.append(
                "default decoder unexpectedly contains StrategyND attention projection"
            )
        return {
            "applicable": False,
            "gather_count": len(gathers),
            "kernel_count": len(kernels),
            "passed": not violations,
            "scoped_instruction_count": len(scoped),
            "violations": violations,
        }

    expected_kernel_count = 8 * layers
    if len(gathers) != layers:
        violations.append(
            "StrategyND attention gather count drifted: "
            f"expected={layers} observed={len(gathers)}"
        )
    if len(kernels) != expected_kernel_count:
        violations.append(
            "StrategyND attention partial count drifted: "
            f"expected={expected_kernel_count} observed={len(kernels)}"
        )
    logical_operand = HloShape("bf16", (8, 1, 6144))
    logical_result = HloShape("bf16", (4, 8, 1, 6144))
    folded_result = HloShape("bf16", (32, 1, 6144))
    malformed_gathers = tuple(
        instruction.name
        for instruction in gathers
        if instruction.opcode != "all-gather"
        or instruction.operand_shapes != (logical_operand,)
        or instruction.result_shapes
        not in ((logical_result,), (folded_result,))
        or "dimensions={0}" not in instruction.raw_line
        or not instruction.use_global_device_ids
    )
    if malformed_gathers:
        violations.append(
            "StrategyND attention gathers lost exact LP4 geometry: "
            f"{malformed_gathers}"
        )
    logical_call = (
        (
            HloShape("bf16", (1, 512)),
            HloShape("u8", (6144, 512)),
            HloShape("f32", (48, 4)),
        ),
        (HloShape("bf16", (1, 6144)),),
    )
    folded_call = (
        (
            HloShape("bf16", (8, 512)),
            HloShape("u8", (6144, 512)),
            HloShape("f32", (8, 128)),
        ),
        (HloShape("bf16", (8, 6144)),),
    )
    malformed_kernels = tuple(
        instruction.name
        for instruction in kernels
        if (instruction.operand_shapes, instruction.result_shapes)
        not in (logical_call, folded_call)
    )
    if malformed_kernels:
        violations.append(
            "StrategyND attention partials lost exact K512 geometry: "
            f"{malformed_kernels[:8]}"
        )

    flow = _HloValueFlow(module)
    kernel_keys = tuple(
        (instruction.computation, instruction.name)
        for instruction in kernels
    )
    gather_keys = tuple(
        (instruction.computation, instruction.name)
        for instruction in gathers
    )
    shape_only_opcodes = frozenset(
        {
            "all-gather",
            "bitcast",
            "concatenate",
            "copy",
            "fusion",
            "get-tuple-element",
            "optimization-barrier",
            "parameter",
            "reshape",
            "slice",
            "tuple",
        }
    )
    kernel_flow = flow.provenance(
        kernel_keys,
        allowed_target_opcodes=shape_only_opcodes,
    )
    gather_source_bits = {
        key: kernel_flow.get(key, 0) for key in gather_keys
    }
    malformed_lineage = tuple(
        key[1]
        for key, bits in gather_source_bits.items()
        if bits.bit_count() != 8
    )
    if malformed_lineage:
        violations.append(
            "StrategyND attention gathers do not consume exactly eight K512 "
            f"partials: {malformed_lineage}"
        )
    kernel_gather_counts = [
        sum(bool(bits & (1 << index)) for bits in gather_source_bits.values())
        for index in range(len(kernel_keys))
    ]
    partial_gather_bijection = (
        len(gather_source_bits) == layers
        and len(kernel_gather_counts) == expected_kernel_count
        and all(count == 1 for count in kernel_gather_counts)
        and not malformed_lineage
    )
    if not partial_gather_bijection:
        violations.append(
            "StrategyND K512 partials and attention gathers are not bijective"
        )
    gather_operands = tuple(
        (instruction.computation, instruction.operand_names[0])
        for instruction in gathers
        if len(instruction.operand_names) == 1
    )
    exclusive_sources, exclusive_source_violations = flow.exclusive_sources(
        gather_operands,
        permitted_sources=frozenset(kernel_keys),
        allowed_opcodes=shape_only_opcodes - {"all-gather"},
    )
    exclusive_partial_dataflow = (
        len(gather_operands) == len(gathers)
        and exclusive_sources == frozenset(kernel_keys)
        and not exclusive_source_violations
    )
    if not exclusive_partial_dataflow:
        violations.append(
            "StrategyND gather operands are not exclusively assembled from "
            "the declared K512 partials: "
            f"{exclusive_source_violations}"
        )

    gather_flow = flow.provenance(gather_keys)
    entry_roots = tuple(
        (instruction.computation, instruction.name)
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )
    live_gather_bits = 0
    for root in entry_roots:
        live_gather_bits |= gather_flow.get(root, 0)
    expected_live_bits = (1 << len(gather_keys)) - 1
    all_gathers_live = (
        len(entry_roots) == 1 and live_gather_bits == expected_live_bits
    )
    if not all_gathers_live:
        violations.append(
            "StrategyND attention gather output is not live at the decoder root"
        )
    return {
        "all_gathers_live": all_gathers_live,
        "applicable": True,
        "expected_gather_count": layers,
        "expected_kernel_count": expected_kernel_count,
        "exclusive_partial_dataflow": exclusive_partial_dataflow,
        "exclusive_source_violations": exclusive_source_violations,
        "gather_count": len(gathers),
        "gather_names": [instruction.name for instruction in gathers],
        "gather_source_counts": {
            key[1]: bits.bit_count()
            for key, bits in gather_source_bits.items()
        },
        "kernel_count": len(kernels),
        "kernel_name": kernel_name,
        "partial_gather_bijection": partial_gather_bijection,
        "passed": not violations,
        "scoped_instruction_count": len(scoped),
        "violations": violations,
    }


def _expected_tpu_decoder_reductions(
    *,
    layers: int,
    dense_layers: int,
    sparse_layers: int,
    pregathered_b512_attention: bool,
    feature_reconstruct_down_fp32: bool,
    complete_token_path: bool,
    split_residual_state: bool,
    token_observation_candidates: int,
    local_parallel_size: int = 4,
    strategy_nd_attention_projection: bool = False,
    dense_final_layout_convolution: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return the protected GLM TPU all-reduce arities and logical shapes."""

    dense_reductions = 0 if dense_final_layout_convolution else dense_layers
    residual_reductions = dense_reductions + (
        0 if strategy_nd_attention_projection else layers
    )
    result_shapes = {"bf16[2,1,6144]": sparse_layers}
    if residual_reductions:
        result_shapes["bf16[1,6144]"] = residual_reductions
    if pregathered_b512_attention:
        # The selected-cache path removes all three owner-split result/LSE/
        # validity gathers. TPU XLA had rewritten the latter two families to
        # 78 f32[256] and 78 u32[1,1,128] all-reduce components, including
        # their tuple launch fusion. The remaining attention-output and MLP
        # reductions are single-result, as is the new selected-cache sum.
        arities = {
            "1": 2 * layers - (
                dense_layers if dense_final_layout_convolution else 0
            ) + (
                0 if strategy_nd_attention_projection else layers
            )
        }
        result_shapes["bf16[1,2048,640]"] = layers
    else:
        # The accepted full-decoder lowering launch-fuses the 312 logical
        # attention/MLP components into a plan-specific physical arity
        # histogram.  LP2 exposes four more singleton launches than LP4 while
        # preserving all 312 logical components; this is the exact lowering
        # acquired from the sealed PP16 final-layout schedule.
        if local_parallel_size == 4:
            arities = {"1": 277, "2": 16, "3": 1}
        elif local_parallel_size == 2:
            arities = {"1": 285, "2": 12, "3": 1}
        else:
            raise PlanValidationError(
                "TPU decoder reduction contract requires LP2 or LP4"
            )
        attention_lse_width = 64 * local_parallel_size
        result_shapes.update(
            {
                f"f32[{attention_lse_width}]": layers,
                "u32[1,1,128]": layers,
            }
        )
    if feature_reconstruct_down_fp32:
        arities["1"] += sparse_layers
        result_shapes["f32[8,6144]"] = sparse_layers
    if complete_token_path:
        arities["1"] += 3
        embedding_shape = (
            "bf16[1,1,6144]" if split_residual_state else "bf16[1,6144]"
        )
        result_shapes[embedding_shape] = result_shapes.get(embedding_shape, 0) + 1
        if token_observation_candidates == 1:
            lane_shape = str(local_parallel_size)
            result_shapes.update(
                {f"bf16[{lane_shape}]": 1, f"s32[{lane_shape}]": 1}
            )
    return arities, result_shapes


def validate_decoder_step_hlo(
    optimized_hlo: str,
    *,
    stablehlo: str | None = None,
    config: DecoderStepConfig,
    schedule: PipelineSchedule,
    groups: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
    backend_contract: str,
    feature_output_tile: int | None = None,
    feature_fuse_route_weighting: bool = False,
    feature_reconstruct_down_fp32: bool = False,
    dsa_query_backend: StageLinearBackend | None = None,
    dsa_query_exact_association: bool = False,
    dsa_head_key_exact_association: bool = False,
    attention_projection_backend: AttentionProjectionBackend = "separate",
    complete_token_path: bool = False,
    token_observation_candidates: int = 1,
    split_residual_state: bool = False,
    prefill_index_repair: bool = False,
    main_rope_table_enabled: bool = False,
    pregathered_b512_attention: bool = False,
    strategy_nd_attention_projection: bool = False,
    dense_final_layout_convolution: bool = False,
) -> dict[str, Any]:
    """Reject non-local collectives, count drift, and dead batch rows."""

    if backend_contract not in {"cpu_reference", *_TPU_DECODER_BACKEND_CONTRACTS}:
        raise PlanValidationError("decoder HLO backend contract is unknown")
    if feature_output_tile is None:
        feature_output_tile = (
            PROMOTED_FEATURE_OUTPUT_TILE
            if backend_contract in _FEATURE_DECODER_BACKEND_CONTRACTS
            else REFERENCE_FEATURE_OUTPUT_TILE
        )
    if feature_output_tile not in (128, 256):
        raise PlanValidationError("feature output tile must be 128 or 256")
    if not isinstance(feature_fuse_route_weighting, bool):
        raise PlanValidationError("feature route-weight fusion flag must be boolean")
    if not isinstance(feature_reconstruct_down_fp32, bool):
        raise PlanValidationError(
            "feature FP32 reconstruction flag must be boolean"
        )
    if not isinstance(prefill_index_repair, bool):
        raise PlanValidationError("prefill index-repair flag must be boolean")
    if not isinstance(main_rope_table_enabled, bool):
        raise PlanValidationError("main-RoPE table HLO flag must be boolean")
    if not isinstance(pregathered_b512_attention, bool):
        raise PlanValidationError(
            "pregathered-B512 attention HLO flag must be boolean"
        )
    if not isinstance(strategy_nd_attention_projection, bool):
        raise PlanValidationError(
            "StrategyND attention-projection HLO flag must be boolean"
        )
    if not isinstance(dense_final_layout_convolution, bool):
        raise PlanValidationError(
            "dense final-layout convolution HLO flag must be boolean"
        )
    if strategy_nd_attention_projection and not pregathered_b512_attention:
        raise PlanValidationError(
            "StrategyND attention-projection HLO requires pregathered-B512"
        )
    if pregathered_b512_attention and backend_contract != (
        "tpu_v4_pp8_pallas_feature_linear"
    ):
        raise PlanValidationError(
            "pregathered-B512 attention requires the PP8 Pallas-linear HLO contract"
        )
    if feature_reconstruct_down_fp32 and feature_fuse_route_weighting:
        raise PlanValidationError(
            "feature FP32 reconstruction is incompatible with fused route "
            "weighting"
        )
    if dsa_query_backend is None:
        dsa_query_backend = (
            "pallas"
            if backend_contract == "tpu_v4_pp8_pallas_feature_linear"
            else "reference"
        )
    if dsa_query_backend not in ("reference", "pallas"):
        raise PlanValidationError("DSA query HLO backend is unknown")
    if not isinstance(dsa_query_exact_association, bool):
        raise PlanValidationError(
            "exact DSA query association flag must be boolean"
        )
    if dsa_query_exact_association and dsa_query_backend != "reference":
        raise PlanValidationError(
            "exact DSA query association requires the reference backend"
        )
    if not isinstance(dsa_head_key_exact_association, bool):
        raise PlanValidationError(
            "decoder exact DSA head/key association flag must be boolean"
        )
    if dsa_head_key_exact_association and not dsa_query_exact_association:
        raise PlanValidationError(
            "exact DSA head/key association requires exact DSA query"
        )
    if (
        config.dsa_score_default_precision
        and not dsa_head_key_exact_association
    ):
        raise PlanValidationError(
            "default DSA score precision requires exact DSA head/key inputs"
        )
    if attention_projection_backend not in (
        "separate",
        "fused_n82_convolution",
    ):
        raise PlanValidationError(
            "attention projection HLO backend is unknown"
        )
    if not isinstance(complete_token_path, bool):
        raise PlanValidationError("complete token-path flag must be boolean")
    if not isinstance(split_residual_state, bool):
        raise PlanValidationError("split residual-state flag must be boolean")
    if (
        not isinstance(token_observation_candidates, int)
        or isinstance(token_observation_candidates, bool)
        or not 1 <= token_observation_candidates <= 64
    ):
        raise PlanValidationError(
            "token observation candidate width must be in 1..64"
        )
    if token_observation_candidates != 1 and not complete_token_path:
        raise PlanValidationError(
            "token observation candidates require the complete token path"
        )
    if (
        backend_contract not in _FEATURE_DECODER_BACKEND_CONTRACTS
        and feature_output_tile != 128
    ):
        raise PlanValidationError(
            "a non-default feature output tile requires a feature backend"
        )
    if (
        feature_fuse_route_weighting
        and backend_contract not in _FEATURE_DECODER_BACKEND_CONTRACTS
    ):
        raise PlanValidationError(
            "feature route-weight fusion requires a feature backend"
        )
    if (
        feature_reconstruct_down_fp32
        and backend_contract not in _FEATURE_DECODER_BACKEND_CONTRACTS
    ):
        raise PlanValidationError(
            "feature FP32 reconstruction requires a feature backend"
        )
    skeleton = PipelineSkeletonConfig(
        config.stage_count,
        config.local_parallel_size,
        config.hidden_size,
        config.selected_width,
    )
    canonical_groups = _canonical_groups(groups, skeleton)
    canonical_pairs = _canonical_pairs(pairs, canonical_groups, skeleton)
    module = parse_hlo_module(optimized_hlo)
    collectives = module.collectives
    by_opcode: dict[str, list[Any]] = {}
    for collective in collectives:
        by_opcode.setdefault(collective.opcode, []).append(collective)
    full_layers = sum(
        layer.indexer_kind == "full"
        for stage in schedule.stages
        for layer in stage.layers
    )
    layers = schedule.layer_count
    dense_layers = sum(
        layer.mlp_kind == "dense"
        for stage in schedule.stages
        for layer in stage.layers
    )
    reduction_arity_counts: dict[str, int] = {}
    expected_reduction_arity_counts: dict[str, int] = {}
    reduction_result_shape_counts: dict[str, int] = {}
    expected_reduction_result_shape_counts: dict[str, int] = {}
    reduction_component_count = 0
    expected_reduction_component_count = 0
    sparse_layers = layers - dense_layers
    if backend_contract == "cpu_reference":
        expected_gathers = 4 * layers + 3 * full_layers + (
            2 if complete_token_path else 0
        )
        expected_reductions = 2 * layers + (
            1 if complete_token_path else 0
        )
    else:
        expected_pipeline_geometry = {
            "tpu_v4_pp8_reference": (8, 4),
            "tpu_v4_pp8_pallas_feature": (8, 4),
            "tpu_v4_pp8_pallas_feature_linear": (8, 4),
            "tpu_v4_pp16_reference": (16, 2),
            "tpu_v4_pp16_pallas_feature": (16, 2),
        }[backend_contract]
        if (
            (config.stage_count, config.local_parallel_size),
            config.hidden_size,
            config.selected_width,
            layers,
            dense_layers,
            sparse_layers,
        ) != (expected_pipeline_geometry, 6144, 2048, 78, 3, 75):
            raise PlanValidationError(
                f"the {backend_contract} lowering contract is pinned to its "
                "complete 78-layer GLM-5.2 pipeline geometry"
            )

        # Gate C and complete-decoder HLO establish both semantic result shapes
        # and exact physical launch arities. Pin both views: checking only one
        # would admit a missing logical result or a launch-fusion regression.
        # TPU XLA lowers each four-element top-1 all-gather to a one-hot local
        # all-reduce.  The complete path therefore adds no physical gather,
        # but adds embedding plus score/id singleton reductions.
        expected_gathers = (
            (0 if pregathered_b512_attention else 2 * layers)
            + (layers if strategy_nd_attention_projection else 0)
            + (dense_layers if dense_final_layout_convolution else 0)
            + 3 * full_layers
        )
        (
            expected_reduction_arity_counts,
            expected_reduction_result_shape_counts,
        ) = _expected_tpu_decoder_reductions(
            layers=layers,
            dense_layers=dense_layers,
            sparse_layers=sparse_layers,
            pregathered_b512_attention=pregathered_b512_attention,
            strategy_nd_attention_projection=(
                strategy_nd_attention_projection
            ),
            dense_final_layout_convolution=(
                dense_final_layout_convolution
            ),
            feature_reconstruct_down_fp32=feature_reconstruct_down_fp32,
            complete_token_path=complete_token_path,
            split_residual_state=split_residual_state,
            token_observation_candidates=token_observation_candidates,
            local_parallel_size=config.local_parallel_size,
        )
        expected_reductions = sum(expected_reduction_arity_counts.values())
        expected_reduction_component_count = sum(
            int(arity) * count
            for arity, count in expected_reduction_arity_counts.items()
        )
        reductions = by_opcode.get("all-reduce", ())
        reduction_arity_counts = {
            str(arity): count
            for arity, count in sorted(
                Counter(len(item.result_shapes) for item in reductions).items()
            )
        }
        def reduction_shape_signature(
            item: HloInstruction, shape: HloShape
        ) -> str:
            if (
                pregathered_b512_attention
                and item.op_name
                and "greenfield_selected_cache_lp4_exchange"
                in item.op_name.split("/")
                and shape == HloShape("bf16", (1, 4, 512, 640))
            ):
                # TPU may preserve the Pallas B512 block fold around the
                # all-reduce.  It is the same 2,048x640 logical segment and
                # is accepted only in the exact named exchange scope.
                return "bf16[1,2048,640]"
            return (
                f"{shape.dtype}["
                + ",".join(str(dimension) for dimension in shape.dimensions)
                + "]"
            )

        result_shapes = Counter(
            reduction_shape_signature(item, shape)
            for item in reductions
            for shape in item.result_shapes
        )
        reduction_result_shape_counts = dict(sorted(result_shapes.items()))
        reduction_component_count = sum(result_shapes.values())
        if complete_token_path and token_observation_candidates != 1:
            for dtype in ("bf16", "s32"):
                candidate_shapes = [
                    shape
                    for item in reductions
                    for shape in item.result_shapes
                    if shape.dtype == dtype
                    and prod(shape.dimensions)
                    == config.local_parallel_size * token_observation_candidates
                ]
                if len(candidate_shapes) == 1:
                    shape = candidate_shapes[0]
                    signature = (
                        f"{shape.dtype}["
                        + ",".join(
                            str(dimension) for dimension in shape.dimensions
                        )
                        + "]"
                    )
                    expected_reduction_result_shape_counts[signature] = 1
    expected_permutes = 2 * config.stage_count + (
        1 if complete_token_path else 0
    )
    violations = []
    observed_counts = {
        opcode: len(values) for opcode, values in sorted(by_opcode.items())
    }
    expected_counts = {
        "all-gather": expected_gathers,
        "all-reduce": expected_reductions,
        "collective-permute": expected_permutes,
    }
    if observed_counts != expected_counts:
        violations.append(
            f"decoder collective counts drifted: expected={expected_counts} "
            f"observed={observed_counts}"
        )
    if backend_contract in _TPU_DECODER_BACKEND_CONTRACTS:
        if reduction_arity_counts != expected_reduction_arity_counts:
            violations.append(
                "decoder physical all-reduce tuple arities drifted: "
                f"expected={expected_reduction_arity_counts} "
                f"observed={reduction_arity_counts}"
            )
        if reduction_component_count != expected_reduction_component_count:
            violations.append(
                "decoder logical all-reduce component count drifted: "
                f"expected={expected_reduction_component_count} "
                f"observed={reduction_component_count}"
            )
        if (
            reduction_result_shape_counts
            != expected_reduction_result_shape_counts
        ):
            violations.append(
                "decoder all-reduce logical result shapes drifted: "
                f"expected={expected_reduction_result_shape_counts} "
                f"observed={reduction_result_shape_counts}"
            )
    expected_groups = tuple(tuple(group) for group in canonical_groups)
    for collective in (
        *by_opcode.get("all-gather", ()),
        *by_opcode.get("all-reduce", ()),
    ):
        if collective.replica_groups != expected_groups:
            violations.append(
                f"decoder collective {collective.name} escaped local groups"
            )
        if collective.maximum_group_size != config.local_parallel_size:
            violations.append(
                f"decoder collective {collective.name} has non-local width"
            )
    expected_pair_multiset = sorted(canonical_pairs * expected_permutes)
    observed_pair_multiset = sorted(
        pair
        for collective in by_opcode.get("collective-permute", ())
        for pair in collective.source_target_pairs
    )
    if observed_pair_multiset != expected_pair_multiset:
        violations.append("decoder transport pairs/counts drifted")
    residual_transport_dimensions = (
        (2, 1, config.hidden_size)
        if split_residual_state
        else (1, config.hidden_size)
    )
    residual_transport_dtype = (
        "f32" if backend_contract == "cpu_reference" else "bf16"
    )
    residual_transports = [
        collective
        for collective in by_opcode.get("collective-permute", ())
        if any(
            shape.dtype == residual_transport_dtype
            and shape.dimensions == residual_transport_dimensions
            for shape in collective.operand_shapes
        )
    ]
    if len(residual_transports) != config.stage_count:
        violations.append(
            "decoder residual transport count/shape drifted: expected "
            f"{config.stage_count} {residual_transport_dtype}"
            f"{residual_transport_dimensions}, "
            f"found {len(residual_transports)}"
        )
    complete_token_collective_contract: dict[str, Any] = {
        "applicable": False,
        "passed": True,
        "violations": [],
    }
    if complete_token_path:
        complete_token_collective_contract = (
            _validate_complete_token_collective_lowering(
                module,
                expected_groups=expected_groups,
                expected_pairs=canonical_pairs,
                backend_contract=backend_contract,
                token_observation_candidates=token_observation_candidates,
                dsa_query_exact_association=dsa_query_exact_association,
                main_rope_table_enabled=main_rope_table_enabled,
                local_parallel_size=config.local_parallel_size,
            )
        )
        violations.extend(complete_token_collective_contract["violations"])
    live_tensor_contract = _classify_decoder_live_tensor_shapes(
        module,
        config=config,
        full_indexer_layers=full_layers,
        backend_contract=backend_contract,
        prefill_index_repair=prefill_index_repair,
        dsa_head_key_exact_association=(
            dsa_head_key_exact_association
        ),
        strategy_nd_attention_projection=(
            strategy_nd_attention_projection
        ),
    )
    forbidden_shapes = live_tensor_contract["forbidden_shapes"]
    violations.extend(live_tensor_contract["violations"])
    if forbidden_shapes:
        violations.append("decoder contains dead-row/full-pod live tensors")
    forbidden_full_vocab = []
    local_vocab = config.vocab_size // config.local_parallel_size
    full_vocab_dimensions = (
        (config.vocab_size,),
        (1, config.vocab_size),
        (config.local_parallel_size, local_vocab),
        (config.local_parallel_size, 1, local_vocab),
        (1, config.local_parallel_size, local_vocab),
    )
    for collective in collectives:
        for shape in collective.result_shapes:
            if (
                shape.dtype in ("bf16", "f32")
                and shape.dimensions in full_vocab_dimensions
            ):
                forbidden_full_vocab.append(
                    f"{collective.name}:{shape.dtype}["
                    + ",".join(str(value) for value in shape.dimensions)
                    + "]"
                )
    if complete_token_path and forbidden_full_vocab:
        violations.append(
            "decoder reconstructs full vocabulary logits: "
            f"{forbidden_full_vocab}"
        )
    if module.num_partitions not in (None, config.total_devices):
        violations.append(
            f"decoder expected {config.total_devices} partitions, "
            f"found {module.num_partitions}"
        )
    pallas_feature_contract: dict[str, Any] = {}
    if backend_contract in _FEATURE_DECODER_BACKEND_CONTRACTS:
        pallas_feature_contract = _validate_pallas_feature_decoder_calls(
            optimized_hlo,
            sparse_layers=sparse_layers,
            feature_output_tile=feature_output_tile,
            fuse_route_weighting=feature_fuse_route_weighting,
            reconstruct_down_fp32=feature_reconstruct_down_fp32,
            local_parallel_size=config.local_parallel_size,
        )
        violations.extend(pallas_feature_contract["violations"])
    pallas_stage_linear_contract: dict[str, Any] = {}
    if backend_contract == "tpu_v4_pp8_pallas_feature_linear":
        pallas_stage_linear_contract = (
            _validate_pallas_stage_linear_decoder_calls(
                optimized_hlo,
                layers=layers,
                dense_layers=dense_layers,
                full_indexer_layers=full_layers,
                dsa_query_backend=dsa_query_backend,
                dsa_head_key_exact_association=(
                    dsa_head_key_exact_association
                ),
                attention_projection_backend=attention_projection_backend,
                strategy_nd_attention_projection=(
                    strategy_nd_attention_projection
                ),
                dense_final_layout_convolution=(
                    dense_final_layout_convolution
                ),
                prefill_index_repair=prefill_index_repair,
                module=module,
            )
        )
        violations.extend(pallas_stage_linear_contract["violations"])
    dsa_query_association_contract: dict[str, Any] = {}
    dsa_head_key_association_contract: dict[str, Any] = {
        "applicable": False,
        "passed": True,
        "violations": [],
    }
    if backend_contract != "cpu_reference":
        dsa_query_association_contract = (
            _validate_dsa_query_decoder_association(
                optimized_hlo,
                full_indexer_layers=full_layers,
                local_parallel_size=config.local_parallel_size,
                dsa_indexer_heads=config.dsa_indexer_heads,
                index_key_width=config.index_key_width,
                backend=dsa_query_backend,
                exact_association=dsa_query_exact_association,
            )
        )
        violations.extend(dsa_query_association_contract["violations"])
        dsa_head_key_association_contract = (
            _validate_dsa_head_key_decoder_association(
                optimized_hlo,
                full_indexer_layers=full_layers,
                maximum_full_indexer_slots=(
                    config.maximum_full_indexer_slots
                ),
                exact_association=dsa_head_key_exact_association,
                prefill_index_repair=prefill_index_repair,
            )
        )
        dsa_head_key_association_contract["applicable"] = True
        violations.extend(
            dsa_head_key_association_contract["violations"]
        )
    fused_qkv_a_contract: dict[str, Any] = {}
    if attention_projection_backend == "fused_n82_convolution":
        fused_qkv_a_contract = _validate_fused_qkv_a_decoder_association(
            optimized_hlo,
            layers=layers,
            prefill_index_repair=prefill_index_repair,
            dsa_head_key_exact_association=(
                dsa_head_key_exact_association
            ),
            module=module,
        )
        violations.extend(fused_qkv_a_contract["violations"])
    main_rope_table_contract = _validate_main_rope_table_hlo(
        module,
        config=config,
        layers=layers,
        enabled=main_rope_table_enabled,
    )
    violations.extend(main_rope_table_contract["violations"])
    pregathered_attention_contract = (
        _validate_pregathered_b512_attention_hlo(
            optimized_hlo,
            module=module,
            config=config,
            layers=layers,
            enabled=pregathered_b512_attention,
        )
    )
    violations.extend(pregathered_attention_contract["violations"])
    strategy_nd_attention_contract = (
        _validate_strategy_nd_attention_projection_hlo(
            module,
            layers=layers,
            enabled=strategy_nd_attention_projection,
        )
    )
    violations.extend(strategy_nd_attention_contract["violations"])
    strategy_nd_attention_stablehlo_contract = (
        validate_strategy_nd_attention_stablehlo(
            stablehlo,
            layers=layers,
            enabled=strategy_nd_attention_projection,
        )
    )
    violations.extend(
        strategy_nd_attention_stablehlo_contract["violations"]
    )
    dense_final_layout_contract = (
        _validate_dense_final_layout_convolution_hlo(
            optimized_hlo,
            stablehlo,
            dense_layers=dense_layers,
            enabled=dense_final_layout_convolution,
            groups=canonical_groups,
            module=module,
        )
    )
    violations.extend(dense_final_layout_contract["violations"])
    return {
        "backend_contract": backend_contract,
        "collective_count": len(collectives),
        "collective_counts": observed_counts,
        "expected_collective_counts": expected_counts,
        "all_reduce_component_count": reduction_component_count,
        "expected_all_reduce_component_count": (
            expected_reduction_component_count
        ),
        "all_reduce_arity_counts": reduction_arity_counts,
        "expected_all_reduce_arity_counts": expected_reduction_arity_counts,
        "all_reduce_result_shape_counts": reduction_result_shape_counts,
        "expected_all_reduce_result_shape_counts": (
            expected_reduction_result_shape_counts
        ),
        "forbidden_shapes": forbidden_shapes,
        "live_tensor_contract": live_tensor_contract,
        "feature_output_tile": feature_output_tile,
        "feature_fuse_route_weighting": feature_fuse_route_weighting,
        "feature_reconstruct_down_fp32": feature_reconstruct_down_fp32,
        "complete_token_path": complete_token_path,
        "split_residual_state": split_residual_state,
        "prefill_index_repair": prefill_index_repair,
        "residual_transport_dimensions": list(
            residual_transport_dimensions
        ),
        "residual_transport_dtype": residual_transport_dtype,
        "residual_transport_count": len(residual_transports),
        "residual_transports": [
            _compact_collective_record(item) for item in residual_transports
        ],
        "token_observation_candidates": token_observation_candidates,
        "complete_token_collective_contract": (
            complete_token_collective_contract
        ),
        "forbidden_full_vocab_logits": forbidden_full_vocab,
        "full_indexer_layers": full_layers,
        "layer_count": layers,
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "pallas_feature_contract": pallas_feature_contract,
        "pallas_stage_linear_contract": pallas_stage_linear_contract,
        "dsa_query_association_contract": (
            dsa_query_association_contract
        ),
        "dsa_query_exact_association": dsa_query_exact_association,
        "dsa_head_key_exact_association": (
            dsa_head_key_exact_association
        ),
        "dsa_score_default_precision": (
            config.dsa_score_default_precision
        ),
        "dsa_head_key_association_contract": (
            dsa_head_key_association_contract
        ),
        "attention_projection_backend": attention_projection_backend,
        "main_rope_table_enabled": main_rope_table_enabled,
        "main_rope_table_contract": main_rope_table_contract,
        "pregathered_b512_attention": pregathered_b512_attention,
        "strategy_nd_attention_projection": (
            strategy_nd_attention_projection
        ),
        "dense_final_layout_convolution": dense_final_layout_convolution,
        "dense_final_layout_convolution_contract": (
            dense_final_layout_contract
        ),
        "pregathered_b512_attention_contract": (
            pregathered_attention_contract
        ),
        "strategy_nd_attention_projection_contract": (
            strategy_nd_attention_contract
        ),
        "strategy_nd_attention_stablehlo_contract": (
            strategy_nd_attention_stablehlo_contract
        ),
        "fused_qkv_a_contract": fused_qkv_a_contract,
        "passed": not violations,
        "violations": violations,
    }


def validate_layer0_residual_discriminator_hlo(
    optimized_hlo: str,
    *,
    config: DecoderStepConfig,
    groups: Sequence[Sequence[int]],
    variant_name: str,
    main_rope_table_enabled: bool = False,
) -> dict[str, Any]:
    """Fail closed on one isolated layer-0 arithmetic probe arm.

    The production decoder is validated independently by its exact contract.
    Each diagnostic executable has no pipeline transport and may replay only
    layer 0. Keeping every arm in a separate executable prevents TPU XLA from
    tuple-fusing their hidden reductions or next-layer normalization.
    """

    combine_variant_flags = {
        name: (attention_fp32, dense_fp32)
        for name, attention_fp32, dense_fp32 in (
            LAYER0_RESIDUAL_DISCRIMINATOR_VARIANTS
        )
    }
    if variant_name in combine_variant_flags:
        discriminator_kind = "combine_precision"
        attention_fp32, dense_fp32 = combine_variant_flags[variant_name]
        virtual_association = None
    elif variant_name in LAYER0_VIRTUAL_TP32_DISCRIMINATOR_VARIANTS:
        discriminator_kind = "virtual_tp32"
        attention_fp32 = False
        dense_fp32 = False
        virtual_association = variant_name
    elif variant_name in LAYER0_ATTENTION_SCHEDULE_DISCRIMINATOR_VARIANTS:
        discriminator_kind = "attention_schedule"
        attention_fp32 = False
        dense_fp32 = False
        virtual_association = None
    elif variant_name == "attention_output_association_control":
        discriminator_kind = "attention_output_association"
        attention_fp32 = False
        dense_fp32 = False
        virtual_association = None
    elif variant_name.startswith("attention_output_") and (
        variant_name.removeprefix("attention_output_")
        in VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
    ):
        discriminator_kind = "attention_output_association"
        attention_fp32 = False
        dense_fp32 = False
        virtual_association = variant_name.removeprefix(
            "attention_output_"
        )
    elif variant_name in LAYER0_STRATEGY_ND_ROW0_ASSOCIATION_VARIANTS:
        discriminator_kind = "strategy_nd_row0_association"
        attention_fp32 = False
        dense_fp32 = False
        virtual_association = (
            STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
            if variant_name == "strategy_nd_row0_both"
            else None
        )
    else:
        raise PlanValidationError(
            f"unknown layer-0 discriminator variant: {variant_name}"
        )

    skeleton = PipelineSkeletonConfig(
        config.stage_count,
        config.local_parallel_size,
        config.hidden_size,
        config.selected_width,
    )
    canonical_groups = _canonical_groups(groups, skeleton)
    module = parse_hlo_module(optimized_hlo)
    collectives = module.collectives
    allowed_opcodes = {"all-gather", "all-reduce"}
    unexpected_collectives = [
        _compact_collective_record(item)
        for item in collectives
        if item.opcode not in allowed_opcodes
    ]
    escaped_collectives = [
        _compact_collective_record(item)
        for item in collectives
        if item.replica_groups != canonical_groups
        or item.maximum_group_size != config.local_parallel_size
    ]
    by_opcode = Counter(item.opcode for item in collectives)
    monolithic_cache_gather_scope = (
        "greenfield_replicated_monolithic_attention_cache_gather"
    )

    def in_named_scope(op_name: str | None, scope: str) -> bool:
        return bool(op_name) and scope in op_name.split("/")

    monolithic_cache_gathers = tuple(
        item
        for item in collectives
        if in_named_scope(item.op_name, monolithic_cache_gather_scope)
    )
    monolithic_attention_scope = "greenfield_replicated_monolithic_attention"
    monolithic_attention_scope_present = any(
        in_named_scope(instruction.op_name, monolithic_attention_scope)
        for instruction in module.instructions
    )
    owner_split_gather_scopes = (
        "greenfield_owner_split_attention_output_gather",
        "greenfield_owner_split_attention_lse_gather",
        "greenfield_owner_split_attention_validity_gather",
    )
    owner_split_gathers = tuple(
        item
        for item in collectives
        if any(
            in_named_scope(item.op_name, scope)
            for scope in owner_split_gather_scopes
        )
    )
    owner_split_output_gathers = tuple(
        item
        for item in collectives
        if in_named_scope(item.op_name, owner_split_gather_scopes[0])
    )
    strategy_nd_gather_scope = (
        "greenfield_strategy_nd_row0_association_gather"
    )
    strategy_nd_gathers = tuple(
        item
        for item in collectives
        if in_named_scope(item.op_name, strategy_nd_gather_scope)
    )

    def is_strategy_nd_partials_gather(item: Any) -> bool:
        if item.opcode != "all-gather":
            return False
        operand_shape = HloShape("bf16", (8, 1, config.hidden_size))
        logical_result = HloShape(
            "bf16",
            (config.local_parallel_size, 8, 1, config.hidden_size),
        )
        flattened_result = HloShape(
            "bf16",
            (config.local_parallel_size * 8, 1, config.hidden_size),
        )
        return bool(
            operand_shape in item.operand_shapes
            and any(
                shape in {logical_result, flattened_result}
                for shape in item.result_shapes
            )
        )

    strategy_nd_shaped_gathers = tuple(
        item for item in collectives
        if is_strategy_nd_partials_gather(item)
    )
    strategy_nd_attention_gathers = tuple(
        item
        for item in strategy_nd_gathers
        if in_named_scope(
            item.op_name,
            "greenfield_strategy_nd_row0_attention_output",
        )
    )
    strategy_nd_dense_gathers = tuple(
        item
        for item in strategy_nd_gathers
        if in_named_scope(
            item.op_name,
            "greenfield_strategy_nd_row0_dense_down",
        )
    )

    def is_cache_shaped_gather(item: Any) -> bool:
        if item.opcode != "all-gather":
            return False
        operands = tuple(
            shape
            for shape in item.operand_shapes
            if shape.dtype == "bf16"
            and len(shape.dimensions) == 3
            and shape.dimensions[-1] == config.packed_cache_width
        )
        for result in item.result_shapes:
            if result.dtype != "bf16":
                continue
            if (
                len(result.dimensions) == 4
                and result.dimensions[0] == config.local_parallel_size
                and result.dimensions[-1] == config.packed_cache_width
                and any(
                    operand.dimensions == result.dimensions[1:]
                    for operand in operands
                )
            ):
                return True
            # TPU SPMD canonicalizes ``all_gather(..., tiled=False)`` by
            # folding the logical replica dimension into cache page axis 0.
            if (
                len(result.dimensions) == 3
                and any(
                    result.dimensions[0]
                    == operand.dimensions[0] * config.local_parallel_size
                    and result.dimensions[1:] == operand.dimensions[1:]
                    for operand in operands
                )
            ):
                return True
        return False

    cache_shaped_gathers = tuple(
        item
        for item in collectives
        if is_cache_shaped_gather(item)
    )
    result_shapes = Counter(
        f"{shape.dtype}["
        + ",".join(str(value) for value in shape.dimensions)
        + "]"
        for item in collectives
        for shape in item.result_shapes
    )

    custom_calls = tuple(
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    )
    attention_bf16_name = "greenfield_fp8_block_matmul_m8_k4096_n6144"
    attention_f32_name = (
        "greenfield_fp8_block_matmul_f32_m8_k4096_n6144"
    )
    dense_bf16_name = (
        "greenfield_fp8_fused_block_swiglu_"
        "m8_h6144_i3072_o6144"
    )
    dense_f32_name = f"{dense_bf16_name}_downf32"
    virtual_attention_name = (
        "greenfield_fp8_strategy_nd_o_m8_k512_n6144"
    )
    virtual_dense_name = (
        "greenfield_fp8_fused_block_swiglu_"
        "m8_h6144_i384_o6144"
    )
    fp32_boundary_name = "greenfield_fp32_to_bf16_r8_h6144"
    kernel_counts = {
        attention_bf16_name: sum(
            attention_bf16_name in line
            and attention_f32_name not in line
            for line in custom_calls
        ),
        attention_f32_name: sum(
            attention_f32_name in line for line in custom_calls
        ),
        dense_bf16_name: sum(
            dense_bf16_name in line and dense_f32_name not in line
            for line in custom_calls
        ),
        dense_f32_name: sum(dense_f32_name in line for line in custom_calls),
        virtual_attention_name: sum(
            virtual_attention_name in line for line in custom_calls
        ),
        virtual_dense_name: sum(
            virtual_dense_name in line for line in custom_calls
        ),
        fp32_boundary_name: sum(
            fp32_boundary_name in line for line in custom_calls
        ),
    }
    host_markers = sorted(
        marker
        for marker in (
            "host_callback",
            "xla_python_cpu_callback",
            "xla_ffi_python_cpu_callback",
            "outside_compilation",
        )
        if marker in optimized_hlo
    )
    entry_roots = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )
    expected_root_shapes = (
        HloShape("s32", (1, config.selected_width)),
        HloShape("f32", (1, config.selected_width)),
        HloShape("s32", (1, 1)),
        HloShape("bf16", (1, config.hidden_size)),
        HloShape("pred", (1, 1)),
    )
    normalization_scope = (
        f"greenfield_layer1_input_norm_{variant_name}"
    )
    normalization_scope_present = normalization_scope in optimized_hlo
    scoped_instructions = tuple(
        instruction
        for instruction in module.instructions
        if instruction.op_name is not None
        and normalization_scope in instruction.op_name
    )
    multi_hidden_results = tuple(
        instruction.to_dict()
        for instruction in scoped_instructions
        if sum(
            shape == HloShape("bf16", (1, config.hidden_size))
            for shape in instruction.result_shapes
        )
        > 1
    )
    multi_scalar_reductions = tuple(
        instruction.to_dict()
        for instruction in scoped_instructions
        if instruction.opcode in {"fusion", "reduce"}
        and sum(
            shape == HloShape("f32", ())
            for shape in instruction.result_shapes
        )
        > 1
    )
    if discriminator_kind in {"combine_precision", "attention_schedule"} or (
        discriminator_kind in {
            "attention_output_association",
            "strategy_nd_row0_association",
        }
        and virtual_association is None
    ):
        expected_kernel_counts = {
            attention_bf16_name: int(not attention_fp32),
            attention_f32_name: int(attention_fp32),
            dense_bf16_name: int(not dense_fp32),
            dense_f32_name: int(dense_fp32),
            virtual_attention_name: 0,
            virtual_dense_name: 0,
            fp32_boundary_name: int(attention_fp32) + int(dense_fp32),
        }
        expected_fp32_combines = int(attention_fp32) + int(dense_fp32)
        expected_hidden_collective_shapes = None
        virtual_association_scope = None
        virtual_association_scope_present = True
    elif discriminator_kind == "attention_output_association":
        expected_kernel_counts = {
            attention_bf16_name: 0,
            attention_f32_name: 0,
            dense_bf16_name: 1,
            dense_f32_name: 0,
            virtual_attention_name: 8,
            virtual_dense_name: 0,
            fp32_boundary_name: 0,
        }
        expected_fp32_combines = 0
        assert virtual_association is not None
        dcp_first = virtual_association.startswith("dcp_then_model_")
        expected_hidden_collective_shapes = {
            "bf16[1,6144]": 3 if dcp_first else 2,
            "bf16[8,1,6144]": 0 if dcp_first else 1,
        }
        virtual_association_scope = (
            f"greenfield_virtual_tp32_{virtual_association}"
        )
        virtual_association_scope_present = (
            virtual_association_scope in optimized_hlo
        )
    elif discriminator_kind == "strategy_nd_row0_association":
        expected_kernel_counts = {
            attention_bf16_name: 0,
            attention_f32_name: 0,
            dense_bf16_name: 0,
            dense_f32_name: 0,
            virtual_attention_name: 8,
            virtual_dense_name: 8,
            fp32_boundary_name: 0,
        }
        expected_fp32_combines = 0
        expected_hidden_collective_shapes = None
        virtual_association_scope = (
            "greenfield_strategy_nd_row0_association"
        )
        virtual_association_scope_present = (
            virtual_association_scope in optimized_hlo
        )
    else:
        expected_kernel_counts = {
            attention_bf16_name: 0,
            attention_f32_name: 0,
            dense_bf16_name: 0,
            dense_f32_name: 0,
            virtual_attention_name: 8,
            virtual_dense_name: 8,
            fp32_boundary_name: 0,
        }
        expected_fp32_combines = 0
        assert virtual_association is not None
        dcp_first = virtual_association.startswith("dcp_then_model_")
        expected_hidden_collective_shapes = {
            "bf16[1,6144]": 3 if dcp_first else 1,
            "bf16[8,1,6144]": 0 if dcp_first else 2,
        }
        virtual_association_scope = (
            f"greenfield_virtual_tp32_{virtual_association}"
        )
        virtual_association_scope_present = (
            virtual_association_scope in optimized_hlo
        )
    violations = []
    if module.num_partitions not in (None, config.total_devices):
        violations.append(
            "layer-0 discriminator partition count drifted: "
            f"expected={config.total_devices} observed={module.num_partitions}"
        )
    if unexpected_collectives:
        violations.append(
            "layer-0 discriminator contains transport/global communication"
        )
    if escaped_collectives:
        violations.append("layer-0 discriminator escaped PP8 local groups")
    minimum_all_gathers = (
        3
        if discriminator_kind
        in {
            "attention_schedule",
            "attention_output_association",
            "strategy_nd_row0_association",
        }
        else 5
    )
    if not minimum_all_gathers <= by_opcode.get("all-gather", 0) <= 20:
        violations.append(
            "layer-0 discriminator DSA/attention gather count drifted: "
            f"{by_opcode.get('all-gather', 0)}"
        )
    if by_opcode.get("all-reduce", 0) < 3:
        violations.append("layer-0 discriminator lost local reductions")
    if kernel_counts != expected_kernel_counts:
        violations.append(
            "layer-0 discriminator kernel association drifted: "
            f"expected={expected_kernel_counts} observed={kernel_counts}"
        )
    observed_fp32_combines = sum(
        count
        for shape, count in result_shapes.items()
        if shape == "f32[1,6144]"
    )
    if observed_fp32_combines != expected_fp32_combines:
        violations.append(
            "layer-0 discriminator FP32 local-combine count drifted: "
            f"expected={expected_fp32_combines} "
            f"observed={observed_fp32_combines}"
        )
    if expected_hidden_collective_shapes is not None:
        observed_hidden_collective_shapes = {
            shape: result_shapes.get(shape, 0)
            for shape in expected_hidden_collective_shapes
        }
        if (
            observed_hidden_collective_shapes
            != expected_hidden_collective_shapes
        ):
            violations.append(
                "layer-0 discriminator virtual TP32 collective association "
                "drifted: "
                f"expected={expected_hidden_collective_shapes} "
                f"observed={observed_hidden_collective_shapes}"
            )
        if not virtual_association_scope_present:
            violations.append(
                "layer-0 discriminator lost its virtual TP32 association scope"
            )
    expects_strategy_nd_row0 = (
        variant_name == "strategy_nd_row0_both"
    )
    if expects_strategy_nd_row0:
        if len(strategy_nd_gathers) != 2:
            violations.append(
                "layer-0 StrategyND challenger must contain exactly two "
                f"scoped partial gathers, found {len(strategy_nd_gathers)}"
            )
        if len(strategy_nd_shaped_gathers) != 2:
            violations.append(
                "layer-0 StrategyND global partial-gather count/shapes drifted: "
                f"{len(strategy_nd_shaped_gathers)}"
            )
        if any(
            item not in strategy_nd_gathers
            for item in strategy_nd_shaped_gathers
        ):
            violations.append(
                "layer-0 StrategyND contains an unscoped partial gather"
            )
        if len(strategy_nd_attention_gathers) != 1:
            violations.append(
                "layer-0 StrategyND attention partial-gather count drifted: "
                f"{len(strategy_nd_attention_gathers)}"
            )
        if len(strategy_nd_dense_gathers) != 1:
            violations.append(
                "layer-0 StrategyND dense partial-gather count drifted: "
                f"{len(strategy_nd_dense_gathers)}"
            )
        if not virtual_association_scope_present:
            violations.append(
                "layer-0 discriminator lost its StrategyND row-zero scope"
            )
    elif strategy_nd_gathers or strategy_nd_shaped_gathers:
        violations.append(
            "layer-0 control unexpectedly contains StrategyND partial gathers"
        )
    if (
        len(entry_roots) != 1
        or entry_roots[0].result_shapes != expected_root_shapes
    ):
        violations.append("layer-0 discriminator lost its single-row output")
    if multi_hidden_results:
        violations.append(
            "layer-0 discriminator tuple-fused multiple hidden rows"
        )
    if multi_scalar_reductions:
        violations.append(
            "layer-0 discriminator tuple-fused multiple RMS reductions"
        )
    if not normalization_scope_present:
        violations.append(
            "layer-0 discriminator lost the scoped layer-1 normalization"
        )
    expects_monolithic_attention = (
        variant_name == "replicated_monolithic_attention"
    )
    if expects_monolithic_attention:
        if not monolithic_attention_scope_present:
            violations.append(
                "layer-0 discriminator lost replicated monolithic attention"
            )
        if len(monolithic_cache_gathers) != 1:
            violations.append(
                "layer-0 discriminator must have exactly one scoped full-cache "
                f"gather, found {len(monolithic_cache_gathers)}"
            )
        elif not is_cache_shaped_gather(monolithic_cache_gathers[0]):
            violations.append(
                "layer-0 discriminator monolithic cache-gather shape drifted"
            )
        if len(cache_shaped_gathers) != 1:
            violations.append(
                "layer-0 monolithic challenger full-cache gather count drifted: "
                f"{len(cache_shaped_gathers)}"
            )
        if owner_split_gathers:
            violations.append(
                "layer-0 monolithic challenger retained owner-split partial gathers"
            )
    elif monolithic_attention_scope_present or monolithic_cache_gathers:
        violations.append(
            "layer-0 control unexpectedly contains monolithic attention"
        )
    elif discriminator_kind in {
        "attention_schedule",
        "attention_output_association",
        "strategy_nd_row0_association",
    }:
        # TPU SPMD rewrites the LSE all-gather to an unnamed f32[256]
        # all-reduce and removes the validity gather.  The BF16 attention-output
        # gather survives with its scope, so pin it and independently reject the
        # distinctive logical full-cache gather used by the challenger. TPU SPMD
        # may flatten its replica and page dimensions, so the shape guard checks
        # the exact LP4 growth from operand to result as well as the packed width.
        if len(owner_split_output_gathers) != 1:
            violations.append(
                "layer-0 attention control lost its owner-split output gather"
            )
        if cache_shaped_gathers:
            violations.append(
                "layer-0 attention control contains a full-cache gather"
            )
    main_rope_table_contract = _validate_main_rope_table_hlo(
        module,
        config=config,
        layers=1,
        enabled=main_rope_table_enabled,
    )
    violations.extend(main_rope_table_contract["violations"])
    if host_markers:
        violations.append(
            f"layer-0 discriminator contains host execution: {host_markers}"
        )
    return {
        "collective_counts": dict(sorted(by_opcode.items())),
        "collective_result_shape_counts": dict(sorted(result_shapes.items())),
        "escaped_collectives": escaped_collectives,
        "host_markers": host_markers,
        "kernel_counts": kernel_counts,
        "main_rope_table_contract": main_rope_table_contract,
        "main_rope_table_enabled": main_rope_table_enabled,
        "monolithic_attention_scope_present": (
            monolithic_attention_scope_present
        ),
        "monolithic_cache_gathers": [
            _compact_collective_record(item)
            for item in monolithic_cache_gathers
        ],
        "cache_shaped_gathers": [
            _compact_collective_record(item)
            for item in cache_shaped_gathers
        ],
        "owner_split_gathers": [
            _compact_collective_record(item) for item in owner_split_gathers
        ],
        "owner_split_output_gathers": [
            _compact_collective_record(item)
            for item in owner_split_output_gathers
        ],
        "strategy_nd_gathers": [
            _compact_collective_record(item) for item in strategy_nd_gathers
        ],
        "strategy_nd_attention_gather_count": len(
            strategy_nd_attention_gathers
        ),
        "strategy_nd_dense_gather_count": len(strategy_nd_dense_gathers),
        "strategy_nd_shaped_gather_count": len(
            strategy_nd_shaped_gathers
        ),
        "expected_kernel_counts": expected_kernel_counts,
        "expected_fp32_local_combine_count": expected_fp32_combines,
        "expected_hidden_collective_shapes": (
            expected_hidden_collective_shapes
        ),
        "entry_root_shapes": [
            shape.to_dict()
            for root in entry_roots
            for shape in root.result_shapes
        ],
        "multi_hidden_results": list(multi_hidden_results),
        "multi_scalar_reductions": list(multi_scalar_reductions),
        "normalization_scope": normalization_scope,
        "normalization_scope_present": normalization_scope_present,
        "num_partitions": module.num_partitions,
        "observed_fp32_local_combine_count": observed_fp32_combines,
        "discriminator_kind": discriminator_kind,
        "passed": not violations,
        "unexpected_collectives": unexpected_collectives,
        "variant_name": variant_name,
        "virtual_tp32_association_scope": virtual_association_scope,
        "virtual_tp32_association_scope_present": (
            virtual_association_scope_present
        ),
        "violations": violations,
    }


def validate_layer0_ingredients_observer_hlo(
    optimized_hlo: str,
    *,
    config: DecoderStepConfig,
    groups: Sequence[Sequence[int]],
    main_rope_table_enabled: bool = False,
) -> dict[str, Any]:
    """Fail closed on the single diagnostic layer-0 ingredient replay."""

    skeleton = PipelineSkeletonConfig(
        config.stage_count,
        config.local_parallel_size,
        config.hidden_size,
        config.selected_width,
    )
    canonical_groups = _canonical_groups(groups, skeleton)
    module = parse_hlo_module(optimized_hlo)
    collectives = module.collectives
    allowed_opcodes = {"all-gather", "all-reduce"}
    unexpected_collectives = [
        _compact_collective_record(item)
        for item in collectives
        if item.opcode not in allowed_opcodes
    ]
    escaped_collectives = [
        _compact_collective_record(item)
        for item in collectives
        if item.replica_groups != canonical_groups
        or item.maximum_group_size != config.local_parallel_size
    ]
    by_opcode = Counter(item.opcode for item in collectives)
    custom_calls = tuple(
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    )
    kernel_names = {
        "attention_production": "greenfield_fp8_block_matmul_m8_k4096_n6144",
        "attention_virtual": "greenfield_fp8_strategy_nd_o_m8_k512_n6144",
        "dense_production": (
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
        ),
        "dense_virtual": (
            "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
        ),
        "fp32_boundary": "greenfield_fp32_to_bf16_r8_h6144",
    }
    kernel_counts = {
        name: sum(value in line for line in custom_calls)
        for name, value in kernel_names.items()
    }
    expected_kernel_counts = {
        "attention_production": 1,
        "attention_virtual": 8,
        "dense_production": 1,
        "dense_virtual": 8,
        "fp32_boundary": 0,
    }
    entry_roots = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_line.lstrip().startswith("ROOT ")
    )
    hidden = config.hidden_size
    selected = config.selected_width
    expected_root_shapes = (
        HloShape("s32", (1, selected)),
        HloShape("f32", (1, selected)),
        HloShape("s32", (1, 1)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, 640)),
        HloShape("s32", (1, selected)),
        HloShape("s32", (1, 1)),
        HloShape("bf16", (1, selected, 640)),
        HloShape("pred", (1, 1)),
        HloShape("bf16", (1, 64, 512)),
        HloShape("f32", (1, 64)),
        HloShape("pred", (1, 1)),
        HloShape("bf16", (1, 64, 512)),
        HloShape("f32", (1, 64)),
        HloShape("pred", (1, 1)),
        HloShape("bf16", (1, 16, 256)),
        HloShape("bf16", (1, 4096)),
        HloShape("bf16", (1, 8, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, 8, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("bf16", (1, hidden)),
        HloShape("pred", (1, 1)),
    )
    host_markers = sorted(
        marker
        for marker in (
            "host_callback",
            "xla_python_cpu_callback",
            "xla_ffi_python_cpu_callback",
            "outside_compilation",
        )
        if marker in optimized_hlo
    )
    violations = []
    if module.num_partitions not in (None, config.total_devices):
        violations.append(
            "layer-0 ingredient observer partition count drifted: "
            f"expected={config.total_devices} observed={module.num_partitions}"
        )
    if unexpected_collectives:
        violations.append(
            "layer-0 ingredient observer contains transport/global communication"
        )
    if escaped_collectives:
        violations.append("layer-0 ingredient observer escaped PP8 local groups")
    if not 5 <= by_opcode.get("all-gather", 0) <= 20:
        violations.append(
            "layer-0 ingredient observer DSA/attention gather count drifted: "
            f"{by_opcode.get('all-gather', 0)}"
        )
    if by_opcode.get("all-reduce", 0) < 3:
        violations.append("layer-0 ingredient observer lost local reductions")
    if kernel_counts != expected_kernel_counts:
        violations.append(
            "layer-0 ingredient observer kernel set drifted: "
            f"expected={expected_kernel_counts} observed={kernel_counts}"
        )
    if (
        len(entry_roots) != 1
        or entry_roots[0].result_shapes != expected_root_shapes
    ):
        violations.append(
            "layer-0 ingredient observer lost its pinned one-row outputs"
        )
    if "greenfield_layer0_ingredients_layer1_norm" not in optimized_hlo:
        violations.append(
            "layer-0 ingredient observer lost the scoped layer-1 normalization"
        )
    main_rope_table_contract = _validate_main_rope_table_hlo(
        module,
        config=config,
        layers=1,
        enabled=main_rope_table_enabled,
        required_entry_root_indices=(
            (LAYER0_INGREDIENT_NAMES.index("attention_output_input"),)
            if main_rope_table_enabled
            else ()
        ),
    )
    violations.extend(main_rope_table_contract["violations"])
    if host_markers:
        violations.append(
            f"layer-0 ingredient observer contains host execution: {host_markers}"
        )
    return {
        "collective_counts": dict(sorted(by_opcode.items())),
        "entry_root_shapes": [
            shape.to_dict()
            for root in entry_roots
            for shape in root.result_shapes
        ],
        "escaped_collectives": escaped_collectives,
        "expected_kernel_counts": expected_kernel_counts,
        "expected_root_shapes": [
            shape.to_dict() for shape in expected_root_shapes
        ],
        "host_markers": host_markers,
        "ingredient_names": list(LAYER0_INGREDIENT_NAMES),
        "kernel_counts": kernel_counts,
        "main_rope_table_contract": main_rope_table_contract,
        "main_rope_table_enabled": main_rope_table_enabled,
        "num_partitions": module.num_partitions,
        "passed": not violations,
        "unexpected_collectives": unexpected_collectives,
        "violations": violations,
    }


def _attention_weights(
    weight: Any,
    slot: int,
    projection_backend: AttentionProjectionBackend,
) -> AttentionFp8Weights:
    base = f"attention.slot_{slot:02d}"
    common = {
        "q_a_norm_weight": weight(f"{base}.q_a_norm"),
        "q_b_bits": weight(f"{base}.q_b.weight_bits"),
        "q_b_scale": weight(f"{base}.q_b.scale_inv"),
        "kv_a_norm_weight": weight(f"{base}.kv_a_norm"),
        "kv_b_bits": weight(f"{base}.kv_b.weight_bits"),
        "kv_b_scale": weight(f"{base}.kv_b.scale_inv"),
        "o_bits": weight(f"{base}.o.weight_bits"),
        "o_scale": weight(f"{base}.o.scale_inv"),
    }
    if projection_backend == "separate":
        return AttentionFp8Weights(
            q_a_bits=weight(f"{base}.q_a.weight_bits"),
            q_a_scale=weight(f"{base}.q_a.scale_inv"),
            kv_a_bits=weight(f"{base}.kv_a.weight_bits"),
            kv_a_scale=weight(f"{base}.kv_a.scale_inv"),
            **common,
        )
    if projection_backend != "fused_n82_convolution":
        raise ValueError("decoder attention projection backend is unknown")
    return AttentionFp8Weights(
        q_a_bits=None,
        q_a_scale=None,
        kv_a_bits=None,
        kv_a_scale=None,
        qkv_a_bits=weight(f"{base}.qkv_a.weight_bits"),
        qkv_a_scale=weight(f"{base}.qkv_a.scale_inv"),
        **common,
    )


def _dsa_weights(weight: Any, slot: int) -> DsaFp8Weights:
    base = f"indexer.slot_{slot:02d}"
    return DsaFp8Weights(
        weight(f"{base}.wq_b.weight_bits"),
        weight(f"{base}.wq_b.scale_inv"),
        weight(f"{base}.wk.weight_bits"),
        weight(f"{base}.wk.scale_inv"),
        weight(f"{base}.key_norm_weight"),
        weight(f"{base}.key_norm_bias"),
        weight(f"{base}.head_weight"),
    )


def _dense_weights(
    weight: Any,
    slot: int,
    *,
    final_layout_convolution: bool = False,
) -> DenseFp8Weights:
    base = f"dense.slot_{slot:02d}"
    if final_layout_convolution:
        return DenseFp8Weights(
            weight(f"{base}.merged_gate_up.weight_bits_in_out"),
            weight(f"{base}.merged_gate_up.scale_inv_in_out"),
            None,
            None,
            weight(f"{base}.down.weight_bits_in_out"),
            weight(f"{base}.down.scale_inv_in_out"),
        )
    return DenseFp8Weights(
        weight(f"{base}.gate.weight_bits"),
        weight(f"{base}.gate.scale_inv"),
        weight(f"{base}.up.weight_bits"),
        weight(f"{base}.up.scale_inv"),
        weight(f"{base}.down.weight_bits"),
        weight(f"{base}.down.scale_inv"),
    )


def _moe_weights(weight: Any, slot: int) -> MoeFp8Weights:
    base = f"sparse.slot_{slot:02d}"
    return MoeFp8Weights(
        weight(f"{base}.router_weight"),
        weight(f"{base}.correction_bias"),
        weight(f"{base}.experts.gate_proj.weight_bits"),
        weight(f"{base}.experts.gate_proj.scale_inv"),
        weight(f"{base}.experts.up_proj.weight_bits"),
        weight(f"{base}.experts.up_proj.scale_inv"),
        weight(f"{base}.experts.down_proj.weight_bits"),
        weight(f"{base}.experts.down_proj.scale_inv"),
        weight(f"{base}.shared.gate_proj.weight_bits"),
        weight(f"{base}.shared.gate_proj.scale_inv"),
        weight(f"{base}.shared.up_proj.weight_bits"),
        weight(f"{base}.shared.up_proj.scale_inv"),
        weight(f"{base}.shared.down_proj.weight_bits"),
        weight(f"{base}.shared.down_proj.scale_inv"),
    )


def _execute_stage(
    stage: StageExecution,
    values: tuple[Any, Any, Any, Any],
    *,
    weight: Any,
    local_slot: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    axis_name: str,
    axis_groups: tuple[tuple[int, ...], ...],
    config: DecoderStepConfig,
    dsa_contract: DsaNumericalContract,
    mla_contract: MlaNumericalContract,
    moe_contract: GlmMoeNumericalContract,
    cache_layout: StageLocalKvLayout,
    block_shape: tuple[int, int],
    sparse_moe_backend: SparseMoeBackend,
    pallas_moe_config: Fp8BlockMatmulConfig | None,
    pallas_moe_fuse_route_weighting: bool,
    pallas_moe_reconstruct_down_fp32: bool,
    linear_backend: StageLinearBackend,
    dsa_query_backend: StageLinearBackend,
    dsa_query_weight_aliases: tuple[tuple[Any, ...], ...] | None,
    dsa_recurrent_wk_weights: tuple[Any, ...] | None,
    attention_projection_backend: AttentionProjectionBackend,
    pregathered_b512_attention: bool,
    strategy_nd_attention_projection: bool,
    dense_final_layout_convolution: bool,
    main_rope_table_row: Any | None = None,
    dsa_observation: Any | None = None,
    dsa_internal_observation: DsaInternalObservation | None = None,
    layer_residual_observation: Any | None = None,
    prefill_index_inputs: Any | None = None,
) -> tuple[Any, ...]:
    import jax.numpy as jnp
    from jax import lax

    residual, kv_cache, index_cache, metadata = values
    full_slot = 0
    for layer in stage.layers:
        if layer_residual_observation is not None:
            layer_residual_observation = layer_residual_observation.at[
                layer.layer_id
            ].set(residual[0])
        attention = _attention_weights(
            weight,
            layer.stage_slot,
            attention_projection_backend,
        )
        input_norm = weight(
            f"attention.slot_{layer.stage_slot:02d}.input_norm"
        )
        post_norm = weight(
            f"attention.slot_{layer.stage_slot:02d}.post_norm"
        )
        if layer.indexer_kind == "full":
            dsa = _dsa_weights(weight, full_slot)
            query_weight_aliases = (
                None
                if dsa_query_weight_aliases is None
                else tuple(
                    alias[full_slot] for alias in dsa_query_weight_aliases
                )
            )
            recurrent_wk_weight = (
                None
                if dsa_recurrent_wk_weights is None
                else dsa_recurrent_wk_weights[full_slot]
            )
            layer_index_cache = index_cache[full_slot]
            current_full_slot = full_slot
            full_slot += 1
        else:
            dsa = None
            query_weight_aliases = None
            recurrent_wk_weight = None
            layer_index_cache = index_cache[0]
            current_full_slot = None
        if layer.mlp_kind == "dense":
            assert layer.dense_slot is not None
            dense = _dense_weights(
                weight,
                layer.dense_slot,
                final_layout_convolution=dense_final_layout_convolution,
            )
            moe = None
        else:
            assert layer.sparse_slot is not None
            dense = None
            moe = _moe_weights(weight, layer.sparse_slot)
        producer_valid = (
            jnp.asarray(True)
            if layer.indexer_kind == "full"
            else metadata[0, config.producer_index]
            == jnp.int32(layer.index_state_producer_layer)
        )
        incoming_valid = (
            (metadata[0, config.health_index] == jnp.int32(1))
            & producer_valid
        )[None]
        result = stage_local_transformer_layer_fp8_mapped(
            residual,
            kv_cache[layer.stage_slot],
            layer_index_cache,
            metadata[:, : config.selected_width],
            metadata[:, config.count_index],
            position,
            block_tables,
            context_lengths,
            input_norm,
            post_norm,
            attention,
            dsa,
            dense,
            moe,
            incoming_valid,
            local_slot,
            axis_name=axis_name,
            indexer_kind=layer.indexer_kind,
            mlp_kind=layer.mlp_kind,
            dsa_contract=dsa_contract,
            mla_contract=mla_contract,
            moe_contract=moe_contract,
            cache_layout=cache_layout,
            axis_index_groups=axis_groups,
            block_shape=block_shape,
            sparse_moe_backend=sparse_moe_backend,
            pallas_moe_config=pallas_moe_config,
            pallas_moe_fuse_route_weighting=(
                pallas_moe_fuse_route_weighting
            ),
            pallas_moe_reconstruct_down_fp32=(
                pallas_moe_reconstruct_down_fp32
            ),
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            dsa_query_weight_aliases=query_weight_aliases,
            dsa_precomputed_wk_weight=recurrent_wk_weight,
            dsa_head_key_exact_association=(
                dsa_recurrent_wk_weights is not None
            ),
            dsa_score_precision=(
                "default"
                if config.dsa_score_default_precision
                else "highest"
            ),
            attention_projection_backend=attention_projection_backend,
            pregathered_b512_attention=pregathered_b512_attention,
            main_rope_table_row=main_rope_table_row,
        )
        if current_full_slot is not None and prefill_index_inputs is not None:
            prefill_index_inputs = prefill_index_inputs.at[
                current_full_slot
            ].set(result.dsa_internals.normalized_hidden[0])
        residual = result.output
        if layer_residual_observation is not None:
            layer_residual_observation = layer_residual_observation.at[
                layer.layer_id + 1
            ].set(residual[0])
        kv_cache = kv_cache.at[layer.stage_slot].set(result.kv_cache)
        if current_full_slot is not None:
            index_cache = index_cache.at[current_full_slot].set(
                result.index_cache
            )
            metadata = metadata.at[0, config.producer_index].set(
                jnp.int32(layer.layer_id)
            )
            if dsa_observation is not None:
                dsa_observation = dsa_observation.at[
                    current_full_slot, : config.selected_width
                ].set(result.selected_positions[0])
                dsa_observation = dsa_observation.at[
                    current_full_slot,
                    config.selected_width : 2 * config.selected_width,
                ].set(
                    lax.bitcast_convert_type(
                        result.selected_scores[0], jnp.int32
                    )
                )
                dsa_observation = dsa_observation.at[
                    current_full_slot, 2 * config.selected_width
                ].set(result.selected_valid_counts[0])
                dsa_observation = dsa_observation.at[
                    current_full_slot, 2 * config.selected_width + 1
                ].set(jnp.int32(layer.layer_id))
            if dsa_internal_observation is not None:
                internals = result.dsa_internals
                dsa_internal_observation = DsaInternalObservation(
                    dsa_internal_observation.normalized_hidden.at[
                        current_full_slot
                    ].set(internals.normalized_hidden[0]),
                    dsa_internal_observation.q_a_state.at[
                        current_full_slot
                    ].set(internals.q_a_state[0]),
                    dsa_internal_observation.query.at[
                        current_full_slot
                    ].set(internals.query[0]),
                    dsa_internal_observation.head_weights.at[
                        current_full_slot
                    ].set(internals.head_weights[0]),
                    dsa_internal_observation.current_key.at[
                        current_full_slot
                    ].set(internals.current_key[0]),
                    dsa_internal_observation.producer_layer_ids.at[
                        current_full_slot
                    ].set(jnp.int32(layer.layer_id)),
                )
        metadata = metadata.at[:, : config.selected_width].set(
            result.selected_positions
        )
        metadata = metadata.at[:, config.count_index].set(
            result.selected_valid_counts
        )
        metadata = metadata.at[0, config.health_index].set(
            result.contract_valid[0].astype(jnp.int32)
        )
    metadata = metadata.at[0, config.visited_index].set(
        jnp.bitwise_or(
            metadata[0, config.visited_index],
            jnp.int32(1 << stage.assignment.stage_id),
        )
    )
    values = (residual, kv_cache, index_cache, metadata)
    if dsa_observation is not None:
        values = (*values, dsa_observation)
    if dsa_internal_observation is not None:
        values = (*values, dsa_internal_observation)
    if layer_residual_observation is not None:
        values = (*values, layer_residual_observation)
    if prefill_index_inputs is not None:
        values = (*values, prefill_index_inputs)
    return values


def _execute_stage_split(
    stage: StageExecution,
    values: tuple[Any, Any, Any, Any],
    *,
    weight: Any,
    local_slot: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    axis_name: str,
    axis_groups: tuple[tuple[int, ...], ...],
    config: DecoderStepConfig,
    dsa_contract: DsaNumericalContract,
    mla_contract: MlaNumericalContract,
    moe_contract: GlmMoeNumericalContract,
    cache_layout: StageLocalKvLayout,
    block_shape: tuple[int, int],
    sparse_moe_backend: SparseMoeBackend,
    pallas_moe_config: Fp8BlockMatmulConfig | None,
    pallas_moe_fuse_route_weighting: bool,
    pallas_moe_reconstruct_down_fp32: bool,
    linear_backend: StageLinearBackend,
    dsa_query_backend: StageLinearBackend,
    dsa_query_weight_aliases: tuple[tuple[Any, ...], ...] | None,
    dsa_recurrent_wk_weights: tuple[Any, ...] | None,
    attention_projection_backend: AttentionProjectionBackend,
    pregathered_b512_attention: bool,
    strategy_nd_attention_projection: bool,
    dense_final_layout_convolution: bool,
    main_rope_table_row: Any | None = None,
    dsa_observation: Any | None = None,
    dsa_internal_observation: DsaInternalObservation | None = None,
    layer_residual_observation: Any | None = None,
    prefill_index_inputs: Any | None = None,
) -> tuple[Any, ...]:
    """Execute one stage with the accepted hidden/residual state association."""

    import jax.numpy as jnp
    from jax import lax

    residual_state, kv_cache, index_cache, metadata = values
    if residual_state.shape != (2, 1, config.hidden_size):
        raise ValueError("split decoder state must be [hidden,residual]")
    hidden_states = residual_state[0]
    residual = residual_state[1]
    full_slot = 0

    def combined_boundary() -> Any:
        return (
            hidden_states.astype(jnp.float32)
            + residual.astype(jnp.float32)
        ).astype(hidden_states.dtype)

    for layer in stage.layers:
        if layer_residual_observation is not None:
            layer_residual_observation = layer_residual_observation.at[
                layer.layer_id
            ].set(combined_boundary()[0])
        attention = _attention_weights(
            weight,
            layer.stage_slot,
            attention_projection_backend,
        )
        input_norm = weight(
            f"attention.slot_{layer.stage_slot:02d}.input_norm"
        )
        post_norm = weight(
            f"attention.slot_{layer.stage_slot:02d}.post_norm"
        )
        if layer.indexer_kind == "full":
            dsa = _dsa_weights(weight, full_slot)
            query_weight_aliases = (
                None
                if dsa_query_weight_aliases is None
                else tuple(
                    alias[full_slot] for alias in dsa_query_weight_aliases
                )
            )
            recurrent_wk_weight = (
                None
                if dsa_recurrent_wk_weights is None
                else dsa_recurrent_wk_weights[full_slot]
            )
            layer_index_cache = index_cache[full_slot]
            current_full_slot = full_slot
            full_slot += 1
        else:
            dsa = None
            query_weight_aliases = None
            recurrent_wk_weight = None
            layer_index_cache = index_cache[0]
            current_full_slot = None
        if layer.mlp_kind == "dense":
            assert layer.dense_slot is not None
            dense = _dense_weights(
                weight,
                layer.dense_slot,
                final_layout_convolution=dense_final_layout_convolution,
            )
            moe = None
        else:
            assert layer.sparse_slot is not None
            dense = None
            moe = _moe_weights(weight, layer.sparse_slot)
        producer_valid = (
            jnp.asarray(True)
            if layer.indexer_kind == "full"
            else metadata[0, config.producer_index]
            == jnp.int32(layer.index_state_producer_layer)
        )
        incoming_valid = (
            (metadata[0, config.health_index] == jnp.int32(1))
            & producer_valid
        )[None]
        result = stage_local_transformer_layer_fp8_split_mapped(
            hidden_states,
            residual,
            kv_cache[layer.stage_slot],
            layer_index_cache,
            metadata[:, : config.selected_width],
            metadata[:, config.count_index],
            position,
            block_tables,
            context_lengths,
            input_norm,
            post_norm,
            attention,
            dsa,
            dense,
            moe,
            incoming_valid,
            local_slot,
            axis_name=axis_name,
            indexer_kind=layer.indexer_kind,
            mlp_kind=layer.mlp_kind,
            dsa_contract=dsa_contract,
            mla_contract=mla_contract,
            moe_contract=moe_contract,
            cache_layout=cache_layout,
            axis_index_groups=axis_groups,
            block_shape=block_shape,
            sparse_moe_backend=sparse_moe_backend,
            pallas_moe_config=pallas_moe_config,
            pallas_moe_fuse_route_weighting=pallas_moe_fuse_route_weighting,
            pallas_moe_reconstruct_down_fp32=(
                pallas_moe_reconstruct_down_fp32
            ),
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            dsa_query_weight_aliases=query_weight_aliases,
            dsa_precomputed_wk_weight=recurrent_wk_weight,
            dsa_head_key_exact_association=(
                dsa_recurrent_wk_weights is not None
            ),
            dsa_score_precision=(
                "default"
                if config.dsa_score_default_precision
                else "highest"
            ),
            attention_projection_backend=attention_projection_backend,
            pregathered_b512_attention=pregathered_b512_attention,
            virtual_tp32_reduction_association=(
                STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
                if strategy_nd_attention_projection
                else None
            ),
            virtual_tp32_attention_only=(
                strategy_nd_attention_projection
            ),
            main_rope_table_row=main_rope_table_row,
            dense_final_layout_convolution=dense_final_layout_convolution,
        )
        if current_full_slot is not None and prefill_index_inputs is not None:
            prefill_index_inputs = prefill_index_inputs.at[
                current_full_slot
            ].set(result.dsa_internals.normalized_hidden[0])
        hidden_states = result.hidden_states
        residual = result.residual
        if layer_residual_observation is not None:
            layer_residual_observation = layer_residual_observation.at[
                layer.layer_id + 1
            ].set(combined_boundary()[0])
        kv_cache = kv_cache.at[layer.stage_slot].set(result.kv_cache)
        if current_full_slot is not None:
            index_cache = index_cache.at[current_full_slot].set(
                result.index_cache
            )
            metadata = metadata.at[0, config.producer_index].set(
                jnp.int32(layer.layer_id)
            )
            if dsa_observation is not None:
                dsa_observation = dsa_observation.at[
                    current_full_slot, : config.selected_width
                ].set(result.selected_positions[0])
                dsa_observation = dsa_observation.at[
                    current_full_slot,
                    config.selected_width : 2 * config.selected_width,
                ].set(
                    lax.bitcast_convert_type(
                        result.selected_scores[0], jnp.int32
                    )
                )
                dsa_observation = dsa_observation.at[
                    current_full_slot, 2 * config.selected_width
                ].set(result.selected_valid_counts[0])
                dsa_observation = dsa_observation.at[
                    current_full_slot, 2 * config.selected_width + 1
                ].set(jnp.int32(layer.layer_id))
            if dsa_internal_observation is not None:
                internals = result.dsa_internals
                dsa_internal_observation = DsaInternalObservation(
                    dsa_internal_observation.normalized_hidden.at[
                        current_full_slot
                    ].set(internals.normalized_hidden[0]),
                    dsa_internal_observation.q_a_state.at[
                        current_full_slot
                    ].set(internals.q_a_state[0]),
                    dsa_internal_observation.query.at[
                        current_full_slot
                    ].set(internals.query[0]),
                    dsa_internal_observation.head_weights.at[
                        current_full_slot
                    ].set(internals.head_weights[0]),
                    dsa_internal_observation.current_key.at[
                        current_full_slot
                    ].set(internals.current_key[0]),
                    dsa_internal_observation.producer_layer_ids.at[
                        current_full_slot
                    ].set(jnp.int32(layer.layer_id)),
                )
        metadata = metadata.at[:, : config.selected_width].set(
            result.selected_positions
        )
        metadata = metadata.at[:, config.count_index].set(
            result.selected_valid_counts
        )
        metadata = metadata.at[0, config.health_index].set(
            result.contract_valid[0].astype(jnp.int32)
        )
    metadata = metadata.at[0, config.visited_index].set(
        jnp.bitwise_or(
            metadata[0, config.visited_index],
            jnp.int32(1 << stage.assignment.stage_id),
        )
    )
    outputs: tuple[Any, ...] = (
        jnp.stack((hidden_states, residual), axis=0),
        kv_cache,
        index_cache,
        metadata,
    )
    if dsa_observation is not None:
        outputs = (*outputs, dsa_observation)
    if dsa_internal_observation is not None:
        outputs = (*outputs, dsa_internal_observation)
    if layer_residual_observation is not None:
        outputs = (*outputs, layer_residual_observation)
    if prefill_index_inputs is not None:
        outputs = (*outputs, prefill_index_inputs)
    return outputs


def build_decoder_step_program(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    state_layout: DecoderStateLayout,
    weight_layout: DecoderRuntimeWeightLayout,
    groups: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
    *,
    devices: Sequence[Any] | None = None,
    axis_name: str = "device",
    sparse_moe_backend: SparseMoeBackend = "reference",
    feature_output_tile: int | None = None,
    feature_fuse_route_weighting: bool = False,
    feature_reconstruct_down_fp32: bool = False,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    dsa_query_exact_association: bool = False,
    dsa_head_key_exact_association: bool = False,
    dsa_score_default_precision: bool = False,
    attention_projection_backend: AttentionProjectionBackend = "separate",
    main_rope_table_enabled: bool = False,
    pregathered_b512_attention: bool = False,
    strategy_nd_attention_projection: bool = False,
    dense_final_layout_convolution: bool = False,
    complete_token_path: bool = False,
    observe_dsa_events: bool = False,
    observe_dsa_internals: bool = False,
    observe_layer_residuals: bool = False,
    observe_prefill_index_inputs: bool = False,
    build_layer0_residual_discriminator: bool = False,
    layer0_residual_discriminator_kind: (
        Layer0ResidualDiscriminatorKind
    ) = "combine_precision",
    build_layer0_ingredients_observer: bool = False,
    split_residual_state: bool = False,
) -> DecoderStepProgram:
    """Build, but do not compile, one all-stage decoder step.

    ``complete_token_path`` adds the sharded embedding and final
    norm/logits/greedy-token boundaries. ``split_residual_state`` preserves
    the accepted decoder's two live BF16 residual components so fused norms
    consume their unrounded FP32 sum. Both default off so the protected
    transformer-body executable and its input/output contract remain intact.
    """

    if not axis_name:
        raise PlanValidationError("decoder axis name must be explicit")
    if sparse_moe_backend not in ("reference", "pallas_feature"):
        raise PlanValidationError("decoder sparse MoE backend is unknown")
    if feature_output_tile is None:
        feature_output_tile = (
            PROMOTED_FEATURE_OUTPUT_TILE
            if sparse_moe_backend == "pallas_feature"
            else REFERENCE_FEATURE_OUTPUT_TILE
        )
    if feature_output_tile not in (128, 256):
        raise PlanValidationError("feature output tile must be 128 or 256")
    if not isinstance(feature_fuse_route_weighting, bool):
        raise PlanValidationError("feature route-weight fusion flag must be boolean")
    if not isinstance(feature_reconstruct_down_fp32, bool):
        raise PlanValidationError(
            "feature FP32 reconstruction flag must be boolean"
        )
    if sparse_moe_backend == "reference" and feature_output_tile != 128:
        raise PlanValidationError(
            "a non-default feature output tile requires pallas_feature"
        )
    if feature_fuse_route_weighting and sparse_moe_backend != "pallas_feature":
        raise PlanValidationError(
            "feature route-weight fusion requires pallas_feature"
        )
    if (
        feature_reconstruct_down_fp32
        and sparse_moe_backend != "pallas_feature"
    ):
        raise PlanValidationError(
            "feature FP32 reconstruction requires pallas_feature"
        )
    if feature_reconstruct_down_fp32 and feature_fuse_route_weighting:
        raise PlanValidationError(
            "feature FP32 reconstruction is incompatible with fused route "
            "weighting"
        )
    if linear_backend not in ("reference", "pallas"):
        raise PlanValidationError("decoder FP8 linear backend is unknown")
    if dsa_query_backend is None:
        dsa_query_backend = linear_backend
    if dsa_query_backend not in ("reference", "pallas"):
        raise PlanValidationError("decoder DSA query backend is unknown")
    if not isinstance(dsa_query_exact_association, bool):
        raise PlanValidationError(
            "decoder exact DSA query association flag must be boolean"
        )
    if dsa_query_exact_association and dsa_query_backend != "reference":
        raise PlanValidationError(
            "exact DSA query association requires the reference backend"
        )
    if not isinstance(dsa_head_key_exact_association, bool):
        raise PlanValidationError(
            "decoder exact DSA head/key association flag must be boolean"
        )
    if dsa_head_key_exact_association and not dsa_query_exact_association:
        raise PlanValidationError(
            "exact DSA head/key association requires exact DSA query"
        )
    if not isinstance(dsa_score_default_precision, bool):
        raise PlanValidationError(
            "decoder DSA score-precision flag must be boolean"
        )
    if dsa_score_default_precision and not dsa_head_key_exact_association:
        raise PlanValidationError(
            "default DSA score precision requires exact DSA head/key inputs"
        )
    if attention_projection_backend not in (
        "separate",
        "fused_n82_convolution",
    ):
        raise PlanValidationError(
            "decoder attention projection backend is unknown"
        )
    if not isinstance(main_rope_table_enabled, bool):
        raise PlanValidationError("decoder main-RoPE table flag must be boolean")
    if not isinstance(pregathered_b512_attention, bool):
        raise PlanValidationError(
            "decoder pregathered-B512 attention flag must be boolean"
        )
    if not isinstance(strategy_nd_attention_projection, bool):
        raise PlanValidationError(
            "decoder StrategyND attention-projection flag must be boolean"
        )
    if not isinstance(dense_final_layout_convolution, bool):
        raise PlanValidationError(
            "decoder dense final-layout convolution flag must be boolean"
        )
    if not isinstance(complete_token_path, bool):
        raise PlanValidationError("complete token-path flag must be boolean")
    if not isinstance(split_residual_state, bool):
        raise PlanValidationError("split residual-state flag must be boolean")
    if pregathered_b512_attention and not (
        plan.name is PlanName.PP8_LP4
        and plan.local_parallel_size == 4
        and main_rope_table_enabled
        and complete_token_path
        and split_residual_state
        and linear_backend == "pallas"
        and dsa_query_exact_association
        and dsa_head_key_exact_association
        and dsa_score_default_precision
        and attention_projection_backend == "fused_n82_convolution"
    ):
        raise PlanValidationError(
            "pregathered-B512 attention requires the protected PP8 exact "
            "split-token Pallas/main-RoPE/DSA/fused-qkv path"
        )
    if strategy_nd_attention_projection and not pregathered_b512_attention:
        raise PlanValidationError(
            "StrategyND attention projection requires the protected "
            "pregathered-B512 attention path"
        )
    if strategy_nd_attention_projection and not split_residual_state:
        raise PlanValidationError(
            "StrategyND attention projection requires split residual state"
        )
    if dense_final_layout_convolution and not (
        strategy_nd_attention_projection
        and linear_backend == "pallas"
        and split_residual_state
    ):
        raise PlanValidationError(
            "dense final-layout convolution requires the protected StrategyND "
            "split Pallas path"
        )
    if not isinstance(observe_dsa_events, bool):
        raise PlanValidationError("DSA event-observation flag must be boolean")
    if not isinstance(observe_dsa_internals, bool):
        raise PlanValidationError("DSA internal-observation flag must be boolean")
    if not isinstance(observe_layer_residuals, bool):
        raise PlanValidationError("layer residual-observation flag must be boolean")
    if not isinstance(observe_prefill_index_inputs, bool):
        raise PlanValidationError(
            "prefill index-input observation flag must be boolean"
        )
    if not isinstance(build_layer0_residual_discriminator, bool):
        raise PlanValidationError(
            "layer-0 residual discriminator flag must be boolean"
        )
    if not isinstance(build_layer0_ingredients_observer, bool):
        raise PlanValidationError(
            "layer-0 ingredient-observer flag must be boolean"
        )
    if layer0_residual_discriminator_kind not in (
        "combine_precision",
        "virtual_tp32",
        "attention_schedule",
        "attention_output_association",
        "strategy_nd_row0_association",
    ):
        raise PlanValidationError(
            "layer-0 residual discriminator kind is unknown"
        )
    if (
        not build_layer0_residual_discriminator
        and layer0_residual_discriminator_kind != "combine_precision"
    ):
        raise PlanValidationError(
            "non-default layer-0 discriminator kind requires its build flag"
        )
    if (
        layer0_residual_discriminator_kind
        in {"attention_schedule", "strategy_nd_row0_association"}
        and not main_rope_table_enabled
    ):
        raise PlanValidationError(
            "attention/StrategyND discriminator requires the proven main-RoPE table"
        )
    if (
        layer0_residual_discriminator_kind
        == "attention_output_association"
        and not main_rope_table_enabled
    ):
        raise PlanValidationError(
            "attention-output association discriminator requires the proven "
            "main-RoPE table"
        )
    if observe_dsa_events and not complete_token_path:
        raise PlanValidationError(
            "DSA event observation requires the complete-token path"
        )
    if observe_layer_residuals and not observe_dsa_events:
        raise PlanValidationError(
            "layer residual observation requires the isolated DSA observer"
        )
    if observe_dsa_internals and not observe_dsa_events:
        raise PlanValidationError(
            "DSA internal observation requires the isolated DSA observer"
        )
    if observe_dsa_internals and observe_layer_residuals:
        raise PlanValidationError(
            "DSA internal and layer-residual diagnostics must be isolated"
        )
    if observe_prefill_index_inputs and not complete_token_path:
        raise PlanValidationError(
            "prefill index-input observation requires the complete-token path"
        )
    if observe_prefill_index_inputs and (
        observe_dsa_events
        or observe_dsa_internals
        or observe_layer_residuals
    ):
        raise PlanValidationError(
            "prefill index-input observation must be isolated"
        )
    if build_layer0_residual_discriminator and not (
        complete_token_path and split_residual_state
    ):
        raise PlanValidationError(
            "layer-0 residual discriminator requires the complete split-token path"
        )
    if build_layer0_ingredients_observer and not (
        complete_token_path
        and split_residual_state
        and linear_backend == "pallas"
        and dsa_query_exact_association
        and dsa_head_key_exact_association
        and dsa_score_default_precision
        and attention_projection_backend == "fused_n82_convolution"
    ):
        raise PlanValidationError(
            "layer-0 ingredient observer requires the proven complete split-token "
            "BF16 Pallas/DSA/fused-qkv path"
        )
    if (
        main_rope_table_enabled
        and build_layer0_residual_discriminator
        and layer0_residual_discriminator_kind
        not in {
            "attention_schedule",
            "attention_output_association",
            "strategy_nd_row0_association",
        }
    ):
        raise PlanValidationError(
            "main-RoPE table execution admits only the isolated attention-"
            "schedule/output-association/StrategyND discriminator"
        )
    expected_expert_layout = (
        COMPLETE_EXPERT_RUNTIME_LAYOUT
        if sparse_moe_backend == "reference"
        else feature_expert_runtime_layout(plan.local_parallel_size)
    )
    if weight_layout.routed_expert_layout != expected_expert_layout:
        raise PlanValidationError(
            "decoder sparse MoE backend and runtime expert layout disagree: "
            f"backend={sparse_moe_backend!r} "
            f"expected={expected_expert_layout!r} "
            f"observed={weight_layout.routed_expert_layout!r}"
        )
    expected_attention_layout = (
        SEPARATE_QKV_A_RUNTIME_LAYOUT
        if attention_projection_backend == "separate"
        else FUSED_QKV_A_N82_RUNTIME_LAYOUT
    )
    if weight_layout.attention_projection_layout != expected_attention_layout:
        raise PlanValidationError(
            "decoder attention backend and runtime layout disagree: "
            f"backend={attention_projection_backend!r} "
            f"expected={expected_attention_layout!r} "
            f"observed={weight_layout.attention_projection_layout!r}"
        )
    expected_dense_layout = (
        FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT
        if dense_final_layout_convolution
        else LEGACY_DENSE_RUNTIME_LAYOUT
    )
    if weight_layout.dense_projection_layout != expected_dense_layout:
        raise PlanValidationError(
            "decoder dense backend and runtime layout disagree: "
            f"expected={expected_dense_layout!r} "
            f"observed={weight_layout.dense_projection_layout!r}"
        )
    hashes = (
        schedule.plan_hash,
        state_layout.plan_hash,
        weight_layout.plan_hash,
    )
    if hashes != (plan.plan_hash,) * 3:
        raise PlanValidationError("decoder components belong to different plans")
    if state_layout.schedule_hash != schedule.schedule_hash or (
        weight_layout.schedule_hash != schedule.schedule_hash
    ):
        raise PlanValidationError("decoder component schedule hashes disagree")
    geometry = plan.geometry
    pallas_moe_config = None
    if sparse_moe_backend == "pallas_feature":
        output_scale_blocks = feature_output_tile // 128
        pallas_moe_config = Fp8BlockMatmulConfig(
            block_shape=geometry.fp8_block_shape,
            row_tile=8,
            output_tile=(
                geometry.fp8_block_shape[0] * output_scale_blocks
            ),
            contraction_tile=geometry.fp8_block_shape[1] * 4,
        )
    config = DecoderStepConfig(
        stage_count=plan.pipeline_stages,
        local_parallel_size=plan.local_parallel_size,
        hidden_size=geometry.hidden_size,
        selected_width=geometry.dsa_top_k,
        context_capacity=state_layout.context_capacity,
        maximum_layer_slots=max(stage.layer_count for stage in schedule.stages),
        maximum_full_indexer_slots=max(
            stage.full_indexer_count for stage in state_layout.stages
        ),
        logical_page_size=state_layout.logical_page_size,
        local_rows_per_page=state_layout.stages[0].local_rows_per_page,
        packed_cache_width=state_layout.stages[0].packed_kv_width,
        index_key_width=state_layout.stages[0].index_key_width,
        dsa_indexer_heads=geometry.dsa_indexer_heads,
        vocab_size=geometry.vocab_size,
        main_rope_table_width=geometry.qk_rope_head_dim,
        dsa_score_default_precision=dsa_score_default_precision,
    )
    skeleton = PipelineSkeletonConfig(
        config.stage_count,
        config.local_parallel_size,
        config.hidden_size,
        config.selected_width,
    )
    canonical_groups = _canonical_groups(groups, skeleton)
    canonical_pairs = _canonical_pairs(pairs, canonical_groups, skeleton)
    stage_by_rank, slot_by_rank = _partition_maps(canonical_groups, skeleton)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    if len(runtime_devices) != config.total_devices:
        raise PlanValidationError(
            f"decoder expected {config.total_devices} devices, "
            f"found {len(runtime_devices)}"
        )
    if all(hasattr(device, "id") for device in runtime_devices):
        for stage, group in zip(schedule.stages, canonical_groups, strict=True):
            observed_ids = tuple(int(runtime_devices[rank].id) for rank in group)
            if observed_ids != stage.assignment.device_ids:
                raise PlanValidationError(
                    f"decoder runtime devices disagree with physical stage "
                    f"{stage.assignment.stage_id}: expected="
                    f"{stage.assignment.device_ids} observed={observed_ids}"
                )
    mesh = Mesh(np.asarray(runtime_devices, dtype=object), (axis_name,))
    stage_map = jnp.asarray(stage_by_rank, dtype=jnp.int32)
    slot_map = jnp.asarray(slot_by_rank, dtype=jnp.int32)
    axis_groups = tuple(tuple(group) for group in canonical_groups)
    dsa_contract = DsaNumericalContract(
        hidden_size=geometry.hidden_size,
        q_lora_rank=geometry.q_lora_rank,
        num_heads=geometry.dsa_indexer_heads,
        head_dim=geometry.dsa_indexer_head_dim,
        rotary_dim=geometry.qk_rope_head_dim,
        top_k=geometry.dsa_top_k,
    )
    mla_contract = MlaNumericalContract(
        num_heads=geometry.attention_heads,
        kv_lora_rank=geometry.kv_lora_rank,
        qk_nope_head_dim=geometry.qk_nope_head_dim,
        qk_rope_head_dim=geometry.qk_rope_head_dim,
        qk_head_dim=(
            geometry.qk_nope_head_dim + geometry.qk_rope_head_dim
        ),
        v_head_dim=geometry.v_head_dim,
        packed_cache_width=config.packed_cache_width,
        top_k=geometry.dsa_top_k,
    )
    moe_contract = GlmMoeNumericalContract(
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.moe_intermediate_size,
        num_experts=geometry.num_routed_experts,
        top_k=geometry.routed_top_k,
        stage_size=plan.local_parallel_size,
        fp8_block_shape=geometry.fp8_block_shape,
    )
    cache_layout = StageLocalKvLayout(
        logical_page_size=config.logical_page_size,
        local_parallel_size=config.local_parallel_size,
        packed_cache_width=config.packed_cache_width,
    )
    main_rope_table_host = None
    main_rope_table_digest = None
    main_rope_table_bytes = 0
    if main_rope_table_enabled:
        main_rope_table_host = build_rotary_table_host(
            config.context_capacity,
            rotary_dim=geometry.qk_rope_head_dim,
            theta=8_000_000.0,
        )
        main_rope_table_digest = rotary_table_sha256(main_rope_table_host)
        main_rope_table_bytes = int(main_rope_table_host.nbytes)

    local_vocab = geometry.vocab_size // config.local_parallel_size

    def mapped_impl(
        local_weights: Mapping[str, Any],
        local_exact_dsa_weights: Any | None,
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any | None,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any | None,
    ) -> tuple[Any, ...]:
        rank = lax.axis_index(axis_name)
        stage_id = stage_map[rank]
        local_slot = slot_map[rank]
        residual = local_residual_container[0]
        kv_cache = local_kv_container[0]
        index_cache = local_index_container[0]
        metadata = local_metadata_container[0]
        dsa_observation = None
        dsa_internal_observation = None
        token_observation = None
        layer_residual_observation = None
        prefill_index_inputs = None
        main_rope_table_row = None
        if main_rope_table_enabled:
            if (
                local_main_rope_table is None
                or local_main_rope_table.shape
                != (config.context_capacity, geometry.qk_rope_head_dim)
                or local_main_rope_table.dtype != jnp.bfloat16
            ):
                raise PlanValidationError(
                    "decoder main-RoPE table asset is invalid"
                )
            safe_rope_position = jnp.clip(
                position[0],
                jnp.int32(0),
                jnp.int32(config.context_capacity - 1),
            )
            with jax.named_scope("greenfield_main_rope_table_lookup"):
                main_rope_table_row = local_main_rope_table[
                    safe_rope_position
                ]
        elif local_main_rope_table is not None:
            raise PlanValidationError(
                "default decoder cannot consume a main-RoPE table"
            )
        if observe_prefill_index_inputs:
            prefill_index_inputs = jnp.zeros(
                (
                    config.maximum_full_indexer_slots,
                    config.hidden_size,
                ),
                dtype=residual.dtype,
            )
        if observe_dsa_events:
            dsa_observation = jnp.full(
                (
                    config.maximum_full_indexer_slots,
                    config.dsa_observation_width,
                ),
                -1,
                dtype=jnp.int32,
            )
            token_observation = jnp.full(
                (config.token_observation_width,),
                -1,
                dtype=jnp.int32,
            )
            if observe_dsa_internals:
                slots = config.maximum_full_indexer_slots
                dsa_internal_observation = DsaInternalObservation(
                    jnp.zeros(
                        (slots, config.hidden_size), dtype=residual.dtype
                    ),
                    jnp.zeros(
                        (slots, dsa_contract.q_lora_rank),
                        dtype=residual.dtype,
                    ),
                    jnp.zeros(
                        (
                            slots,
                            dsa_contract.num_heads,
                            dsa_contract.head_dim,
                        ),
                        dtype=jnp.float32,
                    ),
                    jnp.zeros(
                        (slots, dsa_contract.num_heads), dtype=jnp.float32
                    ),
                    jnp.zeros(
                        (slots, dsa_contract.head_dim), dtype=jnp.float32
                    ),
                    jnp.full((slots,), -1, dtype=jnp.int32),
                )
            if observe_layer_residuals:
                layer_residual_observation = jnp.zeros(
                    (geometry.num_layers + 1, geometry.hidden_size),
                    dtype=residual.dtype,
                )

        def weight(name: str) -> Any:
            return local_weights[name][0]

        local_dsa_query_weight_aliases = local_exact_dsa_weights
        local_dsa_recurrent_wk_weights = None
        if dsa_head_key_exact_association:
            if (
                local_exact_dsa_weights is None
                or len(local_exact_dsa_weights) != 2
            ):
                raise PlanValidationError(
                    "exact DSA head/key execution requires query and wk state"
                )
            (
                local_dsa_query_weight_aliases,
                local_dsa_recurrent_wk_weights,
            ) = local_exact_dsa_weights
        dsa_query_weight_aliases = (
            None
            if local_dsa_query_weight_aliases is None
            else tuple(
                tuple(value[0] for value in alias)
                for alias in local_dsa_query_weight_aliases
            )
        )
        dsa_recurrent_wk_weights = (
            None
            if local_dsa_recurrent_wk_weights is None
            else tuple(value[0] for value in local_dsa_recurrent_wk_weights)
        )
        if dsa_query_exact_association and (
            dsa_query_weight_aliases is None
            or len(dsa_query_weight_aliases) != 4
        ):
            raise PlanValidationError(
                "exact DSA query execution requires four owner aliases"
            )
        if dsa_head_key_exact_association and (
            dsa_recurrent_wk_weights is None
            or len(dsa_recurrent_wk_weights)
            != config.maximum_full_indexer_slots
        ):
            raise PlanValidationError(
                "exact DSA head/key execution lost local wk owners"
            )

        execute_stage = (
            _execute_stage_split
            if split_residual_state
            else _execute_stage
        )

        def normalize_final(final_residual: Any) -> Any:
            if split_residual_state:
                normalized, _ = fused_add_rms_norm(
                    final_residual[0],
                    final_residual[1],
                    weight("global.final_norm"),
                    epsilon=1e-5,
                )
                return normalized
            return final_norm(
                final_residual,
                weight("global.final_norm"),
                epsilon=1e-5,
            )

        next_token = jnp.full((1,), -1, dtype=jnp.int32)
        if complete_token_path:
            assert local_token_container is not None
            token_id = local_token_container[0, 0]
            active = metadata[0, config.active_index] == jnp.int32(1)
            should_embed = active & (stage_id == jnp.int32(0))

            def embed_token(values: tuple[Any, Any]) -> tuple[Any, Any]:
                current_residual, current_metadata = values
                local_start = local_slot * jnp.int32(local_vocab)
                local_end = local_start + jnp.int32(local_vocab)
                token_valid = (
                    (token_id >= jnp.int32(0))
                    & (token_id < jnp.int32(config.vocab_size))
                )
                owns_token = (
                    token_valid
                    & (token_id >= local_start)
                    & (token_id < local_end)
                )
                local_id = jnp.clip(
                    token_id - local_start,
                    jnp.int32(0),
                    jnp.int32(local_vocab - 1),
                )
                local_row = weight("global.embedding")[local_id][None, :]
                local_row = jnp.where(
                    owns_token,
                    local_row,
                    jnp.zeros_like(local_row),
                )
                embedded = lax.psum(
                    local_row,
                    axis_name,
                    axis_index_groups=axis_groups,
                )
                if split_residual_state:
                    current_residual = jnp.stack(
                        (embedded, jnp.zeros_like(embedded)), axis=0
                    )
                else:
                    current_residual = embedded
                current_metadata = current_metadata.at[
                    0, config.visited_index
                ].set(jnp.int32(0))
                current_metadata = current_metadata.at[
                    0, config.health_index
                ].set(
                    (
                        current_metadata[0, config.health_index]
                        == jnp.int32(1)
                    ).astype(jnp.int32)
                    * token_valid.astype(jnp.int32)
                )
                return current_residual, current_metadata

            residual, metadata = lax.cond(
                should_embed,
                embed_token,
                lambda values: values,
                (residual, metadata),
            )

        for hop, stage in enumerate(schedule.stages):
            active = metadata[0, config.active_index] == jnp.int32(1)
            should_execute = active & (stage_id == jnp.int32(hop))
            if observe_dsa_events:
                assert dsa_observation is not None
                if observe_layer_residuals:
                    assert layer_residual_observation is not None
                    (
                        residual,
                        kv_cache,
                        index_cache,
                        metadata,
                        dsa_observation,
                        layer_residual_observation,
                    ) = lax.cond(
                        should_execute,
                        lambda values, stage=stage: execute_stage(
                            stage,
                            values[:4],
                            weight=weight,
                            local_slot=local_slot,
                            position=position,
                            block_tables=block_tables,
                            context_lengths=context_lengths,
                            axis_name=axis_name,
                            axis_groups=axis_groups,
                            config=config,
                            dsa_contract=dsa_contract,
                            mla_contract=mla_contract,
                            moe_contract=moe_contract,
                            cache_layout=cache_layout,
                            block_shape=geometry.fp8_block_shape,
                            sparse_moe_backend=sparse_moe_backend,
                            pallas_moe_config=pallas_moe_config,
                            pallas_moe_fuse_route_weighting=(
                                feature_fuse_route_weighting
                            ),
                            pallas_moe_reconstruct_down_fp32=(
                                feature_reconstruct_down_fp32
                            ),
                            linear_backend=linear_backend,
                            dsa_query_backend=dsa_query_backend,
                            dsa_query_weight_aliases=(
                                dsa_query_weight_aliases
                            ),
                            dsa_recurrent_wk_weights=(
                                dsa_recurrent_wk_weights
                            ),
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
                            pregathered_b512_attention=(
                                pregathered_b512_attention
                            ),
                            strategy_nd_attention_projection=(
                                strategy_nd_attention_projection
                            ),
                            dense_final_layout_convolution=(
                                dense_final_layout_convolution
                            ),
                            main_rope_table_row=main_rope_table_row,
                            dsa_observation=values[4],
                            layer_residual_observation=values[5],
                        ),
                        lambda values: values,
                        (
                            residual,
                            kv_cache,
                            index_cache,
                            metadata,
                            dsa_observation,
                            layer_residual_observation,
                        ),
                    )
                elif observe_dsa_internals:
                    assert dsa_internal_observation is not None
                    (
                        residual,
                        kv_cache,
                        index_cache,
                        metadata,
                        dsa_observation,
                        dsa_internal_observation,
                    ) = lax.cond(
                        should_execute,
                        lambda values, stage=stage: execute_stage(
                            stage,
                            values[:4],
                            weight=weight,
                            local_slot=local_slot,
                            position=position,
                            block_tables=block_tables,
                            context_lengths=context_lengths,
                            axis_name=axis_name,
                            axis_groups=axis_groups,
                            config=config,
                            dsa_contract=dsa_contract,
                            mla_contract=mla_contract,
                            moe_contract=moe_contract,
                            cache_layout=cache_layout,
                            block_shape=geometry.fp8_block_shape,
                            sparse_moe_backend=sparse_moe_backend,
                            pallas_moe_config=pallas_moe_config,
                            pallas_moe_fuse_route_weighting=(
                                feature_fuse_route_weighting
                            ),
                            pallas_moe_reconstruct_down_fp32=(
                                feature_reconstruct_down_fp32
                            ),
                            linear_backend=linear_backend,
                            dsa_query_backend=dsa_query_backend,
                            dsa_query_weight_aliases=(
                                dsa_query_weight_aliases
                            ),
                            dsa_recurrent_wk_weights=(
                                dsa_recurrent_wk_weights
                            ),
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
                            pregathered_b512_attention=(
                                pregathered_b512_attention
                            ),
                            strategy_nd_attention_projection=(
                                strategy_nd_attention_projection
                            ),
                            dense_final_layout_convolution=(
                                dense_final_layout_convolution
                            ),
                            main_rope_table_row=main_rope_table_row,
                            dsa_observation=values[4],
                            dsa_internal_observation=values[5],
                        ),
                        lambda values: values,
                        (
                            residual,
                            kv_cache,
                            index_cache,
                            metadata,
                            dsa_observation,
                            dsa_internal_observation,
                        ),
                    )
                else:
                    (
                        residual,
                        kv_cache,
                        index_cache,
                        metadata,
                        dsa_observation,
                    ) = lax.cond(
                        should_execute,
                        lambda values, stage=stage: execute_stage(
                            stage,
                            values[:4],
                            weight=weight,
                            local_slot=local_slot,
                            position=position,
                            block_tables=block_tables,
                            context_lengths=context_lengths,
                            axis_name=axis_name,
                            axis_groups=axis_groups,
                            config=config,
                            dsa_contract=dsa_contract,
                            mla_contract=mla_contract,
                            moe_contract=moe_contract,
                            cache_layout=cache_layout,
                            block_shape=geometry.fp8_block_shape,
                            sparse_moe_backend=sparse_moe_backend,
                            pallas_moe_config=pallas_moe_config,
                            pallas_moe_fuse_route_weighting=(
                                feature_fuse_route_weighting
                            ),
                            pallas_moe_reconstruct_down_fp32=(
                                feature_reconstruct_down_fp32
                            ),
                            linear_backend=linear_backend,
                            dsa_query_backend=dsa_query_backend,
                            dsa_query_weight_aliases=(
                                dsa_query_weight_aliases
                            ),
                            dsa_recurrent_wk_weights=(
                                dsa_recurrent_wk_weights
                            ),
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
                            pregathered_b512_attention=(
                                pregathered_b512_attention
                            ),
                            strategy_nd_attention_projection=(
                                strategy_nd_attention_projection
                            ),
                            dense_final_layout_convolution=(
                                dense_final_layout_convolution
                            ),
                            main_rope_table_row=main_rope_table_row,
                            dsa_observation=values[4],
                        ),
                        lambda values: values,
                        (
                            residual,
                            kv_cache,
                            index_cache,
                            metadata,
                            dsa_observation,
                        ),
                    )
            elif observe_prefill_index_inputs:
                assert prefill_index_inputs is not None
                (
                    residual,
                    kv_cache,
                    index_cache,
                    metadata,
                    prefill_index_inputs,
                ) = lax.cond(
                    should_execute,
                    lambda values, stage=stage: execute_stage(
                        stage,
                        values[:4],
                        weight=weight,
                        local_slot=local_slot,
                        position=position,
                        block_tables=block_tables,
                        context_lengths=context_lengths,
                        axis_name=axis_name,
                        axis_groups=axis_groups,
                        config=config,
                        dsa_contract=dsa_contract,
                        mla_contract=mla_contract,
                        moe_contract=moe_contract,
                        cache_layout=cache_layout,
                        block_shape=geometry.fp8_block_shape,
                        sparse_moe_backend=sparse_moe_backend,
                        pallas_moe_config=pallas_moe_config,
                        pallas_moe_fuse_route_weighting=(
                            feature_fuse_route_weighting
                        ),
                        pallas_moe_reconstruct_down_fp32=(
                            feature_reconstruct_down_fp32
                        ),
                        linear_backend=linear_backend,
                        dsa_query_backend=dsa_query_backend,
                        dsa_query_weight_aliases=dsa_query_weight_aliases,
                        dsa_recurrent_wk_weights=dsa_recurrent_wk_weights,
                        attention_projection_backend=(
                            attention_projection_backend
                        ),
                        pregathered_b512_attention=(
                            pregathered_b512_attention
                        ),
                        strategy_nd_attention_projection=(
                            strategy_nd_attention_projection
                        ),
                        dense_final_layout_convolution=(
                            dense_final_layout_convolution
                        ),
                        main_rope_table_row=main_rope_table_row,
                        prefill_index_inputs=values[4],
                    ),
                    lambda values: values,
                    (
                        residual,
                        kv_cache,
                        index_cache,
                        metadata,
                        prefill_index_inputs,
                    ),
                )
            else:
                residual, kv_cache, index_cache, metadata = lax.cond(
                    should_execute,
                    lambda values, stage=stage: execute_stage(
                        stage,
                        values,
                        weight=weight,
                        local_slot=local_slot,
                        position=position,
                        block_tables=block_tables,
                        context_lengths=context_lengths,
                        axis_name=axis_name,
                        axis_groups=axis_groups,
                        config=config,
                        dsa_contract=dsa_contract,
                        mla_contract=mla_contract,
                        moe_contract=moe_contract,
                        cache_layout=cache_layout,
                        block_shape=geometry.fp8_block_shape,
                        sparse_moe_backend=sparse_moe_backend,
                        pallas_moe_config=pallas_moe_config,
                        pallas_moe_fuse_route_weighting=(
                            feature_fuse_route_weighting
                        ),
                        pallas_moe_reconstruct_down_fp32=(
                            feature_reconstruct_down_fp32
                        ),
                        linear_backend=linear_backend,
                        dsa_query_backend=dsa_query_backend,
                        dsa_query_weight_aliases=dsa_query_weight_aliases,
                        dsa_recurrent_wk_weights=dsa_recurrent_wk_weights,
                        attention_projection_backend=(
                            attention_projection_backend
                        ),
                        pregathered_b512_attention=(
                            pregathered_b512_attention
                        ),
                        strategy_nd_attention_projection=(
                            strategy_nd_attention_projection
                        ),
                        dense_final_layout_convolution=(
                            dense_final_layout_convolution
                        ),
                        main_rope_table_row=main_rope_table_row,
                    ),
                    lambda values: values,
                    (residual, kv_cache, index_cache, metadata),
                )
            if complete_token_path and hop == config.stage_count - 1:
                if observe_dsa_events:

                    def sample_token_observer(
                        values: tuple[Any, Any, Any],
                    ) -> tuple[Any, Any, Any]:
                        (
                            final_residual,
                            final_metadata,
                            current_token_observation,
                        ) = values
                        normalized = normalize_final(final_residual)
                        local_logits = vocabulary_logits(
                            normalized,
                            weight("global.lm_head"),
                        )
                        finite = jnp.all(jnp.isfinite(local_logits))
                        safe_logits = jnp.where(
                            jnp.isfinite(local_logits),
                            local_logits,
                            jnp.asarray(
                                -jnp.inf, dtype=local_logits.dtype
                            ),
                        )
                        candidate_width = config.token_observation_candidates
                        local_scores, local_indices = lax.top_k(
                            safe_logits[0], candidate_width
                        )
                        local_indices = local_indices.astype(jnp.int32)
                        global_indices = (
                            local_slot * jnp.int32(local_vocab)
                            + local_indices
                        )
                        candidate_score = jnp.where(
                            finite,
                            local_scores,
                            jnp.full_like(local_scores, -jnp.inf),
                        )
                        candidate_index = jnp.where(
                            finite,
                            global_indices,
                            jnp.full_like(
                                global_indices,
                                jnp.int32(config.vocab_size),
                            ),
                        )
                        scores = lax.all_gather(
                            candidate_score,
                            axis_name,
                            axis=0,
                            axis_index_groups=axis_groups,
                        )
                        indices = lax.all_gather(
                            candidate_index,
                            axis_name,
                            axis=0,
                            axis_index_groups=axis_groups,
                        )
                        flat_scores = scores.reshape(
                            1,
                            config.local_parallel_size * candidate_width,
                        )
                        flat_indices = indices.reshape(
                            1,
                            config.local_parallel_size * candidate_width,
                        )
                        index_order = jnp.argsort(
                            flat_indices, axis=1, stable=True
                        )
                        sorted_scores = jnp.take_along_axis(
                            flat_scores, index_order, axis=1
                        )
                        sorted_indices = jnp.take_along_axis(
                            flat_indices, index_order, axis=1
                        )
                        selected_scores, selected_slots = lax.top_k(
                            sorted_scores, candidate_width
                        )
                        selected_indices = jnp.take_along_axis(
                            sorted_indices, selected_slots, axis=1
                        ).astype(jnp.int32)
                        chosen = selected_indices[0, :1]
                        current_token_observation = jnp.concatenate(
                            (
                                selected_indices[0],
                                lax.bitcast_convert_type(
                                    selected_scores[0].astype(jnp.float32),
                                    jnp.int32,
                                ),
                            )
                        )
                        head_valid = jnp.all(
                            indices < jnp.int32(config.vocab_size)
                        )
                        final_metadata = final_metadata.at[
                            0, config.health_index
                        ].set(
                            (
                                final_metadata[0, config.health_index]
                                == jnp.int32(1)
                            ).astype(jnp.int32)
                            * head_valid.astype(jnp.int32)
                        )
                        return (
                            chosen,
                            final_metadata,
                            current_token_observation,
                        )

                    assert token_observation is not None
                    next_token, metadata, token_observation = lax.cond(
                        should_execute,
                        sample_token_observer,
                        lambda values: (next_token, values[1], values[2]),
                        (residual, metadata, token_observation),
                    )
                else:

                    def sample_token(
                        values: tuple[Any, Any],
                    ) -> tuple[Any, Any]:
                        final_residual, final_metadata = values
                        normalized = normalize_final(final_residual)
                        local_logits = vocabulary_logits(
                            normalized,
                            weight("global.lm_head"),
                        )
                        finite = jnp.all(jnp.isfinite(local_logits))
                        safe_logits = jnp.where(
                            jnp.isfinite(local_logits),
                            local_logits,
                            jnp.asarray(
                                -jnp.inf, dtype=local_logits.dtype
                            ),
                        )
                        local_index = jnp.argmax(
                            safe_logits[0], axis=0
                        ).astype(jnp.int32)
                        local_score = safe_logits[0, local_index][None]
                        global_index = (
                            local_slot * jnp.int32(local_vocab) + local_index
                        )[None]
                        candidate_score = jnp.where(
                            finite,
                            local_score,
                            jnp.asarray(
                                [-jnp.inf], dtype=local_score.dtype
                            ),
                        )
                        candidate_index = jnp.where(
                            finite,
                            global_index,
                            jnp.asarray(
                                [config.vocab_size], dtype=jnp.int32
                            ),
                        )
                        scores = lax.all_gather(
                            candidate_score,
                            axis_name,
                            axis=0,
                            axis_index_groups=axis_groups,
                        )
                        indices = lax.all_gather(
                            candidate_index,
                            axis_name,
                            axis=0,
                            axis_index_groups=axis_groups,
                        )
                        winning_score = jnp.max(scores, axis=0)
                        chosen = jnp.min(
                            jnp.where(
                                scores == winning_score[None, ...],
                                indices,
                                jnp.int32(config.vocab_size),
                            ),
                            axis=0,
                        ).astype(jnp.int32)
                        head_valid = jnp.all(
                            indices < jnp.int32(config.vocab_size)
                        )
                        final_metadata = final_metadata.at[
                            0, config.health_index
                        ].set(
                            (
                                final_metadata[0, config.health_index]
                                == jnp.int32(1)
                            ).astype(jnp.int32)
                            * head_valid.astype(jnp.int32)
                        )
                        return chosen, final_metadata

                    next_token, metadata = lax.cond(
                        should_execute,
                        sample_token,
                        lambda values: (next_token, values[1]),
                        (residual, metadata),
                    )
            residual = lax.ppermute(residual, axis_name, canonical_pairs)
            metadata = lax.ppermute(metadata, axis_name, canonical_pairs)
            if complete_token_path and hop == config.stage_count - 1:
                next_token = lax.ppermute(
                    next_token, axis_name, canonical_pairs
                )
        outputs = (
            residual[None, ...],
            kv_cache[None, ...],
            index_cache[None, ...],
            metadata[None, ...],
            next_token[None, ...],
            position + jnp.ones_like(position),
            block_tables,
            context_lengths + jnp.ones_like(context_lengths),
        )
        if not observe_dsa_events:
            if not observe_prefill_index_inputs:
                return outputs
            assert prefill_index_inputs is not None
            return (*outputs, prefill_index_inputs[None, ...])
        assert dsa_observation is not None
        assert token_observation is not None
        observation_outputs = (
            *outputs,
            dsa_observation[None, ...],
            token_observation[None, ...],
        )
        if not observe_layer_residuals:
            if not observe_dsa_internals:
                return observation_outputs
            assert dsa_internal_observation is not None
            return (
                *observation_outputs,
                DsaInternalObservation(
                    *(value[None, ...] for value in dsa_internal_observation)
                ),
            )
        assert layer_residual_observation is not None
        return (
            *observation_outputs,
            layer_residual_observation[None, ...],
        )

    def mapped_body(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
    ) -> tuple[Any, Any, Any, Any]:
        values = mapped_impl(
            local_weights,
            None,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            None,
            position,
            block_tables,
            context_lengths,
            None,
        )
        return values[:4]

    def mapped_token(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
    ) -> tuple[Any, ...]:
        return mapped_impl(
            local_weights,
            None,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            local_token_container,
            position,
            block_tables,
            context_lengths,
            None,
        )

    def mapped_body_exact_query(
        local_weights: Mapping[str, Any],
        local_query_weight_aliases: tuple[tuple[Any, ...], ...],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
    ) -> tuple[Any, Any, Any, Any]:
        values = mapped_impl(
            local_weights,
            local_query_weight_aliases,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            None,
            position,
            block_tables,
            context_lengths,
            None,
        )
        return values[:4]

    def mapped_token_exact_query(
        local_weights: Mapping[str, Any],
        local_query_weight_aliases: tuple[tuple[Any, ...], ...],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
    ) -> tuple[Any, ...]:
        return mapped_impl(
            local_weights,
            local_query_weight_aliases,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            local_token_container,
            position,
            block_tables,
            context_lengths,
            None,
        )

    def mapped_body_main_rope(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any,
    ) -> tuple[Any, Any, Any, Any]:
        values = mapped_impl(
            local_weights,
            None,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            None,
            position,
            block_tables,
            context_lengths,
            local_main_rope_table,
        )
        return values[:4]

    def mapped_token_main_rope(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any,
    ) -> tuple[Any, ...]:
        return mapped_impl(
            local_weights,
            None,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            local_token_container,
            position,
            block_tables,
            context_lengths,
            local_main_rope_table,
        )

    def mapped_body_exact_query_main_rope(
        local_weights: Mapping[str, Any],
        local_query_weight_aliases: tuple[tuple[Any, ...], ...],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any,
    ) -> tuple[Any, Any, Any, Any]:
        values = mapped_impl(
            local_weights,
            local_query_weight_aliases,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            None,
            position,
            block_tables,
            context_lengths,
            local_main_rope_table,
        )
        return values[:4]

    def mapped_token_exact_query_main_rope(
        local_weights: Mapping[str, Any],
        local_query_weight_aliases: tuple[tuple[Any, ...], ...],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any,
    ) -> tuple[Any, ...]:
        return mapped_impl(
            local_weights,
            local_query_weight_aliases,
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            local_token_container,
            position,
            block_tables,
            context_lengths,
            local_main_rope_table,
        )

    def mapped_layer0_residual_discriminator_impl(
        local_weights: Mapping[str, Any],
        local_exact_dsa_weights: Any | None,
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any | None,
        *,
        variant_name: str,
        reconstruct_attention_output_fp32: bool,
        reconstruct_dense_down_fp32: bool,
        virtual_tp32_reduction_association: (
            VirtualTp32ReductionAssociation | None
        ),
        virtual_tp32_attention_only: bool,
        replicated_monolithic_attention: bool,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Replay one isolated layer-0 arithmetic association."""

        rank = lax.axis_index(axis_name)
        stage_id = stage_map[rank]
        local_slot = slot_map[rank]
        residual_state = local_residual_container[0]
        kv_cache = local_kv_container[0]
        index_cache = local_index_container[0]
        metadata = local_metadata_container[0]
        main_rope_table_row = None
        if main_rope_table_enabled:
            if (
                local_main_rope_table is None
                or local_main_rope_table.shape
                != (config.context_capacity, geometry.qk_rope_head_dim)
                or local_main_rope_table.dtype != jnp.bfloat16
            ):
                raise PlanValidationError(
                    "layer-0 discriminator main-RoPE table is invalid"
                )
            safe_rope_position = jnp.clip(
                position[0],
                jnp.int32(0),
                jnp.int32(config.context_capacity - 1),
            )
            with jax.named_scope("greenfield_main_rope_table_lookup"):
                main_rope_table_row = local_main_rope_table[
                    safe_rope_position
                ]
        elif local_main_rope_table is not None:
            raise PlanValidationError(
                "default layer-0 discriminator cannot consume a main-RoPE table"
            )

        def weight(name: str) -> Any:
            return local_weights[name][0]

        token_id = local_token_container[0, 0]
        local_start = local_slot * jnp.int32(local_vocab)
        local_end = local_start + jnp.int32(local_vocab)
        token_valid = (
            (token_id >= jnp.int32(0))
            & (token_id < jnp.int32(config.vocab_size))
        )
        owns_token = (
            token_valid
            & (token_id >= local_start)
            & (token_id < local_end)
        )
        local_id = jnp.clip(
            token_id - local_start,
            jnp.int32(0),
            jnp.int32(local_vocab - 1),
        )
        local_row = weight("global.embedding")[local_id][None, :]
        local_row = jnp.where(
            owns_token, local_row, jnp.zeros_like(local_row)
        )
        embedded = lax.psum(
            local_row,
            axis_name,
            axis_index_groups=axis_groups,
        )

        selected_positions = jnp.full(
            (config.selected_width,), -1, dtype=jnp.int32
        )
        selected_scores = jnp.full(
            (config.selected_width,), -jnp.inf, dtype=jnp.float32
        )
        selected_count = jnp.zeros((1,), dtype=jnp.int32)
        normalized_hidden = jnp.zeros(
            (config.hidden_size,), dtype=residual_state.dtype
        )
        contract_valid = jnp.zeros((1,), dtype=jnp.bool_)

        local_query_weight_aliases = local_exact_dsa_weights
        local_recurrent_wk_weights = None
        if dsa_head_key_exact_association:
            if (
                local_exact_dsa_weights is None
                or len(local_exact_dsa_weights) != 2
            ):
                raise PlanValidationError(
                    "layer-0 discriminator requires query and wk state"
                )
            (
                local_query_weight_aliases,
                local_recurrent_wk_weights,
            ) = local_exact_dsa_weights
        query_weight_aliases = (
            None
            if local_query_weight_aliases is None
            else tuple(
                tuple(value[0] for value in alias)
                for alias in local_query_weight_aliases
            )
        )
        recurrent_wk_weights = (
            None
            if local_recurrent_wk_weights is None
            else tuple(value[0] for value in local_recurrent_wk_weights)
        )

        def execute_layer0(
            reconstruct_attention_output_fp32: bool,
            reconstruct_dense_down_fp32: bool,
            virtual_tp32_reduction_association: (
                VirtualTp32ReductionAssociation | None
            ),
            virtual_tp32_attention_only: bool,
            replicated_monolithic_attention: bool,
        ) -> Any:
            stage = schedule.stages[0]
            layer = stage.layers[0]
            if (
                layer.layer_id != 0
                or layer.indexer_kind != "full"
                or layer.mlp_kind != "dense"
                or layer.dense_slot is None
            ):
                raise PlanValidationError(
                    "layer-0 discriminator requires the pinned dense/full layer"
                )
            attention = _attention_weights(
                weight,
                layer.stage_slot,
                attention_projection_backend,
            )
            return stage_local_transformer_layer_fp8_split_mapped(
                embedded,
                jnp.zeros_like(embedded),
                kv_cache[layer.stage_slot],
                index_cache[0],
                metadata[:, : config.selected_width],
                metadata[:, config.count_index],
                position,
                block_tables,
                context_lengths,
                weight(
                    f"attention.slot_{layer.stage_slot:02d}.input_norm"
                ),
                weight(
                    f"attention.slot_{layer.stage_slot:02d}.post_norm"
                ),
                attention,
                _dsa_weights(weight, 0),
                _dense_weights(weight, layer.dense_slot),
                None,
                (
                    metadata[0, config.health_index] == jnp.int32(1)
                )[None],
                local_slot,
                axis_name=axis_name,
                indexer_kind="full",
                mlp_kind="dense",
                dsa_contract=dsa_contract,
                mla_contract=mla_contract,
                moe_contract=moe_contract,
                cache_layout=cache_layout,
                axis_index_groups=axis_groups,
                block_shape=geometry.fp8_block_shape,
                sparse_moe_backend=sparse_moe_backend,
                pallas_moe_config=pallas_moe_config,
                pallas_moe_fuse_route_weighting=(
                    feature_fuse_route_weighting
                ),
                pallas_moe_reconstruct_down_fp32=(
                    feature_reconstruct_down_fp32
                ),
                linear_backend=linear_backend,
                dsa_query_backend=dsa_query_backend,
                dsa_query_weight_aliases=(
                    None
                    if query_weight_aliases is None
                    else tuple(alias[0] for alias in query_weight_aliases)
                ),
                dsa_precomputed_wk_weight=(
                    None
                    if recurrent_wk_weights is None
                    else recurrent_wk_weights[0]
                ),
                dsa_head_key_exact_association=(
                    recurrent_wk_weights is not None
                ),
                dsa_score_precision=(
                    "default"
                    if config.dsa_score_default_precision
                    else "highest"
                ),
                attention_projection_backend=attention_projection_backend,
                main_rope_table_row=main_rope_table_row,
                reconstruct_attention_output_fp32=(
                    reconstruct_attention_output_fp32
                ),
                reconstruct_dense_down_fp32=(
                    reconstruct_dense_down_fp32
                ),
                virtual_tp32_reduction_association=(
                    virtual_tp32_reduction_association
                ),
                virtual_tp32_attention_only=virtual_tp32_attention_only,
                replicated_monolithic_attention=(
                    replicated_monolithic_attention
                ),
            )

        def stage0_branch(
            values: tuple[Any, Any, Any, Any, Any],
        ) -> tuple[Any, Any, Any, Any, Any]:
            del values
            candidate = execute_layer0(
                reconstruct_attention_output_fp32,
                reconstruct_dense_down_fp32,
                virtual_tp32_reduction_association,
                virtual_tp32_attention_only,
                replicated_monolithic_attention,
            )
            stage = schedule.stages[0]
            if len(stage.layers) < 2:
                raise PlanValidationError(
                    "layer-0 discriminator requires layer 1 on stage 0"
                )
            next_layer = stage.layers[1]
            next_input_norm = weight(
                f"attention.slot_{next_layer.stage_slot:02d}.input_norm"
            )
            with jax.named_scope(
                f"greenfield_layer1_input_norm_{variant_name}"
            ):
                normalized = fused_add_rms_norm(
                    candidate.hidden_states,
                    candidate.residual,
                    next_input_norm,
                    epsilon=1e-5,
                )[0][0]
            return (
                candidate.selected_positions[0],
                candidate.selected_scores[0],
                candidate.selected_valid_counts,
                normalized,
                candidate.contract_valid & token_valid[None],
            )

        values = (
            selected_positions,
            selected_scores,
            selected_count,
            normalized_hidden,
            contract_valid,
        )
        values = lax.cond(
            stage_id == jnp.int32(0),
            stage0_branch,
            lambda current: current,
            values,
        )
        return tuple(value[None, ...] for value in values)

    def make_mapped_layer0_residual_discriminator(
        variant_name: str,
        reconstruct_attention_output_fp32: bool,
        reconstruct_dense_down_fp32: bool,
        virtual_tp32_reduction_association: (
            VirtualTp32ReductionAssociation | None
        ),
        virtual_tp32_attention_only: bool,
        replicated_monolithic_attention: bool,
    ) -> Any:
        if dsa_query_exact_association:
            def mapped_exact_query(
                local_weights: Mapping[str, Any],
                local_exact_dsa_weights: Any,
                local_residual_container: Any,
                local_kv_container: Any,
                local_index_container: Any,
                local_metadata_container: Any,
                local_token_container: Any,
                position: Any,
                block_tables: Any,
                context_lengths: Any,
                local_main_rope_table: Any | None = None,
            ) -> tuple[Any, Any, Any, Any, Any]:
                return mapped_layer0_residual_discriminator_impl(
                    local_weights,
                    local_exact_dsa_weights,
                    local_residual_container,
                    local_kv_container,
                    local_index_container,
                    local_metadata_container,
                    local_token_container,
                    position,
                    block_tables,
                    context_lengths,
                    local_main_rope_table,
                    variant_name=variant_name,
                    reconstruct_attention_output_fp32=(
                        reconstruct_attention_output_fp32
                    ),
                    reconstruct_dense_down_fp32=(
                        reconstruct_dense_down_fp32
                    ),
                    virtual_tp32_reduction_association=(
                        virtual_tp32_reduction_association
                    ),
                    virtual_tp32_attention_only=virtual_tp32_attention_only,
                    replicated_monolithic_attention=(
                        replicated_monolithic_attention
                    ),
                )

            mapped_exact_query.__name__ = (
                f"mapped_layer0_residual_discriminator_{variant_name}"
            )
            return mapped_exact_query

        def mapped(
            local_weights: Mapping[str, Any],
            local_residual_container: Any,
            local_kv_container: Any,
            local_index_container: Any,
            local_metadata_container: Any,
            local_token_container: Any,
            position: Any,
            block_tables: Any,
            context_lengths: Any,
            local_main_rope_table: Any | None = None,
        ) -> tuple[Any, Any, Any, Any, Any]:
            return mapped_layer0_residual_discriminator_impl(
                local_weights,
                None,
                local_residual_container,
                local_kv_container,
                local_index_container,
                local_metadata_container,
                local_token_container,
                position,
                block_tables,
                context_lengths,
                local_main_rope_table,
                variant_name=variant_name,
                reconstruct_attention_output_fp32=(
                    reconstruct_attention_output_fp32
                ),
                reconstruct_dense_down_fp32=reconstruct_dense_down_fp32,
                virtual_tp32_reduction_association=(
                    virtual_tp32_reduction_association
                ),
                virtual_tp32_attention_only=virtual_tp32_attention_only,
                replicated_monolithic_attention=(
                    replicated_monolithic_attention
                ),
            )

        mapped.__name__ = (
            f"mapped_layer0_residual_discriminator_{variant_name}"
        )
        return mapped

    def mapped_layer0_ingredients_observer(
        local_weights: Mapping[str, Any],
        local_exact_dsa_weights: Any,
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        local_main_rope_table: Any | None = None,
    ) -> tuple[Any, ...]:
        """Replay layer 0 once and return primitive, unsaturated boundaries."""

        rank = lax.axis_index(axis_name)
        stage_id = stage_map[rank]
        local_slot = slot_map[rank]
        residual_state = local_residual_container[0]
        kv_cache = local_kv_container[0]
        index_cache = local_index_container[0]
        metadata = local_metadata_container[0]
        main_rope_table_row = None
        if main_rope_table_enabled:
            if (
                local_main_rope_table is None
                or local_main_rope_table.shape
                != (config.context_capacity, geometry.qk_rope_head_dim)
                or local_main_rope_table.dtype != jnp.bfloat16
            ):
                raise PlanValidationError(
                    "layer-0 ingredient main-RoPE table is invalid"
                )
            safe_rope_position = jnp.clip(
                position[0],
                jnp.int32(0),
                jnp.int32(config.context_capacity - 1),
            )
            with jax.named_scope("greenfield_main_rope_table_lookup"):
                main_rope_table_row = local_main_rope_table[
                    safe_rope_position
                ]
        elif local_main_rope_table is not None:
            raise PlanValidationError(
                "default layer-0 ingredient observer cannot consume a "
                "main-RoPE table"
            )

        def weight(name: str) -> Any:
            return local_weights[name][0]

        local_heads = mla_contract.num_heads // config.local_parallel_size
        sentinel_values = (
            jnp.full((config.selected_width,), -1, dtype=jnp.int32),
            jnp.full((config.selected_width,), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros(
                (mla_contract.packed_cache_width,), dtype=residual_state.dtype
            ),
            jnp.full((config.selected_width,), -1, dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros(
                (config.selected_width, mla_contract.packed_cache_width),
                dtype=residual_state.dtype,
            ),
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.zeros(
                (mla_contract.num_heads, mla_contract.kv_lora_rank),
                dtype=residual_state.dtype,
            ),
            jnp.full((mla_contract.num_heads,), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.zeros(
                (mla_contract.num_heads, mla_contract.kv_lora_rank),
                dtype=residual_state.dtype,
            ),
            jnp.full((mla_contract.num_heads,), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.bool_),
            jnp.zeros(
                (local_heads, mla_contract.v_head_dim),
                dtype=residual_state.dtype,
            ),
            jnp.zeros(
                (local_heads * mla_contract.v_head_dim,),
                dtype=residual_state.dtype,
            ),
            jnp.zeros(
                (8, config.hidden_size), dtype=residual_state.dtype
            ),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros(
                (8, config.hidden_size), dtype=residual_state.dtype
            ),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((config.hidden_size,), dtype=residual_state.dtype),
            jnp.zeros((1,), dtype=jnp.bool_),
        )

        def stage0_branch(_: tuple[Any, ...]) -> tuple[Any, ...]:
            stage = schedule.stages[0]
            layer = stage.layers[0]
            if (
                layer.layer_id != 0
                or layer.indexer_kind != "full"
                or layer.mlp_kind != "dense"
                or layer.dense_slot is None
                or len(stage.layers) < 2
            ):
                raise PlanValidationError(
                    "layer-0 ingredient observer requires dense/full layers 0 and 1"
                )
            if len(local_exact_dsa_weights) != 2:
                raise PlanValidationError(
                    "layer-0 ingredient observer requires query and wk state"
                )
            local_query_weight_aliases, local_recurrent_wk_weights = (
                local_exact_dsa_weights
            )
            query_weight_aliases = tuple(
                tuple(value[0] for value in alias)
                for alias in local_query_weight_aliases
            )
            recurrent_wk_weights = tuple(
                value[0] for value in local_recurrent_wk_weights
            )

            token_id = local_token_container[0, 0]
            local_start = local_slot * jnp.int32(local_vocab)
            local_end = local_start + jnp.int32(local_vocab)
            token_valid = (
                (token_id >= jnp.int32(0))
                & (token_id < jnp.int32(config.vocab_size))
            )
            owns_token = (
                token_valid
                & (token_id >= local_start)
                & (token_id < local_end)
            )
            local_id = jnp.clip(
                token_id - local_start,
                jnp.int32(0),
                jnp.int32(local_vocab - 1),
            )
            local_embedding = weight("global.embedding")[local_id][None, :]
            local_embedding = jnp.where(
                owns_token, local_embedding, jnp.zeros_like(local_embedding)
            )
            embedded = lax.psum(
                local_embedding,
                axis_name,
                axis_index_groups=axis_groups,
            )
            observed = stage_local_transformer_layer_fp8_split_mapped(
                embedded,
                jnp.zeros_like(embedded),
                kv_cache[layer.stage_slot],
                index_cache[0],
                metadata[:, : config.selected_width],
                metadata[:, config.count_index],
                position,
                block_tables,
                context_lengths,
                weight(
                    f"attention.slot_{layer.stage_slot:02d}.input_norm"
                ),
                weight(
                    f"attention.slot_{layer.stage_slot:02d}.post_norm"
                ),
                _attention_weights(
                    weight,
                    layer.stage_slot,
                    attention_projection_backend,
                ),
                _dsa_weights(weight, 0),
                _dense_weights(weight, layer.dense_slot),
                None,
                (metadata[0, config.health_index] == jnp.int32(1))[None],
                local_slot,
                axis_name=axis_name,
                indexer_kind="full",
                mlp_kind="dense",
                dsa_contract=dsa_contract,
                mla_contract=mla_contract,
                moe_contract=moe_contract,
                cache_layout=cache_layout,
                axis_index_groups=axis_groups,
                block_shape=geometry.fp8_block_shape,
                sparse_moe_backend=sparse_moe_backend,
                pallas_moe_config=pallas_moe_config,
                pallas_moe_fuse_route_weighting=(
                    feature_fuse_route_weighting
                ),
                pallas_moe_reconstruct_down_fp32=(
                    feature_reconstruct_down_fp32
                ),
                linear_backend=linear_backend,
                dsa_query_backend=dsa_query_backend,
                dsa_query_weight_aliases=tuple(
                    alias[0] for alias in query_weight_aliases
                ),
                dsa_precomputed_wk_weight=recurrent_wk_weights[0],
                dsa_head_key_exact_association=True,
                dsa_score_precision="default",
                attention_projection_backend=attention_projection_backend,
                main_rope_table_row=main_rope_table_row,
                capture_ingredients=True,
            )
            candidate = observed.result
            ingredients = observed.ingredients
            attention = ingredients.attention
            dense = ingredients.dense
            next_layer = stage.layers[1]
            with jax.named_scope("greenfield_layer0_ingredients_layer1_norm"):
                layer1_normalized = fused_add_rms_norm(
                    candidate.hidden_states,
                    candidate.residual,
                    weight(
                        f"attention.slot_{next_layer.stage_slot:02d}.input_norm"
                    ),
                    epsilon=1e-5,
                )[0]
            return (
                candidate.selected_positions[0],
                candidate.selected_scores[0],
                candidate.selected_valid_counts,
                ingredients.normalized_input[0],
                ingredients.combined_residual[0],
                attention.current_cache_row[0],
                attention.owner_selected_positions[0],
                attention.owner_selected_valid_counts,
                attention.owner_selected_cache_values[0],
                attention.owner_selected_cache_valid,
                attention.sparse_partial_output[0],
                attention.sparse_partial_logsumexp[0],
                attention.sparse_partial_valid,
                attention.combined_attention_output[0],
                attention.combined_attention_logsumexp[0],
                attention.combined_attention_valid,
                attention.value_states[0],
                attention.output_input[0],
                attention.virtual_output_partials[:, 0],
                attention.local_output_update[0],
                attention.reduced_output_update[0],
                ingredients.normalized_mlp[0],
                ingredients.post_attention_residual[0],
                dense.virtual_down_partials[:, 0],
                dense.local_down_update[0],
                dense.reduced_down_update[0],
                ingredients.next_hidden[0],
                layer1_normalized[0],
                (
                    candidate.contract_valid
                    & attention.owner_selected_cache_valid
                    & attention.sparse_partial_valid
                    & attention.combined_attention_valid
                    & token_valid[None]
                ),
            )

        values = lax.cond(
            stage_id == jnp.int32(0),
            stage0_branch,
            lambda current: current,
            sentinel_values,
        )
        return tuple(value[None, ...] for value in values)

    def mapped_prefill_index_repair(
        local_weights: Mapping[str, Any],
        local_materialized_wk: tuple[Any, ...],
        local_index_container: Any,
        local_history_container: Any,
        block_tables: Any,
    ) -> Any:
        rank = lax.axis_index(axis_name)
        stage_id = stage_map[rank]
        local_slot = slot_map[rank]
        index_cache = local_index_container[0]
        prompt_history = local_history_container[:, 0]

        def weight(name: str) -> Any:
            return local_weights[name][0]

        def stage_branch(stage: StageExecution) -> Any:
            def repair(value: Any) -> Any:
                full_slot = 0
                repaired = value
                for layer in stage.layers:
                    if layer.indexer_kind != "full":
                        continue
                    dsa = _dsa_weights(weight, full_slot)
                    layer_cache = repair_stage_local_prompt_index_cache(
                        repaired[full_slot],
                        prompt_history[:, full_slot],
                        block_tables,
                        local_materialized_wk[full_slot][0],
                        dsa.key_norm_weight,
                        dsa.key_norm_bias,
                        local_slot,
                        contract=dsa_contract,
                        logical_page_size=config.logical_page_size,
                        local_rows_per_page=config.local_rows_per_page,
                        prompt_chunk=2048,
                        physical_rows=64,
                    )
                    repaired = repaired.at[full_slot].set(layer_cache)
                    full_slot += 1
                return repaired

            return repair

        branches = tuple(stage_branch(stage) for stage in schedule.stages)
        index_cache = lax.switch(stage_id, branches, index_cache)
        return index_cache[None, ...]

    maximum_full_indexer_slots = config.maximum_full_indexer_slots

    prefill_index_weight_names = tuple(
        (
            f"indexer.slot_{full_slot:02d}.wk.weight_bits",
            f"indexer.slot_{full_slot:02d}.wk.scale_inv",
        )
        for full_slot in range(maximum_full_indexer_slots)
    )
    dsa_query_weight_names = tuple(
        (
            f"indexer.slot_{full_slot:02d}.wq_b.weight_bits",
            f"indexer.slot_{full_slot:02d}.wq_b.scale_inv",
        )
        for full_slot in range(maximum_full_indexer_slots)
    )

    def mapped_dsa_query_weight_decode_fp32(
        local_wq_b_bits: tuple[Any, ...],
        local_wq_b_scales: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return tuple(
            dequantize_fp8_bits_block_weight(
                wq_b_bits[0],
                wq_b_scale[0],
                block_shape=geometry.fp8_block_shape,
                output_dtype=jnp.float32,
            )[None, ...]
            for wq_b_bits, wq_b_scale in zip(
                local_wq_b_bits, local_wq_b_scales, strict=True
            )
        )

    def mapped_prefill_index_weight_decode_bf16(
        local_wk_bits: tuple[Any, ...],
        local_wk_scales: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        values = []
        for wk_bits, wk_scale in zip(
            local_wk_bits, local_wk_scales, strict=True
        ):
            decoded = decode_stage_local_prefill_index_wk_bf16(
                wk_bits[0],
                wk_scale[0],
                contract=dsa_contract,
                fp8_block_shape=geometry.fp8_block_shape,
            )
            values.append(decoded[None, ...])
        return tuple(values)

    def mapped_prefill_index_weight_promote_fp32(
        local_wk_bf16: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        return tuple(
            promote_stage_local_prefill_index_wk(
                wk_bf16[0], contract=dsa_contract
            )[None, ...]
            for wk_bf16 in local_wk_bf16
        )

    weight_specs = {
        spec.name: P(axis_name, *(None for _ in spec.shape))
        for spec in weight_layout.specs
    }
    residual_spec = (
        P(axis_name, None, None, None)
        if split_residual_state
        else P(axis_name, None, None)
    )
    kv_spec = P(axis_name, None, None, None, None)
    index_spec = P(axis_name, None, None, None, None)
    metadata_spec = P(axis_name, None, None)
    token_spec = P(axis_name, None)
    dsa_observation_spec = P(axis_name, None, None)
    token_observation_spec = P(axis_name, None)
    layer_residual_observation_spec = P(axis_name, None, None)
    prefill_index_input_spec = P(axis_name, None, None)
    dsa_internal_observation_spec = DsaInternalObservation(
        P(axis_name, None, None),
        P(axis_name, None, None),
        P(axis_name, None, None, None),
        P(axis_name, None, None),
        P(axis_name, None, None),
        P(axis_name, None),
    )
    common_specs = (
        weight_specs,
        residual_spec,
        kv_spec,
        index_spec,
        metadata_spec,
    )
    materialized_dsa_query_specs = tuple(
        P(axis_name, None, None)
        for _ in range(maximum_full_indexer_slots)
    )
    dsa_query_alias_specs = tuple(
        materialized_dsa_query_specs for _ in range(4)
    )
    materialized_wk_specs = tuple(
        P(axis_name, None, None)
        for _ in range(maximum_full_indexer_slots)
    )
    if dsa_query_exact_association:
        exact_dsa_weight_specs: Any = dsa_query_alias_specs
        if dsa_head_key_exact_association:
            exact_dsa_weight_specs = (
                dsa_query_alias_specs,
                materialized_wk_specs,
            )
        common_specs = (
            weight_specs,
            exact_dsa_weight_specs,
            residual_spec,
            kv_spec,
            index_spec,
            metadata_spec,
        )
    if complete_token_path:
        input_specs = (*common_specs, token_spec, P(), P(), P())
        if dsa_query_exact_association:
            mapped = (
                mapped_token_exact_query_main_rope
                if main_rope_table_enabled
                else mapped_token_exact_query
            )
        else:
            mapped = (
                mapped_token_main_rope
                if main_rope_table_enabled
                else mapped_token
            )
        output_specs = (
            residual_spec,
            kv_spec,
            index_spec,
            metadata_spec,
            token_spec,
            P(),
            P(),
            P(),
        )
        if observe_dsa_events:
            output_specs = (
                *output_specs,
                dsa_observation_spec,
                token_observation_spec,
            )
            if observe_layer_residuals:
                output_specs = (
                    *output_specs,
                    layer_residual_observation_spec,
                )
            elif observe_dsa_internals:
                output_specs = (
                    *output_specs,
                    dsa_internal_observation_spec,
                )
        elif observe_prefill_index_inputs:
            output_specs = (*output_specs, prefill_index_input_spec)
    else:
        input_specs = (*common_specs, P(), P(), P())
        if dsa_query_exact_association:
            mapped = (
                mapped_body_exact_query_main_rope
                if main_rope_table_enabled
                else mapped_body_exact_query
            )
        else:
            mapped = (
                mapped_body_main_rope
                if main_rope_table_enabled
                else mapped_body
            )
        output_specs = (residual_spec, kv_spec, index_spec, metadata_spec)
    if main_rope_table_enabled:
        input_specs = (*input_specs, P())
    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=output_specs,
        check_vma=False,
    )
    layer0_residual_discriminators: tuple[tuple[str, Any], ...] = ()
    if build_layer0_residual_discriminator:
        layer0_residual_discriminators = tuple(
            (
                variant_name,
                jax.shard_map(
                    make_mapped_layer0_residual_discriminator(
                        variant_name,
                        reconstruct_attention_output_fp32,
                        reconstruct_dense_down_fp32,
                        virtual_tp32_reduction_association,
                        virtual_tp32_attention_only,
                        replicated_monolithic_attention,
                    ),
                    mesh=mesh,
                    in_specs=input_specs,
                    out_specs=(
                        P(axis_name, None),
                        P(axis_name, None),
                        P(axis_name, None),
                        P(axis_name, None),
                        P(axis_name, None),
                    ),
                    check_vma=False,
                ),
            )
            for (
                variant_name,
                reconstruct_attention_output_fp32,
                reconstruct_dense_down_fp32,
                virtual_tp32_reduction_association,
                virtual_tp32_attention_only,
                replicated_monolithic_attention,
            ) in _layer0_residual_discriminator_specs(
                layer0_residual_discriminator_kind
            )
        )
    layer0_ingredients_observer = None
    if build_layer0_ingredients_observer:
        ingredient_output_specs = (
            P(axis_name, None),  # selected_positions
            P(axis_name, None),  # selected_scores
            P(axis_name, None),  # selected_valid_counts
            P(axis_name, None),  # normalized_input
            P(axis_name, None),  # combined_residual
            P(axis_name, None),  # current_cache_row
            P(axis_name, None),  # owner_selected_positions
            P(axis_name, None),  # owner_selected_valid_counts
            P(axis_name, None, None),  # owner_selected_cache_values
            P(axis_name, None),  # owner_selected_cache_valid
            P(axis_name, None, None),  # sparse_partial_output
            P(axis_name, None),  # sparse_partial_logsumexp
            P(axis_name, None),  # sparse_partial_valid
            P(axis_name, None, None),  # combined_attention_output
            P(axis_name, None),  # combined_attention_logsumexp
            P(axis_name, None),  # combined_attention_valid
            P(axis_name, None, None),  # value_states
            P(axis_name, None),  # attention_output_input
            P(axis_name, None, None),  # attention_virtual_partials
            P(axis_name, None),  # attention_local_update
            P(axis_name, None),  # attention_reduced_update
            P(axis_name, None),  # normalized_mlp
            P(axis_name, None),  # post_attention_residual
            P(axis_name, None, None),  # dense_virtual_partials
            P(axis_name, None),  # dense_local_update
            P(axis_name, None),  # dense_reduced_update
            P(axis_name, None),  # next_hidden
            P(axis_name, None),  # layer1_normalized
            P(axis_name, None),  # contract_valid
        )
        if len(ingredient_output_specs) != len(LAYER0_INGREDIENT_NAMES):
            raise PlanValidationError(
                "layer-0 ingredient names and shard specs drifted"
            )
        layer0_ingredients_observer = jax.shard_map(
            mapped_layer0_ingredients_observer,
            mesh=mesh,
            in_specs=input_specs,
            out_specs=ingredient_output_specs,
            check_vma=False,
        )
    repair_prefill_index_cache = None
    materialize_dsa_query_weights_fp32 = None
    decode_prefill_index_weights_bf16 = None
    promote_prefill_index_weights_fp32 = None
    if dsa_query_exact_association:
        raw_query_weight_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        materialize_dsa_query_weights_fp32 = jax.shard_map(
            mapped_dsa_query_weight_decode_fp32,
            mesh=mesh,
            in_specs=(raw_query_weight_specs, raw_query_weight_specs),
            out_specs=materialized_dsa_query_specs,
            check_vma=False,
        )
    if observe_prefill_index_inputs:
        raw_wk_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        raw_wk_scale_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        decode_prefill_index_weights_bf16 = jax.shard_map(
            mapped_prefill_index_weight_decode_bf16,
            mesh=mesh,
            in_specs=(raw_wk_specs, raw_wk_scale_specs),
            out_specs=materialized_wk_specs,
            check_vma=False,
        )
        promote_prefill_index_weights_fp32 = jax.shard_map(
            mapped_prefill_index_weight_promote_fp32,
            mesh=mesh,
            in_specs=(materialized_wk_specs,),
            out_specs=materialized_wk_specs,
            check_vma=False,
        )
        repair_prefill_index_cache = jax.shard_map(
            mapped_prefill_index_repair,
            mesh=mesh,
            in_specs=(
                weight_specs,
                materialized_wk_specs,
                index_spec,
                P(None, axis_name, None, None),
                P(),
            ),
            out_specs=index_spec,
            check_vma=False,
        )
    return DecoderStepProgram(
        config=config,
        execute=execute,
        mesh=mesh,
        input_specs=input_specs,
        groups=canonical_groups,
        pairs=canonical_pairs,
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        state_layout_hash=state_layout.state_layout_hash,
        weight_layout_hash=weight_layout.layout_hash,
        sparse_moe_backend=sparse_moe_backend,
        feature_output_tile=feature_output_tile,
        feature_fuse_route_weighting=feature_fuse_route_weighting,
        feature_reconstruct_down_fp32=feature_reconstruct_down_fp32,
        linear_backend=linear_backend,
        dsa_query_backend=dsa_query_backend,
        dsa_query_exact_association=dsa_query_exact_association,
        dsa_head_key_exact_association=(
            dsa_head_key_exact_association
        ),
        dsa_score_default_precision=dsa_score_default_precision,
        attention_projection_backend=attention_projection_backend,
        main_rope_table_enabled=main_rope_table_enabled,
        pregathered_b512_attention=pregathered_b512_attention,
        strategy_nd_attention_projection=(
            strategy_nd_attention_projection
        ),
        dense_final_layout_convolution=dense_final_layout_convolution,
        main_rope_table_host=main_rope_table_host,
        main_rope_table_sha256=main_rope_table_digest,
        main_rope_table_bytes_per_device=main_rope_table_bytes,
        complete_token_path=complete_token_path,
        observe_dsa_events=observe_dsa_events,
        observe_dsa_internals=observe_dsa_internals,
        observe_layer_residuals=observe_layer_residuals,
        observe_prefill_index_inputs=observe_prefill_index_inputs,
        prefill_index_weight_names=(
            prefill_index_weight_names
            if observe_prefill_index_inputs
            else ()
        ),
        dsa_query_weight_names=(
            dsa_query_weight_names if dsa_query_exact_association else ()
        ),
        materialize_dsa_query_weights_fp32=(
            materialize_dsa_query_weights_fp32
        ),
        decode_prefill_index_weights_bf16=(
            decode_prefill_index_weights_bf16
        ),
        promote_prefill_index_weights_fp32=(
            promote_prefill_index_weights_fp32
        ),
        repair_prefill_index_cache=repair_prefill_index_cache,
        layer0_residual_discriminators=layer0_residual_discriminators,
        layer0_residual_discriminator_kind=(
            layer0_residual_discriminator_kind
        ),
        layer0_ingredients_observer=layer0_ingredients_observer,
        split_residual_state=split_residual_state,
    )
