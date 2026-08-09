"""One-step all-stage raw-FP8 decoder program over the global device mesh."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, NamedTuple

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
from ..kernels.reference.linear import vocabulary_logits
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.reference.prefill_index import (
    materialize_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from ..kernels.reference.rmsnorm import final_norm, fused_add_rms_norm
from ..kernels.stage_local import StageLinearBackend
from ..model.schedule import PipelineSchedule, StageExecution
from ..model.state import DecoderStateLayout
from ..model.weights import (
    COMPLETE_EXPERT_RUNTIME_LAYOUT,
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
)
from ..sharding.hlo_contract import (
    HloInstruction,
    HloModule,
    HloShape,
    parse_hlo_module,
)
from ..types import ExecutionPlan
from .pipeline import (
    PipelineSkeletonConfig,
    _canonical_groups,
    _canonical_pairs,
    _partition_maps,
)

REFERENCE_FEATURE_OUTPUT_TILE = 128
PROMOTED_FEATURE_OUTPUT_TILE = 256
TOKEN_OBSERVATION_CANDIDATES = 16
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
) -> tuple[dict[str, int], dict[str, int], int]:
    """Count forbidden shapes, admitting exact repair-rooted occurrences."""

    repair_computations = (
        _prefill_index_repair_computations(module)
        if prefill_index_repair
        else set()
    )
    allowed: Counter[str] = Counter()
    forbidden: Counter[str] = Counter()
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            signature = (
                f"{shape.dtype}["
                + ",".join(str(value) for value in shape.dimensions)
                + "]"
            )
            if signature not in forbidden_signatures:
                continue
            if (
                signature in repair_allowed_signatures
                and instruction.computation in repair_computations
            ):
                allowed[signature] += 1
            else:
                forbidden[signature] += 1
    return (
        dict(sorted(allowed.items())),
        dict(sorted(forbidden.items())),
        len(repair_computations),
    )


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
    attention_projection_backend: AttentionProjectionBackend
    complete_token_path: bool
    observe_dsa_events: bool
    observe_dsa_internals: bool
    observe_layer_residuals: bool
    observe_prefill_index_inputs: bool
    prefill_index_weight_names: tuple[tuple[str, str], ...]
    materialize_prefill_index_weights: Any | None
    repair_prefill_index_cache: Any | None
    split_residual_state: bool


def _validate_pallas_feature_decoder_calls(
    optimized_hlo: str,
    *,
    sparse_layers: int,
    feature_output_tile: int = PROMOTED_FEATURE_OUTPUT_TILE,
    fuse_route_weighting: bool = False,
    reconstruct_down_fp32: bool = False,
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
    selected_name = "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512"
    if feature_output_tile != 128:
        selected_name += f"_ot{feature_output_tile}"
    if reconstruct_down_fp32:
        selected_name += "_downf32"
    if fuse_route_weighting:
        selected_name += "_wsum"
    kernel_names = (
        selected_name,
        "greenfield_fp8_block_up_gate_m8_k6144_n512",
        "greenfield_fp8_block_matmul_m8_k512_n6144",
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
        if line.count("u8[256,6144,512]") < 2
        or "u8[256,512,6144]" not in line
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
        kernel_names[1]: ("u8[512,6144]",),
        kernel_names[2]: ("u8[6144,512]",),
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
        "bf16[256,6144,512]",
        "f32[256,6144,512]",
        "bf16[256,512,6144]",
        "f32[256,512,6144]",
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
            "f8e4m3fn[512,6144]",
            "f8e4m3fn[6144,512]",
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
    (
        allowed_repair_shapes,
        forbidden_shape_counts,
        repair_computation_count,
    ) = _classify_scoped_shape_occurrences(
        module,
        forbidden_signatures=forbidden_candidates,
        repair_allowed_signatures={"f32[32,6144]"},
        prefill_index_repair=prefill_index_repair,
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
        "forbidden_shapes": list(forbidden_shapes),
        "prefill_index_repair": prefill_index_repair,
        "prefill_index_repair_computation_count": repair_computation_count,
        "passed": not violations,
        "required_shapes": required_shapes,
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
    violations = []
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
    attention_projection_backend: AttentionProjectionBackend = "separate",
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
        "greenfield_fp8_block_matmul_m8_k4096_n6144": layers,
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": layers,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": layers,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": (
            full_indexer_layers if dsa_query_backend == "pallas" else 0
        ),
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128": full_indexer_layers,
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144": (
            dense_layers
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
    allowed_prefill_index_repair_shapes = []
    allowed_dsa_score_shapes = []
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
        "body_records": body_records,
        "expected_score_body_count": expected_body_count,
        "forbidden_shapes": forbidden_shapes,
        "local_context_width": local_context_width,
        "passed": not body_violations and not forbidden_shapes,
        "prefill_index_repair": prefill_index_repair,
        "prefill_index_repair_computation_count": len(
            repair_computations
        ),
        "score_body_count": len(valid_computations),
        "score_dimensions": list(score_dimensions),
        "violations": body_violations,
    }


def _validate_complete_token_collective_lowering(
    module: Any,
    *,
    expected_groups: Sequence[Sequence[int]],
    expected_pairs: Sequence[Sequence[int]],
    backend_contract: str,
    token_observation_candidates: int = 1,
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
        == 4 * token_observation_candidates
    )
    token_id_exchange = tuple(
        item
        for item in reductions
        if len(item.result_shapes) == 1
        and item.result_shapes[0].dtype == "s32"
        and prod(item.result_shapes[0].dimensions)
        == 4 * token_observation_candidates
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
                != 4 * token_observation_candidates
            ):
                violations.append(
                    f"complete-token {label} exchange operand shape drifted"
                )
            if collective.replica_groups != canonical_groups:
                violations.append(
                    f"complete-token {label} exchange escaped PP8 local groups"
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
    accepted_token_return_op_names = (
        "jit(mapped_token)/shard_map/ppermute",
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


def validate_decoder_step_hlo(
    optimized_hlo: str,
    *,
    config: DecoderStepConfig,
    schedule: PipelineSchedule,
    groups: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
    backend_contract: str,
    feature_output_tile: int | None = None,
    feature_fuse_route_weighting: bool = False,
    feature_reconstruct_down_fp32: bool = False,
    dsa_query_backend: StageLinearBackend | None = None,
    attention_projection_backend: AttentionProjectionBackend = "separate",
    complete_token_path: bool = False,
    token_observation_candidates: int = 1,
    split_residual_state: bool = False,
    prefill_index_repair: bool = False,
) -> dict[str, Any]:
    """Reject non-local collectives, count drift, and dead batch rows."""

    if backend_contract not in (
        "cpu_reference",
        "tpu_v4_pp8_reference",
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
    ):
        raise PlanValidationError("decoder HLO backend contract is unknown")
    if feature_output_tile is None:
        feature_output_tile = (
            PROMOTED_FEATURE_OUTPUT_TILE
            if backend_contract
            in (
                "tpu_v4_pp8_pallas_feature",
                "tpu_v4_pp8_pallas_feature_linear",
            )
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
        backend_contract
        not in (
            "tpu_v4_pp8_pallas_feature",
            "tpu_v4_pp8_pallas_feature_linear",
        )
        and feature_output_tile != 128
    ):
        raise PlanValidationError(
            "a non-default feature output tile requires a feature backend"
        )
    if feature_fuse_route_weighting and backend_contract not in (
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
    ):
        raise PlanValidationError(
            "feature route-weight fusion requires a feature backend"
        )
    if feature_reconstruct_down_fp32 and backend_contract not in (
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
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
    reduction_arity_counts: dict[str, int] = {}
    expected_reduction_arity_counts: dict[str, int] = {}
    reduction_result_shape_counts: dict[str, int] = {}
    expected_reduction_result_shape_counts: dict[str, int] = {}
    reduction_component_count = 0
    expected_reduction_component_count = 0
    sparse_layers = 0
    if backend_contract == "cpu_reference":
        expected_gathers = 4 * layers + 3 * full_layers + (
            2 if complete_token_path else 0
        )
        expected_reductions = 2 * layers + (
            1 if complete_token_path else 0
        )
    else:
        dense_layers = sum(
            layer.mlp_kind == "dense"
            for stage in schedule.stages
            for layer in stage.layers
        )
        sparse_layers = layers - dense_layers
        if (
            config.stage_count,
            config.local_parallel_size,
            config.hidden_size,
            config.selected_width,
            layers,
            dense_layers,
            sparse_layers,
        ) != (8, 4, 6144, 2048, 78, 3, 75):
            raise PlanValidationError(
                "the TPU v4 PP8 reference lowering contract is pinned to "
                "the complete 78-layer GLM-5.2 decoder"
            )

        # Gate C established the semantic TPU rewrite: every attention body
        # contributes three reduction results and every MLP contributes one.
        # The first complete 78-layer optimized HLO then established the exact
        # physical lowering.  All 312 logical results are present, but XLA
        # launch-fuses 78 padded-u32 results into 43 single-result, 16
        # two-result, and one three-result all-reduces.  Pin both views: merely
        # expecting 4 * layers physical instructions falsely reports 18
        # missing collectives, while checking only logical arity would allow a
        # physical launch-count regression to pass unnoticed.
        # TPU XLA lowers each four-element top-1 all-gather to a one-hot local
        # all-reduce.  The complete path therefore adds no physical gather,
        # but adds embedding plus score/id singleton reductions.
        expected_gathers = 2 * layers + 3 * full_layers
        expected_reduction_arity_counts = {"1": 277, "2": 16, "3": 1}
        if feature_reconstruct_down_fp32:
            expected_reduction_arity_counts["1"] += sparse_layers
        if complete_token_path:
            expected_reduction_arity_counts["1"] += 3
        expected_reductions = sum(expected_reduction_arity_counts.values())
        expected_reduction_component_count = sum(
            int(arity) * count
            for arity, count in expected_reduction_arity_counts.items()
        )
        expected_reduction_result_shape_counts = {
            "bf16[1,6144]": (
                layers + dense_layers
            ),
            "bf16[2,1,6144]": sparse_layers,
            "f32[256]": layers,
            "u32[1,1,128]": layers,
        }
        if feature_reconstruct_down_fp32:
            expected_reduction_result_shape_counts[
                "f32[8,6144]"
            ] = sparse_layers
        if complete_token_path:
            embedding_shape = (
                "bf16[1,1,6144]"
                if split_residual_state
                else "bf16[1,6144]"
            )
            expected_reduction_result_shape_counts[embedding_shape] = (
                expected_reduction_result_shape_counts.get(
                    embedding_shape, 0
                )
                + 1
            )
            if token_observation_candidates == 1:
                expected_reduction_result_shape_counts.update(
                    {"bf16[4]": 1, "s32[4]": 1}
                )

        reductions = by_opcode.get("all-reduce", ())
        reduction_arity_counts = {
            str(arity): count
            for arity, count in sorted(
                Counter(len(item.result_shapes) for item in reductions).items()
            )
        }
        result_shapes = Counter(
            (
                f"{shape.dtype}["
                + ",".join(str(dimension) for dimension in shape.dimensions)
                + "]"
            )
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
                    == 4 * token_observation_candidates
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
    if backend_contract in (
        "tpu_v4_pp8_reference",
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
    ):
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
            )
        )
        violations.extend(complete_token_collective_contract["violations"])
    live_tensor_contract = _classify_decoder_live_tensor_shapes(
        module,
        config=config,
        full_indexer_layers=full_layers,
        backend_contract=backend_contract,
        prefill_index_repair=prefill_index_repair,
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
    if backend_contract in (
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
    ):
        pallas_feature_contract = _validate_pallas_feature_decoder_calls(
            optimized_hlo,
            sparse_layers=sparse_layers,
            feature_output_tile=feature_output_tile,
            fuse_route_weighting=feature_fuse_route_weighting,
            reconstruct_down_fp32=feature_reconstruct_down_fp32,
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
                attention_projection_backend=attention_projection_backend,
                prefill_index_repair=prefill_index_repair,
                module=module,
            )
        )
        violations.extend(pallas_stage_linear_contract["violations"])
    dsa_query_association_contract: dict[str, Any] = {}
    if backend_contract != "cpu_reference":
        dsa_query_association_contract = (
            _validate_dsa_query_decoder_association(
                optimized_hlo,
                full_indexer_layers=full_layers,
                local_parallel_size=config.local_parallel_size,
                dsa_indexer_heads=config.dsa_indexer_heads,
                index_key_width=config.index_key_width,
                backend=dsa_query_backend,
            )
        )
        violations.extend(dsa_query_association_contract["violations"])
    fused_qkv_a_contract: dict[str, Any] = {}
    if attention_projection_backend == "fused_n82_convolution":
        fused_qkv_a_contract = _validate_fused_qkv_a_decoder_association(
            optimized_hlo,
            layers=layers,
            prefill_index_repair=prefill_index_repair,
            module=module,
        )
        violations.extend(fused_qkv_a_contract["violations"])
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
        "attention_projection_backend": attention_projection_backend,
        "fused_qkv_a_contract": fused_qkv_a_contract,
        "passed": not violations,
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


def _dense_weights(weight: Any, slot: int) -> DenseFp8Weights:
    base = f"dense.slot_{slot:02d}"
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
    attention_projection_backend: AttentionProjectionBackend,
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
            layer_index_cache = index_cache[full_slot]
            current_full_slot = full_slot
            full_slot += 1
        else:
            dsa = None
            layer_index_cache = index_cache[0]
            current_full_slot = None
        if layer.mlp_kind == "dense":
            assert layer.dense_slot is not None
            dense = _dense_weights(weight, layer.dense_slot)
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
            attention_projection_backend=attention_projection_backend,
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
    attention_projection_backend: AttentionProjectionBackend,
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
            layer_index_cache = index_cache[full_slot]
            current_full_slot = full_slot
            full_slot += 1
        else:
            dsa = None
            layer_index_cache = index_cache[0]
            current_full_slot = None
        if layer.mlp_kind == "dense":
            assert layer.dense_slot is not None
            dense = _dense_weights(weight, layer.dense_slot)
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
            attention_projection_backend=attention_projection_backend,
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
    attention_projection_backend: AttentionProjectionBackend = "separate",
    complete_token_path: bool = False,
    observe_dsa_events: bool = False,
    observe_dsa_internals: bool = False,
    observe_layer_residuals: bool = False,
    observe_prefill_index_inputs: bool = False,
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
    if attention_projection_backend not in (
        "separate",
        "fused_n82_convolution",
    ):
        raise PlanValidationError(
            "decoder attention projection backend is unknown"
        )
    if not isinstance(complete_token_path, bool):
        raise PlanValidationError("complete token-path flag must be boolean")
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
    if not isinstance(split_residual_state, bool):
        raise PlanValidationError("split residual-state flag must be boolean")
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
    expected_expert_layout = (
        COMPLETE_EXPERT_RUNTIME_LAYOUT
        if sparse_moe_backend == "reference"
        else FEATURE_EXPERT_RUNTIME_LAYOUT
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

    local_vocab = geometry.vocab_size // config.local_parallel_size

    def mapped_impl(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        local_token_container: Any | None,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
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
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
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
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
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
                            attention_projection_backend=(
                                attention_projection_backend
                            ),
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
                        attention_projection_backend=(
                            attention_projection_backend
                        ),
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
                        attention_projection_backend=(
                            attention_projection_backend
                        ),
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
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            None,
            position,
            block_tables,
            context_lengths,
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
            local_residual_container,
            local_kv_container,
            local_index_container,
            local_metadata_container,
            local_token_container,
            position,
            block_tables,
            context_lengths,
        )

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

    def mapped_prefill_index_weight_materialization(
        local_wk_bits: tuple[Any, ...],
        local_wk_scales: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        values = []
        for wk_bits, wk_scale in zip(
            local_wk_bits, local_wk_scales, strict=True
        ):
            materialized = materialize_stage_local_prefill_index_wk(
                wk_bits[0],
                wk_scale[0],
                contract=dsa_contract,
                fp8_block_shape=geometry.fp8_block_shape,
            )
            values.append(materialized[None, ...])
        return tuple(values)

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
    if complete_token_path:
        input_specs = (*common_specs, token_spec, P(), P(), P())
        mapped = mapped_token
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
        mapped = mapped_body
        output_specs = (residual_spec, kv_spec, index_spec, metadata_spec)
    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=output_specs,
        check_vma=False,
    )
    repair_prefill_index_cache = None
    materialize_prefill_index_weights = None
    if observe_prefill_index_inputs:
        materialized_wk_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        raw_wk_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        raw_wk_scale_specs = tuple(
            P(axis_name, None, None)
            for _ in range(maximum_full_indexer_slots)
        )
        materialize_prefill_index_weights = jax.shard_map(
            mapped_prefill_index_weight_materialization,
            mesh=mesh,
            in_specs=(raw_wk_specs, raw_wk_scale_specs),
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
        attention_projection_backend=attention_projection_backend,
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
        materialize_prefill_index_weights=(
            materialize_prefill_index_weights
        ),
        repair_prefill_index_cache=repair_prefill_index_cache,
        split_residual_state=split_residual_state,
    )
