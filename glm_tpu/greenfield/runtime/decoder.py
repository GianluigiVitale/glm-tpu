"""One-step all-stage raw-FP8 decoder program over the global device mesh."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import PlanValidationError
from ..kernels.layer import (
    AttentionFp8Weights,
    DenseFp8Weights,
    DsaFp8Weights,
    MoeFp8Weights,
    SparseMoeBackend,
    stage_local_transformer_layer_fp8_mapped,
)
from ..kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.stage_local import StageLinearBackend
from ..model.schedule import PipelineSchedule, StageExecution
from ..model.state import DecoderStateLayout
from ..model.weights import (
    COMPLETE_EXPERT_RUNTIME_LAYOUT,
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
)
from ..sharding.hlo_contract import parse_hlo_module
from ..types import ExecutionPlan
from .pipeline import (
    PipelineSkeletonConfig,
    _canonical_groups,
    _canonical_pairs,
    _partition_maps,
)


@dataclass(frozen=True, slots=True)
class DecoderStepConfig:
    stage_count: int
    local_parallel_size: int
    hidden_size: int
    selected_width: int
    maximum_layer_slots: int
    maximum_full_indexer_slots: int
    logical_page_size: int
    local_rows_per_page: int
    packed_cache_width: int
    index_key_width: int

    def __post_init__(self) -> None:
        for field in (
            "stage_count",
            "local_parallel_size",
            "hidden_size",
            "selected_width",
            "maximum_layer_slots",
            "maximum_full_indexer_slots",
            "logical_page_size",
            "local_rows_per_page",
            "packed_cache_width",
            "index_key_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"decoder {field} must be positive")
        if self.logical_page_size != (
            self.local_parallel_size * self.local_rows_per_page
        ):
            raise PlanValidationError("decoder local page geometry is inconsistent")

    @property
    def total_devices(self) -> int:
        return self.stage_count * self.local_parallel_size

    @property
    def metadata_width(self) -> int:
        # Positions, count, producer, visited, health, active.
        return self.selected_width + 5

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
    linear_backend: StageLinearBackend


def _validate_pallas_feature_decoder_calls(
    optimized_hlo: str,
    *,
    sparse_layers: int,
) -> dict[str, Any]:
    """Pin every production feature-MoE kernel and reject weight overlays."""

    kernel_names = (
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512",
        "greenfield_fp8_block_up_gate_m8_k6144_n512",
        "greenfield_fp8_block_matmul_m8_k512_n6144",
    )
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if 'custom_call_target="tpu_custom_call"' in line
    ]
    calls_by_kernel = {
        name: [line for line in custom_calls if name in line]
        for name in kernel_names
    }
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

    selected_name = kernel_names[0]
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
        "kernel_counts": kernel_counts,
        "passed": not violations,
        "violations": violations,
    }


def _validate_pallas_stage_linear_decoder_calls(
    optimized_hlo: str,
    *,
    layers: int,
    dense_layers: int,
    full_indexer_layers: int,
) -> dict[str, Any]:
    """Pin every raw-FP8 attention/dense projection replacing an overlay."""

    expected_kernel_counts = {
        "greenfield_fp8_block_matmul_m8_k6144_n2048": layers,
        "greenfield_fp8_block_matmul_m8_k2048_n4096": layers,
        "greenfield_fp8_block_matmul_m8_k6144_n640": layers,
        "greenfield_fp8_block_matmul_m8_k4096_n6144": layers,
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512": layers,
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256": layers,
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024": full_indexer_layers,
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
    forbidden_shapes = tuple(
        f"{dtype}[{shape}]"
        for dtype in ("bf16", "f32")
        for shape in (
            "2048,6144",
            "4096,2048",
            "576,6144",
            "6144,4096",
            "7168,512",
            "1024,2048",
            "128,6144",
            "3072,6144",
            "6144,3072",
        )
    )
    forbidden_overlays = [
        shape for shape in forbidden_shapes if shape in optimized_hlo
    ]
    forbidden_formatted_shapes = tuple(
        f"f8e4m3fn[{shape}]"
        for shape in (
            "2048,6144",
            "4096,2048",
            "6144,4096",
            "7168,512",
            "1024,2048",
            "128,6144",
        )
    )
    forbidden_formatted_overlays = [
        shape for shape in forbidden_formatted_shapes if shape in optimized_hlo
    ]
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
        "expected_kernel_counts": expected_kernel_counts,
        "forbidden_decoded_weight_overlays": forbidden_overlays,
        "forbidden_formatted_weight_overlays": forbidden_formatted_overlays,
        "kernel_counts": kernel_counts,
        "passed": not violations,
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
) -> dict[str, Any]:
    """Reject non-local collectives, count drift, and dead batch rows."""

    if backend_contract not in (
        "cpu_reference",
        "tpu_v4_pp8_reference",
        "tpu_v4_pp8_pallas_feature",
        "tpu_v4_pp8_pallas_feature_linear",
    ):
        raise PlanValidationError("decoder HLO backend contract is unknown")
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
        expected_gathers = 4 * layers + 3 * full_layers
        expected_reductions = 2 * layers
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
        expected_gathers = 2 * layers + 3 * full_layers
        expected_reduction_arity_counts = {"1": 277, "2": 16, "3": 1}
        expected_reductions = sum(expected_reduction_arity_counts.values())
        expected_reduction_component_count = sum(
            int(arity) * count
            for arity, count in expected_reduction_arity_counts.items()
        )
        expected_reduction_result_shape_counts = {
            "bf16[1,6144]": layers + dense_layers,
            "bf16[2,1,6144]": sparse_layers,
            "f32[256]": layers,
            "u32[1,1,128]": layers,
        }

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
    expected_permutes = 2 * config.stage_count
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
    forbidden_shapes = []
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            if shape.dimensions in (
                (config.total_devices, config.hidden_size),
                (config.total_devices, 1, config.hidden_size),
                (config.total_devices, config.selected_width),
                (config.total_devices, 1, config.selected_width),
            ):
                forbidden_shapes.append(
                    {"instruction": instruction.name, "shape": shape.to_dict()}
                )
    if forbidden_shapes:
        violations.append("decoder contains dead-row/full-pod live tensors")
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
            )
        )
        violations.extend(pallas_stage_linear_contract["violations"])
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
        "full_indexer_layers": full_layers,
        "layer_count": layers,
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "pallas_feature_contract": pallas_feature_contract,
        "pallas_stage_linear_contract": pallas_stage_linear_contract,
        "passed": not violations,
        "violations": violations,
    }


def _attention_weights(
    weight: Any,
    slot: int,
) -> AttentionFp8Weights:
    base = f"attention.slot_{slot:02d}"
    return AttentionFp8Weights(
        weight(f"{base}.q_a.weight_bits"),
        weight(f"{base}.q_a.scale_inv"),
        weight(f"{base}.q_a_norm"),
        weight(f"{base}.q_b.weight_bits"),
        weight(f"{base}.q_b.scale_inv"),
        weight(f"{base}.kv_a.weight_bits"),
        weight(f"{base}.kv_a.scale_inv"),
        weight(f"{base}.kv_a_norm"),
        weight(f"{base}.kv_b.weight_bits"),
        weight(f"{base}.kv_b.scale_inv"),
        weight(f"{base}.o.weight_bits"),
        weight(f"{base}.o.scale_inv"),
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
    linear_backend: StageLinearBackend,
) -> tuple[Any, Any, Any, Any]:
    import jax.numpy as jnp

    residual, kv_cache, index_cache, metadata = values
    full_slot = 0
    for layer in stage.layers:
        attention = _attention_weights(weight, layer.stage_slot)
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
            linear_backend=linear_backend,
        )
        residual = result.output
        kv_cache = kv_cache.at[layer.stage_slot].set(result.kv_cache)
        if current_full_slot is not None:
            index_cache = index_cache.at[current_full_slot].set(
                result.index_cache
            )
            metadata = metadata.at[0, config.producer_index].set(
                jnp.int32(layer.layer_id)
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
    return residual, kv_cache, index_cache, metadata


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
    linear_backend: StageLinearBackend = "reference",
) -> DecoderStepProgram:
    """Build, but do not compile, the complete all-stage decode-step map."""

    if not axis_name:
        raise PlanValidationError("decoder axis name must be explicit")
    if sparse_moe_backend not in ("reference", "pallas_feature"):
        raise PlanValidationError("decoder sparse MoE backend is unknown")
    if linear_backend not in ("reference", "pallas"):
        raise PlanValidationError("decoder FP8 linear backend is unknown")
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
    config = DecoderStepConfig(
        stage_count=plan.pipeline_stages,
        local_parallel_size=plan.local_parallel_size,
        hidden_size=geometry.hidden_size,
        selected_width=geometry.dsa_top_k,
        maximum_layer_slots=max(stage.layer_count for stage in schedule.stages),
        maximum_full_indexer_slots=max(
            stage.full_indexer_count for stage in state_layout.stages
        ),
        logical_page_size=state_layout.logical_page_size,
        local_rows_per_page=state_layout.stages[0].local_rows_per_page,
        packed_cache_width=state_layout.stages[0].packed_kv_width,
        index_key_width=state_layout.stages[0].index_key_width,
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

    def mapped(
        local_weights: Mapping[str, Any],
        local_residual_container: Any,
        local_kv_container: Any,
        local_index_container: Any,
        local_metadata_container: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
    ) -> tuple[Any, Any, Any, Any]:
        rank = lax.axis_index(axis_name)
        stage_id = stage_map[rank]
        local_slot = slot_map[rank]
        residual = local_residual_container[0]
        kv_cache = local_kv_container[0]
        index_cache = local_index_container[0]
        metadata = local_metadata_container[0]

        def weight(name: str) -> Any:
            return local_weights[name][0]

        for hop, stage in enumerate(schedule.stages):
            active = metadata[0, config.active_index] == jnp.int32(1)
            should_execute = active & (stage_id == jnp.int32(hop))
            residual, kv_cache, index_cache, metadata = lax.cond(
                should_execute,
                lambda values, stage=stage: _execute_stage(
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
                    linear_backend=linear_backend,
                ),
                lambda values: values,
                (residual, kv_cache, index_cache, metadata),
            )
            residual = lax.ppermute(residual, axis_name, canonical_pairs)
            metadata = lax.ppermute(metadata, axis_name, canonical_pairs)
        return (
            residual[None, ...],
            kv_cache[None, ...],
            index_cache[None, ...],
            metadata[None, ...],
        )

    weight_specs = {
        spec.name: P(axis_name, *(None for _ in spec.shape))
        for spec in weight_layout.specs
    }
    residual_spec = P(axis_name, None, None)
    kv_spec = P(axis_name, None, None, None, None)
    index_spec = P(axis_name, None, None, None, None)
    metadata_spec = P(axis_name, None, None)
    input_specs = (
        weight_specs,
        residual_spec,
        kv_spec,
        index_spec,
        metadata_spec,
        P(),
        P(),
        P(),
    )
    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=(residual_spec, kv_spec, index_spec, metadata_spec),
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
        linear_backend=linear_backend,
    )
