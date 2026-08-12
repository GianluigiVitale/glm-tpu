"""Correctness-first teacher-forced prefill over the complete PP8 token step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..errors import PlanValidationError
from ..model.schedule import PipelineSchedule
from ..sharding.hlo_contract import HloModule, parse_hlo_module
from .decoder import (
    DecoderStepProgram,
    _is_prefill_index_repair_op_name,
    validate_decoder_step_hlo,
)

PrefillBackendContract = Literal[
    "cpu_reference",
    "tpu_v4_pp8_reference",
    "tpu_v4_pp8_pallas_feature",
    "tpu_v4_pp8_pallas_feature_linear",
]


@dataclass(frozen=True, slots=True)
class TeacherForcedPrefillProgram:
    """One compiled prompt scan with no per-token host or stage dispatch."""

    decoder: DecoderStepProgram
    execute: Any
    index_repair_backend: Literal["none", "physical_m64_chunk"]
    prompt_length: int


_OUTER_PREFILL_LOOP_OP_NAME = "jit(execute)/while"
_FUSED_QKV_LOOP_OP_NAME_SUFFIX = (
    "/one_row_fused_qkv_a_n82_convolution/while"
)
_PREFILL_INDEX_REPAIR_LOOP_OP_NAME_SUFFIX = (
    "/physical_m64_prompt_index_key_chunk/while"
)
_PREFILL_INDEX_REPAIR_CHUNK = 2048
_PREFILL_INDEX_REPAIR_PHYSICAL_ROWS = 64


def _prefill_index_repair_chunk_count(prompt_length: int) -> int:
    return (
        prompt_length + _PREFILL_INDEX_REPAIR_CHUNK - 1
    ) // _PREFILL_INDEX_REPAIR_CHUNK


def validate_prefill_index_weight_materialization_hlo(
    optimized_hlo: str,
    *,
    decoder: DecoderStepProgram,
    phase: Literal["combined", "decode_bf16", "promote_fp32"] = "combined",
) -> dict[str, Any]:
    """Pin one completed stage-local repair-weight adapter phase.

    DB519 proves that placing this arithmetic inside the large repair
    executable changes the result.  TPU evidence additionally requires BF16
    decode and FP32 promotion to finish as separate executables.
    """

    if phase not in ("combined", "decode_bf16", "promote_fp32"):
        raise ValueError(f"unknown prefill wk materialization phase: {phase}")

    module = parse_hlo_module(optimized_hlo)
    slots = decoder.config.maximum_full_indexer_slots
    weight_shape = (
        decoder.config.index_key_width,
        decoder.config.hidden_size,
    )
    flat_weight_shape = (weight_shape[0] * weight_shape[1],)
    global_flat_weight_shape = (
        decoder.config.total_devices * flat_weight_shape[0],
    )
    scale_shape = tuple((dimension + 127) // 128 for dimension in weight_shape)

    def parameter_matches(instruction: Any, dtype: str, tail: tuple[int, ...]) -> bool:
        return (
            instruction.computation.startswith("ENTRY ")
            and instruction.raw_opcode == "parameter"
            and len(instruction.result_shapes) == 1
            and instruction.result_shapes[0].dtype == dtype
            and instruction.result_shapes[0].dimensions[-len(tail) :] == tail
        )

    def weight_value_shape(shape: Any) -> bool:
        return (
            shape.dimensions[-2:] == weight_shape
            or shape.dimensions
            in (flat_weight_shape, global_flat_weight_shape)
        )

    raw_parameters = tuple(
        instruction
        for instruction in module.instructions
        if parameter_matches(instruction, "u8", weight_shape)
    )
    scale_parameters = tuple(
        instruction
        for instruction in module.instructions
        if parameter_matches(instruction, "f32", scale_shape)
    )
    bf16_parameters = tuple(
        instruction
        for instruction in module.instructions
        if instruction.computation.startswith("ENTRY ")
        and instruction.raw_opcode == "parameter"
        and len(instruction.result_shapes) == 1
        and instruction.result_shapes[0].dtype == "bf16"
        and weight_value_shape(instruction.result_shapes[0])
    )
    bf16_round_count = sum(
        sum(
            shape.dtype == "bf16" and weight_value_shape(shape)
            for shape in instruction.result_shapes
        )
        for instruction in module.instructions
        if instruction.raw_opcode in ("convert", "fusion")
    )
    fp32_promotion_count = sum(
        sum(
            shape.dtype == "f32" and weight_value_shape(shape)
            for shape in instruction.result_shapes
        )
        for instruction in module.instructions
        if instruction.raw_opcode in ("convert", "fusion")
        and any(shape.dtype == "bf16" for shape in instruction.operand_shapes)
    )
    collectives = tuple(module.collectives)
    lowered = optimized_hlo.lower()
    host_markers = tuple(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in lowered
    )
    violations = []
    expected_raw_parameters = 0 if phase == "promote_fp32" else slots
    expected_scale_parameters = 0 if phase == "promote_fp32" else slots
    expected_bf16_parameters = slots if phase == "promote_fp32" else 0
    if len(raw_parameters) != expected_raw_parameters:
        violations.append(
            "prefill wk materializer raw parameter count drifted: "
            f"expected={expected_raw_parameters} observed={len(raw_parameters)}"
        )
    if len(scale_parameters) != expected_scale_parameters:
        violations.append(
            "prefill wk materializer scale parameter count drifted: "
            f"expected={expected_scale_parameters} observed={len(scale_parameters)}"
        )
    if len(bf16_parameters) != expected_bf16_parameters:
        violations.append(
            "prefill wk materializer BF16 parameter count drifted: "
            f"expected={expected_bf16_parameters} "
            f"observed={len(bf16_parameters)}"
        )
    if phase != "promote_fp32" and bf16_round_count < slots:
        violations.append(
            "prefill wk materializer lost its BF16 adaptation rounds: "
            f"expected_at_least={slots} observed={bf16_round_count}"
        )
    if phase == "promote_fp32" and bf16_round_count:
        violations.append("prefill wk promoter contains a BF16 adaptation round")
    if phase != "decode_bf16" and fp32_promotion_count < slots:
        violations.append(
            "prefill wk materializer lost its FP32 promotions: "
            f"expected_at_least={slots} observed={fp32_promotion_count}"
        )
    if phase == "decode_bf16" and fp32_promotion_count:
        violations.append("prefill wk BF16 decoder contains an FP32 promotion")
    if collectives:
        violations.append("prefill wk materializer contains a collective")
    if host_markers:
        violations.append("prefill wk materializer contains a host callback")
    return {
        "backend": {
            "combined": "external_stage_local_bf16_then_fp32",
            "decode_bf16": "external_stage_local_raw_fp8_to_bf16",
            "promote_fp32": "external_stage_local_bf16_to_fp32",
        }[phase],
        "bf16_round_count": bf16_round_count,
        "bf16_parameter_count": len(bf16_parameters),
        "collective_count": len(collectives),
        "expected_slot_count": slots,
        "fp32_promotion_count": fp32_promotion_count,
        "host_markers": list(host_markers),
        "passed": not violations,
        "phase": phase,
        "raw_parameter_count": len(raw_parameters),
        "scale_parameter_count": len(scale_parameters),
        "violations": violations,
    }


def validate_teacher_forced_prefill_loops(
    optimized_hlo: str,
    *,
    expected_fused_qkv_internal_loops: int,
    expected_prefill_index_repair_loops: int = 0,
) -> dict[str, Any]:
    """Classify every physical prefill loop and reject unknown control flow.

    XLA lowers each zero-spatial fused qkv-a convolution to a bounded internal
    ``while`` on TPU.  Those loops are distinct from the single outer prompt
    scan.  Metadata must prove both identities; count-only exemptions would
    allow unrelated sequential work to enter the prefill executable.
    """

    if (
        not isinstance(expected_fused_qkv_internal_loops, int)
        or isinstance(expected_fused_qkv_internal_loops, bool)
        or expected_fused_qkv_internal_loops < 0
    ):
        raise PlanValidationError(
            "expected fused qkv-a internal loop count must be non-negative"
        )
    if (
        not isinstance(expected_prefill_index_repair_loops, int)
        or isinstance(expected_prefill_index_repair_loops, bool)
        or expected_prefill_index_repair_loops < 0
    ):
        raise PlanValidationError(
            "expected prefill index-repair loop count must be non-negative"
        )
    return _classify_teacher_forced_prefill_loops(
        parse_hlo_module(optimized_hlo),
        expected_fused_qkv_internal_loops=(
            expected_fused_qkv_internal_loops
        ),
        expected_prefill_index_repair_loops=(
            expected_prefill_index_repair_loops
        ),
    )


def _classify_teacher_forced_prefill_loops(
    module: HloModule,
    *,
    expected_fused_qkv_internal_loops: int,
    expected_prefill_index_repair_loops: int = 0,
) -> dict[str, Any]:
    loops = tuple(
        instruction
        for instruction in module.instructions
        if instruction.raw_opcode == "while"
    )
    outer_loops = tuple(
        instruction
        for instruction in loops
        if instruction.op_name == _OUTER_PREFILL_LOOP_OP_NAME
    )
    fused_qkv_internal_loops = tuple(
        instruction
        for instruction in loops
        if instruction.op_name is not None
        and instruction.op_name.startswith(
            f"{_OUTER_PREFILL_LOOP_OP_NAME}/body/"
        )
        and instruction.op_name.endswith(_FUSED_QKV_LOOP_OP_NAME_SUFFIX)
    )
    prefill_index_repair_loops = tuple(
        instruction
        for instruction in loops
        if (
            instruction.op_name is not None
            and instruction.op_name.endswith(
                _PREFILL_INDEX_REPAIR_LOOP_OP_NAME_SUFFIX
            )
        )
        or _is_prefill_index_repair_op_name(instruction.op_name)
    )
    classified_indices = {
        instruction.index
        for instruction in (
            outer_loops
            + fused_qkv_internal_loops
            + prefill_index_repair_loops
        )
    }
    unclassified_loops = tuple(
        instruction
        for instruction in loops
        if instruction.index not in classified_indices
    )
    violations = []
    if len(outer_loops) != 1:
        violations.append(
            "teacher-forced prefill must lower to exactly one outer device "
            f"loop, found {len(outer_loops)}"
        )
    if len(fused_qkv_internal_loops) != expected_fused_qkv_internal_loops:
        violations.append(
            "teacher-forced prefill fused qkv-a internal loop count drifted: "
            f"expected={expected_fused_qkv_internal_loops} "
            f"observed={len(fused_qkv_internal_loops)}"
        )
    if len(prefill_index_repair_loops) != expected_prefill_index_repair_loops:
        violations.append(
            "teacher-forced prefill index-repair loop count drifted: "
            f"expected={expected_prefill_index_repair_loops} "
            f"observed={len(prefill_index_repair_loops)}"
        )
    if unclassified_loops:
        violations.append(
            "teacher-forced prefill contains unclassified physical loops: "
            f"{len(unclassified_loops)}"
        )

    def identities(instructions: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [
            {
                "computation": instruction.computation,
                "instruction": instruction.name,
                "op_name": instruction.op_name,
            }
            for instruction in instructions
        ]

    return {
        "expected_fused_qkv_internal_loop_count": (
            expected_fused_qkv_internal_loops
        ),
        "expected_total_loop_count": (
            1
            + expected_fused_qkv_internal_loops
            + expected_prefill_index_repair_loops
        ),
        "fused_qkv_internal_loop_count": len(fused_qkv_internal_loops),
        "fused_qkv_internal_loops": identities(fused_qkv_internal_loops),
        "loop_count": len(loops),
        "outer_loop_count": len(outer_loops),
        "outer_loops": identities(outer_loops),
        "passed": not violations,
        "prefill_index_repair_loop_count": len(
            prefill_index_repair_loops
        ),
        "prefill_index_repair_loops": identities(
            prefill_index_repair_loops
        ),
        "expected_prefill_index_repair_loop_count": (
            expected_prefill_index_repair_loops
        ),
        "unclassified_loops": identities(unclassified_loops),
        "violations": violations,
    }


def validate_stage_local_prefill_index_repair_hlo(
    module: HloModule,
    *,
    program: TeacherForcedPrefillProgram,
    schedule: PipelineSchedule,
    backend_contract: PrefillBackendContract,
) -> dict[str, Any]:
    """Pin the DB518 repair without weakening the recurrent decoder."""

    decoder = program.decoder
    full_indexer_layers = sum(
        layer.indexer_kind == "full"
        for stage in schedule.stages
        for layer in stage.layers
    )
    chunk_count = _prefill_index_repair_chunk_count(program.prompt_length)
    expected_calls = full_indexer_layers * chunk_count

    def has_result_shape(
        instruction: Any, dtype: str, dimensions: tuple[int, ...]
    ) -> bool:
        return any(
            shape.dtype == dtype and shape.dimensions == dimensions
            for shape in instruction.result_shapes
        )

    scoped = tuple(
        instruction
        for instruction in module.instructions
        if _is_prefill_index_repair_op_name(instruction.op_name)
    )
    projection_opcodes = (
        ("dot", "convolution")
        if backend_contract == "cpu_reference"
        else ("convolution",)
    )
    projections = tuple(
        instruction
        for instruction in scoped
        if instruction.raw_opcode in projection_opcodes
        and has_result_shape(
            instruction,
            "f32",
            (
                _PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,
                decoder.config.index_key_width,
            ),
        )
    )
    exact_projection_operands = tuple(
        instruction
        for instruction in projections
        if len(instruction.operand_shapes) >= 2
        and instruction.operand_shapes[0].dtype
        in (
            ("bf16", "f32")
            if backend_contract == "cpu_reference"
            else ("bf16",)
        )
        and instruction.operand_shapes[0].dimensions
        == (
            _PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,
            decoder.config.hidden_size,
        )
        and instruction.operand_shapes[1].dtype == "f32"
        and instruction.operand_shapes[1].dimensions
        == (
            decoder.config.index_key_width,
            decoder.config.hidden_size,
        )
    )
    physical_sqrt = tuple(
        instruction
        for instruction in scoped
        if instruction.raw_opcode in ("sqrt", "fusion")
        and has_result_shape(
            instruction,
            "f32",
            (_PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,),
        )
        and instruction.op_name is not None
        and "/sqrt" in instruction.op_name
    )
    physical_affine = tuple(
        instruction
        for instruction in scoped
        if instruction.raw_opcode in ("add", "fusion")
        and has_result_shape(
            instruction,
            "f32",
            (
                _PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,
                decoder.config.index_key_width,
            ),
        )
        and instruction.op_name is not None
        and "/add" in instruction.op_name
    )
    grouped_sqrt = tuple(
        instruction
        for instruction in scoped
        if instruction.raw_opcode in ("sqrt", "fusion")
        and has_result_shape(
            instruction,
            "f32",
            (
                _PREFILL_INDEX_REPAIR_CHUNK
                // _PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,
                _PREFILL_INDEX_REPAIR_PHYSICAL_ROWS,
            ),
        )
        and instruction.op_name is not None
        and "/sqrt" in instruction.op_name
    )
    cache_writes = tuple(
        instruction
        for instruction in module.instructions
        if instruction.raw_opcode in ("scatter", "dynamic-update-slice")
        and _is_prefill_index_repair_op_name(instruction.op_name)
    )
    repair_collectives = tuple(
        instruction
        for instruction in module.collectives
        if _is_prefill_index_repair_op_name(instruction.op_name)
    )
    materialized_wk_parameters = tuple(
        instruction
        for instruction in module.instructions
        if instruction.raw_opcode == "parameter"
        and len(instruction.result_shapes) == 1
        and instruction.result_shapes[0].dtype == "f32"
        and instruction.result_shapes[0].dimensions[-2:]
        == (decoder.config.index_key_width, decoder.config.hidden_size)
    )
    repair_weight_rounds = tuple(
        instruction
        for instruction in scoped
        if instruction.raw_opcode in ("convert", "fusion")
        and any(
            shape.dtype == "bf16"
            and shape.dimensions
            in (
                (
                    decoder.config.index_key_width,
                    decoder.config.hidden_size,
                ),
                (
                    decoder.config.index_key_width
                    * decoder.config.hidden_size,
                ),
            )
            for shape in instruction.result_shapes
        )
    )
    history_shape_values = {
        (shape.dtype, shape.dimensions): shape
        for instruction in module.instructions
        for shape in instruction.result_shapes
        if shape.dtype == "bf16"
        if program.prompt_length in shape.dimensions
        and decoder.config.hidden_size in shape.dimensions
    }
    history_shapes = tuple(
        history_shape_values[key].to_dict()
        for key in sorted(history_shape_values)
    )
    full_pod_history_shapes = tuple(
        shape
        for shape in history_shapes
        if backend_contract != "cpu_reference"
        and decoder.config.total_devices in shape["dimensions"]
    )
    lowered = "\n".join(
        instruction.raw_line.lower() for instruction in scoped
    )
    forbidden_markers = tuple(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in lowered
    )
    violations: list[str] = []
    if len(projections) != expected_calls:
        violations.append(
            "prefill index repair projection count drifted: "
            f"expected={expected_calls} observed={len(projections)}"
        )
    if len(exact_projection_operands) != expected_calls:
        lhs_requirement = (
            "backend-lowered M64"
            if backend_contract == "cpu_reference"
            else "BF16-M64"
        )
        violations.append(
            "prefill index repair lost exact "
            f"{lhs_requirement}/FP32-wk operands: "
            f"expected={expected_calls} "
            f"observed={len(exact_projection_operands)}"
        )
    if backend_contract != "cpu_reference":
        if any(
            "dim_labels=bf_oi->bf" not in instruction.raw_line
            for instruction in projections
        ):
            violations.append(
                "prefill index repair convolution dimension labels drifted"
            )
        expected_physical_sqrt = 2 * expected_calls
        if len(physical_sqrt) != expected_physical_sqrt:
            violations.append(
                "prefill index repair physical key-norm sqrt count drifted: "
                f"expected={expected_physical_sqrt} "
                f"observed={len(physical_sqrt)}"
            )
        if len(physical_affine) < expected_calls:
            violations.append(
                "prefill index repair physical key-norm affine count drifted: "
                f"expected_at_least={expected_calls} "
                f"observed={len(physical_affine)}"
            )
        if len(cache_writes) < expected_calls:
            violations.append(
                "prefill index repair cache-write count drifted: "
                f"expected_at_least={expected_calls} "
                f"observed={len(cache_writes)}"
            )
    if grouped_sqrt:
        violations.append(
            "prefill index repair retained grouped [32,64] key-norm sqrt"
        )
    if repair_collectives:
        violations.append("prefill index repair contains a collective")
    if len(materialized_wk_parameters) < (
        decoder.config.maximum_full_indexer_slots
    ):
        violations.append(
            "prefill index repair lost its external FP32 wk parameters: "
            f"expected_at_least={decoder.config.maximum_full_indexer_slots} "
            f"observed={len(materialized_wk_parameters)}"
        )
    if repair_weight_rounds:
        violations.append(
            "prefill index repair rematerializes raw wk inside the executable"
        )
    if not history_shapes:
        violations.append("prefill index repair lost its stage-local history")
    if full_pod_history_shapes:
        violations.append(
            "prefill index repair materializes full-pod prompt history"
        )
    if forbidden_markers:
        violations.append("prefill index repair contains host callbacks")

    def identities(instructions: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [
            {
                "instruction": instruction.name,
                "op_name": instruction.op_name,
                "operand_shapes": [
                    shape.to_dict() for shape in instruction.operand_shapes
                ],
                "result_shapes": [
                    shape.to_dict() for shape in instruction.result_shapes
                ],
            }
            for instruction in instructions
        ]

    return {
        "backend": "physical_m64_chunk",
        "cache_write_count": len(cache_writes),
        "chunk_count": chunk_count,
        "expected_call_count": expected_calls,
        "exact_projection_operand_count": len(exact_projection_operands),
        "forbidden_markers": list(forbidden_markers),
        "full_indexer_layer_count": full_indexer_layers,
        "full_pod_history_shapes": list(full_pod_history_shapes),
        "grouped_sqrt_count": len(grouped_sqrt),
        "history_estimated_bytes_per_device": (
            program.prompt_length
            * decoder.config.maximum_full_indexer_slots
            * decoder.config.hidden_size
            * 2
        ),
        "history_shapes": list(history_shapes),
        "materialized_wk_parameter_count": len(materialized_wk_parameters),
        "passed": not violations,
        "physical_affine_count": len(physical_affine),
        "physical_sqrt_count": len(physical_sqrt),
        "projection_count": len(projections),
        "projections": identities(projections),
        "repair_collectives": identities(repair_collectives),
        "repair_weight_round_count": len(repair_weight_rounds),
        "violations": violations,
    }


def validate_teacher_forced_prefill_hlo(
    optimized_hlo: str,
    *,
    program: TeacherForcedPrefillProgram,
    schedule: PipelineSchedule,
    backend_contract: PrefillBackendContract,
) -> dict[str, Any]:
    """Pin one device loop around one complete, topology-local token step."""

    decoder = program.decoder
    index_repair_enabled = (
        program.index_repair_backend == "physical_m64_chunk"
    )
    if index_repair_enabled != decoder.observe_prefill_index_inputs:
        raise PlanValidationError(
            "teacher-forced prefill index-repair identity drifted"
        )
    decoder_contract = validate_decoder_step_hlo(
        optimized_hlo,
        config=decoder.config,
        schedule=schedule,
        groups=decoder.groups,
        pairs=decoder.pairs,
        backend_contract=backend_contract,
        feature_output_tile=decoder.feature_output_tile,
        feature_fuse_route_weighting=(
            decoder.feature_fuse_route_weighting
        ),
        feature_reconstruct_down_fp32=(
            decoder.feature_reconstruct_down_fp32
        ),
        dsa_query_backend=decoder.dsa_query_backend,
        dsa_query_exact_association=(
            decoder.dsa_query_exact_association
        ),
        dsa_head_key_exact_association=(
            decoder.dsa_head_key_exact_association
        ),
        attention_projection_backend=(
            decoder.attention_projection_backend
        ),
        complete_token_path=True,
        split_residual_state=decoder.split_residual_state,
        prefill_index_repair=index_repair_enabled,
        main_rope_table_enabled=decoder.main_rope_table_enabled,
        pregathered_b512_attention=(
            decoder.pregathered_b512_attention
        ),
    )
    module = parse_hlo_module(optimized_hlo)
    expected_fused_qkv_internal_loops = (
        schedule.layer_count
        if decoder.attention_projection_backend == "fused_n82_convolution"
        else 0
    )
    full_indexer_layers = sum(
        layer.indexer_kind == "full"
        for stage in schedule.stages
        for layer in stage.layers
    )
    expected_prefill_index_repair_loops = (
        full_indexer_layers
        * _prefill_index_repair_chunk_count(program.prompt_length)
        if index_repair_enabled
        else 0
    )
    loop_contract = _classify_teacher_forced_prefill_loops(
        module,
        expected_fused_qkv_internal_loops=(
            expected_fused_qkv_internal_loops
        ),
        expected_prefill_index_repair_loops=(
            expected_prefill_index_repair_loops
        ),
    )
    index_repair_contract: dict[str, Any] = {
        "backend": "none",
        "passed": True,
        "violations": [],
    }
    if index_repair_enabled:
        index_repair_contract = (
            validate_stage_local_prefill_index_repair_hlo(
                module,
                program=program,
                schedule=schedule,
                backend_contract=backend_contract,
            )
        )
    prompt_shape_parameter_count = sum(
        shape.dtype == "s32"
        and shape.dimensions == (program.prompt_length,)
        for instruction in module.instructions
        if instruction.raw_opcode == "parameter"
        for shape in instruction.result_shapes
    )
    dead_prompt_rows = [
        {
            "instruction": instruction.name,
            "shape": instruction.result_shapes[0].to_dict(),
        }
        for instruction in module.instructions
        if instruction.raw_opcode == "parameter"
        and len(instruction.result_shapes) == 1
        and instruction.result_shapes[0].dtype == "s32"
        and instruction.result_shapes[0].dimensions
        in (
            (decoder.config.total_devices, program.prompt_length),
            (decoder.config.total_devices, 1, program.prompt_length),
        )
    ]
    lowered = optimized_hlo.lower()
    host_transfer_markers = tuple(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in lowered
    )
    host_transfer_opcodes = tuple(
        sorted(
            {
                instruction.raw_opcode
                for instruction in module.instructions
                if instruction.raw_opcode
                in {
                    "infeed",
                    "outfeed",
                    "recv",
                    "recv-done",
                    "send",
                    "send-done",
                }
            }
        )
    )
    violations = list(decoder_contract["violations"])
    violations.extend(loop_contract["violations"])
    violations.extend(index_repair_contract["violations"])
    if prompt_shape_parameter_count == 0:
        violations.append(
            "teacher-forced prefill lost its one-dimensional prompt token input"
        )
    if dead_prompt_rows:
        violations.append(
            "teacher-forced prefill contains a full-pod prompt-row tensor"
        )
    if host_transfer_markers or host_transfer_opcodes:
        violations.append(
            "teacher-forced prefill contains host-staged execution markers"
        )
    return {
        "backend_contract": backend_contract,
        "dead_prompt_rows": dead_prompt_rows,
        "decoder_contract": decoder_contract,
        "host_transfer_markers": list(host_transfer_markers),
        "host_transfer_opcodes": list(host_transfer_opcodes),
        "index_repair_backend": program.index_repair_backend,
        "index_repair_contract": index_repair_contract,
        "loop_count": loop_contract["loop_count"],
        "loop_contract": loop_contract,
        "outer_loop_count": loop_contract["outer_loop_count"],
        "passed": not violations,
        "prompt_length": program.prompt_length,
        "prompt_shape_parameter_count": prompt_shape_parameter_count,
        "violations": violations,
    }


def build_teacher_forced_prefill_program(
    decoder: DecoderStepProgram,
    *,
    prompt_length: int,
) -> TeacherForcedPrefillProgram:
    """Build an exact sequential prefill reference for short-context Gate D.

    The scan teacher-forces each prompt token into the complete decoder step.
    Its final token output is therefore the model's first greedy token after
    the prompt, while its KV/index state contains every prompt position.
    """

    if not decoder.complete_token_path:
        raise PlanValidationError(
            "teacher-forced prefill requires a complete-token decoder"
        )
    if getattr(decoder, "observe_dsa_events", False):
        raise PlanValidationError(
            "teacher-forced prefill requires the observation-free decoder"
        )
    if (
        not isinstance(prompt_length, int)
        or isinstance(prompt_length, bool)
        or prompt_length <= 0
    ):
        raise PlanValidationError("prefill prompt length must be positive")
    if prompt_length >= decoder.config.context_capacity:
        raise PlanValidationError(
            "prefill prompt must leave capacity for at least one generated token"
        )

    import jax.numpy as jnp
    from jax import lax

    total_devices = decoder.config.total_devices

    def execute(
        weights: Any,
        residual: Any,
        kv_cache: Any,
        index_cache: Any,
        metadata: Any,
        prompt_tokens: Any,
        position: Any,
        block_tables: Any,
        context_lengths: Any,
        materialized_index_wk: tuple[Any, ...] | None = None,
        dsa_query_weight_aliases: tuple[tuple[Any, ...], ...] | None = None,
        main_rope_table: Any | None = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        if tuple(prompt_tokens.shape) != (prompt_length,):
            raise PlanValidationError(
                "teacher-forced prompt token shape differs from the compiled length"
            )
        if prompt_tokens.dtype != jnp.dtype(jnp.int32):
            raise PlanValidationError(
                "teacher-forced prompt tokens must have dtype int32"
            )
        if decoder.main_rope_table_enabled != (main_rope_table is not None):
            raise PlanValidationError(
                "teacher-forced prefill main-RoPE table state drifted"
            )
        initial_prediction = jnp.full(
            (total_devices, 1), -1, dtype=jnp.int32
        )
        initial = (
            residual,
            kv_cache,
            index_cache,
            metadata,
            position,
            block_tables,
            context_lengths,
            initial_prediction,
        )

        def scan_step(
            carry: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
            prompt_token: Any,
        ) -> tuple[
            tuple[Any, Any, Any, Any, Any, Any, Any, Any], Any | None
        ]:
            (
                current_residual,
                current_kv,
                current_index,
                current_metadata,
                current_position,
                current_blocks,
                current_lengths,
                _,
            ) = carry
            distributed_token = jnp.broadcast_to(
                jnp.asarray(prompt_token, dtype=jnp.int32),
                (total_devices, 1),
            )
            decoder_state = (
                current_residual,
                current_kv,
                current_index,
                current_metadata,
                distributed_token,
                current_position,
                current_blocks,
                current_lengths,
            )
            if decoder.dsa_query_exact_association:
                if dsa_query_weight_aliases is None:
                    raise PlanValidationError(
                        "exact DSA prefill requires four query aliases"
                    )
                exact_dsa_weights: Any = dsa_query_weight_aliases
                if decoder.dsa_head_key_exact_association:
                    if materialized_index_wk is None:
                        raise PlanValidationError(
                            "exact DSA head/key prefill requires FP32 wk owners"
                        )
                    exact_dsa_weights = (
                        dsa_query_weight_aliases,
                        materialized_index_wk,
                    )
                decoder_inputs = (
                    weights,
                    exact_dsa_weights,
                    *decoder_state,
                )
                if decoder.main_rope_table_enabled:
                    decoder_inputs = (*decoder_inputs, main_rope_table)
                output = decoder.execute(*decoder_inputs)
            else:
                if dsa_query_weight_aliases is not None:
                    raise PlanValidationError(
                        "default prefill must not receive query aliases"
                    )
                decoder_inputs = (weights, *decoder_state)
                if decoder.main_rope_table_enabled:
                    decoder_inputs = (*decoder_inputs, main_rope_table)
                output = decoder.execute(*decoder_inputs)
            next_carry = (
                output[0],
                output[1],
                output[2],
                output[3],
                output[5],
                output[6],
                output[7],
                output[4],
            )
            observation = (
                output[8]
                if decoder.observe_prefill_index_inputs
                else None
            )
            return next_carry, observation

        final, prompt_index_inputs = lax.scan(
            scan_step, initial, prompt_tokens, unroll=1
        )
        if decoder.observe_prefill_index_inputs:
            if decoder.repair_prefill_index_cache is None:
                raise PlanValidationError(
                    "prefill index repair executable is unavailable"
                )
            if materialized_index_wk is None:
                raise PlanValidationError(
                    "prefill index repair requires externally materialized wk"
                )
            final = (
                final[0],
                final[1],
                decoder.repair_prefill_index_cache(
                    weights,
                    materialized_index_wk,
                    final[2],
                    prompt_index_inputs,
                    final[5],
                ),
                final[3],
                final[4],
                final[5],
                final[6],
                final[7],
            )
        elif materialized_index_wk is not None:
            raise PlanValidationError(
                "default prefill must not receive materialized repair weights"
            )
        return (
            final[0],
            final[1],
            final[2],
            final[3],
            final[7],
            final[4],
            final[5],
            final[6],
        )

    return TeacherForcedPrefillProgram(
        decoder=decoder,
        execute=execute,
        index_repair_backend=(
            "physical_m64_chunk"
            if decoder.observe_prefill_index_inputs
            else "none"
        ),
        prompt_length=prompt_length,
    )
