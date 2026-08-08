"""Correctness-first teacher-forced prefill over the complete PP8 token step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..errors import PlanValidationError
from ..model.schedule import PipelineSchedule
from ..sharding.hlo_contract import HloModule, parse_hlo_module
from .decoder import DecoderStepProgram, validate_decoder_step_hlo


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
    prompt_length: int


_OUTER_PREFILL_LOOP_OP_NAME = "jit(execute)/while"
_FUSED_QKV_LOOP_OP_NAME_SUFFIX = (
    "/one_row_fused_qkv_a_n82_convolution/while"
)


def validate_teacher_forced_prefill_loops(
    optimized_hlo: str,
    *,
    expected_fused_qkv_internal_loops: int,
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
    return _classify_teacher_forced_prefill_loops(
        parse_hlo_module(optimized_hlo),
        expected_fused_qkv_internal_loops=(
            expected_fused_qkv_internal_loops
        ),
    )


def _classify_teacher_forced_prefill_loops(
    module: HloModule,
    *,
    expected_fused_qkv_internal_loops: int,
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
    classified_indices = {
        instruction.index
        for instruction in outer_loops + fused_qkv_internal_loops
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
        "expected_total_loop_count": 1 + expected_fused_qkv_internal_loops,
        "fused_qkv_internal_loop_count": len(fused_qkv_internal_loops),
        "fused_qkv_internal_loops": identities(fused_qkv_internal_loops),
        "loop_count": len(loops),
        "outer_loop_count": len(outer_loops),
        "outer_loops": identities(outer_loops),
        "passed": not violations,
        "unclassified_loops": identities(unclassified_loops),
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
        attention_projection_backend=(
            decoder.attention_projection_backend
        ),
        complete_token_path=True,
        split_residual_state=decoder.split_residual_state,
    )
    module = parse_hlo_module(optimized_hlo)
    expected_fused_qkv_internal_loops = (
        schedule.layer_count
        if decoder.attention_projection_backend == "fused_n82_convolution"
        else 0
    )
    loop_contract = _classify_teacher_forced_prefill_loops(
        module,
        expected_fused_qkv_internal_loops=(
            expected_fused_qkv_internal_loops
        ),
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
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        if tuple(prompt_tokens.shape) != (prompt_length,):
            raise PlanValidationError(
                "teacher-forced prompt token shape differs from the compiled length"
            )
        if prompt_tokens.dtype != jnp.dtype(jnp.int32):
            raise PlanValidationError(
                "teacher-forced prompt tokens must have dtype int32"
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
            tuple[Any, Any, Any, Any, Any, Any, Any, Any], None
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
            output = decoder.execute(
                weights,
                current_residual,
                current_kv,
                current_index,
                current_metadata,
                distributed_token,
                current_position,
                current_blocks,
                current_lengths,
            )
            return (
                output[0],
                output[1],
                output[2],
                output[3],
                output[5],
                output[6],
                output[7],
                output[4],
            ), None

        final, _ = lax.scan(scan_step, initial, prompt_tokens, unroll=1)
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
        prompt_length=prompt_length,
    )
