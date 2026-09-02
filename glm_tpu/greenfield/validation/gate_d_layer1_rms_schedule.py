"""Contracts and classification for the bounded layer-1 RMS schedule discriminator.

The captured-RMS probe's optimized-HLO contract and the accepted-schedule
fingerprint live here so the sealed-archive driver and the probe share one
implementation.  ``fp32_carry_schedule`` selects the accepted-schedule arm
(FP32 carry, one FP32 barrier on the ``[32,6144]`` sum, ``f32[32]`` scheduled
reduction); ``False`` selects the control arm.
"""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any

import ml_dtypes
import numpy as np

from glm_tpu.greenfield.validation.hlo_dependency import (
    _instruction_key,
    _shape_signatures,
    _value_depends_on,
)

DB548_LAYER1_NORMALIZED_SHA256 = (
    "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
)
ACCEPTED_LAYER1_NORMALIZED_SHA256 = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _compare_bits(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    if expected.shape != observed.shape or expected.dtype != np.uint16 or (
        observed.dtype != np.uint16
    ):
        raise ValueError("layer-1 RMS comparison shape/dtype drifted")
    expected_f32 = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_f32 = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    absolute = np.abs(observed_f32 - expected_f32)
    mismatch = expected != observed
    return {
        "elementwise_exact": bool(np.array_equal(expected, observed)),
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": (
            int(np.flatnonzero(mismatch)[0]) if np.any(mismatch) else None
        ),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()),
        "mismatch_count": int(np.count_nonzero(mismatch)),
        "observed_sha256": _array_sha256(observed),
        "shape": list(expected.shape),
    }


def _exact_accepted_rms_schedule(value: Any) -> bool:
    if value.raw_opcode != "fusion" or _shape_signatures(
        value.result_shapes
    ) != ("f32[32]",):
        return False
    marker = "backend_config="
    if marker not in value.raw_line:
        return False
    try:
        config = json.loads(value.raw_line.split(marker, 1)[1])
    except json.JSONDecodeError:
        return False
    window = config.get("window_config", {})
    megacore = config.get("megacore_config", {})
    return bool(
        window.get("kernel_window_bounds") == []
        and window.get("output_window_bounds") == ["2", "48"]
        and window.get("input_window_bounds") == []
        and window.get("iteration_bounds") == ["2", "1"]
        and window.get("cost_model_type") == "COST_MODEL_TYPE_INVALID"
        and window.get("is_mask") is False
        and window.get("pad_input_on_minor_dim") == "0"
        and window.get("pad_output_on_minor_dim") == "0"
        and megacore.get("megacore_split_dim") == "0"
        and megacore.get("megacore_allreduce_bytes") == "4096"
    )


def audit_layer1_rms_schedule_optimized_hlo(
    optimized_hlo: str,
    *,
    fp32_carry_schedule: bool,
) -> dict[str, Any]:
    """Pin post-scheduling liveness; StableHLO pins exact arithmetic."""

    from glm_tpu.greenfield.sharding.hlo_contract import (
        COLLECTIVE_OPCODES,
        parse_hlo_module,
    )

    module = parse_hlo_module(optimized_hlo)
    violations: list[str] = []
    async_collectives = [
        item.name
        for item in module.instructions
        if any(
            item.raw_opcode == f"{opcode}-{suffix}"
            for opcode in COLLECTIVE_OPCODES
            for suffix in ("start", "done")
        )
    ]
    collectives = list(module.collectives)
    gathers = [
        item
        for item in collectives
        if "greenfield_captured_rms_strategy_gather"
        in (item.op_name or "").split("/")
    ]
    gather = gathers[0] if len(gathers) == 1 else None
    gather_dimensions_exact = False
    if gather is not None:
        sanitized_gather = re.sub(r"/\*.*?\*/", "", gather.raw_line)
        sanitized_gather = re.sub(
            r'"(?:\\.|[^"\\])*"', '""', sanitized_gather
        )
        sanitized_gather = re.sub(r"\s+", "", sanitized_gather)
        gather_dimensions_exact = re.findall(
            r"\bdimensions=\{([^}]*)\}", sanitized_gather
        ) == ["0"]
    if module.num_partitions != 4 or module.num_replicas not in (None, 1):
        violations.append("captured RMS optimized module cardinality drifted")
    if async_collectives:
        violations.append(f"async collectives are forbidden: {async_collectives}")
    if len(collectives) != 1 or len(gathers) != 1:
        violations.append(
            "captured RMS requires one scoped gather: "
            f"collectives={len(collectives)} gathers={len(gathers)}"
        )
    if gather is not None and (
        gather.opcode != "all-gather"
        or gather.replica_groups != ((0, 1, 2, 3),)
        or not gather.use_global_device_ids
        or _shape_signatures(gather.operand_shapes)
        != ("bf16[1,8,1,6144]",)
        or _shape_signatures(gather.result_shapes)
        not in {("bf16[4,8,1,6144]",), ("bf16[32,1,6144]",)}
        or not gather_dimensions_exact
    ):
        violations.append("captured RMS gather geometry drifted")
    roots = [
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
        and item.raw_line.lstrip().startswith("ROOT ")
    ]
    root = roots[0] if len(roots) == 1 else None
    if root is None or _shape_signatures(root.result_shapes) != (
        "bf16[1,6144]",
    ):
        violations.append("captured RMS ENTRY result drifted")
    entry_parameters = [
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ") and item.raw_opcode == "parameter"
    ]
    expected_parameter_shapes = (
        "bf16[1,6144]",
        "bf16[1,6144]",
        "bf16[1,8,1,6144]",
        "bf16[6144]",
    )
    if (
        len(entry_parameters) != 4
        or tuple(sorted(
            shape
            for item in entry_parameters
            for shape in _shape_signatures(item.result_shapes)
        ))
        != expected_parameter_shapes
    ):
        violations.append("captured RMS ENTRY inputs drifted")

    by_key = {_instruction_key(item): item for item in module.instructions}
    layout_opcodes = {
        "bitcast",
        "copy",
        "optimization-barrier",
        "reshape",
        "slice",
    }

    def layout_predecessor(value: Any) -> Any | None:
        if value.raw_opcode not in layout_opcodes or len(value.operand_names) != 1:
            return None
        predecessor = by_key.get(
            (value.computation, value.operand_names[0])
        )
        if (
            predecessor is None
            or len(value.result_shapes) != 1
            or len(predecessor.result_shapes) != 1
            or value.result_shapes[0].dtype != predecessor.result_shapes[0].dtype
        ):
            return None
        output_elements = value.result_shapes[0].element_count
        input_elements = predecessor.result_shapes[0].element_count
        if value.raw_opcode == "slice":
            sanitized = re.sub(
                r'"(?:\\.|[^"\\])*"', '""', value.raw_line
            )
            sanitized = re.sub(r"/\*.*?\*/", "", sanitized)
            sanitized = re.sub(r"\s+", "", sanitized)
            exact_row_zero = bool(
                _shape_signatures(predecessor.result_shapes)
                == ("bf16[32,6144]",)
                and _shape_signatures(value.result_shapes)
                == ("bf16[1,6144]",)
                and re.findall(
                    r"\bslice=\{\[0:1\],\[0:6144\]\}",
                    sanitized,
                )
                == ["slice={[0:1],[0:6144]}"]
            )
            if not exact_row_zero:
                return None
        elif output_elements != input_elements:
            return None
        return predecessor

    def unwrap_layout(value: Any) -> Any:
        seen: set[tuple[str, str]] = set()
        current = value
        while _instruction_key(current) not in seen:
            seen.add(_instruction_key(current))
            predecessor = layout_predecessor(current)
            if predecessor is None:
                return current
            current = predecessor
        return current

    output_fusion = unwrap_layout(root) if root is not None else None
    exact_output_fusion = bool(
        output_fusion is not None and output_fusion.raw_opcode == "fusion"
    )
    exact_result_binding = bool(
        exact_output_fusion
        and gather is not None
        and _value_depends_on(module, root, gather)
        and all(
            _value_depends_on(module, root, parameter)
            for parameter in entry_parameters
        )
    )
    if not exact_result_binding:
        violations.append("captured RMS result liveness drifted")
    scheduled_reductions = [
        item
        for item in module.instructions
        if item.computation.startswith("ENTRY ")
        and _shape_signatures(item.result_shapes) == ("f32[32]",)
        and "greenfield_captured_rms_layer1"
        in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
    ]
    accepted_scheduled_reductions = [
        item for item in scheduled_reductions if _exact_accepted_rms_schedule(item)
    ]

    def exact_schedule_output_edge(schedule: Any) -> bool:
        if output_fusion is None:
            return False
        matches = 0
        for operand_name in output_fusion.operand_names:
            operand = by_key.get((output_fusion.computation, operand_name))
            if operand is None:
                continue
            candidate = unwrap_layout(operand)
            candidate_layout_line = re.sub(
                r'"(?:\\.|[^"\\])*"', '""', candidate.raw_line
            )
            candidate_layout_line = re.sub(
                r"/\*.*?\*/", "", candidate_layout_line
            )
            if (
                candidate.raw_opcode == "bitcast"
                and _shape_signatures(candidate.operand_shapes)
                == ("f32[32]",)
                and _shape_signatures(candidate.result_shapes)
                == ("f32[1]",)
                and len(candidate.operand_names) == 1
                and re.search(
                    r"=\s*f32\[1\]\{0:T\(128\)S\(3\)\}\s+bitcast\(",
                    candidate_layout_line,
                )
                is not None
            ):
                predecessor = by_key.get(
                    (candidate.computation, candidate.operand_names[0])
                )
                if predecessor is not None:
                    predecessor_layout_line = re.sub(
                        r'"(?:\\.|[^"\\])*"', '""', predecessor.raw_line
                    )
                    predecessor_layout_line = re.sub(
                        r"/\*.*?\*/", "", predecessor_layout_line
                    )
                    if re.search(
                        r"=\s*f32\[32\]\{0:T\(128\)S\(3\)\}\s+fusion\(",
                        predecessor_layout_line,
                    ) is not None:
                        candidate = predecessor
            if (
                not fp32_carry_schedule
                and _instruction_key(candidate) == _instruction_key(schedule)
            ):
                matches += 1
                continue
            if (
                candidate.raw_opcode == "fusion"
                and "greenfield_captured_rms_layer1"
                in (candidate.op_name or "")
                and (candidate.op_name or "").split("/")[-1] == "rsqrt"
                and len(candidate.operand_names) == 1
                and (
                    source := by_key.get(
                        (candidate.computation, candidate.operand_names[0])
                    )
                )
                is not None
                and _instruction_key(unwrap_layout(source))
                == _instruction_key(schedule)
            ):
                matches += 1
        return matches == 1

    exact_scheduled_reduction_binding = bool(
        not fp32_carry_schedule
        or (
            len(scheduled_reductions) == 1
            and len(accepted_scheduled_reductions) == 1
            and root is not None
            and gather is not None
            and _value_depends_on(
                module, accepted_scheduled_reductions[0], gather
            )
            and exact_schedule_output_edge(accepted_scheduled_reductions[0])
        )
    )
    if not exact_scheduled_reduction_binding:
        violations.append("accepted captured RMS schedule/liveness drifted")
    if any(item.raw_opcode == "convolution" for item in module.instructions):
        violations.append("captured RMS replay contains a convolution")
    custom_calls = [
        item.name
        for item in module.instructions
        if item.raw_opcode == "custom-call"
        and 'custom_call_target="tpu_custom_call"' in item.raw_line
    ]
    if custom_calls:
        violations.append(f"captured RMS replay contains Pallas calls: {custom_calls}")
    forbidden = [
        marker
        for marker in ("host_callback", "xla_python_cpu_callback", " outfeed(")
        if marker in optimized_hlo
    ]
    if forbidden:
        violations.append(f"captured RMS replay contains host effects: {forbidden}")
    return {
        "accepted_scheduled_reduction_values": [
            item.name for item in accepted_scheduled_reductions
        ],
        "async_collectives": async_collectives,
        "collective_count": len(collectives),
        "exact_output_fusion": exact_output_fusion,
        "exact_result_binding": exact_result_binding,
        "exact_scheduled_reduction_binding": exact_scheduled_reduction_binding,
        "live_rows": 1,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "performance_claim": False,
        "fp32_carry_schedule": fp32_carry_schedule,
        "passed": not violations,
        "violations": violations,
    }



def audit_layer1_rms_schedule_stablehlo(
    stablehlo: str, *, fp32_carry_schedule: bool
) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.stablehlo_dense_convolution import (
        validate_captured_dense_rms_stablehlo,
    )

    return validate_captured_dense_rms_stablehlo(
        stablehlo, split_layer1_rms=fp32_carry_schedule
    )


def classify_layer1_rms_schedule_outputs(
    *,
    control_bits: np.ndarray,
    schedule_bits: np.ndarray,
    db548_bits: np.ndarray,
    accepted_bits: np.ndarray,
) -> dict[str, Any]:
    """Compare both arms against their sealed references.

    The control arm must reproduce the DB548 greenfield row exactly or the
    harness is refused.  The accepted-schedule arm is exact iff it reproduces
    the accepted legacy row.
    """

    for name, value in (
        ("control", control_bits),
        ("schedule", schedule_bits),
        ("db548", db548_bits),
        ("accepted", accepted_bits),
    ):
        value = np.asarray(value)
        if value.shape != (6144,) or value.dtype != np.uint16:
            raise ValueError(f"layer-1 RMS {name} row geometry drifted")
    if _array_sha256(db548_bits) != DB548_LAYER1_NORMALIZED_SHA256:
        raise ValueError("DB548 layer-1 reference row drifted")
    if _array_sha256(accepted_bits) != ACCEPTED_LAYER1_NORMALIZED_SHA256:
        raise ValueError("accepted layer-1 reference row drifted")
    control = _compare_bits(np.asarray(db548_bits), np.asarray(control_bits))
    schedule = _compare_bits(np.asarray(accepted_bits), np.asarray(schedule_bits))
    schedule_vs_db548 = _compare_bits(np.asarray(db548_bits), np.asarray(schedule_bits))
    harness_admissible = bool(control["elementwise_exact"])
    schedule_exact = bool(harness_admissible and schedule["elementwise_exact"])
    return {
        "accepted_layer1_normalized_sha256": ACCEPTED_LAYER1_NORMALIZED_SHA256,
        "control_vs_db548": control,
        "db548_layer1_normalized_sha256": DB548_LAYER1_NORMALIZED_SHA256,
        "harness_admissible": harness_admissible,
        "schedule_arm_exact": schedule_exact,
        "schedule_vs_accepted": schedule,
        "schedule_vs_db548": schedule_vs_db548,
    }
