"""Fail-closed HLO proof for the integrated layer-0 boundary diagnostic."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping, Sequence

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    lint_hlo,
    parse_hlo_module,
)
from .association_fingerprint import validate_strategy_nd_reduction
from .dense_rms_replay import (
    _exact_accepted_rms_schedule,
    _shape_signature,
    _validate_exact_dense_rms_value_flow,
)


INTEGRATED_DENSE_RMS_STABLEHLO_SHA256 = (
    "0ae728d6d6ddedbc7d2818975f07b03bc518f26950091bdb592c6574c07853b2"
)
INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256 = (
    "7086b1d0209ce509c7b513f7e4f8ce5c9a232ca658f784617f23ef710fc67568"
)


def integrated_dense_rms_hlo_policy(
    member_device_ids: Sequence[int],
) -> HloContractPolicy:
    """Require the diagnostic's exact full-pod collective scope and group."""

    members = tuple(int(value) for value in member_device_ids)
    if members != tuple(range(32)):
        raise BenchmarkValidationError(
            "integrated dense RMS requires physical ids 0..31"
        )
    return HloContractPolicy(
        name="strategy-nd-integrated-dense-rms",
        total_devices=32,
        repeated_region_patterns=(
            r"integrated_dense_rms_strategy_nd_collective",
        ),
        maximum_repeated_collective_group_size=32,
        expected_repeated_replica_groups=(members,),
        expected_collectives=(CollectiveExpectation("all-reduce", 1),),
        partition_id_to_device_id=members,
        forbidden_row_width_pairs=(),
        allow_full_pod_repeated_collectives=True,
    )


def validate_integrated_dense_rms_stablehlo(
    stablehlo: str,
    *,
    split_layer1_rms: bool = False,
) -> Mapping[str, Any]:
    """Pin the complete exact eight-input StableHLO graph byte-for-byte."""

    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    expected = (
        INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256
        if split_layer1_rms
        else INTEGRATED_DENSE_RMS_STABLEHLO_SHA256
    )
    digest = sha256(stablehlo.encode()).hexdigest()
    if digest != expected:
        raise BenchmarkValidationError(
            "integrated dense RMS StableHLO SHA-256 drifted: "
            f"expected={expected} found={digest}"
        )
    return {
        "exact_graph_sha256": digest,
        "exact_input_schema": True,
        "exact_predense_rms": True,
        "exact_final_layout_dense": True,
        "exact_strategy_nd_collective": True,
        "exact_layer1_rms": True,
        "exact_live_result": True,
        "split_layer1_rms": split_layer1_rms,
        "passed": True,
        "performance_claim": False,
        "violations": [],
    }


def validate_integrated_dense_rms_hlo(
    optimized_hlo: str,
    member_device_ids: Sequence[int],
    *,
    split_layer1_rms: bool = False,
) -> Mapping[str, Any]:
    """Bind real contractions through one BF16 StrategyND and layer-1 ROOT."""

    from scripts.greenfield.probe_layer0_dense_convolution import (
        _validate_optimized_hlo,
    )

    members = tuple(int(value) for value in member_device_ids)
    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    report = lint_hlo(
        parse_hlo_module(optimized_hlo),
        integrated_dense_rms_hlo_policy(members),
    )
    report.raise_for_violations()
    reductions = tuple(
        item
        for item in report.module.collectives
        if item.opcode == "all-reduce"
    )
    if len(reductions) != 1:
        raise BenchmarkValidationError(
            "integrated dense RMS requires exactly one all-reduce"
        )
    reduction = reductions[0]
    algorithm = validate_strategy_nd_reduction(report, reduction)
    entry = tuple(
        item
        for item in report.module.instructions
        if item.computation.startswith("ENTRY ")
    )
    roots = tuple(
        item for item in entry if item.raw_line.lstrip().startswith("ROOT ")
    )
    if len(roots) != 1 or _shape_signature(roots[0]) != (
        ("u16", (1, 6144)),
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS ENTRY result drifted"
        )
    scheduled = tuple(
        item
        for item in entry
        if "integrated_dense_rms_layer1_norm" in (item.op_name or "")
        and (item.op_name or "").split("/")[-1] == "reduce_sum"
        and _exact_accepted_rms_schedule(item)
    )
    if len(scheduled) != 1:
        raise BenchmarkValidationError(
            "integrated dense RMS accepted layer-1 schedule drifted"
        )
    expected_schedule_shape = (
        (("f32", (32,)),)
        if split_layer1_rms
        else (("f32", (32,)), ("f32", (32, 6144)))
    )
    if _shape_signature(scheduled[0]) != expected_schedule_shape:
        raise BenchmarkValidationError(
            "integrated dense RMS layer-1 schedule form drifted"
        )
    contraction = _validate_optimized_hlo(
        optimized_hlo,
        compile_rows=32,
        final_dense_layout=True,
        dense_envelope=True,
        integrated_dense_rms=True,
        accepted_gate_singleton=True,
        accepted_gate_dequant_fusion=True,
    )
    required_contraction = (
        "accepted_down_schedule",
        "accepted_gate_up_schedule",
        "exact_accepted_kernel_geometry",
        "exact_accepted_weight_layout",
        "exact_activation_graph",
        "exact_carried_residual_binding",
        "exact_collective_input_binding",
        "exact_gate_dequant_fusion_boundary",
        "exact_gate_singleton_external_boundary",
        "exact_packed_weight_lineage",
        "exact_result_binding",
    )
    if not contraction.get("passed") or not all(
        contraction.get(name) is True for name in required_contraction
    ):
        raise BenchmarkValidationError(
            f"integrated dense contraction HLO drifted: {contraction}"
        )
    exact_rms = _validate_exact_dense_rms_value_flow(
        report,
        reduction,
        scheduled[0],
        roots[0],
        integrated_dense=True,
        require_split_output_fusion=split_layer1_rms,
    )
    if (
        exact_rms.get("residual_source_mode")
        != "integrated_attention_plus_combined_residual"
        or not all(
            exact_rms.get(name) is True
            for name in (
                "exact_collective_input",
                "exact_reduction_operand_graph",
                "exact_weighted_operand_graph",
                "exact_result_binding",
            )
        )
        or (
            split_layer1_rms
            and not all(
                exact_rms.get(name) is True
                for name in (
                    "exact_accepted_scheduled_reduction",
                    "split_output_fusion_exact",
                    "split_recompute_exact",
                )
            )
        )
    ):
        raise BenchmarkValidationError(
            f"integrated dense layer-1 HLO drifted: {exact_rms}"
        )
    if any(
        marker in optimized_hlo
        for marker in ("host_callback", "xla_python_cpu_callback", " outfeed(")
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS contains a host effect"
        )
    return {
        "accepted_scheduled_reduction": scheduled[0].name,
        "collective_algorithm": dict(algorithm),
        "collective_count": len(report.module.collectives),
        "contraction": contraction,
        **exact_rms,
        "live_rows": 1,
        "num_partitions": report.module.num_partitions,
        "num_replicas": report.module.num_replicas,
        "passed": True,
        "performance_claim": False,
        "split_layer1_rms": split_layer1_rms,
        "violations": [],
    }
