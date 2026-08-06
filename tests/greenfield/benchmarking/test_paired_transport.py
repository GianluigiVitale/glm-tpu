from __future__ import annotations

from glm_tpu.greenfield.benchmarking import (
    PairedTransportConfig,
    validate_paired_transport_hlo,
)
from glm_tpu.greenfield.benchmarking.transport_chain import TransportKind
from glm_tpu.greenfield.types import PlanName


def _pairs() -> tuple[tuple[int, int], ...]:
    return tuple(
        (lane * 8 + stage, lane * 8 + (stage + 1) % 8)
        for lane in range(4)
        for stage in range(8)
    )


def test_paired_pallas_hlo_requires_one_communicating_call_per_stage() -> None:
    config = PairedTransportConfig(
        plan=PlanName.PP8_LP4,
        kind=TransportKind.PALLAS_REMOTE_COPY,
        warmup_iterations=1,
        measured_iterations=1,
    )
    line = (
        '%call = (bf16[1,6144], s32[1,2052]) custom-call(%r, %m), '
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_stage_remote_copy_bf16_6144_s32_2052"}, '
        'backend_config={"has_communication":true}'
    )
    hlo = "HloModule paired, num_partitions=32\n" + "\n".join([line] * 8)
    result = validate_paired_transport_hlo(
        hlo,
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert result["passed"], result
    assert result["kernel_custom_call_count"] == 8


def test_paired_pallas_hlo_rejects_dead_rows_and_missing_call() -> None:
    config = PairedTransportConfig(
        plan=PlanName.PP8_LP4,
        kind=TransportKind.PALLAS_REMOTE_COPY,
        warmup_iterations=1,
        measured_iterations=1,
    )
    result = validate_paired_transport_hlo(
        "HloModule bad\n%x = bf16[32,6144] parameter(0)",
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert not result["passed"]
    assert result["forbidden_dead_rows"] == ["bf16[32,6144]"]
