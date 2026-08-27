from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking import (
    PairedTransportConfig,
    PairedTransportKind,
    pack_paired_transport_payload,
    unpack_paired_transport_payload,
    validate_paired_transport_hlo,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
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
        kind=PairedTransportKind.PALLAS_REMOTE_COPY,
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
        kind=PairedTransportKind.PALLAS_REMOTE_COPY,
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


def _production_config(
    kind: PairedTransportKind = PairedTransportKind.PACKED_DEVICE_RESIDENT,
) -> PairedTransportConfig:
    return PairedTransportConfig(
        plan=PlanName.PP8_LP4,
        kind=kind,
        residual_shape=(2, 1, 6144),
        metadata_shape=(1, 2053),
        warmup_iterations=1,
        measured_iterations=1,
    )


def _permute_hlo(shape: str, count: int) -> str:
    pairs = (
        "{"
        + ",".join(f"{{{source},{target}}}" for source, target in _pairs())
        + "}"
    )
    lines = ["HloModule paired, num_partitions=32", "ENTRY main {"]
    for index in range(count):
        lines.append(f"  %p{index} = {shape} parameter({index})")
        lines.append(
            f"  %moved{index} = {shape} collective-permute(%p{index}), "
            f"channel_id={index + 1}, source_target_pairs={pairs}"
        )
    lines.append("}")
    return "\n".join(lines)


def test_production_pack_round_trip_preserves_every_bit() -> None:
    config = _production_config()
    assert config.packed_width == 16_394
    residual = jnp.linspace(
        -2.0,
        2.0,
        num=np.prod(config.residual_shape),
        dtype=jnp.bfloat16,
    ).reshape(config.residual_shape)
    residual_bits = np.asarray(residual).view(np.uint16).copy()
    metadata = np.arange(np.prod(config.metadata_shape), dtype=np.int32)
    metadata[::4] *= -1
    metadata[0] = np.iinfo(np.int32).min
    metadata[-1] = np.iinfo(np.int32).max
    metadata_array = jnp.asarray(metadata.reshape(config.metadata_shape))

    packed = pack_paired_transport_payload(residual, metadata_array, config=config)
    restored_residual, restored_metadata = unpack_paired_transport_payload(
        packed,
        config=config,
    )

    np.testing.assert_array_equal(
        np.asarray(restored_residual).view(np.uint16),
        residual_bits,
    )
    np.testing.assert_array_equal(
        np.asarray(restored_metadata), np.asarray(metadata_array)
    )


def test_production_pack_refuses_shape_or_dtype_drift() -> None:
    config = _production_config()
    residual = jnp.zeros(config.residual_shape, dtype=jnp.bfloat16)
    metadata = jnp.zeros(config.metadata_shape, dtype=jnp.int32)
    with pytest.raises(BenchmarkValidationError, match="residual shape/dtype"):
        pack_paired_transport_payload(
            residual.astype(jnp.float32), metadata, config=config
        )
    with pytest.raises(BenchmarkValidationError, match="metadata shape/dtype"):
        pack_paired_transport_payload(residual, metadata[:, :-1], config=config)


def test_packed_hlo_requires_one_exact_production_payload_per_stage() -> None:
    config = _production_config()
    result = validate_paired_transport_hlo(
        _permute_hlo("u16[16394]", 8),
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert result["passed"], result
    assert result["collective_count"] == 8
    assert result["packed_permute_count"] == 8
    assert result["separate_residual_permute_count"] == 0
    assert result["separate_metadata_permute_count"] == 0


def test_packed_hlo_rejects_wrong_width_and_separate_metadata() -> None:
    config = _production_config()
    wrong_width = validate_paired_transport_hlo(
        _permute_hlo("u16[16393]", 8),
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert not wrong_width["passed"]
    separate = validate_paired_transport_hlo(
        _permute_hlo("u16[16394]", 8) + "\n" + _permute_hlo("s32[1,2053]", 1),
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert not separate["passed"]


def test_packed_hlo_rejects_production_batch_32_dead_rows() -> None:
    config = _production_config()
    hlo = (
        _permute_hlo("u16[16394]", 8)
        + "\n%dead_residual = bf16[2,32,6144] parameter(9)"
        + "\n%dead_metadata = s32[32,2053] parameter(10)"
    )
    result = validate_paired_transport_hlo(
        hlo,
        config=config,
        physical_pairs=_pairs(),
        total_devices=32,
    )
    assert not result["passed"]
    assert result["forbidden_dead_rows"] == [
        "bf16[2,32,6144]",
        "s32[32,2053]",
    ]


def test_current_jax_preserves_exact_packed_pp8_transport_hlo() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking import (
    PairedTransportConfig,
    PairedTransportKind,
    benchmark_paired_transport,
    build_paired_transport,
)
from glm_tpu.greenfield.types import PlanName

pairs = tuple(
    (lane * 8 + stage, lane * 8 + (stage + 1) % 8)
    for lane in range(4)
    for stage in range(8)
)
result = {}
for kind in (
    PairedTransportKind.DEVICE_RESIDENT,
    PairedTransportKind.PACKED_DEVICE_RESIDENT,
):
    config = PairedTransportConfig(
        plan=PlanName.PP8_LP4,
        kind=kind,
        residual_shape=(2, 1, 6144),
        metadata_shape=(1, 2053),
        warmup_iterations=1,
        measured_iterations=2,
    )
    compiled = build_paired_transport(
        config,
        pairs,
        enforce_hlo_contract=(
            kind is PairedTransportKind.PACKED_DEVICE_RESIDENT
        ),
    )
    measured = benchmark_paired_transport(compiled)
    result[kind.value] = {
        "checksum": measured["first_addressable_checksum"],
        "collective_count": measured["hlo"]["collective_count"],
        "packed_count": measured["hlo"]["packed_permute_count"],
        "separate_metadata_count": measured["hlo"]["separate_metadata_permute_count"],
        "separate_residual_count": measured["hlo"]["separate_residual_permute_count"],
    }
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    separate = result["device_resident"]
    packed = result["packed_device_resident"]
    assert separate["checksum"] == packed["checksum"]
    assert separate["collective_count"] == 16
    # CPU XLA promotes the separate BF16 residual permutes to f32. The static
    # tests above seal the TPU BF16 shape; this forced-CPU test seals semantics
    # and the physical launch-count reduction.
    assert separate["separate_metadata_count"] == 8
    assert packed["collective_count"] == 8
    assert packed["packed_count"] == 8
    assert packed["separate_residual_count"] == 0
    assert packed["separate_metadata_count"] == 0
