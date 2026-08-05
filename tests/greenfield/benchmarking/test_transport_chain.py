from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.benchmarking.transport_chain import (
    TransportChainConfig,
    TransportKind,
    transport_chain_hlo_policy,
    validate_transport_pairs,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.types import PlanName


def ring_pairs(stage_count: int, lane_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            lane * stage_count + stage,
            lane * stage_count + (stage + 1) % stage_count,
        )
        for lane in range(lane_count)
        for stage in range(stage_count)
    )


def test_transport_config_and_pair_cycles_fail_closed() -> None:
    with pytest.raises(BenchmarkValidationError, match="only PP8"):
        TransportChainConfig(
            plan=PlanName.WS32_2D,
            kind=TransportKind.DEVICE_RESIDENT,
            rows=1,
            width=8,
            dtype="bfloat16",
        )
    pairs = list(ring_pairs(8, 4))
    pairs[0] = (pairs[0][0], pairs[0][0])
    with pytest.raises(BenchmarkValidationError):
        validate_transport_pairs(pairs, total_devices=32, stage_count=8)


def test_transport_policy_requires_every_pair_for_every_hop() -> None:
    config = TransportChainConfig(
        plan=PlanName.PP8_LP4,
        kind=TransportKind.DEVICE_RESIDENT,
        rows=1,
        width=8,
        dtype="bfloat16",
        warmup_iterations=1,
        measured_iterations=1,
    )
    pairs = ring_pairs(8, 4)
    policy = transport_chain_hlo_policy(
        config,
        pairs,
        total_devices=32,
        partition_id_to_device_id=tuple(range(32)),
    )
    assert len(policy.expected_collective_permute_pairs) == 8 * 32
    counts = {item.opcode: item.count for item in policy.expected_collectives}
    assert counts["collective-permute"] == 8
    assert set(counts.values()) == {0, 8}


def test_current_jax_preserves_exact_pp8_transport_hlo() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.transport_chain import (
    TransportChainConfig,
    TransportKind,
    benchmark_transport_chain,
    build_transport_chain,
)
from glm_tpu.greenfield.types import PlanName

pairs = tuple(
    (lane * 8 + stage, lane * 8 + (stage + 1) % 8)
    for lane in range(4)
    for stage in range(8)
)
result = {}
for kind in TransportKind:
    config = TransportChainConfig(
        plan=PlanName.PP8_LP4,
        kind=kind,
        rows=1,
        width=8,
        dtype="float32",
        warmup_iterations=1,
        measured_iterations=2,
    )
    compiled = build_transport_chain(config, pairs)
    measured = benchmark_transport_chain(compiled)
    result[kind.value] = {
        "checksums_match": measured["first_addressable_checksum"] == measured["last_addressable_checksum"],
        "counts": measured["hlo"]["collective_counts"],
        "pairs": [
            pair
            for item in measured["hlo"]["collectives"]
            for pair in item["physical_source_target_pairs"]
        ],
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
    assert result["control"]["counts"] == {}
    assert result["device_resident"]["counts"] == {"collective-permute": 8}
    assert result["control"]["checksums_match"]
    assert result["device_resident"]["checksums_match"]
    expected_pairs = [list(pair) for pair in ring_pairs(8, 4)] * 8
    assert sorted(result["device_resident"]["pairs"]) == sorted(expected_pairs)
