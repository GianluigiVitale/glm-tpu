from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.benchmarking.collective_chain import (
    CollectiveChainConfig,
    CollectiveKind,
    collective_chain_hlo_policy,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


def test_config_rejects_invalid_collective_contracts() -> None:
    with pytest.raises(BenchmarkValidationError, match="divisible"):
        CollectiveChainConfig(
            kind=CollectiveKind.ALL_TO_ALL,
            group_size=8,
            rows=1,
            width=6145,
            dtype="bfloat16",
        )
    with pytest.raises(BenchmarkValidationError, match="dtype"):
        CollectiveChainConfig(
            kind=CollectiveKind.ALL_REDUCE,
            group_size=4,
            rows=1,
            width=6144,
            dtype="float16",
        )


def test_protected_contract_refuses_short_or_underwarmed_runs() -> None:
    config = CollectiveChainConfig(
        kind=CollectiveKind.ALL_REDUCE,
        group_size=4,
        rows=1,
        width=8,
        dtype="bfloat16",
        chain_length=3,
        warmup_iterations=1,
        measured_iterations=2,
    )
    with pytest.raises(BenchmarkValidationError, match="chain_length"):
        config.require_protected_contract()


def test_collective_permute_policy_repeats_every_physical_ring_edge() -> None:
    config = CollectiveChainConfig(
        kind=CollectiveKind.COLLECTIVE_PERMUTE,
        group_size=4,
        rows=1,
        width=8,
        dtype="bfloat16",
        chain_length=3,
        warmup_iterations=1,
        measured_iterations=1,
    )
    groups = ((0, 2, 3, 1), (4, 6, 7, 5))
    policy = collective_chain_hlo_policy(config, groups, total_devices=8)
    assert len(policy.expected_collective_permute_pairs) == 3 * 8
    assert policy.expected_collective_permute_pairs[:8] == (
        (0, 2),
        (2, 3),
        (3, 1),
        (1, 0),
        (4, 6),
        (6, 7),
        (7, 5),
        (5, 4),
    )


def test_current_jax_optimized_hlo_preserves_every_dependent_operation() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.collective_chain import (
    CollectiveChainConfig,
    CollectiveKind,
    benchmark_collective_chain,
    build_collective_chain,
)

# Non-identity executable assignment proves HLO partition ids are remapped to
# physical device ids before groups and permute edges are accepted.
groups = ((0, 2, 3, 1), (4, 6, 7, 5))
result = {}
for kind in CollectiveKind:
    config = CollectiveChainConfig(
        kind=kind,
        group_size=4,
        rows=1,
        width=8,
        dtype="bfloat16",
        chain_length=3,
        warmup_iterations=1,
        measured_iterations=2,
    )
    compiled = build_collective_chain(config, groups)
    measured = benchmark_collective_chain(compiled)
    result[kind.value] = {
        "checksums_match": (
            measured["first_addressable_checksum"]
            == measured["last_addressable_checksum"]
        ),
        "counts": measured["hlo"]["collective_counts"],
        "result_arities": [
            len(instruction.result_shapes)
            for instruction in compiled.hlo_report.module.collectives
        ],
    }
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_opcode = {
        "all_gather": "all-gather",
        "all_reduce": "all-reduce",
        "all_to_all": "all-to-all",
        "collective_permute": "collective-permute",
        "fused_tuple_all_reduce": "all-reduce",
        "reduce_scatter": "reduce-scatter",
    }
    assert result["control"]["counts"] == {}
    for kind, opcode in expected_opcode.items():
        assert result[kind]["counts"] == {opcode: 3}
        assert result[kind]["checksums_match"]
    assert result["fused_tuple_all_reduce"]["result_arities"] == [2, 2, 2]
    assert result["all_to_all"]["result_arities"] == [4, 4, 4]


def test_exact_75_operation_chain_for_every_required_group_size() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.collective_chain import (
    CollectiveChainConfig,
    CollectiveKind,
    benchmark_collective_chain,
    build_collective_chain,
)

result = {}
for size in (2, 4, 8, 32):
    groups = tuple(
        tuple(range(start, start + size)) for start in range(0, 32, size)
    )
    config = CollectiveChainConfig(
        kind=CollectiveKind.ALL_REDUCE,
        group_size=size,
        rows=1,
        width=16,
        dtype="bfloat16",
        chain_length=75,
        warmup_iterations=1,
        measured_iterations=1,
    )
    compiled = build_collective_chain(config, groups)
    measured = benchmark_collective_chain(compiled)
    result[str(size)] = {
        "checksum": measured["first_addressable_checksum"],
        "counts": measured["hlo"]["collective_counts"],
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
    assert set(result) == {"2", "4", "8", "32"}
    assert all(item["counts"] == {"all-reduce": 75} for item in result.values())
    assert all(len(item["checksum"]) == 64 for item in result.values())
