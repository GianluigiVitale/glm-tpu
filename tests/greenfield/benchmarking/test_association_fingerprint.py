from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    STRATEGY_ND_ALGORITHM,
    StrategyNdFingerprintConfig,
    _BALANCED_FOUR_WAY_TREES,
    _candidate_output_bits,
    analyze_strategy_nd_fingerprint,
    array_sha256,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    generate_strategy_nd_input_bits,
    validate_strategy_nd_fingerprint_hlo,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError


def _coordinates() -> dict[int, tuple[int, int, int]]:
    return {
        device_id: (device_id % 2, (device_id // 2) % 4, device_id // 8)
        for device_id in range(32)
    }


def _strategy_nd_hlo(algorithm: dict[str, object]) -> str:
    group = ",".join(map(str, range(32)))
    backend = json.dumps(
        {"collective_algorithm_config": algorithm}, separators=(",", ":")
    )
    return f'''HloModule fingerprint, num_partitions=32, replica_count=1

add {{
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT sum = bf16[] add(x, y)
}}

ENTRY main {{
  input = bf16[1,6144] parameter(0)
  ROOT reduced = bf16[1,6144] all-reduce(input), replica_groups={{{{{group}}}}}, use_global_device_ids=true, to_apply=add, metadata={{op_name="jit(fingerprint)/shard_map/strategy_nd_association_fingerprint/psum"}}, backend_config={backend}
}}
'''


def test_input_bank_is_deterministic_finite_and_raw_hash_is_shape_bound() -> None:
    config = StrategyNdFingerprintConfig(trials=3, width=96, seed=17)
    first = generate_strategy_nd_input_bits(config)
    second = generate_strategy_nd_input_bits(config)
    assert first.shape == (3, 32, 96)
    assert first.dtype == np.uint16
    assert np.array_equal(first, second)
    assert np.isfinite(bfloat16_bits_to_float32(first)).all()
    assert array_sha256(first) == array_sha256(second)
    assert array_sha256(first) != array_sha256(first.reshape(3, 16, 192))


def test_bfloat16_round_trip_and_tie_to_even() -> None:
    values = np.asarray([1.0, -2.0, 1.00390625, 1.01171875], dtype=np.float32)
    bits = float32_to_bfloat16_bits(values)
    recovered = bfloat16_bits_to_float32(bits)
    assert recovered.tolist() == [1.0, -2.0, 1.0, 1.015625]


def test_hlo_contract_pins_exact_strategy_nd_backend() -> None:
    report, algorithm = validate_strategy_nd_fingerprint_hlo(
        _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)), tuple(range(32))
    )
    assert report.valid
    assert report.to_dict()["collective_counts"] == {"all-reduce": 1}
    assert algorithm == STRATEGY_ND_ALGORITHM

    drifted = dict(STRATEGY_ND_ALGORITHM)
    drifted["strategy"] = "StrategyRing"
    with pytest.raises(BenchmarkValidationError, match="byte-pinned"):
        validate_strategy_nd_fingerprint_hlo(
            _strategy_nd_hlo(drifted), tuple(range(32))
        )


def test_offline_analyzer_recovers_known_axis_pincer_family() -> None:
    config = StrategyNdFingerprintConfig(trials=4, width=96, seed=23)
    inputs = generate_strategy_nd_input_bits(config)
    decoded = bfloat16_bits_to_float32(inputs)
    topology_values = np.empty((4, 4, 2, 4, 96), dtype=np.float32)
    coordinates = _coordinates()
    for device_id in range(32):
        x, y, z = coordinates[device_id]
        topology_values[:, y, x, z, :] = decoded[:, device_id, :]
    outputs = _candidate_output_bits(
        topology_values,
        (1, 2, 0),
        {0: _BALANCED_FOUR_WAY_TREES[1], 2: _BALANCED_FOUR_WAY_TREES[2]},
    )
    analysis = analyze_strategy_nd_fingerprint(
        inputs,
        outputs,
        tuple(range(32)),
        coordinates,
        block_width=32,
    )
    assert analysis["candidate_count"] == 54
    assert analysis["union_exact_column_count"] == 96
    assert analysis["uncovered_columns"] == []
    assert analysis["top_candidates"][0]["exact_columns"] == 96


def test_offline_analyzer_rejects_invalid_band_width_and_coordinate_alias() -> None:
    narrow_config = StrategyNdFingerprintConfig(trials=2, width=95, seed=31)
    narrow_inputs = generate_strategy_nd_input_bits(narrow_config)
    with pytest.raises(BenchmarkValidationError, match="divisible by three"):
        analyze_strategy_nd_fingerprint(
            narrow_inputs,
            np.zeros((2, 95), dtype=np.uint16),
            tuple(range(32)),
            _coordinates(),
            block_width=5,
        )

    config = StrategyNdFingerprintConfig(trials=2, width=96, seed=37)
    inputs = generate_strategy_nd_input_bits(config)
    aliased_coordinates = _coordinates()
    aliased_coordinates[31] = aliased_coordinates[30]
    with pytest.raises(BenchmarkValidationError, match="bijectively map"):
        analyze_strategy_nd_fingerprint(
            inputs,
            np.zeros((2, 96), dtype=np.uint16),
            tuple(range(32)),
            aliased_coordinates,
            block_width=32,
        )


def test_forced_cpu_executable_preserves_one_collective_and_raw_bits() -> None:
    program = r'''
import json
from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    StrategyNdFingerprintConfig,
    build_strategy_nd_fingerprint,
    execute_strategy_nd_fingerprint,
    generate_strategy_nd_input_bits,
)
config = StrategyNdFingerprintConfig(trials=2, width=96, seed=29)
inputs = generate_strategy_nd_input_bits(config)
compiled = build_strategy_nd_fingerprint(
    config, tuple(range(32)), enforce_hlo_contract=False
)
outputs, capture = execute_strategy_nd_fingerprint(compiled, inputs)
print(json.dumps({
    "counts": compiled.hlo_report.to_dict()["collective_counts"],
    "input": capture["input_bits_sha256"],
    "output": capture["output_bits_sha256"],
    "shape": list(outputs.shape),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["counts"] == {"all-reduce": 1}
    assert result["shape"] == [2, 96]
    assert len(result["input"]) == len(result["output"]) == 64


def test_protected_fingerprint_contract_is_exact() -> None:
    StrategyNdFingerprintConfig().require_protected_contract()
    with pytest.raises(BenchmarkValidationError, match="32 trials"):
        StrategyNdFingerprintConfig(trials=8).require_protected_contract()
