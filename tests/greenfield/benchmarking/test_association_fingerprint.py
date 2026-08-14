from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    STRATEGY_ND_ALGORITHM,
    StrategyNdFingerprintConfig,
    _BALANCED_FOUR_WAY_TREES,
    _candidate_output_bits,
    accepted_tp32_model_axis_device_ids,
    analyze_m32_strategy_nd_fingerprint,
    analyze_strategy_nd_fingerprint,
    array_sha256,
    bfloat16_bits_to_float32,
    float32_to_bfloat16_bits,
    generate_strategy_nd_input_bits,
    model_axis_to_physical_input_bits,
    replay_db533_strategy_nd_row0_bits,
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
  input = bf16[32,6144]{{1,0:T(8,128)(2,1)S(3)}} parameter(0)
  ROOT reduced = bf16[32,6144]{{1,0:T(8,128)(2,1)S(3)}} all-reduce(input), replica_groups={{{{{group}}}}}, use_global_device_ids=true, to_apply=add, metadata={{op_name="jit(fingerprint)/shard_map/strategy_nd_association_fingerprint/psum"}}, backend_config={backend}
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


def test_dense_replay_reorders_model_axis_and_matches_db533_row_zero() -> None:
    model_axis_device_ids = (
        0, 8, 16, 24, 2, 10, 18, 26,
        4, 12, 20, 28, 6, 14, 22, 30,
        1, 9, 17, 25, 3, 11, 19, 27,
        5, 13, 21, 29, 7, 15, 23, 31,
    )
    rng = np.random.default_rng(550)
    values = rng.standard_normal((32, 6144), dtype=np.float32)
    model_bits = float32_to_bfloat16_bits(values)
    physical = model_axis_to_physical_input_bits(
        model_bits, model_axis_device_ids
    )
    for model_position, device_id in enumerate(model_axis_device_ids):
        np.testing.assert_array_equal(physical[device_id], model_bits[model_position])

    values_by_physical = bfloat16_bits_to_float32(physical).reshape(
        4, 4, 2, 6144
    ).transpose(1, 2, 0, 3)

    def add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return bfloat16_bits_to_float32(
            float32_to_bfloat16_bits(left + right)
        )

    def four(source: np.ndarray, cross: bool) -> np.ndarray:
        pairs = ((0, 3), (1, 2)) if cross else ((0, 1), (2, 3))
        return add(
            add(source[pairs[0][0]], source[pairs[0][1]]),
            add(source[pairs[1][0]], source[pairs[1][1]]),
        )

    y_reduced = np.concatenate(
        (
            four(values_by_physical[..., :2048], False),
            four(values_by_physical[..., 2048:4096], True),
            four(values_by_physical[..., 4096:], False),
        ),
        axis=-1,
    )
    x_reduced = add(y_reduced[0], y_reduced[1])
    expected = float32_to_bfloat16_bits(
        np.concatenate(
            tuple(
                four(
                    x_reduced[..., start : start + 256],
                    bool((start // 256) % 2),
                )
                for start in range(0, 6144, 256)
            ),
            axis=-1,
        )
    )
    actual = replay_db533_strategy_nd_row0_bits(
        model_bits, model_axis_device_ids
    )
    np.testing.assert_array_equal(actual, expected)

    with pytest.raises(BenchmarkValidationError, match="bijectively"):
        model_axis_to_physical_input_bits(model_bits, (0,) * 32)
    with pytest.raises(BenchmarkValidationError, match="width 6144"):
        replay_db533_strategy_nd_row0_bits(
            model_bits[:, :96], model_axis_device_ids
        )


REAL_DB550_PARTIALS = Path(
    os.environ.get(
        "GLM_TEST_DB550_DENSE_PARTIALS",
        "/home/gianl/glm-run/"
        "greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z/"
        "dense_partials_capture/dense_partials.npz",
    )
)
REAL_DB533_FINGERPRINT_HLO = Path(
    os.environ.get(
        "GLM_TEST_DB533_FINGERPRINT_HLO",
        "/home/gianl/glm-run/"
        "greenfield_collective_association_20260811T213152133863450Z/hlo/"
        "strategy_nd_association_bfloat16_32x6144.optimized_hlo.txt",
    )
)


@pytest.mark.skipif(not REAL_DB550_PARTIALS.is_file(), reason="DB550 artifact absent")
def test_real_db550_dense_partials_replay_pins_current_software_result() -> None:
    with np.load(REAL_DB550_PARTIALS, allow_pickle=False) as payload:
        model_bits = np.ascontiguousarray(
            payload["accepted_dense_partials_bfloat16_bits"]
        ).reshape(32, 6144)
    model_axis_device_ids = (
        0, 8, 16, 24, 2, 10, 18, 26,
        4, 12, 20, 28, 6, 14, 22, 30,
        1, 9, 17, 25, 3, 11, 19, 27,
        5, 13, 21, 29, 7, 15, 23, 31,
    )
    replayed = replay_db533_strategy_nd_row0_bits(
        model_bits, model_axis_device_ids
    )
    assert replayed.shape == (6144,)
    assert replayed.dtype == np.uint16
    assert replayed[2795] == 47808
    assert array_sha256(replayed) == (
        "ca1abfa2f57271f4dd4ae5db7c1bf0b8ced7b02cc8106dcb4a8cbcd84124032a"
    )


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

    with pytest.raises(BenchmarkValidationError, match="sorted global"):
        validate_strategy_nd_fingerprint_hlo(
            _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)), tuple(reversed(range(32)))
        )

    wrong_layout = _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)).replace(
        "{1,0:T(8,128)(2,1)S(3)}", "{1,0}"
    )
    with pytest.raises(BenchmarkValidationError, match="TPU result layout"):
        validate_strategy_nd_fingerprint_hlo(wrong_layout, tuple(range(32)))

    wrong_operand_layout = _strategy_nd_hlo(
        dict(STRATEGY_ND_ALGORITHM)
    ).replace(
        "input = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} parameter(0)",
        "input = bf16[32,6144]{0,1} parameter(0)",
    )
    with pytest.raises(BenchmarkValidationError, match="operand.*layout"):
        validate_strategy_nd_fingerprint_hlo(
            wrong_operand_layout, tuple(range(32))
        )

    wrong_reducer = _strategy_nd_hlo(
        dict(STRATEGY_ND_ALGORITHM)
    ).replace("ROOT sum = bf16[] add(x, y)", "ROOT sum = bf16[] maximum(x, y)")
    with pytest.raises(BenchmarkValidationError, match="exact scalar BF16 add"):
        validate_strategy_nd_fingerprint_hlo(wrong_reducer, tuple(range(32)))

    async_collective = _strategy_nd_hlo(
        dict(STRATEGY_ND_ALGORITHM)
    ).replace(" all-reduce(input)", " all-reduce-start(input)")
    with pytest.raises(BenchmarkValidationError, match="synchronous"):
        validate_strategy_nd_fingerprint_hlo(async_collective, tuple(range(32)))

    rogue_operand = _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)).replace(
        "  ROOT reduced = bf16[32,6144]",
        "  rogue = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} negate(input)\n"
        "  ROOT reduced = bf16[32,6144]",
    ).replace(" all-reduce(input)", " all-reduce(rogue)")
    with pytest.raises(BenchmarkValidationError, match="ENTRY"):
        validate_strategy_nd_fingerprint_hlo(rogue_operand, tuple(range(32)))

    dead_reduction = _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)).replace(
        "  ROOT reduced =", "  reduced ="
    )
    entry_end = dead_reduction.rfind("\n}\n")
    dead_reduction = (
        dead_reduction[:entry_end]
        + "\n  ROOT rogue = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} negate(input)"
        + dead_reduction[entry_end:]
    )
    with pytest.raises(BenchmarkValidationError, match="ENTRY"):
        validate_strategy_nd_fingerprint_hlo(dead_reduction, tuple(range(32)))

    metadata_layout_decoy = _strategy_nd_hlo(dict(STRATEGY_ND_ALGORITHM)).replace(
        "input = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} parameter(0)",
        "input = bf16[32,6144]{0,1} parameter(0), "
        'metadata={op_name="{1,0:T(8,128)(2,1)S(3)}"}',
    )
    with pytest.raises(BenchmarkValidationError, match="ENTRY"):
        validate_strategy_nd_fingerprint_hlo(
            metadata_layout_decoy, tuple(range(32))
        )

    duplicate_reducer_parameter = _strategy_nd_hlo(
        dict(STRATEGY_ND_ALGORITHM)
    ).replace("y = bf16[] parameter(1)", "y = bf16[] parameter(0)")
    with pytest.raises(BenchmarkValidationError, match="exact scalar BF16 add"):
        validate_strategy_nd_fingerprint_hlo(
            duplicate_reducer_parameter, tuple(range(32))
        )


@pytest.mark.skipif(
    not REAL_DB533_FINGERPRINT_HLO.is_file(), reason="DB533 TPU HLO absent"
)
def test_hlo_contract_replays_sha_pinned_db533_tpu_lowering() -> None:
    hlo = REAL_DB533_FINGERPRINT_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == (
        "1ed443ae17894373a37eba47768e2ff5a1d4b30f632187abd82927e8fe06e537"
    )
    report, algorithm = validate_strategy_nd_fingerprint_hlo(
        hlo, tuple(range(32))
    )
    assert report.valid
    assert report.to_dict()["collective_counts"] == {"all-reduce": 1}
    assert algorithm == STRATEGY_ND_ALGORITHM


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


def test_m32_analyzer_replays_every_physical_row_independently() -> None:
    config = StrategyNdFingerprintConfig(trials=4, width=96, seed=24)
    inputs = generate_strategy_nd_input_bits(config)
    decoded = bfloat16_bits_to_float32(inputs)
    topology_values = np.empty((4, 4, 2, 4, 96), dtype=np.float32)
    coordinates = _coordinates()
    for device_id in range(32):
        x, y, z = coordinates[device_id]
        topology_values[:, y, x, z, :] = decoded[:, device_id, :]
    first = _candidate_output_bits(
        topology_values,
        (0, 1, 2),
        {0: _BALANCED_FOUR_WAY_TREES[0], 2: _BALANCED_FOUR_WAY_TREES[1]},
    )
    second = _candidate_output_bits(
        topology_values,
        (2, 0, 1),
        {0: _BALANCED_FOUR_WAY_TREES[2], 2: _BALANCED_FOUR_WAY_TREES[0]},
    )
    outputs = np.empty((4, 32, 96), dtype=np.uint16)
    outputs[:, :16, :] = first[:, None, :]
    outputs[:, 16:, :] = second[:, None, :]
    analysis = analyze_m32_strategy_nd_fingerprint(
        inputs,
        outputs,
        tuple(range(32)),
        coordinates,
        block_width=32,
    )
    assert analysis["compile_bucket_rows"] == 32
    assert analysis["trials_per_physical_row"] == 4
    assert analysis["unique_row_output_count"] == 2
    assert analysis["minimum_row_union_exact_column_count"] == 96
    assert analysis["maximum_row_union_exact_column_count"] == 96
    assert analysis["rows"][0]["physical_row"] == 0
    assert analysis["rows"][16]["physical_row"] == 16
    assert analysis["rows"][0]["top_candidates"][0]["exact_columns"] == 96
    assert analysis["rows"][16]["top_candidates"][0]["exact_columns"] == 96


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
from dataclasses import dataclass, replace
import json
import numpy as np
from glm_tpu.greenfield.benchmarking.association_fingerprint import (
    StrategyNdFingerprintConfig,
    accepted_tp32_model_axis_device_ids,
    build_strategy_nd_fingerprint,
    execute_strategy_nd_fingerprint,
    generate_strategy_nd_input_bits,
)
config = StrategyNdFingerprintConfig(trials=2, width=96, seed=29)
inputs = generate_strategy_nd_input_bits(config)
compiled = build_strategy_nd_fingerprint(
    config, tuple(range(32)), enforce_hlo_contract=False
)
accepted_model_axis = accepted_tp32_model_axis_device_ids()
@dataclass(frozen=True)
class FakeV4Device:
    id: int
    coords: tuple[int, int, int]
    core_on_chip: int = 0
    device_kind: str = "TPU v4"
    platform: str = "tpu"
fake_v4_devices = [
    FakeV4Device(
        device_id,
        (device_id % 2, (device_id // 2) % 4, device_id // 8),
    )
    for device_id in range(32)
]
accepted_v4_model_axis = accepted_tp32_model_axis_device_ids(fake_v4_devices)
class CountingExecutable:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
    def __call__(self, value):
        self.calls += 1
        return self.inner(value)
counting = CountingExecutable(compiled.compiled)
outputs, capture = execute_strategy_nd_fingerprint(
    replace(compiled, compiled=counting), inputs
)
print(json.dumps({
    "calls": counting.calls,
    "accepted_model_axis": list(accepted_model_axis),
    "accepted_v4_model_axis": list(accepted_v4_model_axis),
    "counts": compiled.hlo_report.to_dict()["collective_counts"],
    "deterministic": capture["repeated_output_bits_sha256"] == capture["output_bits_sha256"],
    "input": capture["input_bits_sha256"],
    "replica_trials": len(capture["local_replica_output_sha256_by_trial"]),
    "rows_equal": bool(np.all(outputs == outputs[:, :1, :])),
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
    assert result["accepted_model_axis"] == list(range(32))
    assert result["accepted_v4_model_axis"] == [
        0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30,
        1, 9, 17, 25, 3, 11, 19, 27, 5, 13, 21, 29, 7, 15, 23, 31,
    ]
    assert result["calls"] == 4  # two M32 trials plus the identical repeat bank
    assert result["counts"] == {"all-reduce": 1}
    assert result["deterministic"]
    assert result["replica_trials"] == 2
    assert result["rows_equal"]
    assert result["shape"] == [2, 32, 96]
    assert len(result["input"]) == len(result["output"]) == 64


def test_protected_fingerprint_contract_is_exact() -> None:
    StrategyNdFingerprintConfig().require_protected_contract()
    with pytest.raises(BenchmarkValidationError, match="32 trials"):
        StrategyNdFingerprintConfig(trials=8).require_protected_contract()
