from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.association_fingerprint import array_sha256
from glm_tpu.greenfield.benchmarking.integrated_dense_rms import (
    POST_ATTENTION_NORM_RAW_SHA256,
    load_integrated_dense_rms_inputs,
    model_axis_weights_to_physical,
    validate_integrated_checkpoint_success,
)
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import (
    INTEGRATED_DENSE_RMS_STABLEHLO_SHA256,
    integrated_dense_rms_hlo_policy,
    validate_integrated_dense_rms_stablehlo,
)
from glm_tpu.greenfield.validation.strategy_nd_integrated_dense_rms import (
    CHECKPOINT_SUCCESS_SHA256,
    _recompute_comparison,
    _validate_capture,
)


REPO = Path(__file__).resolve().parents[3]
REAL_SOURCE = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
REAL_STAGE0_SLOT0 = Path(
    "/home/gianl/gcs-models/checkpoints/greenfield/glm52/runtime_feature/"
    "PP8_LP4/greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z/"
    "base_decoder_runtime_feature/stage_00/device_slot_00.safetensors"
)


def test_model_axis_weights_are_bijectively_mapped_to_physical_ids() -> None:
    value = np.arange(32 * 2, dtype=np.int32).reshape(32, 2)
    mapping = tuple(reversed(range(32)))
    physical = model_axis_weights_to_physical(value, mapping)
    for source_rank, physical_id in enumerate(mapping):
        assert np.array_equal(physical[physical_id], value[source_rank])
    with pytest.raises(ValueError, match="bijectively cover"):
        model_axis_weights_to_physical(value, tuple([0] * 32))


@pytest.mark.skipif(
    not REAL_SOURCE.is_file() or not REAL_STAGE0_SLOT0.is_file(),
    reason="protected DB548/checkpoint source absent",
)
def test_real_integrated_source_is_exactly_pinned() -> None:
    from safetensors import safe_open

    with safe_open(REAL_STAGE0_SLOT0, framework="np") as handle:
        post_norm = np.ascontiguousarray(
            handle.get_tensor("attention.slot_00.post_norm")
        )
    assert __import__("hashlib").sha256(post_norm.tobytes()).hexdigest() == (
        POST_ATTENTION_NORM_RAW_SHA256
    )
    inputs = load_integrated_dense_rms_inputs(
        REAL_SOURCE,
        post_norm,
    )
    assert inputs.accepted_layer1_bits.shape == (6144,)


def test_exact_forced_32_cpu_stablehlo_and_mutation_refusal() -> None:
    code = r'''
from glm_tpu.greenfield.benchmarking.integrated_dense_rms import build_integrated_dense_rms
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import validate_integrated_dense_rms_stablehlo

compiled = build_integrated_dense_rms(tuple(range(32)), validate_hlo=False)
contract = validate_integrated_dense_rms_stablehlo(compiled.stablehlo)
assert contract["passed"]
from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import integrated_dense_rms_hlo_policy
from glm_tpu.greenfield.sharding.hlo_contract import lint_hlo, parse_hlo_module
report = lint_hlo(parse_hlo_module(compiled.optimized_hlo), integrated_dense_rms_hlo_policy(tuple(range(32))))
assert report.valid, [item.to_dict() for item in report.violations]
try:
    validate_integrated_dense_rms_stablehlo(compiled.stablehlo.replace("9.99999974E-6", "2.99999974E-6", 1))
except ValueError:
    pass
else:
    raise AssertionError("StableHLO mutation was accepted")
print(contract["exact_graph_sha256"])
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO),
            "XLA_FLAGS": "--xla_force_host_platform_device_count=32",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert INTEGRATED_DENSE_RMS_STABLEHLO_SHA256 in completed.stdout


def test_integrated_policy_requires_the_exact_scope() -> None:
    policy = integrated_dense_rms_hlo_policy(tuple(range(32)))
    assert policy.repeated_region_patterns == (
        r"integrated_dense_rms_strategy_nd_collective",
    )
    with pytest.raises(ValueError, match="physical ids"):
        integrated_dense_rms_hlo_policy(tuple(reversed(range(32))))


def test_checkpoint_success_pin_and_missing_mutated_refusals(
    tmp_path: Path,
) -> None:
    marker = REAL_STAGE0_SLOT0.parents[2] / "SUCCESS"
    if not marker.is_file():
        pytest.skip("protected checkpoint marker absent")
    assert __import__("hashlib").sha256(marker.read_bytes()).hexdigest() == (
        CHECKPOINT_SUCCESS_SHA256
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="checkpoint SUCCESS drifted"):
        validate_integrated_checkpoint_success(checkpoint)
    shutil.copy2(marker, checkpoint / "SUCCESS")
    assert validate_integrated_checkpoint_success(checkpoint) == (
        CHECKPOINT_SUCCESS_SHA256
    )
    (checkpoint / "SUCCESS").write_bytes(marker.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="checkpoint SUCCESS drifted"):
        validate_integrated_checkpoint_success(checkpoint)


def test_integrated_comparison_and_capture_are_recomputed_from_arrays() -> None:
    expected = np.arange(6144, dtype=np.uint16)
    exact = _recompute_comparison(expected.copy(), expected)
    assert exact["classification"] == "integrated_dense_rms_exact_accepted"
    assert exact["mismatch_count"] == 0
    digest = array_sha256(expected)
    capture = {
        "invocation_count": 2,
        "local_replica_output_sha256": [digest] * 4,
        "output_bits_sha256": digest,
        "repeated_local_replica_output_sha256": [digest] * 4,
        "repeated_output_bits_sha256": digest,
    }
    _validate_capture(capture, expected)
    capture["invocation_count"] = True
    with pytest.raises(ValueError, match="deterministic capture"):
        _validate_capture(capture, expected)


def test_protected_integrated_wrapper_is_default_off_and_success_last() -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    assert "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_RMS_REPLAY:-0" in wrapper
    assert "--mode strategy_nd_integrated_dense_rms" in wrapper
    assert "validate_strategy_nd_integrated_dense_rms" in wrapper
    assert '"$RMS_REPLAY" "$INTEGRATED_REPLAY"' in wrapper
    assert wrapper.index("strict_census post") < wrapper.index(
        '"$REMOTE_PREFIX/SUCCESS" >/dev/null'
    )
