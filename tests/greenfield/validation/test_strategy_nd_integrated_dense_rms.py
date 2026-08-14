from __future__ import annotations

import os
from hashlib import sha256
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
    INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256,
    integrated_dense_rms_hlo_policy,
    validate_integrated_dense_rms_hlo,
    validate_integrated_dense_rms_stablehlo,
)
from glm_tpu.greenfield.validation.strategy_nd_integrated_dense_rms import (
    CHECKPOINT_SUCCESS_SHA256,
    _recompute_comparison,
    _validate_capture,
    _validate_hlo_prevalidation,
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
REAL_SPLIT_TPU_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_INTEGRATED_SPLIT_TPU_HLO",
        "/home/gianl/glm-run/"
        "greenfield_recover_integrated_split_hlo_20260814T192500000000000Z/"
        "recovery_hlo/worker0/"
        "module_0012.jit_integrated.cl_914450892.after_codegen.txt",
    )
)
REAL_SPLIT_TPU_HLO_SHA256 = (
    "212aa36a9587ff390e6b0c18b654187d96e158eb896eaece1af51cda35df4e27"
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
split = build_integrated_dense_rms(
    tuple(range(32)), validate_hlo=False, split_layer1_rms=True
)
split_contract = validate_integrated_dense_rms_stablehlo(
    split.stablehlo, split_layer1_rms=True
)
assert split_contract["passed"] and split_contract["split_layer1_rms"] is True
assert split.split_layer1_rms is True
try:
    validate_integrated_dense_rms_stablehlo(split.stablehlo)
except ValueError:
    pass
else:
    raise AssertionError("split StableHLO passed the tuple-schedule contract")
print(split_contract["exact_graph_sha256"])
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
    assert INTEGRATED_DENSE_SPLIT_RMS_STABLEHLO_SHA256 in completed.stdout


def test_integrated_policy_requires_the_exact_scope() -> None:
    policy = integrated_dense_rms_hlo_policy(tuple(range(32)))
    assert policy.repeated_region_patterns == (
        r"integrated_dense_rms_strategy_nd_collective",
    )
    with pytest.raises(ValueError, match="physical ids"):
        integrated_dense_rms_hlo_policy(tuple(reversed(range(32))))


@pytest.mark.skipif(
    not REAL_SPLIT_TPU_HLO.is_file(),
    reason="protected split integrated TPU HLO absent",
)
def test_real_split_tpu_hlo_and_row_recompute_mutations() -> None:
    from jaxlib import xla_client

    hlo = REAL_SPLIT_TPU_HLO.read_text()
    assert sha256(hlo.encode()).hexdigest() == REAL_SPLIT_TPU_HLO_SHA256
    contract = validate_integrated_dense_rms_hlo(
        hlo,
        tuple(range(32)),
        split_layer1_rms=True,
    )
    assert contract["exact_accepted_scheduled_reduction"] is True
    assert contract["split_output_fusion_exact"] is True
    assert contract["split_recompute_exact"] is True
    replacements = (
        (
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
            "slice(%param_1.98), slice={[0:1], [0:6144]}",
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} "
            "slice(%param_1.98), slice={[1:2], [0:6144]}",
        ),
        (
            "%add.55 = f32[1,6144]{1,0:T(1,128)} "
            "add(%convert_element_type.145, %convert_element_type.144)",
            "%add.55 = f32[1,6144]{1,0:T(1,128)} "
            "add(%convert_element_type.145, %convert_element_type.145)",
        ),
        (
            "%param_1.98 = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} parameter(1)",
            "%param_1.98 = bf16[32,6144]{1,0} parameter(1)",
        ),
        (
            "%slice.36 = bf16[1,6144]{1,0:T(2,128)(2,1)} slice(",
            "%slice.36 = bf16[1,6144]{1,0} slice(",
        ),
        (
            "%convert_element_type.145 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert_element_type.145 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%add.55 = f32[1,6144]{1,0:T(1,128)} add(",
            "%add.55 = f32[1,6144]{1,0} add(",
        ),
        (
            "%mul.113 = f32[1,6144]{1,0:T(1,128)} broadcast(",
            "%mul.113 = f32[1,6144]{1,0} broadcast(",
        ),
        (
            "%mul.111 = f32[1,6144]{1,0:T(1,128)} multiply(",
            "%mul.111 = f32[1,6144]{1,0} multiply(",
        ),
        (
            "%convert_element_type.141 = bf16[1,6144]{1,0:T(2,128)(2,1)} convert(",
            "%convert_element_type.141 = bf16[1,6144]{1,0} convert(",
        ),
        (
            "%convert.2 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert.2 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%mul.112 = bf16[1,6144]{1,0:T(2,128)(2,1)} broadcast(",
            "%mul.112 = bf16[1,6144]{1,0} broadcast(",
        ),
        (
            "%convert.3 = f32[1,6144]{1,0:T(1,128)} convert(",
            "%convert.3 = f32[1,6144]{1,0} convert(",
        ),
        (
            "%mul.109 = f32[1,6144]{1,0:T(1,128)} multiply(",
            "%mul.109 = f32[1,6144]{1,0} multiply(",
        ),
        (
            "%convert.4 = bf16[1,6144]{1,0:T(2,128)(2,1)} convert(",
            "%convert.4 = bf16[1,6144]{1,0} convert(",
        ),
        (
            "ROOT %bitcast_convert_type.4 = u16[1,6144]{1,0:T(2,128)(2,1)} "
            "bitcast-convert(",
            "ROOT %bitcast_convert_type.4 = u16[1,6144]{1,0} bitcast-convert(",
        ),
        (
            "ROOT %multiply_bitcast-convert_fusion = "
            "u16[1,6144]{1,0:T(2,128)(2,1)} fusion(",
            "ROOT %multiply_bitcast-convert_fusion = u16[1,6144]{1,0} fusion(",
        ),
    )
    mutations = tuple(hlo.replace(old, new, 1) for old, new in replacements)
    assert all(mutation != hlo for mutation in mutations)
    for mutation in mutations:
        xla_client._xla.hlo_module_from_text(mutation)
        with pytest.raises(ValueError):
            validate_integrated_dense_rms_hlo(
                mutation,
                tuple(range(32)),
                split_layer1_rms=True,
            )


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
    split_exact = _recompute_comparison(
        expected.copy(), expected, split_layer1_rms=True
    )
    assert split_exact["classification"] == (
        "integrated_dense_split_rms_exact_accepted"
    )
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


def test_hlo_prevalidation_record_is_exact_and_non_promoting() -> None:
    record = {
        "optimized_hlo_sha256": "1" * 64,
        "performance_claim": False,
        "split_layer1_rms": True,
        "stablehlo_sha256": "2" * 64,
        "validated": False,
    }
    _validate_hlo_prevalidation(
        record,
        optimized_hlo_sha256="1" * 64,
        stablehlo_sha256="2" * 64,
        split_layer1_rms=True,
    )
    for key, value in (
        ("validated", True),
        ("performance_claim", True),
        ("split_layer1_rms", False),
        ("optimized_hlo_sha256", "3" * 64),
    ):
        mutation = dict(record)
        mutation[key] = value
        with pytest.raises(ValueError, match="prevalidation record drifted"):
            _validate_hlo_prevalidation(
                mutation,
                optimized_hlo_sha256="1" * 64,
                stablehlo_sha256="2" * 64,
                split_layer1_rms=True,
            )


def test_protected_integrated_wrapper_is_default_off_and_success_last() -> None:
    wrapper = (
        REPO / "scripts/greenfield/run_strategy_nd_dense_replay.sh"
    ).read_text()
    assert "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_RMS_REPLAY:-0" in wrapper
    assert (
        "GLM_GREENFIELD_STRATEGY_ND_INTEGRATED_SPLIT_RMS_REPLAY:-0"
        in wrapper
    )
    assert "--integrated-split-layer1-rms" in wrapper
    assert "--mode strategy_nd_integrated_dense_rms" in wrapper
    assert "validate_strategy_nd_integrated_dense_rms" in wrapper
    assert '"$RMS_REPLAY" "$INTEGRATED_REPLAY"' in wrapper
    assert wrapper.index("strict_census post") < wrapper.index(
        '"$REMOTE_PREFIX/SUCCESS" >/dev/null'
    )
    runner = (
        REPO / "scripts/greenfield/microbench_collectives.py"
    ).read_text()
    start = runner.index("def _run_strategy_nd_integrated_dense_rms(")
    end = runner.index("\ndef _run_strategy_nd_fingerprint(", start)
    integrated = runner[start:end]
    assert "validate_hlo=False" in integrated
    assert "hlo_prevalidation.json" in integrated
    assert integrated.index("hlo_prevalidation.json") < integrated.index(
        "execute_integrated_dense_rms("
    )
