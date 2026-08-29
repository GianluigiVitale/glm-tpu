from __future__ import annotations

import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
)
from glm_tpu.greenfield.benchmarking.pp16_feature2_numerical import (
    _bitwise_mismatches,
    compare_feature2_numerical_capture,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)

ROOT = Path("/home/gianl/gcs-models")
TOKEN_ORACLE = ROOT / (
    "oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)
DSA_ORACLE = ROOT / (
    "oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
LAYER1 = ROOT / (
    "oracles/greenfield/glm52/dsa_internals/8k/layer1/"
    "greenfield_layer1_dsa_internal_comparison_20260808T115135394251231Z/"
    "internals.npz"
)
DB529 = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/"
    "inputs/internal"
)
DB550 = ROOT / (
    "results/greenfield_layer0_dense_partial_capture_"
    "20260813T200736889447458Z/dense_partial_capture.npz"
)
PROTECTED_CAPTURE = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_prefill_numerical_20260829T001056567299421Z/"
    "result.npz"
)
SOURCES_AVAILABLE = all(
    path.exists() for path in (TOKEN_ORACLE, DSA_ORACLE, LAYER1, DB529, DB550)
)


def test_compile_and_numerical_entrypoints_are_fail_closed() -> None:
    repo = Path(__file__).resolve().parents[3]
    shared = (repo / "scripts/greenfield/acquire_pp16_feature2_prefill.py").read_text()
    numerical = (
        repo / "scripts/greenfield/execute_pp16_feature2_prefill.py"
    ).read_text()
    assert "return run_feature2(parse_args(), execute_main=False)" in shared
    assert "return run_feature2(parse_args(), execute_main=True)" in numerical
    assert shared.count("main_compiled(*main_arguments)") == 1
    assert shared.index("feature2 executable canonical HLO drifted") < shared.index(
        "main_compiled(*main_arguments)"
    )
    assert shared.index("_validate_preexecution_memory(memory_after_compile)") < (
        shared.index("main_compiled(*main_arguments)")
    )
    assert "--expected-main-stablehlo-sha256" in numerical
    assert "--expected-main-canonical-hlo-sha256" in numerical
    assert "--expected-main-canonical-hlo-byte-count" in numerical
    assert "--expected-main-stack-frame-reference-count" in numerical
    assert "--expected-jax-version" in numerical
    assert "--expected-jaxlib-version" in numerical
    assert "--expected-libtpu-version" in numerical
    assert "warmup" not in numerical.lower()

    wrapper = (
        repo / "scripts/greenfield/run_pp16_feature2_prefill_numerical.sh"
    ).read_text()
    assert "GLM_GREENFIELD_PP16_FEATURE2_NUMERICAL:-0" in wrapper
    assert "GLM_GREENFIELD_PP16_FEATURE2_MODE:-off} == execute_once" in wrapper
    assert "warmups=0 invocations=1" in wrapper
    assert '--result-npz "$RUN_DIR/result.npz"' in wrapper
    assert "ACQUIRED_MAIN_STABLE_SHA=127bf089" in wrapper
    assert "ACQUIRED_MAIN_CANONICAL_SHA=fb5aaf02" in wrapper
    assert "ACQUIRED_MAIN_CANONICAL_BYTES=7870521" in wrapper
    assert "ACQUIRED_MAIN_STACK_FRAME_REFERENCES=16170" in wrapper
    assert "ACQUIRED_JAX_VERSION=0.10.1" in wrapper
    assert "ACQUIRED_JAXLIB_VERSION=0.10.1" in wrapper
    assert "ACQUIRED_LIBTPU_VERSION=0.0.41" in wrapper
    assert "bench/results.db" not in wrapper
    assert "gsutil" not in wrapper
    assert "upload_ledger_no_clobber" in wrapper
    assert "One or more URLs matched no objects." in wrapper
    assert 'verify_final_object_set "$terminal_status"' in wrapper
    assert '--if-generation-match="$generation"' in wrapper
    create = wrapper.index("gcloud storage cp --if-generation-match=0")
    assert wrapper.rindex("terminal_publication_started=1", 0, create) < create
    assert wrapper.rindex("terminal_remote_name=$terminal_status", 0, create) < create
    assert '2>"$RUN_DIR/terminal_create.receipt.stderr"' in wrapper[create:]
    assert '>"$RUN_DIR/terminal_create.stdout"' in wrapper[create:]
    assert 'sync -f "$RUN_DIR/terminal_create.receipt.stderr"' in wrapper[create:]
    assert "[[ -n $terminal_remote_name && -s $receipt ]] || return 1" in wrapper
    assert 'gcloud storage cp --no-clobber "$terminal_path"' not in wrapper


def test_exact_comparator_rejects_signed_zero_and_one_bit_float_drift() -> None:
    positive_zero = np.asarray([0.0], dtype=np.float32)
    negative_zero = np.asarray([-0.0], dtype=np.float32)
    assert positive_zero[0] == negative_zero[0]
    assert _bitwise_mismatches(positive_zero, negative_zero) == 1

    baseline = np.asarray([1.0], dtype=np.float32)
    one_bit = baseline.view(np.uint32).copy()
    one_bit[0] ^= np.uint32(1)
    assert _bitwise_mismatches(baseline, one_bit.view(np.float32)) == 1


def _capture(path: Path) -> None:
    from safetensors import safe_open

    from glm_tpu.greenfield.validation.short_context_dsa_oracle import (
        inspect_short_context_dsa_oracle,
    )

    manifest = inspect_short_context_dsa_oracle(DSA_ORACLE)
    with safe_open(
        DSA_ORACLE / manifest["files"]["tensors"]["filename"], framework="np"
    ) as handle:
        positions = np.ascontiguousarray(
            handle.get_tensor("selected_positions")[0, 1][None, :]
        )
        scores = np.ascontiguousarray(
            handle.get_tensor("selected_scores")[0, 1][None, :]
        )
        valid = np.ascontiguousarray(handle.get_tensor("valid_counts")[0, 1][None])
    with np.load(LAYER1, allow_pickle=False) as handle:
        key_bits = np.ascontiguousarray(
            np.asarray(handle["accepted__current_key"], dtype=ml_dtypes.bfloat16).view(
                np.uint16
            )
        )
    with np.load(DB550, allow_pickle=False) as handle:
        order = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        _, carried = derive_expected_dense_boundary_bits(
            handle["dense_virtual_partials_bfloat16_bits"],
            handle["post_attention_residual_bfloat16_bits"],
            order,
        )
    layer1_cache = np.zeros((2, 16, 256, 128), dtype=np.uint16)
    layer1_cache[1, 15, 219] = key_bits
    np.savez(
        path,
        event1_positions=positions,
        event1_valid_counts=valid,
        event1_scores=scores,
        current_carried_halves_bfloat16_bits=np.stack(
            (carried[:, :3072], carried[:, 3072:])
        ),
        current_attention_query_owners_bfloat16_bits=np.zeros(
            (2, 1, 32, 256), dtype=np.uint16
        ),
        current_kv_a_bfloat16_bits=np.zeros((1, 576), dtype=np.uint16),
        layer0_kv_cache_owners_bfloat16_bits=np.zeros(
            (2, 16, 256, 640), dtype=np.uint16
        ),
        layer0_index_cache_owners_bfloat16_bits=np.zeros(
            (2, 16, 256, 128), dtype=np.uint16
        ),
        layer1_index_cache_owners_bfloat16_bits=layer1_cache,
        carried_liveness_digest_owners=np.zeros((2, 2), dtype=np.uint32),
        contract_valid=np.ones((1,), dtype=np.bool_),
    )


def _compare(path: Path) -> dict[str, object]:
    return compare_feature2_numerical_capture(
        path,
        token_oracle_dir=TOKEN_ORACLE,
        dsa_oracle_dir=DSA_ORACLE,
        layer1_internal_reference=LAYER1,
        db529_internal_dir=DB529,
        db550_boundary=DB550,
    )


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
def test_feature2_numerical_comparator_accepts_only_exact_capture(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture)
    report = _compare(capture)
    assert report["exact"] is True
    assert report["status"] == "NUMERICAL_EXACT"
    assert set(report["mismatch_counts"].values()) == {0}

    with np.load(capture, allow_pickle=False) as handle:
        values = {name: np.asarray(handle[name]).copy() for name in handle.files}
    values["event1_positions"][0, 0] ^= np.int32(1)
    np.savez(capture, **values)
    rejected = _compare(capture)
    assert rejected["exact"] is False
    assert rejected["status"] == "NUMERICAL_REJECTED"
    assert rejected["mismatch_counts"]["event1_positions"] == 1


@pytest.mark.skipif(not SOURCES_AVAILABLE, reason="protected sources unavailable")
def test_feature2_numerical_comparator_rejects_schema_drift(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _capture(capture)
    with np.load(capture, allow_pickle=False) as handle:
        values = {name: np.asarray(handle[name]).copy() for name in handle.files}
    values.pop("contract_valid")
    np.savez(capture, **values)
    with pytest.raises(BenchmarkValidationError, match="keys drifted"):
        _compare(capture)


def test_feature2_rejection_recovery_is_cpu_only_distinct_and_default_off() -> None:
    recovery = (
        Path(__file__).parents[3]
        / "scripts/greenfield/recover_pp16_feature2_numerical_rejection.sh"
    )
    text = recovery.read_text()
    environment = {
        "GLM_GREENFIELD_PP16_FEATURE2_RECOVER": "0",
        "GLM_GREENFIELD_PP16_FEATURE2_RECOVERY_MODE": "off",
    }
    completed = subprocess.run(
        ["bash", str(recovery)],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "RUN_PIN=363a52b7c8a4201ffbbb899352b7159c26a95b80" in text
    assert (
        "ORIGINAL_LEDGER_SHA="
        "d857d2e98cf18eb505d447acd52acf062106af75e83ac0918fa64c13105e96c2"
    ) in text
    assert (
        "CAPTURE_SHA="
        "be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
    ) in text
    assert (
        "POST_CENSUS_SHA="
        "cd1f25488e876acf475068131c5eb7f1ca25dabea158c553de86483aa8a40798"
    ) in text
    assert "RECOVERY_REMOTE=$APPROVED_BUCKET/results/$RECOVERY_TAG" in text
    assert "RECOVERY_MODE == validate_only" in text
    assert text.index("if [[ $RECOVERY_MODE == validate_only ]]") < text.index(
        'upload_preterminal "$RECOVERY_DIR/recovery.evidence.sha256"'
    )
    assert "VALIDATED_ONLY" in text
    assert '"$SOURCE_REMOTE/NUMERICAL_REJECTED"' in text
    assert '"$SOURCE_REMOTE/SUCCESS"' in text
    assert '"in_process_release_passed": False' in text
    assert '"terminal_cleanup_authentication_passed": True' in text
    assert "72_812_032" in text
    assert '"event1_positions": 1852' in text
    assert '"carried_bfloat16_bits": 968' in text
    assert (
        "3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2"
    ) in text
    assert (
        "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
    ) in text
    assert "--if-generation-match=0" in text
    assert 'gcloud storage rm --if-generation-match="$generation"' in text
    assert "rollback_unverified_terminal" in text
    assert "terminal_verified=1\ntrap - EXIT" in text
    assert '"cat", f"{uri}#{generation}"' in text
    assert 'metadata.get("crc32c_hash", metadata.get("crc32c"))' in text
    assert '[[ -s $receipt ]] || return 1' in text
    assert 'assert matches == [(remote, generation)]' in text
    assert 'sync -f "$RECOVERY_DIR/upload_receipts/terminal_create.stderr"' in text
    assert '"gate_d_passed": False' in text
    assert '"performance_claim": False' in text
    assert "JAX_PLATFORMS=cpu" in text
    assert "gcloud compute" not in text
    assert "TPU_VISIBLE_DEVICES" not in text
    assert "gcloud storage cp --no-clobber" not in text


@pytest.mark.skipif(
    not SOURCES_AVAILABLE or not PROTECTED_CAPTURE.is_file(),
    reason="protected feature2 numerical sources unavailable",
)
def test_protected_feature2_rejection_exact_identities_are_executable() -> None:
    report = _compare(PROTECTED_CAPTURE)
    assert report["status"] == "NUMERICAL_REJECTED"
    assert report["exact"] is False
    assert report["capture_sha256"] == (
        "be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
    )
    assert report["mismatch_counts"] == {
        "carried_bfloat16_bits": 968,
        "contract_valid": 0,
        "event1_positions": 1852,
        "event1_scores": 2048,
        "event1_valid_counts": 0,
        "layer1_current_key_bfloat16_bits": 0,
    }
    assert report["captured_array_sha256"][
        "current_carried_halves_bfloat16_bits"
    ] == "3f6c86ed6e96a59adfe706a522297bf83c2ed0802a36ede9f06a88cf6f3f53d2"
    assert report["expected_sha256"]["carried_bfloat16_bits"] == (
        "35a601b7f174eb9204848757f709549a31e82774309929f4071c61849626044c"
    )
