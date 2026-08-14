from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.dense_boundary import (
    AcceptedDenseBoundaryCaptureConfig,
    DenseBoundaryComparisonConfig,
    LAYER_NAME,
    PROBE_CLASSIFICATION,
    PROBE_KIND,
    capture_accepted_dense_boundary,
    compare_dense_boundary_candidate,
)


WIDTH = 6144
LEGACY_HASH = "1" * 40
ORACLE_HASH = "2" * 40
PROBE_HASH = "3" * 40
ACCEPTED_TAG = "accepted-dense-boundary-test"
PROBE_TAG = "dense-convolution-test"
PROBE_RUN_ID = 540


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _manifest_sha256(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(values, dtype=ml_dtypes.bfloat16)
    ).view(np.uint16)


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    mismatch = expected != observed
    count = int(np.count_nonzero(mismatch))
    expected_float = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_float = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    error = np.abs(expected_float - observed_float)
    return {
        "elementwise_exact": count == 0,
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": int(np.flatnonzero(mismatch)[0]) if count else None,
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": count,
        "observed_sha256": _array_sha256(observed),
        "shape": [WIDTH],
    }


def _write_source(
    path: Path,
    dense: np.ndarray,
    residual: np.ndarray,
    **overrides: object,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "artifact_kind": np.asarray("glm52_legacy_dense_boundary"),
        "format_version": np.asarray(1, dtype=np.int64),
        "capture_mode": np.asarray("dense_boundary"),
        "process_index": np.asarray(0, dtype=np.int64),
        "process_count": np.asarray(8, dtype=np.int64),
        "layer_name": np.asarray(LAYER_NAME),
        "position": np.asarray(8155, dtype=np.int32),
        "source_row": np.asarray(0, dtype=np.int32),
        "run_tag": np.asarray(ACCEPTED_TAG),
        "code_hash": np.asarray(LEGACY_HASH),
        "oracle_pin": np.asarray(ORACLE_HASH),
        "model_id": np.asarray("zai-org/GLM-5.2-FP8"),
        "dense_update": dense,
        "dense_update__dtype": np.asarray("bfloat16"),
        "post_attention_residual": residual,
        "post_attention_residual__dtype": np.asarray("bfloat16"),
    }
    fields.update({name: np.asarray(value) for name, value in overrides.items()})
    np.savez(path, **fields)


def _capture(tmp_path: Path, dense: np.ndarray, residual: np.ndarray) -> Path:
    source = tmp_path / "source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_source(
        source / f"internal.{safe_layer}.position8155.proc0.npz",
        dense,
        residual,
    )
    output = tmp_path / "accepted"
    capture_accepted_dense_boundary(
        AcceptedDenseBoundaryCaptureConfig(
            source_dump_dir=source,
            output_dir=output,
            expected_run_tag=ACCEPTED_TAG,
            expected_legacy_code_hash=LEGACY_HASH,
            expected_oracle_pin=ORACLE_HASH,
        )
    )
    return output


def _write_probe(
    root: Path,
    *,
    dense: np.ndarray,
    residual: np.ndarray,
    accepted_layer1: np.ndarray,
    candidate_layer1: np.ndarray,
) -> tuple[str, str, str, str]:
    root.mkdir(parents=True)
    tensor_path = root / "dense_convolution.npz"
    np.savez(
        tensor_path,
        accepted_layer1_normalized_bfloat16_bits=accepted_layer1,
        dense_update_bfloat16_bits=dense.reshape(1, WIDTH),
        layer1_normalized_bfloat16_bits=candidate_layer1,
        normalized_mlp_bfloat16_bits=np.zeros((1, WIDTH), dtype=np.uint16),
        post_attention_residual_bfloat16_bits=residual.reshape(1, WIDTH),
    )
    runner = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": PROBE_HASH,
        "exact": False,
        "exact_arms": [],
        "layer1_comparison": _comparison(accepted_layer1, candidate_layer1),
        "performance_claim": False,
        "position": 8155,
        "status": "SUCCESS",
    }
    runner_path = root / "runner.json"
    runner_path.write_text(json.dumps(runner, sort_keys=True) + "\n")
    summary = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": PROBE_HASH,
        "elapsed_seconds": 8,
        "exact_arms": [],
        "performance_claim": False,
        "results_db_run_id": PROBE_RUN_ID,
        "status": "SUCCESS",
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    success_path = root / "SUCCESS"
    success_path.write_text(
        "\n".join(
            (
                f"artifact_kind={PROBE_KIND}",
                f"code_hash={PROBE_HASH}",
                f"results_db_run_id={PROBE_RUN_ID}",
                f"classification={PROBE_CLASSIFICATION}",
                "exact_arms=none",
                "performance_claim=false",
                f"evidence_sha256={'4' * 64}",
                f"remote_objects_sha256={'5' * 64}",
                f"remote_prefix=gs://driftbench-dsv4-uc/results/{PROBE_TAG}",
            )
        )
        + "\n"
    )
    return tuple(
        _file_sha256(path)
        for path in (runner_path, tensor_path, summary_path, success_path)
    )


def _config(
    tmp_path: Path,
    *,
    accepted: Path,
    probe: Path,
    hashes: tuple[str, str, str, str],
    dense: np.ndarray,
    residual: np.ndarray,
    accepted_layer1: np.ndarray,
    candidate_layer1: np.ndarray,
    output: str = "comparison",
) -> DenseBoundaryComparisonConfig:
    return DenseBoundaryComparisonConfig(
        accepted_capture_dir=accepted,
        probe_dir=probe,
        output_dir=tmp_path / output,
        expected_accepted_capture_file_sha256=_file_sha256(
            accepted / "capture.json"
        ),
        expected_probe_runner_sha256=hashes[0],
        expected_probe_tensor_sha256=hashes[1],
        expected_probe_summary_sha256=hashes[2],
        expected_probe_success_sha256=hashes[3],
        expected_accepted_run_tag=ACCEPTED_TAG,
        expected_legacy_code_hash=LEGACY_HASH,
        expected_oracle_pin=ORACLE_HASH,
        expected_probe_code_hash=PROBE_HASH,
        expected_probe_tag=PROBE_TAG,
        expected_probe_run_id=PROBE_RUN_ID,
        expected_probe_dense_update_sha256=_array_sha256(dense),
        expected_probe_post_attention_residual_sha256=_array_sha256(residual),
        expected_accepted_layer1_sha256=_array_sha256(accepted_layer1),
        expected_probe_layer1_sha256=_array_sha256(candidate_layer1),
    )


def _fixture_values() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dense = _bits(np.linspace(-2, 2, WIDTH, dtype=np.float32))
    residual = _bits(np.linspace(1, 3, WIDTH, dtype=np.float32))
    accepted_layer1 = _bits(np.linspace(-1, 1, WIDTH, dtype=np.float32))
    candidate_layer1 = accepted_layer1.copy()
    candidate_layer1[1] ^= np.uint16(1)
    return dense, residual, accepted_layer1, candidate_layer1


def test_capture_seals_both_exact_rows(tmp_path: Path) -> None:
    dense, residual, _, _ = _fixture_values()
    output = _capture(tmp_path, dense, residual)
    manifest = json.loads((output / "capture.json").read_text())
    assert manifest["capture_mode"] == "dense_boundary"
    assert set(manifest["tensors"]) == {
        "dense_update",
        "post_attention_residual",
    }
    with np.load(output / "dense_boundary.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["dense_update_bfloat16_bits"], dense)
        np.testing.assert_array_equal(
            payload["post_attention_residual_bfloat16_bits"], residual
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"run_tag": "wrong"}, "run_tag"),
        ({"source_row": 1}, "source_row"),
        ({"dense_update__dtype": "float32"}, "dense_update__dtype"),
    ],
)
def test_capture_refuses_source_drift(
    tmp_path: Path, override: dict[str, object], match: str
) -> None:
    dense, residual, _, _ = _fixture_values()
    source = tmp_path / "source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_source(
        source / f"internal.{safe_layer}.position8155.proc0.npz",
        dense,
        residual,
        **override,
    )
    with pytest.raises(ValueError, match=match):
        capture_accepted_dense_boundary(
            AcceptedDenseBoundaryCaptureConfig(
                source_dump_dir=source,
                output_dir=tmp_path / "accepted",
                expected_run_tag=ACCEPTED_TAG,
                expected_legacy_code_hash=LEGACY_HASH,
                expected_oracle_pin=ORACLE_HASH,
            )
        )


@pytest.mark.parametrize(
    ("dense_exact", "classification", "boundary"),
    [
        (True, "dense_update_exact_layer1_fused_norm_open", "layer1_fused_add_rmsnorm"),
        (False, "dense_mlp_output_nonexact", "dense_mlp_input_or_arithmetic"),
    ],
)
def test_comparison_classifies_first_open_boundary(
    tmp_path: Path,
    dense_exact: bool,
    classification: str,
    boundary: str,
) -> None:
    candidate_dense, residual, accepted_layer1, candidate_layer1 = _fixture_values()
    accepted_dense = candidate_dense.copy()
    if not dense_exact:
        accepted_dense[7] ^= np.uint16(1)
    accepted = _capture(tmp_path, accepted_dense, residual)
    probe = tmp_path / PROBE_TAG
    hashes = _write_probe(
        probe,
        dense=candidate_dense,
        residual=residual,
        accepted_layer1=accepted_layer1,
        candidate_layer1=candidate_layer1,
    )
    result = compare_dense_boundary_candidate(
        _config(
            tmp_path,
            accepted=accepted,
            probe=probe,
            hashes=hashes,
            dense=candidate_dense,
            residual=residual,
            accepted_layer1=accepted_layer1,
            candidate_layer1=candidate_layer1,
        )
    )
    assert result["classification"] == classification
    assert result["first_open_boundary"] == boundary
    assert result["post_attention_residual"]["elementwise_exact"] is True
    assert result["probe"] == {
        "code_hash": PROBE_HASH,
        "run_id": PROBE_RUN_ID,
        "runner_sha256": hashes[0],
        "success_sha256": hashes[3],
        "summary_sha256": hashes[2],
        "tag": PROBE_TAG,
        "tensor_sha256": hashes[1],
    }


def test_comparison_classifies_residual_and_refuses_layer1_verdict_drift(
    tmp_path: Path,
) -> None:
    dense, residual, accepted_layer1, candidate_layer1 = _fixture_values()
    accepted_residual = residual.copy()
    accepted_residual[9] ^= np.uint16(1)
    accepted = _capture(tmp_path, dense, accepted_residual)
    probe = tmp_path / PROBE_TAG
    hashes = _write_probe(
        probe,
        dense=dense,
        residual=residual,
        accepted_layer1=accepted_layer1,
        candidate_layer1=candidate_layer1,
    )
    result = compare_dense_boundary_candidate(
        _config(
            tmp_path,
            accepted=accepted,
            probe=probe,
            hashes=hashes,
            dense=dense,
            residual=residual,
            accepted_layer1=accepted_layer1,
            candidate_layer1=candidate_layer1,
            output="residual-drift",
        )
    )
    assert result["classification"] == "post_attention_residual_nonexact"
    assert result["first_open_boundary"] == "layer0_post_attention_residual"
    assert result["post_attention_residual"]["elementwise_exact"] is False
    assert result["post_attention_residual"]["mismatch_count"] == 1

    runner_path = probe / "runner.json"
    runner = json.loads(runner_path.read_text())
    runner["layer1_comparison"]["mismatch_count"] += 1
    runner_path.write_text(json.dumps(runner, sort_keys=True) + "\n")
    changed_hashes = (_file_sha256(runner_path), *hashes[1:])
    with pytest.raises(ValueError, match="layer1 verdict drifted"):
        compare_dense_boundary_candidate(
            _config(
                tmp_path,
                accepted=accepted,
                probe=probe,
                hashes=changed_hashes,
                dense=dense,
                residual=residual,
                accepted_layer1=accepted_layer1,
                candidate_layer1=candidate_layer1,
                output="layer1-drift",
            )
        )


def test_capture_and_comparison_are_append_only(tmp_path: Path) -> None:
    dense, residual, accepted_layer1, candidate_layer1 = _fixture_values()
    accepted = _capture(tmp_path, dense, residual)
    with pytest.raises(FileExistsError, match="append-only"):
        capture_accepted_dense_boundary(
            AcceptedDenseBoundaryCaptureConfig(
                source_dump_dir=tmp_path / "source",
                output_dir=accepted,
                expected_run_tag=ACCEPTED_TAG,
                expected_legacy_code_hash=LEGACY_HASH,
                expected_oracle_pin=ORACLE_HASH,
            )
        )
    probe = tmp_path / PROBE_TAG
    hashes = _write_probe(
        probe,
        dense=dense,
        residual=residual,
        accepted_layer1=accepted_layer1,
        candidate_layer1=candidate_layer1,
    )
    config = _config(
        tmp_path,
        accepted=accepted,
        probe=probe,
        hashes=hashes,
        dense=dense,
        residual=residual,
        accepted_layer1=accepted_layer1,
        candidate_layer1=candidate_layer1,
    )
    compare_dense_boundary_candidate(config)
    with pytest.raises(FileExistsError, match="append-only"):
        compare_dense_boundary_candidate(config)


def test_protected_wrapper_pins_dense_boundary_sources_and_cleanup() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    launcher = (
        repo / "scripts/greenfield/run_capture_legacy_layer0_dense_boundary.sh"
    ).read_text()
    assert "GLM_GREENFIELD_DSA_INTERNALS_MODE=dense_boundary" in launcher
    assert "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k" in launcher
    for exact in (
        "LEGACY_PIN=8443ea64f4574335091130f0e4f1dfef258c91f7",
        "OBSERVER_COMMIT_DISTANCE=10",
        "DENSE_CONVOLUTION_RUN_ID=540",
        "DENSE_CONVOLUTION_CODE_HASH=2f63779309b25c71c1cc7d35ff97715ae4bf631e",
        "DENSE_CONVOLUTION_RUNNER_SHA=876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91",
        "DENSE_CONVOLUTION_TENSOR_SHA=2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9",
        "DENSE_CONVOLUTION_SUMMARY_SHA=9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449",
        "DENSE_CONVOLUTION_SUCCESS_SHA=d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799",
    ):
        assert exact in wrapper
    assert '"post_attention_residual_nonexact"' in wrapper
    assert '"layer0_post_attention_residual"' in wrapper
    assert '"dense_boundary_residual_exact": str(residual_exact).lower()' in wrapper
    assert wrapper.index("protected DB540 live DB identity drifted") < wrapper.index(
        "strict_census pre"
    )
    assert wrapper.index("strict_census post") < wrapper.index(
        "freezing fresh DSA-oracle evidence"
    )
    assert wrapper.index("validate_exact_remote_object_set(root, prefix, listing)") < (
        wrapper.index('(root / "SUCCESS").write_text')
    )


def _run_dense_boundary_terminal(
    root: Path,
    *,
    residual_exact: bool,
    mutation: str | None = None,
) -> subprocess.CompletedProcess[str]:
    wrapper = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    marker = (
        'PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python - \\\n'
        '  "$RUN_DIR" "$REMOTE_PREFIX" "$PIN" \\\n'
    )
    start = wrapper.index(marker) + len(marker)
    start = wrapper.index("from hashlib import sha256", start)
    body = wrapper[start : wrapper.index("\nPY\n", start)]

    oracle_pin = "b3c25df47ac98783912dc658878181ec0a8ae16d"
    (root / "oracle").mkdir(parents=True)
    (root / "oracle" / "manifest.json").write_text(
        json.dumps({"artifact_kind": "unit", "manifest_sha256": "a" * 64})
    )
    (root / "evidence_sha256.json").write_text("{}\n")
    (root / "remote_objects.json").write_text("{}\n")
    (root / "dsa_exact_comparison.json").write_text('{"exact": true}\n')

    capture: dict[str, object] = {
        "artifact_kind": "glm52_accepted_dense_boundary_capture",
        "capture_layout": "replicated_logical_live_rows",
        "capture_mode": "dense_boundary",
        "capture_process_indices": [0],
        "diagnostic_only": True,
        "format_version": 1,
        "layer_name": LAYER_NAME,
        "legacy_code_hash": LEGACY_HASH,
        "model_id": "zai-org/GLM-5.2-FP8",
        "oracle_pin": oracle_pin,
        "performance_claim": False,
        "position": 8155,
        "process_count": 8,
        "process_files": [{
            "byte_count": 1,
            "path": "source.proc0.npz",
            "process_index": 0,
            "sha256": "4" * 64,
        }],
        "run_tag": "unit-dense-boundary",
        "tensor_file": {
            "byte_count": 1,
            "filename": "dense_boundary.npz",
            "sha256": "5" * 64,
        },
        "tensors": {
            "dense_update": {"shape": [WIDTH], "sha256": "6" * 64},
            "post_attention_residual": {
                "shape": [WIDTH],
                "sha256": "7" * 64,
            },
        },
    }
    capture["manifest_sha256"] = _manifest_sha256(capture)
    zero = np.zeros((WIDTH,), dtype=np.uint16)
    dense = _comparison(zero, zero)
    residual_observed = zero.copy()
    if not residual_exact:
        residual_observed[9] = np.uint16(1)
    residual = _comparison(zero, residual_observed)
    comparison: dict[str, object] = {
        "accepted_capture_manifest_sha256": capture["manifest_sha256"],
        "artifact_kind": "glm52_accepted_greenfield_dense_boundary_comparison",
        "classification": (
            "dense_update_exact_layer1_fused_norm_open"
            if residual_exact
            else "post_attention_residual_nonexact"
        ),
        "dense_update": dense,
        "diagnostic_only": True,
        "first_open_boundary": (
            "layer1_fused_add_rmsnorm"
            if residual_exact
            else "layer0_post_attention_residual"
        ),
        "format_version": 1,
        "legacy_code_hash": LEGACY_HASH,
        "oracle_pin": oracle_pin,
        "performance_claim": False,
        "position": 8155,
        "post_attention_residual": residual,
        "probe": {
            "code_hash": "2f63779309b25c71c1cc7d35ff97715ae4bf631e",
            "run_id": 540,
            "runner_sha256": "876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91",
            "success_sha256": "d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799",
            "summary_sha256": "9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449",
            "tag": "greenfield_layer0_dense_convolution_20260813T005213127235575Z",
            "tensor_sha256": "2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9",
        },
        "status": "SUCCESS",
    }
    comparison["manifest_sha256"] = _manifest_sha256(comparison)
    if mutation == "boolean_error":
        residual["max_abs_error"] = True
        comparison["manifest_sha256"] = _manifest_sha256(comparison)
    elif mutation == "invalid_sha":
        residual["expected_sha256"] = "not-a-sha"
        comparison["manifest_sha256"] = _manifest_sha256(comparison)
    elif mutation == "comparison_manifest":
        comparison["manifest_sha256"] = "0" * 64
    elif mutation == "capture_manifest":
        capture["manifest_sha256"] = "0" * 64
    elif mutation == "dense_bool":
        dense["elementwise_exact"] = 1
        comparison["manifest_sha256"] = _manifest_sha256(comparison)

    capture_dir = root / "dense_boundary_capture"
    comparison_dir = root / "dense_boundary_comparison"
    capture_dir.mkdir()
    comparison_dir.mkdir()
    (capture_dir / "capture.json").write_text(json.dumps(capture))
    (comparison_dir / "comparison.json").write_text(json.dumps(comparison))
    args = [
        str(root),
        "gs://unit/result",
        "a" * 40,
        LEGACY_HASH,
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
        "0",
        "dense_boundary",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "unit-tag",
        "unused",
        "0" * 64,
        "0" * 64,
        "0" * 64,
        "0" * 64,
        oracle_pin,
    ]
    return subprocess.run(
        [sys.executable, "-", *args],
        input=body,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("residual_exact", [True, False])
def test_dense_boundary_terminal_accepts_exact_numerical_schemas(
    tmp_path: Path,
    residual_exact: bool,
) -> None:
    completed = _run_dense_boundary_terminal(
        tmp_path / str(residual_exact), residual_exact=residual_exact
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "boolean_error",
        "invalid_sha",
        "comparison_manifest",
        "capture_manifest",
        "dense_bool",
    ],
)
def test_dense_boundary_terminal_refuses_schema_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    completed = _run_dense_boundary_terminal(
        tmp_path / mutation,
        residual_exact=False,
        mutation=mutation,
    )
    assert completed.returncode != 0
