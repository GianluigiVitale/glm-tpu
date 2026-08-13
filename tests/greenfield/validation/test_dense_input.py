from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.validation.dense_input import (
    AcceptedDenseInputCaptureConfig,
    CAPTURE_KIND,
    COMPARISON_KIND,
    DenseInputComparisonConfig,
    LAYER_NAME,
    MODEL_ID,
    POSITION,
    PROBE_CLASSIFICATION,
    PROBE_CODE_HASH,
    PROBE_KIND,
    PROBE_TAG,
    SOURCE_KIND,
    WIDTH,
    capture_accepted_dense_input,
    compare_dense_input_candidate,
)


LEGACY_PIN = "1" * 40
ORACLE_PIN = "2" * 40
RUN_TAG = "dense-input-unit"


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_source(root: Path, bits: np.ndarray) -> Path:
    source = root / "source"
    source.mkdir()
    safe_layer = LAYER_NAME.replace(".", "_")
    path = source / f"internals.{safe_layer}.position{POSITION}.proc0.npz"
    np.savez(
        path,
        artifact_kind=np.asarray(SOURCE_KIND),
        format_version=np.asarray(1, dtype=np.int64),
        capture_mode=np.asarray("dense_input"),
        process_index=np.asarray(0, dtype=np.int64),
        process_count=np.asarray(8, dtype=np.int64),
        layer_name=np.asarray(LAYER_NAME),
        position=np.asarray(POSITION, dtype=np.int32),
        source_row=np.asarray(0, dtype=np.int32),
        run_tag=np.asarray(RUN_TAG),
        code_hash=np.asarray(LEGACY_PIN),
        oracle_pin=np.asarray(ORACLE_PIN),
        model_id=np.asarray(MODEL_ID),
        normalized_mlp=np.ascontiguousarray(bits),
        normalized_mlp__dtype=np.asarray("bfloat16"),
    )
    return source


def _capture(root: Path, bits: np.ndarray) -> Path:
    source = _write_source(root, bits)
    output = root / "capture"
    capture_accepted_dense_input(
        AcceptedDenseInputCaptureConfig(
            source_dump_dir=source,
            output_dir=output,
            expected_run_tag=RUN_TAG,
            expected_legacy_code_hash=LEGACY_PIN,
            expected_oracle_pin=ORACLE_PIN,
        )
    )
    return output


def _write_probe(root: Path, candidate: np.ndarray) -> tuple[Path, dict[str, str]]:
    probe = root / PROBE_TAG
    probe.mkdir()
    runner = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": PROBE_CODE_HASH,
        "exact": False,
        "exact_arms": [],
        "performance_claim": False,
        "position": POSITION,
        "status": "SUCCESS",
    }
    (probe / "runner.json").write_text(
        json.dumps(runner, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "artifact_kind": PROBE_KIND,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": PROBE_CODE_HASH,
        "elapsed_seconds": 8,
        "exact_arms": [],
        "performance_claim": False,
        "results_db_run_id": 540,
        "status": "SUCCESS",
    }
    (probe / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (probe / "SUCCESS").write_text(
        "\n".join(
            (
                f"artifact_kind={PROBE_KIND}",
                f"code_hash={PROBE_CODE_HASH}",
                "results_db_run_id=540",
                f"classification={PROBE_CLASSIFICATION}",
                "exact_arms=none",
                "performance_claim=false",
                f"evidence_sha256={'3' * 64}",
                f"remote_objects_sha256={'4' * 64}",
                f"remote_prefix=gs://driftbench-dsv4-uc/results/{PROBE_TAG}",
            )
        )
        + "\n"
    )
    zeros = np.zeros((1, WIDTH), dtype=np.uint16)
    np.savez(
        probe / "dense_convolution.npz",
        accepted_layer1_normalized_bfloat16_bits=np.zeros(WIDTH, dtype=np.uint16),
        dense_update_bfloat16_bits=zeros,
        layer1_normalized_bfloat16_bits=np.zeros(WIDTH, dtype=np.uint16),
        normalized_mlp_bfloat16_bits=candidate.reshape(1, WIDTH),
        post_attention_residual_bfloat16_bits=zeros,
    )
    return probe, {
        name: _file_sha(probe / filename)
        for name, filename in {
            "runner": "runner.json",
            "tensor": "dense_convolution.npz",
            "summary": "summary.json",
            "success": "SUCCESS",
        }.items()
    }


def _comparison_config(
    root: Path, capture: Path, probe: Path, hashes: dict[str, str], candidate: np.ndarray
) -> DenseInputComparisonConfig:
    return DenseInputComparisonConfig(
        accepted_capture_dir=capture,
        probe_dir=probe,
        output_dir=root / "comparison",
        expected_accepted_capture_file_sha256=_file_sha(capture / "capture.json"),
        expected_probe_runner_sha256=hashes["runner"],
        expected_probe_tensor_sha256=hashes["tensor"],
        expected_probe_summary_sha256=hashes["summary"],
        expected_probe_success_sha256=hashes["success"],
        expected_accepted_run_tag=RUN_TAG,
        expected_legacy_code_hash=LEGACY_PIN,
        expected_oracle_pin=ORACLE_PIN,
        expected_probe_normalized_mlp_sha256=_array_sha(candidate),
    )


def test_capture_seals_exact_dense_input(tmp_path: Path) -> None:
    bits = np.arange(WIDTH, dtype=np.uint16)
    output = _capture(tmp_path, bits)
    manifest = json.loads((output / "capture.json").read_text())
    assert manifest["artifact_kind"] == CAPTURE_KIND
    assert manifest["capture_mode"] == "dense_input"
    assert manifest["capture_process_indices"] == [0]
    assert manifest["tensor"]["tensor_sha256"] == _array_sha(bits)
    with np.load(output / "dense_input.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["normalized_mlp_bfloat16_bits"], bits)


def test_capture_refuses_source_contract_drift(tmp_path: Path) -> None:
    bits = np.zeros(WIDTH, dtype=np.uint16)
    source = _write_source(tmp_path, bits)
    path = next(source.glob("*.npz"))
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["capture_mode"] = np.asarray("dense_boundary")
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="capture_mode"):
        capture_accepted_dense_input(
            AcceptedDenseInputCaptureConfig(
                source_dump_dir=source,
                output_dir=tmp_path / "capture",
                expected_run_tag=RUN_TAG,
                expected_legacy_code_hash=LEGACY_PIN,
                expected_oracle_pin=ORACLE_PIN,
            )
        )


def test_comparison_classifies_exact_dense_input(tmp_path: Path) -> None:
    bits = np.arange(WIDTH, dtype=np.uint16)
    capture = _capture(tmp_path, bits)
    probe, hashes = _write_probe(tmp_path, bits)
    result = compare_dense_input_candidate(
        _comparison_config(tmp_path, capture, probe, hashes, bits)
    )
    assert result["artifact_kind"] == COMPARISON_KIND
    assert result["classification"] == "normalized_mlp_exact_dense_arithmetic_open"
    assert result["first_open_boundary"] == "dense_mlp_or_cross_layer_fusion"
    assert result["normalized_mlp"]["elementwise_exact"] is True
    assert result["probe"] == {
        "code_hash": PROBE_CODE_HASH,
        "run_id": 540,
        "runner_sha256": hashes["runner"],
        "success_sha256": hashes["success"],
        "summary_sha256": hashes["summary"],
        "tag": PROBE_TAG,
        "tensor_sha256": hashes["tensor"],
    }


def test_comparison_classifies_pre_dense_divergence(tmp_path: Path) -> None:
    accepted = np.arange(WIDTH, dtype=np.uint16)
    candidate = accepted.copy()
    candidate[17] ^= np.uint16(1)
    capture = _capture(tmp_path, accepted)
    probe, hashes = _write_probe(tmp_path, candidate)
    result = compare_dense_input_candidate(
        _comparison_config(tmp_path, capture, probe, hashes, candidate)
    )
    assert result["classification"] == "normalized_mlp_nonexact"
    assert result["first_open_boundary"] == "post_attention_add_rmsnorm"
    assert result["normalized_mlp"]["mismatch_count"] == 1
    assert result["normalized_mlp"]["first_mismatch_index"] == 17


def test_comparison_refuses_pinned_probe_drift(tmp_path: Path) -> None:
    bits = np.arange(WIDTH, dtype=np.uint16)
    capture = _capture(tmp_path, bits)
    probe, hashes = _write_probe(tmp_path, bits)
    config = _comparison_config(tmp_path, capture, probe, hashes, bits)
    (probe / "runner.json").write_text("{}\n")
    with pytest.raises(ValueError, match="runner file hash drifted"):
        compare_dense_input_candidate(config)


def test_comparison_refuses_capture_manifest_drift(tmp_path: Path) -> None:
    bits = np.arange(WIDTH, dtype=np.uint16)
    capture = _capture(tmp_path, bits)
    probe, hashes = _write_probe(tmp_path, bits)
    config = _comparison_config(tmp_path, capture, probe, hashes, bits)
    manifest = json.loads((capture / "capture.json").read_text())
    manifest["position"] = POSITION - 1
    (capture / "capture.json").write_text(json.dumps(manifest) + "\n")
    config = replace(
        config,
        expected_accepted_capture_file_sha256=_file_sha(capture / "capture.json"),
    )
    with pytest.raises(ValueError, match="manifest hash drifted"):
        compare_dense_input_candidate(config)


def test_protected_wrapper_pins_dense_input_sources_and_cleanup() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    launcher = (
        repo / "scripts/greenfield/run_capture_legacy_layer0_dense_input.sh"
    ).read_text()
    assert "GLM_GREENFIELD_DSA_INTERNALS_MODE=dense_input" in launcher
    assert "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k" in launcher
    for exact in (
        "LEGACY_PIN=0c2f7f28a075a51f5eb51dc98bbb74e363d3290f",
        "OBSERVER_COMMIT_DISTANCE=11",
        "DENSE_CONVOLUTION_RUN_ID=540",
        "DENSE_CONVOLUTION_CODE_HASH=2f63779309b25c71c1cc7d35ff97715ae4bf631e",
        "DENSE_CONVOLUTION_RUNNER_SHA=876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91",
        "DENSE_CONVOLUTION_TENSOR_SHA=2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9",
        "DENSE_CONVOLUTION_SUMMARY_SHA=9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449",
        "DENSE_CONVOLUTION_SUCCESS_SHA=d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799",
    ):
        assert exact in wrapper
    assert wrapper.index("protected DB540 live DB identity drifted") < wrapper.index(
        "strict_census pre"
    )
    assert wrapper.index("strict_census post") < wrapper.index(
        "freezing fresh DSA-oracle evidence"
    )
    assert wrapper.index("validate_exact_remote_object_set(root, prefix, listing)") < (
        wrapper.index('(root / "SUCCESS").write_text')
    )


def test_protected_db540_replay(tmp_path: Path) -> None:
    probe = Path(
        "/home/gianl/glm-run/"
        "greenfield_layer0_dense_convolution_20260813T005213127235575Z"
    )
    if not all(
        (probe / name).is_file()
        for name in ("runner.json", "dense_convolution.npz", "summary.json", "SUCCESS")
    ):
        pytest.skip("protected DB540 evidence is not mounted")
    with np.load(probe / "dense_convolution.npz", allow_pickle=False) as payload:
        bits = np.ascontiguousarray(payload["normalized_mlp_bfloat16_bits"][0])
    capture = _capture(tmp_path, bits)
    result = compare_dense_input_candidate(
        DenseInputComparisonConfig(
            accepted_capture_dir=capture,
            probe_dir=probe,
            output_dir=tmp_path / "comparison",
            expected_accepted_capture_file_sha256=_file_sha(capture / "capture.json"),
            expected_probe_runner_sha256=(
                "876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91"
            ),
            expected_probe_tensor_sha256=(
                "2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9"
            ),
            expected_probe_summary_sha256=(
                "9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449"
            ),
            expected_probe_success_sha256=(
                "d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799"
            ),
            expected_accepted_run_tag=RUN_TAG,
            expected_legacy_code_hash=LEGACY_PIN,
            expected_oracle_pin=ORACLE_PIN,
        )
    )
    assert result["classification"] == "normalized_mlp_exact_dense_arithmetic_open"
    assert result["normalized_mlp"]["elementwise_exact"] is True


def _db540_preflight_program() -> str:
    wrapper = Path(__file__).resolve().parents[3] / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    return next(program for program in programs if "protected DB540 run is absent" in program)


def _write_db540(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (
          run_id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL, model TEXT NOT NULL,
          model_revision TEXT, harness_git TEXT, fork_git TEXT, env_json TEXT,
          pod TEXT, note TEXT
        );
        CREATE TABLE items (
          id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, benchmark TEXT NOT NULL,
          item_id TEXT NOT NULL, asked_utc TEXT NOT NULL, prompt TEXT NOT NULL,
          gold TEXT, raw_output TEXT, extracted TEXT, correct INTEGER, score REAL,
          n_prompt_tokens INTEGER, n_gen_tokens INTEGER, latency_ms REAL, seed INTEGER,
          finish_reason TEXT, truncated INTEGER
        );
        CREATE TABLE summary (
          id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, benchmark TEXT NOT NULL,
          created_utc TEXT NOT NULL, n INTEGER, metric TEXT, value REAL,
          card_value REAL, delta REAL, note TEXT
        );
        """
    )
    environment = {
        "GLM_ENGINE": "greenfield_dense_convolution_probe",
        "checkpoint_manifest_sha256": (
            "de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134"
        ),
        "classification": "accepted_dense_convolution_nonexact",
        "db538_tensor_sha256": (
            "e801d5471697fefd1477c46603698289de93818d08d214bdf56e576f52819e0e"
        ),
        "greenfield_code_hash": PROBE_CODE_HASH,
        "greenfield_run_tag": PROBE_TAG,
    }
    connection.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
        (
            540,
            "2026-08-13T00:53:15+00:00",
            "zai-org/GLM-5.2-FP8:greenfield-layer0-dense-convolution",
            "native-jax-db538-dense-convolution-v1",
            "2f63779",
            "b3c25df47",
            json.dumps(environment, sort_keys=True),
            "db-v4-64-od",
            "Protected layer-0 dense convolution discriminator; no performance claim.",
        ),
    )
    connection.execute(
        "INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            1824,
            540,
            "greenfield_layer0_dense_convolution",
            "position8155",
            "2026-08-13T00:53:15+00:00",
            "Sealed exact StrategyND attention boundary at first 8K decode row.",
            "Exact accepted BF16 layer-1 normalized hidden [6144].",
            '{"classification": "accepted_dense_convolution_nonexact", '
            '"exact_arms": [], "mismatch_counts": '
            '{"accepted_dense_convolution": 1073}}',
            "none",
            0,
            0.0,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    connection.execute(
        "INSERT INTO summary VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            771,
            540,
            "greenfield_layer0_dense_convolution",
            "2026-08-13T00:53:15+00:00",
            1,
            "probe_contract_valid",
            1.0,
            None,
            None,
            "Diagnostic layer-0 arithmetic classification only; no decoder claim.",
        ),
    )
    connection.commit()
    connection.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "none",
        "run_harness",
        "run_environment",
        "item_raw",
        "item_extracted",
        "item_prompt",
        "summary_n",
        "summary_note",
    ),
)
def test_db540_preflight_authenticates_complete_rows(
    tmp_path: Path, mutation: str
) -> None:
    database = tmp_path / "results.db"
    _write_db540(database)
    statements = {
        "run_harness": "UPDATE runs SET harness_git='rogue' WHERE run_id=540",
        "run_environment": "UPDATE runs SET env_json='{}' WHERE run_id=540",
        "item_raw": "UPDATE items SET raw_output='rogue' WHERE run_id=540",
        "item_extracted": "UPDATE items SET extracted='rogue' WHERE run_id=540",
        "item_prompt": "UPDATE items SET prompt='rogue' WHERE run_id=540",
        "summary_n": "UPDATE summary SET n=2 WHERE run_id=540",
        "summary_note": "UPDATE summary SET note='rogue' WHERE run_id=540",
    }
    if mutation != "none":
        connection = sqlite3.connect(database)
        connection.execute(statements[mutation])
        connection.commit()
        connection.close()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _db540_preflight_program(),
            str(database),
            "540",
            PROBE_TAG,
            PROBE_CODE_HASH,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert (completed.returncode == 0) is (mutation == "none")


def _terminal_program() -> str:
    wrapper = Path(__file__).resolve().parents[3] / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    return next(program for program in programs if "dense-input numerical" in program)


def _manifest(value: dict[str, object]) -> dict[str, object]:
    value = dict(value)
    value["manifest_sha256"] = sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return value


def _write_terminal_fixture(root: Path, *, exact: bool) -> tuple[Path, Path]:
    (root / "oracle").mkdir(parents=True)
    (root / "dense_input_capture").mkdir()
    (root / "dense_input_comparison").mkdir()
    (root / "oracle" / "manifest.json").write_text(
        json.dumps({"artifact_kind": "oracle", "manifest_sha256": "5" * 64})
    )
    (root / "evidence_sha256.json").write_text("{}\n")
    (root / "remote_objects.json").write_text("{}\n")
    (root / "dsa_exact_comparison.json").write_text('{"exact": true}\n')
    observed = (
        "082125fead43b25f10686705c1b6473153f4092dd5bc476f8e01a86629f0758f"
    )
    expected = observed if exact else "6" * 64
    capture = _manifest({
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "replicated_logical_live_row",
        "capture_mode": "dense_input",
        "capture_process_indices": [0],
        "diagnostic_only": True,
        "legacy_code_hash": "0c2f7f28a075a51f5eb51dc98bbb74e363d3290f",
        "oracle_pin": "b3c25df47ac98783912dc658878181ec0a8ae16d",
        "performance_claim": False,
        "position": 8155,
        "layer_name": LAYER_NAME,
        "tensor": {"shape": [6144], "tensor_sha256": expected},
    })
    comparison = _manifest({
        "accepted_capture_manifest_sha256": capture["manifest_sha256"],
        "artifact_kind": COMPARISON_KIND,
        "classification": (
            "normalized_mlp_exact_dense_arithmetic_open"
            if exact
            else "normalized_mlp_nonexact"
        ),
        "diagnostic_only": True,
        "first_open_boundary": (
            "dense_mlp_or_cross_layer_fusion"
            if exact
            else "post_attention_add_rmsnorm"
        ),
        "normalized_mlp": {
            "elementwise_exact": exact,
            "expected_sha256": expected,
            "first_mismatch_index": None if exact else 7,
            "max_abs_error": 0.0 if exact else 0.015625,
            "mean_abs_error": 0.0 if exact else 0.0001,
            "mismatch_count": 0 if exact else 3,
            "observed_sha256": observed,
            "shape": [6144],
        },
        "performance_claim": False,
        "probe": {
            "code_hash": PROBE_CODE_HASH,
            "run_id": 540,
            "runner_sha256": (
                "876353e2504d728343223f03be9092a08d2662924ceb4b88d6f09563101bad91"
            ),
            "success_sha256": (
                "d4c01377daae55ea23b329b1b6dc819b9595dc80f7d35521d0cbdd7d28caa799"
            ),
            "summary_sha256": (
                "9b277ca495d3b9e9ce497d2bf78a520a2672f133e68228d14a10937d4a4b1449"
            ),
            "tag": PROBE_TAG,
            "tensor_sha256": (
                "2cdf128976eb7e04d5c84e012f066d1766361af57896cd22f83c01c7d704e1b9"
            ),
        },
        "status": "SUCCESS",
    })
    capture_path = root / "dense_input_capture" / "capture.json"
    comparison_path = root / "dense_input_comparison" / "comparison.json"
    capture_path.write_text(json.dumps(capture))
    comparison_path.write_text(json.dumps(comparison))
    return capture_path, comparison_path


def _run_terminal(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _terminal_program(),
            str(root),
            "gs://driftbench-dsv4-uc/unit",
            "7" * 40,
            "0c2f7f28a075a51f5eb51dc98bbb74e363d3290f",
            "1",
            "2",
            "294",
            "1",
            "1",
            "0",
            "0",
            "0",
            "dense_input",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("exact", (True, False))
def test_terminal_accepts_consistent_dense_input_verdict(
    tmp_path: Path, exact: bool
) -> None:
    _write_terminal_fixture(tmp_path, exact=exact)
    completed = _run_terminal(tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        "exact_nonzero_count",
        "exact_first_index",
        "exact_nonzero_error",
        "exact_unequal_sha",
        "nonexact_zero_count",
        "nonexact_missing_index",
        "nonexact_zero_error",
        "nonexact_equal_sha",
    ),
)
def test_terminal_refuses_contradictory_dense_input_verdict(
    tmp_path: Path, mutation: str
) -> None:
    exact = mutation.startswith("exact_")
    capture_path, comparison_path = _write_terminal_fixture(tmp_path, exact=exact)
    capture = json.loads(capture_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    normalized = comparison["normalized_mlp"]
    observed = normalized["observed_sha256"]
    mutations = {
        "exact_nonzero_count": ("mismatch_count", 1),
        "exact_first_index": ("first_mismatch_index", 0),
        "exact_nonzero_error": ("mean_abs_error", 0.001),
        "exact_unequal_sha": ("expected_sha256", "8" * 64),
        "nonexact_zero_count": ("mismatch_count", 0),
        "nonexact_missing_index": ("first_mismatch_index", None),
        "nonexact_zero_error": ("mean_abs_error", 0.0),
        "nonexact_equal_sha": ("expected_sha256", observed),
    }
    name, value = mutations[mutation]
    normalized[name] = value
    if name == "expected_sha256":
        capture["tensor"]["tensor_sha256"] = value
        capture.pop("manifest_sha256")
        capture = _manifest(capture)
        comparison["accepted_capture_manifest_sha256"] = capture["manifest_sha256"]
        capture_path.write_text(json.dumps(capture))
    comparison.pop("manifest_sha256")
    comparison = _manifest(comparison)
    comparison_path.write_text(json.dumps(comparison))
    completed = _run_terminal(tmp_path)
    assert completed.returncode != 0
