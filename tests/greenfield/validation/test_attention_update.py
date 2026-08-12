from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.attention_update import (
    ATTENTION_ARMS,
    AcceptedAttentionUpdateCaptureConfig,
    AttentionUpdateComparisonConfig,
    LAYER_NAME,
    PROBE_CLASSIFICATION,
    PROBE_KIND,
    PROBE_RUN_ID,
    capture_accepted_attention_update,
    compare_attention_update_candidates,
    validate_exact_remote_object_set,
)


LEGACY_HASH = "1" * 40
ORACLE_HASH = "2" * 40
PROBE_HASH = "3" * 40
ACCEPTED_TAG = "accepted-attention-update-test"
PROBE_TAG = "projection-reduction-test"
WIDTH = 6144


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(values, dtype=ml_dtypes.bfloat16)
    ).view(np.uint16)


def _write_source(path: Path, bits: np.ndarray, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, np.ndarray] = {
        "artifact_kind": np.asarray("glm52_legacy_attention_update"),
        "format_version": np.asarray(1, dtype=np.int64),
        "capture_mode": np.asarray("attention_update"),
        "process_index": np.asarray(0, dtype=np.int64),
        "process_count": np.asarray(8, dtype=np.int64),
        "layer_name": np.asarray(LAYER_NAME),
        "position": np.asarray(8155, dtype=np.int32),
        "source_row": np.asarray(0, dtype=np.int32),
        "run_tag": np.asarray(ACCEPTED_TAG),
        "code_hash": np.asarray(LEGACY_HASH),
        "oracle_pin": np.asarray(ORACLE_HASH),
        "model_id": np.asarray("zai-org/GLM-5.2-FP8"),
        "attention_update": bits,
        "attention_update__dtype": np.asarray("bfloat16"),
    }
    fields.update({name: np.asarray(value) for name, value in overrides.items()})
    np.savez(path, **fields)


def _capture(tmp_path: Path, bits: np.ndarray) -> Path:
    source = tmp_path / "source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_source(
        source / f"internal.{safe_layer}.position8155.proc0.npz",
        bits,
    )
    output = tmp_path / "accepted"
    capture_accepted_attention_update(
        AcceptedAttentionUpdateCaptureConfig(
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
    local: np.ndarray,
    strategy: np.ndarray,
    dense_only_drift: bool = False,
) -> tuple[str, str, str, str]:
    root.mkdir(parents=True)
    arrays: dict[str, np.ndarray] = {}
    arms: dict[str, object] = {}
    for mechanism, arm_names in ATTENTION_ARMS.items():
        for index, arm in enumerate(arm_names):
            value = local if mechanism == "local" else strategy
            if dense_only_drift and mechanism == "local" and index == 1:
                value = value.copy()
                value[0, 0] ^= np.uint16(1)
            arrays[f"attention_update_bfloat16_bits__{arm}"] = value
            arms[arm] = {
                "attention_strategy_nd": mechanism == "strategy_nd",
                "dense_strategy_nd": arm.endswith("strategy_dense"),
                "hlo": {"contract": {"passed": True}},
                "value_comparison": {
                    "elementwise_exact": True,
                    "expected_sha256": (
                        "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
                    ),
                    "first_mismatch_index": None,
                    "max_abs_error": 0.0,
                    "mean_abs_error": 0.0,
                    "mismatch_count": 0,
                    "observed_sha256": (
                        "79a6e290274ef470b518a6de894b44d2929ad6861d1751f0915aa4eb20cf2e9d"
                    ),
                    "shape": [16384],
                },
            }
    tensor_path = root / "projection_reduction.npz"
    np.savez(tensor_path, **arrays)
    runner = {
        "artifact_kind": PROBE_KIND,
        "arms": arms,
        "classification": PROBE_CLASSIFICATION,
        "code_hash": PROBE_HASH,
        "exact_arms": [],
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
        "elapsed_seconds": 20,
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
            )
        )
        + "\n"
    )
    return (
        _file_sha256(runner_path),
        _file_sha256(tensor_path),
        _file_sha256(summary_path),
        _file_sha256(success_path),
    )


def _comparison_config(
    tmp_path: Path,
    *,
    accepted: Path,
    probe: Path,
    runner_sha: str,
    tensor_sha: str,
    summary_sha: str,
    success_sha: str,
    output_name: str = "comparison",
) -> AttentionUpdateComparisonConfig:
    return AttentionUpdateComparisonConfig(
        accepted_capture_dir=accepted,
        probe_dir=probe,
        output_dir=tmp_path / output_name,
        expected_accepted_capture_file_sha256=_file_sha256(
            accepted / "capture.json"
        ),
        expected_probe_runner_sha256=runner_sha,
        expected_probe_tensor_sha256=tensor_sha,
        expected_probe_summary_sha256=summary_sha,
        expected_probe_success_sha256=success_sha,
        expected_accepted_run_tag=ACCEPTED_TAG,
        expected_legacy_code_hash=LEGACY_HASH,
        expected_oracle_pin=ORACLE_HASH,
        expected_probe_code_hash=PROBE_HASH,
        expected_probe_tag=PROBE_TAG,
    )


def test_capture_seals_exact_bfloat16_row_and_provenance(tmp_path: Path) -> None:
    bits = _bits(np.linspace(-1.0, 1.0, WIDTH, dtype=np.float32))
    output = _capture(tmp_path, bits)
    manifest = json.loads((output / "capture.json").read_text())
    assert manifest["capture_layout"] == "replicated_logical_live_row"
    assert manifest["capture_process_indices"] == [0]
    assert manifest["legacy_code_hash"] == LEGACY_HASH
    assert manifest["oracle_pin"] == ORACLE_HASH
    assert manifest["process_files"][0]["process_index"] == 0
    assert manifest["performance_claim"] is False
    with np.load(output / "attention_update.npz", allow_pickle=False) as payload:
        np.testing.assert_array_equal(
            payload["attention_update_bfloat16_bits"], bits
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"run_tag": "wrong"}, "run_tag"),
        ({"source_row": 1}, "source_row"),
        ({"attention_update__dtype": "float32"}, "attention_update__dtype"),
    ],
)
def test_capture_refuses_source_provenance_drift(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    source = tmp_path / "source"
    safe_layer = LAYER_NAME.replace(".", "_")
    _write_source(
        source / f"internal.{safe_layer}.position8155.proc0.npz",
        _bits(np.zeros(WIDTH, dtype=np.float32)),
        **override,
    )
    with pytest.raises(ValueError, match=match):
        capture_accepted_attention_update(
            AcceptedAttentionUpdateCaptureConfig(
                source_dump_dir=source,
                output_dir=tmp_path / "accepted",
                expected_run_tag=ACCEPTED_TAG,
                expected_legacy_code_hash=LEGACY_HASH,
                expected_oracle_pin=ORACLE_HASH,
            )
        )


def test_comparison_selects_strategy_nd_projection(tmp_path: Path) -> None:
    accepted_bits = _bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))
    accepted = _capture(tmp_path, accepted_bits)
    local = accepted_bits.reshape(1, WIDTH).copy()
    local[0, 5] ^= np.uint16(1)
    strategy = accepted_bits.reshape(1, WIDTH).copy()
    probe = tmp_path / PROBE_TAG
    runner_sha, tensor_sha, summary_sha, success_sha = _write_probe(
        probe, local=local, strategy=strategy
    )
    result = compare_attention_update_candidates(
        _comparison_config(
            tmp_path,
            accepted=accepted,
            probe=probe,
            runner_sha=runner_sha,
            tensor_sha=tensor_sha,
            summary_sha=summary_sha,
            success_sha=success_sha,
        )
    )
    assert result["classification"] == "strategy_nd_attention_projection_exact"
    assert result["exact_candidates"] == ["strategy_nd"]
    assert result["first_open_boundary"] == "after_attention_projection"
    assert result["performance_claim"] is False


def test_comparison_classifies_unresolved(tmp_path: Path) -> None:
    accepted_bits = _bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))
    accepted = _capture(tmp_path, accepted_bits)
    local = accepted_bits.reshape(1, WIDTH).copy()
    strategy = accepted_bits.reshape(1, WIDTH).copy()
    local[0, 5] ^= np.uint16(1)
    strategy[0, 9] ^= np.uint16(1)
    probe = tmp_path / PROBE_TAG
    runner_sha, tensor_sha, summary_sha, success_sha = _write_probe(
        probe, local=local, strategy=strategy
    )
    result = compare_attention_update_candidates(
        _comparison_config(
            tmp_path,
            accepted=accepted,
            probe=probe,
            runner_sha=runner_sha,
            tensor_sha=tensor_sha,
            summary_sha=summary_sha,
            success_sha=success_sha,
        )
    )
    assert result["classification"] == \
        "attention_projection_arithmetic_unresolved"
    assert result["exact_candidates"] == []
    assert result["first_open_boundary"] == "attention_projection"


def test_comparison_refuses_dense_only_drift_and_probe_hash_drift(
    tmp_path: Path,
) -> None:
    accepted_bits = _bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))
    accepted = _capture(tmp_path, accepted_bits)
    candidate = accepted_bits.reshape(1, WIDTH).copy()
    probe = tmp_path / PROBE_TAG
    runner_sha, tensor_sha, summary_sha, success_sha = _write_probe(
        probe,
        local=candidate,
        strategy=candidate,
        dense_only_drift=True,
    )
    with pytest.raises(ValueError, match="changed with dense-only choice"):
        compare_attention_update_candidates(
            _comparison_config(
                tmp_path,
                accepted=accepted,
                probe=probe,
                runner_sha=runner_sha,
                tensor_sha=tensor_sha,
                summary_sha=summary_sha,
                success_sha=success_sha,
                output_name="dense-drift",
            )
        )
    with pytest.raises(ValueError, match="runner file hash drifted"):
        compare_attention_update_candidates(
            _comparison_config(
                tmp_path,
                accepted=accepted,
                probe=probe,
                runner_sha="f" * 64,
                tensor_sha=tensor_sha,
                summary_sha=summary_sha,
                success_sha=success_sha,
                output_name="hash-drift",
            )
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (("position", 999), "identity drifted"),
        (("attention_strategy_nd", False), "prerequisite failed"),
        (("expected_sha256", "0" * 64), "prerequisite failed"),
    ],
)
def test_comparison_refuses_semantically_wrong_probe(
    tmp_path: Path,
    mutation: tuple[str, object],
    match: str,
) -> None:
    accepted_bits = _bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))
    accepted = _capture(tmp_path, accepted_bits)
    candidate = accepted_bits.reshape(1, WIDTH).copy()
    probe = tmp_path / PROBE_TAG
    _write_probe(probe, local=candidate, strategy=candidate)
    runner_path = probe / "runner.json"
    runner = json.loads(runner_path.read_text())
    name, value = mutation
    if name == "position":
        runner[name] = value
    elif name == "attention_strategy_nd":
        runner["arms"]["strategy_attention_local_dense"][name] = value
    else:
        runner["arms"]["local_attention_local_dense"][
            "value_comparison"
        ][name] = value
    runner_path.write_text(json.dumps(runner, sort_keys=True) + "\n")
    config = _comparison_config(
        tmp_path,
        accepted=accepted,
        probe=probe,
        runner_sha=_file_sha256(runner_path),
        tensor_sha=_file_sha256(probe / "projection_reduction.npz"),
        summary_sha=_file_sha256(probe / "summary.json"),
        success_sha=_file_sha256(probe / "SUCCESS"),
        output_name=f"mutation-{name}",
    )
    with pytest.raises(ValueError, match=match):
        compare_attention_update_candidates(config)


def test_capture_loader_refuses_empty_process_ledger(tmp_path: Path) -> None:
    bits = _bits(np.linspace(-2.0, 2.0, WIDTH, dtype=np.float32))
    accepted = _capture(tmp_path, bits)
    manifest_path = accepted / "capture.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["process_files"] = []
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256(
        json.dumps(
            manifest_without_hash,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    probe = tmp_path / PROBE_TAG
    hashes = _write_probe(
        probe,
        local=bits.reshape(1, WIDTH),
        strategy=bits.reshape(1, WIDTH),
    )
    config = _comparison_config(
        tmp_path,
        accepted=accepted,
        probe=probe,
        runner_sha=hashes[0],
        tensor_sha=hashes[1],
        summary_sha=hashes[2],
        success_sha=hashes[3],
        output_name="empty-ledger",
    )
    config = AttentionUpdateComparisonConfig(
        **{
            name: (
                _file_sha256(manifest_path)
                if name == "expected_accepted_capture_file_sha256"
                else getattr(config, name)
            )
            for name in config.__dataclass_fields__
        }
    )
    with pytest.raises(ValueError, match="process ledger drifted"):
        compare_attention_update_candidates(config)


def test_remote_object_set_refuses_planted_extra(tmp_path: Path) -> None:
    (tmp_path / "capture.json").write_text("{}\n")
    prefix = "gs://approved/run"
    with pytest.raises(ValueError, match=r"extra=\['stale.bin'\]"):
        validate_exact_remote_object_set(
            tmp_path,
            prefix,
            [f"{prefix}/capture.json", f"{prefix}/stale.bin"],
        )
    validate_exact_remote_object_set(
        tmp_path,
        prefix,
        [f"{prefix}/capture.json"],
    )


def test_wrapper_publishes_success_only_after_nonterminal_archive_gate() -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    object_gate = script.index("validate_exact_remote_object_set(root, prefix, listing)")
    success_create = script.index('(root / "SUCCESS").write_text')
    upload_marker = 'gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS"'
    success_upload = script.index(upload_marker)
    assert object_gate < success_create < success_upload
    assert "gcloud storage cp" not in script[success_upload + len(upload_marker):]
