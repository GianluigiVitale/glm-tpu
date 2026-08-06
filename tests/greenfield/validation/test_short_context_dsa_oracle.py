from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from glm_tpu.greenfield.validation.short_context_dsa_oracle import (
    ShortContextDsaOracleConfig,
    capture_short_context_dsa_oracle,
    inspect_short_context_dsa_oracle,
)
from glm_tpu.greenfield.validation.short_context_oracle import (
    capture_short_context_oracle,
)
from tests.greenfield.validation.test_short_context_oracle import (
    _config as _token_config,
    _patch_tokenizer,
)


def _write_dump(
    path: Path,
    *,
    step: int,
    event: int,
    position: int,
    width: int,
    corrupt_set: bool = False,
    pad_noise: bool = False,
) -> None:
    rows = 4
    count = position + 1
    indices = np.full((rows, width), -1, np.int32)
    scores = np.full((rows, width), -np.inf, np.float32)
    selected = np.arange(count - 1, -1, -1, dtype=np.int32)
    if corrupt_set:
        selected[-1] = selected[0]
    indices[0, :count] = selected
    scores[0, :count] = np.arange(count, 0, -1, dtype=np.float32)
    if pad_noise:
        for row in range(1, rows):
            stale_count = min(row + 1, width)
            indices[row, :stale_count] = np.arange(
                stale_count - 1, -1, -1, dtype=np.int32
            )
            scores[row, :stale_count] = np.arange(
                stale_count, 0, -1, dtype=np.float32
            )
    positions = np.asarray([position, 0, 0, 0], np.int32)
    valid = np.asarray([True, False, False, False], np.bool_)
    req_ids = np.zeros((rows,), np.int32)
    distribution = np.asarray([1, 1, 1], np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        step_index=np.asarray(step, np.int64),
        event_index=np.asarray(event, np.int64),
        stash_key=np.asarray("topk_indices"),
        process_index=np.asarray(0, np.int64),
        process_count=np.asarray(8, np.int64),
        topk_indices=indices,
        topk_indices__dtype=np.asarray("int32"),
        positions=positions,
        positions__dtype=np.asarray("int32"),
        valid=valid,
        valid__dtype=np.asarray("bool"),
        req_ids=req_ids,
        req_ids__dtype=np.asarray("int32"),
        distribution=distribution,
        distribution__dtype=np.asarray("int32"),
        topk_scores=scores,
        topk_scores__dtype=np.asarray("float32"),
    )


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    corrupt_set: bool = False,
    pad_noise: bool = False,
) -> ShortContextDsaOracleConfig:
    _patch_tokenizer(monkeypatch)
    token_config = _token_config(tmp_path)
    capture_short_context_oracle(token_config)
    dump_prefix = "/tmp/fresh_dsa.npz"
    connection = sqlite3.connect(token_config.results_db)
    environment = json.loads(
        connection.execute("SELECT env_json FROM runs WHERE run_id=1").fetchone()[0]
    )
    environment["os_env"] = {
        "GLM_DCP": "1",
        "GLM_DCP_SCATTER_IMPL": "pageloop",
        "GLM_DSA_DCP": "1",
        "GLM_DSA_DCP_SCATTER_IMPL": "flat",
        "GLM_DSA_DUMP_TOPK": dump_prefix,
        "GLM_DSA_DUMP_TOPK_EVENTS": "all",
        "GLM_DSA_MODE": "pallas_decode",
        "GLM_DSA_SCORER": "xla",
        "GLM_EXPECT_CODE_HASH": "b" * 9,
        "GLM_LOAD_CHECKSUM": "1",
        "GLM_LOAD_NAN_CHECK": "1",
        "GLM_MLA_DCP": "1",
        "GLM_PWAL_NAN_CHECK": "1",
        "GLM_STATE_HASH_REF": "/tmp/golden.json",
        "GLM_WK_OOB_DIR": "/home/gianl/gcs-models/models/GLM-5.2-FP8",
        "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
    }
    connection.execute(
        "UPDATE runs SET env_json=? WHERE run_id=1",
        (json.dumps(environment),),
    )
    connection.commit()
    connection.close()
    source = tmp_path / "dumps" / "worker_02"
    for step_offset, position in enumerate((5, 6)):
        for event in range(2):
            _write_dump(
                source
                / (
                    "fresh_dsa.step"
                    f"{step_offset + 2:04d}.evt{event:02d}.proc0.npz"
                ),
                step=step_offset + 2,
                event=event,
                position=position,
                width=8,
                corrupt_set=corrupt_set and step_offset == 0 and event == 0,
                pad_noise=pad_noise,
            )
    return ShortContextDsaOracleConfig(
        results_db=token_config.results_db,
        token_oracle_dir=token_config.output_dir,
        source_dump_dir=tmp_path / "dumps",
        output_dir=tmp_path / "dsa_oracle",
        capture_code_hash="a" * 40,
        source_capture_code_hash="a" * 40,
        legacy_repository_pin="b" * 40,
        token_oracle_manifest_sha256=json.loads(
            (token_config.output_dir / "manifest.json").read_text()
        )["manifest_sha256"],
        run_id=1,
        item_row_id=2,
        expected_harness_git="abcdef0",
        expected_fork_git="1234567",
        expected_benchmark="passkey",
        expected_model_uri="gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
        expected_prompt_tokens=5,
        expected_generated_tokens=3,
        expected_seed=9,
        expected_gold="42",
        expected_oob_dir=(
            "/home/gianl/gcs-models/models/GLM-5.2-FP8"
        ),
        expected_dump_prefix=dump_prefix,
        expected_process_count=8,
        first_source_step=2,
        decode_step_count=2,
        first_decode_position=5,
        selected_width=8,
        producer_layer_ids=(0, 4),
    )


def test_short_context_dsa_oracle_roundtrip_and_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    manifest = capture_short_context_dsa_oracle(config)
    assert inspect_short_context_dsa_oracle(config.output_dir) == manifest
    assert manifest["event_contract"]["producer_layer_ids"] == [0, 4]
    assert len(manifest["source_dump_files"]) == 4
    with pytest.raises(FileExistsError, match="append-only"):
        capture_short_context_dsa_oracle(config)


def test_short_context_dsa_oracle_excludes_invalid_pad_row_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch, pad_noise=True)
    manifest = capture_short_context_dsa_oracle(config)
    assert manifest["event_contract"]["padded_row_policy"] == (
        "excluded_by_valid_mask_and_provenance_counted"
    )
    assert {
        record["padded_non_sentinel_row_count"]
        for record in manifest["source_dump_files"]
    } == {3}
    assert inspect_short_context_dsa_oracle(config.output_dir) == manifest


def test_short_context_dsa_oracle_refuses_set_and_tensor_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_root = tmp_path / "bad"
    bad_root.mkdir()
    bad = _config(bad_root, monkeypatch, corrupt_set=True)
    with pytest.raises(ValueError, match="selected set"):
        capture_short_context_dsa_oracle(bad)

    good_root = tmp_path / "good"
    good_root.mkdir()
    good = _config(good_root, monkeypatch)
    capture_short_context_dsa_oracle(good)
    with (good.output_dir / "dsa_events.safetensors").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="file size mismatch"):
        inspect_short_context_dsa_oracle(good.output_dir)


def test_short_context_dsa_oracle_refuses_token_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    wrong = replace(config, token_oracle_manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="manifest pin drifted"):
        capture_short_context_dsa_oracle(wrong)


def test_short_context_dsa_oracle_refuses_missing_repair_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    connection = sqlite3.connect(config.results_db)
    environment = json.loads(
        connection.execute("SELECT env_json FROM runs WHERE run_id=1").fetchone()[0]
    )
    del environment["os_env"]["GLM_WK_OOB_GOLDEN"]
    connection.execute(
        "UPDATE runs SET env_json=? WHERE run_id=1",
        (json.dumps(environment),),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="protocol drifted"):
        capture_short_context_dsa_oracle(config)
