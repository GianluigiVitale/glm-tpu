from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from glm_tpu.greenfield.validation.short_context_oracle import (
    ShortContextOracleConfig,
    capture_short_context_oracle,
    inspect_short_context_oracle,
)


class _Tokenizer:
    vocab_size = 154_880

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in value]}

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(chr(value) for value in values)


def _source(tmp_path: Path) -> tuple[Path, Path]:
    database = tmp_path / "results.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY,
            created_utc TEXT NOT NULL,
            model TEXT NOT NULL,
            model_revision TEXT,
            harness_git TEXT,
            fork_git TEXT,
            env_json TEXT,
            pod TEXT,
            note TEXT
        );
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL,
            benchmark TEXT NOT NULL,
            item_id TEXT NOT NULL,
            asked_utc TEXT NOT NULL,
            prompt TEXT NOT NULL,
            gold TEXT,
            raw_output TEXT,
            extracted TEXT,
            correct INTEGER,
            score REAL,
            n_prompt_tokens INTEGER,
            n_gen_tokens INTEGER,
            latency_ms REAL,
            seed INTEGER,
            finish_reason TEXT,
            truncated INTEGER
        );
        """
    )
    environment = json.dumps(
        {
            "model": "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
            "prompt_prefix_ids": [154822, 154824],
            "protocol": "raw",
            "temperature": 0.0,
        }
    )
    connection.execute(
        "INSERT INTO runs VALUES (1,'now',?,NULL,?,?,?,'pod','note')",
        (
            "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
            "abcdef0",
            "1234567",
            environment,
        ),
    )
    connection.execute(
        "INSERT INTO items VALUES "
        "(2,1,'passkey','t0','now','abc','42',' xy','xy',"
        "1,1.0,5,3,1.0,9,NULL,NULL)"
    )
    connection.commit()
    connection.close()
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        (tokenizer_root / filename).write_text(filename)
    return database, tokenizer_root


def _config(tmp_path: Path) -> ShortContextOracleConfig:
    database, tokenizer_root = _source(tmp_path)
    return ShortContextOracleConfig(
        results_db=database,
        tokenizer_root=tokenizer_root,
        output_dir=tmp_path / "oracle",
        capture_code_hash="a" * 40,
        legacy_repository_pin="b" * 40,
        run_id=1,
        item_row_id=2,
        expected_harness_git="abcdef0",
        expected_fork_git="1234567",
        expected_benchmark="passkey",
        expected_model_uri=(
            "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
        ),
        expected_prompt_tokens=5,
        expected_generated_tokens=3,
        expected_seed=9,
        expected_gold="42",
    )


def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )


def test_short_context_oracle_roundtrip_and_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    config = _config(tmp_path)
    manifest = capture_short_context_oracle(config)
    assert inspect_short_context_oracle(config.output_dir) == manifest
    assert manifest["prompt_token_count"] == 5
    assert manifest["generated_token_count"] == 3
    assert manifest["source"]["source_row_sha256"]
    with pytest.raises(FileExistsError, match="append-only"):
        capture_short_context_oracle(config)


def test_short_context_oracle_refuses_identity_and_file_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    config = _config(tmp_path)
    capture_short_context_oracle(config)
    with (config.output_dir / "tokens.safetensors").open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="size mismatch"):
        inspect_short_context_oracle(config.output_dir)

    second_root = tmp_path / "second"
    second_root.mkdir()
    wrong = replace(_config(second_root), expected_gold="wrong")
    with pytest.raises(ValueError, match="identity drifted"):
        capture_short_context_oracle(wrong)
