from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from glm_tpu.greenfield.validation.short_context_logprob_oracle import (
    ShortContextLogprobOracleConfig,
    capture_short_context_logprob_oracle,
    inspect_short_context_logprob_oracle,
    normalize_sample_logprobs,
)
from glm_tpu.greenfield.validation.short_context_oracle import (
    ShortContextOracleConfig,
    capture_short_context_oracle,
)


@dataclass
class _Logprob:
    logprob: float
    rank: int


class _Tokenizer:
    vocab_size = 154_880

    def __call__(self, value: str, *, add_special_tokens: bool) -> dict:
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in value]}

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(chr(value) for value in values)


def _token_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    database = tmp_path / "source.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL,
            model TEXT NOT NULL, model_revision TEXT, harness_git TEXT,
            fork_git TEXT, env_json TEXT, pod TEXT, note TEXT
        );
        CREATE TABLE items (
            id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL,
            benchmark TEXT NOT NULL, item_id TEXT NOT NULL,
            asked_utc TEXT NOT NULL, prompt TEXT NOT NULL, gold TEXT,
            raw_output TEXT, extracted TEXT, correct INTEGER, score REAL,
            n_prompt_tokens INTEGER, n_gen_tokens INTEGER, latency_ms REAL,
            seed INTEGER, finish_reason TEXT, truncated INTEGER
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
        "(2,1,'passkey','t0','now','abc','xy',' xy','xy',"
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
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _Tokenizer(),
    )
    return capture_short_context_oracle(
        ShortContextOracleConfig(
            results_db=database,
            tokenizer_root=tokenizer_root,
            output_dir=tmp_path / "token_oracle",
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
            expected_gold="xy",
        )
    )


def test_normalize_sample_logprobs_orders_by_rank_and_refuses_width() -> None:
    positions = [
        {
            5: _Logprob(-3.0, 3),
            3: _Logprob(-1.0, 1),
            4: _Logprob(-2.0, 2),
        }
    ]
    ids, scores, ranks = normalize_sample_logprobs(positions, top_k=3)
    np.testing.assert_array_equal(ids, [[3, 4, 5]])
    np.testing.assert_array_equal(ranks, [[1, 2, 3]])
    np.testing.assert_allclose(scores, [[-1.0, -2.0, -3.0]])
    with pytest.raises(ValueError, match="width"):
        normalize_sample_logprobs(positions, top_k=4)


def test_short_context_logprob_oracle_roundtrip_and_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token_manifest = _token_oracle(tmp_path, monkeypatch)
    generated = np.asarray([32, 120, 121], dtype=np.int32)
    candidates = np.asarray(
        [[32, 1, 2, 3], [120, 4, 5, 6], [121, 7, 8, 9]], dtype=np.int32
    )
    scores = np.asarray(
        [[-0.1, -0.2, -0.3, -0.4]] * 3, dtype=np.float32
    )
    ranks = np.broadcast_to(np.arange(1, 5, dtype=np.int32), (3, 4)).copy()
    config = ShortContextLogprobOracleConfig(
        token_oracle_dir=tmp_path / "token_oracle",
        output_dir=tmp_path / "logprob_oracle",
        capture_code_hash="c" * 40,
        legacy_repository_pin="d" * 40,
        token_oracle_manifest_sha256=token_manifest["manifest_sha256"],
        source_run_id=3,
        source_item_row_id=4,
        source_harness_git="abcdef0",
        source_fork_git="1234567",
        source_launcher_git="e" * 40,
        top_k=4,
        step_count=3,
    )
    manifest = capture_short_context_logprob_oracle(
        config,
        generated_token_ids=generated,
        candidate_token_ids=candidates,
        candidate_logprobs=scores,
        candidate_ranks=ranks,
    )
    assert inspect_short_context_logprob_oracle(config.output_dir) == manifest
    assert manifest["top_k"] == 4
    assert "not raw logits" in manifest["diagnostic_semantics"]
    with pytest.raises(FileExistsError, match="append-only"):
        capture_short_context_logprob_oracle(
            config,
            generated_token_ids=generated,
            candidate_token_ids=candidates,
            candidate_logprobs=scores,
            candidate_ranks=ranks,
        )

    bad_root = tmp_path / "bad"
    bad_config = ShortContextLogprobOracleConfig(
        token_oracle_dir=tmp_path / "token_oracle",
        output_dir=bad_root,
        capture_code_hash="c" * 40,
        legacy_repository_pin="d" * 40,
        token_oracle_manifest_sha256=token_manifest["manifest_sha256"],
        source_run_id=3,
        source_item_row_id=4,
        source_harness_git="abcdef0",
        source_fork_git="1234567",
        source_launcher_git="e" * 40,
        top_k=4,
        step_count=3,
    )
    with pytest.raises(ValueError, match="token prefix"):
        capture_short_context_logprob_oracle(
            bad_config,
            generated_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
            candidate_token_ids=candidates,
            candidate_logprobs=scores,
            candidate_ranks=ranks,
        )
