"""Spec §23.1: legacy long-context workloads rebuilt from provenance, fail-closed."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from glm_tpu.greenfield.validation import (
    LongContextOracleConfig,
    capture_long_context_oracle,
    inspect_long_context_oracle,
)
from glm_tpu.greenfield.validation import long_context_oracle as module

RESULTS_DB = Path("/home/gianl/glm-tpu/bench/results.db")
TOKENIZER_ROOT = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
LEGACY_BENCH = Path("/home/gianl/glm-tpu/bench")
CODE_HASH = "0" * 40
LEGACY_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
MODEL_URI = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"


def _available() -> bool:
    return (
        RESULTS_DB.is_file()
        and all((TOKENIZER_ROOT / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"))
        and (LEGACY_BENCH / "glm_longctx.py").is_file()
    )


def _row(run_id: int, benchmark: str) -> dict:
    connection = sqlite3.connect(f"file:{RESULTS_DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT i.id, i.seed, i.gold, i.n_prompt_tokens, i.n_gen_tokens, r.harness_git, r.fork_git "
            "FROM items i JOIN runs r ON r.run_id = i.run_id WHERE i.run_id = ? AND i.benchmark = ?",
            (run_id, benchmark),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return dict(row)


def _passkey_config(tmp_path: Path, depth: float, **overrides) -> LongContextOracleConfig:
    row = _row(403, f"passkey_L128000_d{depth}")
    values = dict(
        kind="passkey",
        results_db=RESULTS_DB,
        tokenizer_root=TOKENIZER_ROOT,
        legacy_bench_root=LEGACY_BENCH,
        output_dir=tmp_path / "oracle",
        capture_code_hash=CODE_HASH,
        legacy_repository_pin=LEGACY_PIN,
        run_id=403,
        item_row_id=int(row["id"]),
        expected_harness_git=row["harness_git"],
        expected_fork_git=row["fork_git"],
        expected_benchmark=f"passkey_L128000_d{depth}",
        expected_model_uri=MODEL_URI,
        expected_prompt_tokens=int(row["n_prompt_tokens"]),
        expected_generated_tokens=int(row["n_gen_tokens"]),
        expected_seed=int(row["seed"]),
        expected_context_length=128000,
        expected_depth=depth,
        expected_gold=row["gold"],
    )
    values.update(overrides)
    return LongContextOracleConfig(**values)


def test_long_context_oracle_config_refuses_inconsistent_identity(tmp_path: Path) -> None:
    base = dict(
        kind="passkey", results_db=tmp_path / "db", tokenizer_root=tmp_path, legacy_bench_root=tmp_path,
        output_dir=tmp_path / "o", capture_code_hash=CODE_HASH, legacy_repository_pin=LEGACY_PIN,
        run_id=403, item_row_id=1, expected_harness_git="a4a17ac", expected_fork_git="b3c25df47",
        expected_benchmark="passkey_L128000_d0.0", expected_model_uri=MODEL_URI,
        expected_prompt_tokens=127363, expected_generated_tokens=20, expected_seed=1,
        expected_context_length=128000, expected_depth=0.0, expected_gold="705269",
    )
    LongContextOracleConfig(**base)
    with pytest.raises(ValueError, match="length/depth"):
        LongContextOracleConfig(**{**base, "expected_depth": 0.05})
    with pytest.raises(ValueError, match="digit-string gold"):
        LongContextOracleConfig(**{**base, "expected_gold": "x"})
    with pytest.raises(ValueError, match="approved bucket"):
        LongContextOracleConfig(**{**base, "expected_model_uri": "gs://other/m"})
    e0 = {**base, "kind": "e0", "expected_benchmark": "dsa_throughput_ctx262144", "expected_prompt_tokens": 262144,
          "expected_context_length": 262144, "expected_depth": None, "expected_gold": None, "expected_generated_tokens": 256}
    LongContextOracleConfig(**e0)
    with pytest.raises(ValueError, match="no depth or gold"):
        LongContextOracleConfig(**{**e0, "expected_gold": "1"})
    with pytest.raises(ValueError, match="equal the context length"):
        LongContextOracleConfig(**{**e0, "expected_prompt_tokens": 262143})


@pytest.mark.skipif(not _available(), reason="legacy provenance DB, tokenizer or bench unavailable")
def test_e0_oracle_rebuilds_the_legacy_256k_prompt_bit_exactly(tmp_path: Path) -> None:
    row = _row(402, "dsa_throughput_ctx262144")
    config = LongContextOracleConfig(
        kind="e0", results_db=RESULTS_DB, tokenizer_root=TOKENIZER_ROOT, legacy_bench_root=LEGACY_BENCH,
        output_dir=tmp_path / "oracle", capture_code_hash=CODE_HASH, legacy_repository_pin=LEGACY_PIN,
        run_id=402, item_row_id=int(row["id"]), expected_harness_git=row["harness_git"],
        expected_fork_git=row["fork_git"], expected_benchmark="dsa_throughput_ctx262144",
        expected_model_uri=MODEL_URI, expected_prompt_tokens=262144, expected_generated_tokens=256,
        expected_seed=int(row["seed"]), expected_context_length=262144,
    )
    manifest = capture_long_context_oracle(config)
    assert manifest["kind"] == "e0" and manifest["prompt_token_count"] == 262144
    assert manifest["generated_token_count"] == 256
    assert manifest["prompt_prefix_ids"] == [154822, 154824]
    inspected = inspect_long_context_oracle(config.output_dir)
    assert inspected["manifest_sha256"] == manifest["manifest_sha256"]
    rebuild = json.loads((config.output_dir / "rebuild.json").read_text())
    assert rebuild["seed"] == int(row["seed"]) and rebuild["sample_low"] == 256
    with pytest.raises(FileExistsError):
        capture_long_context_oracle(config)


@pytest.mark.skipif(not _available(), reason="legacy provenance DB, tokenizer or bench unavailable")
def test_passkey_oracle_rebuilds_the_legacy_128k_prompt_and_refuses_drift(tmp_path: Path) -> None:
    config = _passkey_config(tmp_path, 1.0)
    manifest = capture_long_context_oracle(config)
    assert manifest["prompt_token_count"] == 127363 and manifest["generated_token_count"] == 20
    assert manifest["source"]["gold"] == "891482"
    inspected = inspect_long_context_oracle(config.output_dir)
    assert inspected["rebuild"]["stored_prompt_field_reproduced"] is True
    assert "not verified" in inspected["rebuild"]["prompt_ids_provenance"]
    assert "diagnostic" in inspected["rebuild"]["generated_ids_provenance"]
    assert inspected["legacy_bench_git"]["harness_git"] == "a4a17ac"
    assert len(inspected["legacy_bench_git"]["repository_head"]) == 40
    text = (config.output_dir / "prompt.txt").read_text(encoding="utf-8")
    assert text.endswith("What is the secret passcode? The secret passcode is")
    assert "The secret passcode is 891482." in text
    # Tampering with any sealed file is refused by inspection.
    (config.output_dir / "prompt.txt").write_text(text[:-1], encoding="utf-8")
    with pytest.raises(ValueError, match="file drifted"):
        inspect_long_context_oracle(config.output_dir)
    # A wrong gold or seed is refused before anything is written.
    for overrides, message in (
        ({"expected_gold": "891483"}, "identity drifted"),
        ({"expected_seed": 16797346}, "identity drifted"),
    ):
        wrong = _passkey_config(tmp_path / "wrong", 1.0, **overrides)
        with pytest.raises(ValueError, match=message):
            capture_long_context_oracle(wrong)
        assert not wrong.output_dir.exists()


def test_legacy_bench_loader_refuses_model_execution_imports(tmp_path: Path) -> None:
    """The bench prompt builders are utilities; loading them must fail closed if
    they pull a model-execution package across the greenfield boundary."""
    import sys

    forbidden = "tpu_inference"
    if forbidden in sys.modules:
        pytest.skip("forbidden package already loaded in this interpreter")
    bench = tmp_path / "bench"
    bench.mkdir()
    (tmp_path / f"{forbidden}.py").write_text("VALUE = 1\n", encoding="utf-8")
    (bench / "glm_longctx.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        f"import {forbidden}\n"
        "PROMPT_PREFIX_IDS = [154822, 154824]\n",
        encoding="utf-8",
    )
    (bench / "dsa_throughput.py").write_text("PROMPT_PREFIX_IDS = [154822, 154824]\n", encoding="utf-8")
    for name in ("extract.py", "provenance.py", "engine.py"):
        (bench / name).write_text("", encoding="utf-8")
    try:
        with pytest.raises(ImportError, match="model-execution boundary"):
            module._load_legacy_bench(bench)
    finally:
        sys.modules.pop(forbidden, None)
        for name in list(sys.modules):
            if name.startswith("greenfield_legacy_bench_"):
                sys.modules.pop(name, None)

