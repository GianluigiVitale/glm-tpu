"""Seal one accepted legacy short-context result as a token oracle.

The capture reads only an append-only provenance database and tokenizer files.
It never imports either the legacy or greenfield model execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import numpy as np


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_short_context_legacy_oracle"
MODEL_ID = "zai-org/GLM-5.2-FP8"
MODEL_VOCAB_SIZE = 154_880
PROMPT_PREFIX_IDS = (154_822, 154_824)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _manifest_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.asarray(value, dtype="<i4").tobytes()).hexdigest()


def _validate_digest(value: str, name: str, lengths: tuple[int, ...]) -> None:
    if len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "byte_count": path.stat().st_size,
        "filename": path.name,
        "sha256": _sha256_file(path),
    }


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only oracle file exists: {path}")
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


@dataclass(frozen=True, slots=True)
class ShortContextOracleConfig:
    results_db: Path
    tokenizer_root: Path
    output_dir: Path
    capture_code_hash: str
    legacy_repository_pin: str
    run_id: int
    item_row_id: int
    expected_harness_git: str
    expected_fork_git: str
    expected_benchmark: str
    expected_model_uri: str
    expected_prompt_tokens: int
    expected_generated_tokens: int
    expected_seed: int
    expected_gold: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_db", Path(self.results_db))
        object.__setattr__(self, "tokenizer_root", Path(self.tokenizer_root))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "run_id",
            "item_row_id",
            "expected_prompt_tokens",
            "expected_generated_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.expected_seed, int) or isinstance(
            self.expected_seed, bool
        ):
            raise ValueError("expected_seed must be an integer")
        for name in (
            "expected_harness_git",
            "expected_fork_git",
            "expected_benchmark",
            "expected_model_uri",
            "expected_gold",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        _validate_digest(self.capture_code_hash, "capture_code_hash", (40, 64))
        _validate_digest(
            self.legacy_repository_pin, "legacy_repository_pin", (40, 64)
        )
        _validate_digest(
            self.expected_harness_git,
            "expected_harness_git",
            tuple(range(7, 65)),
        )
        _validate_digest(
            self.expected_fork_git,
            "expected_fork_git",
            tuple(range(7, 65)),
        )
        if not self.expected_model_uri.startswith(
            "gs://driftbench-dsv4-uc/"
        ):
            raise ValueError("expected_model_uri must use the approved bucket")


def _read_source_row(config: ShortContextOracleConfig) -> dict[str, Any]:
    connection = sqlite3.connect(
        f"file:{config.results_db}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        integrity = [row[0] for row in connection.execute("pragma integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"legacy results DB integrity failed: {integrity}")
        row = connection.execute(
            """
            SELECT
                i.id AS item_row_id,
                i.run_id,
                i.benchmark,
                i.item_id,
                i.asked_utc,
                i.prompt,
                i.gold,
                i.raw_output,
                i.extracted,
                i.correct,
                i.score,
                i.n_prompt_tokens,
                i.n_gen_tokens,
                i.latency_ms,
                i.seed,
                i.finish_reason,
                i.truncated,
                r.created_utc AS run_created_utc,
                r.model,
                r.model_revision,
                r.harness_git,
                r.fork_git,
                r.env_json,
                r.pod,
                r.note
            FROM items AS i
            JOIN runs AS r ON r.run_id = i.run_id
            WHERE i.run_id = ? AND i.id = ?
            """,
            (config.run_id, config.item_row_id),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("legacy short-context oracle row is missing")
    source = dict(row)
    expected = {
        "benchmark": config.expected_benchmark,
        "correct": 1,
        "fork_git": config.expected_fork_git,
        "gold": config.expected_gold,
        "harness_git": config.expected_harness_git,
        "item_row_id": config.item_row_id,
        "model": config.expected_model_uri,
        "n_gen_tokens": config.expected_generated_tokens,
        "n_prompt_tokens": config.expected_prompt_tokens,
        "run_id": config.run_id,
        "seed": config.expected_seed,
    }
    mismatches = {
        name: {"expected": value, "observed": source.get(name)}
        for name, value in expected.items()
        if source.get(name) != value
    }
    if mismatches:
        raise ValueError(f"legacy short-context row identity drifted: {mismatches}")
    if not source.get("prompt") or not source.get("raw_output"):
        raise ValueError("legacy short-context row lacks prompt or raw output")
    environment = json.loads(source["env_json"])
    environment_expected = {
        "model": config.expected_model_uri,
        "prompt_prefix_ids": list(PROMPT_PREFIX_IDS),
        "protocol": "raw",
        "temperature": 0.0,
    }
    environment_mismatches = {
        name: {"expected": value, "observed": environment.get(name)}
        for name, value in environment_expected.items()
        if environment.get(name) != value
    }
    if environment_mismatches:
        raise ValueError(
            "legacy short-context generation protocol drifted: "
            f"{environment_mismatches}"
        )
    source["env_json"] = environment
    return source


def capture_short_context_oracle(
    config: ShortContextOracleConfig,
) -> dict[str, Any]:
    """Capture exact prompt/output token IDs from one accepted legacy row."""

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only short-context oracle exists: {config.output_dir}"
        )
    if not config.results_db.is_file():
        raise FileNotFoundError(config.results_db)
    required_tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    for filename in required_tokenizer_files:
        if not (config.tokenizer_root / filename).is_file():
            raise FileNotFoundError(config.tokenizer_root / filename)

    from safetensors.numpy import save_file
    from transformers import AutoTokenizer

    source = _read_source_row(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_root,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt_ids = np.asarray(
        [
            *PROMPT_PREFIX_IDS,
            *tokenizer(
                source["prompt"], add_special_tokens=False
            )["input_ids"],
        ],
        dtype=np.int32,
    )
    generated_ids = np.asarray(
        tokenizer(
            source["raw_output"], add_special_tokens=False
        )["input_ids"],
        dtype=np.int32,
    )
    if prompt_ids.shape != (config.expected_prompt_tokens,):
        raise ValueError("tokenized legacy prompt length drifted")
    if generated_ids.shape != (config.expected_generated_tokens,):
        raise ValueError("tokenized legacy output length drifted")
    if not np.array_equal(prompt_ids[:2], np.asarray(PROMPT_PREFIX_IDS)):
        raise ValueError("legacy prompt prefix drifted")
    for name, value in (
        ("prompt", prompt_ids),
        ("generated", generated_ids),
    ):
        if np.any(value < 0) or np.any(value >= MODEL_VOCAB_SIZE):
            raise ValueError(f"legacy {name} token IDs escaped model vocabulary")
    roundtrip = tokenizer.decode(
        generated_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if roundtrip != source["raw_output"]:
        raise ValueError("legacy output token roundtrip is not exact")

    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "tokens.safetensors"
    partial = tensor_path.with_suffix(".safetensors.partial")
    save_file(
        {
            "generated_token_ids": generated_ids,
            "prompt_token_ids": prompt_ids,
        },
        partial,
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "model_id": MODEL_ID,
        },
    )
    partial.replace(tensor_path)
    prompt_path = config.output_dir / "prompt.txt"
    output_path = config.output_dir / "raw_output.txt"
    source_path = config.output_dir / "source_row.json"
    _write_text_once(prompt_path, source["prompt"])
    _write_text_once(output_path, source["raw_output"])
    _write_text_once(
        source_path,
        json.dumps(source, indent=2, sort_keys=True) + "\n",
    )
    source_row_sha256 = sha256(
        _canonical_json(source).encode("utf-8")
    ).hexdigest()
    tokenizer_files = {
        filename: _file_record(config.tokenizer_root / filename)
        for filename in required_tokenizer_files
    }
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "capture_code_hash": config.capture_code_hash,
        "files": {
            "prompt": _file_record(prompt_path),
            "raw_output": _file_record(output_path),
            "source_row": _file_record(source_path),
            "tokens": _file_record(tensor_path),
        },
        "format_version": FORMAT_VERSION,
        "generated_token_count": int(generated_ids.size),
        "generated_token_ids_sha256": _array_sha256(generated_ids),
        "legacy_repository_pin_at_capture": config.legacy_repository_pin,
        "model_id": MODEL_ID,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "prompt_prefix_ids": list(PROMPT_PREFIX_IDS),
        "prompt_token_count": int(prompt_ids.size),
        "prompt_token_ids_sha256": _array_sha256(prompt_ids),
        "source": {
            "benchmark": source["benchmark"],
            "fork_git": source["fork_git"],
            "gold": source["gold"],
            "harness_git": source["harness_git"],
            "item_id": source["item_id"],
            "item_row_id": source["item_row_id"],
            "model_uri": source["model"],
            "run_id": source["run_id"],
            "seed": source["seed"],
            "source_row_sha256": source_row_sha256,
        },
        "tokenizer_files": tokenizer_files,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _write_text_once(
        config.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    inspect_short_context_oracle(config.output_dir)
    return manifest


def inspect_short_context_oracle(output_dir: Path) -> dict[str, Any]:
    """Verify every token-oracle identity without opening the source DB."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("not a greenfield short-context legacy oracle")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported short-context oracle format")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("short-context oracle manifest checksum mismatch")
    if manifest.get("model_id") != MODEL_ID or (
        manifest.get("model_vocab_size") != MODEL_VOCAB_SIZE
    ):
        raise ValueError("short-context oracle model identity drifted")
    if manifest.get("prompt_prefix_ids") != list(PROMPT_PREFIX_IDS):
        raise ValueError("short-context oracle prompt-prefix contract drifted")
    _validate_digest(
        manifest.get("capture_code_hash", ""),
        "capture_code_hash",
        (40, 64),
    )
    _validate_digest(
        manifest.get("legacy_repository_pin_at_capture", ""),
        "legacy_repository_pin_at_capture",
        (40, 64),
    )
    expected_files = {
        "prompt": "prompt.txt",
        "raw_output": "raw_output.txt",
        "source_row": "source_row.json",
        "tokens": "tokens.safetensors",
    }
    if {
        role: record.get("filename")
        for role, record in manifest.get("files", {}).items()
    } != expected_files:
        raise ValueError("short-context oracle file identities drifted")
    for record in manifest["files"].values():
        path = output_dir / record["filename"]
        if path.stat().st_size != record["byte_count"]:
            raise ValueError("short-context oracle file size mismatch")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError("short-context oracle file checksum mismatch")
    source = json.loads(
        (output_dir / manifest["files"]["source_row"]["filename"]).read_text()
    )
    if sha256(_canonical_json(source).encode("utf-8")).hexdigest() != (
        manifest["source"]["source_row_sha256"]
    ):
        raise ValueError("short-context oracle source-row checksum mismatch")
    expected_source = {
        "benchmark": source["benchmark"],
        "fork_git": source["fork_git"],
        "gold": source["gold"],
        "harness_git": source["harness_git"],
        "item_id": source["item_id"],
        "item_row_id": source["item_row_id"],
        "model_uri": source["model"],
        "run_id": source["run_id"],
        "seed": source["seed"],
        "source_row_sha256": manifest["source"]["source_row_sha256"],
    }
    if manifest["source"] != expected_source:
        raise ValueError("short-context oracle source identity drifted")
    if source.get("correct") != 1 or (
        source.get("n_prompt_tokens") != manifest["prompt_token_count"]
    ) or source.get("n_gen_tokens") != manifest["generated_token_count"]:
        raise ValueError("short-context oracle accepted-result identity drifted")
    environment = source.get("env_json")
    if not isinstance(environment, dict) or {
        "model": environment.get("model"),
        "prompt_prefix_ids": environment.get("prompt_prefix_ids"),
        "protocol": environment.get("protocol"),
        "temperature": environment.get("temperature"),
    } != {
        "model": source["model"],
        "prompt_prefix_ids": list(PROMPT_PREFIX_IDS),
        "protocol": "raw",
        "temperature": 0.0,
    }:
        raise ValueError("short-context oracle generation protocol drifted")
    tokenizer_files = manifest.get("tokenizer_files")
    required_tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    if not isinstance(tokenizer_files, dict) or (
        set(tokenizer_files) != required_tokenizer_files
    ):
        raise ValueError("short-context oracle tokenizer identities drifted")
    for filename, record in tokenizer_files.items():
        if record.get("filename") != filename or not isinstance(
            record.get("byte_count"), int
        ) or record["byte_count"] <= 0:
            raise ValueError("short-context oracle tokenizer record drifted")
        _validate_digest(record.get("sha256", ""), filename, (64,))
    if (output_dir / manifest["files"]["prompt"]["filename"]).read_text() != (
        source["prompt"]
    ):
        raise ValueError("short-context oracle prompt text drifted")
    if (
        output_dir / manifest["files"]["raw_output"]["filename"]
    ).read_text() != source["raw_output"]:
        raise ValueError("short-context oracle raw output drifted")
    tensor_path = output_dir / manifest["files"]["tokens"]["filename"]
    with safe_open(tensor_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "model_id": MODEL_ID,
        }:
            raise ValueError("short-context oracle tensor metadata drifted")
        if set(handle.keys()) != {"generated_token_ids", "prompt_token_ids"}:
            raise ValueError("short-context oracle tensor keys drifted")
        generated = handle.get_tensor("generated_token_ids")
        prompt = handle.get_tensor("prompt_token_ids")
    if prompt.shape != (manifest["prompt_token_count"],) or (
        generated.shape != (manifest["generated_token_count"],)
    ):
        raise ValueError("short-context oracle token shapes drifted")
    if prompt.dtype != np.int32 or generated.dtype != np.int32:
        raise ValueError("short-context oracle token dtype drifted")
    if _array_sha256(prompt) != manifest["prompt_token_ids_sha256"] or (
        _array_sha256(generated) != manifest["generated_token_ids_sha256"]
    ):
        raise ValueError("short-context oracle token checksum mismatch")
    if prompt[:2].tolist() != manifest["prompt_prefix_ids"]:
        raise ValueError("short-context oracle prompt prefix mismatch")
    if np.any(prompt < 0) or np.any(prompt >= MODEL_VOCAB_SIZE) or (
        np.any(generated < 0) or np.any(generated >= MODEL_VOCAB_SIZE)
    ):
        raise ValueError("short-context oracle contains invalid token IDs")
    return manifest
