"""Seal compact legacy sample-logprob evidence for the 2K token oracle.

This module never imports the legacy model or vLLM.  The model-facing capture
script converts vLLM's response objects into primitive arrays before calling
this validator.  Log probabilities are used only as a diagnostic: subtracting
two values from the same row preserves their raw-logit margin, but the values
are not mislabeled as absolute logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .short_context_oracle import inspect_short_context_oracle


FORMAT_VERSION = 1
ARTIFACT_KIND = "greenfield_short_context_legacy_logprobs"
MODEL_VOCAB_SIZE = 154_880


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


def _array_hash(value: np.ndarray, dtype: str) -> str:
    return sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


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


def normalize_sample_logprobs(
    positions: Sequence[Mapping[int, object]],
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert vLLM-like per-position mappings to rank-ordered arrays.

    Values need only expose ``logprob`` and ``rank`` attributes.  Keeping this
    boundary structural makes the validator CPU-testable without importing
    vLLM or either execution engine.
    """

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 1:
        raise ValueError("top_k must be an integer greater than one")
    ids = np.empty((len(positions), top_k), dtype=np.int32)
    scores = np.empty((len(positions), top_k), dtype=np.float32)
    ranks = np.empty((len(positions), top_k), dtype=np.int32)
    for row_index, position in enumerate(positions):
        if len(position) != top_k:
            raise ValueError(
                f"logprob row {row_index} width differs from top_k: "
                f"{len(position)} != {top_k}"
            )
        rows: list[tuple[int, float, int]] = []
        for token_id, value in position.items():
            rank = getattr(value, "rank", None)
            logprob = getattr(value, "logprob", None)
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("logprob token IDs must be integers")
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise ValueError("every logprob candidate must have an integer rank")
            try:
                score = float(logprob)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "every candidate must have a numeric logprob"
                ) from error
            rows.append((token_id, score, rank))
        rows.sort(key=lambda row: row[2])
        ids[row_index] = [row[0] for row in rows]
        scores[row_index] = [row[1] for row in rows]
        ranks[row_index] = [row[2] for row in rows]
    return ids, scores, ranks


@dataclass(frozen=True, slots=True)
class ShortContextLogprobOracleConfig:
    token_oracle_dir: Path
    output_dir: Path
    capture_code_hash: str
    legacy_repository_pin: str
    token_oracle_manifest_sha256: str
    source_run_id: int
    source_item_row_id: int
    source_harness_git: str
    source_fork_git: str
    source_launcher_git: str
    top_k: int = 16
    step_count: int = 15

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_oracle_dir", Path(self.token_oracle_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in ("source_run_id", "source_item_row_id", "top_k", "step_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k <= 1:
            raise ValueError("top_k must be greater than one")
        for name in (
            "capture_code_hash",
            "legacy_repository_pin",
            "token_oracle_manifest_sha256",
        ):
            _validate_digest(getattr(self, name), name, (40, 64))
        for name in (
            "source_harness_git",
            "source_fork_git",
            "source_launcher_git",
        ):
            _validate_digest(getattr(self, name), name, tuple(range(7, 65)))


def _validate_arrays(
    *,
    expected_tokens: np.ndarray,
    generated_token_ids: np.ndarray,
    candidate_token_ids: np.ndarray,
    candidate_logprobs: np.ndarray,
    candidate_ranks: np.ndarray,
    top_k: int,
    step_count: int,
) -> None:
    expected_shape = (step_count, top_k)
    if generated_token_ids.shape != (step_count,):
        raise ValueError("generated token vector shape drifted")
    for name, value in (
        ("candidate_token_ids", candidate_token_ids),
        ("candidate_logprobs", candidate_logprobs),
        ("candidate_ranks", candidate_ranks),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape drifted: {value.shape} != {expected_shape}")
    if not np.array_equal(generated_token_ids, expected_tokens[:step_count]):
        raise ValueError("legacy diagnostic token prefix differs from sealed oracle")
    if np.any(candidate_token_ids < 0) or np.any(
        candidate_token_ids >= MODEL_VOCAB_SIZE
    ):
        raise ValueError("candidate token ID is outside the model vocabulary")
    if any(np.unique(row).size != top_k for row in candidate_token_ids):
        raise ValueError("candidate token IDs must be unique within each row")
    expected_ranks = np.broadcast_to(
        np.arange(1, top_k + 1, dtype=np.int32), expected_shape
    )
    if not np.array_equal(candidate_ranks, expected_ranks):
        raise ValueError("candidate ranks must be exactly 1..top_k")
    if not np.all(np.isfinite(candidate_logprobs)):
        raise ValueError("candidate logprobs must be finite")
    if np.any(candidate_logprobs[:, 1:] > candidate_logprobs[:, :-1]):
        raise ValueError("candidate logprobs must be non-increasing by rank")
    if not np.array_equal(candidate_token_ids[:, 0], generated_token_ids):
        raise ValueError("greedy generated token must be the rank-one candidate")


def capture_short_context_logprob_oracle(
    config: ShortContextLogprobOracleConfig,
    *,
    generated_token_ids: np.ndarray,
    candidate_token_ids: np.ndarray,
    candidate_logprobs: np.ndarray,
    candidate_ranks: np.ndarray,
) -> dict[str, Any]:
    """Validate and append-only seal one compact legacy logprob capture."""

    if config.output_dir.exists():
        raise FileExistsError(
            f"append-only short-context logprob oracle exists: {config.output_dir}"
        )
    token_manifest = inspect_short_context_oracle(config.token_oracle_dir)
    if token_manifest["manifest_sha256"] != config.token_oracle_manifest_sha256:
        raise ValueError("sealed token oracle manifest differs from the pinned hash")

    from safetensors.numpy import load_file, save_file

    token_arrays = load_file(
        str(config.token_oracle_dir / token_manifest["files"]["tokens"]["filename"])
    )
    expected_tokens = np.asarray(token_arrays["generated_token_ids"], dtype=np.int32)
    generated_token_ids = np.asarray(generated_token_ids, dtype=np.int32)
    candidate_token_ids = np.asarray(candidate_token_ids, dtype=np.int32)
    candidate_logprobs = np.asarray(candidate_logprobs, dtype=np.float32)
    candidate_ranks = np.asarray(candidate_ranks, dtype=np.int32)
    if config.step_count > expected_tokens.size:
        raise ValueError("logprob step_count exceeds the sealed token oracle")
    _validate_arrays(
        expected_tokens=expected_tokens,
        generated_token_ids=generated_token_ids,
        candidate_token_ids=candidate_token_ids,
        candidate_logprobs=candidate_logprobs,
        candidate_ranks=candidate_ranks,
        top_k=config.top_k,
        step_count=config.step_count,
    )

    prompt_token_count = int(token_manifest["prompt_token_count"])
    decode_positions = np.arange(
        prompt_token_count - 1,
        prompt_token_count - 1 + config.step_count,
        dtype=np.int32,
    )
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "logprobs.safetensors"
    save_file(
        {
            "candidate_logprobs": candidate_logprobs,
            "candidate_ranks": candidate_ranks,
            "candidate_token_ids": candidate_token_ids,
            "decode_positions": decode_positions,
            "generated_token_ids": generated_token_ids,
        },
        str(tensor_path),
    )
    arrays = {
        "candidate_logprobs_sha256": _array_hash(candidate_logprobs, "<f4"),
        "candidate_ranks_sha256": _array_hash(candidate_ranks, "<i4"),
        "candidate_token_ids_sha256": _array_hash(candidate_token_ids, "<i4"),
        "decode_positions_sha256": _array_hash(decode_positions, "<i4"),
        "generated_token_ids_sha256": _array_hash(generated_token_ids, "<i4"),
    }
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "arrays": arrays,
        "capture_code_hash": config.capture_code_hash,
        "diagnostic_semantics": (
            "FP32 sample log-probabilities; within-row differences preserve "
            "raw-logit margins but absolute values are not raw logits"
        ),
        "decode_position_end_inclusive": int(decode_positions[-1]),
        "decode_position_start": int(decode_positions[0]),
        "files": {"tensors": _file_record(tensor_path)},
        "format_version": FORMAT_VERSION,
        "legacy_repository_pin": config.legacy_repository_pin,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "source": {
            "fork_git": config.source_fork_git,
            "harness_git": config.source_harness_git,
            "item_row_id": config.source_item_row_id,
            "launcher_git": config.source_launcher_git,
            "run_id": config.source_run_id,
        },
        "step_count": config.step_count,
        "token_oracle": {
            "generated_token_ids_sha256": token_manifest[
                "generated_token_ids_sha256"
            ],
            "manifest_sha256": token_manifest["manifest_sha256"],
            "prompt_token_ids_sha256": token_manifest["prompt_token_ids_sha256"],
        },
        "top_k": config.top_k,
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inspect_short_context_logprob_oracle(config.output_dir)


def inspect_short_context_logprob_oracle(output_dir: Path) -> dict[str, Any]:
    """Fail closed while inspecting a sealed legacy logprob artifact."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("short-context logprob artifact kind drifted")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("short-context logprob format version drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("short-context logprob manifest hash mismatch")
    tensor_record = manifest.get("files", {}).get("tensors", {})
    tensor_path = output_dir / str(tensor_record.get("filename", ""))
    if not tensor_path.is_file():
        raise FileNotFoundError(tensor_path)
    if tensor_path.stat().st_size != tensor_record.get("byte_count"):
        raise ValueError("short-context logprob tensor size mismatch")
    if _sha256_file(tensor_path) != tensor_record.get("sha256"):
        raise ValueError("short-context logprob tensor hash mismatch")

    from safetensors.numpy import load_file

    tensors = load_file(str(tensor_path))
    expected_keys = {
        "candidate_logprobs",
        "candidate_ranks",
        "candidate_token_ids",
        "decode_positions",
        "generated_token_ids",
    }
    if set(tensors) != expected_keys:
        raise ValueError("short-context logprob tensor keys drifted")
    _validate_arrays(
        expected_tokens=np.asarray(tensors["generated_token_ids"], dtype=np.int32),
        generated_token_ids=np.asarray(tensors["generated_token_ids"], dtype=np.int32),
        candidate_token_ids=np.asarray(tensors["candidate_token_ids"], dtype=np.int32),
        candidate_logprobs=np.asarray(tensors["candidate_logprobs"], dtype=np.float32),
        candidate_ranks=np.asarray(tensors["candidate_ranks"], dtype=np.int32),
        top_k=int(manifest["top_k"]),
        step_count=int(manifest["step_count"]),
    )
    positions = np.asarray(tensors["decode_positions"], dtype=np.int32)
    if positions.shape != (int(manifest["step_count"]),) or np.any(
        np.diff(positions) != 1
    ):
        raise ValueError("decode positions are not one contiguous sequence")
    if int(positions[0]) != manifest.get("decode_position_start") or int(
        positions[-1]
    ) != manifest.get("decode_position_end_inclusive"):
        raise ValueError("decode position bounds differ from the manifest")
    expected_hashes = {
        "candidate_logprobs_sha256": _array_hash(
            tensors["candidate_logprobs"], "<f4"
        ),
        "candidate_ranks_sha256": _array_hash(tensors["candidate_ranks"], "<i4"),
        "candidate_token_ids_sha256": _array_hash(
            tensors["candidate_token_ids"], "<i4"
        ),
        "decode_positions_sha256": _array_hash(tensors["decode_positions"], "<i4"),
        "generated_token_ids_sha256": _array_hash(
            tensors["generated_token_ids"], "<i4"
        ),
    }
    if manifest.get("arrays") != expected_hashes:
        raise ValueError("short-context logprob array hash mismatch")
    return manifest
