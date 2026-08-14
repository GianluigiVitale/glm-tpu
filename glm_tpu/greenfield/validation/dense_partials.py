"""Seal accepted layer-0 pre-reduction partials and compare with DB548."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any

import ml_dtypes
import numpy as np


SOURCE_KIND = "glm52_legacy_dense_partials"
CAPTURE_KIND = "glm52_accepted_dense_partials_capture"
COMPARISON_KIND = "glm52_accepted_greenfield_dense_partials_comparison"
MODEL_ID = "zai-org/GLM-5.2-FP8"
LAYER_NAME = "model.layers.0.self_attn.attn"
POSITION = 8155
OWNERS = 4
VIRTUAL_RANKS = 8
WIDTH = 6144
SHAPE = (OWNERS, VIRTUAL_RANKS, 1, WIDTH)
PROTECTED_PROMPT_SHA256 = (
    "22f1f5075bfc7a93aeef9d7d1edbb9e0a6aa6ebd28286840ac1dac5a7c1b4896"
)
PROTECTED_RAW_OUTPUTS = (
    " 881446. Do not forget it. There and back again. The grass is green",
    " 881446. Do not forget it. The grass is green. The sky is blue",
)


@dataclass(frozen=True, slots=True)
class DensePartialsCaptureConfig:
    source_dump_dir: Path
    db548_dir: Path
    output_dir: Path
    expected_run_tag: str
    expected_legacy_code_hash: str
    expected_oracle_pin: str
    expected_db548_runner_sha256: str
    expected_db548_tensor_sha256: str
    expected_db548_summary_sha256: str
    expected_db548_success_sha256: str
    expected_model_id: str = MODEL_ID
    expected_layer_name: str = LAYER_NAME
    expected_position: int = POSITION
    expected_process_count: int = 8

    def __post_init__(self) -> None:
        if not self.expected_run_tag.strip():
            raise ValueError("expected run tag must be non-empty")
        if self.expected_process_count != 8:
            raise ValueError("accepted dense-partial capture requires eight hosts")


@dataclass(frozen=True, slots=True)
class DensePartialsRollbackConfig:
    results_db: Path
    run_tag: str
    expected_harness_git: str
    expected_fork_git: str
    expected_legacy_pin: str
    expected_legacy_short: str
    expected_oracle_pin: str
    expected_remote_prefix: str
    expected_dump_prefix: str
    expected_internal_dump_prefix: str
    expected_internal_mode: str = "dense_partial"
    expected_prompt_sha256: str = PROTECTED_PROMPT_SHA256
    expected_raw_outputs: tuple[str, ...] = PROTECTED_RAW_OUTPUTS


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _manifest_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _require_digest(value: str, *, length: int, name: str) -> None:
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a lowercase digest")


def _scalar(payload: Any, name: str) -> Any:
    value = payload[name]
    if value.shape != ():
        raise ValueError(f"dense-partial field {name} is not scalar")
    return value.item()


def _decode(bits: np.ndarray) -> np.ndarray:
    if bits.dtype != np.dtype(np.uint16):
        raise ValueError("dense-partial BF16 storage is not uint16")
    return np.ascontiguousarray(bits).view(ml_dtypes.bfloat16).astype(np.float32)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _load_accepted_partials(
    config: DensePartialsCaptureConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    errors = sorted(config.source_dump_dir.rglob("*.INTERNAL.ERROR.*"))
    if errors:
        raise ValueError(f"dense-partial error sentinel exists: {errors[0]}")
    safe_layer = config.expected_layer_name.replace("/", "_").replace(".", "_")
    paths = sorted(config.source_dump_dir.rglob(
        f"*.{safe_layer}.position{config.expected_position}.proc*.rank*.npz"))
    if len(paths) != OWNERS * VIRTUAL_RANKS:
        raise ValueError("dense-partial capture file count drifted")
    expected_keys = {
        "artifact_kind", "format_version", "capture_mode", "process_index",
        "process_count", "model_rank", "owner", "virtual_rank",
        "layer_name", "position", "source_row", "run_tag", "code_hash",
        "oracle_pin", "model_id", "dense_partial", "dense_partial__dtype",
    }
    bits = np.empty(SHAPE, dtype=np.uint16)
    ranks: set[int] = set()
    process_counts = {value: 0 for value in range(config.expected_process_count)}
    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_keys:
                raise ValueError(f"{path}: dense-partial key set drifted")
            expected_scalars = {
                "artifact_kind": SOURCE_KIND,
                "format_version": 1,
                "capture_mode": "dense_partial",
                "process_count": config.expected_process_count,
                "layer_name": config.expected_layer_name,
                "position": config.expected_position,
                "source_row": 0,
                "run_tag": config.expected_run_tag,
                "code_hash": config.expected_legacy_code_hash,
                "oracle_pin": config.expected_oracle_pin,
                "model_id": config.expected_model_id,
                "dense_partial__dtype": "bfloat16",
            }
            for name, expected in expected_scalars.items():
                observed = _scalar(payload, name)
                if observed != expected:
                    raise ValueError(f"{path}: {name}={observed!r} != {expected!r}")
            process_index = int(_scalar(payload, "process_index"))
            rank = int(_scalar(payload, "model_rank"))
            owner = int(_scalar(payload, "owner"))
            virtual_rank = int(_scalar(payload, "virtual_rank"))
            if not 0 <= process_index < config.expected_process_count:
                raise ValueError(f"{path}: invalid process index")
            if rank in ranks or not 0 <= rank < OWNERS * VIRTUAL_RANKS:
                raise ValueError(f"{path}: invalid/duplicate model rank")
            if (owner, virtual_rank) != divmod(rank, VIRTUAL_RANKS):
                raise ValueError(f"{path}: owner/virtual-rank mapping drifted")
            row = np.ascontiguousarray(payload["dense_partial"])
            if row.shape != (WIDTH,) or row.dtype != np.dtype(np.uint16):
                raise ValueError(f"{path}: dense-partial tensor drifted")
            if not np.isfinite(_decode(row)).all():
                raise ValueError(f"{path}: dense-partial tensor is non-finite")
            bits[owner, virtual_rank, 0] = row
            ranks.add(rank)
            process_counts[process_index] += 1
            records.append({
                "byte_count": path.stat().st_size,
                "model_rank": rank,
                "path": path.relative_to(config.source_dump_dir).as_posix(),
                "process_index": process_index,
                "sha256": _file_sha256(path),
            })
    if ranks != set(range(OWNERS * VIRTUAL_RANKS)):
        raise ValueError("dense-partial rank coverage is incomplete")
    if set(process_counts.values()) != {4}:
        raise ValueError("dense-partial per-host coverage drifted")
    return bits, sorted(records, key=lambda value: value["model_rank"])


def _load_db548(config: DensePartialsCaptureConfig) -> tuple[np.ndarray, dict[str, Any]]:
    filenames = {
        "runner": "runner.json",
        "tensor": "dense_partial_capture.npz",
        "summary": "summary.json",
        "success": "SUCCESS",
    }
    expected_hashes = {
        "runner": config.expected_db548_runner_sha256,
        "tensor": config.expected_db548_tensor_sha256,
        "summary": config.expected_db548_summary_sha256,
        "success": config.expected_db548_success_sha256,
    }
    records: dict[str, Any] = {}
    for name, filename in filenames.items():
        path = config.db548_dir / filename
        if not path.is_file() or _file_sha256(path) != expected_hashes[name]:
            raise ValueError(f"DB548 {name} identity drifted")
        records[name] = {
            "filename": filename,
            "file_sha256": expected_hashes[name],
        }
    runner = json.loads((config.db548_dir / filenames["runner"]).read_text())
    if (
        runner.get("artifact_kind") != "glm52_layer0_dense_partial_capture"
        or runner.get("status") != "SUCCESS"
        or runner.get("capture_partials") is not True
        or runner.get("performance_claim") is not False
    ):
        raise ValueError("DB548 runner contract drifted")
    with np.load(config.db548_dir / filenames["tensor"], allow_pickle=False) as payload:
        key = "dense_virtual_partials_bfloat16_bits"
        if key not in payload.files:
            raise ValueError("DB548 dense-partial tensor key is absent")
        bits = np.ascontiguousarray(payload[key])
    if bits.shape != SHAPE or bits.dtype != np.dtype(np.uint16):
        raise ValueError("DB548 dense-partial tensor geometry drifted")
    if not np.isfinite(_decode(bits)).all():
        raise ValueError("DB548 dense-partial tensor is non-finite")
    return bits, records


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    mismatch = expected != observed
    count = int(np.count_nonzero(mismatch))
    flat = np.flatnonzero(mismatch)
    error = np.abs(_decode(expected) - _decode(observed))
    per_rank = []
    for owner in range(OWNERS):
        for virtual_rank in range(VIRTUAL_RANKS):
            local = mismatch[owner, virtual_rank, 0]
            local_indices = np.flatnonzero(local)
            per_rank.append({
                "first_mismatch_index": (
                    int(local_indices[0]) if local_indices.size else None),
                "mismatch_count": int(np.count_nonzero(local)),
                "model_rank": owner * VIRTUAL_RANKS + virtual_rank,
                "owner": owner,
                "virtual_rank": virtual_rank,
            })
    return {
        "elementwise_exact": count == 0,
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_flat_index": int(flat[0]) if count else None,
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": count,
        "observed_sha256": _array_sha256(observed),
        "per_rank": per_rank,
        "shape": list(SHAPE),
    }


def _validate_config(config: DensePartialsCaptureConfig) -> None:
    _require_digest(config.expected_legacy_code_hash, length=40,
                    name="legacy code hash")
    _require_digest(config.expected_oracle_pin, length=40, name="oracle pin")
    for name, value in (
        ("DB548 runner", config.expected_db548_runner_sha256),
        ("DB548 tensor", config.expected_db548_tensor_sha256),
        ("DB548 summary", config.expected_db548_summary_sha256),
        ("DB548 success", config.expected_db548_success_sha256),
    ):
        _require_digest(value, length=64, name=name)


def _capture_record(
    config: DensePartialsCaptureConfig,
    accepted: np.ndarray,
    db548: np.ndarray,
    process_files: list[dict[str, Any]],
    tensor_path: Path,
) -> dict[str, Any]:
    capture: dict[str, Any] = {
        "artifact_kind": CAPTURE_KIND,
        "capture_layout": "model_rank_owner4_virtual8_live_row",
        "capture_mode": "dense_partial",
        "diagnostic_only": True,
        "format_version": 1,
        "layer_name": config.expected_layer_name,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "model_id": config.expected_model_id,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "process_count": config.expected_process_count,
        "process_files": process_files,
        "run_tag": config.expected_run_tag,
        "tensor": {
            "accepted_tensor_sha256": _array_sha256(accepted),
            "byte_count": tensor_path.stat().st_size,
            "db548_tensor_sha256": _array_sha256(db548),
            "file_sha256": _file_sha256(tensor_path),
            "filename": tensor_path.name,
            "shape": list(SHAPE),
        },
    }
    capture["manifest_sha256"] = _manifest_sha256(capture)
    return capture


def _comparison_record(
    config: DensePartialsCaptureConfig,
    capture: dict[str, Any],
    accepted: np.ndarray,
    db548: np.ndarray,
    db548_records: dict[str, Any],
) -> dict[str, Any]:
    comparison = _comparison(accepted, db548)
    result: dict[str, Any] = {
        "artifact_kind": COMPARISON_KIND,
        "accepted_capture_manifest_sha256": capture["manifest_sha256"],
        "classification": (
            "accepted_dense_partials_exact_db548"
            if comparison["elementwise_exact"]
            else "accepted_dense_partials_nonexact_db548"),
        "comparison": comparison,
        "db548_files": db548_records,
        "diagnostic_only": True,
        "format_version": 1,
        "legacy_code_hash": config.expected_legacy_code_hash,
        "oracle_pin": config.expected_oracle_pin,
        "performance_claim": False,
        "position": config.expected_position,
        "run_tag": config.expected_run_tag,
        "status": "SUCCESS",
    }
    result["manifest_sha256"] = _manifest_sha256(result)
    return result


def validate_dense_partials_artifacts(
    config: DensePartialsCaptureConfig,
) -> dict[str, Any]:
    """Recompute every sealed tensor and record before terminal publication."""
    _validate_config(config)
    expected_files = {"capture.json", "comparison.json", "dense_partials.npz"}
    observed_files = {
        path.name for path in config.output_dir.iterdir() if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError("dense-partial sealed file set drifted")
    accepted, process_files = _load_accepted_partials(config)
    db548, db548_records = _load_db548(config)
    tensor_path = config.output_dir / "dense_partials.npz"
    with np.load(tensor_path, allow_pickle=False) as payload:
        expected_keys = {
            "accepted_dense_partials_bfloat16_bits",
            "db548_dense_partials_bfloat16_bits",
        }
        if set(payload.files) != expected_keys:
            raise ValueError("dense-partial sealed NPZ key set drifted")
        sealed_accepted = np.ascontiguousarray(
            payload["accepted_dense_partials_bfloat16_bits"])
        sealed_db548 = np.ascontiguousarray(
            payload["db548_dense_partials_bfloat16_bits"])
    if (
        sealed_accepted.shape != SHAPE
        or sealed_db548.shape != SHAPE
        or sealed_accepted.dtype != np.dtype(np.uint16)
        or sealed_db548.dtype != np.dtype(np.uint16)
        or not np.array_equal(sealed_accepted, accepted)
        or not np.array_equal(sealed_db548, db548)
    ):
        raise ValueError("dense-partial sealed NPZ tensors drifted")
    expected_capture = _capture_record(
        config, accepted, db548, process_files, tensor_path)
    capture = json.loads((config.output_dir / "capture.json").read_text())
    if capture != expected_capture:
        raise ValueError("dense-partial capture record drifted")
    expected_comparison = _comparison_record(
        config, capture, accepted, db548, db548_records)
    comparison = json.loads(
        (config.output_dir / "comparison.json").read_text())
    if comparison != expected_comparison:
        raise ValueError("dense-partial comparison record drifted")
    return comparison


def capture_and_compare_dense_partials(
    config: DensePartialsCaptureConfig,
) -> dict[str, Any]:
    """Seal accepted partials and directly classify the DB548 boundary."""
    _validate_config(config)
    if config.output_dir.exists():
        raise FileExistsError(f"append-only capture exists: {config.output_dir}")
    accepted, process_files = _load_accepted_partials(config)
    db548, db548_records = _load_db548(config)
    config.output_dir.mkdir(parents=True)
    tensor_path = config.output_dir / "dense_partials.npz"
    _atomic_npz(
        tensor_path,
        accepted_dense_partials_bfloat16_bits=accepted,
        db548_dense_partials_bfloat16_bits=db548,
    )
    capture = _capture_record(
        config, accepted, db548, process_files, tensor_path)
    _atomic_json(config.output_dir / "capture.json", capture)
    result = _comparison_record(
        config, capture, accepted, db548, db548_records)
    _atomic_json(config.output_dir / "comparison.json", result)
    return validate_dense_partials_artifacts(config)


def _rollback_environment(config: DensePartialsRollbackConfig) -> dict[str, Any]:
    os_environment = {
        "DISABLE_WEIGHT_REQUANTIZATION": "1",
        "GLM_ASYNC_SCHED": "0",
        "GLM_DCP": "1",
        "GLM_DCP_SCATTER_IMPL": "pageloop",
        "GLM_DSA_BT_WIDTH": "owned",
        "GLM_DSA_DCP": "1",
        "GLM_DSA_DCP_PREFILL_ATTN": "segment",
        "GLM_DSA_DCP_SCATTER_IMPL": "flat",
        "GLM_DSA_DUMP_INTERNALS": config.expected_internal_dump_prefix,
        "GLM_DSA_DUMP_INTERNALS_CODE_HASH": config.expected_legacy_pin,
        "GLM_DSA_DUMP_INTERNALS_LAYER": LAYER_NAME,
        "GLM_DSA_DUMP_INTERNALS_MODE": config.expected_internal_mode,
        "GLM_DSA_DUMP_INTERNALS_MODEL_ID": MODEL_ID,
        "GLM_DSA_DUMP_INTERNALS_ORACLE_PIN": config.expected_oracle_pin,
        "GLM_DSA_DUMP_INTERNALS_POSITION": str(POSITION),
        "GLM_DSA_DUMP_INTERNALS_RUN_TAG": config.run_tag,
        "GLM_DSA_DUMP_TOPK": config.expected_dump_prefix,
        "GLM_DSA_DUMP_TOPK_EVENTS": "all",
        "GLM_DSA_DUMP_TOPK_SKIP_WARMUP": "1",
        "GLM_DSA_MERGE_IMPL": "v2",
        "GLM_DSA_MODE": "pallas_decode",
        "GLM_DSA_OWNED_SEG_IMPL": "v2",
        "GLM_DSA_SCORER": "xla",
        "GLM_DSA_SEG_GATHER_IMPL": "v2",
        "GLM_EXPECT_CODE_HASH": config.expected_legacy_short,
        "GLM_GREENFIELD_ACCEPTED_DECODE_PROJECTION_CAPTURE": "0",
        "GLM_GREENFIELD_ACCEPTED_PREFILL_PROJECTION_CAPTURE": "0",
        "GLM_GREENFIELD_DSA_INTERNALS_CAPTURE": "1",
        "GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID": "0",
        "GLM_GREENFIELD_DSA_INTERNALS_MODE": config.expected_internal_mode,
        "GLM_GREENFIELD_DSA_INTERNALS_POSITION": str(POSITION),
        "GLM_GREENFIELD_MAIN_CACHE_CAPTURE": "0",
        "GLM_GREENFIELD_PROMPT_CACHE_CAPTURE": "0",
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE": "8k",
        "GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG": config.run_tag,
        "GLM_GREENFIELD_SHORT_DSA_REMOTE_PREFIX": (
            config.expected_remote_prefix),
        "GLM_LOAD_CHECKSUM": "1",
        "GLM_LOAD_NAN_CHECK": "1",
        "GLM_LOG_STATS": "1",
        "GLM_MLA_DCP": "1",
        "GLM_PWAL_NAN_CHECK": "1",
        "GLM_STATE_HASH_REF": "/tmp/golden.json",
        "GLM_TP": "32",
        "GLM_WK_OOB_DIR": "/home/gianl/gcs-models/models/GLM-5.2-FP8",
        "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
        "GLM_WRITE_PROBE": "1",
        "JAX_SHARE_BINARY_BETWEEN_HOSTS": "1",
        "JAX_SHARE_BINARY_BETWEEN_HOSTS_TIMEOUT_MS": "120000",
        "MODEL_IMPL_TYPE": "vllm",
        "NEW_MODEL_DESIGN": "1",
        "OMP_NUM_THREADS": "1",
        "REQUANTIZE_WEIGHT_DTYPE": "float8_e4m3fn",
        "RUNAI_STREAMER_CONCURRENCY": "32",
        "RUNAI_STREAMER_MEMORY_LIMIT": "34359738368",
        "TPU_DISABLE_DSA_INDEXER": "1",
        "TPU_MULTIHOST_BACKEND": "ray",
    }
    return {
        "attention_path": "dsa-sparse:pallas_decode",
        "base_seed": 12345,
        "benchmark": "longctx_passkey",
        "depths": [0.5],
        "dp_attention": False,
        "eos_ids": [154820, 154827, 154829],
        "expert_parallel": True,
        "gmu": 0.9,
        "lengths": [8192],
        "load_format": "runai_streamer",
        "max_batched_tokens": 2048,
        "max_len": 8704,
        "max_new": 20,
        "max_seqs": 1,
        "model": "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
        "num_gpu_blocks": 24,
        "os_env": os_environment,
        "prompt_mode": "raw_completion",
        "prompt_prefix_ids": [154822, 154824],
        "protocol": "raw",
        "stop_ids": [154820, 154827, 154829, 154828],
        "stub": False,
        "temperature": 0.0,
        "tp": 32,
        "trials": 1,
    }


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def rollback_dense_partial_oracle_run(
    config: DensePartialsRollbackConfig,
) -> str:
    """Delete only an exact dense-partial/boundary run prefix."""
    if not config.run_tag.strip():
        raise ValueError("rollback run tag must be non-empty")
    if config.expected_internal_mode not in {"dense_partial", "dense_boundary"}:
        raise ValueError("rollback internal mode is unsupported")
    _require_digest(config.expected_legacy_pin, length=40,
                    name="rollback legacy pin")
    _require_digest(config.expected_oracle_pin, length=40,
                    name="rollback oracle pin")
    _require_digest(config.expected_prompt_sha256, length=64,
                    name="rollback prompt")
    if not config.expected_raw_outputs or any(
            not isinstance(value, str) or not value
            for value in config.expected_raw_outputs):
        raise ValueError("rollback raw-output set is invalid")
    if config.expected_legacy_short != config.expected_legacy_pin[:len(
            config.expected_legacy_short)]:
        raise ValueError("rollback legacy short hash drifted")
    connection = sqlite3.connect(config.results_db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        matches: list[tuple[Any, ...]] = []
        for row in connection.execute(
            "SELECT run_id,created_utc,model,model_revision,harness_git,"
            "fork_git,env_json,pod,note FROM runs WHERE model=? AND note=?",
            (
                "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
                f"fresh flat all-event DSA oracle {config.run_tag}",
            ),
        ):
            try:
                environment = json.loads(row[6])
            except (TypeError, json.JSONDecodeError):
                continue
            if environment.get("os_env", {}).get(
                    "GLM_GREENFIELD_SHORT_DSA_ORACLE_TAG") == config.run_tag:
                matches.append((*row, environment))
        if not matches:
            connection.rollback()
            return "NO_PROVISIONAL_DB_RUN"
        if len(matches) != 1:
            connection.rollback()
            raise ValueError("refusing ambiguous dense-partial DB rollback")
        row = matches[0]
        run_id = int(row[0])
        environment = row[9]
        if (
            not _valid_timestamp(row[1])
            or row[2] != "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
            or row[3] is not None
            or row[4] != config.expected_harness_git
            or row[5] != config.expected_fork_git
            or environment != _rollback_environment(config)
            or row[7] != "db-v4-64-od"
            or row[8] != f"fresh flat all-event DSA oracle {config.run_tag}"
        ):
            connection.rollback()
            raise ValueError("refusing non-identical dense-partial run rollback")
        items = connection.execute(
            "SELECT benchmark,item_id,asked_utc,prompt,gold,raw_output,"
            "extracted,correct,score,n_prompt_tokens,n_gen_tokens,latency_ms,"
            "seed,finish_reason,truncated FROM items WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        summaries = connection.execute(
            "SELECT benchmark,created_utc,n,metric,value,card_value,delta,note "
            "FROM summary WHERE run_id=? ORDER BY id",
            (run_id,),
        ).fetchall()
        if len(items) > 1 or len(summaries) > 2:
            connection.rollback()
            raise ValueError("refusing non-prefix dense-partial DB rollback")
        asked_utc: str | None = None
        if items:
            item = items[0]
            asked_utc = item[2]
            latency = item[11]
            if (
                item[0] != "passkey_L8192_d0.5"
                or item[1] != "t0"
                or not _valid_timestamp(asked_utc)
                or sha256(item[3].encode("utf-8")).hexdigest()
                != config.expected_prompt_sha256
                or item[4] != "881446"
                or item[5] not in config.expected_raw_outputs
                or item[6] != "881446"
                or item[7] != 1
                or item[8] != 1.0
                or item[9] != 8155
                or item[10] != 20
                or not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(latency)
                or latency <= 0.0
                or item[12] != 1093997
                or item[13] is not None
                or item[14] is not None
            ):
                connection.rollback()
                raise ValueError("refusing non-identical dense-partial item rollback")
        if summaries and not items:
            connection.rollback()
            raise ValueError("refusing summary without dense-partial item")
        expected_summaries = [
            (
                "passkey_L8192_d0.5", 1, "acc", 100.0,
                None, None, "",
            ),
            (
                "longctx_passkey", 0, "acc", 100.0,
                None, None,
                "aggregate over 1 cells x 1 trials; per-cell rows = passkey_L*_d*",
            ),
        ]
        for observed, expected in zip(summaries, expected_summaries, strict=False):
            created_utc = observed[1]
            if (
                not _valid_timestamp(created_utc)
                or (observed[0], *observed[2:]) != expected
                or asked_utc is None
                or not 0.0 <= (
                    datetime.fromisoformat(created_utc)
                    - datetime.fromisoformat(asked_utc)
                ).total_seconds() <= 60.0
            ):
                connection.rollback()
                raise ValueError("refusing non-prefix dense-partial summary rollback")
        connection.execute("DELETE FROM summary WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM items WHERE run_id=?", (run_id,))
        connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        if connection.execute(
            "SELECT COUNT(*) FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()[0] != 0:
            connection.rollback()
            raise ValueError("dense-partial DB rollback did not remove run")
        connection.commit()
        return f"ROLLED_BACK_PROVISIONAL_DB_RUN={run_id}"
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()
