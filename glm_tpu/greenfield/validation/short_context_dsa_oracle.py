"""Seal recurrent short-context legacy DSA events as an independent oracle."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence

import numpy as np

from .short_context_oracle import inspect_short_context_oracle


FORMAT_VERSION = 2
ARTIFACT_KIND = "greenfield_short_context_legacy_dsa_oracle"
MODEL_ID = "zai-org/GLM-5.2-FP8"
TIE_POLICY = "descending_score_then_lowest_global_position"
PADDED_ROW_POLICY = "excluded_by_valid_mask_and_provenance_counted"
_DUMP_NAME = re.compile(
    r"\.step(?P<step>[0-9]{4})\.evt(?P<event>[0-9]{2})"
    r"\.proc(?P<process>[0-9]+)\.npz$"
)
_PAYLOAD_KEYS = (
    "distribution",
    "positions",
    "req_ids",
    "topk_indices",
    "topk_scores",
    "valid",
)
_EXPECTED_DTYPES = {
    "distribution": np.dtype(np.int32),
    "positions": np.dtype(np.int32),
    "req_ids": np.dtype(np.int32),
    "topk_indices": np.dtype(np.int32),
    "topk_scores": np.dtype(np.float32),
    "valid": np.dtype(np.bool_),
}


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
    array = np.asarray(value)
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return sha256(little.tobytes(order="C")).hexdigest()


def _validate_digest(value: str, name: str, lengths: tuple[int, ...]) -> None:
    if len(value) not in lengths or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")


def _write_text_once(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"append-only DSA oracle file exists: {path}")
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "byte_count": path.stat().st_size,
        "filename": path.name,
        "sha256": _sha256_file(path),
    }


@dataclass(frozen=True, slots=True)
class ShortContextDsaOracleConfig:
    results_db: Path
    token_oracle_dir: Path
    source_dump_dir: Path
    output_dir: Path
    capture_code_hash: str
    source_capture_code_hash: str
    legacy_repository_pin: str
    token_oracle_manifest_sha256: str
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
    expected_oob_dir: str
    expected_dump_prefix: str
    expected_process_count: int
    first_source_step: int
    decode_step_count: int
    first_decode_position: int
    selected_width: int
    producer_layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "results_db",
            "token_oracle_dir",
            "source_dump_dir",
            "output_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        object.__setattr__(
            self, "producer_layer_ids", tuple(self.producer_layer_ids)
        )
        for name in (
            "run_id",
            "item_row_id",
            "expected_prompt_tokens",
            "expected_generated_tokens",
            "expected_process_count",
            "first_source_step",
            "decode_step_count",
            "first_decode_position",
            "selected_width",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.producer_layer_ids or any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in self.producer_layer_ids
        ):
            raise ValueError("producer_layer_ids must be non-negative integers")
        if len(set(self.producer_layer_ids)) != len(self.producer_layer_ids):
            raise ValueError("producer_layer_ids must be unique")
        for name in (
            "capture_code_hash",
            "source_capture_code_hash",
            "legacy_repository_pin",
            "token_oracle_manifest_sha256",
        ):
            _validate_digest(getattr(self, name), name, (40, 64))
        for name in ("expected_harness_git", "expected_fork_git"):
            _validate_digest(
                getattr(self, name), name, tuple(range(7, 65))
            )
        if not self.expected_dump_prefix.startswith("/tmp/"):
            raise ValueError("legacy DSA dump prefix must be under /tmp")
        if not self.expected_oob_dir.startswith(
            "/home/gianl/gcs-models/models/"
        ):
            raise ValueError(
                "legacy WK repair source must use the approved read-only model mount"
            )
        if not self.expected_model_uri.startswith("gs://driftbench-dsv4-uc/"):
            raise ValueError("legacy model URI must use the approved bucket")

    @property
    def event_count(self) -> int:
        return len(self.producer_layer_ids)


def _read_source_row(config: ShortContextDsaOracleConfig) -> dict[str, Any]:
    connection = sqlite3.connect(
        f"file:{config.results_db}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        integrity = [row[0] for row in connection.execute("pragma integrity_check")]
        if integrity != ["ok"]:
            raise ValueError(f"legacy results DB integrity failed: {integrity}")
        row = connection.execute(
            """
            SELECT i.*, r.created_utc AS run_created_utc, r.model,
                   r.model_revision, r.harness_git, r.fork_git, r.env_json,
                   r.pod, r.note
            FROM items AS i JOIN runs AS r ON r.run_id = i.run_id
            WHERE i.run_id = ? AND i.id = ?
            """,
            (config.run_id, config.item_row_id),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise ValueError("legacy DSA source row is missing")
    source = dict(row)
    expected = {
        "benchmark": config.expected_benchmark,
        "correct": 1,
        "fork_git": config.expected_fork_git,
        "gold": config.expected_gold,
        "harness_git": config.expected_harness_git,
        "id": config.item_row_id,
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
        raise ValueError(f"legacy DSA source identity drifted: {mismatches}")
    environment = json.loads(source["env_json"])
    os_environment = environment.get("os_env", {})
    required_os_environment = {
        "GLM_DCP": "1",
        "GLM_DCP_SCATTER_IMPL": "pageloop",
        "GLM_DSA_DCP": "1",
        "GLM_DSA_DCP_SCATTER_IMPL": "flat",
        "GLM_DSA_DUMP_TOPK": config.expected_dump_prefix,
        "GLM_DSA_DUMP_TOPK_EVENTS": "all",
        "GLM_DSA_MODE": "pallas_decode",
        "GLM_DSA_SCORER": "xla",
        "GLM_EXPECT_CODE_HASH": config.legacy_repository_pin[:9],
        "GLM_LOAD_CHECKSUM": "1",
        "GLM_LOAD_NAN_CHECK": "1",
        "GLM_MLA_DCP": "1",
        "GLM_PWAL_NAN_CHECK": "1",
        "GLM_STATE_HASH_REF": "/tmp/golden.json",
        "GLM_WK_OOB_DIR": config.expected_oob_dir,
        "GLM_WK_OOB_GOLDEN": "/tmp/golden.json",
    }
    environment_expected = {
        "model": config.expected_model_uri,
        "prompt_prefix_ids": [154_822, 154_824],
        "protocol": "raw",
        "temperature": 0.0,
    }
    environment_mismatches = {
        name: {"expected": value, "observed": environment.get(name)}
        for name, value in environment_expected.items()
        if environment.get(name) != value
    }
    os_mismatches = {
        name: {"expected": value, "observed": os_environment.get(name)}
        for name, value in required_os_environment.items()
        if os_environment.get(name) != value
    }
    if environment_mismatches or os_mismatches:
        raise ValueError(
            "legacy DSA protocol drifted: "
            f"run={environment_mismatches} os={os_mismatches}"
        )
    source["env_json"] = environment
    return source


def _load_dump(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        keys = set(handle.files)
        expected_keys = {
            "event_index",
            "process_count",
            "process_index",
            "stash_key",
            "step_index",
            *(_PAYLOAD_KEYS),
            *(f"{key}__dtype" for key in _PAYLOAD_KEYS),
        }
        if keys != expected_keys:
            raise ValueError(f"legacy DSA dump keys drifted: {path}")
        return {key: np.asarray(handle[key]).copy() for key in handle.files}


def _validate_selected_row(
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    position: int,
    selected_width: int,
    context: str,
) -> int:
    """Validate the exact causal top-k representation for one live row."""

    valid_count = min(position + 1, selected_width)
    live_indices = indices[:valid_count]
    live_scores = scores[:valid_count]
    unique_count = int(np.unique(live_indices).size)
    if unique_count != valid_count or np.any(live_indices < 0) or np.any(
        live_indices > position
    ):
        raise ValueError(f"{context} selected set is not unique and causal")
    if position + 1 <= selected_width and not np.array_equal(
        np.sort(live_indices), np.arange(position + 1, dtype=np.int32)
    ):
        raise ValueError(f"{context} selected set is not the exact causal prefix")
    if np.any(indices[valid_count:] != -1) or not np.all(
        np.isneginf(scores[valid_count:])
    ):
        raise ValueError(f"{context} sentinel tail drifted")
    if not np.all(np.isfinite(live_scores)) or np.any(
        live_scores[1:] > live_scores[:-1]
    ):
        raise ValueError(f"{context} scores are not finite descending values")
    ties = live_scores[1:] == live_scores[:-1]
    if np.any(live_indices[1:][ties] < live_indices[:-1][ties]):
        raise ValueError(f"{context} lowest-position tie order drifted")
    return valid_count


def _validate_dump(
    payload: Mapping[str, np.ndarray],
    *,
    config: ShortContextDsaOracleConfig,
    step: int,
    event: int,
    position: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    scalars = {
        "step_index": step,
        "event_index": event,
        "process_count": config.expected_process_count,
    }
    for name, expected in scalars.items():
        value = payload[name]
        if value.shape != () or value.dtype != np.int64 or int(value) != expected:
            raise ValueError(f"legacy DSA {name} drifted at step/event {step}/{event}")
    if payload["process_index"].shape != () or (
        payload["process_index"].dtype != np.int64
    ):
        raise ValueError("legacy DSA process index metadata drifted")
    process_index = int(payload["process_index"])
    if not 0 <= process_index < config.expected_process_count:
        raise ValueError("legacy DSA process index escaped process count")
    if payload["stash_key"].shape != () or str(payload["stash_key"]) != (
        "topk_indices"
    ):
        raise ValueError("legacy DSA stash identity drifted")
    for name, dtype in _EXPECTED_DTYPES.items():
        if payload[name].dtype != dtype or str(payload[f"{name}__dtype"]) != str(
            dtype
        ):
            raise ValueError(f"legacy DSA dtype drifted for {name}")
    indices = payload["topk_indices"]
    scores = payload["topk_scores"]
    if indices.ndim != 2 or indices.shape[1] != config.selected_width or (
        scores.shape != indices.shape
    ):
        raise ValueError("legacy DSA selected tensor shape drifted")
    rows = indices.shape[0]
    for name in ("positions", "req_ids", "valid"):
        if payload[name].shape != (rows,):
            raise ValueError(f"legacy DSA row metadata shape drifted: {name}")
    live = np.flatnonzero(payload["valid"])
    if live.tolist() != [0]:
        raise ValueError("legacy DSA decode dump must contain one live row zero")
    if int(payload["positions"][0]) != position or int(payload["req_ids"][0]) != 0:
        raise ValueError("legacy DSA live row position/request drifted")
    valid_count = _validate_selected_row(
        indices[0],
        scores[0],
        position=position,
        selected_width=config.selected_width,
        context="legacy DSA",
    )
    live_indices = indices[0, :valid_count]
    live_scores = scores[0, :valid_count]
    # Legacy's batch bucket is larger than the one live request. Invalid
    # rows are explicitly outside the DSA verdict and may retain stale
    # selections from earlier scheduler work. This is the established
    # dsa_topk_diff contract: align and compare only ``valid`` rows while
    # reporting pad-row payloads as provenance. Requiring sentinel payloads
    # here would reject a correct live row based on semantically dead state.
    padded_non_sentinel_rows = 0
    if rows > 1:
        padded_non_sentinel_rows = int(
            np.count_nonzero(
                np.any(indices[1:] != -1, axis=1)
                | ~np.all(np.isneginf(scores[1:]), axis=1)
            )
        )
    return (
        live_indices.copy(),
        live_scores.copy(),
        valid_count,
        padded_non_sentinel_rows,
    )


def capture_short_context_dsa_oracle(
    config: ShortContextDsaOracleConfig,
) -> dict[str, Any]:
    """Validate, compact, and seal a fresh flat legacy DSA capture."""

    if config.output_dir.exists():
        raise FileExistsError(f"append-only DSA oracle exists: {config.output_dir}")
    token_manifest = inspect_short_context_oracle(config.token_oracle_dir)
    if token_manifest["manifest_sha256"] != config.token_oracle_manifest_sha256:
        raise ValueError("sealed token oracle manifest pin drifted")
    source = _read_source_row(config)
    token_source = json.loads((config.token_oracle_dir / "source_row.json").read_text())
    for name in ("prompt", "raw_output", "gold", "seed"):
        if source[name] != token_source[name]:
            raise ValueError(f"fresh DSA run disagrees with token oracle: {name}")

    groups: dict[tuple[int, int], list[Path]] = {}
    for path in sorted(config.source_dump_dir.rglob("*.npz")):
        match = _DUMP_NAME.search(path.name)
        if match is None:
            continue
        key = (int(match.group("step")), int(match.group("event")))
        groups.setdefault(key, []).append(path)
    selected_positions = np.full(
        (config.decode_step_count, config.event_count, config.selected_width),
        -1,
        dtype=np.int32,
    )
    selected_scores = np.full(
        selected_positions.shape, -np.inf, dtype=np.float32
    )
    valid_counts = np.zeros(
        (config.decode_step_count, config.event_count), dtype=np.int32
    )
    decode_positions = np.arange(
        config.first_decode_position,
        config.first_decode_position + config.decode_step_count,
        dtype=np.int32,
    )
    source_files = []
    replica_layout: set[tuple[str, int]] | None = None
    for step_offset, position in enumerate(decode_positions.tolist()):
        step = config.first_source_step + step_offset
        for event in range(config.event_count):
            paths = groups.get((step, event), [])
            if not paths:
                raise ValueError(f"legacy DSA dump missing step/event {step}/{event}")
            canonical_payload = None
            event_replicas = set()
            canonical_values = None
            for path in paths:
                payload = _load_dump(path)
                values = _validate_dump(
                    payload,
                    config=config,
                    step=step,
                    event=event,
                    position=position,
                )
                relative = path.relative_to(config.source_dump_dir).as_posix()
                process_index = int(payload["process_index"])
                event_replicas.add((str(Path(relative).parent), process_index))
                source_files.append(
                    {
                        "byte_count": path.stat().st_size,
                        "event_index": event,
                        "live_row_indices": [0],
                        "padded_non_sentinel_row_count": values[3],
                        "path": relative,
                        "process_count": int(payload["process_count"]),
                        "process_index": process_index,
                        "row_count": int(payload["valid"].size),
                        "sha256": _sha256_file(path),
                        "step_index": step,
                    }
                )
                comparable = tuple(payload[name] for name in _PAYLOAD_KEYS)
                if canonical_payload is None:
                    canonical_payload = comparable
                    canonical_values = values
                elif any(
                    not np.array_equal(left, right)
                    for left, right in zip(canonical_payload, comparable, strict=True)
                ):
                    raise ValueError("legacy DSA replicated dump payload drifted")
            if replica_layout is None:
                replica_layout = event_replicas
            elif event_replicas != replica_layout:
                raise ValueError("legacy DSA source replica coverage drifted")
            assert canonical_values is not None
            indices, scores, count, _ = canonical_values
            selected_positions[step_offset, event, :count] = indices
            selected_scores[step_offset, event, :count] = scores
            valid_counts[step_offset, event] = count

    config.output_dir.mkdir(parents=True)
    from safetensors.numpy import save_file

    tensor_path = config.output_dir / "dsa_events.safetensors"
    partial = tensor_path.with_suffix(".safetensors.partial")
    tensors = {
        "decode_positions": decode_positions,
        "producer_layer_ids": np.asarray(config.producer_layer_ids, np.int32),
        "selected_positions": selected_positions,
        "selected_scores": selected_scores,
        "valid_counts": valid_counts,
    }
    save_file(
        tensors,
        partial,
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "model_id": MODEL_ID,
            "tie_policy": TIE_POLICY,
        },
    )
    partial.replace(tensor_path)
    source_path = config.output_dir / "source_row.json"
    _write_text_once(source_path, json.dumps(source, indent=2, sort_keys=True) + "\n")
    source_row_sha256 = sha256(
        _canonical_json(source).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
                "shape": list(value.shape),
            }
            for name, value in tensors.items()
        },
        "capture_code_hash": config.capture_code_hash,
        "event_contract": {
            "decode_step_count": config.decode_step_count,
            "event_count": config.event_count,
            "first_decode_position": config.first_decode_position,
            "first_source_step": config.first_source_step,
            "padded_row_policy": PADDED_ROW_POLICY,
            "producer_layer_ids": list(config.producer_layer_ids),
            "selected_width": config.selected_width,
            "tie_policy": TIE_POLICY,
        },
        "files": {
            "source_row": _file_record(source_path),
            "tensors": _file_record(tensor_path),
        },
        "format_version": FORMAT_VERSION,
        "legacy_repository_pin_at_capture": config.legacy_repository_pin,
        "model_id": MODEL_ID,
        "repair_contract": {
            "golden_manifest": "/tmp/golden.json",
            "oob_checkpoint_dir": config.expected_oob_dir,
        },
        "source": {
            "benchmark": source["benchmark"],
            "dump_prefix": config.expected_dump_prefix,
            "fork_git": source["fork_git"],
            "harness_git": source["harness_git"],
            "item_id": source["item_id"],
            "item_row_id": source["id"],
            "process_count": config.expected_process_count,
            "replicas": [
                {"directory": directory, "process_index": process}
                for directory, process in sorted(replica_layout or ())
            ],
            "run_id": source["run_id"],
            "source_row_sha256": source_row_sha256,
        },
        "source_capture_code_hash": config.source_capture_code_hash,
        "source_dump_files": source_files,
        "token_oracle": {
            "generated_token_ids_sha256": token_manifest[
                "generated_token_ids_sha256"
            ],
            "manifest_sha256": token_manifest["manifest_sha256"],
            "prompt_token_ids_sha256": token_manifest["prompt_token_ids_sha256"],
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _write_text_once(
        config.output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    inspect_short_context_dsa_oracle(config.output_dir)
    return manifest


def inspect_short_context_dsa_oracle(output_dir: Path) -> dict[str, Any]:
    """Verify the sealed compact DSA artifact without legacy dependencies."""

    from safetensors import safe_open

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND or (
        manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("unsupported short-context DSA oracle")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("short-context DSA oracle manifest checksum mismatch")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("short-context DSA oracle model identity drifted")
    for name in (
        "capture_code_hash",
        "source_capture_code_hash",
        "legacy_repository_pin_at_capture",
    ):
        _validate_digest(manifest.get(name, ""), name, (40, 64))
    if manifest.get("event_contract", {}).get("tie_policy") != TIE_POLICY:
        raise ValueError("short-context DSA oracle tie policy drifted")
    if manifest.get("event_contract", {}).get("padded_row_policy") != (
        PADDED_ROW_POLICY
    ):
        raise ValueError("short-context DSA oracle padded-row policy drifted")
    expected_files = {
        "source_row": "source_row.json",
        "tensors": "dsa_events.safetensors",
    }
    if {
        role: record.get("filename")
        for role, record in manifest.get("files", {}).items()
    } != expected_files:
        raise ValueError("short-context DSA oracle file identities drifted")
    for record in manifest["files"].values():
        path = output_dir / record["filename"]
        if path.stat().st_size != record["byte_count"]:
            raise ValueError("short-context DSA oracle file size mismatch")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError("short-context DSA oracle file checksum mismatch")
    source = json.loads((output_dir / "source_row.json").read_text())
    if sha256(_canonical_json(source).encode("utf-8")).hexdigest() != (
        manifest["source"]["source_row_sha256"]
    ):
        raise ValueError("short-context DSA oracle source-row checksum mismatch")
    repair_contract = manifest.get("repair_contract", {})
    oob_dir = repair_contract.get("oob_checkpoint_dir", "")
    if not isinstance(oob_dir, str) or not oob_dir.startswith(
        "/home/gianl/gcs-models/models/"
    ) or repair_contract.get("golden_manifest") != "/tmp/golden.json":
        raise ValueError("short-context DSA oracle repair contract drifted")
    source_os_environment = source.get("env_json", {}).get("os_env", {})
    if source_os_environment.get("GLM_WK_OOB_DIR") != oob_dir or (
        source_os_environment.get("GLM_WK_OOB_GOLDEN")
        != repair_contract["golden_manifest"]
    ):
        raise ValueError("short-context DSA oracle repair provenance drifted")
    contract = manifest["event_contract"]
    steps = int(contract["decode_step_count"])
    events = int(contract["event_count"])
    width = int(contract["selected_width"])
    expected_shapes = {
        "decode_positions": (steps,),
        "producer_layer_ids": (events,),
        "selected_positions": (steps, events, width),
        "selected_scores": (steps, events, width),
        "valid_counts": (steps, events),
    }
    expected_dtypes = {
        "decode_positions": np.dtype(np.int32),
        "producer_layer_ids": np.dtype(np.int32),
        "selected_positions": np.dtype(np.int32),
        "selected_scores": np.dtype(np.float32),
        "valid_counts": np.dtype(np.int32),
    }
    tensor_path = output_dir / "dsa_events.safetensors"
    with safe_open(tensor_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
            "model_id": MODEL_ID,
            "tie_policy": TIE_POLICY,
        }:
            raise ValueError("short-context DSA tensor metadata drifted")
        if set(handle.keys()) != set(expected_shapes):
            raise ValueError("short-context DSA tensor keys drifted")
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    for name, value in tensors.items():
        record = manifest["arrays"].get(name)
        if value.shape != expected_shapes[name] or value.dtype != expected_dtypes[name]:
            raise ValueError(f"short-context DSA tensor contract drifted: {name}")
        if record != {
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
            "shape": list(value.shape),
        }:
            raise ValueError(f"short-context DSA tensor checksum drifted: {name}")
    expected_positions = np.arange(
        int(contract["first_decode_position"]),
        int(contract["first_decode_position"]) + steps,
        dtype=np.int32,
    )
    if not np.array_equal(tensors["decode_positions"], expected_positions) or (
        tensors["producer_layer_ids"].tolist()
        != contract["producer_layer_ids"]
    ):
        raise ValueError("short-context DSA step/event mapping drifted")
    for step, position in enumerate(expected_positions.tolist()):
        for event in range(events):
            indices = tensors["selected_positions"][step, event]
            scores = tensors["selected_scores"][step, event]
            count = _validate_selected_row(
                indices,
                scores,
                position=position,
                selected_width=width,
                context="short-context DSA",
            )
            if int(tensors["valid_counts"][step, event]) != count:
                raise ValueError("short-context DSA valid-count contract drifted")
    source_files = manifest.get("source_dump_files")
    if not isinstance(source_files, list) or len(source_files) < steps * events:
        raise ValueError("short-context DSA source-file coverage drifted")
    expected_pairs = {
        (int(record["step_index"]), int(record["event_index"]))
        for record in source_files
    }
    if expected_pairs != {
        (int(contract["first_source_step"]) + step, event)
        for step in range(steps)
        for event in range(events)
    }:
        raise ValueError("short-context DSA source step/event coverage drifted")
    for record in source_files:
        _validate_digest(record.get("sha256", ""), "source dump", (64,))
        if int(record.get("byte_count", 0)) <= 0:
            raise ValueError("short-context DSA source-file record drifted")
        row_count = int(record.get("row_count", 0))
        padded_non_sentinel = int(
            record.get("padded_non_sentinel_row_count", -1)
        )
        if (
            record.get("live_row_indices") != [0]
            or row_count <= 0
            or not 0 <= padded_non_sentinel < row_count
        ):
            raise ValueError("short-context DSA padded-row provenance drifted")
    return manifest
