"""Authenticate the sealed full-width PP16 feature2 numerical rejection."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

import google_crc32c
import numpy as np

from ..errors import BenchmarkValidationError
from ..validation.prompt_index_cache import inspect_legacy_prompt_index_cache
from .pp16_feature2_hlo import validate_feature2_sealed_hlo_archive_identity
from .pp16_feature2_numerical import (
    compare_feature2_full_width_numerical_capture,
    validate_feature2_in_process_cleanup,
)

SOURCE_TAG = "greenfield_pp16_feature2_prefill_numerical_20260829T051119686986506Z"
SOURCE_CODE_HASH = "e404abee10a770f74cf1b22a7dd9d7ba904add06"
SOURCE_LEDGER_SHA256 = (
    "05bbbfbbf97c97c678701dddc46e685c3cf4ad7700ad3f76633970a37280055c"
)
SOURCE_CAPTURE_SHA256 = (
    "e514fc28e9d8c30bc7de9d70f01ae04002666494446e2f4a06a7fe6c66901e65"
)
HALF_WIDTH_CAPTURE_SHA256 = (
    "be3dda446cfbde547af49dd5ea371af690b553bcc8414907add1b0a2503b9d0d"
)
SOURCE_POST_CENSUS_SHA256 = (
    "455e26851f9c979ecfab1d5a608764e1aa4ced224a615bf7f7049f35f7e3b1ae"
)
SOURCE_TERMINAL_SHA256 = (
    "60b605c8897a9b128f58567470960df6fb6caef0cb2979016de7f27713e8b9ad"
)
SOURCE_TERMINAL_GENERATION = "1787980539108624"
SOURCE_TERMINAL_CRC32C = "jCuZbg=="
SOURCE_MAIN_STABLEHLO_SHA256 = (
    "6c1c69d76c3d121ed4f84cb85fe0091d1605ae43d0d5707e3d52ba2cdd310ad4"
)
SOURCE_MAIN_RAW_OPTIMIZED_HLO_SHA256 = (
    "9e8641ba1de0e41e5c55695dff7a3545169e534f72e890c67ef235111ffd2daf"
)
SOURCE_MAIN_CANONICAL_HLO_SHA256 = (
    "9e933384f340eef45b0479f740379356831feb792a046d11db266f5d69c719a5"
)
SOURCE_MAIN_CANONICAL_HLO_BYTES = 6_558_627
SOURCE_MAIN_STACK_FRAME_REFERENCES = 14_561
EXPECTED_MISMATCH_COUNTS = {
    "carried_bfloat16_bits": 968,
    "contract_valid": 0,
    "event1_positions": 1852,
    "event1_scores": 2048,
    "event1_valid_counts": 0,
    "layer1_current_key_bfloat16_bits": 0,
    "layer1_dsa_head_weights_float32": 64,
    "layer1_dsa_query_float32": 8192,
    "layer1_normalized_hidden_bfloat16_bits": 2,
    "layer1_q_a_state_bfloat16_bits": 94,
}
COMMON_CAPTURE_ARRAYS = (
    "carried_liveness_digest_owners",
    "contract_valid",
    "current_attention_query_owners_bfloat16_bits",
    "current_carried_halves_bfloat16_bits",
    "current_kv_a_bfloat16_bits",
    "event1_positions",
    "event1_scores",
    "event1_valid_counts",
    "layer0_index_cache_owners_bfloat16_bits",
    "layer0_kv_cache_owners_bfloat16_bits",
    "layer1_index_cache_owners_bfloat16_bits",
)
DB518_MANIFEST_SHA256 = (
    "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab"
)
DB518_CACHE_SHA256 = "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
DB518_TENSOR_FILE_SHA256 = (
    "36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a"
)
CANDIDATE_LOGICAL_CACHE_SHA256 = (
    "35350ca026c437def5c958b3382ea3aabc97747d3fbdea8205e1aa336cd48d7a"
)
CACHE_MISMATCH_COORDINATES_SHA256 = (
    "72dc17d4ec3fa5473f05cd33f34cc2eda522413d8ca73e4b93e81b2556c6aaf3"
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_crc32c(record: dict[str, Any]) -> str:
    """Return one unambiguous gcloud CRC32C spelling or fail closed."""

    values = [
        str(record[key])
        for key in ("crc32c_hash", "crc32c")
        if record.get(key) not in (None, "")
    ]
    if not values or len(set(values)) != 1:
        raise BenchmarkValidationError(
            "feature2 recovery remote CRC32C is absent or conflicting"
        )
    return values[0]


def _arrays_bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.ascontiguousarray(left)
    right_array = np.ascontiguousarray(right)
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes() == right_array.tobytes()
    )


def _parse_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*").removeprefix("./")
        pure = PurePosixPath(relative)
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in entries
        ):
            raise BenchmarkValidationError(
                "feature2 recovery source ledger is unsafe or duplicated"
            )
        entries[relative] = digest
    if len(entries) != 26:
        raise BenchmarkValidationError(
            "feature2 recovery source ledger cardinality drifted"
        )
    return entries


def _eight_census_hosts(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    hosts = [line.split()[1] for line in lines if line.startswith("CENSUS_OK ")]
    if (
        len(hosts) != 8
        or len(set(hosts)) != 8
        or any("CENSUS_BAD" in line or "CENSUS_BUSY" in line for line in lines)
    ):
        raise BenchmarkValidationError(
            "feature2 recovery census is not authenticated 8/8 zero work"
        )
    return sorted(hosts)


def _terminal_authentication(source: Path, source_remote: str) -> dict[str, Any]:
    terminal = source / "NUMERICAL_REJECTED"
    receipt = source / "terminal_create.receipt.stderr"
    upload_metadata_path = source / "terminal_upload.describe.json"
    rollback_metadata_path = source / "terminal_rollback.describe.json"
    if any(
        not path.is_file()
        for path in (
            terminal,
            receipt,
            upload_metadata_path,
            rollback_metadata_path,
        )
    ):
        raise BenchmarkValidationError(
            "feature2 recovery terminal source files are incomplete"
        )
    if _sha256_file(terminal) != SOURCE_TERMINAL_SHA256:
        raise BenchmarkValidationError("feature2 recovery terminal bytes drifted")
    upload_metadata = json.loads(upload_metadata_path.read_text())
    rollback_metadata = json.loads(rollback_metadata_path.read_text())
    if upload_metadata != rollback_metadata:
        raise BenchmarkValidationError(
            "feature2 recovery terminal metadata snapshots disagree"
        )
    expected_uri = f"{source_remote}/NUMERICAL_REJECTED"
    generation = str(upload_metadata.get("generation", ""))
    matches = re.findall(r"(gs://[^\s]+)#([0-9]+)", receipt.read_text())
    checksum = google_crc32c.Checksum(terminal.read_bytes())
    crc32c = base64.b64encode(checksum.digest()).decode("ascii")
    remote_crc32c = _metadata_crc32c(upload_metadata)
    if (
        matches != [(expected_uri, SOURCE_TERMINAL_GENERATION)]
        or generation != SOURCE_TERMINAL_GENERATION
        or int(upload_metadata.get("size", -1)) != terminal.stat().st_size
        or crc32c != SOURCE_TERMINAL_CRC32C
        or remote_crc32c != SOURCE_TERMINAL_CRC32C
    ):
        raise BenchmarkValidationError(
            "feature2 recovery terminal receipt/generation/CRC drifted"
        )
    record = json.loads(terminal.read_text())
    bound_keys = (
        "artifact_kind",
        "comparison_sha256",
        "evidence_sha256",
        "status",
        "summary_sha256",
    )
    raw = json.dumps(
        {key: record[key] for key in bound_keys},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    expected_links = {
        "comparison_sha256": _sha256_file(source / "comparison.json"),
        "evidence_sha256": _sha256_file(source / "evidence.sha256"),
        "summary_sha256": _sha256_file(source / "summary.json"),
    }
    if (
        record.get("artifact_kind") != "greenfield_pp16_feature2_numerical_terminal"
        or record.get("status") != "NUMERICAL_REJECTED"
        or any(record.get(key) != value for key, value in expected_links.items())
        or record.get("marker_self_sha256") != sha256(raw).hexdigest()
    ):
        raise BenchmarkValidationError(
            "feature2 recovery terminal self/linkage contract drifted"
        )
    return {
        "crc32c": crc32c,
        "generation": generation,
        "marker_self_sha256": record["marker_self_sha256"],
        "receipt_sha256": _sha256_file(receipt),
        "remote_uri": expected_uri,
        "sha256": SOURCE_TERMINAL_SHA256,
        "size": terminal.stat().st_size,
        "upload_metadata_sha256": _sha256_file(upload_metadata_path),
    }


def authenticate_feature2_full_width_rejection(
    source_run_dir: Path,
    *,
    source_remote: str,
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    layer1_internal_reference: Path,
    db529_internal_dir: Path,
    db550_boundary: Path,
    half_width_capture: Path,
    db518_prompt_cache_dir: Path,
    results_db: Path,
    recovery_tag: str,
) -> dict[str, Any]:
    """Recompute the immutable local rejection without opening a TPU backend."""

    source = Path(source_run_dir)
    if source.name != SOURCE_TAG or not source.is_dir():
        raise BenchmarkValidationError("feature2 recovery source tag/path drifted")
    ledger_path = source / "evidence.sha256"
    if _sha256_file(ledger_path) != SOURCE_LEDGER_SHA256:
        raise BenchmarkValidationError("feature2 recovery source ledger drifted")
    ledger = _parse_ledger(ledger_path)
    for relative, expected in ledger.items():
        path = source / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise BenchmarkValidationError(
                f"feature2 recovery source object drifted: {relative}"
            )
    if _sha256_file(source / "result.npz") != SOURCE_CAPTURE_SHA256:
        raise BenchmarkValidationError("feature2 recovery source capture drifted")

    runner = json.loads((source / "runner.json").read_text())
    if (
        runner.get("status") != "NUMERICAL_CAPTURED"
        or runner.get("code_hash") != SOURCE_CODE_HASH
        or runner.get("graph_sha256")
        != "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d"
        or runner.get("main_executed") is not True
        or runner.get("main_execution_count") != 1
        or runner.get("compile_only") is not False
        or runner.get("full_width_rounded_then_slice") is not True
        or runner.get("sealed_boundary_capture") is not True
        or runner.get("numerical_claim") is not False
        or runner.get("performance_claim") is not False
    ):
        raise BenchmarkValidationError(
            "feature2 recovery numerical runner boundary drifted"
        )
    if runner.get("physical_group") != {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "local_device_count_visible": 4,
        "mesh_device_count": 2,
    }:
        raise BenchmarkValidationError("feature2 recovery LP2 group drifted")
    runtime = runner.get("runtime_pins", {})
    if (
        {name: runtime.get(name) for name in ("jax", "jaxlib", "libtpu")}
        != {"jax": "0.10.1", "jaxlib": "0.10.1", "libtpu": "0.0.41"}
        or runtime.get("backend_platform") != "tpu"
        or runtime.get("device_kind") != "TPU v4"
        or not runtime.get("platform_version")
    ):
        raise BenchmarkValidationError("feature2 recovery runtime pins drifted")

    main = runner.get("hlo", {}).get("feature2_main", {})
    canonical = main.get("execution_canonical_hlo", {})
    identity = validate_feature2_sealed_hlo_archive_identity(
        source / "hlo" / main.get("stablehlo", {}).get("filename", ""),
        source / "hlo" / main.get("optimized_hlo", {}).get("filename", ""),
        source / "hlo" / canonical.get("filename", ""),
        expected_stablehlo_sha256=SOURCE_MAIN_STABLEHLO_SHA256,
        expected_canonical_sha256=SOURCE_MAIN_CANONICAL_HLO_SHA256,
        expected_canonical_bytes=SOURCE_MAIN_CANONICAL_HLO_BYTES,
        expected_canonicalizer_version=1,
        expected_stripped_stack_frame_references=(SOURCE_MAIN_STACK_FRAME_REFERENCES),
    )
    if (
        identity["optimized_hlo_sha256"] != SOURCE_MAIN_RAW_OPTIMIZED_HLO_SHA256
        or main.get("optimized_hlo", {}).get("sha256")
        != SOURCE_MAIN_RAW_OPTIMIZED_HLO_SHA256
        or canonical.get("canonicalizer_code_hash") != SOURCE_CODE_HASH
    ):
        raise BenchmarkValidationError("feature2 recovery HLO identity drifted")

    comparison = compare_feature2_full_width_numerical_capture(
        source / "result.npz",
        token_oracle_dir=token_oracle_dir,
        dsa_oracle_dir=dsa_oracle_dir,
        layer1_internal_reference=layer1_internal_reference,
        db529_internal_dir=db529_internal_dir,
        db550_boundary=db550_boundary,
    )
    if (
        comparison.get("status") != "NUMERICAL_REJECTED"
        or comparison.get("exact") is not False
        or comparison.get("capture_sha256") != SOURCE_CAPTURE_SHA256
        or comparison.get("comparison_schema") != "full_width_sealed_boundaries_v2"
        or comparison.get("mismatch_counts") != EXPECTED_MISMATCH_COUNTS
    ):
        raise BenchmarkValidationError("feature2 recovery numerical rejection drifted")

    half_width_capture = Path(half_width_capture)
    if _sha256_file(half_width_capture) != HALF_WIDTH_CAPTURE_SHA256:
        raise BenchmarkValidationError(
            "feature2 recovery half-width source capture drifted"
        )

    db518_manifest, accepted_prompt_cache = inspect_legacy_prompt_index_cache(
        db518_prompt_cache_dir,
        expected_manifest_sha256=DB518_MANIFEST_SHA256,
    )
    if db518_manifest.get("prompt_index_key_bfloat16_sha256") != DB518_CACHE_SHA256:
        raise BenchmarkValidationError("feature2 recovery DB518 cache identity drifted")
    if db518_manifest.get("tensor_file", {}).get("sha256") != DB518_TENSOR_FILE_SHA256:
        raise BenchmarkValidationError(
            "feature2 recovery DB518 tensor-file identity drifted"
        )

    with (
        np.load(source / "result.npz", allow_pickle=False) as capture,
        np.load(half_width_capture, allow_pickle=False) as half_width,
        np.load(layer1_internal_reference, allow_pickle=False) as expected,
    ):
        common_arrays = tuple(sorted(set(capture.files) & set(half_width.files)))
        if common_arrays != COMMON_CAPTURE_ARRAYS:
            raise BenchmarkValidationError(
                "feature2 recovery common capture schema drifted"
            )
        common_array_sha256: dict[str, str] = {}
        for name in COMMON_CAPTURE_ARRAYS:
            current = np.ascontiguousarray(capture[name])
            previous = np.ascontiguousarray(half_width[name])
            if not _arrays_bitwise_equal(current, previous):
                raise BenchmarkValidationError(
                    f"feature2 recovery full/half-width output drifted: {name}"
                )
            common_array_sha256[name] = sha256(current.tobytes()).hexdigest()

        layer0_index_owners = np.ascontiguousarray(
            capture["layer0_index_cache_owners_bfloat16_bits"]
        )
        positions = np.arange(8_155, dtype=np.int64)
        rows = positions % 512
        candidate_prompt_cache = np.ascontiguousarray(
            layer0_index_owners[rows // 256, positions // 512, rows % 256]
        )
        candidate_prompt_cache_sha256 = sha256(
            candidate_prompt_cache.tobytes()
        ).hexdigest()
        cache_mismatch_coordinates = np.ascontiguousarray(
            np.argwhere(candidate_prompt_cache != accepted_prompt_cache)
        )
        cache_mismatch_coordinates_sha256 = sha256(
            cache_mismatch_coordinates.tobytes()
        ).hexdigest()
        if (
            candidate_prompt_cache.shape != (8_155, 128)
            or candidate_prompt_cache.dtype != np.uint16
            or candidate_prompt_cache_sha256 != CANDIDATE_LOGICAL_CACHE_SHA256
            or cache_mismatch_coordinates.shape != (71, 2)
            or cache_mismatch_coordinates_sha256 != CACHE_MISMATCH_COORDINATES_SHA256
            or np.unique(cache_mismatch_coordinates[:, 0]).size != 71
            or int(cache_mismatch_coordinates[:, 1].max()) != 63
            or cache_mismatch_coordinates[0].tolist() != [113, 35]
            or int(candidate_prompt_cache[113, 35]) != 47_091
            or int(accepted_prompt_cache[113, 35]) != 47_092
        ):
            raise BenchmarkValidationError(
                "feature2 recovery candidate/DB518 cache localization drifted"
            )

        normalized = np.ascontiguousarray(
            capture["current_normalized_hidden_owners_bfloat16_bits"]
        )
        q_a = np.ascontiguousarray(capture["current_q_a_state_owners_bfloat16_bits"])
        query = np.ascontiguousarray(capture["current_dsa_query_owners"])
        head_weights = np.ascontiguousarray(capture["current_dsa_head_weights_owners"])
        expected_normalized = np.ascontiguousarray(
            expected["accepted__normalized_hidden"]
        )
    normalized_owner_mismatches = []
    normalized_owner_indices = []
    for owner in range(2):
        local = normalized[owner].reshape(expected_normalized.shape)
        mismatches = np.argwhere(local != expected_normalized).reshape(-1)
        normalized_owner_mismatches.append(int(mismatches.size))
        normalized_owner_indices.append([int(value) for value in mismatches])
    if (
        not _arrays_bitwise_equal(normalized[0], normalized[1])
        or not _arrays_bitwise_equal(q_a[0], q_a[1])
        or not _arrays_bitwise_equal(query[0], query[1])
        or not _arrays_bitwise_equal(head_weights[0], head_weights[1])
        or normalized_owner_mismatches != [1, 1]
        or normalized_owner_indices != [[2795], [2795]]
        or [int(normalized[owner, 0, 2795]) for owner in range(2)] != [48422, 48422]
        or int(expected_normalized[2795]) != 48423
    ):
        raise BenchmarkValidationError(
            "feature2 recovery sealed-boundary localization drifted"
        )

    memory = runner.get("memory", {})
    generated_code_size = main.get("memory_analysis", {}).get(
        "generated_code_size_in_bytes"
    )
    cleanup = validate_feature2_in_process_cleanup(
        memory.get("after_cleanup", []),
        generated_code_size_bytes=generated_code_size,
    )
    if (
        cleanup.get("mode") != "generated_code_resident_until_process_exit"
        or cleanup.get("bytes_in_use_per_device") != [66_017_792, 66_017_792]
        or cleanup.get("generated_code_size_bytes") != 64_264_704
        or cleanup.get("residual_over_generated_code_bytes") != 1_753_088
    ):
        raise BenchmarkValidationError("feature2 recovery cleanup drifted")
    source_census = source / "census_post.txt"
    if _sha256_file(source_census) != SOURCE_POST_CENSUS_SHA256:
        raise BenchmarkValidationError("feature2 recovery source census drifted")
    source_hosts = _eight_census_hosts(source_census)

    connection = sqlite3.connect(f"file:{Path(results_db)}?mode=ro", uri=True)
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    max_run_id = connection.execute("SELECT max(run_id) FROM runs").fetchone()[0]
    source_pattern = f"%{SOURCE_TAG}%"
    source_matches = connection.execute(
        "SELECT count(*) FROM runs WHERE model LIKE ? OR coalesce(note, '') LIKE ? "
        "OR coalesce(env_json, '') LIKE ?",
        (source_pattern, source_pattern, source_pattern),
    ).fetchone()[0]
    recovery_pattern = f"%{recovery_tag}%"
    recovery_matches = connection.execute(
        "SELECT count(*) FROM runs WHERE model LIKE ? OR coalesce(note, '') LIKE ? "
        "OR coalesce(env_json, '') LIKE ?",
        (recovery_pattern, recovery_pattern, recovery_pattern),
    ).fetchone()[0]
    connection.close()
    if integrity != "ok" or source_matches != 0 or recovery_matches != 0:
        raise BenchmarkValidationError(
            "feature2 recovery DB integrity/no-row contract drifted"
        )

    summary = json.loads((source / "summary.json").read_text())
    expected_summary_cleanup = cleanup | {
        "post_process_authenticated_zero_work_hosts": 8,
        "post_process_census_sha256": SOURCE_POST_CENSUS_SHA256,
        "terminal_cleanup_gate": "authenticated post-process 8/8 zero work",
    }
    if (
        summary.get("status") != "NUMERICAL_REJECTED"
        or summary.get("exact") is not False
        or summary.get("mismatch_counts") != EXPECTED_MISMATCH_COUNTS
        or summary.get("main_execution_count") != 1
        or summary.get("numerical_claim") is not False
        or summary.get("performance_claim") is not False
        or summary.get("cleanup") != expected_summary_cleanup
    ):
        raise BenchmarkValidationError("feature2 recovery source summary drifted")

    return {
        "artifact_kind": (
            "greenfield_pp16_feature2_full_width_rejection_local_authentication"
        ),
        "cleanup": cleanup,
        "db518_cache_localization": {
            "accepted_cache_sha256": DB518_CACHE_SHA256,
            "actual_bits": 47_091,
            "candidate_cache_sha256": CANDIDATE_LOGICAL_CACHE_SHA256,
            "earliest_hidden_index": 35,
            "earliest_position": 113,
            "expected_bits": 47_092,
            "manifest_sha256": DB518_MANIFEST_SHA256,
            "mismatch_coordinate_count": int(cache_mismatch_coordinates.shape[0]),
            "mismatch_coordinates_sha256": cache_mismatch_coordinates_sha256,
            "mismatch_position_count": int(
                np.unique(cache_mismatch_coordinates[:, 0]).size
            ),
            "maximum_hidden_index": int(cache_mismatch_coordinates[:, 1].max()),
            "tensor_file_sha256": DB518_TENSOR_FILE_SHA256,
        },
        "comparison": comparison,
        "full_width_vs_half_width": {
            "bitwise_equal": True,
            "common_array_count": len(common_array_sha256),
            "common_array_sha256": common_array_sha256,
            "half_width_capture_sha256": HALF_WIDTH_CAPTURE_SHA256,
        },
        "database": {
            "integrity": integrity,
            "max_run_id": max_run_id,
            "recovery_tag_matches": recovery_matches,
            "source_tag_matches": source_matches,
        },
        "hlo": {
            "canonical_sha256": SOURCE_MAIN_CANONICAL_HLO_SHA256,
            "raw_optimized_sha256": SOURCE_MAIN_RAW_OPTIMIZED_HLO_SHA256,
            "stablehlo_sha256": SOURCE_MAIN_STABLEHLO_SHA256,
        },
        "ledger_entries": ledger,
        "normalized_hidden_localization": {
            "actual_bits_per_owner": [48422, 48422],
            "expected_bits": 48423,
            "hidden_index": 2795,
            "mismatches_per_owner": normalized_owner_mismatches,
            "owners_bitwise_equal": True,
        },
        "source_census_hosts": source_hosts,
        "source_code_hash": SOURCE_CODE_HASH,
        "source_ledger_sha256": SOURCE_LEDGER_SHA256,
        "source_tag": SOURCE_TAG,
        "status": "NUMERICAL_REJECTED",
        "terminal": _terminal_authentication(source, source_remote),
    }
