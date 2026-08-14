"""Terminal recomputation for the protected StrategyND-to-RMS replay."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarking import (
    ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    ACCEPTED_TP32_MODEL_AXIS_RECIPE,
    DENSE_RMS_SOURCE_CODE_HASH,
    DENSE_RMS_SOURCE_NPZ_SHA256,
    DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256,
    DENSE_RMS_SOURCE_RUNNER_SHA256,
    DENSE_RMS_SOURCE_SUCCESS_SHA256,
    DENSE_RMS_SOURCE_SUMMARY_SHA256,
    DENSE_RMS_SOURCE_TAG,
    STRATEGY_ND_ALGORITHM,
    array_sha256,
    load_dense_rms_inputs,
    model_axis_to_physical_input_bits,
    validate_strategy_nd_dense_rms_hlo,
)
from ..sharding.stablehlo_dense_convolution import (
    validate_strategy_nd_dense_rms_stablehlo,
)
from ..topology.discover import validate_target_v4_64
from ..types import PhysicalTopology
from .strategy_nd_dense_replay import (
    EXPECTED_FLEET_LOCAL_DEVICE_IDS,
    EXPECTED_HOSTNAMES,
    EXPECTED_MODEL_AXIS_DEVICE_IDS,
    EXPECTED_TOPOLOGY_HASH,
    _validate_source_files,
)


EXPECTED_SOURCE = {
    "db550_npz_sha256": ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    "db550_raw_sha256": ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    "db550_run_id": 550,
    "db550_tag": (
        "greenfield_legacy_layer0_dense_partials_p8155_"
        "20260814T100132090917640Z"
    ),
    "rms_code_hash": DENSE_RMS_SOURCE_CODE_HASH,
    "rms_npz_sha256": DENSE_RMS_SOURCE_NPZ_SHA256,
    "rms_remote_objects_sha256": DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256,
    "rms_runner_sha256": DENSE_RMS_SOURCE_RUNNER_SHA256,
    "rms_success_sha256": DENSE_RMS_SOURCE_SUCCESS_SHA256,
    "rms_summary_sha256": DENSE_RMS_SOURCE_SUMMARY_SHA256,
    "rms_tag": DENSE_RMS_SOURCE_TAG,
}
EXPECTED_RECORD_FIELDS = {
    "association_dense_replay",
    "association_dense_rms_replay",
    "association_fingerprint",
    "captured_utc",
    "code_hash",
    "fleet_local_device_ids_in_runtime_order",
    "hostname",
    "jax_process_index",
    "jax_version",
    "launch_process_id",
    "matrix",
    "mechanism_only",
    "mode",
    "run_tag",
    "schema_version",
    "topology",
    "topology_hash",
}
EXPECTED_REPLAY_FIELDS = {
    "accepted_model_axis_device_ids",
    "accepted_model_axis_recipe",
    "artifact_manifest",
    "capture",
    "collective_algorithm",
    "collective_groups",
    "comparison",
    "diagnostic_only",
    "fleet_hashes",
    "fleet_hlo_hashes",
    "fleet_stablehlo_hashes",
    "member_device_ids",
    "optimized_hlo_contract",
    "optimized_hlo_sha256",
    "performance_claim",
    "source",
    "stablehlo_contract",
    "stablehlo_sha256",
}
EXPECTED_CAPTURE_FIELDS = {
    "determinism_repeat_invocations",
    "distributed_input_sha256",
    "invocation_count",
    "local_replica_output_sha256",
    "output_bits_sha256",
    "repeated_local_replica_output_sha256",
    "repeated_output_bits_sha256",
}
EXPECTED_COMPARISON_FIELDS = {
    "classification",
    "elementwise_exact",
    "expected_hidden_2795_bfloat16_bits",
    "expected_raw_sha256",
    "first_mismatch_index",
    "max_absolute_error",
    "mean_absolute_error",
    "mismatch_count",
    "observed_hidden_2795_bfloat16_bits",
    "observed_raw_sha256",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_int_list(value: object, expected: list[int]) -> bool:
    return bool(
        type(value) is list
        and all(type(item) is int for item in value)
        and value == expected
    )


def _validate_capture_record(
    capture: object,
    expected_physical: np.ndarray,
    output: np.ndarray,
) -> None:
    distributed = np.ascontiguousarray(
        np.broadcast_to(expected_physical[:, None, :], (32, 32, 6144))
    )
    output_sha = array_sha256(output)
    if not (
        type(capture) is dict
        and set(capture) == EXPECTED_CAPTURE_FIELDS
        and type(capture["determinism_repeat_invocations"]) is int
        and capture["determinism_repeat_invocations"] == 1
        and type(capture["invocation_count"]) is int
        and capture["invocation_count"] == 2
        and capture["distributed_input_sha256"] == array_sha256(distributed)
        and _is_sha256(capture["distributed_input_sha256"])
        and capture["output_bits_sha256"] == output_sha
        and capture["repeated_output_bits_sha256"] == output_sha
        and _is_sha256(capture["output_bits_sha256"])
        and _is_sha256(capture["repeated_output_bits_sha256"])
        and type(capture["local_replica_output_sha256"]) is list
        and len(capture["local_replica_output_sha256"]) == 4
        and all(
            type(value) is str and value == output_sha
            for value in capture["local_replica_output_sha256"]
        )
        and type(capture["repeated_local_replica_output_sha256"]) is list
        and capture["repeated_local_replica_output_sha256"]
        == capture["local_replica_output_sha256"]
    ):
        raise ValueError("dense RMS deterministic capture drifted")


def _terminal_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or "=" not in line:
            raise ValueError("dense RMS source SUCCESS is malformed")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ValueError("dense RMS source SUCCESS has duplicate fields")
        values[key] = value
    return values


def _validate_rms_source_files(source_dir: Path) -> Any:
    expected = {
        "SUCCESS": DENSE_RMS_SOURCE_SUCCESS_SHA256,
        "dense_partial_capture.npz": DENSE_RMS_SOURCE_NPZ_SHA256,
        "remote_objects.json": DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256,
        "runner.json": DENSE_RMS_SOURCE_RUNNER_SHA256,
        "summary.json": DENSE_RMS_SOURCE_SUMMARY_SHA256,
    }
    observed = {path.name for path in source_dir.iterdir() if path.is_file()}
    if observed != set(expected):
        raise ValueError("dense RMS sealed source file set drifted")
    for name, digest in expected.items():
        if _file_sha256(source_dir / name) != digest:
            raise ValueError(f"dense RMS sealed source file drifted: {name}")
    runner = json.loads((source_dir / "runner.json").read_text())
    summary = json.loads((source_dir / "summary.json").read_text())
    success = _terminal_values(source_dir / "SUCCESS")
    if not (
        runner.get("artifact_kind") == "glm52_layer0_dense_partial_capture"
        and runner.get("capture_partials") is True
        and runner.get("classification") == "accepted_m32_dense_partials_captured"
        and runner.get("code_hash") == DENSE_RMS_SOURCE_CODE_HASH
        and runner.get("compile_rows") == 32
        and runner.get("dense_envelope") is True
        and runner.get("final_dense_layout") is True
        and runner.get("performance_claim") is False
        and runner.get("status") == "SUCCESS"
        and summary.get("artifact_kind") == runner["artifact_kind"]
        and summary.get("classification") == runner["classification"]
        and summary.get("code_hash") == DENSE_RMS_SOURCE_CODE_HASH
        and summary.get("runner_sha256") == DENSE_RMS_SOURCE_RUNNER_SHA256
        and summary.get("tensor_sha256") == DENSE_RMS_SOURCE_NPZ_SHA256
        and summary.get("results_db_run_id") is None
        and summary.get("status") == "SUCCESS"
        and success.get("artifact_kind") == runner["artifact_kind"]
        and success.get("classification") == runner["classification"]
        and success.get("code_hash") == DENSE_RMS_SOURCE_CODE_HASH
        and success.get("runner_sha256") == DENSE_RMS_SOURCE_RUNNER_SHA256
        and success.get("tensor_sha256") == DENSE_RMS_SOURCE_NPZ_SHA256
        and success.get("results_db_run_id") == "none"
        and success.get("performance_claim") == "false"
    ):
        raise ValueError("dense RMS sealed source semantics drifted")
    return load_dense_rms_inputs(source_dir / "dense_partial_capture.npz")


def _recompute_comparison(
    observed: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    mismatch_indices = np.flatnonzero(observed != expected)
    first = None if not len(mismatch_indices) else int(mismatch_indices[0])
    observed_values = (
        observed.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    expected_values = (
        expected.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    error = np.abs(observed_values - expected_values)
    observed_sha = _raw_sha256(observed)
    return {
        "classification": (
            "global_strategy_nd_rms_exact_accepted"
            if first is None
            else (
                "global_strategy_nd_rms_matches_db548_control"
                if observed_sha
                == "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
                else "global_strategy_nd_rms_nonexact_new_result"
            )
        ),
        "elementwise_exact": first is None,
        "expected_hidden_2795_bfloat16_bits": int(expected[2795]),
        "expected_raw_sha256": _raw_sha256(expected),
        "first_mismatch_index": first,
        "max_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": int(len(mismatch_indices)),
        "observed_hidden_2795_bfloat16_bits": int(observed[2795]),
        "observed_raw_sha256": observed_sha,
    }


def validate_strategy_nd_dense_rms_replay(
    run_dir: Path,
    *,
    expected_code_hash: str,
    expected_run_tag: str,
) -> dict[str, Any]:
    """Reload every fleet/source/artifact byte and recompute the verdict."""

    host_paths = sorted((run_dir / "host_records").glob("collective.rank*.json"))
    records = [json.loads(path.read_text()) for path in host_paths]
    if len(records) != 8:
        raise ValueError("dense RMS replay requires eight host records")
    for launch_index, (path, record) in enumerate(
        zip(host_paths, records, strict=True)
    ):
        if path.name != f"collective.rank{launch_index}.json":
            raise ValueError("dense RMS host-record filename coverage drifted")
        if set(record) != EXPECTED_RECORD_FIELDS:
            raise ValueError("dense RMS top-level record schema drifted")
        if type(record["captured_utc"]) is not str:
            raise ValueError("dense RMS captured timestamp type drifted")
        try:
            captured = datetime.fromisoformat(record["captured_utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("dense RMS captured timestamp drifted") from error
        if captured.tzinfo is None or captured.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("dense RMS captured timestamp is not UTC")
        if not (
            record["launch_process_id"] == launch_index
            and type(record["launch_process_id"]) is int
            and type(record["jax_process_index"]) is int
            and record["hostname"] == EXPECTED_HOSTNAMES[launch_index]
            and type(record["hostname"]) is str
            and record["run_tag"] == expected_run_tag
            and type(record["run_tag"]) is str
            and record["code_hash"] == expected_code_hash
            and type(record["code_hash"]) is str
            and record["mode"] == "strategy_nd_dense_rms_replay"
            and record["schema_version"] == 4
            and type(record["schema_version"]) is int
            and record["mechanism_only"] is True
            and type(record["jax_version"]) is str
            and record["jax_version"]
            and type(record["topology"]) is dict
            and _is_sha256(record["topology_hash"])
        ):
            raise ValueError("dense RMS host identity/provenance drifted")
    if {record["jax_process_index"] for record in records} != set(range(8)):
        raise ValueError("dense RMS JAX process coverage drifted")
    if len({record["jax_version"] for record in records}) != 1:
        raise ValueError("dense RMS JAX versions differ")
    topologies = []
    for record in records:
        topology = PhysicalTopology.from_dict(record["topology"])
        validate_target_v4_64(topology)
        process_index = record["jax_process_index"]
        if (
            topology.topology_hash != EXPECTED_TOPOLOGY_HASH
            or record["topology_hash"] != EXPECTED_TOPOLOGY_HASH
            or type(record["fleet_local_device_ids_in_runtime_order"]) is not list
            or any(
                not _is_int_list(value, expected)
                for value, expected in zip(
                    record["fleet_local_device_ids_in_runtime_order"],
                    EXPECTED_FLEET_LOCAL_DEVICE_IDS,
                    strict=True,
                )
            )
            or len(record["fleet_local_device_ids_in_runtime_order"]) != 8
            or [
                device.device_id
                for device in topology.devices
                if device.process_index == process_index
            ]
            != EXPECTED_FLEET_LOCAL_DEVICE_IDS[process_index]
        ):
            raise ValueError("dense RMS topology/runtime ownership drifted")
        topologies.append(topology)
    if any(
        topology.to_dict() != topologies[0].to_dict()
        for topology in topologies[1:]
    ):
        raise ValueError("dense RMS fleet topologies differ")
    if any(
        record["association_dense_replay"] is not None
        or record["association_fingerprint"] is not None
        or record["matrix"]
        for record in records
    ):
        raise ValueError("dense RMS record contains another benchmark mode")

    items = [record["association_dense_rms_replay"] for record in records]
    if any(not isinstance(item, dict) or set(item) != EXPECTED_REPLAY_FIELDS for item in items):
        raise ValueError("dense RMS nested replay schema drifted")
    reference = items[0]
    for field in EXPECTED_REPLAY_FIELDS - {"artifact_manifest"}:
        if any(item[field] != reference[field] for item in items[1:]):
            raise ValueError(f"dense RMS fleet field differs: {field}")
    if reference["source"] != EXPECTED_SOURCE:
        raise ValueError("dense RMS source provenance drifted")
    if (
        not _is_int_list(
            reference["accepted_model_axis_device_ids"],
            list(EXPECTED_MODEL_AXIS_DEVICE_IDS),
        )
        or reference["accepted_model_axis_recipe"]
        != ACCEPTED_TP32_MODEL_AXIS_RECIPE
        or not _is_int_list(reference["member_device_ids"], list(range(32)))
        or type(reference["collective_groups"]) is not list
        or len(reference["collective_groups"]) != 1
        or not _is_int_list(reference["collective_groups"][0], list(range(32)))
        or reference["collective_algorithm"] != STRATEGY_ND_ALGORITHM
        or reference["diagnostic_only"] is not True
        or reference["performance_claim"] is not False
        or type(reference["accepted_model_axis_recipe"]) is not str
        or type(reference["collective_algorithm"]) is not dict
        or type(reference["source"]) is not dict
        or type(reference["artifact_manifest"]) is not dict
        or type(reference["optimized_hlo_contract"]) is not dict
        or type(reference["stablehlo_contract"]) is not dict
        or not _is_sha256(reference["optimized_hlo_sha256"])
        or not _is_sha256(reference["stablehlo_sha256"])
    ):
        raise ValueError("dense RMS execution identity drifted")
    for name in ("fleet_hashes",):
        if any(
            len(values) != 8 or len(set(values)) != 1
            for values in reference[name].values()
        ):
            raise ValueError(f"dense RMS fleet agreement failed: {name}")
    for name in ("fleet_hlo_hashes", "fleet_stablehlo_hashes"):
        values = reference[name]
        expected = (
            reference["optimized_hlo_sha256"]
            if name == "fleet_hlo_hashes"
            else reference["stablehlo_sha256"]
        )
        if len(values) != 8 or len(set(values)) != 1 or values[0] != expected:
            raise ValueError(f"dense RMS fleet HLO agreement failed: {name}")

    owners = [
        (record, item)
        for record, item in zip(records, items, strict=True)
        if item["artifact_manifest"]
    ]
    if len(owners) != 1 or owners[0][0]["jax_process_index"] != 0:
        raise ValueError("dense RMS artifacts require one JAX-process-zero owner")
    manifest = json.loads((run_dir / "rms_replay" / "manifest.json").read_text())
    if manifest != owners[0][1]["artifact_manifest"]:
        raise ValueError("dense RMS artifact manifest differs from owner record")
    expected_artifacts = {
        "accepted_layer1_bits": ((6144,), np.dtype(np.uint16)),
        "hardware_layer1_bits": ((6144,), np.dtype(np.uint16)),
        "physical_input_bits": ((32, 6144), np.dtype(np.uint16)),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected_artifacts.items():
        record = manifest.get(name)
        path = run_dir / "rms_replay" / f"{name}.npy"
        value = np.load(path, allow_pickle=False)
        if not (
            isinstance(record, dict)
            and set(record)
            == {"array_sha256", "dtype", "file", "file_sha256", "shape"}
            and record["file"] == path.name
            and tuple(record["shape"]) == shape
            and record["dtype"] == dtype.str
            and value.shape == shape
            and value.dtype == dtype
            and record["file_sha256"] == _file_sha256(path)
            and record["array_sha256"] == array_sha256(value)
        ):
            raise ValueError(f"dense RMS artifact failed recomputation: {name}")
        arrays[name] = np.ascontiguousarray(value)

    model_bits = _validate_source_files(run_dir / "source")
    rms_inputs = _validate_rms_source_files(run_dir / "source_rms")
    expected_physical = model_axis_to_physical_input_bits(
        model_bits, EXPECTED_MODEL_AXIS_DEVICE_IDS
    )
    if not np.array_equal(arrays["physical_input_bits"], expected_physical):
        raise ValueError("dense RMS physical input does not derive from DB550")
    if not np.array_equal(
        arrays["accepted_layer1_bits"], rms_inputs.accepted_layer1_bits
    ):
        raise ValueError("dense RMS accepted target does not derive from source")
    comparison = _recompute_comparison(
        arrays["hardware_layer1_bits"], arrays["accepted_layer1_bits"]
    )
    recorded_comparison = reference["comparison"]
    if not (
        type(recorded_comparison) is dict
        and set(recorded_comparison) == EXPECTED_COMPARISON_FIELDS
        and type(recorded_comparison["classification"]) is str
        and type(recorded_comparison["elementwise_exact"]) is bool
        and type(recorded_comparison["expected_hidden_2795_bfloat16_bits"])
        is int
        and _is_sha256(recorded_comparison["expected_raw_sha256"])
        and (
            recorded_comparison["first_mismatch_index"] is None
            or type(recorded_comparison["first_mismatch_index"]) is int
        )
        and type(recorded_comparison["max_absolute_error"]) is float
        and type(recorded_comparison["mean_absolute_error"]) is float
        and type(recorded_comparison["mismatch_count"]) is int
        and type(recorded_comparison["observed_hidden_2795_bfloat16_bits"])
        is int
        and _is_sha256(recorded_comparison["observed_raw_sha256"])
    ):
        raise ValueError("dense RMS numerical comparison schema drifted")
    if (
        json.loads((run_dir / "rms_replay" / "comparison.json").read_text())
        != comparison
        or recorded_comparison != comparison
    ):
        raise ValueError("dense RMS numerical comparison failed recomputation")
    _validate_capture_record(
        reference["capture"],
        expected_physical,
        arrays["hardware_layer1_bits"],
    )
    expected_fleet = {
        "accepted_target": array_sha256(arrays["accepted_layer1_bits"]),
        "output": array_sha256(arrays["hardware_layer1_bits"]),
        "physical_input": array_sha256(arrays["physical_input_bits"]),
        "rms_source_file": DENSE_RMS_SOURCE_NPZ_SHA256,
        "source_file": ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    }
    if {
        key: values[0] for key, values in reference["fleet_hashes"].items()
    } != expected_fleet:
        raise ValueError("dense RMS fleet hashes do not match artifacts")

    label = "strategy_nd_dense_rms_bfloat16_32x6144"
    hlo_dir = run_dir / "hlo"
    stable_path = hlo_dir / f"{label}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{label}.optimized_hlo.txt"
    contract_path = hlo_dir / f"{label}.hlo_contract.json"
    if {path.name for path in hlo_dir.iterdir() if path.is_file()} != {
        stable_path.name,
        optimized_path.name,
        contract_path.name,
    }:
        raise ValueError("dense RMS HLO artifact set drifted")
    stablehlo = stable_path.read_text()
    optimized_hlo = optimized_path.read_text()
    if (
        sha256(stablehlo.encode()).hexdigest() != reference["stablehlo_sha256"]
        or sha256(optimized_hlo.encode()).hexdigest()
        != reference["optimized_hlo_sha256"]
    ):
        raise ValueError("dense RMS HLO SHA-256 drifted")
    stable_contract = validate_strategy_nd_dense_rms_stablehlo(stablehlo)
    hlo_report, algorithm, optimized_contract = validate_strategy_nd_dense_rms_hlo(
        optimized_hlo, tuple(range(32))
    )
    if (
        stable_contract != reference["stablehlo_contract"]
        or optimized_contract != reference["optimized_hlo_contract"]
        or algorithm != STRATEGY_ND_ALGORITHM
        or json.loads(contract_path.read_text())
        != {
            "collective_algorithm": STRATEGY_ND_ALGORITHM,
            "optimized": optimized_contract,
            "stablehlo": stable_contract,
            "valid": True,
        }
        or not hlo_report.valid
    ):
        raise ValueError("dense RMS HLO contract failed terminal replay")
    if any(
        type(comparison[field]) is not float or not math.isfinite(comparison[field])
        for field in ("max_absolute_error", "mean_absolute_error")
    ):
        raise ValueError("dense RMS numerical errors are not finite floats")

    return {
        "artifact_kind": "glm52_strategy_nd_dense_rms_replay",
        "classification": comparison["classification"],
        "code_hash": expected_code_hash,
        "diagnostic_only": True,
        "elementwise_exact": comparison["elementwise_exact"],
        "expected_hidden_2795_bfloat16_bits": comparison[
            "expected_hidden_2795_bfloat16_bits"
        ],
        "expected_raw_sha256": comparison["expected_raw_sha256"],
        "mismatch_count": comparison["mismatch_count"],
        "observed_hidden_2795_bfloat16_bits": comparison[
            "observed_hidden_2795_bfloat16_bits"
        ],
        "observed_raw_sha256": comparison["observed_raw_sha256"],
        "optimized_hlo_sha256": reference["optimized_hlo_sha256"],
        "performance_claim": False,
        "run_tag": expected_run_tag,
        "source": dict(EXPECTED_SOURCE),
        "stablehlo_sha256": reference["stablehlo_sha256"],
        "status": "SUCCESS",
        "topology_hash": EXPECTED_TOPOLOGY_HASH,
    }
