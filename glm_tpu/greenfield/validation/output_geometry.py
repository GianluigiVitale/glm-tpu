"""Terminal recomputation for the protected output-geometry discriminator."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarking import (
    ACCEPTED_TP32_MODEL_AXIS_RECIPE,
    DENSE_RMS_SOURCE_CODE_HASH,
    DENSE_RMS_SOURCE_NPZ_SHA256,
    DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256,
    DENSE_RMS_SOURCE_RUNNER_SHA256,
    DENSE_RMS_SOURCE_SUCCESS_SHA256,
    DENSE_RMS_SOURCE_SUMMARY_SHA256,
    DENSE_RMS_SOURCE_TAG,
    M1_AUTO_BF16_LAYOUT,
    M32_BF16_LAYOUT,
    OUTPUT_GEOMETRY_DENSE_ROW_RAW_SHA256,
    OUTPUT_GEOMETRY_INVERSE_RAW_SHA256,
    array_sha256,
    compare_output_geometry_arrays,
    load_output_geometry_inputs,
    validate_output_geometry_hlo,
    validate_output_geometry_stablehlo,
)
from ..topology.discover import validate_target_v4_64
from ..types import PhysicalTopology
from .strategy_nd_dense_replay import (
    EXPECTED_FLEET_LOCAL_DEVICE_IDS,
    EXPECTED_HOSTNAMES,
    EXPECTED_MODEL_AXIS_DEVICE_IDS,
    EXPECTED_TOPOLOGY_HASH,
)
from .strategy_nd_dense_rms_replay import (
    _file_sha256,
    _is_int_list,
    _is_sha256,
    _validate_rms_source_files,
)


EXPECTED_SOURCE = {
    "dense_row_raw_sha256": OUTPUT_GEOMETRY_DENSE_ROW_RAW_SHA256,
    "inverse_raw_sha256": OUTPUT_GEOMETRY_INVERSE_RAW_SHA256,
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
    "association_fingerprint",
    "association_output_geometry",
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
    "comparison",
    "diagnostic_only",
    "fleet_hashes",
    "fleet_hlo_hashes",
    "fleet_stablehlo_hashes",
    "member_device_ids",
    "optimized_hlo_contract",
    "optimized_hlo_sha256",
    "output_layouts",
    "performance_claim",
    "source",
    "stablehlo_contract",
    "stablehlo_sha256",
}
EXPECTED_CAPTURE_FIELDS = {
    "invocation_count",
    "local_replica_sha256",
    "output_sha256",
    "repeated_local_replica_sha256",
    "repeated_output_sha256",
}
OUTPUT_NAMES = ("m32_control", "m1_auto", "m1_m32_tile")


def _validate_host_records(
    records: list[dict[str, Any]],
    paths: list[Path],
    *,
    expected_code_hash: str,
    expected_run_tag: str,
) -> None:
    if len(records) != 8:
        raise ValueError("output geometry requires eight host records")
    topologies: list[PhysicalTopology] = []
    for launch_index, (path, record) in enumerate(
        zip(paths, records, strict=True)
    ):
        if path.name != f"collective.rank{launch_index}.json":
            raise ValueError("output-geometry host filename coverage drifted")
        if set(record) != EXPECTED_RECORD_FIELDS:
            raise ValueError("output-geometry top-level schema drifted")
        try:
            captured = datetime.fromisoformat(record["captured_utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("output-geometry timestamp drifted") from error
        if (
            captured.tzinfo is None
            or captured.utcoffset() != timezone.utc.utcoffset(None)
        ):
            raise ValueError("output-geometry timestamp is not UTC")
        if not (
            type(record["captured_utc"]) is str
            and type(record["launch_process_id"]) is int
            and record["launch_process_id"] == launch_index
            and type(record["jax_process_index"]) is int
            and 0 <= record["jax_process_index"] < 8
            and type(record["hostname"]) is str
            and record["hostname"] == EXPECTED_HOSTNAMES[launch_index]
            and type(record["jax_version"]) is str
            and bool(record["jax_version"])
            and type(record["run_tag"]) is str
            and record["run_tag"] == expected_run_tag
            and type(record["code_hash"]) is str
            and record["code_hash"] == expected_code_hash
            and record["mode"] == "strategy_nd_output_geometry"
            and type(record["schema_version"]) is int
            and record["schema_version"] == 6
            and record["mechanism_only"] is True
            and record["association_dense_replay"] is None
            and record["association_fingerprint"] is None
            and record["matrix"] == []
            and type(record["topology"]) is dict
            and record["topology_hash"] == EXPECTED_TOPOLOGY_HASH
        ):
            raise ValueError("output-geometry host provenance drifted")
        topology = PhysicalTopology.from_dict(record["topology"])
        validate_target_v4_64(topology)
        if (
            topology.topology_hash != EXPECTED_TOPOLOGY_HASH
            or record["fleet_local_device_ids_in_runtime_order"]
            != EXPECTED_FLEET_LOCAL_DEVICE_IDS
            or [
                device.device_id
                for device in topology.devices
                if device.process_index == record["jax_process_index"]
            ]
            != EXPECTED_FLEET_LOCAL_DEVICE_IDS[record["jax_process_index"]]
        ):
            raise ValueError("output-geometry topology ownership drifted")
        topologies.append(topology)
    if len({record["jax_version"] for record in records}) != 1:
        raise ValueError("output-geometry JAX versions differ")
    if {record["jax_process_index"] for record in records} != set(range(8)):
        raise ValueError("output-geometry JAX process coverage drifted")
    if any(
        topology.to_dict() != topologies[0].to_dict()
        for topology in topologies[1:]
    ):
        raise ValueError("output-geometry fleet topologies differ")


def _load_artifacts(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    expected = {
        "accepted_bits": ((6144,), np.dtype(np.uint16)),
        "m1_auto_bits": ((1, 6144), np.dtype(np.uint16)),
        "m1_m32_tile_bits": ((1, 6144), np.dtype(np.uint16)),
        "m32_control_bits": ((32, 6144), np.dtype(np.uint16)),
    }
    artifact_dir = run_dir / "output_geometry"
    observed_files = {
        path.name for path in artifact_dir.iterdir() if path.is_file()
    }
    if observed_files != {
        "accepted_bits.npy",
        "comparison.json",
        "m1_auto_bits.npy",
        "m1_m32_tile_bits.npy",
        "m32_control_bits.npy",
        "manifest.json",
    }:
        raise ValueError("output-geometry artifact file set drifted")
    if set(manifest) != set(expected):
        raise ValueError("output-geometry artifact set drifted")
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected.items():
        record = manifest[name]
        path = run_dir / "output_geometry" / f"{name}.npy"
        value = np.load(path, allow_pickle=False)
        if not (
            type(record) is dict
            and set(record)
            == {"array_sha256", "dtype", "file", "file_sha256", "shape"}
            and record["file"] == path.name
            and record["shape"] == list(shape)
            and record["dtype"] == dtype.str
            and value.shape == shape
            and value.dtype == dtype
            and record["file_sha256"] == _file_sha256(path)
            and record["array_sha256"] == array_sha256(value)
        ):
            raise ValueError(f"output-geometry artifact drifted: {name}")
        arrays[name] = np.ascontiguousarray(value)
    return arrays


def _validate_capture(
    capture: object,
    outputs: dict[str, np.ndarray],
) -> None:
    hashes = {
        "m32_control": array_sha256(outputs["m32_control_bits"]),
        "m1_auto": array_sha256(outputs["m1_auto_bits"]),
        "m1_m32_tile": array_sha256(outputs["m1_m32_tile_bits"]),
    }
    if not (
        type(capture) is dict
        and set(capture) == EXPECTED_CAPTURE_FIELDS
        and type(capture["invocation_count"]) is int
        and capture["invocation_count"] == 2
        and capture["output_sha256"] == hashes
        and capture["repeated_output_sha256"] == hashes
    ):
        raise ValueError("output-geometry deterministic capture drifted")
    for field in ("local_replica_sha256", "repeated_local_replica_sha256"):
        groups = capture[field]
        if not (
            type(groups) is dict
            and set(groups) == set(OUTPUT_NAMES)
            and all(
                type(groups[name]) is list
                and len(groups[name]) == 4
                and all(
                    type(value) is str and value == hashes[name]
                    for value in groups[name]
                )
                for name in OUTPUT_NAMES
            )
        ):
            raise ValueError(f"output-geometry local capture drifted: {field}")


def validate_output_geometry_replay(
    run_dir: Path,
    *,
    expected_code_hash: str,
    expected_run_tag: str,
) -> dict[str, Any]:
    """Reload every source, HLO, fleet record and numerical artifact."""

    host_paths = sorted((run_dir / "host_records").glob("collective.rank*.json"))
    records = [json.loads(path.read_text()) for path in host_paths]
    _validate_host_records(
        records,
        host_paths,
        expected_code_hash=expected_code_hash,
        expected_run_tag=expected_run_tag,
    )
    items = [record["association_output_geometry"] for record in records]
    if any(type(item) is not dict or set(item) != EXPECTED_REPLAY_FIELDS for item in items):
        raise ValueError("output-geometry nested schema drifted")
    reference = items[0]
    for field in EXPECTED_REPLAY_FIELDS - {"artifact_manifest"}:
        if any(item[field] != reference[field] for item in items[1:]):
            raise ValueError(f"output-geometry fleet field differs: {field}")
    if not (
        _is_int_list(
            reference["accepted_model_axis_device_ids"],
            list(EXPECTED_MODEL_AXIS_DEVICE_IDS),
        )
        and reference["accepted_model_axis_recipe"]
        == ACCEPTED_TP32_MODEL_AXIS_RECIPE
        and _is_int_list(reference["member_device_ids"], list(range(32)))
        and reference["diagnostic_only"] is True
        and reference["performance_claim"] is False
        and reference["source"] == EXPECTED_SOURCE
        and reference["output_layouts"]
        == {
            "m1_auto": M1_AUTO_BF16_LAYOUT,
            "m1_m32_tile": M32_BF16_LAYOUT,
            "m32_control": M32_BF16_LAYOUT,
        }
        and _is_sha256(reference["stablehlo_sha256"])
        and _is_sha256(reference["optimized_hlo_sha256"])
    ):
        raise ValueError("output-geometry execution identity drifted")
    for field, expected in (
        ("fleet_hlo_hashes", reference["optimized_hlo_sha256"]),
        ("fleet_stablehlo_hashes", reference["stablehlo_sha256"]),
    ):
        values = reference[field]
        if not (
            type(values) is list
            and len(values) == 8
            and all(type(value) is str and value == expected for value in values)
        ):
            raise ValueError(f"output-geometry fleet HLO agreement failed: {field}")
    if not (
        type(reference["fleet_hashes"]) is dict
        and all(
            type(values) is list
            and len(values) == 8
            and len(set(values)) == 1
            and all(_is_sha256(value) for value in values)
            for values in reference["fleet_hashes"].values()
        )
    ):
        raise ValueError("output-geometry fleet tensor agreement failed")

    owners = [
        (record, item)
        for record, item in zip(records, items, strict=True)
        if item["artifact_manifest"]
    ]
    if len(owners) != 1 or owners[0][0]["jax_process_index"] != 0:
        raise ValueError("output-geometry artifacts need one process-zero owner")
    manifest_path = run_dir / "output_geometry" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest != owners[0][1]["artifact_manifest"]:
        raise ValueError("output-geometry manifest differs from owner record")
    arrays = _load_artifacts(run_dir, manifest)
    source_inputs = load_output_geometry_inputs(
        run_dir / "source_rms" / "dense_partial_capture.npz",
        EXPECTED_MODEL_AXIS_DEVICE_IDS,
    )
    _validate_rms_source_files(run_dir / "source_rms")
    if not np.array_equal(arrays["accepted_bits"], source_inputs.accepted_bits):
        raise ValueError("output-geometry accepted artifact drifted from source")
    outputs = {
        "m32_control": arrays["m32_control_bits"],
        "m1_auto": arrays["m1_auto_bits"],
        "m1_m32_tile": arrays["m1_m32_tile_bits"],
    }
    comparison = compare_output_geometry_arrays(outputs, source_inputs.accepted_bits)
    stored_comparison = json.loads(
        (run_dir / "output_geometry" / "comparison.json").read_text()
    )
    if reference["comparison"] != comparison or stored_comparison != comparison:
        raise ValueError("output-geometry comparison failed recomputation")
    _validate_capture(reference["capture"], arrays)

    expected_hashes = {
        "accepted": array_sha256(source_inputs.accepted_bits),
        "carried": array_sha256(source_inputs.carried_bits),
        "dense": array_sha256(source_inputs.dense_bits),
        "inverse": array_sha256(source_inputs.inverse),
        "m1_auto": array_sha256(outputs["m1_auto"]),
        "m1_m32_tile": array_sha256(outputs["m1_m32_tile"]),
        "m32_control": array_sha256(outputs["m32_control"]),
        "source_file": DENSE_RMS_SOURCE_NPZ_SHA256,
        "weight": array_sha256(source_inputs.weight_bits),
    }
    if reference["fleet_hashes"] != {
        name: [digest] * 8 for name, digest in expected_hashes.items()
    }:
        raise ValueError("output-geometry fleet hashes failed recomputation")

    label = "strategy_nd_output_geometry_bfloat16_m32_m1_tiled"
    hlo_dir = run_dir / "hlo"
    expected_hlo_files = {
        f"{label}.hlo_contract.json",
        f"{label}.hlo_prevalidation.json",
        f"{label}.optimized_hlo.txt",
        f"{label}.stablehlo.mlir",
    }
    if {
        path.name for path in hlo_dir.iterdir() if path.is_file()
    } != expected_hlo_files:
        raise ValueError("output-geometry HLO file set drifted")
    stablehlo_path = run_dir / "hlo" / f"{label}.stablehlo.mlir"
    optimized_hlo_path = run_dir / "hlo" / f"{label}.optimized_hlo.txt"
    stablehlo = stablehlo_path.read_text()
    optimized_hlo = optimized_hlo_path.read_text()
    stable_contract = validate_output_geometry_stablehlo(stablehlo)
    optimized_contract = validate_output_geometry_hlo(optimized_hlo)
    if not (
        _file_sha256(stablehlo_path) == reference["stablehlo_sha256"]
        and _file_sha256(optimized_hlo_path) == reference["optimized_hlo_sha256"]
        and reference["stablehlo_contract"] == stable_contract
        and reference["optimized_hlo_contract"] == optimized_contract
    ):
        raise ValueError("output-geometry HLO evidence drifted")
    prevalidation = json.loads(
        (run_dir / "hlo" / f"{label}.hlo_prevalidation.json").read_text()
    )
    if prevalidation != {
        "optimized_hlo_sha256": reference["optimized_hlo_sha256"],
        "performance_claim": False,
        "stablehlo_sha256": reference["stablehlo_sha256"],
        "validated": False,
    }:
        raise ValueError("output-geometry prevalidation record drifted")
    contract = json.loads(
        (run_dir / "hlo" / f"{label}.hlo_contract.json").read_text()
    )
    if contract != {
        "optimized": optimized_contract,
        "stablehlo": stable_contract,
        "valid": True,
    }:
        raise ValueError("output-geometry HLO contract artifact drifted")

    decisive = comparison["pairwise"]["m1_m32_tile_vs_m32_control"]
    accepted_control = comparison["pairwise"]["m32_control_vs_accepted"]
    return {
        "artifact_kind": "glm52_strategy_nd_output_geometry_replay",
        "classification": comparison["classification"],
        "code_hash": expected_code_hash,
        "diagnostic_only": True,
        "elementwise_exact": decisive["elementwise_exact"],
        "expected_raw_sha256": decisive["expected_raw_sha256"],
        "first_mismatch_index": decisive["first_mismatch_index"],
        "m32_control_exact_accepted": accepted_control["elementwise_exact"],
        "m32_control_mismatch_count": accepted_control["mismatch_count"],
        "mismatch_count": decisive["mismatch_count"],
        "observed_raw_sha256": decisive["observed_raw_sha256"],
        "optimized_hlo_sha256": reference["optimized_hlo_sha256"],
        "performance_claim": False,
        "run_tag": expected_run_tag,
        "source": dict(reference["source"]),
        "stablehlo_sha256": reference["stablehlo_sha256"],
        "topology_hash": EXPECTED_TOPOLOGY_HASH,
    }
