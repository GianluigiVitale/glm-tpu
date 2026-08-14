"""Terminal validation for the protected DB550 StrategyND dense replay."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..topology.discover import validate_target_v4_64
from ..types import PhysicalTopology

from ..benchmarking.association_fingerprint import (
    ACCEPTED_DENSE_PARTIALS_KEYS,
    ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    ACCEPTED_DENSE_PARTIALS_SHAPE,
    ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256,
    ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256,
    ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256,
    ACCEPTED_TP32_MODEL_AXIS_RECIPE,
    STRATEGY_ND_ALGORITHM,
    array_sha256,
    model_axis_to_physical_input_bits,
    replay_db533_strategy_nd_row0_bits,
    validate_strategy_nd_fingerprint_hlo,
)


EXPECTED_MODEL_AXIS_DEVICE_IDS = (
    0, 8, 16, 24, 2, 10, 18, 26,
    4, 12, 20, 28, 6, 14, 22, 30,
    1, 9, 17, 25, 3, 11, 19, 27,
    5, 13, 21, 29, 7, 15, 23, 31,
)
EXPECTED_TOPOLOGY_HASH = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
EXPECTED_HOSTNAMES = tuple(f"t1v-n-6c15e171-w-{index}" for index in range(8))
EXPECTED_FLEET_LOCAL_DEVICE_IDS = [
    list(range(process_index * 4, process_index * 4 + 4))
    for process_index in range(8)
]
EXPECTED_RECORD_FIELDS = {
    "association_dense_replay",
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
    "config",
    "diagnostic_only",
    "fleet_hashes",
    "fleet_hlo_hashes",
    "hlo",
    "member_device_ids",
    "optimized_hlo_sha256",
    "performance_claim",
    "source",
}
EXPECTED_REPLAY_TYPES = {
    "accepted_model_axis_device_ids": list,
    "accepted_model_axis_recipe": str,
    "artifact_manifest": dict,
    "capture": dict,
    "collective_algorithm": dict,
    "collective_groups": list,
    "comparison": dict,
    "config": dict,
    "diagnostic_only": bool,
    "fleet_hashes": dict,
    "fleet_hlo_hashes": list,
    "hlo": dict,
    "member_device_ids": list,
    "optimized_hlo_sha256": str,
    "performance_claim": bool,
    "source": dict,
}
EXPECTED_SOURCE = {
    "accepted_decode_projection_hlo_gzip_sha256": ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256,
    "accepted_decode_projection_hlo_raw_sha256": ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256,
    "accepted_decode_projection_manifest_sha256": ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256,
    "accepted_dense_partials_raw_sha256": ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    "capture_file_sha256": "9b6a4a6d608a0e88f9fbda3bc68be77e2ee43a0dda35a361a32c76f22fda01e6",
    "capture_manifest_sha256": "21c178989ae2fa40df9d34922dc9736c4f4ba87d35dd3f7878a22485ef7184e4",
    "comparison_manifest_sha256": "4238b9dcde7305a9f7a7cf35719eba6fe0b6c14d2a7a31480a8b9c192c245050",
    "comparison_sha256": "92707ccac80a337bcae0c527148fc9133383067d25add36eb0b36f7e7c9198de",
    "db_item_id": 1834,
    "db_run_id": 550,
    "db533_analysis_sha256": "e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108",
    "db533_code_hash": "a9e6307bdad70b883ba82456fbf8f4bdf8db5ac6",
    "db533_hlo_contract_sha256": "966a5dd8be19c409ac616dc194fe8e7dc78ae6c253132c7a73dcdeb8e0c040bc",
    "db533_item_id": 1818,
    "db533_run_id": 533,
    "db533_success_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "db533_summary_sha256": "3ca82073f69fbe56526e1765594c7e8e9a738c73eb2a62b2df6d9a0c4d3136b7",
    "npz_sha256": ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    "remote_objects_sha256": "663adbf1a11c32bbfc28b1030a0c329080d29fcf96fe4a2d77169e64e8858a05",
    "success_sha256": "9605aa5c0f76fd9a5ec9b8e78b1aa720111cbc0633d7b962834004537f321b23",
    "tag": "greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z",
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _load_source(path: Path) -> np.ndarray:
    if not path.is_file() or _file_sha256(path) != ACCEPTED_DENSE_PARTIALS_NPZ_SHA256:
        raise ValueError("sealed DB550 source NPZ SHA-256 drifted")
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != ACCEPTED_DENSE_PARTIALS_KEYS:
            raise ValueError("sealed DB550 source NPZ keys drifted")
        accepted = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[0]])
        db548 = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[1]])
    if (
        accepted.shape != ACCEPTED_DENSE_PARTIALS_SHAPE
        or db548.shape != ACCEPTED_DENSE_PARTIALS_SHAPE
        or accepted.dtype != np.uint16
        or db548.dtype != np.uint16
        or _raw_sha256(accepted) != ACCEPTED_DENSE_PARTIALS_RAW_SHA256
        or _raw_sha256(db548) != ACCEPTED_DENSE_PARTIALS_RAW_SHA256
        or not np.array_equal(accepted, db548)
    ):
        raise ValueError("sealed DB550 source tensor contract drifted")
    return accepted.reshape(32, 6144)


def _validate_source_files(source_dir: Path) -> np.ndarray:
    expected = {
        "SUCCESS": EXPECTED_SOURCE["success_sha256"],
        "capture.json": EXPECTED_SOURCE["capture_file_sha256"],
        "comparison.json": EXPECTED_SOURCE["comparison_sha256"],
        "dense_partials.npz": EXPECTED_SOURCE["npz_sha256"],
        "remote_objects.json": EXPECTED_SOURCE["remote_objects_sha256"],
    }
    observed = {path.name for path in source_dir.iterdir() if path.is_file()}
    if observed != set(expected):
        raise ValueError("sealed DB550 source file set drifted")
    for name, digest in expected.items():
        if _file_sha256(source_dir / name) != digest:
            raise ValueError(f"sealed DB550 source file drifted: {name}")
    capture = json.loads((source_dir / "capture.json").read_text())
    comparison = json.loads((source_dir / "comparison.json").read_text())
    if (
        capture.get("manifest_sha256") != EXPECTED_SOURCE["capture_manifest_sha256"]
        or comparison.get("manifest_sha256")
        != EXPECTED_SOURCE["comparison_manifest_sha256"]
        or comparison.get("classification") != "accepted_dense_partials_exact_db548"
        or not comparison.get("comparison", {}).get("elementwise_exact")
        or comparison.get("comparison", {}).get("mismatch_count") != 0
    ):
        raise ValueError("sealed DB550 capture/comparison semantic contract drifted")
    return _load_source(source_dir / "dense_partials.npz")


def _validate_db533_source_files(source_dir: Path) -> None:
    expected = {
        "SUCCESS": EXPECTED_SOURCE["db533_success_sha256"],
        "analysis.json": EXPECTED_SOURCE["db533_analysis_sha256"],
        "hlo_contract.json": EXPECTED_SOURCE["db533_hlo_contract_sha256"],
        "summary.json": EXPECTED_SOURCE["db533_summary_sha256"],
    }
    observed = {path.name for path in source_dir.iterdir() if path.is_file()}
    if observed != set(expected):
        raise ValueError("sealed DB533 source file set drifted")
    for name, digest in expected.items():
        if _file_sha256(source_dir / name) != digest:
            raise ValueError(f"sealed DB533 source file drifted: {name}")
    analysis = json.loads((source_dir / "analysis.json").read_text())
    summary = json.loads((source_dir / "summary.json").read_text())
    row0 = analysis.get("rows", [{}])[0]
    if (
        analysis.get("code_hash") != EXPECTED_SOURCE["db533_code_hash"]
        or analysis.get("accepted_model_axis_device_ids")
        != list(EXPECTED_MODEL_AXIS_DEVICE_IDS)
        or analysis.get("minimum_row_union_exact_column_count") != 6144
        or analysis.get("maximum_row_union_exact_column_count") != 6144
        or row0.get("physical_row") != 0
        or row0.get("row_output_bits_sha256")
        != "7239b23e8ccba21dd74cb11342b3a3cb8e79c9ebff35d03203da2c61898f51dc"
        or summary.get("results_db_run_id") != EXPECTED_SOURCE["db533_run_id"]
        or summary.get("code_hash") != EXPECTED_SOURCE["db533_code_hash"]
        or summary.get("mode") != "strategy_nd_fingerprint"
    ):
        raise ValueError("sealed DB533 association semantic contract drifted")


def _recompute_comparison(
    output_bits: np.ndarray, software_row0: np.ndarray
) -> dict[str, Any]:
    row0 = np.ascontiguousarray(output_bits[0, 0])
    mismatch_indices = np.flatnonzero(row0 != software_row0)
    first = None if not len(mismatch_indices) else int(mismatch_indices[0])
    row_mismatch_counts = [
        int(np.count_nonzero(row != software_row0)) for row in output_bits[0]
    ]
    return {
        "classification": (
            "hardware_row0_exact_db533_software"
            if first is None
            else "hardware_row0_differs_db533_software"
        ),
        "hardware_hidden_2795_bfloat16_bits": int(row0[2795]),
        "hardware_row0_array_sha256": array_sha256(row0),
        "hardware_row0_raw_sha256": _raw_sha256(row0),
        "hidden_index": 2795,
        "row0_exact": first is None,
        "row0_first_mismatch_index": first,
        "row0_mismatch_count": row_mismatch_counts[0],
        "row_mismatch_counts": row_mismatch_counts,
        "software_hidden_2795_bfloat16_bits": int(software_row0[2795]),
        "software_row0_array_sha256": array_sha256(software_row0),
        "software_row0_raw_sha256": _raw_sha256(software_row0),
    }


def validate_strategy_nd_dense_replay(
    run_dir: Path,
    *,
    expected_code_hash: str,
    expected_run_tag: str,
) -> dict[str, Any]:
    """Reload and recompute the complete eight-host replay evidence."""

    host_paths = sorted((run_dir / "host_records").glob("collective.rank*.json"))
    records = [json.loads(path.read_text()) for path in host_paths]
    if len(records) != 8:
        raise ValueError("dense replay requires exactly eight host records")
    for launch_index, (path, record) in enumerate(zip(host_paths, records, strict=True)):
        if path.name != f"collective.rank{launch_index}.json":
            raise ValueError("dense replay host-record filename coverage drifted")
        if set(record) != EXPECTED_RECORD_FIELDS:
            raise ValueError("dense replay top-level record schema drifted")
        if (
            type(record["launch_process_id"]) is not int
            or record["launch_process_id"] != launch_index
            or type(record["jax_process_index"]) is not int
            or record["hostname"] != EXPECTED_HOSTNAMES[launch_index]
        ):
            raise ValueError("dense replay filename/launch/hostname binding drifted")
        try:
            captured = datetime.fromisoformat(record["captured_utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("dense replay captured timestamp drifted") from error
        if captured.tzinfo is None or captured.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("dense replay captured timestamp is not UTC")
        if (
            record["run_tag"] != expected_run_tag
            or record["mechanism_only"] is not True
            or not isinstance(record["jax_version"], str)
            or not record["jax_version"]
        ):
            raise ValueError("dense replay run/mechanism/software provenance drifted")
    if {record["jax_process_index"] for record in records} != set(range(8)):
        raise ValueError("dense replay JAX process coverage drifted")
    if {record["launch_process_id"] for record in records} != set(range(8)):
        raise ValueError("dense replay launch process coverage drifted")
    if len({record["hostname"] for record in records}) != 8:
        raise ValueError("dense replay hostnames are not unique")
    if {record["code_hash"] for record in records} != {expected_code_hash}:
        raise ValueError("dense replay code hash drifted")
    if {record["mode"] for record in records} != {"strategy_nd_dense_replay"}:
        raise ValueError("dense replay mode drifted")
    if {record["schema_version"] for record in records} != {3}:
        raise ValueError("dense replay schema version drifted")
    if len({record["jax_version"] for record in records}) != 1:
        raise ValueError("dense replay JAX versions differ")
    topologies: list[PhysicalTopology] = []
    for record in records:
        try:
            topology = PhysicalTopology.from_dict(record["topology"])
            validate_target_v4_64(topology)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("dense replay target-v4-64 topology drifted") from error
        if (
            topology.topology_hash != record["topology_hash"]
            or topology.topology_hash != EXPECTED_TOPOLOGY_HASH
            or record["fleet_local_device_ids_in_runtime_order"]
            != EXPECTED_FLEET_LOCAL_DEVICE_IDS
        ):
            raise ValueError("dense replay topology hash/local ordering drifted")
        expected_local_ids = [
            device.device_id
            for device in topology.devices
            if device.process_index == record["jax_process_index"]
        ]
        if expected_local_ids != EXPECTED_FLEET_LOCAL_DEVICE_IDS[
            record["jax_process_index"]
        ]:
            raise ValueError("dense replay JAX process/topology ownership drifted")
        topologies.append(topology)
    if any(topology.to_dict() != topologies[0].to_dict() for topology in topologies[1:]):
        raise ValueError("dense replay fleet topology records differ")
    if any(record["matrix"] or record["association_fingerprint"] for record in records):
        raise ValueError("dense replay record contains another benchmark mode")

    items = [record.get("association_dense_replay") for record in records]
    if any(not isinstance(item, dict) for item in items):
        raise ValueError("dense replay payload is missing")
    for item in items:
        if set(item) != EXPECTED_REPLAY_FIELDS or any(
            type(item[field]) is not expected_type
            for field, expected_type in EXPECTED_REPLAY_TYPES.items()
        ):
            raise ValueError("dense replay nested payload schema/type drifted")
    reference = items[0]
    stable_fields = (
        "accepted_model_axis_device_ids",
        "accepted_model_axis_recipe",
        "capture",
        "collective_algorithm",
        "collective_groups",
        "comparison",
        "config",
        "diagnostic_only",
        "fleet_hashes",
        "fleet_hlo_hashes",
        "hlo",
        "member_device_ids",
        "optimized_hlo_sha256",
        "performance_claim",
        "source",
    )
    for field in stable_fields:
        if any(item[field] != reference[field] for item in items[1:]):
            raise ValueError(f"dense replay fleet field differs: {field}")
    if reference["source"] != EXPECTED_SOURCE:
        raise ValueError("dense replay DB550 source provenance drifted")
    if tuple(reference["accepted_model_axis_device_ids"]) != EXPECTED_MODEL_AXIS_DEVICE_IDS:
        raise ValueError("dense replay accepted model-axis mapping drifted")
    if reference["accepted_model_axis_recipe"] != ACCEPTED_TP32_MODEL_AXIS_RECIPE:
        raise ValueError("dense replay accepted model-axis recipe drifted")
    if reference["member_device_ids"] != list(range(32)):
        raise ValueError("dense replay physical member ids drifted")
    if reference["collective_groups"] != [list(range(32))]:
        raise ValueError("dense replay collective group drifted")
    if reference["config"] != {"seed": 1196575821, "trials": 1, "width": 6144}:
        raise ValueError("dense replay execution config drifted")
    if reference["collective_algorithm"] != STRATEGY_ND_ALGORITHM:
        raise ValueError("dense replay StrategyND algorithm drifted")
    if not reference["diagnostic_only"] or reference["performance_claim"]:
        raise ValueError("dense replay incorrectly claims promotion/performance")
    if not reference["hlo"]["valid"] or reference["hlo"]["violations"]:
        raise ValueError("dense replay HLO contract failed")
    if reference["hlo"]["collective_counts"] != {"all-reduce": 1}:
        raise ValueError("dense replay does not contain one physical all-reduce")
    capture = reference["capture"]
    if any(
        capture.get(field) != value
        for field, value in {
            "compile_bucket_rows": 32,
            "determinism_repeat_invocations": 1,
            "input_rows_replicated": True,
            "invocation_count": 2,
            "measured_trial_invocations": 1,
        }.items()
    ) or capture["output_bits_sha256"] != capture["repeated_output_bits_sha256"]:
        raise ValueError("dense replay determinism/capture contract drifted")
    for key, values in reference["fleet_hashes"].items():
        if len(values) != 8 or len(set(values)) != 1:
            raise ValueError(f"dense replay fleet hash agreement failed: {key}")
    if (
        len(reference["fleet_hlo_hashes"]) != 8
        or len(set(reference["fleet_hlo_hashes"])) != 1
        or reference["fleet_hlo_hashes"][0] != reference["optimized_hlo_sha256"]
    ):
        raise ValueError("dense replay fleet optimized HLO agreement failed")

    owners = [
        (record, item)
        for record, item in zip(records, items, strict=True)
        if item["artifact_manifest"]
    ]
    if len(owners) != 1 or owners[0][0]["jax_process_index"] != 0:
        raise ValueError("dense replay artifacts require exactly one process-zero owner")
    manifest = json.loads((run_dir / "replay" / "manifest.json").read_text())
    if manifest != owners[0][1]["artifact_manifest"]:
        raise ValueError("dense replay artifact manifest differs from process-zero record")
    expected_artifacts = {
        "hardware_output_bits": ((1, 32, 6144), np.dtype(np.uint16)),
        "physical_input_bits": ((1, 32, 6144), np.dtype(np.uint16)),
        "software_row0_bits": ((6144,), np.dtype(np.uint16)),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected_artifacts.items():
        record = manifest.get(name)
        path = run_dir / "replay" / f"{name}.npy"
        if not isinstance(record, dict) or set(record) != {
            "array_sha256", "dtype", "file", "file_sha256", "shape"
        }:
            raise ValueError(f"dense replay artifact record drifted: {name}")
        value = np.load(path, allow_pickle=False)
        if (
            record["file"] != path.name
            or tuple(record["shape"]) != shape
            or record["dtype"] != dtype.str
            or value.shape != shape
            or value.dtype != dtype
            or record["file_sha256"] != _file_sha256(path)
            or record["array_sha256"] != array_sha256(value)
        ):
            raise ValueError(f"dense replay artifact failed recomputation: {name}")
        arrays[name] = np.ascontiguousarray(value)

    model_bits = _validate_source_files(run_dir / "source")
    _validate_db533_source_files(run_dir / "source_db533")
    expected_input = model_axis_to_physical_input_bits(
        model_bits, EXPECTED_MODEL_AXIS_DEVICE_IDS
    )[None, ...]
    expected_software = replay_db533_strategy_nd_row0_bits(
        model_bits, EXPECTED_MODEL_AXIS_DEVICE_IDS
    )
    if not np.array_equal(arrays["physical_input_bits"], expected_input):
        raise ValueError("dense replay physical input does not derive from DB550 model order")
    if not np.array_equal(arrays["software_row0_bits"], expected_software):
        raise ValueError("dense replay software row-zero result failed recomputation")
    recomputed = _recompute_comparison(
        arrays["hardware_output_bits"], expected_software
    )
    comparison_file = json.loads((run_dir / "replay" / "comparison.json").read_text())
    if comparison_file != recomputed or reference["comparison"] != recomputed:
        raise ValueError("dense replay numerical comparison failed recomputation")
    if array_sha256(arrays["physical_input_bits"]) != capture["input_bits_sha256"]:
        raise ValueError("dense replay capture input hash drifted")
    if array_sha256(arrays["hardware_output_bits"]) != capture["output_bits_sha256"]:
        raise ValueError("dense replay capture output hash drifted")
    replica_hash = array_sha256(arrays["hardware_output_bits"][0])
    if (
        capture.get("local_replica_output_sha256_by_trial")
        != [[replica_hash] * 4]
        or capture.get("repeated_local_replica_output_sha256_by_trial")
        != [[replica_hash] * 4]
    ):
        raise ValueError("dense replay local replica hashes drifted")
    expected_fleet = {
        "input": array_sha256(arrays["physical_input_bits"]),
        "output": array_sha256(arrays["hardware_output_bits"]),
        "software": array_sha256(arrays["software_row0_bits"]),
        "source_file": ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
        "source_raw": ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    }
    if {
        key: values[0] for key, values in reference["fleet_hashes"].items()
    } != expected_fleet:
        raise ValueError("dense replay fleet hashes do not match reloaded artifacts")

    hlo_path = (
        run_dir / "hlo" /
        "strategy_nd_dense_partials_bfloat16_32x6144.optimized_hlo.txt"
    )
    contract_path = (
        run_dir / "hlo" /
        "strategy_nd_dense_partials_bfloat16_32x6144.hlo_contract.json"
    )
    if {path.name for path in (run_dir / "hlo").iterdir() if path.is_file()} != {
        hlo_path.name,
        contract_path.name,
    }:
        raise ValueError("dense replay HLO artifact set drifted")
    hlo_text = hlo_path.read_text()
    if sha256(hlo_text.encode()).hexdigest() != reference["optimized_hlo_sha256"]:
        raise ValueError("dense replay optimized HLO SHA-256 drifted")
    hlo_report, algorithm = validate_strategy_nd_fingerprint_hlo(
        hlo_text, tuple(range(32))
    )
    if hlo_report.to_dict() != reference["hlo"] or algorithm != STRATEGY_ND_ALGORITHM:
        raise ValueError("dense replay optimized HLO failed terminal validation")
    if json.loads(contract_path.read_text()) != {
        "collective_algorithm": STRATEGY_ND_ALGORITHM,
        "decode_shape_admissible": True,
        "hlo": reference["hlo"],
        "valid": True,
    }:
        raise ValueError("dense replay HLO contract artifact drifted")

    return {
        "artifact_kind": "glm52_strategy_nd_dense_partials_replay",
        "classification": recomputed["classification"],
        "code_hash": expected_code_hash,
        "diagnostic_only": True,
        "hardware_hidden_2795_bfloat16_bits": recomputed[
            "hardware_hidden_2795_bfloat16_bits"
        ],
        "hardware_row0_raw_sha256": recomputed["hardware_row0_raw_sha256"],
        "optimized_hlo_sha256": reference["optimized_hlo_sha256"],
        "performance_claim": False,
        "row0_exact": recomputed["row0_exact"],
        "row0_mismatch_count": recomputed["row0_mismatch_count"],
        "run_tag": expected_run_tag,
        "software_hidden_2795_bfloat16_bits": recomputed[
            "software_hidden_2795_bfloat16_bits"
        ],
        "software_row0_raw_sha256": recomputed["software_row0_raw_sha256"],
        "source": dict(EXPECTED_SOURCE),
        "status": "SUCCESS",
        "topology_hash": records[0]["topology_hash"],
    }
