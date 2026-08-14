"""Terminal recomputation for the integrated one-rank-per-chip discriminator."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmarking import (
    ACCEPTED_SOURCE_VALIDITY,
    ACCEPTED_TP32_MODEL_AXIS_RECIPE,
    CHECKPOINT_SUCCESS_SHA256,
    DENSE_RMS_SOURCE_NPZ_SHA256,
    DENSE_RMS_SOURCE_TAG,
    IntegratedDenseWeights,
    array_sha256,
    load_integrated_dense_rms_inputs,
    model_axis_weights_to_physical,
    validate_integrated_dense_rms_hlo,
    validate_integrated_dense_rms_stablehlo,
    validate_integrated_checkpoint_success,
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
    EXPECTED_COMPARISON_FIELDS,
    _file_sha256,
    _is_int_list,
    _is_sha256,
    _raw_sha256,
    _validate_rms_source_files,
)


CHECKPOINT_MANIFEST_SHA256 = (
    "de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134"
)
LEGACY_INTEGRATED_CODE_HASH = "d7872b582181e8c2518da2d8785b109a733e92b1"
LEGACY_INTEGRATED_RUN_TAG = (
    "greenfield_strategy_nd_integrated_dense_rms_20260814T174146122417710Z"
)
EXPECTED_SOURCE = {
    "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
    "checkpoint_success_sha256": CHECKPOINT_SUCCESS_SHA256,
    "rms_npz_sha256": DENSE_RMS_SOURCE_NPZ_SHA256,
    "rms_tag": DENSE_RMS_SOURCE_TAG,
}
EXPECTED_RECORD_FIELDS = {
    "association_dense_replay",
    "association_fingerprint",
    "association_integrated_dense_rms",
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
    "checkpoint_records",
    "collective_groups",
    "comparison",
    "diagnostic_only",
    "final_layout_records",
    "fleet_hashes",
    "fleet_hlo_hashes",
    "fleet_stablehlo_hashes",
    "member_device_ids",
    "optimized_hlo_contract",
    "optimized_hlo_sha256",
    "performance_claim",
    "physical_weight_hashes",
    "source",
    "stablehlo_contract",
    "stablehlo_sha256",
    "split_layer1_rms",
}
EXPECTED_CAPTURE_FIELDS = {
    "invocation_count",
    "local_replica_output_sha256",
    "output_bits_sha256",
    "repeated_local_replica_output_sha256",
    "repeated_output_bits_sha256",
}
EXPECTED_HLO_PREVALIDATION_FIELDS = {
    "optimized_hlo_sha256",
    "performance_claim",
    "split_layer1_rms",
    "stablehlo_sha256",
    "validated",
}


def _validate_hlo_prevalidation(
    record: object,
    *,
    optimized_hlo_sha256: str,
    stablehlo_sha256: str,
    split_layer1_rms: bool,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
    accepted_source_context: bool = False,
) -> None:
    expected_fields = set(EXPECTED_HLO_PREVALIDATION_FIELDS)
    if preceding_attention_collective:
        expected_fields.add("preceding_attention_collective")
    if split_predense_rms:
        expected_fields.add("split_predense_rms")
    if accepted_source_context:
        expected_fields.add("accepted_source_context")
    if not (
        type(record) is dict
        and set(record) == expected_fields
        and record["optimized_hlo_sha256"] == optimized_hlo_sha256
        and record["stablehlo_sha256"] == stablehlo_sha256
        and (
            not preceding_attention_collective
            or record["preceding_attention_collective"] is True
        )
        and (
            not split_predense_rms
            or record["split_predense_rms"] is True
        )
        and (
            not accepted_source_context
            or record["accepted_source_context"] is True
        )
        and record["split_layer1_rms"] is split_layer1_rms
        and record["validated"] is False
        and record["performance_claim"] is False
    ):
        raise ValueError("integrated dense HLO prevalidation record drifted")


def _recompute_comparison(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
    accepted_source_context: bool = False,
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
    prefix = (
        "integrated_dense_accepted_source_context"
        if accepted_source_context
        else "integrated_dense_predense_split_rms"
        if split_predense_rms
        else "integrated_dense_ordinal_rms"
        if preceding_attention_collective
        else "integrated_dense_split_rms"
        if split_layer1_rms
        else "integrated_dense_rms"
    )
    return {
        "classification": (
            f"{prefix}_exact_accepted"
            if first is None
            else f"{prefix}_matches_db548_control"
            if observed_sha
            == "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
            else f"{prefix}_matches_rejected_db549"
            if observed_sha
            == "229dc8ace9bfa31fce6d6ccabc9fca49ccc55f30b9d1dd6f97a032f5117b812f"
            else f"{prefix}_nonexact_new_result"
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


def _validate_capture(capture: object, output: np.ndarray) -> None:
    output_sha = array_sha256(output)
    if not (
        type(capture) is dict
        and set(capture) == EXPECTED_CAPTURE_FIELDS
        and type(capture["invocation_count"]) is int
        and capture["invocation_count"] == 2
        and capture["output_bits_sha256"] == output_sha
        and capture["repeated_output_bits_sha256"] == output_sha
        and type(capture["local_replica_output_sha256"]) is list
        and len(capture["local_replica_output_sha256"]) == 4
        and all(
            type(value) is str and value == output_sha
            for value in capture["local_replica_output_sha256"]
        )
        and capture["repeated_local_replica_output_sha256"]
        == capture["local_replica_output_sha256"]
    ):
        raise ValueError("integrated dense deterministic capture drifted")


def validate_strategy_nd_integrated_dense_rms(
    run_dir: Path,
    *,
    checkpoint_root: Path,
    expected_code_hash: str,
    expected_run_tag: str,
    expected_split_layer1_rms: bool = False,
    expected_preceding_attention_collective: bool = False,
    expected_split_predense_rms: bool = False,
    expected_accepted_source_context: bool = False,
) -> dict[str, Any]:
    """Reload every source/fleet/artifact byte and recompute the verdict."""

    from scripts.greenfield.probe_layer0_dense_convolution import (
        _FINAL_DENSE_LAYOUT_RECORDS,
        _pack_dense_final_layout,
    )
    from scripts.greenfield.probe_layer0_projection_reduction import (
        _load_weights,
    )

    if not isinstance(expected_split_layer1_rms, bool):
        raise ValueError("integrated dense RMS split expectation must be boolean")
    if not isinstance(expected_preceding_attention_collective, bool):
        raise ValueError("preceding attention expectation must be boolean")
    if not isinstance(expected_split_predense_rms, bool):
        raise ValueError("pre-dense split expectation must be boolean")
    if not isinstance(expected_accepted_source_context, bool):
        raise ValueError("accepted source-context expectation must be boolean")
    if expected_preceding_attention_collective and not expected_split_layer1_rms:
        raise ValueError("preceding attention collective requires split RMS")
    if expected_split_predense_rms and not expected_split_layer1_rms:
        raise ValueError("pre-dense split RMS requires layer-1 split RMS")
    if expected_split_predense_rms and expected_preceding_attention_collective:
        raise ValueError("pre-dense split RMS and rejected ordinal arms are disjoint")
    if expected_accepted_source_context and any(
        (
            expected_split_layer1_rms,
            expected_preceding_attention_collective,
            expected_split_predense_rms,
        )
    ):
        raise ValueError("accepted source context is a disjoint integrated arm")
    legacy_schema = bool(
        not expected_split_layer1_rms
        and not expected_preceding_attention_collective
        and not expected_split_predense_rms
        and not expected_accepted_source_context
        and expected_code_hash == LEGACY_INTEGRATED_CODE_HASH
        and expected_run_tag == LEGACY_INTEGRATED_RUN_TAG
    )
    host_paths = sorted((run_dir / "host_records").glob("collective.rank*.json"))
    records = [json.loads(path.read_text()) for path in host_paths]
    if len(records) != 8:
        raise ValueError("integrated dense RMS requires eight host records")
    topologies = []
    for launch_index, (path, record) in enumerate(
        zip(host_paths, records, strict=True)
    ):
        if (
            path.name != f"collective.rank{launch_index}.json"
            or set(record) != EXPECTED_RECORD_FIELDS
        ):
            raise ValueError("integrated dense host-record schema drifted")
        try:
            captured = datetime.fromisoformat(record["captured_utc"])
        except (TypeError, ValueError) as error:
            raise ValueError("integrated dense timestamp drifted") from error
        if not (
            captured.tzinfo is not None
            and captured.utcoffset() == timezone.utc.utcoffset(None)
            and type(record["launch_process_id"]) is int
            and record["launch_process_id"] == launch_index
            and type(record["jax_process_index"]) is int
            and record["hostname"] == EXPECTED_HOSTNAMES[launch_index]
            and record["run_tag"] == expected_run_tag
            and record["code_hash"] == expected_code_hash
            and record["mode"] == "strategy_nd_integrated_dense_rms"
            and type(record["schema_version"]) is int
            and record["schema_version"] == 5
            and record["mechanism_only"] is True
            and type(record["jax_version"]) is str
            and bool(record["jax_version"])
            and record["association_dense_replay"] is None
            and record["association_fingerprint"] is None
            and record["matrix"] == []
            and _is_sha256(record["topology_hash"])
        ):
            raise ValueError("integrated dense host identity drifted")
        topology = PhysicalTopology.from_dict(record["topology"])
        validate_target_v4_64(topology)
        process_index = record["jax_process_index"]
        if not (
            topology.topology_hash == EXPECTED_TOPOLOGY_HASH
            and record["topology_hash"] == EXPECTED_TOPOLOGY_HASH
            and type(record["fleet_local_device_ids_in_runtime_order"]) is list
            and len(record["fleet_local_device_ids_in_runtime_order"]) == 8
            and all(
                _is_int_list(value, expected)
                for value, expected in zip(
                    record["fleet_local_device_ids_in_runtime_order"],
                    EXPECTED_FLEET_LOCAL_DEVICE_IDS,
                    strict=True,
                )
            )
            and [
                device.device_id
                for device in topology.devices
                if device.process_index == process_index
            ]
            == EXPECTED_FLEET_LOCAL_DEVICE_IDS[process_index]
        ):
            raise ValueError("integrated dense topology ownership drifted")
        topologies.append(topology)
    if (
        {record["jax_process_index"] for record in records} != set(range(8))
        or len({record["jax_version"] for record in records}) != 1
        or any(
            topology.to_dict() != topologies[0].to_dict()
            for topology in topologies[1:]
        )
    ):
        raise ValueError("integrated dense fleet agreement drifted")

    items = [record["association_integrated_dense_rms"] for record in records]
    expected_replay_fields = set(EXPECTED_REPLAY_FIELDS)
    if legacy_schema:
        expected_replay_fields.remove("split_layer1_rms")
    if expected_preceding_attention_collective:
        expected_replay_fields.add("preceding_attention_collective")
    if expected_split_predense_rms:
        expected_replay_fields.add("split_predense_rms")
    if expected_accepted_source_context:
        expected_replay_fields.add("accepted_source_context")
    if any(
        type(item) is not dict or set(item) != expected_replay_fields
        for item in items
    ):
        raise ValueError("integrated dense nested schema drifted")
    reference = items[0]
    for field in expected_replay_fields - {"artifact_manifest"}:
        if any(item[field] != reference[field] for item in items[1:]):
            raise ValueError(f"integrated dense fleet field differs: {field}")
    if not (
        reference["source"] == EXPECTED_SOURCE
        and _is_int_list(
            reference["accepted_model_axis_device_ids"],
            list(EXPECTED_MODEL_AXIS_DEVICE_IDS),
        )
        and reference["accepted_model_axis_recipe"]
        == ACCEPTED_TP32_MODEL_AXIS_RECIPE
        and _is_int_list(reference["member_device_ids"], list(range(32)))
        and reference["collective_groups"] == [list(range(32))]
        and reference["diagnostic_only"] is True
        and reference["performance_claim"] is False
        and (
            not expected_preceding_attention_collective
            or reference["preceding_attention_collective"] is True
        )
        and (
            not expected_split_predense_rms
            or reference["split_predense_rms"] is True
        )
        and (
            not expected_accepted_source_context
            or reference["accepted_source_context"] is True
        )
        and (
            legacy_schema
            or reference["split_layer1_rms"] is expected_split_layer1_rms
        )
        and _is_sha256(reference["optimized_hlo_sha256"])
        and _is_sha256(reference["stablehlo_sha256"])
    ):
        raise ValueError("integrated dense execution identity drifted")
    for name in ("fleet_hlo_hashes", "fleet_stablehlo_hashes"):
        values = reference[name]
        expected = (
            reference["optimized_hlo_sha256"]
            if name == "fleet_hlo_hashes"
            else reference["stablehlo_sha256"]
        )
        if (
            type(values) is not list
            or len(values) != 8
            or len(set(values)) != 1
            or values[0] != expected
        ):
            raise ValueError(f"integrated dense fleet HLO drifted: {name}")

    owners = [
        (record, item)
        for record, item in zip(records, items, strict=True)
        if item["artifact_manifest"]
    ]
    if len(owners) != 1 or owners[0][0]["jax_process_index"] != 0:
        raise ValueError("integrated dense artifacts lack one process-zero owner")
    artifact_dir = run_dir / "integrated_dense_rms"
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest != owners[0][1]["artifact_manifest"]:
        raise ValueError("integrated dense artifact manifest drifted")
    arrays: dict[str, np.ndarray] = {}
    for name in ("accepted_layer1_bits", "hardware_layer1_bits"):
        path = artifact_dir / f"{name}.npy"
        value = np.load(path, allow_pickle=False)
        artifact = manifest.get(name)
        if not (
            type(artifact) is dict
            and set(artifact)
            == {"array_sha256", "dtype", "file", "file_sha256", "shape"}
            and artifact["file"] == path.name
            and artifact["shape"] == [6144]
            and artifact["dtype"] == np.dtype(np.uint16).str
            and value.shape == (6144,)
            and value.dtype == np.uint16
            and artifact["file_sha256"] == _file_sha256(path)
            and artifact["array_sha256"] == array_sha256(value)
        ):
            raise ValueError(f"integrated dense artifact drifted: {name}")
        arrays[name] = np.ascontiguousarray(value)

    _validate_rms_source_files(run_dir / "source_rms")
    validate_integrated_checkpoint_success(checkpoint_root)
    weights, checkpoint_records = _load_weights(
        checkpoint_root,
        manifest_sha256=CHECKPOINT_MANIFEST_SHA256,
    )
    packed, packed_records = _pack_dense_final_layout(weights)
    if not (
        checkpoint_records == reference["checkpoint_records"]
        and packed_records == _FINAL_DENSE_LAYOUT_RECORDS
        and reference["final_layout_records"] == _FINAL_DENSE_LAYOUT_RECORDS
    ):
        raise ValueError("integrated dense checkpoint provenance drifted")
    inputs = load_integrated_dense_rms_inputs(
        run_dir / "source_rms" / "dense_partial_capture.npz",
        weights["attention.slot_00.post_norm"],
    )
    physical_weights = IntegratedDenseWeights(
        **{
            name: model_axis_weights_to_physical(
                value.reshape((32, 1) + value.shape[2:]),
                EXPECTED_MODEL_AXIS_DEVICE_IDS,
            )
            for name, value in zip(
                ("merged_bits", "merged_scale", "down_bits", "down_scale"),
                packed,
                strict=True,
            )
        }
    )
    physical_weight_hashes = {
        name: array_sha256(getattr(physical_weights, name))
        for name in ("merged_bits", "merged_scale", "down_bits", "down_scale")
    }
    if reference["physical_weight_hashes"] != physical_weight_hashes:
        raise ValueError("integrated dense physical weight hashes drifted")
    if not np.array_equal(arrays["accepted_layer1_bits"], inputs.accepted_layer1_bits):
        raise ValueError("integrated dense target differs from sealed source")
    comparison = _recompute_comparison(
        arrays["hardware_layer1_bits"],
        arrays["accepted_layer1_bits"],
        split_layer1_rms=expected_split_layer1_rms,
        preceding_attention_collective=expected_preceding_attention_collective,
        split_predense_rms=expected_split_predense_rms,
        accepted_source_context=expected_accepted_source_context,
    )
    recorded = reference["comparison"]
    if not (
        type(recorded) is dict
        and set(recorded) == EXPECTED_COMPARISON_FIELDS
        and type(recorded["classification"]) is str
        and type(recorded["elementwise_exact"]) is bool
        and type(recorded["expected_hidden_2795_bfloat16_bits"]) is int
        and _is_sha256(recorded["expected_raw_sha256"])
        and (
            recorded["first_mismatch_index"] is None
            or type(recorded["first_mismatch_index"]) is int
        )
        and type(recorded["max_absolute_error"]) is float
        and math.isfinite(recorded["max_absolute_error"])
        and type(recorded["mean_absolute_error"]) is float
        and math.isfinite(recorded["mean_absolute_error"])
        and type(recorded["mismatch_count"]) is int
        and type(recorded["observed_hidden_2795_bfloat16_bits"]) is int
        and _is_sha256(recorded["observed_raw_sha256"])
        and recorded == comparison
        and json.loads((artifact_dir / "comparison.json").read_text())
        == comparison
    ):
        raise ValueError("integrated dense comparison failed recomputation")
    _validate_capture(reference["capture"], arrays["hardware_layer1_bits"])
    expected_fleet = {
        "accepted_target": array_sha256(arrays["accepted_layer1_bits"]),
        "attention_update": array_sha256(inputs.attention_update_bits),
        "combined_residual": array_sha256(inputs.combined_residual_bits),
        "layer1_norm": array_sha256(inputs.layer1_norm_bits),
        "output": array_sha256(arrays["hardware_layer1_bits"]),
        "post_attention_norm": array_sha256(inputs.post_attention_norm_bits),
        **(
            {
                "accepted_source_validity": array_sha256(
                    ACCEPTED_SOURCE_VALIDITY
                )
            }
            if expected_accepted_source_context
            else {}
        ),
        **{
            f"physical_{name}": digest
            for name, digest in physical_weight_hashes.items()
        },
    }
    if not (
        type(reference["fleet_hashes"]) is dict
        and set(reference["fleet_hashes"]) == set(expected_fleet)
        and all(
            type(values) is list
            and len(values) == 8
            and values == [expected_fleet[name]] * 8
            for name, values in reference["fleet_hashes"].items()
        )
    ):
        raise ValueError("integrated dense fleet hashes drifted")

    label = (
        "strategy_nd_integrated_dense_accepted_source_context_bfloat16_32x6144"
        if expected_accepted_source_context
        else "strategy_nd_integrated_dense_predense_split_rms_bfloat16_32x6144"
        if expected_split_predense_rms
        else "strategy_nd_integrated_dense_ordinal_rms_bfloat16_32x6144"
        if expected_preceding_attention_collective
        else "strategy_nd_integrated_dense_split_rms_bfloat16_32x6144"
        if expected_split_layer1_rms
        else "strategy_nd_integrated_dense_rms_bfloat16_32x6144"
    )
    hlo_dir = run_dir / "hlo"
    stable_path = hlo_dir / f"{label}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{label}.optimized_hlo.txt"
    contract_path = hlo_dir / f"{label}.hlo_contract.json"
    prevalidation_path = hlo_dir / f"{label}.hlo_prevalidation.json"
    expected_hlo_files = {
        stable_path.name,
        optimized_path.name,
        contract_path.name,
    }
    if not legacy_schema:
        expected_hlo_files.add(prevalidation_path.name)
    if {path.name for path in hlo_dir.iterdir() if path.is_file()} != expected_hlo_files:
        raise ValueError("integrated dense HLO artifact set drifted")
    stablehlo = stable_path.read_text()
    optimized_hlo = optimized_path.read_text()
    if not (
        sha256(stablehlo.encode()).hexdigest() == reference["stablehlo_sha256"]
        and sha256(optimized_hlo.encode()).hexdigest()
        == reference["optimized_hlo_sha256"]
    ):
        raise ValueError("integrated dense HLO SHA-256 drifted")
    stable_contract = validate_integrated_dense_rms_stablehlo(
        stablehlo,
        split_layer1_rms=expected_split_layer1_rms,
        preceding_attention_collective=expected_preceding_attention_collective,
        split_predense_rms=expected_split_predense_rms,
        accepted_source_context=expected_accepted_source_context,
    )
    optimized_contract = validate_integrated_dense_rms_hlo(
        optimized_hlo,
        tuple(range(32)),
        split_layer1_rms=expected_split_layer1_rms,
        preceding_attention_collective=expected_preceding_attention_collective,
        split_predense_rms=expected_split_predense_rms,
        accepted_source_context=expected_accepted_source_context,
    )
    if legacy_schema:
        stable_contract = dict(stable_contract)
        optimized_contract = dict(optimized_contract)
        stable_contract.pop("split_layer1_rms")
        optimized_contract.pop("split_layer1_rms")
    if not legacy_schema:
        _validate_hlo_prevalidation(
            json.loads(prevalidation_path.read_text()),
            optimized_hlo_sha256=reference["optimized_hlo_sha256"],
            stablehlo_sha256=reference["stablehlo_sha256"],
            split_layer1_rms=expected_split_layer1_rms,
            preceding_attention_collective=(
                expected_preceding_attention_collective
            ),
            split_predense_rms=expected_split_predense_rms,
            accepted_source_context=expected_accepted_source_context,
        )
    if not (
        stable_contract == reference["stablehlo_contract"]
        and optimized_contract == reference["optimized_hlo_contract"]
        and json.loads(contract_path.read_text())
        == {
            "optimized": optimized_contract,
            "stablehlo": stable_contract,
            "valid": True,
        }
    ):
        raise ValueError("integrated dense HLO terminal replay drifted")

    result = {
        "artifact_kind": (
            "glm52_strategy_nd_integrated_dense_accepted_source_context"
            if expected_accepted_source_context
            else "glm52_strategy_nd_integrated_dense_predense_split_rms"
            if expected_split_predense_rms
            else "glm52_strategy_nd_integrated_dense_ordinal_rms"
            if expected_preceding_attention_collective
            else "glm52_strategy_nd_integrated_dense_split_rms"
            if expected_split_layer1_rms
            else "glm52_strategy_nd_integrated_dense_rms"
        ),
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
        "split_layer1_rms": expected_split_layer1_rms,
        "stablehlo_sha256": reference["stablehlo_sha256"],
        "status": "SUCCESS",
        "topology_hash": EXPECTED_TOPOLOGY_HASH,
    }
    if expected_preceding_attention_collective:
        result["preceding_attention_collective"] = True
    if expected_split_predense_rms:
        result["split_predense_rms"] = True
    if expected_accepted_source_context:
        result["accepted_source_context"] = True
    return result
