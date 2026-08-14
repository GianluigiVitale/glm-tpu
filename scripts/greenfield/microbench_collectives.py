#!/usr/bin/env python3
"""Run the protected dependent collective-chain matrix on all TPU hosts.

Every host executes the same ordered matrix and writes one local append-only
record.  Process zero also writes optimized HLO and lint reports.  A launcher
must provide exact code provenance, ownership, archive, DB linkage, and pre/
post-run fleet census; this program provides the device/HLO/latency mechanism
record and never claims model throughput.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    ACCEPTED_SOURCE_VALIDITY,
    ACCEPTED_DENSE_PARTIALS_KEYS,
    ACCEPTED_DENSE_PARTIALS_NPZ_SHA256,
    ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    ACCEPTED_DENSE_PARTIALS_SHAPE,
    ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256,
    ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256,
    ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256,
    ACCEPTED_TP32_MODEL_AXIS_RECIPE,
    CHECKPOINT_SUCCESS_SHA256,
    CollectiveChainConfig,
    CollectiveKind,
    DENSE_RMS_SOURCE_CODE_HASH,
    DENSE_RMS_SOURCE_NPZ_SHA256,
    DENSE_RMS_SOURCE_REMOTE_OBJECTS_SHA256,
    DENSE_RMS_SOURCE_RUNNER_SHA256,
    DENSE_RMS_SOURCE_SUCCESS_SHA256,
    DENSE_RMS_SOURCE_SUMMARY_SHA256,
    DENSE_RMS_SOURCE_TAG,
    IntegratedDenseWeights,
    NATIVE_SOURCE_ATTENDED_LATENT_KEY,
    NATIVE_SOURCE_NPZ_SHA256,
    NATIVE_SOURCE_REMOTE_OBJECTS_SHA256,
    NATIVE_SOURCE_RUNNER_SHA256,
    NATIVE_SOURCE_SUCCESS_SHA256,
    NATIVE_SOURCE_SUMMARY_SHA256,
    NATIVE_SOURCE_TAG,
    StrategyNdFingerprintConfig,
    accepted_tp32_model_axis_device_ids,
    array_sha256,
    assemble_native_source_weights,
    benchmark_collective_chain,
    build_collective_chain,
    build_strategy_nd_dense_rms_replay,
    build_integrated_dense_rms,
    build_strategy_nd_fingerprint,
    execute_strategy_nd_dense_rms_replay,
    execute_integrated_dense_rms,
    execute_native_source_context,
    execute_strategy_nd_fingerprint,
    generate_strategy_nd_input_bits,
    load_dense_rms_inputs,
    load_integrated_dense_rms_inputs,
    model_axis_weights_to_physical,
    native_source_inputs,
    validate_integrated_checkpoint_success,
    model_axis_to_physical_input_bits,
    replay_db533_strategy_nd_row0_bits,
    validate_strategy_nd_fingerprint_hlo,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    collective_groups_for_size,
    discover_physical_topology,
    validate_target_v4_64,
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return result


def _shape(value: str) -> tuple[int, int]:
    result = _csv_ints(value)
    if len(result) != 2:
        raise argparse.ArgumentTypeError("shape must be ROWS,WIDTH")
    return result


def _operations(value: str) -> tuple[CollectiveKind, ...]:
    try:
        result = tuple(
            CollectiveKind(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("operations must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "chain",
            "strategy_nd_fingerprint",
            "strategy_nd_dense_replay",
            "strategy_nd_dense_rms_replay",
            "strategy_nd_integrated_dense_rms",
        ),
        default="chain",
    )
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", default="db-v4-64-od")
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=_csv_ints, default=(2, 4, 8, 32))
    parser.add_argument(
        "--operations",
        type=_operations,
        default=tuple(CollectiveKind),
    )
    parser.add_argument("--shape", type=_shape, default=(2, 6144))
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "int32"),
        default="bfloat16",
    )
    parser.add_argument("--chain-length", type=int, default=75)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--association-trials", type=int, default=32)
    parser.add_argument(
        "--association-replay-input",
        type=Path,
        help="sealed DB550 dense-partial NPZ for strategy_nd_dense_replay",
    )
    parser.add_argument(
        "--association-rms-input",
        type=Path,
        help="sealed residual/norm/target NPZ for strategy_nd_dense_rms_replay",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="exact local final-layout checkpoint root for integrated dense RMS",
    )
    parser.add_argument(
        "--checkpoint-manifest-sha256",
        help="exact runtime manifest hash for integrated dense RMS",
    )
    parser.add_argument(
        "--integrated-split-layer1-rms",
        action="store_true",
        help="use the accepted scalar-only layer-1 RMS schedule",
    )
    parser.add_argument(
        "--integrated-preceding-attention-collective",
        action="store_true",
        help="recreate the accepted attention-then-dense collective ordinal",
    )
    parser.add_argument(
        "--integrated-split-predense-rms",
        action="store_true",
        help="use the accepted scalar-only pre-dense RMS schedule",
    )
    parser.add_argument(
        "--integrated-accepted-source-context",
        action="store_true",
        help="recreate the accepted embedding/predicate/attention source graph",
    )
    parser.add_argument(
        "--integrated-native-source-context",
        action="store_true",
        help="run native embedding and attention producers in the bounded graph",
    )
    parser.add_argument(
        "--native-attention-input",
        type=Path,
        help="SHA-pinned DB537 attention arithmetic NPZ",
    )
    parser.add_argument(
        "--allow-unprotected-test-config",
        action="store_true",
        help="permit fewer than the protected 75/200/1000 contract",
    )
    return parser.parse_args()


def _raw_array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _first_mismatch(actual: np.ndarray, expected: np.ndarray) -> int | None:
    mismatches = np.flatnonzero(np.asarray(actual) != np.asarray(expected))
    return None if not len(mismatches) else int(mismatches[0])


def _run_strategy_nd_dense_replay(
    args: argparse.Namespace,
    jax: Any,
    multihost_utils: Any,
    topology: Any,
) -> dict[str, Any]:
    """Replay DB550's real partials through one exact accepted M32 reduction."""

    source_path = args.association_replay_input
    if source_path is None or not source_path.is_file():
        raise RuntimeError("StrategyND dense replay requires a local sealed input NPZ")
    source_file_sha = _file_sha256(source_path)
    if source_file_sha != ACCEPTED_DENSE_PARTIALS_NPZ_SHA256:
        raise RuntimeError("DB550 dense-partial NPZ SHA-256 drifted")
    with np.load(source_path, allow_pickle=False) as payload:
        if tuple(payload.files) != ACCEPTED_DENSE_PARTIALS_KEYS:
            raise RuntimeError("DB550 dense-partial NPZ key order/set drifted")
        accepted = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[0]])
        db548 = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[1]])
    for label, value in (("accepted", accepted), ("DB548", db548)):
        if value.shape != ACCEPTED_DENSE_PARTIALS_SHAPE or value.dtype != np.uint16:
            raise RuntimeError(f"{label} dense-partial geometry/dtype drifted")
        if _raw_array_sha256(value) != ACCEPTED_DENSE_PARTIALS_RAW_SHA256:
            raise RuntimeError(f"{label} dense-partial raw SHA-256 drifted")
    if not np.array_equal(accepted, db548):
        raise RuntimeError("DB550 accepted and DB548 dense partials are not bitwise equal")

    physical_ids = tuple(sorted(device.device_id for device in topology.devices))
    if physical_ids != tuple(range(32)):
        raise RuntimeError("accepted M32 replay requires contiguous physical ids 0..31")
    model_axis_device_ids = accepted_tp32_model_axis_device_ids(jax.devices())
    model_bits = accepted.reshape(32, 6144)
    physical_bits = model_axis_to_physical_input_bits(
        model_bits, model_axis_device_ids
    )
    input_bits = physical_bits[None, ...]
    software_row0 = replay_db533_strategy_nd_row0_bits(
        model_bits, model_axis_device_ids
    )
    config = StrategyNdFingerprintConfig(trials=1, width=6144)
    label = "strategy_nd_dense_partials_bfloat16_32x6144"

    multihost_utils.sync_global_devices(f"greenfield-replay-start-{label}")
    compiled = build_strategy_nd_fingerprint(
        config,
        physical_ids,
        devices=jax.devices(),
        enforce_hlo_contract=False,
    )
    hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
    fleet_hlo_hashes = _fleet_digest(
        multihost_utils,
        hlo_sha256,
        label="replay optimized HLO",
        num_processes=args.num_processes,
    )
    hlo_report, algorithm = validate_strategy_nd_fingerprint_hlo(
        compiled.optimized_hlo, physical_ids
    )
    output_bits, capture = execute_strategy_nd_fingerprint(compiled, input_bits)
    row0 = np.ascontiguousarray(output_bits[0, 0])
    row_mismatch_counts = [
        int(np.count_nonzero(row != software_row0)) for row in output_bits[0]
    ]
    first = _first_mismatch(row0, software_row0)
    comparison = {
        "classification": (
            "hardware_row0_exact_db533_software"
            if first is None
            else "hardware_row0_differs_db533_software"
        ),
        "hardware_hidden_2795_bfloat16_bits": int(row0[2795]),
        "hardware_row0_array_sha256": array_sha256(row0),
        "hardware_row0_raw_sha256": _raw_array_sha256(row0),
        "hidden_index": 2795,
        "row0_exact": first is None,
        "row0_first_mismatch_index": first,
        "row0_mismatch_count": row_mismatch_counts[0],
        "row_mismatch_counts": row_mismatch_counts,
        "software_hidden_2795_bfloat16_bits": int(software_row0[2795]),
        "software_row0_array_sha256": array_sha256(software_row0),
        "software_row0_raw_sha256": _raw_array_sha256(software_row0),
    }
    stable_hashes = {
        "input": array_sha256(input_bits),
        "output": array_sha256(output_bits),
        "software": array_sha256(software_row0),
        "source_file": source_file_sha,
        "source_raw": ACCEPTED_DENSE_PARTIALS_RAW_SHA256,
    }
    fleet_hashes = {
        key: _fleet_digest(
            multihost_utils,
            value,
            label=f"replay {key}",
            num_processes=args.num_processes,
        )
        for key, value in stable_hashes.items()
    }

    artifact_manifest: dict[str, Any] = {}
    artifact_dir = args.output.parent / "replay"
    hlo_dir = args.output.parent / "hlo"
    if jax.process_index() == 0:
        hlo_dir.mkdir(parents=True, exist_ok=True)
        hlo_path = hlo_dir / f"{label}.optimized_hlo.txt"
        contract_path = hlo_dir / f"{label}.hlo_contract.json"
        hlo_path.write_text(compiled.optimized_hlo)
        _atomic_write(
            contract_path,
            {
                "collective_algorithm": algorithm,
                "decode_shape_admissible": True,
                "hlo": hlo_report.to_dict(),
                "valid": True,
            },
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "physical_input_bits": artifact_dir / "physical_input_bits.npy",
            "hardware_output_bits": artifact_dir / "hardware_output_bits.npy",
            "software_row0_bits": artifact_dir / "software_row0_bits.npy",
        }
        _atomic_save(paths["physical_input_bits"], input_bits)
        _atomic_save(paths["hardware_output_bits"], output_bits)
        _atomic_save(paths["software_row0_bits"], software_row0)
        _atomic_write(artifact_dir / "comparison.json", comparison)
        for name, path in paths.items():
            value = np.load(path, allow_pickle=False)
            artifact_manifest[name] = {
                "array_sha256": array_sha256(value),
                "dtype": value.dtype.str,
                "file": path.name,
                "file_sha256": _file_sha256(path),
                "shape": list(value.shape),
            }
        _atomic_write(artifact_dir / "manifest.json", artifact_manifest)

    multihost_utils.sync_global_devices(f"greenfield-replay-end-{label}")
    print(
        "GREENFIELD_STRATEGY_ND_DENSE_REPLAY_OK "
        f"launch_process={args.process_id} jax_process={jax.process_index()} "
        f"classification={comparison['classification']} "
        f"mismatches={comparison['row0_mismatch_count']} hlo={hlo_sha256}",
        flush=True,
    )
    return {
        "accepted_model_axis_device_ids": list(model_axis_device_ids),
        "accepted_model_axis_recipe": ACCEPTED_TP32_MODEL_AXIS_RECIPE,
        "artifact_manifest": artifact_manifest,
        "capture": capture,
        "collective_algorithm": algorithm,
        "collective_groups": [list(physical_ids)],
        "comparison": comparison,
        "config": config.to_dict(),
        "diagnostic_only": True,
        "fleet_hashes": fleet_hashes,
        "fleet_hlo_hashes": fleet_hlo_hashes,
        "hlo": hlo_report.to_dict(),
        "member_device_ids": list(physical_ids),
        "optimized_hlo_sha256": hlo_sha256,
        "performance_claim": False,
        "source": {
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
            "npz_sha256": source_file_sha,
            "remote_objects_sha256": "663adbf1a11c32bbfc28b1030a0c329080d29fcf96fe4a2d77169e64e8858a05",
            "success_sha256": "9605aa5c0f76fd9a5ec9b8e78b1aa720111cbc0633d7b962834004537f321b23",
            "tag": "greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z",
        },
    }


def _run_strategy_nd_dense_rms_replay(
    args: argparse.Namespace,
    jax: Any,
    multihost_utils: Any,
    topology: Any,
) -> dict[str, Any]:
    """Consume the physical StrategyND result in the exact layer-1 boundary."""

    source_path = args.association_replay_input
    rms_source_path = args.association_rms_input
    if source_path is None or not source_path.is_file():
        raise RuntimeError("dense RMS replay requires the sealed DB550 NPZ")
    if rms_source_path is None or not rms_source_path.is_file():
        raise RuntimeError("dense RMS replay requires the sealed RMS source NPZ")
    source_file_sha = _file_sha256(source_path)
    if source_file_sha != ACCEPTED_DENSE_PARTIALS_NPZ_SHA256:
        raise RuntimeError("DB550 dense-partial NPZ SHA-256 drifted")
    with np.load(source_path, allow_pickle=False) as payload:
        if tuple(payload.files) != ACCEPTED_DENSE_PARTIALS_KEYS:
            raise RuntimeError("DB550 dense-partial NPZ keys drifted")
        accepted = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[0]])
        db548 = np.ascontiguousarray(payload[ACCEPTED_DENSE_PARTIALS_KEYS[1]])
    if any(
        value.shape != ACCEPTED_DENSE_PARTIALS_SHAPE
        or value.dtype != np.uint16
        or _raw_array_sha256(value) != ACCEPTED_DENSE_PARTIALS_RAW_SHA256
        for value in (accepted, db548)
    ) or not np.array_equal(accepted, db548):
        raise RuntimeError("DB550 dense-partial tensor contract drifted")
    rms_inputs = load_dense_rms_inputs(rms_source_path)
    if _file_sha256(rms_source_path) != DENSE_RMS_SOURCE_NPZ_SHA256:
        raise RuntimeError("dense RMS auxiliary source hash drifted")

    physical_ids = tuple(sorted(device.device_id for device in topology.devices))
    if physical_ids != tuple(range(32)):
        raise RuntimeError("dense RMS replay requires physical ids 0..31")
    model_axis_device_ids = accepted_tp32_model_axis_device_ids(jax.devices())
    model_bits = accepted.reshape(32, 6144)
    physical_bits = model_axis_to_physical_input_bits(
        model_bits, model_axis_device_ids
    )
    label = "strategy_nd_dense_rms_bfloat16_32x6144"
    multihost_utils.sync_global_devices(f"greenfield-rms-replay-start-{label}")
    compiled = build_strategy_nd_dense_rms_replay(
        physical_ids,
        devices=jax.devices(),
        enforce_optimized_hlo_contract=True,
    )
    stablehlo_sha = sha256(compiled.stablehlo.encode()).hexdigest()
    optimized_hlo_sha = sha256(compiled.optimized_hlo.encode()).hexdigest()
    fleet_stablehlo_hashes = _fleet_digest(
        multihost_utils,
        stablehlo_sha,
        label="dense RMS StableHLO",
        num_processes=args.num_processes,
    )
    fleet_hlo_hashes = _fleet_digest(
        multihost_utils,
        optimized_hlo_sha,
        label="dense RMS optimized HLO",
        num_processes=args.num_processes,
    )
    output_bits, capture = execute_strategy_nd_dense_rms_replay(
        compiled, physical_bits, rms_inputs
    )
    expected_bits = rms_inputs.accepted_layer1_bits
    mismatch_indices = np.flatnonzero(output_bits != expected_bits)
    first = None if not len(mismatch_indices) else int(mismatch_indices[0])
    observed_values = (
        output_bits.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    expected_values = (
        expected_bits.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    absolute_error = np.abs(observed_values - expected_values)
    output_raw_sha = _raw_array_sha256(output_bits)
    expected_raw_sha = _raw_array_sha256(expected_bits)
    db548_control_sha = (
        "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
    )
    classification = (
        "global_strategy_nd_rms_exact_accepted"
        if first is None
        else (
            "global_strategy_nd_rms_matches_db548_control"
            if output_raw_sha == db548_control_sha
            else "global_strategy_nd_rms_nonexact_new_result"
        )
    )
    comparison = {
        "classification": classification,
        "elementwise_exact": first is None,
        "expected_hidden_2795_bfloat16_bits": int(expected_bits[2795]),
        "expected_raw_sha256": expected_raw_sha,
        "first_mismatch_index": first,
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error, dtype=np.float64)),
        "mismatch_count": int(len(mismatch_indices)),
        "observed_hidden_2795_bfloat16_bits": int(output_bits[2795]),
        "observed_raw_sha256": output_raw_sha,
    }
    stable_hashes = {
        "accepted_target": array_sha256(expected_bits),
        "output": array_sha256(output_bits),
        "physical_input": array_sha256(physical_bits),
        "rms_source_file": DENSE_RMS_SOURCE_NPZ_SHA256,
        "source_file": source_file_sha,
    }
    fleet_hashes = {
        key: _fleet_digest(
            multihost_utils,
            value,
            label=f"dense RMS replay {key}",
            num_processes=args.num_processes,
        )
        for key, value in stable_hashes.items()
    }

    artifact_manifest: dict[str, Any] = {}
    replay_dir = args.output.parent / "rms_replay"
    hlo_dir = args.output.parent / "hlo"
    if jax.process_index() == 0:
        hlo_dir.mkdir(parents=True, exist_ok=True)
        stablehlo_path = hlo_dir / f"{label}.stablehlo.mlir"
        optimized_hlo_path = hlo_dir / f"{label}.optimized_hlo.txt"
        contract_path = hlo_dir / f"{label}.hlo_contract.json"
        stablehlo_path.write_text(compiled.stablehlo)
        optimized_hlo_path.write_text(compiled.optimized_hlo)
        _atomic_write(
            contract_path,
            {
                "collective_algorithm": compiled.collective_algorithm,
                "optimized": compiled.optimized_hlo_contract,
                "stablehlo": compiled.stablehlo_contract,
                "valid": True,
            },
        )
        paths = {
            "accepted_layer1_bits": replay_dir / "accepted_layer1_bits.npy",
            "hardware_layer1_bits": replay_dir / "hardware_layer1_bits.npy",
            "physical_input_bits": replay_dir / "physical_input_bits.npy",
        }
        _atomic_save(paths["accepted_layer1_bits"], expected_bits)
        _atomic_save(paths["hardware_layer1_bits"], output_bits)
        _atomic_save(paths["physical_input_bits"], physical_bits)
        _atomic_write(replay_dir / "comparison.json", comparison)
        for name, path in paths.items():
            value = np.load(path, allow_pickle=False)
            artifact_manifest[name] = {
                "array_sha256": array_sha256(value),
                "dtype": value.dtype.str,
                "file": path.name,
                "file_sha256": _file_sha256(path),
                "shape": list(value.shape),
            }
        _atomic_write(replay_dir / "manifest.json", artifact_manifest)

    multihost_utils.sync_global_devices(f"greenfield-rms-replay-end-{label}")
    print(
        "GREENFIELD_STRATEGY_ND_DENSE_RMS_REPLAY_OK "
        f"launch_process={args.process_id} jax_process={jax.process_index()} "
        f"classification={classification} mismatches={len(mismatch_indices)} "
        f"hlo={optimized_hlo_sha}",
        flush=True,
    )
    return {
        "accepted_model_axis_device_ids": list(model_axis_device_ids),
        "accepted_model_axis_recipe": ACCEPTED_TP32_MODEL_AXIS_RECIPE,
        "artifact_manifest": artifact_manifest,
        "capture": capture,
        "collective_algorithm": dict(compiled.collective_algorithm),
        "collective_groups": [list(physical_ids)],
        "comparison": comparison,
        "diagnostic_only": True,
        "fleet_hashes": fleet_hashes,
        "fleet_hlo_hashes": fleet_hlo_hashes,
        "fleet_stablehlo_hashes": fleet_stablehlo_hashes,
        "member_device_ids": list(physical_ids),
        "optimized_hlo_contract": dict(compiled.optimized_hlo_contract),
        "optimized_hlo_sha256": optimized_hlo_sha,
        "performance_claim": False,
        "source": {
            "db550_npz_sha256": source_file_sha,
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
        },
        "stablehlo_contract": dict(compiled.stablehlo_contract),
        "stablehlo_sha256": stablehlo_sha,
    }


def _run_strategy_nd_integrated_dense_rms(
    args: argparse.Namespace,
    jax: Any,
    multihost_utils: Any,
    topology: Any,
) -> dict[str, Any]:
    """Keep real rank-local contractions, StrategyND, and both RMS boundaries live."""

    from scripts.greenfield.probe_layer0_dense_convolution import (
        _FINAL_DENSE_LAYOUT_RECORDS,
        _pack_dense_final_layout,
    )
    from scripts.greenfield.probe_layer0_projection_reduction import (
        _load_weights,
    )

    if (
        args.association_rms_input is None
        or not args.association_rms_input.is_file()
        or args.checkpoint_root is None
        or not args.checkpoint_root.is_dir()
        or not args.checkpoint_manifest_sha256
    ):
        raise RuntimeError(
            "integrated dense RMS requires the sealed source and checkpoint"
        )
    validate_integrated_checkpoint_success(args.checkpoint_root)
    physical_ids = tuple(sorted(device.device_id for device in topology.devices))
    if physical_ids != tuple(range(32)):
        raise RuntimeError("integrated dense RMS requires physical ids 0..31")
    native_source_context = bool(args.integrated_native_source_context)
    weights, checkpoint_records = _load_weights(
        args.checkpoint_root,
        manifest_sha256=args.checkpoint_manifest_sha256,
        include_embedding=native_source_context,
    )
    packed, packed_records = _pack_dense_final_layout(weights)
    if packed_records != _FINAL_DENSE_LAYOUT_RECORDS:
        raise RuntimeError("integrated dense final-layout records drifted")
    inputs = load_integrated_dense_rms_inputs(
        args.association_rms_input,
        weights["attention.slot_00.post_norm"],
    )
    model_axis_device_ids = accepted_tp32_model_axis_device_ids(jax.devices())
    physical_weights = IntegratedDenseWeights(
        **{
            name: model_axis_weights_to_physical(
                value.reshape((32, 1) + value.shape[2:]),
                model_axis_device_ids,
            )
            for name, value in zip(
                ("merged_bits", "merged_scale", "down_bits", "down_scale"),
                packed,
                strict=True,
            )
        }
    )
    native_inputs = None
    native_weights = None
    if native_source_context:
        native_path = args.native_attention_input
        if (
            native_path is None
            or _file_sha256(native_path)
            != NATIVE_SOURCE_NPZ_SHA256
        ):
            raise RuntimeError("DB537 native attention source drifted")
        with np.load(native_path, allow_pickle=False) as payload:
            if NATIVE_SOURCE_ATTENDED_LATENT_KEY not in payload.files:
                raise RuntimeError("DB537 exact attended latent is absent")
            attended_latent_bits = np.ascontiguousarray(
                payload[NATIVE_SOURCE_ATTENDED_LATENT_KEY]
            )
        native_inputs = native_source_inputs(attended_latent_bits, inputs)
        native_weights = assemble_native_source_weights(
            weights,
            physical_weights,
            model_axis_device_ids,
        )
    split_layer1_rms = bool(args.integrated_split_layer1_rms)
    preceding_attention_collective = bool(
        args.integrated_preceding_attention_collective
    )
    split_predense_rms = bool(args.integrated_split_predense_rms)
    accepted_source_context = bool(args.integrated_accepted_source_context)
    label = (
        "strategy_nd_integrated_dense_native_source_context_bfloat16_32x6144"
        if native_source_context
        else "strategy_nd_integrated_dense_accepted_source_context_bfloat16_32x6144"
        if accepted_source_context
        else "strategy_nd_integrated_dense_predense_split_rms_bfloat16_32x6144"
        if split_predense_rms
        else "strategy_nd_integrated_dense_ordinal_rms_bfloat16_32x6144"
        if preceding_attention_collective
        else "strategy_nd_integrated_dense_split_rms_bfloat16_32x6144"
        if split_layer1_rms
        else "strategy_nd_integrated_dense_rms_bfloat16_32x6144"
    )
    multihost_utils.sync_global_devices(f"greenfield-integrated-start-{label}")
    compiled = build_integrated_dense_rms(
        physical_ids,
        devices=jax.devices(),
        validate_hlo=False,
        split_layer1_rms=split_layer1_rms,
        preceding_attention_collective=preceding_attention_collective,
        split_predense_rms=split_predense_rms,
        accepted_source_context=accepted_source_context,
        native_source_context=native_source_context,
    )
    stablehlo_sha = sha256(compiled.stablehlo.encode()).hexdigest()
    optimized_hlo_sha = sha256(compiled.optimized_hlo.encode()).hexdigest()
    replay_dir = args.output.parent / "integrated_dense_rms"
    hlo_dir = args.output.parent / "hlo"
    stablehlo_path = hlo_dir / f"{label}.stablehlo.mlir"
    optimized_hlo_path = hlo_dir / f"{label}.optimized_hlo.txt"
    contract_path = hlo_dir / f"{label}.hlo_contract.json"
    if jax.process_index() == 0:
        _atomic_write_text(stablehlo_path, compiled.stablehlo)
        _atomic_write_text(optimized_hlo_path, compiled.optimized_hlo)
        _atomic_write(
            hlo_dir / f"{label}.hlo_prevalidation.json",
            {
                "optimized_hlo_sha256": optimized_hlo_sha,
                "performance_claim": False,
                "split_layer1_rms": split_layer1_rms,
                **(
                    {"native_source_context": True}
                    if native_source_context
                    else {}
                ),
                **(
                    {"accepted_source_context": True}
                    if accepted_source_context
                    else {}
                ),
                **(
                    {"split_predense_rms": True}
                    if split_predense_rms
                    else {}
                ),
                "stablehlo_sha256": stablehlo_sha,
                "validated": False,
                **(
                    {"preceding_attention_collective": True}
                    if preceding_attention_collective
                    else {}
                ),
                },
            )
    # The native acquisition intentionally fails in the validator below.
    # Do not let another process tear down distributed JAX before process 0
    # has durably written the graphs needed for local validator iteration.
    multihost_utils.sync_global_devices(
        f"greenfield-integrated-hlo-persisted-{label}"
    )
    from glm_tpu.greenfield.benchmarking.integrated_dense_rms_hlo import (
        validate_integrated_dense_rms_hlo,
        validate_integrated_dense_rms_stablehlo,
    )

    stablehlo_contract = validate_integrated_dense_rms_stablehlo(
        compiled.stablehlo,
        split_layer1_rms=split_layer1_rms,
        preceding_attention_collective=preceding_attention_collective,
        split_predense_rms=split_predense_rms,
        accepted_source_context=accepted_source_context,
        native_source_context=native_source_context,
    )
    optimized_hlo_contract = validate_integrated_dense_rms_hlo(
        compiled.optimized_hlo,
        physical_ids,
        split_layer1_rms=split_layer1_rms,
        preceding_attention_collective=preceding_attention_collective,
        split_predense_rms=split_predense_rms,
        accepted_source_context=accepted_source_context,
        native_source_context=native_source_context,
    )
    compiled = replace(
        compiled,
        stablehlo_contract=stablehlo_contract,
        optimized_hlo_contract=optimized_hlo_contract,
    )
    fleet_stablehlo_hashes = _fleet_digest(
        multihost_utils,
        stablehlo_sha,
        label="integrated dense RMS StableHLO",
        num_processes=args.num_processes,
    )
    fleet_hlo_hashes = _fleet_digest(
        multihost_utils,
        optimized_hlo_sha,
        label="integrated dense RMS optimized HLO",
        num_processes=args.num_processes,
    )
    if native_source_context:
        if native_weights is None or native_inputs is None:
            raise RuntimeError("native source inputs were not assembled")
        output_bits, capture = execute_native_source_context(
            compiled,
            native_weights,
            native_inputs,
        )
        expected_bits = native_inputs.accepted_layer1_bits
    else:
        output_bits, capture = execute_integrated_dense_rms(
            compiled,
            physical_weights,
            inputs,
        )
        expected_bits = inputs.accepted_layer1_bits
    mismatch_indices = np.flatnonzero(output_bits != expected_bits)
    first = None if not len(mismatch_indices) else int(mismatch_indices[0])
    observed_values = (
        output_bits.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    expected_values = (
        expected_bits.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    absolute_error = np.abs(observed_values - expected_values)
    output_raw_sha = _raw_array_sha256(output_bits)
    expected_raw_sha = _raw_array_sha256(expected_bits)
    db548_control_sha = (
        "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005"
    )
    db549_rejected_sha = (
        "229dc8ace9bfa31fce6d6ccabc9fca49ccc55f30b9d1dd6f97a032f5117b812f"
    )
    classification_prefix = (
        "integrated_dense_native_source_context"
        if native_source_context
        else "integrated_dense_accepted_source_context"
        if accepted_source_context
        else "integrated_dense_predense_split_rms"
        if split_predense_rms
        else "integrated_dense_ordinal_rms"
        if preceding_attention_collective
        else "integrated_dense_split_rms"
        if split_layer1_rms
        else "integrated_dense_rms"
    )
    classification = (
        f"{classification_prefix}_exact_accepted"
        if first is None
        else f"{classification_prefix}_matches_db548_control"
        if output_raw_sha == db548_control_sha
        else f"{classification_prefix}_matches_rejected_db549"
        if output_raw_sha == db549_rejected_sha
        else f"{classification_prefix}_nonexact_new_result"
    )
    comparison = {
        "classification": classification,
        "elementwise_exact": first is None,
        "expected_hidden_2795_bfloat16_bits": int(expected_bits[2795]),
        "expected_raw_sha256": expected_raw_sha,
        "first_mismatch_index": first,
        "max_absolute_error": float(np.max(absolute_error)),
        "mean_absolute_error": float(np.mean(absolute_error, dtype=np.float64)),
        "mismatch_count": int(len(mismatch_indices)),
        "observed_hidden_2795_bfloat16_bits": int(output_bits[2795]),
        "observed_raw_sha256": output_raw_sha,
    }
    physical_weight_hashes = {
        name: array_sha256(getattr(physical_weights, name))
        for name in ("merged_bits", "merged_scale", "down_bits", "down_scale")
    }
    if native_source_context:
        assert native_weights is not None
        physical_weight_hashes.update(
            {
                name: array_sha256(getattr(native_weights, name))
                for name in (
                    "embedding",
                    "kv_b_bits",
                    "kv_b_scale",
                    "o_bits_in_out",
                    "o_scale_in_out",
                )
            }
        )
    stable_hashes = {
        "accepted_target": array_sha256(expected_bits),
        "layer1_norm": array_sha256(inputs.layer1_norm_bits),
        "post_attention_norm": array_sha256(inputs.post_attention_norm_bits),
        "output": array_sha256(output_bits),
        **(
            {}
            if native_source_context
            else {
                "attention_update": array_sha256(
                    inputs.attention_update_bits
                ),
                "combined_residual": array_sha256(
                    inputs.combined_residual_bits
                ),
            }
        ),
        **(
            {
                "attended_latent": array_sha256(
                    native_inputs.attended_latent_bits
                ),
                "native_source_token_ids": array_sha256(
                    native_inputs.token_ids
                ),
            }
            if native_source_context and native_inputs is not None
            else {}
        ),
        **(
            {"accepted_source_validity": array_sha256(ACCEPTED_SOURCE_VALIDITY)}
            if accepted_source_context
            else {}
        ),
        **{
            f"physical_{name}": digest
            for name, digest in physical_weight_hashes.items()
        },
    }
    fleet_hashes = {
        key: _fleet_digest(
            multihost_utils,
            value,
            label=f"integrated dense RMS {key}",
            num_processes=args.num_processes,
        )
        for key, value in stable_hashes.items()
    }
    artifact_manifest: dict[str, Any] = {}
    if jax.process_index() == 0:
        _atomic_write(
            contract_path,
            {
                "optimized": dict(compiled.optimized_hlo_contract),
                "stablehlo": dict(compiled.stablehlo_contract),
                "valid": True,
            },
        )
        paths = {
            "accepted_layer1_bits": replay_dir / "accepted_layer1_bits.npy",
            "hardware_layer1_bits": replay_dir / "hardware_layer1_bits.npy",
        }
        for name, path in paths.items():
            _atomic_save(
                path,
                expected_bits if name == "accepted_layer1_bits" else output_bits,
            )
            value = np.load(path, allow_pickle=False)
            artifact_manifest[name] = {
                "array_sha256": array_sha256(value),
                "dtype": value.dtype.str,
                "file": path.name,
                "file_sha256": _file_sha256(path),
                "shape": list(value.shape),
            }
        _atomic_write(replay_dir / "comparison.json", comparison)
        _atomic_write(replay_dir / "manifest.json", artifact_manifest)
    multihost_utils.sync_global_devices(f"greenfield-integrated-end-{label}")
    print(
        "GREENFIELD_STRATEGY_ND_INTEGRATED_DENSE_RMS_OK "
        f"launch_process={args.process_id} jax_process={jax.process_index()} "
        f"classification={classification} mismatches={len(mismatch_indices)} "
        f"hlo={optimized_hlo_sha}",
        flush=True,
    )
    return {
        "accepted_model_axis_device_ids": list(model_axis_device_ids),
        "accepted_model_axis_recipe": ACCEPTED_TP32_MODEL_AXIS_RECIPE,
        "artifact_manifest": artifact_manifest,
        "capture": dict(capture),
        "checkpoint_records": checkpoint_records,
        "collective_groups": [list(physical_ids)],
        "comparison": comparison,
        "diagnostic_only": True,
        "final_layout_records": packed_records,
        "fleet_hashes": fleet_hashes,
        "fleet_hlo_hashes": fleet_hlo_hashes,
        "fleet_stablehlo_hashes": fleet_stablehlo_hashes,
        "member_device_ids": list(physical_ids),
        "optimized_hlo_contract": dict(compiled.optimized_hlo_contract),
        "optimized_hlo_sha256": optimized_hlo_sha,
        "performance_claim": False,
        "physical_weight_hashes": physical_weight_hashes,
        "source": {
            "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
            "checkpoint_success_sha256": CHECKPOINT_SUCCESS_SHA256,
            "rms_npz_sha256": DENSE_RMS_SOURCE_NPZ_SHA256,
            "rms_tag": DENSE_RMS_SOURCE_TAG,
            **(
                {
                    "native_npz_sha256": NATIVE_SOURCE_NPZ_SHA256,
                    "native_remote_objects_sha256": (
                        NATIVE_SOURCE_REMOTE_OBJECTS_SHA256
                    ),
                    "native_runner_sha256": NATIVE_SOURCE_RUNNER_SHA256,
                    "native_success_sha256": NATIVE_SOURCE_SUCCESS_SHA256,
                    "native_summary_sha256": NATIVE_SOURCE_SUMMARY_SHA256,
                    "native_tag": NATIVE_SOURCE_TAG,
                }
                if native_source_context
                else {}
            ),
        },
        "stablehlo_contract": dict(compiled.stablehlo_contract),
        "stablehlo_sha256": stablehlo_sha,
        "split_layer1_rms": split_layer1_rms,
        **(
            {"native_source_context": True}
            if native_source_context
            else {}
        ),
        **(
            {"accepted_source_context": True}
            if accepted_source_context
            else {}
        ),
        **({"split_predense_rms": True} if split_predense_rms else {}),
        **(
            {"preceding_attention_collective": True}
            if preceding_attention_collective
            else {}
        ),
    }


def _run_strategy_nd_fingerprint(
    args: argparse.Namespace,
    jax: Any,
    multihost_utils: Any,
    topology: Any,
) -> dict[str, Any]:
    config = StrategyNdFingerprintConfig(trials=args.association_trials)
    if not args.allow_unprotected_test_config:
        config.require_protected_contract()
    physical_ids = tuple(sorted(device.device_id for device in topology.devices))
    if physical_ids != tuple(range(32)):
        raise RuntimeError(
            "accepted M32 fingerprint requires contiguous physical ids 0..31"
        )
    members = physical_ids
    groups = (members,)
    accepted_model_axis_device_ids = accepted_tp32_model_axis_device_ids(
        jax.devices()
    )
    label = "strategy_nd_association_bfloat16_32x6144"
    multihost_utils.sync_global_devices(f"greenfield-fingerprint-start-{label}")
    input_bits = generate_strategy_nd_input_bits(config)
    compiled = build_strategy_nd_fingerprint(
        config,
        members,
        devices=jax.devices(),
        enforce_hlo_contract=False,
    )
    hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
    fleet_hlo_hashes = _fleet_digest(
        multihost_utils,
        hlo_sha256,
        label="optimized HLO",
        num_processes=args.num_processes,
    )
    artifact_dir = args.output.parent / "hlo"
    if jax.process_index() == 0:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / f"{label}.optimized_hlo.txt").write_text(
            compiled.optimized_hlo
        )
    try:
        hlo_report, algorithm = validate_strategy_nd_fingerprint_hlo(
            compiled.optimized_hlo, members
        )
    except Exception as error:
        if jax.process_index() == 0:
            _atomic_write(
                artifact_dir / f"{label}.hlo_contract.json",
                {
                    "accepted_decode_projection_hlo_gzip_sha256": (
                        ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256
                    ),
                    "accepted_decode_projection_hlo_raw_sha256": (
                        ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256
                    ),
                    "accepted_decode_projection_manifest_sha256": (
                        ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256
                    ),
                    "decode_shape_admissible": False,
                    "decode_tree_claim": False,
                    "error": repr(error),
                    "generic_hlo": compiled.hlo_report.to_dict(),
                    "source_hlo_role": "decode_32_rows",
                    "valid": False,
                },
            )
        raise
    if jax.process_index() == 0:
        _atomic_write(
            artifact_dir / f"{label}.hlo_contract.json",
            {
                "accepted_decode_projection_hlo_gzip_sha256": (
                    ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256
                ),
                "accepted_decode_projection_hlo_raw_sha256": (
                    ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256
                ),
                "accepted_decode_projection_manifest_sha256": (
                    ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256
                ),
                "collective_algorithm": algorithm,
                "decode_shape_admissible": True,
                "decode_tree_claim": False,
                "hlo": hlo_report.to_dict(),
                "source_hlo_role": "decode_32_rows",
                "valid": True,
            },
        )
    output_bits, capture = execute_strategy_nd_fingerprint(compiled, input_bits)
    fleet_input_hashes = _fleet_digest(
        multihost_utils,
        capture["input_bits_sha256"],
        label="fingerprint input bits",
        num_processes=args.num_processes,
    )
    fleet_output_hashes = _fleet_digest(
        multihost_utils,
        capture["output_bits_sha256"],
        label="fingerprint output bits",
        num_processes=args.num_processes,
    )
    artifact_manifest: dict[str, Any] = {}
    if jax.process_index() == 0:
        association_dir = args.output.parent / "association"
        input_path = association_dir / "input_bits.npy"
        output_path = association_dir / "output_bits.npy"
        _atomic_save(input_path, input_bits)
        _atomic_save(output_path, output_bits)
        artifact_manifest = {
            "input_bits": {
                "array_sha256": array_sha256(input_bits),
                "dtype": input_bits.dtype.str,
                "file": input_path.name,
                "file_sha256": _file_sha256(input_path),
                "shape": list(input_bits.shape),
            },
            "output_bits": {
                "array_sha256": array_sha256(output_bits),
                "dtype": output_bits.dtype.str,
                "file": output_path.name,
                "file_sha256": _file_sha256(output_path),
                "shape": list(output_bits.shape),
            },
        }
        _atomic_write(association_dir / "manifest.json", artifact_manifest)
    multihost_utils.sync_global_devices(f"greenfield-fingerprint-end-{label}")
    print(
        "GREENFIELD_ASSOCIATION_FINGERPRINT_OK "
        f"launch_process={args.process_id} jax_process={jax.process_index()} "
        f"trials={config.trials} input={capture['input_bits_sha256']} "
        f"output={capture['output_bits_sha256']} hlo={hlo_sha256}",
        flush=True,
    )
    return {
        "accepted_model_axis_device_ids": list(accepted_model_axis_device_ids),
        "accepted_model_axis_recipe": ACCEPTED_TP32_MODEL_AXIS_RECIPE,
        "accepted_decode_projection_hlo_gzip_sha256": (
            ACCEPTED_DECODE_PROJECTION_HLO_GZIP_SHA256
        ),
        "accepted_decode_projection_hlo_raw_sha256": (
            ACCEPTED_DECODE_PROJECTION_HLO_RAW_SHA256
        ),
        "accepted_decode_projection_manifest_sha256": (
            ACCEPTED_DECODE_PROJECTION_MANIFEST_SHA256
        ),
        "artifact_manifest": artifact_manifest,
        "capture": capture,
        "collective_algorithm": algorithm,
        "collective_groups": [list(group) for group in groups],
        "config": config.to_dict(),
        "diagnostic_only": True,
        "decode_shape_admissible": True,
        "decode_tree_claim": False,
        "fleet_hlo_hashes": fleet_hlo_hashes,
        "fleet_input_bits_hashes": fleet_input_hashes,
        "fleet_output_bits_hashes": fleet_output_hashes,
        "hlo": hlo_report.to_dict(),
        "member_device_ids": list(members),
        "optimized_hlo_sha256": hlo_sha256,
        "source_hlo_role": "decode_32_rows",
    }


def _discover_runtime_topology(
    jax: Any,
    multihost_utils: Any,
    *,
    slice_name: str,
    num_processes: int,
) -> Any:
    local_ids = np.asarray(
        [device.id for device in jax.local_devices()], dtype=np.int32
    )
    fleet_local_ids = np.asarray(
        multihost_utils.process_allgather(local_ids)
    ).reshape(num_processes, jax.local_device_count())
    observed_local_order = {
        int(device_id): local_index
        for process_row in fleet_local_ids
        for local_index, device_id in enumerate(process_row.tolist())
    }
    topology = discover_physical_topology(
        jax.devices(),
        slice_name=slice_name,
        observed_local_order=observed_local_order,
    )
    validate_target_v4_64(topology)
    return topology, fleet_local_ids


def _fleet_digest(
    multihost_utils: Any,
    digest_hex: str,
    *,
    label: str,
    num_processes: int,
) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    fleet = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        num_processes, len(digest)
    )
    values = [row.tobytes().hex() for row in fleet]
    if len(set(values)) != 1:
        raise RuntimeError(f"hosts disagree on {label}: {values}")
    return values


def main() -> int:
    args = parse_args()
    if args.num_processes != 8 or not 0 <= args.process_id < args.num_processes:
        raise ValueError("protected collective benchmark requires process ids 0..7")
    if tuple(args.groups) != tuple(sorted(set(args.groups))) or set(args.groups) - {
        2,
        4,
        8,
        32,
    }:
        raise ValueError("groups must be unique sorted values from 2,4,8,32")
    strategy_nd_modes = {
        "strategy_nd_fingerprint",
        "strategy_nd_dense_replay",
        "strategy_nd_dense_rms_replay",
        "strategy_nd_integrated_dense_rms",
    }
    if args.mode in strategy_nd_modes and (
        tuple(args.groups) != (32,)
        or tuple(args.operations) != (CollectiveKind.ALL_REDUCE,)
        or tuple(args.shape) != (32, 6144)
        or args.dtype != "bfloat16"
    ):
        raise ValueError(
            "StrategyND fingerprint requires groups=32, operation=all_reduce, "
            "shape=32,6144, and dtype=bfloat16"
        )
    if args.mode in {
        "strategy_nd_dense_replay",
        "strategy_nd_dense_rms_replay",
        "strategy_nd_integrated_dense_rms",
    } and args.association_trials != 1:
        raise ValueError("StrategyND real-data replay requires association-trials=1")
    if args.mode not in {
        "strategy_nd_dense_replay",
        "strategy_nd_dense_rms_replay",
    } and args.association_replay_input:
        raise ValueError("association replay input is valid only for real-data replay")
    rms_modes = {
        "strategy_nd_dense_rms_replay",
        "strategy_nd_integrated_dense_rms",
    }
    if args.mode not in rms_modes and args.association_rms_input:
        raise ValueError("association RMS input is valid only for dense RMS replay")
    if args.mode in rms_modes and not args.association_rms_input:
        raise ValueError("dense RMS replay requires association-rms-input")
    if args.mode == "strategy_nd_integrated_dense_rms":
        if not args.checkpoint_root or not args.checkpoint_manifest_sha256:
            raise ValueError("integrated dense RMS requires exact checkpoint inputs")
    elif args.checkpoint_root or args.checkpoint_manifest_sha256:
        raise ValueError("checkpoint inputs are valid only for integrated dense RMS")
    if args.integrated_split_layer1_rms and args.mode != (
        "strategy_nd_integrated_dense_rms"
    ):
        raise ValueError(
            "integrated split RMS is valid only for integrated dense RMS"
        )
    if args.integrated_preceding_attention_collective and (
        args.mode != "strategy_nd_integrated_dense_rms"
        or not args.integrated_split_layer1_rms
    ):
        raise ValueError(
            "preceding attention collective requires integrated split RMS"
        )
    if args.integrated_split_predense_rms and (
        args.mode != "strategy_nd_integrated_dense_rms"
        or not args.integrated_split_layer1_rms
        or args.integrated_preceding_attention_collective
    ):
        raise ValueError(
            "pre-dense split RMS requires the disjoint integrated layer-1 split arm"
        )
    if args.integrated_accepted_source_context and (
        args.mode != "strategy_nd_integrated_dense_rms"
        or args.integrated_split_layer1_rms
        or args.integrated_preceding_attention_collective
        or args.integrated_split_predense_rms
    ):
        raise ValueError(
            "accepted source context requires the disjoint integrated dense RMS arm"
        )
    if args.integrated_native_source_context and (
        args.mode != "strategy_nd_integrated_dense_rms"
        or args.integrated_accepted_source_context
        or args.integrated_split_layer1_rms
        or args.integrated_preceding_attention_collective
        or args.integrated_split_predense_rms
        or args.native_attention_input is None
        or not args.native_attention_input.is_file()
    ):
        raise ValueError(
            "native source context requires its disjoint integrated mode and DB537 input"
        )
    if args.native_attention_input is not None and not (
        args.integrated_native_source_context
    ):
        raise ValueError(
            "native attention input is valid only for native source context"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    run_tag = os.environ.get("GLM_GREENFIELD_RUN_TAG", "")
    if args.mode in {
        "strategy_nd_dense_replay",
        "strategy_nd_dense_rms_replay",
        "strategy_nd_integrated_dense_rms",
    } and not run_tag:
        raise RuntimeError("StrategyND real-data replay requires GLM_GREENFIELD_RUN_TAG")

    import jax
    from jax.experimental import multihost_utils

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    try:
        if (
            jax.process_count() != 8
            or jax.local_device_count() != 4
            or jax.device_count() != 32
        ):
            raise RuntimeError(
                "protected benchmark requires 8 processes, 4 local chips, and 32 total chips"
            )
        topology, fleet_local_ids = _discover_runtime_topology(
            jax,
            multihost_utils,
            slice_name=args.slice_name,
            num_processes=args.num_processes,
        )
        association_fingerprint = None
        association_dense_replay = None
        association_dense_rms_replay = None
        association_integrated_dense_rms = None
        matrix = []
        if args.mode == "strategy_nd_fingerprint":
            association_fingerprint = _run_strategy_nd_fingerprint(
                args, jax, multihost_utils, topology
            )
        elif args.mode == "strategy_nd_dense_replay":
            association_dense_replay = _run_strategy_nd_dense_replay(
                args, jax, multihost_utils, topology
            )
        elif args.mode == "strategy_nd_dense_rms_replay":
            association_dense_rms_replay = _run_strategy_nd_dense_rms_replay(
                args, jax, multihost_utils, topology
            )
        elif args.mode == "strategy_nd_integrated_dense_rms":
            association_integrated_dense_rms = (
                _run_strategy_nd_integrated_dense_rms(
                    args, jax, multihost_utils, topology
                )
            )
        else:
            for group_size in args.groups:
                groups = collective_groups_for_size(topology, group_size)
                for kind in args.operations:
                    config = CollectiveChainConfig(
                        kind=kind,
                        group_size=group_size,
                        rows=args.shape[0],
                        width=args.shape[1],
                        dtype=args.dtype,
                        chain_length=args.chain_length,
                        warmup_iterations=args.warmup,
                        measured_iterations=args.iterations,
                    )
                    if not args.allow_unprotected_test_config:
                        config.require_protected_contract()
                    label = (
                        f"{kind.value}_g{group_size}_{args.dtype}_"
                        f"{args.shape[0]}x{args.shape[1]}"
                    )
                    multihost_utils.sync_global_devices(f"greenfield-chain-start-{label}")
                    compiled = build_collective_chain(
                        config,
                        groups,
                        devices=jax.devices(),
                        enforce_hlo_contract=False,
                    )
                    hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
                    fleet_hlo_hashes = _fleet_digest(
                        multihost_utils,
                        hlo_sha256,
                        label="optimized HLO",
                        num_processes=args.num_processes,
                    )
                    if jax.process_index() == 0:
                        artifact_dir = args.output.parent / "hlo"
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        (artifact_dir / f"{label}.optimized_hlo.txt").write_text(
                            compiled.optimized_hlo
                        )
                        _atomic_write(
                            artifact_dir / f"{label}.hlo_contract.json",
                            compiled.hlo_report.to_dict(),
                        )
                    if not compiled.hlo_report.valid:
                        print(
                            "GREENFIELD_COLLECTIVE_HLO_REJECTED "
                            + json.dumps(
                                {
                                    "case": label,
                                    "collective_counts": compiled.hlo_report.to_dict()[
                                        "collective_counts"
                                    ],
                                    "hlo_sha256": hlo_sha256,
                                    "violations": [
                                        violation.to_dict()
                                        for violation in compiled.hlo_report.violations
                                    ],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        compiled.hlo_report.raise_for_violations()
                    measured = benchmark_collective_chain(compiled)
                    measured.update(
                        {
                            "collective_groups": [list(group) for group in groups],
                            "fleet_hlo_hashes": fleet_hlo_hashes,
                            "optimized_hlo_sha256": hlo_sha256,
                        }
                    )
                    matrix.append(measured)
                    print(
                        "GREENFIELD_COLLECTIVE_CASE_OK "
                        f"launch_process={args.process_id} "
                        f"jax_process={jax.process_index()} case={label} "
                        f"p50_ms={measured['latency']['p50_ms']:.6f} "
                        f"hlo={hlo_sha256}",
                        flush=True,
                    )
                    multihost_utils.sync_global_devices(f"greenfield-chain-end-{label}")

        record = {
            "association_dense_replay": association_dense_replay,
            "association_fingerprint": association_fingerprint,
            "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_hash": code_hash,
            "fleet_local_device_ids_in_runtime_order": fleet_local_ids.tolist(),
            "hostname": socket.gethostname(),
            "jax_process_index": jax.process_index(),
            "jax_version": jax.__version__,
            "launch_process_id": args.process_id,
            "matrix": matrix,
            "mechanism_only": True,
            "mode": args.mode,
            "run_tag": run_tag,
            "schema_version": (
                5
                if association_integrated_dense_rms is not None
                else 4
                if association_dense_rms_replay is not None
                else 3
                if association_dense_replay is not None
                else 2
                if association_fingerprint is not None
                else 1
            ),
            "topology": topology.to_dict(),
            "topology_hash": topology.topology_hash,
        }
        if association_dense_rms_replay is not None:
            record["association_dense_rms_replay"] = association_dense_rms_replay
        if association_integrated_dense_rms is not None:
            record["association_integrated_dense_rms"] = (
                association_integrated_dense_rms
            )
        _atomic_write(args.output, record)
        print(
            "GREENFIELD_COLLECTIVE_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"host={record['hostname']} cases={len(matrix)} output={args.output}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
