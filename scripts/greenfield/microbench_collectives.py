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
    ACCEPTED_PROMPT_PROJECTION_HLO_SHA256,
    CollectiveChainConfig,
    CollectiveKind,
    StrategyNdFingerprintConfig,
    array_sha256,
    benchmark_collective_chain,
    build_collective_chain,
    build_strategy_nd_fingerprint,
    execute_strategy_nd_fingerprint,
    generate_strategy_nd_input_bits,
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
        choices=("chain", "strategy_nd_fingerprint"),
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
        "--allow-unprotected-test-config",
        action="store_true",
        help="permit fewer than the protected 75/200/1000 contract",
    )
    return parser.parse_args()


def _run_strategy_nd_fingerprint(
    args: argparse.Namespace,
    jax: Any,
    multihost_utils: Any,
    topology: Any,
) -> dict[str, Any]:
    config = StrategyNdFingerprintConfig(trials=args.association_trials)
    if not args.allow_unprotected_test_config:
        config.require_protected_contract()
    groups = collective_groups_for_size(topology, 32)
    if len(groups) != 1:
        raise RuntimeError("StrategyND fingerprint requires one 32-device group")
    members = groups[0]
    label = "strategy_nd_association_bfloat16_1x6144"
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
                    "accepted_prompt_projection_hlo_sha256": (
                        ACCEPTED_PROMPT_PROJECTION_HLO_SHA256
                    ),
                    "decode_tree_claim": False,
                    "error": repr(error),
                    "generic_hlo": compiled.hlo_report.to_dict(),
                    "source_hlo_role": "prefill_2048_rows",
                    "valid": False,
                },
            )
        raise
    if jax.process_index() == 0:
        _atomic_write(
            artifact_dir / f"{label}.hlo_contract.json",
            {
                "accepted_prompt_projection_hlo_sha256": (
                    ACCEPTED_PROMPT_PROJECTION_HLO_SHA256
                ),
                "collective_algorithm": algorithm,
                "decode_tree_claim": False,
                "hlo": hlo_report.to_dict(),
                "source_hlo_role": "prefill_2048_rows",
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
        "accepted_prompt_projection_hlo_sha256": (
            ACCEPTED_PROMPT_PROJECTION_HLO_SHA256
        ),
        "artifact_manifest": artifact_manifest,
        "capture": capture,
        "collective_algorithm": algorithm,
        "collective_groups": [list(group) for group in groups],
        "config": config.to_dict(),
        "diagnostic_only": True,
        "decode_tree_claim": False,
        "fleet_hlo_hashes": fleet_hlo_hashes,
        "fleet_input_bits_hashes": fleet_input_hashes,
        "fleet_output_bits_hashes": fleet_output_hashes,
        "hlo": hlo_report.to_dict(),
        "member_device_ids": list(members),
        "optimized_hlo_sha256": hlo_sha256,
        "source_hlo_role": "prefill_2048_rows",
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
    if args.mode == "strategy_nd_fingerprint" and (
        tuple(args.groups) != (32,)
        or tuple(args.operations) != (CollectiveKind.ALL_REDUCE,)
        or tuple(args.shape) != (1, 6144)
        or args.dtype != "bfloat16"
    ):
        raise ValueError(
            "StrategyND fingerprint requires groups=32, operation=all_reduce, "
            "shape=1,6144, and dtype=bfloat16"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")

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
        matrix = []
        if args.mode == "strategy_nd_fingerprint":
            association_fingerprint = _run_strategy_nd_fingerprint(
                args, jax, multihost_utils, topology
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
            "schema_version": 2 if association_fingerprint is not None else 1,
            "topology": topology.to_dict(),
            "topology_hash": topology.topology_hash,
        }
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
