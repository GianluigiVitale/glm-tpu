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
    CollectiveChainConfig,
    CollectiveKind,
    benchmark_collective_chain,
    build_collective_chain,
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
    parser.add_argument(
        "--allow-unprotected-test-config",
        action="store_true",
        help="permit fewer than the protected 75/200/1000 contract",
    )
    return parser.parse_args()


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
    num_processes: int,
) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    fleet = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        num_processes, len(digest)
    )
    values = [row.tobytes().hex() for row in fleet]
    if len(set(values)) != 1:
        raise RuntimeError(f"hosts disagree on optimized HLO: {values}")
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
        matrix = []
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
                )
                hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
                fleet_hlo_hashes = _fleet_digest(
                    multihost_utils,
                    hlo_sha256,
                    num_processes=args.num_processes,
                )
                measured = benchmark_collective_chain(compiled)
                measured.update(
                    {
                        "collective_groups": [list(group) for group in groups],
                        "fleet_hlo_hashes": fleet_hlo_hashes,
                        "optimized_hlo_sha256": hlo_sha256,
                    }
                )
                matrix.append(measured)
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
            "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_hash": code_hash,
            "fleet_local_device_ids_in_runtime_order": fleet_local_ids.tolist(),
            "hostname": socket.gethostname(),
            "jax_process_index": jax.process_index(),
            "jax_version": jax.__version__,
            "launch_process_id": args.process_id,
            "matrix": matrix,
            "mechanism_only": True,
            "schema_version": 1,
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
