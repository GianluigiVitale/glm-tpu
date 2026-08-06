#!/usr/bin/env python3
"""Run PP8/PP16 device-resident transport chains on all TPU hosts."""

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
    PairedTransportConfig,
    TransportChainConfig,
    TransportKind,
    benchmark_paired_transport,
    benchmark_transport_chain,
    build_paired_transport,
    build_transport_chain,
    validate_compiled_transport,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
    discover_physical_topology,
    group_manifest_hash,
    stage_transfer_lanes,
    stage_transfer_pairs,
    validate_target_v4_64,
)
from glm_tpu.greenfield.types import PlanName  # noqa: E402


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _plans(value: str) -> tuple[PlanName, ...]:
    try:
        result = tuple(PlanName(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    allowed = {PlanName.PP8_LP4, PlanName.PP16_LP2}
    if not result or len(result) != len(set(result)) or set(result) - allowed:
        raise argparse.ArgumentTypeError(
            "plans must be unique values from PP8_LP4,PP16_LP2"
        )
    return result


def _kinds(value: str) -> tuple[TransportKind, ...]:
    try:
        result = tuple(
            TransportKind(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("transport kinds must be non-empty and unique")
    return result


def _payloads(value: str) -> tuple[tuple[str, int, int], ...]:
    result = []
    for raw in value.split(","):
        parts = raw.strip().split(":")
        if len(parts) != 3 or parts[0] not in {"bfloat16", "float32", "int32"}:
            raise argparse.ArgumentTypeError(
                "payloads must use DTYPE:ROWS:WIDTH entries"
            )
        try:
            rows, width = int(parts[1]), int(parts[2])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "payload rows and width must be integers"
            ) from error
        if rows <= 0 or width <= 0:
            raise argparse.ArgumentTypeError("payload dimensions must be positive")
        result.append((parts[0], rows, width))
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("payloads must be non-empty and unique")
    return tuple(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", default="db-v4-64-od")
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plans",
        type=_plans,
        default=(PlanName.PP8_LP4, PlanName.PP16_LP2),
    )
    parser.add_argument(
        "--kinds",
        type=_kinds,
        default=tuple(TransportKind),
    )
    parser.add_argument(
        "--payloads",
        type=_payloads,
        default=(
            ("bfloat16", 1, 6144),
            ("bfloat16", 2, 6144),
            ("bfloat16", 1, 2048),
            ("int32", 1, 2048),
        ),
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--paired-production", action="store_true")
    parser.add_argument("--allow-unprotected-test-config", action="store_true")
    return parser.parse_args()


def _discover_runtime_topology(
    jax: Any,
    multihost_utils: Any,
    *,
    slice_name: str,
    num_processes: int,
) -> tuple[Any, np.ndarray]:
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
        raise ValueError("protected transport benchmark requires process ids 0..7")
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
                "protected transport requires 8 processes, 4 local chips, and 32 chips"
            )
        topology, fleet_local_ids = _discover_runtime_topology(
            jax,
            multihost_utils,
            slice_name=args.slice_name,
            num_processes=args.num_processes,
        )
        matrix = []
        paired_matrix = []
        plan_contracts = {}
        for plan in args.plans:
            groups = (
                build_pp8_lp4_groups(topology)
                if plan is PlanName.PP8_LP4
                else build_pp16_lp2_groups(topology)
            )
            lanes = stage_transfer_lanes(topology, groups)
            pairs = stage_transfer_pairs(topology, groups)
            plan_contracts[plan.value] = {
                "group_manifest_hash": group_manifest_hash(plan, groups),
                "groups": [group.to_dict() for group in groups],
                "physical_lanes": [list(lane) for lane in lanes],
                "physical_pairs": [list(pair) for pair in pairs],
            }
            for dtype, rows, width in args.payloads:
                for kind in args.kinds:
                    config = TransportChainConfig(
                        plan=plan,
                        kind=kind,
                        rows=rows,
                        width=width,
                        dtype=dtype,
                        warmup_iterations=args.warmup,
                        measured_iterations=args.iterations,
                    )
                    if not args.allow_unprotected_test_config:
                        config.require_protected_contract()
                    label = (
                        f"{plan.value.lower()}_{kind.value}_{dtype}_{rows}x{width}"
                    )
                    multihost_utils.sync_global_devices(
                        f"greenfield-transport-start-{label}"
                    )
                    compiled = build_transport_chain(
                        config,
                        pairs,
                        devices=jax.devices(),
                        enforce_hlo_contract=False,
                    )
                    hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
                    fleet_hlo_hashes = _fleet_digest(
                        multihost_utils,
                        hlo_sha256,
                        num_processes=args.num_processes,
                    )
                    if jax.process_index() == 0:
                        artifact_dir = args.output.parent / "hlo"
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        (artifact_dir / f"{label}.optimized_hlo.txt").write_text(
                            compiled.optimized_hlo
                        )
                    validate_compiled_transport(compiled)
                    measured = benchmark_transport_chain(compiled)
                    if jax.process_index() == 0:
                        contract = compiled.hlo_report.to_dict()
                        contract["pallas_remote_copy"] = measured[
                            "pallas_remote_copy"
                        ]
                        _atomic_write(
                            artifact_dir / f"{label}.hlo_contract.json",
                            contract,
                        )
                    measured["fleet_hlo_hashes"] = fleet_hlo_hashes
                    measured["optimized_hlo_sha256"] = hlo_sha256
                    matrix.append(measured)
                    print(
                        "GREENFIELD_TRANSPORT_CASE_OK "
                        f"launch_process={args.process_id} "
                        f"jax_process={jax.process_index()} case={label} "
                        f"p50_ms={measured['latency']['p50_ms']:.6f} "
                        f"hlo={hlo_sha256}",
                        flush=True,
                    )
                    multihost_utils.sync_global_devices(
                        f"greenfield-transport-end-{label}"
                    )
            if args.paired_production:
                for kind in args.kinds:
                    config = PairedTransportConfig(
                        plan=plan,
                        kind=kind,
                        warmup_iterations=args.warmup,
                        measured_iterations=args.iterations,
                    )
                    if not args.allow_unprotected_test_config:
                        config.require_protected_contract()
                    label = f"{plan.value.lower()}_{kind.value}_paired_production"
                    multihost_utils.sync_global_devices(
                        f"greenfield-paired-transport-start-{label}"
                    )
                    compiled = build_paired_transport(
                        config,
                        pairs,
                        devices=jax.devices(),
                    )
                    hlo_sha256 = sha256(compiled.optimized_hlo.encode()).hexdigest()
                    fleet_hlo_hashes = _fleet_digest(
                        multihost_utils,
                        hlo_sha256,
                        num_processes=args.num_processes,
                    )
                    measured = benchmark_paired_transport(compiled)
                    measured["fleet_hlo_hashes"] = fleet_hlo_hashes
                    measured["optimized_hlo_sha256"] = hlo_sha256
                    paired_matrix.append(measured)
                    if jax.process_index() == 0:
                        artifact_dir = args.output.parent / "hlo"
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        (artifact_dir / f"{label}.optimized_hlo.txt").write_text(
                            compiled.optimized_hlo
                        )
                        _atomic_write(
                            artifact_dir / f"{label}.hlo_contract.json",
                            compiled.hlo_contract,
                        )
                    print(
                        "GREENFIELD_PAIRED_TRANSPORT_CASE_OK "
                        f"launch_process={args.process_id} "
                        f"jax_process={jax.process_index()} case={label} "
                        f"p50_ms={measured['latency']['p50_ms']:.6f} "
                        f"hlo={hlo_sha256}",
                        flush=True,
                    )
                    multihost_utils.sync_global_devices(
                        f"greenfield-paired-transport-end-{label}"
                    )

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
            "model_equivalent_compute": False,
            "paired_matrix": paired_matrix,
            "plan_contracts": plan_contracts,
            "schema_version": 1,
            "single_compiled_invocation_per_case": True,
            "stage_dispatch": "device_program_only",
            "topology": topology.to_dict(),
            "topology_hash": topology.topology_hash,
        }
        _atomic_write(args.output, record)
        print(
            "GREENFIELD_TRANSPORT_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"cases={len(matrix)} output={args.output}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
