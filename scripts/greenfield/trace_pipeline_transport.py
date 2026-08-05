#!/usr/bin/env python3
"""Capture fresh fleet XPlanes for exact PP8 and PP16 transport programs."""

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
    TransportChainConfig,
    TransportKind,
    build_transport_chain,
    validate_compiled_transport,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
    discover_physical_topology,
    stage_transfer_lanes,
    stage_transfer_pairs,
    validate_target_v4_64,
)
from glm_tpu.greenfield.types import PlanName  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", default="db-v4-64-od")
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trace-steps", type=int, default=20)
    return parser.parse_args()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
    return topology


def _file_record(path: Path) -> dict[str, Any]:
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    if (
        args.num_processes != 8
        or not 0 <= args.process_id < args.num_processes
        or args.warmup < 1
        or args.trace_steps != 20
    ):
        raise ValueError("protected trace requires 8 processes, warmup, and 20 steps")
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
            raise RuntimeError("protected trace requires the complete v4-64 slice")
        topology = _discover_runtime_topology(
            jax,
            multihost_utils,
            slice_name=args.slice_name,
            num_processes=args.num_processes,
        )
        traces = []
        for plan in (PlanName.PP8_LP4, PlanName.PP16_LP2):
            groups = (
                build_pp8_lp4_groups(topology)
                if plan is PlanName.PP8_LP4
                else build_pp16_lp2_groups(topology)
            )
            lanes = stage_transfer_lanes(topology, groups)
            pairs = stage_transfer_pairs(topology, groups)
            config = TransportChainConfig(
                plan=plan,
                kind=TransportKind.DEVICE_RESIDENT,
                rows=1,
                width=6144,
                dtype="bfloat16",
                warmup_iterations=200,
                measured_iterations=1000,
            )
            compiled = build_transport_chain(
                config,
                pairs,
                devices=jax.devices(),
                enforce_hlo_contract=False,
            )
            validate_compiled_transport(compiled)
            for _ in range(args.warmup):
                jax.block_until_ready(compiled.compiled(compiled.input_value))
            label = plan.value.lower()
            trace_dir = args.trace_root / label
            trace_dir.mkdir(parents=True, exist_ok=False)
            multihost_utils.sync_global_devices(f"greenfield-trace-ready-{label}")
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = 0
            tracing = False
            try:
                jax.profiler.start_trace(
                    str(trace_dir),
                    profiler_options=options,
                )
                tracing = True
                multihost_utils.sync_global_devices(
                    f"greenfield-trace-started-{label}"
                )
                for step in range(args.trace_steps):
                    with jax.profiler.TraceAnnotation(
                        "greenfield_transport_step",
                        step_num=step,
                        plan=plan.value,
                    ):
                        result = compiled.compiled(compiled.input_value)
                        jax.block_until_ready(result)
                jax.profiler.stop_trace()
                tracing = False
            finally:
                if tracing:
                    jax.profiler.stop_trace()
            multihost_utils.sync_global_devices(f"greenfield-trace-stopped-{label}")
            xplanes = tuple(sorted(trace_dir.rglob("*.xplane.pb")))
            if len(xplanes) != 1:
                raise RuntimeError(
                    f"{plan.value} expected one local XPlane, found {len(xplanes)}"
                )
            traces.append(
                {
                    "config": config.to_dict(),
                    "hlo_collective_counts": compiled.hlo_report.to_dict()[
                        "collective_counts"
                    ],
                    "hlo_sha256": sha256(compiled.optimized_hlo.encode()).hexdigest(),
                    "physical_lanes": [list(lane) for lane in lanes],
                    "physical_pairs": [list(pair) for pair in pairs],
                    "trace_steps": args.trace_steps,
                    "xplane": _file_record(xplanes[0]),
                }
            )
            print(
                "GREENFIELD_TRANSPORT_TRACE_PLAN_OK "
                f"launch_process={args.process_id} jax_process={jax.process_index()} "
                f"plan={plan.value} xplane={xplanes[0]}",
                flush=True,
            )

        _atomic_write(
            args.output,
            {
                "captured_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "code_hash": code_hash,
                "hostname": socket.gethostname(),
                "jax_process_index": jax.process_index(),
                "launch_process_id": args.process_id,
                "schema_version": 1,
                "topology_hash": topology.topology_hash,
                "traces": traces,
            },
        )
        print(
            "GREENFIELD_TRANSPORT_TRACE_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"output={args.output}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
