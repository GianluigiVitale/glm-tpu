#!/usr/bin/env python3
"""Capture and cross-host-verify the physical topology for Gate A.

This script is intentionally model-free.  Launch it once on every TPU VM with
the same coordinator, branch pin, and output path.  Each host writes its own
local record; the launcher is responsible for gathering all eight append-only
records into the protected run directory.
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

from glm_tpu.greenfield.topology import (  # noqa: E402
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
    discover_physical_topology,
    group_manifest_hash,
    groups_to_dict,
    validate_target_v4_64,
)
from glm_tpu.greenfield.types import PlanName  # noqa: E402


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--slice-name", default="db-v4-64-od")
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_processes != 8 or not 0 <= args.process_id < args.num_processes:
        raise ValueError("Gate-A topology capture requires process ids 0..7")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )

    import jax
    from jax.experimental import multihost_utils

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    try:
        # TPU JAX may topology-order processes differently from TPU-VM worker
        # suffixes. Preserve both identities; never assume they are equal.
        jax_process_index = jax.process_index()
        topology = discover_physical_topology(
            jax.devices(), slice_name=args.slice_name
        )
        validate_target_v4_64(topology)
        pp8_groups = build_pp8_lp4_groups(topology)
        pp16_groups = build_pp16_lp2_groups(topology)

        local_observed = {device.id for device in jax.local_devices()}
        local_recorded = {
            device.device_id
            for device in topology.devices
            if device.process_index == jax_process_index
        }
        if local_observed != local_recorded:
            raise RuntimeError(
                "local JAX devices disagree with the global physical inventory: "
                f"observed={sorted(local_observed)} recorded={sorted(local_recorded)}"
            )

        contract = {
            "code_hash": code_hash,
            "pp16_lp2": groups_to_dict(PlanName.PP16_LP2, pp16_groups),
            "pp16_lp2_hash": group_manifest_hash(
                PlanName.PP16_LP2, pp16_groups
            ),
            "pp8_lp4": groups_to_dict(PlanName.PP8_LP4, pp8_groups),
            "pp8_lp4_hash": group_manifest_hash(PlanName.PP8_LP4, pp8_groups),
            "topology": topology.to_dict(),
            "topology_hash": topology.topology_hash,
        }
        contract_hash = sha256(_canonical_json(contract).encode("utf-8")).hexdigest()
        digest = np.frombuffer(bytes.fromhex(contract_hash), dtype=np.uint8)
        fleet_digests = np.asarray(multihost_utils.process_allgather(digest))
        fleet_digests = fleet_digests.reshape(args.num_processes, len(digest))
        if not np.all(fleet_digests == fleet_digests[0]):
            observed = [row.tobytes().hex() for row in fleet_digests]
            raise RuntimeError(f"hosts disagree on topology contract: {observed}")
        multihost_utils.sync_global_devices(
            f"greenfield-topology-{contract_hash[:16]}"
        )

        record = {
            "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "contract": contract,
            "contract_hash": contract_hash,
            "fleet_contract_hashes": [
                row.tobytes().hex() for row in fleet_digests
            ],
            "hostname": socket.gethostname(),
            "jax_device_count": jax.device_count(),
            "jax_local_device_count": jax.local_device_count(),
            "jax_process_count": jax.process_count(),
            "jax_process_index": jax_process_index,
            "jax_version": jax.__version__,
            "launch_process_id": args.process_id,
            "local_device_ids": sorted(local_observed),
            "schema_version": 1,
        }
        _atomic_write(args.output, record)
        print(
            "GREENFIELD_TOPOLOGY_OK "
            f"launch_process={args.process_id} jax_process={jax_process_index} "
            f"host={record['hostname']} "
            f"devices={jax.device_count()} local={jax.local_device_count()} "
            f"contract={contract_hash} output={args.output}",
            flush=True,
        )
        return 0
    finally:
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
