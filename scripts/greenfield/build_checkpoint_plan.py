#!/usr/bin/env python3
"""Build an append-only full-checkpoint inventory and final-layout plan.

This is a metadata-only Gate-B operation.  It reads the 141 safetensors
headers and index, never tensor payloads and never initializes JAX/TPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.partitioning import (  # noqa: E402
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    MemoryPolicy,
    build_layout_manifest,
    build_pipeline_plan,
    build_placement_ledger,
    read_source_inventory,
    write_layout_manifest,
    write_source_inventory,
)
from glm_tpu.greenfield.types import (  # noqa: E402
    ModelGeometry,
    PhysicalTopology,
    PlanName,
)


DEFAULT_SOURCE = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
DEFAULT_TOPOLOGY = Path(
    "/home/gianl/glm-run/greenfield_topology_20260805T125842425591441Z/"
    "topology.rank0.json"
)
DEFAULT_SOURCE_URI = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=("PP8_LP4", "PP16_LP2"), required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-uri", default=DEFAULT_SOURCE_URI)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--topology-capture", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--mtp-stage-id", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") != "rewrite/topology-first-decode":
        raise RuntimeError("checkpoint plan must run on the isolated rewrite branch")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("checkpoint plan requires a clean worktree")
    if not args.source_uri.startswith("gs://driftbench-dsv4-uc/"):
        raise RuntimeError("checkpoint source must use the approved bucket")
    if args.output_dir.exists():
        raise FileExistsError(
            f"append-only checkpoint plan output already exists: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()

    topology_record = json.loads(args.topology_capture.read_text())
    contract = topology_record["contract"]
    topology = PhysicalTopology.from_dict(contract["topology"])
    if topology.topology_hash != contract["topology_hash"]:
        raise RuntimeError("topology capture hash does not reproduce")
    plan_name = PlanName(args.plan)
    if plan_name is PlanName.PP8_LP4:
        local_size = 4
        plan_group_hash = contract["pp8_lp4_hash"]
    else:
        local_size = 2
        plan_group_hash = contract["pp16_lp2_hash"]

    config = json.loads((args.source_root / "config.json").read_text())
    geometry = ModelGeometry.from_hf_config(config)
    inventory = read_source_inventory(
        args.source_root,
        model_id=geometry.model_id,
        source_revision=args.source_revision,
    )
    ledger = build_placement_ledger(
        inventory,
        geometry,
        local_parallel_size=local_size,
    )
    memory_policy = MemoryPolicy()
    partition = build_pipeline_plan(
        name=plan_name,
        geometry=geometry,
        topology=topology,
        ledger=ledger,
        memory_policy=memory_policy,
        mtp_stage_id=args.mtp_stage_id,
    )
    layout = build_layout_manifest(
        inventory=inventory,
        ledger=ledger,
        partition=partition,
        source_uri=args.source_uri,
        code_hash=code_hash,
        topology_hash=topology.topology_hash,
        plan_group_hash=plan_group_hash,
    )

    inventory_path = args.output_dir / "source_inventory.json"
    plan_path = args.output_dir / "plan.json"
    layout_path = args.output_dir / "layout_manifest.json"
    write_source_inventory(inventory, inventory_path)
    plan_value = partition.to_dict()
    plan_value["plan_manifest_sha256"] = partition.plan_manifest_sha256
    _write_json(plan_path, plan_value)
    write_layout_manifest(layout, layout_path)
    destination_files = layout["destination_files"]
    summary: dict[str, Any] = {
        "artifact_kind": "greenfield_checkpoint_plan",
        "base_destination_file_count": sum(
            record["load_set"] == BASE_LOAD_SET for record in destination_files
        ),
        "capacity_feasible": partition.capacity_feasible,
        "code_hash": code_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "index_share_crossings": list(partition.index_share_crossings),
        "inventory_sha256": inventory.inventory_sha256,
        "layout_manifest_sha256": layout["manifest_sha256"],
        "maximum_accounted_bytes": max(
            stage.maximum_accounted_bytes for stage in partition.stages
        ),
        "minimum_free_bytes": min(
            stage.minimum_free_bytes for stage in partition.stages
        ),
        "mtp_destination_file_count": sum(
            record["load_set"] == MTP_LOAD_SET for record in destination_files
        ),
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "plan_group_hash": plan_group_hash,
        "plan_id": plan_name.value,
        "plan_manifest_sha256": partition.plan_manifest_sha256,
        "promotion_memory_proven": partition.promotion_memory_proven,
        "source_file_count": len(inventory.files),
        "source_leaf_count": len(inventory.tensors),
        "source_payload_bytes": inventory.payload_bytes,
        "source_revision": inventory.source_revision,
        "topology_hash": topology.topology_hash,
    }
    summary_path = args.output_dir / "summary.json"
    _write_json(summary_path, summary)
    evidence_paths = (inventory_path, plan_path, layout_path, summary_path)
    evidence_path = args.output_dir / "evidence.sha256"
    evidence_path.write_text(
        "".join(
            f"{_sha256_file(path)}  {path.name}\n" for path in evidence_paths
        )
    )
    (args.output_dir / "SUCCESS").write_text(
        f"{layout['manifest_sha256']}  layout_manifest.json\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
