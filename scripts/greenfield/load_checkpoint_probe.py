#!/usr/bin/env python3
"""Verify and direct-load a bounded complete-layout checkpoint probe."""

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
import time
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FullCheckpointLoadExpectation,
    VerifiedPackedCheckpoint,
    build_destination_probe_plans,
    load_final_layout_stage,
    resolve_stage_devices,
)
from glm_tpu.greenfield.errors import CheckpointValidationError  # noqa: E402
from glm_tpu.greenfield.partitioning import inspect_layout_manifest  # noqa: E402


ARTIFACT_KIND = "greenfield_checkpoint_pack_probe"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def verify_probe_artifact(
    *,
    checkpoint_root: Path,
    layout_path: Path,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[Any, ...],
    dict[str, Mapping[str, Any]],
]:
    layout = inspect_layout_manifest(layout_path)
    manifest = json.loads((checkpoint_root / "manifest.json").read_text())
    if manifest.get("artifact_kind") != ARTIFACT_KIND:
        raise CheckpointValidationError("wrong checkpoint probe artifact kind")
    unhashed = dict(manifest)
    observed = unhashed.pop("manifest_sha256", None)
    if observed != sha256(_canonical(unhashed)).hexdigest():
        raise CheckpointValidationError("checkpoint probe manifest self-hash drifted")
    if observed != expected_manifest_sha256:
        raise CheckpointValidationError("checkpoint probe manifest identity drifted")
    expected = {
        "layout_file_sha256": _sha256_file(layout_path),
        "layout_manifest_sha256": layout["manifest_sha256"],
        "plan_group_hash": layout["plan_group_hash"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_revision": layout["source"]["revision"],
        "topology_hash": layout["topology_hash"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointValidationError(f"checkpoint probe drifted at {key}")
    source_names = tuple(item["name"] for item in manifest.get("sources", ()))
    plans = build_destination_probe_plans(layout, source_names=source_names)
    records = {item["filename"]: item for item in manifest.get("files", ())}
    if len(records) != len(plans) or manifest.get("file_count") != len(plans):
        raise CheckpointValidationError("checkpoint probe file count drifted")
    evidence = {}
    total_payload = 0
    for plan in plans:
        relative = f"payload/{plan.filename}"
        record = records.get(relative)
        if record is None:
            raise CheckpointValidationError(
                f"checkpoint probe lacks owner {plan.filename}"
            )
        expected_record = {
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "header_bytes": len(plan.header),
            "payload_bytes": plan.payload_bytes,
            "stage_id": plan.stage_id,
            "tensor_names": [item.name for item in plan.tensors],
        }
        for key, value in expected_record.items():
            if record.get(key) != value:
                raise CheckpointValidationError(
                    f"checkpoint probe owner {plan.filename} drifted at {key}"
                )
        path = checkpoint_root / relative
        if path.stat().st_size != plan.file_bytes:
            raise CheckpointValidationError(
                f"checkpoint probe owner size drifted: {plan.filename}"
            )
        digest = _sha256_file(path)
        if record.get("sha256") != digest:
            raise CheckpointValidationError(
                f"checkpoint probe owner hash drifted: {plan.filename}"
            )
        evidence[plan.filename] = {"sha256": digest}
        total_payload += plan.payload_bytes
    if manifest.get("packed_payload_bytes") != total_payload:
        raise CheckpointValidationError("checkpoint probe payload total drifted")
    return dict(layout), manifest, plans, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--topology-capture", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    branch = subprocess.check_output(
        ["git", "-C", str(REPO), "branch", "--show-current"], text=True
    ).strip()
    if branch not in ("rewrite/topology-first-decode", ""):
        raise RuntimeError("checkpoint probe loader requires the isolated branch")
    code_hash = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("checkpoint probe loader code hash is stale")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("checkpoint probe loader requires a clean worktree")
    if args.output.exists() or args.state_manifest_output.exists():
        raise FileExistsError("checkpoint probe loader outputs are append-only")

    layout, manifest, plans, evidence = verify_probe_artifact(
        checkpoint_root=args.checkpoint_root,
        layout_path=args.layout_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    if manifest["plan_id"] != "PP16_LP2" or manifest["stage_id"] != 0:
        raise RuntimeError("protected probe loader requires PP16 stage 0")
    plan_manifest = layout["plan_manifest"]
    expectation = FullCheckpointLoadExpectation(
        packed_manifest_sha256=manifest["manifest_sha256"],
        layout_manifest_sha256=layout["manifest_sha256"],
        source_inventory_sha256=layout["source"]["inventory_sha256"],
        source_revision=layout["source"]["revision"],
        topology_hash=layout["topology_hash"],
        plan_group_hash=layout["plan_group_hash"],
        plan_manifest_sha256=layout["plan_manifest_sha256"],
        execution_plan_sha256=plan_manifest["execution_plan_sha256"],
        layout_code_hash=layout["code_hash"],
        pack_code_hash=manifest["code_hash"],
        destination=args.destination,
        plan_id="PP16_LP2",
    )
    checkpoint = VerifiedPackedCheckpoint(
        root=args.checkpoint_root / "payload",
        layout=layout,
        packed_manifest=manifest,
        control={},
        plans=plans,
        evidence_by_filename=evidence,
    )

    import jax

    local_devices = tuple(jax.local_devices())
    if (
        jax.default_backend() != "tpu"
        or jax.process_count() != 1
        or jax.device_count() != 4
        or len(local_devices) != 4
        or any(device.device_kind != "TPU v4" for device in local_devices)
    ):
        raise RuntimeError(
            "protected PP16 probe load requires one standalone four-chip TPU v4 host"
        )
    visible = tuple(
        int(item) for item in os.environ.get("TPU_VISIBLE_DEVICES", "").split(",")
        if item
    )
    if visible != (0, 1, 2, 3):
        raise RuntimeError("protected PP16 probe requires visible chips 0,1,2,3")
    resolution = resolve_stage_devices(
        local_devices,
        args.topology_capture,
        expectation,
        stage_id=0,
        visible_device_indices=visible,
    )
    if resolution.captured_device_ids != (0, 1):
        raise RuntimeError("PP16 probe did not resolve captured devices 0,1")
    started = time.perf_counter()
    loaded = load_final_layout_stage(
        checkpoint,
        expectation,
        resolution,
        verify_device_roundtrip=True,
        chunk_bytes=8 * 1024 * 1024,
    )
    load_seconds = time.perf_counter() - started
    load_record = dict(loaded.load_record)
    if (
        load_record["loaded_payload_bytes"] != manifest["packed_payload_bytes"]
        or load_record["device_roundtrip_bytes"] != manifest["packed_payload_bytes"]
        or load_record["loaded_tensor_count"] != 16
        or not load_record["device_roundtrip_verified"]
        or load_record["fp8_device_dequantizations"] != 0
        or load_record["fp8_host_dequantizations"] != 0
        or load_record["host_global_concatenations"] != 0
        or load_record["runtime_checkpoint_reshards"] != 0
    ):
        loaded.close()
        raise RuntimeError("PP16 probe violated the direct-load contract")
    state = dict(loaded.state_manifest)
    _write_json_once(args.state_manifest_output, state)
    memory_after_load = [_memory_stats(device) for device in resolution.devices]
    loaded.close()
    memory_after_release = [_memory_stats(device) for device in resolution.devices]
    value = {
        "artifact_kind": "greenfield_checkpoint_probe_device_load",
        "captured_device_ids": list(resolution.captured_device_ids),
        "captured_process_index": resolution.captured_process_index,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "code_hash": code_hash,
        "device_memory_after_load": memory_after_load,
        "device_memory_after_release": memory_after_release,
        "hostname": socket.gethostname(),
        "jax_version": jax.__version__,
        "load": load_record,
        "load_seconds": load_seconds,
        "manifest_sha256": manifest["manifest_sha256"],
        "performance_claim": False,
        "plan_id": "PP16_LP2",
        "stage_id": 0,
        "state_manifest_file_sha256": _sha256_file(args.state_manifest_output),
        "state_manifest_sha256": state["manifest_sha256"],
        "status": "SUCCESS",
        "topology_hash": layout["topology_hash"],
    }
    _write_json_once(args.output, value)
    print(
        "GREENFIELD_PP16_PROBE_LOAD_OK "
        f"payload={load_record['loaded_payload_bytes']} "
        f"tensors={load_record['loaded_tensor_count']} "
        f"seconds={load_seconds:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
