#!/usr/bin/env python3
"""Load and byte-round-trip one complete PP8/PP16 final-layout stage."""

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
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FullCheckpointLoadExpectation,
    load_final_layout_stage,
    resolve_stage_devices,
    verify_full_packed_checkpoint,
)
from glm_tpu.greenfield.partitioning import BASE_LOAD_SET  # noqa: E402


EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
EXPECTED_BRANCH = "rewrite/topology-first-decode"


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


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"stage-load output is append-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _device_record(device: object) -> dict[str, Any]:
    def value(name: str) -> Any:
        raw = getattr(device, name)
        return raw() if callable(raw) else raw

    return {
        "core_on_chip": int(value("core_on_chip")),
        "coordinates": [int(item) for item in value("coords")],
        "device_kind": str(value("device_kind")),
        "id": int(value("id")),
        "local_hardware_id": int(value("local_hardware_id")),
        "platform": str(value("platform")),
        "process_index": int(value("process_index")),
    }


def _memory_stats(device: object) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--topology-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-manifest-output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--packed-manifest-sha256", required=True)
    parser.add_argument("--layout-manifest-sha256", required=True)
    parser.add_argument("--source-inventory-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--topology-sha256", required=True)
    parser.add_argument("--plan-group-sha256", required=True)
    parser.add_argument("--plan-manifest-sha256", required=True)
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--layout-code-hash", required=True)
    parser.add_argument("--pack-code-hash", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--plan-id", default="PP8_LP4")
    parser.add_argument("--stage-id", type=int)
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--verify-device-roundtrip", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") not in (EXPECTED_BRANCH, ""):
        raise RuntimeError("stage loader requires the isolated branch or detached pin")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale loader code: expected={args.expected_code_hash} found={code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("stage loader requires a clean worktree")
    if not args.verify_device_roundtrip:
        raise RuntimeError("protected full-stage load requires device round-trip")
    if args.chunk_bytes <= 0:
        raise ValueError("chunk bytes must be positive")
    if args.output.exists() or args.state_manifest_output.exists():
        raise FileExistsError("stage-load outputs are append-only")
    expectation = FullCheckpointLoadExpectation(
        packed_manifest_sha256=args.packed_manifest_sha256,
        layout_manifest_sha256=args.layout_manifest_sha256,
        source_inventory_sha256=args.source_inventory_sha256,
        source_revision=args.source_revision,
        topology_hash=args.topology_sha256,
        plan_group_hash=args.plan_group_sha256,
        plan_manifest_sha256=args.plan_manifest_sha256,
        execution_plan_sha256=args.execution_plan_sha256,
        layout_code_hash=args.layout_code_hash,
        pack_code_hash=args.pack_code_hash,
        destination=args.destination,
        plan_id=args.plan_id,
    )
    metadata_started = time.perf_counter()
    checkpoint = verify_full_packed_checkpoint(
        args.checkpoint_root, expectation
    )
    metadata_seconds = time.perf_counter() - metadata_started

    import jax

    local_devices = tuple(jax.local_devices())
    # libtpu requires the complete four-chip host subcube even when PP16
    # places the loaded arrays only on its adjacent two-chip stage.
    if (
        jax.default_backend() != "tpu"
        or jax.process_count() != 1
        or jax.device_count() != 4
        or len(local_devices) != 4
        or any(device.device_kind != "TPU v4" for device in local_devices)
    ):
        raise RuntimeError(
            "protected stage load requires one standalone four-chip TPU v4 host"
        )
    visible_raw = os.environ.get("TPU_VISIBLE_DEVICES", "")
    try:
        visible = tuple(int(item) for item in visible_raw.split(",") if item)
    except ValueError as error:
        raise ValueError(f"invalid TPU_VISIBLE_DEVICES={visible_raw!r}") from error
    if visible != (0, 1, 2, 3):
        raise RuntimeError(
            "protected stage load requires exact visible local devices 0,1,2,3"
        )
    resolution = resolve_stage_devices(
        local_devices,
        args.topology_capture,
        expectation,
        stage_id=args.stage_id,
        visible_device_indices=visible,
    )
    selected = sorted(
        (
            plan
            for plan in checkpoint.plans
            if plan.load_set == BASE_LOAD_SET
            and plan.stage_id == resolution.stage_id
        ),
        key=lambda item: item.device_slot,
    )
    planned_payload_by_slot = [plan.payload_bytes for plan in selected]
    if len(planned_payload_by_slot) != expectation.stage_size:
        raise RuntimeError(
            "resolved stage does not have its exact base owner file count"
        )
    memory_policy = checkpoint.layout["plan_manifest"]["memory_policy"]
    hbm_limit = int(memory_policy["hbm_limit_bytes"])
    load_started = time.perf_counter()
    loaded = load_final_layout_stage(
        checkpoint,
        expectation,
        resolution,
        verify_device_roundtrip=True,
        chunk_bytes=args.chunk_bytes,
    )
    load_seconds = time.perf_counter() - load_started
    load_record = dict(loaded.load_record)
    if (
        not load_record["device_roundtrip_verified"]
        or load_record["device_roundtrip_bytes"]
        != load_record["loaded_payload_bytes"]
        or load_record["fp8_device_dequantizations"] != 0
        or load_record["fp8_host_dequantizations"] != 0
        or load_record["host_global_concatenations"] != 0
        or load_record["runtime_checkpoint_reshards"] != 0
    ):
        loaded.close()
        raise RuntimeError("stage loader violated the direct raw-FP8 contract")
    state_manifest = dict(loaded.state_manifest)
    _write_json_once(args.state_manifest_output, state_manifest)
    state_manifest_file_sha256 = _sha256_file(args.state_manifest_output)
    device_after_load = [_memory_stats(device) for device in resolution.devices]
    for slot, (memory, planned) in enumerate(
        zip(device_after_load, planned_payload_by_slot, strict=True)
    ):
        if memory is None:
            loaded.close()
            raise RuntimeError(f"device slot {slot} returned no HBM statistics")
        bytes_in_use = memory.get("bytes_in_use")
        peak = memory.get("peak_bytes_in_use")
        limit = memory.get("bytes_limit")
        if (
            not isinstance(bytes_in_use, int)
            or bytes_in_use < planned
            or not isinstance(peak, int)
            or peak < bytes_in_use
            or peak >= hbm_limit
            or limit != hbm_limit
        ):
            loaded.close()
            raise RuntimeError(
                f"device slot {slot} HBM contract failed: "
                f"planned={planned} memory={memory} policy_limit={hbm_limit}"
            )
    loaded.close()
    device_after_release = [_memory_stats(device) for device in resolution.devices]
    record = {
        "artifact_kind": "greenfield_full_checkpoint_stage_load",
        "captured_device_ids_in_slot_order": list(
            resolution.captured_device_ids
        ),
        "captured_process_index": resolution.captured_process_index,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "destination": expectation.destination,
            "layout_manifest_sha256": expectation.layout_manifest_sha256,
            "packed_manifest_sha256": expectation.packed_manifest_sha256,
            "source_inventory_sha256": expectation.source_inventory_sha256,
            "source_revision": expectation.source_revision,
        },
        "code_hash": code_hash,
        "device_memory_after_load": device_after_load,
        "device_memory_after_release": device_after_release,
        "hostname": socket.gethostname(),
        "hbm_limit_bytes": hbm_limit,
        "jax": {
            "default_backend": jax.default_backend(),
            "devices_in_runtime_order": [
                _device_record(device) for device in local_devices
            ],
            "devices_in_stage_slot_order": [
                _device_record(device) for device in resolution.devices
            ],
            "version": jax.__version__,
        },
        "load": load_record,
        "metadata_verification_seconds": metadata_seconds,
        "payload_load_and_roundtrip_seconds": load_seconds,
        "plan_group_sha256": expectation.plan_group_hash,
        "plan_id": expectation.plan_id,
        "planned_payload_bytes_by_slot": planned_payload_by_slot,
        "stage_id": resolution.stage_id,
        "state_manifest": {
            "file_sha256": state_manifest_file_sha256,
            "manifest_sha256": state_manifest["manifest_sha256"],
            "path": str(args.state_manifest_output),
            "payload_bytes": state_manifest["payload_bytes"],
            "tensor_count": state_manifest["tensor_count"],
        },
        "status": "SUCCESS",
        "topology_sha256": expectation.topology_hash,
    }
    _write_json_once(args.output, record)
    print(
        "GREENFIELD_FULL_STAGE_LOAD_OK "
        f"host={record['hostname']} stage={resolution.stage_id} "
        f"payload={state_manifest['payload_bytes']} "
        f"tensors={state_manifest['tensor_count']} "
        f"seconds={load_seconds:.3f} "
        f"state={state_manifest['manifest_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
