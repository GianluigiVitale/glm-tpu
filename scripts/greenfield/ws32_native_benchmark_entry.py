"""Explicit native benchmark worker entry under the existing WS32 owner name.

Not an outer supervisor. The controller MUST hold both existing leases, verify
published clean source/storage/fleet vacancy, transport the pinned small request
capsule, observe exact PID/start/boot ownership, collect originals and cleanly
seal. Running this worker directly is not a protected benchmark launch.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

from scripts.greenfield import ws32_native_benchmark_protocol as protocol
from scripts.greenfield import ws32_native_benchmark_transport as transport
from scripts.greenfield.ws32_history_preflight import _plain_path

RUN_ROOT = Path("/home/gianl/glm-run")
TOKENIZER = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-benchmark-request", type=Path, required=True)
    parser.add_argument("--native-benchmark-protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recipe = json.loads((protocol.REPO / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    fields = {
        "checkpoint_root": "CHECKPOINT_ROOT", "checkpoint_transport": "CHECKPOINT_TRANSPORT",
        "checkpoint_manifest_sha256": "CHECKPOINT_MANIFEST_SHA", "checkpoint_success_sha256": "CHECKPOINT_SUCCESS_SHA",
        "strategy_nd_dense_overlay_root": "STRATEGY_ND_DENSE_OVERLAY_ROOT",
        "strategy_nd_dense_overlay_manifest_sha256": "STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA",
        "strategy_nd_dense_overlay_manifest_file_sha256": "STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA",
        "strategy_nd_dense_overlay_success_file_sha256": "STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA",
    }
    for field, key in fields.items():
        value = recipe["GLM_GREENFIELD_WS32_" + key]
        setattr(args, field, Path(value) if field.endswith("_root") else value)
    args.source_inventory = Path("/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json")
    args.topology_capture_root = Path("/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records")
    args.topology_sha256 = "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
    args.topology_fleet_sha256 = "4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301"
    args.mesh_sha256 = "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"
    args.slice_name = "db-v4-64-od"
    args.context_capacity, args.num_processes = protocol.CAPACITY, 8
    return args


def preflight(args: Any) -> tuple[dict, dict]:
    """No backend initialization, file overwrite or TPU access before refusal."""
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    from scripts.greenfield.ws32_native_benchmark_programs import require_source
    _require_clean_code(args.expected_code_hash)
    require_source(protocol.REPO)
    if os.environ.get("GLM_GREENFIELD_NATIVE_BENCHMARK") != "1":
        raise ValueError("native benchmark is default-off")
    root = args.output.parent
    transport._identity(root.name, args.expected_code_hash, args.process_id)
    _plain_path(root)
    if (root.parent != RUN_ROOT or not root.is_dir()
            or args.output.name != f"runner.rank{args.process_id}.json" or args.output.exists()
            or (root / f"native.rank{args.process_id}").exists()
            or (root / f"sessions.rank{args.process_id}").exists()):
        raise ValueError("native worker requires a fresh exact run/rank namespace")
    for path, cap in ((args.native_benchmark_request, protocol.PAYLOAD_CAP),
                      (args.native_benchmark_protocol, 64 << 10)):
        _plain_path(path)
        if not path.is_file() or not 0 < path.stat().st_size <= cap:
            raise ValueError("native protocol/request original missing or oversized")
    # Protocol is in the published source pin, not an arbitrary run-time waiver.
    if args.native_benchmark_protocol.resolve() != protocol.REPO / "configs/greenfield-native-benchmark-protocol.json":
        raise ValueError("native protocol must be the committed campaign registration")
    plan_bytes = args.native_benchmark_protocol.read_bytes()
    if sha256(plan_bytes).hexdigest() != args.protocol_sha256:
        raise ValueError("native protocol bytes differ from launch pin")
    plan = json.loads(plan_bytes)
    payload = json.loads(args.native_benchmark_request.read_bytes())
    protocol.validate(payload, plan)
    for name, expected in plan["tokenizer_files"].items():
        if Path(name).name != name or name not in ("tokenizer.json", "tokenizer_config.json"):
            raise ValueError("native tokenizer pin names differ")
        if sha256((TOKENIZER / name).read_bytes()).hexdigest() != expected:
            raise ValueError("native tokenizer bytes differ")
    if set(plan["tokenizer_files"]) != {"tokenizer.json", "tokenizer_config.json"}:
        raise ValueError("native complete tokenizer identity missing")
    return payload, plan


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload, plan = preflight(args)
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield.ws32_native_benchmark_worker import load_runtime
    from scripts.greenfield.ws32_native_benchmark_requests import RequestStore, execute_requests
    from scripts.greenfield.ws32_native_benchmark_observability import NativeObservability
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from transformers import AutoTokenizer

    started = time.perf_counter()
    jax, mesh, physical_mesh, topology, fleet_sha = original._initialize_runtime(args)
    consensus = original._batched_fleet_all
    def phase(fn):
        result = error = None
        try:
            result = fn()
        except Exception as exc:
            error = exc
        agreed = consensus(error is None)
        if error is not None:
            raise error
        if not agreed:
            raise RuntimeError("native entry peer setup/publication failed")
        return result
    store = phase(lambda: RequestStore(args.output.parent / f"sessions.rank{args.process_id}", args.process_id))
    owner = dict(pid=os.getpid(), hostname=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        start_ticks=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        argv_sha256=sha256(Path("/proc/self/cmdline").read_bytes()).hexdigest())
    record = dict(profile=transport.PROFILE, artifact_kind="ws32_native_benchmark_worker_v1",
        code_hash=args.expected_code_hash, protocol_sha256=args.protocol_sha256,
        launch_process_id=args.process_id, jax_process_index=int(jax.process_index()), owner=owner,
        complete=False, benchmark_quality_proven=False, protected_result_sealed=False)
    phase(lambda: _atomic_json(args.output, record))
    try:
        tokenizer = phase(lambda: AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True,
                                                               trust_remote_code=False))
        loaded = load_runtime(args=args, repo=protocol.REPO,
            root=args.output.parent / f"native.rank{args.process_id}", mesh=mesh,
            physical_mesh=physical_mesh, topology=topology, fleet_sha=fleet_sha,
            consensus=consensus, preserve_memory=store.preserve_memory)
        record["cold_load_compile_seconds"] = loaded.record["cold_load_compile_seconds"]
        def progress(row):
            record["completed_requests"] = row["index"] + 1
            record["last_request_id"] = row["request_id"]
            _atomic_json(args.output, record)
            print(f"NATIVE_BENCHMARK_COMPLETED index={row['index']} tokens={row['generated_tokens']}", flush=True)
        rows = execute_requests(loaded=loaded, payload=payload, plan=plan, store=store,
            tokenizer=tokenizer, deadline=started + plan["tranche_wall_seconds"], after_request=progress,
            observations=NativeObservability(loaded, store))
        record.update(complete=True, completed_requests=len(rows), worker_wall_seconds=time.perf_counter()-started)
        phase(lambda: _atomic_json(args.output, record))
        return 0
    except Exception as exc:
        record.update(complete=False, failure=f"{type(exc).__name__}: {exc}")
        _atomic_json(args.output, record)
        raise
