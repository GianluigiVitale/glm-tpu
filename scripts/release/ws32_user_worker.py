"""Default-off user worker; NOT an outer launcher or deployment authorization.

The protected controller must hold both leases, authenticate idle original
hosts, stage one identical private request and own process observation/sealing.
Uses the original loader and compiled graph admissions without benchmark scoring.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from typing import Any

from glm_tpu import user_request

REPO = Path(__file__).resolve().parents[2]
RUN_ROOT = Path("/home/gianl/glm-run")
TOKENIZER = Path("/home/gianl/gcs-models/models/GLM-5.2-FP8")
TAG = re.compile(r"greenfield_ws32_user_request_[0-9]{8}T[0-9]{15}Z")
PROFILE = "glm_ws32_user_worker_v1"


def identity(tag: str, pin: str, rank: int) -> None:
    if (
        not isinstance(tag, str)
        or TAG.fullmatch(tag) is None
        or not isinstance(pin, str)
        or re.fullmatch(r"[0-9a-f]{40}", pin) is None
        or type(rank) is not int
        or not 0 <= rank < 8
    ):
        raise ValueError("user worker tag/pin/rank differs")


def site_args(args: Any) -> Any:
    """The exact retained site recipe; no runtime knobs or checkpoint fallback."""
    recipe = json.loads(
        (REPO / "configs/greenfield-ws32-batched-acquisition.json").read_bytes()
    )["environment"]
    fields = {
        "checkpoint_root": "CHECKPOINT_ROOT",
        "checkpoint_transport": "CHECKPOINT_TRANSPORT",
        "checkpoint_manifest_sha256": "CHECKPOINT_MANIFEST_SHA",
        "checkpoint_success_sha256": "CHECKPOINT_SUCCESS_SHA",
        "strategy_nd_dense_overlay_root": "STRATEGY_ND_DENSE_OVERLAY_ROOT",
        "strategy_nd_dense_overlay_manifest_sha256": "STRATEGY_ND_DENSE_OVERLAY_MANIFEST_SHA",
        "strategy_nd_dense_overlay_manifest_file_sha256": "STRATEGY_ND_DENSE_OVERLAY_MANIFEST_FILE_SHA",
        "strategy_nd_dense_overlay_success_file_sha256": "STRATEGY_ND_DENSE_OVERLAY_SUCCESS_FILE_SHA",
    }
    for field, key in fields.items():
        value = recipe["GLM_GREENFIELD_WS32_" + key]
        setattr(args, field, Path(value) if field.endswith("_root") else value)
    args.source_inventory = Path(
        "/home/gianl/gcs-models/checkpoints/greenfield/glm52/plans/PP8_LP4/greenfield_checkpoint_plan_pp8_20260805T180552087295643Z/source_inventory.json"
    )
    args.topology_capture_root = Path(
        "/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records"
    )
    args.topology_sha256 = (
        "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
    )
    args.topology_fleet_sha256 = (
        "4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301"
    )
    args.mesh_sha256 = (
        "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"
    )
    args.slice_name = "db-v4-64-od"
    args.context_capacity, args.num_processes = user_request.CAPACITY, 8
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-request", type=Path, required=True)
    parser.add_argument("--request-file-sha256", required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return site_args(parser.parse_args(argv))


def validate_inputs(args: Any) -> dict:
    """Pure CPU preflight of the exact private namespace and immutable payload."""
    root = args.output.parent
    identity(root.name, args.expected_code_hash, args.process_id)
    user_request._plain(root)
    if (
        root.parent != RUN_ROOT
        or not root.is_dir()
        or root.stat().st_uid != os.geteuid()
        or stat.S_IMODE(root.stat().st_mode) & 0o077
        or args.output.name != f"runner.rank{args.process_id}.json"
        or any(
            p.exists() or p.is_symlink()
            for p in (
                args.output,
                root / f"native.rank{args.process_id}",
                root / f"sessions.rank{args.process_id}",
            )
        )
    ):
        raise ValueError("user worker requires a fresh owner-only run/rank namespace")
    if (
        args.user_request != root / "request.json"
        or type(args.wall_seconds) is not int
        or not 1 <= args.wall_seconds <= 86400
    ):
        raise ValueError("user request path or operational deadline differs")
    host, port = args.coordinator_address.rsplit(":", 1)
    ipaddress.ip_address(host)
    if port != "8476":
        raise ValueError("user coordinator port differs")
    raw = user_request.read_bounded(args.user_request, user_request.PAYLOAD_CAP)
    facts = args.user_request.stat()
    if (
        facts.st_uid != os.geteuid()
        or stat.S_IMODE(facts.st_mode) & 0o077
        or sha256(raw).hexdigest() != args.request_file_sha256
    ):
        raise ValueError("user request permissions or original file digest differ")
    value = json.loads(raw)
    user_request.validate(value)
    return value


def preflight(args: Any) -> dict:
    # Refuse default-off before imports that might initialize an accelerator.
    if os.environ.get("GLM_GREENFIELD_USER_REQUEST") != "1":
        raise ValueError("user worker is default-off; protected controller required")
    value = validate_inputs(args)
    if not socket.gethostname().endswith("-w-" + str(args.process_id)):
        raise ValueError("user worker hostname/rank differs")
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    from scripts.greenfield.ws32_native_benchmark_programs import require_source

    _require_clean_code(args.expected_code_hash)
    require_source(REPO)
    for name, expected in user_request.TOKENIZER_FILES.items():
        if (
            sha256(user_request.read_bounded(TOKENIZER / name, 32 << 20)).hexdigest()
            != expected
        ):
            raise ValueError("user tokenizer differs from the frozen profile")
    template = user_request.read_bounded(
        REPO / "reference/hf-repo/chat_template.jinja", 64 << 10
    )
    if sha256(template).hexdigest() != user_request.TEMPLATE_SHA:
        raise ValueError("user chat template differs from the frozen profile")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = preflight(args)
    # Enforce private permissions also when an operator directly invokes this
    # worker incorrectly; this is NOT a substitute for controller admission.
    os.umask(0o077)
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield.ws32_native_benchmark_worker import load_runtime
    from scripts.greenfield.ws32_native_benchmark_requests import RequestStore
    from scripts.greenfield.ws32_native_benchmark_observability import (
        NativeObservability,
    )
    from scripts.greenfield.microbench_fp8_matmul import _atomic_json
    from scripts.release.ws32_user_request import execute_user_request
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
        if type(agreed) is not bool or not agreed:
            raise RuntimeError("user worker peer setup/publication failed")
        return result

    owner = dict(
        pid=os.getpid(),
        hostname=socket.gethostname(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        start_ticks=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        argv_sha256=sha256(Path("/proc/self/cmdline").read_bytes()).hexdigest(),
    )
    record = dict(
        profile=PROFILE,
        artifact_kind=PROFILE,
        code_hash=args.expected_code_hash,
        request_file_sha256=args.request_file_sha256,
        request_sha256=request["request_sha256"],
        wall_seconds=args.wall_seconds,
        launch_process_id=args.process_id,
        jax_process_index=int(jax.process_index()),
        owner=owner,
        complete=False,
        benchmark=False,
        protected_result_sealed=False,
    )
    phase(lambda: _atomic_json(args.output, record))
    try:
        store = phase(
            lambda: RequestStore(
                args.output.parent / f"sessions.rank{args.process_id}", args.process_id
            )
        )
        tokenizer = phase(
            lambda: AutoTokenizer.from_pretrained(
                TOKENIZER, local_files_only=True, trust_remote_code=False
            )
        )
        loaded = load_runtime(
            args=args,
            repo=REPO,
            root=args.output.parent / f"native.rank{args.process_id}",
            mesh=mesh,
            physical_mesh=physical_mesh,
            topology=topology,
            fleet_sha=fleet_sha,
            consensus=consensus,
            preserve_memory=store.preserve_memory,
        )
        record["cold_load_compile_seconds"] = loaded.record["cold_load_compile_seconds"]
        row = execute_user_request(
            loaded=loaded,
            request=request,
            store=store,
            tokenizer=tokenizer,
            observations=NativeObservability(loaded, store),
            deadline=started + args.wall_seconds,
        )
        record.update(
            complete=True,
            generated_tokens=row["generated_tokens"],
            finish_reason=row["finish_reason"],
            token_ids_sha256=row["token_ids_sha256"],
            worker_wall_seconds=time.perf_counter() - started,
        )
        phase(lambda: _atomic_json(args.output, record))
        return 0
    except Exception as exc:
        # Keep primary type and existing partials, never include private prompt
        # contents in a controller/log exception message or retry generation.
        record.update(complete=False, failure_type=type(exc).__name__)
        _atomic_json(args.output, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
