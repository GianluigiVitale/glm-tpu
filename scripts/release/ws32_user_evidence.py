"""Join original user response, cold admission, physical traces and ownership.

The leased controller calls this only after generation-verified collection and
authenticated cleanup. It does not execute a model, score answers or publish a
seal. DB/archive linkage remains a separate transaction after this replay.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

from glm_tpu import user_request
from scripts.release import ws32_user_worker as worker
from scripts.release import ws32_user_transport as transport
from scripts.release.ws32_user_result import read, replay_user_request, same
from scripts.greenfield import ws32_native_benchmark_evidence as cold_evidence
from scripts.greenfield import ws32_native_benchmark_archive as original_archive
from scripts.greenfield.fp8_baseline_guard import validate_fleet


def replay_cold(
    destination: Path, pin: str
) -> tuple[dict, list[dict], tuple[int, ...]]:
    """Same retained loader recipe and original cold validators, without JAX init."""
    from scripts.greenfield import run_short_decoder_ws32 as original
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig

    original._require_clean_code(pin)
    args = worker.site_args(SimpleNamespace())
    captures = tuple(
        json.loads(read(args.topology_capture_root / f"topology.rank{r}.json", 4 << 20))
        for r in range(8)
    )
    topology, captures, fleet_sha = original.validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=args.topology_fleet_sha256,
        slice_name=args.slice_name,
    )
    physical = original.build_ws32_physical_mesh(topology)
    if physical.mesh_hash != args.mesh_sha256:
        raise ValueError("user cold replay physical mesh differs")
    config = Ws32DecoderConfig(
        original._geometry(),
        args.context_capacity,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
    )
    checked = cold_evidence.replay_cold_fleet(
        root=destination,
        repo=worker.REPO,
        pin=pin,
        captures=captures,
        physical_mesh=physical,
        topology_hash=args.topology_sha256,
        fleet_hash=fleet_sha,
        full_index_layers=config.full_index_slots,
    )
    parents = [
        json.loads(read(destination / f"native.rank{r}/runner.json", 16 << 20))
        for r in range(8)
    ]
    return checked, parents, config.full_index_slots


def load_tokenizer():
    """Local pinned files only; never infer source identity from package labels."""
    from transformers import AutoTokenizer

    for name, expected in user_request.TOKENIZER_FILES.items():
        if sha256(read(worker.TOKENIZER / name, 32 << 20)).hexdigest() != expected:
            raise ValueError("user replay tokenizer identity differs")
    if (
        sha256(
            read(worker.REPO / "reference/hf-repo/chat_template.jinja", 64 << 10)
        ).hexdigest()
        != user_request.TEMPLATE_SHA
    ):
        raise ValueError("user replay template identity differs")
    return AutoTokenizer.from_pretrained(
        worker.TOKENIZER, local_files_only=True, trust_remote_code=False
    )


def replay_traces(destination: Path, report: dict, owners: list[dict]) -> dict | None:
    from scripts.analysis.parse_xplane import (
        aggregate_fleet,
        validate_fleet_expectations,
    )

    traces = report["trace_originals"]
    paths = sorted(destination.rglob("*.xplane.pb"))
    if report["generated_tokens"] == 1:
        if traces or paths:
            raise ValueError(
                "prefill-only user response has unexpected trace originals"
            )
        return None
    if len(traces) != 8 or [r["rank"] for r in traces] != list(range(8)):
        raise ValueError("user response needs its own eight rank traces")
    expected_paths = sorted(destination / r["path"] for r in traces)
    if paths != expected_paths:
        raise ValueError("user trace archive contains unexpected or missing XPlanes")
    limit = transport.cold.file_limits()["observer.optimized_hlo.txt"]
    text = read(destination / "native.rank0/observer.optimized_hlo.txt", limit).decode()
    module = re.match(r"HloModule ([A-Za-z0-9_.-]+),", text)
    if module is None:
        raise ValueError("user observer module identity is missing")
    trace = aggregate_fleet(
        destination,
        step_module_re=r"(?<![A-Za-z0-9_])"
        + re.escape(module[1])
        + r"(?![A-Za-z0-9_])",
        allow_single_step=True,
    )
    validate_fleet_expectations(
        trace, n_files=8, n_cores=64, n_hosts=8, cores_per_host=8, steps_per_core=1
    )
    expected_hosts = {
        str(destination / r["path"]): owners[r["rank"]]["hostname"] for r in traces
    }
    same(trace["file_hosts"], expected_hosts, "XPlane host versus authenticated worker")
    return trace


def replay_collected(root: Path, tag: str, pin: str) -> dict:
    worker.identity(tag, pin, 0)
    if root != worker.RUN_ROOT / tag:
        raise ValueError("user replay requires the original run root")
    transport.private_directory(root)
    destination = root / "collected"
    transport.private_directory(destination)
    launch = json.loads(read(root / "launch.json", 64 << 10))
    same(
        {k: launch[k] for k in ("tag", "code_hash", "benchmark")},
        dict(tag=tag, code_hash=pin, benchmark=False),
        "launch identity",
    )
    raw = read(root / "request.json", user_request.PAYLOAD_CAP)
    same(
        sha256(raw).hexdigest(), launch["request_file_sha256"], "launched request file"
    )
    same(len(raw), launch["request_bytes"], "launched request bytes")
    request = json.loads(raw)
    user_request.validate(request)
    for phase in ("pre", "post"):
        validate_fleet(read(root / f"census_{phase}.txt", 64 << 20).decode())
    # Freeze the exact observation history for idempotent archival; no relaunch.
    final_watch = root / "final_watch.jsonl"
    if not final_watch.exists():
        with transport.private_writes():
            transport.outputs._write_once(
                final_watch, read(root / "user_watch.jsonl", 64 << 20)
            )
    original_archive.verify_ownership(root, tag, pin)
    owners = []
    for rank in range(8):
        outer = json.loads(read(destination / f"runner.rank{rank}.json", 2 << 20))
        owner = outer["owner"]
        ended = json.loads(read(destination / f"ended.rank{rank}.json", 16 << 10))
        transport.check_ended(
            ended,
            tag=tag,
            pin=pin,
            rank=rank,
            request_file_sha256=launch["request_file_sha256"],
            host=owner["hostname"],
            boot_id=owner["boot_id"],
        )
        same(
            {
                k: ended[k]
                for k in ("worker_exit_code", "worker_started", "worker_error_type")
            },
            dict(worker_exit_code=0, worker_started=True, worker_error_type=None),
            "terminal worker exit",
        )
        owners.append(owner)
    cold, parents, layers = replay_cold(destination, pin)
    report = replay_user_request(
        root=destination,
        request=request,
        request_file_sha256=launch["request_file_sha256"],
        pin=pin,
        parents=parents,
        tokenizer=load_tokenizer(),
        full_index_layers=layers,
    )
    trace = replay_traces(destination, report, owners)
    report.update(
        cold=cold,
        cold_admission_verified=True,
        worker_ownership_verified=True,
        trace=trace,
        trace_physical_coverage_verified=trace is not None,
        cleanup="authenticated8host_idle",
        execution_code_hash=pin,
        replay_code_hash=pin,
    )
    # This is original replay, not a seal. Archive publication must still finish.
    return report
