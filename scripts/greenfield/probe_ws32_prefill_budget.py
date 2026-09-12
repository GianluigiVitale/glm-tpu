#!/usr/bin/env python3
"""Weight-free §24 budget worker; only the protected FP8 wrapper may launch it.

Reuses runtime topology, voted calls, journal and production budget campaign.
No model loading, checkpoint packing, infrastructure management or model claim.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield.microbench_fp8_matmul import _atomic_json, _git_head
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_worker import BudgetedCalls
from scripts.greenfield.probe_ws32_prefill_moe import (
    TOPOLOGY,
    TOPOLOGY_SHA,
    FLEET_SHA,
    MESH_SHA,
)

KERNEL = "ws32_prefill_budget_baseline"


def kernel_for_tag(tag: str) -> str:
    from scripts.greenfield.prefill_sorted_merge_admission import KERNEL as candidate
    from scripts.greenfield.ws32_rolled_prefill_worker import KERNEL as compiler
    from scripts.greenfield.ws32_dense_canonical_compile import KERNEL as dense_compiler
    from scripts.greenfield.ws32_canonical_prefill_compile import (
        KERNEL as full_compiler,
    )
    from scripts.greenfield.ws32_history_compile import KERNEL as history_compiler
    from scripts.greenfield.ws32_delivery_compile import KERNEL as delivery_compiler
    from scripts.greenfield.ws32_owned_state_compile import KERNEL as owned_compiler
    from scripts.greenfield.ws32_pending_rows_compile import KERNEL as pending_compiler
    from scripts.greenfield.ws32_flat_rows_compile import KERNEL as flat_compiler
    from scripts.greenfield.ws32_capture_barrier_compile import KERNEL as capture_compiler

    for kernel in (KERNEL, candidate, compiler, dense_compiler, full_compiler, history_compiler, delivery_compiler, owned_compiler, pending_compiler, flat_compiler, capture_compiler):
        if re.fullmatch(r"greenfield_fp8_" + kernel + r"_[a-zA-Z0-9_]+", tag):
            return kernel
    raise ValueError("unregistered DSA budget/candidate tag")


def run_root(tag: str) -> Path:
    kernel_for_tag(tag)
    return Path("/home/gianl/glm-run") / tag


def validate_request(args: argparse.Namespace, tag: str) -> None:
    """Refuse stale/unscoped requests before distributed initialization."""
    if (
        REPO != Path("/home/gianl/glm-tpu-topology-rewrite")
        or type(args.process_id) is not int
        or not 0 <= args.process_id < 8
        or not re.fullmatch(r"[0-9a-f]{40}", args.expected_code_hash)
        or _git_head() != args.expected_code_hash
        or args.output_dir != run_root(tag) / f"rank{args.process_id}"
    ):
        raise ValueError("budget worker code/rank/output identity differs")
    if (args.output_dir / "runner.json").exists():
        raise FileExistsError(args.output_dir / "runner.json")
    for name in (
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "TPU_VISIBLE_DEVICES",
    ):
        if os.environ.get(name):
            raise ValueError("budget worker inherited single-process TPU bounds")
    from scripts.greenfield import ws32_rolled_prefill_worker as rolled

    from scripts.greenfield.ws32_dense_canonical_compile import KERNEL as dense_compiler
    from scripts.greenfield.ws32_canonical_prefill_compile import (
        KERNEL as full_compiler,
    )

    from scripts.greenfield.ws32_history_compile import KERNEL as history_compiler

    from scripts.greenfield.ws32_delivery_compile import KERNEL as delivery_compiler
    from scripts.greenfield.ws32_owned_state_compile import KERNEL as owned_compiler
    from scripts.greenfield.ws32_pending_rows_compile import KERNEL as pending_compiler
    from scripts.greenfield.ws32_flat_rows_compile import KERNEL as flat_compiler
    from scripts.greenfield.ws32_capture_barrier_compile import KERNEL as capture_compiler

    if kernel_for_tag(tag) in (rolled.KERNEL, dense_compiler, full_compiler, history_compiler, delivery_compiler, owned_compiler, pending_compiler, flat_compiler, capture_compiler):
        # Local metadata/source refusal precedes TPU runtime startup. No model
        # payload is read here or by the subsequent compiler continuation.
        rolled.compile_mode(
            kernel_for_tag(tag) == dense_compiler,
            full_canonical=kernel_for_tag(tag) == full_compiler,
            history=kernel_for_tag(tag) == history_compiler,
            delivery=kernel_for_tag(tag) == delivery_compiler,
            owned_state=kernel_for_tag(tag) == owned_compiler,
            pending_rows=kernel_for_tag(tag) == pending_compiler,
            flat_rows=kernel_for_tag(tag) == flat_compiler,
            capture_barrier=kernel_for_tag(tag) == capture_compiler,
        ).preparation.read_metadata(REPO)
    elif kernel_for_tag(tag) != KERNEL:
        from scripts.greenfield.prefill_sorted_merge_admission import registration

        registration()


def execute_budget(
    root: Path,
    record: dict,
    *,
    jax: Any,
    mesh: Any,
    physical_mesh: Any,
    consensus: Any,
    sorted_local_merge: bool = False,
) -> None:
    """Voted setup and terminal persistence around the reviewed continuation."""

    def setup():
        all_slots = {
            device: slot
            for slot, device in enumerate(physical_mesh.flattened_device_ids)
        }
        slots = {int(d.id): all_slots[int(d.id)] for d in jax.local_devices()}
        if len(slots) != 4 or len(set(slots.values())) != 4:
            raise ValueError("budget local physical owners differ")
        record["local_device_slots"] = [
            dict(device_id=d, device_slot=s) for d, s in slots.items()
        ]
        protocol, profile, journal_type, _ = worker.contract(sorted_local_merge)
        if record.get("protocol") != protocol or record.get("profile") != profile:
            raise ValueError("budget trusted launch mode and record differ")
        journal = journal_type(
            root / "compile_journal.jsonl",
            {
                k: record[k]
                for k in (
                    "protocol",
                    "profile",
                    "compile_only",
                    "code_hash",
                    "launch_rank",
                )
            },
        )
        return BudgetedCalls(
            root=root,
            record=record,
            consensus=consensus,
            journal=journal,
            local_slots=slots,
        )

    calls = fleet_step(
        "budget_setup", setup, record=record, root=root, consensus=consensus
    )
    try:
        if sorted_local_merge:
            prepared = worker.prepare(calls, mesh, sorted_local_merge=True)
            worker.run_samples(calls, prepared)
        else:
            worker.run_budget_campaign(calls, mesh)
    finally:
        # Every surviving peer follows the same close/hash/publication vote,
        # including after a numerical refusal. No later journal phase is added.
        def close():
            calls.journal.close()
            record["compile_journal_sha256"] = sha256(
                (root / "compile_journal.jsonl").read_bytes()
            ).hexdigest()

        fleet_step(
            "budget_finalize", close, record=record, root=root, consensus=consensus
        )

    # SUCCESS is never set until the journal finalization vote has succeeded.
    def terminal():
        record.update(status="SUCCESS", budget_complete=True)
        _atomic_json(root / "runner.json", record)

    fleet_step(
        "budget_terminal", terminal, record=record, root=root, consensus=consensus
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    tag = os.environ.get("GLM_GREENFIELD_RUN_TAG", "")
    validate_request(args, tag)
    kernel = kernel_for_tag(tag)
    from scripts.greenfield import ws32_rolled_prefill_worker as rolled

    from scripts.greenfield.ws32_dense_canonical_compile import KERNEL as dense_compiler
    from scripts.greenfield.ws32_canonical_prefill_compile import (
        KERNEL as full_compiler,
    )

    canonical_dense = kernel == dense_compiler
    full_canonical = kernel == full_compiler
    from scripts.greenfield.ws32_history_compile import KERNEL as history_compiler

    history = kernel == history_compiler
    from scripts.greenfield.ws32_delivery_compile import KERNEL as delivery_compiler
    from scripts.greenfield.ws32_owned_state_compile import KERNEL as owned_compiler
    from scripts.greenfield.ws32_pending_rows_compile import KERNEL as pending_compiler
    from scripts.greenfield.ws32_flat_rows_compile import KERNEL as flat_compiler
    from scripts.greenfield.ws32_capture_barrier_compile import KERNEL as capture_compiler

    delivery = kernel == delivery_compiler
    owned_state = kernel == owned_compiler
    pending_rows = kernel == pending_compiler
    flat_rows = kernel == flat_compiler
    capture_barrier = kernel == capture_compiler
    compile_only = kernel in (rolled.KERNEL, dense_compiler, full_compiler, history_compiler, delivery_compiler, owned_compiler, pending_compiler, flat_compiler, capture_compiler)
    sorted_local_merge = not compile_only and kernel != KERNEL
    if compile_only:
        mode = rolled.compile_mode(canonical_dense, full_canonical=full_canonical, history=history, delivery=delivery, owned_state=owned_state, pending_rows=pending_rows, flat_rows=flat_rows, capture_barrier=capture_barrier)
        protocol, profile = mode.protocol, mode.profile
    else:
        protocol, profile, _, _ = worker.contract(sorted_local_merge)
    print(f"PREFILL_BUDGET rank={args.process_id} tag={tag} starting", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.num_processes, args.slice_name = 8, "db-v4-64-od"
    args.topology_capture_root = TOPOLOGY
    args.topology_sha256, args.topology_fleet_sha256, args.mesh_sha256 = (
        TOPOLOGY_SHA,
        FLEET_SHA,
        MESH_SHA,
    )
    record = dict(
        status="RUNNING",
        tag=tag,
        kernel=kernel,
        protocol=protocol,
        profile=profile,
        compile_only=compile_only,
        code_hash=args.expected_code_hash,
        launch_rank=args.process_id,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        start_ticks=int(
            Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        ),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        weights_loaded=False,
        model_executable_calls=0,
        model_ttft_measured=False,
        performance_claim=False,
        baseline_only=kernel == KERNEL,
        diagnostic_only=True,
        admission_only=False,
        latency=None,
        warmup=0 if compile_only else probe.WARMUP,
        iterations=0 if compile_only else probe.ITERATIONS,
        programs={},
    )
    if compile_only:
        record.update(prefill_mode=rolled.PREFILL_MODE, numerical_claim=False)
    if sorted_local_merge:
        from scripts.greenfield.prefill_sorted_merge_admission import registration

        record["candidate_registration"] = registration()
    _atomic_json(args.output_dir / "runner.json", record)
    try:
        from scripts.greenfield.run_short_decoder_ws32 import _initialize_runtime

        started = time.monotonic()
        jax, mesh, physical, topology, fleet_sha = _initialize_runtime(args)
        from jax.experimental import multihost_utils

        def consensus(ok):
            return bool(
                np.asarray(
                    multihost_utils.process_allgather(np.asarray(ok, np.int32))
                ).all()
            )

        def runtime_record():
            record.update(
                jax_process_index=jax.process_index(),
                mesh_sha256=physical.mesh_hash,
                physical_device_ids=physical.device_ids,
                topology_sha256=topology.topology_hash,
                topology_fleet_sha256=fleet_sha,
                versions={"jax": version("jax"), "libtpu": version("libtpu")},
                runtime_seconds=time.monotonic() - started,
            )
            if compile_only:
                slots = {
                    device: slot
                    for slot, device in enumerate(physical.flattened_device_ids)
                }
                local = [
                    dict(device_id=int(d.id), device_slot=slots[int(d.id)])
                    for d in jax.local_devices()
                ]
                if len(local) != 4 or len({r["device_slot"] for r in local}) != 4:
                    raise ValueError("rolled compiler physical owners differ")
                record["local_device_slots"] = local

        fleet_step(
            "budget_runtime",
            runtime_record,
            record=record,
            root=args.output_dir,
            consensus=consensus,
        )
        if compile_only:
            rolled.execute_pair(
                args.output_dir,
                record,
                mesh=mesh,
                repo=REPO,
                consensus=consensus,
                canonical_dense=canonical_dense,
                full_canonical=full_canonical,
                history=history,
                delivery=delivery, owned_state=owned_state, pending_rows=pending_rows, flat_rows=flat_rows, capture_barrier=capture_barrier,
            )
        else:
            execute_budget(
                args.output_dir,
                record,
                jax=jax,
                mesh=mesh,
                physical_mesh=physical,
                consensus=consensus,
                sorted_local_merge=sorted_local_merge,
            )
        print(f"PREFILL_BUDGET rank={args.process_id} tag={tag} complete", flush=True)
        return 0
    except Exception as exc:
        record.update(
            status="FAILED", budget_complete=False, error=f"{type(exc).__name__}: {exc}"
        )
        _atomic_json(args.output_dir / "runner.json", record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
