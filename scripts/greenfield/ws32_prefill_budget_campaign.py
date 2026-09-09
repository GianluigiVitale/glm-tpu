#!/usr/bin/env python3
"""Bounded weight-free budget transport inside the protected FP8 wrapper.

The existing wrapper owns leases, idle censuses, DB, archive and final SUCCESS.
This adapter never provisions or manages infrastructure and loads no weights.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import shutil
from typing import Any

import numpy as np

from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield import prefill_budget_evidence as evidence
from scripts.greenfield import prefill_budget_overhead as overhead
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield.probe_ws32_prefill_budget import (
    KERNEL,
    REPO,
    run_root,
    kernel_for_tag,
)
from scripts.greenfield.probe_ws32_prefill_moe import (
    TOPOLOGY,
    TOPOLOGY_SHA,
    FLEET_SHA,
    MESH_SHA,
)
from scripts.greenfield.ws32_prefill_moe_campaign import (
    ssh,
    deploy_existing_workers,
    coordinator_address,
)
from scripts.greenfield.prefill_window_evidence import same_json

MAX_RANK_BYTES = 64 << 20
MAX_LEDGER_BYTES = 64 << 10
# 240s overhead +120s sampling +540s runtime/fixture/compile/tie allowance.
WORKER_SECONDS = 900
SSH_SECONDS = 1080  # bounded child kill/publication allowance, not model wall
NOTE = (
    "Weight-free production32-row DSA long-prefix budgets and fresh-cache/input/"
    "local-consumer overhead. Synthetic exact own-score/ties; no model correctness, "
    "full-prefill throughput, delivered model TTFT or prefill trace claim."
)


def evidence_files() -> tuple[str, ...]:
    originals = tuple(
        c.name + suffix + ".npz"
        for c in probe.cases()
        for suffix in ("_sample0", f"_sample{probe.WARMUP + probe.ITERATIONS - 1}")
    )
    ties = tuple(
        c.name + "_tie.npz"
        for c in probe.cases()
        if c.last_valid_length == c.prompt_length
    )
    return (
        "runner.json",
        "worker.log",
        "compile_journal.jsonl",
        *(
            f"{n}.{s}"
            for n in worker.PROGRAMS
            for s in ("stablehlo.mlir", "optimized_hlo.txt")
        ),
        *originals,
        *ties,
    )


def topology_bindings() -> tuple[Any, tuple[dict, ...]]:
    """Authenticate the existing complete capture, not arbitrary worker claims."""
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import (
        validate_ws32_topology_fleet,
    )
    from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh

    captures = tuple(
        json.loads((TOPOLOGY / f"topology.rank{i}.json").read_text()) for i in range(8)
    )
    topology, ordered, _ = validate_ws32_topology_fleet(
        captures,
        expected_topology_sha256=TOPOLOGY_SHA,
        expected_fleet_sha256=FLEET_SHA,
        slice_name="db-v4-64-od",
    )
    physical = build_ws32_physical_mesh(topology)
    if physical.mesh_hash != MESH_SHA:
        raise ValueError("budget captured physical mesh differs")
    return physical, ordered


def validate_workers(records: list[dict], pin: str, tag: str) -> list[dict[int, int]]:
    run_root(tag)
    kernel = kernel_for_tag(tag)
    sorted_local_merge = kernel != KERNEL
    protocol, profile, _, _ = worker.contract(sorted_local_merge)
    if not re.fullmatch(r"[0-9a-f]{40}", pin) or len(records) != 8:
        raise ValueError("budget needs exact pin and eight workers")
    physical, captures = topology_bindings()
    slots_by_device = {d: i for i, d in enumerate(physical.flattened_device_ids)}
    owners = []
    for rank, record in enumerate(records):
        capture = captures[rank]
        fixed = dict(
            status="SUCCESS",
            budget_complete=True,
            tag=tag,
            kernel=kernel,
            protocol=protocol,
            profile=profile,
            compile_only=False,
            code_hash=pin,
            launch_rank=rank,
            hostname=capture["hostname"],
            jax_process_index=capture["jax_process_index"],
            mesh_sha256=MESH_SHA,
            topology_sha256=TOPOLOGY_SHA,
            topology_fleet_sha256=FLEET_SHA,
            physical_device_ids=physical.device_ids,
            weights_loaded=False,
            model_executable_calls=0,
            model_ttft_measured=False,
            performance_claim=False,
            baseline_only=not sorted_local_merge,
            diagnostic_only=True,
            admission_only=False,
            latency=None,
            warmup=probe.WARMUP,
            iterations=probe.ITERATIONS,
        )
        if sorted_local_merge:
            from scripts.greenfield.prefill_sorted_merge_admission import registration

            fixed["candidate_registration"] = registration()
        elif "candidate_registration" in record:
            raise ValueError("candidate identity supplied to baseline collector")
        same_json({k: record.get(k) for k in fixed}, fixed, "budget fleet identity")
        local = {d: slots_by_device[d] for d in capture["local_device_ids"]}
        same_json(
            record["local_device_slots"],
            [dict(device_id=d, device_slot=s) for d, s in local.items()],
            "budget captured local owners",
        )
        if (
            type(record.get("pid")) is not int
            or record["pid"] <= 0
            or type(record.get("start_ticks")) is not int
            or record["start_ticks"] <= 0
            or not re.fullmatch(r"[0-9a-f-]{36}", record.get("boot_id", ""))
            or "error" in record
            or "phase_error" in record
        ):
            raise ValueError("budget process identity or failure differs")
        phases = record.get("acquisition_phases", {})
        if set(phases) != {
            "budget_runtime",
            "budget_setup",
            "budget_finalize",
            "budget_terminal",
        }:
            raise ValueError("budget outer phase inventory differs")
        for phase in phases.values():
            if (
                phase["status"] != "COMPLETE"
                or phase["error"] is not None
                or not np.isfinite(phase["seconds"])
                or phase["seconds"] < 0
            ):
                raise ValueError("budget outer phase failed")
        same_json(
            record["versions"],
            {"jax": "0.10.1", "libtpu": "0.0.41"},
            "budget runtime versions",
        )
        owners.append(local)
    if len({d for slots in owners for d in slots}) != 32:
        raise ValueError("budget fleet does not cover32 physical owners")
    for name in worker.PROGRAMS:
        for field in ("stablehlo_sha256", "optimized_hlo_sha256", "compiled_memory"):
            values = [
                json.dumps(r["programs"][name][field], sort_keys=True) for r in records
            ]
            if len(set(values)) != 1:
                raise ValueError("budget fleet graph/allocations differ")
    return owners


def overhead_wall(records: list[dict]) -> dict:
    """Max host per aligned sample; first/repeat initialization stay separate."""

    def maxima(values):
        rows = np.asarray(values, np.float64)
        samples = rows.max(axis=0).tolist()
        return dict(
            samples_seconds=samples,
            p50_seconds=float(np.percentile(samples, 50)),
            p99_seconds=float(np.percentile(samples, 99)),
        )

    return dict(
        scope="FRESH_STATE_AND_PLACEMENT_BUDGET_NOT_MODEL_TTFT",
        initialization={
            str(c): dict(
                first_seconds=max(
                    r["budget_overhead"]["initialization"][str(c)][0][
                        "cache_initialization_seconds"
                    ]
                    for r in records
                ),
                repeat_seconds=max(
                    r["budget_overhead"]["initialization"][str(c)][1][
                        "cache_initialization_seconds"
                    ]
                    for r in records
                ),
            )
            for c, _ in probe.CAPACITIES
        },
        input_placement={
            str(n): dict(
                scope=records[0]["budget_overhead"]["input_placement"][str(n)]["scope"],
                **maxima(
                    [
                        r["budget_overhead"]["input_placement"][str(n)][
                            "samples_seconds"
                        ]
                        for r in records
                    ]
                ),
            )
            for n in overhead.INPUT_ROWS
        },
        delivery=next(
            r["budget_overhead"]["delivery"]
            for r in records
            if r["jax_process_index"] == 0
        ),
        delivery_scope=probe.DELIVERY_SCOPE,
    )


def aggregate(root: Path, records: list[dict], pin: str, tag: str) -> dict:
    kernel = kernel_for_tag(tag)
    sorted_local_merge = kernel != KERNEL
    protocol, profile, _, _ = worker.contract(sorted_local_merge)
    slots = validate_workers(records, pin, tag)
    for rank, (record, local) in enumerate(zip(records, slots, strict=True)):
        evidence.validate_files(
            root / "fleet" / f"rank{rank}",
            record,
            slots=local,
            require_overhead=not sorted_local_merge,
            sorted_local_merge=sorted_local_merge,
        )
    wall = evidence.fleet_wall(records)
    extra = dict(overhead_wall=overhead_wall(records)) if not sorted_local_merge else {}
    note = NOTE
    if sorted_local_merge:
        from scripts.greenfield import prefill_sorted_merge_admission as candidate

        extra = dict(
            candidate_selection=candidate.compare(wall),
            candidate_registration=candidate.registration(),
        )
        note = candidate.NOTE
    return dict(
        status="SUCCESS",
        code_hash=pin,
        tag=tag,
        kernel=kernel,
        protocol=protocol,
        profile=profile,
        admission_only=False,
        baseline_only=not sorted_local_merge,
        diagnostic_only=True,
        performance_claim=False,
        latency=None,
        profiler_free_timing=True,
        warmup=probe.WARMUP,
        iterations=probe.ITERATIONS,
        selected_route_case=None,
        device_kind="TPU v4",
        workers=records,
        hlo=dict(
            sha256=records[0]["programs"][worker.PROGRAMS[0]]["optimized_hlo_sha256"],
            contract=dict(passed=True, scope="SYNTHETIC_DSA_EXPERT8_BUDGET"),
        ),
        comparison=dict(passed=None, diagnostic_evidence_complete=True),
        dsa_wall=wall,
        claim_scope=note,
        checksum=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        **extra,
    )


def validate_record(record: dict, pin: str, root: Path) -> None:
    """Actual original replay before DB; no record-only promotion."""
    expected = aggregate(root, record["workers"], pin, root.name)
    same_json(record, expected, "budget aggregate replay")


def publish_rank(tag: str, rank: int) -> None:
    from google.cloud import storage
    from scripts.greenfield.collect_ws32_worker_evidence import (
        digest_file,
        publish_exact,
    )

    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("invalid budget publication rank")
    root = run_root(tag) / f"rank{rank}"
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    receipts, omitted, size = [], [], 0
    # Failures may preserve an intermediate sample not in the successful set.
    allowed = (
        *evidence_files(),
        *(
            c.name + f"_sample{i}.npz"
            for c in probe.cases()
            for i in range(1, probe.WARMUP + probe.ITERATIONS - 1)
        ),
    )
    for name in allowed:
        path = root / name
        if not path.exists():
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or size + path.stat().st_size > MAX_RANK_BYTES - MAX_LEDGER_BYTES * 2
        ):
            omitted.append(name)
            continue
        facts = digest_file(path)
        receipts.append(
            publish_exact(
                bucket,
                f"results/{tag}/workers/rank{rank}/{name}",
                path,
                facts,
                compressed=False,
            )
        )
        size += facts["size"]
    if omitted:
        path = root / "publication_omissions.json"
        _atomic_json(path, dict(omitted=omitted, originals_retained_locally=True))
        receipts.append(
            publish_exact(
                bucket,
                f"results/{tag}/workers/rank{rank}/{path.name}",
                path,
                digest_file(path),
                compressed=False,
            )
        )
    path = root / "worker_receipts.json"
    _atomic_json(path, receipts)
    if path.stat().st_size > MAX_LEDGER_BYTES:
        raise ValueError("budget receipt ledger exceeds fixed ceiling")
    publish_exact(
        bucket,
        f"results/{tag}/workers/rank{rank}/{path.name}",
        path,
        digest_file(path),
        compressed=False,
    )


def collect(tag: str, pin: str) -> dict:
    from google.cloud import storage

    root = run_root(tag)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    ledgers = []
    # Resolve and budget every generation BEFORE downloading any payload.
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        blob = bucket.get_blob(prefix + "worker_receipts.json")
        if blob is None or not 0 < int(blob.size) <= MAX_LEDGER_BYTES:
            raise ValueError("missing/oversized budget receipt ledger")
        raw = blob.download_as_bytes(if_generation_match=int(blob.generation))
        receipts = json.loads(raw)
        if (
            len(receipts) != len(evidence_files())
            or {r["name"] for r in receipts} != {prefix + n for n in evidence_files()}
            or any(type(r["size"]) is not int or r["size"] <= 0 for r in receipts)
            or sum(r["size"] for r in receipts) > MAX_RANK_BYTES
        ):
            raise ValueError("budget original file inventory/size differs")
        for receipt in receipts:
            if (
                not str(receipt["generation"]).isdecimal()
                or int(receipt["generation"]) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", receipt["original_sha256"])
            ):
                raise ValueError("budget receipt generation/digest invalid")
            obj = bucket.blob(receipt["name"], generation=int(receipt["generation"]))
            obj.reload(if_generation_match=int(receipt["generation"]))
            if int(obj.size) != receipt["size"] or obj.crc32c != receipt["crc32c"]:
                raise ValueError("budget original generation size/CRC differs")
        ledgers.append((blob, raw, receipts))
    if shutil.disk_usage(root).free < 8 * MAX_RANK_BYTES + (1 << 30):
        raise ValueError("insufficient bounded budget collection space")
    records = []
    for rank, (blob, raw, receipts) in enumerate(ledgers):
        destination = root / "fleet" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "worker_receipts.json").write_bytes(raw)
        _atomic_json(
            destination / "ledger_source.json",
            dict(
                name=blob.name,
                generation=str(blob.generation),
                size=int(blob.size),
                crc32c=blob.crc32c,
                sha256=sha256(raw).hexdigest(),
            ),
        )
        for receipt in receipts:
            generation = int(receipt["generation"])
            data = bucket.blob(
                receipt["name"], generation=generation
            ).download_as_bytes(if_generation_match=generation)
            if (
                len(data) != receipt["size"]
                or sha256(data).hexdigest() != receipt["original_sha256"]
            ):
                raise ValueError("budget original SHA/length differs")
            (destination / Path(receipt["name"]).name).write_bytes(data)
        records.append(json.loads((destination / "runner.json").read_text()))
    return aggregate(root, records, pin, tag)


def launch_command(tag: str, pin: str, address: str) -> str:
    run_root(tag)
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError("invalid budget launch pin")
    return (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; tag="
        + shlex.quote(tag)
        + "; wt="
        + shlex.quote(str(REPO))
        + "; "
        "[[ $idx =~ ^[0-7]$ ]]; out=/home/gianl/glm-run/$tag/rank$idx; "
        '[[ ! -e "$out" ]]; mkdir -p "$out"; cd "$wt"; '
        'upload(){ JAX_PLATFORMS=cpu PYTHONPATH="$wt" timeout --kill-after=10s 120s '
        "/home/gianl/vllm-env/bin/python -m scripts.greenfield.ws32_prefill_budget_campaign "
        'publish-rank --tag "$tag" --rank "$idx"; }; trap upload EXIT; '
        'GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" '
        f"timeout --kill-after=30s {WORKER_SECONDS}s /home/gianl/vllm-env/bin/python -u "
        "scripts/greenfield/probe_ws32_prefill_budget.py --expected-code-hash "
        + pin
        + " --coordinator-address "
        + shlex.quote(address)
        + ' --process-id "$idx" --output-dir "$out" >"$out/worker.log" 2>&1'
    )


def campaign(tag: str, pin: str) -> None:
    root = run_root(tag)
    if kernel_for_tag(tag) != KERNEL:
        from scripts.greenfield.prefill_sorted_merge_admission import baseline_wall

        baseline_wall()
    deploy_existing_workers(root, pin)
    ssh(
        launch_command(tag, pin, coordinator_address(root)),
        output=root / "fleet_launch.log",
        timeout=SSH_SECONDS,
    )
    record = collect(tag, pin)
    _atomic_json(root / "runner.json", record)
    (root / "hlo").mkdir(exist_ok=True)
    shutil.copyfile(
        root / "fleet/rank0" / f"{worker.PROGRAMS[0]}.optimized_hlo.txt",
        root / "hlo/candidate.optimized_hlo.txt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("campaign", "publish-rank"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pin")
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.mode == "campaign":
        campaign(args.tag, args.pin)
    else:
        publish_rank(args.tag, args.rank)
