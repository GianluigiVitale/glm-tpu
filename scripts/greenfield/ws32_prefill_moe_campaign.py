#!/usr/bin/env python3
"""Real-MoE worker transport/collection inside the existing bounded FP8 wrapper.

This module does not acquire leases, run census, finalize DB or publish terminal
success: run_fp8_matmul_microbench.sh retains those responsibilities. No direct
launch outside that wrapper is authorized by adding this adapter.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.probe_ws32_prefill_moe import (
    CASES,
    FLEET_SHA,
    MESH_SHA,
    ORACLE_SHA,
    PACK_SHA,
    PROTOCOL,
    ROWS,
    TOPOLOGY_SHA,
    check_hlo,
)

KERNEL = "ws32_prefill_moe_admission"
FILES = (
    "runner.json",
    "worker.log",
    "candidate.stablehlo.mlir",
    "candidate.optimized_hlo.txt",
    "reference.optimized_hlo.txt",
    "normal.npz",
    "concentrated.npz",
)


def run_root(tag: str) -> Path:
    if not re.fullmatch(
        r"greenfield_fp8_ws32_prefill_moe_admission_[a-zA-Z0-9_]+", tag
    ):
        raise ValueError("invalid real-MoE admission tag")
    return Path("/home/gianl/glm-run") / tag


def ssh(command: str, *, output: Path, worker: str = "all", timeout: int = 120) -> None:
    with output.open("w") as stream:
        result = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                "db-v4-64-od",
                "--zone",
                "us-central2-b",
                f"--worker={worker}",
                f"--command={command}",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError(f"worker command failed; see {output}")


def validate_workers(records: list[dict[str, Any]], pin: str) -> None:
    if len(records) != 8 or {r["launch_rank"] for r in records} != set(range(8)):
        raise ValueError("need eight unique launch ranks")
    if len({r["hostname"] for r in records}) != 8 or {
        r["jax_process_index"] for r in records
    } != set(range(8)):
        raise ValueError("fleet hostname/process mapping differs")
    if len({r["hlo"]["sha256"] for r in records}) != 1:
        raise ValueError("fleet candidate HLO hashes differ")
    for case in CASES:
        inputs = {tuple(r["cases"][case]["input_sha256"]) for r in records}
        if len(inputs) != 1 or any(
            len(h) != 3 or any(not re.fullmatch("[0-9a-f]{64}", v) for v in h)
            for h in inputs
        ):
            raise ValueError("fleet replicated case inputs differ")
    slots = []
    for r in records:
        if not (
            r["status"] == "SUCCESS"
            and r["protocol"] == PROTOCOL
            and r["code_hash"] == pin
            and r["admission_only"] is True
            and r["performance_claim"] is False
            and r["iterations"] == 0
            and r["latency"] is None
            and r["rows"] == ROWS
            and r["packed_manifest_sha256"] == PACK_SHA
            and r["oracle_manifest_sha256"] == ORACLE_SHA
            and r["mesh_sha256"] == MESH_SHA
            and r["topology_sha256"] == TOPOLOGY_SHA
            and r["topology_fleet_sha256"] == FLEET_SHA
            and r["hlo"]["contract"]["passed"] is True
            and set(r["cases"]) == set(CASES)
            and len(r["local_device_slots"]) == 4
            and len(r["device_memory_stats_including_reference"]) == 4
            and r["pid"] > 0
            and r["start_ticks"] > 0
            and bool(r["boot_id"])
        ):
            raise ValueError("real-MoE worker contract/provenance differs")
        own_slots = {s["device_slot"] for s in r["local_device_slots"]}
        device_ids = np.asarray(r["physical_device_ids"]).reshape(-1).tolist()
        if len(device_ids) != 32 or len(set(device_ids)) != 32:
            raise ValueError("physical device identity differs")
        for s in r["local_device_slots"]:
            if device_ids[s["device_slot"]] != s["device_id"] or not re.fullmatch(
                "[0-9a-f]{64}", s["file_sha256"]
            ):
                raise ValueError("loaded shard/device binding differs")
        memory = r["compiled_memory_estimate"]
        sizes = [
            memory[n]
            for n in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
            )
        ]
        if any(type(n) is not int or n < 0 for n in sizes) or sum(sizes) > 1024**3:
            raise ValueError("compiled one-layer memory budget differs")
        stats = r["device_memory_stats_including_reference"]
        if {s["device_id"] for s in stats} != {device_ids[s] for s in own_slots}:
            raise ValueError("memory evidence does not cover local owners")
        if any(
            not 0 < s["stats"]["peak_bytes_in_use"] < s["stats"]["bytes_limit"]
            for s in stats
        ):
            raise ValueError("measured HBM headroom missing")
        slots.extend(own_slots)
        for c in r["cases"].values():
            if not (
                c["passed"] is True
                and c["fleet_passed"] is True
                and c["reference_vs_legacy"]["passed"] is True
                and len(c["shards"]) == 4
                and {s["device_slot"] for s in c["shards"]} == own_slots
                and all(
                    s["passed"] is True
                    and s["bit_mismatches"] == 0
                    and s["output_sha256"] == s["reference_sha256"]
                    for s in c["shards"]
                )
            ):
                raise ValueError("real-MoE case evidence differs")
    if len(slots) != 32 or set(slots) != set(range(32)):
        raise ValueError("real-MoE fleet does not cover all32 final owners")


def validate_record(record: dict[str, Any], pin: str) -> None:
    if not (
        record["status"] == "SUCCESS"
        and record["code_hash"] == pin
        and record["kernel"] == KERNEL
        and record["protocol"] == PROTOCOL
        and record["admission_only"] is True
        and record["performance_claim"] is False
        and record["baseline_only"] is False
        and record["diagnostic_only"] is False
        and record["latency"] is None
        and record["warmup"] == record["iterations"] == 0
        and record["profiler_free_timing"] is False
    ):
        raise ValueError("real-MoE aggregate classification differs")
    validate_workers(record["workers"], pin)


def validate_files(destination: Path, record: dict[str, Any]) -> None:
    """Recheck original tensor bytes and actual graph, not just worker verdicts."""
    hlo = (destination / "candidate.optimized_hlo.txt").read_text()
    if (
        sha256(hlo.encode()).hexdigest() != record["hlo"]["sha256"]
        or not check_hlo(hlo)["passed"]
    ):
        raise ValueError("candidate HLO bytes/contract differ")
    if (
        sha256((destination / "reference.optimized_hlo.txt").read_bytes()).hexdigest()
        != record["reference_hlo_sha256"]
    ):
        raise ValueError("reference HLO bytes differ")
    for case in CASES:
        shards = record["cases"][case]["shards"]
        with np.load(destination / f"{case}.npz", allow_pickle=False) as arrays:
            expected_keys = {
                f"{kind}_{s['device_id']}"
                for s in shards
                for kind in ("actual", "reference")
            }
            if set(arrays.files) != expected_keys:
                raise ValueError("original output shard inventory differs")
            for s in shards:
                a, b = (
                    arrays[f"{kind}_{s['device_id']}"]
                    for kind in ("actual", "reference")
                )
                if not (
                    a.shape == b.shape == (ROWS, 1536)
                    and a.dtype == b.dtype == np.uint16
                    and np.array_equal(a, b)
                    and np.all((a & 0x7F80) != 0x7F80)
                    and sha256(a.tobytes()).hexdigest() == s["output_sha256"]
                    and sha256(b.tobytes()).hexdigest() == s["reference_sha256"]
                ):
                    raise ValueError("original BF16 output evidence differs")


def publish_rank(tag: str, rank: int) -> None:
    from google.cloud import storage
    from scripts.greenfield.collect_ws32_worker_evidence import (
        digest_file,
        publish_exact,
    )

    if not 0 <= rank < 8:
        raise ValueError("invalid rank")
    root = run_root(tag) / f"rank{rank}"
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    receipts = []
    for name in FILES:
        path = root / name
        if path.is_file():
            receipts.append(
                publish_exact(
                    bucket,
                    f"results/{tag}/workers/rank{rank}/{name}",
                    path,
                    digest_file(path),
                    compressed=False,
                )
            )
    ledger = root / "worker_receipts.json"
    ledger.write_text(json.dumps(receipts, sort_keys=True) + "\n")
    publish_exact(
        bucket,
        f"results/{tag}/workers/rank{rank}/worker_receipts.json",
        ledger,
        digest_file(ledger),
        compressed=False,
    )


def collect(tag: str, pin: str) -> dict[str, Any]:
    from google.cloud import storage

    root = run_root(tag)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")
    records = []
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        ledger_blob = bucket.get_blob(prefix + "worker_receipts.json")
        if ledger_blob is None:
            raise ValueError(f"missing rank{rank} receipt ledger")
        ledger_bytes = ledger_blob.download_as_bytes(
            if_generation_match=ledger_blob.generation
        )
        receipts = json.loads(ledger_bytes)
        if {r["name"] for r in receipts} != {prefix + n for n in FILES} or len(
            receipts
        ) != len(FILES):
            raise ValueError(f"rank{rank} evidence set is incomplete")
        destination = root / "fleet" / f"rank{rank}"
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "worker_receipts.json").write_bytes(ledger_bytes)
        for receipt in receipts:
            generation = int(receipt["generation"])
            blob = bucket.blob(receipt["name"], generation=generation)
            blob.reload(if_generation_match=generation)
            if int(blob.size) != receipt["size"] or blob.crc32c != receipt["crc32c"]:
                raise ValueError("worker object size/CRC receipt mismatch")
            data = blob.download_as_bytes(if_generation_match=generation)
            if sha256(data).hexdigest() != receipt["original_sha256"]:
                raise ValueError("worker object SHA mismatch")
            (destination / Path(receipt["name"]).name).write_bytes(data)
        record = json.loads((destination / "runner.json").read_text())
        if record["launch_rank"] != rank:
            raise ValueError("record rank differs from published path")
        validate_files(destination, record)
        records.append(record)
    validate_workers(records, pin)
    return dict(
        status="SUCCESS",
        code_hash=pin,
        kernel=KERNEL,
        admission_only=True,
        baseline_only=False,
        diagnostic_only=False,
        performance_claim=False,
        latency=None,
        profiler_free_timing=False,
        warmup=0,
        iterations=0,
        selected_route_case=None,
        device_kind="TPU v4",
        protocol=PROTOCOL,
        workers=records,
        hlo={"sha256": records[0]["hlo"]["sha256"], "contract": {"passed": True}},
        comparison={"passed": True},
        checksum=sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    )


def campaign(tag: str, pin: str) -> None:
    root = run_root(tag)
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite") or not re.fullmatch(
        "[0-9a-f]{40}", pin
    ):
        raise ValueError("invalid worktree/pin")
    # Deployment only to existing checked-out worker repositories; no cloning or infrastructure action.
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; wt="
        + shlex.quote(str(REPO))
        + "; pin="
        + pin
        + "; "
        '[[ $idx =~ ^[0-7]$ && -e $wt/.git ]]; [[ -z $(git -C "$wt" status --porcelain) ]]; '
        'if [[ $idx != 0 ]]; then git -C "$wt" fetch -q origin rewrite/topology-first-decode; '
        'git -C "$wt" checkout -q --detach "$pin"; fi; '
        '[[ $(git -C "$wt" rev-parse HEAD) == "$pin" && -z $(git -C "$wt" status --porcelain) ]]; '
        'echo "PREFILL_SYNC_OK $(hostname)"'
    )
    ssh(command, output=root / "fleet_sync.log")
    markers = [
        line.split()[1]
        for line in (root / "fleet_sync.log").read_text().splitlines()
        if line.startswith("PREFILL_SYNC_OK ")
    ]
    if len(markers) != 8 or len(set(markers)) != 8:
        raise ValueError("fleet sync not eight unique hosts")
    ssh("hostname -I | awk '{print $1}'", output=root / "coordinator.log", worker="0")
    import ipaddress

    address = (
        str(
            ipaddress.ip_address(
                (root / "coordinator.log").read_text().strip().splitlines()[-1]
            )
        )
        + ":8476"
    )
    # The shell remains until its bounded child exits and original files publish.
    command = (
        "set -euo pipefail; idx=${HOSTNAME##*-w-}; tag="
        + shlex.quote(tag)
        + "; wt="
        + shlex.quote(str(REPO))
        + "; "
        'out=/home/gianl/glm-run/$tag/rank$idx; mkdir -p "$out"; cd "$wt"; '
        'upload(){ JAX_PLATFORMS=cpu PYTHONPATH="$wt" /home/gianl/vllm-env/bin/python '
        '-m scripts.greenfield.ws32_prefill_moe_campaign publish-rank --tag "$tag" --rank "$idx"; }; trap upload EXIT; '
        'GLM_GREENFIELD_RUN_TAG="$tag" JAX_PLATFORMS=tpu PYTHONPATH="$wt" '
        "timeout --kill-after=30s 600s /home/gianl/vllm-env/bin/python -u scripts/greenfield/probe_ws32_prefill_moe.py "
        "--expected-code-hash "
        + pin
        + " --coordinator-address "
        + shlex.quote(address)
        + ' --process-id "$idx" --output-dir "$out" '
        '>"$out/worker.log" 2>&1'
    )
    ssh(command, output=root / "fleet_launch.log", timeout=780)
    record = collect(tag, pin)
    (root / "runner.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (root / "hlo").mkdir(exist_ok=True)
    (root / "hlo/candidate.optimized_hlo.txt").write_bytes(
        (root / "fleet/rank0/candidate.optimized_hlo.txt").read_bytes()
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
