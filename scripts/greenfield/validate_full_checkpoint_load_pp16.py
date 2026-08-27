#!/usr/bin/env python3
"""Validate and DB-link the protected 16-stage PP16 full-load records."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FullCheckpointLoadExpectation,
    verify_full_packed_checkpoint,
)
from glm_tpu.greenfield.partitioning import BASE_LOAD_SET  # noqa: E402


PACKED_SHA = "13ad2e926b44bee86e620d877d4c266dbacc4f9a59b352fc4f2c8ef534efedb5"
LAYOUT_SHA = "f97de2d8ec525d984963c522696e09c40b9cb478fb562917f0a42fc2083b15f9"
INVENTORY_SHA = "a388627c08c8ff591903deb1fbf3198f43916e64a2295ed0e253f1e44a042fc4"
REVISION = "gcs-object-set-830fd1bf7d8d6b6242895cfd50f5978e5cc5749da42246c19391855e586e9658"
TOPOLOGY_SHA = "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
GROUP_SHA = "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21"
PLAN_SHA = "3c3ea07b0f97a84702e7b2ad370b014ea22c9e69bc7d35ad88e23a4ad82ded16"
EXECUTION_SHA = "079cefe6a646e0c56120c0903c395587bc95bf8dc6d66de0cf9eb029827794c3"
LAYOUT_CODE = "6dc73048fec3b15892e061bbaf1ec5886f22d29c"
PACK_CODE = "3685ee40960f53874e18ad54ec1e8a0c65e53a62"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    return parser.parse_args()


def _semantic_hash(value: dict[str, object]) -> str:
    return sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def main() -> int:
    args = parse_args()
    expectation = FullCheckpointLoadExpectation(
        packed_manifest_sha256=PACKED_SHA,
        layout_manifest_sha256=LAYOUT_SHA,
        source_inventory_sha256=INVENTORY_SHA,
        source_revision=REVISION,
        topology_hash=TOPOLOGY_SHA,
        plan_group_hash=GROUP_SHA,
        plan_manifest_sha256=PLAN_SHA,
        execution_plan_sha256=EXECUTION_SHA,
        layout_code_hash=LAYOUT_CODE,
        pack_code_hash=PACK_CODE,
        destination=args.destination,
        plan_id="PP16_LP2",
    )
    checkpoint = verify_full_packed_checkpoint(
        args.checkpoint_root, expectation
    )
    plans = tuple(
        plan
        for plan in checkpoint.plans
        if plan.load_set == BASE_LOAD_SET
    )
    expected_by_stage = {
        stage: tuple(
            sorted(
                (plan for plan in plans if plan.stage_id == stage),
                key=lambda plan: plan.device_slot,
            )
        )
        for stage in range(16)
    }
    record_paths = sorted((args.run_dir / "host_records").glob("stage_*.json"))
    state_paths = sorted((args.run_dir / "host_records").glob("state_*.json"))
    records = [json.loads(path.read_text()) for path in record_paths]
    states = {
        path.name.replace("state_", "stage_"): json.loads(path.read_text())
        for path in state_paths
    }
    if len(records) != 16 or len(states) != 16:
        raise SystemExit("PP16 fleet requires sixteen runner/state records")
    if {record["stage_id"] for record in records} != set(range(16)):
        raise SystemExit("PP16 fleet stage coverage failed")
    host_counts = Counter(record["hostname"] for record in records)
    if len(host_counts) != 8 or set(host_counts.values()) != {2}:
        raise SystemExit("each PP16 host must contribute exactly two stages")
    if Counter(
        record["captured_process_index"] for record in records
    ) != Counter({process: 2 for process in range(8)}):
        raise SystemExit("each PP16 process must contribute exactly two stages")
    device_ids = [
        device_id
        for record in records
        for device_id in record["captured_device_ids_in_slot_order"]
    ]
    if sorted(device_ids) != list(range(32)) or len(set(device_ids)) != 32:
        raise SystemExit("PP16 fleet physical device coverage failed")

    loaded_files: set[str] = set()
    payload_bytes = 0
    tensor_count = 0
    maximum_peak_hbm = 0
    minimum_largest_free: int | None = None
    stage_summaries = []
    for path, record in zip(record_paths, records, strict=True):
        stage = record["stage_id"]
        if (
            record.get("status") != "SUCCESS"
            or record.get("code_hash") != args.code_hash
            or record.get("plan_id") != "PP16_LP2"
            or record.get("topology_sha256") != TOPOLOGY_SHA
            or record.get("plan_group_sha256") != GROUP_SHA
        ):
            raise SystemExit(f"PP16 stage {stage} identity failed")
        load = record["load"]
        if (
            not load["device_roundtrip_verified"]
            or load["device_roundtrip_bytes"] != load["loaded_payload_bytes"]
            or any(
                load[key]
                for key in (
                    "fp8_device_dequantizations",
                    "fp8_host_dequantizations",
                    "host_global_concatenations",
                    "runtime_checkpoint_reshards",
                )
            )
        ):
            raise SystemExit(f"PP16 stage {stage} direct-load contract failed")
        state = states[path.name]
        unhashed = dict(state)
        observed_state_hash = unhashed.pop("manifest_sha256", None)
        if (
            observed_state_hash != _semantic_hash(unhashed)
            or observed_state_hash
            != record["state_manifest"]["manifest_sha256"]
        ):
            raise SystemExit(f"PP16 stage {stage} state semantic hash failed")
        state_path = (
            args.run_dir
            / "host_records"
            / path.name.replace("stage_", "state_")
        )
        if (
            sha256(state_path.read_bytes()).hexdigest()
            != record["state_manifest"]["file_sha256"]
        ):
            raise SystemExit(f"PP16 stage {stage} state file hash failed")
        owner_plans = expected_by_stage[stage]
        owner_states = sorted(
            state["files"], key=lambda item: item["device_slot"]
        )
        if len(owner_plans) != 2 or len(owner_states) != 2:
            raise SystemExit(f"PP16 stage {stage} owner count failed")
        if record["captured_device_ids_in_slot_order"] != [
            plan.device_id for plan in owner_plans
        ]:
            raise SystemExit(f"PP16 stage {stage} physical owner order failed")
        for plan, item in zip(owner_plans, owner_states, strict=True):
            evidence = checkpoint.evidence_by_filename[plan.filename]
            if (
                item["filename"] != plan.filename
                or item["device_id"] != plan.device_id
                or item["device_slot"] != plan.device_slot
                or item["file_sha256"] != evidence["sha256"]
                or item["payload_bytes"] != plan.payload_bytes
                or item["tensor_count"] != len(plan.tensors)
            ):
                raise SystemExit(f"PP16 loaded owner drifted: {plan.filename}")
            expected_tensors = {
                tensor.name: (
                    tensor.dtype,
                    list(tensor.shape),
                    tensor.byte_count,
                )
                for tensor in plan.tensors
            }
            observed_tensors = {
                tensor["name"]: (
                    tensor["logical_dtype"],
                    tensor["shape"],
                    tensor["byte_count"],
                )
                for tensor in item["tensors"]
            }
            if observed_tensors != expected_tensors or any(
                len(tensor["sha256"]) != 64 for tensor in item["tensors"]
            ):
                raise SystemExit(f"PP16 tensor ledger drifted: {plan.filename}")
            loaded_files.add(plan.filename)
        if (
            state["payload_bytes"] != load["loaded_payload_bytes"]
            or state["tensor_count"] != load["loaded_tensor_count"]
        ):
            raise SystemExit(f"PP16 stage {stage} state/load totals failed")
        memories = record["device_memory_after_load"]
        if len(memories) != 2 or any(memory is None for memory in memories):
            raise SystemExit(f"PP16 stage {stage} HBM stats missing")
        peaks = [memory["peak_bytes_in_use"] for memory in memories]
        free = [memory["largest_free_block_bytes"] for memory in memories]
        if any(value <= 0 for value in peaks + free):
            raise SystemExit(f"PP16 stage {stage} HBM stats invalid")
        maximum_peak_hbm = max(maximum_peak_hbm, *peaks)
        minimum_largest_free = (
            min(free)
            if minimum_largest_free is None
            else min(minimum_largest_free, *free)
        )
        payload_bytes += state["payload_bytes"]
        tensor_count += state["tensor_count"]
        stage_summaries.append(
            {
                "device_ids": record[
                    "captured_device_ids_in_slot_order"
                ],
                "hostname": record["hostname"],
                "load_seconds": record[
                    "payload_load_and_roundtrip_seconds"
                ],
                "peak_hbm_bytes": max(peaks),
                "payload_bytes": state["payload_bytes"],
                "stage_id": stage,
                "state_manifest_sha256": observed_state_hash,
                "tensor_count": state["tensor_count"],
            }
        )
    if (
        loaded_files != {plan.filename for plan in plans}
        or payload_bytes != sum(plan.payload_bytes for plan in plans)
        or tensor_count != sum(len(plan.tensors) for plan in plans)
    ):
        raise SystemExit("PP16 full base checkpoint totals failed")

    sys.path.insert(0, str(REPO / "bench"))
    import provenance as pv

    conn = pv.connect(args.results_db)
    run_id = pv.start_run(
        conn,
        model="zai-org/GLM-5.2-FP8:greenfield-pp16-raw-fp8-final-layout-load",
        revision=REVISION,
        env={
            "GLM_ENGINE": "greenfield",
            "GLM_EXECUTION_PLAN": "PP16_LP2",
            "greenfield_code_hash": args.code_hash,
            "layout_manifest_sha256": LAYOUT_SHA,
            "packed_manifest_sha256": PACKED_SHA,
            "topology_hash": TOPOLOGY_SHA,
        },
        note="Gate-B PP16 full checkpoint direct-load integrity proof; not throughput.",
        harness_repo=str(REPO),
    )
    for row in stage_summaries:
        pv.record_item(
            conn,
            run_id,
            benchmark="greenfield_full_checkpoint_load_pp16",
            item_id=f"stage_{row['stage_id']:02d}",
            prompt=(
                "Directly load one final-owner PP16 stage and "
                "round-trip every device byte."
            ),
            gold=(
                "Exact two physical owners, raw FP8 resident, "
                "full SHA/state/HBM contract."
            ),
            raw_output=json.dumps(row, sort_keys=True),
            extracted=row["state_manifest_sha256"],
            correct=True,
            score=1.0,
        )
    pv.finalize(
        conn,
        run_id,
        benchmark="greenfield_full_checkpoint_load_pp16",
        metric="contract_valid",
        value=1.0,
        note="Load/integrity evidence only; no token latency or throughput claim.",
    )
    conn.close()
    summary = {
        "artifact_kind": "greenfield_full_checkpoint_load_proof",
        "base_file_count": len(plans),
        "checkpoint_destination": args.destination,
        "code_hash": args.code_hash,
        "device_roundtrip_verified": True,
        "host_count": 8,
        "loaded_payload_bytes": payload_bytes,
        "loaded_tensor_count": tensor_count,
        "maximum_peak_hbm_bytes": maximum_peak_hbm,
        "minimum_largest_free_block_bytes": minimum_largest_free,
        "packed_manifest_sha256": PACKED_SHA,
        "performance_claim": False,
        "plan_id": "PP16_LP2",
        "results_db_run_id": run_id,
        "stage_summaries": sorted(
            stage_summaries, key=lambda row: row["stage_id"]
        ),
        "status": "SUCCESS",
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (
        sqlite3.connect(args.results_db) as source,
        sqlite3.connect(args.run_dir / "results_ckpt.db") as snapshot,
    ):
        source.backup(snapshot)
    with sqlite3.connect(args.run_dir / "results_ckpt.db") as snapshot:
        if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise SystemExit("PP16 results DB snapshot failed")
    print(
        f"PP16_FULL_LOAD_VALID stages=16 devices=32 "
        f"payload={payload_bytes} db={run_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
