#!/usr/bin/env python3
"""Validate the smallest complete PP16 decoder compile acquisition.

This artifact proves load, compilation, HLO locality, one-row shape, and 2K
HBM feasibility only. Its single timed iteration is diagnostic and is never a
Gate-D or performance result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


PLAN_HASH = "cefab5e7ed373a0896e7d1471713abb8ef89c9dedca78ee115c8d1fccc7ad172"
SCHEDULE_HASH = "02b0ae76572ee8ec40c850c2b252f6f85f06338ae5f956c2aa130c32b12c8eac"
RUNTIME_MANIFEST_SHA256 = (
    "0f1bb2718a700fb2eee23dc9f172cd9e5cbd1639396d8c8fa8421e1e3b52b6f1"
)
RUNTIME_LAYOUT_HASH = (
    "7764784461cc8614b703e3ff943851068ae63a4b97c9edb1c5a5376bc24399c7"
)
TOPOLOGY_HASH = (
    "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
)
TOPOLOGY_RESULTS_DB_RUN_ID = 555
EXPECTED_JAX_PROCESS_BY_LAUNCH = (3, 5, 1, 2, 0, 6, 7, 4)
EXPECTED_PAYLOAD_BYTES_PER_HOST = 108_693_168_384
EXPECTED_TENSORS_PER_HOST = 868
EXPECTED_GLOBAL_ARRAYS = 217


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _memory_rows(record: Mapping[str, Any], rank: int) -> list[dict[str, int]]:
    rows = record.get("device_memory_after_execute")
    _require(isinstance(rows, list) and len(rows) == 4, f"rank {rank} lacks four HBM rows")
    result: list[dict[str, int]] = []
    for local_slot, row in enumerate(rows):
        _require(isinstance(row, dict), f"rank {rank} HBM row {local_slot} is invalid")
        for field in (
            "bytes_in_use",
            "bytes_limit",
            "largest_free_block_bytes",
            "peak_bytes_in_use",
        ):
            _require(
                isinstance(row.get(field), int) and row[field] > 0,
                f"rank {rank} HBM row {local_slot} lacks {field}",
            )
        _require(
            row["peak_bytes_in_use"] < row["bytes_limit"],
            f"rank {rank} HBM row {local_slot} has no measured headroom",
        )
        result.append(row)
    return result


def validate_records(
    run_dir: Path,
    code_hash: str,
    runtime_kind: str = "pallas_feature",
) -> dict[str, Any]:
    if runtime_kind not in ("pallas_feature", "pallas_feature_linear"):
        raise ValueError("PP16 acquisition runtime kind is unknown")
    linear_backend = (
        "pallas" if runtime_kind == "pallas_feature_linear" else "reference"
    )
    backend_contract = (
        "tpu_v4_pp16_pallas_feature_linear"
        if linear_backend == "pallas"
        else "tpu_v4_pp16_pallas_feature"
    )
    paths = sorted((run_dir / "host_records").glob("decoder.rank*.json"))
    _require(len(paths) == 8, "PP16 acquisition requires exactly eight host records")
    records_by_rank: dict[int, dict[str, Any]] = {}
    for path in paths:
        match = re.fullmatch(r"decoder\.rank([0-7])\.json", path.name)
        _require(match is not None, f"unexpected host-record filename {path.name}")
        rank = int(match.group(1))
        _require(rank not in records_by_rank, f"duplicate rank {rank}")
        value = json.loads(path.read_text())
        _require(isinstance(value, dict), f"rank {rank} record is not an object")
        records_by_rank[rank] = value
    _require(set(records_by_rank) == set(range(8)), "PP16 rank set is incomplete")

    records = [records_by_rank[rank] for rank in range(8)]
    hostnames = set()
    state_layout_hashes = set()
    memory_rows: list[dict[str, int]] = []
    for rank, record in enumerate(records):
        _require(record.get("schema_version") == 18, f"rank {rank} schema drifted")
        _require(
            record.get("artifact_kind")
            == "greenfield_real_78layer_2k_decoder_token_pallas_feature",
            f"rank {rank} artifact kind drifted",
        )
        expected_scalars = {
            "attention_projection_backend": "separate",
            "body_only": False,
            "code_hash": code_hash,
            "complete_token_path": True,
            "context_capacity": 2048,
            "dense_final_layout_convolution": False,
            "dsa_head_key_exact_association": False,
            "dsa_query_exact_association": False,
            "dsa_score_default_precision": False,
            "feature_fuse_route_weighting": False,
            "feature_output_tile": 128,
            "feature_reconstruct_down_fp32": False,
            "iterations": 1,
            "linear_backend": linear_backend,
            "main_rope_table_enabled": False,
            "metadata_passed": True,
            "plan_hash": PLAN_HASH,
            "prefill_index_repair": False,
            "prefill_used": False,
            "pregathered_b512_attention": False,
            "raw_token_claim": False,
            "residual_transport_bytes_per_stage": 24_576,
            "residual_transport_components": 2,
            "runtime_kind": runtime_kind,
            "runtime_layout_hash": RUNTIME_LAYOUT_HASH,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "schedule_hash": SCHEDULE_HASH,
            "sparse_moe_backend": "pallas_feature",
            "split_residual_state": True,
            "strategy_nd_attention_projection": False,
            "token_passed": True,
            "topology_hash": TOPOLOGY_HASH,
            "trace": None,
            "transformer_body_timing_only": False,
            "warmup": 1,
        }
        for field, expected in expected_scalars.items():
            _require(record.get(field) == expected, f"rank {rank} drifted {field}")
        _require(record.get("launch_process_id") == rank, f"rank {rank} launch id drifted")
        _require(
            record.get("jax_process_index")
            == EXPECTED_JAX_PROCESS_BY_LAUNCH[rank],
            f"rank {rank} JAX id drifted from protected topology DB"
            f"{TOPOLOGY_RESULTS_DB_RUN_ID}",
        )
        hostname = record.get("hostname")
        _require(isinstance(hostname, str) and hostname, f"rank {rank} hostname missing")
        hostnames.add(hostname)

        load = record.get("load_record")
        _require(isinstance(load, dict), f"rank {rank} load record missing")
        expected_load = {
            "addressable_device_count": 4,
            "device_roundtrip_bytes": 0,
            "device_roundtrip_verified": False,
            "fp8_device_dequantizations": 0,
            "fp8_host_dequantizations": 0,
            "global_array_count": EXPECTED_GLOBAL_ARRAYS,
            "host_global_concatenations": 0,
            "loaded_payload_bytes": EXPECTED_PAYLOAD_BYTES_PER_HOST,
            "loaded_tensor_count": EXPECTED_TENSORS_PER_HOST,
            "runtime_checkpoint_reshards": 0,
        }
        for field, expected in expected_load.items():
            _require(load.get(field) == expected, f"rank {rank} load drifted {field}")

        state_layout = record.get("state_layout")
        _require(isinstance(state_layout, dict), f"rank {rank} state layout missing")
        _require(
            state_layout.get("context_capacity") == 2048,
            f"rank {rank} state capacity drifted",
        )
        _require(state_layout.get("local_parallel_size") == 2, f"rank {rank} state LP drifted")
        _require(state_layout.get("plan_hash") == PLAN_HASH, f"rank {rank} state plan drifted")
        _require(
            state_layout.get("schedule_hash") == SCHEDULE_HASH,
            f"rank {rank} state schedule drifted",
        )
        _require(len(state_layout.get("stages", [])) == 16, f"rank {rank} state stages drifted")
        state_layout_hash = record.get("state_layout_hash")
        _require(
            isinstance(state_layout_hash, str) and len(state_layout_hash) == 64,
            f"rank {rank} state hash missing",
        )
        state_layout_hashes.add(state_layout_hash)

        contract = record.get("hlo_contract")
        _require(isinstance(contract, dict), f"rank {rank} HLO contract missing")
        contract_expected = {
            "backend_contract": backend_contract,
            "complete_token_path": True,
            "feature_fuse_route_weighting": False,
            "feature_output_tile": 128,
            "feature_reconstruct_down_fp32": False,
            "forbidden_shapes": [],
            "forbidden_full_vocab_logits": [],
            "full_indexer_layers": 21,
            "layer_count": 78,
            "num_partitions": 32,
            "passed": True,
            "residual_transport_count": 16,
            "residual_transport_dimensions": [2, 1, 6144],
            "residual_transport_dtype": "bf16",
            "split_residual_state": True,
            "violations": [],
        }
        for field, expected in contract_expected.items():
            _require(contract.get(field) == expected, f"rank {rank} HLO drifted {field}")
        _require(
            contract.get("collective_counts") == contract.get("expected_collective_counts"),
            f"rank {rank} HLO collective counts drifted",
        )
        for subfield in (
            "complete_token_collective_contract",
            "dsa_head_key_association_contract",
            "dsa_query_association_contract",
            "live_tensor_contract",
            "pallas_feature_contract",
        ):
            sub = contract.get(subfield)
            _require(
                isinstance(sub, dict) and sub.get("passed") is True,
                f"rank {rank} HLO subcontract failed: {subfield}",
            )
        stage_linear = contract.get("pallas_stage_linear_contract")
        if linear_backend == "pallas":
            _require(
                isinstance(stage_linear, dict)
                and stage_linear.get("passed") is True,
                f"rank {rank} Pallas-linear HLO contract failed",
            )
        else:
            _require(
                stage_linear in ({}, None),
                f"rank {rank} unexpectedly claims Pallas-linear HLO",
            )
        pallas = contract["pallas_feature_contract"]
        _require(pallas.get("local_parallel_size") == 2, f"rank {rank} Pallas LP drifted")
        _require(
            pallas.get("local_intermediate") == 1024,
            f"rank {rank} Pallas feature width drifted",
        )

        optimized = record.get("optimized_hlo_sha256")
        stable = record.get("stablehlo_sha256")
        _require(
            isinstance(optimized, str) and len(optimized) == 64,
            f"rank {rank} HLO hash missing",
        )
        _require(
            isinstance(stable, str) and len(stable) == 64,
            f"rank {rank} StableHLO hash missing",
        )
        _require(
            record.get("fleet_hlo_hashes") == [optimized] * 8,
            f"rank {rank} fleet HLO disagreement",
        )
        _require(
            record.get("fleet_stablehlo_hashes") == [stable] * 8,
            f"rank {rank} fleet StableHLO disagreement",
        )

        fleet_ids = record.get("fleet_local_device_ids_in_runtime_order")
        _require(
            isinstance(fleet_ids, list) and len(fleet_ids) == 8,
            f"rank {rank} fleet device map missing",
        )
        _require(
            all(
                isinstance(row, list) and len(row) == 4
                for row in fleet_ids
            ),
            f"rank {rank} fleet device rows drifted",
        )
        _require(
            sorted(device for row in fleet_ids for device in row)
            == list(range(32)),
            f"rank {rank} fleet device ids drifted",
        )

        token = record.get("token_contract")
        _require(isinstance(token, dict), f"rank {rank} token mechanism record missing")
        _require(
            token.get("synthetic_initial_state") is True,
            f"rank {rank} unexpectedly claims a prompt",
        )
        _require(
            token.get("raw_token_sequence") is None,
            f"rank {rank} unexpectedly claims raw tokens",
        )
        _require(token.get("all_active_lanes_equal") is True, f"rank {rank} token lanes disagree")
        _require(token.get("all_in_vocabulary") is True, f"rank {rank} token is out of vocabulary")
        _require(
            token.get("position_contract_passed") is True,
            f"rank {rank} token position drifted",
        )

        wall = record.get("profiler_free_complete_step_wall")
        _require(
            isinstance(wall, dict) and wall.get("count") == 1,
            f"rank {rank} diagnostic wall sample drifted",
        )
        _require(
            isinstance(wall.get("p50_ms"), (int, float))
            and wall["p50_ms"] > 0,
            f"rank {rank} diagnostic wall sample invalid",
        )
        memory_rows.extend(_memory_rows(record, rank))

    _require(len(hostnames) == 8, "PP16 acquisition hostnames are not unique")
    _require(len(state_layout_hashes) == 1, "PP16 state-layout hashes disagree")
    optimized_hashes = {record["optimized_hlo_sha256"] for record in records}
    stable_hashes = {record["stablehlo_sha256"] for record in records}
    _require(len(optimized_hashes) == 1, "PP16 optimized HLO hashes disagree")
    _require(len(stable_hashes) == 1, "PP16 StableHLO hashes disagree")

    contract = records[0]["hlo_contract"]
    summary = {
        "artifact_kind": "greenfield_pp16_short_decoder_compile_acquisition",
        "code_hash": code_hash,
        "collective_counts": contract["collective_counts"],
        "context_capacity": 2048,
        "diagnostic_only": True,
        "gate_d_passed": False,
        "host_count": 8,
        "launch_to_jax_process": {
            str(rank): process_index
            for rank, process_index in enumerate(EXPECTED_JAX_PROCESS_BY_LAUNCH)
        },
        "maximum_compile_seconds": max(float(record["compile_seconds"]) for record in records),
        "maximum_diagnostic_step_ms": max(
            float(record["profiler_free_complete_step_wall"]["p50_ms"])
            for record in records
        ),
        "maximum_load_seconds": max(float(record["load_seconds"]) for record in records),
        "maximum_peak_hbm_bytes": max(row["peak_bytes_in_use"] for row in memory_rows),
        "minimum_largest_free_block_bytes": min(
            row["largest_free_block_bytes"] for row in memory_rows
        ),
        "numerical_claim": False,
        "optimized_hlo_sha256": next(iter(optimized_hashes)),
        "performance_claim": False,
        "physical_chip_count": len(memory_rows),
        "plan_hash": PLAN_HASH,
        "plan_id": "PP16_LP2",
        "results_db_run_id": None,
        "runtime_layout_hash": RUNTIME_LAYOUT_HASH,
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "runtime_kind": runtime_kind,
        "linear_backend": linear_backend,
        "schedule_hash": SCHEDULE_HASH,
        "stablehlo_sha256": next(iter(stable_hashes)),
        "state_layout_hash": next(iter(state_layout_hashes)),
        "status": "SUCCESS",
        "topology_hash": TOPOLOGY_HASH,
        "topology_results_db_run_id": TOPOLOGY_RESULTS_DB_RUN_ID,
        "trace_claim": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument(
        "--runtime-kind",
        choices=("pallas_feature", "pallas_feature_linear"),
        default="pallas_feature",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_records(
        args.run_dir, args.code_hash, runtime_kind=args.runtime_kind
    )
    print(
        "PP16_COMPILE_ACQUISITION_VALID "
        f"hlo={summary['optimized_hlo_sha256']} "
        f"peak_hbm={summary['maximum_peak_hbm_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
