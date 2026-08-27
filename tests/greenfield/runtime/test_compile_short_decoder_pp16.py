from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.greenfield.validate_short_decoder_compile_pp16 import (
    EXPECTED_JAX_PROCESS_BY_LAUNCH,
    PLAN_HASH,
    RUNTIME_LAYOUT_HASH,
    RUNTIME_MANIFEST_SHA256,
    SCHEDULE_HASH,
    TOPOLOGY_HASH,
    validate_records,
)
from scripts.greenfield.seal_short_decoder_compile_pp16 import (
    TOP_LEVEL_EVIDENCE,
    _local_evidence,
    _parse_remote,
)


CODE_HASH = "1" * 40
HLO_HASH = "2" * 64
STABLEHLO_HASH = "3" * 64
STATE_HASH = "4" * 64
REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_short_decoder_compile_pp16.sh"


def _record(rank: int) -> dict[str, object]:
    memory = [
        {
            "bytes_in_use": 28_000_000_000,
            "bytes_limit": 33_000_000_000,
            "largest_free_block_bytes": 5_000_000_000,
            "peak_bytes_in_use": 28_100_000_000,
        }
        for _ in range(4)
    ]
    contract = {
        "backend_contract": "tpu_v4_pp16_pallas_feature",
        "collective_counts": {
            "all-gather": 1,
            "all-reduce": 1,
            "collective-permute": 16,
        },
        "complete_token_collective_contract": {"passed": True},
        "complete_token_path": True,
        "dsa_head_key_association_contract": {"passed": True},
        "dsa_query_association_contract": {"passed": True},
        "expected_collective_counts": {
            "all-gather": 1,
            "all-reduce": 1,
            "collective-permute": 16,
        },
        "feature_fuse_route_weighting": False,
        "feature_output_tile": 128,
        "feature_reconstruct_down_fp32": False,
        "forbidden_full_vocab_logits": [],
        "forbidden_shapes": [],
        "full_indexer_layers": 21,
        "layer_count": 78,
        "live_tensor_contract": {"passed": True},
        "num_partitions": 32,
        "pallas_feature_contract": {
            "local_intermediate": 1024,
            "local_parallel_size": 2,
            "passed": True,
        },
        "passed": True,
        "residual_transport_count": 16,
        "residual_transport_dimensions": [2, 1, 6144],
        "residual_transport_dtype": "bf16",
        "split_residual_state": True,
        "violations": [],
    }
    return {
        "artifact_kind": "greenfield_real_78layer_2k_decoder_token_pallas_feature",
        "attention_projection_backend": "separate",
        "body_only": False,
        "code_hash": CODE_HASH,
        "compile_seconds": 10.0 + rank,
        "complete_token_path": True,
        "context_capacity": 2048,
        "dense_final_layout_convolution": False,
        "device_memory_after_execute": memory,
        "dsa_head_key_exact_association": False,
        "dsa_query_exact_association": False,
        "dsa_score_default_precision": False,
        "feature_fuse_route_weighting": False,
        "feature_output_tile": 128,
        "feature_reconstruct_down_fp32": False,
        "fleet_hlo_hashes": [HLO_HASH] * 8,
        "fleet_local_device_ids_in_runtime_order": [
            list(range(process * 4, process * 4 + 4))
            for process in range(8)
        ],
        "fleet_stablehlo_hashes": [STABLEHLO_HASH] * 8,
        "hlo_contract": contract,
        "hostname": f"pod-w-{rank}",
        "iterations": 1,
        "jax_process_index": EXPECTED_JAX_PROCESS_BY_LAUNCH[rank],
        "launch_process_id": rank,
        "linear_backend": "reference",
        "load_record": {
            "addressable_device_count": 4,
            "device_roundtrip_bytes": 0,
            "device_roundtrip_verified": False,
            "fp8_device_dequantizations": 0,
            "fp8_host_dequantizations": 0,
            "global_array_count": 217,
            "host_global_concatenations": 0,
            "loaded_payload_bytes": 108_693_168_384,
            "loaded_tensor_count": 868,
            "runtime_checkpoint_reshards": 0,
        },
        "load_seconds": 20.0 + rank,
        "main_rope_table_enabled": False,
        "metadata_passed": True,
        "optimized_hlo_sha256": HLO_HASH,
        "plan_hash": PLAN_HASH,
        "prefill_index_repair": False,
        "prefill_used": False,
        "pregathered_b512_attention": False,
        "profiler_free_complete_step_wall": {"count": 1, "p50_ms": 150.0},
        "raw_token_claim": False,
        "residual_transport_bytes_per_stage": 24_576,
        "residual_transport_components": 2,
        "runtime_kind": "pallas_feature",
        "runtime_layout_hash": RUNTIME_LAYOUT_HASH,
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "schedule_hash": SCHEDULE_HASH,
        "schema_version": 18,
        "sparse_moe_backend": "pallas_feature",
        "split_residual_state": True,
        "stablehlo_sha256": STABLEHLO_HASH,
        "state_layout": {
            "context_capacity": 2048,
            "local_parallel_size": 2,
            "plan_hash": PLAN_HASH,
            "schedule_hash": SCHEDULE_HASH,
            "stages": [{} for _ in range(16)],
        },
        "state_layout_hash": STATE_HASH,
        "strategy_nd_attention_projection": False,
        "token_contract": {
            "all_active_lanes_equal": True,
            "all_in_vocabulary": True,
            "position_contract_passed": True,
            "raw_token_sequence": None,
            "synthetic_initial_state": True,
        },
        "token_passed": True,
        "topology_hash": TOPOLOGY_HASH,
        "trace": None,
        "transformer_body_timing_only": False,
        "warmup": 1,
    }


def _write_records(root: Path, records: list[dict[str, object]]) -> None:
    host_records = root / "host_records"
    host_records.mkdir()
    for rank, record in enumerate(records):
        (host_records / f"decoder.rank{rank}.json").write_text(
            json.dumps(record) + "\n"
        )


def test_pp16_compile_acquisition_validator_separates_diagnostic_timing(
    tmp_path: Path,
) -> None:
    _write_records(tmp_path, [_record(rank) for rank in range(8)])

    summary = validate_records(tmp_path, CODE_HASH)

    assert summary["status"] == "SUCCESS"
    assert summary["physical_chip_count"] == 32
    assert summary["performance_claim"] is False
    assert summary["numerical_claim"] is False
    assert summary["gate_d_passed"] is False
    assert summary["results_db_run_id"] is None
    assert summary["topology_results_db_run_id"] == 555
    assert summary["launch_to_jax_process"] == {
        str(rank): process_index
        for rank, process_index in enumerate(EXPECTED_JAX_PROCESS_BY_LAUNCH)
    }


def test_pp16_compile_acquisition_validator_rejects_hlo_locality_drift(
    tmp_path: Path,
) -> None:
    records = [_record(rank) for rank in range(8)]
    broken = copy.deepcopy(records[3])
    broken["hlo_contract"]["forbidden_shapes"] = ["bf16[32,6144]"]
    records[3] = broken
    _write_records(tmp_path, records)

    with pytest.raises(ValueError, match="forbidden_shapes"):
        validate_records(tmp_path, CODE_HASH)


def test_pp16_compile_acquisition_validator_rejects_jax_process_drift(
    tmp_path: Path,
) -> None:
    records = [_record(rank) for rank in range(8)]
    records[0]["jax_process_index"] = 0
    _write_records(tmp_path, records)

    with pytest.raises(ValueError, match="protected topology DB555"):
        validate_records(tmp_path, CODE_HASH)


def test_pp16_compile_acquisition_runner_is_small_default_off_and_protected() -> None:
    source = RUNNER.read_text()

    assert "GLM_GREENFIELD_PP16_COMPILE_ACQUISITION:-0" in source
    assert "--context-capacity 2048 --warmup 1 --iterations 1 --trace-steps 0" in source
    assert "--runtime-kind pallas_feature" in source
    assert "--feature-output-tile 128" in source
    assert "--complete-token-path 1 --split-residual-state 1" in source
    assert "--verify-device-roundtrip 0" in source
    assert "US-CENTRAL2" in source
    assert "gs://driftbench-dsv4-uc" in source
    assert "gs://driftbench-storage" not in source
    assert ".glm_pod_workload.lock" in source
    assert ".glm-tpu-rsync.lock" in source
    assert source.count("strict_census") >= 4
    assert "validate_short_decoder_compile_pp16.py" in source
    assert "seal_short_decoder_compile_pp16.py" in source
    assert "results.db" not in source


def test_pp16_compile_sealer_requires_exact_local_evidence_set(
    tmp_path: Path,
) -> None:
    for folder, count in (("host_records", 8), ("host_logs", 8), ("hlo", 3)):
        root = tmp_path / folder
        root.mkdir()
        for index in range(count):
            (root / f"artifact{index}").write_text("sealed\n")
    for name in TOP_LEVEL_EVIDENCE:
        (tmp_path / name).write_text("sealed\n")

    evidence = _local_evidence(tmp_path)

    assert len(evidence) == 28
    assert _parse_remote(
        "gs://driftbench-dsv4-uc/results/compile-proof"
    ) == ("driftbench-dsv4-uc", "results/compile-proof/")
    with pytest.raises(ValueError, match="approved results bucket"):
        _parse_remote("gs://driftbench-storage/results/compile-proof")
