from __future__ import annotations

import copy
from hashlib import sha256
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
from scripts.greenfield.recover_short_decoder_compile_pp16 import (
    _crc32c_file,
    _parse_results_uri,
    _require_census,
    _source_blob_name,
    _source_local_files,
    prepare,
)


CODE_HASH = "1" * 40
HLO_HASH = "2" * 64
STABLEHLO_HASH = "3" * 64
STATE_HASH = "4" * 64
REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_short_decoder_compile_pp16.sh"
RECOVERY = REPO / "scripts/greenfield/recover_short_decoder_compile_pp16.py"
COMPILER = REPO / "scripts/greenfield/compile_short_decoder.py"


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
        "dsa_query_association_contract": {
            "exact_association": True,
            "exact_chunk_width": 1024,
            "exact_chunks_per_local_owner": 2,
            "expected_runtime_tuple4_reduction_count": 42,
            "expected_tuple4_reduction_fusion_count": 21,
            "forbidden_global_shapes": [],
            "local_owner_shape": "f32[2048,2048]",
            "passed": True,
            "tuple4_reduction_fusion_count": 21,
        },
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
        "attention_projection_backend": "fused_n82_convolution",
        "body_only": False,
        "code_hash": CODE_HASH,
        "compile_seconds": 10.0 + rank,
        "complete_token_path": True,
        "context_capacity": 2048,
        "dense_final_layout_convolution": False,
        "device_memory_after_execute": memory,
        "dsa_head_key_exact_association": False,
        "dsa_query_exact_association": True,
        "dsa_query_materialization_compile_seconds": 1.0,
        "dsa_query_materialization_execute_seconds": 0.5,
        "dsa_query_materialization_hlo_contract": {
            "forbidden_custom_call_targets": [],
            "forbidden_global_shapes": [],
            "forbidden_operations": [],
            "host_markers": [],
            "local_fp32_shape": "f32[2048,2048]",
            "local_raw_shape": "u8[2048,2048]",
            "local_scale_shape": "f32[16,16]",
            "num_partitions": 32,
            "passed": True,
        },
        "dsa_query_materialization_hlo_sha256": "5" * 64,
        "dsa_query_materialization_state": {
            "input_alias_count": 4,
            "local_shards": [
                {
                    "byte_count": 16_777_216,
                    "device_id": local_slot,
                    "sha256": "6" * 64,
                    "slot": slot,
                }
                for slot in range(4)
                for local_slot in range(4)
            ],
            "materialized_bytes_per_device": 67_108_864,
            "slot_count": 4,
            "source": "completed_stage_local_raw_fp8_to_fp32",
        },
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
        "fleet_dsa_query_materialization_hlo_hashes": ["5" * 64] * 8,
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
            "global_array_count": 203,
            "host_global_concatenations": 0,
            "loaded_payload_bytes": 108_707_162_112,
            "loaded_tensor_count": 812,
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


def test_pp16_compile_acquisition_validator_admits_pallas_linear(
    tmp_path: Path,
) -> None:
    records = [_record(rank) for rank in range(8)]
    for record in records:
        record["artifact_kind"] = (
            "greenfield_real_78layer_2k_decoder_token_pallas_feature_linear"
        )
        record["runtime_kind"] = "pallas_feature_linear"
        record["linear_backend"] = "pallas"
        contract = record["hlo_contract"]
        contract["backend_contract"] = "tpu_v4_pp16_pallas_feature_linear"
        contract["pallas_stage_linear_contract"] = {
            "local_parallel_size": 2,
            "passed": True,
        }

    _write_records(tmp_path, records)
    summary = validate_records(
        tmp_path, CODE_HASH, runtime_kind="pallas_feature_linear"
    )

    assert summary["linear_backend"] == "pallas"
    assert summary["runtime_kind"] == "pallas_feature_linear"


def test_pp16_compile_acquisition_runner_is_small_default_off_and_protected() -> None:
    source = RUNNER.read_text()

    assert "GLM_GREENFIELD_PP16_COMPILE_ACQUISITION:-0" in source
    assert "--context-capacity 2048 --warmup 1 --iterations 1 --trace-steps 0" in source
    assert "GLM_GREENFIELD_PP16_RUNTIME_KIND:-pallas_feature" in source
    assert '--runtime-kind ' in source
    assert "--feature-output-tile 128" in source
    assert "--complete-token-path 1 --split-residual-state 1" in source
    assert "--dsa-query-exact-association 1" in source
    assert "--dsa-head-key-exact-association 0" in source
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


class _RecoveryBlob:
    def __init__(self, *, size: int, crc32c: str, generation: int = 1) -> None:
        self.size = size
        self.crc32c = crc32c
        self.generation = generation


class _RecoveryBucket:
    def __init__(self, blobs: dict[str, _RecoveryBlob]) -> None:
        self.blobs = blobs

    def get_blob(self, name: str) -> _RecoveryBlob | None:
        return self.blobs.get(name)


class _RecoveryClient:
    def __init__(self, blobs: dict[str, _RecoveryBlob]) -> None:
        self.blobs = blobs

    def bucket(self, name: str) -> _RecoveryBucket:
        assert name == "driftbench-dsv4-uc"
        return _RecoveryBucket(self.blobs)


def _census() -> str:
    return "".join(f"CENSUS_OK replacement-pod-w-{rank}\n" for rank in range(8))


def _write_recovery_source(root: Path) -> None:
    _write_records(root, [_record(rank) for rank in range(8)])
    host_logs = root / "host_logs"
    host_logs.mkdir()
    for rank in range(8):
        (host_logs / f"decoder.rank{rank}.log").write_text(f"rank={rank}\n")
    hlo = root / "hlo"
    hlo.mkdir()
    for name in ("contract.json", "optimized.txt.gz", "stable.mlir.gz"):
        (hlo / name).write_text(f"{name}\n")
    for name in ("census_pre.txt", "census_post.txt"):
        (root / name).write_text(_census())
    for name in ("execute.txt", "preflight.json", "remote_vacancy.txt", "sync.txt"):
        (root / name).write_text(f"{name}\n")
    (root / "orchestrator.log").write_text("historical workload\n")


def test_pp16_compile_recovery_prepares_generation_pinned_no_tpu_capsule(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_recovery_source(source)
    census = tmp_path / "recovery-census.txt"
    census.write_text(_census())
    source_tag = "greenfield_short_decoder_compile_pp16_acquisition_20260827T132707782908362Z"
    source_uri = f"gs://driftbench-dsv4-uc/results/{source_tag}"
    recovery_tag = "greenfield_short_decoder_compile_pp16_recovery_20260827T170000000000000Z"
    recovery_uri = f"gs://driftbench-dsv4-uc/results/{recovery_tag}"
    _, source_prefix = _parse_results_uri(source_uri)
    local_files = _source_local_files(source)
    blobs = {}
    for relative_name, path in local_files.items():
        name = _source_blob_name(
            relative_name,
            source_prefix=source_prefix,
            source_run_tag=source_tag,
        )
        blobs[name] = _RecoveryBlob(
            size=path.stat().st_size, crc32c=_crc32c_file(path)
        )
    run_dir = tmp_path / "recovered"

    result = prepare(
        source_dir=source,
        run_dir=run_dir,
        recovery_census=census,
        source_remote_prefix=source_uri,
        remote_prefix=recovery_uri,
        source_run_tag=source_tag,
        recovery_run_tag=recovery_tag,
        workload_code_hash=CODE_HASH,
        recovery_code_hash="5" * 40,
        runtime_kind="pallas_feature",
        client=_RecoveryClient(blobs),
    )

    assert result["summary"]["status"] == "SUCCESS"
    assert result["recovery"]["model_workload_rerun"] is False
    assert result["recovery"]["tpu_initialized_by_recovery"] is False
    assert result["recovery"]["performance_claim"] is False
    assert result["recovery"]["gate_d_passed"] is False
    assert len(result["recovery"]["source_objects"]) == 26
    for line in (run_dir / "evidence.sha256").read_text().splitlines():
        expected, relative_name = line.split("  ", 1)
        assert sha256((run_dir / relative_name).read_bytes()).hexdigest() == expected


def test_pp16_compile_recovery_rejects_non_eight_host_census(tmp_path: Path) -> None:
    census = tmp_path / "census.txt"
    census.write_text("CENSUS_OK replacement-pod-w-0\n")

    with pytest.raises(ValueError, match="not authenticated 8/8"):
        _require_census(census)


def test_pp16_compile_recovery_is_create_only_and_approved_bucket_only() -> None:
    source = RECOVERY.read_text()

    assert "if_generation_match=0" in source
    assert "model_workload_rerun\": False" in source
    assert "tpu_initialized_by_recovery\": False" in source
    assert "import jax" not in source
    assert _parse_results_uri(
        "gs://driftbench-dsv4-uc/results/recovery"
    ) == ("driftbench-dsv4-uc", "results/recovery/")
    with pytest.raises(ValueError, match="approved results bucket"):
        _parse_results_uri("gs://driftbench-storage/results/recovery")


def test_pp16_pallas_linear_lowering_is_explicitly_admitted() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _FEATURE_DECODER_BACKEND_CONTRACTS,
        _PALLAS_LINEAR_DECODER_BACKEND_CONTRACTS,
        _TPU_DECODER_BACKEND_CONTRACTS,
    )

    contract = "tpu_v4_pp16_pallas_feature_linear"
    source = COMPILER.read_text()

    assert contract in _TPU_DECODER_BACKEND_CONTRACTS
    assert contract in _FEATURE_DECODER_BACKEND_CONTRACTS
    assert contract in _PALLAS_LINEAR_DECODER_BACKEND_CONTRACTS
    assert '("PP16_LP2", True): "tpu_v4_pp16_pallas_feature_linear"' in source
    assert "PP16 Pallas-linear lowering is not yet an admitted contract" not in source
