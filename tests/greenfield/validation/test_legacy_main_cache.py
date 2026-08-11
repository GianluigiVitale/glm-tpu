from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.legacy_main_cache import (
    LegacyMainCacheComparisonConfig,
    compare_legacy_layer0_main_cache,
)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bits(rows: int, width: int) -> np.ndarray:
    values = np.arange(rows * width, dtype=np.float32).reshape(rows, width)
    return (values / np.float32(128)).astype(ml_dtypes.bfloat16).view(
        np.uint16)


def _write_dump(
    path: Path,
    *,
    process_index: int,
    step_index: int,
    sequence_length: int,
    cache_bits: np.ndarray,
    corrupt_replica: bool = False,
) -> None:
    index = tuple(slice(None, None, None) for _ in cache_bits.shape)
    values: dict[str, np.ndarray] = {
        "step_index":
        np.asarray(step_index, dtype=np.int64),
        "phase":
        np.asarray("postfwd"),
        "num_scheduled_tokens":
        np.asarray(2 if step_index == 2 else 1, dtype=np.int64),
        "process_index":
        np.asarray(process_index, dtype=np.int64),
        "process_count":
        np.asarray(2, dtype=np.int64),
        "layer_indices":
        np.asarray([1], dtype=np.int64),
        "mesh_shape":
        np.asarray("{'data': 1, 'attn_dp': 1, 'attn_dp_expert': 1, "
                   "'expert': 1, 'model': 4, 'dcp': 1}"),
        "meta__block_tables":
        np.asarray([1, 2, 0, 0], dtype=np.int32),
        "meta__seq_lens":
        np.asarray([sequence_length], dtype=np.int32),
        "layer1__sharding":
        np.asarray("P(None, 'dcp')"),
        "layer1__shape":
        np.asarray(cache_bits.shape, dtype=np.int64),
        "layer1__dtype":
        np.asarray("bfloat16"),
        "layer1__nshards":
        np.asarray(2, dtype=np.int64),
    }
    for local in range(2):
        shard = cache_bits.copy()
        if corrupt_replica and local == 1:
            shard.flat[0] ^= np.uint16(1)
        prefix = f"layer1__shard{local}"
        values[f"{prefix}__data"] = shard
        values[f"{prefix}__index"] = np.asarray(str(index))
        values[f"{prefix}__device"] = np.asarray(
            f"TPU_{process_index * 2 + local}(process={process_index},(0,0,0,0))"
        )
    np.savez(path, **values)


def _write_ingredients(
    directory: Path,
    *,
    selected_delta: bool = False,
    current_delta: bool = False,
    padding_delta: bool = False,
) -> tuple[str, str]:
    directory.mkdir()
    logical = _bits(41, 8)
    logical[:, 6:] = 0
    selected_positions = np.asarray([[1, 4, 33], [1, 4, 33]], dtype=np.int32)
    owner_positions = np.asarray([[1, 4, -1], [33, -1, -1]], dtype=np.int32)
    owner_counts = np.asarray([[2], [1]], dtype=np.int32)
    owner_values = np.zeros((2, 3, 8), dtype=np.uint16)
    owner_values[0, :2] = logical[[1, 4]]
    owner_values[1, 0] = logical[33]
    if selected_delta:
        owner_values[0, 1, 3] ^= np.uint16(1)
    current = np.repeat(logical[40:41], 2, axis=0)
    if current_delta:
        current[:, 2] ^= np.uint16(1)
    if padding_delta:
        owner_values[0, 0, 7] = np.uint16(1)
    arrays = {
        "selected_positions": selected_positions,
        "owner_selected_positions": owner_positions,
        "owner_selected_valid_counts": owner_counts,
        "owner_selected_cache_values_bfloat16_bits": owner_values,
        "owner_selected_cache_valid": np.ones((2, 1), dtype=bool),
        "current_cache_row_bfloat16_bits": current,
    }
    contract_arrays = {
        name: {
            "dtype": str(value.dtype),
            "sha256": sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape),
        }
        for name, value in arrays.items()
    }
    contract = {
        "arrays": contract_arrays,
        "code_hash": "d" * 40,
        "decode_position": 40,
        "finite_passed": True,
        "health_passed": True,
        "hlo_sha256": "e" * 64,
        "lane_replication_passed": True,
        "owner_partition_passed": True,
        "owner_union_exact": True,
        "passed": True,
        "selection_exact": True,
        "source_state": "post_teacher_forced_prefill",
    }
    contract_path = directory / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    np.savez(
        directory / "position_40_ingredients.npz",
        decode_position=np.asarray([40], dtype=np.int32),
        **arrays,
    )
    return _file_hash(contract_path), _file_hash(directory /
                                                 "position_40_ingredients.npz")


def _setup(
    tmp_path: Path,
    *,
    selected_delta: bool = False,
    current_delta: bool = False,
    padding_delta: bool = False,
    corrupt_replica: bool = False,
) -> LegacyMainCacheComparisonConfig:
    source = tmp_path / "source"
    source.mkdir()
    logical = _bits(41, 8)
    logical[:, 6:] = 0
    prefill_cache = np.zeros((3, 2, 32, 8), dtype=np.uint16)
    decode_cache = prefill_cache.copy()
    prefill_cache[1].reshape(64, 8)[:40] = logical[:40]
    decode_cache[1].reshape(64, 8)[:41] = logical[:41]
    for process_index in range(2):
        _write_dump(
            source / f"main_cache.postfwd.step0002.proc{process_index}.npz",
            process_index=process_index,
            step_index=2,
            sequence_length=40,
            cache_bits=prefill_cache,
            corrupt_replica=corrupt_replica and process_index == 0,
        )
        _write_dump(
            source / f"main_cache.postfwd.step0003.proc{process_index}.npz",
            process_index=process_index,
            step_index=3,
            sequence_length=41,
            cache_bits=decode_cache,
        )
    contract_hash, tensor_hash = _write_ingredients(
        tmp_path / "ingredients",
        selected_delta=selected_delta,
        current_delta=current_delta,
        padding_delta=padding_delta,
    )
    return LegacyMainCacheComparisonConfig(
        source_dump_dir=source,
        ingredients_dir=tmp_path / "ingredients",
        output_dir=tmp_path / "output",
        capture_code_hash="a" * 40,
        legacy_repository_pin="b" * 40,
        accepted_oracle_pin="c" * 40,
        ingredients_code_hash="d" * 40,
        ingredients_contract_sha256=contract_hash,
        ingredients_tensor_sha256=tensor_hash,
        run_tag="unit-main-cache",
        source_run_id=1,
        source_item_row_id=2,
        expected_process_count=2,
        expected_local_replication=2,
        expected_physical_replication=4,
        expected_mesh_model_size=4,
        expected_prefill_step=2,
        expected_decode_step=3,
        expected_last_chunk_tokens=2,
        expected_prompt_tokens=40,
        expected_current_position=40,
        expected_physical_pages=3,
        expected_logical_page_size=64,
        expected_cache_width=8,
        expected_payload_width=6,
        expected_selected_width=3,
        expected_owner_count=2,
    )


def test_classifies_exact_cache_before_attention_schedule(
        tmp_path: Path) -> None:
    manifest = compare_legacy_layer0_main_cache(_setup(tmp_path))
    assert manifest["classification"] == "cache_exact_attention_schedule_next"
    assert manifest["first_divergent_primitive"] is None
    assert manifest["performance_claim"] is False
    assert len(manifest["source_dump_files"]) == 4
    assert manifest["comparison"]["selected_prefill_cache_rows"][
        "elementwise_exact"] is True
    with np.load(tmp_path / "output" / "comparison.npz") as payload:
        assert payload["selected_positions"].tolist() == [1, 4, 33]
        np.testing.assert_array_equal(
            payload["legacy_selected_cache_bfloat16_bits"],
            payload["greenfield_selected_cache_bfloat16_bits"],
        )


def test_classifies_first_prefill_cache_difference(tmp_path: Path) -> None:
    manifest = compare_legacy_layer0_main_cache(
        _setup(tmp_path, selected_delta=True, current_delta=True))
    assert manifest["classification"] == "prefill_main_cache"
    comparison = manifest["comparison"]["selected_prefill_cache_rows"]
    assert comparison["first_mismatch_position"] == 4
    assert comparison["first_mismatch_dimension"] == 3
    assert comparison["mismatch_count"] == 1


def test_classifies_current_row_after_exact_prefill(tmp_path: Path) -> None:
    manifest = compare_legacy_layer0_main_cache(
        _setup(tmp_path, current_delta=True))
    assert manifest["classification"] == "recurrent_main_cache_producer"
    comparison = manifest["comparison"]["current_decode_cache_row"]
    assert comparison["first_mismatch_position"] == 40
    assert comparison["first_mismatch_dimension"] == 2


def test_rejects_disagreeing_legacy_replicas(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replicas disagree"):
        compare_legacy_layer0_main_cache(_setup(tmp_path,
                                                corrupt_replica=True))


def test_rejects_nonzero_configured_padding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="padding is nonzero"):
        compare_legacy_layer0_main_cache(_setup(tmp_path,
                                                padding_delta=True))


def test_comparison_cli_is_importable() -> None:
    script = (Path(__file__).resolve().parents[3] /
              "scripts/greenfield/compare_legacy_layer0_main_cache.py")
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--ingredients-contract-sha256" in result.stdout


def test_protected_launcher_pins_exact_steps_and_observer() -> None:
    root = Path(__file__).resolve().parents[3]
    launcher = (root /
                "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
                ).read_text()
    wrapper = (root /
               "scripts/greenfield/run_capture_legacy_layer0_main_cache.sh"
               ).read_text()
    assert "GLM_GREENFIELD_MAIN_CACHE_CAPTURE:-0" in launcher
    assert ("LEGACY_PIN=3443515d9d3c42412558b778c608aaf07c6c89ff" in launcher)
    assert "OBSERVER_COMMIT_DISTANCE=1" in launcher
    assert "GLM_DCP_CACHE_DUMP_LAYERS=1" in launcher
    assert "GLM_DCP_CACHE_DUMP_STEPS=4,5" in launcher
    assert "main_cache.postfwd.step0004.proc*.npz" in launcher
    assert "main_cache.postfwd.step0005.proc*.npz" in launcher
    assert 'main_cache_source_count -eq 16' in launcher
    assert "MAIN_CACHE_INGREDIENTS_CONTRACT_SHA=" in launcher
    assert "MAIN_CACHE_INGREDIENTS_TENSOR_SHA=" in launcher
    assert "compare_legacy_layer0_main_cache.py" in launcher
    assert "layer0_main_cache_dsa_event_tensors_exact" in launcher
    assert "GLM_GREENFIELD_MAIN_CACHE_CAPTURE=1" in wrapper
    assert "gs://driftbench-dsv4-uc/oracles/greenfield" in wrapper
