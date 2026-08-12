from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_attention_arithmetic.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_attention_arithmetic_probe.sh"
SPEC = importlib.util.spec_from_file_location("attention_arithmetic_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _custom_call(name: str) -> str:
    return (
        f'%call = bf16[1] custom-call(), custom_call_target="tpu_custom_call", '
        f'backend_config={{"custom_call_config":{{"name":"{name}"}}}}'
    )


def test_attention_arithmetic_hlo_contract_pins_exact_full_h2_calls() -> None:
    arm = next(item for item in MODULE._ARMS if item.name == "pregathered_full_h2_b512")
    names = (
        "greenfield_fp8_block_matmul_m8_k2048_n512",
        "greenfield_fp8_structured_kv_b_q_absorb_h2_p192_l512",
        "greenfield_pregathered_sparse_mla_h2_k2048_b512_w640",
    )
    hlo = "\n".join(
        ["HloModule probe", "ENTRY main {", "bf16[1,16,512] parameter(0)"]
        + [_custom_call(name) for name in names for _ in range(8)]
        + ["}"]
    )
    record = MODULE._validate_hlo(hlo, arm)
    assert record["passed"]
    assert record["pallas_custom_call_count"] == 24

    extra = hlo + "\n" + _custom_call("unexpected_kernel")
    assert not MODULE._validate_hlo(extra, arm)["passed"]
    assert not MODULE._validate_hlo(hlo + "\n all-reduce(", arm)["passed"]
    for opcode in (
        "all-reduce-start",
        "all-gather-done",
        "collective-permute-start",
        "collective-broadcast-done",
    ):
        body, trailer = hlo.rsplit("}", 1)
        async_hlo = (
            body + f'%async = bf16[1] {opcode}(%call), channel_id=1\n}}' + trailer
        )
        record = MODULE._validate_hlo(async_hlo, arm)
        assert not record["passed"]
        assert record["forbidden_collectives"][0]["raw_opcode"] == opcode


def test_attention_arithmetic_segment_reconstruction_is_exact(tmp_path: Path) -> None:
    selected = np.arange(2048, dtype=np.int32)
    owner_positions = np.full((4, 2048), -1, dtype=np.int32)
    owner_counts = np.zeros((4, 1), dtype=np.int32)
    owner_cache = np.zeros((4, 2048, 640), dtype=np.uint16)
    expected_cache = np.zeros((2048, 640), dtype=np.uint16)
    for owner in range(4):
        positions = selected[((selected % 512) // 128) == owner]
        owner_counts[owner, 0] = positions.size
        owner_positions[owner, : positions.size] = positions
        values = np.repeat(positions[:, None].astype(np.uint16), 640, axis=1)
        owner_cache[owner, : positions.size] = values
        expected_cache[positions] = values
    ingredients = tmp_path / "ingredients.npz"
    np.savez(
        ingredients,
        selected_positions=np.repeat(selected[None, :], 4, axis=0),
        selected_valid_counts=np.full((4, 1), 2048, dtype=np.int32),
        normalized_input_bfloat16_bits=np.zeros((4, 6144), dtype=np.uint16),
        owner_selected_positions=owner_positions,
        owner_selected_valid_counts=owner_counts,
        owner_selected_cache_values_bfloat16_bits=owner_cache,
        owner_selected_cache_valid=np.ones((4, 1), dtype=bool),
        combined_attention_output_bfloat16_bits=np.zeros(
            (4, 64, 512), dtype=np.uint16
        ),
        combined_attention_valid=np.ones((4, 1), dtype=bool),
        contract_valid=np.ones((4, 1), dtype=bool),
    )
    normalized, segment, greenfield, ordered_positions, record = MODULE._load_segment(
        ingredients, expected_sha256=MODULE._file_sha256(ingredients)
    )
    assert normalized.shape == (1, 6144)
    assert segment.shape == (1, 2048, 640)
    np.testing.assert_array_equal(segment.view(np.uint16)[0], expected_cache)
    assert greenfield.shape == (64, 512)
    np.testing.assert_array_equal(ordered_positions, selected)
    assert record["owner_counts"] == [512, 512, 512, 512]


def test_attention_arithmetic_accepted_cache_is_sorted_and_pinned(
    tmp_path: Path, monkeypatch
) -> None:
    ordered_positions = np.arange(6099, 8147, dtype=np.int32)
    score_order = np.concatenate(
        (ordered_positions[1024:], ordered_positions[:1024])
    )
    accepted_score_order = np.repeat(score_order[:, None].astype(np.uint16), 640, axis=1)
    accepted_sorted = np.repeat(
        ordered_positions[:, None].astype(np.uint16), 640, axis=1
    )
    greenfield_sorted = accepted_sorted.copy()
    greenfield_sorted[2046, 367] ^= np.uint16(1)
    tensor = tmp_path / "comparison.npz"
    np.savez(
        tensor,
        selected_positions=score_order,
        owner_selected_counts=np.array([512, 516, 515, 505], dtype=np.int32),
        legacy_selected_cache_bfloat16_bits=accepted_score_order,
        greenfield_selected_cache_bfloat16_bits=np.zeros((2048, 640), np.uint16),
        legacy_current_cache_bfloat16_bits=np.zeros((1, 640), np.uint16),
        greenfield_current_cache_bfloat16_bits=np.zeros((1, 640), np.uint16),
        prefill_live_block_table=np.arange(16, dtype=np.int32),
        decode_live_block_table=np.arange(16, dtype=np.int32),
    )
    accepted_sha = MODULE._array_sha256(accepted_score_order)
    fake_manifest_sha = "f" * 64
    monkeypatch.setattr(MODULE, "_ACCEPTED_CACHE_BITS_SHA256", accepted_sha)
    monkeypatch.setattr(MODULE, "_ACCEPTED_CACHE_MANIFEST_SHA256", fake_manifest_sha)
    manifest = tmp_path / "comparison.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "artifact_kind": "glm52_legacy_pp8_layer0_main_cache_comparison",
                "classification": "prefill_main_cache",
                "first_divergent_primitive": "selected_prefill_cache_rows",
                "manifest_sha256": fake_manifest_sha,
                "legacy": {"source_run_id": 530, "source_item_row_id": 1815},
                "numerical_contract": {"current_position": 8155},
                "comparison": {
                    "selected_prefill_cache_rows": {
                        "expected_bfloat16_sha256": accepted_sha,
                        "shape": [2048, 640],
                    }
                },
            }
        )
    )
    segment, record = MODULE._load_accepted_cache_segment(
        tensor,
        manifest,
        tensor_sha256=MODULE._file_sha256(tensor),
        manifest_sha256=MODULE._file_sha256(manifest),
        ordered_positions=ordered_positions,
        greenfield_cache_bits=greenfield_sorted,
        owner_counts=[512, 516, 515, 505],
    )
    np.testing.assert_array_equal(segment.view(np.uint16)[0], accepted_sorted)
    assert record["greenfield_residue"]["mismatch_count"] == 1


def test_attention_arithmetic_bit_comparison_reports_per_head() -> None:
    expected = np.zeros((2, 4), dtype=np.uint16)
    observed = expected.copy()
    observed[0, 1] = np.uint16(1)
    record = MODULE._compare_bits(expected, observed)
    assert not record["elementwise_exact"]
    assert record["mismatch_count"] == 1
    assert record["mismatching_head_count"] == 1
    assert record["per_head_mismatch_count"] == [1, 0]


def test_attention_arithmetic_q_a_reference_is_hash_pinned(tmp_path: Path) -> None:
    path = tmp_path / "q_a.npz"
    expected = np.arange(2048, dtype=np.uint16)
    np.savez(path, accepted_q_a_bfloat16_bits=expected)
    actual = MODULE._load_q_a_reference(
        path, expected_sha256=MODULE._file_sha256(path)
    )
    np.testing.assert_array_equal(actual, expected)


def test_attention_arithmetic_source_contract_uses_sealed_decode_position() -> None:
    ingredient = {
        "decode_position": 8155,
        "main_rope_table_sha256": MODULE._TABLE_SHA256,
    }
    accepted = {
        "capture_mode": "attention_projection",
        "position": 8155,
        "tensors": {"attended_latent_bfloat16_bits": {"shape": [64, 512]}},
    }
    MODULE._validate_source_contract(ingredient, accepted)
    wrong_key = dict(ingredient)
    wrong_key["position"] = wrong_key.pop("decode_position")
    with pytest.raises(RuntimeError, match="source contract drifted"):
        MODULE._validate_source_contract(wrong_key, accepted)


def test_attention_arithmetic_checkpoint_evidence_must_equal_manifest(
    tmp_path: Path,
) -> None:
    files = []
    for slot in range(32):
        stage = slot // 4
        device_slot = slot % 4
        files.append(
            {
                "destination_filename": (
                    f"base_decoder_runtime_feature/stage_{stage:02d}/"
                    f"device_slot_{device_slot:02d}.safetensors"
                ),
                "device_slot": device_slot,
                "file_bytes": 9,
                "header_bytes": 9,
                "header_sha256": MODULE.sha256(b"123456789").hexdigest(),
                "sha256": f"{slot:064x}",
                "stage_id": stage,
                "tensors": [],
            }
        )
    manifest = {
        "artifact_kind": "greenfield_feature_runtime_packed_checkpoint",
        "file_count": 32,
        "files": files,
        "model_id": "zai-org/GLM-5.2-FP8",
        "plan_id": "PP8_LP4",
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    evidence_path = (
        tmp_path
        / "evidence/base_decoder_runtime_feature/stage_00/device_slot_00.safetensors.json"
    )
    evidence_path.parent.mkdir(parents=True)
    tensor_path = (
        tmp_path
        / "base_decoder_runtime_feature/stage_00/device_slot_00.safetensors"
    )
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"123456789")
    altered = dict(files[0])
    altered["sha256"] = "f" * 64
    evidence_path.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="evidence differs from runtime manifest"):
        MODULE._load_weights(
            tmp_path,
            manifest_sha256=MODULE._file_sha256(manifest_path),
        )


def test_attention_arithmetic_wrapper_pins_sources_and_protection() -> None:
    text = WRAPPER.read_text()
    for token in (
        "INGREDIENT_NPZ_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95",
        "ACCEPTED_NPZ_SHA=3a619a0985fbb9ba6e1be9347bbc745c120fa74553ce1de5de830190a29c0a30",
        "ACCEPTED_SUCCESS_SHA=88691576033fea623414034171441c213472ac26a3d5e7a04388cff434a0a47c",
        "ACCEPTED_CACHE_NPZ_SHA=a71213370a9f7a96998761af961ab379c91df1f286676f60b6bbab811efc0924",
        "ACCEPTED_CACHE_JSON_SHA=c06f919df5b0b8bac3eb5466e5bd2aa5b2bbc33ec19995e8a052967117e8b8d9",
        "ACCEPTED_CACHE_SUCCESS_SHA=7a46ae6574d1ae65018be9537ad198cdaca26b8e01bb94afbb61785f4ed7cd10",
        "CHECKPOINT_MANIFEST_SHA=de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134",
        "Q_A_NPZ_SHA=b371ad77313c085268470c950f231a0cf23c4d17c7c7104b9c791cfc9f49d247",
        "Q_A_SUCCESS_SHA=b3cff36beaff7eaaf71c30349007c8c5acb8c7423ae46fb82c40339061a23b11",
        ".glm_pod_workload.lock",
        "strict_census pre",
        "strict_census post",
        "results_ckpt.db",
        "remote_objects.json",
        "performance_claim",
        "--expected-code-hash \"$PIN\"",
        "--accepted-main-cache \"$ACCEPTED_CACHE_NPZ\"",
        "--accepted-main-cache-manifest \"$ACCEPTED_CACHE_JSON\"",
        "--checkpoint-manifest-sha256 \"$CHECKPOINT_MANIFEST_SHA\"",
        "google_crc32c",
        "remote object ledger checksum mismatch",
        "append-only remote prefix already contains objects",
        "correct=bool(runner[\"exact_arms\"])",
        "greenfield_run_tag",
        "rollback_provisional_db",
        "DELETE FROM summary WHERE run_id = ?",
        "terminal_success_done=1",
    ):
        assert token in text
    assert text.index("strict_census post") < text.index("pv.start_run")
    assert text.index("terminal_success_done=1") > text.index(
        "remote SUCCESS checksum mismatch"
    )
