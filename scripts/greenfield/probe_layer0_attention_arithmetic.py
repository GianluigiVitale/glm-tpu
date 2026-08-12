#!/usr/bin/env python3
"""Discriminate layer-0 sparse-attention block and head association on LP4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import ml_dtypes
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_POSITION = 8155
_TABLE_SHA256 = "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
_ACCEPTED_CACHE_BITS_SHA256 = (
    "73298b7d593ce8f29e72bebaf3084ee836ce21135262d404c33d4ab205171be2"
)
_ACCEPTED_CACHE_MANIFEST_SHA256 = (
    "fb47b2e3d42491168f79bfbebe79053b21dbb00364eec689a6bb433a1b7869c9"
)
_WEIGHT_NAMES = (
    "attention.slot_00.qkv_a.weight_bits",
    "attention.slot_00.qkv_a.scale_inv",
    "attention.slot_00.q_a_norm",
    "attention.slot_00.q_b.weight_bits",
    "attention.slot_00.q_b.scale_inv",
    "attention.slot_00.kv_b.weight_bits",
    "attention.slot_00.kv_b.scale_inv",
)
_WEIGHT_CONTRACT = {
    "attention.slot_00.qkv_a.weight_bits": ((32, 6144, 82), "uint8"),
    "attention.slot_00.qkv_a.scale_inv": ((32, 48, 82), "float32"),
    "attention.slot_00.q_a_norm": ((2048,), "bfloat16"),
    "attention.slot_00.q_b.weight_bits": ((4096, 2048), "uint8"),
    "attention.slot_00.q_b.scale_inv": ((32, 16), "float32"),
    "attention.slot_00.kv_b.weight_bits": ((7168, 512), "uint8"),
    "attention.slot_00.kv_b.scale_inv": ((56, 4), "float32"),
}


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    segment_block: int
    attention_heads: int
    split_projection_heads: int


_ARMS = (
    Arm("pregathered_h16_b128", 128, 16, 16),
    Arm("pregathered_h16_b512", 512, 16, 16),
    Arm("pregathered_attention_h2_b128", 128, 2, 16),
    Arm("pregathered_attention_h2_b512", 512, 2, 16),
    Arm("pregathered_full_h2_b512", 512, 2, 2),
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _compare_bits(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    if expected.shape != observed.shape or (
        expected.dtype != np.uint16 or observed.dtype != np.uint16
    ):
        raise ValueError("attention arithmetic comparison contract drifted")
    expected_f32 = expected.view(ml_dtypes.bfloat16).astype(np.float32)
    observed_f32 = observed.view(ml_dtypes.bfloat16).astype(np.float32)
    delta = observed_f32 - expected_f32
    absolute = np.abs(delta)
    per_head = np.count_nonzero(expected != observed, axis=1)
    return {
        "elementwise_exact": bool(np.array_equal(expected, observed)),
        "expected_sha256": _array_sha256(expected),
        "first_mismatch_index": (
            int(np.flatnonzero(expected != observed)[0])
            if np.any(expected != observed)
            else None
        ),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "mean_abs_error": float(absolute.mean()),
        "mean_signed_error": float(delta.mean()),
        "mismatch_count": int(np.count_nonzero(expected != observed)),
        "mismatching_head_count": int(np.count_nonzero(per_head)),
        "observed_sha256": _array_sha256(observed),
        "p99_abs_error": float(np.percentile(absolute, 99)),
        "per_head_mismatch_count": per_head.tolist(),
        "shape": list(expected.shape),
    }


def _load_segment(
    ingredients_path: Path,
    *,
    expected_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if _file_sha256(ingredients_path) != expected_sha256:
        raise RuntimeError("table-on ingredient tensor SHA-256 drifted")
    with np.load(ingredients_path, allow_pickle=False) as payload:
        required = {
            "selected_positions",
            "selected_valid_counts",
            "normalized_input_bfloat16_bits",
            "owner_selected_positions",
            "owner_selected_valid_counts",
            "owner_selected_cache_values_bfloat16_bits",
            "owner_selected_cache_valid",
            "combined_attention_output_bfloat16_bits",
            "combined_attention_valid",
            "contract_valid",
        }
        if not required.issubset(payload.files):
            raise RuntimeError("table-on ingredients lack attention probe fields")
        selected = np.ascontiguousarray(payload["selected_positions"])
        counts = np.ascontiguousarray(payload["selected_valid_counts"])
        normalized_bits = np.ascontiguousarray(
            payload["normalized_input_bfloat16_bits"]
        )
        owner_positions = np.ascontiguousarray(
            payload["owner_selected_positions"]
        )
        owner_counts = np.ascontiguousarray(
            payload["owner_selected_valid_counts"]
        )
        owner_cache_bits = np.ascontiguousarray(
            payload["owner_selected_cache_values_bfloat16_bits"]
        )
        owner_valid = np.ascontiguousarray(
            payload["owner_selected_cache_valid"]
        )
        greenfield_bits = np.ascontiguousarray(
            payload["combined_attention_output_bfloat16_bits"]
        )
        combined_valid = np.ascontiguousarray(
            payload["combined_attention_valid"]
        )
        contract_valid = np.ascontiguousarray(payload["contract_valid"])
    if (
        selected.shape != (4, 2048)
        or counts.shape != (4, 1)
        or normalized_bits.shape != (4, 6144)
        or normalized_bits.dtype != np.uint16
        or owner_positions.shape != (4, 2048)
        or owner_counts.shape != (4, 1)
        or owner_cache_bits.shape != (4, 2048, 640)
        or owner_cache_bits.dtype != np.uint16
        or greenfield_bits.shape != (4, 64, 512)
        or greenfield_bits.dtype != np.uint16
    ):
        raise RuntimeError("table-on ingredient shape/dtype contract drifted")
    if not (
        np.all(selected == selected[0])
        and np.all(counts == counts[0])
        and int(counts[0, 0]) == 2048
        and np.all(normalized_bits == normalized_bits[0])
        and np.all(greenfield_bits == greenfield_bits[0])
        and bool(owner_valid.all())
        and bool(combined_valid.all())
        and bool(contract_valid.all())
    ):
        raise RuntimeError("table-on ingredient replication/health contract failed")

    live_positions = []
    live_cache = []
    owner_count_values = []
    for owner in range(4):
        count = int(owner_counts[owner, 0])
        positions = owner_positions[owner, :count]
        values = owner_cache_bits[owner, :count]
        if (
            count <= 0
            or np.any(positions < 0)
            or np.any(np.diff(positions) <= 0)
            or np.any((positions % 512) // 128 != owner)
            or np.any(owner_positions[owner, count:] != -1)
        ):
            raise RuntimeError(f"owner {owner} selected-cache contract failed")
        live_positions.append(positions)
        live_cache.append(values)
        owner_count_values.append(count)
    joined_positions = np.concatenate(live_positions)
    joined_cache = np.concatenate(live_cache)
    order = np.argsort(joined_positions, kind="stable")
    ordered_positions = np.ascontiguousarray(joined_positions[order], dtype=np.int32)
    ordered_cache_bits = np.ascontiguousarray(joined_cache[order], dtype=np.uint16)
    expected_positions = np.sort(selected[0], kind="stable")
    if (
        ordered_positions.shape != (2048,)
        or np.any(np.diff(ordered_positions) <= 0)
        or not np.array_equal(ordered_positions, expected_positions)
    ):
        raise RuntimeError("owner selected-cache union differs from exact DSA set")
    segment = ordered_cache_bits.view(ml_dtypes.bfloat16)[None, ...]
    normalized = normalized_bits[0].view(ml_dtypes.bfloat16)[None, ...]
    return (
        np.ascontiguousarray(normalized),
        np.ascontiguousarray(segment),
        np.ascontiguousarray(greenfield_bits[0]),
        ordered_positions,
        {
            "owner_counts": owner_count_values,
            "ordered_positions_sha256": _array_sha256(ordered_positions),
            "segment_bfloat16_bits_sha256": _array_sha256(ordered_cache_bits),
        },
    )


def _load_accepted_cache_segment(
    tensor_path: Path,
    manifest_path: Path,
    *,
    tensor_sha256: str,
    manifest_sha256: str,
    ordered_positions: np.ndarray,
    greenfield_cache_bits: np.ndarray,
    owner_counts: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load DB530's accepted rows and put them in production attention order."""

    if _file_sha256(tensor_path) != tensor_sha256:
        raise RuntimeError("accepted main-cache tensor SHA-256 drifted")
    if _file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("accepted main-cache manifest SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text())
    selected_comparison = manifest.get("comparison", {}).get(
        "selected_prefill_cache_rows", {}
    )
    if (
        manifest.get("artifact_kind")
        != "glm52_legacy_pp8_layer0_main_cache_comparison"
        or manifest.get("classification") != "prefill_main_cache"
        or manifest.get("first_divergent_primitive")
        != "selected_prefill_cache_rows"
        or manifest.get("manifest_sha256") != _ACCEPTED_CACHE_MANIFEST_SHA256
        or manifest.get("legacy", {}).get("source_run_id") != 530
        or manifest.get("legacy", {}).get("source_item_row_id") != 1815
        or manifest.get("numerical_contract", {}).get("current_position")
        != _POSITION
        or selected_comparison.get("expected_bfloat16_sha256")
        != _ACCEPTED_CACHE_BITS_SHA256
        or selected_comparison.get("shape") != [2048, 640]
    ):
        raise RuntimeError("accepted main-cache manifest contract drifted")
    with np.load(tensor_path, allow_pickle=False) as payload:
        required = {
            "selected_positions",
            "owner_selected_counts",
            "legacy_selected_cache_bfloat16_bits",
            "greenfield_selected_cache_bfloat16_bits",
            "legacy_current_cache_bfloat16_bits",
            "greenfield_current_cache_bfloat16_bits",
            "prefill_live_block_table",
            "decode_live_block_table",
        }
        if set(payload.files) != required:
            raise RuntimeError("accepted main-cache tensor keys drifted")
        selected_positions = np.ascontiguousarray(payload["selected_positions"])
        accepted_bits = np.ascontiguousarray(
            payload["legacy_selected_cache_bfloat16_bits"]
        )
        observed_owner_counts = np.ascontiguousarray(
            payload["owner_selected_counts"]
        )
    if (
        selected_positions.shape != (2048,)
        or selected_positions.dtype != np.int32
        or np.any(selected_positions < 0)
        or np.unique(selected_positions).size != 2048
        or accepted_bits.shape != (2048, 640)
        or accepted_bits.dtype != np.uint16
        or observed_owner_counts.shape != (4,)
        or observed_owner_counts.dtype != np.int32
        or observed_owner_counts.tolist() != owner_counts
        or _array_sha256(accepted_bits) != _ACCEPTED_CACHE_BITS_SHA256
    ):
        raise RuntimeError("accepted main-cache tensor contract drifted")
    order = np.argsort(selected_positions, kind="stable")
    sorted_positions = np.ascontiguousarray(selected_positions[order])
    sorted_accepted_bits = np.ascontiguousarray(accepted_bits[order])
    if not np.array_equal(sorted_positions, ordered_positions):
        raise RuntimeError("accepted and greenfield DSA position sets differ")
    if greenfield_cache_bits.shape != (2048, 640):
        raise RuntimeError("greenfield selected-cache segment shape drifted")
    mismatch_indices = np.argwhere(sorted_accepted_bits != greenfield_cache_bits)
    if (
        mismatch_indices.shape != (1, 2)
        or mismatch_indices.tolist() != [[2046, 367]]
        or int(sorted_positions[2046]) != 8145
    ):
        raise RuntimeError(
            "table-on selected-cache residue differs from the sealed DB530 recount"
        )
    segment = sorted_accepted_bits.view(ml_dtypes.bfloat16)[None, ...]
    return np.ascontiguousarray(segment), {
        "accepted_score_order_bfloat16_bits_sha256": _array_sha256(accepted_bits),
        "accepted_sorted_bfloat16_bits_sha256": _array_sha256(
            sorted_accepted_bits
        ),
        "greenfield_sorted_bfloat16_bits_sha256": _array_sha256(
            greenfield_cache_bits
        ),
        "greenfield_residue": {
            "column": 367,
            "mismatch_count": 1,
            "position": 8145,
            "sorted_row": 2046,
        },
        "manifest_file_sha256": manifest_sha256,
        "manifest_sha256": _ACCEPTED_CACHE_MANIFEST_SHA256,
        "ordered_positions_sha256": _array_sha256(sorted_positions),
        "tensor_file_sha256": tensor_sha256,
    }


def _load_accepted(
    path: Path, *, expected_sha256: str
) -> np.ndarray:
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError("accepted attention projection NPZ SHA-256 drifted")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {
            "attended_latent_bfloat16_bits",
            "attention_output_bfloat16_bits",
        }:
            raise RuntimeError("accepted attention projection keys drifted")
        latent = np.ascontiguousarray(
            payload["attended_latent_bfloat16_bits"]
        )
    if latent.shape != (64, 512) or latent.dtype != np.uint16:
        raise RuntimeError("accepted attended latent contract drifted")
    return latent


def _load_q_a_reference(path: Path, *, expected_sha256: str) -> np.ndarray:
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError("protected q-a reference NPZ SHA-256 drifted")
    with np.load(path, allow_pickle=False) as payload:
        if "accepted_q_a_bfloat16_bits" not in payload.files:
            raise RuntimeError("protected q-a reference tensor is missing")
        q_a = np.ascontiguousarray(payload["accepted_q_a_bfloat16_bits"])
    if q_a.shape != (2048,) or q_a.dtype != np.uint16:
        raise RuntimeError("protected q-a reference contract drifted")
    return q_a


def _load_weights(
    checkpoint_root: Path, *, manifest_sha256: str
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from safetensors import safe_open

    manifest_path = checkpoint_root / "runtime_manifest.json"
    if _file_sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("checkpoint runtime manifest SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("artifact_kind")
        != "greenfield_feature_runtime_packed_checkpoint"
        or manifest.get("plan_id") != "PP8_LP4"
        or manifest.get("model_id") != "zai-org/GLM-5.2-FP8"
        or manifest.get("file_count") != 32
        or len(manifest.get("files", ())) != 32
    ):
        raise RuntimeError("checkpoint runtime manifest contract drifted")
    manifest_files = {
        record.get("destination_filename"): record
        for record in manifest["files"]
    }
    if len(manifest_files) != 32 or None in manifest_files:
        raise RuntimeError("checkpoint runtime manifest filenames are not unique")
    by_name: dict[str, list[np.ndarray]] = {name: [] for name in _WEIGHT_NAMES}
    records: list[dict[str, Any]] = []
    for slot in range(4):
        relative = Path(
            f"base_decoder_runtime_feature/stage_00/device_slot_{slot:02d}.safetensors"
        )
        tensor_path = checkpoint_root / relative
        evidence_path = checkpoint_root / "evidence" / relative.with_suffix(
            ".safetensors.json"
        )
        manifest_file = manifest_files.get(relative.as_posix())
        if manifest_file is None:
            raise RuntimeError(f"stage-0 slot {slot} is absent from runtime manifest")
        header_bytes = manifest_file.get("header_bytes")
        if (
            manifest_file.get("stage_id") != 0
            or manifest_file.get("device_slot") != slot
            or not isinstance(header_bytes, int)
            or header_bytes <= 8
            or tensor_path.stat().st_size != manifest_file.get("file_bytes")
        ):
            raise RuntimeError(f"stage-0 slot {slot} file metadata drifted")
        with tensor_path.open("rb") as stream:
            observed_header_sha256 = sha256(stream.read(header_bytes)).hexdigest()
        if observed_header_sha256 != manifest_file.get("header_sha256"):
            raise RuntimeError(f"stage-0 slot {slot} safetensors header drifted")
        evidence = json.loads(evidence_path.read_text())
        if evidence != manifest_file:
            raise RuntimeError(
                f"stage-0 slot {slot} evidence differs from runtime manifest"
            )
        evidence_by_name = {item["name"]: item for item in evidence["tensors"]}
        if len(evidence_by_name) != len(evidence["tensors"]):
            raise RuntimeError(f"stage-0 slot {slot} tensor names are not unique")
        if not set(_WEIGHT_NAMES).issubset(evidence_by_name):
            raise RuntimeError(f"stage-0 slot {slot} manifest lacks attention weights")
        with safe_open(tensor_path, framework="np") as handle:
            if not set(_WEIGHT_NAMES).issubset(handle.keys()):
                raise RuntimeError(f"stage-0 slot {slot} lacks attention weights")
            slot_records = []
            for name in _WEIGHT_NAMES:
                value = np.ascontiguousarray(handle.get_tensor(name))
                observed = _array_sha256(value)
                manifest_tensor = evidence_by_name[name]
                expected_shape, expected_dtype = _WEIGHT_CONTRACT[name]
                if (
                    value.shape != expected_shape
                    or str(value.dtype) != expected_dtype
                    or value.nbytes != manifest_tensor.get("byte_count")
                    or manifest_tensor.get("padding") is not False
                    or observed != manifest_tensor.get("sha256")
                ):
                    raise RuntimeError(
                        f"stage-0 tensor manifest contract drifted: slot={slot} {name}"
                    )
                by_name[name].append(value)
                slot_records.append(
                    {
                        "name": name,
                        "sha256": observed,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
        records.append(
            {
                "device_slot": slot,
                "checkpoint_file_sha256": manifest_file["sha256"],
                "destination_filename": manifest_file["destination_filename"],
                "evidence_file_sha256": _file_sha256(evidence_path),
                "filename": str(relative),
                "header_sha256": observed_header_sha256,
                "tensors": slot_records,
            }
        )
    replicated = (
        "attention.slot_00.qkv_a.weight_bits",
        "attention.slot_00.qkv_a.scale_inv",
        "attention.slot_00.q_a_norm",
    )
    for name in replicated:
        if any(not np.array_equal(by_name[name][0], value) for value in by_name[name][1:]):
            raise RuntimeError(f"replicated stage-0 tensor differs across slots: {name}")
    return (
        {
            "qkv_a_bits": by_name[replicated[0]][0],
            "qkv_a_scale": by_name[replicated[1]][0],
            "q_a_norm": by_name[replicated[2]][0],
            "q_b_bits": np.stack(by_name["attention.slot_00.q_b.weight_bits"]),
            "q_b_scale": np.stack(by_name["attention.slot_00.q_b.scale_inv"]),
            "kv_b_bits": np.stack(by_name["attention.slot_00.kv_b.weight_bits"]),
            "kv_b_scale": np.stack(by_name["attention.slot_00.kv_b.scale_inv"]),
        },
        records,
    )


def _pallas_call_lines(optimized_hlo: str) -> list[str]:
    return [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
        and 'custom_call_target="tpu_custom_call"' in line
    ]


def _validate_hlo(optimized_hlo: str, arm: Arm) -> dict[str, Any]:
    from glm_tpu.greenfield.sharding.hlo_contract import (
        COLLECTIVE_OPCODES,
        parse_hlo_module,
    )

    q_output = 4096 if arm.split_projection_heads == 16 else 512
    expected = {
        f"greenfield_fp8_block_matmul_m8_k2048_n{q_output}": (
            1 if arm.split_projection_heads == 16 else 8
        ),
        "greenfield_fp8_structured_kv_b_q_absorb_"
        f"h{arm.split_projection_heads}_p192_l512": (
            1 if arm.split_projection_heads == 16 else 8
        ),
        "greenfield_pregathered_sparse_mla_"
        f"h{arm.attention_heads}_k2048_b{arm.segment_block}_w640": (
            16 // arm.attention_heads
        ),
    }
    calls = _pallas_call_lines(optimized_hlo)
    counts = {
        name: sum(name in line for line in calls) for name in expected
    }
    unexpected = [
        line for line in calls if not any(name in line for name in expected)
    ]
    forbidden_markers = [
        token
        for token in (
            " all-gather(",
            " all-reduce(",
            " all-to-all(",
            " collective-permute(",
            " reduce-scatter(",
            "host_callback",
            "xla_python_cpu_callback",
            " outfeed(",
        )
        if token in optimized_hlo
    ]
    module = parse_hlo_module(optimized_hlo)
    forbidden_collectives = [
        {
            "name": instruction.name,
            "opcode": instruction.opcode,
            "raw_opcode": instruction.raw_opcode,
        }
        for instruction in module.instructions
        if instruction.raw_opcode in COLLECTIVE_OPCODES
        or any(
            instruction.raw_opcode == f"{opcode}-{suffix}"
            for opcode in COLLECTIVE_OPCODES
            for suffix in ("start", "done")
        )
    ]
    forbidden_dead_rows = [
        shape
        for shape in (
            "s32[32,2048]",
            "bf16[32,2048,640]",
            "bf16[32,64,512]",
        )
        if shape in optimized_hlo
    ]
    violations = [
        f"expected {wanted} {name} calls, found {counts[name]}"
        for name, wanted in expected.items()
        if counts[name] != wanted
    ]
    if unexpected:
        violations.append(f"unexpected TPU custom calls: {unexpected}")
    if forbidden_markers:
        violations.append(f"forbidden HLO markers: {forbidden_markers}")
    if forbidden_collectives:
        violations.append(f"forbidden HLO collectives: {forbidden_collectives}")
    if forbidden_dead_rows:
        violations.append(f"forbidden dead-row shapes: {forbidden_dead_rows}")
    one_live_row = "bf16[1,16,512]" in optimized_hlo
    if not one_live_row:
        violations.append("local attended-latent root lacks one live row")
    return {
        "expected_pallas_call_counts": expected,
        "observed_pallas_call_counts": counts,
        "pallas_custom_call_count": len(calls),
        "unexpected_pallas_custom_calls": unexpected,
        "forbidden_collectives": forbidden_collectives,
        "forbidden_markers": forbidden_markers,
        "forbidden_dead_rows": forbidden_dead_rows,
        "one_live_row": one_live_row,
        "passed": not violations,
        "violations": violations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--ingredients", type=Path, required=True)
    parser.add_argument("--ingredients-sha256", required=True)
    parser.add_argument("--ingredients-contract", type=Path, required=True)
    parser.add_argument("--ingredients-contract-sha256", required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--accepted-sha256", required=True)
    parser.add_argument("--accepted-capture", type=Path, required=True)
    parser.add_argument("--accepted-capture-sha256", required=True)
    parser.add_argument("--accepted-main-cache", type=Path, required=True)
    parser.add_argument("--accepted-main-cache-sha256", required=True)
    parser.add_argument("--accepted-main-cache-manifest", type=Path, required=True)
    parser.add_argument("--accepted-main-cache-manifest-sha256", required=True)
    parser.add_argument("--q-a-reference", type=Path, required=True)
    parser.add_argument("--q-a-reference-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    for path, expected, label in (
        (
            args.ingredients_contract,
            args.ingredients_contract_sha256,
            "ingredient contract",
        ),
        (args.accepted_capture, args.accepted_capture_sha256, "accepted capture"),
        (
            args.accepted_main_cache_manifest,
            args.accepted_main_cache_manifest_sha256,
            "accepted main-cache manifest",
        ),
    ):
        if _file_sha256(path) != expected:
            raise RuntimeError(f"{label} SHA-256 drifted")
    ingredient_contract = json.loads(args.ingredients_contract.read_text())
    accepted_capture = json.loads(args.accepted_capture.read_text())
    if (
        ingredient_contract.get("main_rope_table_sha256") != _TABLE_SHA256
        or ingredient_contract.get("position") != _POSITION
        or accepted_capture.get("position") != _POSITION
        or accepted_capture.get("capture_mode") != "attention_projection"
        or accepted_capture.get("tensors", {})
        .get("attended_latent_bfloat16_bits", {})
        .get("shape")
        != [64, 512]
    ):
        raise RuntimeError("attention arithmetic source contract drifted")

    (
        normalized,
        greenfield_segment,
        greenfield_bits,
        ordered_positions,
        segment_record,
    ) = _load_segment(
        args.ingredients, expected_sha256=args.ingredients_sha256
    )
    accepted_segment, accepted_cache_record = _load_accepted_cache_segment(
        args.accepted_main_cache,
        args.accepted_main_cache_manifest,
        tensor_sha256=args.accepted_main_cache_sha256,
        manifest_sha256=args.accepted_main_cache_manifest_sha256,
        ordered_positions=ordered_positions,
        greenfield_cache_bits=np.ascontiguousarray(
            greenfield_segment.view(np.uint16)[0]
        ),
        owner_counts=segment_record["owner_counts"],
    )
    accepted_bits = _load_accepted(
        args.accepted, expected_sha256=args.accepted_sha256
    )
    accepted_q_a_bits = _load_q_a_reference(
        args.q_a_reference, expected_sha256=args.q_a_reference_sha256
    )
    weights, weight_records = _load_weights(
        args.checkpoint_root,
        manifest_sha256=args.checkpoint_manifest_sha256,
    )

    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.pallas import (
        Fp8BlockMatmulConfig,
        SparseMlaConfig,
        fp8_block_matmul,
        fp8_structured_kv_b_q_absorb,
        pregathered_sparse_mla_pallas,
    )
    from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract
    from glm_tpu.greenfield.kernels.reference.qkv_a import (
        FusedQkvAContract,
        one_row_fused_qkv_a_convolution,
    )
    from glm_tpu.greenfield.kernels.reference.rotary import (
        apply_rotary_fp32_final_round,
        build_rotary_table_host,
        rotary_table_sha256,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError("attention arithmetic probe requires one four-chip TPU host")
    mesh = Mesh(np.asarray(jax.local_devices()), ("lp4",))
    replicated = NamedSharding(mesh, P())
    slot_sharding = NamedSharding(mesh, P("lp4", None, None))
    table = build_rotary_table_host(
        8192, rotary_dim=64, theta=8_000_000.0
    )
    if rotary_table_sha256(table) != _TABLE_SHA256:
        raise RuntimeError("main-RoPE table identity drifted")
    rope_row = np.ascontiguousarray(table[_POSITION])

    argument_prefix = (
        jax.device_put(normalized, replicated),
        jax.device_put(weights["qkv_a_bits"], replicated),
        jax.device_put(weights["qkv_a_scale"], replicated),
        jax.device_put(weights["q_a_norm"], replicated),
        jax.device_put(weights["q_b_bits"], slot_sharding),
        jax.device_put(weights["q_b_scale"], slot_sharding),
        jax.device_put(weights["kv_b_bits"], slot_sharding),
        jax.device_put(weights["kv_b_scale"], slot_sharding),
        jax.device_put(rope_row, replicated),
    )
    accepted_arguments = argument_prefix + (
        jax.device_put(accepted_segment, replicated),
    )
    greenfield_arguments = argument_prefix + (
        jax.device_put(greenfield_segment, replicated),
    )
    full_contract = MlaNumericalContract()
    qkv_contract = FusedQkvAContract()
    fp8_config = Fp8BlockMatmulConfig(
        block_shape=(128, 128), output_tile=128, contraction_tile=128
    )

    def project_q_a(
        normalized_value: Any,
        qkv_bits: Any,
        qkv_scale: Any,
        q_norm: Any,
    ) -> Any:
        return one_row_fused_qkv_a_convolution(
            normalized_value,
            qkv_bits,
            qkv_scale,
            q_norm,
            contract=qkv_contract,
        ).q_residual

    def project_heads(
        q_residual: Any,
        q_bits: Any,
        q_scale: Any,
        kv_bits: Any,
        kv_scale: Any,
        rope: Any,
        *,
        heads: int,
    ) -> tuple[Any, Any]:
        projected = fp8_block_matmul(
            q_residual,
            q_bits,
            q_scale,
            config=fp8_config,
        ).reshape(1, heads, full_contract.qk_head_dim)
        q_nope = projected[..., : full_contract.qk_nope_head_dim]
        unrotated = projected[..., full_contract.qk_nope_head_dim :]
        half = full_contract.qk_rope_head_dim // 2
        q_rope = apply_rotary_fp32_final_round(
            unrotated,
            rope[:half][None, None, :],
            rope[half:][None, None, :],
            interleaved=True,
        )
        q_absorbed = fp8_structured_kv_b_q_absorb(
            q_nope,
            kv_bits,
            kv_scale,
            config=fp8_config,
        )
        return q_absorbed, q_rope

    def make_local(arm: Arm) -> Any:
        attention_contract = replace(
            full_contract, num_heads=arm.attention_heads
        )
        attention_config = SparseMlaConfig(segment_block=arm.segment_block)

        def local(
            normalized_value: Any,
            qkv_bits: Any,
            qkv_scale: Any,
            q_norm: Any,
            q_bits_slot: Any,
            q_scale_slot: Any,
            kv_bits_slot: Any,
            kv_scale_slot: Any,
            rope: Any,
            selected_cache: Any,
        ) -> tuple[Any, Any, Any, Any]:
            q_residual = project_q_a(
                normalized_value, qkv_bits, qkv_scale, q_norm
            )
            q_bits = q_bits_slot[0]
            q_scale = q_scale_slot[0]
            kv_bits = kv_bits_slot[0]
            kv_scale = kv_scale_slot[0]
            outputs = []
            q_nope_parts = []
            q_rope_parts = []
            if arm.split_projection_heads == 16:
                q_nope_all, q_rope_all = project_heads(
                    q_residual,
                    q_bits,
                    q_scale,
                    kv_bits,
                    kv_scale,
                    rope,
                    heads=16,
                )
                for start in range(0, 16, arm.attention_heads):
                    q_nope = lax.slice_in_dim(
                        q_nope_all, start, start + arm.attention_heads, axis=1
                    )
                    q_rope = lax.slice_in_dim(
                        q_rope_all, start, start + arm.attention_heads, axis=1
                    )
                    outputs.append(
                        pregathered_sparse_mla_pallas(
                            q_nope,
                            q_rope,
                            selected_cache,
                            jnp.asarray([2048], dtype=jnp.int32),
                            contract=attention_contract,
                            config=attention_config,
                        )
                    )
                q_nope_parts.append(q_nope_all)
                q_rope_parts.append(q_rope_all)
            else:
                for group in range(8):
                    q_start = group * 512
                    kv_start = group * 896
                    q_nope, q_rope = project_heads(
                        q_residual,
                        lax.slice_in_dim(q_bits, q_start, q_start + 512, axis=0),
                        lax.slice_in_dim(q_scale, group * 4, group * 4 + 4, axis=0),
                        lax.slice_in_dim(kv_bits, kv_start, kv_start + 896, axis=0),
                        lax.slice_in_dim(kv_scale, group * 7, group * 7 + 7, axis=0),
                        rope,
                        heads=2,
                    )
                    outputs.append(
                        pregathered_sparse_mla_pallas(
                            q_nope,
                            q_rope,
                            selected_cache,
                            jnp.asarray([2048], dtype=jnp.int32),
                            contract=attention_contract,
                            config=attention_config,
                        )
                    )
                    q_nope_parts.append(q_nope)
                    q_rope_parts.append(q_rope)
            return (
                jnp.concatenate(outputs, axis=1),
                jnp.concatenate(q_nope_parts, axis=1),
                jnp.concatenate(q_rope_parts, axis=1),
                q_residual,
            )

        return local

    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    arm_records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_attended_latent_bfloat16_bits": accepted_bits,
        "accepted_q_a_bfloat16_bits": accepted_q_a_bits,
        "greenfield_attended_latent_bfloat16_bits": greenfield_bits,
        "accepted_selected_cache_bfloat16_bits": accepted_segment.view(np.uint16),
        "greenfield_selected_cache_bfloat16_bits": greenfield_segment.view(
            np.uint16
        ),
    }
    query_bits_by_arm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    exact_arms = []
    greenfield_cache_exact_arms = []
    for arm in _ARMS:
        mapped = jax.shard_map(
            make_local(arm),
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(),
                P("lp4", None, None),
                P("lp4", None, None),
                P("lp4", None, None),
                P("lp4", None, None),
                P(),
                P(),
            ),
            out_specs=(
                P("lp4", None, None),
                P("lp4", None, None),
                P("lp4", None, None),
                P(),
            ),
            check_vma=False,
        )
        lowered = jax.jit(mapped).lower(*accepted_arguments)
        stablehlo = lowered.as_text()
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
        hlo_contract = _validate_hlo(optimized_hlo, arm)
        if not hlo_contract["passed"]:
            raise RuntimeError(f"attention arithmetic HLO failed: {arm.name}")
        stable_path = args.hlo_dir / f"{arm.name}.stablehlo.mlir"
        optimized_path = args.hlo_dir / f"{arm.name}.optimized_hlo.txt"
        stable_path.write_text(stablehlo)
        optimized_path.write_text(optimized_hlo)
        output, q_nope, q_rope, q_a = compiled(*accepted_arguments)
        jax.block_until_ready((output, q_nope, q_rope, q_a))
        output_bits = np.ascontiguousarray(
            np.asarray(output).reshape(64, 512)
        ).view(np.uint16)
        q_nope_bits = np.ascontiguousarray(
            np.asarray(q_nope).reshape(64, 512)
        ).view(np.uint16)
        q_rope_bits = np.ascontiguousarray(
            np.asarray(q_rope).reshape(64, 64)
        ).view(np.uint16)
        q_a_bits = np.ascontiguousarray(
            np.asarray(q_a).reshape(2048)
        ).view(np.uint16)
        q_a_comparison = _compare_bits(
            accepted_q_a_bits.reshape(1, 2048), q_a_bits.reshape(1, 2048)
        )
        if not q_a_comparison["elementwise_exact"]:
            raise RuntimeError(f"protected q-a prerequisite failed: {arm.name}")
        accepted_comparison = _compare_bits(accepted_bits, output_bits)
        greenfield_capture_comparison = _compare_bits(greenfield_bits, output_bits)
        if accepted_comparison["elementwise_exact"]:
            exact_arms.append(arm.name)

        (
            greenfield_output,
            greenfield_q_nope,
            greenfield_q_rope,
            greenfield_q_a,
        ) = compiled(*greenfield_arguments)
        jax.block_until_ready(
            (
                greenfield_output,
                greenfield_q_nope,
                greenfield_q_rope,
                greenfield_q_a,
            )
        )
        greenfield_output_bits = np.ascontiguousarray(
            np.asarray(greenfield_output).reshape(64, 512)
        ).view(np.uint16)
        greenfield_q_nope_bits = np.ascontiguousarray(
            np.asarray(greenfield_q_nope).reshape(64, 512)
        ).view(np.uint16)
        greenfield_q_rope_bits = np.ascontiguousarray(
            np.asarray(greenfield_q_rope).reshape(64, 64)
        ).view(np.uint16)
        greenfield_q_a_bits = np.ascontiguousarray(
            np.asarray(greenfield_q_a).reshape(2048)
        ).view(np.uint16)
        query_input_independence = {
            "q_a": _compare_bits(
                q_a_bits.reshape(1, 2048), greenfield_q_a_bits.reshape(1, 2048)
            ),
            "q_nope": _compare_bits(q_nope_bits, greenfield_q_nope_bits),
            "q_rope": _compare_bits(q_rope_bits, greenfield_q_rope_bits),
        }
        if not all(
            item["elementwise_exact"] for item in query_input_independence.values()
        ):
            raise RuntimeError(
                f"query roots depend on selected-cache input: {arm.name}"
            )
        greenfield_cache_comparison = _compare_bits(
            accepted_bits, greenfield_output_bits
        )
        if greenfield_cache_comparison["elementwise_exact"]:
            greenfield_cache_exact_arms.append(arm.name)
        arm_records[arm.name] = {
            "accepted_comparison": accepted_comparison,
            "attention_heads_per_call": arm.attention_heads,
            "greenfield_cache_comparison": greenfield_cache_comparison,
            "greenfield_capture_comparison": greenfield_capture_comparison,
            "greenfield_cache_vs_greenfield_capture": _compare_bits(
                greenfield_bits, greenfield_output_bits
            ),
            "hlo": {
                "contract": hlo_contract,
                "optimized_sha256": sha256(optimized_hlo.encode()).hexdigest(),
                "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
            },
            "projection_heads_per_call": arm.split_projection_heads,
            "q_a_comparison": q_a_comparison,
            "query_input_independence": query_input_independence,
            "segment_block": arm.segment_block,
        }
        tensor_payload[
            f"attended_latent_bfloat16_bits__{arm.name}__accepted_cache"
        ] = output_bits
        tensor_payload[
            f"attended_latent_bfloat16_bits__{arm.name}__greenfield_cache"
        ] = greenfield_output_bits
        tensor_payload[f"q_nope_bfloat16_bits__{arm.name}"] = q_nope_bits
        tensor_payload[f"q_rope_bfloat16_bits__{arm.name}"] = q_rope_bits
        tensor_payload[f"q_a_bfloat16_bits__{arm.name}"] = q_a_bits
        query_bits_by_arm[arm.name] = (q_nope_bits, q_rope_bits)

    control_query = query_bits_by_arm["pregathered_h16_b512"]
    query_association = {}
    for name, (q_nope_bits, q_rope_bits) in query_bits_by_arm.items():
        query_association[name] = {
            "q_nope_vs_h16": _compare_bits(control_query[0], q_nope_bits),
            "q_rope_vs_h16": _compare_bits(control_query[1], q_rope_bits),
        }
    args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.tensor_output, **tensor_payload)
    record = {
        "artifact_kind": "glm52_layer0_attention_arithmetic_probe",
        "arms": arm_records,
        "backend": jax.default_backend(),
        "classification": (
            "exact_arithmetic_arm_with_upstream_cache_residue"
            if exact_arms and not greenfield_cache_exact_arms
            else "exact_arithmetic_arm_identified"
            if exact_arms
            else "main_query_or_kernel_arithmetic_unresolved"
        ),
        "code_hash": code_hash,
        "diagnostic_only": True,
        "exact_arms": exact_arms,
        "format_version": 2,
        "greenfield_cache_exact_arms": greenfield_cache_exact_arms,
        "ingredients": {
            "contract_sha256": args.ingredients_contract_sha256,
            "tensor_sha256": args.ingredients_sha256,
        },
        "model_id": "zai-org/GLM-5.2-FP8",
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "position": _POSITION,
        "query_association": query_association,
        "segment": {
            "accepted_cache": accepted_cache_record,
            "greenfield_cache": segment_record,
        },
        "source_capture": {
            "capture_sha256": args.accepted_capture_sha256,
            "tensor_sha256": args.accepted_sha256,
        },
        "q_a_reference": {
            "sha256": args.q_a_reference_sha256,
            "tensor_sha256": _array_sha256(accepted_q_a_bits),
        },
        "status": "SUCCESS",
        "tensor_output": {
            "filename": args.tensor_output.name,
            "sha256": _file_sha256(args.tensor_output),
        },
        "weight_records": weight_records,
    }
    _write_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
