#!/usr/bin/env python3
"""Classify the new LP2 K-half N82 arm without touching TPU resources."""

from __future__ import annotations

import argparse
import json
import os
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

RUNTIME_MANIFEST_FILE_SHA256 = (
    "e13ccefb7341756cd68d85e51209eaa8516ea506eac7ace3b4fd0a1d56828032"
)
RUNTIME_MANIFEST_SELF_SHA256 = (
    "b385458f233f21342855ac4c3373429c034a9e40bd85d638b16466199ff66bab"
)
N82_WEIGHT_SHA256 = (
    "6e8b4efd28df17b493b1348a6ac82a05a468379e930a575a917636e1d506855d"
)
N82_SCALE_SHA256 = (
    "3ca2712f5387e086f4251b5c1eea46b4bc2744911ca214c0ab9c7a3d70711cc5"
)
Q_A_NORM_SHA256 = (
    "d436eddb448639bcef313d66d3dac97f8cfd676215bacfa0dc5786701cef7b2f"
)
ACCEPTED_ORACLE_FILE_SHA256 = (
    "79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054"
)
ACCEPTED_NORMALIZED_SHA256 = (
    "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039"
)
ACCEPTED_Q_A_SHA256 = (
    "8e3dc61e24591cef1979e10b4a79613505f92f45df1e25e4fde844fc61255d85"
)
DB518_FILE_SHA256 = (
    "534bacc54d74992f5a8ab4d422f9fa0947523d59325b4bfa272d4fbeb56262f0"
)
FEATURE2_PROGRAM_SHA256 = (
    "cd06938089f186917913de58ac0f6b285009cdb6304a0d2a51fcb498bc12ce43"
)
EXPECTED_CPU = {
    "accepted_full_q_a_sha256": ACCEPTED_Q_A_SHA256,
    "accepted_f32_partial_q_a_sha256": ACCEPTED_Q_A_SHA256,
    "accepted_bf16_partial_q_a_sha256": (
        "5e4ede6ad99d1ecc374f2d9f06efc6d6a7401734ec7f796f0f319286a4dcfd61"
    ),
    "accepted_bf16_partial_mismatches": 656,
    "current_full_q_a_sha256": (
        "c488a3f95ecb476adcc2ea8d0d8db1b6a88944fcfaccb9a85042a3a34b06368c"
    ),
    "current_f32_partial_q_a_sha256": (
        "c488a3f95ecb476adcc2ea8d0d8db1b6a88944fcfaccb9a85042a3a34b06368c"
    ),
    "current_f32_partial_vs_accepted_mismatches": 46,
    "current_f32_partial_vs_captured_mismatches": 1,
}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _read_safetensors_values(path: Path) -> dict[str, np.ndarray]:
    names = {
        "attention.slot_01.qkv_a.weight_bits": np.dtype(np.uint8),
        "attention.slot_01.qkv_a.scale_inv": np.dtype("<f4"),
        "attention.slot_01.q_a_norm": np.dtype(ml_dtypes.bfloat16),
    }
    with path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_bytes))
        data_start = 8 + header_bytes
        result = {}
        for name, dtype in names.items():
            record = header.get(name)
            if not isinstance(record, dict):
                raise TypeError(f"runtime owner lacks {name}")
            start, end = record["data_offsets"]
            handle.seek(data_start + int(start))
            payload = handle.read(int(end) - int(start))
            value = np.frombuffer(payload, dtype=dtype).reshape(
                tuple(int(item) for item in record["shape"])
            )
            result[name] = np.ascontiguousarray(value)
    expected = {
        "attention.slot_01.qkv_a.weight_bits": N82_WEIGHT_SHA256,
        "attention.slot_01.qkv_a.scale_inv": N82_SCALE_SHA256,
        "attention.slot_01.q_a_norm": Q_A_NORM_SHA256,
    }
    for name, expected_sha in expected.items():
        if _array_sha256(result[name]) != expected_sha:
            raise RuntimeError(f"runtime owner {name} SHA-256 drifted")
    return result


def _project_bf16_partials(
    hidden: Any, weight_bits: Any, scale: Any, norm_weight: Any
) -> Any:
    """Rejected control: round both K-half partials before their sum."""

    import jax.numpy as jnp
    from jax import lax

    def project(weight_scale: tuple[Any, Any]) -> Any:
        bits, one_scale = weight_scale
        weight = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
        expanded = jnp.repeat(one_scale, 128, axis=0)[:6144]
        decoded = (
            weight.astype(jnp.float32) * expanded.astype(jnp.float32)
        ).astype(jnp.bfloat16)

        def convolution(left: Any, right: Any) -> Any:
            return lax.conv_general_dilated(
                left,
                right,
                window_strides=(),
                padding=(),
                dimension_numbers=("NC", "IO", "NC"),
                preferred_element_type=jnp.float32,
            )

        first = convolution(hidden[:, :3072], decoded[:3072])
        second = convolution(hidden[:, 3072:], decoded[3072:])
        return (
            first.astype(jnp.bfloat16).astype(jnp.float32)
            + second.astype(jnp.bfloat16).astype(jnp.float32)
        ).astype(jnp.bfloat16)

    projected = lax.map(project, (weight_bits, scale))
    local_q = projected[:, :, :64]
    square_sum = jnp.sum(jnp.square(local_q.astype(jnp.float32)), axis=(0, 2))
    inverse = lax.rsqrt(square_sum / jnp.float32(2048) + jnp.float32(1e-5))
    normalized = (
        local_q.astype(jnp.float32) * inverse[None, :, None]
    ).astype(jnp.bfloat16)
    logical = jnp.transpose(normalized, (1, 0, 2)).reshape(1, 2048)
    return (logical * norm_weight).astype(jnp.bfloat16)


def _bits(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value)).view(np.uint16).reshape(-1)


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    mismatches = np.flatnonzero(expected != observed)
    return {
        "first_mismatch_indices": mismatches[:16].tolist(),
        "mismatch_count": int(mismatches.size),
        "observed_sha256": _array_sha256(observed),
    }


def classify(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError(
            "classifier requires explicit JAX_PLATFORMS=cpu before JAX import"
        )

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.benchmarking.pp16_feature2_qkv_partial import (
        one_row_lp2_feature_partial_qkv_a_convolution,
        validate_lp2_feature_partial_stablehlo,
    )
    from glm_tpu.greenfield.kernels.reference.qkv_a import (
        FusedQkvAContract,
        one_row_fused_qkv_a_convolution,
    )

    if jax.default_backend() != "cpu" or len(jax.devices()) != 2:
        raise RuntimeError("classifier requires exactly two forced CPU devices")
    manifest_path = args.runtime_root / "runtime_manifest.json"
    success_path = args.runtime_root / "SUCCESS"
    if _file_sha256(manifest_path) != RUNTIME_MANIFEST_FILE_SHA256:
        raise RuntimeError("runtime manifest file SHA-256 drifted")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != RUNTIME_MANIFEST_SELF_SHA256:
        raise RuntimeError("runtime manifest self identity drifted")
    if success_path.read_text() != (
        f"{RUNTIME_MANIFEST_SELF_SHA256}  runtime_manifest.json\n"
    ):
        raise RuntimeError("runtime SUCCESS marker drifted")

    owners = []
    for slot in (0, 1):
        owners.append(
            _read_safetensors_values(
                args.runtime_root
                / f"base_decoder_runtime_feature/stage_00/device_slot_{slot:02d}.safetensors"
            )
        )
    for name in owners[0]:
        if not np.array_equal(owners[0][name], owners[1][name]):
            raise RuntimeError(f"runtime N82 owners differ for {name}")
    weight = owners[0]["attention.slot_01.qkv_a.weight_bits"]
    scale = owners[0]["attention.slot_01.qkv_a.scale_inv"]
    norm = owners[0]["attention.slot_01.q_a_norm"]

    if _file_sha256(args.accepted_oracle) != ACCEPTED_ORACLE_FILE_SHA256:
        raise RuntimeError("accepted layer-1 oracle SHA-256 drifted")
    with np.load(args.accepted_oracle, allow_pickle=False) as values:
        accepted_hidden_bits = np.ascontiguousarray(
            values["accepted__normalized_hidden"]
        )
        accepted_q_bits = np.ascontiguousarray(values["accepted__q_a_state"])
    if _array_sha256(accepted_hidden_bits) != ACCEPTED_NORMALIZED_SHA256 or (
        _array_sha256(accepted_q_bits) != ACCEPTED_Q_A_SHA256
    ):
        raise RuntimeError("accepted layer-1 arrays drifted")

    if _file_sha256(args.db518_result) != DB518_FILE_SHA256:
        raise RuntimeError("DB518 result SHA-256 drifted")
    with np.load(args.db518_result, allow_pickle=False) as values:
        current_hidden_owners = np.ascontiguousarray(
            values["current_normalized_hidden_owners_bfloat16_bits"]
        )
        current_q_owners = np.ascontiguousarray(
            values["current_q_a_state_owners_bfloat16_bits"]
        )
        layer1_cache = np.ascontiguousarray(
            values["layer1_index_cache_owners_bfloat16_bits"]
        )
    if not np.array_equal(current_hidden_owners[0], current_hidden_owners[1]) or (
        not np.array_equal(current_q_owners[0], current_q_owners[1])
    ):
        raise RuntimeError("DB518 layer-1 owner observations differ")
    current_hidden_bits = current_hidden_owners[0, 0]
    current_q_bits = current_q_owners[0, 0]

    weight_jax = jnp.asarray(weight)
    scale_jax = jnp.asarray(scale)
    norm_jax = jnp.asarray(norm)

    def full(hidden: Any) -> Any:
        return one_row_fused_qkv_a_convolution(
            hidden,
            weight_jax,
            scale_jax,
            norm_jax,
            contract=FusedQkvAContract(),
        ).q_residual

    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("feature",))

    def partial_mapped(hidden: Any, bits: Any, scales: Any, norm_weight: Any) -> Any:
        return one_row_lp2_feature_partial_qkv_a_convolution(
            hidden,
            bits,
            scales,
            norm_weight,
            axis_name="feature",
            groups=((0, 1),),
        ).q_residual

    partial = jax.shard_map(
        partial_mapped,
        mesh=mesh,
        in_specs=(P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    partial_jit = jax.jit(partial)
    accepted_hidden = jnp.asarray(
        accepted_hidden_bits.view(ml_dtypes.bfloat16)[None, :]
    )
    current_hidden = jnp.asarray(
        current_hidden_bits.view(ml_dtypes.bfloat16)[None, :]
    )
    lowered = partial_jit.lower(
        accepted_hidden, weight_jax, scale_jax, norm_jax
    )
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_contract = validate_lp2_feature_partial_stablehlo(stablehlo)

    results = {
        "accepted_full": _bits(jax.jit(full)(accepted_hidden)),
        "accepted_f32_partial": _bits(partial_jit(accepted_hidden, weight_jax, scale_jax, norm_jax)),
        "accepted_bf16_partial": _bits(
            jax.jit(_project_bf16_partials)(
                accepted_hidden, weight_jax, scale_jax, norm_jax
            )
        ),
        "current_full": _bits(jax.jit(full)(current_hidden)),
        "current_f32_partial": _bits(
            partial_jit(current_hidden, weight_jax, scale_jax, norm_jax)
        ),
    }
    comparisons = {
        "accepted_full_vs_accepted": _comparison(
            accepted_q_bits, results["accepted_full"]
        ),
        "accepted_f32_partial_vs_accepted": _comparison(
            accepted_q_bits, results["accepted_f32_partial"]
        ),
        "accepted_bf16_partial_vs_accepted": _comparison(
            accepted_q_bits, results["accepted_bf16_partial"]
        ),
        "current_full_vs_captured": _comparison(
            current_q_bits, results["current_full"]
        ),
        "current_f32_partial_vs_captured": _comparison(
            current_q_bits, results["current_f32_partial"]
        ),
        "current_f32_partial_vs_accepted": _comparison(
            accepted_q_bits, results["current_f32_partial"]
        ),
    }
    observed_cpu = {
        "accepted_full_q_a_sha256": comparisons[
            "accepted_full_vs_accepted"
        ]["observed_sha256"],
        "accepted_f32_partial_q_a_sha256": comparisons[
            "accepted_f32_partial_vs_accepted"
        ]["observed_sha256"],
        "accepted_bf16_partial_q_a_sha256": comparisons[
            "accepted_bf16_partial_vs_accepted"
        ]["observed_sha256"],
        "accepted_bf16_partial_mismatches": comparisons[
            "accepted_bf16_partial_vs_accepted"
        ]["mismatch_count"],
        "current_full_q_a_sha256": comparisons["current_full_vs_captured"][
            "observed_sha256"
        ],
        "current_f32_partial_q_a_sha256": comparisons[
            "current_f32_partial_vs_captured"
        ]["observed_sha256"],
        "current_f32_partial_vs_accepted_mismatches": comparisons[
            "current_f32_partial_vs_accepted"
        ]["mismatch_count"],
        "current_f32_partial_vs_captured_mismatches": comparisons[
            "current_f32_partial_vs_captured"
        ]["mismatch_count"],
    }
    if observed_cpu != EXPECTED_CPU:
        raise RuntimeError(
            f"LP2 qkv-a CPU classification drifted: {observed_cpu}"
        )

    source = args.feature2_program.read_text()
    if _file_sha256(args.feature2_program) != FEATURE2_PROGRAM_SHA256:
        raise RuntimeError("feature2 coherent-history source SHA-256 drifted")
    ordered_markers = (
        "physical_m64_prompt_index_key_chunk(",
        "final, normalized_history = lax.scan",
        "projected1 = _qkv_a(",
        "layer1_index_cache,",
        "precomputed_normalized=last_normalized,",
        "precomputed_q_residual=projected1.q_residual,",
    )
    cursor = 0
    for marker in ordered_markers:
        index = source.find(marker, cursor)
        if index < 0:
            raise RuntimeError("feature2 coherent-history source lineage drifted")
        cursor = index + len(marker)

    return {
        "artifact_kind": "pp16_feature2_qkv_khalf_cpu_admission_v1",
        "candidate": "LP2_KHALF_N82_F32_PARTIAL_REDUCTION",
        "classification": (
            "FP32_PARTIAL_ARM_CPU_NECESSARY_CONDITION_PASSES;"
            "TPU_PHYSICAL_ASSOCIATION_AND_EVENT1_EXACTNESS_UNRESOLVED"
        ),
        "comparisons": comparisons,
        "cpu_backend": jax.default_backend(),
        "cpu_device_count": len(jax.devices()),
        "gate_d_closed": False,
        "performance_claim": False,
        "protected_tpu_evidence": False,
        "stablehlo": {
            **stablehlo_contract,
            "sha256": sha256(stablehlo.encode()).hexdigest(),
        },
        "state_coherence": {
            "candidate_coherent_history_reusable": True,
            "db518_layer1_index_cache_sha256": _array_sha256(layer1_cache),
            "feature2_program_sha256": FEATURE2_PROGRAM_SHA256,
            "reason": (
                "The candidate changes only the layer-1 qkv-a projection after the "
                "existing normalized-history/M64 index-key cache is complete. The "
                "layer-1 cache and current key depend on normalized hidden plus wk, "
                "not qkv-a; candidate q-a/query/head/event1 must still be recomputed "
                "together in one bounded executable."
            ),
        },
        "source_identities": {
            "accepted_oracle_file_sha256": ACCEPTED_ORACLE_FILE_SHA256,
            "db518_result_file_sha256": DB518_FILE_SHA256,
            "runtime_manifest_file_sha256": RUNTIME_MANIFEST_FILE_SHA256,
            "runtime_manifest_self_sha256": RUNTIME_MANIFEST_SELF_SHA256,
            "n82_weight_sha256": N82_WEIGHT_SHA256,
            "n82_scale_sha256": N82_SCALE_SHA256,
            "q_a_norm_sha256": Q_A_NORM_SHA256,
        },
        "status": "SUCCESS",
        "tpus_used": 0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--accepted-oracle", type=Path, required=True)
    parser.add_argument("--db518-result", type=Path, required=True)
    parser.add_argument("--feature2-program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_output_vacant(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"classifier output already exists: {path}")


def _write_exclusive_json(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def main() -> None:
    args = _parse_args()
    _require_output_vacant(args.output)
    report = classify(args)
    _write_exclusive_json(args.output, report)


if __name__ == "__main__":
    main()
