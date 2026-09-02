#!/usr/bin/env python3
"""CPU-only, hash-bound certificate of the layer-1 RMS scale frontier at position 8155.

From sealed bytes only (no JAX, no TPU, no model): the DB548 greenfield layer-1
normalized row and the accepted legacy layer-1 normalized row are both reproduced
bit-for-bit from the SAME FP32 RMS input vector (dense update + attention update +
pre-attention residual, all exact FP32 sums of BF16 values), the layer-1 input-norm
weight, epsilon 1e-5 and a single BF16 rounding of ``(x * s) * w``; they differ only
in the FP32 scale ``s = rsqrt(mean(x^2) + eps)``.  The certificate reports the exact
admissible ``s`` windows for both rows and where candidate variance-reduction
structures land.  It makes no Gate-D, performance or mechanism-identity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

BF16 = ml_dtypes.bfloat16
HIDDEN = 6144
EPSILON = 1e-5
SOURCES = {
    "db548_capture": Path(
        "/home/gianl/gcs-models/results/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/dense_partial_capture.npz"
    ),
    "db548_rms_replay": Path(
        "/home/gianl/gcs-models/results/greenfield_layer0_captured_rms_replay_20260813T214428668951467Z/captured_rms_replay.npz"
    ),
    "capsule_inputs": Path(
        "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z/candidate-inputs.npz"
    ),
    "accepted_layer1_internals": Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/dsa_internals/8k/recovery/greenfield_legacy_layer1_dsa_internals_recovery_20260808T093405285628860Z/internal_capture/internals.npz"
    ),
    "accepted_dense_partials": Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/dense_partials/8k/greenfield_legacy_layer0_dense_partials_p8155_20260814T100132090917640Z/dense_partials_capture/dense_partials.npz"
    ),
}
EXPECTED_ARRAY_SHA256 = {
    "attention_update": "68afed86921584fb673abb11a563e359a1533210ec2483e71ee79b88c2b0bde7",
    "combined_residual": "02d045b9a0ec5ab22a711bd6a964564f707be0848683381104e83331020e31a3",
    "post_attention_residual": "a105fdbd429adb1d06a70bf71598a72a91d7b6faa83360005487ce11ce099f8e",
    "layer1_input_norm": "10e34f4f99c638b29557526283205071c1ac8f81f168f4a6817e7e1def4b6c87",
    "dense_update": "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc",
    "accepted_dense_partials": "9d9f65dddc7b622875872a33a6522c330c8fb5490c8cba14526553c211516e35",
    "db548_layer1_normalized": "9b52a04e2852719237f4465b28665cbc213b635763303b554bb12345e99a4005",
    "accepted_layer1_normalized": "9936ee1e19049b297fd205292ebc378aee41d59401bbf56497004356998d3039",
}


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f32_of_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bits, dtype=np.uint16).view(BF16).astype(np.float32)


def hex_f32(value: float) -> str:
    return struct.pack(">f", float(np.float32(value))).hex()


def output_row(x: np.ndarray, s: np.float32, w: np.ndarray) -> np.ndarray:
    """Single BF16 rounding of (x * s) * w, all FP32 before the final round."""
    return ((x * s).astype(np.float32) * w).astype(np.float32).astype(BF16).view(np.uint16)


def seq_sum(values: np.ndarray) -> np.float32:
    acc = np.float32(0)
    for value in values:
        acc = np.float32(acc + value)
    return acc


def pairwise_sum(values: np.ndarray) -> np.float32:
    values = np.asarray(values, dtype=np.float32)
    while values.size > 1:
        if values.size % 2:
            values = np.append(values, np.float32(0))
        values = (values[0::2] + values[1::2]).astype(np.float32)
    return values[0]


def lanes(values: np.ndarray, lane_count: int) -> np.ndarray:
    part = np.zeros(lane_count, np.float32)
    for j in range(values.size // lane_count):
        part = (part + values[j * lane_count : (j + 1) * lane_count]).astype(np.float32)
    return part


def blocks(values: np.ndarray, block: int) -> np.ndarray:
    return np.array([seq_sum(values[i : i + block]) for i in range(0, values.size, block)], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    file_hashes = {name: sha256_file(path) for name, path in SOURCES.items()}
    capture = np.load(SOURCES["db548_capture"])
    replay = np.load(SOURCES["db548_rms_replay"])
    capsule = np.load(SOURCES["capsule_inputs"])
    accepted_internals = np.load(SOURCES["accepted_layer1_internals"])
    partials = np.load(SOURCES["accepted_dense_partials"])
    arrays = {
        "attention_update": capture["attention_update_bfloat16_bits"][0],
        "combined_residual": capture["combined_residual_bfloat16_bits"][0],
        "post_attention_residual": capture["post_attention_residual_bfloat16_bits"][0],
        "layer1_input_norm": capture["layer1_input_norm_bfloat16_bits"],
        "dense_update": capsule["rms_hidden_update_bf16_bits"],
        "accepted_dense_partials": partials["accepted_dense_partials_bfloat16_bits"],
        "db548_layer1_normalized": replay["db548_layer1_normalized_bfloat16_bits"],
        "accepted_layer1_normalized": accepted_internals["normalized_hidden"],
    }
    array_hashes = {name: sha256_bytes(value) for name, value in arrays.items()}
    for name, expected in EXPECTED_ARRAY_SHA256.items():
        if array_hashes[name] != expected:
            raise SystemExit(f"{name}: SHA drift {array_hashes[name]} != {expected}")
    if not np.array_equal(capture["accepted_layer1_normalized_bfloat16_bits"], arrays["accepted_layer1_normalized"]):
        raise SystemExit("DB548 capture carries a different accepted layer-1 row")
    if not np.array_equal(partials["db548_dense_partials_bfloat16_bits"], arrays["accepted_dense_partials"]):
        raise SystemExit("DB548 and accepted dense partials differ")

    attention = f32_of_bf16_bits(arrays["attention_update"])
    combined = f32_of_bf16_bits(arrays["combined_residual"])
    post_attention = arrays["post_attention_residual"]
    dense = f32_of_bf16_bits(arrays["dense_update"])
    weight = f32_of_bf16_bits(arrays["layer1_input_norm"])
    db548_row = np.ascontiguousarray(arrays["db548_layer1_normalized"])
    accepted_row = np.ascontiguousarray(arrays["accepted_layer1_normalized"])

    # Lineage identities on the sealed rows.
    residual_identity = bool(
        np.array_equal((attention + combined).astype(np.float32).astype(BF16).view(np.uint16), post_attention)
    )
    row_mismatch_indices = np.flatnonzero(db548_row != accepted_row).tolist()

    # The RMS input: exact FP32 three-term sum (all associations agree bitwise here).
    x_orders = {
        "(dense+attention)+combined": ((dense + attention).astype(np.float32) + combined).astype(np.float32),
        "dense+(attention+combined)": (dense + (attention + combined).astype(np.float32)).astype(np.float32),
    }
    associations_agree = bool(np.array_equal(x_orders["(dense+attention)+combined"], x_orders["dense+(attention+combined)"]))
    x = x_orders["(dense+attention)+combined"]
    var64 = float(np.mean(x.astype(np.float64) ** 2))
    s0 = np.float32(1.0 / np.sqrt(var64 + EPSILON))
    ulp = np.spacing(s0)

    def window(target: np.ndarray) -> list[int]:
        return [k for k in range(-256, 257) if np.array_equal(output_row(x, np.float32(s0 + k * ulp), weight), target)]

    accepted_window = window(accepted_row)
    db548_window = window(db548_row)
    # Alternative carries that must NOT reproduce either row (materialized BF16 residual, double rounding).
    x_bf16_residual = (dense + f32_of_bf16_bits(post_attention)).astype(np.float32)
    var_alt = float(np.mean(x_bf16_residual.astype(np.float64) ** 2))
    s_alt = np.float32(1.0 / np.sqrt(var_alt + EPSILON))
    ulp_alt = np.spacing(s_alt)
    alt_best = {}
    for label, target in (("accepted", accepted_row), ("db548", db548_row)):
        best_single = min(
            int(np.count_nonzero(output_row(x_bf16_residual, np.float32(s_alt + k * ulp_alt), weight) != target))
            for k in range(-256, 257)
        )
        best_double = min(
            int(
                np.count_nonzero(
                    (
                        (x * np.float32(s0 + k * ulp)).astype(np.float32).astype(BF16).astype(np.float32) * weight
                    ).astype(np.float32).astype(BF16).view(np.uint16)
                    != target
                )
            )
            for k in range(-256, 257)
        )
        alt_best[label] = {
            "materialized_bf16_residual_single_round_min_mismatch": best_single,
            "fp32_residual_double_round_min_mismatch": best_double,
        }

    # Where candidate variance structures land (FP32 arithmetic; f64 = exact sum then one rounding).
    squares = (x * x).astype(np.float32)
    sums = {"sequential": seq_sum(squares), "pairwise": pairwise_sum(squares), "exact_f64": np.float32(np.sum(squares.astype(np.float64)))}
    for lane_count in (8, 128, 256, 512, 1024, 2048):
        part = lanes(squares, lane_count)
        sums[f"lanes{lane_count}_then_tree"] = pairwise_sum(part)
        sums[f"lanes{lane_count}_then_sequential"] = seq_sum(part)
    for block in (48, 128, 384, 768, 1536):
        sums[f"blocks{block}_then_tree"] = pairwise_sum(blocks(squares, block))
    structures = []
    for name, total in sums.items():
        for mean_name, mean in (("divide", np.float32(total / np.float32(HIDDEN))), ("multiply", np.float32(total * np.float32(1.0 / HIDDEN)))):
            for rsqrt_name, s in (
                ("one_over_sqrt_f32", np.float32(np.float32(1.0) / np.sqrt(np.float32(mean + np.float32(EPSILON))))),
                ("correctly_rounded_rsqrt", np.float32(1.0 / np.sqrt(np.float64(np.float32(mean + np.float32(EPSILON)))))),
            ):
                offset = int(round((float(s) - float(s0)) / float(ulp)))
                structures.append(
                    {
                        "sum": name,
                        "mean": mean_name,
                        "rsqrt": rsqrt_name,
                        "s_hex": hex_f32(s),
                        "ulp_offset_from_s0": offset,
                        "lands_in": "accepted" if offset in accepted_window else ("db548" if offset in db548_window else "neither"),
                    }
                )

    result = {
        "artifact_kind": "gate_d_layer1_scale_frontier_certificate",
        "schema_version": 1,
        "claim_scope": (
            "CPU-only replay from sealed bytes. Proves that both the DB548 greenfield and the accepted "
            "legacy layer-1 normalized rows at position 8155 are exact functions of the same FP32 RMS input "
            "vector, weight and epsilon under a single output rounding, differing only in the FP32 scale; "
            "reports the admissible scale windows. No Gate-D, performance or reduction-structure-identity claim."
        ),
        "position": 8155,
        "layer": 1,
        "epsilon": EPSILON,
        "sources_sha256": file_hashes,
        "arrays_sha256": array_hashes,
        "lineage": {
            "post_attention_residual_equals_bf16(attention_update+combined_residual)": residual_identity,
            "accepted_dense_partials_equal_db548_partials": True,
            "db548_vs_accepted_row_mismatch_indices": row_mismatch_indices,
            "three_term_fp32_sum_association_invariant": associations_agree,
        },
        "rms_input": {
            "definition": "x = f32(dense_update_bf16) + f32(attention_update_bf16) + f32(combined_residual_bf16), no intermediate BF16 rounding",
            "variance_exact_f64": var64,
            "s0_hex": hex_f32(s0),
            "s0": float(s0),
            "ulp": float(ulp),
        },
        "output_formula": "bf16((x * s) * w) with one rounding (materialized bf16(x*s) does not reproduce either row)",
        "windows": {
            "accepted_ulp_offsets": accepted_window,
            "accepted_s_hex": [hex_f32(s0 + k * ulp) for k in accepted_window],
            "db548_ulp_offsets": db548_window,
            "db548_s_hex": [hex_f32(s0 + k * ulp) for k in db548_window],
            "disjoint": not set(accepted_window) & set(db548_window),
            "gap_ulps": (min(accepted_window) - max(db548_window)) if accepted_window and db548_window else None,
        },
        "rejected_carry_alternatives": alt_best,
        "variance_structures": structures,
    }
    exact_both = bool(accepted_window) and bool(db548_window)
    result["classification"] = (
        ("LAYER1_SCALE_FRONTIER_CERTIFIED;SAME_FP32_RMS_INPUT;ACCEPTED_AND_DB548_ROWS_EXACT_UNDER_SCALE_WINDOWS;"
         "FRONTIER_IS_ONE_FP32_SCALAR;" if exact_both else "INCONCLUSIVE;")
        + "REDUCTION_STRUCTURE_NOT_IDENTIFIED;GATE_D_OPEN"
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["classification"])
    print(json.dumps({k: result["windows"][k] for k in ("accepted_ulp_offsets", "db548_ulp_offsets", "gap_ulps")}))
    print(json.dumps(result["lineage"]))
    print(json.dumps(alt_best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
