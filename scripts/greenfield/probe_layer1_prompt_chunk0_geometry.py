#!/usr/bin/env python3
"""Bounded probe: layer-1 prompt keys of chunk 0 in the legacy prefill geometry.

Executes the layer-0 block for prompt rows ``[0, 2048)`` with every projection,
norm and MLP shaped as the legacy 2,048-row prefill (virtual TP32 owners on one
device, BF16 owner partials combined with the DB533 row-0 association), feeds
the layer-1 normalized rows through the accepted 64-row prompt-key path and
compares the layer-1 index keys bitwise with the sealed legacy layer-1 prompt
cache.  Row 0 is decisive (attention over one key is exact by construction);
rows >= 1 use a plain FP32 causal softmax instead of the legacy Pallas kernel
and are reported only as a residual map.

Controls: the same rows' layer-0 keys must equal the DB518 greenfield layer-0
cache (proven legacy-exact), and the one-row (M=1) variant of row 0 is reported
against the DB518 greenfield layer-1 row.  CPU/HLO evidence is not proof; this
probe makes no decoder, Gate-D, DB or performance claim.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
ARTIFACT_KIND = "greenfield_layer1_prompt_chunk0_legacy_geometry_probe"
CHUNK_ROWS = 2048
PROMPT_ROWS = 8155
MAIN_ROPE_CAPACITY = 8192
MAIN_ROPE_THETA = 8_000_000.0
LAYER0_NAMES = {
    "q_b": "model.layers.0.self_attn.q_b_proj.weight",
    "kv_b": "model.layers.0.self_attn.kv_b_proj.weight",
    "o": "model.layers.0.self_attn.o_proj.weight",
    "gate": "model.layers.0.mlp.gate_proj.weight",
    "up": "model.layers.0.mlp.up_proj.weight",
    "down": "model.layers.0.mlp.down_proj.weight",
}
LAYER0_NORMS = {
    "kv_a_norm": "model.layers.0.self_attn.kv_a_layernorm.weight",
    "post_norm": "model.layers.0.post_attention_layernorm.weight",
}
LAYER1_NAMES = {
    "wk1": "model.layers.1.self_attn.indexer.wk.weight",
}
LAYER1_NORMS = {
    "input_norm1": "model.layers.1.input_layernorm.weight",
    "k_norm1_weight": "model.layers.1.self_attn.indexer.k_norm.weight",
    "k_norm1_bias": "model.layers.1.self_attn.indexer.k_norm.bias",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _cache_rows(owner_bits: np.ndarray, rows: int) -> np.ndarray:
    """DB518 capture layout: owner=(p%512)//256, page=p//512, row=p%256."""

    positions = np.arange(rows)
    local = positions % 512
    return np.ascontiguousarray(owner_bits[local // 256, positions // 512, local % 256])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-index-sha256", required=True)
    parser.add_argument("--legacy-layer1-cache-dir", type=Path, required=True)
    parser.add_argument("--legacy-layer1-manifest-sha256", required=True)
    parser.add_argument("--db518-result", type=Path, required=True)
    parser.add_argument("--db518-result-sha256", required=True)
    parser.add_argument("--softmax-scale", type=float, default=256 ** -0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def _load_checkpoint_tensors(root: Path, expected_index_sha256: str) -> dict[str, np.ndarray]:
    import torch
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    if _sha256_file(index_path) != expected_index_sha256:
        raise RuntimeError("checkpoint index identity drifted")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    wanted = {**LAYER0_NAMES, **LAYER0_NORMS, **LAYER1_NAMES, **LAYER1_NORMS}
    scale_names = {key: f"{name}_scale_inv" for key, name in {**LAYER0_NAMES, **LAYER1_NAMES}.items()}
    tensors: dict[str, np.ndarray] = {}
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for key, name in list(wanted.items()) + [(f"{k}_scale", n) for k, n in scale_names.items()]:
        by_shard.setdefault(weight_map[name], []).append((key, name))
    for shard, items in by_shard.items():
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            for key, name in items:
                value = handle.get_tensor(name)
                if value.dtype == torch.float8_e4m3fn:
                    tensors[key] = value.view(torch.uint8).numpy().copy()
                elif value.dtype == torch.bfloat16:
                    tensors[key] = value.view(torch.uint16).numpy().copy()
                elif value.dtype == torch.float32:
                    tensors[key] = value.numpy().copy()
                else:
                    raise RuntimeError(f"unexpected checkpoint dtype for {name}: {value.dtype}")
    return tensors


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(f"stale code hash: expected={args.expected_code_hash} found={code_hash}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    args.hlo_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.benchmarking import legacy_prefill_geometry as geo
    from glm_tpu.greenfield.benchmarking import legacy_prefill_owner_packing as pk
    from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_layer0_dsa_association_input,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        inspect_legacy_prompt_index_cache,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"probe requires TPU, got {jax.default_backend()}")
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError("probe requires one isolated four-chip TPU-v4 host")
    device = jax.local_devices()[0]

    started = time.time()
    input_manifest, arrays = inspect_layer0_dsa_association_input(
        args.input_dir, expected_manifest_sha256=args.input_manifest_sha256
    )
    legacy_manifest, legacy_layer1_bits = inspect_legacy_prompt_index_cache(
        args.legacy_layer1_cache_dir, expected_manifest_sha256=args.legacy_layer1_manifest_sha256
    )
    if int(legacy_manifest.get("layer_id", 0)) != 1:
        raise RuntimeError("legacy cache artifact is not the layer-1 prompt cache")
    if _sha256_file(args.db518_result) != args.db518_result_sha256:
        raise RuntimeError("DB518 result.npz identity drifted")
    db518 = np.load(args.db518_result)
    greenfield_layer0 = _cache_rows(db518["layer0_index_cache_owners_bfloat16_bits"], CHUNK_ROWS)
    greenfield_layer1 = _cache_rows(db518["layer1_index_cache_owners_bfloat16_bits"], CHUNK_ROWS)
    legacy_layer1 = legacy_layer1_bits[:CHUNK_ROWS]
    tensors = _load_checkpoint_tensors(args.checkpoint_root, args.checkpoint_index_sha256)

    prompt_ids = arrays["prompt_token_ids"]
    if prompt_ids.shape != (PROMPT_ROWS,):
        raise RuntimeError("prompt token geometry drifted")
    embedding_bits = pk.chunk_embeddings(
        prompt_ids,
        arrays["unique_token_ids"],
        arrays["unique_embedding_bfloat16_bits"],
        start=0,
        rows=CHUNK_ROWS,
    )
    qkv_bits, qkv_scale = pk.pack_fused_qkv_a_owners(
        arrays["self_attn__q_a_proj__weight"],
        arrays["self_attn__q_a_proj__weight_scale_inv"],
        arrays["self_attn__kv_a_proj_with_mqa__weight"],
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"],
    )
    qb_bits, qb_scale = pk.pack_q_b_owners(tensors["q_b"], tensors["q_b_scale"])
    w_uk_t, w_uv = pk.absorbed_kv_b_owners(tensors["kv_b"], tensors["kv_b_scale"])
    o_bits, o_scale = pk.pack_o_proj_owners(tensors["o"], tensors["o_scale"])
    gu_bits, gu_scale = pk.pack_gate_up_owners(
        tensors["gate"], tensors["gate_scale"], tensors["up"], tensors["up_scale"]
    )
    down_bits, down_scale = pk.pack_down_owners(tensors["down"], tensors["down_scale"])
    rope_table = build_rotary_table_host(MAIN_ROPE_CAPACITY, rotary_dim=64, theta=MAIN_ROPE_THETA)
    contract = DsaNumericalContract()

    def bf16(bits: np.ndarray) -> Any:
        return jax.device_put(jnp.asarray(bits.view(ml_dtypes.bfloat16)), device)

    def put(value: np.ndarray) -> Any:
        return jax.device_put(jnp.asarray(value), device)

    weights = {
        "input_norm0": bf16(arrays["input_layernorm__weight"]),
        "q_a_norm": bf16(arrays["self_attn__q_a_layernorm__weight"]),
        "kv_a_norm": bf16(tensors["kv_a_norm"]),
        "post_norm": bf16(tensors["post_norm"]),
        "input_norm1": bf16(tensors["input_norm1"]),
        "k_norm1_weight": bf16(tensors["k_norm1_weight"]),
        "k_norm1_bias": bf16(tensors["k_norm1_bias"]),
        "k_norm0_weight": bf16(arrays["self_attn__indexer__k_norm__weight"]),
        "k_norm0_bias": bf16(arrays["self_attn__indexer__k_norm__bias"]),
        "qkv_bits": put(qkv_bits),
        "qkv_scale": put(qkv_scale),
        "qb_bits": put(qb_bits),
        "qb_scale": put(qb_scale),
        "w_uk_t": put(w_uk_t),
        "w_uv": put(w_uv),
        "o_bits": put(o_bits),
        "o_scale": put(o_scale),
        "gu_bits": put(gu_bits),
        "gu_scale": put(gu_scale),
        "down_bits": put(down_bits),
        "down_scale": put(down_scale),
        "wk0_bits": put(arrays["self_attn__indexer__wk__weight"]),
        "wk0_scale": put(arrays["self_attn__indexer__wk__weight_scale_inv"]),
        "wk1_bits": put(tensors["wk1"]),
        "wk1_scale": put(tensors["wk1_scale"]),
        "rope_table": bf16(rope_table.view(np.uint16)),
    }

    from glm_tpu.greenfield.benchmarking.legacy_prefill_chunk_probe import (
        legacy_geometry_chunk_pipeline,
    )

    def pipeline(embedding: Any, positions: Any, w: dict[str, Any]) -> dict[str, Any]:
        return legacy_geometry_chunk_pipeline(
            embedding, positions, w, softmax_scale=args.softmax_scale, contract=contract
        )

    positions_np = np.arange(CHUNK_ROWS, dtype=np.int32)
    embedding_full = bf16(embedding_bits)
    positions_full = put(positions_np)
    lowered = jax.jit(pipeline).lower(embedding_full, positions_full, weights)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    (args.hlo_dir / "legacy_geometry_chunk0.optimized_hlo.txt").write_text(hlo_text)
    (args.hlo_dir / "legacy_geometry_chunk0.stablehlo.mlir").write_text(lowered.as_text())
    result_full = jax.device_get(compiled(embedding_full, positions_full, weights))
    result_row0 = jax.device_get(jax.jit(pipeline)(bf16(embedding_bits[:1]), put(positions_np[:1]), weights))

    def bits(value: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(value)).view(np.uint16)

    keys1_bits = bits(result_full["keys1"])
    keys0_bits = bits(result_full["keys0"])
    keys1_row0_m1 = bits(result_row0["keys1"])[0]
    control_mismatch_rows = int(np.count_nonzero(np.any(keys0_bits != greenfield_layer0, axis=1)))
    legacy_lane_mismatch = (keys1_bits != legacy_layer1)
    greenfield_lane_mismatch = (keys1_bits != greenfield_layer1)
    per_row_legacy = legacy_lane_mismatch.sum(axis=1)
    per_row_greenfield = greenfield_lane_mismatch.sum(axis=1)
    row0_legacy_lanes = int(per_row_legacy[0])
    row0_m1_vs_greenfield_lanes = int(np.count_nonzero(keys1_row0_m1 != greenfield_layer1[0]))
    row0_m1_vs_legacy_lanes = int(np.count_nonzero(keys1_row0_m1 != legacy_layer1[0]))
    row0_m2048_vs_m1_lanes = int(np.count_nonzero(keys1_row0_m1 != keys1_bits[0]))
    hlo_convolution_2048 = len(
        [line for line in hlo_text.splitlines() if "convolution(" in line and "[2048," in line]
    )
    forbidden = [token for token in ("host_callback", "CustomCall(\"xla_python", "python_callback") if token in hlo_text]

    status = "SUCCESS" if control_mismatch_rows == 0 and not forbidden else "FAILED"
    summary = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": code_hash,
        "run_tag": args.run_tag,
        "status": status,
        "control_layer0_keys_vs_db518_mismatched_rows": control_mismatch_rows,
        "forbidden_hlo_tokens": forbidden,
        "hlo": {
            "optimized_sha256": sha256(hlo_text.encode()).hexdigest(),
            "convolution_lines_with_2048_rows": int(hlo_convolution_2048),
        },
        "inputs": {
            "layer0_input_manifest_sha256": input_manifest["manifest_sha256"],
            "legacy_layer1_manifest_sha256": legacy_manifest["manifest_sha256"],
            "legacy_layer1_bits_sha256": legacy_manifest["prompt_index_key_bfloat16_sha256"],
            "db518_result_sha256": args.db518_result_sha256,
            "checkpoint_index_sha256": args.checkpoint_index_sha256,
            "softmax_scale": args.softmax_scale,
            "checkpoint_tensors": [pk.summarize_bits(k, v) for k, v in sorted(tensors.items())],
        },
        "row0": {
            "legacy_geometry_vs_legacy_lanes": row0_legacy_lanes,
            "legacy_geometry_vs_greenfield_db518_lanes": int(per_row_greenfield[0]),
            "one_row_variant_vs_greenfield_db518_lanes": row0_m1_vs_greenfield_lanes,
            "one_row_variant_vs_legacy_lanes": row0_m1_vs_legacy_lanes,
            "legacy_geometry_vs_one_row_variant_lanes": row0_m2048_vs_m1_lanes,
            "attention_row0_db533_vs_pairwise_lanes": int(
                np.count_nonzero(bits(result_full["attention_row0"]) != bits(result_full["attention_pairwise_row0"]))
            ),
        },
        "chunk0_vs_legacy": {
            "rows_exact": int(np.count_nonzero(per_row_legacy == 0)),
            "rows": CHUNK_ROWS,
            "lanes_mismatched": int(legacy_lane_mismatch.sum()),
            "per_row_first_16": per_row_legacy[:16].tolist(),
            "per_64_row_block_mean": [float(per_row_legacy[i : i + 64].mean()) for i in range(0, CHUNK_ROWS, 64)],
        },
        "chunk0_vs_greenfield_db518": {
            "rows_exact": int(np.count_nonzero(per_row_greenfield == 0)),
            "lanes_mismatched": int(greenfield_lane_mismatch.sum()),
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "claim_scope": (
            "Bounded diagnostic of prompt-row geometry; row 0 decisive, rows >= 1 use a non-legacy "
            "softmax; no decoder, Gate-D, DB or performance claim."
        ),
    }
    np.savez(
        args.output / "probe_arrays.npz",
        keys1_bits=keys1_bits,
        keys0_bits=keys0_bits,
        keys1_row0_one_row_bits=keys1_row0_m1,
        legacy_layer1_bits=legacy_layer1,
        greenfield_layer1_bits=greenfield_layer1,
        **{f"{name}_bits": bits(value) for name, value in result_full.items() if name.endswith("_row0")},
        **{f"one_row_{name}_bits": bits(value) for name, value in result_row0.items() if name.endswith("_row0")},
    )
    summary["arrays_sha256"] = _sha256_file(args.output / "probe_arrays.npz")
    (args.output / "runner.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "row0", "chunk0_vs_legacy", "control_layer0_keys_vs_db518_mismatched_rows")}, sort_keys=True))
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
