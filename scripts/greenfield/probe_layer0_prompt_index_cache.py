#!/usr/bin/env python3
"""Compare production one-row layer-0 prompt keys to the accepted cache."""

from __future__ import annotations

import argparse
import gzip
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
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
ARTIFACT_KIND = "greenfield_layer0_prompt_index_cache_comparison"
FORMAT_VERSION = 1


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--prompt-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    from jax import lax
    import ml_dtypes
    from safetensors.numpy import save_file

    from glm_tpu.greenfield.kernels.pallas import (
        Fp8BlockMatmulConfig,
        fp8_block_matmul_f32,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import (
        DsaNumericalContract,
        dsa_index_keys_from_projection,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_layer0_dsa_association_input,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        compare_prompt_index_key_bits,
        inspect_legacy_prompt_index_cache,
        validate_prompt_index_key_probe_hlo,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"prompt index-key probe requires TPU, got {jax.default_backend()}"
        )
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "prompt index-key probe requires one isolated four-chip TPU-v4 host"
        )
    device = jax.local_devices()[0]
    input_manifest, arrays = inspect_layer0_dsa_association_input(
        args.input_dir,
        expected_manifest_sha256=args.input_manifest_sha256,
    )
    cache_manifest, expected_bits = inspect_legacy_prompt_index_cache(
        args.prompt_cache_dir,
        expected_manifest_sha256=args.prompt_cache_manifest_sha256,
    )
    if cache_manifest["layer0_input_manifest_sha256"] != input_manifest[
        "manifest_sha256"
    ]:
        raise RuntimeError("prompt cache and layer-0 input identity disagree")
    prompt_record = input_manifest["arrays"]["prompt_token_ids"]
    if cache_manifest["prompt_token_ids_sha256"] != prompt_record["sha256"]:
        raise RuntimeError("prompt cache token identity disagrees with input")

    unique_ids = arrays["unique_token_ids"]
    prompt_ids = arrays["prompt_token_ids"]
    prompt_rows_host = np.searchsorted(unique_ids, prompt_ids).astype(np.int32)
    if not np.array_equal(unique_ids[prompt_rows_host], prompt_ids):
        raise RuntimeError("prompt embedding-row mapping drifted")

    def bf16_host(name: str) -> np.ndarray:
        return arrays[name].view(ml_dtypes.bfloat16)

    def put(value: np.ndarray) -> Any:
        return jax.device_put(value, device)

    unique_embeddings = put(bf16_host("unique_embedding_bfloat16_bits"))
    prompt_rows = put(prompt_rows_host)
    input_norm_weight = put(bf16_host("input_layernorm__weight"))
    wk_bits = put(arrays["self_attn__indexer__wk__weight"])
    wk_scale = put(arrays["self_attn__indexer__wk__weight_scale_inv"])
    key_norm_weight = put(
        bf16_host("self_attn__indexer__k_norm__weight")
    )
    key_norm_bias = put(bf16_host("self_attn__indexer__k_norm__bias"))
    contract = DsaNumericalContract()
    pallas_config = Fp8BlockMatmulConfig(
        block_shape=(128, 128),
        output_tile=128,
        contraction_tile=128,
    )

    def production_prompt_keys(
        embedding_rows: Any,
        row_indices: Any,
        norm_weight: Any,
        raw_wk_bits: Any,
        raw_wk_scale: Any,
        k_norm_weight: Any,
        k_norm_bias: Any,
    ) -> Any:
        positions = jnp.arange(row_indices.shape[0], dtype=jnp.int32)

        def scan_step(
            completed: Any, values: tuple[Any, Any]
        ) -> tuple[Any, Any]:
            row_index, position = values
            residual = lax.dynamic_index_in_dim(
                embedding_rows,
                row_index,
                axis=0,
                keepdims=True,
            )
            normalized = rms_norm(residual, norm_weight, epsilon=1e-5)
            projected = fp8_block_matmul_f32(
                normalized,
                raw_wk_bits,
                raw_wk_scale,
                config=pallas_config,
                interpret=False,
            )
            key = dsa_index_keys_from_projection(
                projected,
                k_norm_weight,
                k_norm_bias,
                position[None],
                contract=contract,
            ).astype(jnp.bfloat16)
            return completed + jnp.int32(1), key[0]

        completed, keys = lax.scan(
            scan_step,
            jnp.int32(0),
            (row_indices, positions),
            unroll=1,
        )
        return keys, completed

    arguments = (
        unique_embeddings,
        prompt_rows,
        input_norm_weight,
        wk_bits,
        wk_scale,
        key_norm_weight,
        key_norm_bias,
    )
    before_memory = _memory_stats(device)
    started = time.monotonic()
    compiled = jax.jit(production_prompt_keys).lower(*arguments).compile()
    compile_seconds = time.monotonic() - started
    hlo = compiled.as_text()
    hlo_contract = validate_prompt_index_key_probe_hlo(
        hlo,
        prompt_token_count=prompt_ids.size,
        unique_token_count=unique_ids.size,
    )
    if not hlo_contract["passed"]:
        raise RuntimeError(
            f"production prompt index-key HLO contract failed: {hlo_contract}"
        )

    started = time.monotonic()
    observed, completed = compiled(*arguments)
    jax.block_until_ready((observed, completed))
    execute_seconds = time.monotonic() - started
    if int(np.asarray(completed)) != prompt_ids.size:
        raise RuntimeError("prompt key scan did not execute every position")
    observed_bits = np.ascontiguousarray(np.asarray(observed)).view(np.uint16)
    if observed_bits.shape != expected_bits.shape:
        raise RuntimeError("production prompt key output geometry drifted")
    comparison = compare_prompt_index_key_bits(expected_bits, observed_bits)
    chunk_ranges = ((0, 2048), (2048, 4096), (4096, 6144), (6144, 8155))
    chunk_comparisons = {
        f"positions_{start}_{stop - 1}": compare_prompt_index_key_bits(
            expected_bits[start:stop], observed_bits[start:stop]
        )
        for start, stop in chunk_ranges
    }

    hlo_dir = args.output / "hlo"
    hlo_dir.mkdir()
    hlo_path = hlo_dir / "production_prompt_keys.optimized_hlo.txt.gz"
    with gzip.open(hlo_path, "wt", encoding="utf-8") as stream:
        stream.write(hlo)
    tensor_path = args.output / "observed_prompt_index_keys.safetensors"
    save_file(
        {"observed_prompt_index_key_bfloat16_bits": observed_bits},
        str(tensor_path),
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
        },
    )
    result: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "backend": jax.default_backend(),
        "chunk_comparisons": chunk_comparisons,
        "claim_scope": (
            "Diagnostic layer-0 prompt-cache arithmetic only; elapsed time is "
            "not decoder latency and carries no Gate-D or performance claim."
        ),
        "code_hash": code_hash,
        "comparison": comparison,
        "compile_seconds": compile_seconds,
        "device_count": jax.device_count(),
        "device_kind": sorted({item.device_kind for item in jax.devices()}),
        "diagnostic_only": True,
        "execute_seconds": execute_seconds,
        "format_version": FORMAT_VERSION,
        "hlo": {
            "byte_count": hlo_path.stat().st_size,
            "contract": hlo_contract,
            "filename": hlo_path.relative_to(args.output).as_posix(),
            "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
            "sha256": _sha256_file(hlo_path),
        },
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "memory_after": _memory_stats(device),
        "memory_before": before_memory,
        "numerical_contract": {
            "cache_dtype": "bfloat16",
            "input_rms_norm_epsilon": 1e-5,
            "key_layer_norm_epsilon": contract.key_layer_norm_epsilon,
            "linear_backend": "production_pallas_raw_fp8",
            "one_live_row": True,
            "prompt_token_count": int(prompt_ids.size),
            "rotary_dim": contract.rotary_dim,
            "rotary_interleaved": contract.interleaved_rotary,
            "rotary_theta": contract.theta,
        },
        "performance_claim": False,
        "prompt_cache_manifest_sha256": cache_manifest["manifest_sha256"],
        "run_tag": args.run_tag,
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "observed_bfloat16_sha256": _array_sha256(observed_bits),
            "sha256": _sha256_file(tensor_path),
        },
    }
    result["manifest_sha256"] = _manifest_hash(result)
    (args.output / "comparison.json").write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "elementwise_exact": comparison["elementwise_exact"],
                "first_mismatch_position": comparison[
                    "first_mismatch_position"
                ],
                "manifest_sha256": result["manifest_sha256"],
                "mismatch_count": comparison["mismatch_count"],
                "mismatched_position_count": comparison[
                    "mismatched_position_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
