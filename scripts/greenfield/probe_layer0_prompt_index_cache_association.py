#!/usr/bin/env python3
"""Isolate prompt-key projection/chunk and key-LayerNorm association."""

from __future__ import annotations

import argparse
import gzip
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
ARTIFACT_KIND = "greenfield_layer0_prompt_index_cache_association"
FORMAT_VERSION = 1
BASELINE_ARTIFACT_KIND = "greenfield_layer0_prompt_index_cache_comparison"
ACCEPTED_ADAPTED_WK_BYTE_SUM = 193_298_069


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


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
    parser.add_argument("--baseline-comparison-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-comparison-manifest-sha256", required=True
    )
    parser.add_argument(
        "--candidate-set",
        choices=("matrix", "chunk_parameter"),
        default="matrix",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _classify(exact: set[str]) -> str:
    chunk_parameter = (
        "accepted_xla_m2048_chunk_parameter_divide_sqrt" in exact
    )
    if chunk_parameter:
        return "chunk_parameter_association_sufficient"
    pallas_divide = "production_pallas_m1_divide_sqrt" in exact
    xla_divide = "accepted_xla_m2048_divide_sqrt" in exact
    xla_rsqrt = "accepted_xla_m2048_multiply_rsqrt" in exact
    if pallas_divide and not xla_rsqrt:
        return "key_layer_norm_association_sufficient"
    if xla_rsqrt and not pallas_divide:
        return "projection_or_chunk_association_sufficient"
    if xla_divide and not pallas_divide and not xla_rsqrt:
        return "projection_chunk_and_key_layer_norm_both_required"
    if not exact:
        return "declared_matrix_not_sufficient"
    return "multiple_exact_candidates_interaction_not_uniquely_identified"


def main() -> int:
    args = _parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} "
            f"found={code_hash}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    from jax import lax
    import jax.numpy as jnp
    import ml_dtypes
    from safetensors.numpy import load_file, save_file

    from glm_tpu.greenfield.kernels.pallas import (
        Fp8BlockMatmulConfig,
        fp8_block_matmul_f32,
    )
    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        affine_key_layer_norm,
        layer0_prompt_index_key_chunk,
        layer0_prompt_index_keys_chunked,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm
    from glm_tpu.greenfield.kernels.reference.rotary import (
        apply_rotary,
        rotary_cos_sin,
    )
    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_layer0_dsa_association_input,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        compare_prompt_index_key_bits,
        inspect_legacy_prompt_index_cache,
        validate_prompt_index_key_association_hlo,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            "prompt-key association probe requires TPU, got "
            f"{jax.default_backend()}"
        )
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "prompt-key association probe requires one four-chip TPU-v4 host"
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
    baseline_path = args.baseline_comparison_dir / "comparison.json"
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline.get("artifact_kind") != BASELINE_ARTIFACT_KIND
        or _manifest_hash(baseline)
        != args.baseline_comparison_manifest_sha256
        or baseline.get("manifest_sha256")
        != args.baseline_comparison_manifest_sha256
        or baseline.get("prompt_cache_manifest_sha256")
        != cache_manifest["manifest_sha256"]
        or baseline.get("input_manifest_sha256")
        != input_manifest["manifest_sha256"]
        or baseline.get("comparison", {}).get("elementwise_exact") is not False
    ):
        raise RuntimeError("baseline prompt-cache comparison identity drifted")
    baseline_tensor_path = (
        args.baseline_comparison_dir
        / baseline["tensor_file"]["filename"]
    )
    if _sha256_file(baseline_tensor_path) != baseline["tensor_file"]["sha256"]:
        raise RuntimeError("baseline prompt-cache tensor file drifted")
    baseline_arrays = load_file(str(baseline_tensor_path))
    baseline_bits = np.asarray(
        baseline_arrays["observed_prompt_index_key_bfloat16_bits"]
    )
    if compare_prompt_index_key_bits(expected_bits, baseline_bits) != baseline[
        "comparison"
    ]:
        raise RuntimeError("baseline prompt-cache comparison content drifted")
    if cache_manifest["layer0_input_manifest_sha256"] != input_manifest[
        "manifest_sha256"
    ]:
        raise RuntimeError("prompt cache and layer-0 input disagree")

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
    raw_wk_bits = put(arrays["self_attn__indexer__wk__weight"])
    raw_wk_scale = put(arrays["self_attn__indexer__wk__weight_scale_inv"])
    key_norm_weight = put(
        bf16_host("self_attn__indexer__k_norm__weight")
    )
    key_norm_bias = put(bf16_host("self_attn__indexer__k_norm__bias"))
    geometry = Layer0DsaProbeGeometry()
    pallas_config = Fp8BlockMatmulConfig(
        block_shape=(128, 128),
        output_tile=128,
        contraction_tile=128,
    )

    dequantize_bf16 = partial(
        dequantize_fp8_bits_block_weight,
        output_dtype=jnp.bfloat16,
    )
    started = time.monotonic()
    dequant_compiled = jax.jit(dequantize_bf16).lower(
        raw_wk_bits, raw_wk_scale
    ).compile()
    dequant_compile_seconds = time.monotonic() - started
    wk_bf16 = dequant_compiled(raw_wk_bits, raw_wk_scale)
    jax.block_until_ready(wk_bf16)
    wk_fp32 = jax.device_put(wk_bf16.astype(jnp.float32), device)
    jax.block_until_ready(wk_fp32)
    wk_host = np.ascontiguousarray(np.asarray(wk_fp32, dtype=np.float32))
    wk_identity = {
        "byte_sum": int(wk_host.view(np.uint8).sum(dtype=np.uint64)),
        "dtype": str(wk_host.dtype),
        "sha256": _array_sha256(wk_host),
        "shape": list(wk_host.shape),
        "source_association": "raw FP8 -> BF16 -> FP32 accepted adapter",
    }
    if (
        wk_identity["shape"] != [128, 6144]
        or wk_identity["dtype"] != "float32"
        or wk_identity["byte_sum"] != ACCEPTED_ADAPTED_WK_BYTE_SUM
    ):
        raise RuntimeError(f"accepted adapted wk identity drifted: {wk_identity}")

    def pallas_divide_prompt_keys(
        embeddings: Any,
        row_indices: Any,
        norm_weight: Any,
        wk_bits: Any,
        wk_scale: Any,
        k_norm_weight: Any,
        k_norm_bias: Any,
    ) -> Any:
        positions = jnp.arange(row_indices.shape[0], dtype=jnp.int32)

        def scan_step(completed: Any, values: tuple[Any, Any]) -> tuple[Any, Any]:
            row_index, position = values
            residual = lax.dynamic_index_in_dim(
                embeddings, row_index, axis=0, keepdims=True
            )
            normalized = rms_norm(residual, norm_weight, epsilon=1e-5)
            projected = fp8_block_matmul_f32(
                normalized,
                wk_bits,
                wk_scale,
                config=pallas_config,
                interpret=False,
            )
            key = affine_key_layer_norm(
                projected,
                k_norm_weight,
                k_norm_bias,
                epsilon=geometry.key_norm_epsilon,
                mode="divide_sqrt",
            )
            cos, sin = rotary_cos_sin(
                position[None],
                rotary_dim=geometry.rotary_dim,
                theta=geometry.theta,
                dtype=jnp.float32,
            )
            rotated = apply_rotary(
                key[:, : geometry.rotary_dim],
                cos,
                sin,
                interleaved=True,
            )
            stored = jnp.concatenate(
                (rotated, key[:, geometry.rotary_dim :]), axis=-1
            ).astype(jnp.bfloat16)
            return completed + jnp.int32(1), stored[0]

        completed, keys = lax.scan(
            scan_step,
            jnp.int32(0),
            (row_indices, positions),
            unroll=1,
        )
        return keys, completed

    xla_divide = partial(
        layer0_prompt_index_keys_chunked,
        geometry=geometry,
        key_norm_mode="divide_sqrt",
    )
    xla_rsqrt = partial(
        layer0_prompt_index_keys_chunked,
        geometry=geometry,
        key_norm_mode="multiply_rsqrt",
    )
    chunk_inputs: tuple[tuple[Any, Any], ...] | None = None
    if args.candidate_set == "matrix":
        definitions = {
            "production_pallas_m1_divide_sqrt": (
                pallas_divide_prompt_keys,
                (
                    unique_embeddings,
                    prompt_rows,
                    input_norm_weight,
                    raw_wk_bits,
                    raw_wk_scale,
                    key_norm_weight,
                    key_norm_bias,
                ),
                True,
            ),
            "accepted_xla_m2048_divide_sqrt": (
                xla_divide,
                (
                    unique_embeddings,
                    prompt_rows,
                    input_norm_weight,
                    wk_fp32,
                    key_norm_weight,
                    key_norm_bias,
                ),
                False,
            ),
            "accepted_xla_m2048_multiply_rsqrt": (
                xla_rsqrt,
                (
                    unique_embeddings,
                    prompt_rows,
                    input_norm_weight,
                    wk_fp32,
                    key_norm_weight,
                    key_norm_bias,
                ),
                False,
            ),
        }
    else:
        padded_tokens = (
            (prompt_ids.size + geometry.prompt_chunk - 1)
            // geometry.prompt_chunk
            * geometry.prompt_chunk
        )
        prompt_hidden_host = bf16_host(
            "unique_embedding_bfloat16_bits"
        )[prompt_rows_host]
        prompt_hidden_host = np.pad(
            prompt_hidden_host,
            ((0, padded_tokens - prompt_ids.size), (0, 0)),
        ).reshape(-1, geometry.prompt_chunk, geometry.hidden_size)
        position_host = np.arange(padded_tokens, dtype=np.int32).reshape(
            -1, geometry.prompt_chunk
        )
        chunk_inputs = tuple(
            (put(prompt_hidden_host[index]), put(position_host[index]))
            for index in range(prompt_hidden_host.shape[0])
        )
        chunk_function = partial(
            layer0_prompt_index_key_chunk,
            geometry=geometry,
            key_norm_mode="divide_sqrt",
        )
        first_hidden, first_positions = chunk_inputs[0]
        definitions = {
            "accepted_xla_m2048_chunk_parameter_divide_sqrt": (
                chunk_function,
                (
                    first_hidden,
                    first_positions,
                    input_norm_weight,
                    wk_fp32,
                    key_norm_weight,
                    key_norm_bias,
                ),
                False,
            )
        }
    before_memory = _memory_stats(device)
    candidate_records: dict[str, Any] = {}
    candidate_tensors: dict[str, np.ndarray] = {}
    hlo_dir = args.output / "hlo"
    hlo_dir.mkdir()
    for name, (function, arguments, returns_counter) in definitions.items():
        started = time.monotonic()
        compiled = jax.jit(function).lower(*arguments).compile()
        compile_seconds = time.monotonic() - started
        hlo = compiled.as_text()
        contract = validate_prompt_index_key_association_hlo(
            hlo,
            candidate=name,
            prompt_token_count=int(prompt_ids.size),
            unique_token_count=int(unique_ids.size),
        )
        if not contract["passed"]:
            raise RuntimeError(f"prompt-key association HLO failed: {name}: {contract}")
        started = time.monotonic()
        result = compiled(*arguments)
        if chunk_inputs is not None:
            chunk_results = [result]
            for hidden_chunk, positions_chunk in chunk_inputs[1:]:
                chunk_results.append(
                    compiled(
                        hidden_chunk,
                        positions_chunk,
                        *arguments[2:],
                    )
                )
            jax.block_until_ready(chunk_results)
            keys = jnp.concatenate(chunk_results, axis=0)[
                : prompt_ids.size
            ]
            jax.block_until_ready(keys)
        elif returns_counter:
            keys, completed = result
            jax.block_until_ready((keys, completed))
            if int(np.asarray(completed)) != prompt_ids.size:
                raise RuntimeError(f"prompt-key candidate scan incomplete: {name}")
        else:
            keys = result
            jax.block_until_ready(keys)
        execute_seconds = time.monotonic() - started
        bits = np.ascontiguousarray(np.asarray(keys)).view(np.uint16)
        if bits.shape != expected_bits.shape:
            raise RuntimeError(f"prompt-key candidate shape drifted: {name}")
        candidate_tensors[f"{name}_bfloat16_bits"] = bits
        comparison = compare_prompt_index_key_bits(expected_bits, bits)
        baseline_delta = compare_prompt_index_key_bits(baseline_bits, bits)
        hlo_path = hlo_dir / f"{name}.optimized_hlo.txt.gz"
        with gzip.open(hlo_path, "wt", encoding="utf-8") as stream:
            stream.write(hlo)
        candidate_records[name] = {
            "comparison_to_accepted": comparison,
            "comparison_to_production_baseline": baseline_delta,
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            "hlo": {
                "byte_count": hlo_path.stat().st_size,
                "contract": contract,
                "filename": hlo_path.relative_to(args.output).as_posix(),
                "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
                "sha256": _sha256_file(hlo_path),
            },
            "observed_bfloat16_sha256": _array_sha256(bits),
        }

    tensor_path = args.output / "candidate_prompt_index_keys.safetensors"
    save_file(
        candidate_tensors,
        str(tensor_path),
        metadata={
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(FORMAT_VERSION),
        },
    )
    exact_candidates = sorted(
        name
        for name, record in candidate_records.items()
        if record["comparison_to_accepted"]["elementwise_exact"]
    )
    result = {
        "accepted_adapted_wk": wk_identity,
        "artifact_kind": ARTIFACT_KIND,
        "backend": jax.default_backend(),
        "baseline": {
            "comparison_manifest_sha256": baseline["manifest_sha256"],
            "comparison_to_accepted": baseline["comparison"],
            "observed_bfloat16_sha256": baseline["comparison"][
                "observed_bfloat16_sha256"
            ],
        },
        "candidate_set": args.candidate_set,
        "candidates": candidate_records,
        "claim_scope": (
            "Diagnostic layer-0 prompt-key association only; elapsed times "
            "are not decoder latency and carry no Gate-D/performance claim."
        ),
        "code_hash": code_hash,
        "conclusion": {
            "classification": _classify(set(exact_candidates)),
            "exact_candidates": exact_candidates,
        },
        "dequantize_bf16_compile_seconds": dequant_compile_seconds,
        "device_count": jax.device_count(),
        "device_kind": sorted({item.device_kind for item in jax.devices()}),
        "diagnostic_only": True,
        "format_version": FORMAT_VERSION,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "memory_after": _memory_stats(device),
        "memory_before": before_memory,
        "performance_claim": False,
        "prompt_cache_manifest_sha256": cache_manifest["manifest_sha256"],
        "run_tag": args.run_tag,
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
    }
    result["manifest_sha256"] = _manifest_hash(result)
    result_path = args.output / "association.json"
    result_path.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "classification": result["conclusion"]["classification"],
                "exact_candidates": exact_candidates,
                "manifest_sha256": result["manifest_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
