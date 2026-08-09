#!/usr/bin/env python3
"""Compare one accepted prompt-key producer row with the sealed candidate."""

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
ARTIFACT_KIND = "greenfield_accepted_prompt_key_internal_comparison"
FORMAT_VERSION = 1
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
    capture_source = parser.add_mutually_exclusive_group(required=True)
    capture_source.add_argument("--source-dump-dir", type=Path)
    capture_source.add_argument("--accepted-capture-dir", type=Path)
    parser.add_argument("--accepted-capture-manifest-sha256")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--prompt-cache-dir", type=Path, required=True)
    parser.add_argument("--prompt-cache-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--accepted-run-tag")
    parser.add_argument("--greenfield-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument(
        "--layer-name", default="model.layers.0.self_attn.attn"
    )
    parser.add_argument("--position", type=int, default=113)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--expected-accepted-cache-sha256", required=True)
    parser.add_argument("--expected-candidate-cache-sha256")
    parser.add_argument("--expected-cache-mismatch-count", type=int)
    parser.add_argument(
        "--expected-first-cache-mismatch-position", type=int, default=113
    )
    parser.add_argument(
        "--projection-weight-mode",
        choices=("adapted_bf16", "adapted_fp32"),
        default="adapted_bf16",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    accepted_run_tag = args.accepted_run_tag or args.run_tag
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.greenfield_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.greenfield_code_hash} "
            f"found={code_hash}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from glm_tpu.greenfield.kernels.reference.dsa_association import (
        Layer0DsaProbeGeometry,
        layer0_prompt_index_key_gather_cache_chunk,
        layer0_prompt_index_key_gather_cache_states_chunk,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.validation import (
        LegacyPromptKeyInternalConfig,
        compare_prompt_key_internal_states,
        compare_prompt_index_key_bits,
        inspect_layer0_dsa_association_input,
        inspect_legacy_prompt_index_cache,
        inspect_legacy_prompt_key_internal_capture,
        inspect_prompt_key_internal_capture_artifact,
        validate_prompt_index_key_association_hlo,
    )

    if jax.default_backend() != "tpu":
        raise RuntimeError(
            "prompt-key internal comparison requires TPU, got "
            f"{jax.default_backend()}"
        )
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "prompt-key internal comparison requires one four-chip TPU host"
        )
    if args.position < 0 or args.position >= 2048:
        raise RuntimeError("prompt-key comparison position must be in chunk zero")

    device = jax.local_devices()[0]
    if args.source_dump_dir is not None:
        if args.accepted_capture_manifest_sha256 is not None:
            raise ValueError(
                "accepted capture manifest is only valid with an artifact"
            )
        accepted_capture_dir = args.output / "accepted_capture"
        capture, accepted_states = inspect_legacy_prompt_key_internal_capture(
            LegacyPromptKeyInternalConfig(
                source_dump_dir=args.source_dump_dir,
                output_dir=accepted_capture_dir,
                expected_run_tag=accepted_run_tag,
                expected_legacy_code_hash=args.legacy_code_hash,
                expected_oracle_pin=args.oracle_pin,
                expected_layer_name=args.layer_name,
                expected_position=args.position,
                expected_process_count=args.process_count,
                expected_model_id=args.model_id,
            )
        )
        capture_source_kind = "raw_observer_dumps"
    else:
        if args.accepted_capture_manifest_sha256 is None:
            raise ValueError(
                "accepted capture artifact requires its manifest identity"
            )
        assert args.accepted_capture_dir is not None
        capture, accepted_states = inspect_prompt_key_internal_capture_artifact(
            args.accepted_capture_dir,
            expected_manifest_sha256=args.accepted_capture_manifest_sha256,
        )
        expected_capture_identity = {
            "legacy_code_hash": args.legacy_code_hash,
            "oracle_pin": args.oracle_pin,
            "layer_name": args.layer_name,
            "position": args.position,
            "process_count": args.process_count,
            "model_id": args.model_id,
            "run_tag": accepted_run_tag,
        }
        if any(
            capture.get(name) != expected
            for name, expected in expected_capture_identity.items()
        ):
            raise RuntimeError("accepted prompt-key capture lineage drifted")
        capture_source_kind = "sealed_capture_artifact"
    input_manifest, arrays = inspect_layer0_dsa_association_input(
        args.input_dir,
        expected_manifest_sha256=args.input_manifest_sha256,
    )
    cache_manifest, accepted_cache_bits = inspect_legacy_prompt_index_cache(
        args.prompt_cache_dir,
        expected_manifest_sha256=args.prompt_cache_manifest_sha256,
    )
    if cache_manifest["layer0_input_manifest_sha256"] != input_manifest[
        "manifest_sha256"
    ]:
        raise RuntimeError("prompt cache and layer-0 input disagree")
    if cache_manifest["prompt_index_key_bfloat16_sha256"] != (
        args.expected_accepted_cache_sha256
    ) or _array_sha256(accepted_cache_bits) != (
        args.expected_accepted_cache_sha256
    ):
        raise RuntimeError("accepted prompt-cache identity drifted")

    cache_shape = tuple(cache_manifest["source_layout"]["global_cache_shape"])
    live_block_table_host = np.asarray(
        cache_manifest["source_layout"]["live_block_table"], dtype=np.int32
    )
    if cache_shape != (24, 16, 32, 128) or (
        live_block_table_host.shape != (16,)
    ):
        raise RuntimeError("accepted prompt-cache geometry drifted")

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
    input_norm_weight = put(bf16_host("input_layernorm__weight"))
    raw_wk_bits = put(arrays["self_attn__indexer__wk__weight"])
    raw_wk_scale = put(arrays["self_attn__indexer__wk__weight_scale_inv"])
    key_norm_weight = put(
        bf16_host("self_attn__indexer__k_norm__weight")
    )
    key_norm_bias = put(bf16_host("self_attn__indexer__k_norm__bias"))
    live_block_table = put(live_block_table_host)
    geometry = Layer0DsaProbeGeometry()

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
    if wk_identity["shape"] != [128, 6144] or (
        wk_identity["dtype"] != "float32"
    ) or wk_identity["byte_sum"] != ACCEPTED_ADAPTED_WK_BYTE_SUM:
        raise RuntimeError(f"accepted adapted wk identity drifted: {wk_identity}")

    padded_tokens = (
        (prompt_ids.size + geometry.prompt_chunk - 1)
        // geometry.prompt_chunk
        * geometry.prompt_chunk
    )
    padded_rows = np.pad(
        prompt_rows_host, (0, padded_tokens - prompt_ids.size)
    ).reshape(-1, geometry.prompt_chunk)
    positions = np.arange(padded_tokens, dtype=np.int32).reshape(
        -1, geometry.prompt_chunk
    )
    initial_cache = put(np.zeros(cache_shape, dtype=ml_dtypes.bfloat16))
    chunk_suffixes = tuple(
        (
            live_block_table,
            unique_embeddings,
            put(padded_rows[index]),
            put(positions[index]),
            input_norm_weight,
            wk_fp32,
            key_norm_weight,
            key_norm_bias,
        )
        for index in range(padded_rows.shape[0])
    )
    first_arguments = (initial_cache, *chunk_suffixes[0])
    states_function = partial(
        layer0_prompt_index_key_gather_cache_states_chunk,
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
        projection_weight_mode=args.projection_weight_mode,
    )
    cache_function = partial(
        layer0_prompt_index_key_gather_cache_chunk,
        geometry=geometry,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
        projection_weight_mode=args.projection_weight_mode,
    )

    before_memory = _memory_stats(device)
    started = time.monotonic()
    states_compiled = jax.jit(states_function, donate_argnums=(0,)).lower(
        *first_arguments
    ).compile()
    states_compile_seconds = time.monotonic() - started
    started = time.monotonic()
    cache_compiled = jax.jit(cache_function, donate_argnums=(0,)).lower(
        *first_arguments
    ).compile()
    cache_compile_seconds = time.monotonic() - started

    hlo_dir = args.output / "hlo"
    hlo_dir.mkdir()
    state_hlo = states_compiled.as_text()
    cache_hlo = cache_compiled.as_text()
    state_hlo_path = hlo_dir / "prompt_key_states.optimized_hlo.txt.gz"
    cache_hlo_path = hlo_dir / "prompt_key_cache.optimized_hlo.txt.gz"
    for path, text in ((state_hlo_path, state_hlo), (cache_hlo_path, cache_hlo)):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    weight_label = (
        "bf16_weight"
        if args.projection_weight_mode == "adapted_bf16"
        else "fp32_weight"
    )
    candidate_prefix = (
        "accepted_xla_m2048_gather_cache_write_"
        f"{weight_label}_divide_sqrt_source_rope"
    )
    state_candidate = f"{candidate_prefix}_states"
    cache_candidate = candidate_prefix
    state_contract = validate_prompt_index_key_association_hlo(
        state_hlo,
        candidate=state_candidate,
        prompt_token_count=int(prompt_ids.size),
        unique_token_count=int(unique_ids.size),
    )
    cache_contract = validate_prompt_index_key_association_hlo(
        cache_hlo,
        candidate=cache_candidate,
        prompt_token_count=int(prompt_ids.size),
        unique_token_count=int(unique_ids.size),
    )
    if not state_contract["passed"] or not cache_contract["passed"]:
        (hlo_dir / "contract_failure.json").write_text(
            json.dumps(
                {"cache": cache_contract, "states": state_contract},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise RuntimeError("prompt-key producer HLO contract failed")

    started = time.monotonic()
    state_result = states_compiled(*first_arguments)
    jax.block_until_ready(state_result)
    observed_states = {
        "pre_layer_norm_key": np.ascontiguousarray(
            np.asarray(state_result.pre_layer_norm_key[args.position])
        ),
        "pre_rope_key": np.ascontiguousarray(
            np.asarray(state_result.pre_rope_key[args.position])
        ),
        "post_rope_key": np.ascontiguousarray(
            np.asarray(state_result.post_rope_key[args.position])
        ),
    }
    candidate_cache = state_result.index_cache
    for suffix in chunk_suffixes[1:]:
        candidate_cache = cache_compiled(candidate_cache, *suffix)
    jax.block_until_ready(candidate_cache)
    execute_seconds = time.monotonic() - started

    candidate_cache_host = np.ascontiguousarray(np.asarray(candidate_cache))
    flat_cache = candidate_cache_host.reshape(
        cache_shape[0], cache_shape[1] * cache_shape[2], cache_shape[3]
    )
    live_positions = np.arange(prompt_ids.size, dtype=np.int32)
    physical_pages = live_block_table_host[
        live_positions // (cache_shape[1] * cache_shape[2])
    ]
    candidate_keys = np.ascontiguousarray(
        flat_cache[
            physical_pages,
            live_positions % (cache_shape[1] * cache_shape[2]),
        ]
    )
    candidate_bits = candidate_keys.view(np.uint16)
    candidate_sha = _array_sha256(candidate_bits)
    if args.expected_candidate_cache_sha256 is not None and (
        candidate_sha != args.expected_candidate_cache_sha256
    ):
        raise RuntimeError(
            "sealed DB512 candidate cache identity drifted: "
            f"expected={args.expected_candidate_cache_sha256} "
            f"found={candidate_sha}"
        )
    cache_comparison = compare_prompt_index_key_bits(
        accepted_cache_bits, candidate_bits
    )
    if args.expected_cache_mismatch_count is not None and (
        cache_comparison["mismatch_count"]
        != args.expected_cache_mismatch_count
        or cache_comparison["first_mismatch_position"]
        != args.expected_first_cache_mismatch_position
    ):
        raise RuntimeError(
            f"sealed DB512 mismatch identity drifted: {cache_comparison}"
        )

    state_comparison = compare_prompt_key_internal_states(
        accepted_states, observed_states
    )
    accepted_post_bits = np.ascontiguousarray(
        accepted_states["post_rope_key"].astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    observed_post_bits = np.ascontiguousarray(
        observed_states["post_rope_key"].astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    accepted_cache_row = np.ascontiguousarray(
        accepted_cache_bits[args.position]
    )
    observed_cache_row = np.ascontiguousarray(candidate_bits[args.position])
    if not np.array_equal(accepted_post_bits, accepted_cache_row):
        raise RuntimeError(
            "accepted prompt-key observer does not reproduce its cache row"
        )
    if not np.array_equal(observed_post_bits, observed_cache_row):
        raise RuntimeError(
            "greenfield prompt-key producer does not reproduce its cache row"
        )

    tensor_path = args.output / "prompt_key_internal_comparison.npz"
    np.savez(
        tensor_path,
        accepted_cache_row_bfloat16_bits=accepted_cache_row,
        accepted_post_rope_bfloat16_bits=accepted_post_bits,
        accepted_post_rope_key=accepted_states["post_rope_key"],
        accepted_pre_layer_norm_key=accepted_states["pre_layer_norm_key"],
        accepted_pre_rope_key=accepted_states["pre_rope_key"],
        greenfield_cache_row_bfloat16_bits=observed_cache_row,
        greenfield_post_rope_bfloat16_bits=observed_post_bits,
        greenfield_post_rope_key=observed_states["post_rope_key"],
        greenfield_pre_layer_norm_key=observed_states["pre_layer_norm_key"],
        greenfield_pre_rope_key=observed_states["pre_rope_key"],
    )

    result: dict[str, Any] = {
        "accepted_cache": {
            "manifest_sha256": cache_manifest["manifest_sha256"],
            "post_rope_cast_matches_position": True,
            "prompt_index_key_bfloat16_sha256": (
                args.expected_accepted_cache_sha256
            ),
        },
        "accepted_capture": {
            "capture_source_kind": capture_source_kind,
            "capture_process_indices": capture["capture_process_indices"],
            "manifest_sha256": capture["manifest_sha256"],
            "tensor_file_sha256": capture["tensor_file"]["sha256"],
            "source_run_tag": capture["run_tag"],
        },
        "accepted_adapted_wk": wk_identity,
        "artifact_kind": ARTIFACT_KIND,
        "backend": jax.default_backend(),
        "cache_comparison": cache_comparison,
        "claim_scope": (
            "Diagnostic layer-0 prompt-key producer association only; "
            "elapsed times are not decoder latency or performance proof."
        ),
        "code_hash": code_hash,
        "conclusion": {
            "classification": state_comparison["classification"],
            "first_divergent_field": state_comparison[
                "first_divergent_field"
            ],
        },
        "device_count": jax.device_count(),
        "device_kind": sorted({item.device_kind for item in jax.devices()}),
        "diagnostic_only": True,
        "execute_seconds": execute_seconds,
        "format_version": FORMAT_VERSION,
        "greenfield_cache": {
            "post_rope_cast_matches_position": True,
            "prompt_index_key_bfloat16_sha256": candidate_sha,
        },
        "hlo": {
            "cache": {
                "compile_seconds": cache_compile_seconds,
                "contract": cache_contract,
                "filename": cache_hlo_path.relative_to(args.output).as_posix(),
                "optimized_hlo_sha256": sha256(cache_hlo.encode()).hexdigest(),
                "sha256": _sha256_file(cache_hlo_path),
            },
            "states": {
                "compile_seconds": states_compile_seconds,
                "contract": state_contract,
                "filename": state_hlo_path.relative_to(args.output).as_posix(),
                "optimized_hlo_sha256": sha256(state_hlo.encode()).hexdigest(),
                "sha256": _sha256_file(state_hlo_path),
            },
        },
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "layer_name": args.layer_name,
        "legacy_code_hash": args.legacy_code_hash,
        "memory_after": _memory_stats(device),
        "memory_before": before_memory,
        "model_id": args.model_id,
        "oracle_pin": args.oracle_pin,
        "performance_claim": False,
        "position": args.position,
        "projection_weight_mode": args.projection_weight_mode,
        "run_tag": args.run_tag,
        "state_comparison": state_comparison,
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _sha256_file(tensor_path),
        },
        "wk_dequantize_compile_seconds": dequant_compile_seconds,
    }
    result["manifest_sha256"] = _manifest_hash(result)
    result_path = args.output / "comparison.json"
    result_path.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "classification": result["conclusion"]["classification"],
                "first_divergent_field": result["conclusion"][
                    "first_divergent_field"
                ],
                "manifest_sha256": result["manifest_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
