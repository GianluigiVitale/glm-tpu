#!/usr/bin/env python3
"""Compare one accepted prompt-key producer row with the sealed candidate."""

from __future__ import annotations

import argparse
import gzip
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
ARTIFACT_KIND = "greenfield_accepted_prompt_key_internal_comparison"
FORMAT_VERSION = 2
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
    parser.add_argument(
        "--projection-weight-source",
        choices=(
            "materialized_lp4_stage_local",
            "materialized_parameter",
            "raw_fp8_inside_executable",
        ),
        default="materialized_parameter",
    )
    parser.add_argument(
        "--projection-mapping-mode",
        choices=(
            "logical_m2048",
            "physical_m64_lax_map",
            "physical_m64_projection_keynorm_lax_map",
        ),
        default="logical_m2048",
    )
    parser.add_argument(
        "--capture-mode",
        choices=("prompt_key", "prompt_key_input"),
        default="prompt_key",
    )
    return parser.parse_args()


def _projection_weight_source_contract(
    optimized_hlo: str,
    *,
    source: str,
) -> dict[str, Any]:
    """Distinguish DB518's parameter boundary from fused raw-FP8 repair."""

    entry = next(
        (line for line in optimized_hlo.splitlines() if line.startswith("ENTRY ")),
        "",
    )
    f32_parameter_count = len(re.findall(r"f32\[128,6144\]", entry))
    raw_fp8_parameter_count = len(re.findall(r"u8\[128,6144\]", entry))
    bf16_round_count = len(
        re.findall(
            r"= bf16\[(?:128,6144|786432)\]"
            r"(?:\{[^}\n]*\})? convert\([^\n]+\)"
            r"[^\n]*op_name=\"[^\"]*convert_element_type",
            optimized_hlo,
        )
    )
    if source in (
        "materialized_lp4_stage_local",
        "materialized_parameter",
    ):
        violations = []
        if f32_parameter_count != 1:
            violations.append(
                "materialized projection must expose one entry FP32 wk parameter"
            )
        if raw_fp8_parameter_count:
            violations.append(
                "materialized projection must not expose raw FP8 wk at entry"
            )
    elif source == "raw_fp8_inside_executable":
        violations = []
        if f32_parameter_count:
            violations.append(
                "internal projection must not expose an entry FP32 wk parameter"
            )
        if raw_fp8_parameter_count != 1:
            violations.append(
                "internal projection must expose one raw FP8 wk parameter"
            )
        if bf16_round_count < 1:
            violations.append(
                "internal projection lost its explicit BF16 adaptation round"
            )
    else:
        raise ValueError(f"unknown projection weight source {source!r}")
    return {
        "bf16_round_count": bf16_round_count,
        "entry": entry,
        "entry_f32_wk_parameter_count": f32_parameter_count,
        "entry_raw_fp8_wk_parameter_count": raw_fp8_parameter_count,
        "passed": not violations,
        "source": source,
        "violations": violations,
    }


def _run_lp4_materialized_repair(
    *,
    output_dir: Path,
    raw_wk_bits: np.ndarray,
    raw_wk_scale: np.ndarray,
    normalized_history: np.ndarray,
    live_block_table: np.ndarray,
    key_norm_weight: np.ndarray,
    key_norm_bias: np.ndarray,
    accepted_cache_bits: np.ndarray,
    expected_materialized_wk_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the production materializer and cache repair over four TPU lanes."""

    import jax
    import ml_dtypes
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        materialize_stage_local_prefill_index_wk,
        repair_stage_local_prompt_index_cache,
    )
    from glm_tpu.greenfield.runtime import (
        validate_prefill_index_weight_materialization_hlo,
        validate_stage_local_prefill_index_repair_hlo,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    devices = np.asarray(jax.local_devices(), dtype=object)
    if devices.shape != (4,):
        raise RuntimeError("LP4 prompt repair requires four local TPU devices")
    if normalized_history.shape != (accepted_cache_bits.shape[0], 6144):
        raise RuntimeError("LP4 prompt repair normalized history drifted")
    mesh = Mesh(devices, ("lp",))
    sharded_weight = NamedSharding(mesh, P("lp", None, None))
    replicated = NamedSharding(mesh, P())

    bits = jax.device_put(
        np.broadcast_to(raw_wk_bits, (4, *raw_wk_bits.shape)).copy(),
        sharded_weight,
    )
    scales = jax.device_put(
        np.broadcast_to(raw_wk_scale, (4, *raw_wk_scale.shape)).copy(),
        sharded_weight,
    )

    def mapped_materializer(local_bits: Any, local_scales: Any) -> Any:
        return materialize_stage_local_prefill_index_wk(
            local_bits[0], local_scales[0]
        )[None, ...]

    materializer = jax.shard_map(
        mapped_materializer,
        mesh=mesh,
        in_specs=(P("lp", None, None), P("lp", None, None)),
        out_specs=P("lp", None, None),
        check_vma=False,
    )
    compile_started = time.monotonic()
    lowered_materializer = jax.jit(materializer).lower(bits, scales)
    compiled_materializer = lowered_materializer.compile()
    materializer_compile_seconds = time.monotonic() - compile_started
    materializer_hlo = compiled_materializer.as_text()

    config = SimpleNamespace(
        hidden_size=6144,
        index_key_width=128,
        maximum_full_indexer_slots=1,
        total_devices=4,
    )
    decoder = SimpleNamespace(config=config)
    materializer_contract = validate_prefill_index_weight_materialization_hlo(
        materializer_hlo,
        decoder=decoder,
    )
    hlo_dir = output_dir / "hlo"
    hlo_dir.mkdir(parents=True, exist_ok=True)
    hlo_records = {}

    def record_hlo(
        name: str, hlo: str, contract: dict[str, Any]
    ) -> None:
        path = hlo_dir / f"{name}.optimized_hlo.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(hlo)
        hlo_records[name] = {
            "contract": contract,
            "filename": path.relative_to(output_dir).as_posix(),
            "optimized_hlo_sha256": sha256(hlo.encode()).hexdigest(),
            "sha256": _sha256_file(path),
        }

    record_hlo(
        "lp4_wk_materializer", materializer_hlo, materializer_contract
    )
    if not materializer_contract["passed"]:
        raise RuntimeError("LP4 weight materialization HLO contract failed")

    sentinel = ml_dtypes.bfloat16(-32.0)
    cache_host = np.full((4, 24, 128, 128), sentinel)
    cache = jax.device_put(
        cache_host,
        NamedSharding(mesh, P("lp", None, None, None)),
    )
    history = jax.device_put(normalized_history, replicated)
    blocks = jax.device_put(live_block_table[None, :], replicated)
    norm_weight = jax.device_put(key_norm_weight, replicated)
    norm_bias = jax.device_put(key_norm_bias, replicated)

    def mapped_repair_stage_local_prompt_index_cache(
        local_cache: Any,
        local_history: Any,
        local_blocks: Any,
        local_wk: Any,
        local_norm_weight: Any,
        local_norm_bias: Any,
    ) -> Any:
        repaired = repair_stage_local_prompt_index_cache(
            local_cache[0],
            local_history,
            local_blocks,
            local_wk[0],
            local_norm_weight,
            local_norm_bias,
            lax.axis_index("lp"),
        )
        return repaired[None, ...]

    repair = jax.shard_map(
        mapped_repair_stage_local_prompt_index_cache,
        mesh=mesh,
        in_specs=(
            P("lp", None, None, None),
            P(),
            P(),
            P("lp", None, None),
            P(),
            P(),
        ),
        out_specs=P("lp", None, None, None),
        check_vma=False,
    )
    execute_started = time.monotonic()
    materialized = compiled_materializer(bits, scales)
    jax.block_until_ready(materialized)
    materializer_execute_seconds = time.monotonic() - execute_started
    compile_started = time.monotonic()
    lowered_repair = jax.jit(repair).lower(
        cache,
        history,
        blocks,
        materialized,
        norm_weight,
        norm_bias,
    )
    compiled_repair = lowered_repair.compile()
    repair_compile_seconds = time.monotonic() - compile_started
    repair_hlo = compiled_repair.as_text()

    program = SimpleNamespace(
        decoder=decoder,
        prompt_length=int(normalized_history.shape[0]),
    )
    schedule = SimpleNamespace(
        stages=(
            SimpleNamespace(
                layers=(SimpleNamespace(indexer_kind="full"),)
            ),
        )
    )
    repair_contract = validate_stage_local_prefill_index_repair_hlo(
        parse_hlo_module(repair_hlo),
        program=program,
        schedule=schedule,
        backend_contract="tpu_v4_spmd",
    )
    record_hlo("lp4_cache_repair", repair_hlo, repair_contract)
    if not repair_contract["passed"]:
        raise RuntimeError("LP4 cache repair HLO contract failed")

    execute_started = time.monotonic()
    repaired = compiled_repair(
        cache,
        history,
        blocks,
        materialized,
        norm_weight,
        norm_bias,
    )
    jax.block_until_ready(repaired)
    repair_execute_seconds = time.monotonic() - execute_started
    repaired_host = np.ascontiguousarray(np.asarray(repaired))
    repaired_bits = repaired_host.view(np.uint16)
    sentinel_bits = np.asarray(sentinel).view(np.uint16).item()
    changed_rows = np.any(repaired_bits != sentinel_bits, axis=-1)
    expected_changed_rows = np.zeros((4, 24, 128), dtype=np.bool_)
    candidate_bits = np.empty_like(accepted_cache_bits)
    for position in range(accepted_cache_bits.shape[0]):
        logical_page, page_row = divmod(position, 512)
        physical_page = int(live_block_table[logical_page])
        owner, local_row = divmod(page_row, 128)
        if physical_page < 0 or physical_page >= 24:
            raise RuntimeError("LP4 prompt repair block table is invalid")
        expected_changed_rows[owner, physical_page, local_row] = True
        candidate_bits[position] = repaired_bits[
            owner, physical_page, local_row
        ]
    owner_isolation_exact = bool(
        np.array_equal(changed_rows, expected_changed_rows)
    )
    if not owner_isolation_exact:
        raise RuntimeError("LP4 prompt repair wrote outside owner rows")

    materialized_shards = []
    for shard in materialized.addressable_shards:
        host = np.ascontiguousarray(np.asarray(jax.device_get(shard.data)))
        identity = {
            "byte_count": int(host.nbytes),
            "device_id": int(shard.device.id),
            "sha256": _array_sha256(host),
        }
        if identity["sha256"] != expected_materialized_wk_sha256:
            raise RuntimeError("LP4 materialized wk differs across owner lanes")
        materialized_shards.append(identity)

    return candidate_bits, {
        "assembled_cache_elementwise_exact": bool(
            np.array_equal(candidate_bits, accepted_cache_bits)
        ),
        "assembled_cache_sha256": _array_sha256(candidate_bits),
        "hlo": hlo_records,
        "local_device_ids": [int(device.id) for device in devices],
        "materialized_shards": materialized_shards,
        "materializer_compile_seconds": materializer_compile_seconds,
        "materializer_execute_seconds": materializer_execute_seconds,
        "owner_isolation_exact": owner_isolation_exact,
        "per_lane_written_rows": [
            int(changed_rows[lane].sum()) for lane in range(4)
        ],
        "repair_compile_seconds": repair_compile_seconds,
        "repair_execute_seconds": repair_execute_seconds,
    }


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
        layer0_prompt_normalized_hidden_gather_chunk,
    )
    from glm_tpu.greenfield.kernels.reference.fp8 import (
        dequantize_fp8_bits_block_weight,
    )
    from glm_tpu.greenfield.validation import (
        LegacyPromptKeyInternalConfig,
        compare_prompt_key_internal_states,
        compare_prompt_index_key_bits,
        compare_prompt_projection_input,
        inspect_layer0_dsa_association_input,
        inspect_legacy_prompt_index_cache,
        inspect_legacy_prompt_key_internal_capture,
        inspect_prompt_key_internal_capture_artifact,
        validate_prompt_index_key_association_hlo,
        validate_prompt_projection_input_hlo,
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
                expected_capture_mode=args.capture_mode,
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
            expected_capture_mode=args.capture_mode,
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
    accepted_projection_input = accepted_states.get("projection_input")
    accepted_key_states = {
        name: accepted_states[name]
        for name in (
            "pre_layer_norm_key",
            "pre_rope_key",
            "post_rope_key",
        )
    }
    if (args.capture_mode == "prompt_key_input") != (
        accepted_projection_input is not None
    ):
        raise RuntimeError("accepted projection-input capture mode drifted")
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
    common_options = {
        "geometry": geometry,
        "key_norm_mode": "divide_sqrt",
        "rotary_mode": "accepted_source",
        "projection_weight_mode": args.projection_weight_mode,
        "projection_mapping_mode": args.projection_mapping_mode,
    }
    if args.projection_weight_source in (
        "materialized_lp4_stage_local",
        "materialized_parameter",
    ):
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
        states_function = partial(
            layer0_prompt_index_key_gather_cache_states_chunk,
            **common_options,
        )
        cache_function = partial(
            layer0_prompt_index_key_gather_cache_chunk,
            **common_options,
        )
    else:
        chunk_suffixes = tuple(
            (
                live_block_table,
                unique_embeddings,
                put(padded_rows[index]),
                put(positions[index]),
                input_norm_weight,
                raw_wk_bits,
                raw_wk_scale,
                key_norm_weight,
                key_norm_bias,
            )
            for index in range(padded_rows.shape[0])
        )

        def adapted_wk(bits: Any, scale: Any) -> Any:
            return dequantize_fp8_bits_block_weight(
                bits,
                scale,
                output_dtype=jnp.bfloat16,
            ).astype(jnp.float32)

        def states_function(
            cache: Any,
            blocks: Any,
            embeddings: Any,
            rows: Any,
            chunk_positions: Any,
            norm_weight: Any,
            bits: Any,
            scale: Any,
            key_weight: Any,
            key_bias: Any,
        ) -> Any:
            return layer0_prompt_index_key_gather_cache_states_chunk(
                cache,
                blocks,
                embeddings,
                rows,
                chunk_positions,
                norm_weight,
                adapted_wk(bits, scale),
                key_weight,
                key_bias,
                **common_options,
            )

        def cache_function(
            cache: Any,
            blocks: Any,
            embeddings: Any,
            rows: Any,
            chunk_positions: Any,
            norm_weight: Any,
            bits: Any,
            scale: Any,
            key_weight: Any,
            key_bias: Any,
        ) -> Any:
            return layer0_prompt_index_key_gather_cache_chunk(
                cache,
                blocks,
                embeddings,
                rows,
                chunk_positions,
                norm_weight,
                adapted_wk(bits, scale),
                key_weight,
                key_bias,
                **common_options,
            )

    first_arguments = (initial_cache, *chunk_suffixes[0])
    projection_input_function = partial(
        layer0_prompt_normalized_hidden_gather_chunk,
        geometry=geometry,
    )
    projection_input_arguments = (
        unique_embeddings,
        chunk_suffixes[0][2],
        input_norm_weight,
    )

    before_memory = _memory_stats(device)
    projection_input_compiled = None
    projection_input_compile_seconds = None
    if accepted_projection_input is not None:
        started = time.monotonic()
        projection_input_compiled = jax.jit(
            projection_input_function
        ).lower(*projection_input_arguments).compile()
        projection_input_compile_seconds = time.monotonic() - started
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
    hlo_payloads = [(state_hlo_path, state_hlo), (cache_hlo_path, cache_hlo)]
    projection_input_hlo = None
    projection_input_hlo_path = None
    if projection_input_compiled is not None:
        projection_input_hlo = projection_input_compiled.as_text()
        projection_input_hlo_path = (
            hlo_dir / "prompt_projection_input.optimized_hlo.txt.gz"
        )
        hlo_payloads.append((projection_input_hlo_path, projection_input_hlo))
    for path, text in hlo_payloads:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    weight_label = (
        "bf16_weight"
        if args.projection_weight_mode == "adapted_bf16"
        else "fp32_weight"
    )
    projection_labels = {
        "logical_m2048": "m2048",
        "physical_m64_lax_map": "m64_lax_map",
        "physical_m64_projection_keynorm_lax_map": (
            "m64_projection_keynorm_lax_map"
        ),
    }
    projection_label = projection_labels[args.projection_mapping_mode]
    candidate_prefix = (
        f"accepted_xla_{projection_label}_gather_cache_write_"
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
    projection_input_contract = None
    if projection_input_hlo is not None:
        projection_input_contract = validate_prompt_projection_input_hlo(
            projection_input_hlo,
            unique_token_count=int(unique_ids.size),
        )
    state_weight_source_contract = _projection_weight_source_contract(
        state_hlo, source=args.projection_weight_source
    )
    cache_weight_source_contract = _projection_weight_source_contract(
        cache_hlo, source=args.projection_weight_source
    )
    if (
        not state_contract["passed"]
        or not cache_contract["passed"]
        or not state_weight_source_contract["passed"]
        or not cache_weight_source_contract["passed"]
        or (
            projection_input_contract is not None
            and not projection_input_contract["passed"]
        )
    ):
        (hlo_dir / "contract_failure.json").write_text(
            json.dumps(
                {
                    "cache": cache_contract,
                    "cache_weight_source": cache_weight_source_contract,
                    "projection_input": projection_input_contract,
                    "states": state_contract,
                    "states_weight_source": state_weight_source_contract,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise RuntimeError("prompt-key producer HLO contract failed")

    started = time.monotonic()
    observed_projection_input = None
    projection_input_host = None
    if projection_input_compiled is not None:
        projection_input_result = projection_input_compiled(
            *projection_input_arguments
        )
        projection_input_host = np.ascontiguousarray(
            np.asarray(projection_input_result, dtype=np.float32)
        )
        observed_projection_input = np.ascontiguousarray(
            projection_input_host[args.position]
        )
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
    lp4_materialized_repair = None
    if args.projection_weight_source == "materialized_lp4_stage_local":
        if projection_input_compiled is None or projection_input_host is None:
            raise RuntimeError("LP4 prompt repair requires projection inputs")
        normalized_chunks = [
            projection_input_host.astype(ml_dtypes.bfloat16)
        ]
        for suffix in chunk_suffixes[1:]:
            normalized_chunks.append(
                np.ascontiguousarray(
                    np.asarray(
                        projection_input_compiled(
                            unique_embeddings,
                            suffix[2],
                            input_norm_weight,
                        ),
                        dtype=np.float32,
                    )
                ).astype(ml_dtypes.bfloat16)
            )
        normalized_history = np.ascontiguousarray(
            np.concatenate(normalized_chunks, axis=0)[: prompt_ids.size]
        )
        candidate_bits, lp4_materialized_repair = (
            _run_lp4_materialized_repair(
                output_dir=args.output,
                raw_wk_bits=arrays["self_attn__indexer__wk__weight"],
                raw_wk_scale=arrays[
                    "self_attn__indexer__wk__weight_scale_inv"
                ],
                normalized_history=normalized_history,
                live_block_table=live_block_table_host,
                key_norm_weight=bf16_host(
                    "self_attn__indexer__k_norm__weight"
                ),
                key_norm_bias=bf16_host(
                    "self_attn__indexer__k_norm__bias"
                ),
                accepted_cache_bits=accepted_cache_bits,
                expected_materialized_wk_sha256=wk_identity["sha256"],
            )
        )
    else:
        candidate_cache = state_result.index_cache
        for suffix in chunk_suffixes[1:]:
            candidate_cache = cache_compiled(candidate_cache, *suffix)
        jax.block_until_ready(candidate_cache)
        candidate_cache_host = np.ascontiguousarray(
            np.asarray(candidate_cache)
        )
        flat_cache = candidate_cache_host.reshape(
            cache_shape[0],
            cache_shape[1] * cache_shape[2],
            cache_shape[3],
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
    execute_seconds = time.monotonic() - started
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
        accepted_key_states, observed_states
    )
    projection_input_comparison = None
    if accepted_projection_input is not None:
        assert observed_projection_input is not None
        projection_input_comparison = compare_prompt_projection_input(
            accepted_projection_input,
            observed_projection_input,
        )
    classification = state_comparison["classification"]
    first_divergent_field = state_comparison["first_divergent_field"]
    if projection_input_comparison is not None:
        if not projection_input_comparison["elementwise_exact"]:
            classification = "projection_input_association"
            first_divergent_field = "projection_input"
        elif first_divergent_field == "pre_layer_norm_key":
            classification = "projection_lowering_association"
    accepted_post_bits = np.ascontiguousarray(
        accepted_key_states["post_rope_key"].astype(ml_dtypes.bfloat16)
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
    tensor_arrays = {
        "accepted_cache_row_bfloat16_bits": accepted_cache_row,
        "accepted_post_rope_bfloat16_bits": accepted_post_bits,
        "accepted_post_rope_key": accepted_key_states["post_rope_key"],
        "accepted_pre_layer_norm_key": accepted_key_states[
            "pre_layer_norm_key"
        ],
        "accepted_pre_rope_key": accepted_key_states["pre_rope_key"],
        "greenfield_cache_row_bfloat16_bits": observed_cache_row,
        "greenfield_post_rope_bfloat16_bits": observed_post_bits,
        "greenfield_post_rope_key": observed_states["post_rope_key"],
        "greenfield_pre_layer_norm_key": observed_states[
            "pre_layer_norm_key"
        ],
        "greenfield_pre_rope_key": observed_states["pre_rope_key"],
    }
    if accepted_projection_input is not None:
        assert observed_projection_input is not None
        tensor_arrays.update(
            accepted_projection_input=accepted_projection_input,
            greenfield_projection_input=observed_projection_input,
        )
    np.savez(tensor_path, **tensor_arrays)

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
            "capture_mode": capture["capture_mode"],
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
            "classification": classification,
            "first_divergent_field": first_divergent_field,
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
                "weight_source_contract": cache_weight_source_contract,
            },
            "states": {
                "compile_seconds": states_compile_seconds,
                "contract": state_contract,
                "filename": state_hlo_path.relative_to(args.output).as_posix(),
                "optimized_hlo_sha256": sha256(state_hlo.encode()).hexdigest(),
                "sha256": _sha256_file(state_hlo_path),
                "weight_source_contract": state_weight_source_contract,
            },
        },
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "layer_name": args.layer_name,
        "legacy_code_hash": args.legacy_code_hash,
        "lp4_materialized_repair": lp4_materialized_repair,
        "memory_after": _memory_stats(device),
        "memory_before": before_memory,
        "model_id": args.model_id,
        "oracle_pin": args.oracle_pin,
        "performance_claim": False,
        "position": args.position,
        "projection_weight_mode": args.projection_weight_mode,
        "projection_weight_source": args.projection_weight_source,
        "projection_mapping_mode": args.projection_mapping_mode,
        "projection_input_comparison": projection_input_comparison,
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
    if projection_input_hlo is not None:
        assert projection_input_hlo_path is not None
        result["hlo"]["projection_input"] = {
            "compile_seconds": projection_input_compile_seconds,
            "contract": projection_input_contract,
            "filename": projection_input_hlo_path.relative_to(
                args.output
            ).as_posix(),
            "optimized_hlo_sha256": sha256(
                projection_input_hlo.encode()
            ).hexdigest(),
            "sha256": _sha256_file(projection_input_hlo_path),
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
