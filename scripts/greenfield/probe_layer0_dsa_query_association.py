#!/usr/bin/env python3
"""Discriminate physical layer-0 DSA query-projection associations on TPU."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import jax
from jax import lax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.kernels.reference.dsa_association import (  # noqa: E402
    Layer0DsaProbeGeometry,
    apply_rotary,
    rotary_cos_sin,
)
from glm_tpu.greenfield.kernels.reference.fp8 import (  # noqa: E402
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (  # noqa: E402
    Fp8BlockMatmulConfig,
    fp8_block_matmul_f32,
    fp8_block_vector_matmul_f32,
)
from glm_tpu.greenfield.validation.layer0_dsa_association import (  # noqa: E402
    inspect_distributed_q_a_norm_artifact,
    inspect_layer0_dsa_association_input,
)


ARTIFACT_KIND = "glm52_layer0_dsa_query_association"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _compare(actual: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if actual.shape != candidate.shape or actual.dtype != candidate.dtype:
        raise ValueError("query association comparison contract drifted")
    delta = candidate.astype(np.float32) - actual.astype(np.float32)
    absolute = np.abs(delta)
    return {
        "actual_sha256": _array_sha256(actual),
        "candidate_sha256": _array_sha256(candidate),
        "elementwise_exact": bool(np.array_equal(actual, candidate)),
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()),
        "mean_signed": float(delta.mean()),
        "mismatch_count": int(np.count_nonzero(actual != candidate)),
        "p99_abs": float(np.percentile(absolute, 99)),
        "shape": list(actual.shape),
    }


def _dot_rows(lhs: Any, weight: Any, *, highest: bool = False) -> Any:
    precision = lax.Precision.HIGHEST if highest else None
    return lax.dot_general(
        lhs.astype(jnp.float32),
        weight.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=precision,
        preferred_element_type=jnp.float32,
    )


def _apply_query_rope(projected: Any, positions: Any) -> Any:
    geometry = Layer0DsaProbeGeometry()
    query = projected.reshape(
        projected.shape[0], geometry.heads, geometry.head_dim
    )
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=geometry.rotary_dim,
        theta=geometry.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        query[..., : geometry.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=True,
    )
    return jnp.concatenate(
        (rotated, query[..., geometry.rotary_dim :]), axis=-1
    ).astype(jnp.float32)


def _global_query(q_state: Any, weight: Any, positions: Any) -> Any:
    return _apply_query_rope(_dot_rows(q_state, weight), positions)


def _global_query_one_row(q_state: Any, weight: Any, positions: Any) -> Any:
    return _apply_query_rope(
        _dot_rows(q_state[:1], weight), positions[:1]
    )


def _mapped_heads(
    q_state: Any,
    weight: Any,
    positions: Any,
    *,
    mapping: str,
    one_row: bool = False,
    highest: bool = False,
) -> Any:
    geometry = Layer0DsaProbeGeometry()
    query = q_state[:1] if one_row else q_state
    selected_positions = positions[:1] if one_row else positions
    head_weight = weight.reshape(
        geometry.heads, geometry.head_dim, geometry.q_lora_rank
    )

    def project(one_weight: Any) -> Any:
        return _dot_rows(query, one_weight, highest=highest)

    if mapping == "lax_map":
        head_major = lax.map(project, head_weight)
    elif mapping == "vmap":
        head_major = jax.vmap(project)(head_weight)
    elif mapping == "unrolled":
        head_major = jnp.stack(
            tuple(project(head_weight[index]) for index in range(geometry.heads))
        )
    elif mapping == "fori":
        initial = jnp.zeros(
            (geometry.heads, query.shape[0], geometry.head_dim),
            dtype=jnp.float32,
        )

        def body(index: Any, output: Any) -> Any:
            current = lax.dynamic_index_in_dim(
                head_weight, index, axis=0, keepdims=False
            )
            return output.at[index].set(project(current))

        head_major = lax.fori_loop(0, geometry.heads, body, initial)
    else:
        raise ValueError(f"unknown query-head mapping {mapping!r}")
    projected = jnp.transpose(head_major, (1, 0, 2)).reshape(
        query.shape[0], geometry.heads * geometry.head_dim
    )
    return _apply_query_rope(projected, selected_positions)


def _grouped_heads(
    q_state: Any,
    weight: Any,
    positions: Any,
    *,
    groups: int,
    one_row: bool,
) -> Any:
    geometry = Layer0DsaProbeGeometry()
    query = q_state[:1] if one_row else q_state
    selected_positions = positions[:1] if one_row else positions
    output_width = geometry.heads * geometry.head_dim
    if output_width % groups:
        raise ValueError("query output width does not divide into groups")
    group_width = output_width // groups
    grouped = weight.reshape(groups, group_width, geometry.q_lora_rank)
    projected = jnp.concatenate(
        tuple(_dot_rows(query, grouped[index]) for index in range(groups)),
        axis=-1,
    )
    return _apply_query_rope(projected, selected_positions)


def _candidate_functions() -> dict[str, Callable[[Any, Any, Any], Any]]:
    return {
        "global_m32_n4096": _global_query,
        "global_m1_n4096": _global_query_one_row,
        "head_lax_map_m32_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="lax_map"
        ),
        "head_vmap_m32_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="vmap"
        ),
        "head_unrolled_m32_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="unrolled"
        ),
        "head_fori_m32_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="fori"
        ),
        "head_lax_map_highest_m32_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="lax_map", highest=True
        ),
        "head_lax_map_m1_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="lax_map", one_row=True
        ),
        "head_unrolled_m1_n128": lambda q, w, p: _mapped_heads(
            q, w, p, mapping="unrolled", one_row=True
        ),
        "lp4_unrolled_m32_n1024": lambda q, w, p: _grouped_heads(
            q, w, p, groups=4, one_row=False
        ),
        "lp4_unrolled_m1_n1024": lambda q, w, p: _grouped_heads(
            q, w, p, groups=4, one_row=True
        ),
    }


def _pallas_query(
    q_state: Any,
    weight_bits: Any,
    weight_scale: Any,
    positions: Any,
    *,
    groups: int,
) -> Any:
    """Run the current production raw-FP8 query kernel at one/LP4 ownership."""

    geometry = Layer0DsaProbeGeometry()
    output_width = geometry.heads * geometry.head_dim
    if output_width % groups or weight_scale.shape[0] % groups:
        raise ValueError("Pallas query weight does not divide into groups")
    group_width = output_width // groups
    scale_rows = weight_scale.shape[0] // groups
    outputs = []
    for index in range(groups):
        outputs.append(
            fp8_block_matmul_f32(
                q_state[:1],
                lax.dynamic_slice_in_dim(
                    weight_bits,
                    index * group_width,
                    group_width,
                    axis=0,
                ),
                lax.dynamic_slice_in_dim(
                    weight_scale,
                    index * scale_rows,
                    scale_rows,
                    axis=0,
                ),
                config=Fp8BlockMatmulConfig(
                    output_tile=128,
                    contraction_tile=128,
                ),
            )
        )
    return _apply_query_rope(jnp.concatenate(outputs, axis=-1), positions[:1])


def _pallas_vector_query(
    q_state: Any,
    weight_bits: Any,
    weight_scale: Any,
    positions: Any,
    *,
    groups: int,
) -> Any:
    """Run the one-row raw-FP8 vector reduction at global/LP4 ownership."""

    geometry = Layer0DsaProbeGeometry()
    output_width = geometry.heads * geometry.head_dim
    if output_width % groups or weight_scale.shape[0] % groups:
        raise ValueError("Pallas vector weight does not divide into groups")
    group_width = output_width // groups
    scale_rows = weight_scale.shape[0] // groups
    outputs = []
    for index in range(groups):
        outputs.append(
            fp8_block_vector_matmul_f32(
                q_state[:1],
                lax.dynamic_slice_in_dim(
                    weight_bits,
                    index * group_width,
                    group_width,
                    axis=0,
                ),
                lax.dynamic_slice_in_dim(
                    weight_scale,
                    index * scale_rows,
                    scale_rows,
                    axis=0,
                ),
            )
        )
    return _apply_query_rope(jnp.concatenate(outputs, axis=-1), positions[:1])


def _pallas_candidate_functions() -> dict[str, Callable[..., Any]]:
    return {
        "pallas_global_m1_n4096": lambda q, b, s, p: _pallas_query(
            q, b, s, p, groups=1
        ),
        "pallas_lp4_m1_n1024": lambda q, b, s, p: _pallas_query(
            q, b, s, p, groups=4
        ),
        "pallas_vector_global_m1_n4096": lambda q, b, s, p: (
            _pallas_vector_query(q, b, s, p, groups=1)
        ),
        "pallas_vector_lp4_m1_n1024": lambda q, b, s, p: (
            _pallas_vector_query(q, b, s, p, groups=4)
        ),
    }


def _hlo_contract(hlo: str, *, candidate: str) -> dict[str, Any]:
    lowered = hlo.lower()
    forbidden = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "collective-permute",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    expected_width = "128" if "n128" in candidate else (
        "1024" if "n1024" in candidate else "4096"
    )
    decoded_overlay_shapes = [
        shape
        for shape in ("f32[1024,2048]", "f32[4096,2048]")
        if shape in lowered
    ]
    vector_candidate = "pallas_vector" in candidate
    has_vector_dequantizer = (
        "greenfield_fp8_dequantize_f32_n128_k2048" in lowered
    )
    return {
        "candidate": candidate,
        "expected_physical_output_width": int(expected_width),
        "forbidden_operations": forbidden,
        "decoded_overlay_shapes": decoded_overlay_shapes,
        "has_vector_dequantizer": has_vector_dequantizer,
        "hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "passed": (
            not forbidden
            and (not vector_candidate or has_vector_dequantizer)
            and (not vector_candidate or not decoded_overlay_shapes)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-comparison-sha256", required=True)
    parser.add_argument("--capture-tensors-sha256", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--distributed-q-a-norm-dir", type=Path, required=True)
    parser.add_argument("--q-a-manifest-sha256", required=True)
    parser.add_argument("--q-a-code-hash", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    args = parser.parse_args()

    code_hash = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if code_hash != args.expected_code_hash:
        raise SystemExit("query association code hash drifted")
    if args.output.exists() or args.hlo_dir.exists():
        raise FileExistsError("append-only query association output exists")

    comparison_path = args.capture_dir / "comparison.json"
    tensors_path = args.capture_dir / "internals.npz"
    if _file_sha256(comparison_path) != args.capture_comparison_sha256 or (
        _file_sha256(tensors_path) != args.capture_tensors_sha256
    ):
        raise SystemExit("captured DSA internal identity drifted")
    capture = json.loads(comparison_path.read_text())
    if capture["first_divergent_field"] != "query" or (
        not capture["fields"]["normalized_hidden"]["elementwise_exact"]
        or not capture["fields"]["q_a_state"]["elementwise_exact"]
        or capture["runtime"]["backend"] != "tpu"
    ):
        raise SystemExit("captured DSA query discriminator contract drifted")
    with np.load(tensors_path, allow_pickle=False) as payload:
        actual_query = np.asarray(payload["actual__query"], dtype=np.float32)
        captured_q_bits = np.ascontiguousarray(payload["actual__q_a_state"])
    if actual_query.shape != (32, 128) or captured_q_bits.shape != (2048,):
        raise SystemExit("captured query/q-a shape drifted")

    input_manifest, arrays = inspect_layer0_dsa_association_input(
        args.input_dir,
        expected_manifest_sha256=args.input_manifest_sha256,
    )
    q_manifest, q_bits = inspect_distributed_q_a_norm_artifact(
        args.distributed_q_a_norm_dir,
        expected_manifest_sha256=args.q_a_manifest_sha256,
        expected_code_hash=args.q_a_code_hash,
        expected_input_manifest_sha256=input_manifest["manifest_sha256"],
    )
    if not np.array_equal(q_bits[0], captured_q_bits):
        raise SystemExit("sealed distributed q-a row disagrees with capture")

    q_state = jnp.asarray(
        np.ascontiguousarray(q_bits).view(ml_dtypes.bfloat16)
    )
    weight_bits = jnp.asarray(
        arrays["self_attn__indexer__wq_b__weight"]
    )
    weight_scale = jnp.asarray(
        arrays["self_attn__indexer__wq_b__weight_scale_inv"]
    )
    dequantized = jax.jit(
        lambda bits, scale: dequantize_fp8_bits_block_weight(
            bits, scale, output_dtype=jnp.float32
        )
    )(weight_bits, weight_scale)
    jax.block_until_ready(dequantized)
    dequantized_host = np.asarray(dequantized)
    if int(dequantized_host.view(np.uint8).sum(dtype=np.uint64)) != 3765880530:
        raise SystemExit("adapted wq_b byte identity drifted")

    geometry = Layer0DsaProbeGeometry()
    positions = jnp.zeros((geometry.decode_rows,), dtype=jnp.int32).at[0].set(
        jnp.int32(8155)
    )
    args.hlo_dir.mkdir(parents=True)
    records: dict[str, Any] = {}
    for name, function in _candidate_functions().items():
        compiled = jax.jit(function).lower(
            q_state, dequantized, positions
        ).compile()
        hlo = compiled.as_text()
        contract = _hlo_contract(hlo, candidate=name)
        if not contract["passed"]:
            raise SystemExit(f"query association HLO failed: {name}")
        output = compiled(q_state, dequantized, positions)
        jax.block_until_ready(output)
        candidate = np.asarray(output[0], dtype=np.float32)
        records[name] = {
            "comparison": _compare(actual_query, candidate),
            "hlo": contract,
        }
        (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(hlo)
    for name, function in _pallas_candidate_functions().items():
        compiled = jax.jit(function).lower(
            q_state, weight_bits, weight_scale, positions
        ).compile()
        hlo = compiled.as_text()
        contract = _hlo_contract(hlo, candidate=name)
        if not contract["passed"]:
            raise SystemExit(f"query association HLO failed: {name}")
        output = compiled(q_state, weight_bits, weight_scale, positions)
        jax.block_until_ready(output)
        candidate = np.asarray(output[0], dtype=np.float32)
        records[name] = {
            "comparison": _compare(actual_query, candidate),
            "hlo": contract,
        }
        (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(hlo)

    exact = sorted(
        name
        for name, record in records.items()
        if record["comparison"]["elementwise_exact"]
    )
    result = {
        "artifact_kind": ARTIFACT_KIND,
        "status": "SUCCESS",
        "format_version": 1,
        "code_hash": code_hash,
        "diagnostic_only": True,
        "performance_claim": False,
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "device_kind": sorted({device.device_kind for device in jax.devices()}),
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "tensors_sha256": args.capture_tensors_sha256,
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "query_sha256": _array_sha256(actual_query),
            "q_a_sha256": _array_sha256(captured_q_bits),
        },
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "adapted_wq_b": {
            "shape": list(dequantized_host.shape),
            "dtype": str(dequantized_host.dtype),
            "byte_sum": int(
                dequantized_host.view(np.uint8).sum(dtype=np.uint64)
            ),
            "sha256": _array_sha256(dequantized_host),
        },
        "candidates": records,
        "exact_candidates": exact,
        "association_restored": bool(exact),
        "claim_scope": (
            "Bounded layer-0 query arithmetic only; no decoder, Gate-D, "
            "latency, or token-rate claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
