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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import ml_dtypes
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.kernels.reference.dsa_association import (  # noqa: E402
    Layer0DsaProbeGeometry,
    apply_rotary,
    one_row_virtual_tp32_fused_qkv_a_rms_norm,
    pack_legacy_fused_qkv_runtime_weights,
    rotary_cos_sin,
)
from glm_tpu.greenfield.kernels.layer import (  # noqa: E402
    AttentionFp8Weights,
    _project_attention_qkv_a,
)
from glm_tpu.greenfield.kernels.reference.attention import (  # noqa: E402
    MlaNumericalContract,
)
from glm_tpu.greenfield.kernels.reference.dsa import (  # noqa: E402
    DsaNumericalContract,
)
from glm_tpu.greenfield.kernels.reference.fp8 import (  # noqa: E402
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (  # noqa: E402
    Fp8BlockMatmulConfig,
    fp8_block_matmul_f32,
    fp8_block_vector_matmul_f32,
)
from glm_tpu.greenfield.runtime.decoder import (  # noqa: E402
    _validate_fused_qkv_a_decoder_association,
)
from glm_tpu.greenfield.validation.layer0_dsa_association import (  # noqa: E402
    inspect_distributed_q_a_norm_artifact,
    inspect_layer0_dsa_association_input,
)


ARTIFACT_KIND = "glm52_layer0_dsa_query_association"
Q_A_ARTIFACT_KIND = "glm52_layer0_q_a_association"
PRODUCTION_QKV_A_ARTIFACT_KIND = (
    "glm52_layer0_qkv_a_production_association"
)
PHYSICAL_LP4_QUERY_ARTIFACT_KIND = (
    "glm52_layer0_physical_lp4_dsa_query_association"
)
PHYSICAL_LP4_Q_A_BOUNDARY_ARTIFACT_KIND = (
    "glm52_layer0_physical_lp4_dsa_q_a_boundary_association"
)
PHYSICAL_LP4_HEAD_GEOMETRY_ARTIFACT_KIND = (
    "glm52_layer0_physical_lp4_dsa_head_geometry_association"
)


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


def _compare_bfloat16_bits(
    actual: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    if actual.shape != candidate.shape or (
        actual.dtype != np.uint16 or candidate.dtype != np.uint16
    ):
        raise ValueError("q-a association comparison contract drifted")
    actual_value = actual.view(ml_dtypes.bfloat16).astype(np.float32)
    candidate_value = candidate.view(ml_dtypes.bfloat16).astype(np.float32)
    delta = candidate_value - actual_value
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
        "storage_dtype": "uint16",
        "value_dtype": "bfloat16",
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


def _local_query_projection(
    q_state: Any,
    weight: Any,
    *,
    head_dim: int,
    association: str,
) -> Any:
    """Project one physical owner's heads with an explicit association."""

    if q_state.ndim != 2 or q_state.shape[0] != 1:
        raise ValueError("physical query projection requires one live row")
    if weight.ndim != 2 or weight.shape[1] != q_state.shape[1]:
        raise ValueError("physical query owner weight has an invalid shape")
    if weight.shape[0] % head_dim:
        raise ValueError("physical query owner width does not divide into heads")
    if association == "owner_dot":
        return _dot_rows(q_state, weight)
    if association == "head_unrolled":
        heads = weight.shape[0] // head_dim
        head_weight = weight.reshape(heads, head_dim, weight.shape[1])
        head_major = jnp.stack(
            tuple(_dot_rows(q_state, head_weight[index]) for index in range(heads))
        )
        return jnp.transpose(head_major, (1, 0, 2)).reshape(
            1, weight.shape[0]
        )
    raise ValueError(f"unknown physical query association {association!r}")


def _serial_head_query_projection(
    q_state: Any,
    owner_weight: Any,
    *,
    head_dim: int,
) -> Any:
    """Project one owner's heads through a device-resident N128 loop."""

    if q_state.ndim != 2 or q_state.shape[0] != 1:
        raise ValueError("serial physical query requires one live row")
    if owner_weight.ndim != 2 or owner_weight.shape[1] != q_state.shape[1]:
        raise ValueError("serial physical query owner weight has an invalid shape")
    if owner_weight.shape[0] % head_dim:
        raise ValueError("serial physical query owner width is not head-aligned")
    heads = owner_weight.shape[0] // head_dim
    head_weight = owner_weight.reshape(heads, head_dim, owner_weight.shape[1])
    initial = jnp.zeros((heads, head_dim), dtype=jnp.float32)

    def body(index: Any, output: Any) -> Any:
        selected = lax.dynamic_index_in_dim(
            head_weight, index, axis=0, keepdims=False
        )
        projected = _dot_rows(q_state, selected)[0]
        return lax.dynamic_update_index_in_dim(
            output, projected, index, axis=0
        )

    return lax.fori_loop(0, heads, body, initial).reshape(
        1, owner_weight.shape[0]
    )


def _apply_local_query_rope(projected: Any, positions: Any) -> Any:
    """Apply the accepted RoPE to one physical owner's contiguous heads."""

    geometry = Layer0DsaProbeGeometry()
    if projected.ndim != 2 or projected.shape[0] != 1:
        raise ValueError("physical query projection must contain one row")
    if projected.shape[1] % geometry.head_dim:
        raise ValueError("physical query width does not divide into heads")
    local_heads = projected.shape[1] // geometry.head_dim
    query = projected.reshape(1, local_heads, geometry.head_dim)
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
    ).astype(jnp.float32)[0]


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


def _raw_lookup_query(
    q_state: Any,
    weight_bits: Any,
    weight_scale: Any,
    positions: Any,
    *,
    groups: int,
) -> Any:
    """Run source-faithful raw-FP8 lookup/dequant in unrolled N=128 tiles."""

    geometry = Layer0DsaProbeGeometry()
    output_width = geometry.heads * geometry.head_dim
    if output_width % groups or weight_scale.shape[0] % groups:
        raise ValueError("Raw lookup weight does not divide into groups")
    group_width = output_width // groups
    scale_rows = weight_scale.shape[0] // groups
    outputs = []
    for group_index in range(groups):
        group_bits = lax.dynamic_slice_in_dim(
            weight_bits,
            group_index * group_width,
            group_width,
            axis=0,
        )
        group_scale = lax.dynamic_slice_in_dim(
            weight_scale,
            group_index * scale_rows,
            scale_rows,
            axis=0,
        )
        for tile_index in range(group_width // 128):
            tile_bits = lax.dynamic_slice_in_dim(
                group_bits,
                tile_index * 128,
                128,
                axis=0,
            )
            tile_scale = lax.dynamic_slice_in_dim(
                group_scale,
                tile_index,
                1,
                axis=0,
            )
            decoded = dequantize_fp8_bits_block_weight(
                tile_bits,
                tile_scale,
                output_dtype=jnp.float32,
            )
            outputs.append(_dot_rows(q_state[:1], decoded))
    return _apply_query_rope(jnp.concatenate(outputs, axis=-1), positions[:1])


def _raw_materialized_query(
    q_state: Any,
    weight_bits: Any,
    weight_scale: Any,
    positions: Any,
    *,
    groups: int,
) -> Any:
    """Dequantize one complete owner shard behind an explicit boundary."""

    geometry = Layer0DsaProbeGeometry()
    output_width = geometry.heads * geometry.head_dim
    if output_width % groups or weight_scale.shape[0] % groups:
        raise ValueError("Materialized weight does not divide into groups")
    group_width = output_width // groups
    scale_rows = weight_scale.shape[0] // groups
    outputs = []
    for index in range(groups):
        bits = lax.dynamic_slice_in_dim(
            weight_bits,
            index * group_width,
            group_width,
            axis=0,
        )
        scale = lax.dynamic_slice_in_dim(
            weight_scale,
            index * scale_rows,
            scale_rows,
            axis=0,
        )
        decoded = lax.optimization_barrier(
            dequantize_fp8_bits_block_weight(
                bits,
                scale,
                output_dtype=jnp.float32,
            )
        )
        outputs.append(_dot_rows(q_state[:1], decoded))
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
        "raw_lookup_global_m1_n128_tiles": lambda q, b, s, p: (
            _raw_lookup_query(q, b, s, p, groups=1)
        ),
        "raw_lookup_lp4_m1_n128_tiles": lambda q, b, s, p: (
            _raw_lookup_query(q, b, s, p, groups=4)
        ),
        "raw_materialized_global_m1_n4096": lambda q, b, s, p: (
            _raw_materialized_query(q, b, s, p, groups=1)
        ),
        "raw_materialized_lp4_m1_n1024": lambda q, b, s, p: (
            _raw_materialized_query(q, b, s, p, groups=4)
        ),
    }


def _hlo_contract(hlo: str, *, candidate: str) -> dict[str, Any]:
    lowered = hlo.lower()
    forbidden = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
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
    streamed_candidate = (
        "pallas_vector" in candidate or "raw_lookup" in candidate
    )
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
            and (
                "pallas_vector" not in candidate or has_vector_dequantizer
            )
            and (not streamed_candidate or not decoded_overlay_shapes)
        ),
    }


def _physical_lp4_candidate_modes() -> tuple[tuple[str, str, str], ...]:
    """Return source/association pairs for the real four-device discriminator."""

    return (
        ("physical_raw_owner_dot_m1_n1024", "raw", "owner_dot"),
        (
            "physical_raw_head_unrolled_m1_n128",
            "raw",
            "head_unrolled",
        ),
        (
            "physical_predecoded_owner_dot_m1_n1024",
            "predecoded",
            "owner_dot",
        ),
        (
            "physical_predecoded_head_unrolled_m1_n128",
            "predecoded",
            "head_unrolled",
        ),
    )


def _physical_lp4_q_a_boundary_modes() -> tuple[tuple[str, bool, bool], ...]:
    """Return fused q-a round/precision candidates for the physical query."""

    return (
        (
            "physical_fused_q_a_unrounded_default_owner_dot_m1_n1024",
            False,
            False,
        ),
        (
            "physical_fused_q_a_bf16_barrier_default_owner_dot_m1_n1024",
            True,
            False,
        ),
        (
            "physical_fused_q_a_bf16_barrier_highest_owner_dot_m1_n1024",
            True,
            True,
        ),
    )


def _physical_lp4_head_geometry_modes() -> tuple[str, ...]:
    """Return legacy-local and production-compatible N128 candidates."""

    return (
        "physical_single_head_sweep_m1_n128",
        "physical_owner_serial_while_m1_n128",
        "physical_global_gspmd_m1_n4096",
    )


def _physical_lp4_hlo_contract(
    hlo: str,
    *,
    candidate: str,
    source: str,
) -> dict[str, Any]:
    """Require a one-row, one-owner-per-chip program without communication."""

    lowered = hlo.lower()
    forbidden_operations = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    forbidden_global_shapes = [
        shape
        for shape in (
            "u8[4096,2048]",
            "f32[4096,2048]",
            "bf16[32,2048]",
            "f32[32,2048]",
        )
        if shape in lowered
    ]
    head_unrolled = "head_unrolled" in candidate
    required_shapes = {
        "one_live_q_a_row": "bf16[1,2048]" in lowered,
        "local_owner_projection": (
            ("f32[1,128]" in lowered or "f32[128]" in lowered)
            if head_unrolled
            else ("f32[1,1024]" in lowered or "f32[1024]" in lowered)
        ),
        "local_owner_query": "f32[8,128]" in lowered,
        "local_fp32_owner": "f32[1024,2048]" in lowered,
    }
    if source == "raw":
        required_shapes.update(
            {
                "local_raw_owner": "u8[1024,2048]" in lowered,
                "local_raw_scale": "f32[8,16]" in lowered,
            }
        )
    elif source != "predecoded":
        raise ValueError("physical query source is unknown")
    dot_metadata_count = sum(
        "metadata={op_name" in line and "dot_general" in line
        for line in lowered.splitlines()
    )
    return {
        "candidate": candidate,
        "dot_metadata_count": dot_metadata_count,
        "forbidden_global_shapes": forbidden_global_shapes,
        "forbidden_operations": forbidden_operations,
        "hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "local_parallel_size": 4,
        "physical_projection_width": 128 if head_unrolled else 1024,
        "physical_owner_width": 1024,
        "required_shapes": required_shapes,
        "source": source,
        "passed": (
            not forbidden_operations
            and not forbidden_global_shapes
            and all(required_shapes.values())
            and dot_metadata_count > 0
        ),
    }


def _physical_lp4_q_a_boundary_hlo_contract(
    optimized_hlo: str,
    stablehlo: str,
    *,
    backend: str,
    candidate: str,
    enforce_bfloat16_boundary: bool,
    highest: bool,
) -> dict[str, Any]:
    """Require fused q-a plus one physical owner query and its round gate."""

    lowered = optimized_hlo.lower()
    stable_lowered = stablehlo.lower()
    forbidden_operations = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    forbidden_global_query_weights = [
        shape
        for shape in ("u8[4096,2048]", "f32[4096,2048]")
        if shape in lowered
    ]
    required_shapes = {
        "one_live_normalized_row": "bf16[1,6144]" in lowered,
        "one_live_q_a_row": "bf16[1,2048]" in lowered,
        "local_raw_owner": "u8[1024,2048]" in lowered,
        "local_raw_scale": "f32[8,16]" in lowered,
        "local_fp32_owner": "f32[1024,2048]" in lowered,
        "local_owner_projection": (
            "f32[1,1024]" in lowered or "f32[1024]" in lowered
        ),
        "local_owner_query": "f32[8,128]" in lowered,
    }
    barrier_count = stable_lowered.count("stablehlo.optimization_barrier")
    stable_highest = "highest" in stable_lowered
    fused_qkv_a = _validate_fused_qkv_a_decoder_association(
        optimized_hlo,
        layers=1,
    )
    if backend not in ("cpu", "tpu"):
        raise ValueError("physical q-a boundary backend is unknown")
    expected_barriers = 1 + int(enforce_bfloat16_boundary)
    return {
        "backend": backend,
        "candidate": candidate,
        "enforce_bfloat16_boundary": enforce_bfloat16_boundary,
        "forbidden_global_query_weights": forbidden_global_query_weights,
        "forbidden_operations": forbidden_operations,
        "fused_qkv_a": fused_qkv_a,
        "highest": highest,
        "hlo_sha256": sha256(optimized_hlo.encode()).hexdigest(),
        "local_parallel_size": 4,
        "required_shapes": required_shapes,
        "stablehlo_optimization_barrier_count": barrier_count,
        "stablehlo_requests_highest": stable_highest,
        "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
        "passed": (
            not forbidden_operations
            and not forbidden_global_query_weights
            and all(required_shapes.values())
            and (backend != "tpu" or fused_qkv_a["passed"])
            and barrier_count == expected_barriers
            and stable_highest == highest
        ),
    }


def _physical_lp4_head_geometry_hlo_contract(
    hlo: str,
    *,
    candidate: str,
) -> dict[str, Any]:
    """Require a physical N128 dot and only the requested head schedule."""

    if candidate not in _physical_lp4_head_geometry_modes():
        raise ValueError("physical head-geometry candidate is unknown")
    lowered = hlo.lower()
    serial = "serial_while" in candidate
    global_gspmd = "global_gspmd" in candidate
    forbidden_operations = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    forbidden_global_shapes = [
        shape
        for shape in (
            "f32[4096,2048]",
            "bf16[32,2048]",
            "f32[32,2048]",
        )
        if shape in lowered
    ]
    required_shapes = {
        "one_live_q_a_row": "bf16[1,2048]" in lowered,
        "physical_head_projection": (
            ("f32[1,1024]" in lowered or "f32[1024]" in lowered)
            if global_gspmd
            else ("f32[1,128]" in lowered or "f32[128]" in lowered)
        ),
        "local_output": (
            ("f32[1,8,128]" in lowered or "f32[8,128]" in lowered)
            if serial or global_gspmd
            else ("f32[1,128]" in lowered or "f32[128]" in lowered)
        ),
        "local_weight": (
            "f32[1024,2048]" in lowered
            if serial or global_gspmd
            else "f32[128,2048]" in lowered
        ),
    }
    while_count = sum(" while(" in line for line in lowered.splitlines())
    explicit_four_partitions = "num_partitions=4" in lowered
    dot_metadata_count = sum(
        "metadata={op_name" in line and "dot_general" in line
        for line in lowered.splitlines()
    )
    expected_while_count = 1 if serial else 0
    return {
        "candidate": candidate,
        "dot_metadata_count": dot_metadata_count,
        "expected_while_count": expected_while_count,
        "explicit_four_partitions": explicit_four_partitions,
        "forbidden_global_shapes": forbidden_global_shapes,
        "forbidden_operations": forbidden_operations,
        "head_width": 128,
        "hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "global_logical_gspmd": global_gspmd,
        "local_parallel_size": 4,
        "required_shapes": required_shapes,
        "while_count": while_count,
        "passed": (
            not forbidden_operations
            and not forbidden_global_shapes
            and all(required_shapes.values())
            and dot_metadata_count > 0
            and while_count == expected_while_count
            and (not global_gspmd or explicit_four_partitions)
        ),
    }


def _physical_global_gspmd_stablehlo_contract(
    stablehlo: str,
) -> dict[str, Any]:
    """Require global logical shapes with explicit four-way output sharding."""

    lowered = stablehlo.lower()
    lines = lowered.splitlines()
    sharding_tokens = ("mhlo.sharding", "sdy.sharding")
    required_shapes = {
        "global_logical_weight": "tensor<4096x2048xf32>" in lowered,
        "global_logical_query": "tensor<1x32x128xf32>" in lowered,
        "one_live_q_a_row": "tensor<1x2048xbf16>" in lowered,
    }
    explicitly_sharded = {
        "global_weight_lp4": any(
            "tensor<4096x2048xf32>" in line
            and '"lp4"' in line
            and any(token in line for token in sharding_tokens)
            for line in lines
        ),
        "global_query_lp4": any(
            "tensor<1x32x128xf32>" in line
            and '"lp4"' in line
            and any(token in line for token in sharding_tokens)
            for line in lines
        ),
        "q_a_annotated": any(
            "tensor<1x2048xbf16>" in line
            and any(token in line for token in sharding_tokens)
            for line in lines
        ),
    }
    explicit_four_partitions = (
        "mhlo.num_partitions = 4" in lowered
        or "num_partitions=4" in lowered
    )
    sharding_annotation_count = sum(
        lowered.count(token) for token in sharding_tokens
    )
    return {
        "explicit_four_partitions": explicit_four_partitions,
        "explicitly_sharded": explicitly_sharded,
        "required_shapes": required_shapes,
        "sharding_annotation_count": sharding_annotation_count,
        "stablehlo_sha256": sha256(stablehlo.encode()).hexdigest(),
        "passed": (
            all(required_shapes.values())
            and all(explicitly_sharded.values())
            and explicit_four_partitions
            and sharding_annotation_count >= 3
        ),
    }


def _run_physical_lp4_q_a_boundary(
    *,
    args: argparse.Namespace,
    code_hash: str,
    capture: dict[str, Any],
    accepted_query: np.ndarray,
    current_query: np.ndarray,
    captured_normalized_bits: np.ndarray,
    captured_q_bits: np.ndarray,
    weight_bits: Any,
    weight_scale: Any,
    arrays: dict[str, np.ndarray],
    input_manifest: dict[str, Any],
    q_manifest: dict[str, Any],
) -> None:
    """Reproduce the fused q-a-to-query BF16 materialization boundary."""

    geometry = Layer0DsaProbeGeometry()
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.shape != (4,):
        raise SystemExit("physical LP4 q-a boundary requires four TPU devices")
    mesh = Mesh(devices, ("lp4",))
    normalized = jnp.asarray(
        captured_normalized_bits.view(ml_dtypes.bfloat16)[None, :]
    )
    q_a_bits = jnp.asarray(arrays["self_attn__q_a_proj__weight"])
    q_a_scale = jnp.asarray(
        arrays["self_attn__q_a_proj__weight_scale_inv"]
    )
    kv_a_bits = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight"]
    )
    kv_a_scale = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"]
    )
    q_a_norm_weight = jnp.asarray(
        arrays["self_attn__q_a_layernorm__weight"].view(
            ml_dtypes.bfloat16
        )
    )
    packed = jax.jit(
        lambda q_bits, q_scale, kv_bits, kv_scale: (
            pack_legacy_fused_qkv_runtime_weights(
                q_bits,
                q_scale,
                kv_bits,
                kv_scale,
                geometry=geometry,
            )
        )
    )(
        q_a_bits,
        q_a_scale,
        kv_a_bits,
        kv_a_scale,
    )
    jax.block_until_ready(packed)
    packed_bits = lax.bitcast_convert_type(
        packed.sharded_weight, jnp.uint8
    )
    if packed_bits.shape != (32, 6144, 82) or (
        packed.sharded_scale.shape != (32, 48, 82)
    ):
        raise SystemExit("physical fused q-a final-layout setup drifted")

    arguments = (
        jax.device_put(normalized, NamedSharding(mesh, P())),
        jax.device_put(packed_bits, NamedSharding(mesh, P())),
        jax.device_put(packed.sharded_scale, NamedSharding(mesh, P())),
        jax.device_put(q_a_norm_weight, NamedSharding(mesh, P())),
        jax.device_put(weight_bits, NamedSharding(mesh, P("lp4", None))),
        jax.device_put(weight_scale, NamedSharding(mesh, P("lp4", None))),
        jax.device_put(
            jnp.asarray([8155], dtype=jnp.int32),
            NamedSharding(mesh, P()),
        ),
    )
    args.hlo_dir.mkdir(parents=True)
    records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_query": accepted_query,
        "accepted_q_a_bfloat16_bits": captured_q_bits,
        "current_production_query": current_query,
    }
    for name, enforce_boundary, highest in _physical_lp4_q_a_boundary_modes():
        def local_fused_query(
            normalized_input: Any,
            qkv_bits: Any,
            qkv_scale: Any,
            norm_weight: Any,
            query_bits: Any,
            query_scale: Any,
            positions: Any,
            enforce_boundary: bool = enforce_boundary,
            highest: bool = highest,
        ) -> tuple[Any, Any]:
            attention = AttentionFp8Weights(
                q_a_bits=None,
                q_a_scale=None,
                q_a_norm_weight=norm_weight,
                q_b_bits=None,
                q_b_scale=None,
                kv_a_bits=None,
                kv_a_scale=None,
                kv_a_norm_weight=None,
                kv_b_bits=None,
                kv_b_scale=None,
                o_bits=None,
                o_scale=None,
                qkv_a_bits=qkv_bits,
                qkv_a_scale=qkv_scale,
            )
            q_residual, _ = _project_attention_qkv_a(
                normalized_input,
                attention,
                backend="fused_n82_convolution",
                dsa_contract=DsaNumericalContract(),
                mla_contract=MlaNumericalContract(),
                block_shape=(128, 128),
                epsilon=1e-5,
                linear_backend="pallas",
                linear_interpret=False,
            )
            query_input = (
                lax.optimization_barrier(q_residual)
                if enforce_boundary
                else q_residual
            )
            decoded = lax.optimization_barrier(
                dequantize_fp8_bits_block_weight(
                    query_bits,
                    query_scale,
                    output_dtype=jnp.float32,
                )
            )
            projected = _dot_rows(
                query_input,
                decoded,
                highest=highest,
            )
            return _apply_local_query_rope(projected, positions), q_residual

        mapped = jax.shard_map(
            local_fused_query,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(),
                P("lp4", None),
                P("lp4", None),
                P(),
            ),
            out_specs=(P("lp4", None), P()),
            check_vma=False,
        )
        lowered = jax.jit(mapped).lower(*arguments)
        stablehlo = lowered.as_text()
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
        (args.hlo_dir / f"{name}.stablehlo.mlir").write_text(stablehlo)
        (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(
            optimized_hlo
        )
        contract = _physical_lp4_q_a_boundary_hlo_contract(
            optimized_hlo,
            stablehlo,
            backend=jax.default_backend(),
            candidate=name,
            enforce_bfloat16_boundary=enforce_boundary,
            highest=highest,
        )
        if not contract["passed"]:
            raise SystemExit(f"physical LP4 q-a boundary HLO failed: {name}")
        query_output, q_a_output = compiled(*arguments)
        jax.block_until_ready((query_output, q_a_output))
        candidate_query = np.ascontiguousarray(
            np.asarray(query_output), dtype=np.float32
        )
        candidate_q_bits = np.ascontiguousarray(
            np.asarray(q_a_output)
        ).view(np.uint16)[0]
        if candidate_query.shape != (geometry.heads, geometry.head_dim) or (
            candidate_q_bits.shape != captured_q_bits.shape
        ):
            raise SystemExit("physical fused q-a/query output shape drifted")
        records[name] = {
            "accepted_comparison": _compare(
                accepted_query, candidate_query
            ),
            "current_production_comparison": _compare(
                current_query, candidate_query
            ),
            "hlo": contract,
            "q_a_comparison": _compare_bfloat16_bits(
                captured_q_bits, candidate_q_bits
            ),
        }
        tensor_payload[f"query__{name}"] = candidate_query
        tensor_payload[f"q_a_bfloat16_bits__{name}"] = candidate_q_bits

    exact = sorted(
        name
        for name, record in records.items()
        if record["accepted_comparison"]["elementwise_exact"]
    )
    current_matches = sorted(
        name
        for name, record in records.items()
        if record["current_production_comparison"]["elementwise_exact"]
    )
    q_a_exact = sorted(
        name
        for name, record in records.items()
        if record["q_a_comparison"]["elementwise_exact"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.parent / "physical_lp4_q_a_boundary.npz"
    np.savez(tensor_path, **tensor_payload)
    result = {
        "artifact_kind": PHYSICAL_LP4_Q_A_BOUNDARY_ARTIFACT_KIND,
        "association_restored": bool(exact),
        "backend": jax.default_backend(),
        "candidates": records,
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "query_sha256": _array_sha256(accepted_query),
            "tensors_sha256": args.capture_tensors_sha256,
        },
        "claim_scope": (
            "Bounded physical four-chip fused-q-a/query materialization "
            "boundary only; no decoder, Gate-D, latency, or token-rate claim."
        ),
        "code_hash": code_hash,
        "current_observer": {
            "query_sha256": _array_sha256(current_query),
            "sha256": args.current_internal_sha256,
        },
        "current_reproducing_candidates": current_matches,
        "device_count": jax.device_count(),
        "device_kind": sorted({device.device_kind for device in jax.devices()}),
        "diagnostic_only": True,
        "exact_candidates": exact,
        "format_version": 1,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "local_parallel_size": 4,
        "one_live_row": True,
        "performance_claim": False,
        "q_a_exact_candidates": q_a_exact,
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
    }
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _run_physical_lp4_head_geometry(
    *,
    args: argparse.Namespace,
    code_hash: str,
    capture: dict[str, Any],
    accepted_query: np.ndarray,
    current_query: np.ndarray,
    q_state: Any,
    dequantized_host: np.ndarray,
    input_manifest: dict[str, Any],
    q_manifest: dict[str, Any],
) -> None:
    """Compare one-head physical geometry with an eight-head device loop."""

    geometry = Layer0DsaProbeGeometry()
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.shape != (4,):
        raise SystemExit("physical head geometry requires four TPU devices")
    mesh = Mesh(devices, ("lp4",))
    q_argument = jax.device_put(q_state[:1], NamedSharding(mesh, P()))
    position_argument = jax.device_put(
        jnp.asarray([8155], dtype=jnp.int32), NamedSharding(mesh, P())
    )
    owner_heads = dequantized_host.reshape(
        4, 8, geometry.head_dim, geometry.q_lora_rank
    )
    sweep_weights = tuple(
        jax.device_put(
            owner_heads[:, head].reshape(
                4 * geometry.head_dim, geometry.q_lora_rank
            ),
            NamedSharding(mesh, P("lp4", None)),
        )
        for head in range(8)
    )
    owner_weight_argument = jax.device_put(
        dequantized_host, NamedSharding(mesh, P("lp4", None))
    )

    def local_single_head(
        query: Any, weight: Any, positions: Any
    ) -> Any:
        projected = _dot_rows(query, lax.optimization_barrier(weight))
        return _apply_local_query_rope(projected, positions)

    def local_serial_heads(
        query: Any, weight: Any, positions: Any
    ) -> Any:
        projected = _serial_head_query_projection(
            query,
            lax.optimization_barrier(weight),
            head_dim=geometry.head_dim,
        )
        return _apply_local_query_rope(projected, positions)

    sweep_mapped = jax.shard_map(
        local_single_head,
        mesh=mesh,
        in_specs=(P(), P("lp4", None), P()),
        out_specs=P("lp4", None),
        check_vma=False,
    )
    serial_mapped = jax.shard_map(
        local_serial_heads,
        mesh=mesh,
        in_specs=(P(), P("lp4", None), P()),
        out_specs=P("lp4", None),
        check_vma=False,
    )
    args.hlo_dir.mkdir(parents=True)
    records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_query": accepted_query,
        "current_production_query": current_query,
    }

    sweep_name, serial_name, global_name = (
        _physical_lp4_head_geometry_modes()
    )
    sweep_compiled = jax.jit(sweep_mapped).lower(
        q_argument, sweep_weights[0], position_argument
    ).compile()
    sweep_hlo = sweep_compiled.as_text()
    (args.hlo_dir / f"{sweep_name}.optimized_hlo.txt").write_text(sweep_hlo)
    sweep_contract = _physical_lp4_head_geometry_hlo_contract(
        sweep_hlo, candidate=sweep_name
    )
    if not sweep_contract["passed"]:
        raise SystemExit("physical single-head sweep HLO failed")
    sweep_outputs = []
    for weight in sweep_weights:
        output = sweep_compiled(q_argument, weight, position_argument)
        jax.block_until_ready(output)
        sweep_outputs.append(np.asarray(output, dtype=np.float32))
    sweep_candidate = np.ascontiguousarray(
        np.stack(sweep_outputs).transpose(1, 0, 2).reshape(
            geometry.heads, geometry.head_dim
        ),
        dtype=np.float32,
    )
    records[sweep_name] = {
        "accepted_comparison": _compare(accepted_query, sweep_candidate),
        "current_production_comparison": _compare(
            current_query, sweep_candidate
        ),
        "executions": 8,
        "hlo": sweep_contract,
    }
    tensor_payload[f"query__{sweep_name}"] = sweep_candidate

    serial_compiled = jax.jit(serial_mapped).lower(
        q_argument, owner_weight_argument, position_argument
    ).compile()
    serial_hlo = serial_compiled.as_text()
    (args.hlo_dir / f"{serial_name}.optimized_hlo.txt").write_text(serial_hlo)
    serial_contract = _physical_lp4_head_geometry_hlo_contract(
        serial_hlo, candidate=serial_name
    )
    if not serial_contract["passed"]:
        raise SystemExit("physical serial-head HLO failed")
    serial_output = serial_compiled(
        q_argument, owner_weight_argument, position_argument
    )
    jax.block_until_ready(serial_output)
    serial_candidate = np.ascontiguousarray(
        np.asarray(serial_output), dtype=np.float32
    )
    if serial_candidate.shape != (geometry.heads, geometry.head_dim):
        raise SystemExit("physical serial-head output shape drifted")
    records[serial_name] = {
        "accepted_comparison": _compare(accepted_query, serial_candidate),
        "current_production_comparison": _compare(
            current_query, serial_candidate
        ),
        "executions": 1,
        "hlo": serial_contract,
    }
    tensor_payload[f"query__{serial_name}"] = serial_candidate

    global_function = jax.jit(
        _global_query_one_row,
        in_shardings=(
            NamedSharding(mesh, P()),
            NamedSharding(mesh, P("lp4", None)),
            NamedSharding(mesh, P()),
        ),
        out_shardings=NamedSharding(mesh, P(None, "lp4", None)),
    )
    global_lowered = global_function.lower(
        q_argument, owner_weight_argument, position_argument
    )
    global_stablehlo = global_lowered.as_text()
    global_compiled = global_lowered.compile()
    global_hlo = global_compiled.as_text()
    (args.hlo_dir / f"{global_name}.stablehlo.mlir").write_text(
        global_stablehlo
    )
    (args.hlo_dir / f"{global_name}.optimized_hlo.txt").write_text(
        global_hlo
    )
    global_contract = _physical_lp4_head_geometry_hlo_contract(
        global_hlo, candidate=global_name
    )
    stable_contract = _physical_global_gspmd_stablehlo_contract(
        global_stablehlo
    )
    global_contract["stablehlo"] = stable_contract
    global_contract["passed"] = bool(
        global_contract["passed"] and stable_contract["passed"]
    )
    if not global_contract["passed"]:
        raise SystemExit("physical global-logical GSPMD HLO failed")
    global_output = global_compiled(
        q_argument, owner_weight_argument, position_argument
    )
    jax.block_until_ready(global_output)
    global_candidate = np.ascontiguousarray(
        np.asarray(global_output)[0], dtype=np.float32
    )
    if global_candidate.shape != (geometry.heads, geometry.head_dim):
        raise SystemExit("physical global-logical GSPMD output shape drifted")
    records[global_name] = {
        "accepted_comparison": _compare(accepted_query, global_candidate),
        "current_production_comparison": _compare(
            current_query, global_candidate
        ),
        "executions": 1,
        "hlo": global_contract,
    }
    tensor_payload[f"query__{global_name}"] = global_candidate

    exact = sorted(
        name
        for name, record in records.items()
        if record["accepted_comparison"]["elementwise_exact"]
    )
    current_matches = sorted(
        name
        for name, record in records.items()
        if record["current_production_comparison"]["elementwise_exact"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.parent / "physical_lp4_head_geometry.npz"
    np.savez(tensor_path, **tensor_payload)
    result = {
        "artifact_kind": PHYSICAL_LP4_HEAD_GEOMETRY_ARTIFACT_KIND,
        "association_restored": bool(exact),
        "backend": jax.default_backend(),
        "candidates": records,
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "query_sha256": _array_sha256(accepted_query),
            "tensors_sha256": args.capture_tensors_sha256,
        },
        "claim_scope": (
            "Bounded physical four-chip query head geometry only; no decoder, "
            "Gate-D, latency, or token-rate claim."
        ),
        "code_hash": code_hash,
        "current_observer": {
            "query_sha256": _array_sha256(current_query),
            "sha256": args.current_internal_sha256,
        },
        "current_reproducing_candidates": current_matches,
        "device_count": jax.device_count(),
        "device_kind": sorted({device.device_kind for device in jax.devices()}),
        "diagnostic_only": True,
        "exact_candidates": exact,
        "format_version": 1,
        "head_groups_per_owner": 8,
        "head_width": geometry.head_dim,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "local_parallel_size": 4,
        "one_live_row": True,
        "performance_claim": False,
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
    }
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _run_physical_lp4_query(
    *,
    args: argparse.Namespace,
    code_hash: str,
    capture: dict[str, Any],
    accepted_query: np.ndarray,
    current_query: np.ndarray,
    q_state: Any,
    weight_bits: Any,
    weight_scale: Any,
    dequantized_host: np.ndarray,
    input_manifest: dict[str, Any],
    q_manifest: dict[str, Any],
) -> None:
    """Run one real owner shard on each of four local TPU chips."""

    geometry = Layer0DsaProbeGeometry()
    devices = np.asarray(jax.devices(), dtype=object)
    if devices.shape != (4,):
        raise SystemExit("physical LP4 query requires exactly four TPU devices")
    mesh = Mesh(devices, ("lp4",))
    q_argument = jax.device_put(
        q_state[:1], NamedSharding(mesh, P())
    )
    raw_arguments = (
        q_argument,
        jax.device_put(
            weight_bits, NamedSharding(mesh, P("lp4", None))
        ),
        jax.device_put(
            weight_scale, NamedSharding(mesh, P("lp4", None))
        ),
        jax.device_put(
            jnp.asarray([8155], dtype=jnp.int32),
            NamedSharding(mesh, P()),
        ),
    )
    predecoded_arguments = (
        q_argument,
        jax.device_put(
            dequantized_host, NamedSharding(mesh, P("lp4", None))
        ),
        raw_arguments[-1],
    )

    args.hlo_dir.mkdir(parents=True)
    records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_query": accepted_query,
        "current_production_query": current_query,
    }
    for name, source, association in _physical_lp4_candidate_modes():
        if source == "raw":
            def local_raw(
                query: Any,
                bits: Any,
                scale: Any,
                positions: Any,
                association: str = association,
            ) -> Any:
                decoded = lax.optimization_barrier(
                    dequantize_fp8_bits_block_weight(
                        bits, scale, output_dtype=jnp.float32
                    )
                )
                projected = _local_query_projection(
                    query,
                    decoded,
                    head_dim=geometry.head_dim,
                    association=association,
                )
                return _apply_local_query_rope(projected, positions)

            mapped = jax.shard_map(
                local_raw,
                mesh=mesh,
                in_specs=(P(), P("lp4", None), P("lp4", None), P()),
                out_specs=P("lp4", None),
                check_vma=False,
            )
            arguments = raw_arguments
        else:
            def local_predecoded(
                query: Any,
                weight: Any,
                positions: Any,
                association: str = association,
            ) -> Any:
                projected = _local_query_projection(
                    query,
                    lax.optimization_barrier(weight),
                    head_dim=geometry.head_dim,
                    association=association,
                )
                return _apply_local_query_rope(projected, positions)

            mapped = jax.shard_map(
                local_predecoded,
                mesh=mesh,
                in_specs=(P(), P("lp4", None), P()),
                out_specs=P("lp4", None),
                check_vma=False,
            )
            arguments = predecoded_arguments
        compiled = jax.jit(mapped).lower(*arguments).compile()
        hlo = compiled.as_text()
        (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(hlo)
        contract = _physical_lp4_hlo_contract(
            hlo, candidate=name, source=source
        )
        if not contract["passed"]:
            raise SystemExit(f"physical LP4 query HLO failed: {name}")
        output = compiled(*arguments)
        jax.block_until_ready(output)
        candidate = np.ascontiguousarray(np.asarray(output), dtype=np.float32)
        if candidate.shape != (geometry.heads, geometry.head_dim):
            raise SystemExit("physical LP4 query output shape drifted")
        records[name] = {
            "accepted_comparison": _compare(accepted_query, candidate),
            "current_production_comparison": _compare(
                current_query, candidate
            ),
            "hlo": contract,
        }
        tensor_payload[f"candidate__{name}"] = candidate

    exact = sorted(
        name
        for name, record in records.items()
        if record["accepted_comparison"]["elementwise_exact"]
    )
    current_matches = sorted(
        name
        for name, record in records.items()
        if record["current_production_comparison"]["elementwise_exact"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.parent / "physical_lp4_query_candidates.npz"
    np.savez(tensor_path, **tensor_payload)
    result = {
        "artifact_kind": PHYSICAL_LP4_QUERY_ARTIFACT_KIND,
        "association_restored": bool(exact),
        "backend": jax.default_backend(),
        "candidates": records,
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "query_sha256": _array_sha256(accepted_query),
            "tensors_sha256": args.capture_tensors_sha256,
        },
        "claim_scope": (
            "Bounded physical four-chip layer-0 query arithmetic only; no "
            "decoder, Gate-D, latency, or token-rate claim."
        ),
        "code_hash": code_hash,
        "current_observer": {
            "query_sha256": _array_sha256(current_query),
            "sha256": args.current_internal_sha256,
        },
        "current_reproducing_candidates": current_matches,
        "device_count": jax.device_count(),
        "device_kind": sorted(
            {device.device_kind for device in jax.devices()}
        ),
        "diagnostic_only": True,
        "exact_candidates": exact,
        "format_version": 1,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "local_parallel_size": 4,
        "one_live_row": True,
        "performance_claim": False,
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
    }
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _q_a_hlo_contract(hlo: str, *, candidate: str) -> dict[str, Any]:
    lowered = hlo.lower()
    forbidden = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    forbidden_dead_row_shapes = [
        shape
        for shape in (
            "bf16[32,6144]",
            "f32[32,6144]",
            "bf16[32,2048]",
            "f32[32,2048]",
        )
        if shape in lowered
    ]
    required_shapes = {
        "one_live_hidden_row": "bf16[1,6144]" in lowered,
        "one_live_q_a_row": "bf16[1,2048]" in lowered,
        "shard_major_n82_weight": (
            "f8e4m3fn[32,6144,82]" in lowered
        ),
        "shard_major_n82_scale": "f32[32,48,82]" in lowered,
    }
    if "lax_map_convolution" in candidate:
        convolution_lines = [
            line
            for line in lowered.splitlines()
            if " convolution(" in line and "dim_labels=bf_io->bf" in line
        ]
        required_shapes["one_row_n82_convolution"] = any(
            "f32[1,82]" in line for line in convolution_lines
        )
    return {
        "candidate": candidate,
        "forbidden_operations": forbidden,
        "forbidden_dead_row_shapes": forbidden_dead_row_shapes,
        "hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "one_live_row": True,
        "required_shapes": required_shapes,
        "virtual_tensor_shards": 32,
        "passed": (
            not forbidden
            and not forbidden_dead_row_shapes
            and all(required_shapes.values())
        ),
    }


def _q_a_candidate_modes() -> tuple[tuple[str, str], ...]:
    return tuple(
        ("lax_map_convolution", norm_mode)
        for norm_mode in (
            "logical_mean",
            "shard_sum",
            "left_fold",
            "topology_tree",
        )
    )


def _run_q_a_matrix(
    *,
    args: argparse.Namespace,
    code_hash: str,
    capture: dict[str, Any],
    captured_normalized_bits: np.ndarray,
    captured_q_bits: np.ndarray,
    input_manifest: dict[str, Any],
    q_manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    geometry = Layer0DsaProbeGeometry()
    normalized = jnp.asarray(
        captured_normalized_bits.view(ml_dtypes.bfloat16)[None, :]
    )
    q_a_bits = jnp.asarray(arrays["self_attn__q_a_proj__weight"])
    q_a_scale = jnp.asarray(
        arrays["self_attn__q_a_proj__weight_scale_inv"]
    )
    kv_a_bits = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight"]
    )
    kv_a_scale = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"]
    )
    q_a_norm_weight = jnp.asarray(
        arrays["self_attn__q_a_layernorm__weight"].view(
            ml_dtypes.bfloat16
        )
    )
    pack_function = jax.jit(
        lambda q_bits, q_scale, kv_bits, kv_scale: (
            pack_legacy_fused_qkv_runtime_weights(
                q_bits,
                q_scale,
                kv_bits,
                kv_scale,
                geometry=geometry,
            )
        )
    )
    packed = pack_function(q_a_bits, q_a_scale, kv_a_bits, kv_a_scale)
    jax.block_until_ready(packed)
    if packed.sharded_weight.shape != (32, 6144, 82) or (
        packed.sharded_scale.shape != (32, 48, 82)
    ):
        raise SystemExit("virtual q-a shard-major packing drifted")

    args.hlo_dir.mkdir(parents=True)
    records: dict[str, Any] = {}
    tensor_payload: dict[str, np.ndarray] = {
        "accepted_q_a_bfloat16_bits": captured_q_bits,
        "accepted_normalized_hidden_bfloat16_bits": (
            captured_normalized_bits
        ),
    }
    for projection_mode, norm_mode in _q_a_candidate_modes():
        name = f"virtual_{projection_mode}_{norm_mode}_m1_n82"
        function = jax.jit(
            lambda hidden, weight, scale, norm_weight,
            projection_mode=projection_mode,
            norm_mode=norm_mode: (
                one_row_virtual_tp32_fused_qkv_a_rms_norm(
                    hidden,
                    weight,
                    scale,
                    norm_weight,
                    projection_mode=projection_mode,
                    norm_mode=norm_mode,
                    geometry=geometry,
                )
            )
        )
        compiled = function.lower(
            normalized,
            packed.sharded_weight,
            packed.sharded_scale,
            q_a_norm_weight,
        ).compile()
        hlo = compiled.as_text()
        contract = _q_a_hlo_contract(hlo, candidate=name)
        if not contract["passed"]:
            raise SystemExit(f"q-a association HLO failed: {name}")
        output = compiled(
            normalized,
            packed.sharded_weight,
            packed.sharded_scale,
            q_a_norm_weight,
        )
        jax.block_until_ready(output)
        candidate = np.ascontiguousarray(
            np.asarray(output.q_residual)[0]
        ).view(np.uint16)
        companion = np.ascontiguousarray(
            np.asarray(output.qkv_a_companion)[0]
        ).view(np.uint16)
        records[name] = {
            "comparison": _compare_bfloat16_bits(
                captured_q_bits, candidate
            ),
            "companion_bfloat16_bits_sha256": _array_sha256(companion),
            "hlo": contract,
        }
        tensor_payload[f"candidate__{name}"] = candidate
        tensor_payload[f"companion__{name}"] = companion
        (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(hlo)

    exact = sorted(
        name
        for name, record in records.items()
        if record["comparison"]["elementwise_exact"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.parent / "q_a_candidates.npz"
    np.savez(tensor_path, **tensor_payload)
    result = {
        "artifact_kind": Q_A_ARTIFACT_KIND,
        "association_restored": bool(exact),
        "backend": jax.default_backend(),
        "candidates": records,
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "normalized_hidden_sha256": _array_sha256(
                captured_normalized_bits
            ),
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "q_a_sha256": _array_sha256(captured_q_bits),
            "tensors_sha256": args.capture_tensors_sha256,
        },
        "claim_scope": (
            "Bounded layer-0 one-row q-a arithmetic only; no decoder, "
            "Gate-D, latency, or token-rate claim."
        ),
        "code_hash": code_hash,
        "device_count": jax.device_count(),
        "device_kind": sorted(
            {device.device_kind for device in jax.devices()}
        ),
        "diagnostic_only": True,
        "exact_candidates": exact,
        "format_version": 2,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "local_output_width": 82,
        "one_live_row": True,
        "performance_claim": False,
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
        "virtual_tensor_shards": 32,
    }
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _run_production_qkv_a(
    *,
    args: argparse.Namespace,
    code_hash: str,
    capture: dict[str, Any],
    captured_normalized_bits: np.ndarray,
    captured_q_bits: np.ndarray,
    input_manifest: dict[str, Any],
    q_manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Run the integrated final-layout helper against sealed DB502 outputs."""

    if args.db502_dir is None:
        raise SystemExit("production qkv-a requires the sealed DB502 artifact")
    oracle_runner_path = args.db502_dir / "runner.json"
    oracle_tensor_path = args.db502_dir / "q_a_candidates.npz"
    if (
        _file_sha256(oracle_runner_path) != args.db502_runner_sha256
        or _file_sha256(oracle_tensor_path) != args.db502_tensor_sha256
    ):
        raise SystemExit("sealed DB502 production oracle identity drifted")
    oracle = json.loads(oracle_runner_path.read_text())
    oracle_candidate = "virtual_lax_map_convolution_shard_sum_m1_n82"
    if (
        oracle.get("status") != "SUCCESS"
        or oracle.get("code_hash") != args.db502_code_hash
        or oracle_candidate not in oracle.get("exact_candidates", ())
        or not oracle["candidates"][oracle_candidate]["hlo"]["passed"]
    ):
        raise SystemExit("sealed DB502 production oracle contract drifted")
    with np.load(oracle_tensor_path, allow_pickle=False) as values:
        oracle_q_bits = np.ascontiguousarray(
            values["accepted_q_a_bfloat16_bits"]
        )
        oracle_companion_bits = np.ascontiguousarray(
            values[f"companion__{oracle_candidate}"]
        )
    if not np.array_equal(oracle_q_bits, captured_q_bits) or (
        _array_sha256(oracle_companion_bits)
        != oracle["candidates"][oracle_candidate][
            "companion_bfloat16_bits_sha256"
        ]
    ):
        raise SystemExit("sealed DB502 tensor payload drifted")

    geometry = Layer0DsaProbeGeometry()
    normalized = jnp.asarray(
        captured_normalized_bits.view(ml_dtypes.bfloat16)[None, :]
    )
    q_a_bits = jnp.asarray(arrays["self_attn__q_a_proj__weight"])
    q_a_scale = jnp.asarray(
        arrays["self_attn__q_a_proj__weight_scale_inv"]
    )
    kv_a_bits = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight"]
    )
    kv_a_scale = jnp.asarray(
        arrays["self_attn__kv_a_proj_with_mqa__weight_scale_inv"]
    )
    q_a_norm_weight = jnp.asarray(
        arrays["self_attn__q_a_layernorm__weight"].view(
            ml_dtypes.bfloat16
        )
    )
    packed = jax.jit(
        lambda q_bits, q_scale, kv_bits, kv_scale: (
            pack_legacy_fused_qkv_runtime_weights(
                q_bits,
                q_scale,
                kv_bits,
                kv_scale,
                geometry=geometry,
            )
        )
    )(q_a_bits, q_a_scale, kv_a_bits, kv_a_scale)
    jax.block_until_ready(packed)
    packed_bits = lax.bitcast_convert_type(
        packed.sharded_weight, jnp.uint8
    )
    if packed_bits.shape != (32, 6144, 82) or (
        packed.sharded_scale.shape != (32, 48, 82)
    ):
        raise SystemExit("production qkv-a final-layout setup drifted")

    def production(
        hidden: Any,
        weight_bits: Any,
        scale: Any,
        norm_weight: Any,
    ) -> tuple[Any, Any]:
        attention = AttentionFp8Weights(
            q_a_bits=None,
            q_a_scale=None,
            q_a_norm_weight=norm_weight,
            q_b_bits=None,
            q_b_scale=None,
            kv_a_bits=None,
            kv_a_scale=None,
            kv_a_norm_weight=None,
            kv_b_bits=None,
            kv_b_scale=None,
            o_bits=None,
            o_scale=None,
            qkv_a_bits=weight_bits,
            qkv_a_scale=scale,
        )
        q_residual, companion = _project_attention_qkv_a(
            hidden,
            attention,
            backend="fused_n82_convolution",
            dsa_contract=DsaNumericalContract(),
            mla_contract=MlaNumericalContract(),
            block_shape=(128, 128),
            epsilon=1e-5,
            linear_backend="pallas",
            linear_interpret=False,
        )
        assert companion is not None
        return q_residual, companion

    args.hlo_dir.mkdir(parents=True)
    name = "production_fused_n82_convolution_shard_sum"
    function = jax.jit(production)
    compiled = function.lower(
        normalized,
        packed_bits,
        packed.sharded_scale,
        q_a_norm_weight,
    ).compile()
    hlo = compiled.as_text()
    (args.hlo_dir / f"{name}.optimized_hlo.txt").write_text(hlo)
    runtime_contract = _validate_fused_qkv_a_decoder_association(
        hlo,
        layers=1,
    )
    lowered = hlo.lower()
    forbidden_operations = {
        name: lowered.count(name)
        for name in (
            "all-reduce",
            "all-gather",
            "all-to-all",
            "collective-permute",
            "reduce-scatter",
            "host_callback",
            "xla_python_cpu_callback",
        )
        if name in lowered
    }
    hlo_contract = {
        **runtime_contract,
        "candidate": name,
        "forbidden_operations": forbidden_operations,
        "hlo_sha256": sha256(hlo.encode()).hexdigest(),
        "passed": runtime_contract["passed"] and not forbidden_operations,
    }
    if not hlo_contract["passed"]:
        raise SystemExit("production qkv-a HLO contract failed")
    q_residual, companion = compiled(
        normalized,
        packed_bits,
        packed.sharded_scale,
        q_a_norm_weight,
    )
    jax.block_until_ready((q_residual, companion))
    candidate_q_bits = np.ascontiguousarray(
        np.asarray(q_residual)[0]
    ).view(np.uint16)
    candidate_companion_bits = np.ascontiguousarray(
        np.asarray(companion)[0]
    ).view(np.uint16)
    q_comparison = _compare_bfloat16_bits(
        captured_q_bits,
        candidate_q_bits,
    )
    companion_comparison = _compare_bfloat16_bits(
        oracle_companion_bits,
        candidate_companion_bits,
    )
    restored = (
        q_comparison["elementwise_exact"]
        and companion_comparison["elementwise_exact"]
    )
    records = {
        name: {
            "comparison": q_comparison,
            "companion_comparison": companion_comparison,
            "hlo": hlo_contract,
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.parent / "qkv_a_production.npz"
    np.savez(
        tensor_path,
        accepted_q_a_bfloat16_bits=captured_q_bits,
        accepted_db502_companion_bfloat16_bits=oracle_companion_bits,
        production_q_a_bfloat16_bits=candidate_q_bits,
        production_companion_bfloat16_bits=candidate_companion_bits,
    )
    result = {
        "artifact_kind": PRODUCTION_QKV_A_ARTIFACT_KIND,
        "association_restored": restored,
        "backend": jax.default_backend(),
        "candidates": records,
        "capture": {
            "comparison_sha256": args.capture_comparison_sha256,
            "normalized_hidden_sha256": _array_sha256(
                captured_normalized_bits
            ),
            "owner_actual_sha256": capture["owner_actual_sha256"],
            "q_a_sha256": _array_sha256(captured_q_bits),
            "tensors_sha256": args.capture_tensors_sha256,
        },
        "claim_scope": (
            "Bounded production qkv-a helper arithmetic/HLO only; no "
            "checkpoint, decoder, Gate-D, latency, or token-rate claim."
        ),
        "code_hash": code_hash,
        "db502": {
            "candidate": oracle_candidate,
            "code_hash": args.db502_code_hash,
            "runner_sha256": args.db502_runner_sha256,
            "tensor_sha256": args.db502_tensor_sha256,
        },
        "device_count": jax.device_count(),
        "device_kind": sorted(
            {device.device_kind for device in jax.devices()}
        ),
        "diagnostic_only": True,
        "exact_candidates": [name] if restored else [],
        "final_layout_inputs": {
            "scale_shape": [32, 48, 82],
            "weight_bits_shape": [32, 6144, 82],
        },
        "format_version": 1,
        "input_manifest_sha256": input_manifest["manifest_sha256"],
        "local_output_width": 82,
        "one_live_row": True,
        "performance_claim": False,
        "production_helper": True,
        "q_a_manifest_sha256": q_manifest["manifest_sha256"],
        "status": "SUCCESS",
        "tensor_file": {
            "byte_count": tensor_path.stat().st_size,
            "filename": tensor_path.name,
            "sha256": _file_sha256(tensor_path),
        },
        "virtual_tensor_shards": 32,
    }
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=(
            "query",
            "query_lp4",
            "query_lp4_q_a_boundary",
            "query_lp4_head_geometry",
            "q_a",
            "qkv_a_production",
        ),
        default="query",
    )
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-comparison-sha256", required=True)
    parser.add_argument("--capture-tensors-sha256", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--distributed-q-a-norm-dir", type=Path, required=True)
    parser.add_argument("--q-a-manifest-sha256", required=True)
    parser.add_argument("--q-a-code-hash", required=True)
    parser.add_argument("--db502-dir", type=Path)
    parser.add_argument("--db502-code-hash")
    parser.add_argument("--db502-runner-sha256")
    parser.add_argument("--db502-tensor-sha256")
    parser.add_argument("--current-internal-npz", type=Path)
    parser.add_argument("--current-internal-sha256")
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
        captured_normalized_bits = np.ascontiguousarray(
            payload["actual__normalized_hidden"]
        )
    if actual_query.shape != (32, 128) or captured_q_bits.shape != (2048,):
        raise SystemExit("captured query/q-a shape drifted")
    if captured_normalized_bits.shape != (6144,) or (
        captured_normalized_bits.dtype != np.uint16
        or captured_q_bits.dtype != np.uint16
    ):
        raise SystemExit("captured normalized/q-a storage contract drifted")

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

    if args.target == "q_a":
        _run_q_a_matrix(
            args=args,
            code_hash=code_hash,
            capture=capture,
            captured_normalized_bits=captured_normalized_bits,
            captured_q_bits=captured_q_bits,
            input_manifest=input_manifest,
            q_manifest=q_manifest,
            arrays=arrays,
        )
        return
    if args.target == "qkv_a_production":
        required_db502 = (
            args.db502_code_hash,
            args.db502_runner_sha256,
            args.db502_tensor_sha256,
        )
        if any(value is None for value in required_db502):
            raise SystemExit("production qkv-a DB502 identities are incomplete")
        _run_production_qkv_a(
            args=args,
            code_hash=code_hash,
            capture=capture,
            captured_normalized_bits=captured_normalized_bits,
            captured_q_bits=captured_q_bits,
            input_manifest=input_manifest,
            q_manifest=q_manifest,
            arrays=arrays,
        )
        return

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

    if args.target in (
        "query_lp4",
        "query_lp4_q_a_boundary",
        "query_lp4_head_geometry",
    ):
        if (
            args.current_internal_npz is None
            or args.current_internal_sha256 is None
            or not args.current_internal_npz.is_file()
            or _file_sha256(args.current_internal_npz)
            != args.current_internal_sha256
        ):
            raise SystemExit("current physical LP4 observer identity drifted")
        with np.load(args.current_internal_npz, allow_pickle=False) as payload:
            producer_ids = np.asarray(payload["producer_layer_ids"])
            decode_position = np.asarray(payload["decode_position"])
            current_query = np.ascontiguousarray(
                payload["query"][0], dtype=np.float32
            )
        if (
            producer_ids.shape != (21,)
            or int(producer_ids[0]) != 0
            or decode_position.tolist() != [8155]
            or current_query.shape != actual_query.shape
        ):
            raise SystemExit("current physical LP4 observer contract drifted")
        if args.target == "query_lp4":
            _run_physical_lp4_query(
                args=args,
                code_hash=code_hash,
                capture=capture,
                accepted_query=actual_query,
                current_query=current_query,
                q_state=q_state,
                weight_bits=weight_bits,
                weight_scale=weight_scale,
                dequantized_host=dequantized_host,
                input_manifest=input_manifest,
                q_manifest=q_manifest,
            )
        elif args.target == "query_lp4_q_a_boundary":
            _run_physical_lp4_q_a_boundary(
                args=args,
                code_hash=code_hash,
                capture=capture,
                accepted_query=actual_query,
                current_query=current_query,
                captured_normalized_bits=captured_normalized_bits,
                captured_q_bits=captured_q_bits,
                weight_bits=weight_bits,
                weight_scale=weight_scale,
                arrays=arrays,
                input_manifest=input_manifest,
                q_manifest=q_manifest,
            )
        else:
            _run_physical_lp4_head_geometry(
                args=args,
                code_hash=code_hash,
                capture=capture,
                accepted_query=actual_query,
                current_query=current_query,
                q_state=q_state,
                dequantized_host=dequantized_host,
                input_manifest=input_manifest,
                q_manifest=q_manifest,
            )
        return

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
