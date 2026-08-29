"""Complete bounded PP16 feature-sharded Gate-D discriminator graph.

The executable is intentionally limited to stage 0 on one adjacent LP2 pair.
It evaluates all 8,156 candidate-coherent rows through layer 0, carries only
one BF16 hidden half per owner, reconstructs layer-1 prompt index keys with the
accepted physical-M64 association, and stops after the exact layer-1/event-1
DSA scorer.  It neither executes layer-1 attention output nor any later layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import Any, NamedTuple

import numpy as np

from ..errors import BenchmarkValidationError, PlanValidationError
from ..kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.fp8 import dequantize_fp8_bits_block_weight
from ..kernels.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    promote_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from ..kernels.reference.qkv_a import (
    FusedQkvAContract,
    one_row_fused_qkv_a_convolution,
)
from ..kernels.reference.rotary import (
    apply_rotary_fp32_final_round,
    build_rotary_table_host,
    rotary_table_sha256,
)
from ..kernels.stage_local import (
    _stage_fp8_linear,
    stage_local_dense_feature2_fp8_mapped,
    stage_local_dsa_fp8_mapped,
    stage_local_index_share_fp8_mapped,
)
from ..validation.short_context_oracle import inspect_short_context_oracle
from .pp16_feature2_prefill import (
    PP16_FEATURE2_CONTEXT_LENGTH,
    PP16_FEATURE2_CURRENT_POSITION,
    PP16_FEATURE2_MAIN_ROPE_CAPACITY,
    PP16_FEATURE2_MAIN_ROPE_TABLE_SHA256,
    PP16_FEATURE2_MAIN_ROPE_THETA,
    PP16_FEATURE2_MAIN_ROPE_WIDTH,
    PP16_FEATURE2_PREFILL_VALID_ROWS,
    Feature2PrefillGraph,
    load_feature2_prefill_inputs,
    validate_feature2_prefill_graph,
)
from .pp16_feature_sharded_state import (
    feature2_add_rms_gather_mapped,
    feature2_embedding_mapped,
)

_AXIS_NAME = "feature"
_GROUPS = ((0, 1),)
_PAIRS = ((0, 1), (1, 0))
_CACHE_LAYOUT = StageLocalKvLayout(
    logical_page_size=512,
    local_parallel_size=2,
    packed_cache_width=640,
)
_DSA_CONTRACT = DsaNumericalContract()
_MLA_CONTRACT = MlaNumericalContract()
PP16_FEATURE2_OBSERVATION_POSITION = 113
_FEATURE2_DEFAULT_JAXPR_SHA256 = (
    "75deaf2087d62885eb6e0a9a4d26317ad70e405f912d793dbd9bc355de6d856d"
)
_FEATURE2_FULL_WIDTH_JAXPR_SHA256 = (
    "9773c7b150a5b277116b33574b56f40316da24c5fc497d8827edbeb83fde372d"
)
_FEATURE2_POSITION113_JAXPR_SHA256 = (
    "a6ce2233eed467ae85be0a718532f3e4996b1588673b45687173459caa5adbf0"
)


class Feature2PrefillRuntimeInputs(NamedTuple):
    candidate_token_ids: np.ndarray
    candidate_positions: np.ndarray
    block_tables: np.ndarray
    context_lengths: np.ndarray
    current_position: np.ndarray
    main_rope_table: np.ndarray


class Feature2PrefillProgramResult(NamedTuple):
    event1_positions: Any
    event1_valid_counts: Any
    event1_scores: Any
    current_carried_halves: Any
    current_attention_query_owners: Any
    current_kv_a: Any
    current_normalized_hidden_owners: Any
    current_q_a_state_owners: Any
    current_dsa_query_owners: Any
    current_dsa_head_weights_owners: Any
    layer0_kv_cache_owners: Any
    layer0_index_cache_owners: Any
    layer1_index_cache_owners: Any
    carried_liveness_digest_owners: Any
    contract_valid: Any


class Feature2Position113ProgramResult(NamedTuple):
    """Ordinary feature2 roots plus already-live layer-0 position-113 values."""

    event1_positions: Any
    event1_valid_counts: Any
    event1_scores: Any
    current_carried_halves: Any
    current_attention_query_owners: Any
    current_kv_a: Any
    current_normalized_hidden_owners: Any
    current_q_a_state_owners: Any
    current_dsa_query_owners: Any
    current_dsa_head_weights_owners: Any
    layer0_kv_cache_owners: Any
    layer0_index_cache_owners: Any
    layer1_index_cache_owners: Any
    carried_liveness_digest_owners: Any
    contract_valid: Any
    position113_normalized_hidden_owners: Any
    position113_q_a_state_owners: Any
    position113_dsa_query_owners: Any
    position113_dsa_head_weights_owners: Any
    position113_current_key_owners: Any
    position113_selected_positions_owners: Any
    position113_selected_valid_counts_owners: Any
    position113_selected_scores_owners: Any
    position113_observation_count_owners: Any


@dataclass(frozen=True, slots=True)
class Feature2PrefillProgram:
    graph_sha256: str
    full_width_rounded_then_slice: bool
    sealed_boundary_capture: bool
    observe_position_113: bool
    mesh: Any
    weight_specs: Mapping[str, Any]
    materialize_query_weights_fp32: Any
    decode_index_weights_bf16: Any
    promote_index_weights_fp32: Any
    execute: Any


def _validate_feature2_runtime_devices(devices: Sequence[Any]) -> tuple[Any, ...]:
    runtime_devices = tuple(devices)
    if len(runtime_devices) != 2 or tuple(
        int(device.id) for device in runtime_devices
    ) != (0, 1):
        raise PlanValidationError(
            "feature2 executable requires exact adjacent stage-0 devices 0,1"
        )
    coordinates = tuple(
        None
        if getattr(device, "coords", None) is None
        else tuple(int(value) for value in device.coords)
        for device in runtime_devices
    )
    if any(value is not None for value in coordinates) and coordinates != (
        (0, 0, 0),
        (1, 0, 0),
    ):
        raise PlanValidationError(
            "feature2 executable devices 0,1 are not the protected adjacent pair"
        )
    return runtime_devices


def validate_feature2_prefill_jaxpr(
    jaxpr: str,
    *,
    full_width_rounded_then_slice: bool = False,
    observe_position_113: bool = False,
) -> dict[str, Any]:
    """Fail closed on the complete abstract executable before TPU lowering."""

    if not isinstance(jaxpr, str) or not jaxpr.strip():
        raise BenchmarkValidationError("feature2 executable JAXpr is empty")
    if not isinstance(full_width_rounded_then_slice, bool):
        raise TypeError("feature2 JAXpr full-width-then-slice flag must be boolean")
    if not isinstance(observe_position_113, bool):
        raise TypeError("feature2 JAXpr position-113 observer flag must be boolean")
    jaxpr_sha256 = sha256(jaxpr.encode()).hexdigest()
    expected_jaxpr_sha256 = (
        _FEATURE2_POSITION113_JAXPR_SHA256
        if observe_position_113
        else (
            _FEATURE2_FULL_WIDTH_JAXPR_SHA256
            if full_width_rounded_then_slice
            else _FEATURE2_DEFAULT_JAXPR_SHA256
        )
    )
    counts = {
        "all_gather": jaxpr.count("all_gather["),
        "convolution": jaxpr.count("conv_general_dilated["),
        "fp8_attention_o_n3072": jaxpr.count(
            "name=greenfield_fp8_strategy_nd_o_m8_k512_n3072"
        ),
        "fp8_attention_o_n6144": jaxpr.count(
            "name=greenfield_fp8_strategy_nd_o_m8_k512_n6144"
        ),
        "h16_b512_attention": jaxpr.count(
            "name=greenfield_pregathered_sparse_mla_h16_k2048_b512_w640"
        ),
        "observation_position_literal": jaxpr.count(" 113"),
        "physical_m64_projection": jaxpr.count("Precision.DEFAULT, Precision.HIGHEST"),
        "pmin": jaxpr.count("pmin["),
        "ppermute": jaxpr.count("ppermute["),
        "psum": jaxpr.count("psum["),
        "scan": jaxpr.count("scan["),
    }
    expected = {
        "all_gather": 27,
        "convolution": 133 if full_width_rounded_then_slice else 197,
        "fp8_attention_o_n3072": 0 if full_width_rounded_then_slice else 128,
        "fp8_attention_o_n6144": 64 if full_width_rounded_then_slice else 0,
        "h16_b512_attention": 8,
        "observation_position_literal": 4 if observe_position_113 else 0,
        "physical_m64_projection": 4,
        "pmin": 2 if observe_position_113 else 1,
        "ppermute": 12,
        "psum": 16,
        "scan": 18,
    }
    violations = []
    if counts != expected:
        violations.append(
            f"feature2 executable primitive counts drifted: "
            f"expected={expected}, observed={counts}"
        )
    if jaxpr_sha256 != expected_jaxpr_sha256:
        violations.append(
            "feature2 executable causal JAXpr identity drifted: "
            f"expected={expected_jaxpr_sha256}, observed={jaxpr_sha256}"
        )
    forbidden = tuple(
        marker
        for marker in (
            "bf16[32,6144]",
            "bf16[8156,6144]",
            "debug_callback",
            "host_callback",
            "io_callback",
            "pure_callback",
            "h32_k2048",
        )
        if marker in jaxpr
    )
    if forbidden:
        violations.append(
            f"feature2 executable contains forbidden work/state: {forbidden}"
        )
    if "bf16[2,1,3072]" not in jaxpr:
        violations.append("feature2 executable lost its two persistent halves")
    if violations:
        raise BenchmarkValidationError(
            f"feature2 executable JAXpr rejected: {violations}"
        )
    report = {
        **counts,
        "forbidden_markers": [],
        "jaxpr_sha256": jaxpr_sha256,
        "passed": True,
    }
    if observe_position_113:
        report["observe_position_113"] = True
        report["position113_causal_contract"] = {
            "counter_carried_through_all_scans": True,
            "observed_value_count": 8,
            "position": PP16_FEATURE2_OBSERVATION_POSITION,
            "terminal_validity_gated_by_exact_count": True,
        }
    return report


def validate_feature2_prefill_result_abstract(
    result: Feature2PrefillProgramResult | Feature2Position113ProgramResult,
    *,
    observe_position_113: bool = False,
) -> dict[str, Any]:
    """Pin the only admitted terminal/state boundary of the executable."""

    expected = (
        ((1, 2048), "int32"),
        ((1,), "int32"),
        ((1, 2048), "float32"),
        ((2, 1, 3072), "bfloat16"),
        ((2, 1, 32, 256), "bfloat16"),
        ((1, 576), "bfloat16"),
        ((2, 1, 6144), "bfloat16"),
        ((2, 1, 2048), "bfloat16"),
        ((2, 1, 32, 128), "float32"),
        ((2, 1, 32), "float32"),
        ((2, 16, 256, 640), "bfloat16"),
        ((2, 16, 256, 128), "bfloat16"),
        ((2, 16, 256, 128), "bfloat16"),
        ((2, 2), "uint32"),
        ((1,), "bool"),
    )
    if not isinstance(observe_position_113, bool):
        raise TypeError("feature2 terminal position-113 observer flag must be boolean")
    if observe_position_113:
        expected += (
            ((2, 1, 6144), "bfloat16"),
            ((2, 1, 2048), "bfloat16"),
            ((2, 1, 32, 128), "float32"),
            ((2, 1, 32), "float32"),
            ((2, 1, 128), "float32"),
            ((2, 1, 2048), "int32"),
            ((2, 1), "int32"),
            ((2, 1, 2048), "float32"),
            ((2, 1), "int32"),
        )
    observed = tuple((tuple(value.shape), str(value.dtype)) for value in result)
    if observed != expected:
        raise BenchmarkValidationError(
            "feature2 executable terminal boundary drifted: "
            f"expected={expected}, observed={observed}"
        )
    report = {
        "output_count": len(observed),
        "passed": True,
        "sealed_boundary_capture": True,
        "terminal_dtypes": [dtype for _, dtype in observed],
        "terminal_shapes": [list(shape) for shape, _ in observed],
    }
    if observe_position_113:
        report["observe_position_113"] = True
    return report


_EXECUTABLE_WEIGHT_SHAPES: dict[str, tuple[tuple[int, ...], str]] = {
    "global.embedding": ((77440, 6144), "bf16"),
    "attention.slot_00.input_norm": ((6144,), "bf16"),
    "attention.slot_00.post_norm": ((6144,), "bf16"),
    "attention.slot_00.qkv_a.weight_bits": ((32, 6144, 82), "u8"),
    "attention.slot_00.qkv_a.scale_inv": ((32, 48, 82), "f32"),
    "attention.slot_00.q_a_norm": ((2048,), "bf16"),
    "attention.slot_00.q_b.weight_bits": ((8192, 2048), "u8"),
    "attention.slot_00.q_b.scale_inv": ((64, 16), "f32"),
    "attention.slot_00.kv_a_norm": ((512,), "bf16"),
    "attention.slot_00.kv_b.weight_bits": ((14336, 512), "u8"),
    "attention.slot_00.kv_b.scale_inv": ((112, 4), "f32"),
    "attention.slot_00.o.weight_bits": ((6144, 8192), "u8"),
    "attention.slot_00.o.scale_inv": ((48, 64), "f32"),
    "attention.slot_01.input_norm": ((6144,), "bf16"),
    "attention.slot_01.qkv_a.weight_bits": ((32, 6144, 82), "u8"),
    "attention.slot_01.qkv_a.scale_inv": ((32, 48, 82), "f32"),
    "attention.slot_01.q_a_norm": ((2048,), "bf16"),
    "attention.slot_01.q_b.weight_bits": ((8192, 2048), "u8"),
    "attention.slot_01.q_b.scale_inv": ((64, 16), "f32"),
    "indexer.slot_00.wq_b.weight_bits": ((2048, 2048), "u8"),
    "indexer.slot_00.wq_b.scale_inv": ((16, 16), "f32"),
    "indexer.slot_00.wk.weight_bits": ((128, 6144), "u8"),
    "indexer.slot_00.wk.scale_inv": ((1, 48), "f32"),
    "indexer.slot_00.key_norm_weight": ((128,), "bf16"),
    "indexer.slot_00.key_norm_bias": ((128,), "bf16"),
    "indexer.slot_00.head_weight": ((16, 6144), "bf16"),
    "indexer.slot_01.wq_b.weight_bits": ((2048, 2048), "u8"),
    "indexer.slot_01.wq_b.scale_inv": ((16, 16), "f32"),
    "indexer.slot_01.wk.weight_bits": ((128, 6144), "u8"),
    "indexer.slot_01.wk.scale_inv": ((1, 48), "f32"),
    "indexer.slot_01.key_norm_weight": ((128,), "bf16"),
    "indexer.slot_01.key_norm_bias": ((128,), "bf16"),
    "indexer.slot_01.head_weight": ((16, 6144), "bf16"),
    "dense.slot_00.merged_gate_up.weight_bits_in_out": (
        (16, 6144, 768),
        "u8",
    ),
    "dense.slot_00.merged_gate_up.scale_inv_in_out": (
        (16, 48, 768),
        "f32",
    ),
    "dense.slot_00.down.weight_bits_in_out": ((16, 384, 6144), "u8"),
    "dense.slot_00.down.scale_inv_in_out": ((16, 3, 6144), "f32"),
}
_DENSE_SOURCE_NAMES = frozenset(
    {
        "dense.slot_00.gate.weight_bits",
        "dense.slot_00.gate.scale_inv",
        "dense.slot_00.up.weight_bits",
        "dense.slot_00.up.scale_inv",
        "dense.slot_00.down.weight_bits",
        "dense.slot_00.down.scale_inv",
    }
)
_DENSE_FINAL_NAMES = frozenset(
    name
    for name in _EXECUTABLE_WEIGHT_SHAPES
    if ".weight_bits_in_out" in name or ".scale_inv_in_out" in name
)


def validate_feature2_executable_weight_contract(
    graph: Feature2PrefillGraph,
) -> dict[str, Any]:
    """Bind all 39 authenticated sources to the 37 executable leaves."""

    validate_feature2_prefill_graph(graph)
    source_by_slot: dict[int, dict[str, Any]] = {0: {}, 1: {}}
    for read in graph.selected_reads:
        source_by_slot[read.device_slot][read.name] = read
    violations = []
    expected_sources = set(source_by_slot[0])
    if (
        len(expected_sources) != 39
        or expected_sources != set(source_by_slot[1])
        or not _DENSE_SOURCE_NAMES.issubset(expected_sources)
    ):
        violations.append("feature2 executable source owner set drifted")
    transformed = (expected_sources - _DENSE_SOURCE_NAMES) | _DENSE_FINAL_NAMES
    if transformed != set(_EXECUTABLE_WEIGHT_SHAPES):
        violations.append("feature2 executable final-layout leaf set drifted")
    dtype_names = {"BF16": "bf16", "F32": "f32", "U8": "u8"}
    for slot in (0, 1):
        for name, read in source_by_slot[slot].items():
            if name in _DENSE_SOURCE_NAMES:
                continue
            expected = _EXECUTABLE_WEIGHT_SHAPES.get(name)
            observed = (tuple(read.shape), dtype_names.get(read.dtype))
            if expected != observed:
                violations.append(
                    f"feature2 executable owner {slot} leaf {name!r} drifted"
                )
    if violations:
        raise BenchmarkValidationError(
            f"feature2 executable weight contract rejected: {violations}"
        )
    return {
        "dense_final_leaf_count": len(_DENSE_FINAL_NAMES),
        "dense_source_leaf_count": len(_DENSE_SOURCE_NAMES),
        "executable_leaf_count": len(_EXECUTABLE_WEIGHT_SHAPES),
        "passed": True,
        "source_leaf_count": len(expected_sources),
    }


def load_feature2_prefill_runtime_inputs(
    oracle_dir: Any,
) -> Feature2PrefillRuntimeInputs:
    """Load only authenticated tokens plus the deterministic main-RoPE asset."""

    from pathlib import Path

    from safetensors import safe_open

    root = Path(oracle_dir)
    metadata = load_feature2_prefill_inputs(root)
    token_source = next(
        source for source in metadata.sources if source.name == "candidate_token_ids"
    )
    manifest = inspect_short_context_oracle(root)
    token_record = manifest["files"]["tokens"]
    with safe_open(root / token_record["filename"], framework="np") as handle:
        prompt = np.asarray(handle.get_tensor("prompt_token_ids"), dtype=np.int32)
        generated = np.asarray(handle.get_tensor("generated_token_ids"), dtype=np.int32)
    tokens = np.ascontiguousarray(np.concatenate((prompt, generated[:1])))
    if tokens.shape != token_source.shape:
        raise BenchmarkValidationError("feature2 runtime token shape drifted")
    from .pp16_feature2_prefill import _sha256_array

    if _sha256_array(tokens) != token_source.sha256:
        raise BenchmarkValidationError("feature2 runtime token bytes drifted")
    positions = np.arange(PP16_FEATURE2_CONTEXT_LENGTH, dtype=np.int32)
    block_tables = np.arange(16, dtype=np.int32)[None, :]
    context_lengths = np.asarray([PP16_FEATURE2_CONTEXT_LENGTH], dtype=np.int32)
    current_position = np.asarray([PP16_FEATURE2_CURRENT_POSITION], dtype=np.int32)
    table = build_rotary_table_host(
        PP16_FEATURE2_MAIN_ROPE_CAPACITY,
        rotary_dim=PP16_FEATURE2_MAIN_ROPE_WIDTH,
        theta=PP16_FEATURE2_MAIN_ROPE_THETA,
    )
    if rotary_table_sha256(table) != PP16_FEATURE2_MAIN_ROPE_TABLE_SHA256:
        raise BenchmarkValidationError("feature2 runtime main-RoPE table drifted")
    return Feature2PrefillRuntimeInputs(
        tokens,
        positions,
        block_tables,
        context_lengths,
        current_position,
        table,
    )


def _feature_half(weight: Any, local_slot: Any) -> Any:
    import jax.numpy as jnp
    from jax import lax

    if weight.shape != (6144,) or weight.dtype != jnp.bfloat16:
        raise ValueError("feature2 norm owner must be complete BF16 hidden width")
    return lax.dynamic_slice_in_dim(
        weight,
        local_slot.astype(jnp.int32) * jnp.int32(3072),
        3072,
        axis=0,
    )[None, :]


def _update_liveness_digest(
    digest: Any, carried: Any, position: Any, local_slot: Any
) -> Any:
    """Keep every ordered BF16 carried value live without retaining history."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if digest.shape != (2,) or digest.dtype != jnp.uint32:
        raise ValueError("feature2 liveness digest must be u32[2]")
    if carried.shape != (1, 3072) or carried.dtype != jnp.bfloat16:
        raise ValueError("feature2 liveness row must be one BF16 half")
    bits = lax.bitcast_convert_type(carried.reshape(3072), jnp.uint16).astype(
        jnp.uint32
    )
    global_feature = (
        jnp.arange(3072, dtype=jnp.uint32)
        + local_slot.astype(jnp.uint32) * jnp.uint32(3072)
        + jnp.uint32(1)
    )
    position_word = position.astype(jnp.uint32) + jnp.uint32(1)
    with jax.named_scope("greenfield_pp16_feature2_carried_liveness"):
        row0 = jnp.sum(bits * (global_feature * jnp.uint32(2654435761) + position_word))
        row1 = jnp.sum(
            (bits ^ (position_word * jnp.uint32(2246822519)))
            * (global_feature * jnp.uint32(3266489917) + jnp.uint32(1))
        )
        first = digest[0] * jnp.uint32(16777619) + row0
        rotated = lax.bitwise_or(
            lax.shift_left(digest[1], jnp.uint32(5)),
            lax.shift_right_logical(digest[1], jnp.uint32(27)),
        )
        second = rotated ^ row1
    return jnp.stack((first, second))


def _qkv_a(normalized: Any, weight: Any, scale: Any, norm: Any) -> Any:
    return one_row_fused_qkv_a_convolution(
        normalized,
        weight,
        scale,
        norm,
        contract=FusedQkvAContract(),
    )


def _attention_query(
    q_residual: Any,
    q_b_bits: Any,
    q_b_scale: Any,
    main_rope_row: Any,
) -> Any:
    import jax.numpy as jnp

    projected = _stage_fp8_linear(
        q_residual,
        q_b_bits,
        q_b_scale,
        block_shape=(128, 128),
        backend="pallas",
        interpret=False,
    ).reshape(1, 32, _MLA_CONTRACT.qk_head_dim)
    q_nope = projected[..., : _MLA_CONTRACT.qk_nope_head_dim]
    q_rope_input = projected[..., _MLA_CONTRACT.qk_nope_head_dim :]
    half = _MLA_CONTRACT.qk_rope_head_dim // 2
    cos = main_rope_row[:half][None, None, :]
    sin = main_rope_row[half:][None, None, :]
    q_rope = apply_rotary_fp32_final_round(
        q_rope_input,
        cos,
        sin,
        interleaved=True,
    )
    return jnp.concatenate((q_nope, q_rope), axis=-1).astype(jnp.bfloat16)


def _feature2_prefill_mapped(
    local_weights: Mapping[str, Any],
    layer0_query_weight: Any,
    layer1_query_weight: Any,
    layer0_wk_weight: Any,
    layer1_wk_weight: Any,
    candidate_token_ids: Any,
    candidate_positions: Any,
    block_tables: Any,
    context_lengths: Any,
    current_position: Any,
    main_rope_table: Any,
    *,
    full_width_rounded_then_slice: bool = False,
    observe_position_113: bool = False,
) -> Feature2PrefillProgramResult | Feature2Position113ProgramResult:
    import jax.numpy as jnp
    from jax import lax, named_scope

    if not isinstance(full_width_rounded_then_slice, bool):
        raise TypeError("feature2 full-width-then-slice flag must be boolean")
    if not isinstance(observe_position_113, bool):
        raise TypeError("feature2 position-113 observer flag must be boolean")
    if candidate_token_ids.shape != (PP16_FEATURE2_CONTEXT_LENGTH,) or (
        candidate_token_ids.dtype != jnp.int32
    ):
        raise ValueError("feature2 executable requires exact candidate tokens")
    if candidate_positions.shape != candidate_token_ids.shape or (
        candidate_positions.dtype != jnp.int32
    ):
        raise ValueError("feature2 executable requires exact candidate positions")
    if block_tables.shape != (1, 16) or block_tables.dtype != jnp.int32:
        raise ValueError("feature2 executable requires the 16-page block table")
    if context_lengths.shape != (1,) or context_lengths.dtype != jnp.int32:
        raise ValueError("feature2 executable requires one context length")
    if current_position.shape != (1,) or current_position.dtype != jnp.int32:
        raise ValueError("feature2 executable requires one current position")
    if main_rope_table.shape != (8192, 64) or (main_rope_table.dtype != jnp.bfloat16):
        raise ValueError("feature2 executable main-RoPE table drifted")

    local_slot = lax.axis_index(_AXIS_NAME)

    def weight(name: str) -> Any:
        value = local_weights[name][0]
        expected_shape, dtype_name = _EXECUTABLE_WEIGHT_SHAPES[name]
        expected_dtype = {
            "bf16": jnp.bfloat16,
            "f32": jnp.float32,
            "u8": jnp.uint8,
        }[dtype_name]
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise ValueError(f"feature2 executable weight {name!r} drifted")
        return value

    layer0_query = layer0_query_weight[0]
    layer1_query = layer1_query_weight[0]
    layer0_wk = layer0_wk_weight[0]
    layer1_wk = layer1_wk_weight[0]
    for name, value, shape in (
        ("layer0 query", layer0_query, (2048, 2048)),
        ("layer1 query", layer1_query, (2048, 2048)),
        ("layer0 wk", layer0_wk, (128, 6144)),
        ("layer1 wk", layer1_wk, (128, 6144)),
    ):
        if value.shape != shape or value.dtype != jnp.float32:
            raise ValueError(f"feature2 materialized {name} drifted")

    input_norm0_half = _feature_half(weight("attention.slot_00.input_norm"), local_slot)
    post_norm0_half = _feature_half(weight("attention.slot_00.post_norm"), local_slot)
    input_norm1_half = _feature_half(weight("attention.slot_01.input_norm"), local_slot)
    zero_half = jnp.zeros((1, 3072), dtype=jnp.bfloat16)
    layer0_query_aliases = (layer0_query,) * 4
    layer1_query_aliases = (layer1_query,) * 4

    kv_cache = jnp.zeros((16, 256, 640), dtype=jnp.bfloat16)
    layer0_index_cache = jnp.zeros((16, 256, 128), dtype=jnp.bfloat16)
    layer1_index_cache = jnp.zeros((16, 256, 128), dtype=jnp.bfloat16)
    digest = jnp.zeros((2,), dtype=jnp.uint32)
    contract_valid = jnp.ones((1,), dtype=jnp.bool_)
    last_carried = zero_half
    last_normalized = jnp.zeros((1, 6144), dtype=jnp.bfloat16)
    if observe_position_113:
        observer = (
            jnp.zeros((1, 6144), dtype=jnp.bfloat16),
            jnp.zeros((1, 2048), dtype=jnp.bfloat16),
            jnp.zeros((1, 32, 128), dtype=jnp.float32),
            jnp.zeros((1, 32), dtype=jnp.float32),
            jnp.zeros((1, 128), dtype=jnp.float32),
            jnp.zeros((1, 2048), dtype=jnp.int32),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.zeros((1, 2048), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
        )
    else:
        observer = ()

    def run_chunk(
        start: int,
        valid_rows: int,
        kv_value: Any,
        index0_value: Any,
        index1_value: Any,
        digest_value: Any,
        valid_value: Any,
        last_carried_value: Any,
        last_normalized_value: Any,
        observer_value: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        token_chunk = lax.dynamic_slice_in_dim(
            candidate_token_ids, start, valid_rows, axis=0
        )
        position_chunk = lax.dynamic_slice_in_dim(
            candidate_positions, start, valid_rows, axis=0
        )

        def step(carry: Any, values: Any) -> tuple[Any, Any]:
            (
                current_kv_cache,
                current_index_cache,
                current_digest,
                current_valid,
                _,
                _,
                current_observer,
            ) = carry
            token_id, position_scalar = values
            position = position_scalar[None]
            causal_context = (position_scalar + jnp.int32(1))[None]
            rope_row = main_rope_table[position_scalar]
            embedding_half = feature2_embedding_mapped(
                token_id,
                weight("global.embedding"),
                axis_name=_AXIS_NAME,
                pairs=_PAIRS,
            )
            carried0, normalized0 = feature2_add_rms_gather_mapped(
                embedding_half,
                zero_half,
                input_norm0_half,
                axis_name=_AXIS_NAME,
                groups=_GROUPS,
            )
            projected0 = _qkv_a(
                normalized0,
                weight("attention.slot_00.qkv_a.weight_bits"),
                weight("attention.slot_00.qkv_a.scale_inv"),
                weight("attention.slot_00.q_a_norm"),
            )
            dsa0 = stage_local_dsa_fp8_mapped(
                None,
                current_index_cache,
                position,
                block_tables,
                causal_context,
                weight("attention.slot_00.input_norm"),
                None,
                None,
                weight("attention.slot_00.q_a_norm"),
                weight("indexer.slot_00.wq_b.weight_bits"),
                weight("indexer.slot_00.wq_b.scale_inv"),
                weight("indexer.slot_00.wk.weight_bits"),
                weight("indexer.slot_00.wk.scale_inv"),
                weight("indexer.slot_00.key_norm_weight"),
                weight("indexer.slot_00.key_norm_bias"),
                weight("indexer.slot_00.head_weight"),
                local_slot,
                axis_name=_AXIS_NAME,
                contract=_DSA_CONTRACT,
                cache_layout=_CACHE_LAYOUT,
                axis_index_groups=_GROUPS,
                precomputed_normalized=normalized0,
                precomputed_q_residual=projected0.q_residual,
                linear_backend="pallas",
                dsa_query_backend="reference",
                dsa_query_weight_aliases=layer0_query_aliases,
                precomputed_wk_weight=layer0_wk,
                dsa_head_key_exact_association=True,
                dsa_score_precision="default",
            )
            attention0 = stage_local_index_share_fp8_mapped(
                None,
                current_kv_cache,
                dsa0.selected_positions,
                dsa0.valid_counts,
                position,
                block_tables,
                causal_context,
                weight("attention.slot_00.input_norm"),
                None,
                None,
                weight("attention.slot_00.q_a_norm"),
                weight("attention.slot_00.q_b.weight_bits"),
                weight("attention.slot_00.q_b.scale_inv"),
                None,
                None,
                weight("attention.slot_00.kv_a_norm"),
                weight("attention.slot_00.kv_b.weight_bits"),
                weight("attention.slot_00.kv_b.scale_inv"),
                weight("attention.slot_00.o.weight_bits"),
                weight("attention.slot_00.o.scale_inv"),
                local_slot,
                axis_name=_AXIS_NAME,
                contract=_MLA_CONTRACT,
                cache_layout=_CACHE_LAYOUT,
                axis_index_groups=_GROUPS,
                main_rope_table_row=rope_row,
                precomputed_normalized=normalized0,
                precomputed_q_residual=projected0.q_residual,
                precomputed_kv_a=projected0.kv_a_projection,
                sparse_attention_backend="pallas",
                linear_backend="pallas",
                add_residual=False,
                pregathered_b512_attention=True,
                feature_sharded_output=True,
                feature_pairs=_PAIRS,
                feature_full_width_rounded_then_slice=(full_width_rounded_then_slice),
            )
            post_attention, normalized_mlp = feature2_add_rms_gather_mapped(
                attention0.output,
                carried0,
                post_norm0_half,
                axis_name=_AXIS_NAME,
                groups=_GROUPS,
            )
            dense_update = stage_local_dense_feature2_fp8_mapped(
                normalized_mlp,
                weight("dense.slot_00.merged_gate_up.weight_bits_in_out"),
                weight("dense.slot_00.merged_gate_up.scale_inv_in_out"),
                weight("dense.slot_00.down.weight_bits_in_out"),
                weight("dense.slot_00.down.scale_inv_in_out"),
                axis_name=_AXIS_NAME,
                pairs=_PAIRS,
                full_width_rounded_then_slice=(full_width_rounded_then_slice),
            )
            carried1, normalized1 = feature2_add_rms_gather_mapped(
                dense_update,
                post_attention,
                input_norm1_half,
                axis_name=_AXIS_NAME,
                groups=_GROUPS,
            )
            next_digest = _update_liveness_digest(
                current_digest, carried1, position_scalar, local_slot
            )
            next_valid = current_valid & dsa0.contract_valid & attention0.contract_valid
            if observe_position_113:
                should_observe = position_scalar == jnp.int32(
                    PP16_FEATURE2_OBSERVATION_POSITION
                )
                observed_values = (
                    dsa0.internals.normalized_hidden,
                    dsa0.internals.q_a_state,
                    dsa0.internals.query,
                    dsa0.internals.head_weights,
                    dsa0.internals.current_key,
                    dsa0.selected_positions,
                    dsa0.valid_counts,
                    dsa0.selected_scores,
                )
                next_observer = tuple(
                    jnp.where(should_observe, observed, previous)
                    for observed, previous in zip(
                        observed_values,
                        current_observer[:-1],
                        strict=True,
                    )
                ) + (current_observer[-1] + should_observe.astype(jnp.int32)[None],)
            else:
                next_observer = current_observer
            next_carry = (
                attention0.cache,
                dsa0.index_cache,
                next_digest,
                next_valid,
                carried1,
                normalized1,
                next_observer,
            )
            return next_carry, normalized1[0]

        initial = (
            kv_value,
            index0_value,
            digest_value,
            valid_value,
            last_carried_value,
            last_normalized_value,
            observer_value,
        )
        final, normalized_history = lax.scan(
            step,
            initial,
            (token_chunk, position_chunk),
            unroll=1,
        )
        repaired_index1 = repair_stage_local_prompt_index_cache(
            index1_value,
            normalized_history,
            block_tables,
            layer1_wk,
            weight("indexer.slot_01.key_norm_weight"),
            weight("indexer.slot_01.key_norm_bias"),
            local_slot,
            contract=_DSA_CONTRACT,
            logical_page_size=512,
            local_rows_per_page=256,
            prompt_chunk=2048,
            physical_rows=64,
            local_parallel_size=2,
            position_offset=start,
            valid_rows=valid_rows,
        )
        return (
            final[0],
            final[1],
            repaired_index1,
            final[2],
            final[3],
            final[4],
            final[5],
            final[6],
        )

    start = 0
    for valid_rows in PP16_FEATURE2_PREFILL_VALID_ROWS:
        (
            kv_cache,
            layer0_index_cache,
            layer1_index_cache,
            digest,
            contract_valid,
            last_carried,
            last_normalized,
            observer,
        ) = run_chunk(
            start,
            valid_rows,
            kv_cache,
            layer0_index_cache,
            layer1_index_cache,
            digest,
            contract_valid,
            last_carried,
            last_normalized,
            observer,
        )
        start += valid_rows

    projected1 = _qkv_a(
        last_normalized,
        weight("attention.slot_01.qkv_a.weight_bits"),
        weight("attention.slot_01.qkv_a.scale_inv"),
        weight("attention.slot_01.q_a_norm"),
    )
    current_rope_row = main_rope_table[current_position[0]]
    attention_query = _attention_query(
        projected1.q_residual,
        weight("attention.slot_01.q_b.weight_bits"),
        weight("attention.slot_01.q_b.scale_inv"),
        current_rope_row,
    )
    event1 = stage_local_dsa_fp8_mapped(
        None,
        layer1_index_cache,
        current_position,
        block_tables,
        context_lengths,
        weight("attention.slot_01.input_norm"),
        None,
        None,
        weight("attention.slot_01.q_a_norm"),
        weight("indexer.slot_01.wq_b.weight_bits"),
        weight("indexer.slot_01.wq_b.scale_inv"),
        weight("indexer.slot_01.wk.weight_bits"),
        weight("indexer.slot_01.wk.scale_inv"),
        weight("indexer.slot_01.key_norm_weight"),
        weight("indexer.slot_01.key_norm_bias"),
        weight("indexer.slot_01.head_weight"),
        local_slot,
        axis_name=_AXIS_NAME,
        contract=_DSA_CONTRACT,
        cache_layout=_CACHE_LAYOUT,
        axis_index_groups=_GROUPS,
        precomputed_normalized=last_normalized,
        precomputed_q_residual=projected1.q_residual,
        linear_backend="pallas",
        dsa_query_backend="reference",
        dsa_query_weight_aliases=layer1_query_aliases,
        precomputed_wk_weight=layer1_wk,
        dsa_head_key_exact_association=True,
        dsa_score_precision="default",
    )
    source_valid = (
        (candidate_positions[-1] == current_position[0])
        & (context_lengths[0] == current_position[0] + jnp.int32(1))
        & (current_position[0] == jnp.int32(PP16_FEATURE2_CURRENT_POSITION))
    )[None]
    contract_valid = contract_valid & event1.contract_valid & source_valid
    contract_valid = lax.pmin(
        contract_valid.astype(jnp.int32),
        _AXIS_NAME,
        axis_index_groups=_GROUPS,
    ) == jnp.int32(1)
    if observe_position_113:
        observation_count_valid = lax.pmin(
            (observer[-1] == jnp.int32(1)).astype(jnp.int32),
            _AXIS_NAME,
            axis_index_groups=_GROUPS,
        ) == jnp.int32(1)
        contract_valid = contract_valid & observation_count_valid

    def seal_boundary(value: Any, scope: str) -> Any:
        with named_scope(scope):
            return lax.optimization_barrier(value[None, ...])

    sealed_normalized_hidden = seal_boundary(
        event1.internals.normalized_hidden,
        "greenfield_pp16_feature2_sealed_normalized_hidden",
    )
    sealed_q_a_state = seal_boundary(
        event1.internals.q_a_state,
        "greenfield_pp16_feature2_sealed_q_a_state",
    )
    sealed_dsa_query = seal_boundary(
        event1.internals.query,
        "greenfield_pp16_feature2_sealed_dsa_query",
    )
    sealed_dsa_head_weights = seal_boundary(
        event1.internals.head_weights,
        "greenfield_pp16_feature2_sealed_dsa_head_weights",
    )
    ordinary = Feature2PrefillProgramResult(
        event1.selected_positions,
        event1.valid_counts,
        event1.selected_scores,
        last_carried[None, ...],
        attention_query[None, ...],
        projected1.kv_a_projection,
        sealed_normalized_hidden,
        sealed_q_a_state,
        sealed_dsa_query,
        sealed_dsa_head_weights,
        kv_cache[None, ...],
        layer0_index_cache[None, ...],
        event1.index_cache[None, ...],
        digest[None, ...],
        contract_valid,
    )
    if not observe_position_113:
        return ordinary

    def seal_observer(value: Any, scope: str) -> Any:
        with named_scope(scope):
            return lax.optimization_barrier(value[None, ...])

    sealed_observer = tuple(
        seal_observer(value, scope)
        for value, scope in zip(
            observer,
            (
                "greenfield_pp16_feature2_p113_normalized_hidden",
                "greenfield_pp16_feature2_p113_q_a_state",
                "greenfield_pp16_feature2_p113_dsa_query",
                "greenfield_pp16_feature2_p113_dsa_head_weights",
                "greenfield_pp16_feature2_p113_current_key",
                "greenfield_pp16_feature2_p113_selected_positions",
                "greenfield_pp16_feature2_p113_selected_valid_counts",
                "greenfield_pp16_feature2_p113_selected_scores",
                "greenfield_pp16_feature2_p113_observation_count",
            ),
            strict=True,
        )
    )
    return Feature2Position113ProgramResult(*ordinary, *sealed_observer)


def build_feature2_prefill_program(
    graph: Feature2PrefillGraph,
    *,
    devices: Sequence[Any],
    full_width_rounded_then_slice: bool = False,
    observe_position_113: bool = False,
) -> Feature2PrefillProgram:
    """Build the exact LP2 executable and its separate FP32 materializers."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    validate_feature2_prefill_graph(graph)
    validate_feature2_executable_weight_contract(graph)
    if not isinstance(full_width_rounded_then_slice, bool):
        raise PlanValidationError("feature2 full-width-then-slice flag must be boolean")
    if not isinstance(observe_position_113, bool):
        raise PlanValidationError("feature2 position-113 observer flag must be boolean")
    runtime_devices = _validate_feature2_runtime_devices(devices)
    mesh = Mesh(np.asarray(runtime_devices, dtype=object), (_AXIS_NAME,))
    weight_specs = {
        name: P(_AXIS_NAME, *(None for _ in shape))
        for name, (shape, _) in _EXECUTABLE_WEIGHT_SHAPES.items()
    }

    def materialize_query(bits0: Any, scale0: Any, bits1: Any, scale1: Any) -> Any:
        def decode(bits: Any, scale: Any) -> Any:
            return dequantize_fp8_bits_block_weight(
                bits[0], scale[0], output_dtype=jnp.float32
            )[None, ...]

        return decode(bits0, scale0), decode(bits1, scale1)

    query_specs = P(_AXIS_NAME, None, None)
    materialize_query_weights_fp32 = jax.shard_map(
        materialize_query,
        mesh=mesh,
        in_specs=(query_specs, query_specs, query_specs, query_specs),
        out_specs=(query_specs, query_specs),
        check_vma=False,
    )

    def decode_wk(bits0: Any, scale0: Any, bits1: Any, scale1: Any) -> Any:
        def decode(bits: Any, scale: Any) -> Any:
            return decode_stage_local_prefill_index_wk_bf16(bits[0], scale[0])[
                None, ...
            ]

        return decode(bits0, scale0), decode(bits1, scale1)

    wk_specs = P(_AXIS_NAME, None, None)
    decode_index_weights_bf16 = jax.shard_map(
        decode_wk,
        mesh=mesh,
        in_specs=(wk_specs, wk_specs, wk_specs, wk_specs),
        out_specs=(wk_specs, wk_specs),
        check_vma=False,
    )

    def promote_wk(wk0: Any, wk1: Any) -> Any:
        return (
            promote_stage_local_prefill_index_wk(wk0[0])[None, ...],
            promote_stage_local_prefill_index_wk(wk1[0])[None, ...],
        )

    promote_index_weights_fp32 = jax.shard_map(
        promote_wk,
        mesh=mesh,
        in_specs=(wk_specs, wk_specs),
        out_specs=(wk_specs, wk_specs),
        check_vma=False,
    )
    ordinary_result_specs = Feature2PrefillProgramResult(
        P(),
        P(),
        P(),
        P(_AXIS_NAME, None, None),
        P(_AXIS_NAME, None, None, None),
        P(),
        P(_AXIS_NAME, None, None),
        P(_AXIS_NAME, None, None),
        P(_AXIS_NAME, None, None, None),
        P(_AXIS_NAME, None, None),
        P(_AXIS_NAME, None, None, None),
        P(_AXIS_NAME, None, None, None),
        P(_AXIS_NAME, None, None, None),
        P(_AXIS_NAME, None),
        P(),
    )
    result_specs: Any
    if observe_position_113:
        result_specs = Feature2Position113ProgramResult(
            *ordinary_result_specs,
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None, None, None),
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None),
            P(_AXIS_NAME, None, None),
            P(_AXIS_NAME, None),
        )
    else:
        result_specs = ordinary_result_specs
    execute = jax.shard_map(
        partial(
            _feature2_prefill_mapped,
            full_width_rounded_then_slice=full_width_rounded_then_slice,
            observe_position_113=observe_position_113,
        ),
        mesh=mesh,
        in_specs=(
            weight_specs,
            query_specs,
            query_specs,
            wk_specs,
            wk_specs,
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        out_specs=result_specs,
        check_vma=False,
    )
    if set(weight_specs) != set(_EXECUTABLE_WEIGHT_SHAPES):
        raise PlanValidationError("feature2 executable weight spec drifted")
    return Feature2PrefillProgram(
        graph_sha256=graph.graph_sha256,
        full_width_rounded_then_slice=full_width_rounded_then_slice,
        sealed_boundary_capture=True,
        observe_position_113=observe_position_113,
        mesh=mesh,
        weight_specs=weight_specs,
        materialize_query_weights_fp32=materialize_query_weights_fp32,
        decode_index_weights_bf16=decode_index_weights_bf16,
        promote_index_weights_fp32=promote_index_weights_fp32,
        execute=execute,
    )
