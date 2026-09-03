from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa_association import (
    Layer0DsaProbeGeometry,
    layer0_prompt_index_key_from_normalized_boundary_chunk,
    layer0_prompt_index_key_gather_cache_states_chunk,
    layer0_prompt_normalized_hidden_boundary_chunk,
)
from glm_tpu.greenfield.validation.original_db518_normalized_boundary_hlo import (
    _KEY_CONTROL_PARAMETER_SHARDINGS,
    _KEY_CONTROL_STABLEHLO_SHA256,
    _NORMALIZATION_PARAMETER_SHARDINGS,
    _NORMALIZATION_STABLEHLO_SHA256,
    _exact_dot_callee,
    _exact_square_callee,
    _optimized_single_device_placement,
    _require_stable_identity,
    require_completed_normalization_boundary_hlo,
    require_normalized_key_control_boundary_hlo,
)


GEOMETRY = Layer0DsaProbeGeometry(
    prompt_tokens=129,
    prompt_chunk=128,
    decode_rows=2,
    hidden_size=8,
    q_lora_rank=8,
    qkv_a_companion_rank=4,
    legacy_tensor_shards=2,
    heads=2,
    head_dim=4,
    rotary_dim=2,
)


def _inputs():
    unique = jnp.asarray(
        np.linspace(-1.0, 1.0, 5 * 8, dtype=np.float32).reshape(5, 8),
        dtype=jnp.bfloat16,
    )
    rows = jnp.asarray(np.arange(128) % 5, dtype=jnp.int32)
    positions = jnp.arange(128, dtype=jnp.int32)
    input_norm = jnp.asarray(
        np.linspace(0.75, 1.25, 8, dtype=np.float32), dtype=jnp.bfloat16
    )
    wk = jnp.asarray(
        np.linspace(-0.25, 0.25, 4 * 8, dtype=np.float32).reshape(4, 8),
        dtype=jnp.float32,
    )
    key_weight = jnp.asarray([1.0, 0.75, 1.25, 0.5], dtype=jnp.bfloat16)
    key_bias = jnp.asarray([0.0, 0.125, -0.125, 0.25], dtype=jnp.bfloat16)
    return unique, rows, positions, input_norm, wk, key_weight, key_bias


def _replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def _with_single_device_sharding(stable: str, axes: tuple[str, ...]) -> str:
    lines = stable.splitlines(keepends=True)
    assert lines[0].startswith("module @")
    lines.insert(
        1,
        "  sdy.mesh @empty_mesh = <[]> {stablehlo.mesh = {axes = []}}\n",
    )
    main_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("  func.func public @main(")
    )
    main = lines[main_index]
    for argument, sharding in enumerate(axes):
        marker = f"%arg{argument}: "
        start = main.index(marker) + len(marker)
        end = (
            main.index(", %arg", start)
            if argument + 1 < len(axes)
            else main.index(") ->", start)
        )
        main = (
            main[:end]
            + f" {{sdy.sharding = #sdy.sharding<@empty_mesh, {sharding}>}}"
            + main[end:]
        )
    lines[main_index] = main
    return "".join(lines)


def _bypass_exact_while_with_all_input_alternate(optimized: str) -> str:
    """Keep an exact dead while/dot while an alternate all-input graph wins."""

    alternate = """
%alternate_live (p0: bf16[2048,6144], p1: f32[128,6144], p2: bf16[128], p3: bf16[128]) -> f32[32,64,128] {
  %p0 = bf16[2048,6144]{1,0} parameter(0)
  %p1 = f32[128,6144]{1,0} parameter(1)
  %p2 = bf16[128]{0} parameter(2)
  %p3 = bf16[128]{0} parameter(3)
  ROOT %alt = f32[32,64,128]{2,1,0} custom-call(%p0, %p1, %p2, %p3), custom_call_target="adversarial"
}

"""
    assert optimized.count("ENTRY %main.8 (") == 1
    optimized = optimized.replace(
        "ENTRY %main.8 (", alternate + "ENTRY %main.8 (", 1
    )
    marker = "  %concatenate_bitcast_fusion = "
    assert optimized.count(marker) == 1
    alternate_call = (
        "  %alternate_live_call = f32[32,64,128]{2,1,0} "
        "fusion(%normalized_chunk.1, %wk_weight.1, %key_norm_weight.1, "
        "%key_norm_bias.1), kind=kLoop, calls=%alternate_live\n"
    )
    optimized = optimized.replace(marker, alternate_call + marker, 1)
    optimized = _replace_once(
        optimized,
        "fusion(%multiply_power_fusion, %positions.1, %while.7)",
        "fusion(%multiply_power_fusion, %positions.1, %alternate_live_call)",
    )
    return _replace_once(
        optimized,
        "fusion(%while.7), kind=kLoop, calls=%fused_computation.5",
        "fusion(%alternate_live_call), kind=kLoop, calls=%fused_computation.5",
    )


def _node(
    opcode,
    shapes,
    *,
    operands=(),
    parameter=None,
    raw="",
):
    return {
        "opcode": opcode,
        "shapes": shapes,
        "operands": operands,
        "parameter": parameter,
        "raw": raw,
    }


def test_tpu_fused_square_reduce_is_exactly_recognized():
    computation = {
        "root": "%reduce",
        "nodes": {
            "%hidden": _node(
                "parameter", (("bf16", (2048, 6144)),), parameter=0
            ),
            "%valid": _node(
                "parameter", (("pred", (2048,)),), parameter=1
            ),
            "%selected": _node(
                "select",
                (("bf16", (2048, 6144)),),
                operands=("%valid", "%hidden", "%hidden"),
            ),
            "%converted": _node(
                "convert",
                (("f32", (2048, 6144)),),
                operands=("%selected",),
            ),
            "%square": _node(
                "multiply",
                (("f32", (2048, 6144)),),
                operands=("%converted", "%converted"),
            ),
            "%zero": _node("constant", (("f32", ()),), raw="constant(0)"),
            "%reduce": _node(
                "reduce",
                (("f32", (2048,)),),
                operands=("%square", "%zero"),
                raw="dimensions={1}",
            ),
        },
    }
    assert _exact_square_callee(computation)
    computation["nodes"]["%square"]["opcode"] = "add"
    assert not _exact_square_callee(computation)


def test_tpu_convolution_lowered_dot_is_exactly_recognized():
    computation = {
        "root": "%tuple",
        "nodes": {
            "%wk": _node(
                "parameter", (("f32", (128, 6144)),), parameter=0
            ),
            "%hidden": _node(
                "parameter", (("bf16", (32, 64, 6144)),), parameter=1
            ),
            "%index": _node("parameter", (("s32", ()),), parameter=2),
            "%lhs": _node(
                "fusion",
                (("f32", (64, 6144)),),
                operands=("%hidden", "%index"),
            ),
            "%rhs": _node(
                "fusion", (("f32", (128, 6144)),), operands=("%wk",)
            ),
            "%dot": _node(
                "convolution",
                (("f32", (64, 128)),),
                operands=("%lhs", "%rhs"),
                raw=(
                    "convolution(%lhs, %rhs), dim_labels=bf_oi->bf, "
                    "operand_precision={default,highest}, "
                    'metadata={op_name="jit(test)/dot_general"}'
                ),
            ),
            "%sum": _node(
                "reduce", (("f32", (64,)),), operands=("%dot",)
            ),
            "%tuple": _node(
                "tuple",
                (("f32", (64,)), ("f32", (64, 128))),
                operands=("%sum", "%dot"),
            ),
        },
    }
    assert _exact_dot_callee(computation)
    computation["nodes"]["%dot"]["raw"] = computation["nodes"]["%dot"][
        "raw"
    ].replace("default,highest", "default,default")
    assert not _exact_dot_callee(computation)


def test_optimized_tpu_placement_rejects_nonreplicated_argument():
    axes = ("[{}, {}]", "[{}]")
    nodes = {}
    parameters = {}
    for index, sharding in enumerate(axes):
        name = f"%arg{index}"
        parameters[index] = name
        nodes[name] = _node(
            "parameter",
            (("f32", (2,)),),
            parameter=index,
            raw=(
                "parameter(0), sharding={replicated}, "
                'frontend_attributes={xla.sdy.sharding="'
                f"#sdy.sharding<@empty_mesh, {sharding}>"
                '"}'
            ),
        )
    header = (
        "HloModule test, frontend_attributes={xla.sdy.meshes="
        "{empty_mesh = #sdy.mesh<[]>}}\n"
    )
    assert _optimized_single_device_placement(
        header, nodes, parameters, axes
    )
    nodes["%arg1"]["raw"] = nodes["%arg1"]["raw"].replace(
        "sharding={replicated}", "sharding={maximal device=0}"
    )
    assert not _optimized_single_device_placement(
        header, nodes, parameters, axes
    )


def test_completed_boundary_composes_to_original_producer_keys():
    unique, rows, positions, input_norm, wk, key_weight, key_bias = _inputs()
    normalized = layer0_prompt_normalized_hidden_boundary_chunk(
        unique, rows, input_norm, geometry=GEOMETRY
    )
    boundary_keys = layer0_prompt_index_key_from_normalized_boundary_chunk(
        normalized,
        positions,
        wk,
        key_weight,
        key_bias,
        geometry=GEOMETRY,
    )
    states = layer0_prompt_index_key_gather_cache_states_chunk(
        jnp.zeros((1, 1, 128, 4), dtype=jnp.bfloat16),
        jnp.asarray([0], dtype=jnp.int32),
        unique,
        rows,
        positions,
        input_norm,
        wk,
        key_weight,
        key_bias,
        geometry=GEOMETRY,
        key_norm_mode="divide_sqrt",
        rotary_mode="accepted_source",
        projection_weight_mode="adapted_fp32",
        projection_mapping_mode="physical_m64_projection_keynorm_lax_map",
    )
    assert normalized.shape == (128, 8)
    assert normalized.dtype == jnp.bfloat16
    assert boundary_keys.shape == (128, 4)
    assert boundary_keys.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(boundary_keys).view(np.uint16),
        np.asarray(states.index_cache).reshape(128, 4).view(np.uint16),
    )


def test_boundary_rejects_dtype_and_geometry_drift():
    unique, rows, positions, input_norm, wk, key_weight, key_bias = _inputs()
    with pytest.raises(ValueError, match="values must be BF16"):
        layer0_prompt_normalized_hidden_boundary_chunk(
            unique.astype(jnp.float32), rows, input_norm, geometry=GEOMETRY
        )
    normalized = layer0_prompt_normalized_hidden_boundary_chunk(
        unique, rows, input_norm, geometry=GEOMETRY
    )
    with pytest.raises(ValueError, match="positions must be int32"):
        layer0_prompt_index_key_from_normalized_boundary_chunk(
            normalized,
            positions.astype(jnp.int16),
            wk,
            key_weight,
            key_bias,
            geometry=GEOMETRY,
        )
    with pytest.raises(ValueError, match="wk must remain FP32"):
        layer0_prompt_index_key_from_normalized_boundary_chunk(
            normalized,
            positions,
            wk.astype(jnp.bfloat16),
            key_weight,
            key_bias,
            geometry=GEOMETRY,
        )


@pytest.fixture(scope="module")
def real_boundary_hlo():
    geometry = Layer0DsaProbeGeometry()
    normalizer = partial(
        layer0_prompt_normalized_hidden_boundary_chunk, geometry=geometry
    )
    normalizer_lowered = jax.jit(normalizer).lower(
        jax.ShapeDtypeStruct((37, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048,), jnp.int32),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16),
    )
    normalizer_optimized = normalizer_lowered.compile().as_text()
    key_control = partial(
        layer0_prompt_index_key_from_normalized_boundary_chunk,
        geometry=geometry,
    )
    key_lowered = jax.jit(key_control).lower(
        jax.ShapeDtypeStruct((2048, 6144), jnp.bfloat16),
        jax.ShapeDtypeStruct((2048,), jnp.int32),
        jax.ShapeDtypeStruct((128, 6144), jnp.float32),
        jax.ShapeDtypeStruct((128,), jnp.bfloat16),
        jax.ShapeDtypeStruct((128,), jnp.bfloat16),
    )
    key_optimized = key_lowered.compile().as_text()
    return (
        normalizer_optimized,
        normalizer_lowered.as_text(),
        key_optimized,
        key_lowered.as_text(),
    )


def test_real_boundary_hlo_is_admitted(real_boundary_hlo):
    normalizer_optimized, normalizer_stable, key_optimized, key_stable = (
        real_boundary_hlo
    )
    assert require_completed_normalization_boundary_hlo(
        normalizer_optimized, normalizer_stable
    )["passed"]
    assert require_normalized_key_control_boundary_hlo(
        key_optimized, key_stable
    )["passed"]


def test_real_boundary_hlo_admits_exact_tpu_single_device_placement(
    real_boundary_hlo,
):
    _, normalizer_stable, _, key_stable = real_boundary_hlo
    _, normalizer_identity = _require_stable_identity(
        _with_single_device_sharding(
            normalizer_stable, ("[{}, {}]", "[{}]", "[{}]")
        ),
        module_name="jit_layer0_prompt_normalized_hidden_boundary_chunk",
        expected_sha256=_NORMALIZATION_STABLEHLO_SHA256,
        parameter_shardings=_NORMALIZATION_PARAMETER_SHARDINGS,
    )
    _, key_identity = _require_stable_identity(
        _with_single_device_sharding(
            key_stable,
            ("[{}, {}]", "[{}]", "[{}, {}]", "[{}]", "[{}]"),
        ),
        module_name=(
            "jit_layer0_prompt_index_key_from_normalized_boundary_chunk"
        ),
        expected_sha256=_KEY_CONTROL_STABLEHLO_SHA256,
        parameter_shardings=_KEY_CONTROL_PARAMETER_SHARDINGS,
    )
    assert normalizer_identity["stable_single_device_sharding_bound"]
    assert normalizer_identity["stable_parameter_sharding_count"] == 3
    assert normalizer_identity["stable_raw_sha256"] != normalizer_identity[
        "stable_sha256"
    ]
    assert key_identity["stable_single_device_sharding_bound"]
    assert key_identity["stable_parameter_sharding_count"] == 5


@pytest.mark.parametrize(
    "mutation",
    (
        lambda stable: stable.replace("@empty_mesh", "@other_mesh", 1),
        lambda stable: stable.replace("<[]>", '<["feature"=1]>', 1),
        lambda stable: stable.replace("{axes = []}", '{axes = ["feature"]}', 1),
        lambda stable: stable.replace(
            " {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}", "", 1
        ),
        lambda stable: stable.replace("[{}, {}]", "[{}]", 1),
        lambda stable: stable + "  sdy.mesh @extra = <[]>\n",
    ),
)
def test_tpu_single_device_placement_rejects_drift(
    real_boundary_hlo, mutation
):
    _, stable, _, _ = real_boundary_hlo
    placed = _with_single_device_sharding(
        stable, ("[{}, {}]", "[{}]", "[{}]")
    )
    with pytest.raises(RuntimeError):
        _require_stable_identity(
            mutation(placed),
            module_name="jit_layer0_prompt_normalized_hidden_boundary_chunk",
            expected_sha256=_NORMALIZATION_STABLEHLO_SHA256,
            parameter_shardings=_NORMALIZATION_PARAMETER_SHARDINGS,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda optimized, stable: (
            optimized.replace("bf16[37,6144]", "bf16[38,6144]"), stable
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.gather", "stablehlo.add"),
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.rsqrt", "stablehlo.sqrt"),
        ),
        lambda optimized, stable: (optimized, stable + "\nstablehlo.all_reduce"),
        lambda optimized, stable: (
            optimized.replace(
                "fusion(%input_norm_weight.1,",
                "fusion(%gather_convert_fusion,",
                1,
            ),
            stable,
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%add_rsqrt_fusion, %gather_convert_fusion",
                "%ynn_fusion, %gather_convert_fusion",
            ),
            stable,
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%square.0 = f32[2048,6144]{1,0} multiply(",
                "%square.0 = f32[2048,6144]{1,0} add(",
            ),
            stable,
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(stable, "%2 = chlo.square %1", "%2 = chlo.square %11"),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%15 = stablehlo.multiply %12, %14",
                "%15 = stablehlo.multiply %12, %12",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(stable, "dense<6.144000e+03>", "dense<6.143000e+03>"),
        ),
    ),
)
def test_normalization_boundary_hlo_rejects_hostile_mutations(
    real_boundary_hlo, mutation
):
    optimized, stable, _, _ = real_boundary_hlo
    with pytest.raises(RuntimeError):
        require_completed_normalization_boundary_hlo(*mutation(optimized, stable))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda optimized, stable: (
            optimized.replace("bf16[2048,6144]", "bf16[2047,6144]"), stable
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.while", "stablehlo.case"),
        ),
        lambda optimized, stable: (
            optimized,
            stable.replace("stablehlo.dot_general", "stablehlo.multiply"),
        ),
        lambda optimized, stable: (optimized, stable + "\nstablehlo.rsqrt"),
        lambda optimized, stable: (optimized, stable + "\nall-reduce("),
        lambda optimized, stable: (
            optimized.replace(
                "fusion(%key_norm_bias.1)",
                "fusion(%key_norm_weight.1)",
                1,
            ),
            stable,
        ),
        lambda optimized, stable: (
            _bypass_exact_while_with_all_input_alternate(optimized),
            stable,
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%dot_general.4 = f32[64,128]{1,0} dot(",
                "%dot_general.4 = f32[64,128]{1,0} add(",
            ),
            stable,
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%1 = stablehlo.dot_general %0, %arg0",
                "%1 = stablehlo.dot_general %0, %0",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%25 = stablehlo.add %21, %24",
                "%25 = stablehlo.add %21, %20",
            ),
        ),
        lambda optimized, stable: (
            optimized,
            _replace_once(
                stable,
                "%27 = stablehlo.subtract %25, %26",
                "%27 = stablehlo.subtract %25, %25",
            ),
        ),
        lambda optimized, stable: (
            _replace_once(
                optimized,
                "%concatenate.1 = f32[2048,128]{1,0} concatenate(",
                "%concatenate.1 = f32[2048,128]{1,0} add(",
            ),
            stable,
        ),
    ),
)
def test_key_control_boundary_hlo_rejects_hostile_mutations(
    real_boundary_hlo, mutation
):
    _, _, optimized, stable = real_boundary_hlo
    with pytest.raises(RuntimeError):
        require_normalized_key_control_boundary_hlo(*mutation(optimized, stable))
