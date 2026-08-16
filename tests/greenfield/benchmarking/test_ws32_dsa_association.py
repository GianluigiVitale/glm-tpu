from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from glm_tpu.greenfield.benchmarking.ws32_dsa_association import (
    _attribute_positions,
    _stablehlo_body_sha256,
    classify_ws32_dsa_query_head_contract,
    validate_ws32_dsa_component_hlo,
    validate_ws32_dsa_stablehlo,
)
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores
from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    promote_stage_local_prefill_index_wk,
)
from glm_tpu.greenfield.kernels.ws32_layer import (
    ws32_exact_dsa_current_key,
    ws32_grouped_dsa_query_and_head,
)


def _query_stablehlo(alias_count: int) -> str:
    contract = DsaNumericalContract()
    local_heads = contract.num_heads // alias_count
    local_width = local_heads * contract.head_dim

    def candidate(q_state, normalized, aliases, head_weight, position):
        return ws32_grouped_dsa_query_and_head(
            q_state,
            normalized,
            aliases,
            head_weight,
            position,
            contract=contract,
        )

    return jax.jit(candidate).lower(
        jax.ShapeDtypeStruct((1, contract.q_lora_rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, contract.hidden_size), jnp.bfloat16),
        tuple(
            jax.ShapeDtypeStruct(
                (local_width, contract.q_lora_rank), jnp.float32
            )
            for _ in range(alias_count)
        ),
        jax.ShapeDtypeStruct(
            (local_heads, contract.hidden_size), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    ).as_text()


def _component_stablehlo(component: str) -> str:
    contract = DsaNumericalContract()
    if component.startswith("tuple") and component.endswith("_materializer"):
        alias_count = int(component.removeprefix("tuple").split("_", 1)[0])
        width = contract.num_heads * contract.head_dim // alias_count
        return jax.jit(
            lambda bits, scale: dequantize_fp8_bits_block_weight(
                bits, scale, output_dtype=jnp.float32
            )
        ).lower(
            jax.ShapeDtypeStruct((width, contract.q_lora_rank), jnp.uint8),
            jax.ShapeDtypeStruct(
                (width // 128, contract.q_lora_rank // 128), jnp.float32
            ),
        ).as_text()
    if component == "wk_decode":
        return jax.jit(
            lambda bits, scale: decode_stage_local_prefill_index_wk_bf16(
                bits, scale, contract=contract
            )
        ).lower(
            jax.ShapeDtypeStruct(
                (contract.head_dim, contract.hidden_size), jnp.uint8
            ),
            jax.ShapeDtypeStruct(
                (1, contract.hidden_size // 128), jnp.float32
            ),
        ).as_text()
    if component == "wk_promote":
        return jax.jit(
            lambda value: promote_stage_local_prefill_index_wk(
                value, contract=contract
            )
        ).lower(
            jax.ShapeDtypeStruct(
                (contract.head_dim, contract.hidden_size), jnp.bfloat16
            )
        ).as_text()
    if component == "exact_current_key":
        return jax.jit(
            lambda normalized, wk, weight, bias, position: (
                ws32_exact_dsa_current_key(
                    normalized,
                    wk,
                    weight,
                    bias,
                    position,
                    contract=contract,
                )
            )
        ).lower(
            jax.ShapeDtypeStruct((1, contract.hidden_size), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                (contract.head_dim, contract.hidden_size), jnp.float32
            ),
            jax.ShapeDtypeStruct((contract.head_dim,), jnp.bfloat16),
            jax.ShapeDtypeStruct((contract.head_dim,), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ).as_text()
    if component == "default_score":
        return jax.jit(
            lambda query, keys, head: dsa_scores(
                query, keys, head, precision="default"
            )
        ).lower(
            jax.ShapeDtypeStruct(
                (1, contract.num_heads, contract.head_dim), jnp.float32
            ),
            jax.ShapeDtypeStruct((1024, contract.head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, contract.num_heads), jnp.float32),
        ).as_text()
    raise AssertionError(component)


def _tuple4_hlo() -> str:
    return '''HloModule tuple4

body (w0: f32[1024,2048], q: f32[1,2048], w1: f32[1024,2048], w2: f32[1024,2048], w3: f32[1024,2048]) -> (f32[1024], f32[1024], f32[1024], f32[1024]) {
  %w0 = f32[1024,2048] parameter(0)
  %q = f32[1,2048] parameter(1)
  %w1 = f32[1024,2048] parameter(2)
  %w2 = f32[1024,2048] parameter(3)
  %w3 = f32[1024,2048] parameter(4)
  %s0 = f32[1024,1] slice(%w0), slice={[0:1024], [0:1]}
  %r0 = f32[1024] bitcast(%s0)
  %sq = f32[1,1024] slice(%q), slice={[0:1], [0:1024]}
  %rq = f32[1024] bitcast(%sq)
  %a0 = f32[1024] add(%r0, %rq)
  %s1 = f32[1024,1] slice(%w1), slice={[0:1024], [0:1]}
  %r1 = f32[1024] bitcast(%s1)
  %s2 = f32[1024,1] slice(%w2), slice={[0:1024], [0:1]}
  %r2 = f32[1024] bitcast(%s2)
  %s3 = f32[1024,1] slice(%w3), slice={[0:1024], [0:1]}
  %r3 = f32[1024] bitcast(%s3)
  ROOT %root = (f32[1024], f32[1024], f32[1024], f32[1024]) tuple(%a0, %r1, %r2, %r3)
}

ENTRY main (q_in: bf16[1,2048], w0_in: f32[1024,2048], w1_in: f32[1024,2048], w2_in: f32[1024,2048], w3_in: f32[1024,2048]) -> (f32[1024], f32[1024], f32[1024], f32[1024]) {
  %q_in = bf16[1,2048] parameter(0)
  %w0_in = f32[1024,2048] parameter(1)
  %w1_in = f32[1024,2048] parameter(2)
  %w2_in = f32[1024,2048] parameter(3)
  %w3_in = f32[1024,2048] parameter(4)
  %q = f32[1,2048] convert(%q_in)
  ROOT %fusion = (f32[1024], f32[1024], f32[1024], f32[1024]) fusion(%w0_in, %q, %w1_in, %w2_in, %w3_in), calls=body, backend_config={"megacore_config":{"megacore_allreduce_bytes":"16384"}}
}
'''


def _marker_hlo(*, kernel_name: bool = False) -> str:
    kernel_attribute = (
        ', kernel_name="greenfield_marker_kernel"' if kernel_name else ""
    )
    return f'''HloModule marker

ENTRY main (input: f32[1]) -> f32[1] {{
  %input = f32[1] parameter(0)
  ROOT %call = f32[1] custom-call(%input), custom_call_target="tpu_custom_call"{kernel_attribute}, metadata={{op_name="jit(marker)/pallas_call" stack_frame_id=1}}, backend_config={{"aliasing_operands":{{"lists":[]}},"flag_configs":[]}}
}}
'''


def _copy_async_hlo() -> str:
    return '''HloModule copy_async

ENTRY main (input: f32[1,2048]) -> f32[1,2048] {
  %input = f32[1,2048] parameter(0)
  %start = (f32[1,2048], f32[1,2048], u32[]) copy-start(%input)
  ROOT %done = f32[1,2048] copy-done(%start)
}
'''


def _collective_async_hlo() -> str:
    return '''HloModule collective_async

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

ENTRY main (input: f32[1,2048]) -> f32[1,2048] {
  %input = f32[1,2048] parameter(0)
  %start = (f32[1,2048], f32[1,2048]) all-reduce-start(%input), replica_groups={{0,1,2,3}}, to_apply=%add
  ROOT %done = f32[1,2048] all-reduce-done(%start)
}
'''


def _sharded_stablehlo_envelope() -> str:
    return '''module @sharded {
  sdy.mesh @mesh = <["feature"=4]>
  func.func public @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
  func.func private @helper(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}
'''


def test_exact_stablehlo_bodies_and_arithmetic_mutations() -> None:
    for alias_count in (4, 8):
        component = f"tuple{alias_count}_query_head"
        stablehlo = _query_stablehlo(alias_count)
        assert validate_ws32_dsa_stablehlo(
            stablehlo, component=component
        )["exact"]
        changed = stablehlo.replace(
            "precision = [DEFAULT, DEFAULT]",
            "precision = [HIGHEST, HIGHEST]",
            1,
        )
        assert not validate_ws32_dsa_stablehlo(
            changed, component=component
        )["exact"]

        materializer = f"tuple{alias_count}_materializer"
        exact_materializer = _component_stablehlo(materializer)
        assert validate_ws32_dsa_stablehlo(
            exact_materializer, component=materializer
        )["exact"]
        assert not validate_ws32_dsa_stablehlo(
            exact_materializer.replace("stablehlo.multiply", "stablehlo.add", 1),
            component=materializer,
        )["exact"]

    for component in (
        "wk_decode",
        "wk_promote",
        "exact_current_key",
        "default_score",
    ):
        stablehlo = _component_stablehlo(component)
        assert validate_ws32_dsa_stablehlo(
            stablehlo, component=component
        )["exact"]
        marker = {
            "exact_current_key": "stablehlo.divide",
            "default_score": "stablehlo.maximum",
        }.get(component, "stablehlo.convert")
        assert marker in stablehlo
        assert not validate_ws32_dsa_stablehlo(
            stablehlo.replace(marker, "stablehlo.add", 1),
            component=component,
        )["exact"]


def test_stablehlo_envelope_allows_mesh_and_helpers_before_and_after_main() -> None:
    stablehlo = _sharded_stablehlo_envelope()
    digest = _stablehlo_body_sha256(stablehlo)
    assert digest != _stablehlo_body_sha256(
        stablehlo.replace('["feature"=4]', '["feature"=2]')
    )
    assert digest != _stablehlo_body_sha256(
        stablehlo.replace("return %arg0", "%0 = stablehlo.negate %arg0", 1)
    )


def test_stablehlo_envelope_still_refuses_malformed_modules() -> None:
    stablehlo = _sharded_stablehlo_envelope()
    malformed = (
        stablehlo.replace("module @sharded", "not_a_module @sharded", 1),
        stablehlo.replace("func.func public @main", "func.func private @main", 1),
        stablehlo.replace(
            "  func.func public @main",
            "  func.func public @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {\n"
            "    return %arg0 : tensor<1xf32>\n"
            "  }\n  func.func public @main",
            1,
        ),
        stablehlo.rstrip() + "\ntrailing text",
    )
    for value in malformed:
        with pytest.raises(ValueError, match="module envelope drifted"):
            _stablehlo_body_sha256(value)


def test_optimized_tuple_fusion_binds_every_owner_and_q_source() -> None:
    expected = {
        ("bf16", (1, 2048)): 1,
        ("f32", (1024, 2048)): 4,
    }
    accepted = validate_ws32_dsa_component_hlo(
        _tuple4_hlo(),
        expected_entry_parameters=expected,
        required_16k_fusion_shape=(4, 1024),
    )
    assert accepted["passed"], accepted
    assert accepted["sixteen_kib_fusion_parameter_binding"]

    rogue = _tuple4_hlo().replace(
        "  ROOT %fusion =",
        "  %rogue = f32[1,2048] add(%q, %q)\n  ROOT %fusion =",
    ).replace("fusion(%w0_in, %q,", "fusion(%w0_in, %rogue,")
    assert not validate_ws32_dsa_component_hlo(
        rogue,
        expected_entry_parameters=expected,
        required_16k_fusion_shape=(4, 1024),
    )["passed"]

    wrong_bytes = _tuple4_hlo().replace(
        '"megacore_allreduce_bytes":"16384"',
        '"megacore_allreduce_bytes":"8192"',
    )
    assert not validate_ws32_dsa_component_hlo(
        wrong_bytes,
        expected_entry_parameters=expected,
        required_16k_fusion_shape=(4, 1024),
    )["passed"]

    extra = _tuple4_hlo().replace(
        "%q = f32[1,2048] convert(%q_in)",
        "%extra = f32[1,2048] parameter(5)\n"
        "  %q = f32[1,2048] add(%q_in, %extra)",
    )
    assert not validate_ws32_dsa_component_hlo(
        extra,
        expected_entry_parameters=expected,
        required_16k_fusion_shape=(4, 1024),
    )["passed"]


def test_live_slash_marker_matches_named_scope_path_not_dead_decoy() -> None:
    def marked(value: jax.Array) -> jax.Array:
        with jax.named_scope("greenfield_ws32_linear/feature_reduce"):
            return value * value

    optimized = jax.jit(marked).lower(
        jax.ShapeDtypeStruct((1, 2048), jnp.float32)
    ).compile().as_text()
    result = validate_ws32_dsa_component_hlo(
        optimized,
        expected_entry_parameters={("f32", (1, 2048)): 1},
        required_live_markers=("greenfield_ws32_linear/feature_reduce",),
    )
    assert result["passed"], result

    dead_decoy = _tuple4_hlo().replace(
        "  ROOT %root =",
        "  %dead = f32[1,2048] copy(%q), "
        'metadata={op_name="jit(f)/greenfield_ws32_linear/'
        'feature_reduce/mul"}\n  ROOT %root =',
    )
    rejected = validate_ws32_dsa_component_hlo(
        dead_decoy,
        expected_entry_parameters={
            ("bf16", (1, 2048)): 1,
            ("f32", (1024, 2048)): 4,
        },
        required_live_markers=("greenfield_ws32_linear/feature_reduce",),
        required_16k_fusion_shape=(4, 1024),
    )
    assert not rejected["passed"]
    assert rejected["missing_markers"] == [
        "greenfield_ws32_linear/feature_reduce"
    ]


def test_actual_custom_call_and_kernel_attributes_are_scanner_safe() -> None:
    target = validate_ws32_dsa_component_hlo(
        _marker_hlo(),
        expected_entry_parameters={("f32", (1,)): 1},
        required_live_markers=("tpu_custom_call",),
    )
    assert target["passed"], target

    kernel = validate_ws32_dsa_component_hlo(
        _marker_hlo(kernel_name=True),
        expected_entry_parameters={("f32", (1,)): 1},
        required_live_markers=("greenfield_marker_kernel",),
    )
    assert kernel["passed"], kernel


def test_quoted_attribute_decoy_cannot_supply_a_live_marker() -> None:
    decoy = _tuple4_hlo().replace(
        "calls=body, backend_config=",
        'calls=body, metadata={op_name="jit(f)/fusion" '
        'source_file="decoy custom_call_target=\\"greenfield_fake\\""}, '
        "backend_config=",
    )
    rejected = validate_ws32_dsa_component_hlo(
        decoy,
        expected_entry_parameters={
            ("bf16", (1, 2048)): 1,
            ("f32", (1024, 2048)): 4,
        },
        required_live_markers=("greenfield_fake",),
        required_16k_fusion_shape=(4, 1024),
    )
    assert not rejected["passed"]
    assert rejected["missing_markers"] == ["greenfield_fake"]


def test_attribute_scanner_refuses_unsafe_markers_and_malformed_text() -> None:
    with pytest.raises(ValueError, match="marker is not scanner-safe"):
        _attribute_positions('custom_call_target="ok"', 'custom_call_target="')
    with pytest.raises(ValueError, match="unterminated comment/string"):
        _attribute_positions('custom_call_target="unterminated', "kernel_name=")
    with pytest.raises(ValueError, match="unterminated comment/string"):
        _attribute_positions("/* unterminated", "kernel_name=")


def test_local_async_copy_is_not_misclassified_as_a_collective() -> None:
    result = validate_ws32_dsa_component_hlo(
        _copy_async_hlo(),
        expected_entry_parameters={("f32", (1, 2048)): 1},
    )
    assert result["passed"], result
    assert result["async_collectives"] == []


def test_real_async_collective_remains_fail_closed() -> None:
    result = validate_ws32_dsa_component_hlo(
        _collective_async_hlo(),
        expected_entry_parameters={("f32", (1, 2048)): 1},
        expected_collective_groups=(4,),
    )
    assert not result["passed"]
    assert result["async_collectives"] == [
        "all-reduce-done",
        "all-reduce-start",
    ]


def test_query_arm_contract_classifies_only_16k_hypothesis_miss() -> None:
    accepted = validate_ws32_dsa_component_hlo(
        _tuple4_hlo(),
        expected_entry_parameters={
            ("bf16", (1, 2048)): 1,
            ("f32", (1024, 2048)): 4,
        },
        required_16k_fusion_shape=(4, 1024),
    )
    assert classify_ws32_dsa_query_head_contract(accepted) == "PASSED"

    no_fusion = validate_ws32_dsa_component_hlo(
        _tuple4_hlo().replace("16384", "8192"),
        expected_entry_parameters={
            ("bf16", (1, 2048)): 1,
            ("f32", (1024, 2048)): 4,
        },
        required_16k_fusion_shape=(4, 1024),
    )
    assert (
        classify_ws32_dsa_query_head_contract(no_fusion)
        == "HYPOTHESIS_REJECTED"
    )

    invalid = dict(no_fusion)
    invalid["stablehlo"] = {"exact": False}
    assert classify_ws32_dsa_query_head_contract(invalid) == "INVALID"
