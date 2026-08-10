from __future__ import annotations

import importlib.util
from pathlib import Path

import ml_dtypes
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / (
    "scripts/greenfield/probe_layer0_dsa_query_association.py"
)
WRAPPER = REPO_ROOT / (
    "scripts/greenfield/run_layer0_dsa_query_association_probe.sh"
)
SPEC = importlib.util.spec_from_file_location("dsa_query_association", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_query_comparison_records_exact_and_divergent_values() -> None:
    actual = np.arange(12, dtype=np.float32).reshape(3, 4)
    exact = subject._compare(actual, actual.copy())
    assert exact["elementwise_exact"]
    assert exact["mismatch_count"] == 0
    candidate = actual.copy()
    candidate[1, 2] += np.float32(0.25)
    divergent = subject._compare(actual, candidate)
    assert not divergent["elementwise_exact"]
    assert divergent["mismatch_count"] == 1
    assert divergent["max_abs"] == 0.25


def test_query_candidate_matrix_covers_legacy_and_production_shapes() -> None:
    assert set(subject._candidate_functions()) == {
        "global_m32_n4096",
        "global_m1_n4096",
        "head_lax_map_m32_n128",
        "head_vmap_m32_n128",
        "head_unrolled_m32_n128",
        "head_fori_m32_n128",
        "head_lax_map_highest_m32_n128",
        "head_lax_map_m1_n128",
        "head_unrolled_m1_n128",
        "lp4_unrolled_m32_n1024",
        "lp4_unrolled_m1_n1024",
    }
    assert set(subject._pallas_candidate_functions()) == {
        "pallas_global_m1_n4096",
        "pallas_lp4_m1_n1024",
        "pallas_vector_global_m1_n4096",
        "pallas_vector_lp4_m1_n1024",
        "raw_lookup_global_m1_n128_tiles",
        "raw_lookup_lp4_m1_n128_tiles",
        "raw_materialized_global_m1_n4096",
        "raw_materialized_lp4_m1_n1024",
    }
    assert set(subject._physical_lp4_candidate_modes()) == {
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
    }
    assert set(subject._physical_lp4_q_a_boundary_modes()) == {
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
    }
    assert set(subject._physical_lp4_head_geometry_modes()) == {
        "physical_single_head_sweep_m1_n128",
        "physical_owner_serial_while_m1_n128",
        "physical_global_gspmd_m1_n4096",
    }


def test_physical_lp4_query_associations_are_semantically_equal_on_cpu() -> None:
    q_state = subject.jnp.arange(8, dtype=subject.jnp.float32)[None, :]
    weight = subject.jnp.arange(48, dtype=subject.jnp.float32).reshape(6, 8)
    owner = subject._local_query_projection(
        q_state, weight, head_dim=2, association="owner_dot"
    )
    unrolled = subject._local_query_projection(
        q_state, weight, head_dim=2, association="head_unrolled"
    )
    np.testing.assert_array_equal(np.asarray(owner), np.asarray(unrolled))

    hlo = (
        "bf16[1,2048] u8[1024,2048] f32[8,16] "
        "f32[1024,2048] f32[1,1024] f32[8,128]\n"
        "%fusion = f32[1024] fusion(foo), "
        'metadata={op_name="jit(physical)/dot_general"}'
    )
    contract = subject._physical_lp4_hlo_contract(
        hlo,
        candidate="physical_raw_owner_dot_m1_n1024",
        source="raw",
    )
    assert contract["passed"]
    global_weight = subject._physical_lp4_hlo_contract(
        hlo + " f32[4096,2048]",
        candidate="physical_raw_owner_dot_m1_n1024",
        source="raw",
    )
    assert not global_weight["passed"]
    head_hlo = hlo.replace("f32[1,1024]", "f32[128]")
    head = subject._physical_lp4_hlo_contract(
        head_hlo,
        candidate="physical_raw_head_unrolled_m1_n128",
        source="raw",
    )
    assert head["passed"]
    assert head["physical_projection_width"] == 128


def test_physical_q_a_boundary_contract_requires_the_requested_barrier(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subject,
        "_validate_fused_qkv_a_decoder_association",
        lambda hlo, layers: {
            "passed": hlo.startswith("optimized") and layers == 1
        },
    )
    optimized = (
        "optimized bf16[1,6144] bf16[1,2048] u8[1024,2048] "
        "f32[8,16] f32[1024,2048] f32[1024] f32[8,128]"
    )
    rounded = subject._physical_lp4_q_a_boundary_hlo_contract(
        optimized,
        "stablehlo.optimization_barrier stablehlo.optimization_barrier "
        "stablehlo.dot_general "
        "precision = [HIGHEST, HIGHEST]",
        backend="tpu",
        candidate=(
            "physical_fused_q_a_bf16_barrier_highest_owner_dot_m1_n1024"
        ),
        enforce_bfloat16_boundary=True,
        highest=True,
    )
    assert rounded["passed"]
    missing = subject._physical_lp4_q_a_boundary_hlo_contract(
        optimized,
        "stablehlo.dot_general precision = [HIGHEST, HIGHEST]",
        backend="tpu",
        candidate=(
            "physical_fused_q_a_bf16_barrier_highest_owner_dot_m1_n1024"
        ),
        enforce_bfloat16_boundary=True,
        highest=True,
    )
    assert not missing["passed"]
    unexpected = subject._physical_lp4_q_a_boundary_hlo_contract(
        optimized,
        "stablehlo.optimization_barrier stablehlo.optimization_barrier "
        "stablehlo.dot_general",
        backend="tpu",
        candidate=(
            "physical_fused_q_a_unrounded_default_owner_dot_m1_n1024"
        ),
        enforce_bfloat16_boundary=False,
        highest=False,
    )
    assert not unexpected["passed"]


def test_physical_head_geometry_preserves_owner_head_order() -> None:
    q_state = subject.jnp.arange(8, dtype=subject.jnp.float32)[None, :]
    weight = subject.jnp.arange(48, dtype=subject.jnp.float32).reshape(6, 8)
    owner = subject._local_query_projection(
        q_state, weight, head_dim=2, association="owner_dot"
    )
    serial = subject._serial_head_query_projection(
        q_state, weight, head_dim=2
    )
    np.testing.assert_array_equal(np.asarray(owner), np.asarray(serial))

    single_hlo = (
        "bf16[1,2048] f32[128,2048] f32[1,128]\n"
        "%reduce = f32[128] fusion(foo), "
        'metadata={op_name="jit(single)/dot_general"}'
    )
    single = subject._physical_lp4_head_geometry_hlo_contract(
        single_hlo,
        candidate="physical_single_head_sweep_m1_n128",
    )
    assert single["passed"]
    serial_hlo = (
        "bf16[1,2048] f32[1024,2048] f32[1,128] f32[8,128]\n"
        "%loop = while(foo)\n"
        "%reduce = f32[128] fusion(foo), "
        'metadata={op_name="jit(serial)/dot_general"}'
    )
    serial_contract = subject._physical_lp4_head_geometry_hlo_contract(
        serial_hlo,
        candidate="physical_owner_serial_while_m1_n128",
    )
    assert serial_contract["passed"]
    assert serial_contract["while_count"] == 1
    assert not subject._physical_lp4_head_geometry_hlo_contract(
        serial_hlo.replace("%loop = while(foo)", "%loop = add(foo)"),
        candidate="physical_owner_serial_while_m1_n128",
    )["passed"]

    global_hlo = (
        "hlo_module physical, num_partitions=4\n"
        "bf16[1,2048] f32[1024,2048] f32[1,1024] f32[1,8,128]\n"
        "%reduce = f32[1024] fusion(foo), "
        'metadata={op_name="jit(global)/dot_general"}'
    )
    global_contract = subject._physical_lp4_head_geometry_hlo_contract(
        global_hlo,
        candidate="physical_global_gspmd_m1_n4096",
    )
    assert global_contract["passed"]
    assert global_contract["explicit_four_partitions"]
    assert not subject._physical_lp4_head_geometry_hlo_contract(
        global_hlo.replace(", num_partitions=4", ""),
        candidate="physical_global_gspmd_m1_n4096",
    )["passed"]


def test_physical_global_gspmd_stablehlo_requires_logical_sharding() -> None:
    stablehlo = (
        "module attributes {mhlo.num_partitions = 4 : i32}\n"
        'stablehlo.parameter {mhlo.sharding = "{replicated}"} '
        ": tensor<1x2048xbf16>\n"
        'stablehlo.parameter {mhlo.sharding = "{devices=[4,1]0,1,2,3}", '
        'mesh_axis = "lp4"} '
        ": tensor<4096x2048xf32>\n"
        'stablehlo.return {mhlo.sharding = "{devices=[1,4,1]0,1,2,3}", '
        'mesh_axis = "lp4"} '
        ": tensor<1x32x128xf32>"
    )
    contract = subject._physical_global_gspmd_stablehlo_contract(stablehlo)
    assert contract["passed"]
    assert contract["sharding_annotation_count"] == 3
    assert contract["explicit_four_partitions"]
    assert not subject._physical_global_gspmd_stablehlo_contract(
        stablehlo.replace("mhlo.sharding", "attribute", 1)
        .replace("mhlo.sharding", "attribute", 1)
        .replace("mhlo.sharding", "attribute", 1)
    )["passed"]


def test_q_a_candidate_matrix_is_one_row_and_shard_major() -> None:
    assert set(subject._q_a_candidate_modes()) == {
        ("lax_map_convolution", norm)
        for norm in (
            "logical_mean",
            "shard_sum",
            "left_fold",
            "topology_tree",
        )
    }
    accepted = np.asarray(
        [0.0, 1.0, -2.5], dtype=ml_dtypes.bfloat16
    ).view(np.uint16)
    exact = subject._compare_bfloat16_bits(accepted, accepted.copy())
    assert exact["elementwise_exact"]
    candidate = accepted.copy()
    candidate[1] = np.asarray(
        [1.0078125], dtype=ml_dtypes.bfloat16
    ).view(np.uint16)[0]
    divergent = subject._compare_bfloat16_bits(accepted, candidate)
    assert not divergent["elementwise_exact"]
    assert divergent["mismatch_count"] == 1

    hlo = (
        "bf16[1,6144] bf16[1,2048] "
        "f8e4m3fn[32,6144,82] f32[32,48,82]"
    )
    contract = subject._q_a_hlo_contract(
        hlo, candidate="virtual_vmap_shard_sum_m1_n82"
    )
    assert contract["passed"]
    dead = subject._q_a_hlo_contract(
        hlo + " bf16[32,6144]",
        candidate="virtual_vmap_shard_sum_m1_n82",
    )
    assert not dead["passed"]
    collective = subject._q_a_hlo_contract(
        hlo + " reduce-scatter",
        candidate="virtual_vmap_shard_sum_m1_n82",
    )
    assert not collective["passed"]
    convolution = subject._q_a_hlo_contract(
        hlo
        + "\n%convolution = f32[1,82] convolution(foo), "
        "dim_labels=bf_io->bf",
        candidate="virtual_lax_map_convolution_shard_sum_m1_n82",
    )
    assert convolution["passed"]
    missing_convolution = subject._q_a_hlo_contract(
        hlo,
        candidate="virtual_lax_map_convolution_shard_sum_m1_n82",
    )
    assert not missing_convolution["passed"]


def test_protected_query_wrapper_is_bounded_and_fail_closed() -> None:
    source = WRAPPER.read_text()
    for required in (
        "CAPTURE_SUCCESS_SHA=20deb19c",
        "CAPTURE_COMPARISON_SHA=623818dc",
        "CAPTURE_TENSORS_SHA=0a724bad",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "timeout --signal=TERM --kill-after=60 1800",
        "strict_census pre",
        "strict_census post",
        "results_ckpt.db",
        "remote_objects.json",
        "performance_claim",
        "GLM_GREENFIELD_DSA_ASSOCIATION_TARGET",
        "--target \"$TARGET\"",
        "q_a_candidates.npz",
        "qkv_a_production.npz",
        "qkv_a_production_association",
        "DB502_CODE_HASH=c230c11",
        "DB502_RUNNER_SHA=2a77d75d",
        "DB502_TENSOR_SHA=d9b14bdd",
        "DB502_SUCCESS_SHA=de2e080d",
        "CURRENT_INTERNAL_SHA=e1366c58",
        "physical_lp4_query_candidates.npz",
        "physical_lp4_dsa_query_association",
        "query_lp4_q_a_boundary",
        "physical_lp4_q_a_boundary.npz",
        "physical_lp4_dsa_q_a_boundary_association",
        "query_lp4_head_geometry",
        "physical_lp4_head_geometry.npz",
        "physical_lp4_dsa_head_geometry_association",
    ):
        assert required in source
    for forbidden in ("launch_glm_32chip.sh", "glm_longctx.py"):
        assert forbidden not in source


def test_production_qkv_a_probe_reuses_integrated_helper_and_linter() -> None:
    source = SCRIPT.read_text()
    for required in (
        "_project_attention_qkv_a",
        "_validate_fused_qkv_a_decoder_association",
        '"query_lp4_q_a_boundary",',
        '"query_lp4_head_geometry",',
        "production_fused_n82_convolution_shard_sum",
        "companion_comparison",
        "fused_n82_convolution",
    ):
        assert required in source
