from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

from scripts.greenfield.compile_short_decoder import (
    _canonicalize_dsa_internal_observation,
    _canonicalize_layer_residual_observation,
    _encode_bfloat16_bits,
    _expected_stage_visit_mask,
    _load_short_context_dsa_oracle,
    _materialize_global_array,
    _observer_hlo_isolation_contract,
    _protected_short_context_label,
    _raw_token_sequence_contract,
    _runtime_pipeline_groups,
    _validate_dsa_observation_step,
    _validate_completed_step_selected_states,
    _validate_layer0_ingredients,
    _validate_strategy_nd_canary_hlo,
    _validate_token_observation_step,
)
from scripts.greenfield import compile_short_decoder as compile_module


REPO = Path(__file__).resolve().parents[3]
PROTECTED_RUNNER = REPO / "scripts/greenfield/run_short_decoder_compile_pp8.sh"
PROTECTED_8K_RUNNER = (
    REPO / "scripts/greenfield/run_short_decoder_compile_pp8_8k.sh"
)


def test_runtime_pipeline_groups_follow_final_layout_rank_order() -> None:
    physical_groups = (
        (0, 1),
        (8, 9),
        (16, 17),
        (24, 25),
    )
    layout = SimpleNamespace(
        devices=tuple(
            SimpleNamespace(device_id=device_id)
            for group in physical_groups
            for device_id in group
        )
    )
    schedule = SimpleNamespace(
        stages=tuple(
            SimpleNamespace(
                assignment=SimpleNamespace(device_ids=group)
            )
            for group in physical_groups
        )
    )

    groups, pairs = _runtime_pipeline_groups(schedule, layout)

    assert groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert pairs == (
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 0),
        (7, 1),
    )


def test_runtime_pipeline_groups_reject_incomplete_layout() -> None:
    layout = SimpleNamespace(
        devices=(SimpleNamespace(device_id=0), SimpleNamespace(device_id=1))
    )
    schedule = SimpleNamespace(
        stages=(
            SimpleNamespace(
                assignment=SimpleNamespace(device_ids=(0,))
            ),
        )
    )

    with pytest.raises(ValueError, match="cover every runtime-layout rank"):
        _runtime_pipeline_groups(schedule, layout)


def test_expected_stage_visit_mask_is_plan_aware() -> None:
    assert _expected_stage_visit_mask(8) == 255
    assert _expected_stage_visit_mask(16) == 65535


@pytest.mark.parametrize("stage_count", [0, -1, 32])
def test_expected_stage_visit_mask_rejects_unsupported_counts(
    stage_count: int,
) -> None:
    with pytest.raises(ValueError, match="signed int32"):
        _expected_stage_visit_mask(stage_count)


@pytest.mark.parametrize("stage_count", [True, 8.0])
def test_expected_stage_visit_mask_rejects_non_integer_counts(
    stage_count: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        _expected_stage_visit_mask(stage_count)  # type: ignore[arg-type]


def test_protected_runner_pins_fp32_feature_boundary_kernel() -> None:
    source = PROTECTED_RUNNER.read_text()
    assert (
        'if feature_reconstruct_down_fp32:\n'
        '        expected_kernel_counts["greenfield_fp32_to_bf16_r8_h6144"] = 75'
        in source
    )


def test_protected_runner_exposes_fail_closed_device_roundtrip() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--verify-device-roundtrip"' in compiler
    assert "verify_device_roundtrip=args.verify_device_roundtrip" in compiler
    assert (
        "readonly VERIFY_DEVICE_ROUNDTRIP="
        "${GLM_GREENFIELD_RUNTIME_DEVICE_ROUNDTRIP:-0}" in runner
    )
    assert '--verify-device-roundtrip "$verify_device_roundtrip"' in runner
    assert '["device_roundtrip_verified"]\n    != verify_device_roundtrip' in runner
    assert '["device_roundtrip_bytes"]\n    != expected_roundtrip_bytes' in runner


def test_layer0_ingredient_capture_is_default_off_and_protected() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--observe-layer0-ingredients"' in compiler
    assert "build_layer0_ingredients_observer" in compiler
    assert "validate_layer0_ingredients_observer_hlo" in compiler
    assert "${GLM_GREENFIELD_LAYER0_INGREDIENTS:-0}" in runner
    assert "--observe-layer0-ingredients 1" in runner
    assert "layer0_ingredients.optimized_hlo.txt.gz" in compiler
    assert "source_state\": \"post_teacher_forced_prefill" in compiler
    assert "requires the proven table-on 8K" in compiler
    assert "main_rope_table_enabled=args.main_rope_table" in compiler
    assert '"run_tag": greenfield_run_tag' in compiler
    assert "requires GLM_GREENFIELD_RUN_TAG" in compiler
    assert "ingredients_inputs = (*runtime_prefix, *tuple(output))" in compiler
    assert "LAYER0_INGREDIENTS_DIAGNOSTIC_CONTRACT_OK" in runner
    assert "layer-0 ingredients require the proven main-RoPE table" in runner
    assert '"$remote/layer0_ingredients/"' in runner
    paired_dsa_observer_gate = (
        "observe_layer0_discriminator or args.observe_layer0_ingredients"
    )
    assert compiler.count(paired_dsa_observer_gate) == 2


def test_layer0_ingredient_contract_validates_owner_partition() -> None:
    names = compile_module.LAYER0_INGREDIENT_NAMES
    groups = tuple(tuple(range(stage * 4, stage * 4 + 4)) for stage in range(8))
    expected_positions = np.asarray([0, 1, 128, 129, -1, -1, -1, -1], np.int32)
    expected_scores = np.asarray([4, 3, 2, 1, -np.inf, -np.inf, -np.inf, -np.inf], np.float32)
    shapes = {
        "selected_positions": (8,),
        "selected_scores": (8,),
        "selected_valid_counts": (1,),
        "normalized_input": (6,),
        "combined_residual": (6,),
        "current_cache_row": (4,),
        "owner_selected_positions": (8,),
        "owner_selected_valid_counts": (1,),
        "owner_selected_cache_values": (8, 4),
        "owner_selected_cache_valid": (1,),
        "sparse_partial_output": (2, 3),
        "sparse_partial_logsumexp": (2,),
        "sparse_partial_valid": (1,),
        "combined_attention_output": (2, 3),
        "combined_attention_logsumexp": (2,),
        "combined_attention_valid": (1,),
        "value_states": (1, 2),
        "attention_output_input": (2,),
        "attention_virtual_partials": (2, 6),
        "attention_local_update": (6,),
        "attention_reduced_update": (6,),
        "normalized_mlp": (6,),
        "post_attention_residual": (6,),
        "dense_virtual_partials": (2, 6),
        "dense_local_update": (6,),
        "dense_reduced_update": (6,),
        "next_hidden": (6,),
        "layer1_normalized": (6,),
        "contract_valid": (1,),
    }
    integer_names = {
        "selected_positions",
        "selected_valid_counts",
        "owner_selected_positions",
        "owner_selected_valid_counts",
    }
    boolean_names = {
        "owner_selected_cache_valid",
        "sparse_partial_valid",
        "combined_attention_valid",
        "contract_valid",
    }
    float32_names = {
        "selected_scores",
        "sparse_partial_logsumexp",
        "combined_attention_logsumexp",
    }
    observed = {}
    for name in names:
        shape = (32, *shapes[name])
        if name in integer_names:
            fill = -1 if "positions" in name else 0
            observed[name] = np.full(shape, fill, dtype=np.int32)
        elif name in boolean_names:
            observed[name] = np.zeros(shape, dtype=np.bool_)
        elif name in float32_names:
            observed[name] = np.full(shape, -np.inf, dtype=np.float32)
        else:
            observed[name] = np.zeros(shape, dtype=bfloat16)

    active = np.asarray(groups[0], dtype=np.int32)
    for name, value in observed.items():
        if name in boolean_names:
            value[active] = True
        elif name in float32_names:
            value[active] = 0
        elif name not in integer_names:
            value[active] = bfloat16(1)
    observed["selected_positions"][active] = expected_positions
    observed["selected_scores"][active] = expected_scores
    observed["selected_valid_counts"][active] = 4
    observed["owner_selected_positions"][active[0], :2] = (0, 1)
    observed["owner_selected_positions"][active[1], :2] = (128, 129)
    observed["owner_selected_valid_counts"][active[:2]] = 2
    observed["owner_selected_cache_values"][active] = 0
    observed["owner_selected_cache_values"][active[0], :2] = bfloat16(1)
    observed["owner_selected_cache_values"][active[1], :2] = bfloat16(2)

    artifact, contract = _validate_layer0_ingredients(
        observed,
        groups=groups,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_count=4,
        logical_page_size=512,
    )
    assert contract["passed"]
    assert contract["owner_union_exact"]
    assert "owner_selected_cache_values_bfloat16_bits" in artifact

    observed["owner_selected_positions"][active[0], 0] = 128
    _, rejected = _validate_layer0_ingredients(
        observed,
        groups=groups,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_count=4,
        logical_page_size=512,
    )
    assert not rejected["owner_partition_passed"]
    assert not rejected["passed"]


def test_protected_runner_classifies_prefill_loops_fail_closed() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert '["outer_loop_count"] != 1' in runner
    assert '"expected_fused_qkv_internal_loop_count"' in runner
    assert '"fused_qkv_internal_loop_count"' in runner
    assert '"unclassified_loops"' in runner
    assert '["loop_count"] != 1' not in runner


def test_selected_linear_runtime_defaults_to_fused_qkv_gate_b_artifact() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert (
        "greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z"
        in runner
    )
    assert (
        "123394906a153238e464fc096b626c77996b7cf22b95077b9377b8dcafbe699a"
        in runner
    )
    assert (
        "523afb1dc1ff2b954a9795c4deabdc4fd599c244c0bf1a9e3b8f971700548cb4"
        in runner
    )
    assert (
        "protected prefill repair requires the Gate-B-approved fused qkv-a runtime"
        in runner
    )


def test_pp8_runner_supports_default_off_metadata_parent_lineage() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert (
        "readonly FEATURE_SOURCE_METADATA_ONLY="
        "${GLM_GREENFIELD_FEATURE_SOURCE_METADATA_ONLY:-0}" in runner
    )
    assert (
        "SOURCE_ROOT=${GLM_GREENFIELD_SOURCE_CHECKPOINT_ROOT:-" in runner
    )
    assert (
        "SOURCE_RUNTIME_ROOT=${GLM_GREENFIELD_SOURCE_RUNTIME_ROOT:-" in runner
    )
    assert "feature source metadata-only flag must be 0 or 1" in runner
    assert (
        "--feature-source-metadata-only $FEATURE_SOURCE_METADATA_ONLY "
        "--feature-output-tile" in runner
    )
    assert "status --porcelain --untracked-files=no" in runner
    assert "METADATA_SOURCE_SUFFIX=_metaparent" in runner


def test_pp8_runner_reads_restart_safe_prerequisites_from_approved_mount() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert "PREREQUISITE_DIR=/home/gianl/glm-run/" not in runner
    assert runner.count("PREREQUISITE_DIR=/home/gianl/gcs-models/") == 14


def test_pp8_runner_has_non_tpu_full_metadata_preflight() -> None:
    runner = PROTECTED_RUNNER.read_text()

    assert (
        "readonly PREFLIGHT_ONLY="
        "${GLM_GREENFIELD_SHORT_DECODER_PREFLIGHT_ONLY:-0}" in runner
    )
    assert "short decoder preflight-only flag must be 0 or 1" in runner
    assert "scripts/greenfield/inspect_feature_runtime_checkpoint.py" in runner
    assert "SHORT_DECODER_PREFLIGHT_OK" in runner
    assert runner.index("exec 9>/home/gianl/glm-run/.glm_pod_workload.lock") < (
        runner.index("exec 8>/home/gianl/.glm-tpu-rsync.lock")
    )


def test_prefill_index_repair_is_default_off_and_prerequisites_pinned() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--prefill-index-repair"' in compiler
    assert "observe_prefill_index_inputs=True" in compiler
    assert '"prefill_index_repair": args.prefill_index_repair' in compiler
    assert (
        "readonly PREFILL_INDEX_REPAIR="
        "${GLM_GREENFIELD_PREFILL_INDEX_REPAIR:-0}" in runner
    )
    assert '--prefill-index-repair "$prefill_index_repair"' in runner
    assert "prefill index repair requires the protected 8K" in runner
    assert (
        "prefill index repair requires the accepted split residual state"
        in compiler
    )
    assert (
        "prefill index repair must remain isolated from residual observation"
        in compiler
    )
    assert (
        "readonly DSA_INTERNAL_BASELINE_NPZ="
        "${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_NPZ:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_BASELINE_SHA="
        "${GLM_GREENFIELD_DSA_INTERNAL_BASELINE_SHA:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_LAYER0_REFERENCE_NPZ="
        "${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_NPZ:-" in runner
    )
    assert (
        "readonly DSA_INTERNAL_LAYER0_REFERENCE_SHA="
        "${GLM_GREENFIELD_DSA_INTERNAL_LAYER0_REFERENCE_SHA:-" in runner
    )
    assert "args.prefill_index_repair and args.observe_dsa_internals" not in compiler
    assert (
        "LAYER_RESIDUAL_OBSERVER == 0 && $DSA_INTERNAL_OBSERVER == 0"
        not in runner
    )
    assert '"--observe-layer0-residual-variants"' in compiler
    assert '"--observe-layer0-subshard-variants"' in compiler
    assert '"--observe-layer0-attention-schedule-variants"' in compiler
    assert "build_layer0_residual_discriminator" in compiler
    assert "validate_layer0_residual_discriminator_hlo" in compiler
    assert "decoder.layer0_residual_discriminators" in compiler
    assert '"four_distinct_hlo_modules"' in compiler
    assert '"all_distinct_hlo_modules"' in compiler
    assert '"hlo_suite_contract"' in compiler
    assert "for variant_name, compiled_discriminator in" in compiler
    assert "*runtime_prefix, *tuple(output)" in compiler
    assert "LAYER1_CURRENT_NORMALIZED_HIDDEN_SHA256" in compiler
    assert (
        "readonly LAYER0_RESIDUAL_VARIANTS="
        "${GLM_GREENFIELD_LAYER0_RESIDUAL_VARIANTS:-0}" in runner
    )
    assert (
        "readonly LAYER0_SUBSHARD_VARIANTS="
        "${GLM_GREENFIELD_LAYER0_SUBSHARD_VARIANTS:-0}" in runner
    )
    assert (
        "readonly LAYER0_ATTENTION_VARIANTS="
        "${GLM_GREENFIELD_LAYER0_ATTENTION_VARIANTS:-0}" in runner
    )
    assert '--observe-layer0-residual-variants 1' in runner
    assert '--observe-layer0-subshard-variants 1' in runner
    assert '--observe-layer0-attention-schedule-variants 1' in runner
    assert "layer-0 discriminator requires the complete exact recurrent" in runner
    assert "ISOLATED_RESIDUAL_PREREQUISITE_TAG" in runner
    assert "isolated residual direct remote contract hash drifted" in runner
    assert '"dcp_then_model_sequential_bf16"' in compiler
    assert '"model_then_dcp_pairwise_bf16"' in compiler
    assert '"schema_version": 18' in compiler
    assert 'record["schema_version"] for record in records} != {18}' in runner
    assert "results_db_run_id\": 518" in runner
    assert (
        "a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee"
        in runner
    )
    assert "_prefill_keyfix" in runner
    assert 'repair["expected_call_count"] != 84' in runner
    assert 'repair["physical_sqrt_count"] != 168' in runner
    assert 'repair["repair_collectives"]' in runner
    assert 'repair["repair_weight_round_count"] != 0' in runner
    assert "results_db_run_id\": 519" in runner
    assert "results_db_run_id\": 520" in runner
    assert "greenfield_layer0_prompt_key_norm_m64_20260809T200559393031635Z" in runner
    assert "prefill_index_weight_split_prerequisite" in runner
    assert '"--dsa-query-exact-association"' in compiler
    assert (
        "readonly DSA_QUERY_EXACT_ASSOCIATION="
        "${GLM_GREENFIELD_DSA_QUERY_EXACT_ASSOCIATION:-0}" in runner
    )
    assert '--dsa-query-exact-association "$dsa_query_exact_association"' in runner
    assert "dsa_observer.input_specs[state_spec_offset + 4]" in compiler
    assert "dsa_observer.input_specs[5]" not in compiler
    assert "results_db_run_id\": 525" in runner
    assert "physical_owner_tuple4_barrier_m1_n1024" in runner
    assert "results_db_run_id\": 526" in runner
    assert "physical_production_fused_q_a_tuple4_exact_m1_n1024" in runner
    assert (
        "greenfield_layer0_physical_lp4_dsa_query_production_exact_"
        "20260810T080508327295662Z" in runner
    )
    assert "DB526 direct remote SUCCESS hash drifted" in runner
    assert "dsa_query_exact_association_production_prerequisite" in runner
    assert "_queryexact" in runner
    assert '"--dsa-head-key-exact-association"' in compiler
    assert (
        "readonly DSA_HEAD_KEY_EXACT_ASSOCIATION="
        "${GLM_GREENFIELD_DSA_HEAD_KEY_EXACT_ASSOCIATION:-0}" in runner
    )
    assert (
        '--dsa-head-key-exact-association '
        '"$dsa_head_key_exact_association"' in runner
    )
    assert "results_db_run_id\": 527" in runner
    assert "physical_normalized_barrier_materialized_divide_sqrt" in runner
    assert "DB527 direct remote SUCCESS hash drifted" in runner
    assert "dsa_head_key_exact_association_prerequisite" in runner
    assert "_headkeyexact" in runner
    assert "runtime_prefix = (" in compiler
    assert "*runtime_prefix, *observer_current" in compiler
    assert "step_inputs = (*runtime_prefix, *values)" in compiler
    assert "return compiled(*step_inputs)" in compiler
    assert "DB520 direct remote SUCCESS hash drifted" in runner
    assert "external_stage_local_raw_fp8_to_bf16" in runner
    assert "external_stage_local_bf16_to_fp32" in runner
    assert "prefill_wk_materialization_hlo_contract" in runner
    assert '"--dsa-score-default-precision"' in compiler
    assert (
        "readonly DSA_SCORE_DEFAULT_PRECISION="
        "${GLM_GREENFIELD_DSA_SCORE_DEFAULT_PRECISION:-0}" in runner
    )
    assert (
        '--dsa-score-default-precision "$dsa_score_default_precision"'
        in runner
    )
    assert (
        "args.dsa_score_default_precision = bool(\n"
        "        args.dsa_score_default_precision\n"
        "    )"
        in compiler
    )
    assert "default DSA score precision requires the protected 8K" in runner
    assert "default DSA score precision requires the exact repaired" in runner
    assert "results_db_run_id\": 529" in runner
    assert "DB529 direct remote SUCCESS hash drifted" in runner
    assert "dsa_score_default_precision_prerequisite" in runner
    assert "highest_precision_contraction_count" in runner
    assert "_scoredefault" in runner


def test_main_rope_table_is_default_off_and_db531_protected() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--main-rope-table"' in compiler
    assert "main_rope_table_enabled=args.main_rope_table" in compiler
    assert '"main_rope_table_enabled": decoder.main_rope_table_enabled' in compiler
    assert '"main_rope_table_sha256": decoder.main_rope_table_sha256' in compiler
    assert '"main_rope_table_shape": (' in compiler
    assert '"main_rope_table_bytes_per_device": (' in compiler
    assert '"main_rope_table_local_device_sha256": (' in compiler
    assert "decoder main-RoPE table asset is unavailable" in compiler
    assert "device main-RoPE table identity drifted" in compiler
    assert "default decoder materialized a main-RoPE table" in compiler
    assert (
        "readonly MAIN_ROPE_TABLE="
        "${GLM_GREENFIELD_MAIN_ROPE_TABLE:-0}" in runner
    )
    assert '--main-rope-table "$main_rope_table"' in runner
    assert "main-RoPE table requires the protected 8K" in runner
    assert "main-RoPE table requires the exact repaired recurrent DSA chain" in runner
    assert "main-RoPE table admits only the isolated attention" in runner
    assert "greenfield_layer0_main_rope_20260811T072231959104598Z" in runner
    assert (
        "$APPROVED_BUCKET/oracles/greenfield/glm52/"
        "main_rope_association/8k/$MAIN_ROPE_PREREQUISITE_TAG" in runner
    )
    assert "DB531 direct remote SUCCESS hash drifted" in runner
    assert "expected_main_rope_sha256" in runner
    assert "validate_main_rope_hlo" in runner
    assert 'contract["fp32_multiply_count"] < 78 * 8' in runner
    assert 'contract["fp32_combine_count"] < 78 * 4' in runner
    assert 'contract["final_round_count"] < 78 * 2' in runner
    assert "_mainrope" in runner


def test_pregathered_b512_attention_is_default_off_and_db537_protected() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--pregathered-b512-attention"' in compiler
    assert (
        "pregathered_b512_attention=(\n"
        "                args.pregathered_b512_attention\n"
        "            )" in compiler
    )
    assert '"pregathered_b512_attention": (' in compiler
    prefill = (
        REPO / "glm_tpu/greenfield/runtime/prefill.py"
    ).read_text()
    assert "pregathered_b512_attention=(" in prefill
    assert "decoder.pregathered_b512_attention" in prefill
    assert (
        "readonly PREGATHERED_B512_ATTENTION="
        "${GLM_GREENFIELD_PREGATHERED_B512_ATTENTION:-0}" in runner
    )
    assert (
        '--pregathered-b512-attention "$pregathered_b512_attention"'
        in runner
    )
    assert "pregathered-B512 attention requires the protected 8K" in runner
    assert "PREGATHERED_ATTENTION_PREREQUISITE_TAG" in runner
    assert "greenfield_layer0_attention_arithmetic_20260812T114701365714147Z" in runner
    assert "DB537 attention-arithmetic prerequisite drifted" in runner
    assert "DB537 prerequisite DB linkage drifted" in runner
    assert "DB537 direct remote $remote_file hash drifted" in runner
    assert "validate_pregathered_attention_hlo" in runner
    assert 'contract["exchange_count"] != 78' in runner
    assert 'contract["kernel_count"] != 78' in runner
    assert "greenfield_pregathered_b512_attention_prerequisite_db_run" in runner
    assert "if pregathered_b512_attention else None" in runner
    assert "_pregatheredb512" in runner


def test_strategy_nd_attention_projection_is_default_off_and_db539_protected() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()
    decoder = (REPO / "glm_tpu/greenfield/runtime/decoder.py").read_text()
    prefill = (REPO / "glm_tpu/greenfield/runtime/prefill.py").read_text()

    assert '"--strategy-nd-attention-projection"' in compiler
    assert '"GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION", "0"' in compiler
    assert "strategy_nd_attention_projection=False" not in compiler
    assert '"strategy_nd_attention_projection": (' in compiler
    assert "strategy_nd_attention_projection: bool = False" in decoder
    assert "decoder.strategy_nd_attention_projection" in prefill
    assert "decoder_stablehlo = lowered.as_text()" in compiler
    assert "stablehlo=decoder_stablehlo" in compiler
    assert "stablehlo=dsa_observer_stablehlo" in compiler
    assert "stablehlo=prefill_stablehlo" in compiler
    assert 'f"{hlo_stem}.stablehlo.mlir.gz"' in compiler
    assert "validate_strategy_nd_attention_stablehlo" in decoder
    assert (
        "readonly STRATEGY_ND_ATTENTION_PROJECTION="
        "${GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION:-0}" in runner
    )
    assert (
        "export GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION="
        "$STRATEGY_ND_ATTENTION_PROJECTION" in runner
    )
    assert "StrategyND attention projection requires the protected 8K" in runner
    assert "STRATEGY_ND_ATTENTION_PREREQUISITE_TAG" in runner
    assert "greenfield_legacy_layer0_attention_update_p8155_20260812T172809039093068Z" in runner
    assert "DB539 StrategyND attention prerequisite drifted" in runner
    assert "DB539 prerequisite DB linkage drifted" in runner
    assert "DB539 direct remote $remote_file hash drifted" in runner
    assert "validate_strategy_nd_attention_hlo" in runner
    assert 'contract["gather_count"] != 78' in runner
    assert 'contract["kernel_count"] != 624' in runner
    assert 'stable_contract["matched_tree_count"] != 78' in runner
    assert 'stable_contract["expected_kernel_count"] != 624' in runner
    assert "greenfield_fp8_strategy_nd_o_m8_k512_n6144" in runner
    assert 'record["fleet_stablehlo_hashes"][0]' in runner
    assert 'record["fleet_prefill_stablehlo_hashes"][0]' in runner
    assert 'record["fleet_dsa_observer_stablehlo_hashes"][0]' in runner
    assert "greenfield_strategy_nd_attention_projection_prerequisite_db_run" in runner
    assert "539 if strategy_nd_attention_projection else None" in runner
    assert "_strategynd_o" in runner
    assert runner.index("strict_census post") < runner.index("pv.start_run(")


def test_dense_final_layout_convolution_is_default_off_and_db548_selected() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()
    decoder = (REPO / "glm_tpu/greenfield/runtime/decoder.py").read_text()
    prefill = (REPO / "glm_tpu/greenfield/runtime/prefill.py").read_text()

    assert '"--dense-final-layout-convolution"' in compiler
    assert (
        '"--dense-final-layout-convolution",\n'
        '        type=int,\n'
        '        choices=(0, 1),\n'
        '        default=0,' in compiler
    )
    assert "dense_final_layout_convolution: bool = False" in decoder
    assert "decoder.dense_final_layout_convolution" in prefill
    assert (
        "readonly DENSE_FINAL_LAYOUT_CONVOLUTION="
        "${GLM_GREENFIELD_DENSE_FINAL_LAYOUT_CONVOLUTION:-0}" in runner
    )
    assert (
        '--dense-final-layout-convolution "$dense_final_layout_convolution"'
        in runner
    )
    assert "requires the complete protected 8K exactness chain" in runner
    assert (
        "greenfield_layer0_dense_envelope_cross_layer_"
        "20260813T120703034434907Z" in runner
    )
    assert (
        "greenfield_layer0_dense_envelope_split_rms_"
        "20260813T134012434338842Z" in runner
    )
    assert "dense prerequisite contract drifted" in runner
    assert "DB549 split-RMS decision evidence drifted" in runner
    assert "dense final-layout direct remote $remote_file hash drifted" in runner
    assert "validate_dense_final_layout_hlo" in runner
    assert 'expected = 24 if dense_final_layout_convolution else 0' in runner
    assert 'contract["optimized_exact_gate_down_bijection"]' in runner
    assert 'contract["optimized_exact_down_result_liveness"]' in runner
    assert 'contract["stablehlo_exact_arithmetic_contract"]' in runner
    assert 'exact_stable["exact_runtime_u8_bitcast_count"]' in runner
    assert "greenfield_dense_final_layout_convolution_prerequisite_db_run" in runner
    assert "548 if dense_final_layout_convolution else None" in runner
    assert "greenfield_dense_final_layout_convolution_decision_db_run" in runner
    assert "549 if dense_final_layout_convolution else None" in runner
    assert '"dense_final_layout_convolution_prerequisite"' in runner
    assert "_densefinalconv" in runner


def test_protected_runner_seals_archive_before_terminal_success() -> None:
    runner = PROTECTED_RUNNER.read_text()

    for token in (
        "remote_prefix_preflight.txt",
        "rollback_provisional_db",
        "provisional_db_run_id.txt",
        "google_crc32c",
        "remote object ledger checksum mismatch",
        "validate_exact_remote_object_set(root, prefix, listing)",
        "terminal_success_done=1",
        '"greenfield_run_tag": greenfield_run_tag',
    ):
        assert token in runner
    post_census = runner.index("strict_census post")
    db_mutation = runner.index("pv.start_run(")
    bulk_upload = runner.index(
        'gcloud storage rsync --recursive --checksums-only "$RUN_DIR"'
    )
    exact_object_gate = runner.index(
        "validate_exact_remote_object_set(root, prefix, listing)"
    )
    success_create = runner.index('(root / "SUCCESS").write_text')
    success_upload_marker = (
        'gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS"'
    )
    success_upload = runner.index(success_upload_marker)
    terminal = runner.index("terminal_success_done=1")
    assert (
        post_census
        < db_mutation
        < bulk_upload
        < exact_object_gate
        < success_create
        < success_upload
        < terminal
    )
    assert "gcloud storage cp" not in runner[
        success_upload + len(success_upload_marker) :
    ]


def _short_decoder_rollback_program() -> str:
    match = re.search(
        r"rollback_provisional_db\(\) \{.*?<<'PY'\n(?P<program>.*?)\nPY\n\}",
        PROTECTED_RUNNER.read_text(),
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("program")


@pytest.mark.parametrize("state", ("run", "item", "summary"))
def test_short_decoder_rollback_removes_each_committed_prefix(
    tmp_path: Path,
    state: str,
) -> None:
    provenance_path = REPO / "bench/provenance.py"
    specification = importlib.util.spec_from_file_location(
        f"short_decoder_rollback_provenance_{state}", provenance_path
    )
    assert specification is not None and specification.loader is not None
    provenance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(provenance)

    database = tmp_path / f"{state}.db"
    tag = f"rollback_{state}"
    pin = "a" * 40
    runtime_manifest = "b" * 64
    benchmark = "greenfield_78layer_8k_gate-d_pp8_strategynd_o"
    connection = provenance.connect(str(database))
    run_id = provenance.start_run(
        connection,
        model="zai-org/GLM-5.2-FP8:greenfield-78layer-8k-gate-d",
        revision=runtime_manifest,
        env={
            "GLM_ENGINE": "greenfield_pp8_decoder_gate-d",
            "greenfield_code_hash": pin,
            "greenfield_complete_token_path": True,
            "greenfield_run_tag": tag,
            "greenfield_short_context_dsa_oracle": True,
                "greenfield_short_context_oracle": True,
                "greenfield_strategy_nd_attention_projection": True,
                "greenfield_dense_final_layout_convolution": False,
                "runtime_manifest_sha256": runtime_manifest,
        },
        note=(
            "Protected real 78-layer 8K transformer-body compile/run with "
            "sealed test flags."
        ),
        harness_repo=str(REPO),
        fork_repo=None,
    )
    if state in {"item", "summary"}:
        provenance.record_item(
            connection,
            run_id,
            benchmark=benchmark,
            item_id="gate_d_exact_token_and_dsa",
            prompt="sealed",
            gold="exact",
            raw_output="{}",
            extracted="{}",
            correct=True,
            score=1.0,
        )
    if state == "summary":
        provenance.finalize(
            connection,
            run_id,
            benchmark=benchmark,
            metric="contract_valid",
            value=1.0,
            note="Protected test.",
        )
    connection.close()

    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(database),
            tag,
            pin,
            runtime_manifest,
            "8k",
            "1",
            "1",
                "1",
                "1",
                "0",
            ],
        input=_short_decoder_rollback_program(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ROLLED_BACK_PROVISIONAL_DB_RUN=" in result.stdout
    verify = sqlite3.connect(database)
    assert verify.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert verify.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    assert verify.execute("SELECT COUNT(*) FROM summary").fetchone()[0] == 0
    verify.close()


def test_short_decoder_rollback_refuses_nonidentical_run(tmp_path: Path) -> None:
    provenance_path = REPO / "bench/provenance.py"
    specification = importlib.util.spec_from_file_location(
        "short_decoder_rollback_provenance_refusal", provenance_path
    )
    assert specification is not None and specification.loader is not None
    provenance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(provenance)

    database = tmp_path / "refusal.db"
    tag = "rollback_refusal"
    runtime_manifest = "b" * 64
    connection = provenance.connect(str(database))
    provenance.start_run(
        connection,
        model="zai-org/GLM-5.2-FP8:greenfield-78layer-8k-gate-d",
        revision=runtime_manifest,
        env={
            "GLM_ENGINE": "greenfield_pp8_decoder_gate-d",
            "greenfield_code_hash": "c" * 40,
            "greenfield_complete_token_path": True,
            "greenfield_run_tag": tag,
            "greenfield_short_context_dsa_oracle": True,
            "greenfield_short_context_oracle": True,
            "greenfield_strategy_nd_attention_projection": True,
            "greenfield_dense_final_layout_convolution": False,
            "runtime_manifest_sha256": runtime_manifest,
        },
        note=(
            "Protected real 78-layer 8K transformer-body compile/run with "
            "sealed test flags."
        ),
        harness_repo=str(REPO),
        fork_repo=None,
    )
    connection.close()

    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(database),
            tag,
            "a" * 40,
            runtime_manifest,
            "8k",
            "1",
            "1",
            "1",
            "1",
            "0",
        ],
        input=_short_decoder_rollback_program(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing non-identical provisional DB rollback" in result.stderr
    verify = sqlite3.connect(database)
    assert verify.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    verify.close()


def test_layer0_attention_schedule_discriminator_is_sealed_and_default_off() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert "LAYER1_MAIN_ROPE_NORMALIZED_HIDDEN_SHA256" in compiler
    assert '"attention_schedule_control"' in compiler
    assert '"replicated_monolithic_attention"' in compiler
    assert '"discriminator_kind": layer0_discriminator_kind' in compiler
    assert "96fe8d9bf0e8fa43a3f2ab92735854b8" in compiler
    assert "7600e22f3682b8263a5a1771968331f0" in compiler
    assert "LAYER0_ATTENTION_VARIANTS:-0" in runner
    assert "layer-0 attention variants require the table-on" in runner
    assert "ATTENTION_SCHEDULE_PREREQUISITE_TAG" in runner
    assert "20260811T113139003786245Z" in runner
    assert "table-on production-boundary direct remote contract hash drifted" in runner
    assert "table-on production-boundary direct remote NPZ hash drifted" in runner
    assert "attention-schedule diagnostic contract is missing" in runner
    assert "attention-schedule diagnostic contract failed" in runner
    assert "ATTENTION_SCHEDULE_DIAGNOSTIC_CONTRACT_OK" in runner
    assert (
        "attention-schedule diagnostic contract passed; preserving "
        "intentional diagnostic exit" in runner
    )
    assert "_layer0_attention_schedule_variants" in runner


def test_layer0_attention_output_discriminator_is_sealed_and_default_off() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert (
        '"--observe-layer0-attention-output-association-variants"'
        in compiler
    )
    assert '"attention_output_association_control"' in compiler
    assert '"attention_output_dcp_then_model_sequential_bf16"' in compiler
    assert '"attention_output_model_then_dcp_pairwise_bf16"' in compiler
    assert "LAYER0_ATTENTION_OUTPUT_VARIANTS:-0" in runner
    assert (
        "--observe-layer0-attention-output-association-variants 1" in runner
    )
    assert "layer-0 attention-output variants require the table-on" in runner
    assert "ATTENTION_OUTPUT_PREREQUISITE_TAG" in runner
    assert "20260811T142025290162247Z" in runner
    assert "bacc8a785e3a9cfe26162b2452b8b47b" in runner
    assert "ad64fff27fced824d76d82815800f9f9" in runner
    assert "attention-output direct remote contract hash drifted" in runner
    assert "attention-output direct remote NPZ hash drifted" in runner
    assert (
        "attention-output association diagnostic contract is missing" in runner
    )
    assert (
        "attention-output association diagnostic contract failed" in runner
    )
    assert "ATTENTION_OUTPUT_ASSOCIATION_DIAGNOSTIC_CONTRACT_OK" in runner
    assert (
        "attention-output association diagnostic contract passed; preserving "
        "intentional diagnostic exit" in runner
    )
    assert "_layer0_attention_output_variants" in runner


def test_layer0_strategy_nd_row0_discriminator_is_sealed_and_default_off() -> None:
    compiler = (REPO / "scripts/greenfield/compile_short_decoder.py").read_text()
    runner = PROTECTED_RUNNER.read_text()

    assert '"--observe-layer0-strategy-nd-row0-association"' in compiler
    assert '"strategy_nd_row0_control"' in compiler
    assert '"strategy_nd_row0_both"' in compiler
    assert '"strategy_nd_row0_association"' in compiler
    assert '"--strategy-nd-canary-input-bits"' in compiler
    assert '"--strategy-nd-canary-output-bits"' in compiler
    assert "stablehlo_optimization_barrier_count" in compiler
    assert "strategy_nd_canary_contract" in compiler
    assert "LAYER0_STRATEGY_ND_ROW0:-0" in runner
    assert "--observe-layer0-strategy-nd-row0-association 1" in runner
    assert "layer-0 StrategyND row-zero discriminator requires" in runner
    assert "greenfield_collective_association_20260811T213152133863450Z" in runner
    assert "e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108" in runner
    assert "3ca82073f69fbe56526e1765594c7e8e9a738c73eb2a62b2df6d9a0c4d3136b7" in runner
    assert "DB533 direct remote analysis hash drifted" in runner
    assert "DB533 direct remote summary hash drifted" in runner
    assert "DB533 direct remote SUCCESS hash drifted" in runner
    assert "DB533 direct remote canary input hash drifted" in runner
    assert "DB533 direct remote canary output hash drifted" in runner
    assert "StrategyND row-zero diagnostic contract is missing" in runner
    assert "StrategyND row-zero diagnostic contract failed" in runner
    assert "STRATEGY_ND_ROW0_DIAGNOSTIC_CONTRACT_OK" in runner
    assert "_layer0_strategy_nd_row0" in runner


def test_strategy_nd_canary_hlo_requires_one_scoped_lp4_gather() -> None:
    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4))
        for stage in range(8)
    )
    replica_groups = (
        "{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},"
        "{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}"
    )
    gather = (
        "  ROOT %gather = bf16[4,8,1,6144] all-gather(%input), "
        "dimensions={0}, "
        f"replica_groups={replica_groups}, channel_id=1, "
        "use_global_device_ids=true, metadata={op_name=\"jit(canary)/"
        "greenfield_strategy_nd_row0_canary/"
        "greenfield_strategy_nd_row0_association/"
        "greenfield_strategy_nd_row0_association_gather/all_gather\"}"
    )
    for operand_shape in ("bf16[8,1,6144]", "bf16[1,8,1,6144]"):
        hlo = (
            "HloModule canary, replica_count=1, num_partitions=32\n\n"
            f"ENTRY main (input: {operand_shape}) -> bf16[4,8,1,6144] {{\n"
            f"  %input = {operand_shape} parameter(0)\n"
            f"{gather}\n"
            "}\n"
        )
        passed = _validate_strategy_nd_canary_hlo(hlo, groups=groups)
        assert passed["passed"], passed
        assert passed["collective_count"] == 1
        assert passed["scoped_shaped_gather_count"] == 1

    extra = hlo.replace(
        gather,
        gather.replace("ROOT %gather", "%gather")
        + "\n"
        + gather.replace("%gather", "%extra").replace("channel_id=1", "channel_id=2"),
    )
    rejected = _validate_strategy_nd_canary_hlo(extra, groups=groups)
    assert not rejected["passed"]
    assert rejected["collective_count"] == 2


@pytest.mark.parametrize(
    ("capacity", "label"),
    ((2048, "2k"), (8192, "8k")),
)
def test_protected_short_context_label(capacity: int, label: str) -> None:
    assert _protected_short_context_label(capacity) == label


def test_protected_short_context_label_rejects_unsealed_capacity() -> None:
    with pytest.raises(ValueError, match="must be 2048 or 8192"):
        _protected_short_context_label(4096)


def test_protected_8k_runner_pins_paired_oracles() -> None:
    source = PROTECTED_RUNNER.read_text()
    wrapper = PROTECTED_8K_RUNNER.read_text()

    assert "GLM_GREENFIELD_SHORT_DECODER_PROFILE:-2k" in source
    assert "PROMPT_TOKEN_COUNT=8155" in source
    assert "CONTEXT_CAPACITY=8192" in source
    assert (
        "SHORT_CONTEXT_ORACLE_MANIFEST_SHA="
        "e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2"
        in source
    )
    assert (
        "SHORT_CONTEXT_DSA_ORACLE_MANIFEST_SHA="
        "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
        in source
    )
    assert "export GLM_GREENFIELD_SHORT_DECODER_PROFILE=8k" in wrapper


class _Jax:
    def __init__(self, materialized: np.ndarray) -> None:
        self.materialized = materialized
        self.device_get_calls = 0

    def device_get(self, value: object) -> np.ndarray:
        self.device_get_calls += 1
        return self.materialized


class _Multihost:
    def __init__(self, gathered: np.ndarray) -> None:
        self.gathered = gathered
        self.process_allgather_calls = 0
        self.tiled_values: list[bool] = []

    def process_allgather(self, value: object, *, tiled: bool) -> np.ndarray:
        self.process_allgather_calls += 1
        self.tiled_values.append(tiled)
        return self.gathered


def test_materialize_global_array_uses_device_get_when_fully_addressable() -> None:
    expected = np.arange(12, dtype=np.int32).reshape(4, 3)
    value = SimpleNamespace(is_fully_addressable=True, shape=expected.shape)
    jax = _Jax(expected)
    multihost = _Multihost(np.empty((0,), dtype=np.int32))

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 1
    assert multihost.process_allgather_calls == 0
    assert multihost.tiled_values == []


def test_materialize_global_array_gathers_non_addressable_global_array() -> None:
    expected = np.arange(24, dtype=np.int32).reshape(8, 3)
    value = SimpleNamespace(is_fully_addressable=False, shape=expected.shape)
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(expected)

    actual = _materialize_global_array(jax, multihost, value)

    np.testing.assert_array_equal(actual, expected)
    assert jax.device_get_calls == 0
    assert multihost.process_allgather_calls == 1
    assert multihost.tiled_values == [True]


def test_materialize_global_array_fails_closed_on_gather_shape_drift() -> None:
    value = SimpleNamespace(is_fully_addressable=False, shape=(8, 3))
    jax = _Jax(np.empty((0,), dtype=np.int32))
    multihost = _Multihost(np.empty((1, 8, 3), dtype=np.int32))

    with pytest.raises(RuntimeError, match="global array gather changed shape"):
        _materialize_global_array(jax, multihost, value)


def _layer_residual_observation_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[tuple[int, ...], ...],
    SimpleNamespace,
]:
    groups = ((0, 1), (2, 3))
    schedule = SimpleNamespace(
        stages=(
            SimpleNamespace(
                assignment=SimpleNamespace(stage_id=0),
                layers=(
                    SimpleNamespace(layer_id=0),
                    SimpleNamespace(layer_id=1),
                ),
            ),
            SimpleNamespace(
                assignment=SimpleNamespace(stage_id=1),
                layers=(
                    SimpleNamespace(layer_id=2),
                    SimpleNamespace(layer_id=3),
                ),
            ),
        )
    )
    canonical = np.asarray(
        np.arange(1, 16, dtype=np.float32).reshape(5, 3),
        dtype=bfloat16,
    )
    observation = np.zeros((4, 5, 3), dtype=bfloat16)
    writers = ((0,), (0,), (0, 1), (1,), (1,))
    for boundary, writer_stages in enumerate(writers):
        for stage_id in writer_stages:
            observation[list(groups[stage_id]), boundary] = canonical[boundary]
    return observation, canonical, groups, schedule


def test_layer_residual_observer_canonicalizes_exact_stage_writers() -> None:
    observation, expected, groups, schedule = (
        _layer_residual_observation_fixture()
    )

    canonical, contract = _canonicalize_layer_residual_observation(
        observation,
        groups=groups,
        schedule=schedule,
        hidden_size=3,
    )

    np.testing.assert_array_equal(canonical, expected)
    assert contract["passed"]
    assert contract["boundary_count"] == 5
    assert contract["dtype"] == "bfloat16"
    assert contract["storage_byte_order"] == "little"
    assert contract["storage_dtype"] == "<u2"
    assert contract["storage_field"] == "residual_bfloat16_bits"
    assert contract["lane_mismatch_boundaries"] == []
    assert contract["writer_mismatch_boundaries"] == []
    assert contract["nonwriter_nonzero_boundaries"] == []
    assert [record["writer_stages"] for record in contract["boundary_records"]] == [
        [0],
        [0],
        [0, 1],
        [1],
        [1],
    ]


def test_layer_residual_bfloat16_storage_is_portable_uint16_bits() -> None:
    values = np.asarray([[0.0, 1.0, -2.5]], dtype=bfloat16)

    bits = _encode_bfloat16_bits(values)

    assert bits.dtype.str == "<u2"
    assert bits.tolist() == [[0x0000, 0x3F80, 0xC020]]
    assert bits.tobytes() == b"\x00\x00\x80\x3f\x20\xc0"
    with pytest.raises(ValueError, match="requires bfloat16"):
        _encode_bfloat16_bits(values.astype(np.float32))


@pytest.mark.parametrize(
    ("mutation", "contract_key", "expected_boundary"),
    (
        ("lane", "lane_mismatch_boundaries", 1),
        ("writer", "writer_mismatch_boundaries", 2),
        ("nonwriter", "nonwriter_nonzero_boundaries", 1),
    ),
)
def test_layer_residual_observer_fails_closed_on_replication_drift(
    mutation: str,
    contract_key: str,
    expected_boundary: int,
) -> None:
    observation, _, groups, schedule = _layer_residual_observation_fixture()
    if mutation == "lane":
        observation[1, 1, 0] += bfloat16(1)
    elif mutation == "writer":
        observation[list(groups[1]), 2, 0] += bfloat16(1)
    else:
        observation[2, 1, 0] = bfloat16(1)

    _, contract = _canonicalize_layer_residual_observation(
        observation,
        groups=groups,
        schedule=schedule,
        hidden_size=3,
    )

    assert not contract["passed"]
    assert contract[contract_key] == [expected_boundary]


def test_layer_residual_observer_rejects_shape_and_dtype_drift() -> None:
    observation, _, groups, schedule = _layer_residual_observation_fixture()
    with pytest.raises(RuntimeError, match="tensor contract drifted"):
        _canonicalize_layer_residual_observation(
            observation[:, :-1],
            groups=groups,
            schedule=schedule,
            hidden_size=3,
        )
    with pytest.raises(RuntimeError, match="tensor contract drifted"):
        _canonicalize_layer_residual_observation(
            observation.astype(np.float32),
            groups=groups,
            schedule=schedule,
            hidden_size=3,
        )


def _dsa_internal_observation_fixture() -> tuple[
    SimpleNamespace,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    groups = ((0, 1), (2, 3))
    producers = ((0, 1), (6,))
    normalized_hidden = np.zeros((4, 2, 3), dtype=bfloat16)
    q_a_state = np.zeros((4, 2, 2), dtype=bfloat16)
    query = np.zeros((4, 2, 2, 2), dtype=np.float32)
    head_weights = np.zeros((4, 2, 2), dtype=np.float32)
    current_key = np.zeros((4, 2, 2), dtype=np.float32)
    producer_layer_ids = np.full((4, 2), -1, dtype=np.int32)
    fields = (
        normalized_hidden,
        q_a_state,
        query,
        head_weights,
        current_key,
    )
    for stage, (group, stage_producers) in enumerate(
        zip(groups, producers, strict=True)
    ):
        for slot, producer in enumerate(stage_producers):
            producer_layer_ids[list(group), slot] = producer
            for field_index, value in enumerate(fields, start=1):
                value[list(group), slot] = field_index * 10 + producer
    return (
        SimpleNamespace(
            normalized_hidden=normalized_hidden,
            q_a_state=q_a_state,
            query=query,
            head_weights=head_weights,
            current_key=current_key,
            producer_layer_ids=producer_layer_ids,
        ),
        groups,
        producers,
    )


def test_dsa_internal_observer_canonicalizes_stage_slots() -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()

    canonical, contract = _canonicalize_dsa_internal_observation(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        hidden_size=3,
        q_lora_rank=2,
        num_heads=2,
        head_dim=2,
    )

    assert contract["passed"]
    assert contract["event_count"] == 3
    assert contract["producer_layer_ids"] == [0, 1, 6]
    assert contract["lane_mismatches"] == []
    assert contract["padded_slot_mismatches"] == []
    np.testing.assert_array_equal(
        canonical["producer_layer_ids"], np.asarray([0, 1, 6], np.int32)
    )
    assert canonical["normalized_hidden"].shape == (3, 3)
    assert canonical["normalized_hidden"].dtype.name == "bfloat16"
    assert canonical["q_a_state"].shape == (3, 2)
    assert canonical["query"].shape == (3, 2, 2)
    assert canonical["head_weights"].shape == (3, 2)
    assert canonical["current_key"].shape == (3, 2)


@pytest.mark.parametrize("mutation", ("lane", "padded"))
def test_dsa_internal_observer_fails_closed_on_slot_drift(
    mutation: str,
) -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()
    if mutation == "lane":
        observation.query[1, 0, 0, 0] += 1.0
    else:
        observation.current_key[2, 1, 0] = 1.0

    _, contract = _canonicalize_dsa_internal_observation(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        hidden_size=3,
        q_lora_rank=2,
        num_heads=2,
        head_dim=2,
    )

    assert not contract["passed"]
    assert contract[
        "lane_mismatches" if mutation == "lane" else "padded_slot_mismatches"
    ]


def test_dsa_internal_observer_rejects_shape_and_dtype_drift() -> None:
    observation, groups, producers = _dsa_internal_observation_fixture()
    observation.normalized_hidden = observation.normalized_hidden.astype(
        np.float32
    )
    with pytest.raises(RuntimeError, match="normalized_hidden contract drifted"):
        _canonicalize_dsa_internal_observation(
            observation,
            groups=groups,
            stage_producer_layer_ids=producers,
            hidden_size=3,
            q_lora_rank=2,
            num_heads=2,
            head_dim=2,
        )


def test_completed_step_selected_state_uses_next_position_as_exclusive_bound() -> None:
    metadata = np.asarray(
        [
            [2, 0, 1, -1, 3, 99],
            [1, 2, 0, -1, 3, 99],
        ],
        dtype=np.int32,
    )
    record = _validate_completed_step_selected_states(
        metadata,
        selected_width=4,
        count_index=4,
        next_position=3,
        next_context_length=4,
    )
    assert record == {
        "expected_valid_count": 3,
        "next_context_length": 4,
        "next_position": 3,
        "position_context_aligned": True,
        "rows_valid": [True, True],
    }

    future_position = metadata.copy()
    future_position[0, 0] = 3
    assert not all(
        _validate_completed_step_selected_states(
            future_position,
            selected_width=4,
            count_index=4,
            next_position=3,
            next_context_length=4,
        )["rows_valid"]
    )
    assert not all(
        _validate_completed_step_selected_states(
            metadata,
            selected_width=4,
            count_index=4,
            next_position=3,
            next_context_length=3,
        )["rows_valid"]
    )


def test_raw_token_sequence_contract_is_exact_and_reports_first_drift() -> None:
    expected = np.asarray([4, 8, 15, 16, 23, 42], dtype=np.int32)
    exact = _raw_token_sequence_contract([4, 8, 15], expected)
    assert exact == {
        "compared_token_count": 3,
        "exact_prefix_match": True,
        "expected_token_ids": [4, 8, 15],
        "first_mismatch_index": None,
        "observed_token_ids": [4, 8, 15],
        "oracle_token_count": 6,
    }
    drifted = _raw_token_sequence_contract([4, 7, 15], expected)
    assert not drifted["exact_prefix_match"]
    assert drifted["first_mismatch_index"] == 1
    with pytest.raises(ValueError, match="exceeds"):
        _raw_token_sequence_contract([], expected)
    with pytest.raises(ValueError, match="exceeds"):
        _raw_token_sequence_contract(expected.tolist() + [99], expected)


def test_load_short_context_dsa_oracle_requires_exact_manifest_pin(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from safetensors.numpy import save_file

    oracle_dir = Path(str(tmp_path))
    tensors = {
        "decode_positions": np.asarray([4], np.int32),
        "producer_layer_ids": np.asarray([0, 2], np.int32),
        "selected_positions": np.asarray(
            [[[0, 1, 2, 3], [3, 2, 1, 0]]], np.int32
        ),
        "selected_scores": np.asarray(
            [[[4, 3, 2, 1], [4, 3, 2, 1]]], np.float32
        ),
        "valid_counts": np.asarray([[4, 4]], np.int32),
    }
    save_file(tensors, oracle_dir / "dsa_events.safetensors")
    manifest = {
        "files": {"tensors": {"filename": "dsa_events.safetensors"}},
        "manifest_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        compile_module,
        "inspect_short_context_dsa_oracle",
        lambda _: manifest,
    )

    loaded_manifest, loaded_tensors = _load_short_context_dsa_oracle(
        oracle_dir,
        expected_manifest_sha256="a" * 64,
    )
    assert loaded_manifest == manifest
    for name, expected in tensors.items():
        np.testing.assert_array_equal(loaded_tensors[name], expected)
    with pytest.raises(RuntimeError, match="protected pin"):
        _load_short_context_dsa_oracle(
            oracle_dir,
            expected_manifest_sha256="b" * 64,
        )


def _dsa_observation_fixture() -> tuple[
    np.ndarray,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    groups = ((0, 1), (2, 3))
    producers = ((0, 2), (6,))
    expected_positions = np.asarray(
        [[0, 1, 2, -1], [2, 1, 0, -1], [1, 0, 2, -1]],
        np.int32,
    )
    expected_counts = np.asarray([3, 3, 3], np.int32)
    expected_scores = np.asarray(
        [[3, 2, 1, -np.inf], [3, 2, 1, -np.inf], [3, 2, 1, -np.inf]],
        np.float32,
    )
    expected_producers = np.asarray([0, 2, 6], np.int32)
    observation = np.full((4, 2, 10), -1, np.int32)
    event = 0
    for stage, group in enumerate(groups):
        for slot, producer in enumerate(producers[stage]):
            row = np.concatenate(
                (
                    expected_positions[event],
                    expected_scores[event].view(np.int32),
                    np.asarray([expected_counts[event], producer], np.int32),
                )
            )
            observation[list(group), slot] = row
            event += 1
    return (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    )


def test_dsa_observer_reconstructs_stage_slots_and_lane_replication() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    record = _validate_dsa_observation_step(
        observation,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["event_count"] == 3
    assert record["lane_mismatch_stages"] == []
    assert record["padded_slot_mismatches"] == []
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert record["legacy_total_order_match"]
    assert record["legacy_score_bounded_comparison"]["passed"]


def test_dsa_observer_allows_cross_backend_nontie_rank_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    bounded_expected_scores = expected_scores.copy()
    bounded_expected_scores[0, 1] = np.float32(2.99)
    # The executing program assigns a different descending score order to the
    # same exact set. Its scores remain canonical and inside the established
    # cross-program bound, so the raw order difference stays diagnostic.
    drifted[0:2, 0, 0:3] = np.asarray([1, 0, 2], np.int32)
    drifted_scores = np.asarray([3.01, 3.0, 1.0], np.float32).view(np.int32)
    drifted[0:2, 0, 4:7] = drifted_scores
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=bounded_expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert not record["legacy_total_order_match"]
    assert record["legacy_order_mismatch_count"] == 2
    assert record["legacy_score_bounded_comparison"]["passed"]


def test_dsa_observer_records_cross_backend_score_bound_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    drifted[0:2, 0, 4] = np.asarray([3.2], np.float32).view(np.int32)[0]
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert record["passed"]
    assert record["exact_selected_set_and_tail"]
    assert record["actual_device_score_order_and_ties"]
    assert not record["legacy_score_bounded_comparison"]["passed"]
    assert record["legacy_score_bounded_comparison"]["error"]["max_abs"] > 0.19


def test_dsa_observer_refuses_selected_set_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    drifted[0:2, 0, 2] = 3
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert not record["passed"]
    assert not record["exact_selected_set_and_tail"]
    assert record["selected_set_mismatches"][0]["expected_only_first"] == [2]
    assert record["selected_set_mismatches"][0]["observed_only_first"] == [3]


def test_dsa_observer_refuses_lane_order_producer_and_padding_drift() -> None:
    (
        observation,
        groups,
        producers,
        expected_positions,
        expected_scores,
        expected_counts,
        expected_producers,
    ) = _dsa_observation_fixture()
    drifted = observation.copy()
    # Same selected set, but equal executing scores are in wrong position order.
    drifted[0:2, 0, 0:2] = drifted[0:2, 0, 1::-1]
    tied = np.asarray([3.0, 3.0], np.float32).view(np.int32)
    drifted[0:2, 0, 4:6] = tied
    drifted[0:2, 0, 7] = np.asarray([0.0], np.float32).view(np.int32)[0]
    # One replicated lane disagrees, one producer drifts, and dead padding is live.
    drifted[1, 1, 0] = 99
    drifted[2:4, 0, 9] = 7
    drifted[2:4, 1, 0] = 0
    record = _validate_dsa_observation_step(
        drifted,
        groups=groups,
        stage_producer_layer_ids=producers,
        selected_width=4,
        expected_positions=expected_positions,
        expected_scores=expected_scores,
        expected_valid_counts=expected_counts,
        expected_producer_layer_ids=expected_producers,
        decode_position=3,
    )
    assert not record["passed"]
    assert record["legacy_order_mismatch_count"] == 2
    assert record["first_legacy_order_mismatch"]["selected_offset"] == 0
    assert not record["actual_device_score_order_and_ties"]
    assert record["score_contract_mismatches"][0][
        "first_noncanonical_offset"
    ] == 0
    assert record["tail_mismatches"] == [
        {
            "event_index": 0,
            "position_tail_mismatch_count": 0,
            "producer_layer_id": 0,
            "score_tail_mismatch_count": 1,
        }
    ]
    assert record["lane_mismatch_stages"] == [0]
    assert record["producer_mismatches"] == [
        {"event_index": 2, "expected": 6, "observed": 7}
    ]
    assert record["padded_slot_mismatches"] == [{"slot": 1, "stage": 1}]


def _token_observation_fixture() -> tuple[
    np.ndarray, tuple[tuple[int, ...], ...]
]:
    groups = ((0, 1), (2, 3))
    candidate_ids = np.asarray([3, 1, 4, 2], np.int32)
    candidate_scores = np.asarray([10.0, 9.0, 9.0, 8.0], np.float32)
    row = np.concatenate((candidate_ids, candidate_scores.view(np.int32)))
    observation = np.full((4, 8), -1, np.int32)
    observation[list(groups[-1])] = row
    return observation, groups


def test_token_observer_validates_rank_margin_ties_and_inactive_lanes() -> None:
    observation, groups = _token_observation_fixture()
    record = _validate_token_observation_step(
        observation,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=3,
        vocab_size=8,
    )
    assert record["passed"]
    assert record["candidate_ids"] == [3, 1, 4, 2]
    assert record["expected_offset"] == 2
    assert record["expected_rank"] == 3
    assert record["expected_score"] == 9.0
    assert record["top1_top2_margin"] == 1.0
    assert record["top1_expected_margin"] == 1.0
    assert record["lane_replication"]
    assert record["inactive_rows_are_sentinel"]
    assert record["order_and_ties_valid"]
    assert record["winner_matches_output"]


def test_token_observer_refuses_noncanonical_or_corrupt_candidates() -> None:
    observation, groups = _token_observation_fixture()

    wrong_tie_order = observation.copy()
    wrong_tie_order[2:4, [1, 2]] = wrong_tie_order[2:4, [2, 1]]
    record = _validate_token_observation_step(
        wrong_tie_order,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=3,
        vocab_size=8,
    )
    assert not record["passed"]
    assert not record["order_and_ties_valid"]

    corrupt = observation.copy()
    corrupt[0, 0] = 0
    corrupt[3, 0] = 7
    record = _validate_token_observation_step(
        corrupt,
        groups=groups,
        candidate_width=4,
        expected_token_id=4,
        observed_token_id=1,
        vocab_size=8,
    )
    assert not record["passed"]
    assert not record["inactive_rows_are_sentinel"]
    assert not record["lane_replication"]
    assert not record["winner_matches_output"]


def test_observer_hlo_isolation_requires_no_alias_callback_or_collective_drift(
) -> None:
    contract = {
        "all_reduce_arity_counts": {"1": 3},
        "all_reduce_component_count": 3,
        "all_reduce_result_shape_counts": {
            "bf16[4]": 1,
            "bf16[8]": 1,
            "s32[4]": 1,
        },
        "collective_count": 5,
        "collective_counts": {
            "all-gather": 1,
            "all-reduce": 3,
            "collective-permute": 1,
        },
        "complete_token_collective_contract": {
            "score_exchange": [{"result_shapes": ["bf16[4]"]}],
            "token_id_exchange": [{"result_shapes": ["s32[4]"]}],
        },
        "passed": True,
    }
    hlo = "HloModule observer, is_scheduled=true\nENTRY main {}\n"
    exact = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert exact["passed"]
    assert exact["donate_argnums"] == []
    assert exact["non_token_result_shapes_match"]

    widened = {
        **contract,
        "all_reduce_result_shape_counts": {
            "bf16[8]": 1,
            "bf16[64]": 1,
            "s32[64]": 1,
        },
        "complete_token_collective_contract": {
            "score_exchange": [{"result_shapes": ["bf16[64]"]}],
            "token_id_exchange": [{"result_shapes": ["s32[64]"]}],
        },
    }
    wider_token_exchange = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=widened,
    )
    assert wider_token_exchange["passed"]
    assert wider_token_exchange["token_exchange_shape_difference_allowed"]

    widened_non_token_drift = {
        **widened,
        "all_reduce_result_shape_counts": {
            "bf16[16]": 1,
            "bf16[64]": 1,
            "s32[64]": 1,
        },
    }
    non_token_drift = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=widened_non_token_drift,
    )
    assert not non_token_drift["passed"]
    assert not non_token_drift["non_token_result_shapes_match"]

    aliased = _observer_hlo_isolation_contract(
        "HloModule observer, input_output_alias={ {0}: (1, {}, may-alias) }\n",
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert not aliased["passed"]
    callback = _observer_hlo_isolation_contract(
        hlo + "outside_compilation\n",
        production_contract=contract,
        observer_contract=dict(contract),
    )
    assert not callback["passed"]
    drifted = dict(contract)
    drifted["collective_count"] = 4
    collective = _observer_hlo_isolation_contract(
        hlo,
        production_contract=contract,
        observer_contract=drifted,
    )
    assert not collective["passed"]
