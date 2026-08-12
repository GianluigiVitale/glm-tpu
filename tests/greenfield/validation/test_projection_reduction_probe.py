from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/greenfield/probe_layer0_projection_reduction.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_projection_reduction_probe.sh"
SPEC = importlib.util.spec_from_file_location("projection_reduction_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _custom_call(
    name: str,
    index: int,
    result: str,
    operands: tuple[str, ...],
) -> str:
    return (
        f"%{name}.{index} = {result} custom-call({', '.join(operands)}), "
        'custom_call_target="tpu_custom_call", '
        f'metadata={{op_name="jit(probe)/{name}/pallas_call"}}'
    )


def _synthetic_hlo(arm: object) -> str:
    attention_kernel = (
        "greenfield_fp8_block_matmul_m8_k512_n6144"
        if arm.attention_strategy_nd
        else "greenfield_fp8_block_matmul_m8_k4096_n6144"
    )
    dense_kernel = (
        "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
        if arm.dense_strategy_nd
        else "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
    )
    attention_scope = (
        "greenfield_strategy_nd_row0_attention_output"
        if arm.attention_strategy_nd
        else "greenfield_local_attention_output_reduction"
    )
    dense_scope = (
        "greenfield_strategy_nd_row0_dense_down"
        if arm.dense_strategy_nd
        else "greenfield_local_dense_down_reduction"
    )
    lines = [
        "HloModule probe, num_partitions=4",
        "%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {",
        "  %lhs = bf16[] parameter(0)",
        "  %rhs = bf16[] parameter(1)",
        "  ROOT %sum = bf16[] add(%lhs, %rhs)",
        "}",
        "ENTRY main {",
        "  %latent = bf16[1,16,512] parameter(0)",
        "  %kv_bits = u8[7168,512] parameter(1)",
        "  %kv_scale = f32[56,4] parameter(2)",
        "  " + _custom_call(
            "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
            0,
            "bf16[1,16,256]",
            ("%latent", "%kv_bits", "%kv_scale"),
        ),
        "  %value_flat = bf16[1,4096] reshape("
        "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0)",
    ]
    parameter_index = 3
    if arm.attention_strategy_nd:
        expanded = []
        for index in range(8):
            lines.extend(
                [
                    f"  %attention_input.{index} = bf16[1,512] "
                    f"slice(%value_flat), slice={{[0:1], [{index * 512}:{(index + 1) * 512}]}}",
                    f"  %attention_bits.{index} = u8[6144,512] "
                    f"parameter({parameter_index})",
                    f"  %attention_scale.{index} = f32[48,4] "
                    f"parameter({parameter_index + 1})",
                    "  " + _custom_call(
                        attention_kernel,
                        index + 1,
                        "bf16[1,6144]",
                        (
                            f"%attention_input.{index}",
                            f"%attention_bits.{index}",
                            f"%attention_scale.{index}",
                        ),
                    ),
                    f"  %attention_expanded.{index} = bf16[1,1,6144] "
                    f"reshape(%{attention_kernel}.{index + 1})",
                ]
            )
            expanded.append(f"%attention_expanded.{index}")
            parameter_index += 2
        lines.extend(
            [
                "  %attention_stack = bf16[8,1,6144] concatenate("
                + ", ".join(expanded)
                + "), dimensions={0}",
                "  %attention_collective = bf16[4,8,1,6144] "
                "all-gather(%attention_stack), channel_id=1, "
                "replica_groups={{0,1,2,3}}, dimensions={0}, "
                "use_global_device_ids=true, "
                f'metadata={{op_name="jit(probe)/{attention_scope}/all-gather"}}',
                "  %attention_update = bf16[1,6144] "
                "slice(%attention_collective), slice={[0:1], [0:1], [0:1], [0:6144]}",
            ]
        )
    else:
        lines.extend(
            [
                f"  %attention_bits = u8[6144,4096] parameter({parameter_index})",
                f"  %attention_scale = f32[48,32] parameter({parameter_index + 1})",
                "  " + _custom_call(
                    attention_kernel,
                    1,
                    "bf16[1,6144]",
                    ("%value_flat", "%attention_bits", "%attention_scale"),
                ),
                "  %attention_collective = bf16[1,6144] "
                f"all-reduce(%{attention_kernel}.1), channel_id=1, "
                "replica_groups={{0,1,2,3}}, use_global_device_ids=true, "
                f'to_apply=%add, metadata={{op_name="jit(probe)/{attention_scope}/all-reduce"}}',
                "  %attention_update = bf16[1,6144] copy(%attention_collective)",
            ]
        )
        parameter_index += 2
    if arm.dense_strategy_nd:
        expanded = []
        for index in range(8):
            operands = ["%attention_update"]
            for label, shape in (
                ("gate_bits", "u8[384,6144]"),
                ("gate_scale", "f32[3,48]"),
                ("up_bits", "u8[384,6144]"),
                ("up_scale", "f32[3,48]"),
                ("down_bits", "u8[6144,384]"),
                ("down_scale", "f32[48,3]"),
            ):
                name = f"%dense_{label}.{index}"
                lines.append(
                    f"  {name} = {shape} parameter({parameter_index})"
                )
                operands.append(name)
                parameter_index += 1
            lines.extend(
                [
                    "  " + _custom_call(
                        dense_kernel,
                        index + 20,
                        "bf16[1,6144]",
                        tuple(operands),
                    ),
                    f"  %dense_expanded.{index} = bf16[1,1,6144] "
                    f"reshape(%{dense_kernel}.{index + 20})",
                ]
            )
            expanded.append(f"%dense_expanded.{index}")
        lines.extend(
            [
                "  %dense_stack = bf16[8,1,6144] concatenate("
                + ", ".join(expanded)
                + "), dimensions={0}",
                "  %dense_collective = bf16[4,8,1,6144] "
                "all-gather(%dense_stack), channel_id=2, "
                "replica_groups={{0,1,2,3}}, dimensions={0}, "
                "use_global_device_ids=true, "
                f'metadata={{op_name="jit(probe)/{dense_scope}/all-gather"}}',
                "  %dense_update = bf16[1,6144] "
                "slice(%dense_collective), slice={[0:1], [0:1], [0:1], [0:6144]}",
            ]
        )
    else:
        operands = ["%attention_update"]
        for label, shape in (
            ("gate_bits", "u8[3072,6144]"),
            ("gate_scale", "f32[24,48]"),
            ("up_bits", "u8[3072,6144]"),
            ("up_scale", "f32[24,48]"),
            ("down_bits", "u8[6144,3072]"),
            ("down_scale", "f32[48,24]"),
        ):
            name = f"%dense_{label}"
            lines.append(f"  {name} = {shape} parameter({parameter_index})")
            operands.append(name)
            parameter_index += 1
        lines.extend(
            [
                "  " + _custom_call(
                    dense_kernel,
                    20,
                    "bf16[1,6144]",
                    tuple(operands),
                ),
                "  %dense_collective = bf16[1,6144] "
                f"all-reduce(%{dense_kernel}.20), channel_id=2, "
                "replica_groups={{0,1,2,3}}, use_global_device_ids=true, "
                f'to_apply=%add, metadata={{op_name="jit(probe)/{dense_scope}/all-reduce"}}',
                "  %dense_update = bf16[1,6144] copy(%dense_collective)",
            ]
        )
    lines.extend(
        [
            "  %normalized = bf16[1,6144] copy(%attention_update)",
            "  %layer1 = bf16[1,6144] copy(%dense_update)",
            "  ROOT %root = (bf16[1,16,256], bf16[1,6144], "
            "bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple("
            "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0, "
            "%attention_update, %normalized, %dense_update, %layer1)",
            "}",
        ]
    )
    return "\n".join(lines)


@pytest.mark.parametrize("arm", MODULE._ARMS, ids=lambda arm: arm.name)
def test_projection_reduction_hlo_contract_pins_each_arm(arm: object) -> None:
    record = MODULE._validate_hlo(_synthetic_hlo(arm), arm)
    assert record["passed"], record["violations"]
    assert record["collective_count"] == 2
    assert record["pallas_custom_call_count"] == (
        1
        + (8 if arm.attention_strategy_nd else 1)
        + (8 if arm.dense_strategy_nd else 1)
    )


def test_projection_reduction_hlo_contract_fails_closed() -> None:
    arm = MODULE._ARMS[-1]
    hlo = _synthetic_hlo(arm)
    kernel = "greenfield_fp8_block_matmul_m8_k512_n6144"

    suffixed = hlo.replace(kernel, f"{kernel}_suffix", 2)
    assert not MODULE._validate_hlo(suffixed, arm)["passed"]

    escaped = hlo.replace("{{0,1,2,3}}", "{{0,1,2,3,4,5,6,7}}", 1)
    assert not MODULE._validate_hlo(escaped, arm)["passed"]

    wrong_scope = hlo.replace(
        "greenfield_strategy_nd_row0_attention_output",
        "unscoped_attention_output",
        1,
    )
    assert not MODULE._validate_hlo(wrong_scope, arm)["passed"]

    async_hlo = hlo.replace("all-gather(", "all-gather-start(", 1)
    async_record = MODULE._validate_hlo(async_hlo, arm)
    assert not async_record["passed"]
    assert async_record["async_collectives"]

    callback = hlo.replace("ENTRY main {", "ENTRY main {\n%cb = bf16[1] outfeed()")
    assert not MODULE._validate_hlo(callback, arm)["passed"]

    wrong_cardinality = hlo.replace("num_partitions=4", "num_partitions=8", 1)
    assert not MODULE._validate_hlo(wrong_cardinality, arm)["passed"]

    wrong_collective_shape = hlo.replace(
        "%attention_collective = bf16[4,8,1,6144]",
        "%attention_collective = bf16[99]",
        1,
    )
    assert not MODULE._validate_hlo(wrong_collective_shape, arm)["passed"]

    attention_kernel = "greenfield_fp8_block_matmul_m8_k512_n6144"
    wrong_pallas_shape = hlo.replace(
        f"%{attention_kernel}.1 = bf16[1,6144]",
        f"%{attention_kernel}.1 = bf16[3]",
        1,
    )
    assert not MODULE._validate_hlo(wrong_pallas_shape, arm)["passed"]

    disconnected_collective = hlo.replace(
        "all-gather(%attention_stack)",
        "all-gather(%undefined_attention_stack)",
        1,
    )
    disconnected_record = MODULE._validate_hlo(disconnected_collective, arm)
    assert not disconnected_record["passed"]
    assert any(
        "lineage" in violation or "undefined" in violation
        for violation in disconnected_record["violations"]
    )

    rogue_input = hlo.replace(
        "  %value_flat = bf16[1,4096] reshape("
        "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0)",
        "  %value_flat = bf16[1,4096] reshape("
        "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0)\n"
        "  %rogue_attention_input = bf16[1,512] parameter(999)",
        1,
    ).replace(
        "custom-call(%attention_input.0,",
        "custom-call(%rogue_attention_input,",
        1,
    )
    rogue_record = MODULE._validate_hlo(rogue_input, arm)
    assert not rogue_record["passed"]
    assert any(
        "W_UV does not exclusively feed attention projection" in violation
        for violation in rogue_record["violations"]
    )

    tuple_decoy = hlo.replace(
        "  %value_flat = bf16[1,4096] reshape("
        "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0)",
        "  %value_flat = bf16[1,4096] reshape("
        "%greenfield_fp8_structured_kv_b_value_h16_l512_v256.0)\n"
        "  %rogue_value_flat = bf16[1,4096] parameter(998)\n"
        "  %value_pair = (bf16[1,4096], bf16[1,4096]) tuple("
        "%value_flat, %rogue_value_flat)\n"
        "  %decoy_value_flat = bf16[1,4096] get-tuple-element("
        "%value_pair), index=1",
        1,
    ).replace(
        "%attention_input.0 = bf16[1,512] slice(%value_flat)",
        "%attention_input.0 = bf16[1,512] slice(%decoy_value_flat)",
        1,
    )
    tuple_decoy_record = MODULE._validate_hlo(tuple_decoy, arm)
    assert not tuple_decoy_record["passed"]
    assert any(
        "W_UV does not exclusively feed attention projection" in violation
        or "forbidden lineage opcode" in violation
        for violation in tuple_decoy_record["violations"]
    )

    fusion_decoy = hlo.replace(
        "ENTRY main {",
        "%select_rogue (live: bf16[1,6144], rogue: bf16[1,6144]) "
        "-> bf16[1,6144] {\n"
        "  %live = bf16[1,6144] parameter(0)\n"
        "  %rogue = bf16[1,6144] parameter(1)\n"
        "  ROOT %selected = bf16[1,6144] copy(%rogue)\n"
        "}\n"
        "ENTRY main {",
        1,
    ).replace(
        "  %dense_gate_bits.0 = u8[384,6144] parameter(",
        "  %rogue_dense_input = bf16[1,6144] parameter(997)\n"
        "  %decoy_dense_input = bf16[1,6144] fusion("
        "%attention_update, %rogue_dense_input), kind=kLoop, "
        "calls=%select_rogue\n"
        "  %dense_gate_bits.0 = u8[384,6144] parameter(",
        1,
    ).replace(
        f"custom-call(%attention_update, %dense_gate_bits.0",
        f"custom-call(%decoy_dense_input, %dense_gate_bits.0",
        1,
    )
    fusion_decoy_record = MODULE._validate_hlo(fusion_decoy, arm)
    assert not fusion_decoy_record["passed"]
    assert any(
        "attention reduction does not feed dense" in violation
        for violation in fusion_decoy_record["violations"]
    )
    fusion_live_record = MODULE._validate_hlo(
        fusion_decoy.replace(
            "ROOT %selected = bf16[1,6144] copy(%rogue)",
            "ROOT %selected = bf16[1,6144] copy(%live)",
            1,
        ),
        arm,
    )
    assert fusion_live_record["passed"], fusion_live_record["violations"]

    tuple_fusion_decoy = fusion_decoy.replace(
        "-> bf16[1,6144] {",
        "-> (bf16[1,6144], bf16[1,6144]) {",
        1,
    ).replace(
        "ROOT %selected = bf16[1,6144] copy(%rogue)",
        "ROOT %selected = (bf16[1,6144], bf16[1,6144]) "
        "tuple(%live, %rogue)",
        1,
    ).replace(
        "  %decoy_dense_input = bf16[1,6144] fusion("
        "%attention_update, %rogue_dense_input), kind=kLoop, "
        "calls=%select_rogue",
        "  %decoy_dense_pair = (bf16[1,6144], bf16[1,6144]) fusion("
        "%attention_update, %rogue_dense_input), kind=kLoop, "
        "calls=%select_rogue\n"
        "  %decoy_dense_input = bf16[1,6144] get-tuple-element("
        "%decoy_dense_pair), index=1",
        1,
    )
    tuple_fusion_decoy_record = MODULE._validate_hlo(tuple_fusion_decoy, arm)
    assert not tuple_fusion_decoy_record["passed"]
    assert any(
        "attention reduction does not feed dense" in violation
        for violation in tuple_fusion_decoy_record["violations"]
    )
    tuple_fusion_live_record = MODULE._validate_hlo(
        tuple_fusion_decoy.replace("index=1", "index=0", 1), arm
    )
    assert tuple_fusion_live_record["passed"], tuple_fusion_live_record[
        "violations"
    ]

    local_hlo = _synthetic_hlo(MODULE._ARMS[0])
    wrong_reducer = local_hlo.replace(
        "ROOT %sum = bf16[] add(%lhs, %rhs)",
        "ROOT %sum = bf16[] maximum(%lhs, %rhs)",
        1,
    )
    wrong_reducer_record = MODULE._validate_hlo(
        wrong_reducer, MODULE._ARMS[0]
    )
    assert not wrong_reducer_record["passed"]
    assert any(
        "BF16 add reducer" in violation
        for violation in wrong_reducer_record["violations"]
    )


def test_projection_reduction_runtime_imports_resolve() -> None:
    symbols = MODULE._load_runtime_symbols()
    assert len(symbols) == 15
    assert symbols[0].__name__ == "jax"
    assert all(symbol is not None for symbol in symbols)


def test_projection_reduction_association_analysis_is_pinned(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "physical_row": row,
            "trials": 32,
            "uncovered_column_count": 0,
            "union_exact_column_count": 6144,
            "width": 6144,
        }
        for row in range(32)
    ]
    payload = {
        "accepted_model_axis_device_ids": list(
            MODULE._ACCEPTED_MODEL_AXIS_DEVICE_IDS
        ),
        "block_width": 128,
        "code_hash": MODULE._ASSOCIATION_CODE_HASH,
        "compile_bucket_rows": 32,
        "maximum_row_union_exact_column_count": 6144,
        "minimum_row_union_exact_column_count": 6144,
        "rows": rows,
        "total_union_exact_column_count": 32 * 6144,
        "trials_per_physical_row": 32,
        "width": 6144,
    }
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(payload))
    record = MODULE._load_association_analysis(
        path, expected_sha256=MODULE._file_sha256(path)
    )
    assert record["row_count"] == 32

    payload["rows"][0]["union_exact_column_count"] = 6143
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="row 0 association evidence drifted"):
        MODULE._load_association_analysis(
            path, expected_sha256=MODULE._file_sha256(path)
        )


def test_projection_reduction_bit_comparison() -> None:
    expected = np.zeros(8, dtype=np.uint16)
    observed = expected.copy()
    observed[3] = np.uint16(1)
    record = MODULE._compare_bits(expected, observed)
    assert record["mismatch_count"] == 1
    assert record["first_mismatch_index"] == 3
    assert not record["elementwise_exact"]


def test_projection_reduction_checkpoint_evidence_must_equal_manifest(
    tmp_path: Path,
) -> None:
    files = []
    for global_slot in range(32):
        stage = global_slot // 4
        slot = global_slot % 4
        files.append(
            {
                "destination_filename": (
                    f"base_decoder_runtime_feature/stage_{stage:02d}/"
                    f"device_slot_{slot:02d}.safetensors"
                ),
                "device_slot": slot,
                "file_bytes": 9,
                "header_bytes": 9,
                "header_sha256": MODULE.sha256(b"123456789").hexdigest(),
                "sha256": f"{global_slot:064x}",
                "stage_id": stage,
                "tensors": [],
            }
        )
    manifest = {
        "artifact_kind": "greenfield_feature_runtime_packed_checkpoint",
        "file_count": 32,
        "files": files,
        "plan_id": "PP8_LP4",
    }
    manifest_path = tmp_path / "runtime_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    evidence_path = (
        tmp_path
        / "evidence/base_decoder_runtime_feature/stage_00/"
        "device_slot_00.safetensors.json"
    )
    evidence_path.parent.mkdir(parents=True)
    tensor_path = (
        tmp_path
        / "base_decoder_runtime_feature/stage_00/device_slot_00.safetensors"
    )
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(b"123456789")
    altered = dict(files[0])
    altered["sha256"] = "f" * 64
    evidence_path.write_text(json.dumps(altered))
    with pytest.raises(RuntimeError, match="evidence differs from runtime manifest"):
        MODULE._load_weights(
            tmp_path,
            manifest_sha256=MODULE._file_sha256(manifest_path),
        )


def test_projection_reduction_wrapper_pins_protection() -> None:
    text = WRAPPER.read_text()
    for token in (
        "ATTENTION_TENSOR_SHA=7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61",
        "ACCEPTED_NPZ_SHA=3a619a0985fbb9ba6e1be9347bbc745c120fa74553ce1de5de830190a29c0a30",
        "INGREDIENT_NPZ_SHA=c06fe575f0518e981c3099a8033c1978b01fd0cff843ecbbd7a25cebed8d0e95",
        "LAYER1_REFERENCE_SHA=79b813daa8e194b6c9a9ad883a0199f4a938ca4d4ab7277d20a291b480349054",
        "ASSOCIATION_ANALYSIS_SHA=e7e34828365ca3d6cae0052f8d0e2e802143c6ca83810153db3116423f994108",
        "CHECKPOINT_MANIFEST_SHA=de46d38e404c637209f95505291105e89a6e7f95270fe91375a55ea79b5f7134",
        ".glm_pod_workload.lock",
        "strict_census pre",
        "strict_census post",
        "remote_objects.json",
        "google_crc32c",
        "rollback_provisional_db",
        "correct=bool(runner[\"exact_arms\"])",
        "terminal_success_done=1",
    ):
        assert token in text
    assert text.index("strict_census post") < text.index("pv.start_run")


def _rollback_program() -> str:
    match = re.search(
        r"rollback_provisional_db\(\) \{.*?<<'PY'\n(?P<program>.*?)\nPY\n\}",
        WRAPPER.read_text(),
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("program")


@pytest.mark.parametrize("state", ("run", "item", "summary"))
def test_projection_reduction_rollback_removes_each_committed_prefix(
    tmp_path: Path,
    state: str,
) -> None:
    provenance_path = REPO / "bench/provenance.py"
    specification = importlib.util.spec_from_file_location(
        f"projection_rollback_provenance_{state}", provenance_path
    )
    assert specification is not None and specification.loader is not None
    provenance = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(provenance)

    database = tmp_path / f"{state}.db"
    tag = f"rollback_{state}"
    pin = "a" * 40
    connection = provenance.connect(str(database))
    run_id = provenance.start_run(
        connection,
        model="zai-org/GLM-5.2-FP8:greenfield-layer0-projection-reduction",
        revision="native-jax-db537-strategy-nd-v1",
        env={
            "GLM_ENGINE": "greenfield_projection_reduction_probe",
            "greenfield_code_hash": pin,
            "greenfield_run_tag": tag,
        },
        note=(
            "Protected layer-0 projection/reduction discriminator; "
            "no performance claim."
        ),
        harness_repo=str(REPO),
        fork_repo=None,
    )
    if state in {"item", "summary"}:
        provenance.record_item(
            connection,
            run_id,
            benchmark="greenfield_layer0_projection_reduction",
            item_id="position8155",
            prompt="sealed",
            gold="Exact accepted BF16 layer-1 normalized hidden [6144].",
            raw_output="{}",
            extracted="none",
            correct=False,
            score=0.0,
        )
    if state == "summary":
        provenance.finalize(
            connection,
            run_id,
            benchmark="greenfield_layer0_projection_reduction",
            metric="probe_contract_valid",
            value=1.0,
            note="Diagnostic only.",
        )
    connection.close()

    result = subprocess.run(
        [sys.executable, "-", str(database), tag, pin],
        input=_rollback_program(),
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
