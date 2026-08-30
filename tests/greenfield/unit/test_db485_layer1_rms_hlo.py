from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.greenfield.classify_db485_layer1_rms_hlo import (
    _write_append_only_json,
    classify_files,
    classify_texts,
)


ACCEPTED = Path(
    "/home/gianl/glm-run/accepted_db485_compile_only_hlo_"
    "20260830T025924791267740Z/hlo/"
    "jit_step_fun_impl.m32.module_5202.jit_step_fun_impl.cl_914450892."
    "after_codegen.txt.gz"
)
DB518 = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_layer0_db518_numerical_"
    "20260829T115022665987633Z/hlo/feature2_main.optimized_hlo.txt"
)
WS32 = Path(
    "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_acquire_"
    "20260827T003758068665390Z/hlo/decode.optimized_hlo.txt"
)
ACCEPTED_RUN = Path(
    "/home/gianl/glm-run/accepted_db485_compile_only_hlo_"
    "20260830T025924791267740Z"
)
SUCCESS_CAPSULE = Path("docs/artifacts/accepted-db485-compile-only-hlo-success.json")
CAUSALITY_CAPSULE = Path("docs/artifacts/db485-layer1-rms-hlo-causality.json")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _replace_line(text: str, prefix: str, old: str, new: str) -> str:
    lines = text.splitlines(keepends=True)
    matches = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith(prefix)
    ]
    assert len(matches) == 1, (prefix, len(matches))
    index = matches[0]
    assert old in lines[index], (prefix, old)
    lines[index] = lines[index].replace(old, new, 1)
    return "".join(lines)


@pytest.mark.skipif(
    not ACCEPTED.is_file() or not DB518.is_file() or not WS32.is_file(),
    reason="sealed DB485/DB518/WS32 HLO inputs unavailable",
)
def test_db485_layer1_rms_hlo_classification_is_fail_closed() -> None:
    result = classify_files(ACCEPTED, DB518, WS32)
    assert result["status"] == "CLASSIFIED"
    assert result["classification"] == (
        "LOGICAL_BF16_ASSOCIATION_MATCHES;"
        "UNROUNDED_STATE_PHYSICAL_CAUSE_UNRESOLVED;"
        "RMS_REDUCTION_AND_WEIGHTED_OUTPUT_GEOMETRY_REMAIN_DISTINCT"
    )
    assert result["correction_materialization_proven"] is False
    assert result["logical_bf16_association_matches"] is True
    assert result["unrounded_state_adjudicated"] is False
    assert result["accepted"]["rms_association"].startswith("fp32(bf16_dense")
    assert result["db518"]["post_weight_all_gather_shape"] == [1, 6144]
    assert result["db518"]["rms_feature_group_size"] == 2
    assert result["ws32"]["post_weight_all_gather_shape"] == [1, 6144]
    assert result["ws32"]["feature_group_size"] == 4
    assert result["ws32"]["full_dense_value_path_claim"] is False
    assert result["ws32"]["target_boundary_certificate"].startswith(
        "exact BF16 StrategyND row0-tree operands"
    )
    assert result["gate_d_passed"] is False
    assert result["tpu_successor_authorized"] is False
    assert json.loads(CAUSALITY_CAPSULE.read_text()) == result
    assert CAUSALITY_CAPSULE.read_text() == (
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    accepted = gzip.decompress(ACCEPTED.read_bytes()).decode()
    db518 = DB518.read_text()
    ws32 = WS32.read_text()
    with pytest.raises(ValueError, match="accepted dense BF16 result"):
        classify_texts(
            accepted.replace(
                "psum.1101 = bf16[32,6144]",
                "psum.1101 = f32[32,6144]",
                1,
            ),
            db518,
            ws32,
        )
    with pytest.raises(ValueError, match=r"DB518 .*window"):
        classify_texts(
            accepted,
            db518.replace(
                '"output_window_bounds":["1","12"]',
                '"output_window_bounds":["1","13"]',
            ),
            ws32,
        )


@pytest.mark.skipif(
    not ACCEPTED.is_file() or not DB518.is_file() or not WS32.is_file(),
    reason="sealed DB485/DB518/WS32 HLO inputs unavailable",
)
def test_db485_layer1_rms_hlo_rejects_hostile_value_flow_mutations() -> None:
    accepted = gzip.decompress(ACCEPTED.read_bytes()).decode()
    db518 = DB518.read_text()
    ws32 = WS32.read_text()

    mutations = (
        (
            "accepted source swap",
            accepted.replace(
                "fusion(psum.1101, add_rsqrt_fusion.231",
                "fusion(psum.1100, add_rsqrt_fusion.231",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted missing carried round",
            accepted.replace(
                "convert_element_type.23474 = bf16[32,6144]",
                "convert_element_type.23474 = f32[32,6144]",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted wrong RMS weight owner",
            accepted.replace(
                "get-tuple-element(param.2777), index=25",
                "get-tuple-element(param.2777), index=26",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted wrong weighted-call weight",
            accepted.replace(
                "fusion(psum.1101, add_rsqrt_fusion.231, copy-done.1096",
                "fusion(psum.1101, add_rsqrt_fusion.231, copy-done.1095",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted RMS inverse source break",
            accepted.replace(
                "fusion(multiply_reduce_fusion.316), kind=kLoop, "
                "calls=fused_computation.18515",
                "fusion(multiply_reduce_fusion.315), kind=kLoop, "
                "calls=fused_computation.18515",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted RMS reducer source break",
            accepted.replace(
                "fusion(psum.1101, psum.1100, all-reduce.3, "
                "get-tuple-element.44327), kind=kLoop, "
                "calls=fused_computation.12442",
                "fusion(psum.1102, psum.1100, all-reduce.3, "
                "get-tuple-element.44327), kind=kLoop, "
                "calls=fused_computation.12442",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted weighted output loses live consumer",
            accepted.replace(
                "fusion(fusion.6342, bitcast.24623, copy-done.350), kind=kOutput",
                "fusion(fusion.6341, bitcast.24623, copy-done.350), kind=kOutput",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "DB518 wrong local group",
            accepted,
            db518.replace(
                "%psum.335 = f32[]{:T(128)} all-reduce(%get-tuple-element.16181), "
                "channel_id=1, replica_groups={{0,1}}",
                "%psum.335 = f32[]{:T(128)} all-reduce(%get-tuple-element.16181), "
                "channel_id=1, replica_groups={{0,2}}",
                1,
            ),
            ws32,
        ),
        (
            "DB518 square result source break",
            accepted,
            _replace_line(
                db518,
                "%get-tuple-element.16181 = ",
                "get-tuple-element(%multiply_reduce_fusion.131), index=0",
                "get-tuple-element(%multiply_reduce_fusion.130), index=0",
            ),
            ws32,
        ),
        (
            "DB518 unnormalized state source break",
            accepted,
            _replace_line(
                db518,
                "%get-tuple-element.16173 = ",
                "get-tuple-element(%multiply_reduce_fusion.131), index=1",
                "get-tuple-element(%multiply_reduce_fusion.130), index=1",
            ),
            ws32,
        ),
        (
            "DB518 initial RMS source swap",
            accepted,
            db518.replace(
                "fusion(%collective-permute-done.3, %get-tuple-element.17883, "
                "%min.1123, %and.4286, %eq.947)",
                "fusion(%collective-permute-done.3, %get-tuple-element.17884, "
                "%min.1123, %and.4286, %eq.947)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 embedding owner zero break",
            accepted,
            db518.replace(
                "%constant.4603.clone.15 = bf16[]{:T(256)} constant(0)",
                "%constant.4603.clone.15 = bf16[]{:T(256)} constant(1)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 wrong embedding table owner",
            accepted,
            _replace_line(
                db518,
                "%get-tuple-element.17883 = ",
                "get-tuple-element(%wide.wide.param.2), index=25",
                "get-tuple-element(%wide.wide.param.2), index=24",
            ),
            ws32,
        ),
        (
            "DB518 embedding owner fallback break",
            accepted,
            db518.replace(
                "select(%select_n.5840, %dynamic_slice.771, %closed_call.913)",
                "select(%select_n.5840, %dynamic_slice.771, %dynamic_slice.771)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial BF16 association break",
            accepted,
            db518.replace(
                "add(%convert.1576, %convert.1577)",
                "add(%convert.1576, %convert.1576)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial BF16 zero break",
            accepted,
            db518.replace(
                "%constant.4603.clone.5.clone.4 = bf16[]{:T(256)} constant(0)",
                "%constant.4603.clone.5.clone.4 = bf16[]{:T(256)} constant(1)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 complementary zero fallback break",
            accepted,
            db518.replace(
                "select(%select_n.5047.clone.4, %dynamic_slice.615.clone.4, "
                "%closed_call.633.clone.4)",
                "select(%select_n.5047.clone.4, %dynamic_slice.615.clone.4, "
                "%dynamic_slice.615.clone.4)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial corrected-state copy break",
            accepted,
            _replace_line(
                db518,
                "%copy-start.102 = ",
                "copy-start(%copy-done.100)",
                "copy-start(%copy-done.101)",
            ),
            ws32,
        ),
        (
            "DB518 attention copy break",
            accepted,
            _replace_line(
                db518,
                "%copy-start.141 = ",
                "copy-start(%copy-done.140)",
                "copy-start(%copy-done.139)",
            ),
            ws32,
        ),
        (
            "DB518 attention concat source break",
            accepted,
            _replace_line(
                db518,
                "%constant_dynamic-update-slice_fusion.358 = ",
                "fusion(%constant_dynamic-update-slice_fusion.357, %add.14172)",
                "fusion(%constant_dynamic-update-slice_fusion.356, %add.14172)",
            ),
            ws32,
        ),
        (
            "DB518 attention concat value break",
            accepted,
            _replace_line(
                db518,
                "%constant_dynamic-update-slice_fusion.358 = ",
                "fusion(%constant_dynamic-update-slice_fusion.357, %add.14172)",
                "fusion(%constant_dynamic-update-slice_fusion.357, %add.14171)",
            ),
            ws32,
        ),
        (
            "DB518 dense concat source break",
            accepted,
            _replace_line(
                db518,
                "%constant_dynamic-update-slice_fusion.370 = ",
                "fusion(%constant_dynamic-update-slice_fusion.369, %add.14189)",
                "fusion(%constant_dynamic-update-slice_fusion.368, %add.14189)",
            ),
            ws32,
        ),
        (
            "DB518 dense concat value break",
            accepted,
            _replace_line(
                db518,
                "%constant_dynamic-update-slice_fusion.370 = ",
                "fusion(%constant_dynamic-update-slice_fusion.369, %add.14189)",
                "fusion(%constant_dynamic-update-slice_fusion.369, %add.14188)",
            ),
            ws32,
        ),
        (
            "DB518 RMS inverse source break",
            accepted,
            db518.replace(
                "%rsqrt.332 = f32[]{:T(128)S(6)} rsqrt(%add.14192)",
                "%rsqrt.332 = f32[]{:T(128)S(6)} rsqrt(%add.14191)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 wrong RMS weight owner",
            accepted,
            _replace_line(
                db518,
                "%get-tuple-element.17905 = ",
                "get-tuple-element(%wide.wide.param.2), index=47",
                "get-tuple-element(%wide.wide.param.2), index=46",
            ),
            ws32,
        ),
        (
            "DB518 wrong weighted-call weight",
            accepted,
            db518.replace(
                "fusion(%copy-done.114, %get-tuple-element.16173, %rsqrt.332)",
                "fusion(%copy-done.115, %get-tuple-element.16173, %rsqrt.332)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 gathered output loses live root",
            accepted,
            db518.replace(
                "/*index=5*/%get-tuple-element.17863, %copy-done.76, "
                "%copy-done.91",
                "/*index=5*/%get-tuple-element.17863, %copy-done.75, "
                "%copy-done.91",
                1,
            ),
            ws32,
        ),
        (
            "WS32 wrong layer",
            accepted,
            db518,
            ws32.replace(
                "%fused_computation.13416 (",
                "%fused_computation.13499 (",
                1,
            ),
        ),
        (
            "WS32 source swap",
            accepted,
            db518,
            ws32.replace(
                "fusion(%psum.8778, %psum.8781, %add.28843, %add.28848, "
                "%mul.17735)",
                "fusion(%psum.8781, %psum.8778, %add.28843, %add.28848, "
                "%mul.17735)",
                1,
            ),
        ),
        (
            "WS32 embedding source break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%psum.8778 = ",
                "all-reduce(%broadcast_select_fusion.165)",
                "all-reduce(%broadcast_select_fusion.164)",
            ),
        ),
        (
            "WS32 attention slice source break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%slice.34982 = ",
                "slice(%greenfield_fp8_block_matmul_f32_m8_k2048_n1536.78)",
                "slice(%greenfield_fp8_block_matmul_f32_m8_k2048_n1536.77)",
            ),
        ),
        (
            "WS32 attention reduction source break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%psum.8781 = ",
                "all-reduce(%slice.34982)",
                "all-reduce(%slice.34981)",
            ),
        ),
        (
            "WS32 target row selector break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%mul.17735 = ",
                "multiply(%axis_index.17, %constant.25110.clone.1)",
                "multiply(%axis_index.16, %constant.25110.clone.1)",
            ),
        ),
        (
            "WS32 post-attention source swap",
            accepted,
            db518,
            ws32.replace(
                "fusion(%psum.8778, %psum.8781), kind=kLoop, "
                "calls=%fused_computation.13424",
                "fusion(%psum.8781, %psum.8778), kind=kLoop, "
                "calls=%fused_computation.13424",
                1,
            ),
        ),
        (
            "WS32 post-attention inverse source break",
            accepted,
            db518,
            ws32.replace(
                "%rsqrt.1098 = f32[1,1]{1,0:T(1,128)} rsqrt(%add.28822)",
                "%rsqrt.1098 = f32[1,1]{1,0:T(1,128)} rsqrt(%add.28823)",
                1,
            ),
        ),
        (
            "WS32 post-attention wrong RMS weight owner",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%get-tuple-element.26443 = ",
                "get-tuple-element(%param.4144), index=19",
                "get-tuple-element(%param.4144), index=20",
            ),
        ),
        (
            "WS32 post-attention wrong feature group",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%psum.8782 = ",
                "replica_groups={{0,1,2,3}",
                "replica_groups={{0,1,2,4}",
            ),
        ),
        (
            "WS32 dense no longer consumes normalized state",
            accepted,
            db518,
            ws32.replace(
                "%convolution_convert_fusion.11 = bf16[1,768]"
                "{1,0:T(2,128)(2,1)S(3)} fusion(%all_gather.931",
                "%convolution_convert_fusion.11 = bf16[1,768]"
                "{1,0:T(2,128)(2,1)S(3)} fusion(%all_gather.930",
                1,
            ),
        ),
        (
            "WS32 missing normalized round",
            accepted,
            db518,
            ws32.replace(
                "%convert_element_type.20269 = bf16[1,1536]",
                "%convert_element_type.20269 = f32[1,1536]",
                1,
            ),
        ),
        (
            "WS32 dense target producer break",
            accepted,
            db518,
            ws32.replace(
                "add(%get-tuple-element.71301, %get-tuple-element.71299)",
                "add(%get-tuple-element.71300, %get-tuple-element.71298)",
                1,
            ),
        ),
        (
            "WS32 dense tuple source break",
            accepted,
            db518,
            ws32.replace(
                "fusion(%copy_bitcast_fusion.97), kind=kLoop, "
                "calls=%fused_computation.13418",
                "fusion(%copy_bitcast_fusion.96), kind=kLoop, "
                "calls=%fused_computation.13418",
                1,
            ),
        ),
        (
            "WS32 square result source break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%get-tuple-element.71072 = ",
                "get-tuple-element(%multiply_reduce_fusion.455), index=0",
                "get-tuple-element(%multiply_reduce_fusion.454), index=0",
            ),
        ),
        (
            "WS32 unnormalized state source break",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%get-tuple-element.71073 = ",
                "get-tuple-element(%multiply_reduce_fusion.455), index=1",
                "get-tuple-element(%multiply_reduce_fusion.454), index=1",
            ),
        ),
        (
            "WS32 RMS inverse source break",
            accepted,
            db518,
            ws32.replace(
                "%rsqrt.1099 = f32[1,1]{1,0:T(1,128)} rsqrt(%add.28850)",
                "%rsqrt.1099 = f32[1,1]{1,0:T(1,128)} rsqrt(%add.28851)",
                1,
            ),
        ),
        (
            "WS32 wrong RMS weight owner",
            accepted,
            db518,
            _replace_line(
                ws32,
                "%get-tuple-element.26448 = ",
                "get-tuple-element(%param.4144), index=24",
                "get-tuple-element(%param.4144), index=25",
            ),
        ),
        (
            "WS32 wrong output layout",
            accepted,
            db518,
            ws32.replace(
                "%reshape_multiply_fusion.231 = bf16[1,1536]"
                "{1,0:T(2,128)(2,1)S(3)}",
                "%reshape_multiply_fusion.231 = bf16[1,1536]"
                "{1,0:T(8,128)(2,1)S(3)}",
                1,
            ),
        ),
        (
            "WS32 wrong feature group",
            accepted,
            db518,
            ws32.replace(
                "%psum.8783 = f32[1,1]{1,0:T(1,128)} all-reduce(%bitcast.8888), "
                "channel_id=1, replica_groups={{0,1,2,3}",
                "%psum.8783 = f32[1,1]{1,0:T(1,128)} all-reduce(%bitcast.8888), "
                "channel_id=1, replica_groups={{0,1,2,4}",
                1,
            ),
        ),
        (
            "WS32 dead duplicate target",
            accepted,
            db518,
            ws32
            + "\n"
            + next(
                line
                for line in ws32.splitlines()
                if line.lstrip().startswith("%reshape_multiply_fusion.231 = ")
            ),
        ),
        (
            "accepted reducer drops checked square",
            accepted.replace(
                "reduce(pow.1331, constant.14528.clone.445)",
                "reduce(pow.1330, constant.14528.clone.445)",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted normalized value drops inverse",
            accepted.replace(
                "multiply(add.23462, mul.22995)",
                "multiply(add.23462, add.23462)",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted inverse uses wrong epsilon broadcast",
            accepted.replace(
                "add(div.8633, broadcast.29230)",
                "add(div.8633, broadcast.29231)",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted rsqrt drops epsilon",
            accepted.replace(
                "rsqrt(add.26410)",
                "rsqrt(div.8633)",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "accepted weighted value drops weight",
            accepted.replace(
                "multiply(convert.4444, convert.4445)",
                "multiply(convert.4444, convert.4444)",
                1,
            ),
            db518,
            ws32,
        ),
        (
            "DB518 reducer drops checked square",
            accepted,
            db518.replace(
                "reduce(%square.414, %constant.4609.clone.35)",
                "reduce(%square.413, %constant.4609.clone.35)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial reducer drops checked square",
            accepted,
            db518.replace(
                "reduce(%square.410, %constant.4609.clone.29)",
                "reduce(%square.409, %constant.4609.clone.29)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial tuple drops corrected state",
            accepted,
            db518.replace(
                "tuple(%reduce_sum.844, %convert_element_type.8495.clone.4)",
                "tuple(%reduce_sum.844, %add.12644.clone.4)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 initial corrected state drops owner sum",
            accepted,
            db518.replace(
                "convert(%add.12644.clone.4)",
                "convert(%add.12643.clone.4)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 tuple drops checked state",
            accepted,
            db518.replace(
                "tuple(%reduce_sum.848, %add.12639.clone.4)",
                "tuple(%reduce_sum.848, %add.12640.clone.4)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 normalized value drops inverse",
            accepted,
            db518.replace(
                "multiply(%param_1.11640, %mul.9794)",
                "multiply(%param_1.11640, %param_1.11640)",
                1,
            ),
            ws32,
        ),
        (
            "DB518 weighted value drops weight",
            accepted,
            db518.replace(
                "multiply(%convert.1846, %convert.1847)",
                "multiply(%convert.1846, %convert.1846)",
                1,
            ),
            ws32,
        ),
        (
            "WS32 post-attention reducer drops checked square",
            accepted,
            db518,
            ws32.replace(
                "reduce(%square.1590, %constant.25099.clone.644)",
                "reduce(%square.1589, %constant.25099.clone.644)",
                1,
            ),
        ),
        (
            "WS32 post-attention normalized value drops inverse",
            accepted,
            db518,
            ws32.replace(
                "multiply(%add.34747, %mul.25127)",
                "multiply(%add.34747, %add.34747)",
                1,
            ),
        ),
        (
            "WS32 post-attention weighted value drops weight",
            accepted,
            db518,
            ws32.replace(
                "multiply(%convert.11481, %convert.11482)",
                "multiply(%convert.11481, %convert.11481)",
                1,
            ),
        ),
        (
            "WS32 reducer drops checked square",
            accepted,
            db518,
            ws32.replace(
                "reduce(%square.1589, %constant.25099.clone.643)",
                "reduce(%square.1588, %constant.25099.clone.643)",
                1,
            ),
        ),
        (
            "WS32 tuple drops checked state",
            accepted,
            db518,
            ws32.replace(
                "tuple(%reduce_sum.7562, %add.35993.clone.1)",
                "tuple(%reduce_sum.7562, %add.35997.clone.1)",
                1,
            ),
        ),
        (
            "WS32 normalized value drops inverse",
            accepted,
            db518,
            ws32.replace(
                "multiply(%param_1.26801, %mul.25126)",
                "multiply(%param_1.26801, %param_1.26801)",
                1,
            ),
        ),
        (
            "WS32 weighted value drops weight",
            accepted,
            db518,
            ws32.replace(
                "multiply(%convert.11470, %convert.11471)",
                "multiply(%convert.11470, %convert.11470)",
                1,
            ),
        ),
        (
            "WS32 row0-tree first pair self-add",
            accepted,
            db518,
            ws32.replace(
                "add(%convert.11473, %convert.11474)",
                "add(%convert.11473, %convert.11473)",
                1,
            ),
        ),
        (
            "WS32 row0-tree root member break",
            accepted,
            db518,
            ws32.replace(
                "tuple(%bitcast.26036, %bitcast.26038.clone.1, "
                "%bitcast.26037.clone.1, %bitcast.26039.clone.1)",
                "tuple(%bitcast.26038.clone.1, %bitcast.26038.clone.1, "
                "%bitcast.26037.clone.1, %bitcast.26039.clone.1)",
                1,
            ),
        ),
    )
    for _label, mutated_accepted, mutated_db518, mutated_ws32 in mutations:
        assert (mutated_accepted, mutated_db518, mutated_ws32) != (
            accepted,
            db518,
            ws32,
        ), _label
        try:
            classify_texts(mutated_accepted, mutated_db518, mutated_ws32)
        except ValueError:
            pass
        else:
            pytest.fail(f"hostile mutation classified: {_label}")


@pytest.mark.skipif(
    not ACCEPTED_RUN.is_dir() or not SUCCESS_CAPSULE.is_file(),
    reason="accepted DB485 success archive unavailable",
)
def test_accepted_db485_success_capsule_binds_local_terminal_archive() -> None:
    capsule = json.loads(SUCCESS_CAPSULE.read_text())
    manifest = ACCEPTED_RUN / "manifest.json"
    remote_objects_file = ACCEPTED_RUN / "remote_objects.json"
    success = ACCEPTED_RUN / "SUCCESS"
    remote_objects = json.loads(remote_objects_file.read_text())["objects"]

    assert _sha256(manifest) == capsule["manifest_sha256"]
    assert _sha256(remote_objects_file) == capsule["remote_objects_sha256"]
    assert _sha256(success) == capsule["success_file_sha256"]
    assert len(remote_objects) + 2 == capsule["object_count"]
    assert (
        sum(item["size"] for item in remote_objects)
        + remote_objects_file.stat().st_size
        + success.stat().st_size
        == capsule["remote_object_bytes"]
    )

    compressed = ACCEPTED
    raw = gzip.decompress(compressed.read_bytes())
    assert compressed.stat().st_size == capsule["m32_hlo"]["compressed_byte_count"]
    assert _sha256(compressed) == capsule["m32_hlo"]["compressed_sha256"]
    assert len(raw) == capsule["m32_hlo"]["raw_byte_count"]
    assert sha256(raw).hexdigest() == capsule["m32_hlo"]["raw_sha256"]
    assert capsule["workflow"] == {
        "decode_calls": 0,
        "generate_calls": 0,
        "host_count": 8,
        "integrity_passed": True,
        "state_passed": True,
        "zero_work_cleanup_hosts": 8,
    }


def test_append_only_output_refuses_existing_paths_and_dangling_symlinks(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("preserve me\n")
    with pytest.raises(FileExistsError):
        _write_append_only_json(existing, {"wrong": True})
    assert existing.read_text() == "preserve me\n"

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileExistsError):
        _write_append_only_json(dangling, {"wrong": True})
    assert dangling.is_symlink()

    fresh = tmp_path / "fresh.json"
    _write_append_only_json(fresh, {"ok": True})
    assert fresh.read_text() == '{\n  "ok": true\n}\n'
