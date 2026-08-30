#!/usr/bin/env python3
"""Classify the sealed DB485 layer-0-to-layer-1 RMS HLO boundary.

This is an offline, pin-specific mechanism certificate.  It compares the
accepted legacy M32 after-codegen module with the rejected DB518 PP16 module
and the sealed WS32 short-decoder module.  It never imports any execution path
and makes no numerical or Gate-D claim.
"""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ACCEPTED_GZIP_SHA256 = (
    "b0f7f7b90ea229caa73e1f7c29afff01fecb064065384298bc8659fdd11c024b"
)
ACCEPTED_RAW_SHA256 = (
    "5c23a74f03551923b4b6b0cc35e5802c2d26e634a05b9bdbf146aabf7ef972ea"
)
DB518_OPTIMIZED_SHA256 = (
    "634cf81a31a89aaa704a1715ae7d2cb4fd96b588463d76c6061f5c90ab4607ca"
)
WS32_OPTIMIZED_SHA256 = (
    "8f964f9e5bac5141921973facc73e30c0a3eacbf6bbcdb09958d0a7150100ce6"
)


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise ValueError(f"DB485 layer-1 RMS certificate lost {label}")


def _require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise ValueError(
            f"DB485 layer-1 RMS certificate expected one {label}; found {count}"
        )


def _line(text: str, prefix: str, label: str) -> str:
    matches = [line for line in text.splitlines() if line.lstrip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"DB485 layer-1 RMS certificate expected one {label}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _require_order(text: str, needles: tuple[str, ...], label: str) -> None:
    cursor = -1
    for needle in needles:
        position = text.find(needle, cursor + 1)
        if position < 0:
            raise ValueError(f"DB485 layer-1 RMS certificate lost {label}: {needle}")
        cursor = position


def _computation(text: str, name: str) -> str:
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if line == f"{name} {{"
        or (line.startswith(f"{name} (") and line.endswith(" {"))
    ]
    if len(starts) != 1:
        raise ValueError(
            f"DB485 layer-1 RMS computation expected once: {name}; "
            f"found {len(starts)}"
        )
    start = starts[0]
    for stop in range(start + 1, len(lines)):
        if lines[stop] == "}":
            return "\n".join(lines[start : stop + 1])
    raise ValueError(f"DB485 layer-1 RMS computation is unterminated: {name}")


def _sealed_computation(
    text: str, name: str, expected_sha256: str, label: str
) -> str:
    computation = _computation(text, name)
    actual_sha256 = _sha256(computation.encode())
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"DB485 layer-1 RMS certificate {label} computation drifted: "
            f"{actual_sha256}"
        )
    return computation


def classify_texts(accepted: str, db518: str, ws32: str) -> dict[str, Any]:
    """Fail closed on the exact accepted and candidate causal structures."""

    accepted_reduce = _line(
        accepted, "psum.1101 = ", "accepted first-dense reduction"
    )
    for needle, label in (
        ("bf16[32,6144]", "accepted dense BF16 result"),
        ("all-reduce(fusion.6106)", "accepted dense all-reduce"),
        (
            "replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,"
            "16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}}",
            "accepted model group",
        ),
        (
            'op_name="jit(step_fun_impl)/VllmRowParallelLinear/shard_map/psum"',
            "accepted projection identity",
        ),
        ("stack_frame_id=1226", "accepted first dense stack"),
    ):
        _require(accepted_reduce, needle, label)

    accepted_rms = _sealed_computation(
        accepted,
        "fused_computation.12442",
        "e81dd31bd46b479b050760bb3f9f5f096bf25d8acaeea64e5e70d1093d478b4f",
        "accepted RMS reduction",
    )
    for needle, label in (
        ("param_0.55797 = bf16[32,6144]", "accepted dense input type"),
        ("convert(param_0.55797)", "accepted dense input conversion"),
        ("param_1.65259 = bf16[32,6144]", "accepted attention input type"),
        ("convert(param_1.65259)", "accepted attention input conversion"),
        ("param_2.50400 = bf16[32,6144]", "accepted residual input type"),
        (
            "select(broadcast_in_dim.18416, param_2.50400, "
            "broadcast_in_dim.18415)",
            "accepted residual selection mapping",
        ),
        ("convert(select_n.18969)", "accepted residual input conversion"),
        (
            "add(convert_element_type.23476, convert_element_type.23475)",
            "accepted attention plus residual",
        ),
        ("convert(add.26479)", "accepted carried BF16 round"),
        (
            "add(convert_element_type.15476, convert_element_type.15475)",
            "accepted dense plus rounded carried",
        ),
        ("multiply(add.22533, add.22533)", "accepted RMS square source"),
        (
            "ROOT reduce.3110 = f32[32]{0:T(128)S(3)} "
            "reduce(pow.1331, constant.14528.clone.445), dimensions={1}",
            "accepted RMS reduction root",
        ),
    ):
        _require(accepted_rms, needle, label)
    _require_order(
        accepted_rms,
        (
            "convert_element_type.23476 =",
            "convert_element_type.23475 =",
            "add.26479 =",
            "convert_element_type.23474 = bf16",
            "convert_element_type.15475 = f32",
            "add.22533 =",
            "pow.1331 =",
            "ROOT reduce.3110 =",
        ),
        "accepted residual/attention/dense round order",
    )

    accepted_weighted = _sealed_computation(
        accepted,
        "fused_computation.12680",
        "dfdd2e342cc966b7d250a007499a01fb72c28bc216a4873d75c3c26088b66deb",
        "accepted weighted RMS",
    )
    for needle, label in (
        ("param_0.55793 = bf16[32,6144]", "accepted weighted dense input"),
        ("convert(param_0.55793)", "accepted weighted dense conversion"),
        ("param_3.28132 = bf16[32,6144]", "accepted weighted attention input"),
        ("convert(param_3.28132)", "accepted weighted attention conversion"),
        ("param_4.13865 = bf16[32,6144]", "accepted weighted residual input"),
        (
            "select(broadcast_in_dim.18406, param_4.13865, "
            "broadcast_in_dim.18405)",
            "accepted weighted residual selection mapping",
        ),
        ("convert(select_n.18964)", "accepted weighted residual conversion"),
        ("convert(add.26475)", "accepted weighted carried round"),
        (
            "add(convert_element_type.17493, convert_element_type.17492)",
            "accepted weighted RMS input",
        ),
        (
            "mul.22995 = f32[32,6144]{1,0:T(8,128)} "
            "broadcast(param_1.63926)",
            "accepted inverse broadcast",
        ),
        (
            "mul.22005 = f32[32,6144]{1,0:T(8,128)} "
            "multiply(add.23462, mul.22995)",
            "accepted inverse normalization multiply",
        ),
        ("convert(mul.22005)", "accepted normalized BF16 round"),
        ("param_2.49521 = bf16[6144]", "accepted layer-1 RMS weight source"),
        (
            "mul.23332 = bf16[32,6144]{1,0:T(8,128)(2,1)} "
            "broadcast(param_2.49521)",
            "accepted RMS weight broadcast",
        ),
        ("convert(mul.23332)", "accepted RMS weight conversion"),
        (
            "mul.22003 = f32[32,6144]{1,0:T(8,128)} "
            "multiply(convert.4444, convert.4445)",
            "accepted corrected weight multiply",
        ),
        (
            '"float_type_correction_info":{"original_type":"BF16"',
            "accepted BF16 correction",
        ),
        (
            "ROOT convert.4446 = bf16[32,6144]{1,0:T(8,128)(2,1)S(3)} "
            "convert(mul.22003)",
            "accepted weighted output",
        ),
    ):
        _require(accepted_weighted, needle, label)
    _require_order(
        accepted_weighted,
        (
            "add.26475 =",
            "convert_element_type.23462 = bf16",
            "add.23462 =",
            "mul.22005 =",
            "convert_element_type.17491 = bf16",
            "mul.22003 =",
            "ROOT convert.4446 =",
        ),
        "accepted weighted value flow",
    )

    accepted_inverse = _sealed_computation(
        accepted,
        "fused_computation.18515",
        "1ccfae9aaf507ea0ea87c3bdfda8ee3354c49b18838bd0037ba0709e06fc64bb",
        "accepted RMS inverse",
    )
    for needle, label in (
        ("constant(0.000162760422)", "accepted hidden-width reciprocal"),
        ("multiply(param_0.55154, broadcast.29388)", "accepted RMS mean"),
        ("constant(1e-05)", "accepted RMS epsilon"),
        (
            "broadcast.29230 = f32[32]{0:T(128)} "
            "broadcast(constant.14530.clone.3)",
            "accepted RMS epsilon broadcast",
        ),
        (
            "add.26410 = f32[32]{0:T(128)} "
            "add(div.8633, broadcast.29230)",
            "accepted RMS epsilon addition",
        ),
        (
            "ROOT rsqrt.1561 = f32[32]{0:T(128)S(3)} rsqrt(add.26410)",
            "accepted RMS inverse",
        ),
    ):
        _require(accepted_inverse, needle, label)

    accepted_reduce_call = _line(
        accepted,
        "multiply_reduce_fusion.316 = ",
        "accepted RMS reduction call",
    )
    for needle, label in (
        ("f32[32]", "accepted RMS reduction shape"),
        (
            "fusion(psum.1101, psum.1100, all-reduce.3, "
            "get-tuple-element.44327)",
            "accepted dense/attention/residual reduction operands",
        ),
        ("calls=fused_computation.12442", "accepted RMS reduction callee"),
        ('"output_window_bounds":["2","48"]', "accepted RMS reduction window"),
        ('"megacore_split_dim":"0"', "accepted RMS reduction split"),
    ):
        _require(accepted_reduce_call, needle, label)

    accepted_call = _line(accepted, "fusion.6342 = ", "accepted weighted call")
    for needle, label in (
        ("bf16[32,6144]{1,0:T(8,128)(2,1)S(3)}", "accepted output tile"),
        (
            "fusion(psum.1101, add_rsqrt_fusion.231, copy-done.1096, "
            "psum.1100, all-reduce.3, get-tuple-element.44327)",
            "accepted weight/dense/inverse/attention/residual operand roles",
        ),
        ("calls=fused_computation.12680", "accepted weighted callee"),
        ('"output_window_bounds":["2","48"]', "accepted output window"),
        ('"megacore_split_dim":"0"', "accepted megacore split"),
    ):
        _require(accepted_call, needle, label)
    for prefix, needles, label in (
        (
            "psum.1100 = ",
            ("bf16[32,6144]", "all-reduce(fusion.6184)", "stack_frame_id=1213"),
            "accepted attention source",
        ),
        (
            "all-reduce.3 = ",
            (
                "bf16[32,6144]",
                "all-reduce(broadcast_select_fusion.312)",
                "stack_frame_id=91",
            ),
            "accepted embedding source",
        ),
        (
            "get-tuple-element.19189 = ",
            ("bf16[6144]", "get-tuple-element(param.2777), index=25"),
            "accepted layer-1 RMS-weight owner",
        ),
        (
            "copy-start.1096 = ",
            ("bf16[6144]", "copy-start(get-tuple-element.19189)"),
            "accepted RMS-weight lineage",
        ),
        (
            "copy-done.1096 = ",
            ("bf16[6144]", "copy-done(copy-start.1096)"),
            "accepted RMS-weight operand",
        ),
        (
            "add_rsqrt_fusion.231 = ",
            (
                "fusion(multiply_reduce_fusion.316)",
                "calls=fused_computation.18515",
                "stack_frame_id=895",
            ),
            "accepted inverse call",
        ),
    ):
        source_line = _line(accepted, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)
    accepted_live_consumer = _line(
        accepted, "fusion.9360 = ", "accepted weighted-output live consumer"
    )
    for needle, label in (
        (
            "bf16[32,1,82]{2,0,1:T(8,128)(2,1)S(3)}",
            "accepted qkv-a output type",
        ),
        ("fusion(fusion.6342", "accepted weighted output consumption"),
        (
            "fusion(fusion.6342, bitcast.24623, copy-done.350)",
            "accepted qkv-a operand roles",
        ),
        ("calls=fused_computation.16511", "accepted qkv-a callee"),
        (
            "DeepSeekV2FusedQkvAProjLinear/shard_map/dot_general",
            "accepted next-layer projection role",
        ),
    ):
        _require(accepted_live_consumer, needle, label)

    accepted_qkv_a_bitcast = _sealed_computation(
        accepted,
        "bitcast_fusion.232",
        "11eac6ac5ba67d1f1f3346c363e7226f6322c2268128c88f9f76dfefc3bccbd6",
        "accepted qkv-a input bitcast",
    )
    for needle, label in (
        (
            "bitcast_input.232 = bf16[32,6144]",
            "accepted qkv-a bitcast input type",
        ),
        (
            "ROOT bitcast.20841 = bf16[32,6144]",
            "accepted qkv-a bitcast output type",
        ),
        ("bitcast(bitcast_input.232)", "accepted qkv-a bitcast value flow"),
    ):
        _require(accepted_qkv_a_bitcast, needle, label)

    accepted_qkv_a = _sealed_computation(
        accepted,
        "fused_computation.16511",
        "b7cdfb915620603e24fa50b03f38aa71f69310a02b1642675cd2446cc5bed969",
        "accepted qkv-a consumer",
    )
    for needle, label in (
        (
            "param_0.48555 = bf16[32,6144]",
            "accepted qkv-a weighted-input type",
        ),
        (
            "fusion(param_0.48555), kind=kLoop, calls=bitcast_fusion.232",
            "accepted qkv-a weighted-input bitcast",
        ),
        ("param_1.55079 = f32[6144,82]", "accepted qkv-a scale type"),
        ("param_2.43308 = f8e4m3fn[6144,82]", "accepted qkv-a weight type"),
        (
            "multiply_convert_fusion.1803 = bf16[6144,82]",
            "accepted qkv-a decoded-weight type",
        ),
        (
            "convolution.3669 = f32[32,82]",
            "accepted qkv-a accumulation type",
        ),
        (
            "convolution(fusion.11239, multiply_convert_fusion.1803)",
            "accepted qkv-a convolution operands",
        ),
        (
            "convert_element_type.21737 = bf16[32,82]",
            "accepted qkv-a rounded output type",
        ),
        (
            "ROOT bitcast.18218 = bf16[32,1,82]",
            "accepted qkv-a live root type",
        ),
    ):
        _require(accepted_qkv_a, needle, label)
    _require_order(
        accepted_qkv_a,
        (
            "param_0.48555 =",
            "fusion.11239 =",
            "multiply_convert_fusion.1803 =",
            "convolution.3669 =",
            "convert_element_type.21737 =",
            "ROOT bitcast.18218 =",
        ),
        "accepted weighted-RMS-to-qkv-a value flow",
    )

    db518_embedding_select = _sealed_computation(
        db518,
        "%fused_computation.1854.clone.clone.clone",
        "783fe7a879d30405f2717f6808d6f4862ab0a9cd5dcfb52f3e3802b51152db86",
        "DB518 embedding owner selection",
    )
    for needle, label in (
        (
            "param_1.11515: bf16[1,77440,6144]",
            "DB518 embedding table source type",
        ),
        (
            "%constant.4603.clone.15 = bf16[]{:T(256)} constant(0)",
            "DB518 embedding BF16 zero fallback",
        ),
        (
            "broadcast(%constant.4603.clone.15)",
            "DB518 embedding BF16 zero broadcast",
        ),
        (
            "select(%select_n.5840, %dynamic_slice.771, %closed_call.913)",
            "DB518 embedding upper-half owner selection",
        ),
        (
            "select(%select_n.5840, %dynamic_slice.770, %closed_call.913)",
            "DB518 embedding lower-half owner selection",
        ),
        (
            "select(%select_n.5837, %select_n.5839, %select_n.5838)",
            "DB518 embedding feature-half selection",
        ),
        ("ROOT %bitcast.5975 = bf16[1,3072]", "DB518 embedding selected root"),
    ):
        _require(db518_embedding_select, needle, label)

    db518_initial_rms = _sealed_computation(
        db518,
        "%fused_computation.1900.clone.clone.clone",
        "1b0a8a34dc95b310cb8ddf0563544a709fbe04b8655ca80de3fa17c284e78c9f",
        "DB518 initial RMS",
    )
    for needle, label in (
        ("param_0.12003: bf16[1,3072]", "DB518 initial embedding source type"),
        (
            "param_1.11516: bf16[1,77440,6144]",
            "DB518 initial embedding-table source type",
        ),
        (
            "%constant.4603.clone.5.clone.4 = bf16[]{:T(256)} constant(0)",
            "DB518 initial BF16 zero fallback",
        ),
        (
            "broadcast(%constant.4603.clone.5.clone.4)",
            "DB518 initial BF16 zero broadcast",
        ),
        (
            "select(%select_n.5047.clone.4, %dynamic_slice.614.clone.4, "
            "%closed_call.633.clone.4)",
            "DB518 initial lower embedding-owner selection",
        ),
        (
            "select(%select_n.5047.clone.4, %dynamic_slice.615.clone.4, "
            "%closed_call.633.clone.4)",
            "DB518 initial upper embedding-owner selection",
        ),
        (
            "select(%select_n.5045.clone.4, %select_n.5046.clone.4, "
            "%select_n.5048.clone.4)",
            "DB518 initial complementary embedding-owner selection",
        ),
        ("bitcast(%select_n.5044.clone.4)", "DB518 initial local embedding shard"),
        (
            "add(%convert.1576, %convert.1577)",
            "DB518 initial zero-masked embedding-owner sum",
        ),
        ('"original_type":"BF16"', "DB518 initial BF16-corrected addition"),
        (
            "%convert_element_type.8495.clone.4 = "
            "f32[1,3072]{1,0:T(1,128)S(3)} "
            "convert(%add.12644.clone.4)",
            "DB518 initial corrected-state conversion",
        ),
        (
            "multiply(%convert_element_type.8495.clone.4, "
            "%convert_element_type.8495.clone.4)",
            "DB518 initial RMS square source",
        ),
        (
            "%reduce_sum.844 = f32[]{:T(128)} "
            "reduce(%square.410, %constant.4609.clone.29), dimensions={0,1}",
            "DB518 initial RMS reduction root source",
        ),
        (
            "ROOT %tuple.2686 = (f32[]{:T(128)}, "
            "f32[1,3072]{1,0:T(1,128)S(3)}) "
            "tuple(%reduce_sum.844, %convert_element_type.8495.clone.4)",
            "DB518 initial RMS live square/state root",
        ),
    ):
        _require(db518_initial_rms, needle, label)
    _require_once(
        db518_initial_rms,
        '"float_type_correction_info"',
        "DB518 initial BF16 correction",
    )

    db518_initial_call = _line(
        db518,
        "%multiply_reduce_fusion.127 = ",
        "DB518 initial RMS reduction call",
    )
    for needle, label in (
        (
            "fusion(%collective-permute-done.3, %get-tuple-element.17883, "
            "%min.1123, %and.4286, %eq.947)",
            "DB518 initial embedding/carried operand roles",
        ),
        (
            "calls=%fused_computation.1900.clone.clone.clone",
            "DB518 initial RMS callee",
        ),
        ('"output_window_bounds":["1","12"]', "DB518 initial RMS window"),
    ):
        _require(db518_initial_call, needle, label)

    for prefix, needles, label in (
        (
            "%get-tuple-element.17883 = ",
            (
                "bf16[1,77440,6144]",
                "get-tuple-element(%wide.wide.param.2), index=25",
            ),
            "DB518 embedding table owner",
        ),
        (
            "%fusion.3474 = ",
            (
                "bf16[1,3072]",
                "fusion(%eq.947, %get-tuple-element.17883, %min.1123, %and.4286)",
                "calls=%fused_computation.1854.clone.clone.clone",
            ),
            "DB518 embedding owner selection",
        ),
        (
            "%collective-permute-start.3 = ",
            (
                "collective-permute-start(%fusion.3474)",
                "source_target_pairs={{0,1},{1,0}}",
                "greenfield_pp16_feature2_embedding_exchange/ppermute",
            ),
            "DB518 embedding exchange",
        ),
        (
            "%collective-permute-done.3 = ",
            (
                "bf16[1,3072]",
                "collective-permute-done(%collective-permute-start.3)",
                "greenfield_pp16_feature2_embedding_exchange/ppermute",
            ),
            "DB518 embedding exchange completion",
        ),
        (
            "%get-tuple-element.15838 = ",
            (
                "f32[1,3072]",
                "get-tuple-element(%multiply_reduce_fusion.127), index=1",
            ),
            "DB518 initial corrected-embedding root",
        ),
        (
            "%copy-start.100 = ",
            ("f32[1,3072]", "copy-start(%get-tuple-element.15838)"),
            "DB518 initial corrected-embedding first copy",
        ),
        (
            "%copy-done.100 = ",
            ("f32[1,3072]", "copy-done(%copy-start.100)"),
            "DB518 initial corrected-embedding first copy completion",
        ),
        (
            "%copy-start.102 = ",
            ("f32[1,3072]", "copy-start(%copy-done.100)"),
            "DB518 initial corrected-embedding target copy",
        ),
        (
            "%copy-done.102 = ",
            ("f32[1,3072]", "copy-done(%copy-start.102)"),
            "DB518 initial corrected-embedding target operand",
        ),
        (
            "%constant_dynamic-update-slice_fusion.358 = ",
            (
                "bf16[1,3072]",
                "fusion(%constant_dynamic-update-slice_fusion.357, "
                "%add.14172)",
                "calls=%fused_computation.1716.clone.clone.clone",
                "greenfield_strategy_nd_feature2_attention_output/concatenate",
            ),
            "DB518 attention result",
        ),
        (
            "%copy-start.140 = ",
            (
                "bf16[1,3072]",
                "copy-start(%constant_dynamic-update-slice_fusion.358)",
            ),
            "DB518 attention first copy",
        ),
        (
            "%copy-done.140 = ",
            ("bf16[1,3072]", "copy-done(%copy-start.140)"),
            "DB518 attention first copy completion",
        ),
        (
            "%copy-start.141 = ",
            ("bf16[1,3072]", "copy-start(%copy-done.140)"),
            "DB518 attention target copy",
        ),
        (
            "%copy-done.141 = ",
            ("bf16[1,3072]", "copy-done(%copy-start.141)"),
            "DB518 attention target operand",
        ),
        (
            "%constant_dynamic-update-slice_fusion.370 = ",
            (
                "bf16[1,3072]",
                "fusion(%constant_dynamic-update-slice_fusion.369, "
                "%add.14189)",
                "calls=%fused_computation.1728.clone.clone.clone",
                "greenfield_strategy_nd_feature2_dense_down/concatenate",
            ),
            "DB518 dense target operand",
        ),
    ):
        source_line = _line(db518, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    _sealed_computation(
        db518,
        "%fused_computation.1716.clone.clone.clone",
        "e4566a7491860b063a3159e95086ee75d4e6b088035891cb0ab385a5044e00cd",
        "DB518 attention concat",
    )
    _sealed_computation(
        db518,
        "%fused_computation.1728.clone.clone.clone",
        "632ae1786a3a5db1cd249d36b428a3ed0e97dcc4b68b88aebedeadc041c06f49",
        "DB518 dense concat",
    )

    db518_rms = _sealed_computation(
        db518,
        "%fused_computation.1890.clone.clone.clone",
        "00a5186da4266e951ee2d80f9a9ea2200fdea6ccc421ca39a89c97b61abdea99",
        "DB518 target RMS",
    )
    for needle, label in (
        (
            "param_0.12224: f32[1,3072]",
            "DB518 BF16-corrected embedding source type",
        ),
        ("param_1.11639: bf16[1,3072]", "DB518 attention source type"),
        ("convert(%param_1.11639)", "DB518 attention source conversion"),
        ("param_2.7065: bf16[1,3072]", "DB518 dense source type"),
        ("convert(%param_2.7065)", "DB518 dense source conversion"),
        (
            "add(%convert_element_type.8493.clone.4, %param_0.12224)",
            "DB518 attention plus embedding",
        ),
        ("convert(%add.12640.clone.4)", "DB518 carried BF16 round"),
        (
            "add(%convert_element_type.8492.clone.4, "
            "%convert_element_type.8490.clone.4)",
            "DB518 dense plus rounded carried",
        ),
        ("multiply(%add.12639.clone.4, %add.12639.clone.4)", "DB518 RMS square source"),
        (
            "%reduce_sum.848 = f32[]{:T(128)} "
            "reduce(%square.414, %constant.4609.clone.35), dimensions={0,1}",
            "DB518 RMS reduction root source",
        ),
        (
            "ROOT %tuple.2754 = (f32[]{:T(128)}, "
            "f32[1,3072]{1,0:T(1,128)S(3)}) "
            "tuple(%reduce_sum.848, %add.12639.clone.4)",
            "DB518 RMS square/state tuple root",
        ),
    ):
        _require(db518_rms, needle, label)
    _require_order(
        db518_rms,
        (
            "%convert_element_type.8493.clone.4 =",
            "%add.12640.clone.4 =",
            "%convert_element_type.8491.clone.4 = bf16",
            "%convert_element_type.8490.clone.4 = f32",
            "%add.12639.clone.4 =",
            "%square.414 =",
            "%reduce_sum.848 =",
            "ROOT %tuple.2754 =",
        ),
        "DB518 residual/attention/dense round order",
    )

    db518_weighted = _sealed_computation(
        db518,
        "%fused_computation.1889.clone.clone.clone",
        "335b564aa878ee909190efbc1e19e72d4eb727b5024a9f131470da51a8a15eeb",
        "DB518 weighted RMS",
    )
    for needle, label in (
        ('"float_type_correction_info"', "DB518 BF16 correction"),
        (
            "broadcast(%param_2.7066), dimensions={}",
            "DB518 inverse broadcast",
        ),
        (
            "multiply(%param_1.11640, %mul.9794)",
            "DB518 inverse normalization multiply",
        ),
        ("convert(%mul.9793)", "DB518 normalized BF16 round"),
        ("param_0.12225 = bf16[1,3072]", "DB518 RMS weight source"),
        ("convert(%param_0.12225)", "DB518 RMS weight conversion"),
        (
            "multiply(%convert.1846, %convert.1847)",
            "DB518 corrected weight multiply",
        ),
        (
            "ROOT %convert.1848 = bf16[1,3072]"
            "{1,0:T(2,128)(2,1)S(3)} convert(%mul.9792)",
            "DB518 weighted half output",
        ),
    ):
        _require(db518_weighted, needle, label)

    db518_call = _line(
        db518,
        "%convert_multiply_fusion.67 = ",
        "DB518 weighted call",
    )
    for needle, label in (
        ("bf16[1,3072]{1,0:T(2,128)(2,1)S(3)}", "DB518 weighted half geometry"),
        (
            "fusion(%copy-done.114, %get-tuple-element.16173, %rsqrt.332)",
            "DB518 weight/state/inverse operand roles",
        ),
        ("calls=%fused_computation.1889.clone.clone.clone", "DB518 weighted callee"),
        ('"output_window_bounds":["1","12"]', "DB518 output window"),
        ('"megacore_split_dim":"1"', "DB518 megacore split"),
    ):
        _require(db518_call, needle, label)

    db518_reduce_call = _line(
        db518,
        "%multiply_reduce_fusion.131 = ",
        "DB518 RMS reduction call",
    )
    for needle, label in (
        (
            "fusion(%copy-done.102, %copy-done.141, "
            "%constant_dynamic-update-slice_fusion.370)",
            "DB518 embedding/attention/dense operand roles",
        ),
        ("calls=%fused_computation.1890.clone.clone.clone", "DB518 RMS callee"),
        ('"output_window_bounds":["1","12"]', "DB518 reduction window"),
    ):
        _require(db518_reduce_call, needle, label)
    db518_feature_reduce = _line(db518, "%psum.335 = ", "DB518 feature reduction")
    for needle, label in (
        ("all-reduce(%get-tuple-element.16181)", "DB518 reduced square source"),
        ("replica_groups={{0,1}}", "DB518 feature group"),
        ("greenfield_pp16_feature2_rms/psum", "DB518 feature-reduction role"),
    ):
        _require(db518_feature_reduce, needle, label)

    for prefix, needles, label in (
        (
            "%get-tuple-element.16181 = ",
            (
                "f32[]",
                "get-tuple-element(%multiply_reduce_fusion.131), index=0",
            ),
            "DB518 RMS square-reduction result",
        ),
        (
            "%get-tuple-element.16173 = ",
            (
                "f32[1,3072]",
                "get-tuple-element(%multiply_reduce_fusion.131), index=1",
            ),
            "DB518 unnormalized RMS state result",
        ),
    ):
        source_line = _line(db518, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    for prefix, needles, label in (
        (
            "%constant.4432.clone.2..sunk.2 = ",
            ("f32[]", "constant(0.000162760422)"),
            "DB518 hidden-width reciprocal",
        ),
        (
            "%constant.4710.clone.1..sunk.2 = ",
            ("f32[]", "constant(1e-05)"),
            "DB518 RMS epsilon",
        ),
        (
            "%div.2725 = ",
            (
                "multiply(%psum.335, %constant.4432.clone.2..sunk.2)",
                "greenfield_pp16_feature2_rms/div",
            ),
            "DB518 RMS mean",
        ),
        (
            "%add.14192 = ",
            (
                "add(%div.2725, %constant.4710.clone.1..sunk.2)",
                "greenfield_pp16_feature2_rms/add",
            ),
            "DB518 RMS epsilon addition",
        ),
        (
            "%rsqrt.332 = ",
            ("rsqrt(%add.14192)", "greenfield_pp16_feature2_rms/rsqrt"),
            "DB518 RMS inverse",
        ),
        (
            "%get-tuple-element.17905 = ",
            (
                "bf16[1,3072]",
                "get-tuple-element(%wide.wide.param.2), index=47",
            ),
            "DB518 layer-1 RMS weight owner",
        ),
        (
            "%copy-start.114 = ",
            ("bf16[1,3072]", "copy-start(%get-tuple-element.17905)"),
            "DB518 layer-1 RMS weight copy",
        ),
        (
            "%copy-done.114 = ",
            ("bf16[1,3072]", "copy-done(%copy-start.114)"),
            "DB518 layer-1 RMS weight operand",
        ),
    ):
        source_line = _line(db518, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    db518_gather = _line(db518, "%all_gather.507 = ", "DB518 post-weight gather")
    for needle, label in (
        ("bf16[1,6144]", "DB518 post-weight full-width gather"),
        ("all-gather(%convert_multiply_fusion.67)", "DB518 gather-after-weight order"),
        ("replica_groups={{0,1}}", "DB518 normalized-feature group"),
        ("greenfield_pp16_feature2_normalized_gather", "DB518 gather role"),
    ):
        _require(db518_gather, needle, label)
    for prefix, needles, label in (
        (
            "%copy-start.76 = ",
            ("bf16[1,6144]", "copy-start(%all_gather.507)"),
            "DB518 gathered-output copy",
        ),
        (
            "%copy-done.76 = ",
            ("bf16[1,6144]", "copy-done(%copy-start.76)"),
            "DB518 gathered-output completion",
        ),
        (
            "%dynamic_update_slice.160 = ",
            ("bf16[2012,6144]", "%all_gather.507"),
            "DB518 gathered-output cache consumer",
        ),
        (
            "ROOT %tuple.2930 = ",
            ("%copy-done.76", "%dynamic_update_slice.160"),
            "DB518 gathered-output live root",
        ),
    ):
        source_line = _line(db518, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    ws32_post_attention_reduce = _sealed_computation(
        ws32,
        "%fused_computation.13424",
        "9f543ac5c782b420e5aa3e71c95796da0baeccdafa9301ed4c05edce27d0e0ed",
        "WS32 post-attention RMS reduction",
    )
    for needle, label in (
        ("param_0.40668: bf16[1,1536]", "WS32 carried-residual source type"),
        ("param_1.37023: bf16[1,1536]", "WS32 attention-update source type"),
        (
            "add(%convert_element_type.21274, %convert_element_type.21273)",
            "WS32 post-attention residual association",
        ),
        ("multiply(%add.34749, %add.34749)", "WS32 post-attention square source"),
        (
            "ROOT %reduce_sum.7563 = f32[]{:T(128)} "
            "reduce(%square.1590, %constant.25099.clone.644), "
            "dimensions={0,1}",
            "WS32 post-attention reduction root",
        ),
    ):
        _require(ws32_post_attention_reduce, needle, label)

    ws32_post_attention_call = _line(
        ws32,
        "%multiply_reduce_fusion.456 = ",
        "WS32 post-attention RMS reduction call",
    )
    for needle, label in (
        (
            "fusion(%psum.8778, %psum.8781)",
            "WS32 embedding/attention operand roles",
        ),
        ("calls=%fused_computation.13424", "WS32 post-attention callee"),
        ('"output_window_bounds":["1","6"]', "WS32 post-attention window"),
        ('"megacore_split_dim":"1"', "WS32 post-attention megacore split"),
    ):
        _require(ws32_post_attention_call, needle, label)

    ws32_post_attention_weighted = _sealed_computation(
        ws32,
        "%fused_computation.13423",
        "606703f16cbb5ffb259083651475fdb640b5fee76285a34f1558ecb8d3101cc5",
        "WS32 post-attention weighted RMS",
    )
    for needle, label in (
        (
            "add(%convert_element_type.21268, %convert_element_type.21267)",
            "WS32 weighted post-attention residual association",
        ),
        (
            "broadcast(%param_1.37018), dimensions={}",
            "WS32 post-attention inverse broadcast",
        ),
        (
            "multiply(%add.34747, %mul.25127)",
            "WS32 post-attention inverse normalization multiply",
        ),
        ("convert(%mul.24804)", "WS32 post-attention normalized BF16 round"),
        ("param_0.32591 = bf16[1536]", "WS32 post-attention RMS weight source"),
        (
            "reshape(%param_0.32591)",
            "WS32 post-attention RMS weight broadcast",
        ),
        (
            "multiply(%convert.11481, %convert.11482)",
            "WS32 post-attention corrected weight multiply",
        ),
        ('"float_type_correction_info"', "WS32 post-attention BF16 correction"),
        (
            "ROOT %convert.11483 = bf16[1,1536]"
            "{1,0:T(2,128)(2,1)S(3)} convert(%mul.24803)",
            "WS32 post-attention output",
        ),
    ):
        _require(ws32_post_attention_weighted, needle, label)

    ws32_post_attention_weighted_call = _line(
        ws32,
        "%reshape_multiply_fusion.232 = ",
        "WS32 weighted post-attention call",
    )
    for needle, label in (
        ("bf16[1,1536]{1,0:T(2,128)(2,1)S(3)}", "WS32 post-attention tile"),
        (
            "fusion(%copy-done.2852, %bitcast.8854, %psum.8778, %psum.8781)",
            "WS32 post-attention weight/inverse/state roles",
        ),
        ("calls=%fused_computation.13423", "WS32 weighted post-attention callee"),
        ('"output_window_bounds":["1","6"]', "WS32 post-attention output window"),
        ('"megacore_split_dim":"1"', "WS32 weighted post-attention split"),
    ):
        _require(ws32_post_attention_weighted_call, needle, label)

    for prefix, needles, label in (
        (
            "%bitcast.8853 = ",
            ("bitcast(%multiply_reduce_fusion.456)",),
            "WS32 post-attention square scalar",
        ),
        (
            "%psum.8782 = ",
            (
                "all-reduce(%bitcast.8853)",
                "replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},"
                "{12,13,14,15},{16,17,18,19},{20,21,22,23},"
                "{24,25,26,27},{28,29,30,31}}",
                "greenfield_ws32_fused_rmsnorm/feature_square_reduce/psum",
            ),
            "WS32 post-attention feature reduction",
        ),
        (
            "%div.7079 = ",
            ("multiply(%psum.8782, %constant.25100.clone.1)",),
            "WS32 post-attention RMS mean",
        ),
        (
            "%add.28822 = ",
            ("add(%div.7079, %constant.25101.clone.1)",),
            "WS32 post-attention epsilon addition",
        ),
        (
            "%rsqrt.1098 = ",
            ("rsqrt(%add.28822)",),
            "WS32 post-attention inverse",
        ),
        (
            "%bitcast.8854 = ",
            ("bitcast(%rsqrt.1098)",),
            "WS32 post-attention weighted inverse operand",
        ),
        (
            "%get-tuple-element.26443 = ",
            ("bf16[1536]", "get-tuple-element(%param.4144), index=19"),
            "WS32 post-attention RMS weight owner",
        ),
        (
            "%copy-start.2852 = ",
            ("bf16[1536]", "copy-start(%get-tuple-element.26443)"),
            "WS32 post-attention RMS weight copy",
        ),
        (
            "%copy-done.2852 = ",
            ("bf16[1536]", "copy-done(%copy-start.2852)"),
            "WS32 post-attention RMS weight operand",
        ),
    ):
        source_line = _line(ws32, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    ws32_dense_input_gather = _line(
        ws32, "%all_gather.931 = ", "WS32 layer-0 dense input gather"
    )
    for needle, label in (
        ("bf16[1,6144]", "WS32 dense input full width"),
        (
            "all-gather(%reshape_multiply_fusion.232)",
            "WS32 dense consumes post-attention normalization",
        ),
        (
            "greenfield_ws32_strategy_nd_dense/hidden_gather/all_gather",
            "WS32 layer-0 dense input role",
        ),
    ):
        _require(ws32_dense_input_gather, needle, label)
    ws32_first_dense_consumer = _line(
        ws32,
        "%convolution_convert_fusion.11 = ",
        "WS32 layer-0 dense first consumer",
    )
    for needle, label in (
        ("fusion(%all_gather.931", "WS32 layer-0 dense normalized input"),
        (
            "greenfield_ws32_strategy_nd_dense/virtual_partials/"
            "greenfield_dense_convolution_virtual_rank_00",
            "WS32 layer-0 dense consumer role",
        ),
    ):
        _require(ws32_first_dense_consumer, needle, label)

    ws32_rms = _sealed_computation(
        ws32,
        "%fused_computation.13416",
        "c164809741db13ce40f3db3df51a54998c59bb206543ca1ef5e3a846535e3232",
        "WS32 target RMS reduction",
    )
    for needle, label in (
        ("param_0.42801: bf16[1,1536]", "WS32 embedding source type"),
        ("param_1.40448: bf16[1,1536]", "WS32 attention source type"),
        ("param_2.30735: bf16[1,1536]", "WS32 first dense candidate type"),
        ("param_3.17824: bf16[1,1536]", "WS32 second dense candidate type"),
        (
            "add(%convert_element_type.23931.clone.1, "
            "%convert_element_type.23930.clone.1)",
            "WS32 attention plus embedding",
        ),
        ("convert(%add.35997.clone.1)", "WS32 carried BF16 round"),
        (
            "add(%convert_element_type.23929.clone.1, "
            "%convert_element_type.23927.clone.1)",
            "WS32 dense plus rounded carried",
        ),
        (
            "multiply(%add.35993.clone.1, %add.35993.clone.1)",
            "WS32 RMS square source",
        ),
        (
            "%reduce_sum.7562 = f32[]{:T(128)} "
            "reduce(%square.1589, %constant.25099.clone.643), "
            "dimensions={0,1}",
            "WS32 RMS reduction root source",
        ),
        (
            "ROOT %tuple.11414 = (f32[]{:T(128)}, "
            "f32[1,1536]{1,0:T(1,128)S(3)}) "
            "tuple(%reduce_sum.7562, %add.35993.clone.1)",
            "WS32 live RMS square/state tuple root",
        ),
    ):
        _require(ws32_rms, needle, label)
    _require_order(
        ws32_rms,
        (
            "%param_2.30735 =",
            "%param_3.17824 =",
            "%select_n.16889.clone.1 =",
            "%convert_element_type.23931.clone.1 =",
            "%convert_element_type.23930.clone.1 =",
            "%add.35997.clone.1 =",
            "%convert_element_type.23928.clone.1 = bf16",
            "%convert_element_type.23927.clone.1 = f32",
            "%add.35993.clone.1 =",
            "%square.1589 =",
            "%reduce_sum.7562 =",
            "ROOT %tuple.11414 =",
        ),
        "WS32 residual/attention/dense round order",
    )

    ws32_weighted = _sealed_computation(
        ws32,
        "%fused_computation.13415",
        "05a0e45ac1d84ae6b489dfc9916131343543a58bc523d1db760dbb590d8fffb8",
        "WS32 target weighted RMS",
    )
    for needle, label in (
        ("param_1.26801: f32[1,1536]", "WS32 unnormalized RMS input"),
        (
            "broadcast(%param_2.21812), dimensions={}",
            "WS32 inverse broadcast",
        ),
        (
            "multiply(%param_1.26801, %mul.25126)",
            "WS32 inverse normalization multiply",
        ),
        ("convert(%mul.24802)", "WS32 normalized BF16 round"),
        ("param_0.32575 = bf16[1536]", "WS32 RMS weight source"),
        ("reshape(%param_0.32575)", "WS32 RMS weight broadcast"),
        (
            "multiply(%convert.11470, %convert.11471)",
            "WS32 corrected weight multiply",
        ),
        ('"float_type_correction_info"', "WS32 BF16 correction"),
        (
            "ROOT %convert.11472 = bf16[1,1536]"
            "{1,0:T(2,128)(2,1)S(3)} convert(%mul.24801)",
            "WS32 weighted shard output",
        ),
    ):
        _require(ws32_weighted, needle, label)
    _require_once(
        ws32_weighted,
        '"float_type_correction_info"',
        "WS32 target BF16 correction",
    )
    _require_order(
        ws32_weighted,
        (
            "%mul.24802 =",
            "%convert_element_type.20269 = bf16",
            "%convert.11470 = f32",
            "%param_0.32575 = bf16",
            "%mul.24801 =",
            "ROOT %convert.11472 =",
        ),
        "WS32 weighted value flow",
    )

    ws32_reduce_call = _line(
        ws32,
        "%multiply_reduce_fusion.455 = ",
        "WS32 layer-0 RMS reduction call",
    )
    for needle, label in (
        (
            "fusion(%psum.8778, %psum.8781, %add.28843, %add.28848, "
            "%mul.17735)",
            "WS32 embedding/attention/dense operand roles",
        ),
        ("calls=%fused_computation.13416", "WS32 RMS callee"),
        ('"output_window_bounds":["1","6"]', "WS32 RMS output window"),
        ('"megacore_split_dim":"1"', "WS32 RMS megacore split"),
    ):
        _require(ws32_reduce_call, needle, label)

    ws32_row0_tree = _sealed_computation(
        ws32,
        "%fused_computation.13418",
        "ba07849f81256e6913e35f86f1f271bda0eccc51a5881a451b4e6532c10db8a9",
        "WS32 dense row0-tree tuple",
    )
    for needle, label in (
        (
            "add(%convert.11473, %convert.11474)",
            "WS32 dense row0-tree first pair",
        ),
        (
            "ROOT %tuple.11544 = (bf16[1,1536]",
            "WS32 dense row0-tree tuple root",
        ),
        (
            "tuple(%bitcast.26036, %bitcast.26038.clone.1, "
            "%bitcast.26037.clone.1, %bitcast.26039.clone.1)",
            "WS32 dense row0-tree exact root members",
        ),
    ):
        _require(ws32_row0_tree, needle, label)

    for prefix, needles, label in (
        (
            "%add_bitcast_fusion.15 = ",
            (
                "fusion(%copy_bitcast_fusion.97)",
                "calls=%fused_computation.13418",
                "greenfield_ws32_strategy_nd_dense/row0_tree/add",
            ),
            "WS32 dense partial tuple",
        ),
        (
            "%get-tuple-element.71300 = ",
            ("get-tuple-element(%add_bitcast_fusion.15), index=2",),
            "WS32 first dense partial 2",
        ),
        (
            "%get-tuple-element.71298 = ",
            ("get-tuple-element(%add_bitcast_fusion.15), index=0",),
            "WS32 first dense partial 0",
        ),
        (
            "%get-tuple-element.71301 = ",
            ("get-tuple-element(%add_bitcast_fusion.15), index=3",),
            "WS32 second dense partial 3",
        ),
        (
            "%get-tuple-element.71299 = ",
            ("get-tuple-element(%add_bitcast_fusion.15), index=1",),
            "WS32 second dense partial 1",
        ),
        (
            "%add.28848 = ",
            (
                "add(%get-tuple-element.71300, %get-tuple-element.71298)",
                "greenfield_ws32_strategy_nd_dense/row0_tree/add",
            ),
            "WS32 first dense target operand",
        ),
        (
            "%add.28843 = ",
            (
                "add(%get-tuple-element.71301, %get-tuple-element.71299)",
                "greenfield_ws32_strategy_nd_dense/row0_tree/add",
            ),
            "WS32 second dense target operand",
        ),
    ):
        source_line = _line(ws32, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    ws32_weighted_call = _line(
        ws32,
        "%reshape_multiply_fusion.231 = ",
        "WS32 weighted call",
    )
    for needle, label in (
        ("bf16[1,1536]{1,0:T(2,128)(2,1)S(3)}", "WS32 output tile"),
        (
            "fusion(%copy-done.2853, %get-tuple-element.71073, "
            "%bitcast.8889)",
            "WS32 weight/state/inverse operand roles",
        ),
        ("calls=%fused_computation.13415", "WS32 weighted callee"),
        ('"output_window_bounds":["1","6"]', "WS32 weighted output window"),
        ('"megacore_split_dim":"1"', "WS32 weighted megacore split"),
    ):
        _require(ws32_weighted_call, needle, label)

    ws32_feature_reduce = _line(ws32, "%psum.8783 = ", "WS32 feature reduction")
    for needle, label in (
        ("all-reduce(%bitcast.8888)", "WS32 reduced square source"),
        (
            "replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},"
            "{12,13,14,15},{16,17,18,19},{20,21,22,23},"
            "{24,25,26,27},{28,29,30,31}}",
            "WS32 feature groups",
        ),
        (
            "greenfield_ws32_fused_rmsnorm/feature_square_reduce/psum",
            "WS32 feature role",
        ),
    ):
        _require(ws32_feature_reduce, needle, label)

    for prefix, needles, label in (
        (
            "%get-tuple-element.71072 = ",
            (
                "f32[]",
                "get-tuple-element(%multiply_reduce_fusion.455), index=0",
            ),
            "WS32 RMS square-reduction result",
        ),
        (
            "%get-tuple-element.71073 = ",
            (
                "f32[1,1536]",
                "get-tuple-element(%multiply_reduce_fusion.455), index=1",
            ),
            "WS32 unnormalized RMS state result",
        ),
        (
            "%constant.25100.clone.1 = ",
            ("f32[1,1]", "constant({ {0.000162760422} })"),
            "WS32 hidden-width reciprocal",
        ),
        (
            "%constant.25101.clone.1 = ",
            ("f32[1,1]", "constant({ {1e-05} })"),
            "WS32 RMS epsilon",
        ),
        (
            "%div.7084 = ",
            ("multiply(%psum.8783, %constant.25100.clone.1)",),
            "WS32 RMS mean",
        ),
        (
            "%add.28850 = ",
            ("add(%div.7084, %constant.25101.clone.1)",),
            "WS32 RMS epsilon addition",
        ),
        (
            "%rsqrt.1099 = ",
            ("rsqrt(%add.28850)",),
            "WS32 RMS inverse",
        ),
        (
            "%bitcast.8889 = ",
            ("bitcast(%rsqrt.1099)",),
            "WS32 weighted inverse operand",
        ),
        (
            "%get-tuple-element.26448 = ",
            ("bf16[1536]", "get-tuple-element(%param.4144), index=24"),
            "WS32 layer-1 RMS weight owner",
        ),
        (
            "%copy-done.2853 = ",
            ("bf16[1536]", "copy-done(%copy-start.2853)"),
            "WS32 RMS-weight operand",
        ),
    ):
        source_line = _line(ws32, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    ws32_gather = _line(ws32, "%all_gather.934 = ", "WS32 post-weight gather")
    for needle, label in (
        ("bf16[1,6144]", "WS32 post-weight full-width gather"),
        ("all-gather(%reshape_multiply_fusion.231)", "WS32 gather-after-weight order"),
        (
            "replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},"
            "{12,13,14,15},{16,17,18,19},{20,21,22,23},"
            "{24,25,26,27},{28,29,30,31}}",
            "WS32 normalized-feature groups",
        ),
        ("greenfield_ws32_exact_dsa/normalized_feature_gather", "WS32 gather role"),
    ):
        _require(ws32_gather, needle, label)
    ws32_gather_consumer = _line(
        ws32, "%convert.3385 = ", "WS32 post-weight live consumer"
    )
    for needle, label in (
        ("f32[1,6144]", "WS32 gathered consumer width"),
        ("convert(%all_gather.934)", "WS32 gathered-output live use"),
    ):
        _require(ws32_gather_consumer, needle, label)

    for prefix, needles, label in (
        (
            "%psum.8778 = ",
            (
                "all-reduce(%broadcast_select_fusion.165)",
                "greenfield_ws32_embedding/expert_owner_reduce/psum",
                "replica_groups={{0,4,8,12,16,20,24,28}",
            ),
            "WS32 embedding lineage",
        ),
        (
            "%slice.34982 = ",
            (
                "f32[1,1536]",
                "slice(%greenfield_fp8_block_matmul_f32_m8_k2048_n1536.78)",
                "slice={[0:1], [0:1536]}",
            ),
            "WS32 attention projection row lineage",
        ),
        (
            "%psum.8781 = ",
            (
                "all-reduce(%slice.34982)",
                "greenfield_ws32_linear/expert_reduce/psum",
                "replica_groups={{0,4,8,12,16,20,24,28}",
            ),
            "WS32 attention lineage",
        ),
        (
            "%mul.17735 = ",
            (
                "s32[]",
                "multiply(%axis_index.17, %constant.25110.clone.1)",
                "greenfield_ws32_complete_decoder/mul",
            ),
            "WS32 target row selector",
        ),
        (
            "%copy-start.2853 = ",
            ("bf16[1536]", "copy-start(%get-tuple-element.26448)"),
            "WS32 RMS-weight lineage",
        ),
    ):
        source_line = _line(ws32, prefix, label)
        for needle in needles:
            _require(source_line, needle, label)

    return {
        "accepted": {
            "dense_projection_result_dtype": "bf16",
            "logical_shape": [32, 6144],
            "megacore_split_dim": 0,
            "output_tile": "T(8,128)(2,1)",
            "output_window_bounds": [2, 48],
            "rms_feature_group_size": 1,
            "rms_local_square_input_shape": [32, 6144],
            "rms_association": (
                "fp32(bf16_dense + bf16(bf16_attention + bf16_residual))"
            ),
        },
        "accepted_qkv_a_boundary": {
            "accumulation_dtype": "f32",
            "consumer_input_dtype": "bf16",
            "decoded_weight_dtype": "bf16",
            "physical_materialization_claim": False,
            "pre_round_fp32_operand_exposed_in_hlo": False,
            "value_flow": "bf16_weighted_rms -> bf16_bitcast -> f32_convolution",
        },
        "artifact_kind": "db485_layer1_rms_hlo_causality_v3",
        "classification": (
            "LOGICAL_BF16_ASSOCIATION_MATCHES;"
            "UNROUNDED_STATE_PHYSICAL_CAUSE_UNRESOLVED;"
            "RMS_REDUCTION_AND_WEIGHTED_OUTPUT_GEOMETRY_REMAIN_DISTINCT"
        ),
        "correction_materialization_proven": False,
        "db518": {
            "logical_weighted_shape_per_owner": [1, 3072],
            "megacore_split_dim": 1,
            "output_tile": "T(2,128)(2,1)",
            "output_window_bounds": [1, 12],
            "post_weight_all_gather_shape": [1, 6144],
            "rms_feature_group_size": 2,
            "rms_local_square_input_shape": [1, 3072],
            "rms_association": (
                "fp32(bf16_dense + "
                "bf16(bf16_attention + "
                "bf16_corrected_embedding_owner_sum_as_fp32))"
            ),
            "source_role_certificate": (
                "initial BF16-corrected embedding-owner sum -> exact copy chain -> "
                "attention copy chain+dense output -> next-layer RMS -> live gather"
            ),
        },
        "gate_d_passed": False,
        "historical_freeze_required": True,
        "logical_bf16_association_matches": True,
        "numerical_claim": False,
        "performance_claim": False,
        "tpu_successor_authorized": False,
        "unrounded_state_adjudicated": False,
        "ws32": {
            "feature_group_size": 4,
            "logical_weighted_shape_per_owner": [1, 1536],
            "megacore_split_dim": 1,
            "normalized_feature_group_size": 4,
            "output_tile": "T(2,128)(2,1)",
            "output_window_bounds": [1, 6],
            "post_weight_all_gather_shape": [1, 6144],
            "rms_local_square_input_shape": [1, 1536],
            "full_dense_value_path_claim": False,
            "pre_dense_boundary_certificate": (
                "embedding+update -> post-update RMS -> one dense input consumer"
            ),
            "rms_association": (
                "fp32(bf16_strategy_nd_row0_tree + "
                "bf16(bf16_update + bf16_embedding))"
            ),
            "target_boundary_certificate": (
                "exact BF16 StrategyND row0-tree operands+rounded carried -> "
                "next-layer RMS"
            ),
        },
    }


def classify_files(
    accepted_hlo: Path, db518_hlo: Path, ws32_hlo: Path
) -> dict[str, Any]:
    accepted_gzip = accepted_hlo.read_bytes()
    if _sha256(accepted_gzip) != ACCEPTED_GZIP_SHA256:
        raise ValueError("accepted DB485 HLO gzip SHA-256 drifted")
    accepted_raw = gzip.decompress(accepted_gzip)
    if _sha256(accepted_raw) != ACCEPTED_RAW_SHA256:
        raise ValueError("accepted DB485 HLO raw SHA-256 drifted")
    db518_raw = db518_hlo.read_bytes()
    if _sha256(db518_raw) != DB518_OPTIMIZED_SHA256:
        raise ValueError("DB518 optimized HLO SHA-256 drifted")
    ws32_raw = ws32_hlo.read_bytes()
    if _sha256(ws32_raw) != WS32_OPTIMIZED_SHA256:
        raise ValueError("WS32 optimized HLO SHA-256 drifted")
    return {
        **classify_texts(
            accepted_raw.decode(errors="strict"),
            db518_raw.decode(errors="strict"),
            ws32_raw.decode(errors="strict"),
        ),
        "sealed_inputs": {
            "accepted_hlo_gzip_sha256": ACCEPTED_GZIP_SHA256,
            "accepted_hlo_raw_sha256": ACCEPTED_RAW_SHA256,
            "db518_optimized_hlo_sha256": DB518_OPTIMIZED_SHA256,
            "ws32_optimized_hlo_sha256": WS32_OPTIMIZED_SHA256,
        },
        "status": "CLASSIFIED",
    }


def _write_append_only_json(output: Path, result: dict[str, Any]) -> None:
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-hlo", type=Path, required=True)
    parser.add_argument("--db518-hlo", type=Path, required=True)
    parser.add_argument("--ws32-hlo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify_files(args.accepted_hlo, args.db518_hlo, args.ws32_hlo)
    _write_append_only_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
